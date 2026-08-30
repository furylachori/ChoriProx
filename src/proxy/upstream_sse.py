"""Standalone upstream OpenAI-compatible SSE parser.

Consumes the byte lines produced by ``response.iter_lines(decode_unicode=False)``
and reconstructs one complete OpenAI Chat Completion dictionary.

This is a standard-library-only leaf module. It does not import Flask, Requests,
proxy.state, pipeline modules, tool parsing, provider compatibility, or client
adapters.

Output shape is always a completed response (``object="chat.completion"``,
``message`` not ``delta``), never a raw upstream delta.
"""

# ============================================================================
# WIRED AND LOAD-BEARING (STREAMING_TICKET_199ece93, Part A).
# For client stream=true requests this module is on the production path:
#   - ProviderCompatStage preserves ctx.wants_stream (upstream body stream=true);
#   - UpstreamStage passes wants_stream=True to call_upstream;
#   - call_upstream (proxy.http_client) requests Requests stream=True and feeds
#     response.iter_lines() into OpenAIStreamAccumulator, reconstructing one
#     complete OpenAI Chat Completion dict before PostProcessStage runs.
# Internal memory-tool requeries (requery.py) remain non-streaming JSON calls
# (intentional asymmetry). Do NOT delete or bypass this module — it is the
# production upstream SSE reconstruction path.
# ============================================================================

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


class UpstreamStreamError(Exception):
    """Base exception for upstream SSE parsing failures.

    Attributes:
        semantic_started: whether semantic output had begun at raise time.
        code: sanitized stable machine-readable error code.
    """

    def __init__(self, message: str, *, code: str, semantic_started: bool) -> None:
        super().__init__(message)
        self.code = code
        self.semantic_started = semantic_started

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, semantic_started={self.semantic_started})"


class UpstreamStreamProtocolError(UpstreamStreamError):
    """Protocol violation that does not concern size limits."""

    pass


class UpstreamStreamLimitError(UpstreamStreamError):
    """A finite parser limit was exceeded."""

    pass


_SANITIZED_CODE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class UpstreamStreamLimits:
    """Finite parser limits. All values must be genuine positive integers.

    Invariants enforced at construction:
        - every value is a positive integer (bools rejected even though bool
          subclasses int);
        - line <= event;
        - content, reasoning, tool_argument <= response.
    """

    max_line_bytes: int = 1_048_576
    max_event_bytes: int = 2_097_152
    max_response_bytes: int = 67_108_864
    max_content_bytes: int = 33_554_432
    max_reasoning_bytes: int = 33_554_432
    max_tool_argument_bytes: int = 16_777_216
    max_tool_calls: int = 128

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"limit {name} must be a positive integer, got {value!r}")
        if self.max_line_bytes > self.max_event_bytes:
            raise ValueError("max_line_bytes must be <= max_event_bytes")
        for name in ("max_content_bytes", "max_reasoning_bytes", "max_tool_argument_bytes"):
            if getattr(self, name) > self.max_response_bytes:
                raise ValueError(f"{name} must be <= max_response_bytes")


DEFAULT_UPSTREAM_STREAM_LIMITS = UpstreamStreamLimits()


@dataclass
class _ToolCall:
    """Per-index tool-call accumulator."""

    call_id: str | None = None
    call_type: str | None = None
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)
    arguments_present: bool = False


class _StagedEvent:
    """Per-event staged snapshot of accumulator state.

    Holds a deep copy of every mutable field plus per-event state
    (``_event_bytes``, ``_event_semantic``, ``_dirty``). All ingestion for one
    JSON event is applied to a staged snapshot. If anything raises during
    staging, the parent is unchanged (atomic discard). The staged has NO
    reference to the parent — ``mark_semantic`` and ``mark_dirty`` are
    staged-local; the parent is updated only via ``_commit_staged_event``.
    """

    __slots__ = (
        "_tool_calls",
        "_tool_argument_bytes",
        "_usage",
        "_usage_present",
        "_top",
        "_content_parts",
        "_content_present",
        "_reasoning_content_parts",
        "_reasoning_content_present",
        "_reasoning_parts",
        "_reasoning_present",
        "_content_bytes",
        "_reasoning_bytes",
        "_role",
        "_finish_reason",
        "_choice_terminal",
        "_has_choice",
        "_event_bytes",
        "_event_semantic",
        "_dirty",
    )

    def __init__(self, source: "OpenAIStreamAccumulator") -> None:
        self._tool_calls = {
            k: _ToolCall(
                call_id=v.call_id,
                call_type=v.call_type,
                name_parts=list(v.name_parts),
                argument_parts=list(v.argument_parts),
                arguments_present=v.arguments_present,
            )
            for k, v in source._tool_calls.items()
        }
        self._tool_argument_bytes = source._tool_argument_bytes
        self._usage = copy.deepcopy(source._usage) if source._usage is not None else None
        self._usage_present = source._usage_present
        self._top = copy.deepcopy(source._top)
        self._content_parts = list(source._content_parts)
        self._content_present = source._content_present
        self._reasoning_content_parts = list(source._reasoning_content_parts)
        self._reasoning_content_present = source._reasoning_content_present
        self._reasoning_parts = list(source._reasoning_parts)
        self._reasoning_present = source._reasoning_present
        self._content_bytes = source._content_bytes
        self._reasoning_bytes = source._reasoning_bytes
        self._role = source._role
        self._finish_reason = source._finish_reason
        self._choice_terminal = source._choice_terminal
        self._has_choice = source._has_choice
        self._event_bytes = 0
        self._event_semantic = False
        self._dirty = False

    def mark_semantic(self) -> None:
        self._event_semantic = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def _event_changed_from(self) -> bool:
        return self._dirty


_CANONICAL_TOP = frozenset({"id", "created", "model", "system_fingerprint"})
_SKIP_TOP = frozenset({"choices", "usage", "object", "error"})
_BOM = b"\xef\xbb\xbf"
_DONE = "[DONE]"


class OpenAIStreamAccumulator:
    """Request-local SSE accumulator.

    Instantiate one per logical parse. No state is shared across instances.
    """

    def __init__(self, *, limits: UpstreamStreamLimits = DEFAULT_UPSTREAM_STREAM_LIMITS) -> None:
        self._limits = limits

        self._semantic_started = False
        self._saw_done = False
        self._after_done = False
        self._first_physical_line = True

        self._pending_data: list[bytes] = []
        self._pending_bytes = 0

        self._has_choice = False
        self._choice_terminal = False
        self._role: str | None = None
        self._finish_reason: str | None = None

        self._content_parts: list[str] = []
        self._content_present = False
        self._content_bytes = 0

        self._reasoning_parts: list[str] = []
        self._reasoning_present = False
        self._reasoning_content_parts: list[str] = []
        self._reasoning_content_present = False
        self._reasoning_bytes = 0

        self._tool_calls: dict[int, _ToolCall] = {}
        self._tool_argument_bytes = 0

        self._usage: dict | None = None
        self._usage_present = False

        self._top: dict[str, Any] = {}

        self._response_bytes = 0

    @property
    def semantic_started(self) -> bool:
        return self._semantic_started

    @property
    def saw_done(self) -> bool:
        return self._saw_done

    def feed_line(self, line: bytes) -> None:
        """Ingest one physical SSE line (without its terminator)."""
        if not isinstance(line, bytes):
            self._raise("invalid_line", "line must be bytes")
        if b"\x00" in line:
            self._raise("invalid_line", "line contains embedded NUL")

        if len(line) > self._limits.max_line_bytes:
            self._raise("limit_exceeded", "line byte limit exceeded")

        if self._first_physical_line:
            self._first_physical_line = False
            if line.startswith(_BOM):
                line = line[len(_BOM):]
        else:
            if line.startswith(_BOM):
                self._raise("invalid_utf8", "UTF-8 BOM only allowed at stream start")

        if line.endswith(b"\r"):
            line = line[:-1]

        if line == b"":
            self._dispatch_event()
            return

        if line.startswith(b":"):
            return

        if line.startswith(b"data:"):
            value = line[5:]
            if value.startswith(b" "):
                value = value[1:]
            added = len(value) + (1 if self._pending_data else 0)
            projected = self._pending_bytes + added
            if projected > self._limits.max_event_bytes:
                self._raise("limit_exceeded", "event byte limit exceeded")
            self._pending_data.append(value)
            self._pending_bytes = projected
            return

        if line.startswith((b"event:", b"id:", b"retry:")):
            return

    def finish(self) -> dict:
        """Flush any trailing event and validate terminal completeness."""
        self._dispatch_event()

        if not self._semantic_started:
            self._raise("incomplete_response", "no semantic output before stream end")
        if not self._has_choice:
            self._raise("incomplete_response", "no choice reconstructed")
        if self._finish_reason is None:
            self._raise("incomplete_response", "no finish reason before stream end")
        for tc in self._tool_calls.values():
            if not tc.call_id:
                self._raise("incomplete_tool_call", "tool call missing id")
            if tc.call_type != "function":
                self._raise("incomplete_tool_call", "tool call missing type function")
            if not tc.name_parts or not "".join(tc.name_parts):
                self._raise("incomplete_tool_call", "tool call missing name")
            if not tc.arguments_present:
                self._raise("incomplete_tool_call", "tool call missing arguments")
        return self._build()

    def _raise(self, code: str, message: str) -> None:
        if not _SANITIZED_CODE.match(code):
            raise AssertionError(f"unsanitized error code: {code!r}")
        if code == "limit_exceeded":
            cls: type[UpstreamStreamError] = UpstreamStreamLimitError
        else:
            cls = UpstreamStreamProtocolError
        raise cls(message, code=code, semantic_started=self._semantic_started)

    def _dispatch_event(self) -> None:
        if not self._pending_data:
            return
        parts = self._pending_data
        self._pending_data = []
        self._pending_bytes = 0

        assembled_bytes = b"\n".join(parts)
        try:
            assembled = assembled_bytes.decode("utf-8")
        except UnicodeDecodeError:
            self._raise("invalid_utf8", "event data is not valid UTF-8")

        stripped = assembled.strip()

        if self._after_done:
            if stripped == _DONE:
                self._on_done()
            else:
                self._raise("malformed_event", "data event after [DONE]")
            return

        if stripped == "":
            return

        if stripped == _DONE:
            self._on_done()
            return

        try:
            payload = json.loads(assembled)
        except json.JSONDecodeError:
            self._raise("malformed_json", "event data is not valid JSON")
        if not isinstance(payload, dict):
            self._raise("malformed_json", "event data must be a JSON object")

        self._on_json(payload, assembled_bytes)

    def _on_done(self) -> None:
        if self._saw_done:
            self._raise("malformed_event", "duplicate [DONE]")
        self._saw_done = True
        self._after_done = True

    def _on_json(self, payload: dict, assembled_bytes: bytes) -> None:
        if self._after_done:
            self._raise("malformed_event", "data event after [DONE]")
        if "error" in payload:
            self._raise("malformed_event", "top-level error event")

        staged = self._clone_for_event()
        staged._event_bytes = len(assembled_bytes)

        self._ingest_top_level_to_staged(payload, staged)
        self._ingest_usage_to_staged(payload.get("usage"), staged)
        self._ingest_choices_to_staged(payload.get("choices"), staged)

        if not staged._event_semantic:
            return

        projected = self._response_bytes + staged._event_bytes
        if projected > self._limits.max_response_bytes:
            self._raise("limit_exceeded", "response byte limit exceeded")

        self._commit_staged_event(staged, projected_response_bytes=projected)

    def _clone_for_event(self) -> _StagedEvent:
        return _StagedEvent(self)

    def _commit_staged_event(
        self,
        staged: _StagedEvent,
        *,
        projected_response_bytes: int,
    ) -> None:
        self._semantic_started = True

        if staged._event_changed_from():
            self._tool_calls = staged._tool_calls
            self._tool_argument_bytes = staged._tool_argument_bytes
            self._usage = staged._usage
            self._usage_present = staged._usage_present
            self._top = staged._top
            self._content_parts = staged._content_parts
            self._content_present = staged._content_present
            self._reasoning_content_parts = staged._reasoning_content_parts
            self._reasoning_content_present = staged._reasoning_content_present
            self._reasoning_parts = staged._reasoning_parts
            self._reasoning_present = staged._reasoning_present
            self._content_bytes = staged._content_bytes
            self._reasoning_bytes = staged._reasoning_bytes
            self._role = staged._role
            self._finish_reason = staged._finish_reason
            self._choice_terminal = staged._choice_terminal
            self._has_choice = staged._has_choice
        self._response_bytes = projected_response_bytes

    def _validate_unicode(self, value: Any) -> None:
        try:
            json.dumps(value, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            self._raise("invalid_unicode", "value is not valid Unicode")

    def _ingest_top_level_to_staged(self, payload: dict, staged: _StagedEvent) -> None:
        for key in _CANONICAL_TOP:
            if key in payload:
                self._ingest_top_field_to_staged(key, payload[key], staged)
        for key, value in payload.items():
            if key in _SKIP_TOP or key in _CANONICAL_TOP:
                continue
            self._ingest_top_field_to_staged(key, value, staged)

    def _ingest_top_field_to_staged(self, key: str, value: Any, staged: _StagedEvent) -> None:
        if value is None:
            return
        self._validate_unicode(value)
        if key in staged._top:
            if staged._top[key] != value:
                # `created` is a wall-clock-second Unix timestamp that some
                # upstreams (e.g. GLM via ollama.com) re-emit per chunk with
                # a fresh value. Strict equality would falsely flag a valid
                # completion as conflicting_metadata → 502. First-seen value
                # wins (self._build emits self._top[key], so whichever value
                # arrived first is preserved). Other canonical fields
                # (`id`/`model`/`system_fingerprint`) AND arbitrary unknown
                # top-level keys still raise on conflict — that guard is the
                # spliced-completion detector for those fields and must remain.
                if key == "created":
                    staged.mark_semantic()
                    return
                self._raise("conflicting_metadata", "conflicting top-level metadata")
            staged.mark_semantic()
            return
        staged.mark_semantic()
        staged._top[key] = value
        staged.mark_dirty()

    def _ingest_usage_to_staged(self, usage: Any, staged: _StagedEvent) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            self._raise("malformed_chunk", "usage must be an object or null")
        self._validate_unicode(usage)
        if staged._usage_present:
            if staged._usage != usage:
                self._raise("conflicting_metadata", "conflicting usage")
            staged.mark_semantic()
            return
        staged.mark_semantic()
        staged._usage = usage
        staged._usage_present = True
        staged.mark_dirty()

    def _ingest_choices_to_staged(self, choices: Any, staged: _StagedEvent) -> None:
        if choices is None:
            return
        if not isinstance(choices, list):
            self._raise("malformed_chunk", "choices must be a list")
        if len(choices) == 0:
            return
        if len(choices) > 1:
            self._raise("unsupported_choice", "multiple choices are not supported")
        choice = choices[0]
        if not isinstance(choice, dict):
            self._raise("malformed_chunk", "choice must be an object")
        index = choice.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index != 0:
            self._raise("unsupported_choice", "only choice index 0 is supported")
        staged._has_choice = True

        delta = choice.get("delta")
        if delta is not None:
            if not isinstance(delta, dict):
                self._raise("malformed_chunk", "delta must be an object")
            self._ingest_delta_to_staged(delta, staged)

        finish = choice.get("finish_reason")
        if finish is not None:
            self._ingest_finish_reason_to_staged(finish, staged)

    def _reject_if_terminal_for_staged(self, kind: str, staged: _StagedEvent) -> None:
        if staged._choice_terminal:
            self._raise("post_terminal_delta", f"{kind} after finish reason")

    def _ingest_delta_to_staged(self, delta: dict, staged: _StagedEvent) -> None:
        if "role" in delta:
            role = delta["role"]
            if role is None:
                pass
            elif isinstance(role, str):
                self._validate_unicode(role)
                if role != "assistant":
                    self._raise("malformed_chunk", "role must be assistant")
                if staged._choice_terminal:
                    # Benign post-terminal role: MiniMax repeats
                    # role:"assistant" in its terminal and duplicate-terminal
                    # chunks. A non-conflicting role carries no new information,
                    # so treat it as a no-op rather than a protocol violation.
                    staged.mark_semantic()
                elif staged._role is None:
                    staged.mark_semantic()
                    staged._role = role
                    staged.mark_dirty()
                elif staged._role != role:
                    self._raise("malformed_chunk", "conflicting role")
                else:
                    staged.mark_semantic()
            else:
                self._raise("malformed_chunk", "role must be a string or null")

        if "content" in delta:
            content = delta["content"]
            if content is None:
                pass
            elif isinstance(content, str):
                self._validate_unicode(content)
                if staged._choice_terminal:
                    # Benign post-terminal empty content: MiniMax sends
                    # content:"" in its terminal / duplicate-terminal chunks.
                    # An empty string adds no output and is a no-op; a NON-empty
                    # content fragment after finish is a real protocol violation.
                    if content != "":
                        self._raise("post_terminal_delta", "content after finish reason")
                    staged.mark_semantic()
                else:
                    raw_len = len(content.encode("utf-8"))
                    projected = staged._content_bytes + raw_len
                    if projected > self._limits.max_content_bytes:
                        self._raise("limit_exceeded", "content byte limit exceeded")
                    staged.mark_semantic()
                    staged._content_parts.append(content)
                    staged._content_present = True
                    staged._content_bytes = projected
                    staged.mark_dirty()
            else:
                self._raise("malformed_chunk", "content must be a string or null")

        if "reasoning_content" in delta:
            self._reject_if_terminal_for_staged("reasoning_content", staged)
            value = delta["reasoning_content"]
            if value is None:
                pass
            elif isinstance(value, str):
                self._validate_unicode(value)
                raw_len = len(value.encode("utf-8"))
                projected = staged._reasoning_bytes + raw_len
                if projected > self._limits.max_reasoning_bytes:
                    self._raise("limit_exceeded", "reasoning byte limit exceeded")
                staged.mark_semantic()
                staged._reasoning_content_parts.append(value)
                staged._reasoning_content_present = True
                staged._reasoning_bytes = projected
                staged.mark_dirty()
            else:
                self._raise("malformed_chunk", "reasoning_content must be a string or null")

        if "reasoning" in delta:
            self._reject_if_terminal_for_staged("reasoning", staged)
            value = delta["reasoning"]
            if value is None:
                pass
            elif isinstance(value, str):
                self._validate_unicode(value)
                raw_len = len(value.encode("utf-8"))
                projected = staged._reasoning_bytes + raw_len
                if projected > self._limits.max_reasoning_bytes:
                    self._raise("limit_exceeded", "reasoning byte limit exceeded")
                staged.mark_semantic()
                staged._reasoning_parts.append(value)
                staged._reasoning_present = True
                staged._reasoning_bytes = projected
                staged.mark_dirty()
            else:
                self._raise("malformed_chunk", "reasoning must be a string or null")

        if "tool_calls" in delta:
            self._reject_if_terminal_for_staged("tool_calls", staged)
            tool_calls = delta["tool_calls"]
            if not isinstance(tool_calls, list):
                self._raise("malformed_chunk", "tool_calls must be a list")
            for fragment in tool_calls:
                self._ingest_tool_call_fragment_to_staged(fragment, staged)

    def _ingest_finish_reason_to_staged(self, finish: Any, staged: _StagedEvent) -> None:
        if not isinstance(finish, str):
            self._raise("malformed_chunk", "finish_reason must be a string")
        self._validate_unicode(finish)
        if staged._finish_reason is None:
            staged.mark_semantic()
            staged._finish_reason = finish
            staged._choice_terminal = True
            staged.mark_dirty()
        elif staged._finish_reason != finish:
            self._raise("malformed_chunk", "conflicting finish_reason")
        else:
            staged.mark_semantic()
            staged._choice_terminal = True

    def _ingest_tool_call_fragment_to_staged(self, fragment: Any, staged: _StagedEvent) -> None:
        if not isinstance(fragment, dict):
            self._raise("malformed_chunk", "tool_call fragment must be an object")
        index = fragment.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            self._raise("malformed_chunk", "tool_call index must be a non-negative integer")

        tc = staged._tool_calls.get(index)
        is_new_call = tc is None
        if is_new_call:
            if len(staged._tool_calls) >= self._limits.max_tool_calls:
                self._raise("limit_exceeded", "too many distinct tool calls")

        if "id" in fragment:
            value = fragment["id"]
            if value is not None:
                if not isinstance(value, str):
                    self._raise("malformed_chunk", "tool_call id must be a string or null")
                self._validate_unicode(value)

        if "type" in fragment:
            value = fragment["type"]
            if value is not None and value != "function":
                self._raise("malformed_chunk", "tool_call type must be function")

        function = fragment.get("function")
        if function is not None:
            if not isinstance(function, dict):
                self._raise("malformed_chunk", "tool_call function must be an object")
            if "name" in function:
                name = function["name"]
                if name is not None:
                    if not isinstance(name, str):
                        self._raise("malformed_chunk", "function name must be a string or null")
                    self._validate_unicode(name)
            if "arguments" in function:
                arguments = function["arguments"]
                if arguments is not None:
                    if not isinstance(arguments, str):
                        self._raise("malformed_chunk", "function arguments must be a string or null")
                    self._validate_unicode(arguments)
                    raw_len = len(arguments.encode("utf-8"))
                    projected_args = staged._tool_argument_bytes + raw_len
                    if projected_args > self._limits.max_tool_argument_bytes:
                        self._raise("limit_exceeded", "tool argument byte limit exceeded")

        staged.mark_semantic()

        if is_new_call:
            tc = _ToolCall()
            staged._tool_calls[index] = tc
            staged.mark_dirty()

        if "id" in fragment:
            value = fragment["id"]
            if value is not None:
                if tc.call_id is None:
                    tc.call_id = value
                    staged.mark_dirty()
                elif tc.call_id != value:
                    self._raise("malformed_chunk", "conflicting tool_call id")

        if "type" in fragment:
            value = fragment["type"]
            if value is not None:
                if tc.call_type is None:
                    tc.call_type = value
                    staged.mark_dirty()
                elif tc.call_type != value:
                    self._raise("malformed_chunk", "conflicting tool_call type")

        if function is not None:
            if "name" in function:
                name = function["name"]
                if name is not None:
                    tc.name_parts.append(name)
                    staged.mark_dirty()
            if "arguments" in function:
                arguments = function["arguments"]
                if arguments is not None:
                    tc.arguments_present = True
                    tc.argument_parts.append(arguments)
                    staged._tool_argument_bytes = staged._tool_argument_bytes + len(arguments.encode("utf-8"))
                    staged.mark_dirty()

    def _build(self) -> dict:
        result: dict[str, Any] = {}

        for key in ("id", "created", "model", "system_fingerprint"):
            if key in self._top:
                result[key] = self._top[key]

        result["object"] = "chat.completion"

        for key, value in self._top.items():
            if key not in _CANONICAL_TOP:
                result[key] = value

        if self._usage_present:
            result["usage"] = self._usage

        message: dict[str, Any] = {}
        message["role"] = self._role if self._role is not None else "assistant"

        if self._content_present:
            message["content"] = "".join(self._content_parts)
        else:
            message["content"] = None

        if self._reasoning_content_present:
            message["reasoning_content"] = "".join(self._reasoning_content_parts)
        if self._reasoning_present:
            message["reasoning"] = "".join(self._reasoning_parts)
        if self._reasoning_present and not self._reasoning_content_present:
            message["reasoning_content"] = "".join(self._reasoning_parts)

        if self._tool_calls:
            tool_calls_out = []
            for index in sorted(self._tool_calls.keys()):
                tc = self._tool_calls[index]
                tool_calls_out.append({
                    "id": tc.call_id,
                    "type": tc.call_type,
                    "function": {
                        "name": "".join(tc.name_parts),
                        "arguments": "".join(tc.argument_parts),
                    },
                })
            message["tool_calls"] = tool_calls_out

        result["choices"] = [
            {
                "index": 0,
                "message": message,
                "finish_reason": self._finish_reason,
            }
        ]
        return result


def reconstruct_openai_sse(
    lines: Iterable[bytes],
    *,
    limits: UpstreamStreamLimits = DEFAULT_UPSTREAM_STREAM_LIMITS,
) -> dict:
    """Reconstruct one complete OpenAI Chat Completion dictionary from SSE lines.

    Each call instantiates a fresh accumulator. No state is shared across calls.
    """
    accumulator = OpenAIStreamAccumulator(limits=limits)
    for line in lines:
        accumulator.feed_line(line)
    return accumulator.finish()
