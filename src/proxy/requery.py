import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from proxy.types import RequeryResult, BackendError
from proxy.http_client import call_upstream
from proxy.utils import safe_content as _safe_content

logger = logging.getLogger(__name__)


_CLOSURE_TOKEN_RE = re.compile(r"^(COMPLETE|PARTIAL|FAILED)$")

_CLOSURE_TERMINATION_MODEL_COMPLETED = "model_completed"


@dataclass
class ClosureFinalizeResult:
    success: bool
    content: Optional[str] = None
    failure_reason: Optional[str] = None


def closure_finalize(
    working_messages: list,
    llama_url: str,
    body: dict,
    llama_body: dict,
    model_cfg,
    auth_headers: Optional[dict],
    ledger: str,
    termination_reason: str,
    timeout_cap: float = 30.0,
    tool_loop_start: float = 0.0,
    max_total_time: float = 900.0,
    parse_tools_fn=None,
) -> ClosureFinalizeResult:
    if not isinstance(ledger, str):
        ledger = ""

    if termination_reason in (_CLOSURE_TERMINATION_MODEL_COMPLETED,):
        timing_clause = "All required memory operations have completed."
    else:
        timing_clause = (
            "The memory tool loop terminated before completing all "
            "operations. Do not state that all operations completed."
        )

    instruction = (
        f"{timing_clause} Termination reason: {termination_reason}. "
        "Produce ONE human-readable completion report beginning at "
        "character zero with exactly one of the tokens COMPLETE, PARTIAL, "
        "or FAILED (uppercase, as the first token), immediately followed "
        "by a single whitespace, newline, or colon character and then "
        "substantive human-readable report text. Do not invoke any tools. "
        "Do not claim success for operations that did not succeed. Do not "
        "return the raw bracket ledger. Execution ledger:\n"
        f"{ledger}"
    )

    safe_messages: list = []
    if isinstance(working_messages, list):
        for m in working_messages:
            if isinstance(m, dict):
                safe_messages.append(copy.deepcopy(m))

    safe_messages.append({"role": "user", "content": instruction})

    closure_body = {
        "model": llama_body.get("model", body.get("model", "default")),
        "messages": safe_messages,
        "stream": False,
    }
    if not isinstance(closure_body.get("messages"), list) or not closure_body["messages"]:
        return ClosureFinalizeResult(success=False, failure_reason="internal_error")

    from proxy.pipeline.provider_compat import apply_provider_compat
    try:
        closure_body = apply_provider_compat(closure_body, model_cfg)
    except Exception as e:
        logger.warning("closure_finalize: provider_compat failed: %s", e)
        return ClosureFinalizeResult(success=False, failure_reason="internal_error")

    closure_body["stream"] = False
    for _k in ("tools", "tool_choice", "parallel_tool_calls",
               "functions", "function_call"):
        closure_body.pop(_k, None)

    remaining_budget = max_total_time - (time.time() - tool_loop_start)
    effective_cap = min(float(timeout_cap) if timeout_cap else 30.0,
                        max(5.0, remaining_budget - 5.0))
    if effective_cap < 5.0:
        effective_cap = 5.0

    try:
        response = call_upstream(
            llama_url,
            closure_body,
            auth_headers or {},
            False,
            timeout_cap=effective_cap,
        )
    except BackendError:
        return ClosureFinalizeResult(success=False, failure_reason="no_response")
    except Exception:
        return ClosureFinalizeResult(success=False, failure_reason="no_response")

    if not isinstance(response, dict):
        return ClosureFinalizeResult(success=False, failure_reason="non_dict")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ClosureFinalizeResult(success=False, failure_reason="no_choices")
    first = choices[0]
    if not isinstance(first, dict):
        return ClosureFinalizeResult(success=False, failure_reason="no_choices")
    fr = first.get("finish_reason")
    if fr in ("tool_calls", "function_call"):
        return ClosureFinalizeResult(success=False, failure_reason="finish_reason_tool_calls")
    message = first.get("message")
    if not isinstance(message, dict):
        return ClosureFinalizeResult(success=False, failure_reason="null_message")

    native_tool_calls = message.get("tool_calls")
    if native_tool_calls is not None and not isinstance(native_tool_calls, list):
        return ClosureFinalizeResult(success=False, failure_reason="tool_calls_native")
    if isinstance(native_tool_calls, list) and native_tool_calls:
        return ClosureFinalizeResult(success=False, failure_reason="tool_calls_native")

    if "function_call" in message and message["function_call"] is not None:
        return ClosureFinalizeResult(success=False, failure_reason="tool_calls_native")

    content = _safe_content(message.get("content"))
    if not isinstance(content, str):
        return ClosureFinalizeResult(success=False, failure_reason="empty_content")
    if not content.strip():
        return ClosureFinalizeResult(success=False, failure_reason="whitespace_only")
    if content.startswith((" ", "\t", "\n", "\r")):
        return ClosureFinalizeResult(success=False, failure_reason="leading_whitespace")
    content_stripped = content.rstrip("\r\n") if content.endswith("\r\n") else content

    if parse_tools_fn is not None:
        try:
            parsed_calls = parse_tools_fn(content_stripped) or []
        except Exception:
            return ClosureFinalizeResult(success=False, failure_reason="tool_calls_parsed")
        if parsed_calls:
            return ClosureFinalizeResult(success=False, failure_reason="tool_calls_parsed")

    if not content_stripped.startswith(("COMPLETE", "PARTIAL", "FAILED")):
        return ClosureFinalizeResult(success=False, failure_reason="missing_prefix")
    if len(content_stripped) > 8 and content_stripped[8] == ":":
        token = "COMPLETE"
        first_sep_idx = 8
    elif len(content_stripped) > 7 and content_stripped[7] == ":":
        token = "PARTIAL"
        first_sep_idx = 7
    elif len(content_stripped) > 6 and content_stripped[6] == ":":
        token = "FAILED"
        first_sep_idx = 6
    else:
        token = None
        for length in (8, 7, 6):
            if len(content_stripped) > length and content_stripped[length] in (
                " ", "\t", "\n", "\r", ":"
            ):
                candidate = content_stripped[:length]
                if _CLOSURE_TOKEN_RE.match(candidate):
                    token = candidate
                    first_sep_idx = length
                    break
    if token is None:
        return ClosureFinalizeResult(success=False, failure_reason="missing_prefix")
    sep = content_stripped[first_sep_idx:first_sep_idx + 1]
    if sep not in (" ", "\t", "\n", "\r", ":"):
        return ClosureFinalizeResult(success=False, failure_reason="missing_prefix")
    after_sep = content_stripped[first_sep_idx + 1:]
    if not after_sep.strip():
        return ClosureFinalizeResult(success=False, failure_reason="missing_prefix")
    if not any(ch.isalnum() for ch in after_sep):
        return ClosureFinalizeResult(success=False, failure_reason="missing_prefix")

    return ClosureFinalizeResult(success=True, content=content_stripped)


def resolve_requery_tools(ctx, llama_body):
    if ctx is not None and hasattr(ctx, '_tools_snapshot') and ctx._tools_snapshot:
        return ctx._tools_snapshot
    return llama_body.get("tools", [])[:]


def requery_llm(
    working_messages: list,
    llama_body: dict,
    body: dict,
    model_cfg,
    llama_url: str,
    auth_headers: Optional[dict],
    http_session=None,
    tool_loop_start: float = 0.0,
    max_total_time: float = 90.0,
    passthrough_calls: list = None,
    parse_tools_fn=None,
    make_tool_calls_fn=None,
    _raw_tool_id_map: dict = None,
    ctx=None,
) -> RequeryResult:
    """Re-query the LLM after memory tool execution."""
    import json
    import time
    from proxy.pipeline.provider_compat import apply_provider_compat
    from proxy.tool_pipeline.tool_registry import registry

    requery_tools = resolve_requery_tools(ctx, llama_body)

    requery_body = {
        "model": llama_body.get("model", body.get("model", "default")),
        "messages": working_messages,
        "tools": requery_tools,
    }

    requery_data = None
    requery_failed = False
    new_tool_calls = None
    requery_content = ""
    should_continue = False

    try:
        logger.info(
            "[TOOL LOOP] Re-querying LLM with %d messages",
            len(working_messages),
        )
        requery_body = apply_provider_compat(requery_body, model_cfg)
        requery_body["stream"] = False

        final_msgs = []
        for msg in requery_body.get("messages", []):
            if msg.get("role") == "assistant":
                has_content = bool(msg.get("content"))
                has_reasoning = bool(msg.get("reasoning_content") or msg.get("reasoning"))
                has_tool_calls = bool(msg.get("tool_calls"))
                if not has_content and not has_tool_calls and not has_reasoning:
                    continue
            final_msgs.append(msg)
        requery_body["messages"] = final_msgs

        _final_msgs = requery_body.get("messages", [])
        _merged_msgs = []
        _merge_count = 0
        for _msg in _final_msgs:
            if (
                _merged_msgs
                and _merged_msgs[-1].get("role") == "user"
                and _msg.get("role") == "user"
            ):
                _merged_msgs[-1]["content"] = (
                    _safe_content(_merged_msgs[-1].get("content"))
                    + "\n\n"
                    + _safe_content(_msg.get("content"))
                )
                _merge_count += 1
            else:
                _merged_msgs.append(_msg)
        if _merge_count:
            logger.info(
                "[TOOL LOOP] Merged %d consecutive user message pair(s) after provider compat",
                _merge_count,
            )
        requery_body["messages"] = _merged_msgs

        if not requery_body.get("messages"):
            logger.error(
                "[TOOL LOOP] Re-query body has no messages after provider compat and cleanup"
            )
            return RequeryResult(
                requery_failed=True,
                requery_data=None,
                working_messages=working_messages,
                _raw_tool_id_map=_raw_tool_id_map,
                new_tool_calls=None,
                requery_content="",
                should_continue=False,
            )

        remaining_budget = max_total_time - (time.time() - tool_loop_start)
        MIN_REQUERY_TIMEOUT = 5
        if remaining_budget < MIN_REQUERY_TIMEOUT + 5:
            logger.warning(
                "[TOOL LOOP] Budget exhausted (%.1fs remaining), stopping",
                remaining_budget,
            )
            return RequeryResult(
                requery_failed=True,
                requery_data=None,
                working_messages=working_messages,
                _raw_tool_id_map=_raw_tool_id_map,
                new_tool_calls=None,
                requery_content="",
                should_continue=False,
            )
        requery_data = call_upstream(
            llama_url, requery_body, auth_headers or {}, False,
            timeout_cap=remaining_budget - 5,
        )

        if not isinstance(requery_data, dict):
            logger.error(
                "[TOOL LOOP] Re-query returned non-dict JSON (%s): %.200s",
                type(requery_data).__name__,
                json.dumps(requery_data, default=str)[:200],
            )
            return RequeryResult(
                requery_failed=True,
                requery_data=None,
                working_messages=working_messages,
                _raw_tool_id_map=_raw_tool_id_map,
                new_tool_calls=None,
                requery_content="",
                should_continue=False,
            )

        if not requery_data.get("choices"):
            logger.warning(
                "[TOOL LOOP] Re-query returned no choices (keys: %s). Response may be empty or malformed.",
                list(requery_data.keys()),
            )
            return RequeryResult(
                requery_failed=True,
                requery_data=None,
                working_messages=working_messages,
                _raw_tool_id_map=_raw_tool_id_map,
                new_tool_calls=None,
                requery_content="",
                should_continue=False,
            )

        requery_message = requery_data["choices"][0].get("message")
        if requery_message is None:
            logger.warning(
                "[TOOL LOOP] Re-query returned null/missing message (choices[0]=%s)",
                requery_data["choices"][0],
            )
            return RequeryResult(
                requery_failed=True,
                requery_data=requery_data,
                working_messages=working_messages,
                _raw_tool_id_map=_raw_tool_id_map,
                new_tool_calls=None,
                requery_content="",
                should_continue=False,
            )
    except BackendError as e:
        _status = e.status_code or "?"
        _body_str = json.dumps(e.error_body, default=str) if e.error_body else ""
        logger.error(
            "[TOOL LOOP] Re-query failed (HTTP %s): %s | body=%s",
            _status,
            e,
            _body_str[:1000],
        )
        return RequeryResult(
            requery_failed=True,
            requery_data=None,
            working_messages=working_messages,
            _raw_tool_id_map=_raw_tool_id_map,
            new_tool_calls=None,
            requery_content="",
            should_continue=False,
        )
    except Exception:
        logger.error("[TOOL LOOP] Re-query failed", exc_info=True)
        return RequeryResult(
            requery_failed=True,
            requery_data=None,
            working_messages=working_messages,
            _raw_tool_id_map=_raw_tool_id_map,
            new_tool_calls=None,
            requery_content="",
            should_continue=False,
        )

    requery_content = _safe_content(requery_message.get("content"))
    _upstream_finish_reason = requery_data["choices"][0].get("finish_reason")

    if not requery_content:
        reasoning = requery_message.get("reasoning_content", "")
        if reasoning:
            requery_content = reasoning

    logger.info(
        "[TOOL LOOP] Re-query response: content=%d chars, tool_calls=%s",
        len(requery_content),
        bool(requery_message.get("tool_calls")),
    )

    native = requery_message.get("tool_calls")
    original_ids = None
    if native is not None and not isinstance(native, list):
        native = None
    if isinstance(native, list) and native:
        new_tool_calls = []
        original_ids = []
        for tc in native:
            try:
                func = tc["function"]
                arguments = func["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                from proxy.tool_parsing import canonicalize_tool_name
                new_tool_calls.append(
                    {"name": canonicalize_tool_name(func["name"]), "arguments": arguments}
                )
                original_ids.append(tc.get("id"))
            except (KeyError, TypeError, json.JSONDecodeError):
                logger.warning(
                    "Failed to parse native tool call in re-query"
                )
    else:
        try:
            new_tool_calls = (
                parse_tools_fn(requery_content) if parse_tools_fn else []
            ) or []
        except Exception:
            new_tool_calls = []

    if new_tool_calls:
        _new_assistant_msg = {
            "role": "assistant",
            "content": requery_content if requery_content else None,
        }
        if make_tool_calls_fn:
            _formatted = make_tool_calls_fn(new_tool_calls, existing_ids=original_ids)
            _new_assistant_msg["tool_calls"] = _formatted
            if _raw_tool_id_map is not None:
                _raw_tool_id_map.clear()
            for raw, fmt in zip(new_tool_calls, _formatted):
                raw_args = json.dumps(
                    raw.get("arguments", {}), sort_keys=True
                )
                _raw_tool_id_map[(raw.get("name"), raw_args)] = fmt.get(
                    "id"
                )
        working_messages.append(_new_assistant_msg)

        remaining_memory = [
            tc
            for tc in new_tool_calls
            if tc.get("name") in registry.get_memory_tool_names()
        ]
        remaining_passthrough = [
            tc
            for tc in new_tool_calls
            if tc.get("name") not in registry.get_memory_tool_names()
        ]

        if remaining_memory:
            should_continue = True
        elif remaining_passthrough:
            pass
        else:
            if not requery_content:
                logger.warning(
                    "[TOOL LOOP] Re-query returned empty content and no tool calls"
                )
                requery_content = "I couldn't find any information about that in my memory."

    return RequeryResult(
        requery_failed=requery_failed,
        requery_data=requery_data,
        working_messages=working_messages,
        _raw_tool_id_map=_raw_tool_id_map,
        new_tool_calls=new_tool_calls,
        requery_content=requery_content,
        should_continue=should_continue,
        upstream_finish_reason=_upstream_finish_reason,
    )


def requery_tools(ctx):
    if ctx is not None and hasattr(ctx, '_tools_snapshot') and ctx._tools_snapshot:
        return ctx._tools_snapshot
    return []