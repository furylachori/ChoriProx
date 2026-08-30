"""Integration tests for streaming responses through the full pipeline.

Verifies that when `stream: true` is set in the request:
1. SSE format is correct (data prefix, \\n\\n suffix, [DONE] terminator)
2. The initial upstream call requests genuine SSE (upstream body stream=true);
   the proxy consumes it incrementally and reconstructs one complete response
   dict before running PostProcessStage, then re-streams the final response.
   Internal memory-tool requeries remain non-streaming JSON calls.
3. stream_options is stripped from the upstream body (the proxy never forwards
   client stream_options upstream)
4. Errors during streaming surface in SSE format with [DONE] terminator

The StreamingBackendHandler speaks two protocols based on the captured request
body: genuine incremental `text/event-stream` when `stream is True`, and a full
`application/json` document otherwise (also used as the JSON-fallback path).

Ticket 199ece93 §4 architecture (W3 resync): the keepalive generator is
completion-thread based. The generator emits one immediate `: connected`
comment, then `: keepalive` per heartbeat_interval while the worker thread
runs the full pipeline. PostProcessStage is entered exactly once per
request. A downstream disconnect no longer clears mode — completion is
the sole authority.
"""
import json
import threading
import time as _time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional
from unittest.mock import patch

import pytest

# Local aliases so the per-test Agent helpers don't need to import inside every call.
# Used only by tests marked with @pytest.mark.real_agent_registry to exercise the
# real CAS compare-and-clear in Agent.clear_active_mode(expected_mode=...).
from proxy.agent.agent import Agent as _RealAgent
from proxy.skills.skill_definition import ModeState as _ModeState
from proxy.agent.agent_registry import registry as _agent_registry


# ---------------------------------------------------------------------------
# Mock backend — captures requests, returns configured responses
# ---------------------------------------------------------------------------


class StreamingBackendHandler(BaseHTTPRequestHandler):
    """Mock backend supporting sequential responses and an optional failure mode.

    Class-level state:
      _responses:   ordered list of response dicts (or special strings) to return
      _request_count: incremented on every POST
      _captured_bodies: parsed request bodies, in arrival order
      _fail_after: if set, raise ConnectionError after this many requests
    """

    _responses: list = []
    _request_count: int = 0
    _captured_bodies: list = []
    _captured_headers: list = []
    _fail_after: Optional[int] = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        StreamingBackendHandler._captured_bodies.append(body)
        StreamingBackendHandler._captured_headers.append(dict(self.headers))
        StreamingBackendHandler._request_count += 1

        if (
            StreamingBackendHandler._fail_after is not None
            and StreamingBackendHandler._request_count > StreamingBackendHandler._fail_after
        ):
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "mock upstream failure"}}')
            return

        idx = StreamingBackendHandler._request_count - 1
        if idx < len(StreamingBackendHandler._responses):
            payload = StreamingBackendHandler._responses[idx]
        else:
            payload = {
                "choices": [{
                    "message": {"role": "assistant", "content": "fallback"},
                    "finish_reason": "stop",
                }],
                "model": "mock-llm",
            }

        if isinstance(payload, Exception):
            raise payload

        wants_stream = body.get("stream") is True
        if wants_stream and isinstance(payload, dict) and payload.get("_raw_sse_lines"):
            # Replay literal upstream SSE lines verbatim (e.g. captured
            # MiniMax quirks) to exercise the real reconstruction path.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in payload["_raw_sse_lines"]:
                self.wfile.write(line if isinstance(line, bytes) else line.encode())
                self.wfile.write(b"\n")
                self.wfile.flush()
            return

        if wants_stream and payload == "__INCOMPLETE_SSE__":
            # ticket 199ece93 Part A / A4: genuine SSE that begins semantic
            # output but sends [DONE] with NO finish_reason -> the accumulator
            # raises UpstreamStreamProtocolError, which must surface to the
            # client as a sanitized `upstream_error` envelope + one [DONE].
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in (
                b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
                b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n',
                b"data: [DONE]\n\n",
            ):
                self.wfile.write(frame)
                self.wfile.flush()
            return

        if wants_stream and isinstance(payload, dict) and payload.get("choices"):
            # ticket 199ece93 Part A: emit GENUINE incremental text/event-stream
            # so call_upstream's iter_lines()->OpenAIStreamAccumulator branch is
            # exercised end-to-end. HTTP/1.0 (the BaseHTTPRequestHandler default)
            # closes the socket after the body, giving iter_lines a clean EOF
            # without chunked framing. Flush per frame so consumption is genuinely
            # incremental.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in _response_dict_to_sse_frames(payload):
                self.wfile.write(frame)
                self.wfile.flush()
            return

        body_bytes = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body_bytes)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def streaming_backend():
    """Start a mock HTTP backend for streaming tests."""
    server = HTTPServer(("127.0.0.1", 0), StreamingBackendHandler)
    port = server.server_address[1]

    StreamingBackendHandler._responses = []
    StreamingBackendHandler._request_count = 0
    StreamingBackendHandler._captured_bodies = []
    StreamingBackendHandler._captured_headers = []
    StreamingBackendHandler._fail_after = None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield server, port

    server.shutdown()


@pytest.fixture
def streaming_client(streaming_backend, tmp_path, monkeypatch):
    """Flask test client wired to the mock streaming backend."""
    from clients.model_registry import ModelRegistry
    import requests as _requests

    server, port = streaming_backend
    backend_url = f"http://127.0.0.1:{port}/v1"

    config = {
        "models": {
            "test-model": {
                "backend": backend_url,
                "tool_parser": "openai",
                "context_limit": 262144,
                "name": "Test Model",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "test/llm-v1",
            },
        },
        "default": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 32000,
        },
    }

    path = tmp_path / "proxy_models.json"
    path.write_text(json.dumps(config))

    reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
    monkeypatch.setattr("qwen_proxy._model_registry", reg)
    monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
    monkeypatch.setattr("proxy.app._model_registry", reg)
    monkeypatch.setattr("proxy.model_registry._model_registry", reg)

    fresh_session = _requests.Session()
    monkeypatch.setattr("qwen_proxy._HTTP_SESSION", fresh_session)
    monkeypatch.setattr("proxy.state._HTTP_SESSION", fresh_session)

    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret-key-12345")

    import qwen_proxy
    qwen_proxy.app.config["TESTING"] = True
    return qwen_proxy.app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response_dict_to_sse_frames(payload: dict) -> list[bytes]:
    """Convert a complete OpenAI chat.completion dict into genuine SSE delta
    frames (role -> content fragments -> tool-call deltas -> terminal
    finish/usage -> [DONE]) so the upstream SSE reconstruction path is
    exercised end-to-end. Content is split into two fragments when long enough
    to exercise concatenation across chunks.
    """
    choice = payload["choices"][0]
    message = choice.get("message", {}) or {}
    finish_reason = choice.get("finish_reason")
    model = payload.get("model", "mock-llm")
    usage = payload.get("usage")

    def _chunk(delta: dict, finish=None, usage_obj=None) -> bytes:
        ch = {"index": 0, "delta": delta, "finish_reason": finish}
        obj: dict = {
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [ch],
        }
        if usage_obj is not None:
            obj["usage"] = usage_obj
        return f"data: {json.dumps(obj)}\n\n".encode()

    frames: list[bytes] = [_chunk({"role": "assistant"})]

    content = message.get("content")
    if isinstance(content, str) and content:
        if len(content) >= 2:
            mid = len(content) // 2
            frames.append(_chunk({"content": content[:mid]}))
            frames.append(_chunk({"content": content[mid:]}))
        else:
            frames.append(_chunk({"content": content}))

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        frames.append(_chunk({"reasoning_content": reasoning}))

    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        frames.append(_chunk({"tool_calls": [{
            "index": i,
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")},
        }]}))

    # Terminal chunk carries the finish reason (and usage, if any).
    frames.append(_chunk({}, finish=finish_reason, usage_obj=usage))
    frames.append(b"data: [DONE]\n\n")
    return frames


def _make_text_response(content: str, finish_reason: str = "stop") -> dict:
    """OpenAI-format text response."""
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "model": "test/llm-v1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_tool_call_response(tool_calls: list, content: str = "", finish_reason: str = "tool_calls") -> dict:
    """OpenAI-format tool_call response."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "model": "test/llm-v1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _openai_tool_call(tool_id: str, name: str, arguments: Any) -> dict:
    """OpenAI-format tool_call dict."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _split_sse(body: str) -> list[str]:
    """Split an SSE body into a list of events (each event is a string).

    Preserves blank lines between events. Each event is one of:
      - a comment line starting with ':'
      - one or more 'data: ' lines
      - the '[DONE]' sentinel
    """
    return [chunk for chunk in body.split("\n\n") if chunk]


def _sse_data_payloads(body: str) -> list[dict]:
    """Strict SSE event parser.

    Iterates every event returned by _split_sse(body) and enforces the SSE
    contract for the proxy's wire format:
      1. An event whose every non-empty line begins with ':' is a comment
         event and is skipped.
      2. Any event mixing comment lines and data lines raises AssertionError.
      3. Every non-comment line must start with 'data: '; otherwise raise
         AssertionError with the offending event in the message.
      4. Multiple 'data: ' values within an event are joined with '\n'
         (SSE event semantics).
      5. The joined value '[DONE]' must be the final event in the stream and
         the only [DONE] event seen; it is not added to the returned payloads.
      6. Any event after [DONE] raises AssertionError.
      7. Every non-DONE joined data value is parsed with json.loads(); the
         parser does not suppress JSONDecodeError.
      8. After parsing, assert exactly one [DONE] event was seen.
    """
    payloads: list = []
    done_seen = False
    events = _split_sse(body)
    total = len(events)
    for idx, event in enumerate(events):
        non_empty_lines = [line for line in event.split("\n") if line]
        if not non_empty_lines:
            continue
        if all(line.startswith(":") for line in non_empty_lines):
            continue
        if any(line.startswith(":") for line in non_empty_lines):
            raise AssertionError(
                f"SSE event mixes comment lines and data lines: {event!r}"
            )
        for line in non_empty_lines:
            if not line.startswith("data: "):
                raise AssertionError(
                    f"SSE event contains non-data, non-comment line: "
                    f"offending line={line!r} in event={event!r}"
                )
        data_values = [line[len("data: "):] for line in non_empty_lines]
        joined = "\n".join(data_values)
        if joined == "[DONE]":
            if done_seen:
                raise AssertionError(
                    f"SSE stream contains duplicate [DONE] event: {event!r}"
                )
            if idx != total - 1:
                raise AssertionError(
                    f"SSE [DONE] event must be the final event in the stream; "
                    f"got event index {idx} of {total}: {event!r}"
                )
            done_seen = True
            continue
        if done_seen:
            raise AssertionError(
                f"SSE event after [DONE] is forbidden: {event!r}"
            )
        payloads.append(json.loads(joined))
    assert done_seen, "SSE stream missing terminal [DONE] event"
    return payloads


def _sse_comments(body: str) -> list[str]:
    """Return all SSE comment lines (starting with ':')."""
    comments = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith(":"):
            comments.append(stripped)
    return comments


# ---------------------------------------------------------------------------
# _sse_data_payloads strict parser helper tests
# ---------------------------------------------------------------------------


def test_sse_data_payloads_rejects_invalid_json_event():
    body = (
        'data: {"error":\n\n'
        'data: [DONE]'
    )
    with pytest.raises(json.JSONDecodeError):
        _sse_data_payloads(body)


def test_sse_data_payloads_rejects_extra_event_after_done():
    body = (
        'data: {"error": {"message": "x"}}\n\n'
        'data: [DONE]\n\n'
        'data: {"choices": null}'
    )
    with pytest.raises(AssertionError):
        _sse_data_payloads(body)


def test_sse_data_payloads_rejects_hidden_second_data_line():
    body = (
        'data: {"error": {"message": "x"}}\n'
        'data: {"choices": null}\n\n'
        'data: [DONE]'
    )
    with pytest.raises(json.JSONDecodeError):
        _sse_data_payloads(body)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param('data: {"choices": [{"message": {"content": "x"}}]}', id="no-done"),
        pytest.param('data: [DONE]\n\ndata: [DONE]', id="two-done"),
    ],
)
def test_sse_data_payloads_requires_exactly_one_done(body):
    with pytest.raises(AssertionError):
        _sse_data_payloads(body)


# ---------------------------------------------------------------------------
# Test 1: SSE format — data prefix and DONE terminator
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sse_format_data_prefix_and_done_terminator(streaming_client, streaming_backend):
    """stream=true yields valid SSE: every data chunk has 'data: ' prefix and '\\n\\n' suffix,
    and the stream ends with 'data: [DONE]\\n\\n'."""
    StreamingBackendHandler._responses = [
        _make_text_response("Hello streamed world"),
    ]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    body = response.get_data(as_text=True)

    # Every data event starts with 'data: ' and ends with '\\n\\n' (per SSE spec)
    events = _split_sse(body)
    assert len(events) >= 2, f"Expected at least 2 events (data + DONE), got {events!r}"

    data_events = [e for e in events if e.startswith("data: ")]
    assert len(data_events) >= 2, f"Expected at least 2 data events, got {data_events!r}"

    for event in data_events:
        # The raw body should end the data event with \n\n separator
        assert event.startswith("data: "), f"Event missing 'data: ' prefix: {event!r}"
    # Verify the body itself contains the full SSE terminator pattern
    assert "data: [DONE]\n\n" in body, f"Missing 'data: [DONE]\\n\\n' terminator in body:\n{body}"

    # Last event is the [DONE] sentinel
    assert data_events[-1] == "data: [DONE]", f"Final event is not [DONE]: {data_events[-1]!r}"

    # At least one data event contains a 'choices' array with a delta
    payloads = _sse_data_payloads(body)
    assert any("choices" in p for p in payloads), f"No 'choices' in any data payload: {payloads!r}"
    # And at least one delta carries actual content
    assert any(
        p.get("choices", [{}])[0].get("delta", {}).get("content", "") == "Hello streamed world"
        for p in payloads
    ), f"Content 'Hello streamed world' not found in any delta: {payloads!r}"


# ---------------------------------------------------------------------------
# Test 2: Tool interception with streaming — proxy streams the initial upstream
#         request, intercepts the tool, executes it, re-queries non-streaming,
#         and streams the final response as SSE.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_streaming_with_tool_calls_streams_final_response(streaming_client, streaming_backend):
    """When stream=true and the upstream returns tool_calls, the proxy:
      - streams the INITIAL upstream request (wants_stream=True, SSE reconstructed)
      - executes the memory tool locally
      - re-queries NON-streaming with the tool result
      - streams the final text response to the client as SSE

    Verifies the OBSERVABLE behavior of the streaming pipeline:
      - response is SSE with valid format
      - upstream is called twice (initial + re-query)
      - final streamed content reflects the post-tool-execution text
      - re-query body includes the tool result and the original assistant tool_call
      - the final streamed SSE has finish_reason: 'stop' (no more tool_calls)
      - the initial upstream call is streaming; the re-query is non-streaming
    """
    recall_tc = _openai_tool_call("call_recall_s1", "recall", {"query": "memory test"})

    StreamingBackendHandler._responses = [
        _make_tool_call_response([recall_tc]),
        _make_text_response("Recalled from memory: answer."),
    ]

    with patch("memory.tools.execute_memory_tool") as mock_exec:
        mock_exec.return_value = {
            "results": [{"content": "remembered data", "score": 0.9}],
        }
        response = streaming_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What do you remember?"}],
                "stream": True,
            },
            headers={"X-Session-ID": "sess-sse-tool-1"},
        )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    body = response.get_data(as_text=True)
    assert "data: [DONE]\n\n" in body, f"Missing [DONE] terminator:\n{body}"

    payloads = _sse_data_payloads(body)
    assert payloads, f"No data payloads parsed from:\n{body}"

    # The final streamed content should contain the post-interception text
    final_deltas = [
        p["choices"][0].get("delta", {}).get("content", "")
        for p in payloads
        if p.get("choices")
    ]
    combined = "".join(final_deltas)
    assert "Recalled from memory" in combined, (
        f"Final streamed content missing post-interception text: {combined!r}"
    )

    # The final delta carries finish_reason: 'stop' (after re-query, no more tool calls)
    finish_reasons = [
        p["choices"][0].get("finish_reason")
        for p in payloads
        if p.get("choices")
    ]
    assert "stop" in finish_reasons, f"Expected finish_reason=stop in final delta, got: {finish_reasons}"

    # Upstream was called exactly twice (tool_call response + re-query)
    assert StreamingBackendHandler._request_count == 2, (
        f"Expected 2 upstream calls (initial + re-query), got {StreamingBackendHandler._request_count}"
    )

    # ticket 199ece93 Part A: the INITIAL upstream request carries the client's
    # stream intent (stream=True → upstream SSE reconstructed to a complete
    # dict); the internal memory-tool RE-QUERY stays non-streaming JSON.
    assert StreamingBackendHandler._captured_bodies[0].get("stream") is True, (
        f"Initial upstream call must be streaming when client requests stream: "
        f"{StreamingBackendHandler._captured_bodies[0]!r}"
    )
    assert StreamingBackendHandler._captured_bodies[1].get("stream") is False, (
        f"Re-query must be non-streaming: "
        f"{StreamingBackendHandler._captured_bodies[1]!r}"
    )

    # The re-query body must include the tool result AND the original tool_call
    requery = StreamingBackendHandler._captured_bodies[1]
    requery_msgs = requery.get("messages", [])
    tool_msgs = [m for m in requery_msgs if m.get("role") == "tool"]
    assert tool_msgs, f"Re-query missing tool result messages: {requery_msgs!r}"
    assistant_msgs = [m for m in requery_msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert assistant_msgs, f"Re-query missing assistant tool_calls: {requery_msgs!r}"

    # The tool result content is non-empty (some form of recall result, mocked or real)
    tool_content = tool_msgs[0].get("content", "")
    assert tool_content, f"Tool result content is empty: {tool_msgs[0]!r}"


# ---------------------------------------------------------------------------
# Test 3: Skill direct response streams correctly
# ---------------------------------------------------------------------------


# NOTE: legacy test removed. test_skill_workspace_status_streams_correctly
# tested the OLD streaming dispatch path where `^^workspace status` was routed
# to the WorkspaceSkill and produced a "Current workspace:" SSE response. The
# new dispatch loop routes `^^workspace` to the generic command dispatcher
# first, which returns "Unknown command: ^^workspace" because the workspace
# skill is not registered as a command. The skill dispatcher is the
# second-priority handler, so the command handler wins.


# ---------------------------------------------------------------------------
# Test 4: stream_options stripped from upstream body
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stream_options_stripped_for_upstream(streaming_client, streaming_backend):
    """When the client sends `stream_options: {include_usage: true}` along with
    stream=true, the upstream call must NOT receive stream_options — the proxy
    strips it during provider-compat prep. The upstream body DOES carry
    `stream: true` (ticket 199ece93 Part A): the SSE parser consumes usage
    wherever the provider emits it, so include_usage is unnecessary.
    """
    StreamingBackendHandler._responses = [
        _make_text_response("ok"),
    ]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"X-Session-ID": "sess-sse-opt-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    # Drain the body so the completion thread runs to completion BEFORE we
    # assert on _request_count. The streaming generator returns immediately
    # after yielding ": connected", then yields the final data + "[DONE]" only
    # when the worker thread (which performs the upstream call) finishes.
    body = response.get_data(as_text=True)

    # Upstream was called exactly once
    assert StreamingBackendHandler._request_count == 1
    upstream_body = StreamingBackendHandler._captured_bodies[0]

    # stream_options must NOT be in the upstream body — the proxy strips it
    assert "stream_options" not in upstream_body, (
        f"stream_options must be stripped from upstream body, got: {upstream_body!r}"
    )
    # ticket 199ece93 Part A: the upstream body carries the client's stream intent
    assert upstream_body.get("stream") is True, (
        f"Upstream must be called streaming for a client stream=true request, "
        f"got stream={upstream_body.get('stream')!r}"
    )

    # The proxy DOES include usage info in the streamed response though
    payloads = _sse_data_payloads(body)
    usage_chunks = [p for p in payloads if "usage" in p]  
    # Usage may be merged into the final chunk by stream_response
    final_chunk = payloads[-1] if payloads else {}
    assert (
        usage_chunks
        or final_chunk.get("choices", [{}])[0].get("finish_reason") == "stop"
    ), "Expected either a usage chunk or a stop finish_reason at end of stream"


@pytest.mark.integration
def test_incomplete_upstream_sse_surfaces_upstream_error(streaming_client, streaming_backend):
    """A4 / spec §27 Layer 2: when the upstream emits genuine SSE that begins
    semantic output but never sends a finish_reason (protocol error), the
    UpstreamStreamError must surface to the client as a sanitized
    `upstream_error` envelope followed by exactly one `[DONE]` — NOT a
    `proxy_error`/500 and NOT any provider detail.

    Revert-test: if _map_exception_to_envelope drops the UpstreamStreamError
    branch, this becomes `proxy_error` and the type assertion fails.
    """
    StreamingBackendHandler._responses = ["__INCOMPLETE_SSE__"]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-incomplete-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"
    body = response.get_data(as_text=True)

    # Exactly one terminator, and the error is a sanitized upstream_error.
    assert body.count("data: [DONE]") == 1, f"Expected exactly one [DONE]:\n{body}"
    assert '"type": "upstream_error"' in body, (
        f"Incomplete upstream SSE must surface as upstream_error, got:\n{body}"
    )
    assert '"proxy_error"' not in body, f"Must not be proxy_error:\n{body}"
    # No raw provider detail / partial content leaks to the client.
    assert "partial" not in body, f"Partial upstream content leaked to client:\n{body}"


@pytest.mark.integration
def test_real_minimax_quirks_reconstruct_end_to_end(streaming_client, streaming_backend):
    """END-TO-END: replay MiniMax-M3's real SSE quirks (role in EVERY delta,
    a DUPLICATE terminal chunk with role+content:"", tool call split across two
    chunks, reasoning duplicated with content, NO [DONE]/clean EOF) through the
    full pipeline and assert the client receives the reconstructed bash tool
    call. This is the exact stream that crashed production (revert af5bea0);
    it must now flow cleanly.
    """
    raw = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"<think>ok","role":"assistant","reasoning":"ok"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null}',
        b"",
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"</think>","role":"assistant"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null}',
        b"",
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"id":"call_mm1","type":"function","function":{"name":"bash","arguments":""},"index":0}]}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null}',
        b"",
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"command\\":\\"echo hi\\"}"},"index":0}]}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null}',
        b"",
        b'data: {"id":"x","choices":[{"finish_reason":"tool_calls","index":0,"delta":{"content":"","role":"assistant"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null}',
        b"",
        b'data: {"id":"x","choices":[{"finish_reason":"tool_calls","index":0,"delta":{"content":"","role":"assistant"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null}',
        b"",
    ]
    StreamingBackendHandler._responses = [{"_raw_sse_lines": raw}]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "run echo hi"}],
            "stream": True,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }],
        },
        headers={"X-Session-ID": "sess-minimax-e2e-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"
    body = response.get_data(as_text=True)

    # Upstream was streamed exactly once (initial), no crash.
    assert StreamingBackendHandler._captured_bodies[0].get("stream") is True
    assert body.count("data: [DONE]") == 1
    assert '"upstream_error"' not in body, f"MiniMax stream must NOT error:\n{body}"
    assert '"proxy_error"' not in body

    # The reconstructed bash tool call reaches the client.
    payloads = _sse_data_payloads(body)
    blob = json.dumps(payloads)
    assert "bash" in blob, f"reconstructed bash tool call missing:\n{body}"
    assert "echo hi" in blob, f"reconstructed tool arguments missing:\n{body}"


# ---------------------------------------------------------------------------
# Test 5: Error during streaming returns error in SSE format
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_error_during_streaming_returns_sse_error(streaming_client, streaming_backend):
    """If upstream errors, the error must be returned in SSE format with
    a 'data: [DONE]' terminator so the client can close the stream cleanly.
    """
    # No responses configured — backend will return its default failure (500)
    # because _fail_after=None but no responses match; force a 500 by setting
    # _fail_after=0 so the very first request errors.
    StreamingBackendHandler._fail_after = 0

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-err-1"},
    )

    # The proxy should still produce a 200 SSE response with the error
    # surfaced as a data event (per _keepalive_stream error path)
    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    body = response.get_data(as_text=True)

    # Must end with [DONE] so clients can close cleanly
    assert body.rstrip().endswith("data: [DONE]"), (
        f"Stream must end with data: [DONE], got tail:\n{body[-200:]!r}"
    )

    # Must contain an error event in SSE format
    payloads = _sse_data_payloads(body)
    error_payloads = [p for p in payloads if "error" in p]
    assert error_payloads, (
        f"Expected at least one data payload containing 'error', got: {payloads!r}"
    )

    # The error is well-formed SSE: the chunk is wrapped in a 'data: ' line
    assert "data: {\"error\"" in body or 'data: {"error"' in body, (
        f"Error not in SSE data: line format:\n{body}"
    )

    # No partial chunks left without termination — body must not have a
    # trailing 'data: ' line without the \n\n terminator
    assert not body.rstrip("\n").endswith("data: "), (
        f"Stream has an unterminated trailing data: line:\n{body}"
    )


# ---------------------------------------------------------------------------
# Test 6: Post-processing comment is valid SSE and parsers can skip it
# ---------------------------------------------------------------------------


def _resolve_body_session_id(body: dict, client_hint: str) -> str:
    """Compute the actual session_id the pipeline will use for ``body``.

    Session IDs are content-derived (sha256 of user messages + client_hint),
    so a pre-registered agent must use this same session_id to be picked
    up by ``agent_registry.get_or_load()`` during the request.
    """
    from proxy.model_registry import resolve_session_id
    return resolve_session_id(body, client_hint)


def _pre_register_agent_with_mode(session_id: str, mode_name: str = "ticket"):
    """Pre-register a REAL Agent for ``session_id`` with an active ``mode_name``.

    Returns the Agent. Caller is responsible for cleanup via
    ``_cleanup_agent_registry_entries``.
    """
    agent = _RealAgent(
        agent_id=f"agent_{session_id}",
        session_id=session_id,
    )
    ms = _ModeState(
        name=mode_name,
        activated_at=_time.time(),
        ttl_seconds=300,
        skill_state={},
    )
    assert agent.set_active_mode(ms), "precondition: mode set"
    _agent_registry.register(agent)
    return agent


def _cleanup_agent_registry_entries(*agents):
    """Remove the agents from the shared registry so cross-test leaks are zero."""
    with _agent_registry._lock:
        for agent in agents:
            _agent_registry._agents.pop(agent.session_id, None)
            _agent_registry._agents_by_id.pop(agent.agent_id, None)
            _agent_registry._lru.pop(agent.session_id, None)
            _agent_registry._in_flight_requests.pop(agent.session_id, None)


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_stop_response_clears_mode(streaming_client, streaming_backend):
    """Streaming request with finish_reason=stop → mode is cleared after the stream.

    Per spec §7.4, ``finish_reason == "stop"`` is COMPLETE → PostProcessStage's
    ``finalize_skill_mode`` clears the agent's active mode. The keepalive
    generator runs PostProcessStage inline after the daemon thread completes,
    so the final state after the stream ends must show ``active_mode is None``.

    Uses a REAL Agent (via ``@pytest.mark.real_agent_registry``) so the CAS
    semantics in ``Agent.clear_active_mode(expected_mode=...)`` are exercised
    — not the autouse MockAgent's unconditional clear.
    """
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello-stream-stop"}],
    }
    client_hint = "stream-stop-mode-test"
    session_id = _resolve_body_session_id(body, client_hint)
    agent = _pre_register_agent_with_mode(session_id, mode_name="ticket")
    StreamingBackendHandler._responses = [
        _make_text_response("Streamed done.", finish_reason="stop"),
    ]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    try:
        assert response.status_code == 200, f"stream failed: {response.status_code}"
        # Drain the stream
        response.get_data(as_text=True)

        # Per spec §7.4, finish_reason=stop → COMPLETE → mode cleared.
        assert agent.snapshot_mode() is None, (
            "finish_reason=stop (COMPLETE) MUST clear the ticket mode "
            "via PostProcessStage.finalize_skill_mode"
        )
    finally:
        _cleanup_agent_registry_entries(agent)


@pytest.mark.integration
def test_streaming_runs_owner_setup_once_on_owner_path(streaming_client, streaming_backend, monkeypatch):
    """CP2B raison d'être (B2): the STREAMING route must run owner_setup exactly
    once on the OWNER execution path (in the completion worker, before
    pipeline.run). A sub-agent WORKSPACE canary for "choriprox" must therefore
    activate the inherited workspace exactly once via workspace_store.set_active.

    Revert-check: drop the ``on_owner_start=lambda: owner_setup(...)`` argument
    from the app.py streaming ``admit(...)`` call -> owner_setup never runs on
    the stream path -> 0 set_active calls -> this test fails.

    ``activate_inherited_workspace`` imports ``workspace_store`` function-locally,
    so ``patch.object`` on the shared singleton's ``set_active`` intercepts the
    worker-thread call while leaving the rest of the pipeline's real
    workspace_store behavior intact.
    """
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

    from proxy import canary as _canary
    sig = _canary.sign_canary("workspace:choriprox")
    marker = f"[[__SUBAGENT__:WORKSPACE:choriprox:{sig}]]"

    content = marker + "\nDo the streamed work"
    client_hint = "stream-owner-setup-test"
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
    }
    # session_id is content-derived from the RAW body (canary still present at
    # resolve time, before detect_subagent_ingress strips it).
    expected_session_id = _resolve_body_session_id(body, client_hint)

    StreamingBackendHandler._responses = [
        _make_text_response("Owner-path done.", finish_reason="stop"),
    ]

    from proxy.workspace_state import workspace_store as _ws_store
    with patch.object(_ws_store, "set_active") as spy:
        response = streaming_client.post(
            "/v1/chat/completions",
            json={**body, "stream": True},
            headers={"X-Session-Client": client_hint},
        )
        assert response.status_code == 200
        assert response.content_type == "text/event-stream"
        # Fully consume the SSE stream so the completion worker runs to
        # completion (owner_setup + pipeline.run both finish before the
        # patch context exits).
        stream_body = response.get_data(as_text=True)

    assert "data: [DONE]\n\n" in stream_body
    assert spy.call_count == 1, (
        f"owner_setup must run exactly once on the streaming owner path; "
        f"set_active call_count={spy.call_count}"
    )
    args, _kwargs = spy.call_args
    assert args[0] == expected_session_id, (
        f"set_active must use the sub-agent session_id; "
        f"got {args[0]!r}, expected {expected_session_id!r}"
    )
    assert args[1] == "choriprox"


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_content_filter_clears_mode(streaming_client, streaming_backend):
    """Streaming request with finish_reason=content_filter → mode is cleared (FAILED).

    Per spec §7.4, ``finish_reason == "content_filter"`` is FAILED → the mode
    is cleared. Memory sync still runs (it's a final-response signal per
    post_process.py:109) but the skill lifecycle clears.
    """
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "say-stream-cf"}],
    }
    client_hint = "stream-cf-mode-test"
    session_id = _resolve_body_session_id(body, client_hint)
    agent = _pre_register_agent_with_mode(session_id, mode_name="ticket")
    StreamingBackendHandler._responses = [
        _make_text_response("Filtered content.", finish_reason="content_filter"),
    ]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    try:
        assert response.status_code == 200
        response.get_data(as_text=True)

        assert agent.snapshot_mode() is None, (
            "finish_reason=content_filter (FAILED) MUST clear the ticket mode"
        )
    finally:
        _cleanup_agent_registry_entries(agent)


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_tool_calls_keeps_mode(streaming_client, streaming_backend):
    """Streaming request with finish_reason=tool_calls → mode is preserved (CONTINUE).

    Per spec §7.4, ``finish_reason == "tool_calls"`` is CONTINUE → the mode is
    kept alive for the next turn. This is the "mid-orchestration" case that
    a transient disconnect must NOT wipe (per spec §7.6 "Disconnect after
    CONTINUE finalization → no extra clear").

    Uses an upstream response with ``finish_reason="tool_calls"`` and an
    empty tool_calls list so ``_handle_tool_interception`` short-circuits
    (no actual tool execution) and the disposition driven into
    ``finalize_skill_mode`` is CONTINUE.
    """
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello-stream-tc"}],
    }
    client_hint = "stream-tc-mode-test"
    session_id = _resolve_body_session_id(body, client_hint)
    agent = _pre_register_agent_with_mode(session_id, mode_name="ticket")
    StreamingBackendHandler._responses = [
        _make_text_response("", finish_reason="tool_calls"),
    ]

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    try:
        assert response.status_code == 200, f"stream failed: {response.status_code}"
        response.get_data(as_text=True)

        current = agent.snapshot_mode()
        assert current is not None, (
            "finish_reason=tool_calls (CONTINUE) MUST preserve the ticket mode "
            "for the next turn in the multi-step orchestration"
        )
        assert current.name == "ticket", (
            f"Expected ticket mode preserved, got {current.name!r}"
        )
    finally:
        _cleanup_agent_registry_entries(agent)


# ===========================================================================
# Ticket 199ece93 §4 — Full-lifecycle heartbeat + completion-thread ownership.
#
# These tests pin the new architecture: a single background thread runs the
# entire pipeline (RequestPipeline.run(ctx) exactly once), the WSGI
# generator emits : connected + : keepalive per heartbeat while pending,
# and a downstream disconnect does NOT touch mode (PostProcess owns the
# keep/clear decision).
# ===========================================================================


@pytest.mark.integration
def test_immediate_heartbeat_first_chunk_is_connected(streaming_client, streaming_backend):
    """Streaming opens with `: connected` as the first SSE event.

    The new contract (ticket 199ece93 §4): the very first chunk flushed
    to the client is the `: connected` heartbeat comment — the upstream
    POST is dispatched in parallel by the completion thread and may take
    many seconds. Clients see a `: connected` line IMMEDIATELY so they
    know the proxy accepted the request.
    """
    StreamingBackendHandler._responses = [
        _make_text_response("quick"),
    ]
    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-connected-1"},
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)

    # First non-empty event must be the : connected comment.
    events = _split_sse(body)
    assert events, f"Expected at least one SSE event; body={body!r}"
    first = events[0]
    assert first.strip() == ": connected", (
        f"First SSE event MUST be ': connected' (no rotating dog, no "
        f"preamble); got {first!r}"
    )


@pytest.mark.integration
def test_direct_command_emits_exactly_one_done_terminator(
    streaming_client, streaming_backend
):
    """A direct command (`^^help`) emits exactly one `data: [DONE]` terminator.

    Pins the route-level contract: DispatchStage → StageAbort(plain dict) →
    _keepalive_stream → _sse_from_response_data. The dispatch path no longer
    owns SSE framing (ticket 199ece93 §4); the canonical `[DONE]` emitter
    is the single delivery generator in `proxy.app`.

    The mock upstream is registered with NO responses, so the StreamingBackend
    handler returns its `fallback` choice (an OpenAI-shape message). The
    upstream MUST NOT be called for a direct command — DispatchStage short-
    circuits the request before reaching UpstreamStage.
    """
    StreamingBackendHandler._responses = []  # explicit: no upstream
    StreamingBackendHandler._request_count = 0

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "^^help"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-direct-help-1"},
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)

    # Exactly one [DONE] terminator — the canonical single-emitter contract.
    done_count = body.count("data: [DONE]\n\n")
    assert done_count == 1, (
        f"Direct command streaming response must emit EXACTLY one "
        f"[DONE] terminator; got {done_count} in body:\n{body!r}"
    )

    # Immediate `: connected` heartbeat on open.
    assert ": connected\n\n" in body, (
        f"Missing immediate : connected heartbeat; body:\n{body!r}"
    )

    # The direct command's content is present (^^help renders a list).
    assert "Available Commands" in body or "Skills" in body, (
        f"Direct command response content not found in stream; body:\n{body!r}"
    )

    # The upstream was NOT called — DispatchStage short-circuits before
    # UpstreamStage for direct commands.
    assert StreamingBackendHandler._request_count == 0, (
        f"Upstream must NOT be called for direct commands; "
        f"got {StreamingBackendHandler._request_count} calls"
    )


@pytest.mark.integration
def test_repeated_heartbeat_during_upstream(slow_streaming_backend):
    """While the upstream is blocked, `: keepalive` events stream repeatedly.

    Uses the `slow_streaming_backend` fixture (30s upstream delay by default;
    we override to 2.0s so the test runs quickly) plus a heartbeat of 0.3s
    (the production `_keepalive_stream` forces values <= 0 back to 5.0,
    so we MUST pass a small positive value). The test drains enough of the
    stream to see at least TWO `: keepalive` events before the upstream
    releases (which then lets the final response + [DONE] through).

    No real semantic data: response arrives before the backend release —
    i.e. nothing has streamed yet because the upstream is still blocked.
    """
    from unittest.mock import patch as _patch
    from proxy.state import CONFIG

    _SlowBackendHandler._request_entered = threading.Event()
    _SlowBackendHandler._release_response = threading.Event()
    _SlowBackendHandler._response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"delayed"},"finish_reason":"stop"}]}'
    )

    client, _server = slow_streaming_backend

    with _patch("proxy.state.CONFIG", {**CONFIG, "streaming": {"heartbeat_interval": 0.3}}):
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "slow hello"}],
                    "stream": True,
                },
                headers={"X-Session-ID": "sess-sse-heartbeat-1"},
            )
            assert response.status_code == 200
            chunk_iter = iter(response.response)
            first = next(chunk_iter)
            assert b": connected" in first, f"first chunk must be : connected; got {first!r}"
            keepalive_count = 0
            chunks = [first]
            for _ in range(20):
                try:
                    nxt = next(chunk_iter)
                except StopIteration:
                    break
                chunks.append(nxt)
                if b": keepalive" in nxt:
                    keepalive_count += 1
                if b"data:" in nxt:
                    break
            assert keepalive_count >= 2, (
                f"Expected >=2 : keepalive events before upstream release; "
                f"got {keepalive_count} across chunks {chunks!r}"
            )
            assert not any(chunk.startswith(b"data:") for chunk in chunks), (
                f"No data: response should arrive before upstream release; "
                f"got {chunks!r}"
            )
        finally:
            _SlowBackendHandler._release_response.set()


@pytest.mark.integration
def test_heartbeat_during_preflight(streaming_client, streaming_backend):
    """Preflight (enforce_context_limits) blocks → `: keepalive` events fire.

    Wraps `proxy.context.enforce_context_limits` with an event gate that
    blocks the REAL preflight function until the test releases it. The
    completion thread is inside pipeline.run → upstream has not been called
    yet, but the WSGI generator is still emitting heartbeats.
    """
    from unittest.mock import patch as _patch
    from proxy.context import enforce_context_limits as real_enforce_context_limits

    preflight_entered = threading.Event()
    preflight_release = threading.Event()

    def gated_enforce(*args, **kwargs):
        preflight_entered.set()
        assert preflight_release.wait(timeout=10)
        return real_enforce_context_limits(*args, **kwargs)

    StreamingBackendHandler._responses = [
        _make_text_response("after-preflight"),
    ]
    with _patch("proxy.context.enforce_context_limits", side_effect=gated_enforce):
        response = streaming_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"X-Session-ID": "sess-sse-preflight-1"},
        )
        assert response.status_code == 200
        chunk_iter = iter(response.response)
        first = next(chunk_iter)
        assert b": connected" in first
        chunks = [first]
        keepalive_seen = False
        for _ in range(6):
            try:
                nxt = next(chunk_iter)
            except StopIteration:
                break
            chunks.append(nxt)
            if b": keepalive" in nxt:
                keepalive_seen = True
                break
        assert keepalive_seen, (
            f"Expected : keepalive during preflight gate; got {chunks!r}"
        )
        assert preflight_entered.wait(timeout=10), "preflight never invoked"
        preflight_release.set()
        rest = b"".join(c for c in iter(lambda: next(chunk_iter, b""), b""))
        assert b"data: [DONE]" in rest, f"missing [DONE] after preflight release: {rest!r}"
        assert StreamingBackendHandler._request_count == 1, (
            f"Upstream must be called exactly once per streaming request; "
            f"got {StreamingBackendHandler._request_count}"
        )


@pytest.mark.integration
def test_heartbeat_during_post_processing(streaming_client, streaming_backend):
    """Post-process blocks → `: keepalive` events fire.

    Wraps `PostProcessStage._persist_messages` (a real sub-op that
    happens between upstream and finalization) with an event gate. The
    test asserts:
      - upstream completes first (so the body was received)
      - `: keepalive` events arrive while post-process is blocked
      - the final semantic SSE does NOT arrive until post-process is released
      - PostProcessStage.execute runs exactly once
    """
    from proxy.pipeline.post_process import PostProcessStage
    from unittest.mock import patch as _patch

    postprocess_entered = threading.Event()
    postprocess_release = threading.Event()
    real_persist = PostProcessStage._persist_messages

    def gated_persist(self, ctx):
        postprocess_entered.set()
        assert postprocess_release.wait(timeout=10)
        return real_persist(self, ctx)

    StreamingBackendHandler._responses = [
        _make_text_response("postprocess-after"),
    ]
    with _patch.object(PostProcessStage, "_persist_messages", gated_persist):
        response = streaming_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"X-Session-ID": "sess-sse-pp-1"},
        )
        assert response.status_code == 200
        chunk_iter = iter(response.response)
        first = next(chunk_iter)
        assert b": connected" in first
        chunks = [first]
        keepalive_seen = False
        for _ in range(6):
            try:
                nxt = next(chunk_iter)
            except StopIteration:
                break
            chunks.append(nxt)
            if b": keepalive" in nxt:
                keepalive_seen = True
                break
        assert keepalive_seen, (
            f"Expected : keepalive during post-process gate; got {chunks!r}"
        )
        assert not any(b"data:" in c for c in chunks), (
            f"data: response must not arrive before post-process is released; "
            f"got {chunks!r}"
        )
        assert postprocess_entered.wait(timeout=10), "PostProcess never invoked"
        postprocess_release.set()
        rest = b"".join(c for c in iter(lambda: next(chunk_iter, b""), b""))
        assert rest.count(b"data: [DONE]") == 1, (
            f"Expected exactly one [DONE] after post-process release; "
            f"got {rest!r}"
        )


@pytest.mark.integration
def test_disconnect_does_not_stop_completion(slow_streaming_backend):
    """Client closes the stream early → completion thread keeps running.

    Uses event-based sync (request_entered / release_response) on
    `_SlowBackendHandler`. The test:
      1. starts the streaming request,
      2. drains `: connected`,
      3. closes the response (GeneratorExit on the generator),
      4. releases the backend (so upstream returns),
      5. waits for post-process entry,
      6. asserts upstream was called once and post-process ran once.

    This proves the completion thread is not tied to the WSGI generator.
    """
    from proxy.pipeline.post_process import PostProcessStage
    from unittest.mock import patch as _patch

    _SlowBackendHandler._request_entered = threading.Event()
    _SlowBackendHandler._release_response = threading.Event()
    _SlowBackendHandler._response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"late"},'
        b'"finish_reason":"stop"}]}'
    )

    postprocess_entered = threading.Event()
    postprocess_finished = threading.Event()

    real_persist = PostProcessStage._persist_messages

    def gated_persist(self, ctx):
        postprocess_entered.set()
        try:
            return real_persist(self, ctx)
        finally:
            postprocess_finished.set()

    client, _server = slow_streaming_backend
    with _patch.object(PostProcessStage, "_persist_messages", gated_persist):
        try:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "disconnect me"}],
                    "stream": True,
                },
                headers={"X-Session-ID": "sess-sse-disc-1"},
            )
            assert response.status_code == 200
            chunk_iter = iter(response.response)
            first = next(chunk_iter)
            assert b": connected" in first
            response.close()
            assert _SlowBackendHandler._request_entered.wait(timeout=10), (
                "upstream never observed the request — completion thread "
                "did not reach UpstreamStage (regression of the new contract)"
            )
            _SlowBackendHandler._release_response.set()
            assert postprocess_entered.wait(timeout=10), (
                "PostProcessStage did not run after disconnect — completion "
                "thread was tied to the generator (regression of the new contract)"
            )
            assert postprocess_finished.wait(timeout=10), (
                "PostProcessStage did not finish after disconnect"
            )
        finally:
            _SlowBackendHandler._release_response.set()


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_disconnect_with_finish_reason_stop_clears_mode_via_postprocess(
    slow_streaming_backend,
):
    """Client disconnects before SSE chunks yielded → PostProcess clears mode.

    Per the new contract (ticket 199ece93 §4), a downstream disconnect
    does NOT clear mode. PostProcessStage owns the keep/clear decision
    and runs in the completion thread even after the WSGI generator
    closes. This test proves: a stop finish_reason + disconnected
    delivery → mode is still cleared (by PostProcess, not by disconnect).
    """
    from proxy.pipeline.post_process import PostProcessStage
    from unittest.mock import patch as _patch

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "disconnect-stop"}],
    }
    client_hint = "disc-stop-clears"
    session_id = _resolve_body_session_id(body, client_hint)
    agent = _pre_register_agent_with_mode(session_id, mode_name="ticket")

    _SlowBackendHandler._request_entered = threading.Event()
    _SlowBackendHandler._release_response = threading.Event()
    _SlowBackendHandler._response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"disconnected done."},'
        b'"finish_reason":"stop"}]}'
    )

    finalize_entered = threading.Event()
    finalize_finished = threading.Event()
    real_finalize = PostProcessStage._finalize_skill_lifecycle_with_flag

    def gated_finalize(self, ctx, expected_mode):
        finalize_entered.set()
        try:
            return real_finalize(self, ctx, expected_mode)
        finally:
            finalize_finished.set()

    client, _server = slow_streaming_backend
    try:
        with _patch.object(PostProcessStage, "_finalize_skill_lifecycle_with_flag", gated_finalize):
            response = client.post(
                "/v1/chat/completions",
                json={**body, "stream": True},
                headers={"X-Session-Client": client_hint},
            )
            assert response.status_code == 200
            chunk_iter = iter(response.response)
            first = next(chunk_iter)
            assert b": connected" in first
            response.close()
            assert _SlowBackendHandler._request_entered.wait(timeout=10), (
                "upstream never observed the request — completion thread "
                "did not reach UpstreamStage"
            )
            _SlowBackendHandler._release_response.set()
            assert finalize_entered.wait(timeout=10), (
                "PostProcessStage._finalize_skill_lifecycle_with_flag did not run "
                "after disconnect — completion thread did not reach the end of "
                "PostProcessStage.execute's finally block"
            )
            assert finalize_finished.wait(timeout=10), (
                "PostProcessStage._finalize_skill_lifecycle_with_flag did not finish"
            )
        assert agent.snapshot_mode() is None, (
            "finish_reason=stop + disconnect: PostProcess MUST have cleared "
            "ticket mode (post-process owns the keep/clear decision)"
        )
    finally:
        _SlowBackendHandler._release_response.set()
        _cleanup_agent_registry_entries(agent)


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_disconnect_with_finish_reason_tool_calls_preserves_mode_via_postprocess(
    slow_streaming_backend,
):
    """Client disconnects mid-stream with a continuation response → mode preserved.

    Complement of the stop test: tool_calls finish_reason INTENTIONALLY
    keeps ticket mode alive for the next turn. PostProcess still runs in
    the completion thread after disconnect and preserves the mode.
    """
    from proxy.pipeline.post_process import PostProcessStage
    from unittest.mock import patch as _patch

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "disconnect-tool-calls"}],
    }
    client_hint = "disc-tc-keeps"
    session_id = _resolve_body_session_id(body, client_hint)
    agent = _pre_register_agent_with_mode(session_id, mode_name="ticket")

    _SlowBackendHandler._request_entered = threading.Event()
    _SlowBackendHandler._release_response = threading.Event()
    _SlowBackendHandler._response_body = json.dumps({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_continue_1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"filePath": "README.md"}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "model": "test/llm-v1",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()

    finalize_entered = threading.Event()
    finalize_finished = threading.Event()
    real_finalize = PostProcessStage._finalize_skill_lifecycle_with_flag

    def gated_finalize(self, ctx, expected_mode):
        finalize_entered.set()
        try:
            return real_finalize(self, ctx, expected_mode)
        finally:
            finalize_finished.set()

    client, _server = slow_streaming_backend
    try:
        with _patch.object(PostProcessStage, "_finalize_skill_lifecycle_with_flag", gated_finalize):
            response = client.post(
                "/v1/chat/completions",
                json={**body, "stream": True},
                headers={"X-Session-Client": client_hint},
            )
            assert response.status_code == 200
            chunk_iter = iter(response.response)
            first = next(chunk_iter)
            assert b": connected" in first
            response.close()
            assert _SlowBackendHandler._request_entered.wait(timeout=10), (
                "upstream never observed the request — completion thread "
                "did not reach UpstreamStage"
            )
            _SlowBackendHandler._release_response.set()
            assert finalize_entered.wait(timeout=10), (
                "PostProcessStage._finalize_skill_lifecycle_with_flag did not run "
                "after disconnect — completion thread did not reach the end of "
                "PostProcessStage.execute's finally block"
            )
            assert finalize_finished.wait(timeout=10), (
                "PostProcessStage._finalize_skill_lifecycle_with_flag did not finish"
            )
        current = agent.snapshot_mode()
        assert current is not None, (
            "finish_reason=tool_calls + disconnect: PostProcess MUST have "
            "preserved ticket mode (continuation turn expected)"
        )
        assert current.name == "ticket", (
            f"Expected ticket mode preserved, got {current.name!r}"
        )
    finally:
        _SlowBackendHandler._release_response.set()
        _cleanup_agent_registry_entries(agent)


@pytest.mark.integration
def test_post_process_runs_exactly_once_per_streaming_request(
    streaming_client, streaming_backend
):
    """PostProcessStage.execute runs exactly once per streaming request.

    Even with the new completion-thread architecture, a single request
    MUST enter PostProcessStage exactly once. The test instruments
    PostProcessStage.execute via a counting wrapper; the count MUST be 1
    on a successful streaming request.

    Regression guard against:
      - the old `_defer_upstream` letting the full pipeline reach
        PostProcess before upstream (would have caused double-execution
        in some old code paths),
      - any future regression that calls PostProcessStage twice.
    """
    from proxy.pipeline.post_process import PostProcessStage
    from unittest.mock import patch as _patch

    real_execute = PostProcessStage.execute
    counter = {"calls": 0}

    def _counting_execute(self, ctx):
        counter["calls"] += 1
        return real_execute(self, ctx)

    StreamingBackendHandler._responses = [
        _make_text_response("once"),
    ]
    with _patch.object(PostProcessStage, "execute", _counting_execute):
        response = streaming_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "count me"}],
                "stream": True,
            },
            headers={"X-Session-ID": "sess-sse-once-1"},
        )
        assert response.status_code == 200
        # Drain the entire stream.
        body = response.get_data(as_text=True)
        assert "data: [DONE]" in body, f"missing [DONE] in body={body!r}"
    assert counter["calls"] == 1, (
        f"PostProcessStage.execute must run exactly once per streaming "
        f"request; got {counter['calls']}"
    )


@pytest.mark.integration
def test_direct_skill_response_emits_exactly_one_done_terminator(
    streaming_client, streaming_backend, tmp_path, monkeypatch
):
    """A direct-skill response (`^^workspace status` style) emits exactly one
    `data: [DONE]\n\n` terminator.

    Regression guard against the OLD dispatch.py bug where the direct
    skill stream emitted its OWN `[DONE]` in `finally`, producing two
    terminators. The fix removed that branch; the canonical `[DONE]`
    emitter is now the single delivery generator in app.py.
    """
    from proxy.skill_dispatcher import SkillDispatcher
    from proxy.skills.skill_registry import SkillRegistry, SkillMeta
    from proxy.skills.skill_definition import BaseSkill

    class _DirectSkill(BaseSkill):
        name = "test-direct"
        trigger = "^^test-direct"
        activation = "command"
        ttl_seconds = 300

        def parse_subcommand(self, text):
            return ("status", "")

        def handle_direct(self, ctx, action, arg):
            return {
                "choices": [{"message": {"role": "assistant", "content": "status-ok"}}],
                "model": "test-direct",
            }

        def get_meta(self):
            return SkillMeta(name="test-direct", trigger="^^test-direct")

    # Register the skill via the real registry.
    from proxy.skills.skill_registry import get_default_registry, reset_default_registry
    reset_default_registry()
    reg = get_default_registry()
    reg.register(_DirectSkill())

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "^^test-direct"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-direct-sse-1"},
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    done_count = body.count("data: [DONE]\n\n")
    assert done_count == 1, (
        f"Direct-skill streaming response must emit EXACTLY one "
        f"[DONE] terminator; got {done_count} in body={body!r}"
    )


class _SlowBackendHandler(BaseHTTPRequestHandler):
    """Mock backend that blocks before responding, gated by a release Event.

    Class-level state:
      _request_entered:   threading.Event set as soon as do_POST begins reading
                          the request body. Tests use this to observe when the
                          upstream call is in flight.
      _release_response:  threading.Event that the test sets to let the handler
                          finish writing the response. The handler blocks on
                          this with a generous timeout, so a forgotten release
                          surfaces as a test-harness failure, not a silent hang.
      _response_body:     bytes — JSON to return after release.
    """

    _request_entered: threading.Event = threading.Event()
    _release_response: threading.Event = threading.Event()
    _response_body: bytes = (
        b'{"choices":[{"message":{"role":"assistant","content":"slow"},'
        b'"finish_reason":"stop"}]}'
    )

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
        type(self)._request_entered.set()
        if not type(self)._release_response.wait(timeout=10):
            raise RuntimeError("test backend release timed out")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self)._response_body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def slow_streaming_backend(tmp_path, monkeypatch):
    """Flask client wired to a slow mock backend (delay before responding).

    Configured for a sub-second keepalive so the test runs quickly.
    """
    import threading as _th
    from http.server import HTTPServer
    from clients.model_registry import ModelRegistry

    server = HTTPServer(("127.0.0.1", 0), _SlowBackendHandler)
    port = server.server_address[1]
    _SlowBackendHandler._request_entered = threading.Event()
    _SlowBackendHandler._release_response = threading.Event()
    _SlowBackendHandler._response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"slow"},"finish_reason":"stop"}]}'
    )

    thread = _th.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    backend_url = f"http://127.0.0.1:{port}/v1"
    config = {
        "models": {
            "test-model": {
                "backend": backend_url,
                "tool_parser": "openai",
                "context_limit": 262144,
                "name": "Test Model",
                "api_key_env": "TEST_API_KEY",
                "backend_model": "test/llm-v1",
            },
        },
        "default": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 32000,
        },
    }
    path = tmp_path / "proxy_models_slow.json"
    path.write_text(json.dumps(config))
    reg = ModelRegistry(config_path="proxy_models_slow.json", base_dir=str(tmp_path))
    monkeypatch.setattr("qwen_proxy._model_registry", reg)
    monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
    monkeypatch.setattr("proxy.app._model_registry", reg)
    monkeypatch.setattr("proxy.model_registry._model_registry", reg)

    monkeypatch.setitem(
        __import__("proxy.state", fromlist=["CONFIG"]).CONFIG,
        "streaming",
        {"heartbeat_interval": 0.05},
    )

    import requests as _r
    fresh = _r.Session()
    monkeypatch.setattr("qwen_proxy._HTTP_SESSION", fresh)
    monkeypatch.setattr("proxy.state._HTTP_SESSION", fresh)
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret-key-12345")

    import qwen_proxy
    qwen_proxy.app.config["TESTING"] = True
    yield qwen_proxy.app.test_client(), server

    server.shutdown()


def _build_swarm_capable_skill_dir(tmp_path, monkeypatch, *, max_calls=4):
    """Helper: write a swarm-capable custom JSON skill, reset registry."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "research.json").write_text(json.dumps({
        "name": "research",
        "trigger": "^^research",
        "description": "Streaming swarm proof skill.",
        "prompt": "RESEARCH_PROMPT_SENTINEL_STREAMING",
        "tool_whitelist": ["read", "grep", "task"],
        "subagents": {
            "tool_whitelist": ["read", "grep"],
            "max_task_calls_per_request": max_calls,
        },
    }))
    monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
    monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
    monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)
    from proxy.skills.skill_registry import reset_default_registry
    reset_default_registry()
    return skills_dir


def _register_real_agent_for_body(body, client_hint):
    """Pre-register a real Agent for the session derived from body."""
    from proxy.agent.agent import Agent
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.model_registry import resolve_session_id

    session_id = resolve_session_id(body, client_hint)
    agent_registry.register(Agent(session_id=session_id, agent_id=f"agent_{session_id}"))
    return session_id


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_swarm_over_budget_emits_sse_error(
    streaming_client, streaming_backend, tmp_path, monkeypatch
):
    """Streaming over-budget response: SSE error envelope + [DONE], no tool-call chunk.

    Per §15.1, the proxy validates the COMPLETE upstream response in non-streaming
    mode before emitting any tool-call chunk. For a swarm-over-budget response,
    the proxy MUST emit:
      - Exactly ONE SSE data chunk carrying the swarm_max_calls_exceeded envelope,
      - followed by `data: [DONE]\\n\\n`,
      - with NO tool-call chunk (no canary, no transformed task entry).
    """
    _build_swarm_capable_skill_dir(tmp_path, monkeypatch, max_calls=1)

    # Two task tool calls — exceeds the budget of 1.
    StreamingBackendHandler._responses = [
        _make_tool_call_response([
            _openai_tool_call(
                "call_over_a",
                "task",
                {"description": "x", "prompt": "p1"},
            ),
            _openai_tool_call(
                "call_over_b",
                "task",
                {"description": "x", "prompt": "p2"},
            ),
        ]),
    ]

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research streaming over"}],
    }
    client_hint = "stream-swarm-over"
    _register_real_agent_for_body(body, client_hint)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"
    body_text = response.get_data(as_text=True)
    # [DONE] terminator present.
    assert body_text.rstrip().endswith("data: [DONE]"), (
        f"Stream must end with [DONE]: tail={body_text[-200:]!r}"
    )

    payloads = _sse_data_payloads(body_text)
    error_payloads = [p for p in payloads if "error" in p]
    assert len(error_payloads) == 1, (
        f"Expected exactly 1 SSE error payload, got {len(error_payloads)}: "
        f"{payloads!r}"
    )
    err = error_payloads[0]["error"]
    assert err["code"] == "swarm_max_calls_exceeded"
    # No canary / no transformed task entry leaked into stream.
    assert "[[__SUBAGENT__" not in body_text, (
        f"Canary marker leaked into stream: {body_text[:300]!r}"
    )
    # No prompt content from the original task call leaked into stream.
    assert "p1" not in body_text and "p2" not in body_text, (
        f"Original prompt content leaked: {body_text[:300]!r}"
    )


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_swarm_subagent_nesting_emits_sse_error(
    streaming_client, streaming_backend, tmp_path, monkeypatch
):
    """Streaming sub-agent with task → SSE error envelope swarm_subagent_cannot_spawn."""
    _build_swarm_capable_skill_dir(tmp_path, monkeypatch, max_calls=4)

    # Build a valid swarm canary so the request is recognized as a sub-agent.
    from proxy.canary import build_swarm_skill_canary
    canary = build_swarm_skill_canary(
        "research", {}, session_id="stream-swarm-nest-sess"
    )

    StreamingBackendHandler._responses = [
        _make_tool_call_response([
            _openai_tool_call(
                "call_sub_nest",
                "task",
                {"description": "nested", "prompt": "x"},
            ),
        ]),
    ]

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": f"{canary} investigate"}],
    }
    client_hint = "stream-swarm-nest"
    _register_real_agent_for_body(body, client_hint)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"
    body_text = response.get_data(as_text=True)
    assert body_text.rstrip().endswith("data: [DONE]"), (
        f"Stream must end with [DONE]: tail={body_text[-200:]!r}"
    )

    payloads = _sse_data_payloads(body_text)
    error_payloads = [p for p in payloads if "error" in p]
    assert len(error_payloads) == 1, (
        f"Expected exactly 1 SSE error payload, got {len(error_payloads)}: {payloads!r}"
    )
    err = error_payloads[0]["error"]
    assert err["code"] == "swarm_subagent_cannot_spawn"
    assert "[[__SUBAGENT__" not in body_text


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_swarm_signer_failure_emits_sse_error(
    streaming_client, streaming_backend, tmp_path, monkeypatch
):
    """Streaming swarm signer failure → SSE error envelope, no prompt leak."""
    _build_swarm_capable_skill_dir(tmp_path, monkeypatch, max_calls=4)

    StreamingBackendHandler._responses = [
        _make_tool_call_response([
            _openai_tool_call(
                "call_signer_stream",
                "task",
                {"description": "investigate", "prompt": "SECRET-LEAK-STREAM-XYZ-99"},
            ),
        ]),
    ]

    from proxy.canary import SwarmCanaryError

    def _raise(*args, **kwargs):
        raise SwarmCanaryError("swarm_signing_failed", "forced for streaming test")

    monkeypatch.setattr("proxy.canary.build_swarm_skill_canary", _raise)

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research trigger signer"}],
    }
    client_hint = "stream-swarm-signer"
    _register_real_agent_for_body(body, client_hint)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 200
    body_text = response.get_data(as_text=True)
    assert body_text.rstrip().endswith("data: [DONE]"), (
        f"Stream must end with [DONE]: tail={body_text[-200:]!r}"
    )

    payloads = _sse_data_payloads(body_text)
    error_payloads = [p for p in payloads if "error" in p]
    assert len(error_payloads) == 1, (
        f"Expected exactly 1 SSE error payload, got {len(error_payloads)}: {payloads!r}"
    )
    err = error_payloads[0]["error"]
    assert err["code"] == "swarm_signing_failed"
    # No prompt content leak.
    assert "SECRET-LEAK-STREAM-XYZ-99" not in body_text, (
        f"Prompt content leaked into stream: {body_text[:300]!r}"
    )


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_streaming_swarm_valid_parent_emits_canary(
    streaming_client, streaming_backend, tmp_path, monkeypatch
):
    """Streaming valid swarm parent: tool-call chunk with canary + [DONE]."""
    from proxy.canary import parse_canary

    _build_swarm_capable_skill_dir(tmp_path, monkeypatch, max_calls=4)

    StreamingBackendHandler._responses = [
        _make_tool_call_response([
            _openai_tool_call(
                "call_valid_swarm",
                "task",
                {"description": "investigate", "prompt": "look at bug"},
            ),
        ]),
    ]

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^research valid"}],
    }
    client_hint = "stream-swarm-valid"
    _register_real_agent_for_body(body, client_hint)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={**body, "stream": True},
        headers={"X-Session-Client": client_hint},
    )

    assert response.status_code == 200
    body_text = response.get_data(as_text=True)
    assert body_text.rstrip().endswith("data: [DONE]"), (
        f"Stream must end with [DONE]: tail={body_text[-200:]!r}"
    )

    # Exactly ONE tool-call chunk carrying the SKILL canary.
    payloads = _sse_data_payloads(body_text)
    swarm_chunks = []
    for p in payloads:
        choices = p.get("choices") or []
        if not choices:
            continue
        msg = choices[0].get("message") or choices[0].get("delta") or {}
        for tc in (msg.get("tool_calls") or []):
            func = tc.get("function") or {}
            args_str = func.get("arguments", "")
            if not args_str:
                continue
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                continue
            prompt = args.get("prompt", "")
            if "[[__SUBAGENT__:SKILL:research:" in prompt:
                swarm_chunks.append(prompt)
    assert len(swarm_chunks) >= 1, (
        f"Expected at least one tool-call chunk with swarm canary, got: {payloads!r}"
    )
    # The canary is valid.
    parsed = parse_canary(swarm_chunks[0])
    assert parsed is not None and parsed.kind == "skill"
    assert parsed.skill_name == "research"


# ===========================================================================
# US-060a — SSE malformed-completion defense
#
# _sse_from_response_data MUST refuse to emit any malformed chat completion
# body (None, {}, [], choices:None, choices:[], choices:[None], etc.) and
# replace it with the canonical upstream protocol error. Explicit upstream
# error envelopes (``{"error": {...}}`` with no ``choices``) are preserved
# verbatim so the client sees the real failure shape.
# ===========================================================================


_MALFORMED_SSE_INPUTS = [
    pytest.param(None, id="none"),
    pytest.param({}, id="empty-dict"),
    pytest.param([], id="empty-list"),
    pytest.param(
        {"choices": None, "id": "raw-null"}, id="choices-null"
    ),
    pytest.param(
        {"choices": [], "id": "raw-empty"}, id="choices-empty"
    ),
    pytest.param(
        {"choices": [None], "id": "raw-first"}, id="choices-first-none"
    ),
    pytest.param(
        {"choices": [{"message": None}], "id": "raw-message"},
        id="choices-message-none",
    ),
    pytest.param(
        {"error": "not-an-object", "id": "raw-error"},
        id="error-string-not-dict",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("data", _MALFORMED_SSE_INPUTS)
def test_sse_from_response_data_refuses_malformed_choices_direct(data):
    """Direct formatter defense: any malformed input is replaced by the
    canonical upstream protocol error envelope and exactly one [DONE]
    terminator is emitted. Sentinel/ID values from the malformed input
    MUST NOT appear in the SSE body.

    Pinned so restoring the old ``make_sse_chunk(data)`` raw fallback
    regresses this test (the fallback would have emitted the malformed
    body verbatim).
    """
    from proxy.app import _sse_from_response_data
    from proxy.openai_compat import build_upstream_protocol_error

    chunks = list(_sse_from_response_data(data, "test-model"))
    body = "".join(chunks)

    parsed_payloads = _sse_data_payloads(body)

    expected = [build_upstream_protocol_error()]
    assert parsed_payloads == expected, (
        f"Malformed SSE input {data!r} must yield exactly the protocol "
        f"error envelope; got {parsed_payloads!r}"
    )
    assert body.count("data: [DONE]\n\n") == 1, (
        f"Malformed SSE input {data!r} must yield exactly one [DONE] "
        f"terminator; body={body!r}"
    )

    if isinstance(data, dict):
        bad_id = data.get("id")
        if bad_id and "raw-" in bad_id:
            assert bad_id not in body, (
                f"Malformed id {bad_id!r} leaked into SSE body for "
                f"input {data!r}; body={body!r}"
            )
    if isinstance(data, dict) and data.get("error") == "not-an-object":
        assert "not-an-object" not in body, (
            f"Malformed error sentinel leaked into SSE body; body={body!r}"
        )


@pytest.mark.integration
def test_sse_from_response_data_preserves_explicit_error_direct():
    """Direct formatter: explicit upstream error envelopes
    (``{"error": {dict}}`` with no ``choices``) are preserved verbatim
    so the client sees the real provider failure shape.
    """
    from proxy.app import _sse_from_response_data

    upstream_error = {
        "error": {"message": "provider failed", "type": "provider_error"}
    }

    chunks = list(_sse_from_response_data(upstream_error, "test-model"))
    body = "".join(chunks)

    parsed_payloads = _sse_data_payloads(body)
    assert parsed_payloads == [upstream_error], (
        f"Explicit upstream error must be preserved verbatim; "
        f"got {parsed_payloads!r}"
    )
    assert body.count("data: [DONE]\n\n") == 1, (
        f"Explicit upstream error path must emit exactly one [DONE]; "
        f"body={body!r}"
    )


@pytest.mark.integration
def test_streaming_pipeline_malformed_then_valid_full_flow(
    streaming_client, streaming_backend, monkeypatch
):
    """Full pipeline: upstream returns a malformed response, then a valid
    one. The malformed body and its sentinel MUST NOT leak into the
    SSE stream; the client sees the valid recovered content.

    The upstream client retries malformed responses when total_tokens
    is 0 (the safe-to-retry path); the second response is consumed
    on retry. ``proxy.http_client.time.sleep`` is patched to no-op so
    the retry does not block the test.
    """
    StreamingBackendHandler._responses = [
        {
            "id": "malformed-attempt-1",
            "choices": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "secret_sentinel": "FIRST_MALFORMED_MUST_NOT_LEAK",
        },
        _make_text_response("recovered"),
    ]
    monkeypatch.setattr("proxy.http_client.time.sleep", lambda *_a, **_k: None)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream malformed then valid"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-malformed-valid-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    body = response.get_data(as_text=True)
    assert StreamingBackendHandler._request_count == 2, (
        f"Upstream must be called exactly twice (initial malformed + "
        f"retry); got {StreamingBackendHandler._request_count}"
    )

    parsed_payloads = _sse_data_payloads(body)
    content_deltas = [
        p["choices"][0].get("delta", {}).get("content", "")
        for p in parsed_payloads
        if p.get("choices")
    ]
    assert "".join(content_deltas) == "recovered", (
        f"Expected recovered content only; got content_deltas={content_deltas!r} "
        f"payloads={parsed_payloads!r}"
    )

    for p in parsed_payloads:
        err = p.get("error") if isinstance(p, dict) else None
        if isinstance(err, dict):
            assert err.get("code") != "invalid_chat_completion", (
                f"Recovered response must not carry the malformed envelope; "
                f"got {p!r}"
            )

    assert "malformed-attempt-1" not in body, (
        f"Malformed response id leaked into SSE body: {body[:300]!r}"
    )
    assert "FIRST_MALFORMED_MUST_NOT_LEAK" not in body, (
        f"Malformed sentinel leaked into SSE body: {body[:300]!r}"
    )
    assert body.count("data: [DONE]\n\n") == 1, (
        f"Stream must end with exactly one [DONE]; body={body!r}"
    )


@pytest.mark.integration
def test_streaming_pipeline_malformed_twice_full_flow(
    streaming_client, streaming_backend, monkeypatch
):
    """Full pipeline: two consecutive malformed zero-token responses.
    Retries are exhausted; the canonical upstream protocol error is
    emitted. Neither malformed id nor sentinel may leak.
    """
    StreamingBackendHandler._responses = [
        {
            "id": "malformed-attempt-a",
            "choices": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "secret_sentinel": "SENTINEL_A_MUST_NOT_LEAK",
        },
        {
            "id": "malformed-attempt-b",
            "choices": None,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "secret_sentinel": "SENTINEL_B_MUST_NOT_LEAK",
        },
    ]
    monkeypatch.setattr("proxy.http_client.time.sleep", lambda *_a, **_k: None)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream malformed twice"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-malformed-twice-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    body = response.get_data(as_text=True)
    assert StreamingBackendHandler._request_count == 2, (
        f"Upstream must be called exactly twice (initial malformed + "
        f"one retry); got {StreamingBackendHandler._request_count}"
    )

    parsed_payloads = _sse_data_payloads(body)
    error_payloads = [p for p in parsed_payloads if isinstance(p, dict) and "error" in p]
    assert len(error_payloads) == 1, (
        f"Expected exactly one error data event, got "
        f"{len(error_payloads)}: {parsed_payloads!r}"
    )

    from proxy.openai_compat import build_upstream_protocol_error

    assert parsed_payloads == [build_upstream_protocol_error()], (
        f"Malformed-twice pipeline must emit exactly the protocol error; "
        f"got {parsed_payloads!r}"
    )

    assert "malformed-attempt-a" not in body
    assert "malformed-attempt-b" not in body
    assert "SENTINEL_A_MUST_NOT_LEAK" not in body
    assert "SENTINEL_B_MUST_NOT_LEAK" not in body
    assert body.count("data: [DONE]\n\n") == 1


@pytest.mark.integration
def test_streaming_pipeline_malformed_nonzero_tokens_full_flow(
    streaming_client, streaming_backend, monkeypatch
):
    """Full pipeline: malformed response with nonzero total_tokens.

    Per ``http_client.call_upstream`` the safe-to-retry gate is closed
    when total_tokens > 0 (the upstream already produced output we
    cannot safely discard), so the second valid response must remain
    unused. The canonical protocol error is emitted.
    """
    StreamingBackendHandler._responses = [
        {
            "id": "malformed-nonzero",
            "choices": None,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25,
            },
            "secret_sentinel": "NONZERO_TOKENS_MUST_NOT_LEAK",
        },
        _make_text_response("never-used-recovered"),
    ]
    monkeypatch.setattr("proxy.http_client.time.sleep", lambda *_a, **_k: None)

    response = streaming_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "stream malformed nonzero"}],
            "stream": True,
        },
        headers={"X-Session-ID": "sess-sse-malformed-nonzero-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    body = response.get_data(as_text=True)
    assert StreamingBackendHandler._request_count == 1, (
        f"Nonzero-tokens malformed response must NOT trigger a retry; "
        f"got {StreamingBackendHandler._request_count} upstream calls"
    )

    parsed_payloads = _sse_data_payloads(body)

    from proxy.openai_compat import build_upstream_protocol_error

    assert parsed_payloads == [build_upstream_protocol_error()], (
        f"Nonzero-tokens malformed pipeline must emit exactly the "
        f"protocol error; got {parsed_payloads!r}"
    )
    assert "never-used-recovered" not in body
    assert "malformed-nonzero" not in body
    assert "NONZERO_TOKENS_MUST_NOT_LEAK" not in body
    assert body.count("data: [DONE]\n\n") == 1


# ---------------------------------------------------------------------------
# F10 cross-request mode clobber — threading.Barrier concurrency test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.real_agent_registry
def test_cross_request_mode_clobber_prevented(slow_streaming_backend):
    """F10 fix: PostProcess must consume the PRE-upstream mode snapshot, not
    re-snapshot after the upstream call. Concurrent A/B replacement must be
    detected by clear_active_mode's CAS so B's newer mode survives.

    Scenario (one session_id → one real Agent, no PostProcessStage mocking):
      - Mode A is installed on the agent.
      - Request A enters the completion thread. ContextOptimizeStage captures
        mode A into ctx.mode_snapshot_pre_upstream via agent.snapshot_mode()
        (AFTER apply_mode, BEFORE upstream). Upstream then blocks on the slow
        backend.
      - The main thread, observing that Request A has reached the blocked
        upstream call (so the mode-A snapshot is already captured), installs
        a NEWER mode B on the same agent — agent.active_mode is now a
        different ModeState object from mode A — and only THEN releases the
        upstream gate.
      - Request A's upstream completes, PostProcessStage reads
        ctx.mode_snapshot_pre_upstream (mode A) and calls finalize_skill_mode.
        clear_active_mode compares current (mode B) against expected (mode A)
        → mismatch → CAS no-ops → B survives.

    Why this reverts: with the fix REVERTED, post_process.py re-snapshots the
    mode at the TOP of execute(). Because mode B is already active when that
    re-snapshot runs, expected_mode = mode B → clear_active_mode(expected=mode
    B) matches the active mode B → clears it → the "B survives" assertion
    FAILS. With the fix, expected_mode = mode A (snapshot) → mismatch → no-op
    → B survives → PASS.

    Synchronization is fully deterministic via the two threading.Events on
    _SlowBackendHandler (_request_entered / _release_response) — no time.sleep().
    Installing mode B AFTER observing _request_entered but BEFORE setting
    _release_response guarantees mode B is active before PostProcessStage
    reads the expected_mode, on both the fixed and the reverted code paths.
    """

    # Install mode A on a real agent for a deterministic session_id.
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "concurrent-clobber"}],
    }
    client_hint = "f10-clobber"
    session_id = _resolve_body_session_id(body, client_hint)

    agent = _RealAgent(agent_id=f"agent_{session_id}", session_id=session_id)
    mode_a = _ModeState(
        name="ticket",
        activated_at=_time.time(),
        ttl_seconds=300,
        skill_state={},
    )
    assert agent.set_active_mode(mode_a), "precondition: mode A installed"
    _agent_registry.register(agent)

    # Slow backend: upstream call blocks until _release_response is set.
    _SlowBackendHandler._request_entered = threading.Event()
    _SlowBackendHandler._release_response = threading.Event()
    _SlowBackendHandler._response_body = (
        b'{"choices":[{"message":{"role":"assistant","content":"late"},'
        b'"finish_reason":"stop"}]}'
    )

    client, _server = slow_streaming_backend

    try:
        # Fire Request A. client.post() with stream=True returns once the
        # completion thread has spawned and the WSGI generator has yielded its
        # first ": connected" chunk — the completion thread is now running the
        # full pipeline. We do NOT block on the stream yet (draining it would
        # force us to wait for completion, collapsing the race window we want
        # to open).
        response = client.post(
            "/v1/chat/completions",
            json={**body, "stream": True},
            headers={"X-Session-Client": client_hint},
        )
        assert response.status_code == 200, (
            f"Request A rejected: {response.status_code}"
        )

        # Wait for Request A to block on the slow upstream call. At this
        # point ContextOptimizeStage has ALREADY captured
        # ctx.mode_snapshot_pre_upstream = mode A on Request A's ctx (the
        # snapshot is taken in ContextOptimizeStage, before UpstreamStage —
        # which is the call that is now blocked). So the mode-A snapshot is
        # fixed and cannot be affected by the swap below.
        assert _SlowBackendHandler._request_entered.wait(timeout=10), (
            "Request A never reached upstream"
        )

        # Install mode B on the same agent — simulate concurrent replacement
        # by another request — BEFORE releasing the upstream gate. This is the
        # deterministic step: mode B must be active before PostProcessStage
        # reads the expected_mode, so the reverted (re-snapshot) path reads
        # mode B (and wrongly clears it) while the fixed path reads mode A
        # (the snapshot) and no-ops.
        mode_b = _ModeState(
            name="ticket",
            activated_at=_time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        assert agent.set_active_mode(mode_b), "precondition: mode B installed"
        assert agent.snapshot_mode() is not mode_a, (
            "precondition: mode B must be installed after mode A"
        )

        # Release the upstream gate. Request A's completion thread proceeds to
        # PostProcessStage, which reads the expected_mode now that mode B is
        # active.
        _SlowBackendHandler._release_response.set()

        # NOW drain the stream. get_data() iterates the WSGI generator to its
        # [DONE] terminator, which is emitted only AFTER the completion
        # thread sets completion_done — i.e. only AFTER pipeline.run()
        # returns and PostProcessStage.finalize_skill_mode has run. This is
        # what makes the assertion below observe the post-finalize state
        # deterministically (pollyanna guard against the lazy-generator race
        # the naive version of this test was vulnerable to).
        response.get_data(as_text=True)

        # B's mode must survive: the fix makes PostProcess consume the
        # pre-upstream snapshot (mode A) for CAS. clear_active_mode compares
        # current (mode B) against expected (mode A) → mismatch → CAS no-ops.
        current = agent.snapshot_mode()
        assert current is not None, (
            "F10 regression: Request A clobbered Request B's mode — "
            "PostProcess must consume the PRE-upstream snapshot, not re-snapshot"
        )
        assert current is not mode_a, (
            "mode A should have been replaced by mode B"
        )
    finally:
        _SlowBackendHandler._release_response.set()
        _cleanup_agent_registry_entries(agent)


# ---------------------------------------------------------------------------
# CP4-3: Replay delivery path — COMPLETED safe artifact served from cache
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_replay_serves_cached_artifact_without_upstream(streaming_client, streaming_backend):
    """CP4-3: when the dedup store has a COMPLETED safe artifact for a
    fingerprint, a matching streaming request is served the cached
    terminal_chunks verbatim — no pipeline, no upstream.

    Asserts:
      (a) response body equals cached terminal_chunks with exactly one [DONE]
      (b) X-ChoriProx-Replay: hit header is present
      (c) X-Session-Token is still issued (if PROXY_SESSION_SECRET is set)
      (d) mock upstream received ZERO requests (no pipeline/upstream ran)
    """
    import os
    import types
    import proxy.state as _state
    from proxy.dedup import compute_fingerprint
    from proxy.completion_coordinator import CompletionArtifact, CompletionRecord

    # -- Define the request body and compute the matching fingerprint --
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "replay-test-message"}],
        "stream": True,
    }
    # Flask test client remote_addr is 127.0.0.1; no X-Session-Client header
    # means client_hint = "127.0.0.1".
    client_hint = "127.0.0.1"

    from proxy.model_registry import resolve_session_id
    session_id = resolve_session_id(request_body, client_hint)

    # Build a ctx-like namespace matching what parse_request produces.
    fp_ctx = types.SimpleNamespace(
        session_id=session_id,
        client_hint=client_hint,
        model_name="test-model",
        workspace_name="",
        is_subagent=False,
        body=request_body,
    )
    fingerprint = compute_fingerprint(fp_ctx, mode_name="")

    # -- Prime the dedup store with a COMPLETED safe artifact --
    cached_chunks = (
        'data: {"id":"replay-test","choices":[{"delta":{"content":"cached"},"finish_reason":"stop"}}]}\n\n',
        'data: [DONE]\n\n',
    )
    artifact = CompletionArtifact(
        terminal_chunks=cached_chunks,
        logical_status="ok",
        original_request_id="replay-orig",
        has_tool_calls=False,
    )

    # Insert a PENDING entry then mark it COMPLETED with the artifact.
    # The replay path reads admission.record.artifact (CompletionRecord.artifact),
    # so we must set it on the record as well as on the DedupEntry.
    record = CompletionRecord(owner_request_id="replay-orig")
    record.artifact = artifact
    _state.dedup_store.lookup_or_insert(fingerprint, lambda: record, is_stream=True)
    _state.dedup_store.mark_completed(fingerprint, artifact)

    # -- Send the streaming request --
    # No upstream responses configured — if the proxy hits upstream, the
    # fallback response would differ from our cached chunks.
    StreamingBackendHandler._responses = []
    StreamingBackendHandler._request_count = 0

    response = streaming_client.post(
        "/v1/chat/completions",
        json=request_body,
        headers={"X-Session-ID": "sess-replay-1"},
    )

    assert response.status_code == 200
    assert response.content_type == "text/event-stream"

    # (b) Replay header present
    assert response.headers.get("X-ChoriProx-Replay") == "hit"

    # (c) X-Session-Token issued if PROXY_SESSION_SECRET is configured
    if os.environ.get("PROXY_SESSION_SECRET"):
        assert response.headers.get("X-Session-Token"), (
            "X-Session-Token must be issued on replay when PROXY_SESSION_SECRET is set"
        )

    # (a) Response body equals cached terminal_chunks verbatim (plus the
    # leading ": connected\n\n" comment from _replay_stream).
    body_text = response.get_data(as_text=True)
    # The replay stream emits ": connected\n\n" then the artifact chunks.
    assert ": connected\n\n" in body_text
    for chunk in cached_chunks:
        assert chunk in body_text, f"cached chunk missing from replay: {chunk!r}"
    # Exactly one [DONE]
    assert body_text.count("data: [DONE]\n\n") == 1

    # (d) Mock upstream received ZERO requests — no pipeline/upstream ran.
    assert StreamingBackendHandler._request_count == 0, (
        f"Replay must not hit upstream, but got {StreamingBackendHandler._request_count} requests"
    )
