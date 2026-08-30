"""Integration tests for tool interception → re-query → final response cycle.

Tests the boundary between upstream LLM response and tool execution:
- Memory tools (recall, search) are intercepted and executed locally
- Passthrough tools (read, bash) are returned to the client
- After tool execution, the pipeline re-queries the LLM with tool results
- The 69-message cap is enforced during tool loops
- Orphaned tool results are cleaned up before re-query
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ToolLoopBackendHandler(BaseHTTPRequestHandler):
    """Mock backend that returns configurable sequential responses.

    _responses: list of response dicts to return in order.
    _request_count: auto-incremented on each POST.
    _captured_bodies: list of parsed request bodies.
    """

    _responses: list = []
    _request_count: int = 0
    _captured_bodies: list = []
    _captured_headers: list = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        ToolLoopBackendHandler._captured_bodies.append(
            json.loads(raw) if raw else {}
        )
        ToolLoopBackendHandler._captured_headers.append(dict(self.headers))

        idx = ToolLoopBackendHandler._request_count
        ToolLoopBackendHandler._request_count += 1

        if idx < len(ToolLoopBackendHandler._responses):
            payload = ToolLoopBackendHandler._responses[idx]
        else:
            payload = {
                "choices": [{
                    "message": {"role": "assistant", "content": "fallback"},
                    "finish_reason": "stop",
                }],
                "model": "mock-llm",
            }

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def tool_loop_backend():
    """Start a mock backend server for tool interception tests."""
    server = HTTPServer(("127.0.0.1", 0), ToolLoopBackendHandler)
    port = server.server_address[1]

    ToolLoopBackendHandler._responses = []
    ToolLoopBackendHandler._request_count = 0
    ToolLoopBackendHandler._captured_bodies = []
    ToolLoopBackendHandler._captured_headers = []

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield server, port

    server.shutdown()


@pytest.fixture
def tool_loop_client(tool_loop_backend, tmp_path, monkeypatch):
    """Flask test client wired to the mock backend for tool interception tests."""
    from clients.model_registry import ModelRegistry
    import requests as _requests

    server, port = tool_loop_backend
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


def _make_tool_call_response(tool_calls, content="", finish_reason="tool_calls"):
    """Build an OpenAI-format response with tool_calls."""
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


def _make_text_response(content, finish_reason="stop"):
    """Build an OpenAI-format response with text content only."""
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "model": "test/llm-v1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _openai_tool_call(tool_id, name, arguments):
    """Build an OpenAI-format tool_call dict."""
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


# ---------------------------------------------------------------------------
# Test 1: Memory tool interception — tool executed locally, re-query happens
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_memory_tool_intercepted_and_requeried(tool_loop_client, tool_loop_backend):
    """When upstream returns a recall tool_call, the proxy intercepts it,
    executes locally, and re-queries the LLM with the tool result.
    The final response has finish_reason='stop'.
    """
    server, port = tool_loop_backend

    recall_tc = _openai_tool_call("call_recall_001", "recall", {"query": "test"})

    ToolLoopBackendHandler._responses = [
        _make_tool_call_response([recall_tc]),
        _make_text_response("Here is what I found in memory."),
    ]

    with patch("memory.tools.execute_memory_tool") as mock_exec:
        mock_exec.return_value = {
            "results": [{"content": "remembered fact", "score": 0.95}],
        }
        resp = tool_loop_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What do you remember?"}],
                "stream": False,
            },
            headers={"X-Session-ID": "sess-memory-001"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data, f"Expected choices in response: {data}"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "memory" in data["choices"][0]["message"]["content"].lower()

    assert mock_exec.call_count == 1
    assert mock_exec.call_args[0][0] == "recall"
    # Show tool result in re-query
    requery_msgs = ToolLoopBackendHandler._captured_bodies[1].get("messages", [])
    tool_results = [m for m in requery_msgs if m.get("role") == "tool"]

    assert ToolLoopBackendHandler._request_count == 2
    requery_body = ToolLoopBackendHandler._captured_bodies[1]
    requery_msgs = requery_body.get("messages", [])
    tool_results = [m for m in requery_msgs if m.get("role") == "tool"]
    assert len(tool_results) >= 1, "Re-query must include tool result messages"
    assert "remembered fact" in tool_results[0].get("content", "")


# ---------------------------------------------------------------------------
# Test 2: Passthrough tool — tool returned to client
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_passthrough_tool_returned_to_client(tool_loop_client, tool_loop_backend):
    """When upstream returns a non-memory tool_call (read), it is returned
    to the client as-is. No re-query happens.
    """
    server, port = tool_loop_backend

    read_tc = _openai_tool_call("call_read_001", "read", {"file_path": "/tmp/test.py"})

    ToolLoopBackendHandler._responses = [
        _make_tool_call_response([read_tc]),
    ]

    resp = tool_loop_client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Read the file"}],
            "stream": False,
        },
        headers={"X-Session-ID": "sess-passthrough-001"},
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data

    msg = data["choices"][0]["message"]
    assert msg.get("tool_calls"), "Response must contain tool_calls for passthrough"
    tc_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
    assert "read" in tc_names

    assert data["choices"][0]["finish_reason"] == "tool_calls"

    assert ToolLoopBackendHandler._request_count == 1, \
        "No re-query should happen for passthrough-only tools"


# ---------------------------------------------------------------------------
# Test 3: Mixed tools — memory intercepted, passthrough returned
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mixed_tools_memory_intercepted_passthrough_returned(tool_loop_client, tool_loop_backend):
    """When upstream returns both recall (memory) and read (passthrough),
    recall is intercepted and executed locally, read is returned to the client,
    and re-query happens with the recall result.
    """
    server, port = tool_loop_backend

    recall_tc = _openai_tool_call("call_recall_mix", "recall", {"query": "mixed test"})
    read_tc = _openai_tool_call("call_read_mix", "read", {"file_path": "/tmp/mix.py"})

    ToolLoopBackendHandler._responses = [
        _make_tool_call_response([recall_tc, read_tc]),
        _make_text_response("Based on memory and the file..."),
    ]

    with patch("memory.tools.execute_memory_tool") as mock_exec:
        mock_exec.return_value = {
            "results": [{"content": "mixed memory result", "score": 0.88}],
        }
        resp = tool_loop_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Check memory and read file"}],
                "stream": False,
            },
            headers={"X-Session-ID": "sess-mixed-001"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data

    finish = data["choices"][0]["finish_reason"]
    msg = data["choices"][0]["message"]

    if finish == "stop":
        assert mock_exec.call_count == 1, "recall should be intercepted"
        assert ToolLoopBackendHandler._request_count == 2, "Re-query should happen"
        requery_body = ToolLoopBackendHandler._captured_bodies[1]
        requery_msgs = requery_body.get("messages", [])
        tool_results = [m for m in requery_msgs if m.get("role") == "tool"]
        assert len(tool_results) >= 1, "Re-query must include recall result"
        assert "mixed memory result" in tool_results[0].get("content", "")
    elif finish == "tool_calls":
        assert msg.get("tool_calls"), "Passthrough tool calls must be present"
        tc_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
        assert "read" in tc_names
    else:
        pytest.fail(f"Unexpected finish_reason: {finish}")


# ---------------------------------------------------------------------------
# Test 4: Tool loop message cap (69 messages)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tool_loop_message_cap_enforced(tool_loop_client, tool_loop_backend):
    """US-023: When message history has 70+ messages and tool calls trigger re-queries,
    the 69-message cap is enforced on every upstream call.
    """
    server, port = tool_loop_backend

    large_messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(35):
        large_messages.append({"role": "user", "content": f"Question {i}"})
        large_messages.append({"role": "assistant", "content": f"Answer {i}"})

    recall_tc = _openai_tool_call("call_cap_001", "recall", {"query": "cap test"})

    ToolLoopBackendHandler._responses = [
        _make_tool_call_response([recall_tc]),
        _make_text_response("Final answer after recall."),
    ]

    with patch("memory.tools.execute_memory_tool") as mock_exec:
        mock_exec.return_value = {"results": [{"content": "cap result", "score": 0.9}]}
        resp = tool_loop_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": large_messages,
                "stream": False,
            },
            headers={"X-Session-ID": "sess-cap-001"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["choices"][0]["finish_reason"] == "stop"

    assert ToolLoopBackendHandler._request_count == 2
    requery_body = ToolLoopBackendHandler._captured_bodies[1]
    requery_msgs = requery_body.get("messages", [])
    assert len(requery_msgs) <= 69, \
        f"Re-query messages must be <= 69, got {len(requery_msgs)}"


# ---------------------------------------------------------------------------
# Test 5: Orphaned tool result cleanup
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_orphaned_tool_results_cleaned_before_requery(tool_loop_client, tool_loop_backend):
    """When the message history contains tool results whose tool_call_id
    doesn't match any assistant tool_calls (orphans from trimming),
    they are cleaned up before re-query — preventing 400 errors from upstream.
    """
    server, port = tool_loop_backend

    orphan_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_openai_tool_call("tc_real_001", "recall", {"query": "real"})],
        },
        {"role": "tool", "tool_call_id": "tc_real_001", "content": '{"results":[]}'},
        {"role": "tool", "tool_call_id": "tc_orphan_999", "content": '{"results":[]}'},
        {"role": "assistant", "content": "Previous answer."},
        {"role": "user", "content": "Follow up question"},
    ]

    recall_tc = _openai_tool_call("call_orphan_001", "recall", {"query": "orphan test"})

    ToolLoopBackendHandler._responses = [
        _make_tool_call_response([recall_tc]),
        _make_text_response("Clean response after orphan removal."),
    ]

    with patch("memory.tools.execute_memory_tool") as mock_exec:
        mock_exec.return_value = {"results": [{"content": "orphan result", "score": 0.85}]}
        resp = tool_loop_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": orphan_messages,
                "stream": False,
            },
            headers={"X-Session-ID": "sess-orphan-001"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data

    assert ToolLoopBackendHandler._request_count == 2, \
        "Re-query should succeed (orphans cleaned, no 400 error)"

    requery_body = ToolLoopBackendHandler._captured_bodies[1]
    requery_msgs = requery_body.get("messages", [])
    for msg in requery_msgs:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            has_matching_assistant = any(
                m.get("role") == "assistant"
                and isinstance(m.get("tool_calls"), list)
                and any(tc.get("id") == tc_id for tc in m["tool_calls"])
                for m in requery_msgs
            )
            if msg.get("tool_call_id") != "tc_orphan_999":
                assert has_matching_assistant, \
                    f"Tool result {tc_id} must have matching assistant tool_call"

    tool_results_in_requery = [
        m for m in requery_msgs if m.get("role") == "tool"
    ]
    orphan_ids = {m.get("tool_call_id") for m in tool_results_in_requery}
    assert "tc_orphan_999" not in orphan_ids, \
        "Orphaned tool result tc_orphan_999 must be cleaned up before re-query"
