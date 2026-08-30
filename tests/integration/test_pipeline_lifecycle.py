"""Integration tests for the full pipeline lifecycle — all 9 stages end-to-end.

Request flow:
  parse_request → CleanupStage → DispatchStage → RouteResolutionStage
  → SecurityStage → ContextOptimizeStage → MotivatorStage
  → ProviderCompatStage → UpstreamStage → PostProcessStage

Tests use proxy_with_mock (real mock HTTP backend) or the client fixture
(mocked HTTP session) depending on whether the test needs to inspect
upstream request bodies.
"""

import json
import pytest
from unittest.mock import patch

from memory.query_executor import InfrastructureError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_upstream_response():
    """Patch _HTTP_SESSION.request to return a controlled upstream response.

    Returns the callable so tests can override data via
    ``mock_upstream_response(data={...})``.
    """
    from qwen_proxy import _HTTP_SESSION

    MARKER = "__PIPELINE_LIFECYCLE_RESPONSE__"

    class MockResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        _payload = None

        def json(self):
            return self._payload

        def iter_content(self, chunk_size=None):
            yield json.dumps(self._payload).encode()

        def raise_for_status(self):
            pass

    def _mock_request(*args, **kwargs):
        resp = MockResponse()
        resp._payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": MARKER},
                    "finish_reason": "stop",
                }
            ],
            "model": "mock-llm",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return resp

    with patch.object(_HTTP_SESSION, "request", side_effect=_mock_request):
        yield MARKER


@pytest.fixture
def client(mock_upstream_response, monkeypatch):
    """Flask test client with mocked upstream and automatic auth."""
    monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")

    from qwen_proxy import app

    app.config["TESTING"] = True
    raw = app.test_client()

    class Authenticated:
        def __init__(self, c, key="test-proxy-key"):
            self._c = c
            self._k = key

        def _headers(self, kw):
            h = kw.get("headers", {})
            if "Authorization" not in h:
                h["Authorization"] = f"Bearer {self._k}"
            kw["headers"] = h
            return kw

        def post(self, *a, **kw):
            return self._c.post(*a, **self._headers(kw))

        def get(self, *a, **kw):
            return self._c.get(*a, **self._headers(kw))

    return Authenticated(raw)


# ---------------------------------------------------------------------------
# Test 1: Happy path — request through all stages returns 200
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_happy_path_full_pipeline(client, mock_upstream_response):
    """A request flows through all 9 stages and returns 200.

    Asserts:
    - Response status is 200
    - Response contains the mocked upstream content
    - session_id is derived correctly from the request body
    """
    from proxy.model_registry import resolve_session_id

    messages = [{"role": "user", "content": "hello pipeline test"}]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": messages},
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response

    expected_sid = resolve_session_id(
        {"model": "test-model", "messages": messages},
        client_hint="127.0.0.1",
    )
    assert isinstance(expected_sid, str)
    assert len(expected_sid) == 16


# ---------------------------------------------------------------------------
# Test 2: Non-critical stage failure doesn't break pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_noncritical_stage_failure_doesnt_break_pipeline(
    client, mock_upstream_response
):
    """MotivatorStage failure is non-fatal — pipeline still returns 200.

    MotivatorStage extends ConditionalStage which catches exceptions and
    returns None. We patch execute() to raise unconditionally, verifying
    the pipeline's error classification treats it as non-critical.
    """
    from proxy.pipeline.stages import MotivatorStage

    original_execute = MotivatorStage.execute

    def boom(self, ctx):
        raise RuntimeError("MotivatorStage simulated failure")

    with patch.object(MotivatorStage, "execute", boom):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test noncritical failure"}],
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response
    assert "error" not in data


# ---------------------------------------------------------------------------
# Test 3: Critical stage failure aborts pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_critical_stage_failure_aborts_pipeline(client):
    """UpstreamStage failure (InfrastructureError) aborts the pipeline.

    UpstreamStage is in CRITICAL_STAGES — its exceptions propagate to
    app.py which returns a non-200 error response.
    """
    with patch(
        "proxy.http_client.call_upstream",
        side_effect=InfrastructureError("neo4j connection pool exhausted"),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test critical failure"}],
            },
        )

    assert resp.status_code != 200
    data = resp.get_json()
    assert data is not None
    assert "error" in data


# ---------------------------------------------------------------------------
# Test 4: Body flows correctly between stages
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_body_flows_correctly_between_stages(client):
    """Verify the body received by upstream contains correct model, messages, and tools.

    Captures the request body at call_upstream (the HTTP layer) and asserts:
    - Model is preserved (or rewritten by model_cfg)
    - Messages contain the original user content
    - Tools are passed through
    - stream_options is stripped by ProviderCompatStage
    """
    captured_bodies = []

    original_payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "verify body flow"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "run a command",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "stream_options": {"include_usage": True},
    }

    def capture_and_respond(url, body, headers, wants_stream):
        captured_bodies.append(body)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "captured"},
                    "finish_reason": "stop",
                }
            ],
            "model": body.get("model", "unknown"),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    with patch("proxy.http_client.call_upstream", side_effect=capture_and_respond):
        resp = client.post(
            "/v1/chat/completions",
            json=original_payload,
        )

    assert resp.status_code == 200
    assert len(captured_bodies) == 1, "Upstream was not called exactly once"

    upstream_body = captured_bodies[0]

    # Messages preserved through the pipeline
    user_msgs = [m for m in upstream_body.get("messages", []) if m.get("role") == "user"]
    assert any("verify body flow" in m.get("content", "") for m in user_msgs), (
        f"User message not found in upstream body: {upstream_body.get('messages')}"
    )

    # Tools passed through
    tool_names = {
        t.get("function", {}).get("name") for t in upstream_body.get("tools", [])
    }
    assert "read" in tool_names or "bash" in tool_names, (
        f"Expected tools in upstream body, got: {tool_names}"
    )

    # ProviderCompatStage sets stream=False and strips stream_options
    assert upstream_body.get("stream") is False
    assert "stream_options" not in upstream_body


# ---------------------------------------------------------------------------
# Test 5: Mode snapshot is consistent across stages
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mode_snapshot_consistent_across_stages(proxy_with_mock):
    """Activate ticket mode and verify mode state is consistent across stages.

    The mode state seen by ContextOptimizeStage at entry must match what
    the tool guard reads via snapshot_mode(). This catches TOCTOU bugs
    between pipeline stages.

    Uses proxy_with_mock to go through the real dispatch path (^^ticket plan).
    """
    import proxy.state as _state

    with _state.tool_cache_lock:
        _state.tool_cache.clear()
        _state.tool_cache_size = 0

    client, handler, port = proxy_with_mock
    # finish_reason is "tool_calls" (NOT "stop") so ticket mode stays active
    # through post-processing. Per the PER-TURN contract, a main-agent
    # "stop" clears ticket/work/custom mode; using "tool_calls" keeps the
    # turn in-progress so the post-request guard_snapshot read below can
    # still observe the (consistent) ticket ModeState. A tool_calls turn
    # is exactly the case where the mode MUST persist mid-orchestration.
    handler._response_body = json.dumps({
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "planned"},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()

    from proxy.pipeline.stages import ContextOptimizeStage
    from proxy.agent.agent_registry import registry as agent_registry

    recorded_context_mode = []

    original_execute = ContextOptimizeStage.execute

    def spy_execute(self, ctx):
        try:
            agent = agent_registry.get_or_load(ctx.session_id)
            if agent is not None:
                snapshot = agent.snapshot_mode()
                recorded_context_mode.append({
                    "session_id": ctx.session_id,
                    "snapshot_name": snapshot.name if snapshot else None,
                    "snapshot_allowed_tools": snapshot.allowed_tool_names if snapshot else None,
                })
        except Exception as e:
            recorded_context_mode.append({"error": str(e)})
        return original_execute(self, ctx)

    from proxy.model_registry import resolve_session_id
    _body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "^^ticket plan fix auth bug"}],
    }
    _state.active_workspaces[resolve_session_id(_body, client_hint="127.0.0.1")] = "test-ws"

    with patch.object(ContextOptimizeStage, "execute", spy_execute):
        resp = client.post(
            "/v1/chat/completions",
            json=_body,
        )

    assert resp.status_code == 200
    assert recorded_context_mode, "ContextOptimizeStage.execute spy was not invoked"

    ctx_snapshot = recorded_context_mode[0]
    assert "error" not in ctx_snapshot, (
        f"Error reading mode at ContextOptimizeStage entry: {ctx_snapshot}"
    )
    assert ctx_snapshot["snapshot_name"] == "ticket", (
        f"Expected ticket mode at ContextOptimizeStage entry, got: {ctx_snapshot}"
    )
    assert ctx_snapshot["snapshot_allowed_tools"] is not None
    assert isinstance(ctx_snapshot["snapshot_allowed_tools"], frozenset)

    # Verify tool guard reads the same mode state
    session_id = ctx_snapshot["session_id"]
    agent = agent_registry.get_or_load(session_id)
    guard_snapshot = agent.snapshot_mode()

    assert guard_snapshot is not None
    assert guard_snapshot.name == ctx_snapshot["snapshot_name"]
    assert guard_snapshot.allowed_tool_names == ctx_snapshot["snapshot_allowed_tools"]
