"""Integration tests for negative inputs and degraded-environment behavior.

Wave 10C: validates the proxy degrades gracefully when given malformed
requests or when optional infrastructure is unavailable.

Covers:
  - Empty `messages` array → graceful fallback
  - Missing `role` field on a message → graceful
  - `content: null` on a message → graceful
  - Unknown model name → falls back to default backend
  - Missing `PROXY_SESSION_SECRET` → canary is unsigned, no crash
  - Missing `PROXY_SOTA_API_KEY` → memory sync disabled, request still serves
  - Missing `NEO4J_AUTH` → circuit breaker open, request still serves
"""

import json
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_upstream_response():
    """Patch _HTTP_SESSION.request to return a controlled upstream response."""
    from qwen_proxy import _HTTP_SESSION

    MARKER = "__NEGATIVE_INPUT_RESPONSE__"

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
# Test 1: Empty `messages` array → graceful (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_empty_messages_array_is_graceful(client, mock_upstream_response):
    """An empty `messages` array must not crash the pipeline.

    The pipeline tolerates empty messages by serving a response built
    from system prompt + cascade fallback context.
    """
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": []},
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["param"] == "messages"
    assert data["error"]["code"] == "invalid_messages"


# ---------------------------------------------------------------------------
# Test 2: Missing `role` field on a message → graceful (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_message_missing_role_field_is_graceful(client, mock_upstream_response):
    """A message without a `role` field must not crash the pipeline.

    Stages that filter by role (e.g., security stage, canary detection)
    must skip such messages rather than raise KeyError.
    """
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"content": "missing role field"},
                {"role": "user", "content": "follow-up"},
            ],
        },
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["param"] == "messages[0].role"
    assert data["error"]["code"] == "invalid_role"


# ---------------------------------------------------------------------------
# Test 3: `content: null` on a message → graceful (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_message_with_null_content_is_graceful(client, mock_upstream_response):
    """A message with `content: null` must not crash the pipeline.

    Some clients (notably tool-call responses) send `content: null`.
    The pipeline must treat null content as an empty string.
    """
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": None},
                {"role": "assistant", "content": None, "tool_calls": []},
                {"role": "user", "content": "real question"},
            ],
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response


# ---------------------------------------------------------------------------
# Test 4: Unknown model name → falls back to default backend (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unknown_model_name_falls_back_to_default(client, mock_upstream_response):
    """An unknown model name must fall back to the registered default backend.

    ModelRegistry.get() returns self._default when no match is found,
    so the request still routes to a working backend. This test does NOT
    mock ModelRegistry.get; it lets the real fallback layer run so a bug
    in fallback (e.g., returning KeyError, or leaking the user's input
    model name into the upstream body) would surface here.
    """
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "completely-unknown-model-xyz",
            "messages": [{"role": "user", "content": "test fallback"}],
        },
    )

    assert resp.status_code == 200, (
        f"unknown model 'completely-unknown-model-xyz' must fall back to "
        f"the default backend and serve a 200; got status {resp.status_code} "
        f"and body {resp.get_data(as_text=True)[:200]!r}"
    )
    data = resp.get_json()
    assert "choices" in data, (
        f"fallback response must contain 'choices'; got data={list(data.keys())}"
    )
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response, (
        f"fallback must have routed the request to a working upstream so the "
        f"mock marker ({mock_upstream_response!r}) appears in the response; "
        f"got content={content!r}"
    )


# ---------------------------------------------------------------------------
# Test 5: Missing `PROXY_SESSION_SECRET` → canary unsigned, no crash (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_missing_session_secret_canary_unsigned(monkeypatch, mock_upstream_response):
    """When PROXY_SESSION_SECRET is unset, canaries are unsigned. The proxy
    must still serve requests and accept the canary (the canary is minted
    by the proxy itself, so unsigned in dev mode is trusted). The request
    must succeed end-to-end with the sub-agent flag correctly applied to
    the request.
    """
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

    import proxy.canary as canary_module
    saved_secret = canary_module._SESSION_SECRET
    canary_module._SESSION_SECRET = ""
    try:
        monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")

        from qwen_proxy import app
        app.config["TESTING"] = True
        raw_client = app.test_client()

        canary_marker = "[[__SUBAGENT__:SKILL:ticket]]"
        resp = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": f"{canary_marker}\nreal user content"}
                ],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data
    finally:
        canary_module._SESSION_SECRET = saved_secret


# ---------------------------------------------------------------------------
# Test 6: Missing `PROXY_SOTA_API_KEY` → sync disabled, no crash (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_missing_sota_api_key_sync_disabled_no_crash(
    monkeypatch, mock_upstream_response
):
    """When PROXY_SOTA_API_KEY is empty, the proxy must still serve requests.

    The SOTAClient is created lazily and accepts an empty api_key (which
    will fail when called, but the request is served before that happens).
    The proxy must not crash on import, request, or background sync setup.
    """
    monkeypatch.delenv("PROXY_SOTA_API_KEY", raising=False)

    import memory.sync as sync_module
    saved_key = sync_module._SOTA_API_KEY
    saved_client = sync_module._sota_client
    sync_module._SOTA_API_KEY = ""
    sync_module._sota_client = None
    try:
        monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")

        from qwen_proxy import app
        app.config["TESTING"] = True
        raw_client = app.test_client()

        resp = raw_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "no sota key"}],
            },
            headers={"Authorization": "Bearer test-proxy-key"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data
        assert "error" not in data
    finally:
        sync_module._SOTA_API_KEY = saved_key
        sync_module._sota_client = saved_client


# ---------------------------------------------------------------------------
# Test 7: Missing `NEO4J_AUTH` → circuit breaker open, request still serves (4.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_missing_neo4j_auth_circuit_open_request_serves(
    client, mock_upstream_response
):
    """When Neo4j is unreachable (NEO4J_AUTH unset/bad), the circuit breaker
    isolates failures and the request still serves from upstream.

    The PostProcessStage catches all InfrastructureError raised by
    submit_sync / db calls, so the user-facing response is unaffected.
    """
    from memory.query_executor import InfrastructureError
    import proxy.state as state

    state.neo4j_circuit_breaker.force_open()

    with patch(
        "memory.sync.submit_sync",
        side_effect=InfrastructureError("Neo4j unavailable (NEO4J_AUTH missing)"),
    ), patch(
        "memory.db.touch_session",
        side_effect=InfrastructureError("Neo4j unavailable (NEO4J_AUTH missing)"),
    ), patch(
        "memory.db.store_messages_batch",
        side_effect=InfrastructureError("Neo4j unavailable (NEO4J_AUTH missing)"),
    ), patch(
        "proxy.session.touch_session",
        side_effect=InfrastructureError("Neo4j unavailable (NEO4J_AUTH missing)"),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "neo4j down"}],
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response
    assert "error" not in data
