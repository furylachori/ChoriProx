"""Integration tests for error propagation across stage boundaries and circuit breaker.

Covers:
  2.4 Error Propagation — AuthConfigError, ContextLimitError, StageAbort,
      cascade fallback, InfrastructureError during sync
  2.5 Circuit Breaker + Pipeline — Neo4j down, HALF_OPEN probe

Request flow:
  parse_request → CleanupStage → DispatchStage → RouteResolutionStage
  → SecurityStage → ContextOptimizeStage → MotivatorStage
  → ProviderCompatStage → UpstreamStage → PostProcessStage
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from memory.query_executor import InfrastructureError
from memory.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_upstream_response():
    """Patch _HTTP_SESSION.request to return a controlled upstream response."""
    from qwen_proxy import _HTTP_SESSION

    MARKER = "__PIPELINE_ERROR_RESPONSE__"

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
# Test 1: AuthConfigError → 401 response (2.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_auth_config_error_returns_401(client):
    """AuthConfigError from prepare_upstream_request → 401 with auth_error type.

    When the model's API key env var is missing, prepare_upstream_request raises
    AuthConfigError. The app catches it and returns 401.
    """
    from proxy.auth import AuthConfigError

    def raise_auth_error(*args, **kwargs):
        raise AuthConfigError("test-model", "MISSING_API_KEY")

    with patch(
        "proxy.pipeline.stage_modules.upstream.prepare_upstream_request",
        side_effect=raise_auth_error,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test auth error"}],
            },
        )

    assert resp.status_code == 401
    data = resp.get_json()
    assert "error" in data
    assert data.get("type") == "auth_error"


# ---------------------------------------------------------------------------
# Test 2: ContextLimitError → 413 response (2.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_limit_error_returns_413(client):
    """Context limit exceeded from enforce_context_limits → 413 via StageAbort.

    enforce_context_limits raises ValueError when context is too large.
    ContextOptimizeStage catches ValueError and raises StageAbort with 413.
    ContextLimitError (from context_v2.py) is also caught via the fallback path.
    # US-044
    """

    def raise_context_limit(*args, **kwargs):
        raise ValueError("Request too large: 99999 tokens exceeds model context limit of 32000 (single message, cannot trim)")

    with patch(
        "proxy.context.enforce_context_limits",
        side_effect=raise_context_limit,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test context limit"}],
            },
        )

    assert resp.status_code == 413
    data = resp.get_json()
    assert "error" in data
    error = data["error"]
    assert isinstance(error, dict)
    assert error.get("type") == "context_limit_exceeded"


# ---------------------------------------------------------------------------
# Test 3: StageAbort from SecurityStage → 400 response (2.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_security_stage_abort_returns_400(client):
    """SecurityStage raising StageAbort → 400 with security_violation type.

    When detect_malicious_content finds threats, SecurityStage raises
    StageAbort with a dict containing error.code=400 and type=security_violation.
    """
    from proxy.pipeline.exceptions import StageAbort
    from proxy.pipeline.stages import SecurityStage

    security_abort = StageAbort({
        "error": {
            "message": "Request blocked: malicious content detected (test_pattern)",
            "type": "security_violation",
            "code": 400,
        }
    })

    def raise_security_abort(self, ctx):
        raise security_abort

    with patch.object(SecurityStage, "execute", raise_security_abort):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test security abort"}],
            },
        )

    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
    error = data["error"]
    assert isinstance(error, dict)
    assert error.get("type") == "security_violation"


# ---------------------------------------------------------------------------
# Test 4: ContextOptimizeStage cascade fallback (2.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cascade_fallback_when_resolve_mode_fails(client, mock_upstream_response):
    """When resolve_mode() raises unexpectedly, cascade fallback fires.

    ContextOptimizeStage catches the exception from resolve_mode/apply_mode
    and falls through to _cascade_apply_default which applies DEFAULT mode.
    The pipeline should still return 200.
    """
    from proxy.pipeline.stages import ContextOptimizeStage

    original_execute = ContextOptimizeStage.execute
    call_count = [0]

    def failing_resolve_mode(*args, **kwargs):
        call_count[0] += 1
        raise RuntimeError("Unexpected error in resolve_mode")

    with patch(
        "proxy.mode.resolve_mode",
        side_effect=failing_resolve_mode,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test cascade fallback"}],
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == mock_upstream_response
    assert call_count[0] >= 1, "resolve_mode was not called"


@pytest.mark.integration
def test_cascade_fallback_minimal_messages(client, mock_upstream_response):
    """When both resolve_mode AND apply_mode fail, minimal fallback fires.

    _cascade_apply_default tries apply_mode(DEFAULT) — if that also fails,
    it builds a minimal context with system prompt + last message.
    The pipeline should still return 200.
    """
    from proxy.pipeline.stages import ContextOptimizeStage

    def always_fail(*args, **kwargs):
        raise RuntimeError("Simulated total failure")

    with patch(
        "proxy.mode.resolve_mode",
        side_effect=always_fail,
    ), patch(
        "proxy.mode.apply_mode",
        side_effect=always_fail,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "test minimal fallback"},
                ],
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data


@pytest.mark.integration
def test_cascade_fallback_empty_context_doesnt_crash(client, mock_upstream_response):
    """When everything fails and messages are empty, pipeline still returns 200.

    The minimal fallback builds ctx.body = {"messages": []} when even the
    fallback prompt import fails. The pipeline must not crash.
    """
    from proxy.pipeline.stages import ContextOptimizeStage

    def always_fail(*args, **kwargs):
        raise RuntimeError("Simulated total failure")

    with patch(
        "proxy.mode.resolve_mode",
        side_effect=always_fail,
    ), patch(
        "proxy.mode.apply_mode",
        side_effect=always_fail,
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["param"] == "messages"
    assert data["error"]["code"] == "invalid_messages"


# ---------------------------------------------------------------------------
# Test 5: InfrastructureError from Neo4j during PostProcessStage (2.4)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_infrastructure_error_during_sync_is_nonfatal(client, mock_upstream_response):
    """InfrastructureError during submit_sync is non-fatal.

    PostProcessStage catches all exceptions from submit_sync and logs them
    as warnings. The response content must be preserved.
    """
    with patch(
        "memory.sync.submit_sync",
        side_effect=InfrastructureError("Neo4j connection pool exhausted"),
    ), patch(
        "memory.db.touch_session",
        side_effect=InfrastructureError("Neo4j connection pool exhausted"),
    ), patch(
        "memory.db.store_messages_batch",
        side_effect=InfrastructureError("Neo4j connection pool exhausted"),
    ), patch(
        "proxy.session.touch_session",
        side_effect=InfrastructureError("Neo4j connection pool exhausted"),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test sync failure"}],
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response
    assert "error" not in data


# ---------------------------------------------------------------------------
# Test 6: Neo4j down → request still succeeds (2.5)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_neo4j_down_request_succeeds(client, mock_upstream_response):
    """When Neo4j is completely down, requests still succeed.

    The circuit breaker isolates Neo4j failures. Memory sync fails silently,
    but the upstream response is returned normally.
    """
    with patch(
        "memory.sync.submit_sync",
        side_effect=InfrastructureError("Connection refused"),
    ), patch(
        "memory.db.touch_session",
        side_effect=InfrastructureError("Connection refused"),
    ), patch(
        "memory.db.store_messages_batch",
        side_effect=InfrastructureError("Connection refused"),
    ), patch(
        "proxy.session.touch_session",
        side_effect=InfrastructureError("Connection refused"),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test neo4j down"}],
            },
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert content == mock_upstream_response


@pytest.mark.integration
def test_postprocess_step_failure_does_not_skip_next_steps(client, mock_upstream_response):
    """US-034: Failure in an early PostProcessStage step must not skip later steps.

    If _sync_chat_history raises, _persist_messages must still execute.
    PostProcessStage.execute() iterates methods and catches exceptions
    per-step, so a failure in step N must not abort step N+1.
    """
    from proxy.pipeline.post_process import PostProcessStage

    sync_called = {"value": False}
    persist_called = {"value": False}

    original_sync = PostProcessStage._sync_chat_history
    original_persist = PostProcessStage._persist_messages

    def failing_sync(self, ctx):
        sync_called["value"] = True
        raise RuntimeError("Simulated sync failure")

    def tracking_persist(self, ctx):
        persist_called["value"] = True
        return original_persist(self, ctx)

    with patch.object(PostProcessStage, "_sync_chat_history", failing_sync), \
         patch.object(PostProcessStage, "_persist_messages", tracking_persist):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "test step resilience"}],
            },
        )

    assert resp.status_code == 200
    assert sync_called["value"], "_sync_chat_history was not called"
    assert persist_called["value"], "_persist_messages must run even when _sync_chat_history fails"


# ---------------------------------------------------------------------------
# Test 7: Circuit breaker HALF_OPEN → exactly one probe (2.5)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_circuit_breaker_half_open_probe():
    """# US-002 In HALF_OPEN state, exactly one probe is allowed and others are rejected.

    The circuit breaker tracks _half_open_pending. When a probe is in-flight,
    subsequent can_execute() calls return False.
    """
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.01, name="test-probe")

    # Trip the circuit: 2 failures
    cb.on_failure(RuntimeError("fail 1"))
    cb.on_failure(RuntimeError("fail 2"))
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout to expire
    import time
    time.sleep(0.02)

    # First call after timeout → HALF_OPEN, probe allowed
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Second call while probe is pending → rejected
    assert cb.can_execute() is False

    # Probe succeeds → circuit closes
    cb.on_success()
    assert cb.state == CircuitState.CLOSED

    # Now requests flow again
    assert cb.can_execute() is True


@pytest.mark.integration
def test_circuit_breaker_half_open_probe_failure_reopens():
    """In HALF_OPEN state, probe failure reopens the circuit."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.01, name="test-reopen")

    # Trip the circuit
    cb.on_failure(RuntimeError("fail 1"))
    cb.on_failure(RuntimeError("fail 2"))
    assert cb.state == CircuitState.OPEN

    import time
    time.sleep(0.02)

    # Transition to HALF_OPEN
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Probe fails → circuit reopens
    cb.on_failure(RuntimeError("probe failed"))
    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 3


@pytest.mark.integration
def test_circuit_breaker_concurrent_probe_rejection():
    """Concurrent requests in HALF_OPEN: one probe, others rejected immediately.

    Simulates two threads racing to enter HALF_OPEN. The first gets the probe,
    the second gets CircuitBreakerOpen.
    """
    import threading

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.01, name="test-concurrent")

    # Trip the circuit
    cb.on_failure(RuntimeError("fail"))
    assert cb.state == CircuitState.OPEN

    import time
    time.sleep(0.02)

    results = {"probe": None, "rejected": None}
    barrier = threading.Barrier(2, timeout=5)

    def probe_thread():
        try:
            barrier.wait()
            if cb.can_execute():
                results["probe"] = "allowed"
            else:
                results["probe"] = "rejected"
        except Exception as e:
            results["probe"] = f"error: {e}"

    def second_thread():
        try:
            barrier.wait()
            # Small delay to ensure first thread gets the probe
            time.sleep(0.001)
            if cb.can_execute():
                results["rejected"] = "allowed"
            else:
                results["rejected"] = "rejected"
        except Exception as e:
            results["rejected"] = f"error: {e}"

    t1 = threading.Thread(target=probe_thread)
    t2 = threading.Thread(target=second_thread)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # One should be allowed (probe), the other rejected
    assert results["probe"] == "allowed"
    assert results["rejected"] == "rejected"

    # Probe succeeds → circuit closes
    cb.on_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.integration
def test_circuit_breaker_open_blocks_requests():
    """OPEN state blocks all requests with CircuitBreakerOpen."""
    cb = CircuitBreaker(failure_threshold=1, name="test-open-block")

    cb.on_failure(RuntimeError("fail"))
    assert cb.state == CircuitState.OPEN

    assert cb.can_execute() is False


@pytest.mark.integration
def test_circuit_breaker_closed_passes_through():
    """CLOSED state allows all requests through."""
    cb = CircuitBreaker(failure_threshold=3, name="test-closed")

    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True
    assert cb.can_execute() is True
    assert cb.can_execute() is True


@pytest.mark.integration
def test_circuit_breaker_decorator_blocks_when_open():
    """The circuit breaker decorator raises CircuitBreakerOpen when circuit is open."""
    cb = CircuitBreaker(failure_threshold=1, name="test-decorator")

    call_count = [0]

    @cb
    def protected_call():
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("first call fails")
        return "success"

    # First failure trips the circuit
    with pytest.raises(RuntimeError):
        protected_call()

    assert cb.state == CircuitState.OPEN

    # Now calls are blocked
    with pytest.raises(CircuitBreakerOpen):
        protected_call()
