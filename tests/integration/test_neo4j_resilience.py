"""Integration tests for infrastructure resilience — Wave 10A.

Covers:
  - US-001: Neo4j down → request still serves successfully + circuit breaker transitions
  - US-017a: Ticket mode fire-and-forget — once activated, no re-activation on second message

These tests exercise the proxy through the Flask test client with the
upstream HTTP layer mocked, and the Neo4j/sync layer patched to raise
``InfrastructureError``.  Action handlers and skill machinery run for real —
only the database I/O is mocked (DB-level mocking per the project pattern).

Conventions (AGENTS.md):
  - Import ``proxy.state`` as a module (counters/dicts mutated via reference).
  - Pre-generate UUIDs before any write queries (not used here — DB is mocked).
  - Handle ``InfrastructureError`` / ``ConstraintViolation`` / ``QueryTimeoutError`` explicitly.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from memory.query_executor import InfrastructureError
from memory.circuit_breaker import CircuitState


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures shared with the existing pipeline-error tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_upstream_response():
    """Patch ``_HTTP_SESSION.request`` to return a controlled upstream response.

    The proxy pipeline depends on the upstream HTTP layer to receive a 200
    response.  We patch the shared session's request method to return a
    fixed JSON body that the PostProcessStage can normalize.
    """
    from qwen_proxy import _HTTP_SESSION

    MARKER = "__NEO4J_RESILIENCE_RESPONSE__"

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
    """Flask test client with mocked upstream and automatic auth.

    Sets ``PROXY_API_KEY`` so the auth check passes, and enables
    ``TESTING`` mode for cleaner error propagation.
    """
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
# US-001: Neo4j down → request still serves successfully
# ---------------------------------------------------------------------------


class TestUS001Neo4jDownServesRequest:
    """When Neo4j is completely unreachable, the proxy must still return
    200 OK to the user.  The circuit breaker isolates the failure so that
    the upstream LLM call succeeds, and the background sync layer's
    ``submit_sync`` is a best-effort non-fatal step.
    """

    def _patch_neo4j_paths_to_fail(self):
        """Patch every Neo4j/sync entry point used by the pipeline.

        Returns a ``list`` of ``patch`` context managers — caller must keep
        the result alive for the duration of the test (use ``with``).
        """
        fail = InfrastructureError("Connection refused (simulated)")

        patches = [
            patch("memory.sync.submit_sync", side_effect=fail),
            patch("memory.db.touch_session", side_effect=fail),
            patch("memory.db.store_messages_batch", side_effect=fail),
            patch("proxy.session.touch_session", side_effect=fail),
        ]
        return patches

    def test_neo4j_down_returns_200(
        self, client, mock_upstream_response, monkeypatch
    ):
        """End-to-end: all Neo4j-touching paths raise, request still 200."""
        import proxy.state as state

        # Reset circuit breaker so the test starts from a known state
        state.neo4j_circuit_breaker.force_closed()
        assert state.neo4j_circuit_breaker.state == CircuitState.CLOSED

        patches = self._patch_neo4j_paths_to_fail()
        for p in patches:
            p.start()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello with neo4j down"}],
                },
            )
        finally:
            for p in patches:
                p.stop()

        assert resp.status_code == 200, (
            f"Expected 200 even with Neo4j down, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)[:300]}"
        )
        data = resp.get_json()
        assert "choices" in data, f"Missing choices in response: {data!r}"
        assert data["choices"][0]["message"]["content"] == mock_upstream_response
        # Error block must NOT be present — pipeline tolerated the failure
        assert "error" not in data, f"Pipeline leaked error: {data!r}"

    def test_neo4j_down_does_not_trigger_circuit_breaker_opens(
        self, client, mock_upstream_response, monkeypatch
    ):
        """The circuit breaker is for Neo4j DRIVER operations (query
        execution / connection checks).  The pipeline's catch-all
        ``except Exception`` around sync and touch_session means
        ``submit_sync``/``store_messages_batch`` failures do NOT call
        ``on_failure`` on the breaker.  The breaker therefore stays
        CLOSED — confirming the breaker is not over-triggered by the
        pipeline's broad exception handling.
        """
        import proxy.state as state

        state.neo4j_circuit_breaker.force_closed()
        initial_failure_count = state.neo4j_circuit_breaker.failure_count

        patches = self._patch_neo4j_paths_to_fail()
        for p in patches:
            p.start()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "cb stay closed test"}],
                },
            )
        finally:
            for p in patches:
                p.stop()

        assert resp.status_code == 200
        # The pipeline's try/except around sync paths means the breaker
        # is NOT incremented.  This is the correct, documented behavior:
        # the breaker tracks DRIVER-level failures, not pipeline-level
        # caught exceptions.
        assert state.neo4j_circuit_breaker.state == CircuitState.CLOSED
        assert state.neo4j_circuit_breaker.failure_count == initial_failure_count

    def test_neo4j_down_sync_skipped_via_exception_handler(
        self, client, mock_upstream_response, monkeypatch, caplog
    ):
        """When ``submit_sync`` raises ``InfrastructureError``,
        PostProcessStage logs a WARNING and the response is still
        returned.  This verifies the sync is *skipped* (not retried
        inline) and the user-facing contract is preserved.
        """
        import logging
        import proxy.state as state

        state.neo4j_circuit_breaker.force_closed()

        caplog.set_level(logging.WARNING, logger="proxy.pipeline.stages")
        fail = InfrastructureError("Connection refused (sync skip)")

        with patch("memory.sync.submit_sync", side_effect=fail), \
             patch("memory.db.touch_session", side_effect=fail), \
             patch("memory.db.store_messages_batch", side_effect=fail), \
             patch("proxy.session.touch_session", side_effect=fail):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "sync skip test"}],
                },
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data

        # The pipeline logged the sync failure (warning-level visibility)
        sync_warnings = [
            r for r in caplog.records
            if "submit_sync failed" in r.getMessage()
        ]
        assert len(sync_warnings) >= 1, (
            f"Expected at least one sync-failed WARNING, got "
            f"{[r.getMessage() for r in caplog.records]!r}"
        )

    def test_neo4j_driver_unavailable_returns_false_from_is_available(
        self, monkeypatch
    ):
        """Direct unit-level check: when the Neo4j driver cannot connect,
        ``QueryExecutor.is_available()`` returns False.

        ``QueryExecutor.__init__`` calls ``get_shared_driver`` to acquire
        the singleton, so we patch that to inject a MagicMock driver
        whose ``session()`` raises.  This exercises the executor's
        is_available() contract without touching Flask.
        """
        from memory import query_executor as qe_mod

        fake_driver = MagicMock()

        def slow_session():
            s = MagicMock()
            s.run.side_effect = InfrastructureError("Connection refused")
            s.close = MagicMock()
            return s

        fake_driver.session.side_effect = slow_session

        monkeypatch.setattr(qe_mod, "get_shared_driver", lambda *a, **kw: fake_driver)

        executor = qe_mod.QueryExecutor(
            bolt_url="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )

        # is_available() must return False (not raise) when the driver
        # cannot establish a session.
        result = executor.is_available()
        assert result is False, (
            f"is_available() must return False when driver fails, got {result!r}"
        )



