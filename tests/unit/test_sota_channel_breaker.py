"""M5-2: SOTA channel must trip the circuit breaker on persistent 429.

Pre-fix: the 429 branch in SOTAChannel.ask() did
`time.sleep(retry_after); continue` without calling
`_sota_breaker.on_failure()`. A persistent 429 stream (e.g. upstream
rate limit) would burn 5 × Retry-After sleeps per call before raising,
with the breaker staying CLOSED the whole time. 5 minutes of wasted
requests per call is exactly what the breaker is supposed to prevent.

Fix: call `_sota_breaker.on_failure()` on EVERY 429 (not just the
terminal attempt), so the breaker actually accumulates failure count
and trips after MAX_RETRIES consecutive 429s. The next call after
tripping raises `CircuitBreakerOpen` immediately instead of paying
the full backoff.
"""

import time
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


class _StubResponse:
    """Mimics an openai API response that raises a 429 on every call."""
    def __init__(self, status_code: int, headers: dict = None, body: str = ""):
        from openai import OpenAIAPIError
        err = OpenAIAPIError(
            message=f"upstream returned {status_code}",
            request=MagicMock(),
            body=None,
        )
        err.status_code = status_code
        err.response = MagicMock()
        err.response.headers = headers or {}
        self._err = err

    def __getattr__(self, name):
        raise self._err


def _make_429_response():
    """Build a stub that raises a 429 with Retry-After: 1 on every call."""
    from openai import APIError as OpenAIAPIError
    err = OpenAIAPIError(
        message="upstream returned 429",
        request=MagicMock(),
        body=None,
    )
    err.status_code = 429
    err.response = MagicMock()
    err.response.headers = {"Retry-After": "1"}

    def _raise(*args, **kwargs):
        raise err

    return _raise


def _build_channel_with_stub(stub_factory):
    """Build a SOTAChannel whose internal client raises the stub on every call.

    Mocks the upstream `client.chat.completions.create` so every attempt
    raises the supplied 429 response.
    """
    from memory.sota_channel import SOTAChannel

    fake_client = MagicMock()
    fake_client.timeout = 30
    fake_client.chat.completions.create.side_effect = stub_factory

    with patch("memory._sota_http.get_sota_client", return_value=fake_client):
        channel = SOTAChannel(
            mode=MagicMock(),  # SOTAMode (enum-like)
            system_prompt="sys",
            prompt="test",
        )
    return channel, fake_client


def test_persistent_429_trips_sota_circuit_breaker():
    """Steady 429 stream: breaker should accumulate failures on every 429.

    With the fix, _sota_breaker.on_failure() is called once per 429
    attempt (5 times across MAX_ATTEMPTS=5 attempts). After 5
    consecutive 429s, the breaker is OPEN. The next call to
    SOTAChannel.ask() raises CircuitBreakerOpen immediately without
    calling the upstream.
    """
    from memory.sota_channel import SOTAChannel, SOTARateLimitError, _sota_breaker
    from memory.circuit_breaker import CircuitBreakerOpen, CircuitState
    from memory._sota_http import SOTAMode

    # Reset breaker state for a clean test. The breaker has no public
    # reset() — directly manipulate private fields under the lock.
    with _sota_breaker._lock:
        _sota_breaker._state = CircuitState.CLOSED
        _sota_breaker._failure_count = 0
        _sota_breaker._last_failure_time = 0.0
        _sota_breaker._half_open_pending = False

    # Stub the upstream to raise 429 on every call.
    fake_client = MagicMock()
    fake_client.timeout = 30
    fake_client.chat.completions.create.side_effect = _make_429_response()

    start = time.monotonic()
    with patch("memory._sota_http.get_sota_client", return_value=fake_client):
        with patch("memory._sota_rate_limiter.claim_slot"):
            channel = SOTAChannel(mode=SOTAMode.SUMMARIZATION, system_prompt="sys")
            with pytest.raises(SOTARateLimitError):
                channel.ask("hello", session_id="test-session-1")
    elapsed = time.monotonic() - start

    # The 5 attempts should each have called on_failure() on the
    # module-level breaker. MAX_ATTEMPTS=5 in SOTAChannel.
    assert _sota_breaker.failure_count == 5, (
        f"Expected 5 on_failure() calls (one per 429 attempt), "
        f"got {_sota_breaker.failure_count}. The breaker never accumulated "
        f"the failure count to trip after MAX_RETRIES consecutive 429s."
    )

    # The breaker should be OPEN now (5 failures at threshold 5).
    assert _sota_breaker.state.name == "OPEN", (
        f"Expected breaker state OPEN after 5 consecutive 429s, "
        f"got {_sota_breaker.state.name}"
    )

    # Total elapsed time should be small: 5 × 1s (Retry-After: 1) + minimal
    # other overhead. With the FIX, the 5th call triggers on_failure
    # which opens the breaker, so the next caller sees CircuitBreakerOpen
    # immediately rather than waiting another 5× Retry-After. The test
    # only exercises the first call (which raises SOTARateLimitError
    # after MAX_ATTEMPTS), so we measure that 5 × 1s elapsed is in
    # the expected range, not 5 × 60s.
    assert elapsed < 30, (
        f"Test took {elapsed:.1f}s — too slow. With Retry-After: 1 and "
        f"5 attempts, expect ~5-10s. If ~5 minutes, the retry-after "
        f"sleeps are not the 1s we stubbed."
    )

    # Subsequent call: the breaker is OPEN so the call raises
    # CircuitBreakerOpen immediately (no upstream call).
    fake_client.chat.completions.create.reset_mock()
    with patch("memory._sota_http.get_sota_client", return_value=fake_client):
        with patch("memory._sota_rate_limiter.claim_slot"):
            channel2 = SOTAChannel(mode=SOTAMode.SUMMARIZATION, system_prompt="sys")
            with pytest.raises(CircuitBreakerOpen):
                channel2.ask("hello", session_id="test-session-2")
    # Upstream was NOT called — the breaker short-circuited.
    assert fake_client.chat.completions.create.call_count == 0, (
        "Upstream was called despite OPEN breaker — the 429 stream "
        "should have been short-circuited."
    )
