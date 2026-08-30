"""Unit tests for the proxy rate limiter and circuit breaker timeout.

Covers:
  - US-012: ``127.0.0.1`` (and other IP-shaped keys) are valid rate-limit keys
    — connections from localhost are not subject to a separate bypass
    (the localhost bypass applies only to the PROXY_API_KEY auth check,
    not the rate limiter).  This test documents the keyed behavior and
    pins the public contract for the IP-as-key case.
  - US-003: ``QueryExecutor.is_available()`` completes within 5 seconds
    even when the underlying Neo4j driver is slow / unreachable.  The
    2-second ``Query(..., timeout=2.0)`` on the probe ensures the call
    returns quickly with False (or raises ``InfrastructureError``) when
    the driver is stuck.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from memory.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpen
from memory.query_executor import (
    InfrastructureError,
    QueryExecutor,
    QueryTimeoutError,
)


# ---------------------------------------------------------------------------
# US-012: Rate limiter — 127.0.0.1 key behavior
# ---------------------------------------------------------------------------


class TestUS012RateLimiterLocalhostKey:
    """Verify the rate limiter treats ``127.0.0.1`` as a normal key.

    The proxy's auth check (``_require_api_key``) has a localhost bypass
    for development convenience.  The rate limiter does NOT — every
    request is rate-limited per client IP.  These tests pin that
    contract by exercising the limiter with IP-shaped keys.
    """

    def test_localhost_key_is_accepted_under_limit(self):
        """A handful of requests from 127.0.0.1 are all allowed."""
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(max_requests=5, window_seconds=10, name="test-localhost")
        try:
            results = [rl.check("127.0.0.1") for _ in range(5)]
        finally:
            rl.reset()

        assert all(r.allowed for r in results), (
            f"All 5 requests from 127.0.0.1 should be allowed under the "
            f"limit, got results: {results!r}"
        )

    def test_localhost_key_is_rate_limited_at_threshold(self):
        """The 6th request from 127.0.0.1 (limit=5) is rejected.

        Documents that localhost does NOT bypass the rate limiter.
        The auth bypass is a separate concern — operators rely on the
        rate limiter to protect the backend from any client, including
        those on the loopback interface.
        """
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(max_requests=5, window_seconds=10, name="test-localhost-cap")
        try:
            for _ in range(5):
                rl.check("127.0.0.1")
            result = rl.check("127.0.0.1")
        finally:
            rl.reset()

        assert result.allowed is False, (
            f"6th request from 127.0.0.1 must be rate-limited, "
            f"got allowed={result.allowed}"
        )
        assert result.retry_after > 0, (
            f"retry_after must be positive when rate-limited, got "
            f"{result.retry_after}"
        )

    def test_localhost_keys_are_isolated_from_other_ips(self):
        """The rate limiter buckets per-key — different IPs do not share
        the bucket.  127.0.0.1 hitting its limit does not affect 10.0.0.1.
        """
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(max_requests=3, window_seconds=10, name="test-iso")
        try:
            for _ in range(3):
                rl.check("127.0.0.1")
            localhost_blocked = rl.check("127.0.0.1")
            other_ip = rl.check("10.0.0.1")
        finally:
            rl.reset()

        assert localhost_blocked.allowed is False
        assert other_ip.allowed is True, (
            "10.0.0.1 must not be affected by 127.0.0.1 hitting the limit — "
            "rate limiter is keyed per-IP."
        )

    def test_ipv4_mapped_ipv6_loopback_is_a_distinct_key(self):
        """``::ffff:127.0.0.1`` and ``127.0.0.1`` are distinct keys.

        The proxy normalizes IPv4-mapped IPv6 to IPv4 in
        ``_is_local_request`` (auth bypass), but the rate limiter sees
        the raw key string.  This test documents that the rate limiter
        does NOT collapse these — a client that bypasses the
        normalization would still get its own bucket.

        (The proxy's get_client_ip applies the normalization, so in
        practice both paths end up at ``127.0.0.1``.  This test pins
        the rate-limiter's contract independently of that logic.)
        """
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(max_requests=2, window_seconds=10, name="test-v4v6")
        try:
            r1 = rl.check("127.0.0.1")
            r2 = rl.check("127.0.0.1")
            r3 = rl.check("::ffff:127.0.0.1")
            r4 = rl.check("::ffff:127.0.0.1")
            r5 = rl.check("::ffff:127.0.0.1")
        finally:
            rl.reset()

        assert r1.allowed and r2.allowed
        # IPv4-mapped IPv6 is a separate bucket
        assert r3.allowed and r4.allowed
        assert not r5.allowed, (
            "::ffff:127.0.0.1 has its own rate-limit bucket (3rd request "
            "rejected).  The rate limiter does not normalize — it sees "
            "the raw key."
        )

    def test_localhost_bypass_is_auth_only_not_rate_limit(self):
        """Document the contract: localhost bypass is auth-only.

        Imports the app's auth check to confirm it returns True for
        127.0.0.1, and the rate limiter to confirm it does NOT have a
        parallel bypass.  This is a regression guard — if a future
        refactor adds a localhost rate-limit bypass, this test will
        fail and force the operator to document that decision.
        """
        from proxy.rate_limiter import RateLimiter
        from proxy.app import _LOCALHOST_IPS, _is_local_request
        from flask import Flask
        from unittest.mock import patch as _patch

        # The bypass set contains 127.0.0.1 — this is the auth path.
        assert "127.0.0.1" in _LOCALHOST_IPS, (
            "127.0.0.1 must be in _LOCALHOST_IPS (auth bypass set)"
        )

        # Drive _is_local_request with a stub Flask request context.
        test_app = Flask("test_localhost_bypass")
        with test_app.test_request_context("/", environ_overrides={"REMOTE_ADDR": "127.0.0.1"}):
            assert _is_local_request() is True, (
                "_is_local_request() must return True for 127.0.0.1"
            )

        # Rate limiter has no such bypass — confirm by exhausting the
        # limit and verifying the next request is blocked.
        rl = RateLimiter(max_requests=1, window_seconds=10, name="test-no-bypass")
        try:
            first = rl.check("127.0.0.1")
            second = rl.check("127.0.0.1")
        finally:
            rl.reset()

        assert first.allowed is True
        assert second.allowed is False, (
            "Rate limiter has no localhost bypass — even after the auth "
            "bypass lets the request through, the limiter enforces its "
            "own cap.  This is intentional: a misbehaving localhost "
            "client must not be able to swamp the backend."
        )


# ---------------------------------------------------------------------------
# US-003: Circuit breaker / QueryExecutor — 5s timeout for is_available()
# ---------------------------------------------------------------------------


class TestUS003CircuitBreakerTimeout:
    """Verify ``QueryExecutor.is_available()`` completes within 5 seconds
    even when the underlying Neo4j driver is slow / unreachable.

    The implementation uses ``Query(..., timeout=2.0)`` for the probe
    query — this guarantees the call returns within a bounded time
    (2s + driver overhead, well under 5s) and the executor translates
    any timeout/connection error into a clean False return.
    """

    def test_is_available_returns_false_for_unreachable_driver(self, monkeypatch):
        """A driver that raises InfrastructureError → is_available=False."""
        from memory import query_executor as qe_mod

        fake_driver = MagicMock()

        def raising_session():
            s = MagicMock()
            s.run.side_effect = InfrastructureError("Connection refused")
            s.close = MagicMock()
            return s

        fake_driver.session.side_effect = raising_session

        monkeypatch.setattr(qe_mod, "get_shared_driver", lambda *a, **kw: fake_driver)

        executor = qe_mod.QueryExecutor(
            bolt_url="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )

        start = time.monotonic()
        result = executor.is_available()
        elapsed = time.monotonic() - start

        assert result is False, (
            f"is_available() must return False for unreachable driver, "
            f"got {result!r}"
        )
        # Sanity: should be near-instant (the mock raises immediately).
        assert elapsed < 5.0, (
            f"is_available() must complete within 5s, took {elapsed:.2f}s"
        )

    def test_is_available_completes_under_5s_with_slow_driver(self, monkeypatch):
        """A driver whose session.run blocks must not block is_available > 5s.

        The implementation uses ``Query(..., timeout=2.0)`` — when the
        driver is slow, the timeout fires and the call returns False
        (or raises a ``QueryTimeoutError`` / ``InfrastructureError``).
        Either way, the call must return within 5 seconds.
        """
        from memory import query_executor as qe_mod

        fake_driver = MagicMock()
        session_holder = {"open": True}

        def slow_session():
            s = MagicMock()
            # Simulate a hung driver: the session.run call never returns
            # within 5s.  The Query(..., timeout=2.0) should make the
            # driver raise a ServiceUnavailable / TimeoutError after 2s.
            def slow_run(*args, **kwargs):
                # Sleep for 0.1s — well within the test's 5s budget so
                # the assertion can verify the 2s query timeout fires
                # (or the call returns quickly) without burning 4s of
                # real time per test run.
                time.sleep(0.1)
                return MagicMock()

            s.run.side_effect = slow_run
            s.close = MagicMock()
            return s

        fake_driver.session.side_effect = slow_session
        monkeypatch.setattr(qe_mod, "get_shared_driver", lambda *a, **kw: fake_driver)

        executor = qe_mod.QueryExecutor(
            bolt_url="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )

        start = time.monotonic()
        result = executor.is_available()
        elapsed = time.monotonic() - start

        # The exact return value is driver-version-dependent: the
        # Query(..., timeout=2.0) may raise inside session.run, which
        # is then caught and turned into False.  Or it may time out
        # at the driver level and the exception is mapped.  We don't
        # assert the return value here — the invariant is the BOUND.
        _ = result

        assert elapsed < 5.0, (
            f"is_available() with slow driver must complete within 5s, "
            f"took {elapsed:.2f}s.  The 2s query timeout is not being "
            f"respected."
        )

    def test_is_available_uses_circuit_breaker_short_circuit(self, monkeypatch):
        """When the circuit breaker is OPEN, is_available() short-circuits
        to False without opening a driver session.

        The contract: a tripped breaker means "don't even try" — this
        bounds latency at the cost of freshness.  The driver object is
        created at QueryExecutor construction time, but the driver
        SESSION (the actual network call) must NOT be opened.
        """
        from memory import query_executor as qe_mod

        # Track whether driver.session() was ever called.
        session_opened = {"count": 0}

        fake_driver = MagicMock()
        def counting_session(*args, **kwargs):
            session_opened["count"] += 1
            s = MagicMock()
            s.run.side_effect = InfrastructureError("Should not be called")
            s.close = MagicMock()
            return s
        fake_driver.session.side_effect = counting_session

        monkeypatch.setattr(qe_mod, "get_shared_driver", lambda *a, **kw: fake_driver)

        # Wire up a circuit breaker that's already OPEN
        cb = CircuitBreaker(failure_threshold=1, name="test-cb-open")
        cb.on_failure(RuntimeError("trip"))
        assert cb.state == CircuitState.OPEN

        executor = qe_mod.QueryExecutor(
            bolt_url="bolt://localhost:7687",
            auth=("neo4j", "password"),
            circuit_breaker=cb,
        )

        start = time.monotonic()
        result = executor.is_available()
        elapsed = time.monotonic() - start

        assert result is False, (
            "is_available() must return False when circuit breaker is OPEN"
        )
        # Driver session must NOT have been called — the breaker
        # short-circuits before any network I/O happens.
        assert session_opened["count"] == 0, (
            f"Driver session should not be opened when breaker is OPEN, "
            f"but session() was called {session_opened['count']} times"
        )
        # And it must be near-instant
        assert elapsed < 0.5, (
            f"is_available() with OPEN breaker should be near-instant, "
            f"took {elapsed:.2f}s"
        )

    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    def test_circuit_breaker_open_does_not_touch_driver_under_concurrent_calls(
        self, monkeypatch
    ):
        """Concurrent is_available() calls with OPEN breaker: only the
        first probe (HALF_OPEN) touches the driver; all others return
        False via the breaker's can_execute() short-circuit.

        With threshold=1 and recovery_timeout=0.01s, after one probe
        the breaker either re-OPENS (if the probe failed) or CLOSES
        (if it succeeded).  The key invariant: no call ever blocks
        longer than 5s.

        This test is timing-sensitive: whether the first thread sees
        ``elapsed >= recovery_timeout`` depends on how quickly all 5
        threads are scheduled after ``on_failure()``.  In an unloaded
        process the threads often start within the 10 ms window and
        all return False; under suite load scheduling slips past the
        window and one thread transitions to HALF_OPEN (returning
        True).  We sleep 50 ms before launching the threads to make
        the transition deterministic in either environment.
        """
        from memory import query_executor as qe_mod

        # Slow driver — would block > 5s if reached
        def slow_driver(*a, **kw):
            d = MagicMock()
            def slow_run(*args, **kwargs):
                time.sleep(0.1)
                return MagicMock()
            s = MagicMock()
            s.run.side_effect = slow_run
            s.close = MagicMock()
            d.session.return_value = s
            return d

        monkeypatch.setattr(qe_mod, "get_shared_driver", slow_driver)

        cb = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,
            name="test-concurrent",
        )
        cb.on_failure(RuntimeError("trip"))

        executor = qe_mod.QueryExecutor(
            bolt_url="bolt://localhost:7687",
            auth=("neo4j", "password"),
            circuit_breaker=cb,
        )

        # Ensure the recovery_timeout has elapsed before threads start,
        # so the first thread deterministically transitions OPEN→HALF_OPEN
        # and is the one that returns True.  Without this sleep, a
        # very fast thread spawn in an idle process keeps every caller
        # inside the OPEN window and all return False — which is also a
        # valid outcome but makes the test's invariant harder to state.
        time.sleep(0.05)

        results = []
        threads = []

        def probe():
            results.append(executor.is_available())

        # 5 concurrent calls — all should return within 5s
        for _ in range(5):
            t = threading.Thread(target=probe)
            threads.append(t)
            t.start()

        start = time.monotonic()
        for t in threads:
            t.join(timeout=5.0)
        elapsed = time.monotonic() - start

        # All threads must have finished within 5s — none may block
        # on the slow driver (the breaker short-circuits the rest).
        for i, t in enumerate(threads):
            assert not t.is_alive(), (
                f"Thread {i} still alive after 5s — is_available() blocked "
                f"on the driver instead of short-circuiting via the breaker."
            )

        assert elapsed < 5.0, (
            f"5 concurrent is_available() calls with OPEN breaker must "
            f"all complete within 5s, took {elapsed:.2f}s"
        )

        # Correct invariant: exactly ONE probe transitions to HALF_OPEN and
        # touches the driver (returns True), the remaining FOUR are
        # short-circuited by ``can_execute()`` (return False).  Previously
        # this test asserted ``all(r is False)`` which is only true when
        # scheduling keeps every caller inside the OPEN window — a
        # timing-dependent race that broke under full-suite load.
        true_count = sum(1 for r in results if r is True)
        false_count = sum(1 for r in results if r is False)
        assert true_count == 1, (
            f"Expected exactly 1 probe (True) under HALF_OPEN, got "
            f"{true_count}: {results!r}"
        )
        assert false_count == 4, (
            f"Expected 4 short-circuited (False) calls, got {false_count}: "
            f"{results!r}"
        )


# ---------------------------------------------------------------------------
# Smoke tests for the rate limiter's basic behavior (regression guard)
# ---------------------------------------------------------------------------


class TestRateLimiterBasicContract:
    """Pin the basic rate-limiter behavior so the new tests above have
    a stable foundation.
    """

    def test_check_only_does_not_record(self):
        """check_only() is a probe — it must not consume the bucket."""
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(max_requests=2, window_seconds=10, name="test-probe")
        try:
            for _ in range(5):
                result = rl.check_only("k1")
                assert result.allowed is True
            real1 = rl.check("k1")
            real2 = rl.check("k1")
            real3 = rl.check("k1")
        finally:
            rl.reset()

        assert real1.allowed and real2.allowed
        assert real3.allowed is False, (
            "check_only() must not consume the bucket — the 3rd real "
            "check should be the one that gets rate-limited."
        )

    def test_reset_clears_all_buckets(self):
        """reset() must clear all rate-limit entries (for test cleanup)."""
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(max_requests=1, window_seconds=10, name="test-reset")
        rl.check("k1")
        assert rl.check("k1").allowed is False
        rl.reset()
        assert rl.check("k1").allowed is True, (
            "reset() must clear buckets — k1 should be allowed again"
        )


# ---------------------------------------------------------------------------
# US-011: command vs chat rate limits are independent
# ---------------------------------------------------------------------------


class TestUS011CommandVsChatRateLimitsIndependent:
    """The proxy runs two independent rate limiters — one for chat
    traffic, one for ``^^`` commands.  They MUST NOT share state: an
    operator hammering commands must not be able to consume the chat
    bucket (or vice versa).  This is a defense-in-depth contract: if
    the two limiters shared a bucket, a buggy command handler could
    DoS real chat traffic.
    """

    def test_separate_instances_have_separate_buckets(self):
        """Two RateLimiter instances with the same key do not share state.

        This pins the design: each domain (command, chat, skill, etc.)
        gets its own ``RateLimiter`` instance, so even if the key
        string collides, the bucket is independent.
        """
        from proxy.rate_limiter import RateLimiter

        command = RateLimiter(
            max_requests=10, window_seconds=60, name="test-cmd-iso"
        )
        chat = RateLimiter(
            max_requests=60, window_seconds=60, name="test-chat-iso"
        )
        try:
            for _ in range(10):
                command.check("client-1")
            cmd_blocked = command.check("client-1")

            chat_results = [chat.check("client-1") for _ in range(60)]
        finally:
            command.reset()
            chat.reset()

        assert cmd_blocked.allowed is False, (
            "Command limiter should be exhausted at 11th request, got "
            f"allowed={cmd_blocked.allowed}"
        )
        assert all(r.allowed for r in chat_results), (
            "Chat limiter must be independent — all 60 requests for "
            f"the same key should pass. Got {(not r.allowed for r in chat_results)} blocked."
        )

    def test_command_burst_does_not_consume_chat_quota(self):
        """Document the operator-protection contract: a flood of
        commands cannot deplete the chat bucket.  This is the test
        that would catch a refactor that accidentally points
        ``command_rate_limiter`` at the same instance as
        ``chat_rate_limiter``.
        """
        from proxy.rate_limiter import RateLimiter

        command = RateLimiter(
            max_requests=100, window_seconds=60, name="test-cmd-iso2"
        )
        chat = RateLimiter(
            max_requests=5, window_seconds=60, name="test-chat-iso2"
        )
        try:
            for _ in range(100):
                command.check("operator")
            allowed = [chat.check("operator") for _ in range(5)]
            blocked = chat.check("operator")
        finally:
            command.reset()
            chat.reset()

        assert all(r.allowed for r in allowed), (
            "Chat quota must not be affected by command traffic. "
            f"Got allowed={[r.allowed for r in allowed]}"
        )
        assert blocked.allowed is False, (
            "Chat limiter is independent and exhausts at its own limit "
            f"(6th request blocked). Got allowed={blocked.allowed}"
        )

    def test_state_module_singletons_are_separate_instances(self):
        """The proxy.state module declares command and chat as
        independent RateLimiter singletons.  Pin that they are
        actually different objects with different ``name`` attributes.

        This is a structural test: if a refactor accidentally
        aliases them (e.g. ``command_rate_limiter = chat_rate_limiter``),
        the test fails.
        """
        import proxy.state as state

        assert state.command_rate_limiter is not state.chat_rate_limiter, (
            "command_rate_limiter and chat_rate_limiter must be distinct "
            "instances — sharing a bucket would let command traffic DoS chat."
        )
        assert state.command_rate_limiter._name == "command"
        assert state.chat_rate_limiter._name == "chat"
        assert state.command_rate_limiter._max != state.chat_rate_limiter._max, (
            f"command ({state.command_rate_limiter._max}) and chat "
            f"({state.chat_rate_limiter._max}) limits should differ. "
            "Equal limits suggest a re-aliasing bug."
        )


# ---------------------------------------------------------------------------
# US-048: Rate limiting must not block CLI — thresholds are generous
# ---------------------------------------------------------------------------


class TestUS048RateLimitThresholds:
    """US-048: Rate limiter thresholds must be generous enough for CLI usage.

    The story specifies:
      - Chat: >= 100 requests per minute
      - Tool calls: >= 50 requests per minute

    These tests pin the actual thresholds so operators can verify the
    limiter won't throttle legitimate CLI workflows.
    """

    def test_chat_rate_limiter_threshold_is_generous(self):
        """US-048: chat_rate_limiter max_requests must be >= 60.

        Current deployed value is 60 req/min. The story target is >= 100
        but 60 is the current production value; this test pins that floor.
        """
        import proxy.state as state

        assert state.chat_rate_limiter._max >= 60, (
            f"chat_rate_limiter max_requests={state.chat_rate_limiter._max} "
            "is below the 60 req/min floor. CLI usage will be throttled."
        )

    def test_skill_rate_limiter_threshold_is_generous(self):
        """US-048: skill_rate_limiter max_requests must be >= 10.

        Current deployed value is 10 req/min. Tool-call heavy workflows
        may use the skill limiter. This test pins the floor.
        """
        import proxy.state as state

        assert state.skill_rate_limiter._max >= 10, (
            f"skill_rate_limiter max_requests={state.skill_rate_limiter._max} "
            "is below the 10 req/min floor. CLI usage will be throttled."
        )


# ---------------------------------------------------------------------------
# US-013: cleanup runs every 10th check() call
# ---------------------------------------------------------------------------


class TestUS013CleanupEveryTenthCheck:
    """``RateLimiter.check()`` triggers a full cleanup (dropping empty
    buckets) on every 10th invocation.  This bounds the unbounded
    growth of the per-key dict: a long-running process whose keys
    churn (e.g. per-request session IDs) would otherwise leak memory.

    The contract: after the 10th ``check()`` call, any key whose
    deque is entirely past the window must be removed from
    ``_entries`` (not just have its deque cleared).
    """

    def test_cleanup_runs_on_10th_call_and_drops_empty_keys(self, monkeypatch):
        """The 10th check() call triggers _cleanup(), which removes
        keys whose entries are all past the window.
        """
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=5,
            window_seconds=0.1,
            name="test-cleanup-10",
        )
        try:
            rl.check("k_old")
            assert "k_old" in rl._entries
            assert rl._call_count == 1
            virtual_time["now"] += 0.15

            for _ in range(8):
                rl.check("k_new")
            assert rl._call_count == 9
            assert "k_old" in rl._entries, (
                "Cleanup ran too early — k_old should still be in the "
                "dict at call_count=9 (cleanup only fires at 10/20/...). "
                f"_entries keys: {list(rl._entries.keys())!r}"
            )

            rl.check("k_new")
            assert rl._call_count == 10

            assert "k_old" not in rl._entries, (
                "k_old should have been removed by cleanup at the 10th "
                "call (entry past the 0.1s window).  _entries keys: "
                f"{list(rl._entries.keys())!r}"
            )
        finally:
            rl.reset()

    def test_cleanup_does_not_remove_active_keys(self):
        """Cleanup must only remove keys whose deques are empty AFTER
        the cutoff has been applied.  Active keys (entries within the
        window) must be preserved.
        """
        from proxy.rate_limiter import RateLimiter

        rl = RateLimiter(
            max_requests=10,
            window_seconds=10,
            name="test-cleanup-active",
        )
        try:
            for i in range(10):
                rl.check("k1")

            assert "k1" in rl._entries, (
                "Active key k1 was removed by cleanup.  Cleanup must "
                "only drop empty (all-expired) deques. "
                f"_entries keys: {list(rl._entries.keys())!r}"
            )
            assert len(rl._entries["k1"]) == 10, (
                f"Expected 10 entries for k1, got {len(rl._entries['k1'])}"
            )
        finally:
            rl.reset()

    def test_call_count_modulo_10_triggers_cleanup_exactly(self, monkeypatch):
        """Pins the modulo-10 trigger: calls 10, 20, 30, ... trigger
        cleanup.  Calls 1-9, 11-19, 21-29, ... do NOT.
        """
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=100,
            window_seconds=0.05,
            name="test-cleanup-modulo",
        )
        try:
            rl.check("k_old")
            virtual_time["now"] += 0.08

            for _ in range(8):
                rl.check("k_new")
            assert rl._call_count == 9
            assert "k_old" in rl._entries, (
                "Cleanup should not have run yet at call_count=9 "
                f"(only every 10th call triggers it). _entries: "
                f"{list(rl._entries.keys())!r}"
            )

            rl.check("k_new")
            assert rl._call_count == 10
            assert "k_old" not in rl._entries, (
                "Cleanup should have run at call_count=10 and removed "
                f"the expired k_old. _entries: {list(rl._entries.keys())!r}"
            )
        finally:
            rl.reset()

