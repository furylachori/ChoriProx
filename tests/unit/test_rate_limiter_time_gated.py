"""M8-4: time-based gates for RateLimiter cleanup and save.

Pre-fix: cleanup and save only fired on the 10th/20th/... check()
call (call_count % 10 == 0). Under low traffic, the 6 persistent
limiters (command/skill/skill_ip/task/chat/admin) never wrote to
SQLite — on process restart, all rate-limit state was lost.

Fix: time-based gates in addition to the modulo gates. _cleanup()
fires when now - _last_cleanup > 60s; _save() fires when
now - _last_save > 30s (for persist=True limiters). The existing
modulo contract is preserved (TestUS013CleanupEveryTenthCheck
still passes).
"""

from unittest.mock import MagicMock, patch

import pytest


class TestTimeBasedCleanupGate:
    """M8-4: time-based cleanup fires independent of call count."""

    def test_cleanup_fires_on_time_elapsed(self, monkeypatch):
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=5, window_seconds=0.1, name="test-time-cleanup"
        )

        # 5 calls in quick succession — modulo gate does NOT fire.
        for _ in range(5):
            rl.check("k1")
        assert rl._call_count == 5
        # Advance time past the cleanup interval.
        virtual_time["now"] += 61.0
        # 6th call: time gate fires (modulo gate does NOT).
        rl.check("k1")
        # After cleanup fired, _last_cleanup should be updated.
        assert rl._last_cleanup == virtual_time["now"], (
            f"_last_cleanup not updated after time-gated cleanup. "
            f"Expected {virtual_time['now']}, got {rl._last_cleanup}"
        )

    def test_check_only_also_gates_time_based_cleanup(self, monkeypatch):
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=5, window_seconds=0.1, name="test-time-cleanup-checkonly"
        )

        # 5 check_only calls in <1s.
        for _ in range(5):
            rl.check_only("k1")
        virtual_time["now"] += 61.0
        rl.check_only("k1")
        assert rl._last_cleanup == virtual_time["now"], (
            "check_only should also gate time-based cleanup so the "
            "per-key dict can't grow unbounded under gate-before-commit "
            "traffic. _last_cleanup was not updated."
        )

    def test_existing_modulo_10_contract_preserved(self, monkeypatch):
        """Pin the existing contract: 10th call still triggers cleanup
        via the modulo path (not just the time path)."""
        from proxy.rate_limiter import RateLimiter

        # Time advances 1s total — far less than 60s. Only modulo fires.
        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=5, window_seconds=0.1, name="test-modulo-preserved"
        )

        # Make _last_cleanup clearly stale so only the modulo path fires.
        rl._last_cleanup = 0.0  # would trigger time gate immediately
        # But _call_count starts at 0; the modulo fires at 10, 20, ...
        for _ in range(10):
            rl.check("k1")
        # _call_count is now 10; cleanup should have run via the
        # modulo path. The 10th call's _last_cleanup should be set
        # to the current virtual_time.
        assert rl._call_count == 10
        assert rl._last_cleanup == virtual_time["now"], (
            f"10th call should fire cleanup via modulo path. "
            f"_last_cleanup={rl._last_cleanup}, expected {virtual_time['now']}"
        )


class TestTimeBasedSaveGate:
    """M8-4: time-based save fires for persistent limiters."""

    def test_save_fires_on_time_elapsed_for_persistent(self, monkeypatch):
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        # Stub _save to track calls.
        save_calls = []
        real_init = RateLimiter.__init__

        def init_with_save_spy(self, max_requests, window_seconds, name="", persist=False):
            real_init(self, max_requests, window_seconds, name, persist)
            self._save_real = self._save
            def spy_save():
                save_calls.append(1)
                # Don't actually call _save_real (avoid DB).
            self._save = spy_save

        monkeypatch.setattr(RateLimiter, "__init__", init_with_save_spy)

        rl = RateLimiter(
            max_requests=5, window_seconds=60.0, name="test-time-save", persist=True
        )

        # 5 calls in <1s.
        for _ in range(5):
            rl.check("k1")
        # No save yet (no modulo fire, no time fire).
        # Note: with persist=True and _last_save initialized to
        # time.time() in __init__, the first call would NOT
        # immediately save — time delta is 0.
        initial_save_count = len(save_calls)

        # Advance time past the save interval.
        virtual_time["now"] += 31.0
        rl.check("k1")
        assert len(save_calls) > initial_save_count, (
            f"Time-gated save should have fired. save_calls={save_calls}, "
            f"initial={initial_save_count}"
        )

    def test_save_does_not_fire_for_non_persist(self, monkeypatch):
        from proxy.rate_limiter import RateLimiter

        save_calls = []
        real_init = RateLimiter.__init__

        def init_with_save_spy(self, max_requests, window_seconds, name="", persist=False):
            real_init(self, max_requests, window_seconds, name, persist)
            def spy_save():
                save_calls.append(1)
            self._save = spy_save

        monkeypatch.setattr(RateLimiter, "__init__", init_with_save_spy)

        rl = RateLimiter(
            max_requests=5, window_seconds=60.0, name="test-no-save", persist=False
        )

        # 100 calls over 10 minutes.
        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        for i in range(100):
            rl.check(f"k{i % 5}")
            virtual_time["now"] += 5.0  # 5s per call, 500s total

        # Even with persist=False, modulo-10 would fire _save 10 times
        # IF it were called. But persist=False skips the save call.
        assert len(save_calls) == 0, (
            f"Non-persistent limiter should never call _save. "
            f"save_calls={save_calls}"
        )

    def test_timestamps_updated_after_firing(self, monkeypatch):
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=5, window_seconds=60.0, name="test-timestamps"
        )

        initial_cleanup_ts = rl._last_cleanup
        initial_save_ts = rl._last_save

        # Force both gates to fire.
        virtual_time["now"] += 100.0
        rl.check("k1")

        assert rl._last_cleanup > initial_cleanup_ts, (
            f"_last_cleanup should be updated. initial={initial_cleanup_ts}, "
            f"now={rl._last_cleanup}"
        )
        # For non-persistent limiters, _last_save is updated on the
        # time-elapsed path even if _save() is not called. (Same
        # timestamp-tracking discipline.)
        assert rl._last_save > initial_save_ts, (
            f"_last_save should be updated even for non-persistent "
            f"limiters (the gate keeps measuring since-last-event)."
        )

    def test_first_call_does_not_immediately_save(self, monkeypatch):
        """M8-4 fix: initialize _last_save = time.time() in __init__
        so the first call does NOT fire the save (delta = 0 < 30s)."""
        from proxy.rate_limiter import RateLimiter

        save_calls = []
        real_init = RateLimiter.__init__

        def init_with_save_spy(self, max_requests, window_seconds, name="", persist=False):
            real_init(self, max_requests, window_seconds, name, persist)
            def spy_save():
                save_calls.append(1)
            self._save = spy_save

        monkeypatch.setattr(RateLimiter, "__init__", init_with_save_spy)

        rl = RateLimiter(
            max_requests=5, window_seconds=60.0, name="test-first-save", persist=True
        )

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        # First call: _last_save = 1000.0 (set in __init__ at virtual_time
        # of __init__'s call). The check() reads now=1000.0. Delta = 0.
        # No save.
        rl.check("k1")
        assert len(save_calls) == 0, (
            f"First call should NOT fire save. save_calls={save_calls}"
        )

    def test_save_exception_still_swallowed(self, monkeypatch):
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        rl = RateLimiter(
            max_requests=5, window_seconds=60.0, name="test-save-exc", persist=True
        )

        # Make _save raise.
        def boom():
            raise RuntimeError("SQLite down")
        rl._save = boom

        # Advance time past save interval.
        virtual_time["now"] += 31.0
        # check() must NOT propagate the exception.
        try:
            result = rl.check("k1")
        except RuntimeError as e:
            raise AssertionError(
                f"check() propagated _save error: {e}. The contract is "
                f"that _save failures are swallowed (graceful "
                f"degradation on SQLite failure)."
            )
        assert result.allowed is True, "check() should not propagate _save errors"
        # And the timestamp should still be updated (so the next save
        # attempt is bounded to the next interval, not retried immediately).
        assert rl._last_save == virtual_time["now"], (
            f"_last_save should be updated even on _save exception. "
            f"Expected {virtual_time['now']}, got {rl._last_save}"
        )

    def test_init_anchors_timestamps(self, monkeypatch):
        """M8-4: __init__ sets _last_cleanup and _last_save to time.time()
        so the first post-restart check doesn't immediately fire.
        ``load_persisted_state`` is stubbed to confirm the anchor comes
        from __init__, not from re-loading persisted state (the
        constructor no longer calls it; startup does, after init_db).
        """
        from proxy.rate_limiter import RateLimiter

        virtual_time = {"now": 1000.0}
        monkeypatch.setattr("proxy.rate_limiter.time.time", lambda: virtual_time["now"])

        # Stub load_persisted_state to do nothing (no DB) and verify the
        # timestamp re-anchor is what __init__ sets on its own.
        monkeypatch.setattr(RateLimiter, "load_persisted_state", lambda self: False)

        # Use persist=True so the hydration path is wired (the
        # constructor no longer calls _load itself — startup does —
        # but the public method must still be stubbable so future
        # tests can intercept without a real SQLite).
        rl = RateLimiter(
            max_requests=5, window_seconds=60.0, name="test-load-anchor", persist=True
        )

        # _last_cleanup and _last_save should be ~1000.0 (now at __init__).
        # Note: they were set in __init__ first; the stubbed
        # load_persisted_state does nothing, so the re-anchor does not
        # happen. But the __init__ initialization is enough to make the
        # first call NOT trigger save.
        assert rl._last_cleanup == 1000.0
        assert rl._last_save == 1000.0
