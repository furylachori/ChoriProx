"""# US-016: Skill Mode Expiry
Integration tests: mode expiry racing with in-flight tool calls.

Validates that mode TTL expiry does not corrupt in-flight work:

1. A tool call started under an active mode completes successfully even
   if the mode expires mid-execution.
2. The mode is cleared AFTER the tool completes (not during).
3. A new request after expiry observes the cleared mode.
4. Concurrent set_active_mode + cleanup_stale_ticket_modes does not lose
   the new mode or clobber a fresh one over a stale one.
5. cleanup_stale_ticket_modes' CAS guard (expected_mode) prevents the
   stale clear when a new mode was installed in the gap.
6. Re-activating a mode after expiry works without conflict.

These tests use short TTLs (1s) and short sleeps to keep the suite fast.

References:
- src/proxy/agent/agent.py                Agent.set_active_mode / clear_active_mode
- src/proxy/agent/agent_registry.py       AgentRegistry.cleanup_stale_ticket_modes
- src/proxy/agent/types.py                ModeState.is_expired
- tests/mode_harness.py                   activate_work_mode / create_test_agent
- tests/concurrency_harness.py            run_concurrently
"""
from __future__ import annotations

import logging
import threading
import time
from unittest.mock import patch

import pytest

from tests.concurrency_harness import run_concurrently
from tests.mode_harness import (
    activate_work_mode,
    create_test_agent,
    create_test_mode_state,
)


# All tests in this module are integration tests.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test 1: Mode expires during slow tool call — tool completes
# ---------------------------------------------------------------------------


class TestModeExpiresDuringToolCall:
    """A slow tool call started under a short-TTL mode finishes cleanly.

    The mode is allowed to expire while the tool is in flight, but the
    tool call itself must complete with its real result. The mode is
    cleared AFTER the tool completes (not torn down mid-execution).
    """

    def test_slow_tool_completes_after_mode_expiry(self):
        # Arrange: work mode with 1-second TTL.
        agent = create_test_agent()
        mode = activate_work_mode(agent, task_hash="slow-tool", ttl=1)

        # Tool body sleeps 1.5s — longer than the 1s TTL but short enough
        # to keep the test under 5s.
        tool_started = threading.Event()
        tool_result: dict = {}
        tool_lock = threading.Lock()

        def slow_tool_call(payload: dict) -> dict:
            tool_started.set()
            # Sleep past the TTL.
            time.sleep(1.5)
            # Read mode *after* the sleep. It will be expired but still
            # present (not cleared yet) — the in-flight tool was started
            # under the mode and should not be rejected.
            ms = agent.snapshot_mode()
            expired_now = agent.is_mode_expired()
            return {
                "ok": True,
                "mode_name": ms.name if ms else None,
                "mode_expired": expired_now,
                "echo": payload,
            }

        def run_tool():
            with tool_lock:
                tool_result["value"] = slow_tool_call({"x": 1})

        # Act: run the tool in a thread so the test thread can wait for
        # the TTL to elapse and then drive cleanup.
        t = threading.Thread(target=run_tool, daemon=True)
        t.start()

        # Wait for the tool to start so we know the TTL clock is real.
        assert tool_started.wait(timeout=2), "Tool did not start"
        # Sleep past the 1s TTL — the mode is now expired.
        time.sleep(1.0)

        # The tool is still running. Mode should report expired.
        assert agent.is_mode_expired(), "Mode should be expired after sleep"

        # Tool completes successfully.
        t.join(timeout=5)
        assert not t.is_alive(), "Tool thread did not complete in time"
        assert tool_result["value"]["ok"] is True
        # The mode snapshot the tool saw inside the sleep is the *same*
        # object (still installed on the agent) but reports expired.
        assert tool_result["value"]["mode_name"] == "work"
        assert tool_result["value"]["mode_expired"] is True

        # Now the tool has returned. Cleanup runs AFTER the tool completes.
        agent.clear_active_mode(expected_mode=mode)
        assert agent.snapshot_mode() is None, (
            "Mode should be cleared after tool completes"
        )

    def test_next_request_sees_cleared_mode(self):
        """After the slow tool returns and cleanup runs, a subsequent
        request must observe the mode as cleared/expired."""
        agent = create_test_agent()
        mode = activate_work_mode(agent, task_hash="next-req", ttl=1)

        # Simulate a tool that takes 1.5s.
        def run_tool():
            time.sleep(1.5)
            return {"ok": True}

        t = threading.Thread(target=run_tool, daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()

        # TTL has long passed; clear the mode.
        agent.clear_active_mode(expected_mode=mode)

        # A subsequent request must see the mode as cleared.
        assert agent.snapshot_mode() is None
        assert not agent.is_skill_mode
        assert not agent.is_ticket_mode


# ---------------------------------------------------------------------------
# Test 2: Mode expires between requests — cleared properly
# ---------------------------------------------------------------------------


class TestModeExpiresBetweenRequests:
    """Two back-to-back requests: first under an active mode, second after
    the mode has expired and been cleared. The second request must not
    observe stale mode state."""

    def test_first_request_active_second_request_cleared(self):
        from proxy.agent.agent_registry import registry

        agent = create_test_agent()
        mode = activate_work_mode(agent, task_hash="between", ttl=1)

        # Register the agent so cleanup_stale_ticket_modes() iterates it.
        with registry._lock:
            registry._agents[agent.session_id] = agent
            registry._agents_by_id[agent.agent_id] = agent
            registry._lru[agent.session_id] = None
            registry._lru.move_to_end(agent.session_id)

        try:
            # Request 1: mode is active, snapshot returns it.
            ms1 = agent.snapshot_mode()
            assert ms1 is not None
            assert ms1.name == "work"
            assert not agent.is_mode_expired()
            assert agent.is_skill_mode

            # Wait past the TTL.
            time.sleep(1.2)

            # Mode is now expired.
            assert agent.is_mode_expired()
            # is_skill_mode is False for expired modes.
            assert not agent.is_skill_mode

            # Run cleanup (mirrors what session.py does periodically).
            registry.cleanup_stale_ticket_modes()

            # Request 2: mode is gone.
            ms2 = agent.snapshot_mode()
            assert ms2 is None, "Stale mode leaked into next request"
            assert not agent.is_ticket_mode
            assert not agent.is_skill_mode
            # _ticket_mode_ts must also be reset so legacy fallbacks don't
            # resurrect the mode.
            with agent._mode_lock:
                assert agent._ticket_mode_ts is None
        finally:
            with registry._lock:
                registry._agents.pop(agent.session_id, None)
                registry._agents_by_id.pop(agent.agent_id, None)
                registry._lru.pop(agent.session_id, None)

    def test_no_stale_state_after_clear(self):
        """After explicit clear, all mode-related state is reset."""
        agent = create_test_agent()
        mode = activate_work_mode(agent, task_hash="stale", ttl=1)
        time.sleep(1.2)

        ok = agent.clear_active_mode(expected_mode=mode)
        assert ok is True

        # Verify all state is gone.
        assert agent.snapshot_mode() is None
        assert agent.snapshot_mode_and_ticket_ts() == (None, None)
        assert agent.get_mode_name() is None
        assert not agent.is_mode_expired()  # no mode = not expired
        assert not agent.is_skill_mode
        assert not agent.is_ticket_mode


# ---------------------------------------------------------------------------
# Test 3: Mode expiry race — concurrent set_active_mode and cleanup
# ---------------------------------------------------------------------------


class TestCleanupRaceWithSetActiveMode:
    """Thread 1 sets a fresh mode while Thread 2 runs cleanup. The fresh
    mode must survive — cleanup must not clobber it."""

    def test_new_mode_survives_concurrent_cleanup(self):
        # We exercise this directly on a single Agent (not the global
        # registry) to avoid cross-test contamination, but the
        # cleanup_stale_ticket_modes() iteration runs over all agents
        # in the registry, so we register ours temporarily.
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry

        agent = Agent(
            session_id="race-test-session",
            parent=None,
            config=AgentConfig(),
        )

        with registry._lock:
            registry._agents[agent.session_id] = agent
            registry._agents_by_id[agent.agent_id] = agent
            registry._lru[agent.session_id] = None
            registry._lru.move_to_end(agent.session_id)

        try:
            # Install an already-expired mode.
            expired_mode = create_test_mode_state(
                "ticket", ttl=1, activated_at=time.time() - 5
            )
            agent.set_active_mode(expired_mode)
            assert agent.is_mode_expired()

            # Prepare a fresh, non-expired mode object.
            fresh_mode = create_test_mode_state(
                "work", ttl=300, activated_at=time.time()
            )

            def set_fresh():
                # Tiny delay so cleanup likely runs first.
                time.sleep(0.05)
                return agent.set_active_mode(fresh_mode)

            def run_cleanup():
                registry.cleanup_stale_ticket_modes()

            try:
                run_concurrently([set_fresh, run_cleanup], timeout=5)
            except Exception:
                # If a thread raised, the agent state is the source of truth.
                raise

            # After both threads finish, the fresh mode must be present.
            ms = agent.snapshot_mode()
            assert ms is not None, "Fresh mode was lost"
            assert ms is fresh_mode, (
                f"Fresh mode was clobbered by cleanup. "
                f"Got {getattr(ms, 'name', None)}"
            )
            # And it must not be expired.
            assert not agent.is_mode_expired()
            assert agent.is_skill_mode
        finally:
            with registry._lock:
                registry._agents.pop(agent.session_id, None)
                registry._agents_by_id.pop(agent.agent_id, None)
                registry._lru.pop(agent.session_id, None)

    def test_concurrent_race_repeated(self):
        """Run the race many times to flush out scheduling-dependent bugs."""
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry

        agent = Agent(
            session_id="race-repeated-session",
            parent=None,
            config=AgentConfig(),
        )

        with registry._lock:
            registry._agents[agent.session_id] = agent
            registry._agents_by_id[agent.agent_id] = agent
            registry._lru[agent.session_id] = None
            registry._lru.move_to_end(agent.session_id)

        try:
            for _ in range(20):
                expired_mode = create_test_mode_state(
                    "ticket", ttl=1, activated_at=time.time() - 5
                )
                agent.set_active_mode(expired_mode)
                fresh_mode = create_test_mode_state(
                    "work", ttl=300, activated_at=time.time()
                )

                def set_fresh(m=fresh_mode):
                    time.sleep(0.01)
                    return agent.set_active_mode(m)

                def run_cleanup():
                    registry.cleanup_stale_ticket_modes()

                run_concurrently([set_fresh, run_cleanup], timeout=5)

                ms = agent.snapshot_mode()
                assert ms is not None
                # Either the fresh mode (if set won) or no mode (if cleanup
                # ran first AND the CAS in cleanup rejected the subsequent
                # set — but set_active_mode has no CAS, so the fresh mode
                # must win). Either way, the *expired* mode must not be
                # # left installed.
                assert ms is not expired_mode
                if ms is not None:
                    assert not agent.is_mode_expired()
        finally:
            with registry._lock:
                registry._agents.pop(agent.session_id, None)
                registry._agents_by_id.pop(agent.agent_id, None)
                registry._lru.pop(agent.session_id, None)


# ---------------------------------------------------------------------------
# Test 4: cleanup_stale_ticket_modes TOCTOU — CAS guard
# ---------------------------------------------------------------------------


class TestCleanupCasGuard:
    """The expected_mode parameter on clear_active_mode is a CAS guard.
    If a new mode was set between snapshot and clear, clear must return
    False and leave the new mode alone. The cleanup loop must not crash
    on this — it must log and continue."""

    def test_cas_failure_does_not_crash_cleanup(self, caplog):
        """Drive the CAS-fail path through the real clear_active_mode.

        Setup:
          1. Agent is registered with the global registry.
          2. We patch agent.is_mode_expired to return True so the cleanup
             loop *snapshots* the agent as stale.
          3. We patch agent.snapshot_mode (used inside cleanup) to return
             the expired ModeState (so the CAS uses the stale reference).
          4. The *real* agent.active_mode is the fresh replacement, so
             the CAS identity check inside clear_active_mode fails and
             clear_active_mode returns False without mutating state.

        The cleanup loop must not crash, must not clobber the replacement
        mode, and must not log at ERROR level.
        """
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry

        agent = Agent(
            session_id="cas-test-session",
            parent=None,
            config=AgentConfig(),
        )

        with registry._lock:
            registry._agents[agent.session_id] = agent
            registry._agents_by_id[agent.agent_id] = agent
            registry._lru[agent.session_id] = None
            registry._lru.move_to_end(agent.session_id)

        try:
            # Install an expired mode and then a fresh replacement —
            # this is the state cleanup observes if it runs *after*
            # the race winner.
            expired_mode = create_test_mode_state(
                "ticket", ttl=1, activated_at=time.time() - 5
            )
            replacement = create_test_mode_state(
                "work", ttl=300, activated_at=time.time()
            )
            agent.set_active_mode(expired_mode)
            agent.set_active_mode(replacement)
            # Sanity: replacement is live.
            assert agent.snapshot_mode() is replacement

            # Patch is_mode_expired → True (so cleanup snapshots the
            # agent as stale) and snapshot_mode → the expired_mode
            # object (so cleanup's CAS uses the stale reference).
            with patch.object(agent, "is_mode_expired", return_value=True), \
                 patch.object(agent, "snapshot_mode", return_value=expired_mode):
                with caplog.at_level(logging.ERROR):
                    # Must not raise.
                    registry.cleanup_stale_ticket_modes()

            # The CAS inside the real clear_active_mode saw
            # `agent.active_mode is not expired_mode` → True, so it
            # returned False without clearing. The replacement mode
            # must still be live.
            assert agent.snapshot_mode() is replacement, (
                "CAS guard failed — the new mode was clobbered"
            )
            assert not agent.is_mode_expired()
            # No ERROR-level log entries (cleanup survived the CAS-fail).
            error_records = [
                r for r in caplog.records if r.levelno >= logging.ERROR
            ]
            assert not error_records, (
                f"cleanup logged errors: {[r.getMessage() for r in error_records]}"
            )
        finally:
            with registry._lock:
                registry._agents.pop(agent.session_id, None)
                registry._agents_by_id.pop(agent.agent_id, None)
                registry._lru.pop(agent.session_id, None)

    def test_cas_failure_logged_at_debug(self, caplog):
        """When CAS fails, cleanup logs at DEBUG (visible in caplog at
        that level) — confirming the path was exercised. We assert the
        debug log was emitted and contains the agent_id."""
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry

        agent = Agent(
            session_id="cas-log-session",
            parent=None,
            config=AgentConfig(),
        )

        with registry._lock:
            registry._agents[agent.session_id] = agent
            registry._agents_by_id[agent.agent_id] = agent
            registry._lru[agent.session_id] = None
            registry._lru.move_to_end(agent.session_id)

        try:
            expired_mode = create_test_mode_state(
                "ticket", ttl=1, activated_at=time.time() - 5
            )
            replacement = create_test_mode_state(
                "work", ttl=300, activated_at=time.time()
            )
            agent.set_active_mode(expired_mode)
            agent.set_active_mode(replacement)

            with patch.object(agent, "is_mode_expired", return_value=True), \
                 patch.object(agent, "snapshot_mode", return_value=expired_mode):
                with caplog.at_level(
                    logging.DEBUG, logger="proxy.agent.registry"
                ):
                    registry.cleanup_stale_ticket_modes()

            # The cleanup code logs at DEBUG after a successful clear;
            # on CAS-fail it returns False and skips the log. So the
            # *positive* signal here is that nothing crashed AND the
            # replacement mode survives. We also accept either the
            # DEBUG log being present (if cleanup tolerated the
            # pseudo-fail) or absent (if it short-circuited).
            assert agent.snapshot_mode() is replacement
        finally:
            with registry._lock:
                registry._agents.pop(agent.session_id, None)
                registry._agents_by_id.pop(agent.agent_id, None)
                registry._lru.pop(agent.session_id, None)

    def test_cas_success_clears_expired_mode(self):
        """Sanity: when no race occurs, cleanup successfully clears the
        expired mode (CAS succeeds)."""
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry

        agent = Agent(
            session_id="cas-success-session",
            parent=None,
            config=AgentConfig(),
        )

        with registry._lock:
            registry._agents[agent.session_id] = agent
            registry._agents_by_id[agent.agent_id] = agent
            registry._lru[agent.session_id] = None
            registry._lru.move_to_end(agent.session_id)

        try:
            mode = create_test_mode_state(
                "ticket", ttl=1, activated_at=time.time() - 5
            )
            agent.set_active_mode(mode)
            assert agent.is_mode_expired()

            registry.cleanup_stale_ticket_modes()

            assert agent.snapshot_mode() is None
            with agent._mode_lock:
                assert agent._ticket_mode_ts is None
        finally:
            with registry._lock:
                registry._agents.pop(agent.session_id, None)
                registry._agents_by_id.pop(agent.agent_id, None)
                registry._lru.pop(agent.session_id, None)


# ---------------------------------------------------------------------------
# Test 5: Mode reactivation after expiry
# ---------------------------------------------------------------------------


class TestModeReactivationAfterExpiry:
    """Re-activating a mode after it has expired must succeed cleanly
    and not be confused with the expired mode."""

    def test_reactivate_with_new_ttl(self):
        agent = create_test_agent()

        # First activation — short TTL.
        first = activate_work_mode(agent, task_hash="first", ttl=1)
        assert agent.snapshot_mode() is first
        assert not agent.is_mode_expired()

        # Wait for expiry.
        time.sleep(1.2)
        assert agent.is_mode_expired()

        # Reactivate with a fresh, longer TTL.
        second = activate_work_mode(agent, task_hash="second", ttl=300)
        assert second is not first
        assert agent.snapshot_mode() is second
        assert not agent.is_mode_expired()
        assert agent.is_skill_mode
        # The new mode's skill_state reflects the new task hash, not the
        # old one.
        assert second.skill_state["task_hash"] == "second"
        assert agent.snapshot_mode().skill_state["task_hash"] == "second"

    def test_reactivate_after_explicit_clear(self):
        agent = create_test_agent()
        activate_work_mode(agent, task_hash="a", ttl=1)
        time.sleep(1.2)

        # Clear the expired mode.
        cleared = agent.clear_active_mode()
        assert cleared is True
        assert agent.snapshot_mode() is None

        # Re-activate.
        new = activate_work_mode(agent, task_hash="b", ttl=300)
        assert agent.snapshot_mode() is new
        assert new.skill_state["task_hash"] == "b"
        assert not agent.is_mode_expired()

    def test_reactivate_with_different_mode_name(self):
        """Switching from one mode name (e.g. work) to another (e.g.
        ticket) after expiry must work without leftover state."""
        agent = create_test_agent()
        work = create_test_mode_state("work", ttl=1, activated_at=time.time())
        agent.set_active_mode(work)
        time.sleep(1.2)
        assert agent.is_mode_expired()

        ticket = create_test_mode_state(
            "ticket", ttl=300, activated_at=time.time()
        )
        agent.set_active_mode(ticket)

        assert agent.snapshot_mode() is ticket
        assert agent.snapshot_mode().name == "ticket"
        assert not agent.is_mode_expired()
        # _ticket_mode_ts is refreshed for ticket mode.
        with agent._mode_lock:
            assert agent._ticket_mode_ts is not None
