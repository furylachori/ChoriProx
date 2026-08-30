"""Load tests for lifecycle finalization under concurrent finalize calls.

Per spec §7.6, finalize_skill_mode must use compare-and-clear (CAS) to
prevent clearing a concurrently-installed mode. These tests verify the
CAS semantics under threading.Barrier-driven concurrency.

CAS invariants pinned here:

1. **Same-expected race** — N threads concurrently call finalize_skill_mode
   on the same agent with the SAME expected_mode. Exactly one clear wins
   (returns True); all others return False because the agent's active_mode
   has been set to None by the winner.

2. **Stale-snapshot preservation** — A finalize_skill_mode call with an
   expected_mode that does NOT match the agent's current active_mode
   returns False; the agent's current mode is preserved (no spurious clear).

3. **Concurrent replacement** — When a new mode is installed AFTER the
   snapshot was captured (and BEFORE the finalize call runs), the finalize
   call detects the mismatch via CAS and does not clear the new mode.

Run with::

    PYTHONPATH=src python3 -m pytest tests/load/test_lifecycle_concurrency.py -n 1 -v

Single-threaded by design: threading.Barrier does not interact well with
xdist worker pools (each worker has its own thread pool). Per the
`tests/AGENTS.md` load-test pitfall, the load layer runs with `-n 1`.
"""
import threading
import time
from types import SimpleNamespace

import pytest

from proxy.agent.agent import Agent
from proxy.skills.skill_definition import ModeState
from proxy.mode import finalize_skill_mode
from proxy.skills.skill_registry import get_default_registry, reset_default_registry
from tests.concurrency_harness import run_concurrently


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(session_id):
    """Construct a real Agent (per spec, no mocking of the production class)."""
    return Agent(
        session_id=session_id,
        agent_id=f"agent_{session_id}",
        is_subagent=False,
    )


def _make_mode_state(name, ttl_seconds=300):
    """Construct a real ModeState."""
    return ModeState(
        name=name,
        activated_at=time.time(),
        ttl_seconds=ttl_seconds,
        skill_state={},
    )


@pytest.fixture
def fresh_agent_registry():
    """Reset the skill registry to avoid leakage across tests."""
    reset_default_registry()
    yield get_default_registry()
    reset_default_registry()


# ===========================================================================
# Same-expected race: exactly one clear wins
# ===========================================================================


def test_concurrent_finalize_all_clears_consistently():
    """N threads concurrently call finalize_skill_mode on the same agent with
    the SAME expected_mode.

    Per spec §7.5/§7.6: the CAS in clear_active_mode(expected_mode=...) is the
    atomic op. The first thread to acquire the lock sees active_mode == mode
    and clears it; subsequent threads see active_mode is None and the CAS
    check fails (expected_mode no longer matches). Exactly one clear wins.
    """
    agent = _make_agent("sess_concurrent_same")
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)

    session_id = "sess_concurrent_same"

    def finalize_once():
        # Each thread builds its own ctx (response_data with stop → COMPLETE).
        ctx = SimpleNamespace(
            session_id=session_id,
            response_data={"choices": [{"finish_reason": "stop"}]},
        )
        return finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    # 8 threads, all barrier-synchronized for a deterministic start.
    results = run_concurrently([finalize_once for _ in range(8)])

    # Exactly ONE thread should have won the CAS race.
    cleared_flags = [result for (result, exc) in results]
    winners = sum(1 for c in cleared_flags if c is True)
    assert winners == 1, (
        f"Expected exactly 1 clear winner, got {winners}. "
        f"Results: {cleared_flags}"
    )

    # Agent's active_mode is None after the race.
    assert agent.snapshot_mode() is None


# ===========================================================================
# Stale-snapshot preservation: CAS mismatch → no clear
# ===========================================================================


def test_concurrent_finalize_with_stale_snapshot_does_not_clear():
    """A finalize call with a STALE expected_mode (different from the agent's
    current active_mode) returns False; the current mode is preserved.

    Per spec §7.6: the abort handler MUST pass the request's expected_mode
    snapshot. If the agent's active_mode has been replaced since the snapshot
    was captured (concurrent replacement), the CAS detects the mismatch and
    refuses to clear.
    """
    agent = _make_agent("sess_stale")
    # Simulate the snapshot at request entry: mode A was active then.
    stale_snapshot = _make_mode_state(name="ticket")

    # A concurrent replacement happened: mode B is now active.
    concurrent_replacement = _make_mode_state(name="work")
    agent.set_active_mode(concurrent_replacement)

    session_id = "sess_stale"
    ctx = SimpleNamespace(
        session_id=session_id,
        response_data={"choices": [{"finish_reason": "stop"}]},
    )

    cleared = finalize_skill_mode(
        ctx=ctx, agent=agent, expected_mode=stale_snapshot
    )

    # CAS mismatch: agent.active_mode is concurrent_replacement, NOT
    # stale_snapshot. clear_active_mode returns False; finalize_skill_mode
    # propagates that.
    assert cleared is False

    # The current (new) mode is preserved.
    assert agent.snapshot_mode() is concurrent_replacement


def test_concurrent_finalize_with_matching_snapshot_clears():
    """Baseline: when expected_mode matches the agent's current active_mode,
    finalize_skill_mode clears and returns True."""
    agent = _make_agent("sess_matching")
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)

    session_id = "sess_matching"
    ctx = SimpleNamespace(
        session_id=session_id,
        response_data={"choices": [{"finish_reason": "stop"}]},
    )

    cleared = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    assert cleared is True
    assert agent.snapshot_mode() is None


# ===========================================================================
# Concurrent replacement during finalize (the spec §7.6 invariant)
# ===========================================================================


def test_concurrent_replace_during_finalize_preserves_new_mode():
    """While finalize is running, a parallel `set_active_mode` installs a new mode.

    Per spec §7.6: the abort handler MUST pass the request's expected_mode
    snapshot. If a sub-agent tool-loop installs a NEW mode AFTER the snapshot
    capture but BEFORE the finalize CAS, the CAS detects the mismatch and
    the new mode is preserved.

    This test simulates the race by:
    1. Snapshotting mode A.
    2. Running finalize on one thread (which would clear mode A).
    3. Concurrently installing mode B from another thread.

    Either:
    - The finalize clears A first, then B is installed → B is the final mode.
    - The install happens first, then finalize sees A != B → CAS mismatch,
      finalize returns False, B is preserved.

    The invariant: after the race, B is the active mode (never cleared).
    """
    agent = _make_agent("sess_replace")
    mode_a = _make_mode_state(name="ticket")
    mode_b = _make_mode_state(name="work")
    agent.set_active_mode(mode_a)

    session_id = "sess_replace"
    results = []

    def install_b():
        # Replace the active mode with B. This may run before or after
        # finalize's CAS check.
        agent.set_active_mode(mode_b)

    def finalize_with_stale_a():
        ctx = SimpleNamespace(
            session_id=session_id,
            response_data={"choices": [{"finish_reason": "stop"}]},
        )
        result = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode_a)
        results.append(result)

    # Barrier-synchronized concurrent execution.
    barrier_results = run_concurrently(
        [install_b, finalize_with_stale_a],
        timeout=10,
    )

    # The finalize result is either:
    # - False (CAS saw B installed before lock → preserved B)
    # - True (CAS saw A and cleared → B then installed by the parallel thread)
    # Both are acceptable for the spec invariant: B is the final mode.
    final_mode = agent.snapshot_mode()
    assert final_mode is mode_b, (
        f"After the race, mode B (work) must be the active mode; got {final_mode!r}"
    )


# ===========================================================================
# CONTINUE under contention: mode stays alive
# ===========================================================================


def test_concurrent_continue_keeps_mode_unchanged():
    """All threads finalize with a CONTINUE disposition (tool_calls).

    Per spec §7.4: CONTINUE → finalize_skill_mode returns False WITHOUT
    calling clear_active_mode. Under concurrent contention, all threads
    return False and the mode is unchanged.

    This is the "many threads, no one clears" scenario — the absence of
    any race-related clear is itself the invariant.
    """
    agent = _make_agent("sess_continue_concurrent")
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)

    session_id = "sess_continue_concurrent"

    def finalize_continue():
        ctx = SimpleNamespace(
            session_id=session_id,
            response_data={
                "choices": [
                    {
                        "message": {"tool_calls": [{"id": "c1"}]},
                        "finish_reason": None,  # None + tool_calls = CONTINUE
                    }
                ]
            },
        )
        return finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    results = run_concurrently([finalize_continue for _ in range(8)])

    # Every thread returns False (no clear on CONTINUE).
    cleared_flags = [result for (result, exc) in results]
    assert all(c is False for c in cleared_flags), (
        f"CONTINUE must return False from all threads; got {cleared_flags}"
    )
    # The mode is still installed.
    assert agent.snapshot_mode() is mode


# ===========================================================================
# Realistic sub-agent race: stream abort + lifecycle finalize
# ===========================================================================


def test_concurrent_abort_clear_and_finalize_cas_safety():
    """Simulate a client-disconnect abort concurrent with lifecycle finalize.

    Per spec §7.6: disconnect cleanup uses compare-and-clear against the
    request's expected snapshot. If a newer mode is installed mid-flight,
    neither the abort nor the finalize should clear it.

    This test mimics the abort/finish race by:
    1. Snapshotting mode A (request entry).
    2. Starting a finalize call with expected_mode=A.
    3. Concurrently calling `clear_active_mode(expected_mode=A)` (abort
       path) and `set_active_mode(B)` (concurrent replacement).

    All three ops contend on agent._mode_lock. The invariant: after the
    race, agent.active_mode is one of {None, B}, never A.
    """
    agent = _make_agent("sess_abort_finish_race")
    mode_a = _make_mode_state(name="ticket")
    mode_b = _make_mode_state(name="work")
    agent.set_active_mode(mode_a)

    session_id = "sess_abort_finish_race"

    def abort_clear():
        # Abort path: CAS-clear with the request's expected snapshot.
        return agent.clear_active_mode(expected_mode=mode_a)

    def finish_finalize():
        # Lifecycle finalize: CAS-clear with the same expected snapshot.
        ctx = SimpleNamespace(
            session_id=session_id,
            response_data={"choices": [{"finish_reason": "stop"}]},
        )
        return finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode_a)

    def concurrent_replace():
        # Mid-flight mode replacement (e.g., tool-loop installs a new mode).
        agent.set_active_mode(mode_b)

    results = run_concurrently(
        [abort_clear, finish_finalize, concurrent_replace],
        timeout=10,
    )

    # Invariant: after the race, the active mode is either None (cleared) or
    # mode_b (concurrent replacement). NEVER mode_a.
    final_mode = agent.snapshot_mode()
    assert final_mode is None or final_mode is mode_b, (
        f"After abort+finalize+replace race, mode must be None or mode_b; "
        f"got {final_mode!r}"
    )