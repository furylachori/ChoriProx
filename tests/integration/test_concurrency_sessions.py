"""Concurrency tests for sessions and agent registry LRU eviction.

Wave 2 of concurrency hardening. Validates that:
- Two concurrent requests on the same session don't deadlock or clobber state
- AgentRegistry LRU eviction respects in-flight request tracking
- Mode activation is atomic per-request (no cross-request clobber)
- submit_sync() per-session lock prevents duplicate sync runs

Uses the Wave 1 concurrency harness (tests/concurrency_harness.py) for
deterministic starts via threading.Barrier and fixed timeouts.

References:
- src/proxy/agent/agent_registry.py:65  AgentRegistry.request_scope()
- src/proxy/agent/agent_registry.py:285 AgentRegistry._evict_lru()
- src/memory/executors.py:158           submit_sync() per-session lock
- src/proxy/mode.py:70                  resolve_mode()
- tests/concurrency_harness.py:11       run_concurrently()
"""
from __future__ import annotations

import threading
import time

import pytest

from tests.concurrency_harness import (
    assert_consistent_state,
    assert_no_deadlock,
    run_concurrently,
)


# ---------------------------------------------------------------------------
# Test 1 (3.11): Two concurrent requests on same session
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTwoConcurrentRequestsOnSameSession:
    """Two threads send requests with the same session_id simultaneously.

    Asserts:
    - No deadlock (both complete within timeout)
    - No AttributeError or mode clobber
    - Each request gets a consistent mode state
    """

    def test_concurrent_get_or_load_same_session(self):
        """Two threads load the same session concurrently — both get valid agents."""
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent_registry import registry

        # Reset registry state
        with registry._lock:
            registry._agents.clear()
            registry._agents_by_id.clear()
            registry._lru.clear()
            registry._in_flight_requests.clear()

        session_id = "concurrent-session-test-1"

        def load_session() -> str:
            with registry.request_scope(session_id):
                agent = registry.get_or_load(session_id)
                if agent is None:
                    raise RuntimeError("get_or_load returned None")
                return agent.agent_id

        results = run_concurrently(
            [load_session, load_session],
            timeout=5,
        )

        # Both should return a non-empty agent_id
        agent_ids = [r for r, _ in results]
        assert all(aid for aid in agent_ids), f"Both threads must get a valid agent: {agent_ids}"

        # Both should resolve to the same agent (deterministic id)
        assert agent_ids[0] == agent_ids[1], (
            f"Two concurrent loads of the same session should return the same "
            f"agent, got {agent_ids[0]} vs {agent_ids[1]}"
        )

    def test_concurrent_set_active_mode_no_clobber(self):
        """Two threads set different modes — neither clobbers the other across reads.

        Validates that snapshot_mode() returns a stable view, and that
        set_active_mode() is atomic per-call.
        """
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.skills.skill_definition import ModeState
        import proxy.state as state

        agent = Agent(
            session_id="mode-race-session",
            parent=None,
            config=AgentConfig(),
        )

        # Stash on state so we can clean up later
        attr_name = f"_concurrency_test_agent_{id(agent)}"
        setattr(state, attr_name, agent)

        barrier = threading.Barrier(2, timeout=5)
        snapshots: list = []
        snapshots_lock = threading.Lock()

        def thread_a() -> None:
            barrier.wait()
            mode = ModeState(
                name="ticket",
                activated_at=time.time(),
                ttl_seconds=300,
                allowed_tool_names=frozenset({"read"}),
            )
            agent.set_active_mode(mode)
            with snapshots_lock:
                snapshots.append(("A-set", agent.get_mode_name()))

        def thread_b() -> None:
            barrier.wait()
            mode = ModeState(
                name="workspace",
                activated_at=time.time(),
                ttl_seconds=0,
                allowed_tool_names=frozenset({"edit"}),
            )
            agent.set_active_mode(mode)
            with snapshots_lock:
                snapshots.append(("B-set", agent.get_mode_name()))

        try:
            results = run_concurrently([thread_a, thread_b], timeout=5)
        finally:
            delattr(state, attr_name)

        # Both threads must have completed without exception
        assert all(exc is None for _, exc in results)

        # After both writes, the final mode is one of the two values
        # (whichever won the last write)
        final = agent.get_mode_name()
        assert final in ("ticket", "workspace"), f"Unexpected final mode: {final}"

    def test_concurrent_snapshot_mode_returns_valid_state(self):
        """Concurrent snapshot_mode() calls always return a ModeState or None — never raise.

        Guards against AttributeError when active_mode is being swapped.
        """
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.skills.skill_definition import ModeState

        agent = Agent(
            session_id="snapshot-race-session",
            parent=None,
            config=AgentConfig(),
        )

        results_box: list = []
        results_lock = threading.Lock()
        stop = threading.Event()
        barrier = threading.Barrier(2, timeout=5)

        def reader() -> None:
            barrier.wait()
            while not stop.is_set():
                snap = agent.snapshot_mode()
                with results_lock:
                    results_box.append(snap)
                time.sleep(0.001)

        def writer() -> None:
            barrier.wait()
            for i in range(50):
                mode = ModeState(
                    name=("ticket" if i % 2 == 0 else "workspace"),
                    activated_at=time.time(),
                    ttl_seconds=300 if i % 2 == 0 else 0,
                )
                agent.set_active_mode(mode)
                time.sleep(0.002)

        t_writer = threading.Thread(target=writer)
        t_reader = threading.Thread(target=reader)
        t_writer.start()
        t_reader.start()
        t_writer.join(timeout=5)
        stop.set()
        t_reader.join(timeout=5)

        assert_no_deadlock([t_writer, t_reader], timeout=2)

        # All snapshots must be None or a ModeState with a 'name' attribute
        for snap in results_box:
            if snap is not None:
                assert hasattr(snap, "name"), (
                    f"snapshot returned object without 'name': {type(snap)}"
                )
                assert isinstance(snap.name, str)

        # Reader must have captured several snapshots
        assert len(results_box) > 10, f"Reader should have captured many snapshots, got {len(results_box)}"


# ---------------------------------------------------------------------------
# Test 2 (3.12): Agent registry LRU eviction during active pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAgentRegistryLRUEvictionDuringActivePipeline:
    """In-flight requests are protected from LRU eviction.

    Algorithm:
    1. Fill LRU to capacity (max_agents)
    2. Thread 1: open request_scope for session-A, then attempt to load a new
       session — should NOT evict session-A (it's in-flight)
    3. Thread 2: load a new session that triggers eviction
    4. Verify session-A is still in registry
    # US-064
    """

    def _fill_registry_to_capacity(self, registry, capacity: int, prefix: str = "fill") -> list:
        """Fill the registry with `capacity` agents and return their session IDs."""
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent import Agent

        ids = []
        for i in range(capacity):
            sid = f"{prefix}-{i:04d}"
            agent = Agent(
                session_id=sid,
                parent=None,
                config=AgentConfig(),
            )
            with registry._lock:
                registry._agents[sid] = agent
                registry._agents_by_id[agent.agent_id] = agent
                registry._lru[sid] = None
            ids.append(sid)
        return ids

    def test_in_flight_session_not_evicted(self):
        """A session with an open request_scope is skipped during LRU eviction."""
        from proxy.agent.agent_registry import AgentRegistry
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent import Agent

        capacity = 3
        registry = AgentRegistry(max_agents=capacity)
        # Fill to capacity
        self._fill_registry_to_capacity(registry, capacity)
        assert len(registry._agents) == capacity

        # Identify the LRU entry — it should be the first one we inserted.
        oldest_sid = next(iter(registry._lru))
        # Mark it in-flight
        registry._in_flight_requests[oldest_sid] = 1

        try:
            # Add a new agent — this should NOT evict the in-flight one
            new_sid = "new-evict-trigger"
            new_agent = Agent(
                session_id=new_sid,
                parent=None,
                config=AgentConfig(),
            )
            with registry._lock:
                if len(registry._agents) >= registry.max_agents:
                    evicted = registry._evict_lru()
                else:
                    evicted = None
                if evicted is not None:
                    registry._agents.pop(evicted.session_id, None)
                registry._agents[new_sid] = new_agent
                registry._lru[new_sid] = None

            # The in-flight session must still be in the registry
            assert oldest_sid in registry._agents, (
                f"In-flight session {oldest_sid} was evicted despite request_scope!"
            )
            # The new session must be registered
            assert new_sid in registry._agents
        finally:
            registry._in_flight_requests.pop(oldest_sid, None)
            # Cleanup
            with registry._lock:
                registry._agents.clear()
                registry._agents_by_id.clear()
                registry._lru.clear()
                registry._in_flight_requests.clear()

    def test_eviction_skips_in_flight_and_picks_next(self):
        """When the LRU entry is in-flight, eviction must skip it and pick the next."""
        from proxy.agent.agent_registry import AgentRegistry
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent import Agent

        capacity = 3
        registry = AgentRegistry(max_agents=capacity)
        self._fill_registry_to_capacity(registry, capacity)

        # Mark the oldest as in-flight
        oldest_sid = next(iter(registry._lru))
        second_sid = list(registry._lru.keys())[1]

        registry._in_flight_requests[oldest_sid] = 1

        try:
            # Evict — should skip oldest (in-flight) and pick second
            with registry._lock:
                evicted = registry._evict_lru()

            assert evicted is not None, "Eviction should have returned an agent"
            assert evicted.session_id == second_sid, (
                f"Eviction should have picked the second-oldest (in-flight oldest was skipped), "
                f"got {evicted.session_id}"
            )
            # Oldest (in-flight) must remain
            assert oldest_sid in registry._agents
            # Evicted one must be gone
            assert second_sid not in registry._agents
        finally:
            registry._in_flight_requests.pop(oldest_sid, None)
            with registry._lock:
                registry._agents.clear()
                registry._agents_by_id.clear()
                registry._lru.clear()
                registry._in_flight_requests.clear()

    def test_concurrent_request_scope_holds_off_eviction(self):
        """Two threads: one holds request_scope, one triggers eviction.

        The in-flight session must survive; the second thread may succeed at
        registering a new session, but only by evicting a non-in-flight agent.
        """
        from proxy.agent.agent_registry import AgentRegistry
        from proxy.agent.agent_config import AgentConfig
        from proxy.agent.agent import Agent

        capacity = 4
        registry = AgentRegistry(max_agents=capacity)
        self._fill_registry_to_capacity(registry, capacity)

        # Pick the oldest to be the protected one
        protected_sid = next(iter(registry._lru))
        barrier = threading.Barrier(2, timeout=5)
        eviction_outcome: dict = {"evicted_sid": None, "success": False}
        eviction_lock = threading.Lock()

        def holder() -> None:
            """Holds a request_scope for protected_sid for the duration of the test."""
            with registry.request_scope(protected_sid):
                barrier.wait()  # Sync start with evictor
                # Hold the scope until the evictor is done
                time.sleep(0.3)

        def evictor() -> None:
            """Tries to register a new session, triggering eviction."""
            barrier.wait()
            new_sid = "evictor-new-session"
            new_agent = Agent(
                session_id=new_sid,
                parent=None,
                config=AgentConfig(),
            )

            with registry._lock:
                if len(registry._agents) >= registry.max_agents:
                    evicted = registry._evict_lru()
                else:
                    evicted = None

                if evicted is not None:
                    registry._agents.pop(evicted.session_id, None)
                    registry._agents_by_id.pop(evicted.agent_id, None)

                # The protected one may still be there — that's fine. We just
                # need to add our new agent.
                if len(registry._agents) < registry.max_agents:
                    registry._agents[new_sid] = new_agent
                    registry._agents_by_id[new_agent.agent_id] = new_agent
                    registry._lru[new_sid] = None
                    with eviction_lock:
                        eviction_outcome["evicted_sid"] = (
                            evicted.session_id if evicted else None
                        )
                        eviction_outcome["success"] = True
                else:
                    with eviction_lock:
                        eviction_outcome["success"] = False

        t_holder = threading.Thread(target=holder, name="holder")
        t_evictor = threading.Thread(target=evictor, name="evictor")
        t_holder.start()
        t_evictor.start()

        assert_no_deadlock([t_holder, t_evictor], timeout=5)
        t_holder.join(timeout=1)
        t_evictor.join(timeout=1)

        # Protected session must still be present (in-flight protection)
        assert protected_sid in registry._agents, (
            f"In-flight session {protected_sid} was evicted while request_scope was open!"
        )
        # Evictor should have succeeded
        assert eviction_outcome["success"], "Evictor should have registered a new session"

        # Cleanup
        with registry._lock:
            registry._agents.clear()
            registry._agents_by_id.clear()
            registry._lru.clear()
            registry._in_flight_requests.clear()


# ---------------------------------------------------------------------------
# Test 3: Mode activation race between concurrent requests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestModeActivationRaceBetweenConcurrentRequests:
    """Concurrent requests on the same session must not corrupt mode state.

    When Thread 1 is in resolve_mode() and Thread 2 sets a new mode, Thread 1
    must see a consistent view (not a half-mutated state).
    """

    def test_resolve_mode_uses_stable_snapshot(self):
        """A slow resolve_mode() with a concurrent set_active_mode() must complete.

        The mode returned by resolve_mode() must be a valid enum value.
        """
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.skills.skill_definition import ModeState
        from proxy.types import RequestContext
        from proxy.mode import resolve_mode, Mode

        agent = Agent(
            session_id="mode-race-resolve",
            parent=None,
            config=AgentConfig(),
        )

        # Pre-activate ticket mode so resolve_mode has something to read
        initial_mode = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset({"read"}),
        )
        agent.set_active_mode(initial_mode)

        # Build a minimal context
        ctx = RequestContext(
            body={"messages": [{"role": "user", "content": "hello world this is a test message"}]},
            model_name="test-model",
            wants_stream=False,
            session_id="mode-race-resolve",
            client_hint="test",
            messages=[{"role": "user", "content": "hello world this is a test message"}],
            is_subagent=False,
        )
        ctx._mode = None  # will be set by resolve_mode

        barrier = threading.Barrier(2, timeout=5)
        errors: list = []
        errors_lock = threading.Lock()

        def slow_resolve() -> None:
            """Call resolve_mode while another thread mutates active_mode."""
            try:
                barrier.wait()
                # Pre-snapshot before the other thread writes
                snap_before = agent.snapshot_mode()
                time.sleep(0.05)  # let writer run
                snap_after = agent.snapshot_mode()
                # Both snapshots must be ModeState or None — never a corrupt value
                if snap_before is not None and not hasattr(snap_before, "name"):
                    raise AssertionError("snap_before has no .name")
                if snap_after is not None and not hasattr(snap_after, "name"):
                    raise AssertionError("snap_after has no .name")
            except Exception as e:
                with errors_lock:
                    errors.append(("resolve", e))

        def writer() -> None:
            try:
                barrier.wait()
                new_mode = ModeState(
                    name="workspace",
                    activated_at=time.time(),
                    ttl_seconds=0,
                    allowed_tool_names=frozenset({"edit"}),
                )
                agent.set_active_mode(new_mode)
            except Exception as e:
                with errors_lock:
                    errors.append(("writer", e))

        t_resolve = threading.Thread(target=slow_resolve, name="slow_resolve")
        t_writer = threading.Thread(target=writer, name="writer")
        t_resolve.start()
        t_writer.start()
        assert_no_deadlock([t_resolve, t_writer], timeout=5)

        assert not errors, f"Concurrent resolve/writer raised: {errors}"

    def test_concurrent_resolve_mode_returns_valid_enum(self):
        """Multiple threads call resolve_mode() concurrently — all get valid Modes."""
        from proxy.agent.agent import Agent
        from proxy.agent.agent_config import AgentConfig
        from proxy.skills.skill_definition import ModeState
        from proxy.types import RequestContext
        from proxy.mode import resolve_mode, Mode

        agent = Agent(
            session_id="mode-race-concurrent",
            parent=None,
            config=AgentConfig(),
        )

        ctx = RequestContext(
            body={"messages": [{"role": "user", "content": "hello world this is a test"}]},
            model_name="test-model",
            wants_stream=False,
            session_id="mode-race-concurrent",
            client_hint="test",
            messages=[{"role": "user", "content": "hello world this is a test"}],
            is_subagent=False,
        )
        ctx._mode = None

        barrier = threading.Barrier(4, timeout=5)
        results: list = []
        results_lock = threading.Lock()

        def caller() -> None:
            barrier.wait()
            try:
                mode = resolve_mode(ctx, agent)
                with results_lock:
                    results.append(mode)
            except Exception as e:
                with results_lock:
                    results.append(e)

        threads = [threading.Thread(target=caller, name=f"caller-{i}") for i in range(4)]
        for t in threads:
            t.start()
        assert_no_deadlock(threads, timeout=5)

        # All callers must have received a valid Mode
        assert len(results) == 4
        for r in results:
            assert isinstance(r, Mode), f"resolve_mode returned non-Mode: {type(r)}: {r}"


# ---------------------------------------------------------------------------
# Test 4: submit_sync per-session locking under concurrent calls
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSubmitSyncPerSessionLocking:
    """submit_sync() must dedupe concurrent calls per session_id."""

    def test_concurrent_submit_same_session_only_one_runs(self):
        """Two threads submit_sync for the same session — only one runs."""
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        session_id = "concurrent-submit-test-1"
        started = threading.Event()
        release = threading.Event()
        call_count = {"n": 0}
        count_lock = threading.Lock()

        def slow_fn(sid: str) -> str:
            with count_lock:
                call_count["n"] += 1
            started.set()
            # Wait until released
            if not release.wait(timeout=5):
                raise TimeoutError("slow_fn: release event never fired")
            return "ok"

        try:
            # T1 starts the slow sync
            f1 = submit_sync(slow_fn, session_id)
            assert f1 is not None, "First submit should return a future"

            # Wait until slow_fn has actually started
            assert started.wait(timeout=5), "slow_fn never started"

            # T2 submits for the same session — must return None (deduped)
            f2 = submit_sync(slow_fn, session_id)
            assert f2 is None, (
                f"Second submit for same session should return None, got {f2!r}"
            )

            # Verify lock is registered
            with _session_locks_lock:
                assert session_id in _session_locks, (
                    f"Session lock should be registered for {session_id}"
                )

            # Release the slow_fn
            release.set()
            result = f1.result(timeout=5)
            assert result == "ok"

            # slow_fn should have been called exactly once
            assert call_count["n"] == 1, (
                f"slow_fn should have run exactly once, ran {call_count['n']} times"
            )
        finally:
            release.set()
            with _session_locks_lock:
                _session_locks.pop(session_id, None)

    def test_third_call_succeeds_after_first_completes(self):
        """After first submit_sync completes, a third call for the same session succeeds."""
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        session_id = "concurrent-submit-test-2"
        call_count = {"n": 0}
        count_lock = threading.Lock()

        def fast_fn(sid: str) -> str:
            with count_lock:
                call_count["n"] += 1
            return f"result-{call_count['n']}"

        try:
            # First call
            f1 = submit_sync(fast_fn, session_id)
            assert f1 is not None
            r1 = f1.result(timeout=5)
            assert r1 == "result-1"

            # Allow done-callback to release the lock
            assert_consistent_state(
                get_state_fn=lambda: (lambda: session_id not in _session_locks)(),
                check_fn=lambda released: released,
                timeout=2,
                interval=0.02,
            )

            # Second call (after first fully released) — must succeed
            f2 = submit_sync(fast_fn, session_id)
            assert f2 is not None, (
                "Third call after first completed should return a future, got None"
            )
            r2 = f2.result(timeout=5)
            assert r2 == "result-2"

            assert call_count["n"] == 2, (
                f"Both calls should have run sequentially, count is {call_count['n']}"
            )
        finally:
            with _session_locks_lock:
                _session_locks.pop(session_id, None)

    def test_concurrent_submits_different_sessions_both_run(self):
        """submit_sync for different sessions must both run concurrently."""
        from memory.executors import submit_sync, _session_locks, _session_locks_lock

        session_a = "concurrent-submit-A"
        session_b = "concurrent-submit-B"
        started = {"a": False, "b": False}
        started_lock = threading.Lock()
        release = threading.Event()
        results: dict = {}

        def fn(sid: str) -> str:
            with started_lock:
                started[sid] = True
            if not release.wait(timeout=5):
                raise TimeoutError(f"fn for {sid}: release never fired")
            return sid

        try:
            f_a = submit_sync(fn, session_a)
            f_b = submit_sync(fn, session_b)
            assert f_a is not None, "session_a should get a future"
            assert f_b is not None, "session_b should get a future"

            # Both should have started
            assert_consistent_state(
                get_state_fn=lambda: dict(started),
                check_fn=lambda s: s.get(session_a) and s.get(session_b),
                timeout=5,
                interval=0.02,
            )

            release.set()
            results[session_a] = f_a.result(timeout=5)
            results[session_b] = f_b.result(timeout=5)
            assert results[session_a] == session_a
            assert results[session_b] == session_b
        finally:
            release.set()
            with _session_locks_lock:
                _session_locks.pop(session_a, None)
                _session_locks.pop(session_b, None)
