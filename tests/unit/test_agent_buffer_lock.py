"""M4-1: _buffer_replace_lock must be per-instance on Agent.

Pre-fix: `_buffer_replace_lock: ClassVar[threading.Lock] = threading.Lock()`
was shared across ALL Agent instances. With 100 concurrent agents, every
`append_message` / `extend_messages` / `messages.setter` call serialized
on one mutex — no parallelism between agents.

Fix: convert to per-instance via `field(default_factory=threading.Lock)`.
Each Agent gets its own lock. Outside state.py's lock-ordering hierarchy
(no cross-lock interactions, ever).

This test verifies:
1. The lock objects are distinct between Agent instances.
2. The lock does NOT survive via the ClassVar (mutating one agent's
   instance attribute does not affect another agent).
3. The internal methods (append_message, extend_messages, messages
   setter) acquire self._buffer_replace_lock, not the class attribute.
"""

import threading
import time

import pytest


pytestmark = pytest.mark.unit


class TestBufferReplaceLockIsPerInstance:
    """M4-1: per-instance lock verification."""

    def test_buffer_replace_lock_is_per_instance(self):
        """Two agents have distinct lock objects."""
        from proxy.agent.agent import Agent

        a = Agent(agent_id="a1", session_id="s1")
        b = Agent(agent_id="b1", session_id="s2")

        assert a._buffer_replace_lock is not b._buffer_replace_lock, (
            "Per-instance lock invariant broken: both agents share "
            "the same lock. Pre-fix bug — ClassVar[threading.Lock] was "
            "shared across all 100+ agents."
        )

    def test_classvar_attribute_no_longer_exists(self):
        """The class no longer carries a shared _buffer_replace_lock."""
        from proxy.agent.agent import Agent

        # After the fix, _buffer_replace_lock is an instance attribute
        # provided by `field(default_factory=threading.Lock)`. It should
        # NOT exist as a class-level attribute (which would be shared
        # across all instances).
        assert not hasattr(Agent, "_buffer_replace_lock") or (
            # If it's a class attribute, it should be a Field marker
            # (not a Lock instance).
            not isinstance(getattr(Agent, "_buffer_replace_lock", None), threading.Lock)
        ), (
            "Agent class still has a Lock instance as class attribute. "
            "This means all instances share the lock — the M4-1 bug is back."
        )

    def test_appends_across_agents_run_in_parallel(self):
        """Parallel appends on different agents don't serialize.

        N agents × M appends each. Each agent acquires its OWN lock,
        so the N groups run fully in parallel. Wall time should be
        roughly 1× the per-agent time, NOT N× (which would indicate
        cross-agent serialization via a shared lock).

        Approach: build a MessageBuffer that blocks while held (sleep
        inside the append). The pre-fix shared lock would serialize
        the 4 agents' appends into one big queue (ratio ~N×). The
        post-fix per-instance lock would let each agent's appends
        run in their own thread independently (ratio ~1×).
        """
        import time as _time
        from proxy.agent.agent import Agent
        from proxy.agent.message_buffer import MessageBuffer

        class _BlockingBuffer(MessageBuffer):
            """MessageBuffer that sleeps briefly per append to amplify
            any cross-agent serialization to wall-time visible."""
            def append(self, message):
                _time.sleep(0.001)  # 1ms per append
                super().append(message)

        n_per_agent = 50
        n_agents = 4

        # First, measure single-agent serial time as a baseline.
        solo = Agent(agent_id="solo", session_id="solo")
        solo._buffer = _BlockingBuffer()
        start = _time.perf_counter()
        for i in range(n_per_agent):
            solo.append_message({"role": "user", "content": f"msg-{i}"})
        solo_time = _time.perf_counter() - start

        # Now: N agents in parallel, each appending n_per_agent messages.
        # Pre-fix (shared lock): parallel_time ≈ n_agents × solo_time
        #   (each append blocks the others because they share the
        #   same lock — total work is 4× the single-agent work, and
        #   no parallelism is possible).
        # Post-fix (per-instance lock): parallel_time ≈ solo_time
        #   (each agent has its own lock; the 4 threads can run
        #   independently and the total wall time is just the
        #   single-agent serial time).
        agents = [
            Agent(agent_id=f"agent-{i}", session_id=f"session-{i}")
            for i in range(n_agents)
        ]
        for a in agents:
            a._buffer = _BlockingBuffer()

        threads = []
        for agent in agents:
            t = threading.Thread(
                target=lambda a=agent: [
                    a.append_message({"role": "user", "content": f"msg-{i}"})
                    for i in range(n_per_agent)
                ],
            )
            threads.append(t)

        start = _time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        parallel_time = _time.perf_counter() - start

        # Threshold: with per-instance locks, the 4 threads can
        # proceed in parallel, so parallel_time should be ~ solo_time
        # (small thread overhead). Pre-fix (shared lock) would have
        # serialized to ~ n_agents × solo_time = 4× solo_time.
        # Threshold of 2.0× solo_time is a clear separator.
        ratio = parallel_time / solo_time if solo_time > 0 else 0
        assert ratio < 2.0, (
            f"Parallel appends took {parallel_time:.3f}s vs solo "
            f"{solo_time:.3f}s — ratio {ratio:.2f}. "
            f"Pre-fix (shared lock) would be ~{n_agents}× slower "
            f"(~{n_agents * solo_time:.3f}s). With per-instance locks, "
            f"the {n_agents} agents should run in parallel and total "
            f"wall time should be near solo_time. Ratio > 2.0 strongly "
            f"suggests the lock is still shared."
        )

        # And the messages are all there.
        for i, agent in enumerate(agents):
            msgs = agent.messages
            assert len(msgs) == n_per_agent, (
                f"agent-{i} lost messages: expected {n_per_agent}, got {len(msgs)}"
            )

    def test_lock_is_inside_the_dataclass_field(self):
        """Sanity: the lock is a field, not a ClassVar, on the dataclass."""
        from dataclasses import fields

        from proxy.agent.agent import Agent

        field_names = {f.name for f in fields(Agent)}
        assert "_buffer_replace_lock" in field_names, (
            f"_buffer_replace_lock is not a dataclass field. "
            f"Fields: {sorted(field_names)}"
        )

    def test_concurrent_appends_to_same_agent_serialize(self):
        """Same agent: concurrent appends are still serialized (no data loss)."""
        from proxy.agent.agent import Agent

        agent = Agent(agent_id="serial", session_id="serial")
        n = 200

        def append_many(start_idx):
            for i in range(start_idx, start_idx + n):
                agent.append_message({"role": "user", "content": f"msg-{i}"})

        t1 = threading.Thread(target=append_many, args=(0,))
        t2 = threading.Thread(target=append_many, args=(n,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # All 400 messages must be there (no lost appends).
        msgs = agent.messages
        assert len(msgs) == 2 * n, (
            f"Lost messages under concurrent appends to the SAME agent. "
            f"Expected {2 * n}, got {len(msgs)}. The per-instance lock "
            f"must still serialize same-agent appends — the fix moves "
            f"isolation to the cross-agent level, not the intra-agent level."
        )
