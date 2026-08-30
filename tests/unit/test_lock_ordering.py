"""Pinning tests for lock-ordering hierarchy.

Verifies that lock positions match the documented hierarchy.
Revert-test: changing make_lock(7, ...) to make_lock(8, ...) in agent.py
makes test_agent_source_uses_position_7 FAIL.
"""
import inspect
import os
import threading

import pytest

from proxy.lock_ordering import OrderedLock, make_lock


def test_make_lock_with_position_7():
    """Agent._mode_lock uses position 7 — verify make_lock accepts it."""
    lock = make_lock(7, "Agent._mode_lock", threading.Lock)
    # In non-debug mode, make_lock returns lock_type() — a _thread.lock instance.
    assert hasattr(lock, "acquire") and hasattr(lock, "release")


def test_ordered_lock_position_7():
    """OrderedLock(7) must report the correct position and name."""
    lock = OrderedLock(7, "Agent._mode_lock")
    assert lock.position == 7
    assert lock.name == "Agent._mode_lock"
    lock.acquire()
    lock.release()


def test_full_lock_hierarchy_positions():
    """All 15 positions (0-14) must be usable without error."""
    positions = [
        (0, "AgentRegistry._lock"),
        (1, "context_config_lock"),
        (2, "chat_histories_lock"),
        (3, "chat_summaries_lock"),
        (4, "tool_cache_lock"),
        (5, "request_counter_lock"),
        (6, "active_workspace_lock"),
        (7, "Agent._mode_lock"),
        (8, "AgentTelemetry._lock"),
        (9, "vfs_lock"),
        (10, "_chain_state_lock"),
        (11, "canary_verify_cache_lock"),
        (12, "log_overflow_count_lock"),
        (13, "recovery_lock"),
        (14, "dedup_lock"),
    ]
    for pos, name in positions:
        lock = make_lock(pos, name, threading.Lock)
        assert lock is not None


def test_agent_source_uses_position_7():
    """Verify agent.py __post_init__ calls make_lock(7, ...) — not 8, not 6.

    This is the revert-test: changing make_lock(7, ...) to make_lock(8, ...)
    in agent.py makes this test FAIL because it inspects the source code.
    """
    from proxy.agent.agent import Agent

    source = inspect.getsource(Agent.__post_init__)
    assert "make_lock(7," in source, (
        "Agent.__post_init__ must call make_lock(7, ...) for position 7. "
        f"Source:\n{source}"
    )


def test_lock_ordering_debug_mode_no_violation():
    """In debug mode, acquiring locks in ascending order must not raise."""
    old = os.environ.get("PROXY_DEBUG_LOCK_ORDERING")
    os.environ["PROXY_DEBUG_LOCK_ORDERING"] = "1"

    import importlib
    import proxy.lock_ordering as lo
    importlib.reload(lo)

    locks = [lo.make_lock(i, f"lock_{i}", threading.Lock) for i in range(15)]
    for lock in locks:
        lock.acquire()
    for lock in reversed(locks):
        lock.release()

    # Clean up
    if old is None:
        del os.environ["PROXY_DEBUG_LOCK_ORDERING"]
    else:
        os.environ["PROXY_DEBUG_LOCK_ORDERING"] = old
    importlib.reload(lo)


def test_lock_ordering_debug_mode_detects_violation():
    """In debug mode, acquiring a lower position while holding higher must raise."""
    old = os.environ.get("PROXY_DEBUG_LOCK_ORDERING")
    os.environ["PROXY_DEBUG_LOCK_ORDERING"] = "1"

    import importlib
    import proxy.lock_ordering as lo
    importlib.reload(lo)

    high_lock = lo.make_lock(9, "vfs_lock", threading.Lock)
    low_lock = lo.make_lock(2, "chat_histories_lock", threading.Lock)

    high_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="Lock ordering violation"):
            low_lock.acquire()
    finally:
        high_lock.release()

    # Clean up
    if old is None:
        del os.environ["PROXY_DEBUG_LOCK_ORDERING"]
    else:
        os.environ["PROXY_DEBUG_LOCK_ORDERING"] = old
    importlib.reload(lo)


def test_dedup_lock_uses_position_14():
    """Revert-test: state.py must use make_lock(14, ...) for dedup_lock."""
    import inspect
    import proxy.state as state_mod
    source = inspect.getsource(state_mod)
    assert 'make_lock(14,' in source, (
        "dedup_lock must be at position 14 in the lock hierarchy"
    )
