"""Runtime lock-ordering enforcement for debug mode.

When ``PROXY_DEBUG_LOCK_ORDERING=1`` is set, every lock defined in the
ordering hierarchy (state.py:86-93) is replaced with an ``OrderedLock``
wrapper that asserts ascending acquisition order per-thread.

Lock hierarchy (from state.py):
    #0 AgentRegistry._lock
    #1 context_config_lock
    #2 chat_histories_lock
    #3 chat_summaries_lock
    #4 tool_cache_lock
    #5 request_counter_lock
    #6 active_workspace_lock
    #7 Agent._mode_lock
    #8 AgentTelemetry._lock
    #9 vfs_lock
    #10 _chain_state_lock
    #11 canary_verify_cache_lock
    #12 log_overflow_count_lock     ← leaf lock; never acquires other locks
    #13 recovery_lock              (RLock, leaf — memory.db_recovery)
    #14 dedup_lock                 (leaf — DedupStore keyed completion map)

Rule: never acquire a lower-numbered lock while holding a higher-numbered one.
"""
import os
import threading
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

_LOCK_ORDERING_ENABLED = os.environ.get("PROXY_DEBUG_LOCK_ORDERING", "") == "1"

_thread_local = threading.local()


class OrderedLock:
    """Drop-in replacement for ``threading.RLock`` that enforces lock ordering.

    In debug mode (``PROXY_DEBUG_LOCK_ORDERING=1``), every ``acquire()``
    checks that no lock with a *higher* position number is currently held
    by this thread.  Violations raise ``RuntimeError`` immediately instead
    of silently deadlocking.

    Re-entrancy is fully supported: the same thread may acquire the same
    lock multiple times (RLock semantics), and the ordering check is
    skipped for re-entrant acquisitions.
    """

    def __init__(self, position: int, name: str):
        self._position = position
        self._name = name
        self._lock = threading.RLock()

    @property
    def position(self) -> int:
        return self._position

    @property
    def name(self) -> str:
        return self._name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if _LOCK_ORDERING_ENABLED:
            held = getattr(_thread_local, "held_locks", None)
            if held is None:
                held = []
                _thread_local.held_locks = held

            is_reentrant = any(p == self._position for p, _ in held)

            if held and not is_reentrant:
                highest_held_pos = max(p for p, _ in held)
                if self._position < highest_held_pos:
                    held_names = [(p, n) for p, n in held]
                    logger.error(
                        "LOCK ORDERING VIOLATION: acquiring #%d (%s) while "
                        "holding #%d — held locks: %s",
                        self._position,
                        self._name,
                        highest_held_pos,
                        held_names,
                    )
                    raise RuntimeError(
                        f"Lock ordering violation: acquiring #{self._position} "
                        f"({self._name}) while holding #{highest_held_pos}"
                    )

        if timeout == -1:
            result = self._lock.acquire(blocking=blocking)
        else:
            result = self._lock.acquire(blocking=blocking, timeout=timeout)

        if result and _LOCK_ORDERING_ENABLED:
            held = getattr(_thread_local, "held_locks", None)
            if held is None:
                held = []
                _thread_local.held_locks = held
            held.append((self._position, self._name))
        return result

    def release(self) -> None:
        if _LOCK_ORDERING_ENABLED:
            held = getattr(_thread_local, "held_locks", None)
            if held:
                for i in range(len(held) - 1, -1, -1):
                    if held[i][0] == self._position:
                        held.pop(i)
                        break
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
        return False

    def __repr__(self) -> str:
        return f"OrderedLock(#{self._position} {self._name})"


def make_lock(
    position: int,
    name: str,
    lock_type: type = threading.RLock,
    enabled: Optional[bool] = None,
) -> Union[threading.Lock, OrderedLock]:
    """Factory: returns ``OrderedLock`` when debug mode is on, ``lock_type`` otherwise.

    Args:
        position: Lock position in the hierarchy (0-14).
        name: Human-readable name for debugging.
        lock_type: Lock class to use in production (default ``threading.RLock``).
        enabled: Override for debug mode detection (default: env var).

    Returns:
        ``OrderedLock`` in debug mode, ``lock_type()`` in production.
    """
    if enabled is None:
        enabled = _LOCK_ORDERING_ENABLED
    if enabled:
        return OrderedLock(position, name)
    return lock_type()
