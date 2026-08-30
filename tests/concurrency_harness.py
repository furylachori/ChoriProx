"""Concurrency harness — deterministic threading helpers for tests.

Uses threading.Barrier for synchronized starts and fixed timeouts
to eliminate platform-dependent flakiness.
"""

import threading
import time


def run_concurrently(callables, timeout=10):
    """Run N callables in threads, collect results/exceptions.

    Uses threading.Barrier for deterministic start — all threads block
    until every thread has called barrier.wait(), then they all proceed
    simultaneously.

    Returns list of (result, exception) tuples, one per callable.
    If a callable returns normally, result is the return value and exception is None.
    If a callable raises, result is None and exception is the raised exception.
    """
    n = len(callables)
    if n == 0:
        return []

    barrier = threading.Barrier(n)
    results = [None] * n
    exceptions = [None] * n

    def worker(idx, fn):
        try:
            barrier.wait(timeout=timeout)
            results[idx] = fn()
        except Exception as e:
            exceptions[idx] = e

    threads = [
        threading.Thread(target=worker, args=(i, fn))
        for i, fn in enumerate(callables)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)

    # Re-raise any thread exceptions in the main thread
    for exc in exceptions:
        if exc is not None:
            raise exc

    return list(zip(results, exceptions))


def assert_no_deadlock(threads, timeout=5):
    """Join threads with timeout. Fail if any thread is still alive."""
    alive = []
    for t in threads:
        t.join(timeout=timeout)
        if t.is_alive():
            alive.append(t)
    if alive:
        names = [getattr(t, "name", repr(t)) for t in alive]
        raise AssertionError(
            f"Potential deadlock: {len(alive)} thread(s) still alive after "
            f"{timeout}s: {', '.join(names)}"
        )


def assert_consistent_state(get_state_fn, check_fn, timeout=5, interval=0.1):
    """Poll a state function and assert invariant holds after concurrent ops.

    Args:
        get_state_fn: Callable returning the current state to check.
        check_fn: Callable(state) -> bool. Must return True when invariant holds.
        timeout: Max seconds to poll before failing.
        interval: Seconds between polls.

    Raises:
        AssertionError if check_fn never returns True within timeout.
    """
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        state = get_state_fn()
        last_state = state
        if check_fn(state):
            return
        time.sleep(interval)
    raise AssertionError(
        f"State invariant violated for {timeout}s. Last state: {last_state!r}"
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_run_concurrently_basic():
    results = run_concurrently([
        lambda: 1 + 1,
        lambda: 2 + 2,
        lambda: 3 + 3,
    ])
    values = [r for r, _ in results]
    assert sorted(values) == [2, 4, 6]


def test_run_concurrently_barrier_sync():
    """Verify all threads start at the same time."""
    import threading as _thr
    started = []
    lock = _thr.Lock()

    def worker():
        with lock:
            started.append(time.monotonic())

    run_concurrently([worker, worker, worker, worker, worker])
    assert len(started) == 5
    # All started within a small window (barrier guarantees near-simultaneous start)
    spread = max(started) - min(started)
    assert spread < 1.0, f"Thread start spread too large: {spread:.3f}s"


def test_run_concurrently_exception_propagation():
    """Exceptions from threads propagate to main thread."""
    import pytest
    with pytest.raises(ValueError, match="boom"):
        run_concurrently([
            lambda: 42,
            lambda: (_ for _ in ()).throw(ValueError("boom")),
        ])


def test_assert_no_deadlock_passes():
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait(timeout=2)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    assert_no_deadlock(threads, timeout=3)


def test_assert_consistent_state_immediate():
    assert_consistent_state(
        get_state_fn=lambda: {"count": 5},
        check_fn=lambda s: s["count"] == 5,
        timeout=1,
    )


def test_assert_consistent_state_eventual():
    """State that becomes consistent after a short delay."""
    state = {"ready": False}

    def flip():
        time.sleep(0.15)
        state["ready"] = True

    t = threading.Thread(target=flip)
    t.start()
    assert_consistent_state(
        get_state_fn=lambda: state,
        check_fn=lambda s: s["ready"],
        timeout=2,
        interval=0.05,
    )
    t.join()
