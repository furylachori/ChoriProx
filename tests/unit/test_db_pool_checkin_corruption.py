"""T3: corruption during checkin poisons the source and wakes a concurrent
waiter parked on the pool condition for `_retiring`. Deterministic (Event/probe
based, no time.sleep for the race window)."""
import sqlite3
import threading
import time

import pytest

import memory.db_pool as db_pool


class _ParkProbe:
    """Wrap a threading.Condition; set `parked` when the target thread parks."""
    def __init__(self, real, ident_holder, parked_event):
        self._real = real
        self._holder = ident_holder
        self._parked = parked_event
    def __enter__(self):
        return self._real.__enter__()
    def __exit__(self, *a):
        return self._real.__exit__(*a)
    def wait(self, timeout=None):
        if threading.get_ident() == self._holder["v"]:
            self._parked.set()
        return self._real.wait(timeout)
    def wait_for(self, predicate, timeout=None):
        """Mirror Condition.wait_for through this wrapper's wait method."""
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = time.monotonic() + waittime
                else:
                    waittime = endtime - time.monotonic()
                if waittime <= 0:
                    break
            self.wait(waittime)
            result = predicate()
        return result
    def notify(self, n=1):
        return self._real.notify(n)
    def notify_all(self):
        return self._real.notify_all()
    def acquire(self, *a, **k):
        return self._real.acquire(*a, **k)
    def release(self):
        return self._real.release()


class _RollbackCorruptConn:
    """Wraps a real sqlite conn; rollback() raises a corruption-classified error."""
    def __init__(self, real):
        self._real = real
        self.closed = False
    def rollback(self):
        raise sqlite3.DatabaseError("database disk image is malformed")
    def close(self):
        self.closed = True
        try:
            self._real.close()
        except Exception:
            pass
    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.integration
def test_corruption_on_checkin_rollback_wakes_waiter(tmp_path):
    db_path = str(tmp_path / "checkin-corrupt.db")
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (v INTEGER)")
    seed.close()

    pool = db_pool.ConnectionPool(db_path, pool_size=4, lazy=True)
    lease = pool.checkout(validate=False)
    # Force rollback to raise a corruption error during checkin.
    object.__setattr__(lease, "_conn", _RollbackCorruptConn(lease._conn))

    assert pool._retiring is False

    waiter_ident = {"v": None}
    waiter_parked = threading.Event()
    waiter_saw_retiring = threading.Event()
    pool._condition = _ParkProbe(pool._condition, waiter_ident, waiter_parked)

    def waiter():
        waiter_ident["v"] = threading.get_ident()
        with pool._condition:
            deadline_iters = 0
            while not pool._retiring and deadline_iters < 100:
                pool._condition.wait(timeout=5.0)
                deadline_iters += 1
            if pool._retiring:
                waiter_saw_retiring.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    assert waiter_parked.wait(timeout=5.0), "waiter never parked on the pool condition"

    # Corruption here: checkin -> rollback raises malformed -> _retiring set + notify.
    pool.checkin(lease)

    assert waiter_saw_retiring.wait(timeout=5.0), (
        "waiter was not woken when checkin corruption set _retiring"
    )
    assert pool._retiring is True
    assert pool._corruption_error is not None
    assert lease._released is True
    t.join(timeout=5.0)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# Slice 5A — _ParkProbe.wait_for must instrument the actual wait path
# ---------------------------------------------------------------------------


def test_park_probe_wait_for_initial_true_does_not_wait_checkin_corruption():
    """An initially true predicate returns without signaling a park."""
    parked = threading.Event()
    holder = {"v": threading.get_ident()}
    predicate_calls = [0]
    probe = _ParkProbe(
        threading.Condition(),
        holder,
        parked,
    )

    def predicate():
        predicate_calls[0] += 1
        return True

    with probe:
        result = probe.wait_for(predicate, timeout=5.0)

    assert result is True
    assert predicate_calls == [1]
    assert not parked.is_set()


def test_park_probe_wait_for_false_predicate_enters_wait_checkin_corruption():
    """A false predicate must release the condition lock and really wait."""
    parked = threading.Event()
    holder = {"v": None}
    ready = {"value": False}
    results = []
    errors = []
    probe = _ParkProbe(
        threading.Condition(),
        holder,
        parked,
    )

    def waiter():
        holder["v"] = threading.get_ident()
        try:
            with probe:
                results.append(
                    probe.wait_for(
                        lambda: ready["value"],
                        timeout=5.0,
                    )
                )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()

    parked_observed = False
    try:
        parked_observed = parked.wait(timeout=5.0)
    finally:
        with probe:
            ready["value"] = True
            probe.notify_all()
        thread.join(timeout=5.0)

    assert parked_observed, "wait_for never entered the actual wait path"
    assert not thread.is_alive()
    assert errors == []
    assert results == [True]


def test_park_probe_wait_for_false_then_true_waits_checkin_corruption():
    """A false-then-true predicate must wait between its evaluations."""
    parked = threading.Event()
    holder = {"v": threading.get_ident()}
    predicate_calls = [0]
    wait_calls = []
    real_condition = threading.Condition()
    original_wait = real_condition.wait

    def recording_wait(timeout=None):
        wait_calls.append(timeout)
        return False

    real_condition.wait = recording_wait
    probe = _ParkProbe(
        real_condition,
        holder,
        parked,
    )

    def predicate():
        predicate_calls[0] += 1
        return predicate_calls[0] >= 2

    try:
        with probe:
            result = probe.wait_for(predicate, timeout=1.0)
    finally:
        real_condition.wait = original_wait

    assert result is True
    assert predicate_calls == [2]
    assert wait_calls == [1.0]
    assert parked.is_set()
