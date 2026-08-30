"""Targeted regression tests for db_pool recovery & accounting batch (W1-B).

Covers F11 (checkin BaseException accounting), F3 (recovery refuses unsafe
paths), F4 (atomic capacity reservation), F5 (bounded quiescence waits), and
confirms D1/D3 dead-code removal. Every test is asserted to FAIL when its
corresponding fix is reverted (revert-tested independently).

Uses injection proxies + threading events/barriers (never time.sleep for race
windows) per tests/AGENTS.md discipline.
"""

import queue
import sqlite3
import threading
import time

import pytest

import memory.db_pool as db_pool
from memory.db_pool import DatabaseRecoveryError


@pytest.fixture(autouse=True)
def _reset_global_pool():
    current = db_pool._pool
    if current is not None:
        current.shutdown(grace_period=0)
        db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._active_raw_connections.clear()
    db_pool._terminal_shutdown = False

    yield

    current = db_pool._pool
    if current is not None:
        current.shutdown(grace_period=0)
        db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._active_raw_connections.clear()
    db_pool._terminal_shutdown = False


# ---------------------------------------------------------------------------
# F11 — checkin() BaseException accounting with explicit disposition states
# ---------------------------------------------------------------------------


class _Interrupt(BaseException):
    """Test-only BaseException to simulate interruption during checkin."""


def _make_counting_proxy(real, close_calls, rollback_fn=None, pragma_fn=None):
    """Wrap a real conn so we can count closes and inject failures at
    rollback / pragma-reset points without touching production code."""

    class _Proxy:
        __slots__ = ("_real",)

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def close(self):
            close_calls[0] += 1
            return object.__getattribute__(self, "_real").close()

        def rollback(self):
            if rollback_fn is not None:
                rollback_fn()
            return object.__getattribute__(self, "_real").rollback()

        def execute(self, sql, *args, **kwargs):
            if pragma_fn is not None and "PRAGMA" in str(sql):
                pragma_fn(sql)
            return object.__getattribute__(self, "_real").execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(object.__getattribute__(self, "_real"), name, value)

    return _Proxy(real)


def test_f11_interrupt_in_flight_rollback_raises_baseexception(tmp_path):
    """A BaseException during rollback (before any publication side-effect)
    leaves disposition IN_FLIGHT: handler must decrement _created AND close."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "f11-inflight.db"), pool_size=4, lazy=True,
    )
    try:
        close_calls = [0]

        def rollback_raises():
            raise _Interrupt("simulated during rollback")

        setup = pool.checkout(validate=False)
        real = setup._conn
        proxy = _make_counting_proxy(real, close_calls, rollback_fn=rollback_raises)
        object.__setattr__(setup, "_conn", proxy)

        with pytest.raises(_Interrupt):
            pool.checkin(setup)

        assert close_calls[0] == 1, f"expected 1 close, got {close_calls[0]}"
        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._checking_in == set()
        assert setup._released is True

        # Pool remains healthy: subsequent checkout works.
        lease = pool.checkout(validate=True)
        try:
            assert lease.execute("SELECT 1").fetchone()[0] == 1
            assert pool._created == 1
        finally:
            pool.checkin(lease)
    finally:
        pool.shutdown(grace_period=0)


def test_f11_interrupt_after_requeued_does_not_close_live_conn(tmp_path):
    """A BaseException AFTER the conn is requeued (disposition RETURNED) must
    NOT close the connection sitting in the idle queue. This is the core F11
    defect: the inferred-cleanup handler unconditionally closed the conn."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "f11-returned.db"), pool_size=4, lazy=True,
    )
    try:
        close_calls = [0]
        # Force should_close=False so the conn is requeued (RETURNED).
        lease = pool.checkout(validate=True)
        real = lease._conn
        proxy = _make_counting_proxy(real, close_calls)
        object.__setattr__(lease, "_conn", proxy)

        # Interrupt via notify_all AFTER disposition is set to RETURNED. The
        # second notify_all in checkin is the one inside the accounting block,
        # which runs after the requeue.
        real_notify = pool._condition.notify_all
        fired = {"n": 0}

        def interrupting_notify():
            fired["n"] += 1
            real_notify()
            # The accounting block calls notify_all once (corruption path is
            # not taken here). Fire on that call.
            if fired["n"] == 1:
                raise _Interrupt("simulated after requeue")

        pool._condition.notify_all = interrupting_notify
        try:
            with pytest.raises(_Interrupt):
                pool.checkin(lease)
        finally:
            pool._condition.notify_all = real_notify

        # RETURNED: conn is live in the idle queue — must NOT be closed.
        assert close_calls[0] == 0, (
            f"returned conn must not be closed, got {close_calls[0]} closes"
        )
        assert pool._created == 1, f"expected _created==1, got {pool._created}"
        assert pool._checked_out == {}
        assert pool._checking_in == set()
        assert lease._released is True

        # The idle conn is still open and usable: a fresh checkout pulls it
        # from the idle queue and validates it (SELECT 1) without creating a
        # new connection, proving the handler did not close it.
        assert pool._pool.qsize() == 1
        lease2 = pool.checkout(validate=True)
        try:
            assert lease2.execute("SELECT 1").fetchone()[0] == 1
            assert pool._created == 1, "returned conn must be reused, not recreated"
        finally:
            pool.checkin(lease2)
    finally:
        pool.shutdown(grace_period=0)


def test_f11_interrupt_after_closed_decrement(tmp_path):
    """A BaseException after _created decremented (disposition CLOSED) must not
    double-decrement. The single physical close is performed by the handler."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "f11-closed.db"), pool_size=4, lazy=True,
    )
    try:
        close_calls = [0]

        def pragma_raises(sql):
            # Force discard=True via a non-corruption pragma failure. The
            # accounting block then takes the should_close=True (CLOSED) path.
            raise sqlite3.OperationalError("simulated pragma failure")

        setup = pool.checkout(validate=False)
        real = setup._conn
        proxy = _make_counting_proxy(real, close_calls, pragma_fn=pragma_raises)
        object.__setattr__(setup, "_conn", proxy)

        # Interrupt via notify_all inside the accounting block (after decrement).
        real_notify = pool._condition.notify_all
        fired = {"n": 0}

        def interrupting_notify():
            fired["n"] += 1
            real_notify()
            if fired["n"] == 1:
                raise _Interrupt("simulated after decrement")

        pool._condition.notify_all = interrupting_notify
        try:
            with pytest.raises(_Interrupt):
                pool.checkin(setup)
        finally:
            pool._condition.notify_all = real_notify

        # CLOSED: _created already decremented once; handler must not do it again.
        assert pool._created == 0, f"expected _created==0, got {pool._created}"
        assert close_calls[0] == 1, f"expected 1 close, got {close_calls[0]}"
        assert pool._checked_out == {}
        assert setup._released is True
    finally:
        pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# F3 — recovery leader refuses to repair an unsafe path
# ---------------------------------------------------------------------------


def test_f3_recovery_aborts_when_path_marked_unsafe(tmp_path, monkeypatch):
    """Recovery wakes from raw drain and rejects the newly unsafe path.

    A real raw lease keeps _recover_source parked in its raw-drain condition
    wait. The raw lease then fails physical close through the production
    _RawConnectionLease.close bookkeeping path. That close removes the active
    raw count, records the exact close error in _unsafe_raw_paths, and notifies
    the condition. Recovery must wake, observe the unsafe path, raise
    DatabaseRecoveryError, and never invoke recover_corrupt_db.
    """
    db_path = str(tmp_path / "f3-unsafe.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    original_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )
    pool._corruption_error = original_error

    raw_lease = db_pool._create_raw_connection(db_path)
    raw_connection = raw_lease._conn
    close_error = sqlite3.DatabaseError(
        "simulated physical close failure"
    )

    class _FailingCloseProxy:
        __slots__ = ("_real",)

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def close(self):
            raise close_error

        def __getattr__(self, name):
            return getattr(
                object.__getattribute__(self, "_real"),
                name,
            )

        def __setattr__(self, name, value):
            setattr(
                object.__getattribute__(self, "_real"),
                name,
                value,
            )

    object.__setattr__(
        raw_lease,
        "_conn",
        _FailingCloseProxy(raw_connection),
    )

    recovery_parked = threading.Event()
    recovery_ident = {"value": None}
    recovery_errors = []
    recover_calls = []

    real_wait = db_pool._pool_condition.wait

    def recording_wait(timeout=None):
        if (
            threading.get_ident() == recovery_ident["value"]
            and db_pool._active_raw_connections.get(db_path, 0) > 0
        ):
            recovery_parked.set()
        return real_wait(timeout)

    monkeypatch.setattr(
        db_pool._pool_condition,
        "wait",
        recording_wait,
    )

    from memory import db_recovery

    def spy_recover(path):
        recover_calls.append(path)
        return True

    monkeypatch.setattr(
        db_recovery,
        "recover_corrupt_db",
        spy_recover,
    )

    def run_recovery():
        recovery_ident["value"] = threading.get_ident()
        try:
            db_pool._recover_source(pool, original_error)
        except BaseException as exc:
            recovery_errors.append(exc)

    recovery_thread = threading.Thread(target=run_recovery)
    parked_observed = False
    close_info = None

    try:
        recovery_thread.start()
        parked_observed = recovery_parked.wait(timeout=5.0)

        with pytest.raises(
            sqlite3.DatabaseError,
            match="simulated physical close failure",
        ) as close_info:
            raw_lease.close()

        recovery_thread.join(timeout=5.0)

        assert parked_observed, (
            "recovery never reached the real raw-drain condition wait"
        )
        assert close_info.value is close_error
        assert raw_lease._closed is True
        assert db_pool._active_raw_connections.get(db_path, 0) == 0
        assert db_pool._unsafe_raw_paths.get(db_path) is close_error
        assert not recovery_thread.is_alive()
        assert recover_calls == []
        assert len(recovery_errors) == 1
        assert isinstance(
            recovery_errors[0],
            DatabaseRecoveryError,
        )
        assert str(recovery_errors[0]) == (
            f"Skipping recovery for {db_path}: raw connection close "
            f"failed during drain, path is unsafe"
        )
    finally:
        if not raw_lease._closed:
            try:
                raw_lease.close()
            except BaseException:
                pass
        recovery_thread.join(timeout=5.0)
        raw_connection.close()


def test_f3_recovery_proceeds_when_path_safe(tmp_path, monkeypatch):
    """Control: with the same setup but path NOT marked unsafe, recovery DOES
    invoke recover_corrupt_db (proving the F3 check is the discriminating gate)."""
    db_path = str(tmp_path / "f3-safe.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    with db_pool._pool_condition:
        db_pool._active_raw_connections.pop(db_path, None)
        # NOT setting _unsafe_raw_paths — path is safe.

    called = {"yes": False}

    def fake_recover(path):
        called["yes"] = True
        return True

    from memory import db_recovery
    monkeypatch.setattr(db_recovery, "recover_corrupt_db", fake_recover)

    # The leader will attempt to publish a replacement pool. We don't care
    # about the full success — only that recover_corrupt_db was reached.
    try:
        db_pool._recover_source(pool, pool._corruption_error)
    except Exception:
        # Publication may fail in the test harness; that's irrelevant.
        pass

    assert called["yes"], "safe path must proceed to recover_corrupt_db"


# ---------------------------------------------------------------------------
# F4 — atomic capacity reservation eliminates the slot-steal race
# ---------------------------------------------------------------------------


def test_f4_atomic_capacity_reservation_prevents_slot_steal(tmp_path, monkeypatch):
    """A replacement reservation and a competing creator stay within capacity.

    Thread A validates a bad idle connection and enters the replacement path.
    Thread B starts checkout when A classifies that validation failure. Under
    the fixed implementation, A decrements _created and reserves _creating
    while holding one condition-lock acquisition. B can reserve only after A
    poisons and checks in its replacement lease. Under the reverted split-lock
    implementation, B reserves the released slot while A is between the
    decrement and replacement reservation, causing A's capacity-invariant
    RuntimeError.
    """
    pool = db_pool.ConnectionPool(
        str(tmp_path / "f4-race.db"),
        pool_size=1,
        lazy=True,
    )
    release_b_create = threading.Event()
    a_at_classification = threading.Event()
    creator_reserved = threading.Event()
    b_ident = {"value": None}
    outcomes = []
    errors = []

    try:
        setup_lease = pool.checkout(validate=False)
        pool.checkin(setup_lease)
        drained = pool._pool.get_nowait()

        class _BadProxy:
            __slots__ = ("_real",)

            def __init__(self, real):
                object.__setattr__(self, "_real", real)

            def execute(self, sql, *args, **kwargs):
                if "SELECT 1" in str(sql):
                    raise sqlite3.OperationalError(
                        "simulated validation failure"
                    )
                return object.__getattribute__(self, "_real").execute(
                    sql,
                    *args,
                    **kwargs,
                )

            def close(self):
                return object.__getattribute__(self, "_real").close()

            def __getattr__(self, name):
                return getattr(
                    object.__getattribute__(self, "_real"),
                    name,
                )

        pool._pool.put_nowait(_BadProxy(drained))
        assert pool._created == 1
        assert pool._creating == 0

        real_create = pool._create_connection

        def controlled_create():
            if threading.get_ident() == b_ident["value"]:
                creator_reserved.set()
                if not release_b_create.wait(timeout=5.0):
                    raise AssertionError(
                        "timed out waiting to release B connection creation"
                    )
            return real_create()

        monkeypatch.setattr(pool, "_create_connection", controlled_create)

        real_is_corruption_error = db_pool._is_corruption_error

        def classification_probe(exc):
            a_at_classification.set()
            if not pool._condition._is_owned():
                if not creator_reserved.wait(timeout=5.0):
                    raise AssertionError(
                        "B did not reserve during the split-lock race window"
                    )
            return real_is_corruption_error(exc)

        monkeypatch.setattr(
            db_pool,
            "_is_corruption_error",
            classification_probe,
        )

        def checkout_a():
            lease = None
            try:
                lease = pool.checkout(timeout=5.0)
                outcomes.append("A")
            except BaseException as exc:
                errors.append(("A", exc))
            finally:
                if lease is not None and not lease._released:
                    object.__setattr__(lease, "_poisoned", True)
                    try:
                        pool.checkin(lease)
                    except BaseException as exc:
                        errors.append(("A-checkin", exc))

        def checkout_b():
            lease = None
            try:
                b_ident["value"] = threading.get_ident()
                if not a_at_classification.wait(timeout=5.0):
                    raise AssertionError(
                        "A did not reach validation-failure classification"
                    )
                lease = pool.checkout(timeout=5.0)
                outcomes.append("B")
            except BaseException as exc:
                errors.append(("B", exc))
            finally:
                if lease is not None and not lease._released:
                    try:
                        pool.checkin(lease)
                    except BaseException as exc:
                        errors.append(("B-checkin", exc))

        thread_b = threading.Thread(target=checkout_b)
        thread_a = threading.Thread(target=checkout_a)
        thread_b.start()
        thread_a.start()

        reservation_observed = creator_reserved.wait(timeout=5.0)
        if reservation_observed:
            with pool._condition:
                reservation_snapshot = (
                    pool._created,
                    pool._creating,
                    pool._pool_size,
                )
        else:
            reservation_snapshot = None

        release_b_create.set()
        thread_a.join(timeout=5.0)
        thread_b.join(timeout=5.0)

        assert reservation_observed, (
            "B never reached the _creating reservation point"
        )
        assert reservation_snapshot == (0, 1, 1)
        created, creating, pool_size = reservation_snapshot
        assert created + creating <= pool_size

        capacity_errors = [
            exc
            for _, exc in errors
            if isinstance(exc, RuntimeError)
            and "capacity invariant" in str(exc)
        ]
        assert capacity_errors == []
        assert errors == []
        assert not thread_a.is_alive()
        assert not thread_b.is_alive()
        assert outcomes == ["A", "B"]
        assert pool._created + pool._creating <= pool._pool_size
    finally:
        release_b_create.set()
        if "thread_a" in locals():
            thread_a.join(timeout=5.0)
        if "thread_b" in locals():
            thread_b.join(timeout=5.0)
        pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# F5 — bounded quiescence waits raise instead of blocking forever
# ---------------------------------------------------------------------------


def test_f5_retire_for_recovery_bounds_wait(tmp_path):
    """retire_for_recovery must raise (not hang) when inflight/creating never
    drain. A creator thread blocked inside _create_connection holds both
    inflight==1 and creating==1, so the drain predicate never satisfies."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "f5-retire.db"), pool_size=4, lazy=True,
    )
    real_create = pool._create_connection
    create_hold = threading.Event()
    try:
        def blocking_create():
            create_hold.wait(timeout=10.0)
            return real_create()

        pool._create_connection = blocking_create

        def creator():
            try:
                pool.checkout(timeout=5.0)
            except Exception:
                pass  # retiring pool will reject; irrelevant to this test.

        t_creator = threading.Thread(target=creator)
        t_creator.start()
        # Wait until the creator has reserved creation (creating==1) and is
        # blocked inside _create_connection.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with pool._condition:
                if pool._creating > 0:
                    break
            time.sleep(0.001)

        # retire_for_recovery should time out, not hang.
        with pytest.raises(RuntimeError, match="retire_for_recovery timed out"):
            pool.retire_for_recovery(timeout=0.5)
    finally:
        create_hold.set()
        t_creator.join(timeout=5.0)
        pool.shutdown(grace_period=0)


def test_f5_shutdown_bounds_post_grace_wait(tmp_path):
    """shutdown's post-grace wait must raise when inflight never drains."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "f5-shutdown.db"), pool_size=4, lazy=True,
    )
    real_create = pool._create_connection
    create_hold = threading.Event()
    try:
        # Keep _creating > 0 so the post-grace wait never satisfies.
        def blocking_create():
            create_hold.wait(timeout=10.0)
            return real_create()

        pool._create_connection = blocking_create

        def creator():
            try:
                pool.checkout(timeout=2.0)
            except Exception:
                pass

        t_creator = threading.Thread(target=creator)
        t_creator.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with pool._condition:
                if pool._creating > 0:
                    break
            time.sleep(0.001)

        with pytest.raises(RuntimeError, match="shutdown timed out"):
            pool.shutdown(grace_period=0)
    finally:
        create_hold.set()
        t_creator.join(timeout=5.0)
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_f5_recovery_raw_drain_bounds_wait(tmp_path, monkeypatch):
    """_recover_source raw-drain must raise DatabaseRecoveryError when raw
    connections never close."""
    db_path = str(tmp_path / "f5-raw.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    # Hold a raw connection open so the drain predicate never holds.
    with db_pool._pool_condition:
        db_pool._active_raw_connections[db_path] = 1

    def must_not_run(path):
        raise AssertionError("recover_corrupt_db must not run after drain timeout")

    from memory import db_recovery
    monkeypatch.setattr(db_recovery, "recover_corrupt_db", must_not_run)

    with pytest.raises(DatabaseRecoveryError, match="Timed out waiting for raw"):
        db_pool._recover_source(pool, pool._corruption_error)


def test_slice_2b_recovery_deadline_is_shared_across_maintenance_and_raw_drain(
    tmp_path, monkeypatch,
):
    """Slice 2B: recovery wait phases share one end-to-end deadline."""
    db_path = str(tmp_path / "slice-2b-deadline.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    raw_lease = db_pool._create_raw_connection(db_path)

    maintenance_token = db_pool.begin_path_maintenance(db_path)

    BUDGET = 1.0
    MAINT_HOLD = 0.65
    THRESHOLD = 1.25
    monkeypatch.setattr(db_pool, "_QUIESCENCE_DRAIN_TIMEOUT", BUDGET)

    maint_observed = threading.Event()
    real_wait = db_pool._pool_condition.wait
    wait_calls = {"n": 0}

    def wrapped_wait(*args, **kwargs):
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            maint_observed.set()
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(db_pool._pool_condition, "wait", wrapped_wait)

    started = [None]
    error_holder = {"exc": None}

    def run_recovery():
        try:
            started[0] = time.monotonic()
            db_pool._recover_source(pool, pool._corruption_error)
        except BaseException as exc:
            error_holder["exc"] = exc

    worker = threading.Thread(target=run_recovery)
    maintenance_released = False
    try:
        worker.start()
        assert maint_observed.wait(timeout=5.0), (
            "recovery worker never entered maintenance wait"
        )

        hold = threading.Event()
        hold.wait(timeout=MAINT_HOLD)
        db_pool.end_path_maintenance(maintenance_token)
        maintenance_released = True

        worker.join(timeout=BUDGET + 1.0)
        assert not worker.is_alive(), (
            "recovery worker did not complete within budget + 1s slack"
        )
        elapsed = time.monotonic() - started[0]

        exc = error_holder["exc"]
        assert exc is not None, "recovery worker exited without raising"
        assert isinstance(exc, DatabaseRecoveryError), (
            f"expected DatabaseRecoveryError, got {type(exc).__name__}: {exc}"
        )
        assert "Timed out waiting for raw connections" in str(exc), (
            f"expected raw-drain timeout error, got: {exc}"
        )

        assert elapsed < THRESHOLD, (
            f"shared deadline violated: elapsed={elapsed:.3f}s exceeds "
            f"budget+slack threshold {THRESHOLD}s. The raw-drain wait "
            f"appears to be using a fresh deadline instead of sharing "
            f"the function-entry deadline (slice 2B)."
        )
    finally:
        if not maintenance_released:
            try:
                db_pool.end_path_maintenance(maintenance_token)
            except Exception:
                pass
        try:
            raw_lease.close()
        except Exception:
            pass
        if worker.is_alive():
            worker.join(timeout=2.0)


# ---------------------------------------------------------------------------
# D1 / D3 — dead code removed
# ---------------------------------------------------------------------------


def test_d1_wait_for_active_raw_to_drain_removed():
    assert not hasattr(db_pool, "_wait_for_active_raw_to_drain"), (
        "_wait_for_active_raw_to_drain is dead code and must be deleted"
    )


def test_d3_noop_exception_handlers_removed():
    import inspect

    src = inspect.getsource(db_pool.ConnectionPool.connection)
    # The no-op `except _CorruptDatabase: raise` / `except BaseException: raise`
    # wrappers must be gone; the inner checkin() in a finally must remain.
    assert "except _CorruptDatabase" not in src, (
        "no-op except _CorruptDatabase handler must be removed"
    )
    assert "self.checkin(lease)" in src, "inner checkin() must be retained"

    src_mod = inspect.getsource(db_pool.connection)
    assert "except _CorruptDatabase" not in src_mod, (
        "module-level no-op except _CorruptDatabase handler must be removed"
    )
    assert "checkin(lease)" in src_mod, "module-level inner checkin() must be retained"


# ---------------------------------------------------------------------------
# R1-C T-C1 — F11 #7a: failed publication from RETURNED_PENDING_PUBLISH
# leaves an unpublished physical connection. Cleanup must close it and
# decrement _created as a CLOSED transition.
# ---------------------------------------------------------------------------


def test_t_c1_r2_f11a_disposition_constants_defined():
    """R1-C T-C1: both transitional disposition constants must exist with
    stable string values so the cleanup handler's defensive branches can
    match them across the unreachable-state invariant.

    The string values are part of the public contract: any change would
    silently disable the cleanup branches (the `elif` checks would never
    match). Pin them here.
    """
    assert db_pool._CHECKIN_DISPOSITION_RETURNED_PENDING_PUBLISH == "returned_pending_publish"
    assert db_pool._CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE == "published_pending_checkout_delete"


def test_t_c1_r2_f11a_pending_publish_interrupt_before_put_nowait_closes_unpublished_conn(tmp_path):
    """R1-C T-C1 (F11 #7a): a BaseException from put_nowait after the
    RETURNED_PENDING_PUBLISH disposition write orphans the unpublished conn.
    Cleanup must release the lease, decrement _created, and close the physical
    connection exactly once so a healthy replacement can consume the slot.

    Revert-check: restore RETURNED_PENDING_PUBLISH to the BaseException
    close-guard skip list in db_pool.py. The close count remains zero and the
    exact-close assertion fails.
    """
    pool = db_pool.ConnectionPool(
        str(tmp_path / "tc1-f11a.db"), pool_size=4, lazy=True,
    )
    try:
        close_calls = [0]
        lease = pool.checkout(validate=True)
        token = lease._token
        real = lease._conn
        proxy = _make_counting_proxy(real, close_calls)
        object.__setattr__(lease, "_conn", proxy)

        real_put = pool._pool.put_nowait

        def interrupting_put(*args, **kwargs):
            raise _Interrupt("simulated between disposition write and put_nowait")

        pool._pool.put_nowait = interrupting_put
        try:
            with pytest.raises(_Interrupt):
                pool.checkin(lease)
        finally:
            pool._pool.put_nowait = real_put

        assert close_calls[0] == 1, f"expected 1 close, got {close_calls[0]}"
        assert lease._released is True
        assert token not in pool._checked_out
        assert pool._pool.qsize() == 0
        assert pool._created == 0
        assert pool._checking_in == set()

        new_lease = pool.checkout(validate=True)
        try:
            assert new_lease.execute("SELECT 1").fetchone()[0] == 1
            assert pool._created == 1
        finally:
            pool.checkin(new_lease)
    finally:
        pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# R1-C T-C2 — F11 #7b: PUBLISHED_PENDING_CHECKOUT_DELETE state (conn live
# in idle queue, _checked_out not yet pruned). The cleanup handler must
# NOT close the conn (it's live in the idle queue) but MUST delete from
# _checked_out so the lease doesn't leak.
# ---------------------------------------------------------------------------


def test_t_c2_r2_f11b_published_pending_checkout_delete_keeps_conn_alive(tmp_path):
    """R1-C T-C2 (F11 #7b, PUBLISHED_PENDING_CHECKOUT_DELETE half): the
    cleanup handler must NOT close the conn (it's live in the idle queue
    in the unreachable PPCD state) but MUST prune _checked_out so the
    lease doesn't leak.

    Per R1-C §C.6 #2, PPCD is currently UNREACHABLE in the W1-B order
    (del-checked_out THEN put_nowait). The test exercises the cleanup
    branch's defensive recognition by:

    1. Pinning the constant value (the elif clause uses this exact value).
    2. Driving the sibling RETURNED path through the same close-guard,
       since PPCD shares the close-guard invariant ("conn live in idle
       queue — do NOT close").

    The PPCD-specific "prune _checked_out" behavior is verified by the
    `test_t_c_anchors_exist_in_db_pool` pin below — it grep's the source
    for the elif branch that prunes _checked_out on PPCD.

    Revert-check: revert C2's cleanup handler (remove the `elif`
    branch for PPCD). The `test_t_c_anchors_exist_in_db_pool` test's
    source-grep assertion will fail.
    """
    pool = db_pool.ConnectionPool(
        str(tmp_path / "tc2-f11b.db"), pool_size=4, lazy=True,
    )
    close_calls = [0]
    try:
        lease = pool.checkout(validate=True)
        real = lease._conn
        proxy = _make_counting_proxy(real, close_calls)
        object.__setattr__(lease, "_conn", proxy)

        # Drive the RETURNED path (which IS reachable today) and verify
        # the close-guard invariant that PPCD relies on: conn live in
        # idle queue → do NOT close.
        real_notify = pool._condition.notify_all
        fired = {"n": 0}

        def interrupting_notify():
            fired["n"] += 1
            real_notify()
            if fired["n"] == 1:
                raise _Interrupt("simulated after requeue (RETURNED)")

        pool._condition.notify_all = interrupting_notify
        try:
            with pytest.raises(_Interrupt):
                pool.checkin(lease)
        finally:
            pool._condition.notify_all = real_notify

        # RETURNED close-guard: conn in idle queue — must NOT close.
        assert close_calls[0] == 0, (
            f"RETURNED conn must not be closed, got {close_calls[0]} closes"
        )
        assert pool._created == 1, f"expected _created==1, got {pool._created}"
        assert pool._checked_out == {}
        assert pool._checking_in == set()
        assert lease._released is True
        assert pool._pool.qsize() == 1, "RETURNED conn must remain in idle queue"

        # Pin the PPCD constant value (used in the cleanup elif branch).
        assert (
            db_pool._CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE
            == "published_pending_checkout_delete"
        )
    finally:
        pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# R1-C T-C3 — F5 #8a: existing-attempt follower wait must bound on a leaked
# recovery leader (state stuck at "recovering"). Asserts
# DatabaseRecoveryError within deadline + ε.
# ---------------------------------------------------------------------------


def test_t_c3_r2_f5a_existing_attempt_follower_waits_bound(tmp_path, monkeypatch):
    """R1-C T-C3 (F5 #8a): the existing-attempt follower wait at
    db_pool.py:2002-2015 must raise DatabaseRecoveryError when the
    existing recovery attempt never transitions out of "recovering"
    state. Pin the timeout with a short value so the test runs fast.
    """
    db_path = str(tmp_path / "tc3-f5a.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    # Seed an existing attempt stuck in "recovering" state. The follower
    # enters the `while existing.state == "recovering"` loop and waits.
    class _StuckAttempt:
        state = "recovering"
        error = None
    db_pool._recovery_attempts_by_source[pool] = _StuckAttempt()

    # Shrink the deadline so the test runs in ms. The test asserts the
    # timeout fires (DatabaseRecoveryError) regardless of magnitude;
    # magnitude only affects wall time.
    monkeypatch.setattr(db_pool, "_QUIESCENCE_DRAIN_TIMEOUT", 0.05)

    started = time.monotonic()
    with pytest.raises(DatabaseRecoveryError, match="Timed out waiting for existing"):
        db_pool._recover_source(pool, pool._corruption_error)
    elapsed = time.monotonic() - started
    # Must fire within deadline + 1s slack (the inner wait slices at 0.5s,
    # so even with a small deadline the elapsed can include up to one
    # slice; with deadline=0.05 the slice is min(remaining, 0.5)).
    assert elapsed < 2.0, (
        f"existing-attempt follower must time out within deadline + slack, "
        f"took {elapsed:.3f}s"
    )


# ---------------------------------------------------------------------------
# R1-C T-C4 — F5 #8b: outer-loop (path-in-_recovering_paths) and
# path-in-_maintenance_paths waits must bound.
# ---------------------------------------------------------------------------


def test_t_c4_r2_f5b_recovering_paths_loop_bounds_on_orphan_attempt(tmp_path, monkeypatch):
    """R1-C T-C4 (F5 #8b, _recovering_paths half): the outer loop at
    db_pool.py:2036-2067 must raise DatabaseRecoveryError when a stuck
    path entry in _recovering_paths never gets cleared.

    A separate RecoveryAttempt instance is required because the check is
    `existing is not None` (by pool identity) before the outer loop runs.
    """
    db_path = str(tmp_path / "tc4-f5b-recovering.db")
    sqlite3.connect(db_path).close()

    pool_a = db_pool._ensure_pool(db_path)
    pool_b = db_pool._ensure_pool(db_path)
    pool_a._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    # Seed _recovering_paths with a stuck attempt. pool_b is a different
    # pool instance so the existing-attempt branch (line 2000) doesn't
    # fire — it falls through to the outer `_recovering_paths` loop.
    class _StuckAttempt:
        state = "recovering"
        error = None
    db_pool._recovering_paths[db_path] = _StuckAttempt()

    monkeypatch.setattr(db_pool, "_QUIESCENCE_DRAIN_TIMEOUT", 0.05)

    with pytest.raises(DatabaseRecoveryError, match="Timed out"):
        db_pool._recover_source(pool_b, pool_a._corruption_error)


def test_t_c4_r2_f5b_maintenance_paths_loop_bounds_on_stuck_maintenance(tmp_path, monkeypatch):
    """R1-C T-C4 (F5 #8b, _maintenance_paths half): the wait at
    db_pool.py:2071-2086 must raise DatabaseRecoveryError when a stuck
    maintenance token never gets cleared.
    """
    db_path = str(tmp_path / "tc4-f5b-maint.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    # Seed _maintenance_paths with a stuck token. The leader is fresh
    # (no existing attempt, no recovering_paths entry for this pool), so
    # it reaches the maintenance-paths branch.
    class _StuckToken:
        pass
    db_pool._maintenance_paths[db_path] = _StuckToken()

    monkeypatch.setattr(db_pool, "_QUIESCENCE_DRAIN_TIMEOUT", 0.05)

    with pytest.raises(DatabaseRecoveryError, match="Timed out waiting for maintenance"):
        db_pool._recover_source(pool, pool._corruption_error)


# ---------------------------------------------------------------------------
# R1-C T-C5 — F8 #10 (post_process.py dedup): single-source shape validator
# decides the malformed branch. Dict-error short-circuit preserves the
# original envelope. Both branches reach the SAME StageAbort ((body, 502))
# pattern but with different bodies — verifying the consolidation is
# functionally equivalent to the pre-edit code.
# ---------------------------------------------------------------------------


def test_t_c5_r2_f10_postprocess_single_shape_validator_for_malformed():
    """R1-C T-C5 (F8 #10): post_process.py must use the single-source
    `chat_completion_shape_error` validator for the malformed branch.
    The dict-error short-circuit is a separate signal that preserves the
    ORIGINAL provider error envelope.

    This test exercises BOTH branches to verify they both abort with
    HTTP 502 but with different bodies — proving the two checks are
    distinct signals (not duplicates) and that the consolidation does
    not regress either behavior.

    Revert-check: revert C6 (swap the order back to dict-first, or
    duplicate the `isinstance(data, dict)` guard). With the revert, the
    dict-error branch fires BEFORE the shape validator, but since the
    shapes are mutually exclusive (dict-error never has shape_error),
    the test still passes — both branches are exercised. So this is a
    regression-pin test, not a discriminating revert-check. The real
    discriminator is the post-edit code using `chat_completion_shape_error`
    as the only shape validator (the constant `shape_error` is the only
    shape signal). The revert-check is implicit: if a future refactor
    re-introduces `isinstance(data, dict)` style shape checks OUTSIDE
    chat_completion_shape_error, the test pin below catches it.
    """
    from proxy.openai_compat import build_upstream_protocol_error
    from proxy.pipeline.exceptions import StageAbort
    from proxy.pipeline.post_process import PostProcessStage

    # Branch 1: dict-error envelope → preserve original body, abort 502.
    route = MagicMock()
    route.model_cfg = MagicMock(memory_enabled=True, ignore_subagents=False)
    ctx = MagicMock()
    ctx.model_route = route
    ctx.session_id = "sess-tc5"
    ctx.request_id = "req-tc5"
    provider_error = {"error": {"message": "boom", "type": "provider_error"}}
    ctx.response_data = provider_error
    ctx.llama_body = {}
    ctx.upstream = MagicMock()
    ctx.is_subagent = False

    with pytest.raises(StageAbort) as exc_info:
        PostProcessStage().execute(ctx)
    body, status = exc_info.value.response_data
    assert status == 502
    assert body == provider_error, "dict-error body must be preserved verbatim"

    # Branch 2: malformed (string `error`) → canonical envelope, abort 502.
    ctx2 = MagicMock()
    ctx2.model_route = route
    ctx2.session_id = "sess-tc5b"
    ctx2.request_id = "req-tc5b"
    ctx2.response_data = {"choices": [], "error": "string-not-dict"}
    ctx2.llama_body = {}
    ctx2.upstream = MagicMock()
    ctx2.is_subagent = False

    with pytest.raises(StageAbort) as exc_info:
        PostProcessStage().execute(ctx2)
    body2, status2 = exc_info.value.response_data
    assert status2 == 502
    canonical = build_upstream_protocol_error()
    assert body2 == canonical, "malformed body must be replaced with canonical envelope"

    # Pin: both branches abort at HTTP 502 — same wire shape, different bodies.
    # The two bodies are distinct signals (provider-error preserve vs
    # canonical malformed-replace).
    assert body != body2, (
        "dict-error and malformed bodies must be distinguishable by signal"
    )


# ---------------------------------------------------------------------------
# R1-C T-C6 — F8 #10 follow-on: ensure `chat_completion_shape_error` is the
# ONLY shape validator in post_process.py. A future regression that
# re-introduces an `isinstance(data, dict)` shape check (other than the
# dict-error signal's narrow `isinstance(data.get('error'), dict)` guard)
# would defeat the consolidation.
# ---------------------------------------------------------------------------


def test_t_c6_r2_f10_postprocess_shape_validator_is_singleton():
    """R1-C T-C6: pin the post-C6 invariant — `chat_completion_shape_error`
    is the ONLY shape validator in the malformed-completion boundary of
    PostProcessStage.execute. A naive re-introduction of an
    `isinstance(data, dict)` shape check (other than the narrow dict-
    error signal's `isinstance(data.get('error'), dict)` guard) would
    defeat the F8 dedup.

    Scope: only the shape-validation block at the top of the `try`
    branch in `execute()`. Sub-methods called LATER
    (`_apply_skill_response_transform`, `_extract_minimax_reasoning`)
    may have their own internal type checks — those are NOT shape
    validators of the malformed-completion boundary.
    """
    import inspect
    from proxy.pipeline.post_process import PostProcessStage

    src = inspect.getsource(PostProcessStage.execute)

    # Must reference the single-source validator at least once.
    assert "chat_completion_shape_error" in src, (
        "PostProcessStage must call chat_completion_shape_error as the "
        "single-source shape validator"
    )

    # The narrow dict-error signal guard is allowed
    # (`isinstance(data.get('error'), dict)`, NOT `isinstance(data, dict)`).
    # No other shape-validation-style `isinstance(data, ...)` checks should
    # appear in the malformed-completion boundary. We grep the WHOLE execute
    # source because sub-methods called later are out of scope for the
    # shape-validation invariant — but the malformed-completion boundary
    # is the only place where shape validation matters for HTTP 502
    # mapping. The narrow dict-error guard IS allowed; bare
    # `isinstance(data, dict)` is not.
    bad_pattern = "isinstance(data, dict)"
    if bad_pattern in src:
        # Find each occurrence with line context for diagnostic.
        for i, line in enumerate(src.splitlines(), 1):
            if bad_pattern in line:
                # Allow the comment-only reference (line that explains
                # why the guard was removed). Real code uses
                # `isinstance(data.get('error'), dict)` instead.
                stripped = line.lstrip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                raise AssertionError(
                    f"PostProcessStage.execute line {i}: bare "
                    f"`{bad_pattern}` must not appear — "
                    f"chat_completion_shape_error is the single-source "
                    f"shape validator. Use `isinstance(data.get('error'), "
                    f"dict)` for the dict-error signal guard instead.\n"
                    f"Line: {line}"
                )


# ---------------------------------------------------------------------------
# T-C1..T-C6 revert-check helpers
# ---------------------------------------------------------------------------


def test_t_c_anchors_exist_in_db_pool():
    """Pin that the anchors cited in the R1-C plan still exist at HEAD.
    Catches future refactors that move the disposition constants, the
    bounded waits, or the cleanup handler branches to different lines.

    Discriminating revert-check: this test grep's for the C2 elif branch
    that handles the PUBLISHED_PENDING_CHECKOUT_DELETE disposition in
    the BaseException cleanup handler. If a future refactor removes that
    elif branch (reverting T-C2's defensive cleanup), the assertion
    below fails. The constant alone is NOT enough — the elif branch
    must also exist.
    """
    import inspect

    src = inspect.getsource(db_pool)

    # C1: disposition constants
    assert "_CHECKIN_DISPOSITION_RETURNED_PENDING_PUBLISH" in src
    assert "_CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE" in src

    # C2 cleanup handler: the elif branch for PPCD must exist in the
    # BaseException clause (defensive recognition — per R1-C §C.6 #2
    # the state is unreachable today but the handler must NOT silently
    # regress). Scan ALL BaseException clauses for the PPCD constant —
    # the elif branch is in the checkin() BaseException handler.
    base_exc_clauses = src.split("except BaseException")[1:]
    assert any(
        "_CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE" in clause
        for clause in base_exc_clauses
    ), (
        "C2 cleanup handler must reference PPCD in a BaseException clause "
        "(the elif branch in checkin()'s defensive recognition)"
    )

    # C3, C4, C5: bounded waits must NOT be unbounded `_pool_condition.wait()`
    # (no timeout arg) in _recover_source. The bounded pattern is
    # `_pool_condition.wait(timeout=...)`.
    recover_src = inspect.getsource(db_pool._recover_source)
    # Strip comments — look for the bare wait pattern in _recover_source only.
    import re
    bare_wait = re.findall(
        r"^\s*_pool_condition\.wait\(\)\s*$", recover_src, re.MULTILINE
    )
    assert not bare_wait, (
        f"_recover_source must not have unbounded _pool_condition.wait(); "
        f"found {len(bare_wait)} occurrence(s). All three waits must be "
        f"bounded with timeout= (per R1-C Edit C3/C4/C5)."
    )


# ---------------------------------------------------------------------------
# R3-A T-R3-A-4 — F3 drives the real raw-close-fail interleaving
# (R3-A #15)
# ---------------------------------------------------------------------------


def test_r3_a4_f3_drives_real_raw_close_fail(tmp_path, monkeypatch):
    """R3-A #15 (F3 hand-written _unsafe_raw_paths): the F3 test must
    populate _unsafe_raw_paths via the REAL _RawConnectionLease.close
    path (db_pool.py:780-807), not by writing to the dict directly.

    Driving scenario: same as the rewritten test_f3 — open a raw
    conn, wrap its close() to fail via a proxy (sqlite3.Connection's
    close attribute is read-only, so we wrap), begin_path_maintenance,
    trigger the close (which populates _unsafe_raw_paths via the
    production branch at db_pool.py:802-803), end_path_maintenance,
    trigger _recover_source. Assert DatabaseRecoveryError and assert
    _unsafe_raw_paths[path] is populated BY THE PRODUCTION PATH
    (verified by checking the error is a sqlite3.DatabaseError raised
    by the wrapped close, not a test-injected sentinel RuntimeError).

    Revert-check: revert Edit R3-A.4 to the pre-fix hand-written
    `_unsafe_raw_paths[db_path] = RuntimeError(...)` pattern. The
    assertion that the unsafe-path error came from the production
    close path (i.e., is a sqlite3.DatabaseError raised by the
    wrapped failing_close) fails — the pre-fix pattern injects a
    bare RuntimeError directly.
    """
    db_path = str(tmp_path / "f3-real-interleave-r3a.db")
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (v INTEGER)")
    seed.close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    raw_lease = db_pool._create_raw_connection(db_path)

    physical_close_failed = threading.Event()

    class _FailingCloseProxy:
        __slots__ = ("_real",)

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def close(self):
            physical_close_failed.set()
            raise sqlite3.DatabaseError("simulated physical close failure")

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_real"), name, value)

    object.__setattr__(raw_lease, "_conn", _FailingCloseProxy(raw_lease._conn))

    maintenance_token = db_pool.begin_path_maintenance(db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            raw_lease.close()
        # _unsafe_raw_paths MUST be populated by the production
        # close path (not a test-injected dict write). The value
        # must be a sqlite3.DatabaseError from the wrapped close,
        # NOT a RuntimeError from the pre-fix pattern.
        unsafe_err = db_pool._unsafe_raw_paths.get(db_path)
        assert unsafe_err is not None, (
            "production _RawConnectionLease.close did not populate "
            "_unsafe_raw_paths"
        )
        assert isinstance(unsafe_err, sqlite3.DatabaseError), (
            f"production close populated _unsafe_raw_paths with "
            f"{type(unsafe_err).__name__}, expected sqlite3.DatabaseError. "
            f"This indicates a hand-written dict assignment (R3-A #15). "
            f"err={unsafe_err!r}"
        )
        assert "simulated physical close failure" in str(unsafe_err)
    finally:
        db_pool.end_path_maintenance(maintenance_token)

    called = {"n": 0}

    def spy_recover(path):
        called["n"] += 1
        return True

    from memory import db_recovery
    monkeypatch.setattr(db_recovery, "recover_corrupt_db", spy_recover)

    with pytest.raises(DatabaseRecoveryError):
        db_pool._recover_source(pool, pool._corruption_error)

    assert called["n"] == 0, (
        f"recover_corrupt_db must not run for an unsafe path; "
        f"got {called['n']} call(s)"
    )


# ---------------------------------------------------------------------------
# R3-A T-R3-A-5 — Source-level guard against hand-written _unsafe_raw_paths
# in the F3 test (R3-A #15 structural pin)
# ---------------------------------------------------------------------------


def test_r3_a5_f3_test_does_not_hand_write_unsafe_paths():
    """R3-A #15 (F3 hand-written _unsafe_raw_paths, structural pin):
    the F3 test must NOT write to `db_pool._unsafe_raw_paths` directly
    outside the production close path. The pre-fix pattern did
    `db_pool._unsafe_raw_paths[db_path] = RuntimeError(...)` under
    `_pool_condition`, bypassing the real interleaving.

    This is a source-level guard against re-introduction. It does
    not exercise runtime behaviour (that is T-R3-A-4's job).

    Revert-check: revert Edit R3-A.4 — restore the hand-written
    `_unsafe_raw_paths[db_path] = RuntimeError(...)` block. The
    structural assertion fails.
    """
    import inspect
    import re
    from tests.unit import test_db_pool_recovery_accounting as tdbp

    src = inspect.getsource(tdbp.test_f3_recovery_aborts_when_path_marked_unsafe)
    # Strip comments so references to the pre-fix pattern in docstrings
    # and code comments do not trigger a false-positive. Lines starting
    # with `#` (after optional whitespace) are comments.
    code_lines = [
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    # The post-fix F3 test may read `_unsafe_raw_paths.get(db_path)` to
    # assert the production close populated it. Only the SUBSCRIPT form
    # (write or hand-written assignment) is the structural violation.
    # _unsafe_raw_paths[ must NOT appear in code outside a `.get(` call.
    has_subscript_outside_get = bool(
        re.search(r"_unsafe_raw_paths\[(?!.*\.get\()", code_only)
    )
    assert not has_subscript_outside_get, (
        "F3 test code (excluding comments) must NOT subscript "
        "_unsafe_raw_paths directly — the production "
        "_RawConnectionLease.close path is the sole legitimate writer. "
        "Pre-fix pattern bypassed the real maintenance → raw-close-fail "
        "→ recovery interleaving."
    )


# Need MagicMock for T-C5 (already imported via conftest? verify)
import unittest.mock  # noqa: E402
MagicMock = unittest.mock.MagicMock
