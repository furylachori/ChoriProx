"""Regression tests for lazy runtime-pool creation during SQLite startup.

Every checkout returns a ``_ConnectionLease`` carrying a generation token.
The pool tracks ownership by token, not by ``id(conn)``. Tests use the
lease directly except where they specifically need to inspect or replace
the underlying physical connection.
"""

import logging
import queue
import sqlite3
import threading
import time

import pytest

import memory.db_pool as db_pool
from memory.db_migrations import init_db
from memory.db_summaries import store_message_summary


class _ParkProbe:
    """Wrap a threading.Condition; set `parked` when the target thread parks."""
    def __init__(self, real, ident_holder, parked_event):
        self._real = real; self._holder = ident_holder; self._parked = parked_event
    def __enter__(self): return self._real.__enter__()
    def __exit__(self, *a): return self._real.__exit__(*a)
    def wait(self, timeout=None):
        if threading.get_ident() == self._holder["v"]: self._parked.set()
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
    def notify(self, n=1): return self._real.notify(n)
    def notify_all(self): return self._real.notify_all()
    def acquire(self, *a, **k): return self._real.acquire(*a, **k)
    def release(self): return self._real.release()

def test_park_probe_wait_for_initial_true_does_not_wait_lazy_startup():
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


def test_park_probe_wait_for_false_predicate_enters_wait_lazy_startup():
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


def test_park_probe_wait_for_false_then_true_waits_lazy_startup():
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


class _EventWaitProbe:
    def __init__(self, real, ident_holder, parked_event):
        self._real = real; self._holder = ident_holder; self._parked = parked_event
    def wait(self, timeout=None):
        if threading.get_ident() == self._holder["v"]: self._parked.set()
        return self._real.wait(timeout)
    def set(self): return self._real.set()
    def clear(self): return self._real.clear()
    def is_set(self): return self._real.is_set()

class _ContendLock:
    def __init__(self, real, contended_event):
        self._real = real; self._contended = contended_event
    def acquire(self, blocking=True, timeout=-1):
        if self._real.acquire(blocking=False): return True
        self._contended.set()
        return (self._real.acquire(blocking) if timeout == -1
                else self._real.acquire(blocking, timeout))
    def release(self): return self._real.release()
    def __enter__(self): self.acquire(); return self
    def __exit__(self, *a): self.release()


@pytest.fixture(autouse=True)
def _reset_global_pool():
    current = db_pool._pool
    if current is not None:
        current.shutdown(grace_period=0)
        db_pool._pool = None
    db_pool._atexit_registered = False

    yield

    current = db_pool._pool
    if current is not None:
        current.shutdown(grace_period=0)
        db_pool._pool = None
    db_pool._atexit_registered = False


def test_ensure_pool_does_not_open_connections_before_first_checkout(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lazy-startup.db"
    monkeypatch.setenv("PROXY_TEST_DB", str(db_path))

    db_pool._ensure_pool(str(db_path))

    pool = db_pool._pool
    assert pool is not None
    assert pool._db_path == str(db_path)
    assert pool._created == 0
    assert pool._pool.qsize() == 0
    assert pool._checked_out == {}

    lease = pool.checkout()
    try:
        assert lease.execute("SELECT 1").fetchone()[0] == 1
        assert pool._created == 1
        assert len(pool._checked_out) == 1
        assert pool._pool.qsize() == 0
    finally:
        pool.checkin(lease)

    assert pool._created == 1
    assert pool._checked_out == {}
    assert pool._pool.qsize() == 1


def test_file_backed_init_finishes_migrations_before_runtime_pool(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "file-backed-init.db"
    monkeypatch.setenv("PROXY_TEST_DB", str(db_path))

    with sqlite3.connect(str(db_path)) as seed:
        from memory.db_migrations import SCHEMA_SQL

        seed.executescript(SCHEMA_SQL)
        seed.execute(
            "INSERT INTO messages "
            "(session_id, message_index, role, content, content_hash, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "g3-session",
                0,
                "user",
                "seed message",
                "seed-hash",
                "2026-07-12T00:00:00Z",
                "2026-07-12T00:00:00Z",
            ),
        )
        seed.commit()

    assert db_pool._pool is None
    assert init_db(str(db_path)) is True

    # File-backed migration must not create the runtime pool.
    assert db_pool._pool is None

    with sqlite3.connect(str(db_path)) as conn:
        session = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            ("g3-session",),
        ).fetchone()
        indexes = conn.execute(
            "PRAGMA index_list(message_summaries)"
        ).fetchall()
        unique_columns = []
        for index in indexes:
            if index[2]:
                columns = conn.execute(
                    f"PRAGMA index_info({index[1]})"
                ).fetchall()
                unique_columns.append(
                    tuple(column[2] for column in columns)
                )

    assert session == ("g3-session",)
    assert ("session_id", "exchange_index") in unique_columns
    assert ("session_id", "message_index") not in unique_columns

    assert store_message_summary(
        session_id="g3-session",
        exchange_index=0,
        role="user",
        summary="first summary",
        tool_summary="",
        content_hash="hash-first",
        model_used="g3-test",
        token_estimate=3,
        end_message_index=1,
        message_index=0,
    ) is True

    pool = db_pool._pool
    assert pool is not None
    assert pool._db_path == str(db_path)
    assert pool._created == 1
    assert pool._pool.qsize() == 1
    assert pool._checked_out == {}

    assert store_message_summary(
        session_id="g3-session",
        exchange_index=0,
        role="user",
        summary="updated summary",
        tool_summary="",
        content_hash="hash-updated",
        model_used="g3-test",
        token_estimate=4,
        end_message_index=1,
        message_index=0,
    ) is True

    with db_pool.connection() as conn:
        rows = conn.execute(
            "SELECT summary, content_hash "
            "FROM message_summaries "
            "WHERE session_id = ? AND exchange_index = ?",
            ("g3-session", 0),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["summary"] == "updated summary"
    assert rows[0]["content_hash"] == "hash-updated"

    pool = db_pool._pool
    assert pool is not None
    assert pool._checked_out == {}
    assert pool._created == 1
    assert pool._pool.qsize() == 1


def test_memory_init_keeps_post_migration_schema_alive(monkeypatch):
    monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

    assert db_pool._pool is None
    assert init_db(":memory:") is True

    pool = db_pool._pool
    assert pool is not None
    assert pool._db_path == ":memory:"
    assert pool._created == 1
    assert pool._pool.qsize() == 1
    assert pool._checked_out == {}

    assert store_message_summary(
        session_id="memory-session",
        exchange_index=0,
        role="user",
        summary="memory summary",
        tool_summary="",
        content_hash="memory-hash",
        model_used="g3-test",
        token_estimate=2,
        end_message_index=1,
        message_index=0,
    ) is True

    with db_pool.connection() as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'message_summaries'"
        ).fetchone()
        indexes = conn.execute(
            "PRAGMA index_list(message_summaries)"
        ).fetchall()
        unique_columns = []
        for index in indexes:
            if index["unique"]:
                columns = conn.execute(
                    f"PRAGMA index_info({index['name']})"
                ).fetchall()
                unique_columns.append(
                    tuple(column["name"] for column in columns)
                )
        row = conn.execute(
            "SELECT summary FROM message_summaries "
            "WHERE session_id = ? AND exchange_index = ?",
            ("memory-session", 0),
        ).fetchone()

    assert table is not None
    assert ("session_id", "exchange_index") in unique_columns
    assert row["summary"] == "memory summary"
    assert pool._checked_out == {}


def test_repeated_file_backed_init_preserves_data_without_preopening_pool(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "repeated-init.db"
    monkeypatch.setenv("PROXY_TEST_DB", str(db_path))

    assert init_db(str(db_path)) is True
    assert db_pool._pool is None

    assert store_message_summary(
        session_id="repeat-session",
        exchange_index=0,
        role="user",
        summary="before repeat",
        tool_summary="",
        content_hash="repeat-before",
        model_used="g3-test",
        token_estimate=2,
        end_message_index=1,
        message_index=0,
    ) is True

    first_pool = db_pool._pool
    assert first_pool is not None

    # init_db is also used by schema self-healing paths. It must preserve
    # committed data. This test does not authorize replacing live owners.
    assert first_pool._checked_out == {}
    assert init_db(str(db_path)) is True

    with db_pool.connection() as conn:
        row = conn.execute(
            "SELECT summary, content_hash FROM message_summaries "
            "WHERE session_id = ? AND exchange_index = ?",
            ("repeat-session", 0),
        ).fetchone()

    assert row["summary"] == "before repeat"
    assert row["content_hash"] == "repeat-before"

    assert store_message_summary(
        session_id="repeat-session",
        exchange_index=0,
        role="user",
        summary="after repeat",
        tool_summary="",
        content_hash="repeat-after",
        model_used="g3-test",
        token_estimate=2,
        end_message_index=1,
        message_index=0,
    ) is True

    with db_pool.connection() as conn:
        rows = conn.execute(
            "SELECT summary, content_hash FROM message_summaries "
            "WHERE session_id = ? AND exchange_index = ?",
            ("repeat-session", 0),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["summary"] == "after repeat"
    assert rows[0]["content_hash"] == "repeat-after"


class TestPoolMetric:
    """The fixed utilization metric: configured capacity, threshold-crossing
    warnings, counter-integrity under failure modes, and shutdown invariants."""

    def test_first_lazy_checkout_does_not_warn_high_utilization(
        self, tmp_path, caplog
    ):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "utilization.db"),
            pool_size=48,
            lazy=True,
        )
        try:
            with caplog.at_level(logging.WARNING, logger="memory.db_pool"):
                lease = pool.checkout()
            try:
                assert pool._created == 1
                assert len(pool._checked_out) == 1
                assert "Pool utilization high" not in caplog.text
                with pool._lock:
                    (
                        created_count,
                        checked_out_count,
                        idle_count,
                        utilization,
                    ) = pool._pool_stats_locked()
                assert checked_out_count == 1
                assert idle_count == 0
                assert created_count == 1
                assert utilization < 0.5
            finally:
                pool.checkin(lease)
        finally:
            pool.shutdown(grace_period=0)

    def test_eager_pool_created_count_matches_capacity(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "eager.db"),
            pool_size=4,
            lazy=False,
        )
        try:
            assert pool._created == 4
            assert pool._pool.qsize() == 4
            assert len(pool._checked_out) == 0
        finally:
            pool.shutdown(grace_period=0)

    def test_eager_pool_cannot_grow_beyond_capacity(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "eager-cap.db"),
            pool_size=4,
            lazy=False,
        )
        try:
            leases = [pool.checkout() for _ in range(4)]
            assert pool._created == 4
            assert pool._pool.qsize() == 0
            assert len(pool._checked_out) == 4
            with pytest.raises(RuntimeError, match="Connection pool exhausted"):
                pool.checkout(timeout=0.1)
            assert pool._created == 4
            for lease in leases:
                pool.checkin(lease)
        finally:
            pool.shutdown(grace_period=0)

    def test_validation_replacement_preserves_created_count(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "validate.db"),
            pool_size=4,
            lazy=True,
        )
        try:
            lease = pool.checkout()
            pool.checkin(lease)
            assert pool._pool.qsize() == 1
            assert pool._created == 1

            # Close the idle conn externally — the next validate=True
            # checkout will detect the failure, replace it, and `_created`
            # must remain at 1 (counter-integrity: discard + recreate).
            idle_lease = pool._pool.get_nowait()
            idle_lease.close()
            try:
                pool._pool.put_nowait(idle_lease)
            except queue.Full:
                pass

            replacement = pool.checkout(validate=True)
            try:
                assert pool._created == 1
                assert len(pool._checked_out) == 1
                result = replacement.execute("SELECT 1").fetchone()[0]
                assert result == 1
            finally:
                pool.checkin(replacement)
            assert pool._created == 1
            assert pool._pool.qsize() == 1
        finally:
            pool.shutdown(grace_period=0)

    def test_deferred_connection_rolls_back_uncommitted_write_on_checkin(
        self, tmp_path
    ):
        """With ``isolation_level=""``, an uncommitted INSERT is rolled back
        at checkin time. Other connections see zero rows.

        Verifies the contract: ordinary DML starts a deferred transaction,
        checkin rolls back residual work.
        """
        pool = db_pool.ConnectionPool(
            str(tmp_path / "deferred.db"),
            pool_size=4,
            lazy=True,
        )
        try:
            # First write: in a fresh connection, insert a row.
            with pool.connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS deferred_t (value INTEGER)"
                )
                conn.execute("INSERT INTO deferred_t VALUES (1)")
                # No commit — relying on checkin rollback.

            # Second checkin: a separate lease reads from a separate
            # physical conn. It must see zero rows because the prior
            # transaction was rolled back at checkin.
            verify = pool.checkout(validate=True)
            try:
                rows = verify.execute(
                    "SELECT COUNT(*) FROM deferred_t"
                ).fetchone()
                assert rows[0] == 0, (
                    f"expected 0 rows after rollback-on-checkin, got {rows[0]}"
                )
            finally:
                pool.checkin(verify)
        finally:
            pool.shutdown(grace_period=0)

    def test_manual_commit_persists_write(self, tmp_path):
        """A commit() inside the connection block persists the write.
        Verifies deferred-transaction semantics: ordinary DML needs an
        explicit commit, transaction() contexts handle that for callers
        that need cross-call atomicity."""
        pool = db_pool.ConnectionPool(
            str(tmp_path / "manual-commit.db"),
            pool_size=4,
            lazy=True,
        )
        try:
            with pool.connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS manual_t (value INTEGER)"
                )
                conn.execute("INSERT INTO manual_t VALUES (42)")
                conn.commit()

            # A separate lease reads back the row.
            verify = pool.checkout(validate=True)
            try:
                rows = verify.execute(
                    "SELECT COUNT(*) FROM manual_t"
                ).fetchone()
                assert rows[0] == 1, (
                    f"expected 1 row after commit, got {rows[0]}"
                )
            finally:
                pool.checkin(verify)
        finally:
            pool.shutdown(grace_period=0)

    def test_failed_checkin_discards_and_decrements(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "failed-checkin.db"),
            pool_size=4,
            lazy=True,
        )
        try:
            real_conn_holder = []

            class _CloseCountingProxy:
                __slots__ = ("_real",)

                def __init__(self, real):
                    self._real = real
                    real_conn_holder.append(real)

                def execute(self, *args, **kwargs):
                    return self._real.execute(*args, **kwargs)

                def rollback(self):
                    return self._real.rollback()

                def close(self):
                    close_calls[0] += 1
                    return self._real.close()

            close_calls = [0]
            first_setup = pool.checkout(validate=False)
            pool.checkin(first_setup)
            drained = pool._pool.get_nowait()
            proxy = _CloseCountingProxy(drained)
            pool._pool.put_nowait(proxy)

            # Acquire the proxy via validate=True — proxy's execute falls
            # through to the real conn so SELECT 1 succeeds.
            lease = pool.checkout(validate=True)

            # Externally break the underlying real conn so checkin's
            # rollback() raises and the failure branch fires.
            real_conn_holder[0].close()
            pool.checkin(lease)

            # The failure branch must have called close on the proxy.
            assert close_calls[0] == 1, (
                f"expected close in failure branch, got {close_calls[0]} calls"
            )

            # Counter-integrity: pool state must be clean.
            assert pool._created == 0
            assert pool._checked_out == {}
            assert pool._pool.qsize() == 0

            # Next checkout creates a healthy replacement.
            replacement = pool.checkout(validate=True)
            try:
                assert pool._created == 1
                replacement.execute("SELECT 1").fetchone()
            finally:
                pool.checkin(replacement)
        finally:
            pool.shutdown(grace_period=0)

    def test_pragma_failure_discards_and_closes(self, tmp_path):
        """PRAGMA reset failure on checkin must close the connection."""
        pool = db_pool.ConnectionPool(
            str(tmp_path / "pragma-fail.db"),
            pool_size=4,
            lazy=True,
        )
        try:
            real_conn_holder = []

            class _PragmaFailingProxy:
                __slots__ = ("_real",)

                def __init__(self, real):
                    self._real = real
                    real_conn_holder.append(real)

                def execute(self, sql, *args, **kwargs):
                    if "busy_timeout" in str(sql):
                        raise sqlite3.OperationalError("simulated PRAGMA failure")
                    return self._real.execute(sql, *args, **kwargs)

                def rollback(self):
                    return self._real.rollback()

                def close(self):
                    close_calls[0] += 1
                    return self._real.close()

            close_calls = [0]
            setup = pool.checkout(validate=False)
            pool.checkin(setup)
            drained = pool._pool.get_nowait()
            proxy = _PragmaFailingProxy(drained)
            pool._pool.put_nowait(proxy)

            lease = pool.checkout(validate=True)
            pool.checkin(lease)

            assert close_calls[0] == 1
            assert pool._created == 0
            assert pool._checked_out == {}
            assert pool._checking_in == set()
            assert pool._pool.qsize() == 0
        finally:
            pool.shutdown(grace_period=0)

    def test_eager_pool_closes_partial_connections_when_creation_fails(
        self, tmp_path, monkeypatch
    ):
        """Eager construction that fails partway through must close any
        connections that were already created and leave the pool empty."""
        path = str(tmp_path / "partial-eager.db")
        real_close_calls = [0]
        created_calls = [0]

        def fake_create(self):
            idx = created_calls[0]
            created_calls[0] += 1
            if idx < 2:
                class _PartialProxy:
                    def __init__(self, real):
                        self._real = real

                    def close(self):
                        real_close_calls[0] += 1
                        return self._real.close()

                    def execute(self, *args, **kwargs):
                        return self._real.execute(*args, **kwargs)

                    def __getattr__(self, name):
                        return getattr(self._real, name)

                return _PartialProxy(
                    sqlite3.connect(path, check_same_thread=False, isolation_level="")
                )
            raise sqlite3.OperationalError("simulated create failure")

        monkeypatch.setattr(
            db_pool.ConnectionPool,
            "_create_connection",
            fake_create,
        )

        with pytest.raises(sqlite3.OperationalError):
            db_pool.ConnectionPool(path, pool_size=4, lazy=False)

        # Both conns created before the failure must have been closed.
        assert real_close_calls[0] == 2, (
            f"expected 2 closes (one per partial-create), got {real_close_calls[0]}"
        )
        assert created_calls[0] == 3, (
            f"expected 3 create calls (2 ok + 1 fail), got {created_calls[0]}"
        )

    def test_warning_emitted_once_per_saturation_episode(self, tmp_path, caplog):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "saturation.db"),
            pool_size=10,
            lazy=True,
        )
        leases = []
        try:
            # 9/10 = 90% crosses the 0.80 fire threshold.
            with caplog.at_level(logging.WARNING, logger="memory.db_pool"):
                for _ in range(9):
                    leases.append(pool.checkout())

            warns = [
                r
                for r in caplog.records
                if r.name == "memory.db_pool"
                and "Pool utilization high" in r.getMessage()
            ]
            assert len(warns) == 1

            # 10/10 — already saturated → no new warning (hysteresis held).
            with caplog.at_level(logging.WARNING, logger="memory.db_pool"):
                leases.append(pool.checkout())
            warns = [
                r
                for r in caplog.records
                if r.name == "memory.db_pool"
                and "Pool utilization high" in r.getMessage()
            ]
            assert len(warns) == 1

            # Return 4 → 6/10 = 60% ≤ 0.70 reset → flag clears.
            for lease in leases[:4]:
                pool.checkin(lease)
            leases = leases[4:]
            assert pool._utilization_warning_active is False

            # Push back above 80% (3 more → 9/10 = 90%) → second warning.
            with caplog.at_level(logging.WARNING, logger="memory.db_pool"):
                for _ in range(3):
                    leases.append(pool.checkout())
            warns = [
                r
                for r in caplog.records
                if r.name == "memory.db_pool"
                and "Pool utilization high" in r.getMessage()
            ]
            assert len(warns) == 2

            latest = warns[-1]
            msg = latest.getMessage()
            assert "checked_out=" in msg
            assert "created=" in msg
            assert "idle=" in msg
            assert "capacity=10" in msg
        finally:
            for lease in leases:
                pool.checkin(lease)
            assert pool._utilization_warning_active is False
            pool.shutdown(grace_period=0)

    def test_shutdown_resets_counters_and_idle_state(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "shutdown-counters.db"),
            pool_size=4,
            lazy=True,
        )
        leases = [pool.checkout() for _ in range(3)]
        pool.checkin(leases[0])
        pool.checkin(leases[1])
        assert pool._created == 3
        assert len(pool._checked_out) == 1
        assert pool._pool.qsize() == 2

        pool.shutdown(grace_period=0)
        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._pool.qsize() == 0
        assert pool._utilization_warning_active is False

    def test_shutdown_idempotent(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "shutdown-idempotent.db"),
            pool_size=4,
            lazy=True,
        )
        lease = pool.checkout()
        pool.checkin(lease)
        pool.shutdown(grace_period=0)
        # Second call must not raise and must remain clean.
        pool.shutdown(grace_period=0)
        assert pool._created == 0
        assert pool._checked_out == {}

    def test_shutdown_closes_checked_out_after_grace(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "shutdown-grace.db"),
            pool_size=4,
            lazy=True,
        )
        lease = pool.checkout()
        assert pool._created == 1
        # Caller deliberately does NOT checkin.
        pool.shutdown(grace_period=0.1)
        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._pool.qsize() == 0
        # The physical conn was force-closed by shutdown. The lease has
        # been released and now refuses to dispatch.
        with pytest.raises(RuntimeError, match="Connection lease is no longer active"):
            lease.execute("SELECT 1")
        # The underlying physical conn was closed; operating on it directly
        # raises ProgrammingError.
        with pytest.raises(sqlite3.ProgrammingError):
            lease._conn.execute("SELECT 1")


class TestPoolConcurrency:
    """Concurrent checkout/shutdown safety. Uses threading.Event/Barrier for
    deterministic synchronization (no timing-only sequential tests).

    sqlite3.Connection is built in C; its `execute` and `close` attributes
    are read-only. Concurrency tests therefore wrap the real conn in a
    Python-level proxy so we can intercept specific methods on Event
    synchronization points without modifying production code.
    """

    class _ConnProxy:
        """Proxy that wraps a real sqlite3.Connection.

        Used to gate `close()` or `execute("SELECT 1")` on Events. Other
        methods (rollback, PRAGMA, etc.) pass through to the real conn.
        Note: no ``__slots__`` so tests can assign bound methods (e.g.,
        ``proxy.close = gated_close``) when they need to gate on a sync
        point that the pool's own flow controls.
        """

        def __init__(self, real, on_close=None, on_execute_select1=None):
            self._real = real
            self._on_close = on_close
            self._on_execute_select1 = on_execute_select1

        def execute(self, sql, *args, **kwargs):
            if self._on_execute_select1 is not None and "SELECT 1" in str(sql):
                return self._on_execute_select1(self._real.execute, sql, *args, **kwargs)
            return self._real.execute(sql, *args, **kwargs)

        def close(self):
            if self._on_close is not None:
                return self._on_close()
            return self._real.close()

        def rollback(self):
            return self._real.rollback()

    def test_checkout_after_shutdown_is_rejected(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "closed.db"),
            pool_size=4,
            lazy=True,
        )

        pool.shutdown(grace_period=0)

        with pytest.raises(
            RuntimeError,
            match="Connection pool is retiring",
        ):
            pool.checkout(timeout=0.01)

        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._pool.qsize() == 0

    def test_concurrent_second_shutdown_waits_for_completion(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "concurrent-shutdown.db"),
            pool_size=4,
            lazy=True,
        )

        # Acquire a real conn and swap it for a proxy whose close is gated.
        setup_lease = pool.checkout(validate=False)
        pool.checkin(setup_lease)
        drained = pool._pool.get_nowait()

        close_started = threading.Event()
        release_close = threading.Event()

        def gated_close():
            close_started.set()
            if not release_close.wait(timeout=5.0):
                raise RuntimeError("test: close gate not released")
            return drained.close()

        proxy = self._ConnProxy(drained, on_close=gated_close)
        pool._pool.put_nowait(proxy)

        # Acquire the proxy through normal checkout (validation passes).
        lease = pool.checkout(validate=True)
        assert pool._created == 1
        assert len(pool._checked_out) == 1

        shutdown_a_done = threading.Event()
        shutdown_a_entered = threading.Event()
        shutdown_b_done = threading.Event()
        shutdown_b_entered = threading.Event()
        shutdown_a_errors: list = []
        shutdown_b_errors: list = []

        def first_shutdown():
            shutdown_a_entered.set()
            try:
                pool.shutdown(grace_period=0)
            except Exception as e:
                shutdown_a_errors.append(e)
            finally:
                shutdown_a_done.set()

        def second_shutdown():
            b_ident["v"] = threading.get_ident()
            shutdown_b_entered.set()
            try:
                pool.shutdown(grace_period=0)
            except Exception as e:
                shutdown_b_errors.append(e)
            finally:
                shutdown_b_done.set()

        t1 = threading.Thread(target=first_shutdown)
        t1.start()
        assert shutdown_a_entered.wait(timeout=5.0)

        # Wait for first shutdown to reach the close gate.
        assert close_started.wait(timeout=5.0), "first shutdown didn't reach close"

        b_ident = {"v": None}
        b_parked = threading.Event()
        pool._shutdown_complete = _EventWaitProbe(pool._shutdown_complete, b_ident, b_parked)

        t2 = threading.Thread(target=second_shutdown)
        t2.start()
        assert shutdown_b_entered.wait(timeout=5.0)

        # Second shutdown MUST park on _shutdown_complete while first is in progress.
        assert b_parked.wait(timeout=5.0), "second shutdown never parked on _shutdown_complete"

        # Release the close gate.
        release_close.set()

        # Both must finish without exception.
        assert shutdown_a_done.wait(timeout=5.0), "first shutdown didn't finish"
        assert shutdown_b_done.wait(timeout=5.0), "second shutdown didn't finish"
        assert not shutdown_a_errors, f"first shutdown raised: {shutdown_a_errors}"
        assert not shutdown_b_errors, f"second shutdown raised: {shutdown_b_errors}"

        t1.join(timeout=3)
        t2.join(timeout=3)
        assert not t1.is_alive()
        assert not t2.is_alive()

        # Final invariants.
        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._pool.qsize() == 0

    def test_shutdown_racing_checkout_cannot_leak_a_connection(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "race.db"),
            pool_size=4,
            lazy=True,
        )

        # Swap in a proxy whose SELECT 1 is gated so the test can interleave
        # shutdown between acquire and validate. Lease registration is
        # deferred until validation passes, so during the gate the conn
        # is "in flight" — neither in the queue nor in _checked_out.
        # Shutdown therefore has no conn to close, but the test must still
        # verify there is no leak: pool invariants and the in-flight
        # conn must both be cleaned up.
        setup_lease = pool.checkout(validate=False)
        pool.checkin(setup_lease)
        drained = pool._pool.get_nowait()

        validate_started = threading.Event()
        release_validate = threading.Event()
        shutdown_done = threading.Event()

        def gated_execute_select1(real_exec, sql, *args, **kwargs):
            validate_started.set()
            if not release_validate.wait(timeout=5.0):
                raise sqlite3.OperationalError(
                    "test: validate gate not released"
                )
            return real_exec(sql, *args, **kwargs)

        proxy = self._ConnProxy(
            drained,
            on_execute_select1=gated_execute_select1,
        )

        pool._pool.put_nowait(proxy)

        checkout_outcome: list = []

        def do_checkout():
            try:
                lease = pool.checkout(timeout=5.0, validate=True)
                checkout_outcome.append(("ok", lease))
                try:
                    pool.checkin(lease)
                except Exception:
                    pass
            except Exception as e:
                checkout_outcome.append(("err", str(e)))

        t1 = threading.Thread(target=do_checkout)
        t1.start()

        # Wait until checkout reaches the validate gate.
        assert validate_started.wait(timeout=5.0), "checkout didn't reach validate"

        # Start shutdown while checkout is blocked inside validate.
        t2 = threading.Thread(
            target=lambda: (pool.shutdown(grace_period=0), shutdown_done.set())
        )
        t2.start()

        # Release validate so the in-flight checkout can unwind.
        release_validate.set()

        # Shutdown waits for admitted checkouts to drain; after release
        # it can complete.
        assert shutdown_done.wait(timeout=5.0), "shutdown didn't complete"

        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive()
        assert not t2.is_alive()

        # Exactly one error from checkout, no successful lease.
        errors = [o for o in checkout_outcome if o[0] == "err"]
        oks = [o for o in checkout_outcome if o[0] == "ok"]
        assert not oks, (
            f"checkout returned a lease while shutdown claimed it: {checkout_outcome}"
        )
        assert len(errors) == 1, (
            f"expected exactly one checkout error, got {checkout_outcome}"
        )
        err_msg = errors[0][1]
        assert ("retiring" in err_msg) or ("replaced" in err_msg), (
            f"unexpected error message: {err_msg}"
        )

        # Final invariants.
        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._checking_in == set()
        assert pool._pool.qsize() == 0

    def test_validation_replacement_cannot_exceed_capacity(self, tmp_path):
        pool = db_pool.ConnectionPool(
            str(tmp_path / "cap.db"),
            pool_size=1,
            lazy=True,
        )
        try:
            # Inject an invalid idle conn. Validation will fail when picked up.
            setup_lease = pool.checkout(validate=False)
            pool.checkin(setup_lease)
            drained = pool._pool.get_nowait()

            def failing_select1(real_exec, sql, *args, **kwargs):
                if "SELECT 1" in str(sql):
                    raise sqlite3.OperationalError("simulated validation failure")
                return real_exec(sql, *args, **kwargs)

            bad = self._ConnProxy(drained, on_execute_select1=failing_select1)
            pool._pool.put_nowait(bad)
            assert pool._created == 1

            # Instrument every successful _create_connection so we can prove
            # that at most one physical connection was created during the
            # validate-replacement race.
            real_create = pool._create_connection
            live_connections = [0]
            max_live_connections = [0]
            create_count = [0]

            class _LiveConnWrapper:
                def __init__(self, real):
                    self._real = real
                    live_connections[0] += 1
                    create_count[0] += 1
                    if live_connections[0] > max_live_connections[0]:
                        max_live_connections[0] = live_connections[0]

                def close(self):
                    live_connections[0] -= 1
                    return self._real.close()

                def execute(self, *args, **kwargs):
                    return self._real.execute(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._real, name)

            def instrumented_create():
                result = real_create()
                return _LiveConnWrapper(result)

            pool._create_connection = instrumented_create

            # Sample max-checked-out synchronously after each checkout
            # returns. Lease registration is the last step in checkout,
            # so by the time checkout returns, _checked_out has the
            # expected count.
            max_checked_out = [0]

            def record_max():
                cur = len(pool._checked_out)
                if cur > max_checked_out[0]:
                    max_checked_out[0] = cur

            # First checkout: validation fails on the bad proxy, then
            # atomic replacement creates a new conn and registers a
            # lease. The replacement transfers the existing slot — no
            # decrement/increment.
            first = pool.checkout(timeout=5.0)
            try:
                record_max()
                assert pool._created == 1
                assert len(pool._checked_out) == 1
            finally:
                pool.checkin(first)
            assert pool._pool.qsize() == 1
            assert pool._created == 1

            # Second checkout: must reuse the single physical conn (now
            # idle in the queue). It must not trigger _create_connection.
            second = pool.checkout(timeout=5.0)
            try:
                record_max()
                assert pool._created == 1
            finally:
                pool.checkin(second)

            # Live-connection budget: at most one physical conn ever
            # existed at any moment.
            assert max_live_connections[0] == 1, (
                f"max_live_connections={max_live_connections[0]}; expected 1"
            )
            # _create_connection was called only once (for the
            # replacement). The second checkout reused the idle conn.
            assert create_count[0] == 1, (
                f"create_count={create_count[0]}; expected 1"
            )
            # At no point was more than one lease checked out.
            assert max_checked_out[0] == 1, (
                f"max_checked_out={max_checked_out[0]}; expected 1"
            )
            assert pool._created <= pool._pool_size
        finally:
            pool.shutdown(grace_period=0)

    def test_concurrent_duplicate_checkin_only_one_proceeds(self, tmp_path):
        """Two threads call checkin(lease) concurrently. Exactly one thread's
        rollback() is invoked; the other is rejected after the first
        releases the operation lock."""
        pool = db_pool.ConnectionPool(
            str(tmp_path / "concurrent-dup.db"),
            pool_size=4,
            lazy=True,
        )
        try:
            rollback_calls = [0]
            pragma_calls = [0]
            rollback_started = threading.Event()
            release_rollback = threading.Event()

            class _CountingProxy(self._ConnProxy):
                def rollback(self):
                    rollback_calls[0] += 1
                    rollback_started.set()
                    if not release_rollback.wait(timeout=5.0):
                        raise RuntimeError("test: rollback gate not released")
                    return self._real.rollback()

                def execute(self, sql, *args, **kwargs):
                    if "PRAGMA" in str(sql):
                        pragma_calls[0] += 1
                    return self._real.execute(sql, *args, **kwargs)

            setup_lease = pool.checkout(validate=False)
            pool.checkin(setup_lease)
            drained = pool._pool.get_nowait()
            proxy = _CountingProxy(drained)
            pool._pool.put_nowait(proxy)

            lease = pool.checkout(validate=True)
            assert pool._created == 1
            assert pool._checking_in == set()

            second_finished = threading.Event()

            def first_checkin():
                pool.checkin(lease)

            def second_checkin():
                pool.checkin(lease)
                second_finished.set()

            t1 = threading.Thread(target=first_checkin)
            t1.start()

            # Wait for first checkin to enter rollback (where the gate is).
            assert rollback_started.wait(timeout=5.0), "first checkin didn't reach rollback"

            # Verify the lease's token is in _checking_in (first thread
            # still owns the checkin critical section).
            with pool._condition:
                assert lease._token in pool._checking_in, (
                    "first checkin should hold the lease token in _checking_in"
                )

            _cin_contended = threading.Event()
            object.__setattr__(lease, "_operation_lock", _ContendLock(lease._operation_lock, _cin_contended))

            t2 = threading.Thread(target=second_checkin)
            t2.start()

            # Second checkin must contend the operation lock (blocked by first checkin).
            assert _cin_contended.wait(timeout=5.0), "second checkin never contended the op lock"

            # Release first checkin's rollback gate.
            release_rollback.set()
            t1.join(timeout=5.0)
            assert not t1.is_alive()

            # Second checkin completes after first releases op lock.
            assert second_finished.wait(timeout=5.0), (
                "second checkin didn't complete"
            )
            assert not t2.is_alive()

            # Exactly one rollback total. PRAGMA invocations: the first
            # checkin ran two PRAGMA statements (busy_timeout +
            # wal_autocheckpoint). The second checkin was rejected
            # before reaching PRAGMA.
            assert rollback_calls[0] == 1
            assert pragma_calls[0] == 2, (
                f"pragma called {pragma_calls[0]} times, expected 2"
            )
            assert lease._released is True
            assert pool._created == 1
            assert pool._checked_out == {}
            assert pool._checking_in == set()
            assert pool._pool.qsize() == 1
        finally:
            pool.shutdown(grace_period=0)

    def test_stale_lease_cannot_rollback_rechecked_out_connection(
        self, tmp_path
    ):
        """A lease returned to the pool cannot roll back a newer
        borrower's transaction on the same physical conn."""
        pool = db_pool.ConnectionPool(
            str(tmp_path / "stale-lease.db"),
            pool_size=1,
            lazy=True,
        )
        try:
            lease_a = pool.checkout()
            physical = lease_a._conn
            pool.checkin(lease_a)
            assert lease_a._released is True

            lease_b = pool.checkout()
            assert lease_b._conn is physical
            assert lease_b._token != lease_a._token

            # Borrower's active transaction.
            lease_b.execute("CREATE TABLE IF NOT EXISTS stale_t (value INTEGER)")
            lease_b.execute("BEGIN")
            lease_b.execute("INSERT INTO stale_t VALUES (1)")

            # Stale duplicate checkin. The first one already returned
            # the conn to the queue; this duplicate must be rejected.
            pool.checkin(lease_a)

            # Borrower's transaction must still be intact.
            assert lease_b.in_transaction is True, (
                "stale checkin rolled back the active transaction"
            )
            lease_b.execute("COMMIT")
            pool.checkin(lease_b)

            # The inserted row is persisted.
            verify = pool.checkout(validate=True)
            try:
                rows = verify.execute(
                    "SELECT COUNT(*) FROM stale_t"
                ).fetchone()
                assert rows[0] == 1, (
                    f"expected 1 row, got {rows[0]}"
                )
            finally:
                pool.checkin(verify)
        finally:
            pool.shutdown(grace_period=0)

    def test_checkout_raises_when_shutdown_claims_after_validation(
        self, tmp_path
    ):
        """If shutdown claims and closes the connection between
        validate-success and the final ownership check, checkout must
        raise RuntimeError, not return a closed lease."""
        pool = db_pool.ConnectionPool(
            str(tmp_path / "checkout-claim.db"),
            pool_size=4,
            lazy=True,
        )

        # Setup BEFORE overriding the helper. The setup checkout must
        # complete normally (no gate).
        setup_lease = pool.checkout(validate=False)
        pool.checkin(setup_lease)
        drained = pool._pool.get_nowait()
        pool._pool.put_nowait(drained)

        # Override the final ownership helper to gate on two events so
        # the test can interleave shutdown.
        validation_finished = threading.Event()
        allow_checkout_finalization = threading.Event()

        original_record = pool._record_checkout_and_maybe_warn

        def gated_record(lease):
            validation_finished.set()
            if not allow_checkout_finalization.wait(timeout=5.0):
                raise RuntimeError("test: finalization gate not released")
            return original_record(lease)

        pool._record_checkout_and_maybe_warn = gated_record

        checkout_outcome: list = []

        def do_checkout():
            try:
                lease = pool.checkout(timeout=5.0, validate=True)
                checkout_outcome.append(("ok", lease))
                try:
                    pool.checkin(lease)
                except Exception:
                    pass
            except Exception as e:
                checkout_outcome.append(("err", str(e)))

        t1 = threading.Thread(target=do_checkout)
        t1.start()

        # Wait for validation to complete.
        assert validation_finished.wait(timeout=5.0), (
            "checkout didn't reach final ownership check"
        )

        # Release the finalization gate so the checkout can finish
        # (it will see the shutdown state and raise). Then run shutdown
        # to completion.
        allow_checkout_finalization.set()
        pool.shutdown(grace_period=0)

        t1.join(timeout=5)
        assert not t1.is_alive()

        # Checkout must have raised the shutdown error.
        outcomes = [o[0] for o in checkout_outcome]
        assert "err" in outcomes, (
            f"checkout did not raise; outcomes={checkout_outcome}"
        )
        err_msg = next(o[1] for o in checkout_outcome if o[0] == "err")
        assert "retiring" in err_msg, (
            f"unexpected error message: {err_msg}"
        )

        # Final invariants.
        assert pool._created == 0
        assert pool._checked_out == {}
        assert pool._checking_in == set()
        assert pool._pool.qsize() == 0

    @pytest.mark.parametrize(
        "bad_size",
        [0, -1, True, 1.5, "4"],
    )
    def test_invalid_pool_size_rejected(self, tmp_path, bad_size):
        with pytest.raises(ValueError, match="pool_size must be a positive integer"):
            db_pool.ConnectionPool(
                str(tmp_path / f"size-{bad_size}.db"),
                pool_size=bad_size,
                lazy=True,
            )


def test_valid_pool_size_accepted(tmp_path):
    """Positive integer pool_size values must construct successfully."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "valid.db"),
        pool_size=4,
        lazy=True,
    )
    assert pool._pool_size == 4
    pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# Mandatory focused tests (corrective patch §21)
# ---------------------------------------------------------------------------


def _reset_db_pool_manager():
    """Test-only helper: tear down the global pool state safely."""
    with db_pool._pool_condition:
        pool = db_pool._pool
        db_pool._pool = None

    if pool is not None:
        pool.shutdown(grace_period=0)

    with db_pool._pool_condition:
        assert not db_pool._active_raw_connections
        assert db_pool._active_recovery_leaders == 0
        db_pool._recovery_attempts_by_source.clear()
        db_pool._recovering_paths.clear()
        db_pool._maintenance_paths.clear()
        db_pool._terminal_shutdown = False
        db_pool._pool_condition.notify_all()


@pytest.fixture(autouse=False)
def reset_db_pool():
    _reset_db_pool_manager()
    yield
    _reset_db_pool_manager()


def test_global_checkout_balances_transferred_admission(tmp_path):
    """`_checkout_global` transfers one admission to the selected pool and
    `pool.checkout` must release it exactly once on the success path."""
    db_path = str(tmp_path / "balance-success.db")
    monkey = db_pool._pool
    assert monkey is None
    lease = db_pool._checkout_global(db_path, validate=True, timeout=5.0)
    try:
        assert lease._pool._inflight == 0
        assert lease._pool._creating == 0
    finally:
        db_pool.return_connection(lease)


def test_global_checkout_creation_failure_balances_admission(tmp_path, monkeypatch):
    """If `_create_connection` raises, the transferred admission must still
    be released by `pool.checkout`."""
    db_path = str(tmp_path / "balance-createfail.db")
    monkey = db_pool._pool
    assert monkey is None

    def boom(self):
        raise sqlite3.OperationalError("simulated create failure")

    monkeypatch.setattr(db_pool.ConnectionPool, "_create_connection", boom)
    with pytest.raises(sqlite3.OperationalError):
        db_pool._checkout_global(db_path, validate=True, timeout=5.0)

    assert db_pool._pool is not None
    assert db_pool._pool._inflight == 0
    assert db_pool._pool._creating == 0


def test_global_checkout_validation_failure_balances_admission(
    tmp_path, monkeypatch
):
    """If physical connection creation fails (a creation-time error),
    the transferred admission must be released by checkout."""
    db_path = str(tmp_path / "balance-validatefail.db")
    monkey = db_pool._pool
    assert monkey is None

    def boom(self):
        raise sqlite3.OperationalError("simulated create failure")

    monkeypatch.setattr(db_pool.ConnectionPool, "_create_connection", boom)
    with pytest.raises(sqlite3.OperationalError):
        db_pool._checkout_global(db_path, validate=True, timeout=5.0)

    assert db_pool._pool is not None
    assert db_pool._pool._inflight == 0
    assert db_pool._pool._creating == 0


def test_checkout_retirement_balances_admission(tmp_path):
    """A retire during checkin must release the transferred admission once
    the lease is returned. The pool's created count is zero."""
    db_path = str(tmp_path / "balance-retire.db")
    monkey = db_pool._pool
    assert monkey is None
    lease = db_pool._checkout_global(db_path, validate=True, timeout=5.0)
    pool = lease._pool
    # Mark retiring before checkin. Checkin observes _retiring and
    # closes the conn. The key invariant: the inflight counter is
    # balanced regardless of the retiring state.
    with pool._condition:
        pool._retiring = True
        pool._condition.notify_all()
    pool.checkin(lease)
    assert pool._inflight == 0
    assert pool._creating == 0
    assert pool._created == 0
    assert pool._checked_out == {}


class _CountingPhysical:
    """Count rollback/PRAGMA/conn-close calls on the physical conn."""

    def __init__(self, real):
        self._real = real
        self.rollback_calls = 0
        self.pragmas = 0
        self.physical_executes = 0

    def rollback(self):
        self.rollback_calls += 1
        return self._real.rollback()

    def execute(self, sql, *args, **kwargs):
        self.physical_executes += 1
        if "PRAGMA" in str(sql):
            self.pragmas += 1
        return self._real.execute(sql, *args, **kwargs)


def test_duplicate_checkin_waits_then_returns_stale(tmp_path):
    """A second duplicate checkin waits for the first operation lock, then
    sees lease._released and returns without a second rollback."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "dup-checkin.db"),
        pool_size=4,
        lazy=True,
    )
    try:
        rollback_calls = [0]
        rollback_started = threading.Event()
        release_rollback = threading.Event()

        class _Proxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args, **kwargs):
                return self._real.execute(sql, *args, **kwargs)

            def rollback(self):
                rollback_calls[0] += 1
                rollback_started.set()
                if not release_rollback.wait(timeout=5.0):
                    raise RuntimeError("rollback gate not released")
                return self._real.rollback()

        setup = pool.checkout(validate=False)
        pool.checkin(setup)
        drained = pool._pool.get_nowait()
        pool._pool.put_nowait(_Proxy(drained))

        lease = pool.checkout(validate=True)
        second_finished = threading.Event()

        def first():
            pool.checkin(lease)

        def second():
            pool.checkin(lease)
            second_finished.set()

        t1 = threading.Thread(target=first)
        t1.start()
        assert rollback_started.wait(timeout=5.0)

        _cin_contended = threading.Event()
        object.__setattr__(lease, "_operation_lock", _ContendLock(lease._operation_lock, _cin_contended))

        t2 = threading.Thread(target=second)
        t2.start()

        # Second must contend the operation lock (blocked by first checkin).
        assert _cin_contended.wait(timeout=5.0), "second checkin never contended the op lock"

        release_rollback.set()
        t1.join(timeout=5.0)
        assert not t1.is_alive()

        assert second_finished.wait(timeout=5.0)
        assert not t2.is_alive()

        # Exactly one rollback; second saw released and exited without
        # a duplicate.
        assert rollback_calls[0] == 1
        assert lease._released is True
        assert pool._checked_out == {}
        assert pool._checking_in == set()
    finally:
        pool.shutdown(grace_period=0)


def test_execute_waits_for_checkin_and_never_reaches_physical_conn(
    tmp_path,
):
    """While checkin owns the operation lock, a concurrent execute waits
    and never touches the physical connection's execute method."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "execute-gate.db"),
        pool_size=4,
        lazy=True,
    )
    try:
        execute_physical = [0]
        rollback_started = threading.Event()
        release_rollback = threading.Event()
        execute_attempted = threading.Event()
        execute_released = threading.Event()

        class _Proxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args, **kwargs):
                execute_physical[0] += 1
                return self._real.execute(sql, *args, **kwargs)

            def rollback(self):
                rollback_started.set()
                if not release_rollback.wait(timeout=5.0):
                    raise RuntimeError("rollback gate not released")
                return self._real.rollback()

        setup = pool.checkout(validate=False)
        pool.checkin(setup)
        drained = pool._pool.get_nowait()
        pool._pool.put_nowait(_Proxy(drained))

        lease = pool.checkout(validate=True)
        # Reset to ignore the checkout's SELECT 1.
        execute_physical[0] = 0

        execute_done = threading.Event()
        execute_physical_during_checkin = [None]

        def do_execute():
            execute_attempted.set()
            execute_released.wait(timeout=5.0)
            try:
                lease.execute("SELECT 1")
            except Exception:
                pass
            execute_done.set()

        def do_checkin():
            # This runs in the main thread. It holds the operation
            # lock through rollback + reset + PRAGMA + condition.
            pool.checkin(lease)

        t_exec = threading.Thread(target=do_execute)
        t_exec.start()
        assert execute_attempted.wait(timeout=5.0)

        # Snapshot the counter, then start checkin.
        execute_physical_before = execute_physical[0]

        t_checkin = threading.Thread(target=do_checkin)
        t_checkin.start()

        # Wait for checkin to enter rollback gate.
        assert rollback_started.wait(timeout=5.0)

        # The execute should still be blocked. The physical counter
        # should NOT have increased beyond the pre-checkin value.
        assert execute_physical[0] == execute_physical_before, (
            f"execute reached physical conn during checkin: "
            f"{execute_physical[0]} (was {execute_physical_before})"
        )

        # Release the rollback gate so checkin can complete.
        release_rollback.set()

        t_checkin.join(timeout=5.0)
        assert not t_checkin.is_alive()

        # Now the execute can run.
        execute_released.set()
        t_exec.join(timeout=5.0)
        assert not t_exec.is_alive()
    finally:
        pool.shutdown(grace_period=0)


def test_shutdown_waits_for_creation_owner_after_grace(tmp_path, monkeypatch):
    """After grace elapses, shutdown must still wait for the physical
    creator to unwind (the creator's create() may take time and shutdown
    must not zero out _creating)."""
    db_path = str(tmp_path / "shutdown-wait-create.db")
    pool = db_pool._ensure_pool(db_path)

    creation_in_progress = threading.Event()
    release_creation = threading.Event()
    original_create = pool._create_connection

    def slow_create(self):
        creation_in_progress.set()
        if not release_creation.wait(timeout=5.0):
            raise RuntimeError("create gate not released")
        return original_create()

    monkeypatch.setattr(db_pool.ConnectionPool, "_create_connection", slow_create)

    _sd_ident = {"v": None}
    _sd_parked = threading.Event()
    pool._condition = _ParkProbe(pool._condition, _sd_ident, _sd_parked)

    # Drain the pool so the next checkout must create.
    while not pool._pool.empty():
        pool._pool.get_nowait()
    with pool._condition:
        pool._created = 0

    # Start a checkout that will block inside create_connection.
    checkout_done = threading.Event()
    checkout_outcome = []

    def do_checkout():
        try:
            lease = pool.checkout(timeout=5.0, validate=True)
            checkout_outcome.append(("ok", lease))
        except Exception as e:
            checkout_outcome.append(("err", str(e)))
        finally:
            checkout_done.set()

    t_checkout = threading.Thread(target=do_checkout)
    t_checkout.start()

    # Wait until the creation has started.
    assert creation_in_progress.wait(timeout=5.0)
    assert pool._creating == 1

    # Now start shutdown. It should wait for the in-flight creation.
    shutdown_done = threading.Event()

    def do_shutdown():
        _sd_ident["v"] = threading.get_ident()
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass
        shutdown_done.set()

    t_shutdown = threading.Thread(target=do_shutdown)
    t_shutdown.start()

    # Shutdown must park on the pool condition while the creation is in progress.
    assert _sd_parked.wait(timeout=5.0), "shutdown never parked on the pool condition"
    # _creating must still be 1 (NOT reset to 0 by shutdown).
    assert pool._creating == 1

    # Release the creation; both threads should now finish.
    release_creation.set()

    assert checkout_done.wait(timeout=5.0)
    assert shutdown_done.wait(timeout=5.0)
    assert not t_checkout.is_alive()
    assert not t_shutdown.is_alive()


def test_shutdown_never_forces_creating_counter_to_zero(tmp_path, monkeypatch):
    """Shutdown must not directly assign ``self._creating = 0``. A creator
    still in-flight keeps the counter at 1 until it unwinds."""
    db_path = str(tmp_path / "shutdown-no-zero.db")
    pool = db_pool._ensure_pool(db_path)

    creation_started = threading.Event()
    release_creation = threading.Event()
    original_create = pool._create_connection

    def slow_create(self):
        creation_started.set()
        if not release_creation.wait(timeout=5.0):
            raise RuntimeError("create gate not released")
        return original_create()

    monkeypatch.setattr(db_pool.ConnectionPool, "_create_connection", slow_create)

    _sd_ident = {"v": None}
    _sd_parked = threading.Event()
    pool._condition = _ParkProbe(pool._condition, _sd_ident, _sd_parked)

    # Drain the pool so the next checkout must create.
    while not pool._pool.empty():
        pool._pool.get_nowait()
    with pool._condition:
        pool._created = 0

    checkout_done = threading.Event()

    def do_checkout():
        try:
            pool.checkout(timeout=5.0, validate=True)
        except Exception:
            pass
        finally:
            checkout_done.set()

    t_checkout = threading.Thread(target=do_checkout)
    t_checkout.start()
    assert creation_started.wait(timeout=5.0)
    assert pool._creating == 1

    # Trigger shutdown — it must not zero _creating while the
    # creator is in progress.
    shutdown_done = threading.Event()

    def do_shutdown():
        _sd_ident["v"] = threading.get_ident()
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass
        shutdown_done.set()

    t_shutdown = threading.Thread(target=do_shutdown)
    t_shutdown.start()
    # Shutdown must park on the pool condition while the creation is in progress.
    assert _sd_parked.wait(timeout=5.0), "shutdown never parked on the pool condition"
    # _creating must still be 1.
    assert pool._creating == 1

    release_creation.set()
    assert checkout_done.wait(timeout=5.0)
    assert shutdown_done.wait(timeout=5.0)
    assert not t_shutdown.is_alive()


def test_shutdown_rethrows_owner_error_to_waiters(tmp_path):
    """If the shutdown owner's invariants fail, the exception is stored
    and re-raised to subsequent waiters."""
    pool = db_pool.ConnectionPool(
        str(tmp_path / "shutdown-rethrow.db"),
        pool_size=4,
        lazy=True,
    )
    try:
        # Force the invariant failure by closing the pool's internal
        # queue under it. We patch ``qsize`` to return a non-zero
        # value when the queue is empty.
        def bad_qsize(self):
            return 1

        original_qsize = type(pool._pool).qsize
        type(pool._pool).qsize = bad_qsize
        try:
            pool.shutdown(grace_period=0)
        except AssertionError as exc:
            assert "shutdown invariants" in str(exc)
        finally:
            type(pool._pool).qsize = original_qsize

        # A second shutdown call must re-raise the stored error.
        with pytest.raises(AssertionError, match="shutdown invariants"):
            pool.shutdown(grace_period=0)
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Maintenance drain, superseded recovery, terminal shutdown discrimination
# ---------------------------------------------------------------------------


def test_begin_path_maintenance_releases_token_on_pool_shutdown_failure(
    tmp_path, monkeypatch
):
    """A failure inside the detached pool's shutdown must still release
    the maintenance token. Otherwise the path is wedged forever."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "shutdown-fail.db")

    # Install a real pool so maintenance can detach it.
    pool = db_pool._ensure_pool(db_path)

    def bad_shutdown(self, grace_period=0):
        raise RuntimeError("simulated shutdown failure")

    monkeypatch.setattr(db_pool.ConnectionPool, "shutdown", bad_shutdown)

    try:
        with pytest.raises(RuntimeError, match="simulated shutdown failure"):
            db_pool.begin_path_maintenance(db_path, drain_timeout=2.0)
    finally:
        # The token must be released even though shutdown raised.
        with db_pool._pool_condition:
            assert db_path not in db_pool._maintenance_paths, (
                "maintenance token leaked on shutdown failure"
            )
        # Cleanup
        try:
            monkeypatch.undo()
        except Exception:
            pass


def test_begin_path_maintenance_releases_token_on_drain_timeout(
    tmp_path, monkeypatch
):
    """A drain timeout must release the token and reattach the pool,
    not leave the path wedged forever."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "drain-timeout.db")
    pool = db_pool._ensure_pool(db_path)

    # Hold a lease open so the drain cannot complete.
    lease = pool.checkout(validate=True)
    try:
        with pytest.raises(RuntimeError, match="drain timed out"):
            db_pool.begin_path_maintenance(db_path, drain_timeout=0.2)
    finally:
        pool.checkin(lease)

    with db_pool._pool_condition:
        assert db_path not in db_pool._maintenance_paths, (
            "maintenance token leaked on drain timeout"
        )
        # Pool must be reattached (or replaced) — not left in a half-detached state.
        assert db_pool._pool is not None
        assert db_pool._pool._db_path == db_path

    # The pool is back in service.
    lease2 = pool.checkout(validate=True)
    pool.checkin(lease2)


def test_maintenance_drain_does_not_force_invalidate_unrelated_leases(
    tmp_path,
):
    """Maintenance begins a *drain* state that rejects new checkouts but
    does NOT mark the pool _retiring. Already-issued leases continue to
    operate normally through their own operation lock."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "drain-keep-lease.db")
    pool = db_pool._ensure_pool(db_path)

    # Take a lease, then start maintenance in another thread.
    lease = pool.checkout(validate=True)
    token_holder = []

    def start_maint():
        try:
            tok = db_pool.begin_path_maintenance(db_path, drain_timeout=2.0)
            token_holder.append(tok)
        except Exception as exc:
            token_holder.append(exc)

    t = threading.Thread(target=start_maint)
    t.start()

    # Wait for the pool to enter maintenance_draining.
    deadline = time.monotonic() + 5.0
    with pool._condition:
        while not pool._maintenance_draining and time.monotonic() < deadline:
            pool._condition.wait(timeout=0.1)
        assert pool._maintenance_draining, "pool should be draining"

    # Pool must NOT be retiring.
    assert pool._retiring is False, (
        "maintenance must not mark pool retiring"
    )

    # Existing lease must still operate.
    row = lease.execute("SELECT 42").fetchone()
    assert row[0] == 42

    # New checkouts must be rejected.
    with pytest.raises(RuntimeError, match="Connection pool is retiring"):
        pool.checkout(timeout=0.2)

    # Release the lease so the drain can complete.
    pool.checkin(lease)

    t.join(timeout=3.0)
    assert not t.is_alive()
    # The token must be held in _maintenance_paths (caller will release).
    assert isinstance(token_holder[0], db_pool._MaintenanceToken)
    with db_pool._pool_condition:
        assert db_path in db_pool._maintenance_paths

    # Clean up: release the token. The pool is shut down, so we just
    # remove the entry; we do not reattach.
    db_pool.end_path_maintenance(token_holder[0])


def test_superseded_source_does_not_rewrite_newer_pool_file(
    tmp_path, monkeypatch
):
    """If a source pool becomes stale (a newer same-path generation is
    now global), recovery must retire the source but NOT touch the
    file. The newer generation's readers must not see their data
    overwritten."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "superseded.db")
    # Pre-create a healthy db at the path.
    sqlite3.connect(db_path).close()

    # Force the file to be a "do not touch" marker so we can detect
    # whether recovery rewrote it.
    marker = b"MUST_NOT_BE_OVERWRITTEN"
    with open(db_path, "wb") as f:
        f.write(marker)

    source_pool = db_pool._ensure_pool(db_path)
    # Simulate the source observing corruption.
    source_pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    # A newer same-path pool replaces the global.
    new_pool = db_pool.ConnectionPool(db_path, pool_size=4, lazy=True)
    with db_pool._pool_condition:
        db_pool._pool = new_pool

    # Patch recover_corrupt_db to RAISE if called: supersession must
    # skip file repair entirely.
    def must_not_run(path):
        raise AssertionError("superseded source must not call recover_corrupt_db")

    from memory import db_recovery
    original = db_recovery.recover_corrupt_db
    db_recovery.recover_corrupt_db = must_not_run
    try:
        # Trigger recovery on the source pool. This is the path that
        # would have rewritten the file in the buggy implementation.
        db_pool._recover_source(
            source_pool,
            source_pool._corruption_error,
        )
    finally:
        db_recovery.recover_corrupt_db = original

    # Verify the file was not touched.
    with open(db_path, "rb") as f:
        assert f.read() == marker, (
            "superseded source must not rewrite the database file"
        )

    # The newer pool must still be the global.
    assert db_pool._pool is new_pool


def test_transactions_preserve_primary_exception_under_checkin_failure(
    tmp_path, monkeypatch
):
    """If a transaction body raises and the subsequent return_connection
    also raises, the body's exception must still propagate (not be
    replaced by the checkin error)."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "tx-preserve.db")
    pool = db_pool._ensure_pool(db_path)

    class _PrimaryError(Exception):
        pass

    def bad_return_connection(conn):
        raise RuntimeError("simulated return_connection failure")

    monkeypatch.setattr(
        db_pool, "return_connection", bad_return_connection
    )

    with pytest.raises(_PrimaryError):
        with db_pool.transaction(db_path):
            raise _PrimaryError("body error")


def test_transactions_checkin_failure_propagates_after_successful_commit(
    tmp_path, monkeypatch
):
    """If a transaction commits successfully but the return_connection
    fails, the body's success is preserved AND the checkin error
    propagates. The user knows the commit succeeded (they see the
    data) but also knows the cleanup leaked."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "tx-success-fail.db")
    pool = db_pool._ensure_pool(db_path)

    def bad_return_connection(conn):
        raise RuntimeError("simulated return_connection failure")

    monkeypatch.setattr(
        db_pool, "return_connection", bad_return_connection
    )

    # The body runs without raising. The checkin error must propagate.
    with pytest.raises(RuntimeError, match="simulated return_connection failure"):
        with db_pool.transaction(db_path):
            pass

    # Verify the table from initial schema is still reachable: the
    # commit succeeded even though the cleanup failed.
    with db_pool._checkout_global(db_path) as lease:
        row = lease.execute("SELECT 1").fetchone()
        assert row[0] == 1


def test_terminal_shutdown_deadline_skips_unquiesced_paths(
    tmp_path, monkeypatch
):
    """Terminal shutdown with a 0.5s deadline while a maintenance
    token is still active must not hang. The shutdown proceeds
    without the path being quiesced, and the WAL checkpoint is
    skipped because the structured result reports ``quiesced=False``."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "terminal-deadline.db")
    pool = db_pool._ensure_pool(db_path)

    # Register a maintenance token by hand to simulate a long-running
    # migration.
    token = db_pool._MaintenanceToken(db_path)
    with db_pool._pool_condition:
        db_pool._maintenance_paths[db_path] = token

    # Run terminal shutdown with a tight deadline. The maintenance
    # path is not released, so the wait should bail out.
    deadline = time.monotonic() + 0.5
    result = db_pool._begin_terminal_shutdown(deadline=deadline)
    # Must return a structured _TerminalDetach with the detached pool.
    assert isinstance(result, db_pool._TerminalDetach)
    assert isinstance(result.pool, db_pool.ConnectionPool)
    # The path was not quiesced.
    assert result.quiesced is False
    assert db_path in result.outstanding_paths
    assert db_pool._terminal_shutdown is True


def test_execute_with_close_path_under_concurrent_checkin(tmp_path):
    """Cursor registration must happen under the operation lock so a
    concurrent checkin cannot fail to close the new cursor. With
    the fix, even if the body finishes immediately and checkin begins
    right after, _cursors has the cursor."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "execute-cursor.db")
    pool = db_pool.ConnectionPool(db_path, pool_size=2, lazy=True)
    try:
        lease = pool.checkout(validate=True)
        try:
            # Use the real execute() and call checkin immediately.
            cursor = lease.execute("SELECT 1")
            # The cursor is registered in lease._cursors under the
            # operation lock; this is the production path.
            assert cursor in lease._cursors
        finally:
            pool.checkin(lease)
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_begin_path_maintenance_does_not_release_token_when_path_unattached(
    tmp_path, monkeypatch
):
    """If the global pool is not on the maintenance path, the token
    is published but no pool is detached. The token is then released
    by the caller. No pool-state mutation occurs."""

    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "no-detach.db")
    # Global is a pool for a different path.
    other_path = str(tmp_path / "other.db")
    other_pool = db_pool._ensure_pool(other_path)

    token = db_pool.begin_path_maintenance(db_path)
    with db_pool._pool_condition:
        assert db_path in db_pool._maintenance_paths
        # Global pool is unchanged (still other_pool).
        assert db_pool._pool is other_pool

    db_pool.end_path_maintenance(token)
    with db_pool._pool_condition:
        assert db_path not in db_pool._maintenance_paths
        assert db_pool._pool is other_pool


# ---------------------------------------------------------------------------
# Discriminating tests for the new fixes (replace the non-discriminating
# versions of test_checkin_holds_operation_lock_through_rollback_and_reset,
# test_shutdown_positive_grace_allows_voluntary_checkin,
# test_global_checkout_validation_failure_balances_admission, and
# test_close_all_connections_does_not_erase_concurrent_replacement).
# ---------------------------------------------------------------------------


def test_checkin_holds_operation_lock_through_rollback_and_reset_discriminating(
    tmp_path,
):
    """Pin the complete checkin lock publication sequence.

    The test verifies that `ConnectionPool.checkin(lease)`:

    1. Holds the operation lock continuously for the entire
       observable success path (rollback + reset + PRAGMA +
       final state publication), with exactly one outermost
       acquisition and one outermost release.
    2. Visits every required phase in exact order, including the
       cursor close (depth 2), the two distinct PRAGMA phases,
       and the final pool-state publication under both the
       operation lock and the pool condition.
    3. Releases the lock at the end so a different thread
       (not the check-in thread) can acquire it.

    A test-only instrumented wrapper around the real
    `threading.RLock` tracks ownership transitions. A
    delegating `_PhaseObservingConnection` wraps the physical
    connection to record the rollback/reset/PRAGMA phases.
    Delegating `_PhaseObservingSet`, `_PhaseObservingDict`,
    `_PhaseObservingQueue`, and `_PhaseObservingCondition`
    wrappers record the pool-state publication phases. A
    delegating `_PhaseObservingCursor` wraps the raw cursor
    to record the cursor close at depth 2. A class-level
    property replaces the `_released` slot descriptor to
    record the `lease._released` write without changing
    production.

    Callbacks only record and never assert. Production
    intentionally catches `Exception` around cursor closure,
    rollback, and reset operations, so all assertions
    occur after `pool.checkin(lease)` returns. The
    cross-thread post-return probe proves the lock is free
    (a reentrant lock can be reacquired by its current owner,
    so a same-thread check would be insufficient).

    No `time.sleep`, `os.sched_yield`, witness polling, or
    scheduler timing window is used."""
    import memory.db_pool as db_pool

    class _LockOwnershipTracker:
        """Wraps a real `threading.RLock` to track ownership
        transitions. Delegates to the real lock for actual
        locking behavior. Provides `__enter__`/`__exit__`
        explicitly because the special-method protocol
        bypasses `__getattr__` for dunder methods.
        """
        def __init__(self, real_lock):
            self._real = real_lock
            self._owner_ident = None
            self._depth = 0
            self.outermost_acquisitions = 0
            self.outermost_releases = 0
            self.transition_violations = []

        def acquire(self, blocking=True, timeout=-1):
            current_ident = threading.get_ident()
            if self._depth < 0:
                self.transition_violations.append(
                    f"acquire started at negative depth {self._depth}"
                )
            if self._depth == 0 and self._owner_ident is not None:
                self.transition_violations.append(
                    "depth is zero but tracker still has owner "
                    f"{self._owner_ident}"
                )
            if self._depth > 0 and self._owner_ident is None:
                self.transition_violations.append(
                    f"depth is {self._depth} but tracker has no owner"
                )

            result = self._real.acquire(
                blocking=blocking,
                timeout=timeout,
            )
            if not result:
                return False

            if self._depth == 0:
                if self._owner_ident is not None:
                    self.transition_violations.append(
                        "outermost acquire found recorded owner "
                        f"{self._owner_ident}"
                    )
                self.outermost_acquisitions += 1
                self._owner_ident = current_ident
            elif self._owner_ident != current_ident:
                self.transition_violations.append(
                    "reentrant acquire owner mismatch "
                    f"(current={current_ident} "
                    f"tracker={self._owner_ident})"
                )

            self._depth += 1
            return True

        def release(self):
            current_ident = threading.get_ident()

            if self._depth <= 0:
                self.transition_violations.append(
                    f"release started at non-positive depth {self._depth}"
                )
            if self._owner_ident != current_ident:
                self.transition_violations.append(
                    "release owner mismatch "
                    f"(current={current_ident} "
                    f"tracker={self._owner_ident})"
                )
            if self._depth > 0 and self._owner_ident is None:
                self.transition_violations.append(
                    f"release started at depth {self._depth} "
                    "without a recorded owner"
                )

            self._real.release()
            self._depth -= 1

            if self._depth == 0:
                self.outermost_releases += 1
                self._owner_ident = None
            elif self._depth < 0:
                self.transition_violations.append(
                    f"release produced negative depth {self._depth}"
                )
            elif self._owner_ident is None:
                self.transition_violations.append(
                    f"release left depth {self._depth} "
                    "without a recorded owner"
                )

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
            return False

        def _is_owned(self):
            return self._real._is_owned()

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _PhaseRecorder:
        """Defers all assertions until after `pool.checkin(lease)`
        returns, because production intentionally catches
        `Exception` around cursor closure, rollback, and reset
        operations. Records visits and violations without
        raising."""
        def __init__(self, tracker, condition_descriptor):
            self.tracker = tracker
            self.condition_descriptor = condition_descriptor
            self.visits = []
            self.violations = []

        def visit(
            self, phase, *, expected_depth,
            require_condition=False,
        ):
            current_ident = threading.get_ident()
            tracker_owner = self.tracker._owner_ident
            tracker_depth = self.tracker._depth
            outermost_acq = self.tracker.outermost_acquisitions
            outermost_rel = self.tracker.outermost_releases
            condition_owned = False
            if require_condition:
                condition_owned = self._condition_is_owned()
            self.visits.append({
                "phase": phase,
                "current_ident": current_ident,
                "tracker_owner": tracker_owner,
                "tracker_depth": tracker_depth,
                "outermost_acquisitions": outermost_acq,
                "outermost_releases": outermost_rel,
                "condition_owned": condition_owned,
            })
            if tracker_owner != current_ident:
                self.violations.append(
                    f"{phase}: owner mismatch "
                    f"(current={current_ident} tracker={tracker_owner})"
                )
            if tracker_depth != expected_depth:
                self.violations.append(
                    f"{phase}: depth {tracker_depth} != expected "
                    f"{expected_depth}"
                )
            if outermost_acq != 1:
                self.violations.append(
                    f"{phase}: outermost_acquisitions {outermost_acq} "
                    f"!= 1"
                )
            if outermost_rel != 0:
                self.violations.append(
                    f"{phase}: outermost_releases {outermost_rel} "
                    f"!= 0"
                )
            if require_condition and not condition_owned:
                self.violations.append(
                    f"{phase}: pool condition not owned"
                )

        def _condition_is_owned(self):
            try:
                return self.condition_descriptor._is_owned()
            except Exception:
                return False

    class _PhaseObservingCursor:
        """Wraps the raw `sqlite3.Cursor` to record the cursor
        close phase. The real `_CursorLease.close()` path
        re-enters the operation lock (depth 2)."""
        def __init__(self, real_cursor, recorder):
            self._real = real_cursor
            self._recorder = recorder

        def close(self):
            self._real.close()
            self._recorder.visit(
                "cursor.close",
                expected_depth=2,
                require_condition=False,
            )

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _PhaseObservingConnection:
        """Wraps the physical `sqlite3.Connection` to record
        rollback, reset, and PRAGMA phases. Each callback
        delegates first, then records. The `execute()`
        observer normalizes SQL to distinguish the two
        PRAGMA calls."""
        def __init__(self, real_conn, recorder):
            self._real = real_conn
            self._recorder = recorder

        def execute(self, sql, *args, **kwargs):
            result = self._real.execute(sql, *args, **kwargs)
            normalized = " ".join(str(sql).lower().split())
            if normalized == "pragma busy_timeout = 5000":
                phase = "pragma.busy_timeout"
            elif normalized == "pragma wal_autocheckpoint = 1000":
                phase = "pragma.wal_autocheckpoint"
            else:
                phase = f"connection.execute:{normalized}"
            self._recorder.visit(
                phase,
                expected_depth=1,
                require_condition=False,
            )
            return result

        def rollback(self):
            result = self._real.rollback()
            self._recorder.visit(
                "rollback",
                expected_depth=1,
                require_condition=False,
            )
            return result

        @property
        def row_factory(self):
            return self._real.row_factory

        @row_factory.setter
        def row_factory(self, value):
            self._real.row_factory = value
            self._recorder.visit(
                "row_factory.reset",
                expected_depth=1,
                require_condition=False,
            )

        @property
        def text_factory(self):
            return self._real.text_factory

        @text_factory.setter
        def text_factory(self, value):
            self._real.text_factory = value
            self._recorder.visit(
                "text_factory.reset",
                expected_depth=1,
                require_condition=False,
            )

        @property
        def isolation_level(self):
            return self._real.isolation_level

        @isolation_level.setter
        def isolation_level(self, value):
            self._real.isolation_level = value
            self._recorder.visit(
                "isolation_level.reset",
                expected_depth=1,
                require_condition=False,
            )

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _PhaseObservingSet:
        """Wraps `pool._checking_in` (a `set`) to record add
        and discard phases."""
        def __init__(self, real_set, recorder):
            self._real = real_set
            self._recorder = recorder

        def add(self, item):
            result = self._real.add(item)
            self._recorder.visit(
                "checking_in.add",
                expected_depth=1,
                require_condition=True,
            )
            return result

        def discard(self, item):
            result = self._real.discard(item)
            self._recorder.visit(
                "checking_in.discard",
                expected_depth=1,
                require_condition=True,
            )
            return result

        def __contains__(self, item):
            return item in self._real

        def __iter__(self):
            return iter(self._real)

        def __len__(self):
            return len(self._real)

        def __bool__(self):
            return bool(self._real)

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _PhaseObservingDict:
        """Wraps `pool._checked_out` (a `dict`) to record
        deletion phases. Does not record ordinary `get()`
        calls because check-in performs multiple identity
        checks."""
        def __init__(self, real_dict, recorder):
            self._real = real_dict
            self._recorder = recorder

        def __getitem__(self, key):
            return self._real[key]

        def __setitem__(self, key, value):
            self._real[key] = value

        def __delitem__(self, key):
            del self._real[key]
            self._recorder.visit(
                "checked_out.delete",
                expected_depth=1,
                require_condition=True,
            )

        def __contains__(self, key):
            return key in self._real

        def __iter__(self):
            return iter(self._real)

        def __len__(self):
            return len(self._real)

        def __bool__(self):
            return bool(self._real)

        def get(self, key, default=None):
            return self._real.get(key, default)

        def values(self):
            return self._real.values()

        def items(self):
            return self._real.items()

        def keys(self):
            return self._real.keys()

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _PhaseObservingQueue:
        """Wraps `pool._pool` (a `queue.Queue`) to record the
        put_nowait phase. The underlying original queue must
        receive the exact object production passed."""
        def __init__(self, real_queue, recorder):
            self._real = real_queue
            self._recorder = recorder

        def put_nowait(self, item):
            self._real.put_nowait(item)
            self._recorder.visit(
                "queue.put_nowait",
                expected_depth=1,
                require_condition=True,
            )

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _PhaseObservingCondition:
        """Wraps `pool._condition` (a `threading.Condition`) to
        record `notify_all` phases. `__enter__` and `__exit__`
        delegate to the original condition."""
        def __init__(self, real_condition, recorder):
            self._real = real_condition
            self._recorder = recorder

        def __enter__(self):
            return self._real.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._real.__exit__(exc_type, exc_val, exc_tb)

        def acquire(self, *args, **kwargs):
            return self._real.acquire(*args, **kwargs)

        def release(self):
            self._real.release()

        def wait(self, *args, **kwargs):
            return self._real.wait(*args, **kwargs)

        def wait_for(self, *args, **kwargs):
            return self._real.wait_for(*args, **kwargs)

        def notify(self, *args, **kwargs):
            return self._real.notify(*args, **kwargs)

        def notify_all(self, *args, **kwargs):
            result = self._real.notify_all(*args, **kwargs)
            self._recorder.visit(
                "condition.notify_all",
                expected_depth=1,
                require_condition=True,
            )
            return result

        def __getattr__(self, name):
            return getattr(self._real, name)

    pool = None
    lease = None
    cursor = None
    original_lock = None
    original_conn = None
    original_raw_cursor = None
    original_pool_queue = None
    original_checking_in = None
    original_checked_out = None
    original_condition = None
    original_reset_method = None
    released_slot_descriptor = None
    released_descriptor_installed = False
    tracker = None
    recorder = None
    observing_cursor = None
    observing_connection = None
    observing_checking_in = None
    observing_checked_out = None
    observing_queue = None
    observing_condition = None
    installed_property = None
    lock_installed = False
    cursor_installed = False
    connection_installed = False
    checking_in_installed = False
    checked_out_installed = False
    queue_installed = False
    condition_installed = False
    utilization_wrapper_installed = False
    primary_exception = None
    primary_traceback = None

    try:
        pool = db_pool.ConnectionPool(
            str(tmp_path / "checkin-holds-lock-disc.db"),
            pool_size=4, lazy=True,
        )
        lease = pool.checkout(validate=True)
        cursor = lease.execute("SELECT 1")
        assert cursor in lease._cursors, (
            "test setup error: cursor not registered in lease._cursors"
        )

        original_lock = lease._operation_lock
        original_conn = lease._conn
        original_raw_cursor = cursor._cursor
        original_pool_queue = pool._pool
        original_checking_in = pool._checking_in
        original_checked_out = pool._checked_out
        original_condition = pool._condition
        original_reset_method = (
            pool._maybe_reset_utilization_warning_locked
        )
        released_slot_descriptor = (
            db_pool._ConnectionLease.__dict__["_released"]
        )

        tracker = _LockOwnershipTracker(original_lock)
        recorder = _PhaseRecorder(tracker, original_condition)

        observing_cursor = _PhaseObservingCursor(
            original_raw_cursor,
            recorder,
        )
        observing_connection = _PhaseObservingConnection(
            original_conn,
            recorder,
        )
        observing_checking_in = _PhaseObservingSet(
            original_checking_in,
            recorder,
        )
        observing_checked_out = _PhaseObservingDict(
            original_checked_out,
            recorder,
        )
        observing_queue = _PhaseObservingQueue(
            original_pool_queue,
            recorder,
        )
        observing_condition = _PhaseObservingCondition(
            original_condition,
            recorder,
        )

        def get_released(observed_lease):
            return released_slot_descriptor.__get__(
                observed_lease,
                db_pool._ConnectionLease,
            )

        def set_released(observed_lease, value):
            released_slot_descriptor.__set__(
                observed_lease,
                value,
            )
            if observed_lease is lease:
                recorder.visit(
                    "lease._released",
                    expected_depth=1,
                    require_condition=True,
                )

        installed_property = property(
            get_released,
            set_released,
        )

        def utilization_wrapper(*args, **kwargs):
            result = original_reset_method(*args, **kwargs)
            recorder.visit(
                "utilization.reset",
                expected_depth=1,
                require_condition=True,
            )
            return result

        # Establish the success-path precondition before installing
        # observers. No checkin phase runs until every observer is
        # installed and verified.
        object.__setattr__(
            pool,
            "_utilization_warning_active",
            True,
        )

        # Install every already-constructed observer in one
        # contiguous block. Set each marker only after its
        # corresponding installation succeeds.
        object.__setattr__(
            lease,
            "_operation_lock",
            tracker,
        )
        lock_installed = True

        object.__setattr__(
            cursor,
            "_cursor",
            observing_cursor,
        )
        cursor_installed = True

        object.__setattr__(
            lease,
            "_conn",
            observing_connection,
        )
        connection_installed = True

        object.__setattr__(
            pool,
            "_checking_in",
            observing_checking_in,
        )
        checking_in_installed = True

        object.__setattr__(
            pool,
            "_checked_out",
            observing_checked_out,
        )
        checked_out_installed = True

        object.__setattr__(
            pool,
            "_pool",
            observing_queue,
        )
        queue_installed = True

        object.__setattr__(
            pool,
            "_condition",
            observing_condition,
        )
        condition_installed = True

        object.__setattr__(
            pool,
            "_maybe_reset_utilization_warning_locked",
            utilization_wrapper,
        )
        utilization_wrapper_installed = True

        # The class-wide descriptor is installed last to minimize
        # its process-wide lifetime.
        setattr(
            db_pool._ConnectionLease,
            "_released",
            installed_property,
        )
        released_descriptor_installed = True

        # Verify every observer was installed before checkin.
        assert lease._operation_lock is tracker
        assert lease._conn is observing_connection
        assert cursor._cursor is observing_cursor
        assert cursor in lease._cursors
        assert pool._checking_in is observing_checking_in
        assert pool._checked_out is observing_checked_out
        assert pool._pool is observing_queue
        assert pool._condition is observing_condition
        assert (
            pool._maybe_reset_utilization_warning_locked
            is utilization_wrapper
        )
        assert (
            db_pool._ConnectionLease.__dict__["_released"]
            is installed_property
        )

        # Run the real checkin. Callbacks only record; all
        # assertions occur after this returns.
        pool.checkin(lease)

        # Cross-thread post-return lock probe. A reentrant
        # lock can be reacquired by its current owner, so a
        # same-thread check is insufficient. The probe runs
        # in a different thread and proves the lock was
        # released.
        probe_result = []
        probe_exceptions = []
        def probe():
            try:
                acquired = original_lock.acquire(blocking=False)
                if acquired:
                    original_lock.release()
                probe_result.append(acquired)
            except BaseException as exc:
                probe_exceptions.append(exc)

        t_probe = threading.Thread(target=probe)
        t_probe.start()
        t_probe.join(timeout=5.0)
        assert not t_probe.is_alive(), (
            "probe thread did not finish"
        )
        assert not probe_exceptions, (
            f"probe raised: {probe_exceptions!r}"
        )
        assert probe_result == [True], (
            f"expected exactly one successful cross-thread probe, "
            f"got {probe_result!r}"
        )

        # Deferred phase-list assertion. Exact equality rejects
        # missing, duplicate, reordered, merged, or skipped
        # phases.
        expected_phases = [
            "checking_in.add",
            "cursor.close",
            "rollback",
            "row_factory.reset",
            "text_factory.reset",
            "isolation_level.reset",
            "pragma.busy_timeout",
            "pragma.wal_autocheckpoint",
            "checking_in.discard",
            "checked_out.delete",
            "lease._released",
            "queue.put_nowait",
            "utilization.reset",
            "condition.notify_all",
        ]
        actual_phases = [v["phase"] for v in recorder.visits]
        assert actual_phases == expected_phases, (
            f"phase trace mismatch:\n"
            f"  expected: {expected_phases}\n"
            f"  actual:   {actual_phases}"
        )

        # Deferred oracle assertions. Every bookkeeping
        # anomaly recorded as a violation by the recorder or
        # the tracker must be empty.
        assert recorder.violations == [], (
            f"recorder violations: {recorder.violations!r}"
        )
        assert tracker.transition_violations == [], (
            f"tracker violations: {tracker.transition_violations!r}"
        )
        assert tracker.outermost_acquisitions == 1, (
            f"expected 1 outermost acquisition, got "
            f"{tracker.outermost_acquisitions}"
        )
        assert tracker.outermost_releases == 1, (
            f"expected 1 outermost release, got "
            f"{tracker.outermost_releases}"
        )
        assert tracker._depth == 0, (
            f"expected depth 0 after checkin, got {tracker._depth}"
        )
        assert tracker._owner_ident is None, (
            f"expected no owner after checkin, got "
            f"{tracker._owner_ident}"
        )
        assert pool._utilization_warning_active is False, (
            f"expected _utilization_warning_active False, got "
            f"{pool._utilization_warning_active}"
        )
        assert lease._released is True, (
            f"expected lease._released True, got {lease._released}"
        )
        assert original_checking_in == set(), (
            f"expected original checking_in to be empty, got "
            f"{original_checking_in!r}"
        )
        assert lease._token not in original_checked_out, (
            f"expected lease._token {lease._token} not in "
            f"original_checked_out"
        )
        assert original_pool_queue.qsize() == 1, (
            f"expected original_pool_queue to have 1 item, got "
            f"{original_pool_queue.qsize()}"
        )
        assert pool._inflight == 0, (
            f"expected _inflight 0, got {pool._inflight}"
        )
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__
    finally:
        cleanup_failures = []

        def attempt_cleanup(label, action):
            try:
                action()
            except BaseException as exc:
                cleanup_failures.append((label, exc))

        # Restore the process-wide descriptor first.
        if released_descriptor_installed:
            attempt_cleanup(
                "restore _ConnectionLease._released descriptor",
                lambda: setattr(
                    db_pool._ConnectionLease,
                    "_released",
                    released_slot_descriptor,
                ),
            )

        # Restore the remaining observers in the prescribed order.
        if condition_installed:
            attempt_cleanup(
                "restore pool condition",
                lambda: object.__setattr__(
                    pool,
                    "_condition",
                    original_condition,
                ),
            )

        if utilization_wrapper_installed:
            attempt_cleanup(
                "remove utilization wrapper",
                lambda: pool.__dict__.pop(
                    "_maybe_reset_utilization_warning_locked",
                    None,
                ),
            )

        if queue_installed:
            attempt_cleanup(
                "restore pool queue",
                lambda: object.__setattr__(
                    pool,
                    "_pool",
                    original_pool_queue,
                ),
            )

        if checked_out_installed:
            attempt_cleanup(
                "restore checked-out mapping",
                lambda: object.__setattr__(
                    pool,
                    "_checked_out",
                    original_checked_out,
                ),
            )

        if checking_in_installed:
            attempt_cleanup(
                "restore checking-in set",
                lambda: object.__setattr__(
                    pool,
                    "_checking_in",
                    original_checking_in,
                ),
            )

        if cursor_installed:
            attempt_cleanup(
                "restore raw cursor",
                lambda: object.__setattr__(
                    cursor,
                    "_cursor",
                    original_raw_cursor,
                ),
            )

        if connection_installed:
            attempt_cleanup(
                "restore lease connection",
                lambda: object.__setattr__(
                    lease,
                    "_conn",
                    original_conn,
                ),
            )

        if lock_installed:
            attempt_cleanup(
                "restore lease operation lock",
                lambda: object.__setattr__(
                    lease,
                    "_operation_lock",
                    original_lock,
                ),
            )

        if pool is not None:
            attempt_cleanup(
                "shutdown pool",
                lambda: pool.shutdown(grace_period=0),
            )

        cleanup_exception = None
        if cleanup_failures:
            cleanup_details = "; ".join(
                f"{label}: {exc!r}"
                for label, exc in cleanup_failures
            )
            cleanup_exception = RuntimeError(
                "test cleanup failed after all cleanup attempts: "
                f"{cleanup_details}"
            )
            cleanup_exception.cleanup_failures = tuple(
                cleanup_failures
            )

        if primary_exception is not None:
            if cleanup_exception is not None:
                raise primary_exception.with_traceback(
                    primary_traceback
                ) from cleanup_exception
            raise primary_exception.with_traceback(
                primary_traceback
            )

        if cleanup_exception is not None:
            raise cleanup_exception


def test_shutdown_positive_grace_allows_voluntary_checkin_discriminating(
    tmp_path,
):
    """Shutdown must wait for a real in-flight operation (not just a
    checked-out lease). The operation lock is held through the
    shutdown observation window, proving the operation is genuinely
    in progress."""
    import memory.db_pool as db_pool

    pool = db_pool.ConnectionPool(
        str(tmp_path / "shutdown-grace-voluntary-disc.db"),
        pool_size=4, lazy=True,
    )

    _sd_ident = {"v": None}
    _sd_parked = threading.Event()
    pool._condition = _ParkProbe(pool._condition, _sd_ident, _sd_parked)

    try:
        lease = pool.checkout(validate=True)

        operation_started = threading.Event()
        operation_release = threading.Event()
        checkin_done = threading.Event()

        def voluntary():
            # Hold the operation lock so shutdown must wait for a
            # real in-flight operation, not just a checked-out lease.
            with lease._operation_lock:
                operation_started.set()
                # Run a real DB operation.
                lease._conn.execute("SELECT 1")
                operation_release.wait(timeout=2.0)
            try:
                pool.checkin(lease)
            finally:
                checkin_done.set()

        t = threading.Thread(target=voluntary)
        t.start()
        assert operation_started.wait(timeout=2.0)

        shutdown_done = threading.Event()
        def do_shutdown():
            _sd_ident["v"] = threading.get_ident()
            pool.shutdown(grace_period=0.5)
            shutdown_done.set()

        t_shutdown = threading.Thread(target=do_shutdown)
        t_shutdown.start()

        # Shutdown must park on the pool condition while the operation is in flight.
        assert _sd_parked.wait(timeout=5.0), "shutdown never parked on the pool condition"

        operation_release.set()
        assert checkin_done.wait(timeout=2.0)
        assert shutdown_done.wait(timeout=5.0)
        t_shutdown.join(timeout=1.0)
        t.join(timeout=1.0)
        assert not t_shutdown.is_alive()
        assert not t.is_alive()
        assert pool._created == 0
        assert pool._checked_out == {}
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_global_checkout_validation_failure_balances_admission_discriminating(
    tmp_path, monkeypatch
):
    """Discriminating version: real connection creation succeeds
    (the corruption probe is hit on the validation SELECT 1, NOT
    during creation). The transferred admission must be released."""
    import memory.db_pool as db_pool
    from memory.db_pool import DatabaseRecoveryError

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "validate-fail-disc.db")
    pool = db_pool._ensure_pool(db_path)

    real_create = db_pool.ConnectionPool._create_connection

    class _ValidationFailingConn:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, *args, **kwargs):
            # The validation SELECT 1 (no pragma) raises. The PRAGMA
            # statements in _create_connection still pass because
            # they are filtered by "SELECT 1" check below.
            if "SELECT 1" in str(sql) and "PRAGMA" not in str(sql):
                raise sqlite3.OperationalError(
                    "database disk image is malformed"
                )
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def patched(self):
        return _ValidationFailingConn(real_create(self))

    monkeypatch.setattr(
        db_pool.ConnectionPool, "_create_connection", patched
    )
    # Make the recovery step fail so the checkout surfaces.
    monkeypatch.setattr(
        "memory.db_recovery.recover_corrupt_db", lambda p: False
    )

    # The validate path fails. _checkout_global retries via the
    # recovery path, which also fails, and the helper raises.
    with pytest.raises(DatabaseRecoveryError):
        db_pool._checkout_global(db_path, validate=True, timeout=5.0)

    # The pool's admitted count must be 0.
    assert pool._inflight == 0
    assert pool._creating == 0


def test_close_all_connections_does_not_erase_concurrent_replacement_discriminating(
    tmp_path,
):
    """Real race: close_all_connections detaches pool_a, then a
    concurrent thread publishes a replacement via _ensure_pool.
    The replacement published AFTER the detach must survive."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "close-all-race-disc.db")
    pool_a = db_pool._ensure_pool(db_path)

    # Wrap pool_a.shutdown with a barrier so we can publish the
    # replacement AFTER close_all_connections has detached pool_a
    # but BEFORE pool_a.shutdown completes.
    real_shutdown = pool_a.shutdown
    detach_signaled = threading.Event()
    replacement_published = threading.Event()
    shutdown_can_finish = threading.Event()

    def intercepted_shutdown(*args, **kwargs):
        # close_all_connections has already set _pool = None and is
        # now calling pool_a.shutdown(). Signal the publisher thread
        # to publish the replacement, then wait for it to finish.
        detach_signaled.set()
        replacement_published.wait(timeout=3.0)
        shutdown_can_finish.set()
        return real_shutdown(*args, **kwargs)

    pool_a.shutdown = intercepted_shutdown

    close_done = threading.Event()
    close_result = {}

    def do_close():
        db_pool.close_all_connections()
        close_result["pool_a_shutdown"] = pool_a._shutting_down
        close_done.set()

    def do_publish():
        detach_signaled.wait(timeout=3.0)
        # Publish a replacement via the real _ensure_pool path.
        new_pool = db_pool._ensure_pool(db_path)
        replacement_published.set()
        close_result["replacement"] = new_pool

    t_close = threading.Thread(target=do_close)
    t_publish = threading.Thread(target=do_publish)
    t_publish.start()
    t_close.start()
    assert close_done.wait(timeout=5.0)
    shutdown_can_finish.wait(timeout=3.0)
    t_close.join(timeout=1.0)
    t_publish.join(timeout=1.0)
    assert not t_close.is_alive()
    assert not t_publish.is_alive()

    # pool_a was shut down.
    assert close_result["pool_a_shutdown"] is True
    # The replacement is the current global.
    assert db_pool._pool is not None
    assert db_pool._pool is not pool_a
    assert db_pool._pool._db_path == db_path


# ---------------------------------------------------------------------------
# Production-path migration tests (db_summaries / batch_manager)
# ---------------------------------------------------------------------------


def test_db_summaries_ensure_extraction_schema_releases_probe_lease(
    tmp_path, monkeypatch
):
    """After the probe detects a missing extraction_batches table,
    the probe lease MUST be released before init_db runs. The path
    maintenance token blocks concurrent same-path checkouts during
    the migration. After init_db, the new pool is published and the
    schema is readable.
    """
    import memory.db_pool as db_pool
    import memory.db_summaries as db_summaries
    from memory import db_migrations

    db_path = str(tmp_path / "ds-summaries-prod.db")
    # Pre-initialize the database on the test path so the
    # post-init_db pool is bound to db_path. The maintenance drain
    # shuts down the post-init pool; a subsequent _checkout_global
    # would otherwise create a new pool on the default path.
    monkeypatch.setattr(db_pool, "_get_db_path", lambda: db_path)

    db_migrations.init_db(db_path)
    db_pool.close_all_connections()

    # Drop the table after the pool is closed.
    c = sqlite3.connect(db_path)
    c.execute("DROP TABLE extraction_batches")
    c.commit()
    c.close()

    checkout_count = [0]
    init_count = [0]
    original_connection = db_pool.connection

    def counting_connection(*args, **kwargs):
        checkout_count[0] += 1
        return original_connection(*args, **kwargs)

    monkeypatch.setattr(db_pool, "connection", counting_connection)

    import memory.db as _db_module
    original_init = _db_module.init_db

    def counting_init(*args, **kwargs):
        init_count[0] += 1
        return original_init(*args, **kwargs)

    monkeypatch.setattr(_db_module, "init_db", counting_init)

    # Call the helper directly to verify the probe-init-verify flow.
    db_summaries._ensure_extraction_schema()

    # Two probe checkouts: one for the missing-schema probe, one for
    # the post-init_db verification.
    assert checkout_count[0] >= 2, (
        f"expected probe + post-init probe, got {checkout_count[0]}"
    )
    # init_db was called at least once.
    assert init_count[0] >= 1
    # The init_db call MUST have used the same path the probe saw.
    # _ensure_extraction_schema now resolves the path from the
    # probe's pool and passes it explicitly.

    # After the call, a direct INSERT via the pool on the same
    # path works. This is the production recovery path: a probe
    # detected a missing table, init_db rebuilt it, and a subsequent
    # call inserts into the repaired schema.
    from memory.db_pool import immediate_transaction
    with immediate_transaction(pooled=True) as conn:
        assert conn._pool._db_path == db_path
        cursor = conn.execute(
            "INSERT INTO extraction_batches (session_id, "
            "start_exchange_index, end_exchange_index, extraction_json, "
            "status, created_at, updated_at, start_message_index, "
            "end_message_index) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "real-prod-session", 0, 5, None, "pending",
                "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", 0, 5,
            ),
        )
        assert cursor.rowcount == 1
        assert cursor.lastrowid is not None


def test_batch_manager_ensure_extraction_schema_blocks_concurrent_checkout(
    tmp_path, monkeypatch
):
    """While the runtime migration is running (init_db holding the
    maintenance token), concurrent same-path checkouts are blocked.
    After init_db returns, the new pool is published and a
    subsequent checkout succeeds.
    """
    import memory.db_pool as db_pool
    import memory.batch_manager as batch_manager
    from memory import db_migrations

    db_path = str(tmp_path / "bm-prod.db")
    monkeypatch.setattr(db_pool, "_get_db_path", lambda: db_path)
    db_migrations.init_db(db_path)
    db_pool.close_all_connections()

    c = sqlite3.connect(db_path)
    c.execute("DROP TABLE extraction_batches")
    c.commit()
    c.close()

    # Patch the imported reference inside db_migrations (the
    # init_db body resolves _create_raw_connection from this
    # local binding, not from db_pool directly).
    init_started = threading.Event()
    init_can_finish = threading.Event()
    init_call_args = []

    real_create_raw = db_migrations._create_raw_connection

    def slow_create_raw(path, **kwargs):
        init_call_args.append(path)
        init_started.set()
        if not init_can_finish.wait(timeout=5.0):
            raise RuntimeError("init gate not released")
        return real_create_raw(path, **kwargs)

    monkeypatch.setattr(db_migrations, "_create_raw_connection", slow_create_raw)

    def trigger_migration():
        try:
            batch_manager._update_batch_status(9999, "pending")
        except Exception:
            pass

    t = threading.Thread(target=trigger_migration)
    t.start()
    assert init_started.wait(timeout=2.0)

    # The maintenance token is held while the migration blocks in
    # _create_raw_connection.
    with db_pool._pool_condition:
        assert db_path in db_pool._maintenance_paths, (
            "path maintenance token not held during init"
        )

    # A concurrent _checkout_global must block on the path
    # maintenance gate. Use a short timeout to detect the block.
    checkout_observed = []
    checkout_entered = threading.Event()
    checkout_done = threading.Event()

    def try_checkout():
        checkout_entered.set()
        try:
            db_pool._checkout_global(
                db_path, validate=True, timeout=0.5,
            )
        except Exception as exc:
            checkout_observed.append(("err", str(exc)))
        else:
            checkout_observed.append(("ok", None))
        finally:
            checkout_done.set()

    t_checkout = threading.Thread(target=try_checkout)
    t_checkout.start()
    # Wait for the checkout to actually attempt the gate so the
    # timeout counts from when it is blocked.
    assert checkout_entered.wait(timeout=2.0)
    # Now the checkout will time out in 0.5s. Wait up to 2s.
    assert checkout_done.wait(timeout=2.0), (
        "checkout thread did not return within 2s"
    )
    assert not t_checkout.is_alive()
    # The checkout must have raised a timeout — i.e., it did NOT
    # acquire a connection while init_db was holding the path.
    assert checkout_observed, "checkout did not run"
    assert checkout_observed[0][0] == "err", (
        "checkout succeeded during init_db maintenance: "
        f"{checkout_observed!r}"
    )
    err_msg = checkout_observed[0][1].lower()
    assert (
        "timed out" in err_msg
        or "deadline exceeded" in err_msg
    ), f"unexpected error: {checkout_observed[0][1]!r}"

    # Release the migration; it completes.
    init_can_finish.set()
    t.join(timeout=5.0)
    assert not t.is_alive()

    # _create_raw_connection was called with the probed path.
    assert len(init_call_args) >= 1
    assert init_call_args[0] == db_path, (
        f"init_db called with {init_call_args[0]!r}, expected {db_path!r}"
    )

    # A subsequent checkout succeeds.
    lease = db_pool._checkout_global(db_path, validate=True, timeout=5.0)
    db_pool.return_connection(lease)


# ---------------------------------------------------------------------------
# Session-token identity-CAS tests
# ---------------------------------------------------------------------------


def test_session_token_completion_does_not_remove_newer_token(
    tmp_path, monkeypatch
):
    """ABA: old submit owns token A, timeout releases token A, new
    submit owns token B, old future completes, old done callback
    attempts release(A) — token B must remain."""
    import memory.executors as executors
    from memory.executors import submit_sync
    from concurrent.futures import ThreadPoolExecutor

    created_pool = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="mem-sync-aba",
    )

    monkeypatch.setattr(executors, "_shutdown_requested", False)
    monkeypatch.setattr(executors, "_shutdown_complete", False)
    monkeypatch.setattr(executors, "_session_locks", {})
    monkeypatch.setattr(executors, "_sync_pool", created_pool)
    monkeypatch.setattr(executors, "_query_pool", None)
    monkeypatch.setattr(executors, "_summary_pool", None)
    monkeypatch.setattr(executors, "_timeout_pool", None)

    job_can_finish = threading.Event()
    job_started = threading.Event()
    job_started_2 = threading.Event()
    job_can_finish_2 = threading.Event()

    try:
        def slow_job(session_id):
            job_started.set()
            job_can_finish.wait(timeout=5.0)

        f_old = submit_sync(slow_job, "session-aba")
        assert f_old is not None
        assert job_started.wait(timeout=2.0)
        token_a = executors._session_locks["session-aba"]

        with executors._session_locks_lock:
            if executors._session_locks.get("session-aba") is token_a:
                del executors._session_locks["session-aba"]

        def slow_job_2(session_id):
            job_started_2.set()
            job_can_finish_2.wait(timeout=5.0)

        f_new = submit_sync(slow_job_2, "session-aba")
        assert f_new is not None
        new_callbacks_done = threading.Event()
        f_new.add_done_callback(
            lambda _future: new_callbacks_done.set()
        )
        assert job_started_2.wait(timeout=2.0)
        token_b = executors._session_locks["session-aba"]
        assert token_b is not token_a

        job_can_finish.set()
        f_old.result(timeout=5.0)

        with executors._session_locks_lock:
            assert executors._session_locks.get("session-aba") is token_b, (
                "old job's done callback deleted the newer token"
            )

        job_can_finish_2.set()
        f_new.result(timeout=5.0)
        assert new_callbacks_done.wait(timeout=2.0)
        assert "session-aba" not in executors._session_locks
    finally:
        job_can_finish.set()
        job_can_finish_2.set()
        created_pool.shutdown(wait=True, cancel_futures=True)
        with executors._session_locks_lock:
            executors._session_locks.clear()


def test_session_token_watchdog_submission_failure_preserves_ownership(
    tmp_path, monkeypatch
):
    """If sync submission succeeds but the watchdog timeout-guard
    submission fails, the ownership token is preserved while the
    sync job is still running."""
    import memory.executors as executors
    from memory.executors import submit_sync
    from concurrent.futures import ThreadPoolExecutor

    created_pool = ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="mem-sync-watchdog",
    )

    monkeypatch.setattr(executors, "_shutdown_requested", False)
    monkeypatch.setattr(executors, "_shutdown_complete", False)
    monkeypatch.setattr(executors, "_session_locks", {})
    monkeypatch.setattr(executors, "_sync_pool", created_pool)
    monkeypatch.setattr(executors, "_query_pool", None)
    monkeypatch.setattr(executors, "_summary_pool", None)
    monkeypatch.setattr(executors, "_timeout_pool", None)

    def bad_get_timeout_pool():
        class _Pool:
            def submit(self, *a, **kw):
                raise RuntimeError("simulated watchdog failure")
            def shutdown(self, *a, **kw):
                pass
        return _Pool()

    monkeypatch.setattr(executors, "_get_timeout_pool", bad_get_timeout_pool)

    job_started = threading.Event()
    job_can_finish = threading.Event()

    def blocking_job(session_id):
        job_started.set()
        job_can_finish.wait(timeout=5.0)

    try:
        future = submit_sync(blocking_job, "session-watchdog")
        assert future is not None
        callbacks_done = threading.Event()
        future.add_done_callback(
            lambda _future: callbacks_done.set()
        )
        assert job_started.wait(timeout=2.0)
        assert "session-watchdog" in executors._session_locks, (
            "token was removed while sync job is still running"
        )

        job_can_finish.set()
        future.result(timeout=5.0)
        assert callbacks_done.wait(timeout=2.0)
        assert "session-watchdog" not in executors._session_locks
    finally:
        job_can_finish.set()
        created_pool.shutdown(wait=True, cancel_futures=True)
        with executors._session_locks_lock:
            executors._session_locks.clear()


# ---------------------------------------------------------------------------
# Unsafe raw close path tracking
# ---------------------------------------------------------------------------


def test_path_unsafe_after_raw_close_failure_blocks_future_checkouts(
    tmp_path, monkeypatch
):
    """When a raw conn's physical close raises, the production
    _RawConnectionLease.close() method decrements the active count,
    records the path as unsafe, and re-raises."""
    import memory.db_pool as db_pool

    db_pool._pool = None
    db_pool._recovery_attempts_by_source.clear()
    db_pool._recovering_paths.clear()
    db_pool._maintenance_paths.clear()
    db_pool._unsafe_raw_paths.clear()
    db_pool._terminal_shutdown = False

    db_path = str(tmp_path / "unsafe-close.db")
    sqlite3.connect(db_path).execute("CREATE TABLE x (v INTEGER)")

    lease = db_pool._create_raw_connection(db_path)
    original_physical = lease._conn

    class _PhysicalCloseFails:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)
        def close(self):
            raise RuntimeError("simulated physical close failure")
        def __getattr__(self, name):
            return getattr(self._real, name)
        def __setattr__(self, name, value):
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(self._real, name, value)

    try:
        object.__setattr__(
            lease, "_conn", _PhysicalCloseFails(original_physical),
        )

        with pytest.raises(RuntimeError, match="simulated physical close failure"):
            lease.close()

        with db_pool._pool_condition:
            assert db_path not in db_pool._active_raw_connections, (
                "active raw count not decremented on close failure"
            )
            assert db_path in db_pool._unsafe_raw_paths

        with pytest.raises(RuntimeError, match="SQLite raw connection failed to close"):
            db_pool._checkout_global(db_path, validate=True, timeout=1.0)
    finally:
        # The physical conn was never actually closed. Close it
        # directly to avoid a leaked file handle.
        try:
            original_physical.close()
        except Exception:
            pass
        with db_pool._pool_condition:
            db_pool._unsafe_raw_paths.pop(db_path, None)
            db_pool._pool_condition.notify_all()
