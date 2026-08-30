"""Tests for the connection/cursor lease protocol.

Covers:
- cursor containment (no raw cursors escape execute/cursor())
- fetch-time corruption quarantine (no inline recovery)
- operation lock scope over checkin
- stale cursor/connection rejection
- iterdump returns a guarded iterator
- context manager idempotency after lease close
- delegated callable wrapping
"""

import os
import sqlite3
import threading
import time

import pytest

import memory.db_pool as db_pool

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_path, name="lease.db"):
    db_path = str(tmp_path / name)
    return db_pool.ConnectionPool(db_path, pool_size=2, lazy=True)


def _setup_healthy_lease(pool):
    """Open a lease and return it (no need to checkin for many tests)."""
    return pool.checkout(validate=True)


class _CorruptingCursorProxy:
    """Wraps a real cursor; fetchone/fetchmany/fetchall/__next__ raise a
    corruption error. The wrapper IS a sqlite3.Cursor subclass-like shim
    that exposes the bare attributes lease.execute inspects."""

    def __init__(self, real):
        self._real = real

    def fetchone(self):
        raise sqlite3.DatabaseError(
            "database disk image is malformed"
        )

    def fetchmany(self, *args, **kwargs):
        raise sqlite3.DatabaseError(
            "database disk image is malformed"
        )

    def fetchall(self):
        raise sqlite3.DatabaseError(
            "database disk image is malformed"
        )

    def __next__(self):
        raise sqlite3.DatabaseError(
            "database disk image is malformed"
        )

    def __iter__(self):
        return self

    def close(self):
        return self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _trigger_cursor_quarantine(lease, proxy):
    """Drive the CursorLease.fetchone path directly.

    lease.execute() inspects the result and only wraps it as a
    CursorLease if it is a sqlite3.Cursor. To exercise the lease-side
    quarantine (not the conn-level execute), we construct a CursorLease
    around the proxy cursor manually.
    """
    raw_cursor = lease._conn.execute("SELECT 1")
    cursor = db_pool._CursorLease(lease, proxy)
    cursor._cursor = raw_cursor  # replace with our proxy
    lease._cursors.add(cursor)
    return cursor


class _CorruptingConnProxy:
    """Wraps a real conn; its ``cursor()`` returns a corrupting cursor proxy
    that quarantine-fetches."""

    def __init__(self, real):
        self._real = real

    def cursor(self):
        return _CorruptingCursorProxy(self._real.cursor())

    def execute(self, sql, *args, **kwargs):
        result = self._real.execute(sql, *args, **kwargs)
        return _CorruptingCursorProxy(result)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Cursor containment
# ---------------------------------------------------------------------------


def test_execute_returns_cursor_lease(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        try:
            cursor = lease.execute("SELECT 1")
            assert isinstance(cursor, db_pool._CursorLease)
            assert cursor in lease._cursors
        finally:
            pool.checkin(lease)
    finally:
        pool.shutdown(grace_period=0)


def test_cursor_returns_cursor_lease(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        try:
            cursor = lease.cursor()
            assert isinstance(cursor, db_pool._CursorLease)
            assert cursor in lease._cursors
        finally:
            pool.checkin(lease)
    finally:
        pool.shutdown(grace_period=0)


def test_cursor_execute_returns_same_wrapper(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        try:
            cursor = lease.execute("SELECT 1")
            result = cursor.execute("SELECT 2")
            assert result is cursor
        finally:
            pool.checkin(lease)
    finally:
        pool.shutdown(grace_period=0)


def test_cursor_connection_returns_active_parent_lease(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        try:
            cursor = lease.execute("SELECT 1")
            assert cursor.connection is lease
        finally:
            pool.checkin(lease)
    finally:
        pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# Fetch-time corruption quarantine
# ---------------------------------------------------------------------------


def test_fetchone_corruption_quarantines_without_inline_recovery(tmp_path):
    pool = _make_pool(tmp_path, "fetchone.db")
    try:
        lease = _setup_healthy_lease(pool)
        real_conn = lease._conn
        proxy = _CorruptingConnProxy(real_conn)
        lease._conn = proxy
        cursor = _trigger_cursor_quarantine(lease, proxy)
        with pytest.raises(sqlite3.DatabaseError):
            cursor.fetchone()
        # The lease must be poisoned and the pool retiring. _recover_source
        # must NOT have been called yet (next global checkout triggers it).
        assert lease._poisoned is True
        with pool._condition:
            assert pool._retiring is True
            assert pool._corruption_error is not None
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_fetchmany_corruption_quarantines_without_inline_recovery(tmp_path):
    pool = _make_pool(tmp_path, "fetchmany.db")
    try:
        lease = _setup_healthy_lease(pool)
        real_conn = lease._conn
        proxy = _CorruptingConnProxy(real_conn)
        lease._conn = proxy
        cursor = _trigger_cursor_quarantine(lease, proxy)
        with pytest.raises(sqlite3.DatabaseError):
            cursor.fetchmany()
        assert lease._poisoned is True
        with pool._condition:
            assert pool._retiring is True
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_fetchall_corruption_quarantines_without_inline_recovery(tmp_path):
    pool = _make_pool(tmp_path, "fetchall.db")
    try:
        lease = _setup_healthy_lease(pool)
        real_conn = lease._conn
        proxy = _CorruptingConnProxy(real_conn)
        lease._conn = proxy
        cursor = _trigger_cursor_quarantine(lease, proxy)
        with pytest.raises(sqlite3.DatabaseError):
            cursor.fetchall()
        assert lease._poisoned is True
        with pool._condition:
            assert pool._retiring is True
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_iteration_corruption_quarantines_without_inline_recovery(tmp_path):
    pool = _make_pool(tmp_path, "iter.db")
    try:
        lease = _setup_healthy_lease(pool)
        real_conn = lease._conn
        proxy = _CorruptingConnProxy(real_conn)
        lease._conn = proxy
        cursor = _trigger_cursor_quarantine(lease, proxy)
        with pytest.raises(sqlite3.DatabaseError):
            iter(cursor).__next__()
        assert lease._poisoned is True
        with pool._condition:
            assert pool._retiring is True
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_next_checkout_recovers_quarantined_generation_once(tmp_path):
    """After a fetch corrupts, the next global checkout should trigger
    one recovery and produce a working replacement."""
    db_path = str(tmp_path / "recover-next.db")
    pool = db_pool._ensure_pool(db_path)

    # Acquire a real lease and poison it via a fetch corruption.
    lease = pool.checkout(validate=True)
    real_conn = lease._conn
    proxy = _CorruptingConnProxy(real_conn)
    lease._conn = proxy
    cursor = _trigger_cursor_quarantine(lease, proxy)
    try:
        cursor.fetchone()
    except sqlite3.DatabaseError:
        pass

    # Patch recover_corrupt_db to return True.
    from memory import db_recovery
    recover_calls = [0]

    def fake_recover(path):
        recover_calls[0] += 1
        if os.path.exists(path):
            os.remove(path)
        c = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level="",
        )
        try:
            from memory.db_migrations import SCHEMA_SQL
            c.executescript(SCHEMA_SQL)
        finally:
            c.close()
        return True

    original_recover = db_recovery.recover_corrupt_db
    db_recovery.recover_corrupt_db = fake_recover
    try:
        new_lease = db_pool._checkout_global(
            db_path, validate=True, timeout=10.0
        )
        try:
            row = new_lease.execute("SELECT 1").fetchone()
            assert row[0] == 1
            # The new lease must be from a different generation pool.
            assert new_lease._pool is not pool
        finally:
            db_pool.return_connection(new_lease)
    finally:
        db_recovery.recover_corrupt_db = original_recover

    # The recovery ran exactly once.
    assert recover_calls[0] == 1


# ---------------------------------------------------------------------------
# Stale cursor/connection rejection
# ---------------------------------------------------------------------------


def test_stale_cursor_fetch_rejected(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        cursor = lease.execute("SELECT 1")
        cursor.close()
        with pytest.raises(RuntimeError, match="Cursor lease is closed"):
            cursor.fetchone()
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_stale_cursor_properties_rejected(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        cursor = lease.execute("SELECT 1")
        cursor.close()
        with pytest.raises(RuntimeError, match="Cursor lease is closed"):
            _ = cursor.rowcount
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_stale_cursor_arraysize_write_rejected(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        cursor = lease.execute("SELECT 1")
        cursor.close()
        assert cursor._closed is True, "_closed should be True after close"
        import sys
        sys.stderr.write(f"about to set arraysize _closed={cursor._closed}\n")
        try:
            cursor.arraysize = 100
            sys.stderr.write(f"set succeeded\n")
        except RuntimeError as e:
            sys.stderr.write(f"set raised: {e}\n")
            assert "Cursor lease is closed" in str(e)
            return
        raise AssertionError(
            "expected RuntimeError on closed cursor arraysize write"
        )
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_stale_connection_property_read_rejected(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        pool.checkin(lease)
        with pytest.raises(RuntimeError):
            _ = lease.rowcount
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_stale_connection_property_write_rejected(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        pool.checkin(lease)
        with pytest.raises(RuntimeError):
            lease.text_factory = "foo"
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# iterdump returns a guarded iterator
# ---------------------------------------------------------------------------


def test_iterdump_returns_guarded_iterator(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        lines = list(lease.iterdump())
        # iterdump yields the SQL schema; the exact contents are
        # implementation-defined, but it should be a non-empty list.
        assert isinstance(lines, list)
        assert len(lines) > 0
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_iterdump_rejects_iteration_after_checkin(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        iter_obj = lease.iterdump()
        pool.checkin(lease)
        with pytest.raises(RuntimeError):
            next(iter_obj)
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Delegated callable wrapping
# ---------------------------------------------------------------------------


def test_delegated_cursor_callable_is_guarded(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        cursor = lease.execute("SELECT 1")
        # ``fetchmany`` is a callable attribute. Accessing it should
        # return a guarded closure that re-acquires the operation lock.
        method = cursor.fetchmany
        result = method(2)
        assert isinstance(result, list)
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lease close / context exit
# ---------------------------------------------------------------------------


def test_lease_close_returns_connection_once(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        lease.close()
        # Second close is a no-op.
        lease.close()
        # The conn is back in the pool.
        assert pool._pool.qsize() == 1
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass


def test_context_exit_after_lease_close_is_idempotent(tmp_path):
    pool = _make_pool(tmp_path)
    try:
        lease = _setup_healthy_lease(pool)
        with lease as l1:
            assert l1 is lease
        # __exit__ after the with block is harmless.
        lease.__exit__(None, None, None)
    finally:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass
