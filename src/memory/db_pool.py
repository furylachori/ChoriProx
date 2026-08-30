"""
Connection pool + transaction lifecycle for memory system.

Pure leaf module — does NOT import memory.db.

Design summary (see AGENTS.md for the full contract):

- Connections use deferred SQLite transactions (``isolation_level=""``).
  ``commit()`` persists, ``rollback()`` undoes, and checkin rolls back
  residual work the caller forgot to commit. ``transaction()``,
  ``immediate_transaction()``, and ``exclusive_transaction()`` own
  explicit commit boundaries. Plain ``connection()`` does not auto-commit.

- Every checkout returns a generation-bearing ``_ConnectionLease`` and a
  pool-bound ``_CursorLease`` for any cursor work. A stale lease cannot
  touch a later borrower, and a closed lease is rejected uniformly.

- Capacity accounting uses three explicit counters: ``_created`` (open
  physical conns), ``_creating`` (reserved open slots), and ``_inflight``
  (admitted checkouts not yet returned). Invariant:
  ``_created + _creating <= _pool_size``; ``_inflight`` only counts
  live admitted checkouts.

- Recovery ownership is exact-source based. Path gating independently
  prevents another pool, raw connection, or migration from opening the
  same file while a recovery is in progress. The selector picks the
  exact pool, transfers one admission to it, and returns — no fallible
  work happens after admission.

- Corruption is reported to the global coordinator via
  ``_RecoveryRequired`` (raised when an existing retiring source
  generation must be recovered) and via inline ``_quarantine_operation_corruption``
  on leases/cursors (sets poisoning + retiring marker, never invokes
  recovery under the lease operation lock). The next global checkout
  performs recovery and retries through the replacement generation.

- Path maintenance tokens (``_MaintenanceToken``) quiesce a path for
  schema migrations or startup repair. The migration ``init_db()``
  acquires a token, detaches the same-path runtime pool, and refuses
  new raw / runtime access until the migration completes.

- The module registers one fixed atexit fallback that places the manager
  in terminal state. Terminal shutdown waits for active recovery leaders
  and raw connections before closing the detached pool and checkpointing.
"""

import atexit
import logging
import os
import queue
import sqlite3
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory.db_pool")

# Utilization thresholds for the "Pool utilization high" warning.
_POOL_HIGH_UTILIZATION_THRESHOLD = 0.80
_POOL_HIGH_UTILIZATION_RESET_THRESHOLD = 0.70

# Default deadline (seconds) for bounded quiescence waits (F5). Mirrors the
# drain timeout used by _drain_for_maintenance so synchronous callers cannot
# wedge forever behind a stuck checkout/creator/raw connection.
_QUIESCENCE_DRAIN_TIMEOUT = 5.0

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "memory.db"


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_pool_condition = threading.Condition(_pool_lock)
_terminal_shutdown = False

# Currently active global pool, or None.
_pool: Optional["ConnectionPool"] = None

# Recovery ownership is exact-source based. Path indexing separately prevents
# another pool or raw connection from opening the same file during recovery.
_recovery_attempts_by_source: dict = {}
_recovering_paths: dict = {}
_active_recovery_leaders = 0

# Migration maintenance excludes pooled and unrelated raw access to a path.
_maintenance_paths: dict = {}

_active_raw_connections: dict = {}

# Paths whose most recent raw-conn physical close failed. The path
# gate fails fast on these to prevent recovery or checkpoint from
# touching a file while a live SQLite handle is still attached.
_unsafe_raw_paths: dict = {}  # canonical path → BaseException

# Set to True after the fixed atexit fallback has been registered.
_atexit_registered = False


# ---------------------------------------------------------------------------
# Exceptions and corruption classification
# ---------------------------------------------------------------------------


class DatabaseRecoveryError(sqlite3.DatabaseError):
    """The corrupt database could not be recovered safely."""


class _CorruptDatabase(sqlite3.DatabaseError):
    """Internal signal carrying the pool generation that observed corruption."""

    def __init__(self, db_path: str, original: sqlite3.DatabaseError):
        super().__init__(str(original))
        self.db_path = db_path
        self.original = original


def _is_corruption_error(exc: BaseException) -> bool:
    """Classify an exception as a SQLite corruption event."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False

    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        primary_code = code & 0xFF
        if primary_code in {
            sqlite3.SQLITE_CORRUPT,
            sqlite3.SQLITE_NOTADB,
        }:
            return True

    message = str(exc).lower()
    return (
        "database disk image is malformed" in message
        or "file is not a database" in message
        or "malformed" in message
        or "not a database" in message
    )


# Explicit connection-disposition states for checkin accounting (F11).
# Set at each publication side-effect so the BaseException cleanup handler
# acts on KNOWN state instead of inferring disposition from _checked_out.
_CHECKIN_DISPOSITION_IN_FLIGHT = "in_flight"  # still in _checked_out, counted in _created
_CHECKIN_DISPOSITION_RETURNED = "returned"     # requeued to idle; _created unchanged
_CHECKIN_DISPOSITION_CLOSED = "closed"         # removed from _created; phys close pending/done
# R2 #7 (F11 transitional gaps): intermediate publication states. Both are
# currently UNREACHABLE in W1-B's checkin order (which is del-checked_out
# THEN put_nowait — see _checkin at line 1269), but the cleanup handler
# must recognize them defensively so a future reorder (put_nowait THEN
# del-checked_out, or pre-del disposition writes) does NOT silently regress
# the F11 fix. Per review §C.6 #2, the constants and cleanup branches are
# INTENTIONAL even though unreachable today.
_CHECKIN_DISPOSITION_RETURNED_PENDING_PUBLISH = "returned_pending_publish"
# Set before put_nowait but after the disposition-flag write: conn is NOT
# in the idle queue yet; _created is unchanged; _checked_out may already be
# pruned (W1-B order) or not (reordered). Failed publication leaves the pool
# owning an unpublished conn, so cleanup must decrement _created and close the
# physical connection outside the condition, treating it as CLOSED.
_CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE = "published_pending_checkout_delete"
# Set after put_nowait but before del-checked_out: conn IS in the idle
# queue; _created unchanged; _checked_out still references the lease.
# Cleanup must NOT close the conn (it is live in the idle queue) but MUST
# prune _checked_out (otherwise the next checkout of this token hands out
# a stale lease).


# ---------------------------------------------------------------------------
# Path canonicalization
# ---------------------------------------------------------------------------


def _canonical_db_path(path: str) -> str:
    if path == ":memory:":
        return path
    return os.path.normcase(
        os.path.realpath(
            os.path.abspath(os.path.expanduser(path))
        )
    )


# ---------------------------------------------------------------------------
# Recovery attempt / maintenance token / recovery-required exception
# ---------------------------------------------------------------------------


class _RecoveryAttempt:
    __slots__ = ("source", "path", "state", "error")

    def __init__(self, source):
        self.source = source
        self.path = source._db_path
        self.state = "recovering"
        self.error = None


class _RecoveryRequired(RuntimeError):
    """A retiring corrupt generation must be recovered before checkout."""

    def __init__(self, source, error: sqlite3.DatabaseError):
        super().__init__("Database generation requires recovery")
        self.source = source
        self.error = error


class _MaintenanceToken:
    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path


# ---------------------------------------------------------------------------
# Path gate
# ---------------------------------------------------------------------------


def _wait_for_path_gate_locked(
    path: str,
    maintenance_token: Optional[_MaintenanceToken] = None,
) -> None:
    """Wait until path recovery/maintenance allows this caller.

    If the path is recorded as ``_unsafe_raw_paths``, this raises
    ``RuntimeError`` immediately: a previous raw-conn physical close
    failed, and the underlying SQLite handle may still be live.
    Subsequent recovery or checkpoint against this path is unsafe.
    """

    while True:
        if _terminal_shutdown:
            raise RuntimeError("Database pool manager is shutting down")

        unsafe = _unsafe_raw_paths.get(path)
        if unsafe is not None:
            raise RuntimeError(
                f"SQLite raw connection failed to close for {path}; "
                f"path is unsafe to access"
            ) from unsafe

        recovery = _recovering_paths.get(path)
        maintenance = _maintenance_paths.get(path)

        recovery_blocked = recovery is not None
        maintenance_blocked = (
            maintenance is not None
            and maintenance is not maintenance_token
        )

        if not recovery_blocked and not maintenance_blocked:
            return

        _pool_condition.wait()


# ---------------------------------------------------------------------------
# Cursor lease
# ---------------------------------------------------------------------------


class _CursorLease:
    """Lease-bound wrapper around a physical sqlite3.Cursor."""

    __slots__ = ("_lease", "_cursor", "_closed")

    def __init__(self, lease, cursor):
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "_closed", False)

    def _require_active_locked(self) -> None:
        if self._closed:
            raise RuntimeError("Cursor lease is closed")
        self._lease._require_active_locked()

    def execute(self, sql, *args, **kwargs):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                result = self._cursor.execute(sql, *args, **kwargs)
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise
        if result is self._cursor:
            return self
        if isinstance(result, sqlite3.Cursor):
            return self
        return result

    def executemany(self, sql, seq_of_params):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                result = self._cursor.executemany(sql, seq_of_params)
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise
        if result is self._cursor:
            return self
        if isinstance(result, sqlite3.Cursor):
            return self
        return result

    def executescript(self, sql_script):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                result = self._cursor.executescript(sql_script)
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise
        if result is self._cursor:
            return self
        if isinstance(result, sqlite3.Cursor):
            return self
        return result

    def fetchone(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                return self._cursor.fetchone()
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise

    def fetchmany(self, size=None):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                if size is None:
                    return self._cursor.fetchmany()
                return self._cursor.fetchmany(size)
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise

    def fetchall(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                return self._cursor.fetchall()
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise

    def close(self):
        with self._lease._operation_lock:
            if self._closed:
                return
            try:
                self._cursor.close()
            except Exception:
                pass
            self._closed = True
            try:
                self._lease._cursors.discard(self)
            except Exception:
                pass

    def setinputsizes(self, sizes):
        with self._lease._operation_lock:
            self._require_active_locked()
            return self._cursor.setinputsizes(sizes)

    def setoutputsize(self, size, column=None):
        with self._lease._operation_lock:
            self._require_active_locked()
            if column is None:
                return self._cursor.setoutputsize(size)
            return self._cursor.setoutputsize(size, column)

    @property
    def rowcount(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            return self._cursor.rowcount

    @property
    def lastrowid(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            return self._cursor.lastrowid

    @property
    def description(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            return self._cursor.description

    @property
    def arraysize(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        with self._lease._operation_lock:
            self._require_active_locked()
            self._cursor.arraysize = value

    @property
    def connection(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            return self._lease

    def __iter__(self):
        return self

    def __next__(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            try:
                row = self._cursor.fetchone()
            except sqlite3.DatabaseError as exc:
                self._lease._quarantine_operation_corruption(exc)
                raise
        if row is None:
            raise StopIteration
        return row

    def __enter__(self):
        with self._lease._operation_lock:
            self._require_active_locked()
            self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        with self._lease._operation_lock:
            if not self._closed:
                try:
                    self._cursor.__exit__(exc_type, exc, tb)
                except Exception:
                    pass
            self._closed = True
            try:
                self._lease._cursors.discard(self)
            except Exception:
                pass
        return False

    def __getattr__(self, name):
        with self._lease._operation_lock:
            self._require_active_locked()
            value = getattr(self._cursor, name)

        if not callable(value):
            return value

        def guarded(*args, **kwargs):
            with self._lease._operation_lock:
                self._require_active_locked()
                try:
                    result = value(*args, **kwargs)
                except sqlite3.DatabaseError as exc:
                    self._lease._quarantine_operation_corruption(exc)
                    raise

            if result is self._cursor:
                return self
            return result

        return guarded

    def __setattr__(self, name, value):
        if name in {"_lease", "_cursor", "_closed"}:
            object.__setattr__(self, name, value)
            return
        # If the class has a data descriptor (property) for this name,
        # let the default behavior invoke its setter.
        attr = getattr(type(self), name, None)
        if isinstance(attr, property):
            object.__setattr__(self, name, value)
            return
        with self._lease._operation_lock:
            setattr(self._cursor, name, value)


# ---------------------------------------------------------------------------
# Connection lease
# ---------------------------------------------------------------------------


class _ConnectionLease:
    """One checkout generation for a physical SQLite connection."""

    __slots__ = (
        "_pool",
        "_conn",
        "_token",
        "_released",
        "_poisoned",
        "_operation_lock",
        "_cursors",
    )

    def __init__(
        self,
        pool: "ConnectionPool",
        conn: sqlite3.Connection,
        token: int,
    ):
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_released", False)
        object.__setattr__(self, "_poisoned", False)
        object.__setattr__(self, "_operation_lock", threading.RLock())
        object.__setattr__(self, "_cursors", set())

    def _require_active_locked(self) -> sqlite3.Connection:
        if self._released or self._poisoned:
            raise RuntimeError("Connection lease is no longer active")
        with self._pool._condition:
            if (
                self._pool._shutting_down
                or self._pool._checked_out.get(self._token) is not self
            ):
                raise RuntimeError("Connection lease is no longer active")
            # Note: _retiring and _maintenance_draining do NOT reject
            # already-issued lease operations. Only an explicit
            # _shutting_down closes the door on existing leases.
        return self._conn

    def _close_all_cursors(self) -> None:
        for cursor in list(self._cursors):
            try:
                cursor.close()
            except Exception:
                pass
        self._cursors.clear()

    def _quarantine_operation_corruption(self, exc: sqlite3.DatabaseError) -> bool:
        """Poison this lease and quarantine its source generation.

        Caller holds self._operation_lock. This function MUST NOT invoke
        _recover_source or wait on module recovery state.
        """
        if not _is_corruption_error(exc):
            return False

        object.__setattr__(self, "_poisoned", True)

        with self._pool._condition:
            if self._pool._corruption_error is None:
                self._pool._corruption_error = exc
            self._pool._retiring = True
            self._pool._condition.notify_all()

        return True

    # -- Explicit operations -------------------------------------------------

    def execute(self, *args, **kwargs):
        with self._operation_lock:
            self._require_active_locked()
            try:
                result = self._conn.execute(*args, **kwargs)
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise
            # Wrap and register the cursor under the operation lock so
            # a concurrent checkin cannot miss the cursor.
            if isinstance(result, sqlite3.Cursor):
                cursor = _CursorLease(self, result)
                self._cursors.add(cursor)
                return cursor
            return result

    def executemany(self, *args, **kwargs):
        with self._operation_lock:
            self._require_active_locked()
            try:
                result = self._conn.executemany(*args, **kwargs)
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise
            if isinstance(result, sqlite3.Cursor):
                cursor = _CursorLease(self, result)
                self._cursors.add(cursor)
                return cursor
            return result

    def executescript(self, *args, **kwargs):
        with self._operation_lock:
            self._require_active_locked()
            try:
                result = self._conn.executescript(*args, **kwargs)
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise
            if isinstance(result, sqlite3.Cursor):
                cursor = _CursorLease(self, result)
                self._cursors.add(cursor)
                return cursor
            return result

    def cursor(self, *args, **kwargs):
        with self._operation_lock:
            self._require_active_locked()
            try:
                result = self._conn.cursor(*args, **kwargs)
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise
            cursor = _CursorLease(self, result)
            self._cursors.add(cursor)
            return cursor

    def commit(self):
        with self._operation_lock:
            self._require_active_locked()
            try:
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise

    def rollback(self):
        with self._operation_lock:
            self._require_active_locked()
            try:
                self._conn.rollback()
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise

    def close(self):
        with self._operation_lock:
            if self._released:
                return
        self._pool.checkin(self)

    def __enter__(self):
        with self._operation_lock:
            self._require_active_locked()
            self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        with self._operation_lock:
            self._require_active_locked()
            # Delegate to the physical context manager. Propagate commit/
            # rollback failures: the caller has already passed the
            # try-block, so a swallowed commit error would leave the
            # database in an uncommitted state and the caller would
            # believe the work succeeded.
            try:
                return self._conn.__exit__(exc_type, exc, tb)
            except sqlite3.DatabaseError as db_exc:
                self._quarantine_operation_corruption(db_exc)
                raise

    def __call__(self, *args, **kwargs):
        with self._operation_lock:
            self._require_active_locked()
            try:
                result = self._conn.execute(*args, **kwargs)
            except sqlite3.DatabaseError as exc:
                self._quarantine_operation_corruption(exc)
                raise
            if isinstance(result, sqlite3.Cursor):
                cursor = _CursorLease(self, result)
                self._cursors.add(cursor)
                return cursor
            return result

    def __setattr__(self, name, value):
        if name in {
            "_pool", "_conn", "_token", "_released", "_poisoned",
            "_operation_lock", "_cursors",
        }:
            object.__setattr__(self, name, value)
            return
        # If the class has a data descriptor (property) for this name,
        # let the default behavior invoke its setter.
        attr = getattr(type(self), name, None)
        if isinstance(attr, property):
            object.__setattr__(self, name, value)
            return
        with self._operation_lock:
            self._require_active_locked()
            setattr(self._conn, name, value)

    def __delattr__(self, name):
        if name in {
            "_pool", "_conn", "_token", "_released", "_poisoned",
            "_operation_lock", "_cursors",
        }:
            object.__delattr__(self, name)
            return
        with self._operation_lock:
            self._require_active_locked()
            delattr(self._conn, name)

    def __getattr__(self, name):
        if name == "blobopen":
            raise NotImplementedError(
                "blobopen is not supported on pooled connection leases"
            )
        with self._operation_lock:
            self._require_active_locked()
            value = getattr(self._conn, name)

        if name == "iterdump":
            return self._wrap_iterdump(value)

        if callable(value):
            return self._wrap_physical_method(name, value)

        return value

    def _wrap_physical_method(self, name, method):
        def _guarded(*args, **kwargs):
            with self._operation_lock:
                self._require_active_locked()
                try:
                    result = method(*args, **kwargs)
                except sqlite3.DatabaseError as exc:
                    self._quarantine_operation_corruption(exc)
                    raise
                if isinstance(result, sqlite3.Cursor):
                    cursor = _CursorLease(self, result)
                    self._cursors.add(cursor)
                    return cursor
                return result
        return _guarded

    def _wrap_iterdump(self, method):
        def guarded_iterdump(*args, **kwargs):
            with self._operation_lock:
                self._require_active_locked()
                iterator = method(*args, **kwargs)

            def iterate():
                while True:
                    with self._operation_lock:
                        self._require_active_locked()
                        try:
                            yield next(iterator)
                        except StopIteration:
                            return
                        except sqlite3.DatabaseError as exc:
                            self._quarantine_operation_corruption(exc)
                            raise

            return iterate()

        return guarded_iterdump


# ---------------------------------------------------------------------------
# Raw migration connection lease
# ---------------------------------------------------------------------------


class _RawConnectionLease:
    """Module-gated raw SQLite connection for the migration layer."""

    __slots__ = ("_conn", "_db_path", "_closed")

    def __init__(self, conn, db_path):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_closed", False)

    def close(self):
        if self._closed:
            return

        # Always publish bookkeeping. A failed physical close leaves
        # the OS handle in an unknown state, but the *manager* state
        # must still reflect that this raw conn is no longer in use.
        # However, the path is no longer safe to recover against or
        # checkpoint: the underlying handle may still be live. Track
        # unsafe paths so the gate fails fast and the WAL checkpoint
        # is skipped.
        physical_error: Optional[BaseException] = None
        try:
            self._conn.close()
        except BaseException as exc:
            physical_error = exc

        object.__setattr__(self, "_closed", True)

        with _pool_condition:
            count = _active_raw_connections.get(self._db_path, 0)
            if count <= 0:
                raise AssertionError("raw connection count underflow")
            if count == 1:
                del _active_raw_connections[self._db_path]
            else:
                _active_raw_connections[self._db_path] = count - 1
            if physical_error is not None:
                _unsafe_raw_paths[self._db_path] = physical_error
            _pool_condition.notify_all()

        if physical_error is not None:
            raise physical_error

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name in {"_conn", "_db_path", "_closed"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._conn, name, value)


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------


class ConnectionPool:
    """Bounded SQLite connection pool with checkout/checkin semantics."""

    def __init__(self, db_path: str, pool_size: int = 48, lazy: bool = True):
        if (
            not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or pool_size <= 0
        ):
            raise ValueError("pool_size must be a positive integer")
        self._db_path = _canonical_db_path(db_path)
        self._pool_size = pool_size
        self._pool = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._created = 0
        self._creating = 0
        self._inflight = 0
        self._retiring = False
        self._shutting_down = False
        # Maintenance drain is a separate state from corruption
        # retirement: it rejects new checkouts but does NOT force-fail
        # already-issued lease operations. Existing application work
        # completes naturally while the migration runs.
        self._maintenance_draining = False
        self._maintenance_drain_complete = threading.Event()
        self._maintenance_drain_complete.set()
        self._shutdown_complete = threading.Event()
        self._shutdown_error: Optional[BaseException] = None
        self._corruption_error: Optional[BaseException] = None
        self._checked_out: dict = {}  # lease_token -> _ConnectionLease
        self._checking_in: set = set()  # lease tokens currently mid-checkin
        self._next_lease_token = 0
        self._exhaustion_count = 0
        self._checkout_latency_log = deque(maxlen=10000)
        self._utilization_warning_active = False

        if not lazy:
            created: list = []
            try:
                for _ in range(pool_size):
                    conn = self._create_connection()
                    created.append(conn)
                    self._pool.put_nowait(conn)
                    with self._condition:
                        self._created += 1
            except Exception:
                for conn in created:
                    try:
                        conn.close()
                    except Exception:
                        pass
                with self._condition:
                    self._created = 0
                while True:
                    try:
                        self._pool.get_nowait()
                    except queue.Empty:
                        break
                raise

    # -----------------------------------------------------------------------
    # Counter helpers
    # -----------------------------------------------------------------------

    def _pool_stats_locked(self) -> tuple:
        created = self._created
        checked_out = len(self._checked_out)
        idle = self._pool.qsize()
        in_use = checked_out
        utilization = min(1.0, in_use / self._pool_size)
        return created, checked_out, idle, utilization

    def _maybe_reset_utilization_warning_locked(self) -> None:
        _, _, _, utilization = self._pool_stats_locked()
        if utilization <= _POOL_HIGH_UTILIZATION_RESET_THRESHOLD:
            self._utilization_warning_active = False

    def _new_lease_locked(
        self,
        conn: sqlite3.Connection,
    ) -> _ConnectionLease:
        self._next_lease_token += 1
        lease = _ConnectionLease(self, conn, self._next_lease_token)
        self._checked_out[lease._token] = lease
        return lease

    def _admit_checkout_locked(self) -> None:
        if self._retiring or self._shutting_down or self._maintenance_draining:
            raise RuntimeError("Connection pool is retiring")
        self._inflight += 1

    def _finish_checkout_locked(self) -> None:
        self._inflight -= 1
        if self._inflight < 0:
            raise AssertionError("negative checkout inflight count")
        self._condition.notify_all()

    # -----------------------------------------------------------------------
    # Physical connection creation
    # -----------------------------------------------------------------------

    def _create_connection(self) -> sqlite3.Connection:
        """Open a physical SQLite connection. Side-effect free."""
        conn = None
        try:
            if self._db_path == ":memory:":
                conn = sqlite3.connect(
                    "file::memory:?cache=shared",
                    uri=True,
                    check_same_thread=False,
                    isolation_level="",
                )
            else:
                conn = sqlite3.connect(
                    self._db_path,
                    check_same_thread=False,
                    isolation_level="",
                )

            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("SELECT 1")
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

            if _is_corruption_error(exc):
                raise _CorruptDatabase(self._db_path, exc) from exc
            raise
        except BaseException:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise

    # -----------------------------------------------------------------------
    # Checkout
    # -----------------------------------------------------------------------

    def checkout(
        self,
        timeout: float = 5.0,
        validate: bool = True,
        *,
        _admitted: bool = False,
    ) -> _ConnectionLease:
        t0 = time.monotonic()
        deadline = t0 + timeout
        admission_pending = False
        conn = None
        physical_owned = False
        creation_reserved = False

        if _admitted:
            # _select_pool_and_admit already incremented _inflight.
            admission_pending = True
        else:
            with self._condition:
                self._admit_checkout_locked()
                admission_pending = True

        try:
            with self._condition:
                while conn is None:
                    if self._retiring or self._shutting_down:
                        raise RuntimeError("Connection pool is retiring")

                    try:
                        conn = self._pool.get_nowait()
                        physical_owned = True
                        break
                    except queue.Empty:
                        pass

                    if self._created + self._creating < self._pool_size:
                        self._creating += 1
                        creation_reserved = True
                        break

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._exhaustion_count += 1
                        raise RuntimeError(
                            f"Connection pool exhausted (size={self._pool_size})"
                        )

                    self._condition.wait(timeout=remaining)

            if creation_reserved:
                try:
                    conn = self._create_connection()
                finally:
                    with self._condition:
                        self._creating -= 1
                        creation_reserved = False
                        if self._creating < 0:
                            raise AssertionError(
                                "negative connection creation count"
                            )
                        self._condition.notify_all()

                with self._condition:
                    if self._retiring or self._shutting_down:
                        try:
                            conn.close()
                        finally:
                            conn = None
                        raise RuntimeError("Connection pool is retiring")

                    self._created += 1
                    physical_owned = True
                    self._condition.notify_all()

            if validate:
                try:
                    conn.execute("SELECT 1")
                except sqlite3.DatabaseError as exc:
                    try:
                        conn.close()
                    finally:
                        conn = None

                    with self._condition:
                        if physical_owned:
                            self._created -= 1
                            physical_owned = False
                        if self._created < 0:
                            raise AssertionError(
                                "negative physical connection count"
                            )

                        if _is_corruption_error(exc):
                            raise _CorruptDatabase(self._db_path, exc) from exc

                        if self._retiring or self._shutting_down:
                            raise RuntimeError(
                                "Connection pool is retiring"
                            )
                        if self._created + self._creating >= self._pool_size:
                            raise RuntimeError(
                                "Connection replacement capacity invariant "
                                "violated"
                            )
                        self._creating += 1
                        creation_reserved = True

                    try:
                        conn = self._create_connection()
                    finally:
                        with self._condition:
                            self._creating -= 1
                            creation_reserved = False
                            if self._creating < 0:
                                raise AssertionError(
                                    "negative connection creation count"
                                )
                            self._condition.notify_all()

                    with self._condition:
                        if self._retiring or self._shutting_down:
                            try:
                                conn.close()
                            finally:
                                conn = None
                            raise RuntimeError(
                                "Connection pool is retiring"
                            )

                        self._created += 1
                        physical_owned = True
                        self._condition.notify_all()

            with self._condition:
                if self._retiring or self._shutting_down:
                    try:
                        conn.close()
                    finally:
                        conn = None
                    self._created -= 1
                    physical_owned = False
                    raise RuntimeError("Connection pool is retiring")

                lease = self._new_lease_locked(conn)
                physical_owned = False

            elapsed = time.monotonic() - t0
            self._checkout_latency_log.append(elapsed)
            if elapsed > 1.0:
                logger.warning("Slow pool checkout: %.2fs", elapsed)

            (
                should_warn,
                utilization,
                checked_out_count,
                created_count,
                idle_count,
            ) = self._record_checkout_and_maybe_warn(lease)

            if should_warn:
                logger.warning(
                    "Pool utilization high: %.0f%% "
                    "(checked_out=%d created=%d idle=%d capacity=%d)",
                    utilization * 100,
                    checked_out_count,
                    created_count,
                    idle_count,
                    self._pool_size,
                )

            return lease
        finally:
            if creation_reserved:
                with self._condition:
                    self._creating -= 1
                    if self._creating < 0:
                        raise AssertionError(
                            "negative connection creation count"
                        )
                    self._condition.notify_all()

            if physical_owned and conn is not None:
                try:
                    conn.close()
                finally:
                    with self._condition:
                        self._created -= 1
                        if self._created < 0:
                            raise AssertionError(
                                "negative physical connection count"
                            )
                        self._condition.notify_all()

            if admission_pending:
                with self._condition:
                    self._finish_checkout_locked()
                admission_pending = False

    def _record_checkout_and_maybe_warn(self, lease: _ConnectionLease) -> tuple:
        with self._condition:
            if (
                self._retiring
                or self._shutting_down
                or self._checked_out.get(lease._token) is not lease
            ):
                raise RuntimeError("Connection pool is retiring")

            (
                created_count,
                checked_out_count,
                idle_count,
                utilization,
            ) = self._pool_stats_locked()

            if (
                utilization > _POOL_HIGH_UTILIZATION_THRESHOLD
                and not self._utilization_warning_active
            ):
                self._utilization_warning_active = True
                should_warn = True
            else:
                should_warn = False

            return should_warn, utilization, checked_out_count, created_count, idle_count

    def checkin(self, lease: _ConnectionLease):
        if not isinstance(lease, _ConnectionLease):
            logger.warning("Ignoring checkin for non-lease connection")
            return

        token = lease._token

        with lease._operation_lock:
            with self._condition:
                if (
                    lease._pool is not self
                    or lease._released
                    or self._checked_out.get(token) is not lease
                    or token in self._checking_in
                ):
                    logger.warning(
                        "Duplicate or stale checkin detected for lease %s, ignoring",
                        token,
                    )
                    return

                self._checking_in.add(token)

            conn = lease._conn
            discard = lease._poisoned
            corruption = None
            # Conn disposition for this checkin. Starts IN_FLIGHT (tracked in
            # _checked_out and counted in _created). Transitioned to CLOSED or
            # RETURNED exactly at the matching publication side-effect so the
            # BaseException handler below can act on known state.
            _checkin_disposition = _CHECKIN_DISPOSITION_IN_FLIGHT

            try:
                try:
                    lease._close_all_cursors()
                except Exception:
                    logger.warning(
                        "Failed to close cursors on checkin, discarding connection",
                        exc_info=True,
                    )
                    discard = True

                if not discard:
                    try:
                        conn.rollback()
                    except sqlite3.DatabaseError as exc:
                        if _is_corruption_error(exc):
                            corruption = exc
                        else:
                            logger.warning(
                                "Rollback failed on checkin",
                                exc_info=True,
                            )
                        discard = True
                    except Exception:
                        logger.warning(
                            "Rollback failed on checkin",
                            exc_info=True,
                        )
                        discard = True

                if not discard:
                    try:
                        conn.row_factory = sqlite3.Row
                        conn.text_factory = str
                        conn.isolation_level = ""
                        conn.execute("PRAGMA busy_timeout = 5000")
                        conn.execute("PRAGMA wal_autocheckpoint = 1000")
                    except sqlite3.DatabaseError as exc:
                        if _is_corruption_error(exc):
                            corruption = exc
                        else:
                            logger.warning(
                                "Connection reset failed on checkin",
                                exc_info=True,
                            )
                        discard = True
                    except Exception:
                        logger.warning(
                            "Connection reset failed on checkin",
                            exc_info=True,
                        )
                        discard = True

                if corruption is not None:
                    object.__setattr__(lease, "_poisoned", True)
                    with self._condition:
                        if self._corruption_error is None:
                            self._corruption_error = corruption
                        self._retiring = True
                        self._condition.notify_all()

                with self._condition:
                    self._checking_in.discard(token)

                    if self._checked_out.get(token) is not lease:
                        object.__setattr__(lease, "_released", True)
                        return

                    del self._checked_out[token]
                    object.__setattr__(lease, "_released", True)

                    should_close = (
                        discard
                        or self._retiring
                        or self._shutting_down
                    )

                    if should_close:
                        self._created -= 1
                        _checkin_disposition = _CHECKIN_DISPOSITION_CLOSED
                        if self._created < 0:
                            raise AssertionError(
                                "negative physical connection count"
                            )
                    else:
                        # R2 #7 (F11 transitional gap, RETURNED_PENDING_PUBLISH):
                        # set the disposition flag BEFORE put_nowait so the
                        # BaseException cleanup handler sees RETURNED_PENDING_PUBLISH
                        # if an exception escapes between this point and the
                        # successful put_nowait that promotes it to RETURNED.
                        # Until publication succeeds, cleanup owns the unpublished
                        # conn and must close it and decrement _created as CLOSED.
                        _checkin_disposition = (
                            _CHECKIN_DISPOSITION_RETURNED_PENDING_PUBLISH
                        )
                        try:
                            self._pool.put_nowait(conn)
                            _checkin_disposition = (
                                _CHECKIN_DISPOSITION_RETURNED
                            )
                        except queue.Full:
                            self._created -= 1
                            _checkin_disposition = _CHECKIN_DISPOSITION_CLOSED
                            if self._created < 0:
                                raise AssertionError(
                                    "negative physical connection count"
                                )
                            should_close = True

                    self._maybe_reset_utilization_warning_locked()
                    self._condition.notify_all()

                if should_close:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except BaseException:
                # Act on KNOWN disposition (set at each publication
                # side-effect above) instead of inferring from _checked_out.
                # IN_FLIGHT: conn still counted and tracked — undo both.
                # CLOSED: _created already decremented here; phys close needed.
                # RETURNED: conn is live in the idle queue — do NOT close it.
                # RETURNED_PENDING_PUBLISH: conn was not published; decrement
                #   _created under the condition and physically close it outside
                #   the condition, treating the unpublished ownership as CLOSED.
                # PUBLISHED_PENDING_CHECKOUT_DELETE: conn IS in idle queue
                #   but _checked_out may still reference the lease — do NOT
                #   close (live in idle queue) but DO prune _checked_out so
                #   the lease doesn't leak. Per review §C.6 #2 these states
                #   are unreachable in the W1-B order but the cleanup
                #   handler MUST recognize them so a future reorder cannot
                #   silently regress the F11 fix.
                with self._condition:
                    self._checking_in.discard(token)
                    if (
                        _checkin_disposition
                        == _CHECKIN_DISPOSITION_IN_FLIGHT
                    ):
                        if self._checked_out.get(token) is lease:
                            del self._checked_out[token]
                        self._created -= 1
                        if self._created < 0:
                            raise AssertionError(
                                "negative physical connection count"
                            )
                    elif (
                        _checkin_disposition
                        == _CHECKIN_DISPOSITION_RETURNED_PENDING_PUBLISH
                    ):
                        self._created -= 1
                        if self._created < 0:
                            raise AssertionError(
                                "negative physical connection count"
                            )
                    elif (
                        _checkin_disposition
                        == _CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE
                    ):
                        if self._checked_out.get(token) is lease:
                            del self._checked_out[token]
                    object.__setattr__(lease, "_released", True)
                    self._condition.notify_all()

                if (
                    _checkin_disposition
                    != _CHECKIN_DISPOSITION_RETURNED
                    and _checkin_disposition
                    != _CHECKIN_DISPOSITION_PUBLISHED_PENDING_CHECKOUT_DELETE
                ):
                    try:
                        conn.close()
                    except Exception:
                        pass
                raise

    @contextmanager
    def connection(self, validate: bool = True):
        """Pool-level connection context."""
        lease = self.checkout(validate=validate)
        with lease._operation_lock:
            lease._require_active_locked()
            try:
                yield lease
            finally:
                self.checkin(lease)

    # -----------------------------------------------------------------------
    # Retirement (used by recovery) and shutdown
    # -----------------------------------------------------------------------

    def _drain_for_maintenance(self, timeout: float = 5.0) -> bool:
        """Reject new checkouts and wait for existing application work to
        return naturally. Returns True on success, False on timeout.

        The pool is NOT marked _retiring. Already-issued leases continue
        to operate normally. The caller (begin_path_maintenance) decides
        what to do if the drain times out.
        """
        self._maintenance_drain_complete.clear()
        try:
            with self._condition:
                self._maintenance_draining = True
                self._condition.notify_all()

            deadline = time.monotonic() + timeout
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._inflight == 0
                        and self._creating == 0
                        and not self._checked_out
                        and not self._checking_in
                    ),
                    timeout=max(0.0, timeout),
                )
                drained = (
                    self._inflight == 0
                    and self._creating == 0
                    and not self._checked_out
                    and not self._checking_in
                )
                if drained:
                    return True
                # Timeout. Drop the drain flag so the application can
                # continue using the pool; the migration caller will
                # abort the migration.
                self._maintenance_draining = False
                return False
        finally:
            self._maintenance_drain_complete.set()

    def _close_idle_connections(self) -> None:
        """Close all idle physical connections. Caller must have drained
        inflight/creating/checked_out. Does NOT close leases.
        """
        with self._condition:
            idle_connections = []
            while True:
                try:
                    idle_connections.append(self._pool.get_nowait())
                except queue.Empty:
                    break
            self._created -= len(idle_connections)
            if self._created < 0:
                self._created = 0

        for conn in idle_connections:
            try:
                conn.close()
            except Exception:
                pass

    def retire_for_recovery(
        self,
        timeout: float = _QUIESCENCE_DRAIN_TIMEOUT,
    ) -> None:
        with self._condition:
            self._retiring = True
            self._condition.notify_all()

            drained = self._condition.wait_for(
                lambda: self._inflight == 0 and self._creating == 0,
                timeout=timeout,
            )
            if not drained:
                raise RuntimeError(
                    "retire_for_recovery timed out waiting for "
                    "inflight/creating to drain"
                )

            idle_connections = []
            while True:
                try:
                    idle_connections.append(self._pool.get_nowait())
                except queue.Empty:
                    break

            self._created -= len(idle_connections)
            if self._created < 0:
                raise AssertionError("negative physical connection count")

            leases = list(self._checked_out.values())
            self._condition.notify_all()

        for conn in idle_connections:
            try:
                conn.close()
            except Exception:
                pass

        for lease in leases:
            claimed = False

            with lease._operation_lock:
                with self._condition:
                    if self._checked_out.get(lease._token) is lease:
                        del self._checked_out[lease._token]
                        self._checking_in.discard(lease._token)
                        self._created -= 1
                        if self._created < 0:
                            raise AssertionError(
                                "negative physical connection count"
                            )
                        object.__setattr__(lease, "_released", True)
                        object.__setattr__(lease, "_poisoned", True)
                        claimed = True
                        self._condition.notify_all()

                if not claimed:
                    continue

                try:
                    lease._close_all_cursors()
                except Exception:
                    pass
                try:
                    lease._conn.rollback()
                except Exception:
                    pass
                try:
                    lease._conn.close()
                except Exception:
                    pass

        with self._condition:
            if (
                self._created != 0
                or self._creating != 0
                or self._inflight != 0
                or self._checked_out
                or self._checking_in
                or self._pool.qsize() != 0
            ):
                raise AssertionError(
                    "pool retirement invariants not established"
                )

    def shutdown(self, grace_period: float = 1.0) -> None:
        with self._condition:
            if self._shutting_down:
                shutdown_owner = False
            else:
                self._shutting_down = True
                self._retiring = True
                self._utilization_warning_active = False
                self._condition.notify_all()
                shutdown_owner = True

        if not shutdown_owner:
            self._shutdown_complete.wait()
            if self._shutdown_error is not None:
                raise self._shutdown_error
            return

        try:
            if grace_period > 0:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._inflight == 0
                            and self._creating == 0
                            and not self._checked_out
                            and not self._checking_in
                        ),
                        timeout=grace_period,
                    )

            # Creators/admitted checkout frames own their counters and
            # must unwind. Bounded deadline so a stuck creator/inflight
            # cannot wedge the caller forever (F5).
            with self._condition:
                drained = self._condition.wait_for(
                    lambda: self._inflight == 0 and self._creating == 0,
                    timeout=_QUIESCENCE_DRAIN_TIMEOUT,
                )
                if not drained:
                    raise RuntimeError(
                        "shutdown timed out waiting for "
                        "inflight/creating to drain"
                    )

                idle_connections = []
                while True:
                    try:
                        idle_connections.append(self._pool.get_nowait())
                    except queue.Empty:
                        break

                self._created -= len(idle_connections)
                if self._created < 0:
                    raise AssertionError(
                        "negative physical connection count"
                    )

                leases = list(self._checked_out.values())

            for conn in idle_connections:
                try:
                    conn.close()
                except Exception:
                    pass

            for lease in leases:
                claimed = False

                with lease._operation_lock:
                    with self._condition:
                        if self._checked_out.get(lease._token) is lease:
                            del self._checked_out[lease._token]
                            self._checking_in.discard(lease._token)
                            self._created -= 1
                            if self._created < 0:
                                raise AssertionError(
                                    "negative physical connection count"
                                )
                            object.__setattr__(lease, "_released", True)
                            object.__setattr__(lease, "_poisoned", True)
                            claimed = True
                            self._condition.notify_all()

                    if not claimed:
                        continue

                    try:
                        lease._close_all_cursors()
                    except Exception:
                        pass
                    try:
                        lease._conn.rollback()
                    except Exception:
                        pass
                    try:
                        lease._conn.close()
                    except Exception:
                        pass

            with self._condition:
                if (
                    self._created != 0
                    or self._creating != 0
                    or self._inflight != 0
                    or self._checked_out
                    or self._checking_in
                    or self._pool.qsize() != 0
                ):
                    raise AssertionError(
                        "pool shutdown invariants not established"
                    )
        except BaseException as exc:
            self._shutdown_error = exc
            raise
        finally:
            self._shutdown_complete.set()


# ---------------------------------------------------------------------------
# Path maintenance
# ---------------------------------------------------------------------------


def begin_path_maintenance(
    db_path: str,
    drain_timeout: float = 5.0,
) -> _MaintenanceToken:
    """Acquires a maintenance token for ``db_path`` and detaches the
    active same-path pool. The token must be released exactly once
    via ``end_path_maintenance(token)`` or by exiting the
    ``path_maintenance`` context manager.

    Maintenance uses a *drain* state, not corruption retirement:
    - new checkouts are rejected
    - already-issued leases continue to operate normally
    - if the drain times out, the maintenance attempt is aborted and
      the token is released automatically (the pool is reattached
      identity-safely)

    On pool shutdown failure, the maintenance token is released so
    the path is not wedged forever. The physical close error is
    surfaced to the caller.
    """

    global _pool
    path = _canonical_db_path(db_path)
    token = _MaintenanceToken(path)
    detached = None

    with _pool_condition:
        if _terminal_shutdown:
            raise RuntimeError("Database pool manager is shutting down")

        while path in _recovering_paths or path in _maintenance_paths:
            _pool_condition.wait()
            if _terminal_shutdown:
                raise RuntimeError("Database pool manager is shutting down")

        _maintenance_paths[path] = token

        if _pool is not None and _pool._db_path == path:
            detached = _pool
            _pool = None

        _pool_condition.notify_all()

    if detached is not None:
        # Drain the existing application work, NOT a hard shutdown.
        # If the drain times out, abort the migration: the pool is
        # reattached identity-safely, and the token is released.
        try:
            drained = detached._drain_for_maintenance(drain_timeout)
        except BaseException:
            drained = False
        if not drained:
            detached._maintenance_draining = False
            with _pool_condition:
                # Identity-safe reattach: only set the global if no
                # newer pool for the same path has been published.
                if _pool is None:
                    _pool = detached
                else:
                    # Newer global exists. The detached generation
                    # is orphaned: shut it down terminal-style so it
                    # is not reused.
                    pass
                if _maintenance_paths.get(path) is token:
                    del _maintenance_paths[path]
                _pool_condition.notify_all()
            if _pool is not detached and detached is not None:
                try:
                    detached.shutdown(grace_period=0)
                except BaseException:
                    logger.warning(
                        "begin_path_maintenance: orphaned pool "
                        "shutdown failed: %s",
                        detached._db_path,
                    )
            raise RuntimeError(
                f"Maintenance drain timed out for {path}; "
                f"aborting migration"
            )

        # Drain succeeded. Close idle conns and shut the pool down.
        try:
            detached._close_idle_connections()
        finally:
            try:
                detached.shutdown(grace_period=0)
            except BaseException:
                # The pool shutdown itself failed. Release the
                # maintenance token so the path is not wedged forever;
                # future operations on the path will re-encounter the
                # underlying error.
                with _pool_condition:
                    if _maintenance_paths.get(path) is token:
                        del _maintenance_paths[path]
                    _pool_condition.notify_all()
                raise

    return token


def end_path_maintenance(token: _MaintenanceToken) -> None:
    with _pool_condition:
        if _maintenance_paths.get(token.path) is not token:
            raise RuntimeError("Invalid path maintenance token")
        del _maintenance_paths[token.path]
        _pool_condition.notify_all()


@contextmanager
def path_maintenance(db_path: str, drain_timeout: float = 5.0):
    """Context-managed wrapper around ``begin_path_maintenance`` /
    ``end_path_maintenance``.

    The token is released automatically when the ``with`` block exits,
    including on exceptions. The token is yielded so the body can pass
    it to ``_checkout_global(..., maintenance_token=token)`` for
    path-gate entry without race.
    """
    token = begin_path_maintenance(db_path, drain_timeout=drain_timeout)
    try:
        yield token
    finally:
        end_path_maintenance(token)


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------


def _select_pool_and_admit(
    db_path: str,
    *,
    maintenance_token: Optional[_MaintenanceToken] = None,
    gate_deadline: Optional[float] = None,
) -> ConnectionPool:
    """Select one exact pool and transfer one checkout admission.

    The path-gate wait honors ``gate_deadline`` (a monotonic time).
    If unset, a 5-second default is used so a held maintenance token
    cannot wedge a checkout indefinitely. The caller's checkout
    timeout is the actual upper bound; the gate deadline defaults
    to that timeout.
    """

    global _pool
    path = _canonical_db_path(db_path)
    if gate_deadline is None:
        gate_deadline = time.monotonic() + 5.0

    while True:
        old_pool = None
        candidate = None

        with _pool_condition:
            # Respect the path-gate wait deadline. The maintenance
            # gate is a long-held state; the caller can wait, but
            # we must not wait forever — the checkout's outer
            # timeout is the actual upper bound.
            while True:
                if _terminal_shutdown:
                    raise RuntimeError("Database pool manager is shutting down")

                unsafe = _unsafe_raw_paths.get(path)
                if unsafe is not None:
                    raise RuntimeError(
                        f"SQLite raw connection failed to close for {path}; "
                        f"path is unsafe to access"
                    ) from unsafe

                recovery = _recovering_paths.get(path)
                maintenance = _maintenance_paths.get(path)

                recovery_blocked = recovery is not None
                maintenance_blocked = (
                    maintenance is not None
                    and maintenance is not maintenance_token
                )

                if not recovery_blocked and not maintenance_blocked:
                    break

                if time.monotonic() >= gate_deadline:
                    raise RuntimeError(
                        f"Path gate deadline exceeded for {path}"
                    )

                _pool_condition.wait(timeout=0.1)

            current = _pool

            if (
                current is not None
                and current._db_path == path
                and current._retiring
                and current._corruption_error is not None
            ):
                raise _RecoveryRequired(
                    current,
                    current._corruption_error,
                )

            reusable = (
                current is not None
                and current._db_path == path
                and not current._retiring
                and not current._shutting_down
            )

            if reusable:
                # Nothing fallible may occur after admission.
                with current._condition:
                    current._admit_checkout_locked()
                return current

            old_pool = current
            candidate = ConnectionPool(path, pool_size=48, lazy=True)
            _pool = candidate

        # All cleanup happens before candidate admission.
        if old_pool is not None and old_pool is not candidate:
            old_pool.shutdown(grace_period=0)

        with _pool_condition:
            # Another thread may have changed the global while old_pool closed.
            if _pool is not candidate:
                candidate.shutdown(grace_period=0)
                continue

            if candidate._retiring or candidate._shutting_down:
                candidate.shutdown(grace_period=0)
                if _pool is candidate:
                    _pool = None
                continue

            # Nothing fallible may occur after this admission.
            with candidate._condition:
                candidate._admit_checkout_locked()
            return candidate


def _ensure_pool(
    db_path: str,
    *,
    maintenance_token: Optional[_MaintenanceToken] = None,
) -> ConnectionPool:
    global _pool
    path = _canonical_db_path(db_path)

    while True:
        old_pool = None

        with _pool_condition:
            _wait_for_path_gate_locked(path, maintenance_token)

            current = _pool
            if (
                current is not None
                and current._db_path == path
                and not current._retiring
                and not current._shutting_down
            ):
                return current

            if (
                current is not None
                and current._db_path == path
                and current._retiring
                and current._corruption_error is not None
            ):
                raise _RecoveryRequired(
                    current,
                    current._corruption_error,
                )

            old_pool = current
            candidate = ConnectionPool(path, pool_size=48, lazy=True)
            _pool = candidate

        if old_pool is not None and old_pool is not candidate:
            old_pool.shutdown(grace_period=0)

        with _pool_condition:
            if _pool is candidate:
                return candidate

        candidate.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# Module-level recovery coordinator
# ---------------------------------------------------------------------------


def _recover_source(
    source_pool: ConnectionPool,
    original_error: sqlite3.DatabaseError,
) -> None:
    """Coordinate corruption recovery for an exact source pool.

    If a newer same-path pool already replaced the global while this
    source was in flight, the source is "superseded" — it has no
    authority to rewrite the database file, because doing so would
    destroy the work of any reader on the newer generation. We retire
    the source pool (so it stops serving) but skip the file-repair and
    let the newer pool initiate its own recovery independently.
    """

    global _pool, _active_recovery_leaders

    path = source_pool._db_path
    leader = False
    superseded = False
    deadline = time.monotonic() + _QUIESCENCE_DRAIN_TIMEOUT

    with _pool_condition:
        if _terminal_shutdown:
            raise DatabaseRecoveryError(
                "Database pool manager is shutting down"
            ) from original_error

        existing = _recovery_attempts_by_source.get(source_pool)
        if existing is not None:
            # R2 #8 (F5 unbounded follower wait, existing-attempt half):
            # bound the wait so a leaked recovery leader (state stuck at
            # "recovering" forever) cannot wedge this follower. Mirrors the
            # pattern at the raw-drain wait below — deadline-based, raises
            # DatabaseRecoveryError on timeout. A spurious wake is fine: the
            # outer `while` re-checks `existing.state` before the next wait.
            while existing.state == "recovering":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DatabaseRecoveryError(
                        f"Timed out waiting for existing recovery attempt "
                        f"for {path} to complete"
                    ) from original_error
                _pool_condition.wait(timeout=min(remaining, 0.5))

            if existing.state == "failed":
                raise DatabaseRecoveryError(
                    f"Recovery failed for {path}"
                ) from existing.error
            return

        # Detect supersession: if the global is a different pool with
        # the same canonical path, this source is stale.
        current = _pool
        if (
            current is not source_pool
            and current is not None
            and current._db_path == path
        ):
            superseded = True
        else:
            # R2 #8 (F5 unbounded follower wait, path-in-_recovering_paths
            # half): bound the outer "path in _recovering_paths" loop too.
            # A leaked attempt whose path key never gets cleared would
            # otherwise wedge this code forever (no notification would
            # change the predicate). On timeout we raise DatabaseRecoveryError
            # — the caller (recovery coordinator) catches this and surfaces
            # it as a transient failure, which is the correct behavior.
            while path in _recovering_paths:
                if time.monotonic() >= deadline:
                    raise DatabaseRecoveryError(
                        f"Timed out waiting for path {path} to leave "
                        f"_recovering_paths"
                    ) from original_error
                path_attempt = _recovering_paths[path]
                # Inner wait: bound the per-attempt wait so a leaked
                # attempt stuck at "recovering" cannot wedge this loop
                # forever. A spurious wake is fine: the inner while
                # re-checks `path_attempt.state` before the next wait.
                while path_attempt.state == "recovering":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DatabaseRecoveryError(
                            f"Timed out waiting for path-attempt recovery "
                            f"for {path} to complete"
                        ) from original_error
                    _pool_condition.wait(timeout=min(remaining, 0.5))

                if _terminal_shutdown:
                    raise DatabaseRecoveryError(
                        "Database pool manager is shutting down"
                    ) from original_error

        if superseded:
            # Mark the attempt record so concurrent followers see it.
            attempt = _RecoveryAttempt(source_pool)
            _recovery_attempts_by_source[source_pool] = attempt
        else:
            # R2 #8 (F5 unbounded maintenance-path wait): bound the wait
            # so a maintenance window that never closes (e.g. a migration
            # that hangs) cannot wedge the recovery leader forever.
            # Mirrors the raw-drain pattern at line ~2040.
            while path in _maintenance_paths:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DatabaseRecoveryError(
                        f"Timed out waiting for maintenance window on "
                        f"path {path} to close"
                    ) from original_error
                _pool_condition.wait(timeout=min(remaining, 0.5))
                if _terminal_shutdown:
                    raise DatabaseRecoveryError(
                        "Database pool manager is shutting down"
                    ) from original_error

            attempt = _RecoveryAttempt(source_pool)
            _recovery_attempts_by_source[source_pool] = attempt
            _recovering_paths[path] = attempt
            _active_recovery_leaders += 1
            leader = True

    if superseded:
        try:
            source_pool.retire_for_recovery(
                timeout=max(0.0, deadline - time.monotonic())
            )
        except BaseException as exc:
            with _pool_condition:
                attempt.state = "failed"
                attempt.error = exc
                if _recovery_attempts_by_source.get(source_pool) is attempt:
                    _recovery_attempts_by_source.pop(source_pool, None)
                _pool_condition.notify_all()
            raise
        else:
            with _pool_condition:
                attempt.state = "succeeded"
                attempt.error = None
                if _recovery_attempts_by_source.get(source_pool) is attempt:
                    _recovery_attempts_by_source.pop(source_pool, None)
                _pool_condition.notify_all()
        return

    # Leader work: retire, drain raw, repair, validate, publish.
    replacement = None
    failure = None
    published = False

    try:
        source_pool.retire_for_recovery(
            timeout=max(0.0, deadline - time.monotonic())
        )

        # Bounded raw-drain (F5): a raw connection that never closes must not
        # wedge the recovery leader forever.
        with _pool_condition:
            while _active_raw_connections.get(path, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DatabaseRecoveryError(
                        f"Timed out waiting for raw connections to drain "
                        f"for {path}"
                    )
                _pool_condition.wait(timeout=min(remaining, 0.5))

            # F3: re-check unsafe paths after the drain wait. A maintenance
            # window may have failed to close a raw conn for this path while
            # we were waiting, which dropped the count to 0 AND marked the
            # path unsafe. We must NOT repair a path whose underlying handle
            # may still be live.
            if _unsafe_raw_paths.get(path) is not None:
                raise DatabaseRecoveryError(
                    f"Skipping recovery for {path}: raw connection close "
                    f"failed during drain, path is unsafe"
                )

        from memory.db_recovery import recover_corrupt_db

        # Re-validate supersession immediately before file repair.
        # Between the initial supersession check and this point, a
        # newer same-path pool may have been selected and published
        # as the global. If so, we must NOT touch the file because
        # the newer generation's readers are attached to the current
        # inode.
        with _pool_condition:
            current = _pool
            superseded_after_gate = (
                current is not source_pool
                and current is not None
                and current._db_path == path
            )
            if superseded_after_gate:
                # The finally block will set attempt.state = "succeeded"
                # and perform terminal bookkeeping (remove path gate,
                # decrement leader count, notify). Do not duplicate it.
                return

            # F3: re-check unsafe paths immediately before repair. The path
            # could have been marked unsafe concurrently since the drain wait.
            if _unsafe_raw_paths.get(path) is not None:
                raise DatabaseRecoveryError(
                    f"Skipping recovery for {path}: raw connection close "
                    f"failed, path is unsafe"
                )

        if not recover_corrupt_db(path):
            raise DatabaseRecoveryError(
                f"recover_corrupt_db returned False for {path}"
            )

        replacement = ConnectionPool(path, pool_size=48, lazy=True)
        test_conn = replacement._create_connection()

        with replacement._condition:
            replacement._pool.put_nowait(test_conn)
            replacement._created = 1

        with _pool_condition:
            if not _terminal_shutdown and _pool is source_pool:
                _pool = replacement
                published = True

        if not published:
            replacement.shutdown(grace_period=0)
            replacement = None
    except BaseException as exc:
        failure = exc
        if replacement is not None:
            try:
                replacement.shutdown(grace_period=0)
            except Exception:
                pass

        with _pool_condition:
            if _pool is source_pool:
                _pool = None
    finally:
        with _pool_condition:
            if failure is None:
                attempt.state = "succeeded"
                attempt.error = None
            else:
                attempt.state = "failed"
                attempt.error = failure

            if _recovering_paths.get(path) is attempt:
                del _recovering_paths[path]

            _active_recovery_leaders -= 1
            if _active_recovery_leaders < 0:
                raise AssertionError(
                    "negative active recovery leader count"
                )

            _pool_condition.notify_all()

    if failure is not None:
        if isinstance(failure, DatabaseRecoveryError):
            raise failure from original_error
        raise DatabaseRecoveryError(
            f"Recovery failed for {path}"
        ) from original_error


# ---------------------------------------------------------------------------
# Global checkout helper
# ---------------------------------------------------------------------------


def _checkout_global(
    db_path: str,
    validate: bool = True,
    timeout: float = 5.0,
    *,
    maintenance_token: Optional[_MaintenanceToken] = None,
) -> _ConnectionLease:
    deadline = time.monotonic() + timeout
    last_corruption = None
    gate_deadline = deadline  # path-gate wait must not exceed checkout timeout

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_corruption is not None:
                raise DatabaseRecoveryError(
                    "Database recovery did not produce a usable pool"
                ) from last_corruption
            raise RuntimeError("Connection pool checkout timed out")

        try:
            pool = _select_pool_and_admit(
                db_path,
                maintenance_token=maintenance_token,
                gate_deadline=gate_deadline,
            )
        except _RecoveryRequired as required:
            last_corruption = required.error
            _recover_source(required.source, required.error)
            continue

        try:
            return pool.checkout(
                timeout=remaining,
                validate=validate,
                _admitted=True,
            )
        except _CorruptDatabase as corrupt:
            last_corruption = corrupt.original
            _recover_source(pool, corrupt.original)
            continue
        except RuntimeError as exc:
            if "retiring" in str(exc):
                with pool._condition:
                    corruption = pool._corruption_error
                if corruption is not None:
                    last_corruption = corruption
                    _recover_source(pool, corruption)
                continue
            raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get_db_path():
    """Get database path from environment variable or default."""
    env_path = os.environ.get("PROXY_TEST_DB")
    if env_path:
        return env_path
    return str(DEFAULT_DB_PATH)


def _create_raw_connection(
    db_path: str,
    *,
    maintenance_token: Optional[_MaintenanceToken] = None,
) -> _RawConnectionLease:
    """Create a non-pooled migration connection."""
    path = _canonical_db_path(db_path)

    with _pool_condition:
        _wait_for_path_gate_locked(path, maintenance_token)
        _active_raw_connections[path] = (
            _active_raw_connections.get(path, 0) + 1
        )

    try:
        if path == ":memory:":
            conn = sqlite3.connect(
                "file::memory:?cache=shared",
                uri=True,
                check_same_thread=False,
                isolation_level="",
            )
        else:
            conn = sqlite3.connect(
                path,
                check_same_thread=False,
                isolation_level="",
            )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return _RawConnectionLease(conn, path)
    except BaseException:
        with _pool_condition:
            count = _active_raw_connections.get(path, 0)
            if count <= 1:
                _active_raw_connections.pop(path, None)
            else:
                _active_raw_connections[path] = count - 1
            _pool_condition.notify_all()
        raise


def get_connection(
    db_path=None,
    pooled: bool = True,
) -> _ConnectionLease:
    path = str(db_path or _get_db_path())
    return _checkout_global(path)


def return_connection(conn: _ConnectionLease):
    """Return a pooled connection to the pool that issued its lease."""
    if not isinstance(conn, _ConnectionLease):
        logger.warning("Ignoring return of non-lease connection")
        return
    conn._pool.checkin(conn)


@contextmanager
def connection(pooled: bool = True, validate: bool = True):
    """Pooled connection context with operation-lock-protected body."""
    lease = _checkout_global(str(_get_db_path()), validate=validate)
    with lease._operation_lock:
        lease._require_active_locked()
        try:
            yield lease
        finally:
            lease._pool.checkin(lease)


def close_all_connections():
    """Legacy compatibility: atomically detach + shutdown the global pool."""

    global _pool

    with _pool_condition:
        pool = _pool
        _pool = None
        _pool_condition.notify_all()

    if pool is not None:
        pool.shutdown(grace_period=0)


# ---------------------------------------------------------------------------
# Transaction rollback helper
# ---------------------------------------------------------------------------


def _rollback_preserving_primary(conn: _ConnectionLease) -> None:
    try:
        if (
            not conn._released
            and not conn._poisoned
            and conn.in_transaction
        ):
            conn.rollback()
    except BaseException:
        logger.warning(
            "Rollback failed while preserving primary transaction error",
            exc_info=True,
        )


def _checkin_preserving_primary(conn: _ConnectionLease) -> None:
    try:
        return_connection(conn)
    except BaseException:
        logger.warning(
            "Checkin failed while preserving primary transaction error",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Transaction contexts
# ---------------------------------------------------------------------------


@contextmanager
def transaction(db_path=None):
    conn = get_connection(db_path)
    primary = None
    try:
        conn.execute("BEGIN")
        yield conn
        conn.commit()
    except BaseException as exc:
        primary = exc
        _rollback_preserving_primary(conn)
        raise
    finally:
        if primary is not None:
            _checkin_preserving_primary(conn)
        else:
            # A checkin failure after a successful commit is a real
            # cleanup failure and must propagate. The body is already
            # done; there is no primary exception to preserve.
            return_connection(conn)


@contextmanager
def exclusive_transaction(db_path=None, pooled: bool = True):
    conn = get_connection(db_path, pooled=pooled)
    primary = None
    try:
        conn.execute("BEGIN EXCLUSIVE")
        yield conn
        conn.commit()
    except BaseException as exc:
        primary = exc
        _rollback_preserving_primary(conn)
        raise
    finally:
        if primary is not None:
            _checkin_preserving_primary(conn)
        else:
            return_connection(conn)


@contextmanager
def immediate_transaction(db_path=None, pooled: bool = True):
    conn = get_connection(db_path, pooled=pooled)
    primary = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException as exc:
        primary = exc
        _rollback_preserving_primary(conn)
        raise
    finally:
        if primary is not None:
            _checkin_preserving_primary(conn)
        else:
            return_connection(conn)


# ---------------------------------------------------------------------------
# Terminal shutdown
# ---------------------------------------------------------------------------


@dataclass
class _TerminalDetach:
    """Structured result of ``_begin_terminal_shutdown``.

    Callers use ``quiesced`` to decide whether the WAL checkpoint is
    safe. Wall-clock inference is unreliable: a path may quiesce
    exactly at the deadline, or a timeout may report clean state on
    return.
    """

    pool: Optional[ConnectionPool]
    quiesced: bool
    outstanding_paths: tuple = ()
    unsafe_paths: tuple = ()


def _begin_terminal_shutdown(
    deadline: Optional[float] = None,
) -> _TerminalDetach:
    """Atomically place the manager in terminal state and detach the pool.

    Waits for active recovery leaders, raw connections, and maintenance
    paths to drain, with a hard deadline. The returned ``_TerminalDetach``
    reports whether quiescence actually completed and which paths were
    outstanding or unsafe. The caller uses ``quiesced`` to decide
    whether a WAL checkpoint is safe — wall-clock inference is not
    reliable.
    """

    global _pool, _terminal_shutdown

    detached: Optional[ConnectionPool] = None
    outstanding_paths: list = []
    unsafe_paths: list = []
    quiesced = False

    with _pool_condition:
        if _terminal_shutdown:
            return _TerminalDetach(
                pool=None, quiesced=False,
                outstanding_paths=(),
                unsafe_paths=tuple(_unsafe_raw_paths.keys()),
            )

        _terminal_shutdown = True
        detached = _pool
        _pool = None
        _pool_condition.notify_all()

        deadline_passed = False
        while (
            _active_recovery_leaders > 0
            or _active_raw_connections
            or _maintenance_paths
        ):
            if deadline is not None:
                if time.monotonic() >= deadline:
                    deadline_passed = True
                    break
                wait_timeout = max(
                    0.0,
                    min(0.5, deadline - time.monotonic()),
                )
                if wait_timeout == 0.0:
                    deadline_passed = True
                    break
            else:
                wait_timeout = 0.5
            _pool_condition.wait(timeout=wait_timeout)

        if deadline_passed:
            outstanding_paths = list(
                set(
                    list(_recovering_paths.keys())
                    + list(_active_raw_connections.keys())
                    + list(_maintenance_paths.keys())
                )
            )
            logger.warning(
                "Terminal shutdown deadline exceeded; "
                "outstanding: %s",
                sorted(outstanding_paths),
            )
        else:
            quiesced = True

        unsafe_paths = list(_unsafe_raw_paths.keys())

    if outstanding_paths:
        logger.warning(
            "Terminal shutdown proceeding with unquiesced paths: %s",
            sorted(outstanding_paths),
        )
    if unsafe_paths:
        logger.warning(
            "Terminal shutdown proceeding with unsafe paths: %s",
            sorted(unsafe_paths),
        )

    return _TerminalDetach(
        pool=detached,
        quiesced=quiesced,
        outstanding_paths=tuple(outstanding_paths),
        unsafe_paths=tuple(unsafe_paths),
    )


@dataclass
class _PoolShutdownResult:
    """Result of ``_shutdown_pool_until``.

    The completion flag and the captured error are set by the
    worker thread that runs ``pool.shutdown`` off the caller's
    stack. The caller uses ``completed`` to decide whether the WAL
    checkpoint is safe to run — wall-clock inference after the
    fact is unreliable.
    """

    completed: bool
    error: Optional[BaseException] = None


def _shutdown_pool_until(
    pool: ConnectionPool,
    deadline: float,
) -> _PoolShutdownResult:
    """Run safe pool shutdown without allowing the caller to exceed
    the shared deadline.

    The worker thread owns the normal shutdown path and waits
    safely for lease operation locks. The caller remains bounded.
    There is no concurrent physical close. If the worker does not
    signal completion before the deadline, the result reports
    ``completed=False`` and the caller must skip post-shutdown
    work (e.g. WAL checkpoint).
    """

    done = threading.Event()
    result = _PoolShutdownResult(completed=False)

    def run_shutdown():
        try:
            pool.shutdown(grace_period=0)
        except BaseException as exc:
            result.error = exc
        finally:
            result.completed = True
            done.set()

    thread = threading.Thread(
        target=run_shutdown,
        daemon=True,
        name="sqlite-pool-terminal-shutdown",
    )
    thread.start()

    remaining = max(0.0, deadline - time.monotonic())
    if not done.wait(timeout=remaining):
        return _PoolShutdownResult(completed=False)

    return result


def _shutdown_global_pool(timeout: float = 10.0) -> None:
    """Fixed atexit fallback. Detaches the pool, quiesces, then checkpoints.

    Honors a single monotonic deadline so process exit cannot hang on
    maintenance or recovery cleanup. The WAL checkpoint is performed
    only when ALL of the following hold:
    - the manager quiesced
    - the pool shutdown completed
    - the pool shutdown did not raise
    - the path is not unsafe
    """

    deadline = time.monotonic() + timeout
    result = _begin_terminal_shutdown(deadline=deadline)
    if result.pool is None:
        return

    path = result.pool._db_path
    pool_shutdown = _shutdown_pool_until(result.pool, deadline)

    if not pool_shutdown.completed:
        logger.warning(
            "db_pool: pool shutdown did not complete before deadline "
            "for %s; skipping WAL checkpoint",
            path,
        )
        return

    if pool_shutdown.error is not None:
        logger.warning(
            "db_pool shutdown raised during terminal cleanup: %s",
            pool_shutdown.error,
        )
        return

    if not result.quiesced:
        logger.warning(
            "db_pool: skipping WAL checkpoint for %s — "
            "terminal shutdown did not quiesce",
            path,
        )
        return

    if path in result.unsafe_paths:
        logger.warning(
            "db_pool: skipping WAL checkpoint for %s — "
            "path is unsafe (raw-conn close previously failed)",
            path,
        )
        return

    if path != ":memory:" and os.path.exists(path):
        try:
            conn = sqlite3.connect(path, check_same_thread=False)
        except Exception as exc:
            logger.warning(
                "db_pool: failed to open %s for checkpoint: %s",
                path,
                exc,
            )
            return
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as exc:
            logger.warning(
                "db_pool: WAL checkpoint failed for %s: %s",
                path,
                exc,
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass


def _register_atexit_fallback() -> None:
    global _atexit_registered
    with _pool_lock:
        if _atexit_registered:
            return
        atexit.register(_shutdown_global_pool)
        _atexit_registered = True


_register_atexit_fallback()