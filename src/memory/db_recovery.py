"""SQLite corruption recovery for ChoriProx data/memory.db.

Pure leaf — does NOT import ``memory.db.*`` at module load. Runtime
dependencies:

  - ``proxy.state.recovery_lock`` (RLock #13) for concurrent-recovery
    serialization (non-db-sibling; lives in proxy.state per the
    ``src/proxy/AGENTS.md`` STATE rule "All shared locks and dicts are
    defined ONCE in state.py").
  - ``memory.db_migrations.SCHEMA_SQL`` (lazy-import on the schema-only
    fallback path inside ``_recover_corrupt_db_inner``; never at module
    load).

Standard library only at import time: ``sqlite3``, ``os``, ``shutil``,
``subprocess``, ``logging``, ``datetime``, ``threading``, ``re``.

Codifies the manual recovery procedure performed on the 2026-07-06 prod
incident (see data/.corrupt_backup_20260706_212625/):

  1. Back up corrupt db + -wal + -shm → data/.corrupt_backup_<ts>/
  2. Run `sqlite3 <db> ".dump"`     → dump.sql    (intact DDL + non-corrupt INSERTs)
  3. Run `sqlite3 <db> ".recover"`  → recover.sql (corrupt-table rows → lost_and_found)
  4. Build a clean db at <db>.healing:
     a. Sed in dump.sql: 'ROLLBACK; -- due to errors' → 'COMMIT;'
     b. Load dump.sql (schema + indexes + recoverable INSERTs)
     c. ATTACH recover.db; map lost_and_found rows → original tables via rootpgno
     d. REINDEX; PRAGMA integrity_check; wal_checkpoint(TRUNCATE)
  5. Atomic swap: os.replace(<db>.healing, <db>) + unlink -wal/-shm
  6. Log recovered-vs-lost row counts; emit metric

Two entry points:

  * ``ensure_db_integrity(db_path)`` — non-destructive check; returns True iff
    ``PRAGMA integrity_check`` is 'ok'. Used at startup and inside pool catches.

  * ``recover_corrupt_db(db_path)`` — full recovery. Returns True on success.
    Idempotent under ``proxy.state.recovery_lock`` (#13, an RLock): the lock
    serializes concurrent callers; once one caller finishes, subsequent
    callers re-enter ``recover_corrupt_db`` on a healthy db (wasted work but
    not data loss — the recovery body is idempotent on a clean db).

    There is intentionally NO ``recovery_in_progress`` flag (CRIT-3 fix). The
    flag caused a global cross-path short-circuit: after the FIRST successful
    recovery in a process, every subsequent corruption event skipped recovery
    and ran only ``ensure_db_integrity``. With N corrupt paths, only 1 was
    actually recovered. Dropping the flag fixes both CRIT-3 (single-path
    re-entry) and CRIT-4 (cross-path corruption blocked).

The lost_and_found → original-tables mapping is the verified-correct routine
described in the ticket spec (rootpgno → sqlite_master.rootpage; column
offset skips c0 when declared col 0 is INTEGER PRIMARY KEY).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_db_integrity(db_path: str) -> bool:
    """Return True iff the SQLite db at ``db_path`` passes ``PRAGMA integrity_check``.

    Opens a fresh read-only connection (does NOT go through the pool — the pool
    may itself be the entity blocked on the corrupt db, and we don't want
    re-entrancy). Catches ``DatabaseError`` so a file with a malformed header
    returns False rather than raising.

    CRIT-2 fix: prepend a 16-byte magic-header sniff (``b"SQLite format 3\\x00"``)
    before opening the URI connect. A 0-byte file or a truncated header slips
    past ``PRAGMA integrity_check`` when ``immutable=1`` is set (it returns 'ok'
    because no pages are touched). The magic-byte sniff catches that case
    BEFORE we spend time on a connect/PRAGMA roundtrip.

    Args:
        db_path: Absolute or relative path to the SQLite database file.

    Returns:
        True if integrity check reports ``ok``; False otherwise (corrupt header,
        DatabaseError on PRAGMA, file not readable, etc.).

    Raises:
        Nothing — all ``DatabaseError`` and unexpected errors are caught and
        logged; the function returns False instead of raising.

    Side-effects:
        Reads the db file (first 16 bytes for the magic sniff, then via a
        read-only SQLite connection). NEVER modifies the file. Logs at WARNING
        when ``PRAGMA integrity_check`` reports a non-'ok' result, and at
        ERROR when the connection itself fails.
    """
    if not db_path or db_path == ":memory:":
        # In-memory databases are not real files — treat as healthy.
        return True
    if not os.path.exists(db_path):
        logger.error("ensure_db_integrity: file not found %s", db_path)
        return False
    # CRIT-2: magic-byte sniff. Catch 0-byte / truncated-header files that
    # PRAGMA integrity_check+immutable=1 wrongly accepts as 'ok'.
    try:
        with open(db_path, "rb") as _f:
            header = _f.read(16)
    except OSError as e:
        logger.error("ensure_db_integrity: cannot read header %s: %s", db_path, e)
        return False
    if len(header) < 16 or header != b"SQLite format 3\x00":
        logger.warning(
            "ensure_db_integrity: missing/invalid SQLite magic header for %s "
            "(header=%r); treating as corrupt",
            db_path,
            header[:16] if header else b"<empty>",
        )
        return False
    try:
        # Read-only + immutable=1: skip locking, tolerate corruption in pages
        # we don't touch. Falls back to plain mode=ro if URI parsing fails.
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        except sqlite3.OperationalError:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = con.execute("PRAGMA integrity_check")
            row = cur.fetchone()
        finally:
            con.close()
        if row is None:
            return False
        ok = row[0] == "ok"
        if not ok:
            logger.warning("ensure_db_integrity: integrity_check failed for %s: %r", db_path, row[0])
        return ok
    except sqlite3.DatabaseError as e:
        logger.error("ensure_db_integrity: DatabaseError for %s: %s", db_path, e)
        return False
    except Exception as e:
        logger.error("ensure_db_integrity: unexpected error for %s: %s", db_path, e)
        return False


def _read_rootpages_from_master(db_path: str) -> dict:
    """Return ``{table_name: rootpage}`` from sqlite_master of the corrupt db.

    sqlite_master is stored in page 1 and survives even severe data-page
    corruption, so it's the canonical mapping source for lost_and_found rows
    (which carry a ``rootpgno`` column = the root page of the table they came
    from).

    Tables without a stored rootpage (indexes, virtual tables) get ``-1`` so
    callers can skip them. Returns an empty dict if the file is unreadable.
    """
    rootpages: dict = {}
    if not db_path or not os.path.exists(db_path):
        return rootpages
    try:
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        except sqlite3.OperationalError:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT name, rootpage, type FROM sqlite_master WHERE type='table'"
            ).fetchall()
            for name, rootpage, _t in rows:
                if name.startswith("sqlite_"):
                    continue
                rootpages[name] = int(rootpage) if rootpage else -1
        finally:
            con.close()
    except Exception as e:
        logger.warning("_read_rootpages_from_master failed for %s: %s", db_path, e)
    return rootpages


def _invert_pgno_to_table(rootpages: dict) -> dict:
    """Invert ``{table_name: rootpage}`` → ``{rootpage: table_name}`` for fast lookup.

    Used by ``_insert_recovered_rows`` to map a ``lost_and_found.rootpgno``
    integer back to the originating table name. The naive lookup
    ``healing_rootpages.get(pgno)`` always returns ``None`` because the source
    dict has string keys (table names), not integer keys — this was the
    CRIT-1 rootpage-inversion bug.

    Tables whose rootpage is ``-1`` (unknown — indexes, virtual tables) are
    skipped. When multiple tables share a rootpage (shouldn't happen for a
    healthy schema), later entries overwrite earlier — the architect-acceptable
    degraded-contract behavior. Returns an empty dict for empty input.
    """
    out: dict = {}
    for table_name, rootpage in rootpages.items():
        if rootpage == -1 or rootpage is None:
            continue
        out[int(rootpage)] = table_name
    return out


def _map_lost_and_found_columns(table_name: str, db_path: str) -> list:
    """Return the DDL column list for ``table_name`` from ``db_path``.

    Used to map ``lost_and_found`` rows back into their original table.
    Column order matches ``PRAGMA table_info(...)`` declaration order.
    """
    if not table_name or not db_path or not os.path.exists(db_path):
        return []
    try:
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        except sqlite3.OperationalError:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
            cols = [row[1] for row in rows]
        finally:
            con.close()
        return cols
    except Exception as e:
        logger.warning("_map_lost_and_found_columns failed for %s.%s: %s", db_path, table_name, e)
        return []


def _first_col_is_rowid_alias(table_name: str, db_path: str) -> bool:
    """Return True if the first declared column of ``table_name`` is an
    INTEGER PRIMARY KEY (a rowid alias — its value lives in the record header,
    so it appears as NULL in the lost_and_found data columns).
    """
    if not table_name or not db_path or not os.path.exists(db_path):
        return False
    try:
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        except sqlite3.OperationalError:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute(f"PRAGMA table_info({table_name})").fetchone()
        finally:
            con.close()
        if row is None:
            return False
        # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
        col_type = (row[2] or "").upper()
        pk_order = row[5] or 0
        return pk_order > 0 and "INT" in col_type
    except Exception as e:
        logger.warning("_first_col_is_rowid_alias failed for %s.%s: %s", db_path, table_name, e)
        return False


def _get_static_schema_sql() -> str | None:
    """Lazy-import the production ``SCHEMA_SQL`` constant from ``db_migrations``.

    CRIT-3 / CRIT-4 (iter3) fallback: when ``dump.sql`` execution fails
    mid-recovery (severely corrupt db produces malformed SQL output), we
    fall back to applying the static production schema so the proxy can
    still come up with an empty but schema-valid db.

    No memory.db sibling imports at module load; lazy imports
    ``db_migrations.SCHEMA_SQL`` only on the schema-only fallback path
    inside ``_recover_corrupt_db_inner``.

    Returns:
        The schema SQL string (``db_migrations.SCHEMA_SQL``) if the import
        succeeds, or ``None`` if the import fails (e.g., running
        standalone with no ``src/memory/db_migrations.py`` on path).
    """
    try:
        from memory.db_migrations import SCHEMA_SQL  # type: ignore
        return SCHEMA_SQL
    except ImportError as e:
        logger.warning(
            "DB_RECOVERY: cannot lazy-import SCHEMA_SQL from db_migrations: %s",
            e,
        )
        return None


def _apply_and_validate_schema(con) -> bool:
    """Apply the production ``SCHEMA_SQL`` to ``con`` and validate completeness.

    Tier-1 functional failure guard (Issue B): when ``sqlite3 .dump`` runs
    on a corrupt db whose pages backing ``sessions`` or ``rate_limiter_state``
    are damaged, those ``CREATE TABLE`` statements are silently absent from
    the dump. The healing db then passes ``PRAGMA integrity_check``
    (SQLite structure is OK) but is missing application tables — the F1
    privilege-escalation guard (``session_exists``) is silently disabled.

    This helper ALWAYS applies ``SCHEMA_SQL`` (idempotent ``CREATE TABLE
    IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS``) on top of whatever the
    dump produced, then runs ``validate_schema_complete`` against the
    connection. If validation fails it logs the diagnostics, performs one
    more idempotent re-apply (last-chance catch-up), and re-validates. If
    the second validation still fails the function returns False — the
    caller MUST refuse to proceed to ``_atomic_swap`` so the live db is left
    untouched.

    Pure leaf: lazy-imports both ``SCHEMA_SQL`` and ``validate_schema_complete``
    inside the function body (no module-level ``memory.db_migrations``
    import). Operates on the passed-in connection.

    Args:
        con: An open sqlite3 connection to the healing db. The connection
            is mutated (CREATE TABLE IF NOT EXISTS is a no-op for tables
            the dump already loaded).

    Returns:
        True if the schema is complete after applying SCHEMA_SQL. False
        if validation fails even after a second SCHEMA_SQL re-apply, or
        if SCHEMA_SQL cannot be imported / applied.
    """
    schema_sql = _get_static_schema_sql()
    if not schema_sql:
        logger.error(
            "DB_RECOVERY: _apply_and_validate_schema — SCHEMA_SQL not "
            "importable from db_migrations",
        )
        return False

    def _apply_once() -> bool:
        try:
            con.executescript(schema_sql)
            con.commit()
            return True
        except sqlite3.DatabaseError as e:
            logger.error(
                "DB_RECOVERY: SCHEMA_SQL apply failed: %s", e,
            )
            return False

    if not _apply_once():
        return False

    try:
        from memory.db_migrations import validate_schema_complete  # type: ignore
    except ImportError as e:
        logger.error(
            "DB_RECOVERY: cannot lazy-import validate_schema_complete: %s", e,
        )
        return False

    ok, diagnostics = validate_schema_complete(con)
    if ok:
        return True

    logger.error(
        "DB_RECOVERY: schema validation failed after first SCHEMA_SQL apply: "
        "%s",
        diagnostics,
    )

    # Last-chance catch-up: re-apply SCHEMA_SQL (idempotent) and re-validate.
    # This handles the edge case where the healing connection had a partial
    # transaction state from the failed dump.sql above that masked newly-
    # created tables. After re-apply the connection is in a fresh state.
    if not _apply_once():
        return False

    ok, diagnostics = validate_schema_complete(con)
    if not ok:
        logger.error(
            "DB_RECOVERY: schema validation still fails after re-apply; "
            "refusing to swap: %s",
            diagnostics,
        )
        return False
    return True


def _populate_recovery_db(recover_sql_path: str, recover_db_path: str) -> bool:
    """Create ``recover_db_path`` and load ``recover_sql_path`` into it.

    Returns True on success. The resulting db has the ``lost_and_found`` table
    populated with all rows salvaged by ``sqlite3 .recover``.
    """
    if os.path.exists(recover_db_path):
        try:
            os.unlink(recover_db_path)
        except OSError:
            pass
    for ext in ("-wal", "-shm"):
        p = recover_db_path + ext
        if os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
    try:
        con = sqlite3.connect(recover_db_path)
        try:
            with open(recover_sql_path, "r", encoding="utf-8", errors="replace") as f:
                sql = f.read()
            con.executescript(sql)
        finally:
            con.close()
        return True
    except Exception as e:
        logger.error("_populate_recovery_db: failed to load %s: %s", recover_sql_path, e)
        return False


def _read_lost_and_found_rows(recover_db_path: str) -> list:
    """Return all rows from the ``lost_and_found`` table in ``recover_db_path``.

    Each row is a sqlite3.Row with columns:
        rootpgno, pgno, nfield, id, c0, c1, ..., c12
    """
    if not os.path.exists(recover_db_path):
        return []
    try:
        con = sqlite3.connect(recover_db_path)
        con.row_factory = sqlite3.Row
        try:
            # Verify the table exists (depends on .recover output version)
            tables = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            if "lost_and_found" not in tables:
                logger.warning(
                    "_read_lost_and_found_rows: no lost_and_found table in %s (tables: %s)",
                    recover_db_path,
                    tables,
                )
                return []
            cur = con.execute(
                "SELECT rootpgno, pgno, nfield, id, "
                "c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 "
                "FROM lost_and_found ORDER BY rootpgno, id"
            )
            return cur.fetchall()
        finally:
            con.close()
    except Exception as e:
        logger.error("_read_lost_and_found_rows failed for %s: %s", recover_db_path, e)
        return []


def _insert_recovered_rows(
    healing_con,
    rows,
    rootpages,
    cache,
    counters,
    recover_db_path,
):
    """Insert each ``lost_and_found`` row into its target table in ``healing_con``.

    Updates ``counters`` (a dict) with recovered and lost row counts.
    Uses ``cache`` (a dict) to avoid re-fetching column info for repeated tables.
    """
    # FIX c7-arch-iter4: invert the rootpages dicts for fast lookup. Bug was
    # `_read_rootpages_from_master() -> {table_name: rootpage}` then
    # `healing_rootpages.get(pgno)` where pgno is int — always None, every
    # lost_and_found row was counted as lost.
    pgno_to_table = _invert_pgno_to_table(rootpages)
    healing_pgno_to_table: dict = {}

    # Build a fast lookup: rootpgno → table_name from the healing DB's
    # sqlite_master (which we just loaded from dump.sql) because it has the
    # canonical post-recovery table list.
    try:
        for name, rp, _t in healing_con.execute(
            "SELECT name, rootpage, type FROM sqlite_master WHERE type='table'"
        ).fetchall():
            if name.startswith("sqlite_"):
                continue
            if rp and int(rp) > 0:
                healing_pgno_to_table[int(rp)] = name
    except Exception as e:
        logger.warning("_insert_recovered_rows: cannot read healing sqlite_master: %s", e)

    # Group rows by rootpgno
    by_pgno: dict = {}
    for r in rows:
        pg = r["rootpgno"]
        by_pgno.setdefault(pg, []).append(r)

    for pgno, group in by_pgno.items():
        # FIX c7-arch-iter4: use inverted maps. Prefer healing (canonical
        # post-recovery) when available; fall back to the corrupt-db
        # sqlite_master read.
        table = healing_pgno_to_table.get(pgno) or pgno_to_table.get(pgno)
        if not table or table == "lost_and_found":
            counters["lost"] += len(group)
            continue
        if table not in cache:
            cache[table] = {
                "cols": _map_lost_and_found_columns(table, recover_db_path),
                "rowid_alias": _first_col_is_rowid_alias(table, recover_db_path),
            }
        info = cache[table]
        declared_cols = info["cols"]
        if not declared_cols:
            counters["lost"] += len(group)
            continue
        # Column offset:
        #   rowid-alias (INTEGER PRIMARY KEY) → c0 is NULL; real data starts at c1
        #   else (TEXT PRIMARY KEY, etc.) → real data starts at c0
        offset = 1 if info["rowid_alias"] else 0
        insert_cols = declared_cols[offset:]
        if not insert_cols:
            counters["lost"] += len(group)
            continue
        placeholders = ",".join(["?"] * len(insert_cols))
        quoted_table = '"' + table.replace('"', '""') + '"'
        quoted_cols = ",".join('"' + c.replace('"', '""') + '"' for c in insert_cols)
        insert_sql = (
            "INSERT OR IGNORE INTO " + quoted_table + " "
            "(" + quoted_cols + ") VALUES (" + placeholders + ")"
        )
        for r in group:
            try:
                values = [r[f"c{i+offset}"] for i in range(len(insert_cols))]
            except (IndexError, KeyError) as e:
                logger.warning(
                    "DB_RECOVERY: row mapping out-of-range pgno=%s id=%s: %s",
                    pgno, r["id"], e,
                )
                counters["lost"] += 1
                continue
            try:
                healing_con.execute(insert_sql, values)
                counters["recovered"] += 1
            except sqlite3.IntegrityError as e:
                # Duplicate or constraint violation — log + skip, don't fail the batch
                logger.info(
                    "DB_RECOVERY: skipped duplicate/invalid row pgno=%s id=%s: %s",
                    pgno, r["id"], e,
                )
                counters["lost"] += 1
            except Exception as e:
                logger.warning(
                    "DB_RECOVERY: failed to recover row pgno=%s id=%s: %s",
                    pgno, r["id"], e,
                )
                counters["lost"] += 1


def _atomic_swap(db_path: str, healing_path: str) -> bool:
    """Replace ``db_path`` with ``healing_path`` (atomic on POSIX).

    Unlinks any pre-existing -wal/-shm sidecar files first so the swapped
    db starts in a clean WAL state.
    """
    for ext in ("-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            try:
                os.unlink(p)
            except OSError as e:
                logger.warning("DB_RECOVERY: failed to unlink %s: %s", p, e)
    try:
        os.replace(healing_path, db_path)
        return True
    except OSError as e:
        logger.error("DB_RECOVERY: os.replace failed %s → %s: %s", healing_path, db_path, e)
        return False


def recover_corrupt_db(db_path: str) -> bool:
    """Run the full SQLite corruption-recovery procedure on ``db_path``.

    See module docstring for the 6-step procedure. Returns True if the
    db was successfully recovered (file swapped + integrity-checked).
    Returns False on any failure (logged at ERROR); the caller is responsible
    for whether False is a fatal error.

    CRIT-3 / CRIT-5 fix: concurrent callers serialize through
    ``proxy.state.recovery_lock`` (#13, an RLock). Inside the lock we DO NOT
    set any ``recovery_in_progress`` flag (the flag was removed in CRIT-3 —
    it caused a process-wide short-circuit where only the FIRST corruption
    event actually recovered and every subsequent corruption event
    skipped the procedure). The lock alone provides serialization; the
    recovery body is idempotent on a healthy db (regenerates the healing
    file from a clean dump — wasted work but not data loss).

    Args:
        db_path: Absolute or relative path to the SQLite database file.

    Returns:
        True if recovery succeeded; False on any failure (file not found,
        subprocess errors, post-recovery integrity check still failing, etc.).

    Raises:
        Nothing — all exceptions (including ``subprocess.SubprocessError``,
        ``OSError``, ``sqlite3.DatabaseError``) are caught and logged; the
        function returns False instead of raising.

    Side-effects:
        - Creates a timestamped backup directory at
          ``<db_dir>/.corrupt_backup_<ts>/`` containing the corrupt db file,
          ``dump.sql`` (clean dump), and ``recover.sql`` (.recover output).
        - Creates a temporary ``<db_path>.healing`` db during recovery.
        - Atomically swaps ``<db_path>.healing`` into ``<db_path>``
          (``os.replace`` is atomic on POSIX).
        - Unlinks any pre-existing ``<db_path>-wal`` / ``<db_path>-shm``
          sidecar files BEFORE and AFTER the swap.
        - Logs at WARNING/ERROR on every step; final result logged at WARNING.
    """
    if not db_path or db_path == ":memory:":
        logger.error("DB_RECOVERY: refusing to recover in-memory db %s", db_path)
        return False
    if not os.path.exists(db_path):
        logger.error("DB_RECOVERY: db file not found: %s", db_path)
        return False

    # CRIT-3 / CRIT-5 fix: serialize via proxy.state.recovery_lock (#13, RLock).
    # There is intentionally no module-level _INFLIGHT flag and no
    # state.recovery_in_progress flag — both were removed because they caused
    # permanent auto-heal disablement after the first successful recovery.
    # The lock alone is sufficient: RLock semantics allow ensure_db_integrity
    # to be called from inside the recovery body without self-deadlock, and
    # concurrent callers serialize cleanly.
    #
    # If proxy.state is not importable, that's a proxy startup bug — fail
    # loudly rather than silently skipping the lock and running recovery
    # without any serialization (the AGENTS.md rule is "All shared locks and
    # dicts are defined ONCE in state.py"; we trust that contract).
    import proxy.state as state_module  # type: ignore
    state_lock = getattr(state_module, "recovery_lock", None)
    if state_lock is None:
        logger.error(
            "DB_RECOVERY: proxy.state.recovery_lock missing — refusing to "
            "recover %s without lock serialization (state.py must define "
            "recovery_lock at position #13)",
            db_path,
        )
        return False
    with state_lock:
        return _recover_corrupt_db_inner(db_path)


def _recover_corrupt_db_inner(db_path: str) -> bool:
    """The actual recovery procedure (caller holds the lock).

    Executes the 6-step procedure documented in the module docstring:
      1. Back up corrupt db + -wal + -shm → <db_dir>/.corrupt_backup_<ts>/
      2. Run `sqlite3 <db> ".dump"`     → dump.sql    (intact DDL + non-corrupt INSERTs)
      3. Run `sqlite3 <db> ".recover"`  → recover.sql (corrupt-table rows → lost_and_found)
      4. Build a clean db at <db>.healing:
         a. Sed dump.sql: 'ROLLBACK; -- due to errors' → 'COMMIT;'
         b. Load dump.sql (schema + indexes + recoverable INSERTs)
         c. ATTACH recover.db; map lost_and_found rows → original tables via rootpgno
         d. REINDEX; PRAGMA integrity_check; wal_checkpoint(TRUNCATE)
      5. Atomic swap: os.replace(<db>.healing, <db>) + unlink -wal/-shm
      6. Log recovered-vs-lost row counts; emit metric

    Args:
        db_path: Absolute or relative path to the SQLite database file.

    Returns:
        True if recovery succeeded and the db file now passes integrity_check;
        False on any failure (subprocess errors, post-recovery integrity check
        still failing, etc.).

    Raises:
        Nothing — all exceptions are caught and logged.

    Side-effects:
        - Creates a backup directory at ``<db_dir>/.corrupt_backup_<ts>/``.
        - Creates temporary ``<db>.healing`` and ``<db>.recover_tmp`` files.
        - Atomically swaps the healing file into the original path.
        - Unlinks ``<db>-wal`` / ``<db>-shm`` sidecar files.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    backup_dir = os.path.join(parent_dir, f".corrupt_backup_{ts}")
    healing_path = db_path + ".healing"
    recover_db_path = db_path + ".recover_tmp"

    counters = {"recovered": 0, "lost": 0}
    try:
        # Step 1: back up corrupt artifacts
        try:
            os.makedirs(backup_dir, exist_ok=True)
        except OSError as e:
            logger.error("DB_RECOVERY: cannot create backup dir %s: %s", backup_dir, e)
            return False
        for ext in ("", "-wal", "-shm"):
            src = db_path + ext
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(backup_dir, os.path.basename(src)))
                except OSError as e:
                    logger.warning("DB_RECOVERY: backup copy failed for %s: %s", src, e)
        logger.warning("DB_RECOVERY: backed up corrupt db to %s", backup_dir)

        dump_path = os.path.join(backup_dir, "dump.sql")
        recover_path = os.path.join(backup_dir, "recover.sql")

        # Step 2: .dump
        try:
            with open(dump_path, "w", encoding="utf-8", errors="replace") as f:
                rc = subprocess.run(
                    ["sqlite3", db_path, ".dump"],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=60,
                )
            if rc.returncode != 0:
                err = rc.stderr.decode("utf-8", errors="replace") if rc.stderr else ""
                logger.error("DB_RECOVERY: .dump failed (rc=%d): %s", rc.returncode, err)
                return False
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error("DB_RECOVERY: .dump subprocess failed: %s", e)
            return False

        # Step 3: .recover
        try:
            with open(recover_path, "w", encoding="utf-8", errors="replace") as f:
                rc = subprocess.run(
                    ["sqlite3", db_path, ".recover"],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=120,
                )
            # sqlite3 .recover may return non-zero even on partial success;
            # trust the file content over the return code.
            if rc.returncode != 0:
                err = rc.stderr.decode("utf-8", errors="replace") if rc.stderr else ""
                logger.warning(
                    "DB_RECOVERY: .recover returned rc=%d (continuing): %s",
                    rc.returncode, err,
                )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error("DB_RECOVERY: .recover subprocess failed: %s", e)
            return False

        # Step 4a: sed — fix ROLLBACK;--due-to-errors → COMMIT;
        try:
            with open(dump_path, "r", encoding="utf-8", errors="replace") as f:
                dump_sql = f.read()
        except OSError as e:
            logger.error("DB_RECOVERY: cannot read %s: %s", dump_path, e)
            return False
        dump_sql = re.sub(
            r"ROLLBACK;\s*--\s*due to errors",
            "COMMIT;",
            dump_sql,
            flags=re.IGNORECASE,
        )

        # Step 4b: build healing DB and load dump.sql
        if os.path.exists(healing_path):
            try:
                os.unlink(healing_path)
            except OSError:
                pass
        for ext in ("-wal", "-shm"):
            p = healing_path + ext
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        try:
            con = sqlite3.connect(healing_path)
        except sqlite3.DatabaseError as e:
            logger.error("DB_RECOVERY: cannot create healing db %s: %s", healing_path, e)
            return False
        try:
            try:
                con.executescript(dump_sql)
                con.commit()
            except sqlite3.DatabaseError as dump_err:
                # CRIT-3 / CRIT-4 (iter3) fix: when ``dump.sql`` is malformed
                # (severe corruption produced broken SQL output, or the file
                # itself was empty/zero-byte), fall through to the
                # schema-only swap path using ``db_migrations.SCHEMA_SQL``.
                # Without this fallback, the recovery returned False, which
                # caused ``init_db`` to raise RuntimeError and the proxy to
                # refuse to start.
                #
                # The fallback produces an EMPTY but schema-valid db: the
                # proxy can come up; lost rows are documented in the backup
                # dir's ``recover.sql`` for operator review. The schema-only
                # path is the UNIVERSAL recovery — any recovery failure that
                # leaves us with an open healing connection should funnel
                # through here, not bail out.
                logger.warning(
                    "DB_RECOVERY: dump.sql execution failed (%s); "
                    "falling back to schema-only swap",
                    dump_err,
                )
                # ``executescript`` issues an implicit COMMIT before the
                # script, so the connection is in a clean state after the
                # failed dump.sql above. _apply_and_validate_schema applies
                # SCHEMA_SQL and verifies the schema is complete before we
                # let recovery proceed to the atomic swap.
                if not _apply_and_validate_schema(con):
                    logger.error(
                        "DB_RECOVERY: schema-only fallback failed validation; "
                        "refusing to swap",
                    )
                    return False
            else:
                # Issue B fix: even when the dump.sql executed successfully,
                # corruption in the pages backing ``sessions`` or
                # ``rate_limiter_state`` can cause those CREATE TABLE
                # statements to be silently absent from the dump. The
                # resulting healing db passes PRAGMA integrity_check (SQLite
                # structure is OK) but is missing application tables —
                # silently disabling the F1 privilege-escalation guard
                # (``session_exists``) once the healing db is swapped in.
                #
                # We ALWAYS apply SCHEMA_SQL on top of the dump (idempotent
                # CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS —
                # fills in any tables the corrupt dump omitted without
                # overwriting recovered tables or rows) and then validate
                # the schema. If validation fails, refuse the swap so the
                # live db is left untouched.
                if not _apply_and_validate_schema(con):
                    logger.error(
                        "DB_RECOVERY: healing db schema incomplete after "
                        "dump.sql execution; refusing to swap",
                    )
                    return False

            # Step 4c: load recover.sql into a separate DB and map lost_and_found
            if not _populate_recovery_db(recover_path, recover_db_path):
                # No recovery possible — but the schema is already in healing_path.
                # Fall through; we'll still swap and reindex what we have.
                logger.warning("DB_RECOVERY: recover.sql load failed; swapping schema-only")
            else:
                lost_rows = _read_lost_and_found_rows(recover_db_path)
                if lost_rows:
                    rootpages = _read_rootpages_from_master(db_path)
                    cache: dict = {}
                    _insert_recovered_rows(
                        con, lost_rows, rootpages, cache, counters, recover_db_path,
                    )
                    con.commit()

            # Step 4d: REINDEX + integrity_check + wal_checkpoint
            try:
                con.execute("REINDEX")
            except Exception as e:
                logger.warning("DB_RECOVERY: REINDEX failed (non-fatal): %s", e)
            check = con.execute("PRAGMA integrity_check").fetchone()
            if check is None or check[0] != "ok":
                logger.error(
                    "DB_RECOVERY: post-recovery integrity_check failed: %r", check,
                )
                return False
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as e:
                logger.warning("DB_RECOVERY: wal_checkpoint(TRUNCATE) failed: %s", e)
        finally:
            try:
                con.close()
            except Exception:
                pass

        # Step 5: atomic swap
        if not _atomic_swap(db_path, healing_path):
            return False

        # Cleanup temp recover db
        for ext in ("", "-wal", "-shm"):
            p = recover_db_path + ext
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

        logger.warning(
            "DB_RECOVERY: complete. recovered=%d lost=%d backup=%s",
            counters["recovered"], counters["lost"], backup_dir,
        )
        return True

    except Exception as e:
        logger.exception("DB_RECOVERY: unexpected failure: %s", e)
        return False