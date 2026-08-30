"""
Schema bootstrap + migrations + startup maintenance for memory system.

Moved from db.py (Tier 4, 4c-2). Pure leaf — imports from db_pool only.
"""

import hashlib
import json
import logging
import os
from typing import Optional

from memory.db_pool import (
    _checkout_global,
    _create_raw_connection,
    _get_db_path,
    path_maintenance,
    return_connection,
)

logger = logging.getLogger("memory.db")

# Required tables for validate_schema_complete. Keep in sync with SCHEMA_SQL below.
# Module-level so the dogfooding metatest (tests/unit/test_db_recovery_schema_guard.py
# TestSchemaCoverageMetatest) can compute coverage.
REQUIRED_TABLES = (
    "messages",
    "conversation_summaries",
    "extraction_batches",
    "tasks",
    "message_summaries",
    "entities",
    "facts",
    "supersessions",
    "memory_log",
    "sessions",
    "rate_limiter_state",
    "tool_call_audit",
)

# Required columns for validate_schema_complete. These are the columns that
# production code reads on chat-completions hot paths, mode-transition paths,
# and SOTA sync paths — derived from grep'd SELECT/INSERT statements across
# src/memory/ and src/proxy/. AGENTS.md test discipline: a missing required
# column silently disables a guard (e.g. F1 ``session_exists`` against an
# absent ``sessions`` table) or causes an immediate AttributeError on first
# read. Module-level so the metatest can compute per-table coverage.
#
# Coverage target: >= 70% of every CREATE TABLE's declared columns (see
# ``test_validator_required_columns_metatest``).
REQUIRED_COLUMNS = (
    # messages — read on every chat-completion (history loads) and SOTA sync
    ("messages", "id"),
    ("messages", "session_id"),
    ("messages", "message_index"),
    ("messages", "role"),
    ("messages", "content"),
    ("messages", "content_hash"),
    ("messages", "created_at"),
    ("messages", "updated_at"),
    ("messages", "exchange_index"),
    # conversation_summaries — read on every chat-completion (context injection)
    ("conversation_summaries", "session_id"),
    ("conversation_summaries", "summary"),
    ("conversation_summaries", "summary_tokens"),
    ("conversation_summaries", "exchange_count"),
    ("conversation_summaries", "last_exchange_index"),
    ("conversation_summaries", "last_exchange_start"),
    ("conversation_summaries", "last_message_index"),
    ("conversation_summaries", "raw_start_message_index"),
    ("conversation_summaries", "ancient_history"),
    ("conversation_summaries", "created_at"),
    ("conversation_summaries", "updated_at"),
    # extraction_batches — read/written by SOTA sync
    ("extraction_batches", "id"),
    ("extraction_batches", "session_id"),
    ("extraction_batches", "start_message_index"),
    ("extraction_batches", "end_message_index"),
    ("extraction_batches", "extraction_json"),
    ("extraction_batches", "status"),
    ("extraction_batches", "error"),
    ("extraction_batches", "created_at"),
    ("extraction_batches", "updated_at"),
    ("extraction_batches", "start_exchange_index"),
    ("extraction_batches", "end_exchange_index"),
    # tasks — schema kept for compatibility with persisted state; reads are
    # delegated to Neo4j, so we cover the full schema here.
    ("tasks", "id"),
    ("tasks", "title"),
    ("tasks", "priority"),
    ("tasks", "status"),
    ("tasks", "parent_id"),
    ("tasks", "blockers"),
    ("tasks", "session_id"),
    ("tasks", "created_at"),
    ("tasks", "updated_at"),
    ("tasks", "completed_at"),
    # message_summaries — read on chat-completion context
    ("message_summaries", "id"),
    ("message_summaries", "session_id"),
    ("message_summaries", "message_index"),
    ("message_summaries", "end_message_index"),
    ("message_summaries", "role"),
    ("message_summaries", "summary"),
    ("message_summaries", "tool_summary"),
    ("message_summaries", "content_hash"),
    ("message_summaries", "model_used"),
    ("message_summaries", "token_estimate"),
    ("message_summaries", "created_at"),
    ("message_summaries", "updated_at"),
    ("message_summaries", "exchange_index"),
    # entities — read by memory tool loop and SOTA reconcile
    ("entities", "id"),
    ("entities", "name"),
    ("entities", "entity_type"),
    ("entities", "summary"),
    ("entities", "session_id"),
    ("entities", "extracted_at"),
    ("entities", "created_at"),
    ("entities", "updated_at"),
    # facts — read by memory tool loop
    ("facts", "id"),
    ("facts", "source_entity_id"),
    ("facts", "target_entity_id"),
    ("facts", "relationship_type"),
    ("facts", "fact"),
    ("facts", "valid_at"),
    ("facts", "invalid_at"),
    ("facts", "session_id"),
    ("facts", "extracted_at"),
    ("facts", "created_at"),
    ("facts", "updated_at"),
    # supersessions — written by SOTA extraction
    ("supersessions", "id"),
    ("supersessions", "old_fact_id"),
    ("supersessions", "new_fact_id"),
    ("supersessions", "reason"),
    ("supersessions", "created_at"),
    # memory_log — written by SOTA sync persistence
    ("memory_log", "memory_id"),
    ("memory_log", "session_id"),
    ("memory_log", "action"),
    ("memory_log", "memory_type"),
    ("memory_log", "topic_key"),
    ("memory_log", "content"),
    ("memory_log", "created_at"),
    # sessions — read on every chat-completion (F1 privilege-escalation guard)
    ("sessions", "session_id"),
    ("sessions", "created_at"),
    # rate_limiter_state — read on every chat-completion (rate limiting)
    ("rate_limiter_state", "limiter_name"),
    ("rate_limiter_state", "key"),
    ("rate_limiter_state", "timestamps"),
    # tool_call_audit — written by observability path on every tool call
    ("tool_call_audit", "id"),
    ("tool_call_audit", "session_id"),
    ("tool_call_audit", "tool_name"),
    ("tool_call_audit", "arguments"),
    ("tool_call_audit", "result_status"),
    ("tool_call_audit", "result_summary"),
    ("tool_call_audit", "timestamp"),
    ("tool_call_audit", "request_id"),
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    content_hash TEXT,
    tool_calls_json TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    is_edited BOOLEAN DEFAULT 0,
    exchange_index INTEGER DEFAULT NULL,
    UNIQUE(session_id, message_index)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_index ON messages(session_id, message_index);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    session_id TEXT NOT NULL PRIMARY KEY,
    summary TEXT,
    summary_tokens INTEGER DEFAULT 0,
    exchange_count INTEGER DEFAULT 0,
    last_exchange_index INTEGER DEFAULT -1,
    last_exchange_start INTEGER,
    last_message_index INTEGER,
    raw_start_message_index INTEGER DEFAULT -1,
    ancient_history TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS extraction_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_index INTEGER NOT NULL,
    end_message_index INTEGER NOT NULL,
    extraction_json TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TEXT,
    updated_at TEXT,
    start_exchange_index INTEGER DEFAULT NULL,
    end_exchange_index INTEGER DEFAULT NULL,
    UNIQUE(session_id, start_message_index, end_message_index)
);

CREATE INDEX IF NOT EXISTS idx_extraction_batches_status ON extraction_batches(status);
CREATE INDEX IF NOT EXISTS idx_extraction_batches_session ON extraction_batches(session_id);
CREATE INDEX IF NOT EXISTS idx_batches_covered ON extraction_batches(session_id, status, start_message_index, end_message_index);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    priority TEXT DEFAULT 'P2',
    status TEXT DEFAULT 'pending',
    parent_id TEXT REFERENCES tasks(id),
    blockers TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

-- Per-exchange summaries (ChoriProx Plus §4.1)
CREATE TABLE IF NOT EXISTS message_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    end_message_index INTEGER DEFAULT 0,
    role TEXT NOT NULL,
    summary TEXT NOT NULL,
    tool_summary TEXT,
    content_hash TEXT NOT NULL,
    model_used TEXT,
    token_estimate INTEGER,
    rendered_full_tokens INTEGER DEFAULT 0,
    rendered_action_tokens INTEGER DEFAULT 0,
    rendered_narrative_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    exchange_index INTEGER DEFAULT NULL,
    UNIQUE(session_id, message_index)
);

CREATE INDEX IF NOT EXISTS idx_msg_summaries_session ON message_summaries(session_id, message_index);
CREATE INDEX IF NOT EXISTS idx_msg_summaries_hash ON message_summaries(content_hash);

-- Entity storage (ChoriProx Plus §5.6)
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    summary TEXT,
    session_id TEXT,
    extracted_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, entity_type, session_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id);

-- Fact/relationship storage (ChoriProx Plus §5.6)
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    fact TEXT NOT NULL,
    valid_at TEXT,
    invalid_at TEXT,
    session_id TEXT,
    extracted_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relationship_type, fact)
);

CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_target ON facts(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(relationship_type);
CREATE INDEX IF NOT EXISTS idx_facts_valid_at ON facts(valid_at);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);

-- Supersession tracking (ChoriProx Plus §5.6)
CREATE TABLE IF NOT EXISTS supersessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    new_fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Recovery/audit log for Neo4j-persisted memories
CREATE TABLE IF NOT EXISTS memory_log (
    memory_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    action TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'event',
    topic_key TEXT,
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

-- Session existence tracking (F1: prevents privilege escalation via forged session IDs)
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rate limiter state persistence (survives restarts)
CREATE TABLE IF NOT EXISTS rate_limiter_state (
    limiter_name TEXT NOT NULL,
    key TEXT NOT NULL,
    timestamps TEXT NOT NULL,
    PRIMARY KEY (limiter_name, key)
);

-- Tool call audit log (#20)
CREATE TABLE IF NOT EXISTS tool_call_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT,
    result_status TEXT,
    result_summary TEXT,
    timestamp TEXT NOT NULL,
    request_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_audit_session ON tool_call_audit(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_audit_timestamp ON tool_call_audit(timestamp);
"""


def validate_schema_complete(conn) -> tuple:
    """Validate a candidate DB has the production schema required at startup.

    Pure-leaf helper used by ``memory.db_recovery._apply_and_validate_schema``
    before it lets the recovery procedure swap a healing db into the live
    ``memory.db`` slot. Uses raw ``sqlite_master`` / ``PRAGMA table_info`` /
    ``PRAGMA index_list`` queries on the passed-in ``conn`` — does NOT import
    ``memory.db`` or ``memory.db_pool``, does NOT open its own connection.

    Checks performed:

      * Every required production table exists.
      * Critical columns on those tables exist (the columns that production
        code reads on startup paths).
      * Critical constraints are present (``PRIMARY KEY (limiter_name, key)``
        on ``rate_limiter_state`` and ``UNIQUE(session_id, message_index)``
        on ``messages``).

    Args:
        conn: An open sqlite3 connection to the candidate DB. The connection
            is used read-only here; nothing in this function mutates schema.

    Returns:
        ``(True, [])`` when the schema is complete.
        ``(False, [diagnostic, ...])`` with one or more human-readable
        diagnostic strings when the schema is missing tables, columns, or
        constraints. The diagnostics are designed to be safe to log at ERROR
        level — they name the missing object, not internal state.
    """
    diagnostics: list = []

    # Required tables — module-level for the dogfooding metatest.
    required_tables = REQUIRED_TABLES

    # Required columns — module-level so the metatest can compute coverage.
    required_columns = REQUIRED_COLUMNS

    try:
        present_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except Exception as e:
        return False, [f"cannot query sqlite_master: {e}"]

    for t in required_tables:
        if t not in present_tables:
            diagnostics.append(f"missing required table: {t}")

    # Column checks only on tables that exist (already-reported missing
    # tables would otherwise produce noisy duplicate diagnostics).
    table_columns_cache: dict = {}

    def _cols(table: str) -> set:
        if table in table_columns_cache:
            return table_columns_cache[table]
        cols = {
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }
        table_columns_cache[table] = cols
        return cols

    for table, col in required_columns:
        if table not in present_tables:
            continue
        try:
            if col not in _cols(table):
                diagnostics.append(
                    f"missing required column: {table}.{col}"
                )
        except Exception as e:
            diagnostics.append(
                f"cannot query table_info({table}): {e}"
            )

    # PRIMARY KEY (limiter_name, key) on rate_limiter_state.
    # PRAGMA index_list returns (seq, name, unique, origin, partial); a
    # composite PRIMARY KEY appears as an internal index with origin='pk'.
    if "rate_limiter_state" in present_tables:
        try:
            indexes = conn.execute(
                "PRAGMA index_list(rate_limiter_state)"
            ).fetchall()
            pk_found = False
            for idx in indexes:
                if len(idx) < 4 or idx[3] != "pk":
                    continue
                idx_name = idx[1]
                cols = [
                    r[2]
                    for r in conn.execute(
                        f"PRAGMA index_info({idx_name})"
                    ).fetchall()
                ]
                if "limiter_name" in cols and "key" in cols:
                    pk_found = True
                    break
            if not pk_found:
                diagnostics.append(
                    "rate_limiter_state missing PRIMARY KEY (limiter_name, key)"
                )
        except Exception as e:
            diagnostics.append(
                f"cannot query index_list(rate_limiter_state): {e}"
            )

    # UNIQUE(session_id, message_index) on messages — same shape, but the
    # unique flag (idx[2] == 1) distinguishes from a plain PRIMARY KEY.
    if "messages" in present_tables:
        try:
            indexes = conn.execute(
                "PRAGMA index_list(messages)"
            ).fetchall()
            unique_found = False
            for idx in indexes:
                if len(idx) < 3 or idx[2] != 1:
                    continue
                idx_name = idx[1]
                cols = [
                    r[2]
                    for r in conn.execute(
                        f"PRAGMA index_info({idx_name})"
                    ).fetchall()
                ]
                if "session_id" in cols and "message_index" in cols:
                    unique_found = True
                    break
            if not unique_found:
                diagnostics.append(
                    "messages missing UNIQUE(session_id, message_index)"
                )
        except Exception as e:
            diagnostics.append(
                f"cannot query index_list(messages): {e}"
            )

    return (len(diagnostics) == 0, diagnostics)


def backfill_sessions_table(conn):
    """Populate sessions through the caller-owned migration connection."""
    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    if count > 0:
        return

    conn.execute(
        """INSERT OR IGNORE INTO sessions (session_id, created_at)
           SELECT DISTINCT session_id, MIN(created_at)
           FROM messages
           GROUP BY session_id"""
    )

    new_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    if new_count > 0:
        logger.info("Backfilled %d sessions from messages table", new_count)


def _needs_exchange_index_migration(conn):
    """Check if extraction_batches still uses message_index-based unique constraint."""
    try:
        indexes = conn.execute("PRAGMA index_list(extraction_batches)").fetchall()
        for idx in indexes:
            if idx["origin"] == "u":
                cols = conn.execute(f"PRAGMA index_info({idx['name']})").fetchall()
                col_names = {c["name"] for c in cols}
                if "start_message_index" in col_names:
                    return True
    except Exception:
        pass
    return False


def _needs_message_summaries_migration(conn):
    """Check if message_summaries still uses message_index-based unique constraint."""
    try:
        indexes = conn.execute("PRAGMA index_list(message_summaries)").fetchall()
        for idx in indexes:
            if idx["origin"] == "u":
                cols = conn.execute(f"PRAGMA index_info({idx['name']})").fetchall()
                col_names = {c["name"] for c in cols}
                if "message_index" in col_names and "exchange_index" not in col_names:
                    return True
    except Exception:
        pass
    return False


def _migrate_add_chain_hash(conn):
    """Add chain_hash column to messages table. Idempotent."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if "chain_hash" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN chain_hash TEXT")
        logger.info("Migration: Added chain_hash column to messages table")


def _migrate_recreate_conversation_summaries(conn):
    """Recreate conversation_summaries with PRIMARY KEY(session_id).

    Drops model_name column (session-scoped). Collapses multi-model rows per
    session into one row: pick the row with MAX(last_exchange_index); tie-break
    by LENGTH(ancient_history). Lossy collapse approved (data is ephemeral,
    7-day retention).

    Idempotent: skips if model_name column absent (already session-scoped).
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
    if "model_name" not in cols:
        return
    conn.execute("BEGIN")
    try:
        conn.execute("""
            CREATE TABLE conversation_summaries_new (
                session_id TEXT NOT NULL PRIMARY KEY,
                summary TEXT,
                summary_tokens INTEGER DEFAULT 0,
                exchange_count INTEGER DEFAULT 0,
                last_exchange_index INTEGER DEFAULT -1,
                last_exchange_start INTEGER,
                last_message_index INTEGER,
                raw_start_message_index INTEGER DEFAULT -1,
                ancient_history TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            INSERT INTO conversation_summaries_new
                (session_id, summary, summary_tokens, exchange_count, last_exchange_index,
                 last_exchange_start, last_message_index, raw_start_message_index,
                 ancient_history, created_at, updated_at)
            SELECT
                session_id, summary, summary_tokens, exchange_count, last_exchange_index,
                last_exchange_start, last_message_index, raw_start_message_index,
                ancient_history, created_at, updated_at
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY session_id
                        ORDER BY last_exchange_index DESC,
                                 LENGTH(COALESCE(ancient_history, '')) DESC
                    ) AS rn
                FROM conversation_summaries
            ) WHERE rn = 1
        """)
        conn.execute("DROP TABLE conversation_summaries")
        conn.execute("ALTER TABLE conversation_summaries_new RENAME TO conversation_summaries")
        conn.execute("COMMIT")
        logger.info("Migration: Collapsed conversation_summaries to PRIMARY KEY(session_id)")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _backfill_one_session(conn, session_id):
    """Backfill chain_hash for a single session. Isolated from siblings so a
    single bad row (e.g. corrupted tool_calls_json) doesn't abort the whole
    migration and leave indexes uncreated downstream.
    """
    from proxy.chain_hash import _message_hash
    rows = conn.execute(
        "SELECT message_index, role, content, tool_calls_json, tool_call_id "
        "FROM messages WHERE session_id = ? ORDER BY message_index",
        (session_id,),
    ).fetchall()
    prev_chain = None
    chunk = []
    for row in rows:
        message_index = row["message_index"]
        role = row["role"]
        is_system = (role or "").lower() == "system"
        if is_system:
            chunk.append((None, session_id, message_index))
            continue
        try:
            msg = {
                "role": role,
                "content": row["content"],
                "tool_calls": json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else None,
            }
            msg_hash = _message_hash(msg)
            if prev_chain is None:
                chain_hash = msg_hash
            else:
                chain_hash = hashlib.sha256(f"{prev_chain}:{msg_hash}".encode("utf-8")).hexdigest()[:16]
            chunk.append((chain_hash, session_id, message_index))
            prev_chain = chain_hash
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            logger.warning(
                "backfill row corrupt session=%s message_index=%s; setting chain_hash=NULL: %s",
                session_id, message_index, e,
            )
            chunk.append((None, session_id, message_index))
        if len(chunk) >= 333:
            conn.executemany(
                "UPDATE messages SET chain_hash = ? WHERE session_id = ? AND message_index = ?",
                chunk,
            )
            chunk = []
    if chunk:
        conn.executemany(
            "UPDATE messages SET chain_hash = ? WHERE session_id = ? AND message_index = ?",
            chunk,
        )
    return len(rows)


def _backfill_chain_hash(conn):
    """Compute chain_hash for existing messages. Per-session sequential pass.

    Skips system messages (chain_hash=NULL) — matches compute_chain() and
    _store_chain_hash. The first non-system message seeds the chain with its
    msg_hash. Chunks per session at 333 messages (999 // 3 params/row) to
    respect the SQLite ~999 parameter limit.

    Guard: skips entirely when no non-system message lacks chain_hash. For
    any session with at least one missing chain_hash, the ENTIRE session is
    recomputed (not just the NULL rows) so incremental gaps don't leave the
    chain internally inconsistent.

    Per-session try/except: a single corrupted row (bad json in tool_calls_json)
    is logged and skipped so the migration does NOT abort mid-loop. If the
    outer function raises, downstream _migrate_add_indexes would not run and
    V2 queries would degrade to full table scans.
    """
    try:
        pending = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chain_hash IS NULL AND role != 'system'"
        ).fetchone()[0]
        if pending == 0:
            return
        sessions = conn.execute(
            "SELECT DISTINCT session_id FROM messages "
            "WHERE chain_hash IS NULL AND role != 'system'"
        ).fetchall()
        total_updated = 0
        for (session_id,) in sessions:
            try:
                updated = _backfill_one_session(conn, session_id)
                total_updated += updated
            except Exception as e:
                logger.warning(
                    "backfill session=%s failed; continuing with next session: %s",
                    session_id, e, exc_info=True,
                )
        if total_updated > 0:
            logger.info("Backfill: computed chain_hash for %d messages", total_updated)
    except Exception as e:
        logger.warning(
            "Backfill aborted before completion; downstream migrations may use "
            "partial state: %s", e, exc_info=True,
        )


def _migrate_add_indexes(conn):
    """Add indexes for V2 chain-hash queries. Idempotent."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_exchange "
        "ON messages(session_id, exchange_index)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_summaries_boundary "
        "ON conversation_summaries(session_id, raw_start_message_index)"
    )


def init_db(db_path=None):
    """Initialize the database schema and apply all migrations.

    Creates all tables and indexes defined in SCHEMA_SQL, performs WAL
    checkpointing, and runs column-level migrations for existing
    installations. Safe to call multiple times (idempotent).

    C7+ auto-heal: before any migration work, run ``ensure_db_integrity``
    on the db file. If integrity_check fails, invoke ``recover_corrupt_db``
    which rebuilds a clean db from the .dump + .recover artifacts and
    atomically swaps it in. Lazy-imports db_recovery to keep this module
    a pure leaf (per src/memory/AGENTS.md).

    Args:
        db_path: Optional database path. Defaults to the configured path.

    Returns:
        bool: True if initialization succeeded, False on failure.
    """
    path = str(db_path or _get_db_path())
    conn = None
    keeper = None
    body_exception: Optional[BaseException] = None
    close_exception: Optional[BaseException] = None
    keeper_exception: Optional[BaseException] = None

    try:
        # path_maintenance is a context manager: the token is released
        # when this block exits, including on any failure path. The
        # only way to leak the token is to call begin_path_maintenance
        # without end_path_maintenance (or its context manager).
        with path_maintenance(path) as maintenance:
            try:
                from memory.db_recovery import (  # lazy import — pure-leaf rule
                    ensure_db_integrity,
                    recover_corrupt_db,
                )
            except ImportError:
                ensure_db_integrity = None  # type: ignore
                recover_corrupt_db = None  # type: ignore

            if ensure_db_integrity is not None:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    try:
                        if not ensure_db_integrity(path):
                            logger.error(
                                "init_db: integrity_check failed for %s; "
                                "attempting auto-recovery",
                                path,
                            )
                            if (
                                recover_corrupt_db is None
                                or not recover_corrupt_db(path)
                            ):
                                raise RuntimeError(
                                    f"DB corruption recovery failed for "
                                    f"{path}; refusing to start"
                                )
                            if not ensure_db_integrity(path):
                                raise RuntimeError(
                                    f"DB still corrupt after recovery at "
                                    f"{path}; refusing to start"
                                )
                    except RuntimeError:
                        raise
                    except Exception as e:
                        logger.error(
                            "init_db: integrity check raised %s", e
                        )
                        raise

            try:
                conn = _create_raw_connection(
                    path,
                    maintenance_token=maintenance,
                )

                # Open the connection FIRST so SQLite can recover any
                # committed-but-uncheckpointed WAL transactions from a prior
                # crash. Then checkpoint to flush recovered data into the
                # main db before applying schema.
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.executescript(SCHEMA_SQL)
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

                backfill_sessions_table(conn)

                # Migration: Add columns to conversation_summaries if missing
                cursor = conn.execute("PRAGMA table_info(conversation_summaries)")
                columns = {row[1] for row in cursor.fetchall()}
                if "last_message_index" not in columns:
                    conn.execute(
                        "ALTER TABLE conversation_summaries ADD COLUMN "
                        "last_message_index INTEGER DEFAULT -1"
                    )
                    logger.info(
                        "Added missing last_message_index column to "
                        "conversation_summaries table"
                    )
                if "last_exchange_start" not in columns:
                    conn.execute(
                        "ALTER TABLE conversation_summaries ADD COLUMN "
                        "last_exchange_start INTEGER DEFAULT 0"
                    )
                    logger.info(
                        "Added missing last_exchange_start column to "
                        "conversation_summaries table"
                    )
                if "ancient_history" not in columns:
                    conn.execute(
                        "ALTER TABLE conversation_summaries ADD COLUMN "
                        "ancient_history TEXT"
                    )
                    logger.info(
                        "Added missing ancient_history column to "
                        "conversation_summaries table"
                    )

                # CR-3: add end_message_index column if missing (existing installs)
                try:
                    conn.execute(
                        "ALTER TABLE message_summaries ADD COLUMN "
                        "end_message_index INTEGER DEFAULT 0"
                    )
                except Exception:
                    pass  # column already exists

                # Migration: create memory_log table if missing (existing installs)
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS memory_log ("
                    " memory_id TEXT PRIMARY KEY,"
                    " session_id TEXT NOT NULL,"
                    " action TEXT NOT NULL,"
                    " memory_type TEXT NOT NULL DEFAULT 'event',"
                    " topic_key TEXT,"
                    " content TEXT NOT NULL DEFAULT '',"
                    " created_at TEXT NOT NULL"
                    ")"
                )

                # Migration: Add exchange_index to messages
                cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(messages)"
                    ).fetchall()
                ]
                if "exchange_index" not in cols:
                    conn.execute(
                        "ALTER TABLE messages ADD COLUMN "
                        "exchange_index INTEGER DEFAULT NULL"
                    )
                    logger.info(
                        "Added missing exchange_index column to messages table"
                    )

                # Migration: Add exchange_index to message_summaries
                cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(message_summaries)"
                    ).fetchall()
                ]
                if "exchange_index" not in cols:
                    conn.execute(
                        "ALTER TABLE message_summaries ADD COLUMN "
                        "exchange_index INTEGER DEFAULT NULL"
                    )
                    logger.info(
                        "Added missing exchange_index column to "
                        "message_summaries table"
                    )

                # Migration: Add exchange indices to extraction_batches
                cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(extraction_batches)"
                    ).fetchall()
                ]
                if "start_exchange_index" not in cols:
                    conn.execute(
                        "ALTER TABLE extraction_batches ADD COLUMN "
                        "start_exchange_index INTEGER DEFAULT NULL"
                    )
                    logger.info(
                        "Added missing start_exchange_index column to "
                        "extraction_batches table"
                    )
                if "end_exchange_index" not in cols:
                    conn.execute(
                        "ALTER TABLE extraction_batches ADD COLUMN "
                        "end_exchange_index INTEGER DEFAULT NULL"
                    )
                    logger.info(
                        "Added missing end_exchange_index column to "
                        "extraction_batches table"
                    )

                # --- Migration 005: exchange_index backfill ---
                needs_backfill = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE exchange_index IS NULL"
                ).fetchone()[0]
                if needs_backfill == 0:
                    needs_correction = conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE "
                        "exchange_index IS NOT NULL AND exchange_index < 0"
                    ).fetchone()[0]
                    if needs_correction == 0:
                        needs_backfill = 0
                    else:
                        needs_backfill = needs_correction
                if needs_backfill > 0:
                    logger.info(
                        "Exchange_index backfill: processing %d messages",
                        needs_backfill,
                    )
                    conn.execute("DROP TABLE IF EXISTS _ei_backfill")
                    try:
                        conn.execute("""
                            CREATE TEMP TABLE _ei_backfill AS
                            SELECT message_index, session_id,
                                   SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                                     OVER (PARTITION BY session_id
                                           ORDER BY message_index) - 1 AS ei
                            FROM messages
                        """)
                        conn.execute("""
                            UPDATE messages SET exchange_index = (
                                SELECT ei FROM _ei_backfill
                                WHERE _ei_backfill.message_index = messages.message_index
                                  AND _ei_backfill.session_id = messages.session_id
                            )
                            WHERE exchange_index IS NULL
                               OR exchange_index != (
                                   SELECT ei FROM _ei_backfill
                                   WHERE _ei_backfill.message_index = messages.message_index
                                     AND _ei_backfill.session_id = messages.session_id
                               )
                        """)
                    finally:
                        conn.execute("DROP TABLE IF EXISTS _ei_backfill")

                try:
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_messages_session_exchange "
                        "ON messages(session_id, exchange_index)"
                    )
                except Exception:
                    pass

                null_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE exchange_index IS NULL"
                ).fetchone()[0]
                if null_count > 0:
                    logger.warning(
                        "Migration 005: %d messages still have NULL exchange_index",
                        null_count,
                    )

                conn.execute("""
                    UPDATE extraction_batches SET
                        start_exchange_index = (
                            SELECT MIN(exchange_index) FROM messages
                            WHERE session_id = extraction_batches.session_id
                              AND message_index = extraction_batches.start_message_index
                        ),
                        end_exchange_index = (
                            SELECT MAX(exchange_index) FROM messages
                            WHERE session_id = extraction_batches.session_id
                              AND message_index = extraction_batches.end_message_index
                        )
                    WHERE start_exchange_index IS NULL OR end_exchange_index IS NULL
                """)

                unmigrated = conn.execute(
                    "SELECT COUNT(*) FROM extraction_batches "
                    "WHERE start_exchange_index IS NULL OR end_exchange_index IS NULL"
                ).fetchone()[0]
                if unmigrated > 0:
                    logger.warning(
                        "Migration 005: %d extraction_batches still have NULL "
                        "exchange indices",
                        unmigrated,
                    )

                conn.execute("""
                    UPDATE message_summaries SET exchange_index = (
                        SELECT m.exchange_index FROM messages m
                        WHERE m.session_id = message_summaries.session_id
                          AND m.message_index = message_summaries.message_index
                    )
                    WHERE exchange_index IS NULL
                """)

                cols = [
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(conversation_summaries)"
                    ).fetchall()
                ]
                if "last_exchange_index" not in cols:
                    conn.execute(
                        "ALTER TABLE conversation_summaries ADD COLUMN "
                        "last_exchange_index INTEGER DEFAULT -1"
                    )
                    logger.info(
                        "Migration 005: Added last_exchange_index column to "
                        "conversation_summaries"
                    )

                conn.commit()

                # --- Recreate extraction_batches if needed ---
                if _needs_exchange_index_migration(conn):
                    conn.execute("BEGIN")
                    try:
                        conn.execute("""CREATE TABLE extraction_batches_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            start_message_index INTEGER NOT NULL,
                            end_message_index INTEGER NOT NULL,
                            extraction_json TEXT,
                            status TEXT DEFAULT 'pending',
                            error TEXT,
                            created_at TEXT,
                            updated_at TEXT,
                            start_exchange_index INTEGER NOT NULL,
                            end_exchange_index INTEGER NOT NULL,
                            UNIQUE(session_id, start_exchange_index, end_exchange_index)
                        )""")
                        before_count = conn.execute(
                            "SELECT COUNT(*) FROM extraction_batches "
                            "WHERE start_exchange_index IS NOT NULL "
                            "AND end_exchange_index IS NOT NULL"
                        ).fetchone()[0]
                        conn.execute("""INSERT OR IGNORE INTO extraction_batches_new
                            (id, session_id, start_exchange_index, end_exchange_index,
                             extraction_json, status, error, created_at, updated_at,
                             start_message_index, end_message_index)
                            SELECT id, session_id, start_exchange_index, end_exchange_index,
                                   extraction_json, status, error, created_at, updated_at,
                                   start_message_index, end_message_index
                            FROM extraction_batches
                            WHERE start_exchange_index IS NOT NULL
                              AND end_exchange_index IS NOT NULL""")
                        after_count = conn.execute(
                            "SELECT COUNT(*) FROM extraction_batches_new"
                        ).fetchone()[0]
                        dropped = before_count - after_count
                        if dropped > 0:
                            logger.warning(
                                "Migration: Dropped %d duplicate extraction_batches "
                                "during table recreation (new UNIQUE constraint on "
                                "exchange_index)", dropped
                            )
                        conn.execute("DROP TABLE extraction_batches")
                        conn.execute(
                            "ALTER TABLE extraction_batches_new "
                            "RENAME TO extraction_batches"
                        )
                        conn.execute(
                            "CREATE INDEX idx_extraction_batches_status "
                            "ON extraction_batches(status)"
                        )
                        conn.execute(
                            "CREATE INDEX idx_extraction_batches_session "
                            "ON extraction_batches(session_id)"
                        )
                        conn.execute(
                            "CREATE INDEX idx_batches_covered ON "
                            "extraction_batches(session_id, status, "
                            "start_exchange_index, end_exchange_index)"
                        )
                        conn.execute("COMMIT")
                        logger.info(
                            "Migration 005: Recreated extraction_batches with "
                            "exchange_index UNIQUE constraint"
                        )
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise

                if _needs_message_summaries_migration(conn):
                    conn.execute("BEGIN")
                    try:
                        conn.execute("""CREATE TABLE message_summaries_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            message_index INTEGER NOT NULL,
                            end_message_index INTEGER DEFAULT 0,
                            role TEXT NOT NULL,
                            summary TEXT NOT NULL,
                            tool_summary TEXT,
                            content_hash TEXT NOT NULL,
                            model_used TEXT,
                            token_estimate INTEGER,
                            rendered_full_tokens INTEGER DEFAULT 0,
                            rendered_action_tokens INTEGER DEFAULT 0,
                            rendered_narrative_tokens INTEGER DEFAULT 0,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            exchange_index INTEGER NOT NULL,
                            UNIQUE(session_id, exchange_index)
                        )""")
                        before_count = conn.execute(
                            "SELECT COUNT(*) FROM message_summaries "
                            "WHERE exchange_index IS NOT NULL"
                        ).fetchone()[0]
                        conn.execute("""INSERT OR IGNORE INTO message_summaries_new
                            (id, session_id, exchange_index, end_message_index, role,
                             summary, tool_summary, content_hash, model_used,
                             token_estimate, rendered_full_tokens,
                             rendered_action_tokens, rendered_narrative_tokens,
                             created_at, updated_at, message_index)
                            SELECT id, session_id, exchange_index, end_message_index,
                                   role, summary, tool_summary, content_hash,
                                   model_used, token_estimate, rendered_full_tokens,
                                   rendered_action_tokens, rendered_narrative_tokens,
                                   created_at, updated_at, message_index
                            FROM message_summaries WHERE exchange_index IS NOT NULL""")
                        after_count = conn.execute(
                            "SELECT COUNT(*) FROM message_summaries_new"
                        ).fetchone()[0]
                        dropped = before_count - after_count
                        if dropped > 0:
                            logger.warning(
                                "Migration: Dropped %d duplicate message_summaries "
                                "during table recreation (new UNIQUE constraint on "
                                "exchange_index)", dropped
                            )
                        conn.execute("DROP TABLE message_summaries")
                        conn.execute(
                            "ALTER TABLE message_summaries_new "
                            "RENAME TO message_summaries"
                        )
                        conn.execute(
                            "CREATE INDEX idx_msg_summaries_session "
                            "ON message_summaries(session_id, exchange_index)"
                        )
                        conn.execute(
                            "CREATE INDEX idx_msg_summaries_hash "
                            "ON message_summaries(content_hash)"
                        )
                        conn.execute("COMMIT")
                        logger.info(
                            "Migration 005: Recreated message_summaries with "
                            "exchange_index UNIQUE constraint"
                        )
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise

                conn.execute("""
                    UPDATE conversation_summaries SET last_exchange_index = (
                        SELECT COALESCE(MAX(exchange_index), -1)
                        FROM message_summaries
                        WHERE message_summaries.session_id =
                              conversation_summaries.session_id
                    )
                """)
                conn.commit()

                _migrate_add_chain_hash(conn)
                _migrate_recreate_conversation_summaries(conn)
                _backfill_chain_hash(conn)
                _migrate_add_indexes(conn)
                conn.commit()

                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA wal_autocheckpoint = 1000")

                if path == ":memory:":
                    keeper = _checkout_global(
                        path,
                        maintenance_token=maintenance,
                    )
            except BaseException as exc:
                body_exception = exc
                raise
            finally:
                # Entire block remains inside path_maintenance. The
                # maintenance token is still held at this point. After
                # a successful body, if either cleanup step fails, the
                # exception propagates to the caller. If the body
                # itself failed, body_exception remains primary.
                if conn is not None:
                    try:
                        conn.close()
                    except BaseException as exc:
                        close_exception = exc
                        logger.error(
                            "init_db: raw conn close failed for %s: %s",
                            path,
                            exc,
                        )
                    finally:
                        conn = None

                if keeper is not None:
                    try:
                        keeper._pool.checkin(keeper)
                    except BaseException as exc:
                        keeper_exception = exc
                        logger.error(
                            "init_db: keeper checkin failed for %s: %s",
                            path,
                            exc,
                        )
                    finally:
                        keeper = None

                if body_exception is None:
                    if close_exception is not None:
                        raise close_exception
                    if keeper_exception is not None:
                        raise keeper_exception

            logger.info("Database schema initialized")
            return True
    except BaseException as exc:
        logger.error("Failed to initialize database: %s", exc)
        # Preserve the established public contract:
        # migration failures return False; cleanup failures following
        # successful migration are surfaced to the caller. Use identity
        # check to distinguish which category the exception belongs to.
        if body_exception is None and (
            exc is close_exception or exc is keeper_exception
        ):
            raise
        return False


def check_integrity(db_path=None):
    """Run SQLite integrity check on the database.

    Executes PRAGMA integrity_check to verify database consistency.
    Typically called on startup to detect corruption.

    Args:
        db_path: Optional database path. Defaults to the configured path.

    Returns:
        bool: True if integrity check passes, False on failure or error.
    """
    path = str(db_path or _get_db_path())
    try:
        conn = _checkout_global(path)
    except Exception as exc:
        logger.error("Integrity check could not acquire connection: %s", exc)
        return False
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        is_ok = bool(result and result[0] == "ok")
        if not is_ok:
            logger.warning(
                "Database integrity check failed: %r",
                result,
            )
        return is_ok
    except Exception as exc:
        logger.error("Integrity check error: %s", exc)
        return False
    finally:
        try:
            return_connection(conn)
        except Exception as exc:
            logger.warning("Integrity check return_connection failed: %s", exc)
