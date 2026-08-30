"""Summary-related database operations for memory system.

Extracted from db.py for modularity. Includes exchange summaries,
extraction batches, exchange metadata, and ancient history.
"""

# IMPORTANT: All imports from memory.db MUST be lazy (inside function bodies)
# to prevent circular import at module load time. db.py re-exports from this
# module at module level, so a module-level `from memory.db import ...` here
# would create an ImportError cycle.
# Pool/transaction imports come from memory.db_pool (pure leaf, no cycle).

import calendar
import logging
import sqlite3
import time
from typing import Optional

from proxy.utils import _now_iso_sqlite

logger = logging.getLogger("memory.db")

TOKEN_RATIO = 0.75


def _ensure_extraction_schema():
    """Detect a missing/unreadable extraction_batches schema and recover.

    The probe is run under a short-lived pool checkout so the rest of
    the system can observe the same schema. The probe lease is
    released BEFORE we call ``init_db`` so ``init_db``'s maintenance
    drain can quiesce the pool. ``init_db`` is called with the same
    path the probe used, not the configured default.

    The ``init_db`` call installs a path-maintenance token that blocks
    other same-path checkouts for the duration of the migration. When
    ``init_db`` returns, the new pool is published and the path gate
    is clear; the post-migration probe verifies the schema is
    actually readable.
    """
    import memory.db as _m
    import memory.db_pool as _p

    needs_init = False
    probe_path: Optional[str] = None

    try:
        with _p.connection(pooled=True) as conn:
            probe_path = conn._pool._db_path
            row = conn.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='extraction_batches'"
            ).fetchone()
            needs_init = row is None
            if not needs_init:
                conn.execute(
                    "SELECT 1 FROM extraction_batches LIMIT 1"
                ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
        needs_init = True

    if not needs_init:
        return

    # Resolve the path to repair. If the probe didn't return a path
    # (e.g. the pool was empty), fall back to the configured default.
    if probe_path is None:
        probe_path = _p._get_db_path()

    logger.warning(
        "extraction_batches schema missing or unreadable on path=%s; "
        "running init_db after releasing probe lease",
        probe_path,
    )

    if not _m.init_db(probe_path):
        raise RuntimeError("Failed to initialize extraction schema")

    with _p.connection(pooled=True) as conn:
        conn.execute(
            "SELECT 1 FROM extraction_batches LIMIT 1"
        ).fetchone()


def get_summary(session_id):
    """[DEPRECATED for v2] Get conversation summary. Returns None on failure.

    In ChoriProx Plus v2, use get_rich_summaries() and get_ancient_history() instead.
    This function remains for v1 rollback compatibility.
    """
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT summary FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["summary"] if row else None
    except Exception as e:
        logger.error("Failed to get summary: %s session_id=%s", e, session_id)
        return None


def record_extraction_batch(
    session_id, start_exchange_index, end_exchange_index,
    extraction_json=None, status="pending",
    start_message_index=None, end_message_index=None
):
    """Record extraction batch. Returns batch_id on success, None on failure."""
    try:
        from memory.db_pool import immediate_transaction
        _ensure_extraction_schema()
        now = _now_iso_sqlite()
        with immediate_transaction(pooled=True) as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO extraction_batches (session_id, start_exchange_index, end_exchange_index, extraction_json, status, created_at, updated_at, start_message_index, end_message_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, start_exchange_index, end_exchange_index, extraction_json, status, now, now, start_message_index, end_message_index),
            )
            rowcount = cursor.rowcount
            lastrowid = cursor.lastrowid
        logger.debug(
            "[DB DEBUG] Insert result: rowcount=%d, lastrowid=%d pool=%s",
            rowcount,
            lastrowid,
            conn._pool._db_path,
        )
        if rowcount == 0:
            existing = get_extraction_batch(session_id, start_exchange_index, end_exchange_index)
            logger.debug(
                "[DB DEBUG] Batch already exists, returning existing id: %s", existing
            )
            return existing["id"] if existing else None
        return lastrowid
    except Exception as e:
        logger.debug("[DB DEBUG] Failed to record extraction batch: %s session_id=%s exchange_indices=[%s,%s]", e, session_id, start_exchange_index, end_exchange_index)
        logger.error("Failed to record extraction batch: %s session_id=%s exchange_indices=[%s,%s]", e, session_id, start_exchange_index, end_exchange_index)
        return None


def get_extraction_batch(session_id, start_exchange, end_exchange):
    """Get existing extraction batch for session_id and exchange indices."""
    try:
        from memory.db_pool import connection
        _ensure_extraction_schema()
        with connection(pooled=True) as conn:
            row = conn.execute(
                """SELECT id, status FROM extraction_batches
                   WHERE session_id = ? AND start_exchange_index = ? AND end_exchange_index = ?
                   LIMIT 1""",
                (session_id, start_exchange, end_exchange),
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to get extraction batch: {e}")
        return None


def get_max_end_message_index(session_id):
    """DEPRECATED: Use get_max_end_exchange_index() instead."""
    return get_max_end_exchange_index(session_id)


def get_messages_in_range(session_id, start_idx, end_idx):
    """Get messages from SQLite within a given index range (inclusive).
    Returns list of dicts with role, content, message_index, content_hash,
    tool_calls_json, tool_call_id, and exchange_index.
    """
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            rows = conn.execute(
                """SELECT message_index, role, content, content_hash,
                          tool_calls_json, tool_call_id, exchange_index
                   FROM messages
                   WHERE session_id = ? AND message_index >= ? AND message_index <= ?
                   ORDER BY message_index""",
                (session_id, start_idx, end_idx),
            ).fetchall()
        return [
            {
                "message_index": r["message_index"],
                "role":         r["role"],
                "content":      r["content"],
                "content_hash": r["content_hash"],
                "tool_calls_json": r["tool_calls_json"],
                "tool_call_id":    r["tool_call_id"],
                "exchange_index":  r["exchange_index"],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error("Failed to get messages in range: %s session_id=%s", e, session_id)
        return []


def get_batch_age(batch_id):
    """Get current age of batch in seconds since creation. Returns None if batch not found."""
    try:
        from memory.db_pool import connection
        _ensure_extraction_schema()
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT created_at FROM extraction_batches WHERE id = ?",
                (batch_id,),
            ).fetchone()
        if not row:
            return None
        created_at = time.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        return time.time() - calendar.timegm(created_at)
    except Exception as e:
        logger.error("Failed to get batch age: %s batch_id=%s", e, batch_id)
        return None


def get_batch_status(batch_id):
    """Get current status of batch. Returns None if batch not found."""
    try:
        from memory.db_pool import connection
        _ensure_extraction_schema()
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT status FROM extraction_batches WHERE id = ?",
                (batch_id,),
            ).fetchone()
        return row["status"] if row else None
    except Exception as e:
        logger.error("Failed to get batch status: %s batch_id=%s", e, batch_id)
        return None


def set_last_summarized_exchange_index(session_id: str, exchange_index: int, conn=None):
    """Track summarization progress by exchange_index, keyed on session_id.

    M5-4: uses INSERT ON CONFLICT so it works on first run when no row
    exists yet (previously this function did a blind UPDATE that affected
    zero rows on first run, leaving the cursor at -1 forever).
    """
    try:
        from memory.db_pool import immediate_transaction
        now = _now_iso_sqlite()

        def _do(c):
            c.execute(
                """INSERT INTO conversation_summaries (session_id, last_exchange_index, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                   last_exchange_index = MAX(last_exchange_index, excluded.last_exchange_index),
                   updated_at = excluded.updated_at""",
                (session_id, exchange_index, now),
            )

        if conn is None:
            with immediate_transaction(pooled=True) as c:
                _do(c)
        else:
            _do(conn)
    except Exception as e:
        logger.error("Failed to set last_exchange_index: %s session_id=%s", e, session_id)


def get_last_summarized_exchange_index(session_id: str, conn=None) -> int:
    """Get the last summarized exchange_index for session_id."""
    try:
        from memory.db_pool import connection

        def _query(c):
            return c.execute(
                "SELECT last_exchange_index FROM conversation_summaries "
                "WHERE session_id = ?",
                (session_id,)
            ).fetchone()

        if conn is None:
            with connection(pooled=True) as c:
                row = _query(c)
        else:
            row = _query(conn)
        return row["last_exchange_index"] if row else -1
    except Exception as e:
        logger.error("Failed to get last_exchange_index: %s session_id=%s", e, session_id)
        return -1


def update_exchange_metadata(session_id: str, exchange_count: int, last_exchange_start: int) -> None:
    """Incremental update — keyed on session_id.
    Uses INSERT ON CONFLICT DO UPDATE."""
    try:
        from memory.db_pool import exclusive_transaction
        now = _now_iso_sqlite()
        with exclusive_transaction(pooled=True) as conn:
            conn.execute(
                """INSERT INTO conversation_summaries (session_id, exchange_count, last_exchange_start, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                   exchange_count = excluded.exchange_count,
                   last_exchange_start = excluded.last_exchange_start,
                   updated_at = excluded.updated_at""",
                (session_id, exchange_count, last_exchange_start, now),
            )
    except Exception as e:
        logger.error("Failed to update exchange metadata: %s session_id=%s", e, session_id)


def get_exchange_metadata(session_id: str) -> tuple:
    """Returns (exchange_count, last_exchange_start) for session_id.
    Defaults to (0, 0) if no row exists."""
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT exchange_count, last_exchange_start FROM conversation_summaries "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return (row[0] or 0, row[1] or 0)
            return (0, 0)
    except Exception as e:
        logger.error("Failed to get exchange metadata: %s session_id=%s", e, session_id)
        return (0, 0)


def get_summary_age(session_id):
    """Get age of last summary in seconds for session_id. Returns None if no summary exists."""
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT updated_at FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            updated_at = time.strptime(row["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
            return time.time() - calendar.timegm(updated_at)
    except Exception as e:
        logger.error("Failed to get summary age: %s session_id=%s", e, session_id)
        return None


def update_ancient_history(session_id: str, ancient_history: str, conn=None) -> bool:
    """Update ancient_history for session_id. Creates row if not exists."""
    try:
        from memory.db_pool import exclusive_transaction
        own_conn = conn is None
        now = _now_iso_sqlite()
        sql = """INSERT INTO conversation_summaries (session_id, ancient_history, updated_at)
                 VALUES (?, ?, ?)
                 ON CONFLICT(session_id) DO UPDATE SET
                 ancient_history = excluded.ancient_history, updated_at = excluded.updated_at"""
        params = (session_id, ancient_history, now)
        if own_conn:
            with exclusive_transaction(pooled=True) as ex_conn:
                ex_conn.execute(sql, params)
        else:
            conn.execute(sql, params)
        return True
    except Exception as e:
        logger.error("Failed to update ancient_history: %s session_id=%s", e, session_id)
        if not own_conn:
            raise
        return False


def get_ancient_history(session_id: str) -> str:
    """Get ancient_history for session_id. Returns empty string if not found."""
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT ancient_history FROM conversation_summaries "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["ancient_history"] if row and row["ancient_history"] else ""
    except Exception as e:
        logger.error("Failed to get ancient_history: %s session_id=%s", e, session_id)
        return ""


def delete_message_summary(session_id: str, exchange_index: int, conn=None) -> bool:
    """Delete a specific message summary by session and exchange_index."""
    try:
        own_conn = conn is None
        if own_conn:
            from memory.db_pool import immediate_transaction
            with immediate_transaction(pooled=True) as conn:
                conn.execute(
                    "DELETE FROM message_summaries WHERE session_id = ? AND exchange_index = ?",
                    (session_id, exchange_index),
                )
        else:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ? AND exchange_index = ?",
                (session_id, exchange_index),
            )
        return True
    except Exception as e:
        logger.error("Failed to delete message summary: %s session_id=%s exchange_index=%s", e, session_id, exchange_index)
        if not own_conn:
            raise
        return False


def get_rich_summaries(session_id: str, before_exchange_index: int = None,
                       before_index: int = None) -> list:
    """Get per-exchange summaries for a session, optionally filtered by exchange_index.
    Returns list of dicts ordered by exchange_index.
    before_index (message_index) is deprecated — use before_exchange_index.
    """
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            conditions = ["session_id = ?"]
            params = [session_id]
            if before_exchange_index is not None:
                conditions.append("exchange_index < ?")
                params.append(before_exchange_index)
            elif before_index is not None:
                conditions.append("message_index < ?")
                params.append(before_index)
            where = " AND ".join(conditions)
            rows = conn.execute(
                f"""SELECT message_index, exchange_index, role, summary, tool_summary, content_hash,
                          token_estimate, rendered_full_tokens, rendered_action_tokens,
                          rendered_narrative_tokens
                   FROM message_summaries
                   WHERE {where}
                   ORDER BY exchange_index""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error("Failed to get rich summaries: %s", e)
        return []


def compute_rendered_tokens(exchange_index, role, summary, tool_summary=None):
    """Pre-compute rendered token estimates at storage time so context assembly is O(1).

    Algorithm from ChoriProx Plus spec §4.1:
    - rendered_full_tokens: "[exchange#{i}] {role}: {summary} {tool_summary}"
    - rendered_action_tokens: "[exchange#{i}] {role}: {summary} {strip_args(tool_summary)}"
    - rendered_narrative_tokens: "[exchange#{i}] {role}: {summary}"
    """
    from memory.prompts.summarize_message import sanitize_tool_calls

    tool_summary = tool_summary or ""

    tools_no_args = tool_summary
    if tool_summary.startswith("[tools:") and tool_summary.endswith("]"):
        inner = tool_summary[7:-1].strip()
        if inner:
            actions = [t.split(":")[0].strip() for t in inner.split(",") if t.strip()]
            tools_no_args = "[tools: " + ", ".join(actions) + "]"
        else:
            tools_no_args = "[tools:]"

    full_line = f"[exchange#{exchange_index}] {role}: {summary} {tool_summary}"
    action_line = f"[exchange#{exchange_index}] {role}: {summary} {tools_no_args}"
    narrative_line = f"[exchange#{exchange_index}] {role}: {summary}"

    rendered_full_tokens = int(len(full_line.split()) / TOKEN_RATIO)
    rendered_action_tokens = int(len(action_line.split()) / TOKEN_RATIO)
    rendered_narrative_tokens = int(len(narrative_line.split()) / TOKEN_RATIO)

    return rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens


def store_message_summary(session_id, exchange_index, role, summary, tool_summary,
                          content_hash, model_used=None, token_estimate=0,
                          end_message_index=0, message_index=None):
    """Store a per-exchange summary. Returns True on success.
    exchange_index is the primary key. message_index is optional and looked up if omitted.
    """
    try:
        from memory.db_pool import immediate_transaction
        with immediate_transaction(pooled=True) as conn:
            if message_index is None:
                row = conn.execute(
                    "SELECT message_index FROM messages WHERE session_id = ? AND exchange_index = ? AND role = 'user' LIMIT 1",
                    (session_id, exchange_index)
                ).fetchone()
                message_index = row["message_index"] if row else 0
            rendered_full, rendered_action, rendered_narrative = compute_rendered_tokens(
                exchange_index, role, summary, tool_summary
            )
            now = _now_iso_sqlite()
            conn.execute(
                """INSERT INTO message_summaries (session_id, message_index, end_message_index, role, summary,
                   tool_summary, content_hash, model_used, token_estimate,
                   rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens,
                   created_at, updated_at, exchange_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, exchange_index) DO UPDATE SET
                   role = excluded.role,
                   summary = excluded.summary,
                   tool_summary = excluded.tool_summary,
                   content_hash = excluded.content_hash,
                   model_used = excluded.model_used,
                   token_estimate = excluded.token_estimate,
                   rendered_full_tokens = excluded.rendered_full_tokens,
                   rendered_action_tokens = excluded.rendered_action_tokens,
                   rendered_narrative_tokens = excluded.rendered_narrative_tokens,
                   updated_at = excluded.updated_at,
                   end_message_index = excluded.end_message_index,
                   exchange_index = excluded.exchange_index
                   WHERE excluded.content_hash != message_summaries.content_hash
                      OR message_summaries.end_message_index < excluded.end_message_index""",
                (session_id, message_index, end_message_index, role, summary, tool_summary,
                 content_hash, model_used, token_estimate,
                 rendered_full, rendered_action, rendered_narrative,
                 now, now, exchange_index),
            )
        return True
    except Exception as e:
        logger.error("Failed to store message summary: %s session_id=%s exchange_index=%s", e, session_id, exchange_index)
        return False


def get_message_summary_by_hash(session_id, exchange_index, content_hash):
    """Check if summary exists with matching content_hash. Returns summary dict or None."""
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                """SELECT id, content_hash FROM message_summaries
                   WHERE session_id = ? AND exchange_index = ? AND content_hash = ?""",
                (session_id, exchange_index, content_hash),
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.error("Failed to check message summary hash: %s session_id=%s", e, session_id)
        return None


def get_message_content_hash(session_id, message_index):
    """Get content_hash of an original message from messages table."""
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT content_hash FROM messages WHERE session_id = ? AND message_index = ?",
                (session_id, message_index),
            ).fetchone()
        return row["content_hash"] if row else None
    except Exception as e:
        logger.error("Failed to get message content hash: %s session_id=%s", e, session_id)
        return None


def get_max_end_exchange_index(session_id):
    """Get the highest end_exchange_index from completed extraction batches."""
    try:
        from memory.db_pool import connection
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT MAX(end_exchange_index) as max_ei FROM extraction_batches WHERE session_id = ? AND status = 'completed'",
                (session_id,)
            ).fetchone()
        return row["max_ei"] if row and row["max_ei"] is not None else -1
    except Exception:
        return -1
