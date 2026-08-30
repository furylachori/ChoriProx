import json
import logging
import sqlite3
from typing import Optional

from proxy.utils import _now_iso_sqlite
from memory._db import get_db as _db

logger = logging.getLogger("memory.sync")


def _ensure_extraction_schema():
    """Detect a missing/unreadable extraction_batches schema and recover.

    See ``db_summaries._ensure_extraction_schema`` for the full
    contract. This is the second of two identical copies of the
    helper, one per module. Both are used by the production call
    sites: ``batch_manager`` exposes the high-level batch lifecycle
    and ``db_summaries`` provides the per-session operations.
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


def _raw_sql(conn, query, params=None):
    try:
        cur = conn.execute(query, params or ())
        return cur
    except Exception as e:
        logger.error(f"Raw SQL error: {e}")
        return None


def _update_batch_status(batch_id, status, error=None):
    from memory.db_pool import immediate_transaction

    _ensure_extraction_schema()

    logger.debug(
        "Attempting to update batch %s status to '%s' (error=%s)",
        batch_id,
        status,
        error,
    )

    with immediate_transaction(pooled=True) as conn:
        if error:
            result = _raw_sql(
                conn,
                "UPDATE extraction_batches SET status = ?, error = ? WHERE id = ?",
                (status, error, batch_id),
            )
        else:
            result = _raw_sql(
                conn,
                "UPDATE extraction_batches SET status = ? WHERE id = ?",
                (status, batch_id),
            )

        if result is None:
            logger.error(
                "Failed to update batch %s status to '%s' - SQL execution failed",
                batch_id,
                status,
            )
            raise RuntimeError(f"Failed to update batch {batch_id} status to {status}")

    logger.info(
        "Batch %s status -> %s%s",
        batch_id,
        status,
        f" (error={error})" if error else "",
    )


def _update_batch_extraction(batch_id, extraction_json, status):
    from memory.db_pool import immediate_transaction

    _ensure_extraction_schema()

    with immediate_transaction(pooled=True) as conn:
        result = _raw_sql(
            conn,
            "UPDATE extraction_batches SET extraction_json = ?, status = ? WHERE id = ?",
            (json.dumps(extraction_json), status, batch_id),
        )
        if result is None:
            logger.error("Failed to update extraction JSON for batch %s - SQL error", batch_id)
        return result


def _claim_batch_for_extraction(batch_id: int) -> bool:
    """M5-3a (Window 2): atomic claim for a batch ready for extraction.

    The batch is eligible for extraction through the full lifecycle
    `pending → processing → extracted`. The production flow runs
    `_prepare_extraction_batch` → `_set_batch_processing` (which sets
    status to 'processing') BEFORE `_persist_extraction_results`
    is called, so at the time this claim runs, the status is
    typically 'processing' (or 'pending' on the first run). The
    'extracted', 'completed', and 'failed' states are excluded —
    'extracted'/'completed' mean another worker already claimed or
    finalized (and would duplicate writes if re-claimed), and
    'failed' is reserved for explicit retry.

    The claim is a true atomic transition: only one thread will see
    rowcount > 0. The WHERE clause must NOT include 'extracted'
    itself — including the post-claim state allows the second
    racer to re-match the row (status='extracted' is in the IN
    list) and report rowcount=1 even when no real transition
    happens, which broke M5-3a's "exactly one wins" invariant
    under heavy parallel load.

    Returns True if the row was claimed (status transitioned to
    'extracted'). False if another worker already processed it,
    the batch is already in a terminal state, or the row doesn't
    exist.
    """
    from memory.db_pool import immediate_transaction

    _ensure_extraction_schema()

    with immediate_transaction(pooled=True) as conn:
        result = _raw_sql(
            conn,
            """UPDATE extraction_batches SET status = 'extracted', updated_at = ?
               WHERE id = ? AND status IN ('pending', 'processing')""",
            (_now_iso_sqlite(), batch_id),
        )
        if result is None:
            logger.error("Failed to claim batch %s for extraction", batch_id)
            return False
        return result.rowcount > 0


def _set_batch_processing(batch_id):
    """Atomically claim a batch by setting status to 'processing'.

    Returns (claimed: bool, was_previously_failed: bool).
    """
    from memory.db_pool import immediate_transaction

    _ensure_extraction_schema()

    with immediate_transaction(pooled=True) as conn:
        row = conn.execute(
            "SELECT status FROM extraction_batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            logger.error("Batch %s not found", batch_id)
            return False, False

        old_status = row["status"]

        result = _raw_sql(
            conn,
            """UPDATE extraction_batches SET status = 'processing', updated_at = ?
               WHERE id = ? AND status IN ('pending', 'failed', 'extracted')""",
            (_now_iso_sqlite(), batch_id),
        )
        if result is None:
            logger.error("Failed to set batch %s to processing - SQL error", batch_id)
            return False, False
        if result.rowcount == 0:
            logger.info("Batch %s already claimed or completed, skipping", batch_id)
            return False, False
        return True, old_status == "failed"


def _update_batch_status_if(
    batch_id,
    from_status,
    to_status,
    error=None,
):
    """M5-3a: conditional UPDATE — set status to `to_status` ONLY IF
    the current status matches `from_status`. Returns True if the
    row was updated, False if the WHERE clause matched no rows
    (another worker won the race, or the row doesn't exist).

    `from_status` may be a single string or a tuple of allowed
    statuses (the SQL is built with `IN (?, ?, ...)` for tuples).
    """
    from memory.db_pool import immediate_transaction

    _ensure_extraction_schema()

    if isinstance(from_status, (tuple, list, set)):
        placeholders = ",".join("?" * len(from_status))
        where_status = f"status IN ({placeholders})"
        params = [batch_id] + list(from_status)
    else:
        where_status = "status = ?"
        params = [batch_id, from_status]

    timestamp = _now_iso_sqlite()
    with immediate_transaction(pooled=True) as conn:
        if error:
            sql = (
                f"UPDATE extraction_batches SET status = ?, error = ?, updated_at = ? "
                f"WHERE id = ? AND {where_status}"
            )
            full_params = [to_status, error, timestamp] + params
        else:
            sql = (
                f"UPDATE extraction_batches SET status = ?, updated_at = ? "
                f"WHERE id = ? AND {where_status}"
            )
            full_params = [to_status, timestamp] + params
        result = _raw_sql(conn, sql, tuple(full_params))
        if result is None:
            logger.error("Failed to conditionally update batch %s status", batch_id)
            return False
        return result.rowcount > 0


def _get_incomplete_batches():
    from memory.db_pool import connection

    _ensure_extraction_schema()

    try:
        with connection(pooled=True) as conn:
            rows = conn.execute(
                """SELECT id, session_id, start_exchange_index, end_exchange_index,
                          start_message_index, end_message_index,
                          extraction_json, status
                   FROM extraction_batches
                    WHERE status IN ('pending', 'failed', 'extracted', 'skipped')
                   ORDER BY id""",
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to query incomplete batches: {e}")
        return []


def _prepare_extraction_batch(session_id, start_exchange, end_exchange,
                              start_message_index=None, end_message_index=None):
    """Find or create extraction batch and claim it. Returns (batch_id, was_previously_failed) or None."""
    existing_batch = _db().get_extraction_batch(session_id, start_exchange, end_exchange)
    if existing_batch:
        batch_id = existing_batch["id"]
        status = existing_batch["status"]
        logger.debug("[SYNC DEBUG] Found existing batch %s with status %s", batch_id, status)
        if status == "completed":
            logger.debug("[SYNC DEBUG] Batch already completed, skipping")
            logger.info("Batch already completed for session %s (%d-%d), skipping", session_id, start_exchange, end_exchange)
            return None

        EXTRACTION_TTL_SECONDS = 600
        batch_age = _db().get_batch_age(batch_id)
        if status != "skipped" and batch_age is not None and batch_age > EXTRACTION_TTL_SECONDS:
            logger.warning(
                "Extraction for batch %s is stale (%ds old). Re-extracting.",
                batch_id, batch_age,
            )
            _update_batch_status(batch_id, "pending")

        logger.info("Resuming existing batch %s with status %s, will re-extract", batch_id, status)

        if status == "skipped":
            from memory.sync import _exchange_range_to_message_range
            start_idx, end_idx = _exchange_range_to_message_range(
                session_id, start_exchange, end_exchange,
            )
            if start_idx is None or end_idx is None:
                _update_batch_status(batch_id, "dead")
                logger.warning(
                    "Batch %s is skipped and exchanges %d-%d have been purged — marked dead session_id=%s",
                    batch_id, start_exchange, end_exchange, session_id
                )
                return None
            existing_msgs = _db().get_messages_in_range(session_id, start_idx, end_idx)
            if not existing_msgs:
                _update_batch_status(batch_id, "dead")
                logger.warning(
                    "Batch %s is skipped and messages %d-%d have been purged — marked dead session_id=%s",
                    batch_id, start_idx, end_idx, session_id
                )
                return None

    else:
        logger.debug("[SYNC DEBUG] Recording new extraction batch...")
        batch_id = _db().record_extraction_batch(
            session_id, start_exchange, end_exchange,
            extraction_json=None, status="pending",
            start_message_index=start_message_index,
            end_message_index=end_message_index,
        )
        if batch_id is None:
            logger.debug("[SYNC DEBUG] Failed to record batch!")
            logger.error("Failed to record batch for session %s", session_id)
            return None
        logger.debug("[SYNC DEBUG] Recorded batch %s", batch_id)

    claimed, was_previously_failed = _set_batch_processing(batch_id)
    if not claimed:
        logger.info("Batch %s already claimed by concurrent sync, skipping", batch_id)
        return None

    return batch_id, was_previously_failed


def _get_messages_for_batch(batch_id):
    """Retrieve messages for a batch's session from SQLite."""
    from memory.db_pool import connection

    _ensure_extraction_schema()

    try:
        with connection(pooled=True) as conn:
            batch = conn.execute(
                "SELECT session_id, end_exchange_index FROM extraction_batches WHERE id=?",
                (batch_id,),
            ).fetchone()
            if not batch:
                return []
            session_id = batch[0]
            end_exchange = batch[1]
            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id=? AND exchange_index <= ?
                   ORDER BY message_index""",
                (session_id, end_exchange),
            ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in rows]
    except Exception as e:
        logger.error("Failed to get messages for batch %s: %s", batch_id, e)
        return []


def _is_covered(session_id, start_exchange, end_exchange):
    """Check if the given range is already covered by a completed batch."""
    try:
        from memory.db_pool import connection

        _ensure_extraction_schema()

        with connection(pooled=True) as conn:
            row = conn.execute(
                """SELECT 1 FROM extraction_batches
                   WHERE session_id = ? AND status = 'completed'
                     AND COALESCE(start_exchange_index, 0) <= ?
                     AND COALESCE(end_exchange_index, 999999999) >= ?
                   LIMIT 1""",
                (session_id, start_exchange, end_exchange),
            ).fetchone()
            return row is not None
    except Exception as e:
        logger.error("Failed to query completed batches: %s", e)
    return False
