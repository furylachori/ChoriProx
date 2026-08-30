"""SQLite CRUD + public façade for the memory DB.

Message/session store (store_message, store_messages_batch, get_last_messages,
touch_session, …) and the re-export surface for the db_pool (connection/
transaction lifecycle) and db_migrations (init_db/SCHEMA_SQL) leaves. Public
callers should use this façade; submodules needing pool/transaction internals
import memory.db_pool directly. See src/memory/AGENTS.md "DB module layout".
"""

import calendar
import hashlib
import json
import os
import sqlite3
import threading
import time

from proxy.utils import _now_iso_sqlite
import logging

# Tier 4 (4c-1): connection pool + transaction lifecycle moved to db_pool.py
# Re-exported here as the public façade. All existing import sites keep working.
from memory.db_pool import (
    ConnectionPool,
    DEFAULT_DB_PATH,
    _create_raw_connection,
    _ensure_pool,
    _get_db_path,
    _pool_lock,
    close_all_connections,
    connection,
    exclusive_transaction,
    get_connection,
    immediate_transaction,
    return_connection,
    transaction,
)

from memory.db_migrations import (
    SCHEMA_SQL,
    _backfill_chain_hash,
    _backfill_one_session,
    _migrate_add_chain_hash,
    _migrate_add_indexes,
    _migrate_recreate_conversation_summaries,
    _needs_exchange_index_migration,
    _needs_message_summaries_migration,
    backfill_sessions_table,
    check_integrity,
    init_db,
)


logger = logging.getLogger("memory.db")

import types as _types


class _DBModule(_types.ModuleType):
    """Façade: forwards _pool reads/writes to db_pool (reassigned by _ensure_pool).

    _pool MUST NOT be a real attribute here — a real attr shadows __getattr__
    and goes stale after _ensure_pool rebinds db_pool._pool. All other moved
    symbols (connection, transaction, …) are real re-exports in __dict__ and
    are found by normal lookup before __getattr__ fires.
    """

    def __getattr__(self, name):
        if name == "_pool":
            from memory import db_pool
            return db_pool._pool
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name == "_pool":
            from memory import db_pool
            db_pool._pool = value
        else:
            super().__setattr__(name, value)


# Install on the real module object (NOT an object replacement) — keeps
# isinstance(memory.db, types.ModuleType) and importlib.reload(memory.db) working.
import sys as _sys
_sys.modules[__name__].__class__ = _DBModule

# Phase 3 — removed

# Global registry of all SQLite connections for shutdown drain
_all_connections: list[sqlite3.Connection] = []
_all_connections_lock = threading.Lock()




def _normalize_content(raw_content):
    """Normalize OpenAI message content to a string.

    Handles all content formats produced by OpenAI messages:
    - None -> ""
    - str -> str
    - list of parts (e.g. [{"type": "text", "text": "..."}]) -> joined text
    """
    if raw_content is None:
        return ""
    elif isinstance(raw_content, list):
        parts = []
        for part in raw_content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    else:
        return str(raw_content)


def store_message(
    session_id,
    message_index,
    role,
    content=None,
    content_hash=None,
    tool_calls_json=None,
    tool_call_id=None,
):
    """Store a single message in the messages table with upsert semantics.

    Uses INSERT with ON CONFLICT to handle duplicates. Marks edited messages
    with is_edited=1 when content hash changes. On failure, logs the error
    and continues (message lost, proxy stays up).

    Args:
        session_id: Session identifier for the conversation.
        message_index: Sequential index of the message within the session.
        role: Message role (e.g., 'user', 'assistant', 'system', 'tool').
        content: Message content text. Normalized from various OpenAI formats.
        content_hash: Optional SHA256 hash prefix for change detection.
        tool_calls_json: Optional JSON string of tool calls made by the model.
        tool_call_id: Optional identifier linking to a specific tool call.

    Returns:
        bool: True if the message was stored successfully, False on failure.
    """
    try:
        # Normalize content to handle all OpenAI message formats
        content = _normalize_content(content)
        # Auto-compute content_hash if not provided, matching store_messages_batch
        if content_hash is None and content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        now = _now_iso_sqlite()
        role_lower = (role or "").lower() or role
        with immediate_transaction(pooled=True) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(exchange_index), -1) FROM messages WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            current_ei = row[0]
            if role_lower == "user":
                current_ei += 1

            try:
                from proxy.chain_hash import _message_hash
                if role_lower == "system":
                    chain_hash = None
                else:
                    # Look back to the most recent NON-NULL chain_hash, matching
                    # compute_chain() which skips system rows and continues chaining
                    # from the prior non-system message. The previous message_index
                    # may be system (chain_hash NULL) — using it would reset the chain.
                    prev_row = conn.execute(
                        "SELECT chain_hash FROM messages "
                        "WHERE session_id = ? AND message_index < ? AND chain_hash IS NOT NULL "
                        "ORDER BY message_index DESC LIMIT 1",
                        (session_id, message_index),
                    ).fetchone()
                    prev_chain_hash = prev_row[0] if prev_row else None
                    msg_for_hash = {
                        "role": role_lower,
                        "content": content,
                        "tool_calls": json.loads(tool_calls_json) if tool_calls_json else None,
                    }
                    msg_hash = _message_hash(msg_for_hash)
                    if prev_chain_hash is None:
                        chain_hash = msg_hash
                    else:
                        chain_hash = hashlib.sha256(
                            f"{prev_chain_hash}:{msg_hash}".encode("utf-8")
                        ).hexdigest()[:16]
            except Exception:
                logger.warning(
                    "chain_hash computation failed; storing NULL session_id=%s idx=%d",
                    session_id, message_index, exc_info=True,
                )
                chain_hash = None

            conn.execute(
                """INSERT INTO messages (session_id, message_index, role, content,
                   content_hash, tool_calls_json, tool_call_id, created_at, exchange_index, chain_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, message_index) DO UPDATE SET
                   content = excluded.content,
                   content_hash = excluded.content_hash,
                   role = excluded.role,
                   chain_hash = excluded.chain_hash,
                   updated_at = excluded.created_at,
                   is_edited = 1
                   WHERE excluded.content_hash IS NOT NULL AND (messages.content_hash IS NULL OR excluded.content_hash != messages.content_hash)""",
                (
                    session_id,
                    message_index,
                    role_lower,
                    content,
                    content_hash,
                    tool_calls_json,
                    tool_call_id,
                    now,
                    current_ei,
                    chain_hash,
                ),
            )
        return True
    except Exception as e:
        logger.error("Failed to store message: %s session_id=%s", e, session_id)
        return False





def get_last_messages(session_id, n=10):
    """Retrieve the last N messages for a session in chronological order.

    Queries messages in descending index order then reverses to return
    them chronologically. Returns an empty list on any failure.

    Args:
        session_id: Session identifier for the conversation.
        n: Maximum number of messages to retrieve. Defaults to 10.

    Returns:
        list[dict]: List of message dictionaries ordered by message_index
            ascending. Empty list on failure or if no messages exist.
    """
    try:
        with connection(pooled=True) as conn:
            rows = conn.execute(
                """SELECT * FROM messages WHERE session_id = ?
                   ORDER BY message_index DESC LIMIT ?""",
                (session_id, n),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]
    except Exception as e:
        logger.error("Failed to get messages: %s session_id=%s", e, session_id)
        return []


def touch_session(session_id: str) -> None:
    """Register a session as existing in the sessions table.

    Uses INSERT OR IGNORE to ensure the session is tracked without
    overwriting existing records. Part of F1 security measure to
    prevent privilege escalation via forged session IDs.

    Args:
        session_id: Session identifier to register.
    """
    try:
        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,)
            )
    except Exception as e:
        logger.warning("Failed to touch session %s: %s", session_id[:8], e)


def session_exists(session_id: str) -> bool:
    """Check if a session exists in the sessions table.

    Part of F1 security measure to prevent privilege escalation via
    forged session IDs. Returns False on any database error.

    Args:
        session_id: Session identifier to check.

    Returns:
        bool: True if the session exists in the database, False otherwise
            or on error.
    """
    try:
        with connection(pooled=True) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            return cursor.fetchone() is not None
    except Exception as e:
        logger.warning("Failed to check session existence %s: %s", session_id[:8], e)
        return False


PROXY_MESSAGE_RETENTION_DAYS = int(os.environ.get("PROXY_MESSAGE_RETENTION_DAYS", "7"))
PROCESSING_TIMEOUT_SECONDS = 300


def reset_stale_processing_batches():
    """Reset extraction batches stuck in 'processing' status back to pending.

    Finds batches that have been in 'processing' status longer than
    PROCESSING_TIMEOUT_SECONDS (300 seconds) and resets them to 'pending'
    with error cleared. Logs a warning when stale batches are found.

    Returns:
        int: Number of batches reset to pending status. Returns 0 on error.
    """
    try:
        with immediate_transaction(pooled=True) as conn:
            cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - PROCESSING_TIMEOUT_SECONDS),
            )
            result = conn.execute(
                """UPDATE extraction_batches
                   SET status = 'pending', error = NULL
                   WHERE status = 'processing' AND updated_at < ?""",
                (cutoff,)
            )
            reset_count = result.rowcount
            if reset_count > 0:
                logger.warning("Reset %d stale processing batches older than %d seconds",
                             reset_count, PROCESSING_TIMEOUT_SECONDS)
            return reset_count
    except Exception as e:
        logger.error("Failed to reset stale processing batches: %s", e)
        return 0


def cleanup_expired_messages():
    """Purge messages older than PROXY_MESSAGE_RETENTION_DAYS.

    Called periodically from cleanup_old_sessions() or app startup.
    SQLite is ephemeral — graph is permanent semantic memory. Also
    cascades deletion to orphaned message_summaries rows.

    Returns:
        int: Number of messages deleted. Returns 0 on error.
    """
    try:
        with immediate_transaction(pooled=True) as conn:
            cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - PROXY_MESSAGE_RETENTION_DAYS * 86400),
            )
            result = conn.execute(
                "DELETE FROM messages WHERE created_at < ?", (cutoff,)
            )
            conn.execute(
                "DELETE FROM message_summaries WHERE NOT EXISTS "
                "(SELECT 1 FROM messages m WHERE m.session_id = message_summaries.session_id AND m.message_index = message_summaries.message_index)"
            )
            deleted = result.rowcount
            if deleted > 0:
                logger.info("Purged %d expired messages (retention=%dd)", deleted, PROXY_MESSAGE_RETENTION_DAYS)
            return deleted
    except Exception as e:
        logger.error("Failed to cleanup expired messages: %s", e)
        return 0


def get_max_message_index(session_id):
    """Get the next message_index for a session from SQLite (source of truth).

    Args:
        session_id: Session identifier for the conversation.

    Returns:
        int: The next available message index (max + 1). Returns 0 on error.
    """
    try:
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(message_index), -1) + 1 AS next_idx FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["next_idx"] if row else 0
    except Exception as e:
        logger.error("Failed to get max message index: %s session_id=%s", e, session_id)
        return 0


def store_messages_batch(session_id, messages, start_index, conn=None):
    """Store multiple messages using INSERT OR REPLACE for idempotency.

    Uses upsert semantics to handle duplicates. Marks edited messages with
    is_edited=1 when content hash, role, tool_calls, or tool_call_id changes.
    Auto-computes content_hash for each message. Tracks exchange_index which
    increments on each role=user message.

    Args:
        session_id: Session identifier for the conversation.
        messages: List of message dictionaries with keys: role, content,
            tool_calls (optional), tool_call_id (optional).
        start_index: Starting message_index for the first message in the batch.
        conn: Optional existing connection (from exclusive_transaction) to
            reuse. When provided, the caller owns the transaction.

    Returns:
        int: Number of messages stored. Returns 0 on error.
    """
    try:
        now = _now_iso_sqlite()

        def _do_store(c):
            cursor = c.cursor()

            row = cursor.execute(
                "SELECT COALESCE(MAX(exchange_index), -1) AS max_ei FROM messages WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            current_exchange_idx = row["max_ei"] if row else -1

            # Look back to the most recent NON-NULL chain_hash, matching
            # compute_chain() which skips system rows and continues chaining
            # from the prior non-system message. The previous message_index
            # may be system (chain_hash NULL) — using it would reset the chain.
            prev_row = cursor.execute(
                "SELECT chain_hash FROM messages "
                "WHERE session_id = ? AND message_index < ? AND chain_hash IS NOT NULL "
                "ORDER BY message_index DESC LIMIT 1",
                (session_id, start_index),
            ).fetchone()
            prev_chain_hash = prev_row[0] if prev_row else None

            for i, msg in enumerate(messages):
                idx = start_index + i
                content = _normalize_content(msg.get("content"))
                tool_calls_json = (
                    json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None
                )
                tool_call_id = msg.get("tool_call_id")
                content_hash = None
                if content:
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                role = (msg.get("role", "") or "").lower() or msg.get("role", "")

                if role == "user":
                    current_exchange_idx += 1

                try:
                    from proxy.chain_hash import _message_hash
                    if role == "system":
                        chain_hash = None
                    else:
                        msg_hash = _message_hash(msg)
                        if prev_chain_hash is None:
                            chain_hash = msg_hash
                        else:
                            chain_hash = hashlib.sha256(
                                f"{prev_chain_hash}:{msg_hash}".encode("utf-8")
                            ).hexdigest()[:16]
                        prev_chain_hash = chain_hash
                except Exception:
                    logger.warning(
                        "chain_hash computation failed; storing NULL session_id=%s idx=%d",
                        session_id, idx, exc_info=True,
                    )
                    chain_hash = None

                cursor.execute(
                    """INSERT INTO messages (session_id, message_index, role, content,
                       content_hash, tool_calls_json, tool_call_id, created_at, exchange_index, chain_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id, message_index) DO UPDATE SET
                       role = excluded.role,
                       content = excluded.content,
                       content_hash = excluded.content_hash,
                       tool_calls_json = excluded.tool_calls_json,
                       tool_call_id = excluded.tool_call_id,
                       updated_at = excluded.created_at,
                       exchange_index = COALESCE(messages.exchange_index, excluded.exchange_index),
                       chain_hash = excluded.chain_hash,
                       is_edited = 1
                       WHERE excluded.content_hash IS NOT NULL AND (
                           messages.content_hash IS NULL
                           OR excluded.content_hash != messages.content_hash
                           OR messages.role != excluded.role
                           OR COALESCE(messages.tool_calls_json, '') != COALESCE(excluded.tool_calls_json, '')
                           OR COALESCE(messages.tool_call_id, '') != COALESCE(excluded.tool_call_id, '')
                       )""",
                    (
                        session_id,
                        idx,
                        role,
                        content,
                        content_hash,
                        tool_calls_json,
                        tool_call_id,
                        now,
                        current_exchange_idx,
                        chain_hash,
                    ),
                )

        if conn is not None:
            _do_store(conn)
        else:
            with exclusive_transaction(pooled=True) as c:
                _do_store(c)
        return len(messages)
    except Exception as e:
        logger.error("Failed to store messages batch: %s session_id=%s", e, session_id)
        return 0








def get_max_exchange_index(session_id):
    """Get the highest exchange_index for a session."""
    try:
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT MAX(exchange_index) as max_ei FROM messages WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            return row["max_ei"] if row and row["max_ei"] is not None else -1
    except Exception:
        return -1


def get_latest_message_index(session_id):
    """Get latest existing message index for session.

    Args:
        session_id: Session identifier for the conversation.

    Returns:
        int: The highest message_index in the session. Returns -1 if no
            messages exist or on error.
    """
    try:
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(message_index), -1) AS latest_idx FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["latest_idx"] if row else -1
    except Exception as e:
        logger.error("Failed to get latest message index: %s session_id=%s", e, session_id)
        return -1


def log_tool_call(session_id, tool_name, arguments, result_status, result_summary="", request_id=None):
    """Log a tool call for audit purposes.

    Records tool invocations in the tool_call_audit table for debugging
    and compliance. Arguments are truncated to 500 chars and result_summary
    to 200 chars to prevent excessive storage.

    Args:
        session_id: Session identifier for the conversation.
        tool_name: Name of the tool that was called.
        arguments: Tool arguments (truncated to 500 chars).
        result_status: Outcome status: 'success', 'error', or 'timeout'.
        result_summary: First 200 chars of the result. Defaults to "".
        request_id: Optional request identifier. Defaults to None.
    """
    try:
        with immediate_transaction(pooled=True) as conn:
            now = _now_iso_sqlite()
            conn.execute(
                """INSERT INTO tool_call_audit 
                   (session_id, tool_name, arguments, result_status, result_summary, timestamp, request_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, tool_name, str(arguments)[:500], result_status, 
                 str(result_summary)[:200], now, request_id)
            )
    except Exception as e:
        logger.debug("Tool audit log failed (non-fatal): %s", e)








# Re-exports for backward compatibility
from memory.db_entities import store_entity, store_fact, store_entities_batch, store_facts_batch
from memory.db_summaries import (
    TOKEN_RATIO,
    get_summary,
    record_extraction_batch,
    get_extraction_batch,
    get_max_end_message_index,
    get_messages_in_range,
    get_batch_age,
    get_batch_status,
    get_last_summarized_exchange_index,
    set_last_summarized_exchange_index,
    update_exchange_metadata,
    get_exchange_metadata,
    get_summary_age,
    update_ancient_history,
    get_ancient_history,
    delete_message_summary,
    get_rich_summaries,
    compute_rendered_tokens,
    store_message_summary,
    get_message_summary_by_hash,
    get_message_content_hash,
    get_max_end_exchange_index,
)
