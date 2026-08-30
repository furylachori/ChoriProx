"""Session management — chat histories, summaries, cleanup."""
import time
import logging

from proxy.state import (
    chat_summaries,
    chat_histories,
    chat_summaries_lock,
    chat_histories_lock,
)
from proxy.workspace_state import workspace_store

logger = logging.getLogger(__name__)

CLEANUP_INTERVAL = 100
# v5: bumped from 1h to 8h. Workspaces are now global (:Workspace nodes outlive
# any single session), so 1h was too aggressive — taking a 2h lunch broke
# workspace context. 8h covers a full workday. Only per-session in-memory state
# and the :Session node + LAST_USED_WORKSPACE edge are evicted, not :Workspace.
SESSION_TTL = 28800
SUMMARY_PROMPT = """
You are a context summarizer. Your ONLY job is to maintain a running summary of this conversation.

CURRENT SUMMARY:
{current_summary}

NEW MESSAGES:
{new_messages}

Instructions:
1. Combine the existing summary with the new messages
2. Preserve ALL critical information: tasks, decisions, code changes, requirements
3. Remove duplicate information, redundant text, and greetings
4. Keep only what is necessary for continuing the conversation
5. Do NOT add analysis, opinions, or extra content
6. Maintain STRICTLY under {max_tokens} tokens

Return ONLY the updated summary text, no explanations, no formatting.
"""


def touch_session(session_id: str):
    """Record current time as last access for a session. Call under chat_histories_lock."""
    workspace_store.touch(session_id)


def get_chat_summary(session_id: str) -> str:
    """Get summary for a session. Acquires chat_summaries_lock."""
    with chat_summaries_lock:
        return chat_summaries.get(session_id, "")





def get_active_session_count_from_db(hours: int = 1) -> int:
    """Count sessions with activity in the last N hours from SQLite."""
    try:
        from memory.db import connection
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM messages WHERE created_at >= ?",
                (cutoff,),
            ).fetchone()
        return row[0] if row else 0
    except Exception:
        return -1  # sentinel for "unavailable"


def cleanup_old_sessions():
    """Evict stale sessions from in-memory cache.

    chat_histories is a cache backed by SQLite. Evicting a session
    just frees RAM — the next request reloads from the DB.

    Lock ordering: acquires chat_histories_lock THEN chat_summaries_lock
    (consistent with documented lock ordering: histories before summaries).
    """
    now = time.time()
    expired = []

    with chat_histories_lock:
        for sid in list(chat_histories.keys()):
            last = workspace_store.get_last_access(sid) or 0
            if now - last > SESSION_TTL:
                expired.append(sid)

    from memory.workspace_persistence import cleanup_neo4j_sessions

    popped_ids = []
    if expired:
        with chat_histories_lock:
            for sid in expired:
                chat_histories.pop(sid, None)
                if workspace_store.get_last_access(sid) is not None:
                    workspace_store.pop_session(sid)
                    popped_ids.append(sid)
        with chat_summaries_lock:
            for sid in expired:
                chat_summaries.pop(sid, None)

    # --- all locks released ---
    # v5: cleanup_neo4j_sessions deletes :Session + edges (not :Workspace)
    if popped_ids:
        cleanup_neo4j_sessions(popped_ids)
    
    logger.info("Evicted %d stale sessions from memory cache", len(expired))

    # Cleanup stale task modes
    from proxy.agent.agent_registry import registry as agent_registry
    agent_registry.cleanup_stale_ticket_modes()

    # Cleanup stale sync locks for evicted sessions
    if expired:
        try:
            import importlib
            sync_mod = importlib.import_module('memory.sync')
            sync_mod._cleanup_sync_locks(set(expired))
        except Exception as e:
            logger.debug("Sync lock cleanup skipped: %s", e)

    # Cleanup stale workspace state (MEM-001: prevents unbounded growth)
    from proxy.state import cleanup_stale_workspaces
    try:
        n = cleanup_stale_workspaces()
        if n > 0:
            logger.info("cleanup_old_sessions: removed %d stale workspace sessions", n)
    except Exception as e:
        logger.debug("Workspace cleanup skipped: %s", e)

    # Cleanup stale rate-limit entries
    from proxy.state import command_rate_limiter, escalation_rate_limiter, skill_rate_limiter, skill_ip_rate_limiter, task_rate_limiter
    command_rate_limiter.cleanup()
    escalation_rate_limiter.cleanup()
    skill_rate_limiter.cleanup()
    skill_ip_rate_limiter.cleanup()
    task_rate_limiter.cleanup()


def reload_session_from_db(session_id: str) -> list:
    """Reload a session's messages from SQLite into chat_histories.

    Returns the loaded messages (or empty list if not in DB).
    Caller should hold chat_histories_lock.
    """
    try:
        from memory.db import get_last_messages
        messages = get_last_messages(session_id, n=500)
        if messages:
            chat_histories[session_id] = messages
            workspace_store.touch(session_id)
            logger.debug("Reloaded session %s from SQLite (%d messages)", session_id, len(messages))
        return messages
    except Exception:
        logger.warning("Failed to reload session %s from SQLite", session_id, exc_info=True)
        return []


def search_chat(session_id: str, query: str, limit: int = 10) -> list:
    ...


def get_last_n_chats(session_id: str, n: int = 5) -> list:
    ...
