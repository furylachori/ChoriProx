"""Cross-cutting eviction coordination.

Called by ChatHistoryCache on eviction. Coordinates mode clearing
across agent registry and workspace state. Uses lazy imports to
avoid circular dependencies.
"""
import logging

logger = logging.getLogger(__name__)


def build_and_process_deferred_clears(evicted_session_ids: list) -> None:
    """Called by ChatHistoryCache on eviction.

    Coordinates mode clearing across agent registry and workspace state.
    Uses lazy imports to avoid circular dependencies.

    Must be called AFTER releasing ``chat_histories_lock`` to prevent
    lock-ordering deadlocks. Acquires ``AgentRegistry._lock`` (position 0)
    to check in-flight status and retrieve agents.

    Also handles workspace state cleanup (F6) for non-in-flight sessions,
    to respect lock ordering.
    """
    if not evicted_session_ids:
        return
    deferred = _build_deferred_clears(evicted_session_ids)
    _process_deferred_clears(deferred)


def _build_deferred_clears(evicted_session_ids: list) -> dict:
    """Build deferred clears for evicted sessions. Returns a dict with
    ``mode_clears`` and ``popped_workspace_ids`` lists.

    Must be called AFTER releasing ``chat_histories_lock`` to prevent
    lock-ordering deadlocks. Acquires ``AgentRegistry._lock`` (position 0)
    to check in-flight status and retrieve agents.

    Workspace state is cleared in-memory (workspace_store.pop_session) but
    the [:LAST_USED_WORKSPACE] edge in Neo4j is intentionally PRESERVED.
    Edge cleanup happens only on explicit workspace leave/deactivate, not
    on chat-history LRU eviction — otherwise a session that briefly leaves
    the in-memory cache loses its sticky workspace and recovery returns
    'default' on the next access.
    """
    if not evicted_session_ids:
        return {"mode_clears": [], "popped_workspace_ids": []}
    deferred = {"mode_clears": [], "popped_workspace_ids": []}
    try:
        from proxy.agent.agent_registry import registry as _agent_registry  # noqa: PLC0415
        from proxy.workspace_state import workspace_store  # noqa: PLC0415
        with _agent_registry._lock:
            for session_id in evicted_session_ids:
                if _agent_registry._in_flight_requests.get(session_id, 0) > 0:
                    logger.debug("Skipping eviction cleanup for in-flight session %s", session_id[:8])
                    continue
                # Pop in-memory workspace state only; keep the Neo4j edge
                # so recovery can re-populate _active on the next access.
                # The edge is removed only on explicit workspace leave
                # (workspace_skill._handle_leave) or permanent session
                # retirement, not on chat-history LRU eviction.
                if workspace_store.get_last_access(session_id) is not None:
                    workspace_store.pop_session(session_id)
                agent = _agent_registry._agents.get(session_id)
                if agent is None:
                    continue
                try:
                    mode_snap = agent.snapshot_mode()
                except (ImportError, AttributeError):
                    mode_snap = None
                if mode_snap is not None:
                    deferred["mode_clears"].append((agent, mode_snap, session_id))
    except Exception:
        logger.exception("Failed to build deferred clears during eviction")
        deferred["popped_workspace_ids"] = []

    return deferred


def _process_deferred_clears(deferred: dict) -> None:
    """Process deferred clears collected during eviction.

    Must be called AFTER releasing ``chat_histories_lock`` to prevent
    lock-ordering deadlocks with on_deactivate callbacks.

    Note: workspace edge cleanup is intentionally NOT triggered here.
    See _build_deferred_clears for rationale.
    """
    for agent, mode_snap, session_id in deferred.get("mode_clears", []):
        try:
            agent.clear_active_mode(expected_mode=mode_snap)
        except Exception:
            logger.exception(
                "Failed to clear active mode for evicted session %s",
                session_id[:8],
            )