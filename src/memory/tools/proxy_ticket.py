"""
Memory tool handlers for proxy_ticket execution.

Contains execute_ticket_tool() and its direct helpers.
requery_llm() lives in proxy/requery.py (same package as proxy.app, eliminating the cross-package coupling).
recall() and search() live in memory/tools/recall.py and memory/tools/search.py.
"""

import logging
import uuid
from typing import Optional

from proxy.context import _get_task_graph_singleton
from memory.query_executor import InfrastructureError

logger = logging.getLogger("memory.tools")

_graph_client = None


def _get_graph():
    global _graph_client
    if _graph_client is None:
        from memory.graph import Neo4jClient

        _graph_client = Neo4jClient()
    return _graph_client


def _generate_id():
    return uuid.uuid4().hex


def _find_parent_workspace(subagent_session_id: str) -> Optional[str]:
    """Find the workspace of a parent agent in ticket mode for sub-agent inheritance.

    Sub-agents spawned via [[__SUBAGENT__:SKILL:ticket:...]] canary inherit
    ticket mode but not the workspace_name (the SKILL canary format doesn't
    carry it). This helper scans the agent registry for a non-subagent
    agent with an active ticket mode and returns its workspace_name.

    Returns the parent's workspace_name, or None if no parent found.
    """
    try:
        from proxy.agent.agent_registry import registry as _registry
        from proxy.workspace_state import DEFAULT_WORKSPACE
    except ImportError:
        return None

    candidates = []
    try:
        agents = getattr(_registry, "_agents", {}) or {}
    except Exception:
        return None

    for sid, agent in list(agents.items()):
        if sid == subagent_session_id:
            continue
        if getattr(agent, "is_subagent", False):
            continue
        try:
            mode = agent.snapshot_mode()
        except Exception:
            continue
        if mode is None:
            continue
        mode_name = getattr(mode, "name", None)
        if mode_name != "ticket":
            continue
        skill_state = getattr(mode, "skill_state", {}) or {}
        ws = skill_state.get("workspace_name")
        if ws and ws != DEFAULT_WORKSPACE:
            try:
                ts = float(getattr(agent, "_ticket_mode_ts", 0) or 0)
            except Exception:
                ts = 0.0
            candidates.append((ts, ws))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


from proxy.tool_pipeline.action_registry import register_action
from proxy.tool_pipeline.actions import (
    CreateTasksAction,
    UpdateTaskAction,
    ListTasksAction,
    GetTaskAction,
    EditAction,
    SetParentAction,
    ClearParentAction,
    ClearBlockedByAction,
    SetPriorityAction,
    AppendReviewNoteAction,
)
register_action("create_tasks", CreateTasksAction)
register_action("update_task", UpdateTaskAction)
register_action("list_tasks", ListTasksAction)
register_action("get_task", GetTaskAction)
register_action("edit", EditAction)
register_action("set_parent", SetParentAction)
register_action("clear_parent", ClearParentAction)
register_action("clear_blocked_by", ClearBlockedByAction)
register_action("set_priority", SetPriorityAction)
register_action("append_review_note", AppendReviewNoteAction)


def execute_ticket_tool(operation, arguments, session_id=None, client_hint=""):
    """Execute a decomposed ticket tool with workspace awareness.

    This is the shared implementation for all decomposed ticket tool
    paths (proxy_ticket_list_tasks, proxy_ticket_create_tasks, etc.).

    Args:
        operation: The action name (e.g., 'list_tasks', 'create_tasks').
        arguments: The parsed tool arguments dict.
        session_id: The current session ID.
        client_hint: Client identifier for cross-session workspace persistence.

    Returns:
        A JSON-serializable result dict with session_id for debugging.
    """
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.tool_pipeline.action_envelope import error_envelope

    caller_agent_id = None
    if session_id:
        try:
            _agent = agent_registry.get_or_load(session_id)
            if _agent is None:
                logger.warning("execute_ticket_tool: agent registry returned None for session %s", session_id)
                in_ticket_or_work_mode = False
            else:
                caller_agent_id = _agent.agent_id
                _mode, _ticket_ts = _agent.snapshot_mode_and_ticket_ts()
                if _mode is None:
                    in_ticket_or_work_mode = _ticket_ts is not None
                else:
                    _mode_name = getattr(_mode, 'name', None)
                    if _mode_name in ('ticket', 'work'):
                        in_ticket_or_work_mode = not _mode.is_expired()
                    else:
                        in_ticket_or_work_mode = False
        except InfrastructureError:
            raise
        except Exception as e:
            logger.warning("execute_ticket_tool: mode check failed: %s", e)
            in_ticket_or_work_mode = False
    else:
        in_ticket_or_work_mode = False

    if not in_ticket_or_work_mode:
        from proxy.workspace_state import workspace_store, DEFAULT_WORKSPACE
        ws_name = DEFAULT_WORKSPACE
        try:
            if session_id:
                ws_name = workspace_store.get_active(session_id, client_hint)
        except Exception:
            logger.warning("execute_ticket_tool: workspace lookup failed", exc_info=True)
        if ws_name and ws_name != DEFAULT_WORKSPACE:
            if operation not in ("list_tasks", "get_task"):
                return error_envelope(
                    "proxy_ticket", operation, "access_denied",
                    (
                        f"proxy_ticket '{operation}' is not available outside ticket or work mode. "
                        "Do not retry this call. If you need task information, "
                        "use list_tasks or get_task (read-only). Tell the user to "
                        "run ^^ticket to enter ticket mode for full task access."
                    ),
                    session_id,
                )
        else:
            return error_envelope(
                "proxy_ticket", operation, "access_denied",
                (
                    "proxy_ticket is not available in this mode. Complete your "
                    "current task with your available tools. Tell the user to "
                    "run ^^ticket if they need task management."
                ),
                session_id,
            )

    # operation is pre-validated by dispatch (from TICKET_TOOL_OPERATIONS)

    # F1: Use singleton to prevent Neo4j connection pool thrashing
    tg = _get_task_graph_singleton()

    # Extract workspace_id from active_mode skill_state
    workspace_id = None
    if session_id:
        try:
            _agent = agent_registry.get_or_load(session_id)
            if _agent is None:
                logger.warning("execute_ticket_tool: agent registry returned None for session %s", session_id)
            else:
                mode_state = _agent.snapshot_mode()
                if mode_state is not None:
                    skill_state = getattr(mode_state, "skill_state", {})
                    workspace_name = skill_state.get("workspace_name")
                else:
                    workspace_name = None
                if not workspace_name:
                    from proxy.workspace_state import workspace_store
                    workspace_name = workspace_store.get_active(session_id, client_hint)
                if (
                    (not workspace_name or workspace_name == "default")
                    and getattr(_agent, "is_subagent", False)
                ):
                    inherited_ws = _find_parent_workspace(session_id)
                    if inherited_ws:
                        workspace_name = inherited_ws
                        logger.info(
                            "execute_ticket_tool: sub-agent %s inherited workspace %r from active parent",
                            session_id[:8], inherited_ws,
                        )
                if workspace_name:
                    workspace_id = workspace_name
        except InfrastructureError:
            raise
        except Exception as e:
            logger.warning("Failed to extract workspace_id from active_mode: %s", e)

    # Dispatch with workspace_id
    from proxy.tool_pipeline.action_registry import dispatch_action
    result = dispatch_action(
        operation, arguments, session_id or "", tg,
        workspace_id=workspace_id, caller_agent_id=caller_agent_id,
        tool="proxy_ticket",
    )
    return result
