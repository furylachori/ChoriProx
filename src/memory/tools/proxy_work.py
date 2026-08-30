"""Memory tool handler for proxy_work execution."""
import logging

from memory.query_executor import InfrastructureError

logger = logging.getLogger("memory.tools")

# Action names accepted by proxy_work
WORK_ACTIONS = {
    "start_work", "stop_work", "report_bug", "review_bug",
    "validate_bug", "resolve_bug", "handoff_task"
}

from proxy.tool_pipeline.action_registry import register_action
from proxy.tool_pipeline.actions import (
    StartWorkAction,
    StopWorkAction,
    ReportBugAction,
    ReviewBugAction,
    ValidateBugAction,
    ResolveBugAction,
    HandoffTaskAction,
)
register_action("start_work", StartWorkAction)
register_action("stop_work", StopWorkAction)
register_action("report_bug", ReportBugAction)
register_action("review_bug", ReviewBugAction)
register_action("validate_bug", ValidateBugAction)
register_action("resolve_bug", ResolveBugAction)
register_action("handoff_task", HandoffTaskAction)


def execute_proxy_work(arguments: dict, session_id: str = None, client_hint: str = "") -> dict:
    """Execute a proxy_work tool call.

    Mode gate: proxy_work is work-mode-only (C3). After deletion of the
    Bug #3 stop_work exemption and the work→ticket TTL fallback, no
    legitimate path exists for ticket-mode MAIN agents to call this
    handler. Default, ticket, and workspace mode callers are rejected
    with mode_inactive. The mode activation comes exclusively from
    `^^work` triggers (or canary-inherited sub-agent activation).

    Dispatches by the 'action' field. Resolves caller_agent_id from the
    session's registered agent and extracts workspace_id from active_mode
    skill_state (or workspace_store fallback).
    """
    from proxy.agent.agent_registry import registry as agent_registry

    _agent = agent_registry.get_or_load(session_id) if session_id else None

    # ---- Mode gate: proxy_work is work-mode-only (C3) ----
    # After C3's deletion of the work→ticket TTL fallback and the Bug #3
    # stop_work exemption, no legitimate path exists for ticket-mode MAIN
    # agents to call proxy_work. Work-mode MAIN agents run work sessions;
    # work-mode sub-agents report bugs and review. Default and ticket
    # mode callers are rejected.
    if _agent is None:
        in_work_mode = False
    else:
        try:
            _mode, _ticket_ts = _agent.snapshot_mode_and_ticket_ts()
            if _mode is None:
                in_work_mode = False
            else:
                _mode_name = getattr(_mode, "name", None)
                in_work_mode = _mode_name == "work" and not _mode.is_expired()
        except InfrastructureError:
            raise
        except Exception as e:
            logger.warning("execute_proxy_work: mode check failed: %s", e)
            in_work_mode = False

    if not in_work_mode:
        from proxy.tool_pipeline.action_envelope import error_envelope
        return error_envelope(
            "proxy_work", None, "mode_inactive",
            (
                "This session is not in work mode. Do not retry this call. "
                "Summarize your progress so far and tell the user to run "
                "^^work <task-hash> to start a work session."
            ),
            session_id,
        )

    # ---- Resolve caller_agent_id (after mode gate) ----
    try:
        caller_agent_id = _agent.agent_id
    except InfrastructureError:
        raise
    except Exception as e:
        logger.warning("execute_proxy_work: agent lookup failed: %s", e)
        caller_agent_id = None

    if caller_agent_id is None:
        from proxy.tool_pipeline.action_envelope import error_envelope
        return error_envelope(
            "proxy_work", None, "agent_not_found",
            "Cannot resolve caller agent_id", session_id,
        )

    action = arguments.get("action", "")
    if not action:
        from proxy.tool_pipeline.action_envelope import error_envelope
        return error_envelope(
            "proxy_work", None, "invalid_input",
            "Missing required parameter: action", session_id,
            valid_actions=sorted(WORK_ACTIONS),
        )
    if action not in WORK_ACTIONS:
        from proxy.tool_pipeline.action_envelope import error_envelope
        return error_envelope(
            "proxy_work", None, "unknown_action",
            f"Unknown action: '{action}'", session_id,
            valid_actions=sorted(WORK_ACTIONS),
        )

    from proxy.context import _get_task_graph_singleton
    tg = _get_task_graph_singleton()

    workspace_id = None
    if _agent is not None:
        try:
            mode_state = _agent.snapshot_mode()
            if mode_state is not None:
                skill_state = getattr(mode_state, "skill_state", {}) or {}
                workspace_name = skill_state.get("workspace_name")
            else:
                workspace_name = None
            if not workspace_name:
                from proxy.workspace_state import workspace_store
                workspace_name = workspace_store.get_active(session_id, client_hint)
            if workspace_name:
                workspace_id = workspace_name
        except InfrastructureError:
            raise
        except Exception as e:
            logger.warning("execute_proxy_work: failed to extract workspace_id: %s", e)

    from proxy.tool_pipeline.action_registry import dispatch_action
    result = dispatch_action(
        action, arguments, session_id or "", tg,
        workspace_id=workspace_id, caller_agent_id=caller_agent_id,
        tool="proxy_work",
    )
    return result


def proxy_work(**kwargs):
    """Wrapper for proxy_work handler. Used by tool dispatcher."""
    session_id = kwargs.pop("session_id", None)
    client_hint = kwargs.pop("client_hint", "")
    return execute_proxy_work(kwargs, session_id=session_id, client_hint=client_hint)
