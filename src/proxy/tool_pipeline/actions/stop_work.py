import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "stop_work"


class StopWorkAction:
    """Close a work session. Does NOT unassign the task."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.work_sessions import close_work_session
        from memory.task_graph import AmbiguousHashError

        short_hash = (arguments.get("short_hash") or "").strip()
        ws_uuid = (arguments.get("ws_uuid") or "").strip()
        new_status = arguments.get("new_status")
        git_commit_sha = arguments.get("git_commit_sha")

        if not short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "stop_work requires: short_hash, ws_uuid", session_id, field="short_hash",
            )
        if not ws_uuid:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "stop_work requires: ws_uuid (get from list_tasks or prior start_work)",
                session_id, field="ws_uuid",
            )
        if not caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "agent_not_found",
                "Cannot resolve caller agent_id", session_id,
            )

        if new_status is not None and new_status not in {"complete", "invalid"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"new_status must be 'complete' or 'invalid' if provided, got '{new_status}'",
                session_id, field="new_status",
            )

        # ---- C3 mode gate: ticket MAIN cannot call stop_work ----
        # Work-mode MAIN runs work sessions; ticket-mode MAIN delegates
        # via the task tool. Without this gate, ticket MAIN calling
        # StopWorkAction.execute directly bypasses the dispatcher layer.
        from proxy.agent.agent_registry import registry as _agent_registry
        try:
            _agent = _agent_registry.get_or_load(session_id) if session_id else None
        except Exception:
            _agent = None
        if _agent is not None:
            _mode_name = _agent.get_mode_name() if hasattr(_agent, "get_mode_name") else None
            if _mode_name == "ticket":
                return error_envelope(
                    TOOL, OPERATION, "blocked",
                    "Work mode actions are blocked for the MAIN agent in ticket mode. "
                    "Delegate to a work-mode sub-agent via the `task` tool.",
                    session_id,
                )

        try:
            result = close_work_session(
                tg=tg,
                ws_uuid=ws_uuid,
                task_short_hash=short_hash,
                agent_id=caller_agent_id,
                workspace_id=workspace_id,
                new_status=new_status,
                git_commit_sha=git_commit_sha,
            )
            if isinstance(result, dict) and result.get("error"):
                return error_envelope(
                    TOOL, OPERATION,
                    result["error"],
                    result.get("message", result["error"]),
                    session_id,
                )
            return success_envelope(TOOL, OPERATION, result if isinstance(result, dict) else {}, session_id)
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Multiple tasks match this hash", session_id, field="short_hash",
            )
        except (InfrastructureError, TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
