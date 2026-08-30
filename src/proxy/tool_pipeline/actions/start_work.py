import logging
import uuid
from typing import Any

from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "start_work"


class StartWorkAction:
    """Claim a task and start a work session."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.work_sessions import create_work_session
        from memory.task_graph import AmbiguousHashError

        short_hash = (arguments.get("short_hash") or "").strip()
        transferred_from_agent = arguments.get("transferred_from_agent")

        if not short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "start_work requires: short_hash", session_id, field="short_hash",
            )

        if not caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "agent_not_found",
                "Cannot resolve caller agent_id", session_id,
            )

        # ---- C3 mode gate: ticket MAIN cannot call start_work ----
        # Work-mode MAIN runs work sessions; ticket-mode MAIN delegates
        # via the task tool. Without this gate, ticket MAIN calling
        # StartWorkAction.execute directly bypasses the dispatcher layer.
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

        ws_uuid = f"ws-{uuid.uuid4().hex}"

        try:
            result = create_work_session(
                tg=tg,
                ws_uuid=ws_uuid,
                task_short_hash=short_hash,
                agent_id=caller_agent_id,
                session_id=session_id or "",
                workspace_id=workspace_id,
                transferred_from_agent=transferred_from_agent,
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
        except InfrastructureError as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
