import logging
import secrets
import uuid as uuid_mod
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError
import proxy.state as state

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "report_bug"


class ReportBugAction:
    """Report a bug found during an active work session."""

    @staticmethod
    def execute(
        arguments: dict,
        session_id: str,
        tg: Any,
        workspace_id: str = None,
        caller_agent_id: str = None,
    ) -> dict:
        from memory.bug_tracker import create_bug
        from memory.task_graph import AmbiguousHashError

        task_short_hash = (arguments.get("task_short_hash") or "").strip()
        title = (arguments.get("title") or "").strip()
        description = (arguments.get("description") or "").strip()
        severity = (arguments.get("severity") or "").strip()

        if not task_short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "report_bug requires: task_short_hash, title, description, severity",
                session_id, field="task_short_hash",
            )
        if not title or len(title) > 200:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "report_bug requires: title (non-empty, ≤200 chars)", session_id,
                field="title",
            )
        if not description or len(description) > 5000:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "report_bug requires: description (non-empty, ≤5000 chars)",
                session_id, field="description",
            )
        if severity not in {"critical", "high", "medium", "low"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"report_bug requires: severity (one of: critical, high, medium, low), got '{severity}'",
                session_id, field="severity",
            )
        if not caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "agent_not_found",
                "Cannot resolve caller agent_id", session_id,
            )

        bug_uuid = f"bug-{uuid_mod.uuid4().hex}"
        bug_short_hash = secrets.token_hex(4)  # 8-char hex

        try:
            result = create_bug(
                tg=tg,
                bug_uuid=bug_uuid,
                bug_short_hash=bug_short_hash,
                task_short_hash=task_short_hash,
                title=title,
                description=description,
                severity=severity,
                caller_agent_id=caller_agent_id,
                workspace_id=workspace_id,
                max_reviews=state.PROXY_BUG_REVIEW_COUNT,
            )
            if isinstance(result, dict) and result.get("error"):
                return error_envelope(
                    TOOL, OPERATION,
                    result["error"],
                    result.get("message", result["error"]),
                    session_id,
                )
            return success_envelope(
                TOOL, OPERATION,
                {
                    "bug_uuid": result["uuid"],
                    "bug_short_hash": result["short_hash"],
                },
                session_id,
            )
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Multiple tasks match this hash", session_id,
                field="task_short_hash",
            )
        except (InfrastructureError, TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
