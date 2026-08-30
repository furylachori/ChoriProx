import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired

from proxy.tool_pipeline.action_envelope import error_envelope, success_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "update_task"


class UpdateTaskAction:
    """Update a task's status by short hash.

    Canonical envelope per spec §11. Loses the old divergent
    ``{completed, status, ...}`` shape; the task's actual status moves
    into ``data.task_status``.
    """

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        short_hash = arguments.get("short_hash", "")
        new_status = arguments.get("new_status", "")

        if not short_hash or not short_hash.strip():
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "update_task requires: short_hash, new_status",
                session_id,
                field="short_hash",
            )
        valid_statuses = {"pending", "in_progress", "complete", "invalid"}
        if new_status not in valid_statuses:
            return error_envelope(
                TOOL, OPERATION, "invalid_status",
                f"new_status must be one of: {', '.join(sorted(valid_statuses))}",
                session_id,
                field="new_status",
            )

        try:
            result = tg.update_task_status(short_hash.strip(), new_status, workspace_id=workspace_id, caller_agent_id=caller_agent_id)
            ok = result.get("ok", False)
            task_status_value = result.get("status", "")
            changed_value = result.get("changed", False)
            if not ok:
                return error_envelope(
                    TOOL, OPERATION, "not_found",
                    f"Task {short_hash.strip()} not found",
                    session_id,
                )
            return success_envelope(
                TOOL,
                OPERATION,
                {
                    "short_hash": short_hash.strip(),
                    "task_status": task_status_value,
                    "changed": changed_value,
                },
                session_id,
            )
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Short hash matched multiple tasks. Use a longer hash or list tasks first.",
                session_id,
            )
        except (TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)