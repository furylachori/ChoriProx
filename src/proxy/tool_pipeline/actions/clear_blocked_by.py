import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "clear_blocked_by"


class ClearBlockedByAction:
    """Remove all blockers from a task."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        short_hash = arguments.get("short_hash")
        if not short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "short_hash is required for 'clear_blocked_by'", session_id,
                field="short_hash",
            )

        try:
            task = tg.get_task_by_short_hash(short_hash.strip(), workspace_id=workspace_id)
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Ambiguous short hash.", session_id, field="short_hash",
            )
        except (TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
        if not task:
            return error_envelope(
                TOOL, OPERATION, "not_found",
                "Task not found.", session_id, field="short_hash",
            )

        try:
            success = tg.update_task(
                short_hash.strip(),
                workspace_id=workspace_id,
                blocked_by=[],
                caller_agent_id=caller_agent_id,
            )
            return success_envelope(
                TOOL, OPERATION,
                {"short_hash": short_hash.strip(), "updated": success},
                session_id,
            )
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Short hash matched multiple tasks.", session_id, field="short_hash",
            )
        except (TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
