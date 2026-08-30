import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "set_priority"


class SetPriorityAction:
    """Change a task's priority."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        # §7.5 — change task priority
        short_hash = arguments.get("short_hash")
        priority = arguments.get("priority")
        if not short_hash or priority is None:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "set_priority requires: short_hash, priority", session_id,
                field="short_hash" if not short_hash else "priority",
            )
        if priority not in {"high", "medium", "low"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"priority must be high/medium/low, got '{priority}'", session_id,
                field="priority",
            )

        try:
            success = tg.update_task(short_hash.strip(), priority=priority, workspace_id=workspace_id, caller_agent_id=caller_agent_id)
            return success_envelope(
                TOOL, OPERATION,
                {
                    "short_hash": short_hash.strip(),
                    "priority": priority,
                    "updated": success,
                },
                session_id,
            )
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Ambiguous short hash.", session_id, field="short_hash",
            )
        except (TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
