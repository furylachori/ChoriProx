"""Read-only get_task action: return the full task record by short_hash.

Ticket 9c151e0f: sub-agents currently cannot verify a specific ticket's state
without mutating it. list_tasks truncates at 16 KB when the workspace has many
tickets. The Phase 3 cleanup sub-agent mutated name fields as a side-channel
lookup and damaged two legit tickets.

get_task(short_hash) returns the full enriched task record (parent, blockers,
review_notes, planning_info) without truncation (planning_info capped at
MAX_GET_TASK_PLANNING_INFO_LEN = 50_000 chars to prevent abuse).

Read-only: no Neo4j writes. The action registry's dispatch_action() catches
AmbiguousHashError and InfrastructureError, so we only need to convert "not
found" into an explicit error dict and sanitize planning_info here.
"""
import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "get_task"


MAX_GET_TASK_PLANNING_INFO_LEN = 50_000


def _truncate_planning_info(value: Any) -> tuple[Any, bool]:
    """Cap planning_info at MAX_GET_TASK_PLANNING_INFO_LEN chars.

    Returns (value, truncated). If value is not a string, passes through
    unchanged with truncated=False. The cap exists to prevent a malicious or
    runaway task record from blowing up the LLM context window; 50 KB is well
    above the 8 KB write cap and matches the typical ticket record.
    """
    if not isinstance(value, str):
        return value, False
    if len(value) <= MAX_GET_TASK_PLANNING_INFO_LEN:
        return value, False
    return value[:MAX_GET_TASK_PLANNING_INFO_LEN], True


class GetTaskAction:
    """Return the full task record (with parent/blockers/review_notes) by short_hash.

    Read-only. Uses TaskGraph.get_task_enriched() to surface parent_short_hash,
    parent_name, blockers, is_blocked, blocked_count, and review_notes alongside
    the raw task fields — the same shape list_tasks exposes PLUS full
    review_notes (list_tasks only returns a summary).
    """

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        short_hash = arguments.get("short_hash", "")
        if not short_hash or not short_hash.strip():
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "short_hash is required for 'get_task'", session_id,
                field="short_hash",
            )

        try:
            task = tg.get_task_enriched(short_hash.strip(), workspace_id=workspace_id)
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Short hash matched multiple tasks. Use a longer hash or list tasks first.",
                session_id, field="short_hash",
            )
        except (TransientError, ServiceUnavailable, SessionExpired) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)

        if not task:
            return error_envelope(
                TOOL, OPERATION, "not_found",
                "Task not found.", session_id, field="short_hash",
            )

        planning_info, planning_info_truncated = _truncate_planning_info(task.get("planning_info"))

        return success_envelope(
            TOOL, OPERATION,
            {
                "short_hash": task.get("short_hash"),
                "uuid": task.get("uuid"),
                "name": task.get("name"),
                "status": task.get("status"),
                "priority": task.get("priority"),
                "summary": task.get("summary", ""),
                "tag": task.get("tag"),
                "planning_info": planning_info,
                "planning_info_truncated": planning_info_truncated,
                "created_at": task.get("created_at"),
                "assigned_agent_id": task.get("assigned_agent_id"),
                "blocked_by": task.get("blockers", []),
                "is_blocked": task.get("is_blocked"),
                "blocked_count": task.get("blocked_count"),
                "parent_short_hash": task.get("parent_short_hash"),
                "parent_name": task.get("parent_name"),
                "review_notes": task.get("review_notes", []),
            },
            session_id,
        )
