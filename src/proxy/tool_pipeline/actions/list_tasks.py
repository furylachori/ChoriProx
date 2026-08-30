import logging
from typing import Any, Optional

from memory.task_graph_queries import get_recent_review_notes
from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "list_tasks"


def _format_review_notes(notes: Optional[list]) -> Optional[str]:
    """Format review notes as a compact string preview (newest first, max 3)."""
    if not notes:
        return None
    lines = [f"[{n.get('ts', '?')}] {n.get('note', '')}" for n in notes]
    return "\n".join(lines)


class ListTasksAction:
    """List actionable tasks with optional status filter."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        status_filter = arguments.get("status_filter")
        if status_filter is not None:
            if not isinstance(status_filter, list):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    "status_filter must be an array", session_id,
                    field="status_filter",
                )

        tasks = tg.get_actionable_tasks(status_filter=status_filter, workspace_id=workspace_id, limit=500)
        store_status = tg.get_store_status() if hasattr(tg, "get_store_status") else {}

        # Register hashes for MODIFY provenance validation
        if session_id:
            tracker = ToolCallTracker.get_instance(session_id)
            hashes = [t.get("short_hash") for t in tasks if t.get("short_hash")]
            tracker.add_seen_hashes(hashes)

        short_hashes = [t["short_hash"] for t in tasks if t.get("short_hash")]
        notes_map = get_recent_review_notes(
            tg._executor, session_id, workspace_id, short_hashes, limit_per_task=3
        )

        for t in tasks:
            notes = notes_map.get(t.get("short_hash"), [])
            t["review_notes_summary"] = _format_review_notes(notes)
            t["planning_info"] = t.get("planning_info")

        return success_envelope(
            TOOL, OPERATION,
            {
                "tasks": [
                    {
                        "short_hash": t.get("short_hash"),
                        "name": t.get("name"),
                        "priority": t.get("priority"),
                        "status": t.get("status"),
                        "summary": t.get("summary", ""),
                        "tag": t.get("tag"),
                        "blocked_by": t.get("blockers", []),
                        "is_blocked": t.get("is_blocked"),
                        "blocked_count": t.get("blocked_count"),
                        "parent_short_hash": t.get("parent_short_hash"),
                        "parent_name": t.get("parent_name"),
                        "planning_info": t.get("planning_info"),
                        "review_notes_summary": t.get("review_notes_summary"),
                    }
                    for t in tasks
                ],
                "total": len(tasks),
                "store_status": store_status,
            },
            session_id,
        )
