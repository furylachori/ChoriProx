"""append_review_note action — append a timestamped review note to a task."""
import logging
from typing import Any, Optional

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "append_review_note"


class AppendReviewNoteAction:
    """Append a timestamped review note to a task."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: Optional[str] = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        short_hash = arguments.get("short_hash")
        note = arguments.get("note")

        if not short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "append_review_note requires: short_hash, note",
                session_id, field="short_hash",
            )

        if not note or not note.strip():
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "append_review_note requires: note (non-empty string)",
                session_id, field="note",
            )

        try:
            result = tg.append_review_note(
                short_hash.strip(),
                note.strip(),
                session_id=session_id or "",
                workspace_id=workspace_id,
            )
            if not result.get("ok"):
                error = result.get("error", "unknown")
                if error == "not_found":
                    return error_envelope(
                        TOOL, OPERATION, "not_found",
                        "Task not found.", session_id,
                        field="short_hash",
                    )
                return error_envelope(
                    TOOL, OPERATION, error,
                    f"Failed to append review note: {error}", session_id,
                    field="short_hash",
                )
            return success_envelope(
                TOOL, OPERATION,
                {
                    "short_hash": short_hash.strip(),
                    "note_count": result.get("note_count"),
                    "updated": True,
                },
                session_id,
            )
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Short hash matched multiple tasks.", session_id,
                field="short_hash",
            )
        except (TransientError, ServiceUnavailable, SessionExpired, InfrastructureError) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
