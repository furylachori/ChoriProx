import logging
import re
from typing import Any

from memory.task_graph_common import MAX_SUMMARY_LEN
from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_ticket"
OPERATION = "edit"


class EditAction:
    """Update task fields: name, summary, priority, blocked_by, parent_hash, new_status, tag, planning_info."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        short_hash = arguments.get("short_hash")
        if not short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "short_hash is required for 'edit'", session_id,
                field="short_hash",
            )
        name = arguments.get("name")
        summary = arguments.get("summary")
        priority = arguments.get("priority")
        blocked_by = arguments.get("blocked_by")
        parent_hash = arguments.get("parent_hash")
        new_status = arguments.get("new_status")
        tag = arguments.get("tag")
        planning_info = arguments.get("planning_info")

        if all(v is None for v in [name, summary, priority, blocked_by, parent_hash, new_status, tag, planning_info]):
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "At least one of name, summary, priority, blocked_by, parent_hash, new_status, tag, or planning_info is required for 'edit'",
                session_id,
            )

        if priority is not None and priority not in {"high", "medium", "low"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"priority must be high/medium/low, got '{priority}'", session_id,
                field="priority",
            )

        if new_status is not None and new_status not in {"pending", "in_progress", "complete", "invalid"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"new_status must be pending/in_progress/complete/invalid, got '{new_status}'",
                session_id, field="new_status",
            )

        if name is not None and len(name.strip()) > 200:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"name must be ≤200 characters, got {len(name.strip())}", session_id,
                field="name",
            )

        if summary is not None and len(summary) > MAX_SUMMARY_LEN:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"summary exceeds {MAX_SUMMARY_LEN} characters (got {len(summary)})",
                session_id, field="summary",
            )

        if planning_info is not None:
            from memory.task_graph_common import MAX_PLANNING_INFO_LEN
            if len(planning_info) > MAX_PLANNING_INFO_LEN:
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"planning_info exceeds {MAX_PLANNING_INFO_LEN} characters (got {len(planning_info)})",
                    session_id, field="planning_info",
                )

        if tag is not None and tag.strip():
            if not re.match(r'^[A-Z]$', tag.strip()):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    f"tag must be a single uppercase letter (A-Z), got '{tag}'",
                    session_id, field="tag",
                )

        existing = None
        if blocked_by is not None or parent_hash is not None:
            try:
                existing = tg.get_task_by_short_hash(short_hash.strip(), workspace_id=workspace_id)
            except AmbiguousHashError:
                return error_envelope(
                    TOOL, OPERATION, "ambiguous_hash",
                    "Short hash matched multiple tasks. Use a longer hash.",
                    session_id, field="short_hash",
                )
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
            if not existing:
                return error_envelope(
                    TOOL, OPERATION, "not_found",
                    "Task not found.", session_id, field="short_hash",
                )

        resolved_blockers = None
        if blocked_by is not None:
            if not isinstance(blocked_by, list):
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    "blocked_by must be an array", session_id, field="blocked_by",
                )
            try:
                resolved_blockers = tg.resolve_blocked_by_hashes(
                    blocked_by, exclude_uuid=existing.get("uuid"),
                    workspace_id=workspace_id
                )
            except ValueError as e:
                return error_envelope(
                    TOOL, OPERATION, "invalid_input",
                    str(e), session_id, field="blocked_by",
                )
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)

        resolved_parent = None
        clear_parent = False
        if parent_hash is not None:
            if not parent_hash:
                clear_parent = True
            else:
                try:
                    resolved_parent = tg.resolve_parent_hash(
                        parent_hash, child_uuid=existing.get("uuid"),
                        workspace_id=workspace_id
                    )
                except ValueError as e:
                    return error_envelope(
                        TOOL, OPERATION, "invalid_input",
                        str(e), session_id, field="parent_hash",
                    )
                except (TransientError, ServiceUnavailable, SessionExpired) as e:
                    return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)

        update_kwargs = {}
        if name is not None:
            update_kwargs["name"] = name.strip()
        if summary is not None:
            update_kwargs["summary"] = summary
        if priority is not None:
            update_kwargs["priority"] = priority
        if tag is not None and tag.strip():
            update_kwargs["tag"] = tag.strip()
        if resolved_blockers is not None:
            update_kwargs["blocked_by"] = resolved_blockers
        if resolved_parent is not None:
            update_kwargs["parent_uuid"] = resolved_parent
        if clear_parent:
            update_kwargs["clear_parent"] = True
        if new_status is not None:
            update_kwargs["new_status"] = new_status
        if planning_info is not None:
            update_kwargs["planning_info"] = planning_info

        try:
            success = tg.update_task(short_hash.strip(), workspace_id=workspace_id, caller_agent_id=caller_agent_id, **update_kwargs)
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
