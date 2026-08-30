"""Shared error-response helpers for proxy_ticket / proxy_work actions.

Centralizes the boilerplate that the task-graph actions emit when they hit
a transient Neo4j driver failure during a call (resolve_parent_hash,
get_task_by_short_hash, etc.). Each helper emits the canonical error
envelope (spec §11) so callers stay focused on the happy path.
"""
from __future__ import annotations

from typing import Any

from proxy.tool_pipeline.action_envelope import error_envelope


def infra_error_envelope(tool: str, operation: str, session_id: str) -> dict:
    """Canonical 503-shaped envelope for transient Neo4j driver failures."""
    return error_envelope(
        tool, operation, "infrastructure_error",
        "Database temporarily unavailable", session_id,
    )


def log_and_emit_infra_error(
    logger: Any,
    tool: str,
    operation: str,
    exc: BaseException,
    session_id: str,
) -> dict:
    """Log a transient Neo4j failure at WARNING and return its canonical envelope.

    Args:
        logger: Logger instance — must support .warning().
        tool: Tool name ("proxy_ticket" or "proxy_work").
        operation: Operation/action name used in the log message.
        exc: The exception that triggered the failure.
        session_id: Session id of the caller (empty string is fine).
    """
    logger.warning(
        "Neo4j infrastructure error in %s/%s: %s", tool, operation, exc,
    )
    return infra_error_envelope(tool, operation, session_id)
