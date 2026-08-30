"""Spec §11 canonical action result envelope helpers.

Every action handler returns one of:
  - success_envelope  -> status="success"
  - partial_envelope  -> status="partial"
  - error_envelope    -> status="error"

These are the ONLY sanctioned shapes. No bare {"error": "..."} dicts.
"""
from __future__ import annotations

from typing import Any, Optional

from proxy.tool_pipeline.error_taxonomy import classify_error


def success_envelope(
    tool: str,
    operation: str,
    data: dict,
    session_id: str,
) -> dict:
    """Canonical success envelope. `data` carries action-specific payload."""
    return {
        "ok": True,
        "tool": tool,
        "operation": operation,
        "status": "success",
        "data": data,
        "error": None,
        "session_id": session_id or "",
    }


def partial_envelope(
    tool: str,
    operation: str,
    data: dict,
    session_id: str,
) -> dict:
    """Canonical partial envelope (e.g. create_tasks with some per-task failures)."""
    return {
        "ok": True,
        "tool": tool,
        "operation": operation,
        "status": "partial",
        "data": data,
        "error": None,
        "session_id": session_id or "",
    }


def error_envelope(
    tool: str,
    operation: Optional[str],
    code: str,
    message: str,
    session_id: str,
    *,
    field: Optional[str] = None,
    valid_actions: Optional[list[str]] = None,
) -> dict:
    """Canonical error envelope.

    `operation` is None when the action is missing or unknown.
    `field` is the offending field name (optional).
    `valid_actions` is included ONLY for missing/unknown action errors.
    """
    category, retryable = classify_error(code)
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
        "category": category,
    }
    if field is not None:
        error["field"] = field
    envelope: dict[str, Any] = {
        "ok": False,
        "tool": tool,
        "operation": operation,
        "status": "error",
        "data": None,
        "error": error,
        "session_id": session_id or "",
    }
    if valid_actions is not None:
        envelope["valid_actions"] = valid_actions
    return envelope
