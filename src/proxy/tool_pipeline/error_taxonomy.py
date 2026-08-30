"""Spec §10 error taxonomy: 4 categories + retryable mapping.

Pure constants + one lookup function. No I/O, no locks.
"""
from __future__ import annotations

CATEGORY_VALIDATION = "validation"
CATEGORY_AUTHORIZATION = "authorization"
CATEGORY_INFRASTRUCTURE = "infrastructure"
CATEGORY_INTERNAL = "internal"

# code -> (category, retryable)
ERROR_CODE_TAXONOMY: dict[str, tuple[str, bool]] = {
    # Validation: caller can fix by changing arguments
    "invalid_input":            (CATEGORY_VALIDATION, True),
    "invalid_action":           (CATEGORY_VALIDATION, True),
    "unknown_action":           (CATEGORY_VALIDATION, True),
    "batch_too_large":          (CATEGORY_VALIDATION, True),
    "ordering_violation":       (CATEGORY_VALIDATION, True),
    "duplicate_name":           (CATEGORY_VALIDATION, True),
    "generic_task_name":        (CATEGORY_VALIDATION, True),
    "invalid_known_tasks":      (CATEGORY_VALIDATION, True),
    "ambiguous":                (CATEGORY_VALIDATION, True),
    "ambiguous_hash":           (CATEGORY_VALIDATION, True),
    "not_found":                (CATEGORY_VALIDATION, True),
    "invalid_status":           (CATEGORY_VALIDATION, True),
    "malformed_arguments":      (CATEGORY_VALIDATION, True),
    "bug_not_open":             (CATEGORY_VALIDATION, True),

    # Authorization: do not retry the same call
    "access_denied":            (CATEGORY_AUTHORIZATION, False),
    "mode_inactive":            (CATEGORY_AUTHORIZATION, False),
    "blocked":                  (CATEGORY_AUTHORIZATION, False),
    "agent_not_found":          (CATEGORY_AUTHORIZATION, False),
    "target_agent_not_found":   (CATEGORY_AUTHORIZATION, False),
    "cannot_handoff_to_self":   (CATEGORY_AUTHORIZATION, False),
    "task_not_found":           (CATEGORY_AUTHORIZATION, False),
    "not_authorized":           (CATEGORY_AUTHORIZATION, False),
    "cannot_validate_own_bug":  (CATEGORY_AUTHORIZATION, False),
    "reviewer_cannot_validate": (CATEGORY_AUTHORIZATION, False),
    "reviews_incomplete":       (CATEGORY_AUTHORIZATION, False),
    "already_validated":        (CATEGORY_AUTHORIZATION, False),
    "not_validated":            (CATEGORY_AUTHORIZATION, False),
    "assign_to_agent_not_found":(CATEGORY_AUTHORIZATION, False),

    # Infrastructure: transient; bounded retry budget separate from validation
    "infrastructure_error":     (CATEGORY_INFRASTRUCTURE, True),
    "query_timeout":            (CATEGORY_INFRASTRUCTURE, True),
    "constraint_violation":     (CATEGORY_INFRASTRUCTURE, True),

    # Internal: do not retry autonomously
    "action_failed":            (CATEGORY_INTERNAL, False),
    "tool_execution_failed":    (CATEGORY_INTERNAL, False),
    "unknown_memory_tool":      (CATEGORY_INTERNAL, False),

    # Retry enforcement (Slice 5)
    "validation_retry_exhausted":     (CATEGORY_VALIDATION, False),
    "authorization_no_retry":         (CATEGORY_AUTHORIZATION, False),
    "infrastructure_retry_exhausted": (CATEGORY_INFRASTRUCTURE, False),
    "internal_no_retry":              (CATEGORY_INTERNAL, False),
}


def classify_error(code: str) -> tuple[str, bool]:
    """Return (category, retryable). Unknown codes -> (internal, False)."""
    return ERROR_CODE_TAXONOMY.get(code, (CATEGORY_INTERNAL, False))
