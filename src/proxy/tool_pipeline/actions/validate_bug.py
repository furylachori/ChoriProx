import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "validate_bug"


class ValidateBugAction:
    """Verdict agent synthesizes reviews for an open bug."""

    @staticmethod
    def execute(
        arguments: dict,
        session_id: str,
        tg: Any,
        workspace_id: str = None,
        caller_agent_id: str = None,
    ) -> dict:
        from memory.bug_tracker import agent_has_reviewed, get_bug_with_reviews, validate_bug

        bug_uuid = (arguments.get("bug_uuid") or "").strip()
        if not bug_uuid:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "bug_uuid is required", session_id, field="bug_uuid",
            )
        if not caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "agent_not_found",
                "Cannot resolve caller agent_id", session_id,
            )

        # Error differentiation per SPEC line 730-744
        bug = get_bug_with_reviews(tg, bug_uuid, workspace_id)
        if bug is not None and bug.get("error") == "infrastructure_error":
            return error_envelope(
                TOOL, OPERATION, "infrastructure_error",
                "Database temporarily unavailable", session_id, field="bug_uuid",
            )
        if bug is None or bug.get("status") != "open":
            return error_envelope(
                TOOL, OPERATION, "bug_not_open",
                "Bug not found or not in 'open' state", session_id, field="bug_uuid",
            )
        if bug.get("reported_by_agent") == caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "cannot_validate_own_bug",
                "Reporter cannot validate their own bug", session_id, field="bug_uuid",
            )
        if agent_has_reviewed(tg, bug_uuid, caller_agent_id):
            return error_envelope(
                TOOL, OPERATION, "reviewer_cannot_validate",
                "Agent has already reviewed this bug; cannot also validate",
                session_id, field="bug_uuid",
            )
        if (bug.get("review_count") or 0) < (bug.get("max_reviews") or 0):
            return error_envelope(
                TOOL, OPERATION, "reviews_incomplete",
                "Bug does not have enough reviews yet to be validated",
                session_id, field="bug_uuid",
            )
        if bug.get("validated_at"):
            return error_envelope(
                TOOL, OPERATION, "already_validated",
                "Bug has already been validated", session_id, field="bug_uuid",
            )

        try:
            result = validate_bug(tg, bug_uuid, caller_agent_id, workspace_id)
            if isinstance(result, dict) and result.get("error"):
                return error_envelope(
                    TOOL, OPERATION,
                    result["error"],
                    result.get("message", result["error"]),
                    session_id,
                )
            return success_envelope(
                TOOL, OPERATION,
                {
                    "bug_uuid": bug_uuid,
                    "avg_confidence": result.get("avg_confidence"),
                    "valid_count": result.get("valid_count"),
                    "total_count": result.get("total"),
                },
                session_id,
            )
        except (TransientError, ServiceUnavailable, SessionExpired, InfrastructureError) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
