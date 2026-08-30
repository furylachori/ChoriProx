import logging
import uuid as uuid_mod
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "review_bug"


class ReviewBugAction:
    """Submit a review for an open bug (merged claim+submit)."""

    @staticmethod
    def execute(
        arguments: dict,
        session_id: str,
        tg: Any,
        workspace_id: str = None,
        caller_agent_id: str = None,
    ) -> dict:
        from memory.bug_tracker import submit_review

        bug_uuid = (arguments.get("bug_uuid") or "").strip()
        confidence = arguments.get("confidence")
        verdict = (arguments.get("verdict") or "").strip()
        proposal = (arguments.get("proposal") or "").strip()

        if not bug_uuid:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "bug_uuid is required", session_id, field="bug_uuid",
            )
        if confidence is None or not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "confidence must be a number 0.0..1.0", session_id, field="confidence",
            )
        if verdict not in {"valid", "not_reproducible", "by_design", "already_fixed"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"verdict must be valid/not_reproducible/by_design/already_fixed, got '{verdict}'",
                session_id, field="verdict",
            )
        if not proposal or len(proposal) > 5000:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "proposal is required and must be ≤5000 chars", session_id,
                field="proposal",
            )
        if not caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "agent_not_found",
                "Cannot resolve caller agent_id", session_id,
            )

        review_uuid = f"review-{uuid_mod.uuid4().hex}"

        try:
            result = submit_review(
                tg=tg,
                review_uuid=review_uuid,
                bug_uuid=bug_uuid,
                confidence=float(confidence),
                verdict=verdict,
                proposal=proposal,
                caller_agent_id=caller_agent_id,
                workspace_id=workspace_id,
            )
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
                    "review_uuid": result["review_uuid"],
                    "review_count": result["review_count"],
                },
                session_id,
            )
        except (TransientError, ServiceUnavailable, SessionExpired, InfrastructureError) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
