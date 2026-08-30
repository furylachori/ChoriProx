import logging
import secrets
import uuid as uuid_mod
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "resolve_bug"


class ResolveBugAction:
    """Human approve/dismiss a validated bug (merged approve+dismiss)."""

    @staticmethod
    def execute(
        arguments: dict,
        session_id: str,
        tg: Any,
        workspace_id: str = None,
        caller_agent_id: str = None,
        is_subagent: bool = False,
    ) -> dict:
        from memory.bug_tracker import get_bug_with_reviews, resolve_bug

        if not is_subagent and session_id:
            # Fail closed: if we cannot VERIFY the caller is a human (agent
            # lookup returns None or raises), treat it as an untrusted
            # sub-agent so the not_authorized gate below applies. Note:
            # _load_from_sqlite swallows all exceptions and returns None, so
            # get_or_load returns None (does NOT raise InfrastructureError)
            # on SQLite degradation — both paths must close the gate.
            try:
                from proxy.agent.agent_registry import registry as _agent_registry
                _agent = _agent_registry.get_or_load(session_id)
            except Exception as e:
                logger.warning(
                    "resolve_bug: agent registry lookup raised; failing closed to sub-agent. "
                    "session_id=%s error=%r",
                    session_id,
                    e,
                )
                is_subagent = True
            else:
                if _agent is None:
                    is_subagent = True
                else:
                    is_subagent = getattr(_agent, 'is_subagent', False)

        bug_uuid = (arguments.get("bug_uuid") or "").strip()
        resolution = (arguments.get("resolution") or "").strip()
        reason = arguments.get("reason")

        if not bug_uuid:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "bug_uuid is required", session_id, field="bug_uuid",
            )
        if resolution not in {"approved", "dismissed"}:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                f"resolution must be 'approved' or 'dismissed', got '{resolution}'",
                session_id, field="resolution",
            )

        if is_subagent:
            return error_envelope(
                TOOL, OPERATION, "not_authorized",
                "Only the human operator can resolve validated bugs. Sub-agents cannot approve or dismiss bugs.",
                session_id,
            )

        # SPEC lines 778-781: error differentiation
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
        if bug.get("validated_at") is None:
            return error_envelope(
                TOOL, OPERATION, "not_validated",
                "Bug must be validated before resolution.", session_id, field="bug_uuid",
            )

        # Pre-generate task UUIDs for the case where resolution='approved'
        task_uuid = f"entity-{uuid_mod.uuid4().hex}"
        task_short_hash = secrets.token_hex(4)

        try:
            result = resolve_bug(
                tg=tg,
                bug_uuid=bug_uuid,
                resolution=resolution,
                reason=reason,
                caller_agent_id=caller_agent_id,
                workspace_id=workspace_id,
                task_uuid=task_uuid,
                task_short_hash=task_short_hash,
            )
            if isinstance(result, dict) and result.get("error"):
                return error_envelope(
                    TOOL, OPERATION,
                    result["error"],
                    result.get("message", result["error"]),
                    session_id,
                )
            data = {
                "bug_uuid": bug_uuid,
                "bug_status": result.get("status"),
            }
            if resolution == "approved" and result.get("task_short_hash"):
                data["task_short_hash"] = result["task_short_hash"]
            return success_envelope(TOOL, OPERATION, data, session_id)
        except (TransientError, ServiceUnavailable, SessionExpired, InfrastructureError) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)
