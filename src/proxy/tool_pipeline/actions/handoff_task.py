import logging
from typing import Any

from neo4j.exceptions import TransientError, ServiceUnavailable, SessionExpired
from memory.query_executor import InfrastructureError

from proxy.tool_pipeline.action_envelope import success_envelope, error_envelope
from proxy.tool_pipeline.actions._errors import log_and_emit_infra_error

logger = logging.getLogger("proxy.tool_pipeline.actions")

TOOL = "proxy_work"
OPERATION = "handoff_task"


class HandoffTaskAction:
    """Transfer task assignment to another agent."""

    @staticmethod
    def execute(arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        from memory.task_graph import AmbiguousHashError

        short_hash = (arguments.get("short_hash") or "").strip()
        to_agent_id = (arguments.get("to_agent_id") or "").strip()
        context = arguments.get("context")

        if not short_hash:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "short_hash is required", session_id, field="short_hash",
            )
        if not to_agent_id:
            return error_envelope(
                TOOL, OPERATION, "invalid_input",
                "to_agent_id is required", session_id, field="to_agent_id",
            )
        if not caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "agent_not_found",
                "Cannot resolve caller agent_id", session_id,
            )

        if to_agent_id == caller_agent_id:
            return error_envelope(
                TOOL, OPERATION, "cannot_handoff_to_self",
                "Cannot handoff to the same agent", session_id, field="to_agent_id",
            )

        from proxy.agent.agent_registry import registry as agent_registry
        target = agent_registry.get_or_load(to_agent_id, caller_session_id=None)
        if target is None:
            return error_envelope(
                TOOL, OPERATION, "target_agent_not_found",
                f"Target agent '{to_agent_id}' not found in registry", session_id,
                field="to_agent_id",
            )

        # Non-idempotent write (SET + reassign) — use execute_transaction_no_retry
        # to avoid driver-internal retry racing with our WHERE clause on a
        # already-reassigned task.
        try:
            inner = tg._executor.execute_transaction_no_retry(
                lambda tx: _handoff_task_tx(tx, short_hash, caller_agent_id, to_agent_id, context, workspace_id, session_id)
            )
        except AmbiguousHashError:
            return error_envelope(
                TOOL, OPERATION, "ambiguous_hash",
                "Multiple tasks match this hash", session_id, field="short_hash",
            )
        except (TransientError, ServiceUnavailable, SessionExpired, InfrastructureError) as e:
            return log_and_emit_infra_error(logger, TOOL, OPERATION, e, session_id)

        if isinstance(inner, dict) and inner.get("error"):
            code = inner["error"]
            if code == "task_not_found":
                return error_envelope(
                    TOOL, OPERATION, "task_not_found",
                    "Task not found or not assigned to caller", session_id,
                    field="short_hash",
                )
            if code == "not_authorized":
                msg = f"Task is assigned to another agent ({inner.get('assigned_to', 'unknown')})"
                return error_envelope(
                    TOOL, OPERATION, "not_authorized", msg, session_id,
                    field="short_hash",
                )
            return error_envelope(
                TOOL, OPERATION, code,
                inner.get("message", code), session_id, field="short_hash",
            )

        return success_envelope(
            TOOL, OPERATION,
            {
                "short_hash": (inner or {}).get("task_hash") or short_hash,
                "to_agent_id": to_agent_id,
                "closed_sessions": int((inner or {}).get("closed_count") or 0),
            },
            session_id,
        )


def _handoff_task_tx(tx, short_hash, caller_agent_id, to_agent_id, context, workspace_id, session_id):
    """Execute the handoff atomically.

    Empty records list → WHERE failed (task not found or not assigned to caller).
    Non-empty list with closed_count=0 → task reassigned, no active sessions existed.
    Non-empty list with closed_count>0 → sessions were marked transferred.

    Returns one of:
      - {"error": "task_not_found"} — task does not exist
      - {"error": "not_authorized", "assigned_to": <id>} — task exists but not assigned to caller
      - {"task_hash": <hash>, "closed_count": <int>} — success
    """
    # v5: workspaces are global. bare_name == workspace_id.
    if not workspace_id:
        workspace_where = ""
    else:
        workspace_where = "AND t.workspace_id = $workspace_id"

    query = f"""
    MATCH (t:Task {{short_hash: $short_hash}})
    WHERE t.deleted_at IS NULL
      {workspace_where}
      AND t.assigned_agent_id = $caller_agent_id
    OPTIONAL MATCH (t)-[:HAS_WORK_SESSION]->(ws:WorkSession {{status: 'active', agent_id: $caller_agent_id}})
    WITH t, collect(ws) AS sessions
    FOREACH (s IN sessions |
      SET s.status = 'transferred',
          s.ended_at = datetime(),
          s.transfer_note = $context
    )
    SET t.assigned_agent_id = $to_agent_id,
        t.status_changed_at = datetime()
    RETURN size(sessions) AS closed_count, t.short_hash AS task_hash
    """
    params = {
        "short_hash": short_hash,
        "caller_agent_id": caller_agent_id,
        "to_agent_id": to_agent_id,
        "context": context,
        "workspace_id": workspace_id,
    }

    records = list(tx.run(query, params))
    if not records:
        # Differentiate: was the task not found, or not assigned to caller?
        check = list(tx.run(f"""
            MATCH (t:Task {{short_hash: $short_hash}})
            WHERE t.deleted_at IS NULL
              {workspace_where}
            RETURN t.assigned_agent_id AS assigned_to
            LIMIT 2
        """, {"short_hash": short_hash, "workspace_id": workspace_id}))
        if not check:
            return {"error": "task_not_found"}
        if len(check) > 1:
            from memory.task_graph import AmbiguousHashError
            raise AmbiguousHashError(len(check))
        return {"error": "not_authorized", "assigned_to": check[0].get("assigned_to")}

    rec = records[0]
    return {
        "task_hash": rec.get("task_hash") or short_hash,
        "closed_count": int(rec.get("closed_count") or 0),
    }
