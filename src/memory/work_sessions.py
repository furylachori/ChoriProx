"""Work session management — atomic claim + lazy stale cleanup.

Functions here wrap the Cypher for the start_work, stop_work, get_active_sessions,
and cleanup_stale_sessions operations. They operate against a TaskGraph's
QueryExecutor (tg._executor) so they share the singleton Neo4j driver and
the circuit breaker.

The Cypher mirrors the SPEC verbatim:
  - start_work  (lines 309-353): atomic claim + work session creation
  - stop_work   (lines 491-509): close specific session by ws_uuid

All writes use parameterized queries — no f-string interpolation of user data.
"""
import logging
from typing import Any, Optional

from memory.query_executor import InfrastructureError

logger = logging.getLogger(__name__)

STALE_SESSION_HOURS = 24


def _workspace_where(workspace_id: Optional[str]) -> str:
    """Build the workspace WHERE fragment for task queries.

    v5: workspaces are global. bare_name == workspace_id.
    IS NULL handling dropped: migration 010 backfills NULLs to 'default'.
    """
    if workspace_id:
        return "AND t.workspace_id = $workspace_id"
    return ""


def create_work_session(
    tg: Any,
    ws_uuid: str,
    task_short_hash: str,
    agent_id: str,
    session_id: str,
    workspace_id: Optional[str] = None,
    transferred_from_agent: Optional[str] = None,
) -> dict:
    """Atomically claim a task and create a WorkSession.

    Mirrors SPEC lines 309-353. Uses execute_transaction_no_retry because
    the operation includes a non-idempotent CREATE on the WorkSession node.

    Returns one of:
      - {"action": "start_work", "ws_uuid": ..., "task_short_hash": ...,
         "task_status": ..., "started_at": ..., "session_id": ...}
      - {"error": "task_not_found", ...}
      - {"error": "not_authorized", "assigned_to": ..., ...}
      - {"error": "task_locked", "active_agent": ..., ...}
      - {"error": "write_conflict", ...}
    """
    workspace_where = _workspace_where(workspace_id)
    short_hash = (task_short_hash or "").strip().lower()
    if not short_hash:
        return {"error": "invalid_input", "message": "short_hash is required"}

    skip_lock_check = False
    pre_check_query = f"""
    MATCH (t:Task {{short_hash: $short_hash}})
    WHERE t.deleted_at IS NULL
      {workspace_where}
    RETURN t.assigned_agent_id AS assigned_agent_id
    LIMIT 2
    """
    try:
        pre_records = tg._executor.execute_read(pre_check_query, {
            "short_hash": short_hash,
            "workspace_id": workspace_id,
        })
    except InfrastructureError as e:
        logger.debug("Pre-check skipped due to Neo4j unavailable: %s", e)
        pre_records = []
    if pre_records:
        pre_assigned_to = pre_records[0].get("assigned_agent_id")
        if pre_assigned_to is not None and pre_assigned_to != agent_id:
            try:
                from proxy.agent.agent_registry import registry as _agent_registry
                _assigned_present = _agent_registry._agents.get(pre_assigned_to) is not None
            except Exception as e:
                logger.debug("Agent registry lookup failed: %s", e)
                _assigned_present = True
            if not _assigned_present:
                skip_lock_check = True

    def _start_work_tx(tx):
        lookup_query = f"""
        MATCH (t:Task {{short_hash: $short_hash}})
        WHERE t.deleted_at IS NULL
          {workspace_where}
        RETURN
          t.short_hash AS short_hash,
          t.status AS status,
          t.assigned_agent_id AS assigned_agent_id
        LIMIT 2
        """
        records = list(tx.run(lookup_query, {
            "short_hash": short_hash,
            "workspace_id": workspace_id,
        }))
        if not records:
            return {"error": "task_not_found"}

        if len(records) > 1:
            from memory.task_graph import AmbiguousHashError
            raise AmbiguousHashError(len(records))

        row = records[0]
        assigned_to = row.get("assigned_agent_id")
        task_status = row.get("status")

        if assigned_to is not None and assigned_to != agent_id:
            return {"error": "not_authorized", "assigned_to": assigned_to}

        if task_status not in ("pending", "in_progress"):
            return {
                "error": "task_not_actionable",
                "status": task_status,
                "message": f"Task status '{task_status}' is not actionable via start_work",
            }

        if not skip_lock_check:
            active_lock_query = f"""
            MATCH (t:Task {{short_hash: $short_hash}})
              -[:HAS_WORK_SESSION]->(ws:WorkSession {{status: 'active'}})
            WHERE ws.agent_id <> $agent_id
              AND ws.started_at >= datetime() - duration({{hours: $stale_hours}})
              {workspace_where}
            RETURN ws.agent_id AS active_agent
            LIMIT 1
            """
            lock_records = list(tx.run(active_lock_query, {
                "short_hash": short_hash,
                "agent_id": agent_id,
                "stale_hours": STALE_SESSION_HOURS,
                "workspace_id": workspace_id,
            }))
            if lock_records:
                return {"error": "task_locked", "active_agent": lock_records[0].get("active_agent")}

        claim_query = f"""
        MATCH (t:Task {{short_hash: $short_hash}})
        WHERE t.deleted_at IS NULL
          {workspace_where}
        FOREACH (s IN [(t)-[:HAS_WORK_SESSION]->(w:WorkSession {{status: 'active'}})
                       WHERE w.started_at < datetime() - duration({{hours: $stale_hours}}) | w] |
          SET s.status = 'abandoned', s.ended_at = datetime()
        )
        SET t.assigned_agent_id = $agent_id,
            t.started_at = coalesce(t.started_at, datetime()),
            t.status = CASE WHEN t.status = 'pending' THEN 'in_progress' ELSE t.status END,
            t.status_changed_at = datetime()
        WITH t
        MERGE (ws:WorkSession {{uuid: $ws_uuid}})
        ON CREATE SET ws.agent_id = $agent_id,
                      ws.session_id = $session_id,
                      ws.started_at = datetime(),
                      ws.turn_count = 0,
                      ws.status = 'active',
                      ws.transferred_from_agent = $transferred_from_agent
        WITH t, ws
        MERGE (t)-[r:HAS_WORK_SESSION]->(ws)
        ON CREATE SET r.created_at = datetime()
        RETURN ws.uuid AS ws_uuid,
               ws.started_at AS session_started_at,
               t.status AS task_status,
               t.started_at AS task_started_at
        """
        result = list(tx.run(claim_query, {
            "short_hash": short_hash,
            "agent_id": agent_id,
            "session_id": session_id or "",
            "ws_uuid": ws_uuid,
            "stale_hours": STALE_SESSION_HOURS,
            "transferred_from_agent": transferred_from_agent,
            "workspace_id": workspace_id,
        }))
        if not result:
            return {"error": "task_not_found"}
        rec = result[0]
        return {
            "action": "start_work",
            "ws_uuid": rec.get("ws_uuid") or ws_uuid,
            "task_short_hash": short_hash,
            "task_status": rec.get("task_status"),
            "started_at": str(rec.get("session_started_at")) if rec.get("session_started_at") else None,
            "session_id": session_id or "",
        }

    try:
        return tg._executor.execute_transaction_no_retry(_start_work_tx)
    except InfrastructureError as e:
        # Detect Neo4j write-write conflict (TransientError surfaced as InfrastructureError).
        # SPEC line 967: must return {"error": "write_conflict", ...} on concurrent claim.
        msg = str(e).lower()
        if "write-write" in msg or "concurrent" in msg or "write conflict" in msg:
            logger.warning("start_work write_conflict for %s: %s", short_hash, e)
            return {"error": "write_conflict", "message": "Concurrent claim; retry."}
        logger.warning("start_work infrastructure error for %s: %s", short_hash, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}


def close_work_session(
    tg: Any,
    ws_uuid: str,
    task_short_hash: str,
    agent_id: str,
    workspace_id: Optional[str] = None,
    new_status: Optional[str] = None,
    git_commit_sha: Optional[str] = None,
) -> dict:
    """Close a specific work session by ws_uuid. Does NOT unassign the task.

    Mirrors SPEC lines 491-509. Atomic: assignment check + session close in a
    single execute_write transaction. Idempotent: uses execute_write.
    Returns one of:
      - {"action": "stop_work", "ws_uuid": ..., "task_status": ..., ...}
      - {"error": "task_not_found", ...}
      - {"error": "not_authorized", "assigned_to": ..., ...}
      - {"error": "no_active_session", ...}
    """
    workspace_where = _workspace_where(workspace_id)
    short_hash = (task_short_hash or "").strip().lower()
    if not short_hash:
        return {"error": "invalid_input", "message": "short_hash is required"}
    if not ws_uuid:
        return {"error": "invalid_input", "message": "ws_uuid is required"}
    if new_status is not None and new_status not in ("complete", "invalid"):
        return {"error": "invalid_input", "message": f"new_status must be 'complete' or 'invalid' if provided, got '{new_status}'"}

    def _stop_work_tx(tx):
        # Single query: verify assignment + close session + update task status.
        # SPEC US-097: stop_work does NOT unassign — t.assigned_agent_id is
        # not modified. We only verify it for authorization.
        close_query = f"""
        MATCH (t:Task {{short_hash: $short_hash}})
        WHERE t.deleted_at IS NULL
          {workspace_where}
          AND t.assigned_agent_id = $agent_id
        OPTIONAL MATCH (t)-[:HAS_WORK_SESSION]->(target_ws:WorkSession
            {{uuid: $ws_uuid, status: 'active', agent_id: $agent_id}})
        WITH t, target_ws
        FOREACH (_ IN CASE WHEN target_ws IS NOT NULL THEN [1] ELSE [] END |
          SET target_ws.status = 'completed',
              target_ws.ended_at = datetime(),
              target_ws.git_commit_sha = $git_commit_sha
        )
        WITH t, target_ws
        WHERE target_ws IS NOT NULL
        SET t.completed_at = CASE WHEN $new_status IN ['complete', 'invalid'] THEN datetime() ELSE t.completed_at END,
            t.status = CASE WHEN $new_status IN ['complete', 'invalid'] THEN $new_status ELSE t.status END,
            t.status_changed_at = CASE WHEN $new_status IS NOT NULL THEN datetime() ELSE t.status_changed_at END
        RETURN target_ws.uuid AS ws_uuid, t.status AS task_status
        """
        records = list(tx.run(close_query, {
            "short_hash": short_hash,
            "ws_uuid": ws_uuid,
            "agent_id": agent_id,
            "new_status": new_status,
            "git_commit_sha": git_commit_sha,
            "workspace_id": workspace_id,
        }))

        if not records:
            # 0 rows: either task not found, task not assigned to caller,
            # or no active session matching ws_uuid. Differentiate.
            check_query = f"""
            MATCH (t:Task {{short_hash: $short_hash}})
            WHERE t.deleted_at IS NULL
              {workspace_where}
            OPTIONAL MATCH (t)-[:HAS_WORK_SESSION]->(ws:WorkSession {{uuid: $ws_uuid}})
            RETURN t.assigned_agent_id AS assigned_to,
                   ws IS NOT NULL AS session_exists
            ORDER BY t.created_at DESC
            LIMIT 2
            """
            check_records = list(tx.run(check_query, {
                "short_hash": short_hash,
                "ws_uuid": ws_uuid,
                "workspace_id": workspace_id,
            }))
            if not check_records:
                return {"error": "task_not_found"}
            if len(check_records) > 1:
                from memory.task_graph import AmbiguousHashError
                raise AmbiguousHashError(len(check_records))
            check = check_records[0]
            assigned_to = check.get("assigned_to")
            if assigned_to != agent_id:
                return {"error": "not_authorized", "assigned_to": assigned_to}
            return {"error": "no_active_session"}

        rec = records[0]
        return {
            "action": "stop_work",
            "ws_uuid": rec.get("ws_uuid") or ws_uuid,
            "task_short_hash": short_hash,
            "task_status": rec.get("task_status"),
        }

    try:
        return tg._executor.execute_transaction_no_retry(_stop_work_tx)
    except InfrastructureError as e:
        logger.warning("stop_work write failed for %s/%s: %s", short_hash, ws_uuid, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}


def get_active_sessions(
    tg: Any,
    task_short_hash: str,
    agent_id: str,
    workspace_id: Optional[str] = None,
) -> list[dict]:
    """Return active WorkSession uuids for an agent on a task.

    Returns a list of dicts with keys: uuid, started_at, session_id.
    """
    workspace_where = _workspace_where(workspace_id)
    short_hash = (task_short_hash or "").strip().lower()
    if not short_hash:
        return []

    query = f"""
    MATCH (t:Task {{short_hash: $short_hash}})
    WHERE t.deleted_at IS NULL
      {workspace_where}
    MATCH (t)-[:HAS_WORK_SESSION]->(ws:WorkSession)
    WHERE ws.status = 'active'
      AND ws.agent_id = $agent_id
    RETURN ws.uuid AS uuid,
           ws.started_at AS started_at,
           ws.session_id AS session_id
    ORDER BY ws.started_at DESC
    """
    try:
        records = tg._executor.execute_read(query, {
            "short_hash": short_hash,
            "agent_id": agent_id,
            "workspace_id": workspace_id,
        })
    except InfrastructureError as e:
        logger.warning("get_active_sessions read failed for %s: %s", short_hash, e)
        return []

    return [
        {
            "uuid": r.get("uuid"),
            "started_at": str(r.get("started_at")) if r.get("started_at") else None,
            "session_id": r.get("session_id") or "",
        }
        for r in records
        if r.get("uuid")
    ]


def cleanup_stale_sessions(
    tg: Any,
    task_short_hash: str,
    agent_id: str,
    workspace_id: Optional[str] = None,
) -> int:
    """Mark sessions >STALE_SESSION_HOURS as 'abandoned'. Returns count.

    Usually called internally by start_work (atomic with claim).
    Exposed for tests and ad-hoc cleanup.
    """
    workspace_where = _workspace_where(workspace_id)
    short_hash = (task_short_hash or "").strip().lower()
    if not short_hash:
        return 0

    query = f"""
    MATCH (t:Task {{short_hash: $short_hash}})
    WHERE t.deleted_at IS NULL
      {workspace_where}
    MATCH (t)-[r:HAS_WORK_SESSION]->(ws:WorkSession {{status: 'active'}})
    WHERE ws.started_at < datetime() - duration({{hours: $stale_hours}})
      AND ws.agent_id = $agent_id
    SET ws.status = 'abandoned', ws.ended_at = datetime()
    RETURN count(ws) AS marked
    """
    try:
        records = tg._executor.execute_write(query, {
            "short_hash": short_hash,
            "stale_hours": STALE_SESSION_HOURS,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
        })
    except InfrastructureError as e:
        logger.warning("cleanup_stale_sessions write failed for %s: %s", short_hash, e)
        return 0
    if not records:
        return 0
    return int(records[0].get("marked") or 0)
