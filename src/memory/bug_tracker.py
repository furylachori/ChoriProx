"""Bug tracking — create/review/validate/resolve bugs and reviews.

Functions here wrap the Cypher for the report_bug, review_bug, validate_bug,
and resolve_bug operations. They operate against a TaskGraph's QueryExecutor
(tg._executor) so they share the singleton Neo4j driver and circuit breaker.

Cypher mirrors the SPEC verbatim:
  - create_bug    (lines 639-661): Bug + FOUND_DURING WorkSession
  - submit_review (lines 672-689): Review + HAS_REVIEW + atomic count increment
  - validate_bug  (lines 703-721): aggregate reviews, set validated_at
  - resolve_bug   (lines 748-775): set status approved/dismissed + create Task

All non-idempotent writes use execute_transaction_no_retry. Reads use
execute_read. All queries are parameterized — no f-string interpolation of
user data.
"""
import logging
from typing import Any, Optional

from memory.query_executor import InfrastructureError

logger = logging.getLogger(__name__)


def _workspace_where(alias: str, workspace_id: Optional[str]) -> str:
    # v5: workspaces are global. bare_name == workspace_id.
    if not workspace_id:
        return ""
    return f"AND {alias}.workspace_id = $workspace_id"


def create_bug(
    tg: Any,
    bug_uuid: str,
    bug_short_hash: str,
    task_short_hash: str,
    title: str,
    description: str,
    severity: str,
    caller_agent_id: str,
    workspace_id: Optional[str],
    max_reviews: int,
) -> dict:
    """Atomically create a Bug + FOUND_DURING link to active WorkSession.

    Mirrors SPEC lines 639-661. Uses execute_transaction_no_retry because
    the operation includes a non-idempotent CREATE on Bug and FOUND_DURING.

    Returns one of:
      - {"uuid": ..., "short_hash": ...} on success
      - {"error": "task_not_found", ...} if no Task matches
      - {"error": "not_authorized", ...} if task is not assigned to caller
      - {"error": "no_active_session", ...} if no active WorkSession for caller
      - {"error": "infrastructure_error", ...} on transient DB failure
    """
    workspace_where = _workspace_where("t", workspace_id)
    short_hash = (task_short_hash or "").strip().lower()

    def _create_bug_tx(tx):
        lookup_query = f"""
        MATCH (t:Task {{short_hash: $short_hash}})
        WHERE t.deleted_at IS NULL
          {workspace_where}
        OPTIONAL MATCH (t)-[:HAS_WORK_SESSION]->(ws:WorkSession {{
            status: 'active', agent_id: $caller_agent_id
        }})
        RETURN t.assigned_agent_id AS assigned_to,
               ws IS NOT NULL AS has_active_session
        LIMIT 2
        """
        records = list(tx.run(lookup_query, {
            "short_hash": short_hash,
            "caller_agent_id": caller_agent_id,
            "workspace_id": workspace_id,
        }))
        if not records:
            return {"error": "task_not_found"}
        if len(records) > 1:
            from memory.task_graph import AmbiguousHashError
            raise AmbiguousHashError(len(records))

        row = records[0]
        assigned_to = row.get("assigned_to")
        # SPEC line 644: strict assignment check — NULL fails (caller must be assigned).
        if assigned_to is None or assigned_to != caller_agent_id:
            return {
                "error": "not_authorized",
                "assigned_to": assigned_to,
                "message": "Task must be assigned to caller before reporting a bug",
            }
        if not row.get("has_active_session"):
            return {"error": "no_active_session"}

        create_query = f"""
        MATCH (t:Task {{short_hash: $short_hash}})
          -[:HAS_WORK_SESSION]->(ws:WorkSession {{
              status: 'active', agent_id: $caller_agent_id
          }})
        WHERE t.deleted_at IS NULL
          {workspace_where}
        CREATE (b:Bug {{
          uuid: $bug_uuid,
          short_hash: $bug_short_hash,
          title: $title,
          description: $description,
          severity: $severity,
          status: 'open',
          max_reviews: $max_reviews,
          review_count: 0,
          reported_by_agent: $caller_agent_id,
          created_at: datetime(),
          workspace_id: $workspace_id
        }})
        CREATE (b)-[:FOUND_DURING {{created_at: datetime()}}]->(ws)
        RETURN b.uuid AS uuid, b.short_hash AS short_hash
        """
        result = list(tx.run(create_query, {
            "short_hash": short_hash,
            "caller_agent_id": caller_agent_id,
            "bug_uuid": bug_uuid,
            "bug_short_hash": bug_short_hash,
            "title": title,
            "description": description,
            "severity": severity,
            "max_reviews": max_reviews,
            "workspace_id": workspace_id,
        }))
        if not result:
            return {"error": "no_active_session"}
        rec = result[0]
        return {"uuid": rec.get("uuid") or bug_uuid, "short_hash": rec.get("short_hash") or bug_short_hash}

    try:
        return tg._executor.execute_transaction_no_retry(_create_bug_tx)
    except InfrastructureError as e:
        logger.warning("create_bug infrastructure error for %s: %s", short_hash, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}


def submit_review(
    tg: Any,
    review_uuid: str,
    bug_uuid: str,
    confidence: float,
    verdict: str,
    proposal: str,
    caller_agent_id: str,
    workspace_id: Optional[str],
) -> dict:
    """Submit a Review for an open bug. Atomic claim+submit (one transaction).

    Mirrors SPEC lines 672-689. Uses execute_transaction_no_retry because the
    operation is non-idempotent (CREATE Review + SET review_count).

    Returns one of:
      - {"review_uuid": ..., "review_count": ...} on success
      - {"error": "bug_not_open", ...} if bug is not open / not found
      - {"error": "max_reviews_reached", ...} if review_count >= max_reviews
      - {"error": "already_reviewed", ...} if same agent already reviewed
      - {"error": "cannot_review_own_bug", ...} if reporter tries to review
      - {"error": "infrastructure_error", ...} on transient DB failure
    """
    workspace_where = _workspace_where("b", workspace_id)

    def _submit_review_tx(tx):
        create_query = f"""
        MATCH (b:Bug {{uuid: $bug_uuid, status: 'open'}})
        WHERE b.review_count < b.max_reviews
          AND NOT EXISTS((b)-[:HAS_REVIEW]->(:Review {{
              reviewer_agent_id: $caller_agent_id
          }}))
          AND b.reported_by_agent <> $caller_agent_id
          {workspace_where}
        CREATE (r:Review {{
          uuid: $review_uuid,
          reviewer_agent_id: $caller_agent_id,
          confidence_score: $confidence,
          verdict: $verdict,
          proposal: $proposal,
          submitted_at: datetime()
        }})
        CREATE (b)-[:HAS_REVIEW {{created_at: datetime()}}]->(r)
        SET b.review_count = b.review_count + 1
        RETURN r.uuid AS review_uuid, b.review_count AS review_count
        """
        records = list(tx.run(create_query, {
            "bug_uuid": bug_uuid,
            "workspace_id": workspace_id,
            "caller_agent_id": caller_agent_id,
            "review_uuid": review_uuid,
            "confidence": confidence,
            "verdict": verdict,
            "proposal": proposal,
        }))
        if not records:
            # Differentiate the cause
            status_query = f"""
            MATCH (b:Bug {{uuid: $bug_uuid}})
            WHERE true
              {workspace_where}
            RETURN b.status AS status,
                   b.review_count AS review_count,
                   b.max_reviews AS max_reviews,
                   b.reported_by_agent AS reporter
            """
            status_recs = list(tx.run(status_query, {
                "bug_uuid": bug_uuid,
                "workspace_id": workspace_id,
            }))
            if not status_recs:
                return {"error": "bug_not_open"}
            s = status_recs[0]
            if s.get("status") != "open":
                return {"error": "bug_not_open"}
            if s.get("reporter") == caller_agent_id:
                return {"error": "cannot_review_own_bug"}
            if (s.get("review_count") or 0) >= (s.get("max_reviews") or 0):
                return {"error": "max_reviews_reached"}
            return {"error": "already_reviewed"}
        rec = records[0]
        return {
            "review_uuid": rec.get("review_uuid") or review_uuid,
            "review_count": int(rec.get("review_count") or 0),
        }

    try:
        return tg._executor.execute_transaction_no_retry(_submit_review_tx)
    except InfrastructureError as e:
        # SPEC line 974: surface write_conflict on concurrent review submission.
        msg = str(e).lower()
        if "write-write" in msg or "concurrent" in msg or "write conflict" in msg:
            logger.warning("submit_review write_conflict for %s: %s", bug_uuid, e)
            return {"error": "write_conflict", "message": "Concurrent review; retry."}
        logger.warning("submit_review infrastructure error for %s: %s", bug_uuid, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}


def validate_bug(
    tg: Any,
    bug_uuid: str,
    caller_agent_id: str,
    workspace_id: Optional[str],
) -> dict:
    """Synthesize reviews for a fully-reviewed open bug. Verdict agent only.

    Mirrors SPEC lines 703-721. Uses execute_transaction_no_retry because SET
    b.validated_at is non-idempotent.

    Returns one of:
      - {"avg_confidence": float, "valid_count": int, "total": int} on success
      - {"error": "bug_not_open", ...} if bug not open / not found
      - {"error": "cannot_validate_own_bug", ...} if caller is reporter
      - {"error": "reviewer_cannot_validate", ...} if caller has reviewed
      - {"error": "reviews_incomplete", ...} if not enough reviews
      - {"error": "already_validated", ...} if validated_at already set
      - {"error": "infrastructure_error", ...} on transient DB failure
    """
    workspace_where = _workspace_where("b", workspace_id)

    def _validate_bug_tx(tx):
        validate_query = f"""
        MATCH (b:Bug {{uuid: $bug_uuid, status: 'open'}})
        WHERE b.reported_by_agent <> $caller_agent_id
          AND NOT EXISTS((b)-[:HAS_REVIEW]->(:Review {{
              reviewer_agent_id: $caller_agent_id
          }}))
          AND b.review_count >= b.max_reviews
          AND b.validated_at IS NULL
          {workspace_where}
        MATCH (b)-[:HAS_REVIEW]->(r:Review)
        WITH b, collect(r) AS reviews,
             avg(r.confidence_score) AS avg_confidence,
             count(r) AS total
        WITH b, reviews, avg_confidence, total,
             size([s IN reviews WHERE s.verdict = 'valid']) AS valid_count
        SET b.avg_confidence = avg_confidence,
            b.valid_reviewers = valid_count,
            b.total_reviewers = total,
            b.validated_at = datetime()
        RETURN b.uuid AS uuid,
               b.status AS status,
               avg_confidence AS avg_confidence,
               valid_count AS valid_count,
               total AS total
        """
        records = list(tx.run(validate_query, {
            "bug_uuid": bug_uuid,
            "caller_agent_id": caller_agent_id,
            "workspace_id": workspace_id,
        }))
        if not records:
            status_query = f"""
            MATCH (b:Bug {{uuid: $bug_uuid}})
            WHERE true
              {workspace_where}
            OPTIONAL MATCH (b)-[:HAS_REVIEW]->(rev:Review)
            RETURN b.status AS status,
                   b.review_count AS review_count,
                   b.max_reviews AS max_reviews,
                   b.reported_by_agent AS reporter,
                   b.validated_at AS validated_at,
                   collect(rev.reviewer_agent_id) AS reviewers
            """
            status_recs = list(tx.run(status_query, {
                "bug_uuid": bug_uuid,
                "workspace_id": workspace_id,
            }))
            if not status_recs:
                return {"error": "bug_not_open"}
            s = status_recs[0]
            if s.get("status") != "open":
                return {"error": "bug_not_open"}
            if s.get("reporter") == caller_agent_id:
                return {"error": "cannot_validate_own_bug"}
            reviewers = s.get("reviewers") or []
            if caller_agent_id in reviewers:
                return {"error": "reviewer_cannot_validate"}
            if (s.get("review_count") or 0) < (s.get("max_reviews") or 0):
                return {"error": "reviews_incomplete"}
            if s.get("validated_at") is not None:
                return {"error": "already_validated"}
            return {"error": "bug_not_open"}
        rec = records[0]
        return {
            "uuid": rec.get("uuid") or bug_uuid,
            "status": rec.get("status"),
            "avg_confidence": float(rec.get("avg_confidence") or 0.0),
            "valid_count": int(rec.get("valid_count") or 0),
            "total": int(rec.get("total") or 0),
        }

    try:
        return tg._executor.execute_transaction_no_retry(_validate_bug_tx)
    except InfrastructureError as e:
        logger.warning("validate_bug infrastructure error for %s: %s", bug_uuid, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}


def resolve_bug(
    tg: Any,
    bug_uuid: str,
    resolution: str,
    reason: Optional[str],
    caller_agent_id: Optional[str],
    workspace_id: Optional[str],
    task_uuid: str,
    task_short_hash: str,
) -> dict:
    """Human approve/dismiss a validated bug. On 'approved', creates a Task.

    Mirrors SPEC lines 748-775. Uses execute_transaction_no_retry because
    the operation includes non-idempotent SET status + optional CREATE Task.

    Returns one of:
      - {"status": ..., "task_short_hash": ...} on success
      - {"error": "bug_not_open", ...} if bug is not open
      - {"error": "not_validated", ...} if validated_at IS NULL
      - {"error": "infrastructure_error", ...} on transient DB failure
    """
    workspace_where = _workspace_where("b", workspace_id)

    def _resolve_bug_tx(tx):
        resolve_query = f"""
        MATCH (b:Bug {{uuid: $bug_uuid, status: 'open'}})
        WHERE b.validated_at IS NOT NULL
          {workspace_where}
        SET b.status = CASE WHEN $resolution = 'approved' THEN 'approved' ELSE 'dismissed' END,
            b.approved_at = CASE WHEN $resolution = 'approved' THEN datetime() ELSE null END,
            b.dismissed_at = CASE WHEN $resolution = 'dismissed' THEN datetime() ELSE null END,
            b.dismissal_reason = $reason
        WITH b
        FOREACH (_ IN CASE WHEN $resolution = 'approved' THEN [1] ELSE [] END |
          CREATE (t:Entity:Task {{
            uuid: $task_uuid,
            short_hash: $task_short_hash,
            name: b.title,
            summary: b.description,
            priority: CASE b.severity
              WHEN 'critical' THEN 'high'
              WHEN 'high' THEN 'high'
              WHEN 'medium' THEN 'medium'
              ELSE 'low' END,
            status: 'pending',
            session_id: null,
            workspace_id: $workspace_id,
            created_at: datetime()
          }})
          CREATE (b)-[:RESOLVED_BY {{approved_at: datetime()}}]->(t)
        )
        RETURN b.uuid AS uuid, b.status AS status
        """
        records = list(tx.run(resolve_query, {
            "bug_uuid": bug_uuid,
            "resolution": resolution,
            "reason": reason,
            "workspace_id": workspace_id,
            "task_uuid": task_uuid,
            "task_short_hash": task_short_hash,
        }))
        if not records:
            status_query = f"""
            MATCH (b:Bug {{uuid: $bug_uuid}})
            WHERE true
              {workspace_where}
            RETURN b.status AS status, b.validated_at AS validated_at
            """
            status_recs = list(tx.run(status_query, {
                "bug_uuid": bug_uuid,
                "workspace_id": workspace_id,
            }))
            if not status_recs:
                return {"error": "bug_not_open"}
            s = status_recs[0]
            if s.get("status") != "open":
                return {"error": "bug_not_open"}
            if s.get("validated_at") is None:
                return {"error": "not_validated"}
            return {"error": "bug_not_open"}
        rec = records[0]
        result = {
            "uuid": rec.get("uuid") or bug_uuid,
            "status": rec.get("status"),
        }
        if resolution == "approved":
            result["task_short_hash"] = task_short_hash
        return result

    try:
        return tg._executor.execute_transaction_no_retry(_resolve_bug_tx)
    except InfrastructureError as e:
        logger.warning("resolve_bug infrastructure error for %s: %s", bug_uuid, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}


def get_bug_with_reviews(
    tg: Any,
    bug_uuid: str,
    workspace_id: Optional[str] = None,
) -> Optional[dict]:
    """Return bug + reviews for ^^ticket bug show / action error differentiation.

    Returns None if the bug is not found. Returns a dict with bug fields plus
    a "reviews" list of {reviewer_agent_id, confidence_score, verdict, proposal}.
    """
    workspace_where = _workspace_where("b", workspace_id)
    bug_query = f"""
    MATCH (b:Bug {{uuid: $bug_uuid}})
    WHERE true
      {workspace_where}
    RETURN b.uuid AS uuid,
           b.short_hash AS short_hash,
           b.title AS title,
           b.description AS description,
           b.severity AS severity,
           b.status AS status,
           b.max_reviews AS max_reviews,
           b.review_count AS review_count,
           b.reported_by_agent AS reported_by_agent,
           b.avg_confidence AS avg_confidence,
           b.valid_reviewers AS valid_reviewers,
           b.total_reviewers AS total_reviewers,
           b.created_at AS created_at,
           b.validated_at AS validated_at,
           b.approved_at AS approved_at,
           b.dismissed_at AS dismissed_at,
           b.dismissal_reason AS dismissal_reason
    """
    try:
        records = tg._executor.execute_read(bug_query, {
            "bug_uuid": bug_uuid,
            "workspace_id": workspace_id,
        })
    except InfrastructureError as e:
        logger.warning("get_bug_with_reviews read failed for %s: %s", bug_uuid, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}
    if not records:
        return None
    bug = dict(records[0])
    reviews_query = f"""
    MATCH (b:Bug {{uuid: $bug_uuid}})
    WHERE true
      {workspace_where}
    MATCH (b)-[:HAS_REVIEW]->(r:Review)
    RETURN r.uuid AS uuid,
           r.reviewer_agent_id AS reviewer_agent_id,
           r.confidence_score AS confidence_score,
           r.verdict AS verdict,
           r.proposal AS proposal,
           r.submitted_at AS submitted_at
    ORDER BY r.submitted_at ASC
    """
    try:
        review_records = tg._executor.execute_read(reviews_query, {
            "bug_uuid": bug_uuid,
            "workspace_id": workspace_id,
        })
    except InfrastructureError as e:
        logger.warning("get_bug_with_reviews reviews read failed for %s: %s", bug_uuid, e)
        review_records = []
    bug["reviews"] = [dict(r) for r in review_records]
    return bug


def get_bug_by_short_hash(
    tg: Any,
    bug_short_hash: str,
    workspace_id: Optional[str] = None,
) -> Optional[dict]:
    """Look up a bug by short_hash. Used by ^^ticket bug show.

    Returns None if not found. Raises AmbiguousHashError if multiple match.
    """
    short_hash = (bug_short_hash or "").strip().lower()
    if not short_hash:
        return None

    workspace_where = _workspace_where("b", workspace_id)
    query = f"""
    MATCH (b:Bug {{short_hash: $short_hash}})
    WHERE true
      {workspace_where}
    RETURN b.uuid AS uuid
    LIMIT 2
    """
    try:
        records = tg._executor.execute_read(query, {
            "short_hash": short_hash,
            "workspace_id": workspace_id,
        })
    except InfrastructureError as e:
        logger.warning("get_bug_by_short_hash read failed for %s: %s", short_hash, e)
        return {"error": "infrastructure_error", "message": "Database temporarily unavailable"}
    if not records:
        return None
    if len(records) > 1:
        from memory.task_graph import AmbiguousHashError
        raise AmbiguousHashError(len(records))
    return get_bug_with_reviews(tg, records[0]["uuid"], workspace_id=workspace_id)


def list_bugs(
    tg: Any,
    workspace_id: Optional[str] = None,
    status_filter: Optional[list[str]] = None,
    limit: int = 100,
) -> list[dict]:
    """List bugs in a workspace, optionally filtered by status.

    Returns a list of dicts with: uuid, short_hash, title, severity, status,
    review_count, max_reviews, created_at, reported_by_agent.
    """
    workspace_where = _workspace_where("b", workspace_id)
    if status_filter:
        status_clause = "AND b.status IN $statuses"
    else:
        status_clause = ""
    query = f"""
    MATCH (b:Bug)
    WHERE true
      {workspace_where}
      {status_clause}
    RETURN b.uuid AS uuid,
           b.short_hash AS short_hash,
           b.title AS title,
           b.severity AS severity,
           b.status AS status,
           b.review_count AS review_count,
           b.max_reviews AS max_reviews,
           b.created_at AS created_at,
           b.reported_by_agent AS reported_by_agent
    ORDER BY b.created_at DESC
    LIMIT $limit
    """
    params: dict[str, Any] = {"limit": limit, "workspace_id": workspace_id}
    if status_filter:
        params["statuses"] = status_filter
    try:
        records = tg._executor.execute_read(query, params)
    except InfrastructureError as e:
        logger.warning("list_bugs read failed for workspace %s: %s", workspace_id, e)
        return []
    return [dict(r) for r in records]


def agent_has_reviewed(
    tg: Any,
    bug_uuid: str,
    caller_agent_id: str,
) -> bool:
    """Check whether the given agent has already submitted a review for the bug.

    Used by ValidateBugAction to differentiate "reviewer_cannot_validate" from
    other error cases during the action-layer post-check.
    """
    query = """
    MATCH (b:Bug {uuid: $bug_uuid})-[:HAS_REVIEW]->(r:Review {
        reviewer_agent_id: $caller_agent_id
    })
    RETURN count(r) > 0 AS has_reviewed
    """
    try:
        records = tg._executor.execute_read(query, {
            "bug_uuid": bug_uuid,
            "caller_agent_id": caller_agent_id,
        })
    except InfrastructureError as e:
        logger.warning("agent_has_reviewed read failed for %s: %s", bug_uuid, e)
        return False
    if not records:
        return False
    return bool(records[0].get("has_reviewed"))
