"""Query operations for task retrieval."""
import logging
from typing import Any, Optional

from memory.query_executor import InfrastructureError

logger = logging.getLogger(__name__)

MAX_CHILD_TASKS = 500


class TaskGraphQueryMixin:
    """Query operations for task retrieval."""

    def get_actionable_tasks(
        self, status_filter: Optional[list[str]] = None, global_list: bool = False,
        workspace_id: Optional[str] = None, limit: int = 500
    ) -> list[dict]:
        if status_filter is None:
            status_filter = ["pending", "in_progress"]

        status_where = "AND t.status IN $status_filter" if status_filter else ""

        workspace_where = ""
        if workspace_id:
            # v5: workspaces are global. bare_name == workspace_id.
            workspace_where = "AND t.workspace_id = $workspace_id"

        query = f"""
        MATCH (t:Task)
        WHERE t.deleted_at IS NULL
          {status_where}
          {workspace_where}
        OPTIONAL MATCH (t)-[:BLOCKED_BY]->(blocker:Task)
        WHERE blocker.deleted_at IS NULL
        OPTIONAL MATCH (t)-[:HAS_PARENT]->(parent:Task)
        WHERE parent.deleted_at IS NULL
        WITH t,
             collect(DISTINCT CASE WHEN blocker.uuid IS NOT NULL THEN {{
               uuid: blocker.uuid,
               short_hash: blocker.short_hash,
               name: blocker.name,
               status: blocker.status
             }} END) AS raw_blockers,
             parent
        WITH t,
             [b IN raw_blockers WHERE b IS NOT NULL] AS blockers,
             parent
        WITH t,
             blockers,
             [b IN blockers WHERE b.status IN ['pending', 'in_progress']] AS active_blockers,
             parent
        RETURN t.uuid AS uuid,
               t.short_hash AS short_hash,
               t.name AS name,
               t.priority AS priority,
               t.status AS status,
               t.summary AS summary,
               t.tag AS tag,
               t.planning_info AS planning_info,
               t.created_at AS created_at,
               t.assigned_agent_id AS assigned_agent_id,
               blockers,
               size(active_blockers) > 0 AS is_blocked,
               size(active_blockers) AS blocked_count,
               parent.uuid AS parent_task_uuid,
               parent.short_hash AS parent_short_hash,
               parent.name AS parent_name
        ORDER BY CASE t.priority
                   WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2
                   WHEN 'low' THEN 3
                 END,
                 CASE WHEN size(active_blockers) > 0 THEN 1 ELSE 0 END,
                 t.name ASC
        LIMIT $limit
        """

        params: dict[str, Any] = {"limit": limit}
        if status_filter:
            params["status_filter"] = status_filter
        if workspace_id:
            params["workspace_id"] = workspace_id

        try:
            records = self._executor.execute_read(query, params)
        except InfrastructureError:
            logger.warning("get_actionable_tasks failed", exc_info=True)
            return []

        return [dict(r) for r in records]

    def get_child_tasks(
        self, parent_hash: str, workspace_id: Optional[str] = None
    ) -> list[dict]:
        if not self.SHORT_HASH_RE.match(parent_hash.strip()):
            logger.warning("get_child_tasks: invalid parent_hash format: %r", parent_hash)
            return []

        workspace_where = ""
        if workspace_id:
            # v5: workspaces are global.
            workspace_where = "AND parent.workspace_id = $workspace_id"

        query = f"""
        MATCH (parent:Task {{short_hash: $parent_hash}})
        WHERE parent.deleted_at IS NULL
          {workspace_where}
        MATCH (child:Task)-[:HAS_PARENT]->(parent)
        WHERE child.deleted_at IS NULL
        RETURN child.uuid AS uuid,
               child.short_hash AS short_hash,
               child.name AS name,
               child.priority AS priority,
               child.status AS status,
               child.summary AS summary,
               child.tag AS tag,
               child.created_at AS created_at,
               child.assigned_agent_id AS assigned_agent_id
        ORDER BY child.status, child.name
        LIMIT $limit
        """

        try:
            params: dict[str, Any] = {"parent_hash": parent_hash, "limit": MAX_CHILD_TASKS}
            if workspace_id:
                params["workspace_id"] = workspace_id

            records = self._executor.execute_read(query, params)
            return [dict(r) for r in records]
        except InfrastructureError:
            raise
        except Exception as e:
            logger.error("get_child_tasks failed: %s", e)
            return []

    def get_task_enriched(self, short_hash: str, workspace_id: Optional[str] = None) -> Optional[dict]:
        workspace_where = ""
        if workspace_id:
            # v5: workspaces are global.
            workspace_where = "AND t.workspace_id = $workspace_id"

        # Guard: skip HAS_NOTE collection when no ReviewNote nodes exist.
        # Avoids Neo4j "unknown relationship type / property key" warnings.
        has_review_notes = False
        try:
            check = self._executor.execute_read(
                "MATCH (n:ReviewNote) RETURN count(n) AS c LIMIT 1"
            )
            # Defensive: tolerate MagicMock / non-numeric results (e.g. in tests
            # where the executor is mocked) and any shape that lacks a numeric
            # "c" key. Treat anything non-positive or non-numeric as "no notes".
            count = check[0]["c"] if check else 0
            has_review_notes = isinstance(count, (int, float)) and count > 0
        except InfrastructureError:
            pass

        if has_review_notes:
            note_match = "OPTIONAL MATCH (t)-[:HAS_NOTE]->(rn:ReviewNote)"
            note_collect = (
                "collect(DISTINCT CASE WHEN rn IS NOT NULL THEN rn END) AS raw_notes"
            )
            note_list = (
                "[n IN raw_notes WHERE n IS NOT NULL | "
                "{ts: n.ts, note: n.note, author: n.author}][0..100] AS review_notes"
            )
        else:
            note_match = ""
            note_collect = "[] AS raw_notes"
            note_list = "[] AS review_notes"

        query = f"""
        MATCH (t:Task {{short_hash: $hash}})
        WHERE t.deleted_at IS NULL
          {workspace_where}
        OPTIONAL MATCH (t)-[:BLOCKED_BY]->(blocker:Task)
        WHERE blocker.deleted_at IS NULL
        OPTIONAL MATCH (t)-[:HAS_PARENT]->(parent:Task)
        WHERE parent.deleted_at IS NULL
        {note_match}
        WITH t,
             collect(DISTINCT CASE WHEN blocker.uuid IS NOT NULL THEN {{
               uuid: blocker.uuid,
               short_hash: blocker.short_hash,
               name: blocker.name,
               status: blocker.status
             }} END) AS raw_blockers,
             parent,
             {note_collect}
        WITH t,
             [b IN raw_blockers WHERE b IS NOT NULL] AS blockers,
             parent,
             raw_notes
        WITH t,
             blockers,
             [b IN blockers WHERE b.status IN ['pending', 'in_progress']] AS active_blockers,
             parent,
             {note_list}
        RETURN t.uuid AS uuid,
               t.short_hash AS short_hash,
               t.name AS name,
               t.priority AS priority,
               t.status AS status,
               t.summary AS summary,
               t.tag AS tag,
               t.planning_info AS planning_info,
               t.created_at AS created_at,
               t.assigned_agent_id AS assigned_agent_id,
               blockers,
               size(active_blockers) > 0 AS is_blocked,
               size(active_blockers) AS blocked_count,
               parent.uuid AS parent_task_uuid,
               parent.short_hash AS parent_short_hash,
               parent.name AS parent_name,
               review_notes
        """

        params: dict[str, Any] = {"hash": short_hash}
        if workspace_id:
            params["workspace_id"] = workspace_id

        try:
            records = self._executor.execute_read(query, params)
        except InfrastructureError:
            logger.warning("get_task_enriched failed", exc_info=True)
            return None

        # Defensive: tolerate mocked executors (e.g. MagicMock in tests) that
        # don't return a real list. Treat anything non-list or empty as "not found"
        # so callers see a clean None instead of a half-formed dict that the
        # formatter renders as a placeholder task.
        if not isinstance(records, list) or not records:
            return None

        return dict(records[0])


def get_recent_review_notes(
    executor, session_id: str, workspace_id: str,
    short_hashes: list, limit_per_task: int = 3
) -> dict:
    """Fetch recent review notes for a list of tasks. Returns {short_hash: [notes]}."""
    if not short_hashes:
        return {}

    # Guard: avoid Neo4j "unknown relationship type / property key" warnings
    # when no ReviewNote nodes exist (migration 005 hasn't run or had no data).
    try:
        from memory.query_executor import InfrastructureError
        check = executor.execute_read(
            "MATCH (n:ReviewNote) RETURN count(n) AS c LIMIT 1"
        )
        if not check or check[0]["c"] == 0:
            return {}
    except InfrastructureError:
        pass  # fall through to main query

    # v5: workspaces are global. bare_name == workspace_id.
    workspace_where = "AND t.workspace_id = $workspace_id" if workspace_id else ""

    query = f"""
    MATCH (t:Task)-[:HAS_NOTE]->(n:ReviewNote)
    WHERE t.short_hash IN $hashes AND t.deleted_at IS NULL {workspace_where}
    WITH t, n ORDER BY n.ts DESC
    WITH t, collect({{ts: n.ts, note: n.note, author: n.author}})[0..$limit] AS recent_notes
    RETURN t.short_hash AS short_hash, recent_notes
    """

    params = {
        "hashes": short_hashes,
        "limit": limit_per_task,
        "workspace_id": workspace_id,
    }

    try:
        records = executor.execute_read(query, params)
    except InfrastructureError:
        logger.warning("get_recent_review_notes failed", exc_info=True)
        return {}

    return {r["short_hash"]: r["recent_notes"] for r in records}
