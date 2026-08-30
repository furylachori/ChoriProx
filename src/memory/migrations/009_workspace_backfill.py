"""Migration 009: Backfill bare-name :Workspace nodes from existing Task data.

v5 rework: workspaces are now global with bare names (no session_id).
This migration:
  Phase 1: Backfill bare-name :Workspace nodes from tasks with already-bare
           workspace_id (post-migration 010 stripped state).
  Phase 2: For any task still at compound workspace_id, strip the prefix
           in-place and MERGE the corresponding bare-name :Workspace node.
           This is the safety net for the 009-before-010 ordering case.
  Phase 3: Set created_by_session on first-create (audit only).

Idempotent: re-running is a no-op on already-migrated data.
"""
from memory.migrations.base import Migration
import logging

logger = logging.getLogger(__name__)

CYPHER_BACKFILL = """
MATCH (t:Task)
WHERE t.workspace_id IS NOT NULL
  AND NOT t.workspace_id CONTAINS ':'
WITH DISTINCT t.workspace_id AS ws_id
ORDER BY ws_id
SKIP $skip LIMIT $limit

MATCH (t:Task {workspace_id: ws_id})
WITH ws_id, min(datetime(t.created_at)) AS earliest_created

MERGE (w:Workspace {workspace_id: ws_id})
ON CREATE SET w.name = ws_id,
              w.created_at = earliest_created,
              w.last_active_at = earliest_created
SET w.last_active_at =
    CASE WHEN earliest_created IS NOT NULL
          AND (w.last_active_at IS NULL OR w.last_active_at < earliest_created)
         THEN earliest_created
         ELSE w.last_active_at
    END

RETURN count(*) AS created
"""

# Safety-net: strip compound IDs and backfill. Used if 009 runs before 010.
CYPHER_STRIP_AND_BACKFILL = """
MATCH (t:Task)
WHERE t.workspace_id IS NOT NULL
  AND t.workspace_id CONTAINS ':'
WITH t, split(t.workspace_id, ':') AS parts
WITH t, parts[-1] AS new_id, parts[0] AS old_sid
WITH t, new_id, old_sid, collect(DISTINCT new_id) AS distinct_new
SET t.workspace_id = new_id
WITH DISTINCT new_id AS ws_id, old_sid
MERGE (w:Workspace {workspace_id: ws_id})
ON CREATE SET w.name = ws_id,
              w.created_at = datetime(),
              w.created_by_session = old_sid
SET w.last_active_at = datetime()
RETURN count(w) AS created
"""


class WorkspaceBackfill(Migration):
    version = 9
    description = "Backfill bare-name :Workspace nodes (v5 global catalog)"

    def up(self, executor):
        BATCH_SIZE = 5000

        # Safety-net phase: if 009 runs before 010 (fresh deploy), strip
        # compound IDs in-place and create bare-name :Workspace nodes.
        result = executor.execute_write(CYPHER_STRIP_AND_BACKFILL)
        safety = result[0].get("created", 0) if result else 0
        logger.info(
            "Migration 009 safety-net: stripped compound IDs and backfilled %d :Workspace nodes",
            safety,
        )

        # Main backfill phase: paginated on distinct bare workspace_ids
        offset = 0
        while True:
            result = executor.execute_write(
                CYPHER_BACKFILL,
                {"skip": offset, "limit": BATCH_SIZE},
            )
            count = result[0].get("created", 0) if result else 0
            logger.info(
                "Backfilled batch offset=%d, created/updated %d :Workspace nodes",
                offset, count,
            )
            if count < BATCH_SIZE:
                break
            offset += BATCH_SIZE

    def down(self, executor):
        # No-op: re-prefixing requires the original session_id mapping which
        # is not preserved. The user must use the documented rollback runbook.
        pass
