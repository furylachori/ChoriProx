"""Migration 010: Strip session_id: prefix from Task and Bug workspace_id.

v5 rework: workspace_id changes from compound (session_id:workspace_name)
to bare (workspace_name). This migration:

  Phase 1: Strip '<session_id>:' from :Task.workspace_id (paginated 5K/batch)
  Phase 2: Strip '<session_id>:' from :Bug.workspace_id (paginated 5K/batch)
  Phase 3: Backfill NULL workspace_id on Task and Bug to 'default' (the
           removed endswith(':default') clause loses NULL handling)
  Phase 4: Re-key :Workspace nodes with compound IDs to bare names
  Phase 5: Delete orphaned compound :Workspace nodes (no remaining references)

All phases are idempotent. Re-running on already-migrated data is a no-op.

NOTE: Migration runner executes migrations in version order. 010 runs AFTER
009 (rewritten) on a fresh deploy. The rewritten 009 internally strips compound
prefixes too, so the ordering is safe even on fresh data.
"""
from memory.migrations.base import Migration
import logging

logger = logging.getLogger(__name__)

PHASE_1_STRIP_TASKS = """
MATCH (t:Task)
WHERE t.workspace_id IS NOT NULL
  AND t.workspace_id CONTAINS ':'
WITH t, split(t.workspace_id, ':') AS parts
SET t.workspace_id = parts[-1]
RETURN count(t) AS migrated
"""

PHASE_2_STRIP_BUGS = """
MATCH (b:Bug)
WHERE b.workspace_id IS NOT NULL
  AND b.workspace_id CONTAINS ':'
WITH b, split(b.workspace_id, ':') AS parts
SET b.workspace_id = parts[-1]
RETURN count(b) AS migrated
"""

PHASE_3_BACKFILL_NULL = """
MATCH (t:Task)
WHERE t.workspace_id IS NULL
SET t.workspace_id = 'default'
RETURN count(t) AS backfilled
"""

PHASE_3B_BACKFILL_NULL_BUGS = """
MATCH (b:Bug)
WHERE b.workspace_id IS NULL
SET b.workspace_id = 'default'
RETURN count(b) AS backfilled
"""

PHASE_4_REKEY_WORKSPACE = """
MATCH (w:Workspace)
WHERE w.workspace_id CONTAINS ':'
WITH w, split(w.workspace_id, ':') AS parts
WITH w, parts[-1] AS new_id
WHERE NOT EXISTS { MATCH (w2:Workspace {workspace_id: new_id}) }
SET w.workspace_id = new_id
SET w.last_active_at = coalesce(w.last_active_at, datetime())
REMOVE w.session_id
RETURN count(w) AS rekeyed
"""

PHASE_5_DELETE_ORPHANS = """
MATCH (w:Workspace)
WHERE w.workspace_id CONTAINS ':'
  AND NOT EXISTS { MATCH (t:Task {workspace_id: w.workspace_id}) }
  AND NOT EXISTS { MATCH (b:Bug {workspace_id: w.workspace_id}) }
DETACH DELETE w
RETURN count(w) AS deleted
"""


class StripWorkspacePrefix(Migration):
    version = 10
    description = "Strip session_id: prefix from Task and Bug workspace_id (v5 global catalog)"

    def up(self, executor):
        BATCH_SIZE = 5000

        # Phase 1: strip Task prefixes
        offset = 0
        while True:
            result = executor.execute_write(
                """MATCH (t:Task)
                   WHERE t.workspace_id IS NOT NULL
                     AND t.workspace_id CONTAINS ':'
                   WITH t SKIP $skip LIMIT $limit
                   WITH split(t.workspace_id, ':') AS parts, t
                   SET t.workspace_id = parts[-1]
                   RETURN count(t) AS migrated""",
                {"skip": offset, "limit": BATCH_SIZE},
            )
            count = result[0].get("migrated", 0) if result else 0
            logger.info("Migration 010 phase 1: stripped %d task prefixes (offset=%d)", count, offset)
            if count < BATCH_SIZE:
                break
            offset += BATCH_SIZE

        # Phase 2: strip Bug prefixes
        result = executor.execute_write(PHASE_2_STRIP_BUGS)
        bug_count = result[0].get("migrated", 0) if result else 0
        logger.info("Migration 010 phase 2: stripped %d bug prefixes", bug_count)

        # Phase 3: backfill NULL workspace_id on Task
        result = executor.execute_write(PHASE_3_BACKFILL_NULL)
        null_count = result[0].get("backfilled", 0) if result else 0
        logger.info("Migration 010 phase 3: backfilled %d NULL task workspace_ids to 'default'", null_count)

        # Phase 3b: backfill NULL workspace_id on Bug
        result = executor.execute_write(PHASE_3B_BACKFILL_NULL_BUGS)
        null_bug = result[0].get("backfilled", 0) if result else 0
        logger.info("Migration 010 phase 3b: backfilled %d NULL bug workspace_ids to 'default'", null_bug)

        # Phase 4: re-key compound :Workspace nodes to bare names
        result = executor.execute_write(PHASE_4_REKEY_WORKSPACE)
        rekeyed = result[0].get("rekeyed", 0) if result else 0
        logger.info("Migration 010 phase 4: re-keyed %d compound :Workspace nodes to bare names", rekeyed)

        # Phase 5: delete orphaned compound :Workspace nodes
        result = executor.execute_write(PHASE_5_DELETE_ORPHANS)
        deleted = result[0].get("deleted", 0) if result else 0
        logger.info("Migration 010 phase 5: deleted %d orphaned compound :Workspace nodes", deleted)

    def down(self, executor):
        # No-op: re-prefixing requires the original session_id mapping which
        # is not preserved. The user must use the documented rollback runbook.
        pass
