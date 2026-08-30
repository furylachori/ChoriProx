"""Migration 008: Bug tracking graph.

Adds:
- Unique constraint on (uuid) for :Bug nodes.
- Composite uniqueness on (short_hash, workspace_id) for :Bug nodes.
- Unique constraint on (uuid) for :Review nodes.
- Index on Bug.status for filter-by-status queries.
- Composite index on (workspace_id, status) for workspace-scoped bug listing.

Both constraints and indexes use IF NOT EXISTS for idempotency.
"""
import logging
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor

logger = logging.getLogger(__name__)


class BugTrackingMigration(Migration):
    version = 8
    description = "Bug tracking graph: Bug and Review nodes, HAS_REVIEW/FOUND_DURING/RESOLVED_BY relationships"

    def up(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "CREATE CONSTRAINT bug_uuid_unique IF NOT EXISTS "
            "FOR (b:Bug) REQUIRE b.uuid IS UNIQUE"
        )
        executor.execute_write(
            "CREATE CONSTRAINT bug_short_hash_workspace_unique IF NOT EXISTS "
            "FOR (b:Bug) REQUIRE (b.short_hash, b.workspace_id) IS UNIQUE"
        )
        executor.execute_write(
            "CREATE CONSTRAINT review_uuid_unique IF NOT EXISTS "
            "FOR (r:Review) REQUIRE r.uuid IS UNIQUE"
        )
        executor.execute_write(
            "CREATE INDEX bug_status IF NOT EXISTS "
            "FOR (b:Bug) ON (b.status)"
        )
        executor.execute_write(
            "CREATE INDEX bug_workspace_status IF NOT EXISTS "
            "FOR (b:Bug) ON (b.workspace_id, b.status)"
        )
        logger.info("Migration 008 applied: bug/review constraints + indexes")
