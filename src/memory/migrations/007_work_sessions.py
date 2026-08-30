"""Migration 007: Work session graph.

Adds:
- Unique constraint on (uuid) for WorkSession nodes.
- Index on (status) for WorkSession nodes, used by start_work stale cleanup queries
  and by stop_work (active session lookup).

Both statements use IF NOT EXISTS for idempotency.
"""
import logging
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor

logger = logging.getLogger(__name__)


class WorkSessionMigration(Migration):
    version = 7
    description = "Work session graph: WorkSession node, HAS_WORK_SESSION relationship"

    def up(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "CREATE CONSTRAINT work_session_uuid_unique IF NOT EXISTS "
            "FOR (ws:WorkSession) REQUIRE ws.uuid IS UNIQUE"
        )
        executor.execute_write(
            "CREATE INDEX work_session_status IF NOT EXISTS "
            "FOR (ws:WorkSession) ON (ws.status)"
        )
        logger.info("Migration 007 applied: work_session_uuid_unique constraint + work_session_status index")
