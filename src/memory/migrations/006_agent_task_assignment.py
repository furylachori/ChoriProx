"""Migration 006: Agent-task assignment foundation.

Adds:
- Unique constraint on (short_hash, workspace_id) for Task nodes.
- Index on assigned_agent_id for fast lookup of agent's active tasks.

Both statements use IF NOT EXISTS for idempotency.
"""
import logging
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor

logger = logging.getLogger(__name__)


class AgentTaskAssignmentMigration(Migration):
    version = 6
    description = "Add assignment foundation: short_hash+workspace uniqueness, assigned_agent index"

    def up(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "CREATE CONSTRAINT task_short_hash_workspace_unique IF NOT EXISTS "
            "FOR (t:Task) REQUIRE (t.short_hash, t.workspace_id) IS UNIQUE"
        )
        executor.execute_write(
            "CREATE INDEX task_assigned_agent IF NOT EXISTS "
            "FOR (t:Task) ON (t.assigned_agent_id)"
        )
        logger.info("Migration 006 applied: task_short_hash_workspace_unique constraint + task_assigned_agent index")
