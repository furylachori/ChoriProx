"""Migration 002: Verify all tasks have parent_task_uuid as top-level property."""
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor


class VerifyBackfill(Migration):
    version = 2
    description = "Verify all tasks have parent_task_uuid as top-level property"

    def up(self, executor: QueryExecutor) -> None:
        rows = executor.execute_read(
            """MATCH (e:Entity)
               WHERE 'Task' IN labels(e)
                 AND e.parent_task_uuid IS NULL
                 AND e.attributes CONTAINS 'parent_task_uuid'
               RETURN count(e) AS unbackfilled"""
        )
        count = rows[0]["unbackfilled"] if rows else 0
        if count > 0:
            raise RuntimeError(
                f"Migration 002: {count} tasks still need backfill. Re-run migration 001."
            )
