"""Migration 003: Remove vestigial Project→HAS_ENTITY relationships and nodes."""
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor


class RemoveProjectRelationships(Migration):
    version = 3
    description = "Remove vestigial Project-HAS_ENTITY relationships, Project nodes, and constraint"

    def up(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "MATCH (:Project)-[r:HAS_ENTITY]->(:Entity) DELETE r"
        )

        executor.execute_write(
            "MATCH (p:Project) DETACH DELETE p"
        )

        executor.execute_write(
            "DROP CONSTRAINT project_name_unique IF EXISTS"
        )
