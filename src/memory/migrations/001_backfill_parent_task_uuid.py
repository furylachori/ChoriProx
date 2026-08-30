"""Migration 001: Copy parent_task_uuid from JSON attributes to top-level property."""
import json
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor


class BackfillParentTaskUuid(Migration):
    version = 1
    description = "Copy parent_task_uuid from JSON attributes to top-level property"

    def up(self, executor: QueryExecutor) -> None:
        while True:
            rows = executor.execute_read(
                """MATCH (e:Entity)
                   WHERE 'Task' IN labels(e)
                     AND e.parent_task_uuid IS NULL
                     AND e.attributes IS NOT NULL
                     AND e.attributes CONTAINS 'parent_task_uuid'
                   RETURN id(e) AS nid, e.attributes AS attrs
                   LIMIT 500"""
            )
            if not rows:
                break
            updates = []
            for row in rows:
                try:
                    attrs = json.loads(row["attrs"])
                    ptu = attrs.get("parent_task_uuid")
                    if ptu:
                        updates.append({"nid": row["nid"], "ptu": ptu})
                except (json.JSONDecodeError, TypeError):
                    continue
            if updates:
                executor.execute_write_no_retry(
                    """UNWIND $updates AS u
                       MATCH (e:Entity) WHERE id(e) = u.nid
                       SET e.parent_task_uuid = u.ptu""",
                    {"updates": updates}
                )
            if not updates and rows:
                null_updates = [{"nid": row["nid"]} for row in rows]
                executor.execute_write_no_retry(
                    """UNWIND $updates AS u
                       MATCH (e:Entity) WHERE id(e) = u.nid
                       SET e.parent_task_uuid = ''""",
                    {"updates": null_updates}
                )
                break
            if not updates:
                break
