"""Migration 004: Transform tasks to graph-native schema."""
import json
import logging
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor
from proxy.utils import _now_iso

logger = logging.getLogger(__name__)


class GraphNativeTasks(Migration):
    version = 4
    description = "Transform tasks to graph-native schema with relationships and promoted properties"

    def up(self, executor: QueryExecutor) -> None:
        self._step0_extract_planning_review(executor)
        self._step1_blocked_by_relationships(executor)
        self._step2_has_parent_relationships(executor)
        self._step3_promote_tag_remove_entity_type(executor)
        self._step4_remove_attributes(executor)
        self._step5_indexes_and_constraints(executor)

    def _step0_extract_planning_review(self, executor: QueryExecutor) -> None:
        while True:
            result = executor.execute_read(
                """MATCH (t:Task) WHERE t.attributes IS NOT NULL
                   AND t._m004_s0 IS NULL
                   RETURN id(t) AS nid, t.attributes AS attrs
                   LIMIT 500"""
            )
            if not result:
                break

            updates = []
            all_nids = []
            for row in result:
                all_nids.append(row["nid"])
                try:
                    attrs = json.loads(row["attrs"]) if isinstance(row["attrs"], str) else {}
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
                planning_info = attrs.get("planning_info")
                review_notes = attrs.get("review_notes")
                if planning_info or review_notes:
                    updates.append({
                        "nid": row["nid"],
                        "planning_info": planning_info,
                        "review_notes": review_notes,
                    })

            if updates:
                executor.execute_write(
                    """UNWIND $updates AS u
                       MATCH (t:Task) WHERE id(t) = u.nid
                       SET t.planning_info = COALESCE(u.planning_info, t.planning_info),
                           t.review_notes = COALESCE(u.review_notes, t.review_notes)""",
                    {"updates": updates},
                )

            executor.execute_write(
                """UNWIND $nids AS nid
                   MATCH (t:Task) WHERE id(t) = nid
                   SET t._m004_s0 = true""",
                {"nids": all_nids},
            )

            if len(result) < 500:
                break

        while True:
            r = executor.execute_write(
                """MATCH (t:Task) WHERE t._m004_s0 IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t._m004_s0
                   RETURN count(*) AS cleaned"""
            )
            cleaned = r[0]["cleaned"] if r else 0
            if cleaned < 500:
                break

    def _step1_blocked_by_relationships(self, executor: QueryExecutor) -> None:
        total_created = 0
        total_orphans = 0

        while True:
            result = executor.execute_read(
                """MATCH (t:Task) WHERE t.attributes IS NOT NULL
                   AND t._m004_s1 IS NULL
                   RETURN t.uuid AS child_uuid, t.attributes AS attrs
                   LIMIT 500"""
            )
            if not result:
                break

            all_uuids = []
            rels = []
            for row in result:
                all_uuids.append(row["child_uuid"])
                try:
                    attrs = json.loads(row["attrs"]) if isinstance(row["attrs"], str) else {}
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
                blocked_by = attrs.get("blocked_by", [])
                for blocker_uuid in blocked_by:
                    rels.append({
                        "child_uuid": row["child_uuid"],
                        "blocker_uuid": blocker_uuid,
                    })

            created = 0
            if rels:
                write_result = executor.execute_write(
                    """UNWIND $rels AS rel
                       MATCH (child:Task {uuid: rel.child_uuid})
                       OPTIONAL MATCH (blocker:Task {uuid: rel.blocker_uuid})
                       WHERE blocker IS NULL OR blocker.deleted_at IS NULL
                       WITH child, blocker, rel
                       WHERE blocker IS NOT NULL
                       MERGE (child)-[r:BLOCKED_BY]->(blocker)
                        ON CREATE SET r.created_at = $now
                        RETURN count(*) as created""",
                    {"rels": rels, "now": _now_iso()},
                )
                created = write_result[0]["created"] if write_result else 0

            orphans = len(rels) - created
            total_created += created
            total_orphans += orphans

            executor.execute_write(
                """UNWIND $uuids AS uuid
                   MATCH (t:Task {uuid: uuid})
                   SET t._m004_s1 = true""",
                {"uuids": all_uuids},
            )

            if len(result) < 500:
                break

        while True:
            r = executor.execute_write(
                """MATCH (t:Task) WHERE t._m004_s1 IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t._m004_s1
                   RETURN count(*) AS cleaned"""
            )
            cleaned = r[0]["cleaned"] if r else 0
            if cleaned < 500:
                break

        if total_orphans > 0:
            logger.warning(
                "Migration 004 step1: %d orphaned blocked_by references (blocker doesn't exist or is deleted)",
                total_orphans,
            )
        logger.info(
            "Migration 004 step1: created %d BLOCKED_BY relationships, %d orphaned references",
            total_created, total_orphans,
        )

    def _step2_has_parent_relationships(self, executor: QueryExecutor) -> None:
        now = _now_iso()
        
        write_result = executor.execute_write(
            """MATCH (child:Task)
               WHERE child.parent_task_uuid IS NOT NULL AND child.parent_task_uuid <> ''
               OPTIONAL MATCH (parent:Task {uuid: child.parent_task_uuid})
               WHERE parent IS NULL OR parent.deleted_at IS NULL
               WITH child, parent
               WHERE parent IS NOT NULL
               MERGE (child)-[r:HAS_PARENT]->(parent)
               ON CREATE SET r.created_at = $now
               RETURN count(*) as created""",
            {"now": now},
        )
        created = write_result[0]["created"] if write_result else 0

        orphan_result = executor.execute_read(
            """MATCH (child:Task)
               WHERE child.parent_task_uuid IS NOT NULL AND child.parent_task_uuid <> ''
               OPTIONAL MATCH (parent:Task {uuid: child.parent_task_uuid})
               WHERE parent IS NULL OR parent.deleted_at IS NOT NULL
               RETURN count(*) as orphans"""
        )
        orphans = orphan_result[0]["orphans"] if orphan_result else 0

        if orphans > 0:
            logger.warning(
                "Migration 004 step2: %d orphaned parent_task_uuid references (parent doesn't exist or is deleted)",
                orphans,
            )

        executor.execute_write(
            """MATCH (t:Task) WHERE t.parent_task_uuid IS NOT NULL
               REMOVE t.parent_task_uuid"""
        )

        logger.info(
            "Migration 004 step2: created %d HAS_PARENT relationships, %d orphaned references",
            created, orphans,
        )

    def _step3_promote_tag_remove_entity_type(self, executor: QueryExecutor) -> None:
        while True:
            result = executor.execute_read(
                """MATCH (t:Task) WHERE t.attributes IS NOT NULL
                   AND t._m004_s3 IS NULL
                   RETURN id(t) AS nid, t.attributes AS attrs
                   LIMIT 500"""
            )
            if not result:
                break

            tag_updates = []
            all_nids = []
            for row in result:
                all_nids.append(row["nid"])
                try:
                    attrs = json.loads(row["attrs"]) if isinstance(row["attrs"], str) else {}
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
                tag = attrs.get("tag")
                if tag:
                    tag_updates.append({"nid": row["nid"], "tag": tag})

            if tag_updates:
                executor.execute_write(
                    """UNWIND $updates AS u
                       MATCH (t:Task) WHERE id(t) = u.nid
                       SET t.tag = u.tag""",
                    {"updates": tag_updates},
                )

            executor.execute_write(
                """UNWIND $nids AS nid
                   MATCH (t:Task) WHERE id(t) = nid
                   SET t._m004_s3 = true""",
                {"nids": all_nids},
            )

            if len(result) < 500:
                break

        while True:
            r = executor.execute_write(
                """MATCH (t:Task) WHERE t._m004_s3 IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t._m004_s3
                   RETURN count(*) AS cleaned"""
            )
            cleaned = r[0]["cleaned"] if r else 0
            if cleaned < 500:
                break

        while True:
            r = executor.execute_write(
                """MATCH (t:Task) WHERE t.entity_type IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t.entity_type
                   RETURN count(*) AS cleaned"""
            )
            cleaned = r[0]["cleaned"] if r else 0
            if cleaned < 500:
                break

    def _step4_remove_attributes(self, executor: QueryExecutor) -> None:
        while True:
            result = executor.execute_write(
                """MATCH (t:Task) WHERE t.attributes IS NOT NULL
                   WITH t LIMIT 500
                   REMOVE t.attributes
                   RETURN count(*) AS cleaned"""
            )
            cleaned = result[0]["cleaned"] if result else 0
            if cleaned < 500:
                break

    def _step5_indexes_and_constraints(self, executor: QueryExecutor) -> None:
        executor.execute_write(
            "CREATE INDEX task_label_short_hash_idx IF NOT EXISTS FOR (t:Task) ON (t.short_hash)"
        )
        executor.execute_write(
            "CREATE INDEX task_label_workspace_idx IF NOT EXISTS FOR (t:Task) ON (t.workspace_id)"
        )
        executor.execute_write(
            "CREATE INDEX task_label_status_idx IF NOT EXISTS FOR (t:Task) ON (t.status)"
        )
        executor.execute_write(
            "CREATE INDEX task_label_created_at_idx IF NOT EXISTS FOR (t:Task) ON (t.created_at)"
        )
        executor.execute_write(
            "CREATE INDEX task_label_tag_idx IF NOT EXISTS FOR (t:Task) ON (t.tag)"
        )

        executor.execute_write(
            "CREATE CONSTRAINT task_uuid_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.uuid IS UNIQUE"
        )

        try:
            executor.execute_write(
                "DROP CONSTRAINT task_name_workspace_unique IF EXISTS"
            )
        except Exception as e:
            logger.debug("Constraint drop failed (may not exist): %s", e)
        executor.execute_write(
            "CREATE CONSTRAINT task_name_workspace_unique IF NOT EXISTS "
            "FOR (t:Task) REQUIRE (t.name, t.workspace_id) IS UNIQUE"
        )

        for idx in ["parent_task_uuid_idx", "short_hash_idx", "task_workspace_idx", "task_status_idx"]:
            try:
                executor.execute_write(f"DROP INDEX {idx} IF EXISTS")
            except Exception as e:
                logger.debug("Constraint drop failed (may not exist): %s", e)
