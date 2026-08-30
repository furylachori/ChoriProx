"""Migration 011: dedup duplicate (name, entity_type) Entity nodes + create uniqueness constraint.

Phase 1: Find `:Entity` nodes with both `name` and `entity_type` set that share
         the same composite key. For each duplicate group, pick a survivor
         (earliest `first_seen`, tie-break `elementId()`), re-point all
         `:RELATES_TO` and `:MENTIONS` relationships from dupes to survivor,
         aggregate `first_seen`/`last_seen`, mark survivor with sentinel
         `_m011_dedup = 'merged'`, and DETACH DELETE the dupes. Atomic per
         group via `execute_transaction_no_retry`. Sentinel-guarded for
         idempotency.

Phase 2: Sentinel cleanup (parity with migration 004 pattern). Removes the
         `_m011_dedup` property so the migration leaves the schema clean.

Phase 3: Create the composite uniqueness constraint
         `(e.name, e.entity_type) IS UNIQUE` on `:Entity`. This is the guard
         against concurrent MERGE races in `store_entities_batch`.

Note: The original plan called for a filtered constraint
`WHERE e.entity_type <> 'Task'`, but Neo4j 2026.05.0 does not support `WHERE`
on `CREATE CONSTRAINT ... IS UNIQUE` (verified). Unfiltered is correct because
`:Entity:Task` nodes have NULL `entity_type` (removed by migration 004) and
NULL values are excluded from composite uniqueness constraints (NULL != NULL).

Caveat: If `up()` raises after the per-group transaction begins, the
`MigrationRunner` writes `_Migration {version: 11, status: 'failed'}` and
`get_current_version()` returns 11 on subsequent boots — the migration
will not auto-retry. Workaround: manually `MATCH (m:_Migration {version:
11}) DELETE m` to clear the failed marker, then restart. This is a
pre-existing runner limitation affecting all migrations 001–010 too;
fixing it is out of scope for migration 011.
"""
import logging
from memory.migrations.base import Migration
from memory.query_executor import QueryExecutor

logger = logging.getLogger(__name__)


CYPHER_FIND_DUPLICATE_GROUPS = """
MATCH (e:Entity)
WHERE e._m011_dedup IS NULL
  AND e.name IS NOT NULL
  AND e.entity_type IS NOT NULL
WITH e.name AS name, e.entity_type AS etype, e
ORDER BY name, etype,
         coalesce(e.first_seen, '9999-99-99T99:99:99Z'),
         elementId(e)
WITH name, etype, collect(e) AS nodes
WHERE size(nodes) > 1
RETURN id(head(nodes)) AS survivor_id,
       elementId(head(nodes)) AS survivor_eid,
       [n IN nodes WHERE elementId(n) <> elementId(head(nodes)) | id(n)] AS dupe_ids
LIMIT $limit
"""


def _merge_duplicate_group(tx, survivor_id: int, dupe_ids: list) -> None:
    """Merge a single (name, entity_type) duplicate group inside a transaction.

    Steps:
      1. Re-point outgoing `:RELATES_TO` from dupes to survivor (MERGE on
         `(relation_type, fact)` identity from `store_relations_batch`).
         Intra-group edges (dupe→dupe or dupe→survivor) are filtered out:
         re-pointing them would create self-loops on the survivor or
         duplicate intra-group relationships that are meaningless after dedup.
      2. Re-point incoming `:RELATES_TO` to dupes → survivor. Same intra-
         group filter applies (survivor→dupe edges are also dropped).
      3. Re-point `:MENTIONS` from memories → survivor (one per (memory, entity)).
         `:Memory` is a different label from `:Entity`, so the `m` nodes are
         never in `dupe_ids` and no intra-group filter is needed here.
      4. Aggregate survivor's `first_seen` (earliest) and `last_seen` (latest)
         across dupes.
      5. Set sentinel `_m011_dedup = 'merged'` on survivor BEFORE deleting
         dupes so a partial transaction leaves the group recoverable.
      6. DETACH DELETE the dupes.
    """
    params = {"dupe_ids": dupe_ids, "survivor_id": survivor_id}

    tx.run(
        """
        UNWIND $dupe_ids AS did
        MATCH (d) WHERE id(d) = did
        MATCH (d)-[r:RELATES_TO]->(other)
        WHERE NOT id(other) IN $dupe_ids
          AND id(other) <> $survivor_id
        WITH d, r, other, $survivor_id AS sid
        MATCH (s) WHERE id(s) = sid
        MERGE (s)-[r2:RELATES_TO {relation_type: r.relation_type, fact: r.fact}]->(other)
        ON CREATE SET r2.source_memory = r.source_memory,
                      r2.created_at = r.created_at,
                      r2.valid_at = r.valid_at,
                      r2.invalid_at = r.invalid_at
        ON MATCH SET r2.valid_at = r.valid_at,
                     r2.invalid_at = r.invalid_at
        """,
        params,
    )

    tx.run(
        """
        UNWIND $dupe_ids AS did
        MATCH (d) WHERE id(d) = did
        MATCH (other)-[r:RELATES_TO]->(d)
        WHERE NOT id(other) IN $dupe_ids
          AND id(other) <> $survivor_id
        WITH d, r, other, $survivor_id AS sid
        MATCH (s) WHERE id(s) = sid
        MERGE (other)-[r2:RELATES_TO {relation_type: r.relation_type, fact: r.fact}]->(s)
        ON CREATE SET r2.source_memory = r.source_memory,
                      r2.created_at = r.created_at,
                      r2.valid_at = r.valid_at,
                      r2.invalid_at = r.invalid_at
        ON MATCH SET r2.valid_at = r.valid_at,
                     r2.invalid_at = r.invalid_at
        """,
        params,
    )

    tx.run(
        """
        UNWIND $dupe_ids AS did
        MATCH (d) WHERE id(d) = did
        MATCH (m)-[r:MENTIONS]->(d)
        WITH m, r, $survivor_id AS sid
        MATCH (s) WHERE id(s) = sid
        MERGE (m)-[r2:MENTIONS]->(s)
        ON CREATE SET r2.created_at = r.created_at
        """,
        params,
    )

    tx.run(
        """
        UNWIND $dupe_ids AS did
        MATCH (d) WHERE id(d) = did
        WITH collect(d) AS dupes, $survivor_id AS sid
        MATCH (s) WHERE id(s) = sid
        SET s.first_seen = reduce(acc = s.first_seen, d IN dupes |
                          CASE WHEN d.first_seen IS NULL THEN acc
                               WHEN acc IS NULL THEN d.first_seen
                               WHEN d.first_seen < acc THEN d.first_seen
                               ELSE acc
                          END),
            s.last_seen = reduce(acc = s.last_seen, d IN dupes |
                          CASE WHEN d.last_seen IS NULL THEN acc
                               WHEN acc IS NULL THEN d.last_seen
                               WHEN d.last_seen > acc THEN d.last_seen
                               ELSE acc
                          END)
        """,
        params,
    )

    tx.run(
        """
        MATCH (s) WHERE id(s) = $sid
        SET s._m011_dedup = 'merged'
        """,
        {"sid": survivor_id},
    )

    tx.run(
        """
        UNWIND $dupe_ids AS did
        MATCH (d) WHERE id(d) = did
        DETACH DELETE d
        """,
        {"dupe_ids": dupe_ids},
    )


class EntityDedupConstraint(Migration):
    version = 11
    description = "Dedup (name, entity_type) Entity duplicates and create uniqueness constraint"

    def up(self, executor: QueryExecutor) -> None:
        BATCH_SIZE = 100

        # Phase 1: Paginated dedup of duplicate groups. Each group is processed
        # in its own transaction via execute_transaction_no_retry so a partial
        # failure leaves the schema recoverable (sentinel marks progress).
        total_groups = 0
        while True:
            rows = executor.execute_read(
                CYPHER_FIND_DUPLICATE_GROUPS,
                {"limit": BATCH_SIZE},
            )
            if not rows:
                break

            for row in rows:
                executor.execute_transaction_no_retry(
                    lambda tx, sid=row["survivor_id"], dids=row["dupe_ids"]: _merge_duplicate_group(tx, sid, dids)
                )
            total_groups += len(rows)

            logger.info(
                "Migration 011 dedup: processed batch of %d duplicate groups (cumulative: %d)",
                len(rows), total_groups,
            )

            if len(rows) < BATCH_SIZE:
                break

        if total_groups > 0:
            logger.info(
                "Migration 011 dedup: merged %d duplicate (name, entity_type) Entity groups",
                total_groups,
            )

        # Phase 2: Sentinel cleanup (parity with migration 004).
        executor.execute_write(
            "MATCH (e:Entity) WHERE e._m011_dedup IS NOT NULL "
            "REMOVE e._m011_dedup RETURN count(e) AS cleaned"
        )

        # Phase 3: Create the composite uniqueness constraint.
        # Unfiltered (no WHERE): Neo4j 2026.05 does not support WHERE on
        # CREATE CONSTRAINT. Task/Bug nodes have NULL entity_type and are
        # auto-excluded from composite uniqueness (NULL != NULL).
        executor.execute_write(
            "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.name, e.entity_type) IS UNIQUE"
        )
        logger.info(
            "Migration 011 applied: entity_name_type_unique constraint + %d duplicate groups merged",
            total_groups,
        )

    def down(self, executor: QueryExecutor) -> None:
        pass