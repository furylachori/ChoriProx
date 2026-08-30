"""Tests for migration 011: Entity dedup + composite uniqueness constraint.

Pins the contract that (name, entity_type) is unique on :Entity nodes,
and that any pre-existing duplicates are merged into a single survivor
with all `:RELATES_TO` and `:MENTIONS` edges re-pointed before the
constraint is installed.

Tests use the `neo4j_test_db` fixture (per-test isolated Enterprise DB on
:7688). If Docker is down, the fixture skips — never remove the skips.

NOTE on test isolation: the harness fixture already runs migration 011
during setup, which installs `entity_name_type_unique` and writes the
`(:_Migration {version: 11, status: 'completed'})` marker. Tests that
need to seed pre-existing duplicates DROP the constraint + DELETE the
migration marker at setup, then call `migration.up()` directly so the
fresh migration sees the duplicates it must dedup.

Marker: `@pytest.mark.integration_neo4j` (consistent with test_graph_store_relations.py).
"""
import threading
from pathlib import Path

import pytest

from memory.graph import Neo4jClient as _RealNeo4jClient


def _drop_constraint_and_marker(executor) -> None:
    """Reset migration 011's side effects so a test can seed duplicates.

    The `neo4j_test_db` fixture already runs migration 011 during setup,
    which creates `entity_name_type_unique`, writes the
    `(:_Migration {version: 11, status: 'completed'})` marker, AND writes
    a `(:MigrationLock {name: 'migration_11', status: 'completed'})` lock
    that prevents the runner from re-running. We drop all three.
    """
    executor.execute_write("DROP CONSTRAINT entity_name_type_unique IF EXISTS")
    executor.execute_write(
        "MATCH (m:_Migration {version: 11}) DETACH DELETE m"
    )
    executor.execute_write(
        "MATCH (lock:MigrationLock {name: 'migration_11'}) DELETE lock"
    )


def _seed_entity(executor, name: str, entity_type: str, first_seen: str | None = None,
                 last_seen: str | None = None) -> None:
    """Create one :Entity node with the given props via raw CREATE.

    Bypasses the composite-key MERGE in `store_entities_batch` so we can
    create true duplicates that the migration must merge.
    """
    props = {"name": name, "entity_type": entity_type, "session_id": "s1"}
    if first_seen is not None:
        props["first_seen"] = first_seen
    if last_seen is not None:
        props["last_seen"] = last_seen
    prop_str = ", ".join(f"{k}: ${k}" for k in props)
    executor.execute_write(
        f"CREATE (e:Entity {{ {prop_str} }})",
        props,
    )


def _seed_relation_for_index(executor, name: str, etype: str, idx: int,
                             tgt: str, fact: str) -> None:
    """Attach a unique RELATES_TO edge from the idx-th duplicate of (name, etype)
    to a fresh target node so dedup yields 3 unique edges after re-pointing.
    """
    executor.execute_write(
        """
        MATCH (d:Entity {name: $name, entity_type: $etype})
        WITH collect(d)[$idx] AS dup
        MATCH (t:Entity {name: $tgt})
        MERGE (dup)-[r:RELATES_TO {relation_type: 'KNOWS', fact: $fact}]->(t)
        ON CREATE SET r.created_at = '2026-01-01T00:00:00Z',
                      r.valid_at = '2026-01-01T00:00:00Z',
                      r.invalid_at = null,
                      r.source_memory = null
        """,
        {"name": name, "etype": etype, "idx": idx, "tgt": tgt, "fact": fact},
    )


@pytest.mark.integration_neo4j
class TestMigration011Dedup:
    def test_dedup_merges_duplicates(self, neo4j_test_db):
        """3 Entity nodes with same (name, entity_type) → 1 survivor, 2 deleted,
        all RELATES_TO edges re-pointed to survivor, sentinel cleaned up."""
        from memory.migrations import EntityDedupConstraint

        # Drop the constraint + marker so the migration can run on a dirty DB.
        _drop_constraint_and_marker(neo4j_test_db)

        # Seed 3 duplicates with distinct first_seen so we can verify aggregation.
        _seed_entity(neo4j_test_db, "foo", "Person", first_seen="2026-01-01T00:00:00Z")
        _seed_entity(neo4j_test_db, "foo", "Person", first_seen="2026-02-01T00:00:00Z")
        _seed_entity(neo4j_test_db, "foo", "Person", first_seen="2026-03-01T00:00:00Z")

        # Each duplicate gets its own outgoing RELATES_TO to a distinct target
        # so that re-pointing all three onto the survivor yields 3 unique edges.
        for i, suffix in enumerate(["a", "b", "c"]):
            tgt = f"target-{suffix}"
            neo4j_test_db.execute_write(
                "CREATE (t:Entity {name: $n, entity_type: 'Person', session_id: 's1'})",
                {"n": tgt},
            )
            _seed_relation_for_index(
                neo4j_test_db, "foo", "Person", i, tgt, f"fact-{suffix}"
            )

        # Run migration 011 directly on the dirty state.
        EntityDedupConstraint().up(neo4j_test_db)

        rows = neo4j_test_db.execute_read(
            "MATCH (e:Entity {name: 'foo', entity_type: 'Person'}) RETURN count(e) AS c"
        )
        assert rows[0]["c"] == 1, f"Expected 1 survivor, got {rows[0]['c']}"

        # Sentinel should be removed in Phase 2.
        sentinel = neo4j_test_db.execute_read(
            "MATCH (e:Entity) WHERE e._m011_dedup IS NOT NULL RETURN count(e) AS c"
        )
        assert sentinel[0]["c"] == 0, "Sentinel property should be removed in Phase 2"

        # 3 RELATES_TO edges from foo → 3 distinct targets (no MERGE collisions).
        rel_count = neo4j_test_db.execute_read(
            """
            MATCH (e:Entity {name: 'foo', entity_type: 'Person'})
                  -[r:RELATES_TO]->()
            RETURN count(r) AS cnt
            """
        )
        assert rel_count[0]["cnt"] == 3, (
            f"Expected 3 RELATES_TO edges re-pointed to survivor, got {rel_count[0]['cnt']}"
        )

        # Survivor retains earliest first_seen.
        survivor = neo4j_test_db.execute_read(
            """MATCH (e:Entity {name: 'foo', entity_type: 'Person'})
               RETURN e.first_seen AS fs"""
        )
        assert survivor[0]["fs"] == "2026-01-01T00:00:00Z", (
            f"Expected earliest first_seen '2026-01-01T00:00:00Z', got {survivor[0]['fs']!r}"
        )

        # Constraint exists (Phase 3).
        constraints = neo4j_test_db.execute_read("SHOW CONSTRAINTS")
        names = {r.get("name") for r in constraints}
        assert "entity_name_type_unique" in names, (
            f"Constraint entity_name_type_unique not found. Got: {names}"
        )

    def test_dedup_handles_null_first_seen(self, neo4j_test_db):
        """3 duplicates with no first_seen property → migration completes, 1 survivor, 2 deleted."""
        from memory.migrations import EntityDedupConstraint

        _drop_constraint_and_marker(neo4j_test_db)

        _seed_entity(neo4j_test_db, "bar", "Concept")
        _seed_entity(neo4j_test_db, "bar", "Concept")
        _seed_entity(neo4j_test_db, "bar", "Concept")

        EntityDedupConstraint().up(neo4j_test_db)

        rows = neo4j_test_db.execute_read(
            "MATCH (e:Entity {name: 'bar', entity_type: 'Concept'}) RETURN count(e) AS c"
        )
        assert rows[0]["c"] == 1, f"Expected 1 survivor, got {rows[0]['c']}"

    def test_dedup_preserves_distinct_entity_types(self, neo4j_test_db):
        """Same name, different entity_type → both survive (different composite keys)."""
        from memory.migrations import EntityDedupConstraint

        _drop_constraint_and_marker(neo4j_test_db)

        _seed_entity(neo4j_test_db, "shared", "Person", first_seen="2026-01-01T00:00:00Z")
        _seed_entity(neo4j_test_db, "shared", "Concept", first_seen="2026-01-01T00:00:00Z")

        EntityDedupConstraint().up(neo4j_test_db)

        rows = neo4j_test_db.execute_read(
            "MATCH (e:Entity {name: 'shared'}) RETURN e.entity_type AS t ORDER BY t"
        )
        types = [r["t"] for r in rows]
        assert types == ["Concept", "Person"], (
            f"Expected both Person and Concept to survive, got {types}"
        )

    def test_dedup_skips_task_entities(self, neo4j_test_db):
        """:Entity:Task and :Entity:Bug nodes have NO entity_type property
        post-004. The migration must leave them alone because NULL is
        excluded from composite uniqueness (NULL != NULL)."""
        from memory.migrations import EntityDedupConstraint

        _drop_constraint_and_marker(neo4j_test_db)

        # Multi-label :Entity:Task and :Entity:Bug nodes WITHOUT entity_type.
        neo4j_test_db.execute_write(
            """
            CREATE (t:Entity:Task {name: 'task-A', uuid: 'task-A-uuid',
                                   status: 'pending', session_id: 's1'}),
                   (b:Entity:Bug {name: 'bug-X', uuid: 'bug-X-uuid',
                                   status: 'open', session_id: 's1'})
            """
        )

        EntityDedupConstraint().up(neo4j_test_db)

        # Both survive untouched.
        task_count = neo4j_test_db.execute_read(
            "MATCH (t:Task) WHERE t.name = 'task-A' RETURN count(t) AS c"
        )
        assert task_count[0]["c"] == 1, "Task node should be untouched"

        bug_count = neo4j_test_db.execute_read(
            "MATCH (b:Bug) WHERE b.name = 'bug-X' RETURN count(b) AS c"
        )
        assert bug_count[0]["c"] == 1, "Bug node should be untouched"

        # Constraint still gets created.
        constraints = neo4j_test_db.execute_read("SHOW CONSTRAINTS")
        names = {r.get("name") for r in constraints}
        assert "entity_name_type_unique" in names, (
            "Migration 011 should still create the constraint even when no Entity "
            "duplicates exist"
        )

    def test_constraint_blocks_concurrent_creates(self, neo4j_test_db):
        """After migration 011 (run by the harness fixture), two simultaneous
        CREATE calls with the same (name, entity_type) must result in only 1
        node — the other fails with ConstraintViolation.

        Uses ``threading.Barrier(2)`` for deterministic simultaneous start
        (NOT sequential — sequential misses the race window entirely).
        """
        from memory.query_executor import ConstraintViolation
        from neo4j.exceptions import ConstraintError as _Neo4jConstraintError

        # Constraint already exists (the harness fixture ran migration 011).
        constraints = neo4j_test_db.execute_read("SHOW CONSTRAINTS")
        names = {r.get("name") for r in constraints}
        assert "entity_name_type_unique" in names, (
            "Test prerequisite: harness must have already run migration 011"
        )

        # Race-prone: seed an Entity node in a separate non-Entity isolation
        # row so two threads will both CREATE the same composite key.
        barrier = threading.Barrier(2)
        results: list[bool | None] = [None, None]
        errors: list[Exception | None] = [None, None]

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=5)
                try:
                    neo4j_test_db.execute_write(
                        "CREATE (e:Entity {name: 'race-target', entity_type: 'X', "
                        "session_id: 's1'})"
                    )
                    results[idx] = True
                except (ConstraintViolation, _Neo4jConstraintError) as e:
                    # _TestQueryExecutor doesn't map errors, so we catch the
                    # raw neo4j ConstraintError too.
                    errors[idx] = e
                    results[idx] = False
            except Exception as e:
                errors[idx] = e
                results[idx] = None

        threads = [
            threading.Thread(target=worker, args=(0,)),
            threading.Thread(target=worker, args=(1,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "Worker thread hung"

        # Exactly one CREATE succeeded.
        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is False)
        assert successes == 1, (
            f"Expected exactly 1 CREATE to succeed, got {successes} "
            f"(results={results}, errors={errors})"
        )
        assert failures == 1, (
            f"Expected exactly 1 CREATE to be blocked, got {failures}"
        )
        assert errors[1 if results[0] else 0] is not None, (
            "Losing thread must have raised ConstraintViolation"
        )

        # Verify exactly 1 node exists.
        rows = neo4j_test_db.execute_read(
            "MATCH (e:Entity {name: 'race-target', entity_type: 'X'}) RETURN count(e) AS c"
        )
        assert rows[0]["c"] == 1, f"Expected 1 surviving node, got {rows[0]['c']}"

    def test_init_schema_creates_constraint(self, neo4j_test_db):
        """Neo4jClient.init_schema() must create entity_name_type_unique via the
        CREATE CONSTRAINT statement (replacing the old DROP statement).

        Drop the constraint first so we can prove init_schema actually CREATES it.
        """
        # Bind a fresh Neo4jClient to the per-test executor. init_schema checks
        # _is_available() (requires _driver + closed circuit breaker) before
        # issuing CREATE statements — so we wire both _executor and _driver.
        client = _RealNeo4jClient.__new__(_RealNeo4jClient)
        client._executor = neo4j_test_db
        client._driver = neo4j_test_db._driver

        # Drop the constraint first (the fixture already created it) so we can
        # prove init_schema CREATES it (not just leaves it alone).
        neo4j_test_db.execute_write("DROP CONSTRAINT entity_name_type_unique IF EXISTS")
        before = {r.get("name") for r in neo4j_test_db.execute_read("SHOW CONSTRAINTS")}
        assert "entity_name_type_unique" not in before, (
            "Pre-condition failed: constraint should be absent after manual drop"
        )

        client.init_schema()

        after = {r.get("name") for r in neo4j_test_db.execute_read("SHOW CONSTRAINTS")}
        assert "entity_name_type_unique" in after, (
            f"Expected entity_name_type_unique constraint after init_schema. Got: {after}"
        )

    def test_migration_idempotent(self, neo4j_test_db):
        """Run migration 011 twice on the same DB → no errors, final state clean."""
        from memory.migrations import EntityDedupConstraint
        from memory.migrations.runner import MigrationRunner

        _drop_constraint_and_marker(neo4j_test_db)

        # Seed one duplicate group.
        _seed_entity(neo4j_test_db, "idem", "Person")
        _seed_entity(neo4j_test_db, "idem", "Person")

        # First run via runner (registers _Migration marker, normal path).
        runner = MigrationRunner(neo4j_test_db)
        runner.register(EntityDedupConstraint())
        runner.run_pending()

        # Second run: registering the same class again is a no-op because the
        # version is already at 11. State should remain clean.
        runner.register(EntityDedupConstraint())
        runner.run_pending()

        rows = neo4j_test_db.execute_read(
            "MATCH (e:Entity {name: 'idem', entity_type: 'Person'}) RETURN count(e) AS c"
        )
        assert rows[0]["c"] == 1

        # Sentinel cleaned.
        sentinel = neo4j_test_db.execute_read(
            "MATCH (e:Entity) WHERE e._m011_dedup IS NOT NULL RETURN count(e) AS c"
        )
        assert sentinel[0]["c"] == 0

        # Constraint still exists.
        constraints = {r.get("name") for r in neo4j_test_db.execute_read("SHOW CONSTRAINTS")}
        assert "entity_name_type_unique" in constraints

    def test_dedup_drops_intra_group_edges(self, neo4j_test_db):
        """Edges between two dupes in the same group must NOT be re-pointed
        as self-loops on the survivor — they are intra-group relationships
        that are meaningless after dedup and would cause self-loop creation.

        Scenario: 3 duplicate (name='intra', entity_type='Person') entities
        with relationships pointing at each other:
          d1 -> d2 (RELATES_TO)
          d2 -> d3 (RELATES_TO)
          d3 -> d1 (RELATES_TO)
        After dedup: only the survivor remains, with NO self-loops.
        """
        from memory.migrations import EntityDedupConstraint

        _drop_constraint_and_marker(neo4j_test_db)

        _seed_entity(neo4j_test_db, "intra", "Person", first_seen="2026-01-01T00:00:00Z")
        _seed_entity(neo4j_test_db, "intra", "Person", first_seen="2026-02-01T00:00:00Z")
        _seed_entity(neo4j_test_db, "intra", "Person", first_seen="2026-03-01T00:00:00Z")

        # Create intra-group edges. Each dupe needs a separate target
        # distinguishable by index, so we use index-based selection.
        neo4j_test_db.execute_write(
            """
            MATCH (a:Entity {name: 'intra', entity_type: 'Person'})
            WITH collect(a) AS nodes
            WITH nodes[0] AS n0, nodes[1] AS n1, nodes[2] AS n2
            // d1 -> d2
            MERGE (n0)-[r1:RELATES_TO {relation_type: 'X', fact: '1->2'}]->(n1)
            // d2 -> d3
            MERGE (n1)-[r2:RELATES_TO {relation_type: 'X', fact: '2->3'}]->(n2)
            // d3 -> d1
            MERGE (n2)-[r3:RELATES_TO {relation_type: 'X', fact: '3->1'}]->(n0)
            """
        )

        EntityDedupConstraint().up(neo4j_test_db)

        # Only the survivor remains.
        rows = neo4j_test_db.execute_read(
            "MATCH (e:Entity {name: 'intra', entity_type: 'Person'}) RETURN count(e) AS c"
        )
        assert rows[0]["c"] == 1, f"Expected 1 survivor, got {rows[0]['c']}"

        # NO self-loops on the survivor.
        self_loops = neo4j_test_db.execute_read(
            """
            MATCH (e:Entity {name: 'intra', entity_type: 'Person'})
                  -[r:RELATES_TO]->(e)
            RETURN count(r) AS c
            """
        )
        assert self_loops[0]["c"] == 0, (
            f"Expected 0 self-loops on survivor, got {self_loops[0]['c']}"
        )

        # NO edges to deleted dupes (they should all be gone).
        edges = neo4j_test_db.execute_read(
            """
            MATCH (e:Entity {name: 'intra', entity_type: 'Person'})
                  -[r:RELATES_TO]->()
            RETURN count(r) AS c
            """
        )
        assert edges[0]["c"] == 0, (
            f"Expected 0 outgoing edges on survivor (intra-group edges dropped), got {edges[0]['c']}"
        )


def test_migration_registered_in_all_three_places():
    """Regression test (D4): migration 011 must be registered in __init__.py,
    sync.py, and neo4j_harness.py. Three-place registration is a recurring
    failure class — every new migration must be wired into all three.
    """
    from memory.migrations import EntityDedupConstraint as _E
    assert _E.version == 11

    _project_root = Path(__file__).resolve().parent.parent.parent
    sync_src = (_project_root / "src/memory/sync.py").read_text()
    assert "EntityDedupConstraint" in sync_src, (
        "sync.py must import EntityDedupConstraint"
    )

    harness_src = (_project_root / "tests/integration/neo4j_harness.py").read_text()
    assert "EntityDedupConstraint" in harness_src, (
        "neo4j_harness.py must register EntityDedupConstraint"
    )

    init_src = (_project_root / "src/memory/migrations/__init__.py").read_text()
    assert "EntityDedupConstraint" in init_src, (
        "__init__.py must export EntityDedupConstraint"
    )