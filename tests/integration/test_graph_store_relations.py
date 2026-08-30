"""CG-2: store_relations_batch must match by (name, entity_type) composite key.

The previous query matched by name only when entity_type was None, creating
a Cartesian product when multiple Entity nodes shared a name. After the fix,
the extraction path always supplies entity types, and the match is exact.
"""
import pytest

from memory.graph import Neo4jClient as _RealNeo4jClient


def _make_g(executor):
    """Construct a Neo4jClient-compatible object wired to the test executor.

    Snapshots the real ``Neo4jClient`` class at import time and binds the two
    batch methods we need onto a fresh instance. Necessary because the autouse
    ``mock_neo4j_for_integration`` fixture in tests/integration/conftest.py
    patches ``memory.graph.Neo4jClient`` to a lambda before each test runs,
    so a direct ``from memory.graph import Neo4jClient`` would yield the lambda.
    """
    out = _RealNeo4jClient.__new__(_RealNeo4jClient)
    out._executor = executor
    return out


@pytest.mark.integration_neo4j
def test_single_type_no_fanout(neo4j_test_db):
    """5 relations between distinct Person entities → 5 edges."""
    g = _make_g(neo4j_test_db)
    entities = [
        {
            "name": f"Person{i}", "entity_type": "Person", "session_id": "s1",
            "first_seen": "2026-01-01T00:00:00Z", "last_seen": "2026-01-01T00:00:00Z",
        }
        for i in range(3)
    ]
    g.store_entities_batch(entities, session_id="s1")
    relations = [
        {
            "source_name": "Person0", "target_name": "Person1",
            "source_entity_type": "Person", "target_entity_type": "Person",
            "relation_type": "KNOWS", "fact": f"Person0 knows Person1 {i}",
            "source_memory_id": None, "created_at": "2026-01-01T00:00:00Z",
            "valid_at": "2026-01-01T00:00:00Z", "invalid_at": None,
        }
        for i in range(5)
    ]
    g.store_relations_batch(relations, session_id="s1")
    rows = neo4j_test_db.execute_read(
        "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt"
    )
    assert rows[0]["cnt"] == 5


@pytest.mark.integration_neo4j
def test_multi_type_no_fanout(neo4j_test_db):
    """Two Person entities and two Project entities (different types, same session) →
    1 relation Person0→Person1 → 1 edge (NOT 4 fan-out).

    Creates true duplicate Entity nodes via raw CREATE (bypassing the
    composite-key MERGE in store_entities_batch) so the test exercises the
    Cartesian-product failure mode the fix prevents. With the old query
    (MATCH by name only), 2 Alice × 2 Bob = 4 candidate endpoint pairs
    would each receive an edge → count == 4. The new composite-key MATCH
    narrows to exactly one (name, entity_type) endpoint per side → count == 1.
    """
    g = _make_g(neo4j_test_db)
    neo4j_test_db.execute_write(
        """
        CREATE (a1:Entity {name: 'Alice', entity_type: 'Person', session_id: 's1'}),
               (a2:Entity {name: 'Alice', entity_type: 'Project', session_id: 's1'}),
               (b1:Entity {name: 'Bob',   entity_type: 'Person', session_id: 's1'}),
               (b2:Entity {name: 'Bob',   entity_type: 'Project', session_id: 's1'})
        """,
    )
    relations = [
        {"source_name": "Alice", "target_name": "Bob",
         "source_entity_type": "Person", "target_entity_type": "Person",
         "relation_type": "KNOWS", "fact": "Alice knows Bob",
         "source_memory_id": None, "created_at": "2026-01-01T00:00:00Z",
         "valid_at": "2026-01-01T00:00:00Z", "invalid_at": None},
    ]
    g.store_relations_batch(relations, session_id="s1")
    rows = neo4j_test_db.execute_read(
        "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt"
    )
    assert rows[0]["cnt"] == 1


@pytest.mark.integration_neo4j
def test_null_type_no_fanout(neo4j_test_db):
    """The original production bug: source_entity_type/target_entity_type = None
    against duplicate Entity nodes produces 4 edges under the old name-only MATCH.

    With the new composite-key MATCH (name, entity_type), passing None for
    entity_type cannot match a node that has a non-null type → 0 edges
    (silent skip, not a fan-out crash). This locks the contract:
    CG-2 requires callers to supply entity types (persistence layer fix).
    """
    g = _make_g(neo4j_test_db)
    neo4j_test_db.execute_write(
        """
        CREATE (a1:Entity {name: 'Alice', entity_type: 'Person', session_id: 's1'}),
               (a2:Entity {name: 'Alice', entity_type: 'Project', session_id: 's1'}),
               (b1:Entity {name: 'Bob',   entity_type: 'Person', session_id: 's1'}),
               (b2:Entity {name: 'Bob',   entity_type: 'Project', session_id: 's1'})
        """,
    )
    relations = [
        {"source_name": "Alice", "target_name": "Bob",
         "source_entity_type": None, "target_entity_type": None,
         "relation_type": "KNOWS", "fact": "Alice knows Bob",
         "source_memory_id": None, "created_at": "2026-01-01T00:00:00Z",
         "valid_at": "2026-01-01T00:00:00Z", "invalid_at": None},
    ]
    g.store_relations_batch(relations, session_id="s1")
    rows = neo4j_test_db.execute_read(
        "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt"
    )
    assert rows[0]["cnt"] == 0


@pytest.mark.integration_neo4j
def test_missing_entities_skipped(neo4j_test_db):
    """2 relations where one is missing → only 1 edge matched."""
    g = _make_g(neo4j_test_db)
    entities = [
        {"name": "Alice", "entity_type": "Person", "session_id": "s1",
         "first_seen": "2026-01-01T00:00:00Z", "last_seen": "2026-01-01T00:00:00Z"},
        {"name": "Bob", "entity_type": "Person", "session_id": "s1",
         "first_seen": "2026-01-01T00:00:00Z", "last_seen": "2026-01-01T00:00:00Z"},
    ]
    g.store_entities_batch(entities, session_id="s1")
    relations = [
        {"source_name": "Alice", "target_name": "Bob",
         "source_entity_type": "Person", "target_entity_type": "Person",
         "relation_type": "KNOWS", "fact": "Alice knows Bob",
         "source_memory_id": None, "created_at": "2026-01-01T00:00:00Z",
         "valid_at": "2026-01-01T00:00:00Z", "invalid_at": None},
        {"source_name": "Alice", "target_name": "MissingEntity",
         "source_entity_type": "Person", "target_entity_type": "Person",
         "relation_type": "KNOWS", "fact": "Alice knows Missing",
         "source_memory_id": None, "created_at": "2026-01-01T00:00:00Z",
         "valid_at": "2026-01-01T00:00:00Z", "invalid_at": None},
    ]
    ok = g.store_relations_batch(relations, session_id="s1")
    assert ok is True
    rows = neo4j_test_db.execute_read(
        "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt"
    )
    assert rows[0]["cnt"] == 1
