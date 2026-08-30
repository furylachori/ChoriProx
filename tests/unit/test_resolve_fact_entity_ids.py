"""CG-2: _resolve_fact_entity_ids must return non-None source_type/target_type
when source_id/target_id are non-None. The Neo4j side relies on this invariant
to operate on the (name, entity_type) composite key.
"""
from unittest.mock import MagicMock

from memory.persistence import _resolve_fact_entity_ids


def _make_conn(db_rows=None):
    """Build a mock SQLite connection with the relevant rows."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = db_rows or []
    return conn


def _call(conn, facts, entity_id_map=None, session_id="s1", now="2026-01-01T00:00:00Z"):
    result = {"errors": []}
    return _resolve_fact_entity_ids(
        conn, entity_id_map or {}, facts, session_id, now, result,
    ), result


class TestResolveFactEntityIds:
    def test_source_type_set_from_current_batch(self):
        conn = _make_conn()
        facts = [{
            "source_entity_name": "Alice", "target_entity_name": "Bob",
            "source_entity_type": "Person", "target_entity_type": "Person",
            "relationship_type": "KNOWS", "fact": "Alice knows Bob",
        }]
        entity_id_map = {("Alice", "Person"): 1, ("Bob", "Person"): 2}
        resolved, _ = _call(conn, facts, entity_id_map)
        assert len(resolved) == 1
        assert resolved[0]["source_type"] == "Person"
        assert resolved[0]["target_type"] == "Person"

    def test_source_type_set_from_db_lookup(self):
        conn = _make_conn(db_rows=[
            {"id": 10, "name": "Alice", "entity_type": "Person"},
            {"id": 20, "name": "Bob", "entity_type": "Person"},
        ])
        facts = [{
            "source_entity_name": "Alice", "target_entity_name": "Bob",
            "relationship_type": "KNOWS", "fact": "Alice knows Bob",
        }]
        resolved, _ = _call(conn, facts)
        assert len(resolved) == 1
        assert resolved[0]["source_type"] == "Person"
        assert resolved[0]["target_type"] == "Person"

    def test_source_type_never_none_when_id_resolved(self):
        conn = _make_conn(db_rows=[
            {"id": 10, "name": "Alice", "entity_type": "Person"},
            {"id": 20, "name": "Bob", "entity_type": "Person"},
        ])
        facts = [{
            "source_entity_name": "Alice", "target_entity_name": "Bob",
            "relationship_type": "KNOWS", "fact": "Alice knows Bob",
        }]
        resolved, _ = _call(conn, facts)
        assert len(resolved) == 1
        rf = resolved[0]
        assert rf["source_id"] is not None
        assert rf["target_id"] is not None
        assert rf["source_type"] is not None
        assert rf["target_type"] is not None
        assert rf["source_type"] == "Person"
        assert rf["target_type"] == "Person"

    def test_missing_entity_skipped(self):
        conn = _make_conn()
        facts = [{
            "source_entity_name": "Alice", "target_entity_name": "Bob",
            "relationship_type": "KNOWS", "fact": "Alice knows Bob",
        }]
        resolved, result = _call(conn, facts)
        assert resolved == []
        assert len(result["errors"]) >= 1

    def test_extraction_type_takes_priority(self):
        conn = _make_conn()
        facts = [{
            "source_entity_name": "Alice", "target_entity_name": "Bob",
            "source_entity_type": "Person", "target_entity_type": "Person",
            "relationship_type": "KNOWS", "fact": "Alice knows Bob",
        }]
        entity_id_map = {("Alice", "Person"): 1, ("Bob", "Person"): 2}
        resolved, _ = _call(conn, facts, entity_id_map)
        assert resolved[0]["source_type"] == "Person"
        assert resolved[0]["target_type"] == "Person"
