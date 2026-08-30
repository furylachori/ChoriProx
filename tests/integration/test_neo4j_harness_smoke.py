"""Smoke test for the Neo4j test harness."""

import pytest


@pytest.mark.integration_neo4j
def test_neo4j_harness_creates_schema(neo4j_test_db):
    """Verify the harness creates the full schema."""
    executor = neo4j_test_db
    rows = executor.execute_read(
        "MATCH (n) WHERE NOT n:_Migration AND NOT n:MigrationLock RETURN count(n) AS cnt"
    )
    assert rows[0]["cnt"] == 0  # empty (excluding migration metadata)


@pytest.mark.integration_neo4j
def test_neo4j_harness_schema_version(neo4j_test_db):
    """Verify the schema version matches the latest migration."""
    import re
    from pathlib import Path

    migrations_dir = Path(__file__).resolve().parents[2] / "src" / "memory" / "migrations"
    versions = [
        int(m.group(1))
        for f in migrations_dir.glob("*.py")
        if (m := re.match(r"^(\d+)_", f.name))
    ]
    assert versions, "No numbered migrations found — migrations dir moved?"
    expected = max(versions)

    executor = neo4j_test_db
    rows = executor.execute_read(
        "MATCH (m:_Migration) RETURN max(m.version) AS version"
    )
    assert rows[0]["version"] == expected


@pytest.mark.integration_neo4j
def test_neo4j_harness_constraints_exist(neo4j_test_db):
    """Verify key constraints were created by migrations."""
    executor = neo4j_test_db
    rows = executor.execute_read("SHOW CONSTRAINTS")
    names = {r.get("name") for r in rows}
    assert "bug_uuid_unique" in names
    assert "review_uuid_unique" in names


def test_real_or_mock_returns_mock_when_neo4j_down(real_or_mock_neo4j):
    """When Neo4j is unavailable, real_or_mock_neo4j returns a MagicMock."""
    from unittest.mock import MagicMock
    if isinstance(real_or_mock_neo4j, MagicMock):
        assert real_or_mock_neo4j.execute_read.return_value == []
    else:
        # Neo4j is running — the fixture returned a real executor
        rows = real_or_mock_neo4j.execute_read("RETURN 1 AS x")
        assert rows[0]["x"] == 1
