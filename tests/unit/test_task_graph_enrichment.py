"""Tests for TaskGraph enrichment and graph-resolution methods."""

import json

import pytest
from unittest.mock import MagicMock

from memory.task_graph import TaskGraph, AmbiguousHashError

pytestmark = pytest.mark.unit


def _setup_operation_mock(mock_client):
    """Configure _operation context manager to delegate to _get_session."""
    ctx = MagicMock()
    ctx.__enter__.side_effect = lambda: mock_client._get_session()
    mock_client._operation.return_value = ctx


# Mock Neo4jClient returns fixed UUIDs — see conftest.py mock_neo4j fixture
@pytest.fixture
def tg(monkeypatch):
    """TaskGraph instance with fully mocked Neo4jClient."""
    mock_client = MagicMock()
    mock_client._is_available.return_value = True

    monkeypatch.setattr("memory.task_graph.Neo4jClient", lambda: mock_client)
    tg_obj = TaskGraph()
    return tg_obj, mock_client, mock_client._executor


def _make_props_record(props: dict):
    """Return a mock Neo4j Record whose .get() works like the real thing."""
    record = MagicMock()
    record.get = lambda key, default=None: props if key == "props" else props.get(key, default)
    return record


class TestResolveBlockedByHashes:
    """Test resolve_blocked_by_hashes validation and resolution."""

    def test_resolve_blocked_by_hashes_success(self, tg):
        """Valid blocker hash resolves to its UUID."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_tasks_by_short_hashes = MagicMock(
            return_value={"abc": {"uuid": "entity-blocker1", "status": "pending"}}
        )

        result = tg_obj.resolve_blocked_by_hashes(["abc"])

        assert result == ["entity-blocker1"]
        tg_obj.get_tasks_by_short_hashes.assert_called_once_with(["abc"], workspace_id=None)

    def test_resolve_blocked_by_hashes_not_found(self, tg):
        """Unknown hash raises ValueError."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_tasks_by_short_hashes = MagicMock(return_value={})

        with pytest.raises(ValueError, match="Blocker not found"):
            tg_obj.resolve_blocked_by_hashes(["xyz"])

    def test_resolve_blocked_by_hashes_ambiguous(self, tg):
        """Ambiguous hash raises AmbiguousHashError."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_tasks_by_short_hashes = MagicMock(
            side_effect=AmbiguousHashError(2)
        )

        with pytest.raises(ValueError, match="Ambiguous hash: 2 tasks match"):
            tg_obj.resolve_blocked_by_hashes(["amb"])

    def test_resolve_blocked_by_hashes_self_block(self, tg):
        """A task cannot block itself."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_tasks_by_short_hashes = MagicMock(
            return_value={"me": {"uuid": "entity-me", "status": "pending"}}
        )

        with pytest.raises(ValueError, match="Task cannot block itself"):
            tg_obj.resolve_blocked_by_hashes(["me"], exclude_uuid="entity-me")

    def test_resolve_blocked_by_hashes_closed_blocker(self, tg):
        """Closed tasks (complete/invalid) cannot be blockers."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_tasks_by_short_hashes = MagicMock(
            return_value={"closed": {"uuid": "entity-closed", "status": "complete"}}
        )

        with pytest.raises(ValueError, match="Closed tasks cannot be blockers"):
            tg_obj.resolve_blocked_by_hashes(["closed"])


class TestResolveParentHash:
    """Test resolve_parent_hash validation and resolution."""

    def test_resolve_parent_hash_success(self, tg):
        """Valid parent hash resolves to its UUID."""
        tg_obj, _mock_client, mock_executor = tg
        tg_obj.get_task_by_short_hash = MagicMock(
            return_value={
                "uuid": "entity-parent1",
                "status": "pending",
            }
        )
        # Mock the HAS_PARENT query
        mock_executor.execute_read.return_value = []

        result = tg_obj.resolve_parent_hash("p1")

        assert result == "entity-parent1"
        tg_obj.get_task_by_short_hash.assert_called_once_with("p1", workspace_id=None)

    def test_resolve_parent_hash_not_found(self, tg):
        """Unknown parent hash raises ValueError."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_task_by_short_hash = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="Parent not found"):
            tg_obj.resolve_parent_hash("unknown")

    def test_resolve_parent_hash_self_parenting(self, tg):
        """A task cannot be its own parent."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_task_by_short_hash = MagicMock(
            return_value={"uuid": "entity-child", "status": "pending"}
        )

        with pytest.raises(ValueError, match="Task cannot be its own parent"):
            tg_obj.resolve_parent_hash("child", child_uuid="entity-child")

    def test_resolve_parent_hash_closed_parent(self, tg):
        """Closed tasks (complete/invalid) cannot be parents."""
        tg_obj, _mock_client, _mock_executor = tg
        tg_obj.get_task_by_short_hash = MagicMock(
            return_value={"uuid": "entity-bad", "status": "invalid"}
        )

        with pytest.raises(ValueError, match="Closed tasks cannot be parents"):
            tg_obj.resolve_parent_hash("bad")


class TestGetChildTasks:
    """Test get_child_tasks uses native HAS_PARENT relationships."""

    def test_get_child_tasks_success(self):
        """Only tasks with HAS_PARENT relationship to the parent are returned."""
        from unittest.mock import patch
        
        with patch("memory.task_graph.Neo4jClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            
            tg_obj = TaskGraph()
            
            # get_child_tasks now uses native relationships, not attributes
            records = [
                {
                    "uuid": "entity-child1",
                    "name": "Child One",
                    "short_hash": "child001",
                    "status": "pending",
                    "priority": "medium",
                    "summary": "",
                    "tag": "",
                    "created_at": "2024-01-01T00:00:00Z",
                },
            ]
            tg_obj._executor.execute_read.return_value = records
            
            children = tg_obj.get_child_tasks("aabb11cc")
            
            assert len(children) == 1
            assert children[0]["name"] == "Child One"

    def test_get_child_tasks_not_found(self, tg):
        """Unknown parent hash returns empty list."""
        tg_obj, _mock_client, mock_executor = tg
        mock_executor.execute_read.return_value = []

        children = tg_obj.get_child_tasks("deadbeef")
        assert children == []


class TestGetActionableTasksEnrichment:
    """Test that get_actionable_tasks returns enriched data with relationships."""

    def test_enrichment_attaches_short_hash(self, tg):
        """Each task gets short_hash from the query."""
        tg_obj, _mock_client, mock_executor = tg

        records = [
            {
                "uuid": "entity-aabb11cc22dd33ee44ff55aa66bb77cc",
                "name": "Test Task",
                "priority": "medium",
                "status": "pending",
                "deleted_at": None,
                "short_hash": "66bb77cc",
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            }
        ]
        mock_executor.execute_read.return_value = records

        result = tg_obj.get_actionable_tasks()

        assert len(result) == 1
        assert result[0]["short_hash"] == "66bb77cc"

    def test_enrichment_attaches_blockers_and_is_blocked(self, tg):
        """Blocker info and is_blocked flag are returned from the query."""
        tg_obj, _mock_client, mock_executor = tg

        records = [
            {
                "uuid": "entity-block1",
                "short_hash": "block001",
                "name": "Blocker Task",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "uuid": "entity-task1",
                "short_hash": "task0001",
                "name": "Blocked Task",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [
                    {"uuid": "entity-block1", "short_hash": "block001", "name": "Blocker Task", "status": "pending"}
                ],
                "is_blocked": True,
                "blocked_count": 1,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
        ]
        mock_executor.execute_read.return_value = records

        result = tg_obj.get_actionable_tasks()

        blocked_task = next(t for t in result if t["uuid"] == "entity-task1")
        assert blocked_task["is_blocked"] is True
        assert blocked_task["blocked_count"] == 1
        assert len(blocked_task["blockers"]) == 1

    def test_enrichment_attaches_parent_fields(self, tg):
        """Child tasks get parent_short_hash and parent_name from the query."""
        tg_obj, _mock_client, mock_executor = tg

        records = [
            {
                "uuid": "entity-abcabcyparent1",
                "short_hash": "parent01",
                "name": "Parent Task",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "uuid": "entity-cccccccccc1",
                "short_hash": "child001",
                "name": "Child Task",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": "entity-abcabcyparent1",
                "parent_short_hash": "parent01",
                "parent_name": "Parent Task",
            },
        ]
        mock_executor.execute_read.return_value = records

        result = tg_obj.get_actionable_tasks()

        child_task = next(t for t in result if t["uuid"] == "entity-cccccccccc1")
        assert child_task["parent_short_hash"] == "parent01"
        assert child_task["parent_name"] == "Parent Task"

    def test_enrichment_sorts_blocked_after_unblocked(self, tg):
        """Within same priority, unblocked tasks appear before blocked ones."""
        tg_obj, _mock_client, mock_executor = tg

        # Records must be in the order that Cypher would return them:
        # ORDER BY priority, blocked_status, name
        records = [
            {
                "uuid": "entity-unblocked",
                "short_hash": "un000001",
                "name": "A Unblocked High",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "uuid": "entity-blocked",
                "short_hash": "bl000001",
                "name": "Z Blocked High",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [{"uuid": "entity-block1", "short_hash": "block001", "name": "Blocker", "status": "pending"}],
                "is_blocked": True,
                "blocked_count": 1,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
        ]
        mock_executor.execute_read.return_value = records

        result = tg_obj.get_actionable_tasks()

        unblocked_idx = next(i for i, t in enumerate(result) if t["uuid"] == "entity-unblocked")
        blocked_idx = next(i for i, t in enumerate(result) if t["uuid"] == "entity-blocked")
        assert unblocked_idx < blocked_idx

    def test_enrichment_preserves_priority_order(self, tg):
        """High priority tasks appear before medium, which appear before low.

        Within each priority tier, unblocked tasks are ordered before blocked.
        """
        tg_obj, _mock_client, mock_executor = tg

        # Records must be in the order that Cypher would return them:
        # ORDER BY priority (high=1, medium=2, low=3), blocked_status (unblocked=0, blocked=1), name ASC
        records = [
            {
                "uuid": "entity-block1",
                "short_hash": "block0001",
                "name": "Blocker Task",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "uuid": "entity-high-unblocked",
                "short_hash": "high0002",
                "name": "High Unblocked",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "uuid": "entity-high-blocked",
                "short_hash": "high0001",
                "name": "High Blocked",
                "priority": "high",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [{"uuid": "entity-block1", "short_hash": "block0001", "name": "Blocker", "status": "pending"}],
                "is_blocked": True,
                "blocked_count": 1,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
            {
                "uuid": "entity-medium",
                "short_hash": "med00001",
                "name": "Medium Task",
                "priority": "medium",
                "status": "pending",
                "deleted_at": None,
                "summary": "",
                "tag": "",
                "created_at": "2024-01-01T00:00:00Z",
                "blockers": [],
                "is_blocked": False,
                "blocked_count": 0,
                "parent_task_uuid": None,
                "parent_short_hash": None,
                "parent_name": None,
            },
        ]
        mock_executor.execute_read.return_value = records

        result = tg_obj.get_actionable_tasks()

        # First task is the standalone high task (first alphabetically)
        # Then the high unblocked task, then the high blocked task
        # Then the medium task (lower priority)
        assert result[0]["uuid"] == "entity-block1"
        assert result[1]["uuid"] == "entity-high-unblocked"
        assert result[2]["uuid"] == "entity-high-blocked"
        assert result[3]["uuid"] == "entity-medium"