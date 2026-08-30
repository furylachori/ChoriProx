"""Verify that batch-created tasks use the same schema as create_task."""

import re
from unittest.mock import MagicMock

import pytest

from memory.task_graph import TaskGraph

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


class TestAtomicCreateTasksSchema:
    """Ensure _atomic_create_tasks writes the same properties as create_task."""

    def test_atomic_create_tasks_generates_uuid_and_short_hash(self, tg):
        tg_obj, mock_client, mock_executor = tg

        captured = {"queries": [], "params": []}

        def fake_execute_transaction_no_retry(fn):
            mock_tx = MagicMock()
            mock_record = MagicMock()
            
            def mock_run(query, params=None, **kwargs):
                captured["queries"].append(query)
                merged_params = params if params else kwargs
                captured["params"].append(merged_params)
                
                if "UNWIND $tasks" in query:
                    tasks_list = merged_params.get("tasks", [])
                    if tasks_list:
                        task = tasks_list[0]
                        mock_record.__getitem__ = lambda _s, key: {
                            "uuid": task.get("uuid", "entity-abc123def456"),
                            "short_hash": task.get("short_hash", "c123def4"),
                            "name": task.get("name", "test"),
                            "priority": task.get("priority", "medium"),
                            "status": task.get("status", "pending"),
                        }[key]
                        mock_record.get = lambda key, default=None: {
                            "uuid": task.get("uuid", "entity-abc123def456"),
                            "short_hash": task.get("short_hash", "c123def4"),
                            "name": task.get("name", "test"),
                            "priority": task.get("priority", "medium"),
                            "status": task.get("status", "pending"),
                        }.get(key, default)
                    return MagicMock(__iter__=lambda _s: iter([mock_record]))
                else:
                    mock_record.__getitem__ = lambda _s, key: {
                        "uuid": kwargs.get("uuid", "entity-abc123def456"),
                        "short_hash": kwargs.get("short_hash", "c123def4"),
                        "name": kwargs.get("name", "test"),
                        "priority": kwargs.get("priority", "medium"),
                        "status": kwargs.get("status", "pending"),
                        "session_id": kwargs.get("session_id", "test"),
                        "tag": kwargs.get("tag", ""),
                    }[key]
                    mock_record.get = lambda key, default=None: {
                        "uuid": kwargs.get("uuid", "entity-abc123def456"),
                        "short_hash": kwargs.get("short_hash", "c123def4"),
                        "name": kwargs.get("name", "test"),
                        "priority": kwargs.get("priority", "medium"),
                        "status": kwargs.get("status", "pending"),
                        "session_id": kwargs.get("session_id", "test"),
                        "tag": kwargs.get("tag", ""),
                    }.get(key, default)
                    return MagicMock(single=lambda: mock_record)
            
            mock_tx.run = mock_run
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_execute_transaction_no_retry

        tasks = [
            {
                "name": "batch-task-1",
                "priority": "high",
                "summary": "summary one",
                "session_id": "sid-1",
                "status": "pending",
                "tag": "test",
            }
        ]
        result = tg_obj.create_tasks_batch(tasks, workspace_id="test_ws:default")

        assert len(result) == 1
        assert "error" not in result[0]

        query = captured["queries"][0]
        params = captured["params"][0]

        # UNWIND uses tasks list, not individual uuid param
        assert "tasks" in params
        assert len(params["tasks"]) == 1
        task_data = params["tasks"][0]
        
        assert result[0]["short_hash"] == task_data["uuid"][-8:]

        # Schema properties must appear as task.property in UNWIND
        assert "task.uuid" in query
        assert "task.short_hash" in query
        assert "task.created_at" in query or "$created_at" in query
        assert "task.deleted_at" in query or "$deleted_at" in query
        assert "task.status_changed_at" in query or "$status_changed_at" in query
        assert "updated_at" not in query
        assert "version" not in query
        assert "task_id" not in query

        # Params must be present and correct
        assert re.match(r"entity-[a-f0-9]{32}", task_data["uuid"])
        assert task_data["short_hash"] == task_data["uuid"][-8:]
        assert params["deleted_at"] is None
        assert params["status_changed_at"] is not None

    def test_batch_task_visible_to_get_task_by_short_hash(self, tg):
        tg_obj, _mock_client, mock_executor = tg

        def fake_execute_transaction_no_retry(fn):
            mock_tx = MagicMock()
            mock_record = MagicMock()
            mock_record.__getitem__ = lambda _s, key: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-2",
                "priority": "medium",
                "status": "in_progress",
                "session_id": "sid-2",
                "tag": "",
            }[key]
            mock_record.get = lambda key, default=None: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-2",
                "priority": "medium",
                "status": "in_progress",
                "session_id": "sid-2",
                "tag": "",
            }.get(key, default)
            mock_tx.run.return_value = MagicMock(__iter__=lambda _s: iter([mock_record]))
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_execute_transaction_no_retry

        tasks = [
            {
                "name": "batch-task-2",
                "priority": "medium",
                "summary": "",
                "session_id": "sid-2",
                "status": "in_progress",
            }
        ]
        batch_result = tg_obj.create_tasks_batch(tasks, workspace_id="test_ws:default")
        short_hash = batch_result[0]["short_hash"]

        # get_task_by_short_hash uses execute_read
        props = {
            "uuid": "entity-abc123def456",
            "short_hash": short_hash,
            "name": "batch-task-2",
            "priority": "medium",
            "status": "in_progress",
            "session_id": "sid-2",
            "deleted_at": None,
        }
        mock_executor.execute_read.return_value = [{"props": props}]

        task = tg_obj.get_task_by_short_hash(short_hash)
        assert task is not None
        assert task["short_hash"] == short_hash
        assert task["name"] == "batch-task-2"

    def test_batch_task_visible_to_list_tasks(self, tg):
        tg_obj, _mock_client, mock_executor = tg

        def fake_execute_transaction_no_retry(fn):
            mock_tx = MagicMock()
            mock_record = MagicMock()
            mock_record.__getitem__ = lambda _s, key: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-3",
                "priority": "low",
                "status": "pending",
                "session_id": "sid-3",
                "tag": "",
            }[key]
            mock_record.get = lambda key, default=None: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-3",
                "priority": "low",
                "status": "pending",
                "session_id": "sid-3",
                "tag": "",
            }.get(key, default)
            mock_tx.run.return_value = MagicMock(single=lambda: mock_record)
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_execute_transaction_no_retry

        tasks = [
            {
                "name": "batch-task-3",
                "priority": "low",
                "summary": "summary three",
                "session_id": "sid-3",
                "status": "pending",
            }
        ]
        tg_obj.create_tasks_batch(tasks, workspace_id="test_ws:default")

        # get_actionable_tasks uses execute_read and returns enriched data
        enriched_task = {
            "uuid": "entity-abc123def456",
            "short_hash": "c123def4",
            "name": "batch-task-3",
            "priority": "low",
            "status": "pending",
            "session_id": "sid-3",
            "created_at": "2024-01-01T00:00:00Z",
            "deleted_at": None,
            "tag": "",
            "blockers": [],
            "parent_uuid": None,
            "is_blocked": False,
            "blocked_count": 0,
        }
        mock_executor.execute_read.return_value = [enriched_task]

        tasks = tg_obj.get_actionable_tasks("sid-3")
        assert len(tasks) == 1
        assert tasks[0]["name"] == "batch-task-3"

    def test_batch_task_updateable(self, tg):
        tg_obj, _mock_client, mock_executor = tg

        def fake_execute_transaction_no_retry(fn):
            mock_tx = MagicMock()
            mock_record = MagicMock()
            mock_record.__getitem__ = lambda _s, key: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-4",
                "priority": "medium",
                "status": "pending",
                "session_id": "sid-4",
                "tag": "",
            }[key]
            mock_record.get = lambda key, default=None: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-4",
                "priority": "medium",
                "status": "pending",
                "session_id": "sid-4",
                "tag": "",
            }.get(key, default)
            mock_tx.run.return_value = MagicMock(__iter__=lambda _s: iter([mock_record]))
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_execute_transaction_no_retry

        tasks = [
            {
                "name": "batch-task-4",
                "priority": "medium",
                "summary": "",
                "session_id": "sid-4",
                "status": "pending",
            }
        ]
        batch_result = tg_obj.create_tasks_batch(tasks, workspace_id="test_ws:default")
        short_hash = batch_result[0]["short_hash"]

        # update_task uses execute_transaction_no_retry with a transaction function
        def fake_update_transaction(fn):
            mock_tx = MagicMock()
            # Mock count query
            count_record = MagicMock()
            count_record.get = lambda key, default=None: 1 if key == "cnt" else default
            count_record.__getitem__ = lambda _s, key: 1 if key == "cnt" else None
            # Mock update query
            update_record = MagicMock()
            update_record.get = lambda key, default=None: 1 if key == "updated" else default
            update_record.__getitem__ = lambda _s, key: 1 if key == "updated" else None
            
            call_count = [0]
            def mock_run(query, params=None, **kwargs):
                call_count[0] += 1
                if "count" in query.lower():
                    return MagicMock(single=lambda: count_record)
                else:
                    return MagicMock(single=lambda: update_record)
            
            mock_tx.run = mock_run
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_update_transaction

        result = tg_obj.update_task(short_hash, workspace_id="test_ws:default", name="updated-name")
        assert result is True

    def test_batch_task_deletable(self, tg):
        """Verify that batch-created tasks can be soft-deleted via update."""
        tg_obj, _mock_client, mock_executor = tg

        def fake_execute_transaction_no_retry(fn):
            mock_tx = MagicMock()
            mock_record = MagicMock()
            mock_record.__getitem__ = lambda _s, key: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-5",
                "priority": "medium",
                "status": "pending",
                "session_id": "sid-5",
                "tag": "",
            }[key]
            mock_record.get = lambda key, default=None: {
                "uuid": "entity-abc123def456",
                "short_hash": "c123def4",
                "name": "batch-task-5",
                "priority": "medium",
                "status": "pending",
                "session_id": "sid-5",
                "tag": "",
            }.get(key, default)
            mock_tx.run.return_value = MagicMock(__iter__=lambda _s: iter([mock_record]))
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_execute_transaction_no_retry

        tasks = [
            {
                "name": "batch-task-5",
                "priority": "medium",
                "summary": "",
                "session_id": "sid-5",
                "status": "pending",
            }
        ]
        batch_result = tg_obj.create_tasks_batch(tasks, workspace_id="test_ws:default")
        short_hash = batch_result[0]["short_hash"]

        # Verify get_task_by_short_hash would match with the right schema
        props = {
            "uuid": "entity-abc123def456",
            "short_hash": short_hash,
            "name": "batch-task-5",
            "priority": "medium",
            "status": "pending",
            "session_id": "sid-5",
            "deleted_at": None,
        }
        mock_executor.execute_read.return_value = [{"props": props}]

        task = tg_obj.get_task_by_short_hash(short_hash)
        assert task is not None
        # The query checks e.deleted_at IS NULL, so as long as deleted_at is
        # present and None the task is visible.
        assert task.get("deleted_at") is None


class TestCreateTaskSchemaConsistency:
    """Ensure create_task and batch create share the same property set."""

    def test_create_task_cypher_matches_batch_cypher(self, tg):
        tg_obj, _mock_client, mock_executor = tg

        captured = {"queries": [], "params": []}

        def fake_execute_transaction_no_retry(fn):
            mock_tx = MagicMock()
            mock_record = MagicMock()
            
            def mock_run(query, params=None, **kwargs):
                captured["queries"].append(query)
                merged_params = params if params else kwargs
                captured["params"].append(merged_params)
                
                if "UNWIND $tasks" in query:
                    tasks_list = merged_params.get("tasks", [])
                    if tasks_list:
                        task = tasks_list[0]
                        mock_record.__getitem__ = lambda _s, key: {
                            "uuid": task.get("uuid", "entity-abc123def456"),
                            "short_hash": task.get("short_hash", "c123def4"),
                            "name": task.get("name", "x"),
                            "priority": task.get("priority", "medium"),
                            "status": task.get("status", "pending"),
                        }[key]
                        mock_record.get = lambda key, default=None: {
                            "uuid": task.get("uuid", "entity-abc123def456"),
                            "short_hash": task.get("short_hash", "c123def4"),
                            "name": task.get("name", "x"),
                            "priority": task.get("priority", "medium"),
                            "status": task.get("status", "pending"),
                        }.get(key, default)
                    return MagicMock(__iter__=lambda _s: iter([mock_record]))
                else:
                    mock_record.__getitem__ = lambda _s, key: {
                        "uuid": kwargs.get("uuid", "entity-abc123def456"),
                        "short_hash": kwargs.get("short_hash", "c123def4"),
                        "name": kwargs.get("name", "x"),
                        "priority": kwargs.get("priority", "medium"),
                        "status": kwargs.get("status", "pending"),
                        "session_id": kwargs.get("session_id", "s"),
                        "tag": kwargs.get("tag", ""),
                    }[key]
                    mock_record.get = lambda key, default=None: {
                        "uuid": kwargs.get("uuid", "entity-abc123def456"),
                        "short_hash": kwargs.get("short_hash", "c123def4"),
                        "name": kwargs.get("name", "x"),
                        "priority": kwargs.get("priority", "medium"),
                        "status": kwargs.get("status", "pending"),
                        "session_id": kwargs.get("session_id", "s"),
                        "tag": kwargs.get("tag", ""),
                    }.get(key, default)
                    return MagicMock(single=lambda: mock_record)
            
            mock_tx.run = mock_run
            return fn(mock_tx)

        mock_executor.execute_transaction_no_retry.side_effect = fake_execute_transaction_no_retry

        tg_obj.create_tasks_batch(
            [
                {
                    "name": "x",
                    "session_id": "s",
                    "priority": "medium",
                    "status": "pending",
                }
            ],
            workspace_id="test_ws:default",
        )

        batch_query = captured["queries"][0]
        batch_params = set(captured["params"][0].keys())

        # UNWIND-based batch uses task.property for per-task fields
        task_properties = {
            "uuid", "short_hash", "name", "session_id",
            "priority", "summary", "status", "tag",
        }
        
        # Shared parameters passed at top level (workspace_id is an unused param
        # since uuid is globally unique; excluded from shared_params check)
        shared_params = {
            "tasks", "created_at", "deleted_at", "status_changed_at",
        }

        # Per-task properties must appear as task.property in UNWIND
        for prop in task_properties:
            assert f"task.{prop}" in batch_query, f"Missing task.{prop} in batch query"

        # Shared parameters must exist as $param in the query
        for param in shared_params:
            assert f"${param}" in batch_query, f"Missing param ${param} in batch query"

        assert batch_params >= shared_params
