"""
Unit tests for CreateTasksAction — Phase 2/3 batch-internal name resolution.
Tests deferred wiring, NFKC normalization, ordering constraints, and known_tasks handling.
"""

import pytest
from unittest.mock import MagicMock, call

from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_tg():
    """Fully mocked TaskGraph with controllable return values."""
    tg = MagicMock()
    tg.is_available.return_value = True
    tg.get_task_by_short_hash.return_value = None
    tg.update_task.return_value = True
    return tg


class TestBlockedByBatchInternalResolution:
    """Tests for blocked_by batch-internal name resolution (§6.2 Phase 2)."""

    def test_blocked_by_batch_internal_wires_via_update_task(self, mock_tg):
        """blocked_by referencing a sibling task should wire after creation via update_task."""
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "blockertask_hash", "uuid": "entity-blockertask-00000001", "name": "BlockerTask", "priority": "high", "status": "pending"},
            {"short_hash": "blockedtask_hash", "uuid": "entity-blockedtask-00000002", "name": "BlockedTask", "priority": "medium", "status": "pending"},
        ]
        arguments = {
            "tasks": [
                {"name": "BlockerTask", "priority": "high"},
                {"name": "BlockedTask", "priority": "medium", "blocked_by": ["BlockerTask"]},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.create_tasks_batch.assert_called_once()
        update_calls = mock_tg.update_task.call_args_list
        assert len(update_calls) >= 1
        wired = False
        for c in update_calls:
            kw = c[1] if c[1] else {}
            if "blocked_by" in kw and "entity-blockertask-00000001" in kw["blocked_by"]:
                wired = True
                break
        assert wired, f"blocked_by wiring not found in update_task calls: {update_calls}"

    def test_blocked_by_string_not_list_returns_error(self, mock_tg):
        """blocked_by as string instead of list should return error (B3 validation)."""
        arguments = {
            "tasks": [
                {"name": "T1", "priority": "high", "blocked_by": "bad"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert result["error"]["code"] == "invalid_input"
        assert "blocked_by must be an array" in result["error"].get("message", "")

    def test_blocked_by_empty_list_allowed(self, mock_tg):
        """Empty blocked_by list should be allowed and skip deferred wiring."""
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "t1_hash", "name": "T1", "priority": "high", "status": "pending"},
        ]
        arguments = {
            "tasks": [
                {"name": "T1", "priority": "high", "blocked_by": []},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        update_calls = mock_tg.update_task.call_args_list
        wiring_calls = []
        for c in update_calls:
            kw = c[1] if c[1] else {}
            if "blocked_by" in kw:
                wiring_calls.append(kw)
        assert len(wiring_calls) == 0, f"Unexpected blocked_by wiring calls: {wiring_calls}"

    def test_blocked_by_ordering_violation(self, mock_tg):
        """blocked_by referencing a task that appears later in batch should return ordering_violation."""
        arguments = {
            "tasks": [
                {"name": "BlockedTask", "priority": "medium", "blocked_by": ["BlockerTask"]},
                {"name": "BlockerTask", "priority": "high"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert result["error"]["code"] == "ordering_violation"
        assert "blocked_by" in result["error"].get("message", "")

    def test_blocked_by_resolution_error_propagates(self, mock_tg):
        """Error from blocked_by resolution should propagate in wiring_errors."""
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "blockertask_hash", "uuid": "entity-blockertask-00000001", "name": "BlockerTask", "priority": "high", "status": "pending"},
            {"short_hash": "blockedtask_hash", "uuid": "entity-blockedtask-00000002", "name": "BlockedTask", "priority": "medium", "status": "pending"},
        ]
        mock_tg.update_task.return_value = False
        arguments = {
            "tasks": [
                {"name": "BlockerTask", "priority": "high"},
                {"name": "BlockedTask", "priority": "medium", "blocked_by": ["BlockerTask"]},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        # Partial-failure scenario: status is "partial" and wiring_errors lives in data
        assert result["status"] == "partial"
        assert "wiring_errors" in result["data"]
        assert len(result["data"]["wiring_errors"]) == 1
        assert result["data"]["wiring_errors"][0]["field"] == "blocked_by"


class TestParentHashBatchInternalResolution:
    """Tests for parent_hash batch-internal name resolution (§6.2 Phase 2)."""

    def test_parent_hash_batch_internal_wires_via_update_task(self, mock_tg):
        """parent_hash referencing a sibling task should wire after creation via update_task."""
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "parenttask_hash", "uuid": "entity-parenttask-00000001", "name": "ParentTask", "priority": "high", "status": "pending"},
            {"short_hash": "childtask_hash", "uuid": "entity-childtask-00000002", "name": "ChildTask", "priority": "medium", "status": "pending"},
        ]
        arguments = {
            "tasks": [
                {"name": "ParentTask", "priority": "high"},
                {"name": "ChildTask", "priority": "medium", "parent_hash": "ParentTask"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.create_tasks_batch.assert_called_once()
        update_calls = mock_tg.update_task.call_args_list
        assert len(update_calls) >= 1
        wired = False
        for c in update_calls:
            kw = c[1] if c[1] else {}
            if kw.get("parent_uuid") == "entity-parenttask-00000001":
                wired = True
                break
        assert wired, f"parent_hash wiring not found in update_task calls: {update_calls}"

    def test_parent_hash_ordering_violation(self, mock_tg):
        """parent_hash referencing a task that appears later in batch should return ordering_violation."""
        arguments = {
            "tasks": [
                {"name": "ChildTask", "priority": "medium", "parent_hash": "ParentTask"},
                {"name": "ParentTask", "priority": "high"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert result["error"]["code"] == "ordering_violation"
        assert "parent_hash" in result["error"].get("message", "")

    def test_parent_hash_known_tasks_wins_over_batch(self, mock_tg):
        """When parent_hash reference is only in known_tasks (not batch), known_tasks hash should be used."""
        known_tasks = [{"name": "SharedTask", "short_hash": "knownhash"}]
        mock_tg.resolve_parent_hash.return_value = "knownhash_resolved"
        mock_tg.get_tasks_by_short_hashes.return_value = {"knownhash": {"uuid": "known-uuid", "short_hash": "knownhash", "name": "SharedTask"}}
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "childhash", "uuid": "entity-childhash-00000001", "name": "ChildTask", "priority": "medium", "status": "pending"},
        ]
        arguments = {
            "tasks": [
                {"name": "ChildTask", "priority": "medium", "parent_hash": "SharedTask"},
            ],
            "known_tasks": known_tasks,
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.resolve_parent_hash.assert_called_with("knownhash", workspace_id=None)

    def test_parent_hash_resolution_error_propagates(self, mock_tg):
        """Error from parent_hash resolution (via resolve_parent_hash) should propagate."""
        mock_tg.resolve_parent_hash.side_effect = ValueError("Parent not found: SharedTask")
        known_tasks = [{"name": "SharedTask", "short_hash": "knownhash"}]
        arguments = {
            "tasks": [
                {"name": "ChildTask", "priority": "medium", "parent_hash": "SharedTask"},
            ],
            "known_tasks": known_tasks,
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert "Parent not found" in result["error"].get("message", "")


class TestTagPreservation:
    """Tests for tag in task creation (§6.3)."""

    def test_tag_preserved_in_task_dict(self, mock_tg):
        """Tag should be preserved as a direct field passed to create_tasks_batch."""
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "t1_hash", "name": "T1", "priority": "high", "status": "pending"},
        ]
        arguments = {
            "tasks": [
                {"name": "T1", "priority": "high", "tag": "A"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.create_tasks_batch.assert_called_once()
        call_args = mock_tg.create_tasks_batch.call_args[0][0]
        assert len(call_args) == 1
        task_dict = call_args[0]
        assert task_dict["tag"] == "A"
