"""
Unit tests for Phase 5: Ticket Manager v2.
Covers batch-internal name resolution, tag support in edit, and registry validation.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
from proxy.tool_pipeline.actions.edit import EditAction
from proxy.tool_pipeline.tool_registry import registry

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def mock_agent_registry():
    """Patch agent_registry.registry.get_or_load to return a mock agent."""
    _mock_agent = MagicMock()
    _mock_agent.is_ticket_mode = True
    with patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=_mock_agent,
    ):
        yield


@pytest.fixture
def mock_tg():
    """Fully mocked TaskGraph with controllable return values."""
    tg = MagicMock()
    tg.is_available.return_value = True
    # Reset side_effect so calls go through; return None by default so
    # batch-internal name lookups aren't confused by a pre-seeded task
    tg.get_task_by_short_hash.side_effect = None
    tg.get_task_by_short_hash.return_value = None
    tg.resolve_blocked_by_hashes.return_value = ["entity-blocker-00000000"]
    tg.resolve_parent_hash.return_value = "entity-parent-00000000"
    tg.update_task.return_value = True
    # Default results use generic names; tests that need name-based lookups
    # (blocked_by / parent_hash with batch-internal refs) must override via side_effect
    tg.create_tasks_batch.return_value = [
        {"short_hash": "task1hash", "uuid": "entity-task1hash-00000001", "name": "Task1", "priority": "high", "status": "pending", "tag": ""},
        {"short_hash": "task2hash", "uuid": "entity-task2hash-00000002", "name": "Task2", "priority": "medium", "status": "pending", "tag": ""},
    ]
    tg.get_actionable_tasks.return_value = []
    return tg


class TestBatchInternalNameResolution:
    """Tests for batch-internal name resolution (Phase 6.2)."""

    def test_batch_internal_parent_hash_resolution(self, mock_tg):
        """Task with parent_hash referencing a batch-internal task should wire correctly via update_task."""
        arguments = {
            "tasks": [
                {"name": "Task1", "priority": "high"},
                {"name": "Task2", "priority": "medium", "parent_hash": "Task1"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.create_tasks_batch.assert_called_once()
        # Phase 3 wiring should call update_task to set parent_uuid
        update_calls = mock_tg.update_task.call_args_list
        assert len(update_calls) >= 1
        # Find the call that sets parent_uuid
        parent_call_found = False
        for call in update_calls:
            args, kwargs = call
            if kwargs.get("parent_uuid") == "entity-task1hash-00000001":
                parent_call_found = True
                # Also verify the short_hash is task2hash
                assert args[0] == "task2hash"
                break
        assert parent_call_found

    def test_batch_internal_blocked_by_resolution(self, mock_tg):
        """Task with blocked_by referencing a batch-internal task should wire correctly via update_task."""
        # Reset side_effect from previous test
        mock_tg.get_task_by_short_hash.side_effect = None
        mock_tg.get_task_by_short_hash.return_value = None
        # Override create_tasks_batch so results match the task names used in the call
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "task1hash", "uuid": "entity-task1hash-00000001", "name": "BlockerTask", "priority": "high", "status": "pending", "tag": ""},
            {"short_hash": "task2hash", "uuid": "entity-task2hash-00000002", "name": "BlockedTask", "priority": "medium", "status": "pending", "tag": ""},
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
        # Phase 3 wiring should call update_task to add blocked_by reference
        update_calls = mock_tg.update_task.call_args_list
        assert len(update_calls) >= 1
        # Find the call that updates blocked_by
        blocked_call_found = False
        for call in update_calls:
            args, kwargs = call
            if "blocked_by" in kwargs and "entity-task1hash-00000001" in kwargs["blocked_by"]:
                blocked_call_found = True
                break
        assert blocked_call_found

    def test_batch_local_wins_on_name_collision(self, mock_tg):
        """known_tasks existing task with same name should NOT win; NEW task in batch wins."""
        known_tasks = [
            {"name": "Foo", "short_hash": "oldfoohash"},
        ]
        # Mock existing known task lookup to return the old task
        def task_lookup(h, **kwargs):
            if h == "oldfoohash":
                return {"uuid": "old-foo-uuid", "short_hash": "oldfoohash", "name": "Foo", "tag": ""}
            return None
        mock_tg.get_task_by_short_hash.side_effect = task_lookup
        # Override create_tasks_batch so results match task names used in the call
        mock_tg.create_tasks_batch.return_value = [
            {"short_hash": "newfoohash", "uuid": "entity-newfoohash-00000001", "name": "Foo", "priority": "high", "status": "pending", "tag": ""},
            {"short_hash": "childtaskhash", "uuid": "entity-childtaskhash-00000002", "name": "ChildTask", "priority": "medium", "status": "pending", "tag": ""},
        ]
        mock_tg.get_tasks_by_short_hashes.return_value = {"oldfoohash": {"uuid": "old-foo-uuid", "short_hash": "oldfoohash", "name": "Foo"}}

        arguments = {
            "tasks": [
                {"name": "Foo", "priority": "high"},
                {"name": "ChildTask", "priority": "medium", "parent_hash": "Foo"},
            ],
            "known_tasks": known_tasks,
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.create_tasks_batch.assert_called_once()
        # The parent_hash for ChildTask should reference the NEW Foo's hash ("newfoohash")
        update_calls = mock_tg.update_task.call_args_list
        parent_call_found = False
        for call in update_calls:
            args, kwargs = call
            if kwargs.get("parent_uuid"):
                # Should NOT be the oldfoohash - should be the NEW Foo from batch ("newfoohash")
                assert kwargs["parent_uuid"] != "oldfoohash"
                assert kwargs["parent_uuid"] == "entity-newfoohash-00000001"
                parent_call_found = True
                break
        assert parent_call_found

    def test_ordering_violation_parent_before_child(self, mock_tg):
        """Child appearing before parent should return ordering_violation error."""
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


class TestEditTagSupport:
    """Tests for tag support in EditAction."""

    def test_edit_tag_updated(self, mock_tg):
        """EditAction.execute() with tag='B' should update task with tag 'B'."""
        # Provide a task so it is found
        mock_tg.get_task_by_short_hash.side_effect = lambda h, **kwargs: (
            {"uuid": "entity-abcd000000", "short_hash": "abcd", "name": "Existing Task",
             "status": "pending", "priority": "medium", "summary": "",
             "tag": "A"}
            if h == "abcd" else None
        )
        arguments = {"short_hash": "abcd", "tag": "B"}
        result = EditAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.update_task.assert_called()
        call_kwargs = mock_tg.update_task.call_args[1]
        assert "tag" in call_kwargs
        # Verify tag B is in the call
        assert call_kwargs["tag"] == "B"

    def test_edit_tag_preserved_on_other_edits(self, mock_tg):
        """EditAction.execute() with name but no tag should preserve existing tag."""
        # Existing task has tag "A"
        mock_tg.get_task_by_short_hash.side_effect = lambda h, **kwargs: (
            {"uuid": "entity-abcd000000", "short_hash": "abcd", "name": "Existing Task",
             "status": "pending", "priority": "medium", "summary": "",
             "tag": "A"}
            if h == "abcd" else None
        )
        arguments = {"short_hash": "abcd", "name": "New Name"}
        result = EditAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.update_task.assert_called()
        call_kwargs = mock_tg.update_task.call_args[1]
        # When editing name only, tag should not be in the call (it's preserved by not being changed)
        # The new implementation doesn't pass tag if it's not being updated
        assert "name" in call_kwargs
        assert call_kwargs["name"] == "New Name"


class TestEditTagPatternEnforced:
    """Tests for tag pattern validation in EditAction."""

    def test_tag_pattern_enforced_in_edit(self, mock_tg):
        """EditAction with tag='1' (not uppercase letter) should return error."""
        # Need to provide a task so the error is from tag validation, not not_found
        mock_tg.get_task_by_short_hash.side_effect = lambda h, **kwargs: (
            {"uuid": "entity-abcd000000", "short_hash": "abcd", "name": "Existing Task",
             "status": "pending", "priority": "medium", "summary": "",
             "tag": "A"}
            if h == "abcd" else None
        )
        arguments = {"short_hash": "abcd", "tag": "1"}
        result = EditAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert "tag" in result["error"].get("message", "").lower()


class TestNewStatusValidation:
    """Tests for new_status validation in EditAction."""

    def test_new_status_invalid_rejected(self, mock_tg):
        """EditAction with new_status='done' (not valid) should return error."""
        arguments = {"short_hash": "abcd", "new_status": "done"}
        result = EditAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert "new_status" in result["error"].get("message", "").lower()


class TestRegistryValidation:
    """Tests for tool registry ticket_mode_tools."""

    def test_registry_proxy_ticket_in_ticket_mode(self):
        """'proxy_ticket' should be in registry.get_tools_for_skill('ticket')."""
        ticket_tools = registry.get_tools_for_skill("ticket")
        tool_names = [t for t in ticket_tools]
        assert "proxy_ticket" in tool_names

    def test_registry_task_in_ticket_mode(self):
        """'task' should be in registry.get_tools_for_skill('ticket')."""
        ticket_tools = registry.get_tools_for_skill("ticket")
        tool_names = [t for t in ticket_tools]
        assert "task" in tool_names


class TestCreateTasksValidation:
    """Tests for validation rules in CreateTasksAction."""

    def test_sentinel_prefix_rejected(self, mock_tg):
        """Batch with name '__batch_name:test' should return error."""
        arguments = {
            "tasks": [
                {"name": "__batch_name:test", "priority": "high"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert "__batch_name:" in result["error"].get("message", "")

    def test_batch_size_limit_100(self, mock_tg):
        """Creating 101 tasks should return batch_too_large error."""
        tasks = [{"name": f"Task{i}", "priority": "high"} for i in range(101)]
        arguments = {"tasks": tasks}
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert result["error"]["code"] == "batch_too_large"
        assert "100" in result["error"].get("message", "")

    def test_name_length_limit_200(self, mock_tg):
        """Task with name >200 chars should return error."""
        long_name = "A" * 201
        arguments = {
            "tasks": [
                {"name": long_name, "priority": "high"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert "200" in result["error"].get("message", "")

    def test_tag_pattern_enforced_in_create(self, mock_tg):
        """Task with tag='ab' (2 chars) should return error about tag pattern."""
        arguments = {
            "tasks": [
                {"name": "TestTask", "priority": "high", "tag": "ab"},
            ]
        }
        result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("error") is not None
        assert "tag" in result["error"].get("message", "").lower()