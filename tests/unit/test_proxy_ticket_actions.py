import pytest
from unittest.mock import MagicMock, ANY, patch
import json

from memory.tools import execute_ticket_tool
from memory.task_graph import AmbiguousHashError

pytestmark = pytest.mark.unit

_mock_mode = MagicMock()
_mock_mode.name = "ticket"
_mock_mode.is_expired.return_value = False
_mock_mode.skill_state = {"workspace_name": "default"}

_mock_agent = MagicMock()
_mock_agent.is_ticket_mode = True
_mock_agent.active_mode = _mock_mode
_mock_agent.snapshot_mode.return_value = _mock_mode
_mock_agent.snapshot_mode_and_ticket_ts.return_value = (_mock_mode, None)


@pytest.fixture(autouse=True)
def mock_agent_registry():
    """Patch agent_registry.registry.get_or_load to return a mock agent."""
    with patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=_mock_agent,
    ):
        yield


@pytest.fixture(autouse=True)
def mock_task_graph_singleton(mock_tg_factory=None):
    """Patch _get_task_graph_singleton to return a mock TaskGraph."""
    tg = MagicMock()
    tg.is_available.return_value = True
    tg.__enter__.return_value = tg
    with patch("proxy.context._get_task_graph_singleton", return_value=tg):
        yield tg


@pytest.fixture
def mock_tg():
    """Fully mocked TaskGraph with controllable return values."""
    tg = MagicMock()
    tg.is_available.return_value = True
    tg.__enter__.return_value = tg  # context-manager returns self
    # Default task lookup for any non-empty hash
    def _default_task_lookup(h, **kwargs):
        if not h:
            return None
        return {
            "uuid": f"entity-{h}00000000",
            "short_hash": h,
            "name": "Task " + h,
            "status": "pending",
            "priority": "medium",
            "summary": "",
            "attributes": json.dumps({"tag": "A"}),
        }
    tg.get_task_by_short_hash.side_effect = _default_task_lookup
    # get_task_enriched left as MagicMock default — individual get_task tests
    # set return_value / side_effect to control behavior.
    tg.resolve_blocked_by_hashes.return_value = ["entity-blocker-00000000"]
    tg.resolve_parent_hash.return_value = "entity-parent-00000000"
    tg.update_task.return_value = True
    return tg


class TestEditAction:
    """Edit Action Tests."""

    def test_edit_name_success(self, mock_tg):
        """Test successful name edit."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "name": "New Name"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.update_task.assert_called()
        call_kwargs = mock_tg.update_task.call_args[1]
        assert call_kwargs.get("name") == "New Name"

    def test_edit_summary_success(self, mock_tg):
        """Test successful summary edit."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "summary": "New summary"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.update_task.assert_called()
        call_kwargs = mock_tg.update_task.call_args[1]
        assert call_kwargs.get("summary") == "New summary"

    def test_edit_priority_success(self, mock_tg):
        """Test successful priority edit."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "priority": "high"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.update_task.assert_called()
        call_kwargs = mock_tg.update_task.call_args[1]
        assert call_kwargs.get("priority") == "high"

    def test_edit_empty_fields_error(self, mock_tg):
        """Test that edit with no fields returns error."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2"}, session_id="sid")
        assert result.get("status") == "error"

    def test_edit_name_collision(self, mock_tg):
        """Test that edit fails when name collision occurs."""
        mock_tg.update_task.return_value = False
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "name": "Dup"}, session_id="sid")
        assert result["data"]["updated"] is False

    def test_edit_blocked_by_replaces_list(self, mock_tg):
        """Test that blocked_by update resolves hashes and passes exclude_uuid."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "blocked_by": ["b1b2"]}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.resolve_blocked_by_hashes.assert_called()
        call_args = mock_tg.resolve_blocked_by_hashes.call_args
        assert call_args[0][0] == ["b1b2"]
        assert "exclude_uuid" in call_args[1]
        call_kwargs = mock_tg.update_task.call_args[1]
        assert "blocked_by" in call_kwargs
        assert call_kwargs["blocked_by"] == ["entity-blocker-00000000"]

    def test_edit_parent_hash_sets_parent(self, mock_tg):
        """Test that parent_hash update resolves hash and sets parent_uuid."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "parent_hash": "p1p2"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.resolve_parent_hash.assert_called()
        call_args = mock_tg.resolve_parent_hash.call_args
        assert call_args[0][0] == "p1p2"
        assert "child_uuid" in call_args[1]
        call_kwargs = mock_tg.update_task.call_args[1]
        assert "parent_uuid" in call_kwargs
        assert call_kwargs["parent_uuid"] == "entity-parent-00000000"

    def test_edit_parent_hash_empty_string_clears(self, mock_tg):
        """Test that empty parent_hash clears the parent relationship."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "parent_hash": ""}, session_id="sid")
        assert result.get("status") != "error"
        call_kwargs = mock_tg.update_task.call_args[1]
        assert call_kwargs.get("clear_parent") is True

    def test_edit_ambiguous_hash(self, mock_tg):
        """Test that ambiguous hash returns error."""
        mock_tg.get_task_by_short_hash.side_effect = AmbiguousHashError(2)
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "blocked_by": ["b1b2"]}, session_id="sid")
        assert result.get("status") == "error"
        assert "ambiguous" in result["error"]["code"].lower()

    def test_edit_planning_info_at_limit_accepted(self, mock_tg):
        """planning_info of exactly MAX_PLANNING_INFO_LEN (8000) chars is accepted."""
        from memory.task_graph_common import MAX_PLANNING_INFO_LEN
        plan = "a" * MAX_PLANNING_INFO_LEN
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "planning_info": plan}, session_id="sid")
        assert result.get("status") != "error", result
        call_kwargs = mock_tg.update_task.call_args[1]
        assert call_kwargs.get("planning_info") == plan
        assert len(call_kwargs["planning_info"]) == MAX_PLANNING_INFO_LEN

    def test_edit_planning_info_below_limit_accepted(self, mock_tg):
        """planning_info one char under the limit (7999) is accepted."""
        from memory.task_graph_common import MAX_PLANNING_INFO_LEN
        plan = "a" * (MAX_PLANNING_INFO_LEN - 1)
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "planning_info": plan}, session_id="sid")
        assert result.get("status") != "error", result
        call_kwargs = mock_tg.update_task.call_args[1]
        assert len(call_kwargs["planning_info"]) == MAX_PLANNING_INFO_LEN - 1

    def test_edit_planning_info_above_limit_rejected(self, mock_tg):
        """planning_info one char over the limit (8001) is rejected with invalid_input."""
        from memory.task_graph_common import MAX_PLANNING_INFO_LEN
        plan = "a" * (MAX_PLANNING_INFO_LEN + 1)
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "planning_info": plan}, session_id="sid")
        assert result.get("error", {}).get("code") == "invalid_input"
        assert str(MAX_PLANNING_INFO_LEN) in result["error"]["message"]
        assert str(len(plan)) in result["error"]["message"]
        mock_tg.update_task.assert_not_called()

    def test_edit_planning_info_large_but_within_limit_accepted(self, mock_tg):
        """Regression: 2026-06-28 architectural-review phase tickets need 3462-4247 chars.

        Old limit was 2000 chars (schemas.py + task_graph_common.MAX_PLANNING_INFO_LEN),
        which silently truncated real plans. New limit is 8000 chars.
        """
        from memory.task_graph_common import MAX_PLANNING_INFO_LEN
        plan = "x" * 4500  # comfortably above the old 2000 limit, below new 8000 limit
        assert plan != "x" * 2000  # sanity: not the old boundary
        assert len(plan) < MAX_PLANNING_INFO_LEN
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("edit", {"action": "edit", "short_hash": "a1b2", "planning_info": plan}, session_id="sid")
        assert result.get("status") != "error", result
        call_kwargs = mock_tg.update_task.call_args[1]
        assert call_kwargs.get("planning_info") == plan


class TestSetParentAction:
    """Set Parent Action Tests."""

    def test_set_parent_success(self, mock_tg):
        """Test successful parent assignment."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_parent", {"action": "set_parent", "short_hash": "a1b2", "parent_hash": "p1p2"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.resolve_parent_hash.assert_called()
        mock_tg.update_task.assert_called()

    def test_set_parent_missing_short_hash(self, mock_tg):
        """Test that missing short_hash returns error."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_parent", {"action": "set_parent", "parent_hash": "p1p2"}, session_id="sid")
        assert result.get("status") == "error"

    def test_set_parent_not_found(self, mock_tg):
        """Test that parent not found returns error."""
        # Make get_task_by_short_hash return None for parent hash "p1p2"
        def _lookup(h, **kwargs):
            if h == "p1p2":
                return None
            return {
                "uuid": f"entity-{h}00000000",
                "short_hash": h,
                "name": "Task " + h,
                "status": "pending",
                "priority": "medium",
                "summary": "",
                "attributes": json.dumps({"tag": "A"}),
            }
        mock_tg.get_task_by_short_hash.side_effect = _lookup
        # resolve_parent_hash needs to raise ValueError when parent not found
        mock_tg.resolve_parent_hash.side_effect = ValueError("Parent not found: p1p2")
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_parent", {"action": "set_parent", "short_hash": "a1b2", "parent_hash": "p1p2"}, session_id="sid")
        assert result.get("status") == "error"

    def test_set_parent_self_parenting(self, mock_tg):
        """Test that self-parenting returns error."""
        mock_tg.resolve_parent_hash.side_effect = ValueError(
            "Task cannot be its own parent"
        )
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_parent", {"action": "set_parent", "short_hash": "a1b2", "parent_hash": "a1b2"}, session_id="sid")
        assert result.get("status") == "error"


class TestClearParentAction:
    """Clear Parent Action Tests."""

    def test_clear_parent_success(self, mock_tg):
        """Test successful parent clearing."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("clear_parent", {"action": "clear_parent", "short_hash": "a1b2"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.get_task_by_short_hash.assert_called()
        mock_tg.update_task.assert_called()

    def test_clear_parent_not_found(self, mock_tg):
        """Test that task not found returns error."""
        mock_tg.get_task_by_short_hash.return_value = None
        # update_task returns False when task not found
        mock_tg.update_task.return_value = False
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("clear_parent", {"action": "clear_parent", "short_hash": "a1b2"}, session_id="sid")
        # When task not found, update_task is called but returns False
        assert result["data"]["updated"] is False


class TestClearBlockedByAction:
    """Clear Blocked By Action Tests."""

    def test_clear_blocked_by_success(self, mock_tg):
        """Test successful blocked_by clearing."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("clear_blocked_by", {"action": "clear_blocked_by", "short_hash": "a1b2"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.get_task_by_short_hash.assert_called()
        mock_tg.update_task.assert_called()

    def test_clear_blocked_by_not_found(self, mock_tg):
        """Test that task not found returns error."""
        mock_tg.get_task_by_short_hash.return_value = None
        # update_task returns False when task not found
        mock_tg.update_task.return_value = False
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("clear_blocked_by", {"action": "clear_blocked_by", "short_hash": "a1b2"}, session_id="sid")
        # When task not found, update_task is called but returns False
        assert result["data"]["updated"] is False


class TestSetPriorityAction:
    """Set Priority Action Tests."""

    def test_set_priority_success(self, mock_tg):
        """Test successful priority change."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_priority", {"action": "set_priority", "short_hash": "a1b2", "priority": "high"}, session_id="sid")
        assert result.get("status") != "error"
        mock_tg.update_task.assert_called_with("a1b2", priority="high", workspace_id="default", caller_agent_id=ANY)

    def test_set_priority_invalid_priority(self, mock_tg):
        """Test that invalid priority returns error."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_priority", {"action": "set_priority", "short_hash": "a1b2", "priority": "urgent"}, session_id="sid")
        assert result.get("status") == "error"

    def test_set_priority_missing(self, mock_tg):
        """Test that missing priority returns error."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("set_priority", {"action": "set_priority", "short_hash": "a1b2"}, session_id="sid")
        assert result.get("status") == "error"


# =============================================================================
# Regression test for ECHO-H3/INDIA H-8 overcorrection in execute_proxy_ticket
# =============================================================================


class TestProxyTicketWorkspaceOverride:
    """Regression tests locking in that execute_proxy_ticket derives workspace
    from workspace_store.get_active(session_id) even when skill_state is empty.

    Pre-fix: when skill_state had no workspace_name AND mode_name was 'ticket',
    the code set workspace_name = DEFAULT_WORKSPACE, ignoring the session's
    actual active workspace.

    Post-fix: workspace_name is always populated from workspace_store.get_active,
    so a session in workspace 'luna-ui' gets luna-ui, not 'default'.
    """

    def test_proxy_ticket_uses_active_workspace(self, mock_tg):
        """When skill_state has no workspace_name, fall back to workspace_store.get_active()."""
        from proxy import state
        session_id = "test-sess-wsfix-pt"
        try:
            # Seed the workspace store: this session is in luna-ui
            state.active_workspaces[session_id] = "luna-ui"

            # Build a mock agent with EMPTY skill_state so the
            # `if not workspace_name:` branch is taken.
            mock_mode = MagicMock()
            mock_mode.name = "ticket"
            mock_mode.is_expired.return_value = False
            mock_mode.skill_state = {}  # no workspace_name → triggers fallback

            mock_agent = MagicMock()
            mock_agent.is_ticket_mode = True
            mock_agent.active_mode = mock_mode
            mock_agent.snapshot_mode.return_value = mock_mode
            mock_agent.snapshot_mode_and_ticket_ts.return_value = (mock_mode, None)

            with patch(
                "proxy.agent.agent_registry.registry.get_or_load",
                return_value=mock_agent,
            ):
                with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
                    result = execute_ticket_tool("set_priority", {"action": "set_priority", "short_hash": "a1b2", "priority": "high"}, session_id=session_id)

            # Result should NOT be an access-denied error (we're in ticket mode)
            assert result.get("status") != "error", result

            # The key assertion: workspace_id uses luna-ui, NOT default.
            # The canonical envelope does not surface workspace_id at the top
            # level (it is only passed as a kwarg to TaskGraph methods), so we
            # verify it via the captured update_task call below.
            call_kwargs = mock_tg.update_task.call_args[1]
            assert call_kwargs["workspace_id"] == "luna-ui", (
                f"expected luna-ui, got {call_kwargs['workspace_id']!r}"
            )
            assert "default" not in call_kwargs["workspace_id"], (
                f"workspace_id should not contain 'default' — got {call_kwargs['workspace_id']!r}"
            )
        finally:
            state.active_workspaces.pop(session_id, None)


# =============================================================================
# Ticket 9c151e0f: read-only get_task action
# =============================================================================


class TestGetTaskAction:
    """Read-only get_task returns the full task record (incl. planning_info in
    full, parent_short_hash, blockers, review_notes) by short_hash.

    Sub-agents currently cannot verify a specific ticket's state without
    mutating it (list_tasks truncates planning_info at 16 KB). The Phase 3
    cleanup sub-agent mutated name fields as a side-channel lookup and damaged
    two legit tickets — get_task fixes that without requiring mutations.
    """

    def test_get_task_returns_full_record(self, mock_tg):
        """get_task returns every field of get_task_enriched() in the response."""
        mock_tg.get_task_enriched.side_effect = None
        mock_tg.get_task_enriched.return_value = {
            "uuid": "entity-a1b20000",
            "short_hash": "a1b2",
            "name": "Investigate auth bug",
            "status": "in_progress",
            "priority": "high",
            "summary": "Auth token rejected for valid users.",
            "tag": "A",
            "planning_info": "Step 1: reproduce.\nStep 2: trace.\nStep 3: fix.",
            "created_at": "2026-06-30T12:00:00Z",
            "assigned_agent_id": "agent-xyz",
            "blockers": [
                {"uuid": "entity-blocker-1", "short_hash": "deadbeef", "name": "Dep", "status": "pending"},
            ],
            "is_blocked": True,
            "blocked_count": 1,
            "parent_short_hash": "parent00",
            "parent_name": "Auth epic",
            "review_notes": [
                {"ts": "2026-06-30T11:00:00Z", "note": "started", "author": "agent-1"},
            ],
        }
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "a1b2"}, session_id="sid")

        assert result.get("status") != "error", result
        assert result["operation"] == "get_task"
        data = result["data"]
        assert data["short_hash"] == "a1b2"
        assert data["uuid"] == "entity-a1b20000"
        assert data["name"] == "Investigate auth bug"
        assert data["status"] == "in_progress"
        assert data["priority"] == "high"
        assert data["summary"] == "Auth token rejected for valid users."
        assert data["tag"] == "A"
        assert data["planning_info"] == "Step 1: reproduce.\nStep 2: trace.\nStep 3: fix."
        assert data["planning_info_truncated"] is False
        assert data["created_at"] == "2026-06-30T12:00:00Z"
        assert data["assigned_agent_id"] == "agent-xyz"
        assert data["is_blocked"] is True
        assert data["blocked_count"] == 1
        assert data["parent_short_hash"] == "parent00"
        assert data["parent_name"] == "Auth epic"
        assert len(data["blocked_by"]) == 1
        assert data["blocked_by"][0]["short_hash"] == "deadbeef"
        assert len(data["review_notes"]) == 1
        assert data["review_notes"][0]["note"] == "started"

    def test_get_task_not_found(self, mock_tg):
        """Missing task → error.code=not_found."""
        mock_tg.get_task_enriched.side_effect = None
        mock_tg.get_task_enriched.return_value = None
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "deadbee"}, session_id="sid")
        assert result["error"]["code"] == "not_found"
        assert "Task not found" in result["error"]["message"]
        assert result["session_id"] == "sid"

    def test_get_task_missing_short_hash(self, mock_tg):
        """Missing/empty short_hash → invalid_input (loud failure, consistent)."""
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task"}, session_id="sid")
        assert result["error"]["code"] == "invalid_input"
        assert "short_hash is required" in result["error"]["message"]

    def test_get_task_ambiguous_hash(self, mock_tg):
        """AmbiguousHashError → error.code=ambiguous_hash with hint to refine."""
        mock_tg.get_task_enriched.side_effect = AmbiguousHashError(2)
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "ab"}, session_id="sid")
        assert result["error"]["code"] == "ambiguous_hash"
        assert "longer hash" in result["error"]["message"]

    def test_get_task_includes_planning_info_in_full(self, mock_tg):
        """planning_info is returned in FULL (no truncation) up to the 50 KB cap.

        Regression for the Phase 3 damage: list_tasks truncates at 16 KB, so
        sub-agents had to guess or mutate. get_task must return planning_info
        in full so sub-agents never need to edit name/summary fields to peek
        inside a long ticket.
        """
        from proxy.tool_pipeline.actions.get_task import MAX_GET_TASK_PLANNING_INFO_LEN

        plan = "x" * (MAX_GET_TASK_PLANNING_INFO_LEN - 100)  # just under the cap
        mock_tg.get_task_enriched.side_effect = None
        mock_tg.get_task_enriched.return_value = {
            "short_hash": "a1b2",
            "name": "Long ticket",
            "status": "pending",
            "planning_info": plan,
        }
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "a1b2"}, session_id="sid")
        assert result.get("status") != "error", result
        assert len(result["data"]["planning_info"]) == len(plan)
        assert result["data"]["planning_info"] == plan
        assert result["data"]["planning_info_truncated"] is False

    def test_get_task_caps_planning_info_above_limit(self, mock_tg):
        """planning_info above the 50 KB cap is truncated and flagged truncated=True.

        Cap prevents a malicious or runaway task record from blowing up the
        LLM context window. 50 KB is well above the 8 KB write cap (task_graph_common.MAX_PLANNING_INFO_LEN).
        """
        from proxy.tool_pipeline.actions.get_task import MAX_GET_TASK_PLANNING_INFO_LEN

        plan = "x" * (MAX_GET_TASK_PLANNING_INFO_LEN + 10_000)
        mock_tg.get_task_enriched.side_effect = None
        mock_tg.get_task_enriched.return_value = {
            "short_hash": "a1b2",
            "name": "Runaway ticket",
            "status": "pending",
            "planning_info": plan,
        }
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "a1b2"}, session_id="sid")
        assert result.get("status") != "error", result
        assert len(result["data"]["planning_info"]) == MAX_GET_TASK_PLANNING_INFO_LEN
        assert result["data"]["planning_info_truncated"] is True

    def test_get_task_is_read_only_no_db_writes(self, mock_tg):
        """get_task must NOT trigger any TaskGraph write method.

        Counts write methods before and after a successful get_task call —
        the diff must be zero. Read-only is the whole point (ticket 9c151e0f).
        """
        mock_tg.get_task_enriched.side_effect = None
        mock_tg.get_task_enriched.return_value = {
            "short_hash": "a1b2",
            "name": "Read-only probe",
            "status": "pending",
        }
        write_methods = [
            "update_task", "create_task", "create_tasks_batch",
            "update_tasks_batch_status", "update_task_status", "delete_task",
            "set_task_blocked_by", "append_review_note", "add_note",
        ]
        for m in write_methods:
            getattr(mock_tg, m).reset_mock()

        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "a1b2"}, session_id="sid")

        for m in write_methods:
            assert getattr(mock_tg, m).call_count == 0, (
                f"get_task must NOT call write method {m!r} (read-only); "
                f"got {getattr(mock_tg, m).call_count} calls"
            )

    def test_get_task_response_shape_compatible_with_list_tasks_items(self, mock_tg):
        """get_task's per-task fields overlap with list_tasks item fields.

        The Phase 3 cleanup sub-agent had to switch between list_tasks and
        get_task — both must produce comparable records so callers can treat
        them uniformly. list_tasks exposes: short_hash, name, priority,
        status, summary, tag, blocked_by, is_blocked, blocked_count,
        parent_short_hash, parent_name. get_task must surface every one of
        these fields under the same names.
        """
        mock_tg.get_task_enriched.side_effect = None
        mock_tg.get_task_enriched.return_value = {
            "short_hash": "a1b2",
            "name": "Cross-shape check",
            "priority": "medium",
            "status": "pending",
            "summary": "x",
            "tag": "B",
            "blockers": [],
            "is_blocked": False,
            "blocked_count": 0,
            "parent_short_hash": None,
            "parent_name": None,
            "planning_info": "...",
        }
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = execute_ticket_tool("get_task", {"action": "get_task", "short_hash": "a1b2"}, session_id="sid")
        data = result["data"]
        for key in (
            "short_hash", "name", "priority", "status", "summary", "tag",
            "is_blocked", "blocked_count",
            "parent_short_hash", "parent_name",
        ):
            assert key in data, f"missing key shared with list_tasks: {key}"
        assert "blocked_by" in data, "get_task must expose blockers as blocked_by to match list_tasks item shape"


# =============================================================================
# Slice 1 acceptance criteria #2, #3, #5 — canonical envelope shape tests.
# Spec §11 (canonical envelope), §17.4 (create_tasks partial status).
#
# These tests pin the post-cutover behavior. Each test corresponds to one of
# the three manual revert-tests called out in the Slice 1 plan:
#   - TestCanonicalEnvelopeMissingAction / TestCanonicalEnvelopeUnknownAction
#     → revert-test #1 (operation propagation; missing/unknown → operation=None
#       or operation=<attempted>, never an empty string).
#   - TestCreateTasksPartialStatus → revert-test #2 (partial-status).
#   - TestUpdateTaskCanonicalEnvelope → revert-test #3 (envelope-on-one-action).
# =============================================================================


def _make_work_mode_agent(agent_id: str = "agent-work"):
    """Build a mock agent whose active mode is 'work' (not expired).

    The autouse `mock_agent_registry` fixture in this file returns a
    TICKET-mode agent. execute_proxy_work requires WORK mode to reach the
    missing/unknown-action branches, so the proxy_work tests below patch
    the registry locally with this builder.
    """
    work_mode = MagicMock()
    work_mode.name = "work"
    work_mode.is_expired.return_value = False
    work_mode.skill_state = {"workspace_name": "default"}

    agent = MagicMock()
    agent.agent_id = agent_id
    agent.snapshot_mode_and_ticket_ts.return_value = (work_mode, None)
    agent.snapshot_mode.return_value = work_mode
    return agent


class TestCanonicalEnvelopeMissingAction:
    """Spec §11: missing action produces a canonical error envelope."""

    def test_proxy_ticket_missing_action_returns_canonical_error(self):
        # Autouse fixtures put _mock_agent in ticket mode → mode gate passes,
        # the missing-action branch in dispatch_action fires.
        result = execute_ticket_tool("", {}, session_id="sid-test")

        assert result["ok"] is False
        assert result["tool"] == "proxy_ticket"
        assert result["operation"] is None  # empty action → operation is None at dispatch
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["error"]["code"] == "invalid_action"
        assert result["error"]["retryable"] is True
        assert result["error"]["category"] == "validation"
        assert "action" in result["error"]["message"].lower()
        assert isinstance(result["valid_actions"], list)
        # ACTION_REGISTRY has 10 ticket + 7 work actions registered at import.
        assert len(result["valid_actions"]) >= 10
        assert result["session_id"] == "sid-test"

    def test_proxy_work_missing_action_returns_canonical_error(self):
        from memory.tools.proxy_work import execute_proxy_work

        work_agent = _make_work_mode_agent()
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=work_agent,
        ):
            result = execute_proxy_work({}, session_id="sid-test")

        assert result["ok"] is False
        assert result["tool"] == "proxy_work"
        assert result["operation"] is None
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["retryable"] is True
        assert result["error"]["category"] == "validation"
        assert "action" in result["error"]["message"].lower()
        assert isinstance(result["valid_actions"], list)
        assert set(result["valid_actions"]) == {
            "start_work", "stop_work", "report_bug",
            "review_bug", "validate_bug", "resolve_bug",
            "handoff_task",
        }
        assert result["session_id"] == "sid-test"


class TestCanonicalEnvelopeUnknownAction:
    """Spec §11: unknown action produces a canonical error envelope."""

    def test_proxy_work_unknown_action_does_not_preserve_bogus_name(self):
        from memory.tools.proxy_work import execute_proxy_work

        work_agent = _make_work_mode_agent()
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=work_agent,
        ):
            result = execute_proxy_work({"action": "bogus"}, session_id="sid-test")

        assert result["ok"] is False
        assert result["tool"] == "proxy_work"
        # Unknown action at the boundary → operation is None. The bogus name
        # is NOT echoed into the envelope (which would be a value-leak).
        assert result["operation"] is None
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["error"]["code"] == "unknown_action"
        assert result["error"]["retryable"] is True
        assert result["error"]["category"] == "validation"
        assert isinstance(result["valid_actions"], list)
        assert set(result["valid_actions"]) == {
            "start_work", "stop_work", "report_bug",
            "review_bug", "validate_bug", "resolve_bug",
            "handoff_task",
        }
        assert result["session_id"] == "sid-test"

    def test_dispatch_action_unknown_returns_canonical_error(self):
        from proxy.tool_pipeline.action_registry import dispatch_action

        mock_tg = MagicMock()
        result = dispatch_action(
            "bogus_action", {}, "sid-dispatch", mock_tg,
            tool="proxy_ticket",
        )

        assert result["ok"] is False
        assert result["tool"] == "proxy_ticket"
        # Unknown action at dispatch layer → operation is None (no attempted
        # action is "claimed" by an action handler).
        assert result["operation"] is None
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["error"]["code"] == "invalid_action"
        assert result["error"]["retryable"] is True
        assert result["error"]["category"] == "validation"
        assert isinstance(result["valid_actions"], list)
        # ACTION_REGISTRY has 10 ticket + 7 work actions.
        assert len(result["valid_actions"]) >= 10
        assert result["session_id"] == "sid-dispatch"


class TestCanonicalEnvelopeProxyTicketRejectsWorkActions:
    """BLOCKER 1: proxy_ticket must NOT dispatch proxy_work actions.

    Both proxy_ticket and proxy_work register into the same ACTION_REGISTRY
    and the ACTION_REGISTRY lookup at the dispatch layer has no tool-name
    domain knowledge. Without an explicit allowlist in execute_proxy_ticket,
    `proxy_ticket(action="report_bug")` reaches ReportBugAction and the
    returned envelope has tool="proxy_work" (because the action knows its
    own tool) — an action reachable through two tools is a security
    boundary violation.
    """

    def test_proxy_ticket_rejects_work_action(self):
        result = execute_ticket_tool("report_bug", {"action": "report_bug"}, session_id="sid")

        assert result["ok"] is False
        # Action reaches dispatch layer → operation is the attempted action (report_bug).
        assert result["operation"] == "report_bug"
        assert result["tool"] == "proxy_work"
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["retryable"] is True
        assert result["error"]["category"] == "validation"
        # valid_actions surfaces the full registry, not just ticket actions
        assert isinstance(result.get("valid_actions"), list) or "valid_actions" not in result
        assert result["session_id"] == "sid"


class TestCreateTasksPartialStatus:
    """Spec §17.4: create_tasks with per-task failures returns status='partial', not 'success'.

    Revert-test #2: if create_tasks.py is reverted to use success_envelope
    when errors/wiring_errors are non-empty, `status == "partial"` fails.
    """

    def test_partial_create_tasks_returns_partial_status(self):
        from proxy.tool_pipeline.actions import CreateTasksAction

        mock_tg = MagicMock()
        # Two tasks submitted; create_tasks_batch returns one success and
        # one per-task error entry (the shape TaskGraph produces when an
        # individual MERGE inside the UNWIND batch fails).
        mock_tg.create_tasks_batch.return_value = [
            {
                "short_hash": "abc12345",
                "name": "Working task",
                "priority": "high",
                "status": "pending",
            },
            {
                "error": "constraint_violation",
                "message": "duplicate name",
            },
        ]

        result = CreateTasksAction.execute(
            arguments={
                "tasks": [
                    {"name": "Working task", "priority": "high"},
                    {"name": "Failing task", "priority": "low"},
                ],
            },
            session_id="sid-partial",
            tg=mock_tg,
            workspace_id="default",
        )

        assert result["ok"] is True
        assert result["status"] == "partial"  # NOT "success"
        assert result["operation"] == "create_tasks"
        assert result["tool"] == "proxy_ticket"
        assert result["error"] is None
        assert isinstance(result["data"], dict)
        assert len(result["data"]["errors"]) > 0
        assert result["data"]["total_errors"] >= 1
        # Sanity: the successful task still appears in `created`.
        assert len(result["data"]["created"]) >= 1
        assert result["data"]["total_created"] >= 1
        assert result["session_id"] == "sid-partial"

    def test_full_success_create_tasks_returns_success_status(self):
        from proxy.tool_pipeline.actions import CreateTasksAction

        mock_tg = MagicMock()
        mock_tg.create_tasks_batch.return_value = [
            {
                "short_hash": "aaa11111",
                "name": "Task A",
                "priority": "high",
                "status": "pending",
            },
            {
                "short_hash": "bbb22222",
                "name": "Task B",
                "priority": "low",
                "status": "pending",
            },
        ]

        result = CreateTasksAction.execute(
            arguments={
                "tasks": [
                    {"name": "Task A", "priority": "high"},
                    {"name": "Task B", "priority": "low"},
                ],
            },
            session_id="sid-success",
            tg=mock_tg,
            workspace_id="default",
        )

        assert result["ok"] is True
        assert result["status"] == "success"
        assert result["operation"] == "create_tasks"
        assert result["tool"] == "proxy_ticket"
        assert result["error"] is None
        assert isinstance(result["data"], dict)
        assert result["data"]["errors"] == []
        assert result["data"]["wiring_errors"] == []
        assert result["data"]["total_created"] == 2
        assert result["data"]["total_errors"] == 0
        assert result["session_id"] == "sid-success"

    def test_wiring_only_partial_returns_partial_status(self):
        """MAJOR 4: wiring-only failures must produce status='partial'.

        The data dict's `total_errors` is `len(errors) + len(wiring_errors)`,
        so a result with empty `errors` but non-empty `wiring_errors` still
        counts as an error budget for the partial envelope. This test pins
        the contract: if the per-task create succeeded for every task, but
        the post-batch wiring (e.g. HAS_PARENT relation) failed, the envelope
        status is 'partial', not 'success'.
        """
        from proxy.tool_pipeline.actions import CreateTasksAction

        mock_tg = MagicMock()
        # All three tasks land clean in `created` (no per-task errors).
        mock_tg.create_tasks_batch.return_value = [
            {
                "short_hash": "aaa11111",
                "uuid": "uuid-aaa11111",
                "name": "Task A",
                "priority": "high",
                "status": "pending",
            },
            {
                "short_hash": "bbb22222",
                "uuid": "uuid-bbb22222",
                "name": "Task B",
                "priority": "low",
                "status": "pending",
            },
            {
                "short_hash": "ccc33333",
                "uuid": "uuid-ccc33333",
                "name": "Task C",
                "priority": "medium",
                "status": "pending",
            },
        ]
        # Force the post-batch deferred-wiring step to fail. Task C
        # references Task A via parent_hash → deferred_wiring gets one
        # entry → update_task(parent_uuid=...) is invoked → returning
        # False appends a wiring_errors entry with
        # error="parent wiring update failed".
        mock_tg.update_task.return_value = False

        result = CreateTasksAction.execute(
            arguments={
                "tasks": [
                    {"name": "Task A", "priority": "high"},
                    {"name": "Task B", "priority": "low"},
                    {"name": "Task C", "priority": "medium", "parent_hash": "Task A"},
                ],
            },
            session_id="sid-wiring-only",
            tg=mock_tg,
            workspace_id="default",
        )

        # The wiring_only invariant: status='partial', not 'success'.
        assert result["ok"] is True
        assert result["status"] == "partial"
        assert result["operation"] == "create_tasks"
        assert result["tool"] == "proxy_ticket"
        assert result["error"] is None
        assert isinstance(result["data"], dict)
        # No per-task create errors; the failure is wiring-only.
        assert result["data"]["errors"] == []
        assert len(result["data"]["wiring_errors"]) >= 1
        # total_errors counts wiring_errors (the fix). Without the fix this
        # would be 0 and the envelope would have been status='success'.
        assert result["data"]["total_errors"] >= 1
        # The successful creates still surface.
        assert result["data"]["total_created"] >= 3
        assert result["session_id"] == "sid-wiring-only"


class TestUpdateTaskCanonicalEnvelope:
    """Spec §11: update_task uses the canonical envelope; the old
    {completed, status, ...} shape is gone.

    Revert-test #3: if update_task.py is reverted to the old shape, the
    `result["ok"]` / `result["tool"]` / `result["operation"]` reads fail
    with KeyError, and the `"completed" not in result` assertion fails.
    """

    def test_update_task_missing_short_hash_returns_canonical_error(self):
        from proxy.tool_pipeline.actions import UpdateTaskAction

        mock_tg = MagicMock()
        result = UpdateTaskAction.execute(
            arguments={},
            session_id="sid-update-err",
            tg=mock_tg,
            workspace_id="default",
        )

        assert result["ok"] is False
        assert result["status"] == "error"
        assert result["operation"] == "update_task"
        assert result["tool"] == "proxy_ticket"
        assert result["data"] is None
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["retryable"] is True
        assert result["error"]["category"] == "validation"
        assert result["error"]["field"] == "short_hash"
        # The old update_task shape had a top-level "completed" boolean —
        # the canonical envelope must NOT carry it.
        assert "completed" not in result
        assert result["session_id"] == "sid-update-err"

    def test_update_task_success_returns_canonical_success(self):
        from proxy.tool_pipeline.actions import UpdateTaskAction

        mock_tg = MagicMock()
        mock_tg.update_task_status.return_value = {
            "ok": True,
            "status": "complete",
            "changed": True,
        }

        result = UpdateTaskAction.execute(
            arguments={"short_hash": "abcd1234", "new_status": "complete"},
            session_id="sid-update-ok",
            tg=mock_tg,
            workspace_id="default",
        )

        assert result["ok"] is True
        # Envelope status is "success" — NOT "complete" (that was the old shape).
        assert result["status"] == "success"
        assert result["operation"] == "update_task"
        assert result["tool"] == "proxy_ticket"
        assert result["error"] is None
        assert isinstance(result["data"], dict)
        # The task's actual status lives HERE, not at the top level.
        assert result["data"]["task_status"] == "complete"
        assert result["data"]["changed"] is True
        assert result["data"]["short_hash"] == "abcd1234"
        # Old shape gone.
        assert "completed" not in result
        assert result["session_id"] == "sid-update-ok"


class TestCreateTasksNonStringName:
    """TICKET-001: non-string name must return invalid_input, not AttributeError.

    Revert-test: if the isinstance(name, str) check is moved back AFTER
    name.strip(), these tests raise AttributeError instead of returning
    an error envelope.
    """

    @pytest.mark.parametrize("bad_name", [123, None, ["list"], {"key": "val"}])
    def test_non_string_name_returns_invalid_input(self, bad_name):
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction

        tg = MagicMock()
        arguments = {"tasks": [{"name": bad_name, "priority": "high"}]}
        result = CreateTasksAction.execute(arguments, "sid", tg)

        assert result["status"] == "error"
        assert result["error"]["code"] == "invalid_input"
        assert result["error"]["field"] == "name"
        assert "non-empty string" in result["error"]["message"]
        tg.create_tasks_batch.assert_not_called()


class TestCreateTasksNonStringNameRejected:
    """TICKET-2026-07-19-001: non-string names return invalid_input, no AttributeError."""

    def test_int_name_returns_invalid_input(self):
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock
        tg = MagicMock()
        result = CreateTasksAction.execute(
            arguments={"tasks": [{"name": 42, "priority": "high"}]},
            session_id="sid-test",
            tg=tg,
            workspace_id="default",
        )
        assert result["status"] == "error"
        assert result["error"]["code"] == "invalid_input"
        assert result["error"].get("field") == "name"
        assert result["error"]["retryable"] is True

    def test_none_name_returns_invalid_input(self):
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock
        tg = MagicMock()
        result = CreateTasksAction.execute(
            arguments={"tasks": [{"name": None, "priority": "high"}]},
            session_id="sid-test",
            tg=tg,
            workspace_id="default",
        )
        assert result["error"]["code"] == "invalid_input"
        assert result["error"].get("field") == "name"

    def test_list_name_returns_invalid_input(self):
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock
        tg = MagicMock()
        result = CreateTasksAction.execute(
            arguments={"tasks": [{"name": ["nested"], "priority": "high"}]},
            session_id="sid-test",
            tg=tg,
            workspace_id="default",
        )
        assert result["error"]["code"] == "invalid_input"
        assert result["error"].get("field") == "name"

    def test_dict_name_returns_invalid_input(self):
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock
        tg = MagicMock()
        result = CreateTasksAction.execute(
            arguments={"tasks": [{"name": {"x": 1}, "priority": "high"}]},
            session_id="sid-test",
            tg=tg,
            workspace_id="default",
        )
        assert result["error"]["code"] == "invalid_input"
        assert result["error"].get("field") == "name"

    def test_valid_string_name_still_accepted(self):
        """Regression check: valid string names are not rejected by the type check."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock
        tg = MagicMock()
        tg.create_tasks_batch.return_value = []
        result = CreateTasksAction.execute(
            arguments={"tasks": [{"name": "Valid task name", "priority": "high"}]},
            session_id="sid-test",
            tg=tg,
            workspace_id="default",
        )
        assert result["status"] == "success"


class TestCreateTasksAssignToFailClosed:
    """TICKET-2026-07-19-002: assign_to registry lookup propagates errors
    (no silent fail-open). The task must NOT be created when registry
    validation fails."""

    def test_infrastructure_error_propagates(self):
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from memory.query_executor import InfrastructureError
        from unittest.mock import MagicMock, patch
        tg = MagicMock()
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=InfrastructureError("neo4j down"),
        ):
            try:
                result = CreateTasksAction.execute(
                    arguments={"tasks": [{"name": "x", "priority": "high", "assign_to": "agent-y"}]},
                    session_id="sid-test",
                    tg=tg,
                    workspace_id="default",
                )
                # If it didn't propagate, it should be a canonical envelope with
                # infrastructure_error code.
                assert result["status"] == "error"
                assert result["error"]["code"] == "infrastructure_error"
                assert result["error"]["category"] == "infrastructure"
            except InfrastructureError:
                pass  # propagating is also acceptable
        # Task must NOT be created
        assert not tg.create_tasks_batch.called, (
            "task was created despite assign_to registry InfrastructureError"
        )

    def test_keyerror_returns_assign_to_lookup_failed(self):
        """KeyError from registry becomes a canonical assign_to_lookup_failed
        envelope (current implementation's choice). The task must NOT be
        created."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock, patch
        tg = MagicMock()
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=KeyError("malformed session_id"),
        ):
            result = CreateTasksAction.execute(
                arguments={"tasks": [{"name": "x", "priority": "high", "assign_to": "agent-y"}]},
                session_id="sid-test",
                tg=tg,
                workspace_id="default",
            )
        assert result["status"] == "error"
        assert result["error"]["code"] == "assign_to_lookup_failed"
        assert result["error"].get("field") == "assign_to"
        assert not tg.create_tasks_batch.called, (
            "task was created despite assign_to registry KeyError"
        )

    def test_valid_assign_to_still_accepted(self):
        """Regression check: when registry returns the agent, task is created."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock, patch
        tg = MagicMock()
        tg.create_tasks_batch.return_value = [
            {"short_hash": "abcd1234", "name": "x", "priority": "high", "status": "pending"}
        ]
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent-y"
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent,
        ):
            result = CreateTasksAction.execute(
                arguments={"tasks": [{"name": "x", "priority": "high", "assign_to": "agent-y"}]},
                session_id="sid-test",
                tg=tg,
                workspace_id="default",
            )
        assert result["status"] == "success"
        assert tg.create_tasks_batch.called

    def test_none_assign_to_skips_validation(self):
        """Regression check: assign_to=None skips registry lookup entirely."""
        from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
        from unittest.mock import MagicMock, patch
        tg = MagicMock()
        tg.create_tasks_batch.return_value = [
            {"short_hash": "abcd1234", "name": "x", "priority": "high", "status": "pending"}
        ]
        # No patch on registry; if it's called, the test fails.
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=AssertionError("registry.get_or_load must NOT be called when assign_to is None"),
        ):
            result = CreateTasksAction.execute(
                arguments={"tasks": [{"name": "x", "priority": "high"}]},
                session_id="sid-test",
                tg=tg,
                workspace_id="default",
            )
        assert result["status"] == "success"
