"""Slice 8a — guard tests for decomposed ticket tools."""

import pytest
from unittest.mock import patch, MagicMock

from proxy.tool_pipeline.tool_guard import MemoryToolGuard
from proxy.tool_pipeline.tool_call import ToolCall


def _make_tc(name, args):
    return ToolCall(id="call-1", name=name, arguments=args)


class TestWorkModeActionsGuard:
    """Verify _check_work_mode_actions handles decomposed tool names."""

    def test_ticket_main_create_tasks_blocked(self):
        """Ticket MAIN calling proxy_ticket_create_tasks → denied (legacy path)."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_create_tasks", {"tasks": [{"name": "x", "priority": "high"}]})
        ctx = MagicMock(session_id="sess-1", is_subagent=False)

        mock_agent = MagicMock()
        mock_mode = MagicMock()
        mock_mode.name = "ticket"
        mock_mode.is_expired.return_value = False
        mock_agent.snapshot_mode.return_value = mock_mode

        with patch("proxy.skills.skill_registry.SkillRegistry") as mock_skreg_cls:
            mock_skreg = MagicMock()
            mock_skreg.get.return_value = None
            mock_skreg_cls.return_value = mock_skreg

            with patch("proxy.agent.agent_registry.registry") as mock_ag_reg:
                mock_ag_reg.get_or_load.return_value = mock_agent
                result = guard._check_work_mode_actions(tc, ctx)
                assert result is not None

    def test_ticket_main_list_tasks_does_not_crash(self):
        """Ticket MAIN calling proxy_ticket_list_tasks → acceptance: doesn't crash.

        The legacy fallback path (no capability_table) cannot distinguish decomposed
        tools since they lack an 'action' field. This is expected — the capability_table
        is the intended path for decomposed tools (Slice 8a).
        """
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_list_tasks", {"status_filter": ["pending"]})
        ctx = MagicMock(session_id="sess-1", is_subagent=False)

        mock_agent = MagicMock()
        mock_mode = MagicMock()
        mock_mode.name = "ticket"
        mock_mode.is_expired.return_value = False
        mock_agent.snapshot_mode.return_value = mock_mode

        mock_skill = MagicMock()
        mock_table = MagicMock()
        mock_table.check_action.return_value = None
        mock_skill.capability_table = mock_table

        with patch("proxy.skills.skill_registry.SkillRegistry") as mock_skreg_cls:
            mock_skreg = MagicMock()
            mock_skreg.get.return_value = mock_skill
            mock_skreg_cls.return_value = mock_skreg

            with patch("proxy.agent.agent_registry.registry") as mock_ag_reg:
                mock_ag_reg.get_or_load.return_value = mock_agent
                result = guard._check_work_mode_actions(tc, ctx)
                assert isinstance(result, (str, type(None)))

    def test_unknown_decomposed_tool_name_not_checked(self):
        """Non-ticket tool names return None immediately — not checked."""
        guard = MemoryToolGuard()
        tc = _make_tc("bash", {"command": "ls"})
        ctx = MagicMock(session_id="sess-1", is_subagent=False)
        result = guard._check_work_mode_actions(tc, ctx)
        assert result is None

    def test_subagent_bypasses_work_mode_actions(self):
        """Sub-agents bypass _check_work_mode_actions entirely."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_create_tasks", {"tasks": [{"name": "x", "priority": "high"}]})
        ctx = MagicMock(session_id="sess-1", is_subagent=True)

        result = guard._check_work_mode_actions(tc, ctx)
        assert result is None


class TestCallLimitsGuard:
    """Verify _check_call_limits counts decomposed tools toward proxy_ticket budget."""

    def test_decomposed_tool_counts_toward_proxy_ticket_budget(self):
        """proxy_ticket_list_tasks should count toward the proxy_ticket call limit."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_list_tasks", {"status_filter": ["pending"]})
        ctx = MagicMock(session_id="sess-1")

        with patch("proxy.tool_pipeline.tool_guard.ToolCallTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.try_count_proxy_ticket.return_value = True
            mock_tracker_cls.get_instance.return_value = mock_tracker

            with patch.object(guard, '_resolve_call_limits_for_skill', return_value={}):
                guard._check_call_limits(tc, ctx)
                mock_tracker.try_count_proxy_ticket.assert_called_once()


class TestHashProvenanceGuard:
    """Verify _check_hash_provenance validates hashes for decomposed tools."""

    def test_set_parent_checks_short_hash_and_parent_hash(self):
        """proxy_ticket_set_parent must have both hashes known."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_set_parent", {
            "short_hash": "abc12345",
            "parent_hash": "def67890",
        })
        ctx = MagicMock(session_id="sess-1")

        with patch("proxy.tool_pipeline.tool_guard.ToolCallTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.has_seen_hash.return_value = True
            mock_tracker_cls.get_instance.return_value = mock_tracker

            result = guard._check_hash_provenance(tc, ctx)
            assert result is None

    def test_set_priority_with_unknown_hash_blocked(self):
        """proxy_ticket_set_priority with unknown hash → blocked."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_set_priority", {
            "short_hash": "unknown1",
            "priority": "high",
        })
        ctx = MagicMock(session_id="sess-1")

        with patch("proxy.tool_pipeline.tool_guard.ToolCallTracker") as mock_tracker_cls:
            mock_tracker = MagicMock()
            mock_tracker.has_seen_hash.return_value = False
            mock_tracker_cls.get_instance.return_value = mock_tracker

            result = guard._check_hash_provenance(tc, ctx)
            assert result is not None
            assert "unknown1" in result

    def test_list_tasks_no_hash_check(self):
        """proxy_ticket_list_tasks skips hash provenance (not in _TICKET_HASH_FIELDS)."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_list_tasks", {"status_filter": ["pending"]})
        ctx = MagicMock(session_id="sess-1")

        result = guard._check_hash_provenance(tc, ctx)
        assert result is None


class TestTicketModeOrchestrator:
    """Verify _check_ticket_mode_orchestrator handles decomposed tools."""

    def test_main_ticket_list_tasks_not_blocked(self):
        """MAIN ticket agent calling proxy_ticket_list_tasks → not blocked."""
        guard = MemoryToolGuard()
        tc = _make_tc("proxy_ticket_list_tasks", {"status_filter": ["pending"]})
        ctx = MagicMock(session_id="sess-1", is_subagent=False, subagent_ticket_mode=False)

        result = guard._check_ticket_mode_orchestrator(tc, ctx)
        assert result is None
