"""Unit tests for work-mode action restriction (US-095) under Slice 8b decomposition.

After Slice 8b, the monolithic `proxy_ticket` tool with an `action` enum is replaced
by 10 action-specific tools (`proxy_ticket_list_tasks`, `proxy_ticket_create_tasks`,
etc.). Work mode MAIN orchestrators can only call read-only tools
(`proxy_ticket_list_tasks`, `proxy_ticket_get_task`, `proxy_ticket_append_review_note`).
"""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import MagicMock, patch

from proxy.tool_pipeline.tool_call import ToolCall
from proxy.tool_pipeline.tool_guard import MemoryToolGuard


def _tc(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=f"call_{uuid.uuid4().hex[:24]}", name=name, arguments=arguments)


@pytest.fixture
def session_id():
    return "test-sess-work-guard"


@pytest.fixture
def work_mode_state():
    ms = MagicMock()
    ms.name = "work"
    ms.is_expired.return_value = False
    return ms


@pytest.fixture
def ticket_mode_state():
    ms = MagicMock()
    ms.name = "ticket"
    ms.is_expired.return_value = False
    return ms


@pytest.fixture
def mock_agent_in_work(work_mode_state):
    a = MagicMock()
    a.snapshot_mode.return_value = work_mode_state
    return a


@pytest.fixture
def mock_agent_in_ticket(ticket_mode_state):
    a = MagicMock()
    a.snapshot_mode.return_value = ticket_mode_state
    return a


class TestWorkModeAllowsReadOnly:
    """MAIN agent in work mode can call read-only decomposed ticket tools."""

    def test_work_mode_allows_list_tasks(self, session_id, mock_agent_in_work):
        tc = _tc(name="proxy_ticket_list_tasks", arguments={})
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent_in_work,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        assert result is None

    def test_work_mode_allows_get_task(self, session_id, mock_agent_in_work):
        tc = _tc(name="proxy_ticket_get_task", arguments={"short_hash": "abc12345"})
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent_in_work,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        assert result is None

    def test_work_mode_allows_append_review_note(self, session_id, mock_agent_in_work):
        tc = _tc(
            name="proxy_ticket_append_review_note",
            arguments={"short_hash": "abc12345", "note": "Done."},
        )
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent_in_work,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        assert result is None


class TestWorkModeBlocksCRUD:
    """MAIN agent in work mode cannot call CRUD decomposed ticket tools."""

    @pytest.mark.parametrize("blocked_tool", [
        "proxy_ticket_create_tasks",
        "proxy_ticket_update_task",
        "proxy_ticket_edit",
        "proxy_ticket_set_parent",
        "proxy_ticket_clear_parent",
        "proxy_ticket_clear_blocked_by",
        "proxy_ticket_set_priority",
    ])
    def test_work_mode_blocks_crud_tools(
        self, blocked_tool, session_id, mock_agent_in_work
    ):
        tc = _tc(name=blocked_tool, arguments={})
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent_in_work,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        assert result is not None, (
            f"Work MAIN should be blocked from {blocked_tool}, got None"
        )
        assert "delegate" in result.lower() or "blocked" in result.lower() or "cannot" in result.lower()


class TestWorkModeGuardExceptions:
    """Work mode guard only applies in work mode (other modes unaffected)."""

    def test_ticket_mode_blocks_create_tasks(self, session_id, mock_agent_in_ticket):
        """US-096: ticket MAIN is also blocked from write tools (delegate via sub-agents)."""
        tc = _tc(
            name="proxy_ticket_create_tasks",
            arguments={"tasks": [{"name": "X", "priority": "high"}]},
        )
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent_in_ticket,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        assert result is not None, (
            "Ticket MAIN should be blocked from proxy_ticket_create_tasks"
        )

    def test_no_session_id_skips_check(self):
        tc = _tc(name="proxy_ticket_create_tasks", arguments={})
        ctx = MagicMock()
        ctx.session_id = ""
        ctx.is_subagent = False
        guard = MemoryToolGuard()
        result = guard._check_work_mode_actions(tc, ctx)
        assert result is None

    def test_non_proxy_ticket_tool_skips_check(self, session_id, mock_agent_in_work):
        tc = _tc(name="proxy_work", arguments={"action": "start_work"})
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=mock_agent_in_work,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        # proxy_work is not restricted by this guard
        assert result is None

    def test_no_agent_returns_none(self, session_id):
        tc = _tc(name="proxy_ticket_create_tasks", arguments={})
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.is_subagent = False
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=None,
        ):
            guard = MemoryToolGuard()
            result = guard._check_work_mode_actions(tc, ctx)
        assert result == "Authorization failed: agent not found for session"


class TestWorkSubProxyTicketReadOnly:
    """Work sub-agents have READ-ONLY proxy_ticket access via CapabilityTable."""

    def _build_work_sub_ctx_and_tc(self, tool_name):
        from skills.work_skill import WorkSkill
        from tests.mode_harness import activate_work_mode, create_test_agent

        agent = create_test_agent(
            session_id="sess-c75-work-sub",
            agent_id="agent-c75-work-sub",
        )
        activate_work_mode(agent, task_hash="deadbeef")
        skill = WorkSkill()
        registry = MagicMock()
        registry.get.return_value = skill
        ctx = MagicMock()
        ctx.session_id = "sess-c75-work-sub"
        ctx.is_subagent = True
        ctx.client_hint = ""
        tc = ToolCall(
            id="t-c75-work-sub",
            name=tool_name,
            arguments={},
            source="native",
            from_cache=False,
        )
        return skill, agent, registry, ctx, tc

    def test_work_sub_list_tasks_allowed(self):
        from proxy.skill_guard import check_sub_agent_proxy_access
        with patch("proxy.skill_guard.is_trusted_skill", return_value=True):
            _, agent, registry, ctx, tc = self._build_work_sub_ctx_and_tc("proxy_ticket_list_tasks")
            result = check_sub_agent_proxy_access(tc, ctx, agent, registry)
        assert result is None, f"work sub proxy_ticket_list_tasks should be allowed; got: {result!r}"

    def test_work_sub_get_task_allowed(self):
        from proxy.skill_guard import check_sub_agent_proxy_access
        with patch("proxy.skill_guard.is_trusted_skill", return_value=True):
            _, agent, registry, ctx, tc = self._build_work_sub_ctx_and_tc("proxy_ticket_get_task")
            result = check_sub_agent_proxy_access(tc, ctx, agent, registry)
        assert result is None, f"work sub proxy_ticket_get_task should be allowed; got: {result!r}"

    def test_work_sub_append_review_note_allowed(self):
        from proxy.skill_guard import check_sub_agent_proxy_access
        with patch("proxy.skill_guard.is_trusted_skill", return_value=True):
            _, agent, registry, ctx, tc = self._build_work_sub_ctx_and_tc("proxy_ticket_append_review_note")
            result = check_sub_agent_proxy_access(tc, ctx, agent, registry)
        assert result is None, f"work sub proxy_ticket_append_review_note should be allowed; got: {result!r}"

    def test_work_sub_create_tasks_denied(self):
        from proxy.skill_guard import check_sub_agent_proxy_access
        with patch("proxy.skill_guard.is_trusted_skill", return_value=True):
            _, agent, registry, ctx, tc = self._build_work_sub_ctx_and_tc("proxy_ticket_create_tasks")
            result = check_sub_agent_proxy_access(tc, ctx, agent, registry)
        assert result is not None, f"work sub proxy_ticket_create_tasks should be denied; got None"
        assert "delegate" in result.lower() or "blocked" in result.lower() or "cannot" in result.lower()

    def test_work_sub_proxy_work_start_work_denied(self):
        from proxy.skill_guard import check_sub_agent_proxy_access
        _, agent, registry, ctx, _ = self._build_work_sub_ctx_and_tc("proxy_ticket_list_tasks")
        tc = ToolCall(
            id="t-c75-work-sub-pw",
            name="proxy_work",
            arguments={"action": "start_work", "short_hash": "abc12345"},
            source="native",
            from_cache=False,
        )
        with patch("proxy.skill_guard.is_trusted_skill", return_value=True):
            result = check_sub_agent_proxy_access(tc, ctx, agent, registry)
        assert result is not None, f"work sub proxy_work(start_work) should be denied; got None"

    def test_work_sub_proxy_work_report_bug_allowed(self):
        from proxy.skill_guard import check_sub_agent_proxy_access
        _, agent, registry, ctx, _ = self._build_work_sub_ctx_and_tc("proxy_ticket_list_tasks")
        tc = ToolCall(
            id="t-c75-work-sub-bug",
            name="proxy_work",
            arguments={"action": "report_bug", "title": "X", "description": "Y", "severity": "low", "task_short_hash": "abc"},
            source="native",
            from_cache=False,
        )
        with patch("proxy.skill_guard.is_trusted_skill", return_value=True):
            result = check_sub_agent_proxy_access(tc, ctx, agent, registry)
        assert result is None, f"work sub proxy_work(report_bug) should be allowed; got: {result!r}"
