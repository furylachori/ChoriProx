"""Unit tests for proxy_work tool execution."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from memory.tools.proxy_work import (
    WORK_ACTIONS,
    execute_proxy_work,
    proxy_work,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def session_id():
    return "test-sess-proxy-work"


@pytest.fixture
def mock_agent():
    a = MagicMock()
    a.agent_id = "agent_test"
    return a


@pytest.fixture
def mock_tg():
    return MagicMock()


@pytest.fixture
def mock_action_result():
    return {"ok": True, "ws_uuid": "ws-abc", "short_hash": "abc12345"}


def _make_mode_state(name: str):
    """Build a mode-state-like object snapshot_mode_and_ticket_ts() returns."""
    ms = MagicMock()
    ms.name = name
    ms.is_expired.return_value = False
    return ms


@pytest.fixture
def patch_agent_in_work_mode(mock_agent):
    """Patch agent_registry so the agent is in 'work' mode."""
    with patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=mock_agent,
    ) as p:
        mock_agent.snapshot_mode_and_ticket_ts.return_value = (_make_mode_state("work"), None)
        mock_agent.snapshot_mode.return_value = _make_mode_state("work")
        yield p


@pytest.fixture
def patch_agent_in_ticket_mode(mock_agent):
    """Patch agent_registry so the agent is in 'ticket' mode."""
    with patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=mock_agent,
    ) as p:
        mock_agent.snapshot_mode_and_ticket_ts.return_value = (_make_mode_state("ticket"), None)
        mock_agent.snapshot_mode.return_value = _make_mode_state("ticket")
        yield p


@pytest.fixture
def patch_agent_no_mode(mock_agent):
    """Patch agent_registry so the agent has NO active mode (default)."""
    with patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=mock_agent,
    ) as p:
        mock_agent.snapshot_mode_and_ticket_ts.return_value = (None, None)
        mock_agent.snapshot_mode.return_value = None
        yield p


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestExecuteProxyWorkDispatch:
    """execute_proxy_work dispatches to the right action handler."""

    def test_dispatches_to_start_work(self, session_id, mock_tg, mock_action_result, patch_agent_in_work_mode):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value=mock_action_result) as mock_dispatch:
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )

        assert result["ok"] is True
        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "start_work"
        assert call_args[0][1]["short_hash"] == "abc12345"
        assert call_args[1]["caller_agent_id"] == "agent_test"

    def test_dispatches_to_stop_work_with_ws_uuid(self, session_id, mock_tg, patch_agent_in_work_mode):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"ok": True}) as mock_dispatch:
            result = execute_proxy_work(
                {"action": "stop_work", "short_hash": "abc12345", "ws_uuid": "ws-xyz"},
                session_id=session_id,
            )

        assert result["ok"] is True
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "stop_work"
        assert call_args[0][1]["ws_uuid"] == "ws-xyz"

    def test_dispatches_to_report_bug(self, session_id, mock_tg, patch_agent_in_work_mode):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"ok": True}) as mock_dispatch:
            result = execute_proxy_work(
                {
                    "action": "report_bug",
                    "task_short_hash": "abc12345",
                    "title": "Login fails on empty input",
                    "description": "Empty email causes 500.",
                    "severity": "high",
                },
                session_id=session_id,
            )

        assert result["ok"] is True
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "report_bug"
        assert call_args[0][1]["severity"] == "high"

    def test_stop_work_blocked_from_ticket_mode(self, session_id, mock_tg, patch_agent_in_ticket_mode):
        """C3: stop_work is BLOCKED from ticket mode (was US-093 allowed).

        Previously US-093 stated "ticket has full proxy_work". C3 reverses
        this: stop_work requires WORK mode. The MAIN agent in ticket mode
        must delegate to a work sub-agent rather than close the session
        directly. Other proxy_work actions (start_work, report_bug, etc.)
        were already blocked; stop_work now joins them.
        """
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"ok": True}):
            result = execute_proxy_work(
                {"action": "stop_work", "short_hash": "abc12345", "ws_uuid": "ws-abc"},
                session_id=session_id,
            )
        assert result["error"]["code"] == "mode_inactive", (
            f"stop_work from ticket mode should be blocked after C3; got: {result}"
        )


class TestExecuteProxyWorkModeGate:
    """Mode gate: proxy_work is only callable in work or ticket mode.

    This is the regression test for the bug where default mode could
    dispatch proxy_work actions. The mode gate enforces: no mode = no
    proxy_work (must use ^^work or ^^ticket to unlock).
    """

    def test_default_mode_returns_access_denied(self, session_id, mock_tg, patch_agent_no_mode):
        """An agent in default mode (no skill active) MUST NOT call proxy_work."""
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )
        assert result["error"]["code"] == "mode_inactive"
        assert "not in work mode" in result["error"]["message"]

    def test_workspace_mode_returns_access_denied(self, session_id, mock_tg, mock_agent):
        """An agent in workspace mode MUST NOT call proxy_work."""
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=mock_agent), \
             patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            mock_agent.snapshot_mode_and_ticket_ts.return_value = (
                _make_mode_state("workspace"), None,
            )
            mock_agent.snapshot_mode.return_value = _make_mode_state("workspace")
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )
        assert result["error"]["code"] == "mode_inactive"

    def test_no_session_id_returns_access_denied(self, mock_tg):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=None,
            )
        assert result["error"]["code"] == "mode_inactive"

    def test_work_mode_passes_mode_gate(self, session_id, mock_tg, mock_action_result, patch_agent_in_work_mode):
        """Work mode agents pass the gate and reach dispatch."""
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value=mock_action_result):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )
        # No error → reached dispatch
        assert result.get("status") != "error" or result.get("ok") is True


class TestExecuteProxyWorkErrors:
    """execute_proxy_work returns proper error responses."""

    def test_no_agent_returns_access_denied(self, session_id, mock_tg):
        """No agent in registry → access_denied (mode gate runs first)."""
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id=session_id,
            )
        assert result["error"]["code"] == "mode_inactive"

    def test_returns_unknown_action_for_invalid_action(self, session_id, mock_tg, patch_agent_in_work_mode):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            result = execute_proxy_work(
                {"action": "fake_action"},
                session_id=session_id,
            )

        assert result["error"]["code"] == "unknown_action"
        assert "start_work" in result["valid_actions"]
        assert "stop_work" in result["valid_actions"]

    def test_returns_invalid_input_for_missing_action(self, session_id, mock_tg, patch_agent_in_work_mode):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            result = execute_proxy_work({}, session_id=session_id)

        assert result["error"]["code"] == "invalid_input"
        assert "Missing required parameter: action" in result["error"]["message"]


class TestExecuteProxyWork:
    """execute_proxy_work() entry-point accepts kwargs from the wrapper."""

    def test_passes_kwargs_through_to_execute_proxy_work(self, mock_tg, patch_agent_in_work_mode):
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"ok": True}):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id="sess-x",
            )

        assert result["ok"] is True


class TestWorkActionsEnum:
    """WORK_ACTIONS is the canonical set of accepted work actions."""

    def test_work_actions_set_is_correct(self):
        assert WORK_ACTIONS == {
            "start_work", "stop_work", "report_bug", "review_bug",
            "validate_bug", "resolve_bug", "handoff_task",
        }

    def test_work_actions_does_not_contain_ticket_actions(self):
        ticket_actions = {
            "create_tasks", "update_task", "list_tasks", "get_task", "edit",
            "set_parent", "clear_parent", "clear_blocked_by",
            "set_priority", "append_review_note",
        }
        assert not (WORK_ACTIONS & ticket_actions)
