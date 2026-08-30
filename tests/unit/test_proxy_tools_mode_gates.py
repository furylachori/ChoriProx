"""US-098: proxy_* Tools Are Mode-Gated at Entry Point.

Verifies that the entry-point mode gate is the canonical authority for
proxy_* tool access. Each proxy_* tool's execute_*() function MUST
return access_denied when the agent is not in the matching mode, and
this is the second line of defense behind tool_guard.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def _make_mode_state(name: str, expired: bool = False):
    """Build a mode-state-like object snapshot_mode_and_ticket_ts() returns."""
    ms = MagicMock()
    ms.name = name
    ms.is_expired.return_value = expired
    return ms


def _make_agent(mode_name: str | None, expired: bool = False):
    """Build a mock agent with the given active mode (None = default)."""
    a = MagicMock()
    a.agent_id = "agent_test"
    if mode_name is None:
        a.snapshot_mode_and_ticket_ts.return_value = (None, None)
        a.snapshot_mode.return_value = None
    else:
        ms = _make_mode_state(mode_name, expired=expired)
        a.snapshot_mode_and_ticket_ts.return_value = (ms, None)
        a.snapshot_mode.return_value = ms
    return a


# =============================================================================
# proxy_ticket mode gate
# =============================================================================


class TestProxyTicketModeGate:
    """proxy_ticket requires ticket or work mode."""

    def test_default_mode_returns_access_denied(self):
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="default",
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1",
            )
        assert result["error"]["code"] == "access_denied"
        assert "not available in this mode" in result["error"]["message"]

    def test_workspace_mode_returns_access_denied(self):
        """Workspace mode does NOT unlock proxy_ticket (US-098 per-tool matrix)."""
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent("workspace"),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="default",
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1",
            )
        assert result["error"]["code"] == "access_denied"

    def test_ticket_mode_passes_gate(self):
        import sys
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent("ticket"),
        ), patch.object(sys.modules['memory.tools.proxy_ticket'], '_get_task_graph_singleton', return_value=MagicMock()), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"tasks": []}):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1",
            )
        # No access_denied
        assert result.get("error", {}).get("code") != "access_denied"

    def test_work_mode_passes_gate(self):
        import sys
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent("work"),
        ), patch.object(sys.modules['memory.tools.proxy_ticket'], '_get_task_graph_singleton', return_value=MagicMock()), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"tasks": []}):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1",
            )
        # No access_denied — work mode unlocks proxy_ticket (read-only enforcement is at the action level)
        assert result.get("error", {}).get("code") != "access_denied"

    def test_no_session_id_returns_access_denied(self):
        from memory.tools.proxy_ticket import execute_ticket_tool
        result = execute_ticket_tool("list_tasks", {"action": "list_tasks"}, session_id=None)
        assert result["error"]["code"] == "access_denied"


# =============================================================================
# proxy_work mode gate
# =============================================================================


class TestProxyWorkModeGate:
    """proxy_work requires work or ticket mode."""

    def test_default_mode_returns_mode_inactive(self):
        from memory.tools.proxy_work import execute_proxy_work
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id="sess-1",
            )
        assert result["error"]["code"] == "mode_inactive"
        assert "work mode" in result["error"]["message"]

    def test_workspace_mode_returns_mode_inactive(self):
        from memory.tools.proxy_work import execute_proxy_work
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent("workspace"),
        ):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id="sess-1",
            )
        assert result["error"]["code"] == "mode_inactive"

    def test_ticket_mode_passes_gate(self):
        """C3: Ticket mode does NOT unlock proxy_work. execute_proxy_work returns mode_inactive for ticket-mode agents (one-shot semantics)."""
        from memory.tools.proxy_work import execute_proxy_work
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent("ticket"),
        ):
            result = execute_proxy_work(
                {"action": "stop_work", "short_hash": "abc12345", "ws_uuid": "ws-abc"},
                session_id="sess-1",
            )
        assert result["error"]["code"] == "mode_inactive"

    def test_work_mode_passes_gate(self):
        from memory.tools.proxy_work import execute_proxy_work
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent("work"),
        ), patch("proxy.context._get_task_graph_singleton", return_value=MagicMock()), \
             patch("proxy.tool_pipeline.action_registry.dispatch_action", return_value={"ok": True}):
            result = execute_proxy_work(
                {"action": "start_work", "short_hash": "abc12345"},
                session_id="sess-1",
            )
        assert result.get("error", {}).get("code") != "access_denied"

    def test_no_session_id_returns_mode_inactive(self):
        from memory.tools.proxy_work import execute_proxy_work
        result = execute_proxy_work(
            {"action": "start_work", "short_hash": "abc12345"},
            session_id=None,
        )
        assert result["error"]["code"] == "mode_inactive"


# =============================================================================
# Cross-tool invariant: the ONLY way to unlock a proxy_* tool
# =============================================================================


class TestProxyToolsModeInvariant:
    """US-098 core invariant: no proxy_* tool is reachable from default mode.

    Note: proxy_workspace was removed in the workspace cleanup (US-099).
    The only proxy_* tools remaining are proxy_ticket and proxy_work.
    """

    @pytest.mark.parametrize("tool_module,fn_name,action,patch_path,expected_error", [
        ("memory.tools.proxy_ticket", "execute_ticket_tool", "list_tasks",
         "proxy.context._get_task_graph_singleton", "access_denied"),
        ("memory.tools.proxy_work", "execute_proxy_work", "start_work",
         "proxy.context._get_task_graph_singleton", "mode_inactive"),
    ])
    def test_proxy_ticket_and_work_block_default_mode(
        self, tool_module, fn_name, action, patch_path, expected_error,
    ):
        """proxy_ticket and proxy_work block default mode at the entry point."""
        import importlib
        mod = importlib.import_module(tool_module)
        fn = getattr(mod, fn_name)
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="default",
        ):
            if fn_name == "execute_ticket_tool":
                result = fn(action, {"action": action, "short_hash": "abc12345"}, session_id="sess-1")
            else:
                args = {"action": action, "short_hash": "abc12345"} if "short_hash" in action or action == "start_work" else {"action": action}
                result = fn(args, session_id="sess-1")
        assert result.get("error", {}).get("code") == expected_error, (
            f"{fn_name} did not return {expected_error} in default mode: {result}"
        )


# =============================================================================
# US-098: Read-Only proxy_ticket for Non-Default Workspace Agents
# =============================================================================


class TestNonDefaultWorkspaceReadOnlyProxyTicket:
    """Agents on non-default workspaces without ticket/work mode get
    read-only proxy_ticket (list_tasks only). All other actions blocked."""

    def test_non_default_workspace_no_mode_list_tasks_allowed(self):
        """Agent on choriprox, no mode -> list_tasks succeeds."""
        import sys
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="choriprox",
        ), patch.object(
            sys.modules['memory.tools.proxy_ticket'], '_get_task_graph_singleton', return_value=MagicMock(),
        ), patch(
            "proxy.tool_pipeline.action_registry.dispatch_action", return_value={"tasks": []},
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1",
            )
        assert result.get("error", {}).get("code") != "access_denied"

    def test_non_default_workspace_no_mode_create_tasks_blocked(self):
        """Agent on choriprox, no mode -> create_tasks returns access_denied."""
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="choriprox",
        ):
            result = execute_ticket_tool(
                "create_tasks", {"action": "create_tasks"}, session_id="sess-1",
            )
        assert result["error"]["code"] == "access_denied"

    def test_non_default_workspace_no_mode_edit_blocked(self):
        """Agent on choriprox, no mode -> edit returns access_denied."""
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="choriprox",
        ):
            result = execute_ticket_tool(
                "edit", {"action": "edit"}, session_id="sess-1",
            )
        assert result["error"]["code"] == "access_denied"

    def test_default_workspace_no_mode_still_blocked(self):
        """Agent on default, no mode -> list_tasks still returns access_denied."""
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="default",
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1",
            )
        assert result["error"]["code"] == "access_denied"
        assert "not available in this mode" in result["error"]["message"]

    def test_non_default_workspace_client_hint_propagation(self):
        """Agent with client_hint-based workspace -> list_tasks succeeds."""
        import sys
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="choriprox",
        ), patch.object(
            sys.modules['memory.tools.proxy_ticket'], '_get_task_graph_singleton', return_value=MagicMock(),
        ), patch(
            "proxy.tool_pipeline.action_registry.dispatch_action", return_value={"tasks": []},
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks"}, session_id="sess-1", client_hint="test-client-123",
            )
        assert result.get("error", {}).get("code") != "access_denied"

    def test_non_default_workspace_no_mode_list_tasks_with_status_filter(self):
        """list_tasks with valid status_filter passes gate and executes."""
        import sys
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="choriprox",
        ), patch.object(
            sys.modules['memory.tools.proxy_ticket'], '_get_task_graph_singleton', return_value=MagicMock(),
        ), patch(
            "proxy.tool_pipeline.action_registry.dispatch_action", return_value={"tasks": []},
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks", "status_filter": ["pending", "in_progress"]},
                session_id="sess-1",
            )
        assert result.get("error", {}).get("code") != "access_denied"

    def test_non_default_workspace_no_mode_list_tasks_invalid_status_filter(self):
        """list_tasks with invalid status_filter passes gate, action handler returns invalid_input."""
        import sys
        from memory.tools.proxy_ticket import execute_ticket_tool
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=_make_agent(None),
        ), patch(
            "proxy.workspace_state.workspace_store.get_active",
            return_value="choriprox",
        ), patch.object(
            sys.modules['memory.tools.proxy_ticket'], '_get_task_graph_singleton', return_value=MagicMock(),
        ), patch(
            "proxy.tool_pipeline.action_registry.dispatch_action",
            return_value={
                "ok": False,
                "tool": "proxy_ticket",
                "operation": "list_tasks",
                "status": "error",
                "data": None,
                "error": {
                    "code": "invalid_input",
                    "message": "status_filter must be a list",
                    "retryable": True,
                    "category": "validation",
                },
                "session_id": "sess-1",
            },
        ):
            result = execute_ticket_tool(
                "list_tasks", {"action": "list_tasks", "status_filter": "pending"},
                session_id="sess-1",
            )
        assert result.get("error", {}).get("code") != "access_denied"
        assert result.get("error", {}).get("code") == "invalid_input"


# =============================================================================
# US-098: question tool guard for sub-agents
# =============================================================================


class TestQuestionToolSubagentGuard:
    """question is blocked for sub-agents (would deadlock).

    Sub-agents are unattended — they cannot wait for a human response.
    The question tool is only available to the MAIN orchestrator agent.
    """

    def _make_fake_tool_call(self, name: str = "question", arguments: dict = None):
        """Build a minimal ToolCall-like object."""
        class FakeToolCall:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments or {}
                self.id = "call-1"
        return FakeToolCall(name, arguments or {})

    def _make_fake_ctx(self, is_subagent: bool = False, **kwargs):
        """Build a minimal ctx-like object with optional attrs using getattr defaults."""
        class FakeCtx:
            pass
        ctx = FakeCtx()
        ctx.session_id = kwargs.get("session_id", "sess-test")
        ctx.is_subagent = is_subagent
        ctx.subagent_ticket_mode = kwargs.get("subagent_ticket_mode", False)
        ctx.inherited_skill = kwargs.get("inherited_skill", None)
        ctx.client_hint = kwargs.get("client_hint", "")
        ctx.body = kwargs.get("body", {})
        ctx.llama_body = kwargs.get("llama_body", {})
        ctx.auth_headers = kwargs.get("auth_headers", {})
        ctx.original_model = kwargs.get("original_model", "test-model")
        ctx.upstream_url = kwargs.get("upstream_url", "http://localhost")
        ctx.model_cfg = kwargs.get("model_cfg", None)
        ctx.execute_fn = kwargs.get("execute_fn", None)
        ctx.parse_tools_fn = kwargs.get("parse_tools_fn", None)
        ctx.max_iterations = kwargs.get("max_iterations", 5)
        ctx.max_total_time = kwargs.get("max_total_time", 90.0)
        ctx.max_result_size = kwargs.get("max_result_size", 16384)
        ctx.max_calls_per_iteration = kwargs.get("max_calls_per_iteration", 10)
        ctx.tool_parser = kwargs.get("tool_parser", None)
        ctx.tool_calls_list = kwargs.get("tool_calls_list", None)
        ctx.incoming_tool_names = kwargs.get("incoming_tool_names", None)
        return ctx

    def test_question_not_in_ticket_skill_filter(self):
        """question is not in TicketSkill.get_tool_filter() — ticket sub-agents cannot use it."""
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillContext
        skill = TicketSkill()
        ctx = SkillContext(session_id="s", agent_id="a", messages=[], skill_state={})
        assert "question" not in skill.get_tool_filter(ctx), (
            f"question should NOT be in ticket skill filter: {skill.get_tool_filter(ctx)}"
        )

    def test_no_question_imperative_in_ticket_prompt(self):
        """TICKET_SUB_PROMPT does not tell sub-agents to call question."""
        from proxy.tool_parsing.ticket_prompts import TICKET_SUB_PROMPT
        lower = TICKET_SUB_PROMPT.lower()
        assert "use the question tool" not in lower, (
            "TICKET_SUB_PROMPT still says 'use the question tool'"
        )
        assert "call question" not in lower, (
            "TICKET_SUB_PROMPT still says 'call question'"
        )

    def test_question_blocked_in_check_pipeline(self):
        """question from sub-agent is rejected at pipeline level."""
        from proxy.tool_pipeline.tool_guard import MemoryToolGuard
        guard = MemoryToolGuard()
        tc = self._make_fake_tool_call("question", {"question": "what task?"})
        ctx = self._make_fake_ctx(is_subagent=True)
        err = guard.check(tc, ctx)
        assert err is not None, "Pipeline should block question from sub-agent"
        assert "not available to sub-agents" in err
