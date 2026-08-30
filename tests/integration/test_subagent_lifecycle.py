"""Integration tests for sub-agent full lifecycle.

Test 3.5: Sub-Agent Full Lifecycle

Verifies the full chain from canary detection to guard, context, and
sync exclusion:
  1. [[__SUBAGENT__:SKILL:ticket]] canary in first user message
     -> detect_subagent_ingress sets is_subagent=True, subagent_ticket_mode=True
     -> MemoryToolGuard._check_hierarchy allows proxy_ticket actions
     -> Context pipeline selects sub-agent system prompt
  2. Sub-agents must NOT trigger submit_sync (US-022) when ignore_subagents=True
  3. Ticket sub-agents have full proxy_ticket access (inherited_skill="ticket")
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proxy.tool_pipeline.tool_guard import MemoryToolGuard
from proxy.tool_pipeline.tool_call import ToolCall
from proxy.canary import detect_subagent_ingress  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(name: str, args: dict, call_id: str = "call_test") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


class _StubCtx:
    """Minimal ToolContext stub for guard tests.

    The MemoryToolGuard reads `is_subagent`, `subagent_ticket_mode`, and
    `inherited_skill` via getattr. We provide the attributes the guard
    inspects without instantiating the real ToolContext dataclass.
    """

    def __init__(
        self,
        session_id: str = "sess-test",
        is_subagent: bool = False,
        subagent_ticket_mode: bool = False,
        inherited_skill=None,
    ):
        self.session_id = session_id
        self.is_subagent = is_subagent
        self.subagent_ticket_mode = subagent_ticket_mode
        self.inherited_skill = inherited_skill


def _patch_agent_registry_no_mode():
    """Patch agent_registry.get_or_load to return an agent with no active mode.

    The guard's mode-allowlist check returns early if mode_state is None.
    We want the hierarchy check to be the only constraint under test.
    """
    fake_agent = MagicMock()
    fake_agent.snapshot_mode.return_value = None  # no mode = no allowlist
    return patch(
        "proxy.tool_pipeline.tool_guard._check_mode_allowlist_needs_agent",
        create=True,
    ) if False else patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        return_value=fake_agent,
    )


# ---------------------------------------------------------------------------
# Test 2.1: Canary → is_subagent=True → tool guard allows ticket tools
# ---------------------------------------------------------------------------


class TestCanaryToGuardChain:
    """End-to-end chain: canary detection sets is_subagent, guard allows tools."""

    # NOTE: legacy tests removed. The four tests in this class (test_canary_sets_subagent_flags,
    # test_guard_allows_proxy_ticket_create_for_ticket_subagent, test_guard_allows_full_proxy_ticket_action_set_for_ticket_subagent,
    # test_context_uses_subagent_prompt_for_ticket_subagent) all relied on the OLD canary
    # format (bare `[[__SUBAGENT__:SKILL:ticket]]` without HMAC sig:ts) and the OLD
    # _check_hierarchy body (which allowed proxy_ticket to sub-agents with subagent_ticket_mode=True
    # without consulting the agent's snapshot_mode). The new contract:
    #  - canary must be HMAC-signed (`[[__SUBAGENT__:SKILL:name:sig:ts]]`)
    #  - _check_hierarchy delegates to check_sub_agent_proxy_access, which uses
    #    the agent's snapshot_mode (not the canary) to determine skill access
    # The new behavior is covered by tests/proxy/test_skill_guard.py and
    # tests/proxy/test_skill_mode.py.

    def test_bare_canary_sets_is_subagent_without_ticket_mode(self):
        """[[__SUBAGENT__]] (bare) sets is_subagent but NOT ticket_mode."""
        from proxy.canary import detect_subagent_ingress

        body = {
            "messages": [
                {"role": "user", "content": "[[__SUBAGENT__]]\nDo something."},
            ],
        }
        result = detect_subagent_ingress(body)
        assert result.is_subagent is True
        assert result.subagent_ticket_mode is False


# ---------------------------------------------------------------------------
# Test 2.2: Sub-agent sync excluded (US-022)
# ---------------------------------------------------------------------------


class TestSubagentSyncExclusion:
    """Sub-agents must NOT trigger submit_sync (per US-022 + ignore_subagents gate)."""

    def test_submit_sync_not_called_for_subagent_with_ignore_subagents(self):
        """When model_cfg.ignore_subagents=True and ctx.is_subagent=True, no submit_sync."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = MagicMock()
        ctx.is_subagent = True
        ctx.session_id = "sess-sub"
        ctx.model_name = "kimi"
        ctx.body = {"messages": []}
        ctx.messages_original = None
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.tool_parser = None
        ctx.model_route.model_cfg = MagicMock()
        ctx.model_route.model_cfg.backend_model = "kimi"
        ctx.model_route.model_cfg.ignore_subagents = True
        ctx.model_route.model_cfg.memory_enabled = True
        ctx.upstream = MagicMock()
        ctx.upstream.url = "http://localhost:9999"
        ctx.upstream.headers = {}
        ctx.upstream.original_model = "kimi"
        ctx.response_data = {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"},
            ],
            "model": "kimi",
        }

        with patch("memory.sync.submit_sync") as mock_submit:
            PostProcessStage().execute(ctx)

        assert not mock_submit.called, (
            f"submit_sync should NOT be called for sub-agent with ignore_subagents=True; "
            f"calls={mock_submit.call_args_list}"
        )

    def test_submit_sync_called_for_non_subagent(self):
        """When ctx.is_subagent=False, submit_sync IS called on final response."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = MagicMock()
        ctx.is_subagent = False
        ctx.session_id = "sess-main"
        ctx.model_name = "kimi"
        ctx.body = {"messages": []}
        ctx.messages_original = None
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.tool_parser = None
        ctx.model_route.model_cfg = MagicMock()
        ctx.model_route.model_cfg.backend_model = "kimi"
        ctx.model_route.model_cfg.ignore_subagents = True
        ctx.model_route.model_cfg.memory_enabled = True
        ctx.upstream = MagicMock()
        ctx.upstream.url = "http://localhost:9999"
        ctx.upstream.headers = {}
        ctx.upstream.original_model = "kimi"
        ctx.response_data = {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"},
            ],
            "model": "kimi",
        }

        with patch("memory.sync.submit_sync") as mock_submit:
            PostProcessStage().execute(ctx)

        assert mock_submit.called, (
            f"submit_sync should be called for non-subagent; "
            f"calls={mock_submit.call_args_list}"
        )

    def test_post_process_gate_logic_subagent_excluded(self):
        """Drive real PostProcessStage.execute(): sub-agent + ignore_subagents=True
        must NOT call submit_sync (US-022)."""
        from proxy.pipeline.post_process import PostProcessStage

        ctx = MagicMock()
        ctx.is_subagent = True
        ctx.session_id = "sess-sub"
        ctx.model_name = "kimi"
        ctx.body = {"messages": []}
        ctx.messages_original = None
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.tool_parser = None
        ctx.model_route.model_cfg = MagicMock()
        ctx.model_route.model_cfg.backend_model = "kimi"
        ctx.model_route.model_cfg.ignore_subagents = True
        ctx.model_route.model_cfg.memory_enabled = True
        ctx.upstream = MagicMock()
        ctx.upstream.url = "http://localhost:9999"
        ctx.upstream.headers = {}
        ctx.upstream.original_model = "kimi"
        ctx.response_data = {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"},
            ],
            "model": "kimi",
        }

        with patch("memory.sync.submit_sync") as mock_submit:
            PostProcessStage().execute(ctx)

        assert not mock_submit.called, (
            f"submit_sync should NOT be called for sub-agent with ignore_subagents=True; "
            f"calls={mock_submit.call_args_list}"
        )

    def test_memory_graph_not_polluted_by_subagent(self):
        """US-022: sub-agents are excluded from the SOTA memory sync path.

        Drive the real PostProcessStage.execute() — when ctx.is_subagent=True
        and model.ignore_subagents=True, submit_sync must NOT be called,
        so no entity extraction or SOTA sync happens.
        """
        from proxy.pipeline.post_process import PostProcessStage

        ctx = MagicMock()
        ctx.is_subagent = True
        ctx.session_id = "sess-sub"
        ctx.model_name = "kimi"
        ctx.body = {"messages": []}
        ctx.messages_original = None
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.tool_parser = None
        ctx.model_route.model_cfg = MagicMock()
        ctx.model_route.model_cfg.backend_model = "kimi"
        ctx.model_route.model_cfg.ignore_subagents = True
        ctx.model_route.model_cfg.memory_enabled = True
        ctx.upstream = MagicMock()
        ctx.upstream.url = "http://localhost:9999"
        ctx.upstream.headers = {}
        ctx.upstream.original_model = "kimi"
        ctx.response_data = {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"},
            ],
            "model": "kimi",
        }

        with patch("memory.sync.submit_sync") as mock_submit:
            PostProcessStage().execute(ctx)

        assert not mock_submit.called, (
            f"submit_sync should NOT be called for sub-agent with ignore_subagents=True; "
            f"calls={mock_submit.call_args_list}"
        )


# ---------------------------------------------------------------------------
# Test 2.3: Ticket sub-agent has full proxy_ticket access
# ---------------------------------------------------------------------------


class TestTicketSubagentFullAccess:
    """Ticket sub-agents (inherited_skill='ticket') have unrestricted proxy_ticket access."""

    # NOTE: legacy tests removed. The four tests in this class (test_ticket_subagent_allowed_for_every_action,
    # test_ticket_subagent_can_use_proxy_work, test_non_ticket_subagent_cannot_modify_tasks,
    # test_non_ticket_subagent_can_list_tasks) all relied on the OLD _check_hierarchy body,
    # which used the `subagent_ticket_mode` canary flag to decide whether to allow
    # proxy_ticket/proxy_work actions. The new _check_hierarchy delegates to
    # check_sub_agent_proxy_access, which uses the agent's snapshot_mode() (not the
    # canary) to determine the active skill and its ProxyAccessLevel. Tests with
    # snapshot_mode()=None now hit the "bare sub-agent" default-deny path.
    # The new contract is covered by tests/proxy/test_skill_guard.py.

    def test_entry_point_gate_does_not_block_ticket_subagent(self):
        """No upstream gate (security stage, etc.) blocks ticket sub-agents.

        We verify the canary detection produces flags that the rest of
        the pipeline accepts (no exception, no early rejection).
        """
        from proxy.canary import detect_subagent_ingress

        body = {
            "messages": [
                {"role": "user", "content": "[[__SUBAGENT__:SKILL:ticket]]\nWork."},
            ],
        }
        # No exception during ingress detection
        result = detect_subagent_ingress(body)
        assert result.is_subagent is True
        # The sub-agent is admitted to the pipeline (no blocked marker)
        assert "blocked" not in str(result).lower()


# ---------------------------------------------------------------------------
# Bonus: tool guard self-reference / work-mode actions tests (regression coverage)
# ---------------------------------------------------------------------------


class TestGuardRegressionCoverage:
    """Quick coverage of other guard branches to ensure no regressions in
    the sub-agent hierarchy check."""

    def test_self_reference_blocks_agent_tool(self):
        """Tools that start with 'agent:' or 'ticket:' are blocked (self-reference)."""
        guard = MemoryToolGuard()
        ctx = _StubCtx(session_id="sess", is_subagent=False)
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=MagicMock(snapshot_mode=lambda: None),
        ):
            tc = _tc("agent:something", {})
            err = guard._check_self_reference(tc, ctx)
        assert err is not None
        assert "self-reference" in err.lower()

    # NOTE: legacy tests removed. The three tests in this section (test_work_subagent_cannot_create_via_proxy_ticket,
    # test_work_subagent_can_list_via_proxy_ticket, test_work_subagent_can_append_review_note) all relied on
    # the OLD _check_hierarchy body, which had per-skill "if inherited_skill == 'work'" branches.
    # The new _check_hierarchy delegates to check_sub_agent_proxy_access, which uses
    # the agent's snapshot_mode() (not inherited_skill) to determine the active skill
    # and its ProxyAccessLevel. Tests with snapshot_mode()=None now hit the
    # "bare sub-agent" default-deny path. The new contract is covered by
    # tests/proxy/test_skill_guard.py.
