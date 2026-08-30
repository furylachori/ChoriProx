"""
Unit tests for mode resolution and application (apply-mode-unification spec).

Covers 3 new functions in src/proxy/mode.py:
  - resolve_mode(ctx, agent) -> Mode
  - apply_mode(ctx, agent, body, mode) -> dict
  - apply_post_process(ctx, agent, data, mode) -> dict

Each test class corresponds to one function; individual tests map 1:1 to spec lines.
Phase 5 refactoring: Mode.TICKET → Mode.SKILL, TTL-based → active_mode-based.
"""

import os
import pytest
import threading
from unittest.mock import MagicMock, patch

from proxy.mode import Mode
# NOTE: legacy prompt imports (OUR_SYSTEM_PROMPT, OUR_SUB_PROMPT, TICKET_MANAGER_PROMPT)
# removed — the legacy prompt-content tests that referenced them have been deleted.
# The new apply_mode delegates to skill_prompts.assemble_system_prompt.


# R1-A test-fix (Edit A1 propagation + Edit A2 None-ownership guard):
# PostProcessStage.execute() reads `ctx.mode_snapshot_pre_upstream` via the
# class-level guard `hasattr(type(ctx), "mode_snapshot_pre_upstream")`. A
# plain MagicMock() does NOT declare this attribute on its class, so the
# guard returns False and `expected_mode` falls through to None — bypassing
# the lifecycle clear and silently dropping the test's per-turn contract.
#
# Declare the field on the MagicMock class so the production code's
# class-level guard accepts MagicMock ctx instances and the test can set
# the instance attribute to a real ModeState for the CAS.
MagicMock.mode_snapshot_pre_upstream = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(body=None, is_subagent=False, subagent_ticket_mode=False,
              seeded_tasks=None, _ticket_activated=False, _mode=None, session_id="sess_test",
              _skill_activated=False):
    """Build a minimal RequestContext-like namespace object."""
    body = body or {"messages": [{"role": "user", "content": "hello"}], "tools": []}
    ctx = MagicMock()
    ctx.body = body
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.subagent_ticket_mode = subagent_ticket_mode
    ctx.seeded_tasks = seeded_tasks or []
    ctx._ticket_activated = _ticket_activated
    ctx._skill_activated = _skill_activated
    ctx._mode = _mode
    ctx.model_cfg = MagicMock()
    ctx.inherited_skill = None
    # Mirror RequestContext: messages is extracted from body["messages"]
    ctx.messages = body.get("messages", [])
    return ctx


class _MockActiveMode:
    """Minimal ModeState-like object for tests."""
    def __init__(self, age_seconds=0):
        import time as _t
        self.name = "ticket"
        self.activated_at = _t.time() - age_seconds
        self.ttl_seconds = 300
        self.skill_state = {}

    def is_expired(self):
        return False


class _MockAgent:
    """Mock agent with active_mode-based skill mode (Phase 5 refactoring)."""

    def __init__(self, is_ticket_mode=False, agent_id="mock_agent"):
        self._mode_lock = threading.RLock()
        self.agent_id = agent_id
        self.config = MagicMock()
        self._deactivation_callbacks = []

        if is_ticket_mode:
            self.active_mode = _MockActiveMode()
        else:
            self.active_mode = None

    def set_active_mode(self, mode_state, expected_mode=None):
        self._set_active_mode_calls = getattr(self, '_set_active_mode_calls', 0) + 1
        self.active_mode = mode_state
        return True

    def clear_active_mode(self, expected_mode=None):
        self._clear_active_mode_calls = getattr(self, '_clear_active_mode_calls', 0) + 1
        old = self.active_mode
        self.active_mode = None
        for cb in self._deactivation_callbacks:
            cb(old)
        return True

    def register_deactivation_callback(self, cb):
        self._deactivation_callbacks.append(cb)

    is_workspace_mode = False

    @property
    def is_ticket_mode(self):
        mode = getattr(self, 'active_mode', None)
        return mode is not None and getattr(mode, 'name', None) == 'ticket'

    def snapshot_mode(self):
        return self.active_mode

    def snapshot_mode_and_ticket_ts(self):
        return self.active_mode, getattr(self, '_ticket_mode_ts', None)

    def get_mode_name(self):
        ms = self.active_mode
        return ms.name if ms else None

    def is_mode_expired(self):
        return False


def _make_agent(is_ticket_mode=False, agent_id="mock_agent", active_mode_age=0):
    """Build a mock agent with active_mode-based is_ticket_mode property."""
    agent = _MockAgent(is_ticket_mode=is_ticket_mode, agent_id=agent_id)
    if is_ticket_mode and active_mode_age > 0:
        agent.active_mode = _MockActiveMode(age_seconds=active_mode_age)
    return agent


# ===========================================================================
# Test resolve_mode(ctx, agent) -> Mode
# Lines 56-74 of spec: _ticket_activated guard, sub-agent inheritance,
#   all 4 Mode values, stale-clear with/without _ticket_activated.
# ===========================================================================

class TestResolveMode:
    """Tests for resolve_mode() per spec lines 56-74."""

    # --- _ticket_activated guard (spec step 2) ---

    # NOTE: legacy test removed. test_stale_clear_fires_without_ticket_activated
    # tested the old per-request stale-clear path that mutated the mock's
    # _clear_active_mode_calls counter. The new resolve_mode delegates to
    # proxy.skill_resolver.resolve_mode; the legacy stale-clear semantics
    # live there now. See tests/proxy/test_skill_mode.py for the new behavior.

    def test_stale_clear_suppressed_with_ticket_activated(self):
        """_skill_activated=True -> stale clear skipped, mode stays SKILL."""
        agent = _make_agent(is_ticket_mode=True)
        ctx = _make_ctx(
            body={"messages": [{"role": "user", "content": "hi"}], "tools": []},
            _ticket_activated=True,    # activating request
            _skill_activated=True,     # blocks stale clear
        )
        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)
        assert getattr(agent, '_clear_active_mode_calls', 0) == 0
        assert mode == Mode.SKILL

    def test_stale_clear_skipped_for_assistant_last_msg(self):
        """Last message is assistant -> stale clear skipped even without flag."""
        agent = _make_agent(is_ticket_mode=True)
        ctx = _make_ctx(
            body={"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "done"},
            ], "tools": []},
            is_subagent=False,
            _ticket_activated=False,
        )
        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)
        assert getattr(agent, '_clear_active_mode_calls', 0) == 0
        assert mode == Mode.SKILL

    # --- Sub-agent inheritance (spec step 3) ---

    # NOTE: legacy test removed. test_subagent_inherits_ticket_mode tested
    # the old SKILL_SUBAGENT activation path that depended on the legacy
    # _activate_skill side effect. The new resolve_mode delegates to
    # proxy.skill_resolver.resolve_mode, which uses a different activation
    # contract. See tests/proxy/test_skill_mode.py::test_subagent_inheritance_skill_mode
    # for the new behavior.

    def test_subagent_no_inheritance_when_flag_false(self):
        """is_subagent=True + subagent_ticket_mode=False -> no inheritance."""
        agent = _make_agent(is_ticket_mode=False)
        ctx = _make_ctx(
            body={"messages": [{"role": "user", "content": "hi"}], "tools": []},
            is_subagent=True,
            subagent_ticket_mode=False,
        )
        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)
        assert mode == Mode.SUBAGENT

    # --- All 4 Mode values (spec step 4) ---

    def test_mode_default_non_subagent_no_ticket(self):
        """Not subagent + not ticket -> DEFAULT."""
        agent = _make_agent(is_ticket_mode=False)
        ctx = _make_ctx(
            body={"messages": [{"role": "user", "content": "hi"}], "tools": []},
            is_subagent=False,
            subagent_ticket_mode=False,
        )
        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)
        assert mode == Mode.DEFAULT

    def test_mode_skill_non_subagent_yes_ticket(self):
        """Not subagent + ticket -> SKILL."""
        agent = _make_agent(is_ticket_mode=True)
        ctx = _make_ctx(
            body={"messages": [{"role": "assistant", "content": "ok"}], "tools": []},
            is_subagent=False,
            subagent_ticket_mode=False,
            _ticket_activated=True,
        )
        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)
        assert mode == Mode.SKILL

    def test_mode_subagent_no_ticket(self):
        """is_subagent=True + no ticket -> SUBAGENT."""
        agent = _make_agent(is_ticket_mode=False)
        ctx = _make_ctx(
            body={"messages": [{"role": "user", "content": "hi"}], "tools": []},
            is_subagent=True,
            subagent_ticket_mode=False,
        )
        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)
        assert mode == Mode.SUBAGENT

    # NOTE: legacy test removed. test_mode_skill_subagent tested the old
    # SKILL_SUBAGENT activation path that depended on the legacy
    # _activate_skill side effect. The new resolve_mode delegates to
    # proxy.skill_resolver.resolve_mode, which uses a different activation
    # contract. See tests/proxy/test_skill_mode.py::test_subagent_inheritance_skill_mode
    # for the new behavior.

    # --- Side-effects (spec steps 5-6) ---

    def test_ctx_mode_set(self):
        """ctx._mode is set to resolved Mode."""
        agent = _make_agent(is_ticket_mode=True)
        ctx = _make_ctx(
            body={"messages": [{"role": "assistant", "content": "ok"}], "tools": []},
            is_subagent=False,
            _ticket_activated=True,
        )
        from proxy.mode import resolve_mode
        resolve_mode(ctx, agent)
        assert ctx._mode == Mode.SKILL

    def test_return_value(self):
        """resolve_mode returns the Mode enum, not just mutates ctx."""
        agent = _make_agent(is_ticket_mode=False)
        ctx = _make_ctx(
            body={"messages": [{"role": "user", "content": "hi"}], "tools": []},
            is_subagent=False,
        )
        from proxy.mode import resolve_mode
        result = resolve_mode(ctx, agent)
        assert isinstance(result, Mode)
        assert result == Mode.DEFAULT


# ===========================================================================
# Test apply_mode(ctx, agent, body, mode) -> dict
# Lines 81-107 of spec: working_dir AFTER prompt selection, model_cfg passed.
# ===========================================================================

class TestApplyMode:
    """Tests for apply_mode() per spec lines 81-107."""

    def setup_method(self):
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def teardown_method(self):
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    # --- G10: working_dir appended after prompt selection ---

    # NOTE: legacy test removed. test_working_dir_appended_after_prompt_skill
    # tested the old prompt structure (OUR_SYSTEM_PROMPT + MANDATORY block +
    # working_dir). The new apply_mode delegates to skill_prompts.assemble_system_prompt
    # which produces a different output structure. See
    # tests/proxy/test_skill_mode.py::test_byte_identical_prompt for the
    # new prompt structure verification.

    def test_working_dir_not_present_when_no_original_system(self):
        """No working_dir injected when original system prompt has none."""
        body = {"messages": [{"role": "user", "content": "hi"}],
                "tools": [{"function": {"name": "read"}}]}
        ctx = _make_ctx(body=body, seeded_tasks=[])
        agent = _make_agent()
        from proxy.mode import apply_mode
        apply_mode(ctx, agent, body, Mode.DEFAULT)
        sys_msg = body["messages"][0]["content"]
        assert "Working directory:" not in sys_msg

    # --- Order: task_context AFTER stable content (KV cache optimization) ---

    # NOTE: legacy test removed. test_task_context_after_working_dir tested
    # the old KV-cache ordering invariants (working_dir → MANDATORY → task_context).
    # The new apply_mode delegates to skill_prompts.assemble_system_prompt and
    # the task context rendering moved there. See tests/proxy/test_skill_mode.py
    # for the new behavior.

    # NOTE: legacy test removed. test_task_seeding_called_for_skill_mode
    # patched the legacy task-context seeding helper (deleted). The new apply_mode
    # is responsible for task context injection now.
    # the new apply_mode. The skill prompt assembly (skill_prompts.assemble_system_prompt)
    # is responsible for task context injection now.



    # --- model_cfg passed correctly (spec step 8: select_context) ---

    # NOTE: legacy test removed. test_model_cfg_forwarded_to_select_context
    # patched proxy.context.select_context (the legacy context-window pruner),
    # which is no longer called by the new apply_mode. The model_cfg is now
    # passed to skill_prompts.assemble_system_prompt and proxy.tool_cache.prepare_tooling_payload.
    # See tests/proxy/test_skill_mode.py::test_apply_mode_skill_active_sets_tools_snapshot
    # for the new model_cfg flow.

    # --- Prompt selection for each mode ---

    # NOTE: legacy prompt-content tests removed. test_prompt_default_includes_our_system,
    # test_prompt_subagent_appends_sub_prompt, test_prompt_skill_uses_ticket_manager_prompt,
    # test_prompt_skill_subagent_uses_focused_prompt, and test_seeded_tasks_injected_for_skill_mode
    # all asserted on the old prompt structure (OUR_SYSTEM_PROMPT from proxy.tool_parsing,
    # MANDATORY block, build_skill_subagent_prompt, build_mandatory_block). The new
    # apply_mode delegates to skill_prompts.assemble_system_prompt which produces
    # a different output. The byte-identical-prompt snapshot in
    # tests/proxy/test_skill_mode.py::test_byte_identical_prompt covers the new structure.

    # --- Tools filtered by prepare_tooling_payload ---

    def test_tools_filtered_for_skill_mode(self):
        """For skill modes, tools are augmented (proxy_ticket added) via the
        LIVE TicketSkill augmentation path.

        Uses a fresh SkillRegistry populated with the REAL builtins so the
        test exercises the production call site
        (SkillRegistry().get(...) -> TicketSkill.get_tool_filter(...) ->
        _get_injectable_memory_tools injection). Asserts on the REAL filter
        values, not on a mock's canned return — a bug in TicketSkill's filter
        or in the augmentation path would now fail this test.
        """
        from proxy.skills.skill_registry import (
            SkillContext,
            SkillRegistry,
            _skill_registry_new__,
        )
        from skills.ticket_skill import TicketSkill

        real_registry = _skill_registry_new__(SkillRegistry)
        real_registry.populate_builtins()

        real_skill = TicketSkill()
        expected_filter = real_skill.get_tool_filter(SkillContext(
            session_id="sess_test",
            agent_id="",
            messages=[],
            skill_state={},
        ))

        body = {"messages": [{"role": "user", "content": "hi"}],
                "tools": [{"function": {"name": "read"}},
                          {"function": {"name": "write"}},
                          {"function": {"name": "bash"}}]}
        ctx = _make_ctx(body=body)
        agent = _make_agent(is_ticket_mode=True)
        from proxy.mode import apply_mode
        with patch("proxy.skills.skill_registry.SkillRegistry",
                   return_value=real_registry):
            apply_mode(ctx, agent, body, Mode.SKILL)

        tool_names = [t.get("function", {}).get("name", "")
                      for t in body.get("tools", [])]

        assert "proxy_ticket_list_tasks" in tool_names, (
            f"proxy_ticket_list_tasks must be injected via the real "
            f"_get_injectable_memory_tools path; got tools={tool_names}"
        )
        assert "proxy_ticket_get_task" in tool_names, (
            f"proxy_ticket_get_task must be injected via the real "
            f"_get_injectable_memory_tools path; got tools={tool_names}"
        )

        assert "write" not in tool_names, (
            f"write must be filtered out by real TicketSkill filter "
            f"{sorted(expected_filter)}; got tools={tool_names}"
        )
        assert "bash" not in tool_names, (
            f"bash must be filtered out by real TicketSkill filter "
            f"{sorted(expected_filter)}; got tools={tool_names}"
        )

        assert "read" in tool_names, (
            f"read must survive the real TicketSkill filter "
            f"{sorted(expected_filter)}; got tools={tool_names}"
        )

    def test_no_ticket_mode_tools_for_default_mode(self):
        """_ticket_mode_tools NOT set for non-skill modes."""
        body = {"messages": [{"role": "user", "content": "hi"}],
                "tools": [{"function": {"name": "read"}}, {"function": {"name": "write"}}, {"function": {"name": "bash"}}]}
        ctx = _make_ctx(body=body)
        agent = _make_agent()
        from proxy.mode import apply_mode
        apply_mode(ctx, agent, body, Mode.DEFAULT)
        assert "_ticket_mode_tools" not in body

    # --- REGRESSION (subagent-ticket-tool): end-to-end subagent ticket inheritance ---

    # NOTE: legacy test removed. test_subagent_ticket_inherits_proxy_ticket_end_to_end
    # tested the old apply_mode + TicketSkill pipeline with mock-side-effects for
    # prepare_tooling_payload and select_context. The new apply_mode delegates to
    # skill_prompts.assemble_system_prompt and the TicketSkill exposes a real
    # ToolingPayload via prepare_tooling_payload. The end-to-end behavior is
    # covered by tests/proxy/test_skill_mode.py
    # (test_subagent_inheritance_skill_mode, test_apply_mode_skill_filters_tools)
    # and tests/integration/test_skill_dispatch_integration.py.

class TestApplyPostProcess:
    """Tests for the PostProcessStage post-process path (apply_skill_response_transform
    + inject_task_canaries + finalize_skill_lifecycle) per spec §7.5.

    Wave 4: apply_post_process was split into 3 ops invoked from
    PostProcessStage.execute(). These tests now exercise PostProcessStage
    end-to-end with the 8 unrelated sub-methods patched to no-ops.
    """

    def setup_method(self):
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def teardown_method(self):
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def _noop_pp_submethods(self):
        """Patch the 8 PostProcessStage sub-methods to no-ops."""
        from contextlib import ExitStack
        from proxy.pipeline.post_process import PostProcessStage as _Stage
        stack = ExitStack()
        noop = lambda self, ctx: None
        stack.enter_context(patch.object(_Stage, "_extract_minimax_reasoning", noop))
        stack.enter_context(patch.object(_Stage, "_scan_write_results", noop))
        stack.enter_context(patch.object(_Stage, "_sync_chat_history", noop))
        stack.enter_context(patch.object(_Stage, "_persist_messages", noop))
        stack.enter_context(patch.object(_Stage, "_populate_vfs_cache", noop))
        stack.enter_context(patch.object(_Stage, "_capture_cache_metrics", noop))
        stack.enter_context(patch.object(_Stage, "_handle_tool_interception", noop))
        stack.enter_context(patch.object(_Stage, "_assign_model_name", noop))
        return stack

    def _patch_get_or_load(self, agent):
        """Return a patcher that makes agent_registry.get_or_load return the test's agent."""
        def _get_or_load(session_id):
            return agent
        return patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=_get_or_load,
        )

    def _make_pp_ctx(self, response_data, session_id="sess_test", is_subagent=False):
        """Build a RequestContext suitable for PostProcessStage.execute()."""
        from unittest.mock import MagicMock
        body = {"messages": [{"role": "user", "content": "hello"}], "tools": []}
        ctx = MagicMock()
        ctx.body = body
        ctx.session_id = session_id
        ctx.is_subagent = is_subagent
        ctx.subagent_ticket_mode = False
        ctx.seeded_tasks = []
        ctx.inherited_skill = None
        ctx._mode = None
        ctx.model_cfg = MagicMock()
        ctx.messages = body["messages"]
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.llama_body = body
        ctx.request_id = "req_pp_unit"
        ctx.client_hint = ""
        ctx.response_data = response_data
        return ctx

    # --- Defensive guard (spec step 1) ---

    def test_none_data_preserves_ticket_mode(self):
        """response_data=None -> clear_active_mode NOT called; mode preserved.

        Per spec §7.5 (Wave 4 fix): when response_data is None (preflight,
        streaming deferred upstream, etc.), no real turn completed. The
        lifecycle finalizer short-circuits and returns False — the mode
        must NOT be unconditionally cleared (which would wipe ticket mode
        mid-streaming before the daemon thread's PostProcessStage runs).
        """
        from unittest.mock import MagicMock, patch
        from proxy.pipeline.post_process import PostProcessStage

        agent = _make_agent(is_ticket_mode=True)
        ctx = self._make_pp_ctx(None, session_id="sess_unit_none")

        # Mock the skill registry to return a skill that allows deactivation.
        # If the finalizer accidentally called clear_active_mode, this mock
        # would record the call — but with the Wave 4 fix it does NOT.
        mock_skill = MagicMock()
        mock_skill.should_deactivate.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent), self._noop_pp_submethods():
            result = PostProcessStage().execute(ctx)

        # Wave 4 fix invariant: data=None short-circuits in
        # finalize_skill_mode BEFORE clear_active_mode is invoked.
        assert getattr(agent, '_clear_active_mode_calls', 0) == 0
        # Active mode is preserved (ticket mode stays installed).
        assert agent.active_mode is not None
        assert agent.active_mode.name == "ticket"
        # execute() still returns None (data was None on entry).
        assert result is None
        # Lifecycle-finalized flag is set (the finally block ran).
        assert ctx.skill_lifecycle_finalized is True

    def test_valid_completion_with_no_active_mode_is_preserved(self):
        """A valid completion is preserved when no skill mode is active."""
        from unittest.mock import MagicMock, patch
        from proxy.pipeline.post_process import PostProcessStage

        agent = _make_agent(is_ticket_mode=False)
        ctx = self._make_pp_ctx({"usage": {}}, session_id="sess_unit_non_dict")
        agent.active_mode = None
        ctx.response_data = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent), self._noop_pp_submethods():
            result = PostProcessStage().execute(ctx)

        # Spec §7.5: lifecycle finalize is idempotent. With no active mode,
        # the call is semantically a no-op even if the MockAgent's counter
        # increments. Assert on the state, not the counter.
        assert agent.active_mode is None
        # The valid completion is preserved by the noop transform/canary.
        assert result is not None

    def test_data_missing_choices_aborts_with_canonical_502(self):
        """US-060a: missing choices fail closed with canonical HTTP 502 data."""
        from unittest.mock import MagicMock, patch
        import pytest
        from proxy.openai_compat import build_upstream_protocol_error
        from proxy.pipeline.exceptions import StageAbort
        from proxy.pipeline.post_process import PostProcessStage

        agent = _make_agent(is_ticket_mode=False)
        ctx = self._make_pp_ctx({"usage": {}}, session_id="sess_unit_no_choices")
        agent.active_mode = None

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent), self._noop_pp_submethods():
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        body, status = exc_info.value.response_data
        assert status == 502
        assert body == build_upstream_protocol_error()
        assert ctx.response_data == build_upstream_protocol_error()
        assert agent.active_mode is None
        assert ctx.skill_lifecycle_finalized is True

    # --- clear_active_mode on stop (spec step 4/5) ---

    def test_clear_on_stop_uses_active_mode(self):
        """finish_reason='stop' -> agent.clear_active_mode() called."""
        from unittest.mock import MagicMock, patch
        from proxy.pipeline.post_process import PostProcessStage

        agent = _make_agent(is_ticket_mode=True)
        ctx = self._make_pp_ctx(
            {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]},
            session_id="sess_unit_clear_stop",
        )

        # Mock the skill registry to return a skill that allows deactivation
        mock_skill = MagicMock()
        mock_skill.should_deactivate.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop").
        ctx.mode_snapshot_pre_upstream = agent.active_mode
        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent), self._noop_pp_submethods():
            PostProcessStage().execute(ctx)

        assert agent._clear_active_mode_calls == 1

    def test_none_finish_reason_with_no_active_mode_does_not_crash(self):
        """A valid completion with no finish reason does not crash without an active mode."""
        from unittest.mock import MagicMock, patch
        from proxy.pipeline.post_process import PostProcessStage

        agent = _make_agent(is_ticket_mode=False)
        agent.active_mode = None
        # Valid completion shape; `finish_reason` is absent and therefore
        # resolves as None. With no active mode, lifecycle clearing is a
        # semantic no-op. Verifies the valid response does not crash
        # PostProcess.
        ctx = self._make_pp_ctx(
            {"choices": [{"message": {"content": None, "tool_calls": []}}]},
            session_id="sess_unit_tool_cont",
        )

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent), self._noop_pp_submethods():
            PostProcessStage().execute(ctx)

        assert agent.active_mode is None

    def test_fresh_agent_loaded_does_not_share_state(self):
        """Each session's agent is independent — different sessions have separate state."""
        from unittest.mock import MagicMock, patch
        from proxy.pipeline.post_process import PostProcessStage

        agent_a = _make_agent(is_ticket_mode=True)
        agent_b = _make_agent(is_ticket_mode=False)
        agent_b.active_mode = None
        ctx_a = self._make_pp_ctx(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
            session_id="sess_unit_a",
        )
        ctx_b = self._make_pp_ctx(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
            session_id="sess_unit_b",
        )

        # Mock the skill registry to return a skill that allows deactivation
        mock_skill = MagicMock()
        mock_skill.should_deactivate.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop") for
        # agent_a. For agent_b, snapshot is None — finalize_skill_mode's
        # None-ownership guard (R2 #1) preserves the absence.
        ctx_a.mode_snapshot_pre_upstream = agent_a.active_mode
        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent_a), self._noop_pp_submethods():
            PostProcessStage().execute(ctx_a)

        # Different agent for ctx_b
        ctx_b.mode_snapshot_pre_upstream = agent_b.active_mode  # None
        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             self._patch_get_or_load(agent_b), self._noop_pp_submethods():
            PostProcessStage().execute(ctx_b)

        # agent_a had an active mode that should now be cleared (per-turn contract).
        assert agent_a.active_mode is None
        # agent_b had no active mode, must remain None (no spurious clear).
        assert agent_b.active_mode is None


# NOTE: TestExtractTaskAnchorMaxChars class removed. The _SUBAGENT_TASK_ANCHOR_MAX_CHARS
# constant and _extract_task_from_messages function were deleted in R1-HIGH-2 (mode.py refactor).
# No replacement tests needed — the functionality was removed.


# NOTE: legacy class removed. TestApplyModeAgentGuard tested an
# agent-None guard that no longer exists in apply_mode — the new
# apply_mode delegates to assemble_system_prompt and assumes a
# non-None agent. See tests/proxy/test_skill_mode.py for the new
# behavior coverage.


# ===========================================================================
# H5: TOCTOU defense — apply_mode uses the snapshot taken at line 194
# ===========================================================================

class TestApplyModeUsesSnapshottedModeName:
    """H5: after agent.snapshot_mode(), apply_mode should use the snapshotted
    ModeState fields (.name, .skill_state) consistently, even if a concurrent
    set_active_mode() mutates the agent's active_mode in place during the
    function body.
    """

    def test_apply_mode_uses_snapshotted_skill_name(self):
        """If agent.active_mode.name is mutated after snapshot, apply_mode
        must still pass the original snapshotted name to prepare_tooling_payload
        and to the skill filter."""
        # Re-import the function in case a prior test (TestExtractTaskAnchorMaxChars)
        # reloaded proxy.mode — ensures we test the current implementation.
        import proxy.mode as _mode_mod
        apply_mode = _mode_mod.apply_mode
        from proxy.types import ToolingPayload

        # Initial mode: "ticket"
        initial_mode = _MockActiveMode()
        initial_mode.name = "ticket"
        initial_mode.skill_state = {"initial_key": "initial_val"}

        # The "mutated" mode: a different ModeState object set on the agent
        # after the snapshot. This represents a concurrent set_active_mode.
        mutated_mode = _MockActiveMode()
        mutated_mode.name = "workspace"
        mutated_mode.skill_state = {"mutated_key": "mutated_val"}

        agent = _make_agent(is_ticket_mode=True)
        agent.active_mode = initial_mode
        assert agent.active_mode is initial_mode
        assert agent.active_mode.name == "ticket"

        # Spy on prepare_tooling_payload (defined in proxy.tool_cache, but
        # imported locally inside apply_mode — patch the source module).
        captured_skill_names = []

        def recorder(body, model_cfg, ticket_mode=False, is_subagent=False, skill_name=None, session_id="", agent_id=""):
            captured_skill_names.append(skill_name)
            return ToolingPayload(
                tools=[], tool_hash=None, cached=False, augmented_tools=[]
            )

        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.get_tool_filter.return_value = frozenset()
        mock_skill.get_system_prompt.return_value = "system-prompt"
        mock_skill.get_subagent_prompt.return_value = ""

        class _SpyRegistry:
            def __init__(self, skill):
                self._skill = skill
            def get(self, name):
                return self._skill if name == "ticket" else None

        body = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        ctx = _make_ctx(body=body)
        ctx._ticket_activated = True
        ctx._skill_activated = True

        # Mutate agent.active_mode in the side_effect to simulate a concurrent
        # set_active_mode() call. If apply_mode re-reads agent.active_mode
        # after the snapshot, it will see "workspace" — but with the H5 fix
        # it uses the snapshotted "ticket".
        def mutating_side_effect(body, model_cfg, ticket_mode=False, is_subagent=False, skill_name=None, session_id="", agent_id=""):
            agent.active_mode = mutated_mode
            return recorder(body, model_cfg, ticket_mode=ticket_mode,
                            is_subagent=is_subagent, skill_name=skill_name)

        with patch("proxy.context.select_context", return_value=None), \
             patch("proxy.tool_cache.prepare_tooling_payload",
                   side_effect=mutating_side_effect), \
             patch("proxy.skills.skill_registry.SkillRegistry",
                   return_value=_SpyRegistry(mock_skill)), \
             patch("proxy.tool_pipeline.tool_registry.registry.get_tools_for_skill",
                   return_value=None):
            apply_mode(ctx, agent, body, Mode.SKILL)

        # The H5 fix: prepare_tooling_payload must have been called with the
        # snapshotted "ticket" name, NOT the mutated "workspace" name.
        assert captured_skill_names, "prepare_tooling_payload was not called"
        assert "workspace" not in captured_skill_names, (
            f"apply_mode used the mutated name 'workspace' instead of the "
            f"snapshotted 'ticket': {captured_skill_names}"
        )
        assert "ticket" in captured_skill_names, (
            f"apply_mode should have used snapshotted 'ticket': {captured_skill_names}"
        )

    # NOTE: legacy test removed. test_apply_mode_defends_against_inplace_mutation_of_snapshotted_modestate
    # tested the H5 TOCTOU defense that captured dict copies of skill_state at
    # snapshot time. The new apply_mode delegates to skill_prompts.assemble_system_prompt
    # which constructs its own SkillContext; the in-place mutation pattern is no
    # longer relevant. The remaining test_apply_mode_uses_snapshotted_skill_name
    # still verifies the core snapshot semantics via prepare_tooling_payload.


# ===========================================================================
# M-D1: TOCTOU defense — REAL concurrency test with threading.Barrier(2)
# ===========================================================================

class TestApplyModeUnderRealConcurrency:
    """M-D1: TOCTOU defense under REAL concurrency, not side_effect mutation.

    Two threads coordinate via threading.Barrier(2):
    - Thread A (apply_mode thread): runs apply_mode(ctx, agent, body, Mode.SKILL).
      apply_mode snapshots agent.active_mode at line 63 of mode.py and uses
      the snapshotted `ms.name` (NOT a re-read of agent.active_mode.name) as
      the `skill_name` argument to prepare_tooling_payload at line 73.
    - Thread B (mutator thread): concurrently mutates agent.active_mode to a
      different ModeState via agent.set_active_mode(mutated_mode). This races
      with apply_mode to demonstrate that the snapshot semantics hold even
      when a real concurrent thread swaps active_mode mid-execution.

    The barrier inside the prepare_tooling_payload recorder ensures Thread B's
    mutation happens AFTER Thread A's snapshot but BEFORE/AT the recorder's
    capture of `skill_name`. The captured skill_name MUST be the snapshotted
    "ticket" name, NOT the mutated "workspace" name.

    Marked with `@pytest.mark.concurrency` so it can be skipped on CI runners
    that lack free-threading support (or otherwise want to opt out of
    concurrency-stress tests).
    """

    @pytest.mark.concurrency
    def test_snapshot_skill_name_under_concurrent_active_mode_swap(self):
        """Concurrent set_active_mode() during apply_mode must not affect the
        `skill_name` passed to prepare_tooling_payload — it must remain the
        SNAPSHOT's name, not the post-mutation value.
        """
        # Re-import the function in case a prior test reloaded proxy.mode —
        # ensures we test the current implementation.
        import proxy.mode as _mode_mod
        apply_mode = _mode_mod.apply_mode
        from proxy.types import ToolingPayload

        # Initial mode: "ticket" (this is what apply_mode will snapshot).
        initial_mode = _MockActiveMode()
        initial_mode.name = "ticket"
        initial_mode.skill_state = {"initial_key": "initial_val"}

        # The "mutated" mode: a different ModeState object set on the agent
        # after the snapshot. If apply_mode re-reads agent.active_mode.name
        # after the snapshot, it would see "workspace".
        mutated_mode = _MockActiveMode()
        mutated_mode.name = "workspace"
        mutated_mode.skill_state = {"mutated_key": "mutated_val"}

        agent = _make_agent(is_ticket_mode=True)
        agent.active_mode = initial_mode
        assert agent.active_mode is initial_mode
        assert agent.active_mode.name == "ticket"

        # Spy on prepare_tooling_payload + barrier for thread coordination.
        captured_skill_names = []
        barrier = threading.Barrier(2, timeout=10.0)

        def recorder(body, model_cfg, ticket_mode=False, is_subagent=False,
                     skill_name=None, session_id="", agent_id=""):
            # We're inside the apply_mode thread, in prepare_tooling_payload.
            # Signal the mutator thread that it may now race the mutation.
            # By the time both threads pass barrier.wait(), apply_mode has
            # already snapshotted active_mode (line 63 of mode.py) and has
            # already bound `skill_name` from the snapshot (line 68).
            try:
                barrier.wait()
            except threading.BrokenBarrierError as e:
                # If the mutator thread died before the barrier, re-raise so
                # the test fails with a clear cause rather than a silent skip.
                raise RuntimeError(
                    f"mutator thread failed to reach barrier: {e!r}"
                ) from e
            captured_skill_names.append(skill_name)
            return ToolingPayload(
                tools=[], tool_hash=None, cached=False, augmented_tools=[]
            )

        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.allowed_tool_names = None
        mock_skill.get_tool_filter.return_value = frozenset()
        mock_skill.get_system_prompt.return_value = "system-prompt"
        mock_skill.get_subagent_prompt.return_value = ""

        class _SpyRegistry:
            def __init__(self, skill):
                self._skill = skill
            def get(self, name):
                return self._skill if name == "ticket" else None

        body = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        ctx = _make_ctx(body=body)
        ctx._ticket_activated = True
        ctx._skill_activated = True

        # Container for mutator thread errors (so we can surface them after join).
        mutator_errors = []

        def mutator():
            """Real concurrent thread that mutates agent.active_mode."""
            try:
                # Wait for apply_mode to reach prepare_tooling_payload's side_effect.
                # At this point, the snapshot has already been taken in apply_mode
                # (line 63 of mode.py) and `skill_name` has been bound from `ms.name`.
                try:
                    barrier.wait()
                except threading.BrokenBarrierError as e:
                    raise RuntimeError(
                        f"apply_mode thread failed to reach barrier: {e!r}"
                    ) from e
                # Mutate AFTER snapshot but DURING prepare_tooling_payload.
                # If apply_mode re-reads agent.active_mode.name here, it would
                # see "workspace" — but the snapshot defense means it uses "ticket".
                agent.set_active_mode(mutated_mode)
            except Exception as e:
                mutator_errors.append(e)

        mutator_thread = threading.Thread(
            target=mutator, name="M-D1-mutator", daemon=True
        )
        mutator_thread.start()

        try:
            with patch("proxy.context.select_context", return_value=None), \
                 patch("proxy.tool_cache.prepare_tooling_payload",
                       side_effect=recorder), \
                 patch("proxy.skills.skill_registry.SkillRegistry",
                       return_value=_SpyRegistry(mock_skill)), \
                 patch("proxy.tool_pipeline.tool_registry.registry.get_tools_for_skill",
                       return_value=None):
                apply_mode(ctx, agent, body, Mode.SKILL)
        finally:
            mutator_thread.join(timeout=10.0)

        # Surface mutator thread errors (e.g., barrier timeout).
        assert not mutator_errors, (
            f"mutator thread raised: {mutator_errors!r}"
        )
        assert not mutator_thread.is_alive(), (
            "mutator thread did not finish within timeout"
        )

        # Confirm the concurrent mutation actually happened — if the mutator
        # never reached agent.active_mode, the test would be a no-op.
        assert agent.active_mode is mutated_mode, (
            "concurrent set_active_mode() did not run; test is non-deterministic. "
            f"active_mode={agent.active_mode!r}, mutated_mode={mutated_mode!r}"
        )

        # The H5/M-D1 fix: prepare_tooling_payload must have been called with
        # the snapshotted "ticket" name, NOT the mutated "workspace" name.
        assert captured_skill_names, "prepare_tooling_payload was not called"
        assert "workspace" not in captured_skill_names, (
            f"apply_mode used the mutated name 'workspace' instead of the "
            f"snapshotted 'ticket': {captured_skill_names}"
        )
        assert "ticket" in captured_skill_names, (
            f"apply_mode should have used snapshotted 'ticket': "
            f"{captured_skill_names}"
        )

    @pytest.mark.concurrency
    def test_snapshot_skill_name_holds_under_repeated_concurrent_mutations(self):
        """Stress: 10 cycles of concurrent set_active_mode() during apply_mode.

        Loops apply_mode 10 times, each time racing with a mutator thread that
        flips active_mode between "ticket" and "workspace". Every iteration
        must record "ticket" (the snapshot at apply_mode entry) — never
        "workspace". Catches races that the single-shot test above might miss
        on intermittent scheduling.
        """
        import proxy.mode as _mode_mod
        apply_mode = _mode_mod.apply_mode
        from proxy.types import ToolingPayload

        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.allowed_tool_names = None
        mock_skill.get_tool_filter.return_value = frozenset()
        mock_skill.get_system_prompt.return_value = "system-prompt"
        mock_skill.get_subagent_prompt.return_value = ""

        class _SpyRegistry:
            def __init__(self, skill):
                self._skill = skill
            def get(self, name):
                return self._skill if name == "ticket" else None

        captured_skill_names = []
        # Mutable holder so the recorder (defined once for the patched mock)
        # can pick up the per-iteration barrier via late binding.
        barrier_holder = [None]

        def recorder(body, model_cfg, ticket_mode=False, is_subagent=False,
                     skill_name=None, session_id="", agent_id=""):
            cur_barrier = barrier_holder[0]
            if cur_barrier is not None:
                try:
                    cur_barrier.wait()
                except threading.BrokenBarrierError as e:
                    raise RuntimeError(
                        f"mutator thread failed to reach barrier: {e!r}"
                    ) from e
            captured_skill_names.append(skill_name)
            return ToolingPayload(
                tools=[], tool_hash=None, cached=False, augmented_tools=[]
            )

        with patch("proxy.context.select_context", return_value=None), \
             patch("proxy.tool_cache.prepare_tooling_payload",
                   side_effect=recorder), \
             patch("proxy.skills.skill_registry.SkillRegistry",
                   return_value=_SpyRegistry(mock_skill)), \
             patch("proxy.tool_pipeline.tool_registry.registry.get_tools_for_skill",
                   return_value=None):
            for i in range(10):
                initial_mode = _MockActiveMode()
                initial_mode.name = "ticket"
                initial_mode.skill_state = {"iter": str(i)}

                mutated_mode = _MockActiveMode()
                mutated_mode.name = "workspace"
                mutated_mode.skill_state = {"iter": str(i), "mutated": True}

                agent = _make_agent(is_ticket_mode=True)
                agent.active_mode = initial_mode

                body = {"messages": [{"role": "user", "content": "hi"}],
                        "tools": []}
                ctx = _make_ctx(body=body)
                ctx._ticket_activated = True
                ctx._skill_activated = True

                barrier = threading.Barrier(2, timeout=10.0)
                barrier_holder[0] = barrier
                mutator_errors = []

                def mutator(_initial=initial_mode, _mutated=mutated_mode,
                            _agent=agent, _errors=mutator_errors,
                            _barrier=barrier):
                    try:
                        _barrier.wait()
                        _agent.set_active_mode(_mutated)
                    except Exception as e:
                        _errors.append(e)

                t = threading.Thread(target=mutator, daemon=True)
                t.start()
                try:
                    apply_mode(ctx, agent, body, Mode.SKILL)
                finally:
                    t.join(timeout=10.0)
                    barrier_holder[0] = None  # release for next iteration
                assert not mutator_errors, (
                    f"iter {i}: mutator thread raised: {mutator_errors!r}"
                )
                assert not t.is_alive(), (
                    f"iter {i}: mutator thread did not finish within timeout"
                )
                assert agent.active_mode is mutated_mode, (
                    f"iter {i}: concurrent set_active_mode() did not run"
                )

        # Across all 10 iterations, every captured skill_name must be "ticket".
        assert len(captured_skill_names) == 10, (
            f"expected 10 recorder calls, got {len(captured_skill_names)}: "
            f"{captured_skill_names}"
        )
        assert all(n == "ticket" for n in captured_skill_names), (
            f"apply_mode used a non-snapshot name in some iteration: "
            f"{captured_skill_names}"
        )
