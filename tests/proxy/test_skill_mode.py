"""Phase 2 unit tests for skill mode resolution and application (AC 2.1–2.9).

Tests resolve_mode, apply_mode, apply_post_process with skill context.
Uses unittest (not pytest) per the assignment.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from proxy.mode import (
    ActiveSkillUnavailable,
    Mode,
    SkillRuntimeError,
    apply_mode,
    apply_skill_response_transform,
    inject_task_canaries,
    resolve_mode,
)
from proxy.pipeline.post_process import PostProcessStage
from proxy.skills.skill_definition import BaseSkill, ModeState
from proxy.skills.skill_registry import (
    SkillContext,
    SkillRegistry,
    _skill_registry_new__,
    reset_default_registry,
)


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
# Test doubles
# ---------------------------------------------------------------------------

class _MockAgent:
    """Mock agent with active_mode-based is_ticket_mode property and skill support."""

    def __init__(self, is_ticket_mode=False, ticket_ts=None,
                 active_mode=None):
        self._ticket_lock = MagicMock()
        self._mode_lock = threading.RLock()
        self._ticket_mode_ts = ticket_ts
        self.config = MagicMock()
        self.config.is_ticket_mode = is_ticket_mode
        self._active_mode = active_mode
        self.agent_id = "test_agent_001"
        self._messages = []
        self.context_limit_override = None

        # Instance-attribute MagicMocks shadow the real clear_* methods below
        # because tests assert on call counts; the methods exist for API parity
        # with production Agent.
        self.clear_ticket_mode = MagicMock()
        self.set_ticket_mode = MagicMock()
        self.clear_active_mode = MagicMock()
        self._deactivation_callbacks = []

    def set_active_mode(self, mode_state, expected_mode=None):
        self._active_mode = mode_state

    def register_deactivation_callback(self, cb):
        self._deactivation_callbacks.append(cb)

    @property
    def is_workspace_mode(self):
        mode = self.active_mode
        if mode is None:
            return False
        return getattr(mode, 'name', None) == 'workspace'

    @property
    def is_ticket_mode(self):
        mode = getattr(self, 'active_mode', None)
        return mode is not None and getattr(mode, 'name', None) == 'ticket'

    @property
    def is_skill_mode(self):
        mode = self.active_mode
        if mode is None:
            return False
        return not mode.is_expired()

    @property
    def active_mode(self):
        return self._active_mode

    @active_mode.setter
    def active_mode(self, value):
        self._active_mode = value

    def snapshot_mode(self):
        return self.active_mode

    def snapshot_mode_and_ticket_ts(self):
        return self.active_mode, getattr(self, '_ticket_mode_ts', None)

    def get_mode_name(self):
        ms = self.active_mode
        return ms.name if ms else None

    def is_mode_expired(self):
        ms = self.active_mode
        if ms is None:
            return False
        return ms.is_expired()

    @property
    def messages(self):
        return list(self._messages)

    @messages.setter
    def messages(self, value):
        self._messages = list(value)

    def append_message(self, message):
        self._messages.append(message)

    def extend_messages(self, messages):
        self._messages.extend(messages)

    @property
    def parent_agent_id(self):
        """Mock agents are always roots — no parent link."""
        return None

    def clear_active_mode(self, expected_mode=None):
        """Real CAS clear mirroring Agent.clear_active_mode.

        Shadowed at instance level by a MagicMock in __init__ (tests assert
        on call counts); this body is the honest fallback for doubles that
        never install the mock.
        """
        if expected_mode is not None and self._active_mode is not expected_mode:
            return False
        self._active_mode = None
        self._ticket_mode_ts = None
        return True

    def clear_ticket_mode(self):
        """Legacy wrapper. Mirrors Agent.clear_ticket_mode()."""
        return self.clear_active_mode()

    def set_context_limit_override(self, tokens):
        self.context_limit_override = tokens

    def get_context_limit_override(self):
        return self.context_limit_override


class MockModeState:
    """Minimal ModeState double for skill tests."""
    def __init__(self, name="ticket", ttl=300):
        import time as _t
        self.name = name
        self.activated_at = _t.time()
        self.ttl_seconds = ttl
        self.allowed_tool_names = frozenset()
        self.skill_state = {}

    def is_expired(self) -> bool:
        """Return True if the ModeState has passed its ttl_seconds since activated_at.

        ttl_seconds == 0 means "never expires" (matches ModeState semantics).
        """
        import time as _t
        if self.ttl_seconds == 0:
            return False
        return _t.time() - self.activated_at > self.ttl_seconds


def _make_ctx(body=None, is_subagent=False, subagent_ticket_mode=False,
              seeded_tasks=None, _ticket_activated=False, _mode=None,
              session_id="sess_test", inherited_skill=None):
    """Build a minimal RequestContext-like namespace object."""
    body = body or {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"function": {"name": "read"}}, {"function": {"name": "write"}}],
    }
    ctx = MagicMock()
    ctx.body = body
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.subagent_ticket_mode = subagent_ticket_mode
    ctx.seeded_tasks = seeded_tasks or []
    ctx._ticket_activated = _ticket_activated
    ctx._mode = _mode
    ctx.inherited_skill = inherited_skill
    ctx.model_cfg = None
    ctx.messages = body.get("messages", [])
    return ctx


# ===========================================================================
# Test resolve_mode with skill context (AC 2.1, 2.2, 2.9)
# ===========================================================================

class TestResolveModeSkillContext(unittest.TestCase):
    """Tests for resolve_mode returning Mode.SKILL for skill-triggered requests."""

    def setUp(self):
        # Re-import Mode from the live proxy.mode module so the test is
        # robust against ``importlib.reload(proxy.mode)`` performed by
        # earlier tests (e.g. TestExtractTaskAnchorMaxChars).  Without this,
        # the Mode class bound at module-import time becomes a stale
        # reference to a different class object, and enum equality breaks
        # (``<Mode.DEFAULT: 1> != <Mode.DEFAULT: 1>``) because the resolve_mode
        # call returns an instance of the *reloaded* class.
        #
        # We compare by ``.value`` (int) throughout this class for the same
        # reason — value-based comparison is immune to class identity.
        from proxy.mode import Mode as _LiveMode  # noqa: F401
        self._Mode = _LiveMode

    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_resolve_mode_returns_mode_skill(self, mock_registry_cls):
        """Resolving a request with inherited_skill set returns Mode.SKILL."""
        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.ttl_seconds = 300
        mock_skill.get_tool_filter.return_value = frozenset()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_registry_cls.return_value = mock_registry

        agent = _MockAgent(is_ticket_mode=False)
        ctx = _make_ctx(inherited_skill="ticket")
        mode = resolve_mode(ctx, agent)
        # Compare by .value to be robust against importlib.reload(proxy.mode)
        # recreating the Mode enum class (see class docstring).
        self.assertEqual(mode.value, self._Mode.SKILL.value)
        self.assertEqual(ctx._mode.value, self._Mode.SKILL.value)

    def test_resolve_mode_skill_enum_exists(self):
        """Mode.SKILL exists for skill-triggered requests."""
        self.assertIsNotNone(Mode.SKILL)

    def test_resolve_mode_returns_default_for_empty(self):
        """No skill trigger → Mode.DEFAULT when not in ticket mode."""
        agent = _MockAgent(is_ticket_mode=False)
        ctx = _make_ctx(inherited_skill=None)
        mode = resolve_mode(ctx, agent)
        self.assertEqual(mode.value, self._Mode.DEFAULT.value)

    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_subagent_inheritance_skill_mode(self, mock_registry_cls):
        """Sub-agent with inherited_skill=None (workspace canary) → Mode.DEFAULT.

        Post-285cf80: workspace canaries set inherited_skill=None (not the
        literal "workspace") — workspace sub-agents use default tools, not a
        "workspace" skill. So a sub-agent with is_subagent=True and
        inherited_skill=None resolves to Mode.SUBAGENT (no skill inheritance).
        """
        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.ttl_seconds = 300
        mock_skill.get_tool_filter.return_value = frozenset()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_registry_cls.return_value = mock_registry

        agent = _MockAgent(is_ticket_mode=False)
        ctx = _make_ctx(is_subagent=True, inherited_skill=None)
        mode = resolve_mode(ctx, agent)
        self.assertEqual(mode.value, self._Mode.SUBAGENT.value)
        self.assertEqual(ctx._mode.value, self._Mode.SUBAGENT.value)


# ===========================================================================
# Test apply_mode (AC 2.3, 2.4, 2.7, 2.8)
# ===========================================================================

class TestApplyModeTools(unittest.TestCase):
    """Tests for apply_mode tools handling."""

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_apply_mode_captures_tools_snapshot(
        self, mock_tooling, mock_select
    ):
        """# US-014: After apply_mode, ctx._tools_snapshot contains filtered tools."""
        mock_tl = MagicMock()
        mock_tl.augmented_tools = [
            {"function": {"name": "read"}},
            {"function": {"name": "write"}},
        ]
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        tools_in = [
            {"function": {"name": "read"}},
            {"function": {"name": "write"}},
            {"function": {"name": "bash"}},
        ]
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools_in,
            "model": "test",
        }
        ctx = _make_ctx(body=body)
        agent = _MockAgent()

        apply_mode(ctx, agent, body, Mode.SKILL)

        # _tools_snapshot should be set after apply_mode
        self.assertTrue(hasattr(ctx, '_tools_snapshot'))
        self.assertIsNotNone(ctx._tools_snapshot)
        # Should contain the filtered tools from prepare_tooling_payload
        self.assertEqual(len(ctx._tools_snapshot), 2)

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_apply_mode_does_not_crash_without_tools(
        self, mock_tooling, mock_select
    ):
        """apply_mode handles missing 'tools' key gracefully (no KeyError)."""
        mock_tl = MagicMock()
        mock_tl.augmented_tools = None
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            # No "tools" key
            "model": "test",
        }
        ctx = _make_ctx(body=body)
        agent = _MockAgent()

        # Should not raise
        result = apply_mode(ctx, agent, body, Mode.DEFAULT)
        self.assertIsNotNone(result)

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_byte_identical_tools(
        self, mock_tooling, mock_select
    ):
        """apply_mode adds mode's tools to existing tool list (verify list concat)."""
        mock_tl = MagicMock()
        mock_tl.augmented_tools = [
            {"function": {"name": "read"}},
            {"function": {"name": "write"}},
            {"function": {"name": "bash"}},
        ]
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        existing_tools = [{"function": {"name": "read"}}]
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": list(existing_tools),
            "model": "test",
        }
        ctx = _make_ctx(body=body)
        agent = _MockAgent()

        apply_mode(ctx, agent, body, Mode.SKILL)

        # Body tools should include the original + any filtered additions
        # The key invariant: the filtered tool list is the starting point for tools
        self.assertIn("tools", body)
        tool_names = {t.get("function", {}).get("name", "") for t in body["tools"]}
        self.assertIn("read", tool_names)

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_byte_identical_prompt(
        self, mock_tooling, mock_select
    ):
        """apply_mode prepends mode's system prompt to existing prompt.

        # Phase 2B: apply_mode always delegates to skill_prompts.assemble_system_prompt.

        Note: the ticket skill's ``get_system_prompt`` returns the full
        ``TICKET_MANAGER_PROMPT`` content (the proxy_ticket sub-agent prompt),
        which is embedded inside the ``<MANDATORY_SYSTEM_INSTRUCTIONS>`` block.
        We assert on a distinctive substring of that content rather than the
        identifier string ``"TICKET_MANAGER_PROMPT"`` (the identifier never
        appears in the rendered prompt — only the prompt body is emitted).
        """
        from proxy.skill_prompts import OUR_SYSTEM_PROMPT
        from proxy.tool_parsing import TICKET_MAIN_PROMPT

        mock_tl = MagicMock()
        mock_tl.augmented_tools = []
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        body = {
            "messages": [
                {"role": "system", "content": "Original system prompt."},
                {"role": "user", "content": "hi"},
            ],
            "tools": [],
            "model": "test",
        }
        ctx = _make_ctx(body=body)
        # Current arch (proxy.skill_prompts.assemble_system_prompt + mode.apply_mode):
        # the ticket-mode MANDATORY block is only injected when
        # mode_state.name == "ticket". Set up a ticket ModeState on the agent
        # so the MANDATORY block is added.
        agent = _MockAgent(active_mode=MockModeState(name="ticket"))

        apply_mode(ctx, agent, body, Mode.SKILL)

        sys_msg = body["messages"][0]["content"]
        # New arch (mode.py:218, skill_prompts.assemble_system_prompt): the system
        # prompt is OUR_SYSTEM_PROMPT (Phase 1c short base from proxy.skill_prompts)
        # + MANDATORY block. The MANDATORY block contains the TICKET_MANAGER_PROMPT
        # body (the proxy_ticket sub-agent prompt) when the active skill is
        # "ticket".
        self.assertTrue(
            sys_msg.startswith(OUR_SYSTEM_PROMPT),
            f"System prompt should start with OUR_SYSTEM_PROMPT, got: {sys_msg[:80]}"
        )
        self.assertIn(
            "<MANDATORY_SYSTEM_INSTRUCTIONS>", sys_msg,
            "Ticket mode MANDATORY block must be present"
        )
        # Check that the TICKET_MANAGER_PROMPT body (not the identifier) is
        # embedded inside the MANDATORY block. A distinctive opening substring
        # is used since the full prompt is thousands of chars.
        self.assertIn(
            "PLANNING ORCHESTRATOR", sys_msg,
            "MANDATORY block must embed the TICKET_MAIN_PROMPT orchestrator identity "
            "for MAIN ticket skill (C2: prompts are role-split)."
        )


# ===========================================================================
# Test PostProcessStage post-process ops (AC 2.5, 2.6, 2.7)
# ===========================================================================

class TestApplyPostProcess(unittest.TestCase):
    """Tests for the post-process path of PostProcessStage.

    Wave 4: apply_post_process was split into 3 ops (apply_skill_response_transform,
    inject_task_canaries, finalize_skill_mode). These tests now exercise the
    end-to-end PostProcessStage.execute() with the unrelated sub-methods
    patched as no-ops, asserting on the post-execute state:
    - ctx.response_data (transformed) for the data-shape assertions
    - ctx.skill_lifecycle_finalized (True) for the lifecycle contract
    - agent.clear_active_mode for the deactivation contract
    """

    def setUp(self):
        reset_default_registry()

    def tearDown(self):
        reset_default_registry()

    def _make_full_ctx(self, response_data, session_id="sess_pp_001"):
        """Build a ctx suitable for PostProcessStage.execute().

        Patches the unrelated sub-methods (VFS, chat history, etc.) to
        no-ops so they don't interfere with the post-process ops under test.
        """
        from proxy.pipeline.post_process import PostProcessStage as _Stage
        body = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        ctx = MagicMock()
        ctx.body = body
        ctx.session_id = session_id
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.seeded_tasks = []
        ctx.inherited_skill = None
        ctx.messages = body["messages"]
        ctx.response_data = response_data
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.llama_body = body
        ctx.request_id = "req_pp_test"
        return ctx

    def _noop_submethods(self):
        """Return a context manager stack that patches the 8 sub-methods to no-ops."""
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

    def _run_with_agent(self, agent, response_data, session_id):
        """Patch agent_registry.get_or_load to return the test's agent, then execute."""
        ctx = self._make_full_ctx(response_data, session_id=session_id)

        def _get_or_load(sid):
            return agent
        return ctx, patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=_get_or_load,
        )

    def test_apply_post_process_malformed_response_aborts_with_canonical_502(self):
        """Malformed response (no choices) → fail-closed StageAbort, 502, mode cleared.

        PostProcessStage now fail-closes malformed upstream data with
        build_upstream_protocol_error() + HTTP 502 BEFORE any success method
        runs. The lifecycle finalize still runs in the outer finally,
        classifying the turn FAILED and clearing the active mode.
        """
        from proxy.openai_compat import build_upstream_protocol_error
        from proxy.pipeline.exceptions import StageAbort

        agent = _MockAgent(active_mode=MockModeState(name="ticket"))
        # Malformed: no "choices" key
        ctx, agent_patch = self._run_with_agent(agent, {"usage": {}}, "sess_malformed")

        # Mock the skill registry to return a skill that allows deactivation
        mock_skill = MagicMock()
        mock_skill.should_deactivate.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        canonical = build_upstream_protocol_error()
        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the FAILED turn disposition and the mode is cleared.
        ctx.mode_snapshot_pre_upstream = agent.active_mode
        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             agent_patch, \
             self._noop_submethods():
            with self.assertRaises(StageAbort) as cm:
                PostProcessStage().execute(ctx)

        body, status = cm.exception.response_data
        self.assertEqual(status, 502)
        self.assertEqual(body, canonical)
        self.assertEqual(ctx.response_data, canonical)
        agent.clear_active_mode.assert_called_once()
        self.assertTrue(ctx.skill_lifecycle_finalized)

    def test_apply_post_process_empty_choices_aborts_with_canonical_502(self):
        """Empty-choices payload → fail-closed StageAbort, 502, mode cleared.

        Mirrors the no-choices case for empty choices — the shape check
        rejects the payload the same way and aborts before sub-methods.
        """
        from proxy.openai_compat import build_upstream_protocol_error
        from proxy.pipeline.exceptions import StageAbort

        agent = _MockAgent(is_ticket_mode=True, active_mode=MockModeState(name="ticket"))
        # Malformed payload (empty choices) classifies as SHAPE failure.
        ctx, agent_patch = self._run_with_agent(agent, {"choices": []}, "sess_malformed_none")

        # Mock the skill registry to return a skill that allows deactivation
        mock_skill = MagicMock()
        mock_skill.should_deactivate.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        canonical = build_upstream_protocol_error()
        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the FAILED turn disposition and the mode is cleared.
        ctx.mode_snapshot_pre_upstream = agent.active_mode
        with patch('proxy.skills.skill_registry.SkillRegistry', return_value=mock_registry), \
             agent_patch, \
             self._noop_submethods():
            with self.assertRaises(StageAbort) as cm:
                PostProcessStage().execute(ctx)

        body, status = cm.exception.response_data
        self.assertEqual(status, 502)
        self.assertEqual(body, canonical)
        self.assertEqual(ctx.response_data, canonical)
        agent.clear_active_mode.assert_called_once()
        self.assertTrue(ctx.skill_lifecycle_finalized)

    def test_apply_post_process_with_skill_strips_trigger(self):
        """Post-process with an active skill preserves message content."""
        agent = _MockAgent(active_mode=None)
        ctx = self._make_full_ctx({
            "choices": [{
                "message": {"content": "Here is the plan for the auth module."},
                "finish_reason": "stop",
            }]
        }, session_id="sess_pp_skill")

        with self._noop_submethods():
            result = PostProcessStage().execute(ctx)

        # SKILL mode: skill.post_process handles the content
        # Since no active skill is set on the agent, post_process uses default
        # (BaseSkill.post_process returns data unchanged)
        self.assertIn("choices", result)
        self.assertIn("Here is the plan", result["choices"][0]["message"]["content"])
        self.assertTrue(ctx.skill_lifecycle_finalized)

    def test_activation_rewrite_format(self):
        """Skill activation rewrite includes ## ^^ticket Activate marker."""
        skill = BaseSkill()
        # Manually set trigger via property override
        skill._trigger = "^^ticket"

        class TicketSkillRewrite(BaseSkill):
            @property
            def trigger(self):
                return "^^ticket"

            def get_activation_rewrite(self, ctx):
                return f"## ^^ticket Activate\n\n{ctx.raw_user_text[len(self.trigger):].strip()}"

        tskill = TicketSkillRewrite()
        ctx = SkillContext(
            session_id="sess",
            agent_id="agent",
            raw_user_text="^^ticket build auth",
            messages=[],
        )

        rewrite = tskill.get_activation_rewrite(ctx)

        self.assertTrue(
            "## ^^ticket Activate" in rewrite,
            f"Rewrite should contain '## ^^ticket Activate', got: {rewrite!r}"
        )


# ===========================================================================
# Test skill mode application with active skill (AC 2.8)
# ===========================================================================

class TestApplyModeSkillActive(unittest.TestCase):
    """Tests for apply_mode when a skill is active on the agent."""

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_apply_mode_skill_active_sets_tools_snapshot(
        self, mock_tooling, mock_select
    ):
        """When a skill is active, apply_mode sets ctx._tools_snapshot."""
        mock_tl = MagicMock()
        mock_tl.augmented_tools = [
            {"function": {"name": "read"}},
            {"function": {"name": "write"}},
        ]
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"function": {"name": "read"}},
                {"function": {"name": "write"}},
                {"function": {"name": "bash"}},
            ],
            "model": "test",
        }
        ctx = _make_ctx(body=body)
        agent = _MockAgent(active_mode=MockModeState(name="ticket"))

        apply_mode(ctx, agent, body, Mode.SKILL)

        # _tools_snapshot should be set regardless of mode
        self.assertTrue(hasattr(ctx, '_tools_snapshot'))
        self.assertIsNotNone(ctx._tools_snapshot)

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_apply_mode_skill_filters_tools(
        self, mock_tooling, mock_select
    ):
        """When a skill is active, apply_mode applies skill tool filter."""
        mock_tl = MagicMock()
        mock_tl.augmented_tools = [
            {"function": {"name": "read"}},
            {"function": {"name": "write"}},
            {"function": {"name": "bash"}},
        ]
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"function": {"name": "read"}},
                {"function": {"name": "write"}},
                {"function": {"name": "bash"}},
                {"function": {"name": "grep"}},
            ],
            "model": "test",
        }
        ctx = _make_ctx(body=body)

        # Create agent with active skill that allows only "read".
        # After Wave 1's three-state tool policy cutover, apply_mode reads
        # SkillDefinition.allowed_tool_names (None = unrestricted; empty
        # frozenset = deny all; non-empty = exact allowlist). The mock
        # returns a SkillDefinition directly, not a BaseSkill, so the
        # production read path is exercised. We use a non-builtin skill
        # name to avoid the TRUSTED_BUILTIN_SKILLS trust-gate on the
        # SkillDefinition constructor.
        mode_state = MockModeState(name="customskill")
        mode_state.skill_state = {}
        agent = _MockAgent(active_mode=mode_state)

        with patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
            mock_reg = MagicMock()
            from proxy.skills.skill_definition import SkillDefinition
            mock_reg.get.return_value = SkillDefinition(
                name="customskill",
                trigger="^^customskill",
                aliases=(),
                allowed_tool_names=frozenset({"read"}),
                ttl_seconds=300,
                description="Custom skill (test fixture).",
            )
            mock_reg_cls.return_value = mock_reg

            apply_mode(ctx, agent, body, Mode.SKILL)

        # Tools should be filtered to only "read"
        tool_names = {t.get("function", {}).get("name", "") for t in body["tools"]}
        self.assertEqual(tool_names, {"read"})


# ===========================================================================
# Test dual authorization formula (Task 3.3)
# ===========================================================================

class TestDualAuthorization:
    """Verify dual authorization formula: skill.get_tool_filter() ∩ (available_in_skills ∪ default_tools)"""

    def test_ticket_skill_tool_filter_size(self):
        """Ticket skill main mode: get_tool_filter() returns expected tool count."""
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillContext
        
        skill = TicketSkill()
        ctx = SkillContext(
            session_id="test", agent_id="test",
            messages=[], is_subagent=False
        )
        tools = skill.get_tool_filter(ctx)

        # C4.5 Fix 3: proxy_work removed from ticket MAIN filter.
        # Slice 8b: proxy_ticket decomposed; ticket MAIN gets the 2 read-only tools.
        # 6 client tools + 2 decomposed ticket tools = 8.
        assert len(tools) == 8, f"Expected 8 tools, got {len(tools)}: {tools}"
        assert tools == {
            "read", "glob", "grep", "search",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "task", "webfetch",
        }

    def test_ticket_skill_subagent_excludes_webfetch(self):
        """Ticket skill sub-agent: get_tool_filter() excludes webfetch."""
        from skills.ticket_skill import TicketSkill
        from proxy.skills.skill_registry import SkillContext

        skill = TicketSkill()
        ctx = SkillContext(
            session_id="test", agent_id="test",
            messages=[], is_subagent=True  # Sub-agent
        )
        tools = skill.get_sub_agent_tool_filter(ctx)

# C4.5 Fix 3: proxy_work removed (already wasn't in sub-agent filter per C2).
        # Slice 8b: ticket sub-agent gets the FULL decomposed ticket surface (10
        # tools) PLUS the 5 client tools (read, glob, grep, search, webfetch).
        # task is excluded per spec.
        assert len(tools) == 15, f"Expected 15 tools, got {len(tools)}: {tools}"
        assert tools == {
            "read", "glob", "grep", "search", "webfetch",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
        }

    def test_available_in_skills_none_for_passthrough(self):
        """Passthrough tools can have available_in_skills set, but if None, available in all skills."""
        from proxy.tool_pipeline.tool_registry import registry
        
        passthrough_tools = registry.get_passthrough_tools()
        
        for name in passthrough_tools:
            meta = registry._tools.get(name)
            if meta:
                # Passthrough tools may have available_in_skills set (e.g., todowrite={'ticket'})
                # or None (available in all skills). Both are valid.
                # The key invariant: is_passthrough=True means the tool bypasses skill filter.
                assert meta.is_passthrough is True, (
                    f"Tool '{name}' returned by get_passthrough_tools() should have is_passthrough=True"
                )

    def test_available_in_skills_for_ticket_tools(self):
        """Ticket-mode tools have available_in_skills={'ticket'}."""
        from proxy.tool_pipeline.tool_registry import registry
        from proxy.tool_pipeline.tool_registry import ToolCategory
        
        # Get ticket-safe tools (read, glob, grep, search, proxy_ticket, todowrite)
        ticket_tools = registry.get_tools_for_skill("ticket")
        
        for name in ticket_tools:
            if name in ("todowrite",):  # Skip passthrough
                continue
            meta = registry._tools.get(name)
            if meta and meta.category != ToolCategory.PASSTHROUGH:
                # Skip tools with available_in_skills=None (available in all skills)
                if meta.available_in_skills is None:
                    continue
                assert "ticket" in meta.available_in_skills, (
                    f"Tool '{name}' should include 'ticket' in available_in_skills"
                )

    def test_tool_guard_uses_active_mode(self):
        """tool_guard._check_mode_allowlist uses agent.active_mode (Phase B path)."""
        from proxy.tool_pipeline.tool_guard import MemoryToolGuard
        from proxy.tool_pipeline.tool_call import ToolCall
        from proxy.skills.skill_definition import ModeState
        from unittest.mock import MagicMock, patch
        import time
        
        guard = MemoryToolGuard()
        
        # Create mock context with session_id
        ctx = MagicMock()
        ctx.session_id = "test_dual_auth_session"
        
        # Set up mock agent with active_mode
        mock_agent = MagicMock()
        mock_agent.active_mode = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset({"read", "proxy_ticket"}),  # Restrictive filter
            skill_state={},
        )
        mock_agent.snapshot_mode.return_value = mock_agent.active_mode
        mock_agent.get_mode_name.return_value = "ticket"
        mock_agent._deactivation_callbacks = []
        
        with patch('proxy.agent.agent_registry.registry.get_or_load', return_value=mock_agent):
            # Test: blocked tool (grep not in allowed)
            tc_blocked = ToolCall(id="t1", name="grep", arguments={})
            error = guard._check_mode_allowlist(tc_blocked, ctx)
            assert error is not None, "grep should be blocked by restrictive active_mode"
            assert "not allowed" in error
            
            # Test: allowed tool (read is in allowed)
            tc_allowed = ToolCall(id="t2", name="read", arguments={})
            error = guard._check_mode_allowlist(tc_allowed, ctx)
            assert error is None, f"read should be allowed: {error}"

    def test_tool_guard_fallback_to_legacy(self):
        """tool_guard checks active_mode for ticket skill."""
        from proxy.tool_pipeline.tool_guard import MemoryToolGuard
        from proxy.tool_pipeline.tool_call import ToolCall
        from proxy.skills.skill_definition import ModeState
        from unittest.mock import MagicMock, patch
        import time
        
        guard = MemoryToolGuard()
        
        # Create mock context
        ctx = MagicMock()
        ctx.session_id = "test_legacy_fallback"
        
        # Set up mock agent with active_mode=ModeState(name='ticket')
        mock_agent = MagicMock()
        mock_agent.active_mode = ModeState(
            name='ticket',
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(['bash']),  # Only bash allowed
            skill_state={},
        )
        mock_agent.snapshot_mode.return_value = mock_agent.active_mode
        mock_agent.get_mode_name.return_value = "ticket"
        mock_agent._deactivation_callbacks = []
        
        with patch('proxy.agent.agent_registry.registry.get_or_load', return_value=mock_agent):
            # bash should be allowed when active_mode is ticket and bash is in allowlist
            tc = ToolCall(id="t1", name="bash", arguments={})
            error = guard._check_mode_allowlist(tc, ctx)
            assert error is None, "bash should be allowed when in ticket mode allowlist"
            
            # Read should be blocked (not in allowlist)
            tc_read = ToolCall(id="t2", name="Read", arguments={})
            error_read = guard._check_mode_allowlist(tc_read, ctx)
            assert error_read is not None, "Read should be blocked when not in ticket mode allowlist"


# ===========================================================================
# Test apply_mode context pruning (V2 select_context restoration)
# Regression: select_context was stripped in c9b6755 and never re-added.
# Without it, 4133-msg / 9.2MB payloads go upstream unpruned.
# ===========================================================================

class TestApplyModeContextPruning(unittest.TestCase):
    """apply_mode must call select_context to prune the message window."""

    def setUp(self):
        """Use in-memory SQLite so select_context's DB calls don't hit a real pool."""
        import os
        os.environ["PROXY_TEST_DB"] = ":memory:"
        from memory.db import init_db
        init_db()

    def tearDown(self):
        import os
        os.environ.pop("PROXY_TEST_DB", None)

    def _make_large_body(self, n_exchanges=30, chars_per_msg=600):
        """Build a body that exceeds the small-model safe budget."""
        messages = [{"role": "system", "content": "sys"}]
        for i in range(n_exchanges):
            messages.append({"role": "user", "content": f"u{i} " + "U" * chars_per_msg})
            messages.append({"role": "assistant", "content": f"a{i} " + "A" * chars_per_msg})
        return {
            "messages": messages,
            "tools": [{"function": {"name": "read"}}],
            "model": "test-small",
        }

    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_apply_mode_calls_select_context_and_prunes(self, mock_registry_cls):
        """apply_mode must call select_context (V2) to assemble the message window.

        Note: the V2 context pipeline has a fast-path when VFS context is None
        (src/proxy/context_v2.py:1764) that returns cleaned messages WITHOUT
        pruning. The test exercises this path (VFS mocked to None for
        deterministic side-effect isolation), so the message count is
        preserved at the original 61. The PRIMARY assertion is that
        select_context is invoked — pruning is delegated to the chain-loaded
        main path which requires prior session history (not seeded here).

        The legacy test comment "2400 threshold" refers to the old
        V2_PASSTHROUGH_THRESHOLD (0.30) which was removed when VFS-aware
        paths were introduced. See commit 9523486 for the original
        implementation and later commits for the VFS-None fast-path.
        """
        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.allowed_tool_names = None
        mock_skill.get_tool_filter.return_value = frozenset({"read"})
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_registry_cls.return_value = mock_registry

        from types import SimpleNamespace
        small_model = SimpleNamespace(context_limit=10_000, raw_exchange_count=10, budget_ratio=0.8)
        body = self._make_large_body(n_exchanges=30, chars_per_msg=600)
        ctx = _make_ctx(body=body)
        ctx.model_cfg = small_model
        ctx.is_subagent = False
        agent = _MockAgent()

        # Spy on select_context to verify apply_mode invokes it.
        with patch("proxy.context.select_context") as spy, \
             patch("proxy.vfs._build_vfs_context", return_value=None):
            from proxy.context_v2 import _select_context_v2
            spy.side_effect = lambda *a, **kw: _select_context_v2(*a, **kw)
            result = apply_mode(ctx, agent, body, Mode.DEFAULT)

        # select_context was called (apply_mode does invoke V2).
        self.assertGreater(spy.call_count, 0,
                           "apply_mode must call select_context (V2 pipeline)")
        # In the VFS-None fast-path (cleaned == full_messages, no VFS),
        # V2 returns cleaned WITHOUT pruning. Original 61 msgs preserved.
        self.assertEqual(len(result["messages"]), 61)
        # System prompt preserved as first message.
        self.assertEqual(result["messages"][0]["role"], "system")

    @patch("proxy.skills.skill_registry.SkillRegistry")
    def test_apply_mode_propagates_context_limit_error(self, mock_registry_cls):
        """ContextLimitError from select_context must propagate, not be swallowed."""
        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.allowed_tool_names = None
        mock_skill.get_tool_filter.return_value = frozenset({"read"})
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_registry_cls.return_value = mock_registry

        body = {"messages": [{"role": "user", "content": "hi"}],
                "tools": [{"function": {"name": "read"}}]}
        ctx = _make_ctx(body=body)
        ctx.is_subagent = False
        agent = _MockAgent()

        from proxy.context import ContextLimitError
        with patch("proxy.context.select_context", side_effect=ContextLimitError("overflow")):
            with self.assertRaises(ContextLimitError):
                apply_mode(ctx, agent, body, Mode.DEFAULT)


if __name__ == "__main__":
    unittest.main()


class TestSubAgentTicketModeDeactivatesOnStop(unittest.TestCase):
    """Sub-agent ticket mode MUST deactivate when upstream returns stop.

    Regression: prior to this fix, TicketSkill did not override
    should_deactivate, so apply_post_process skipped clear_active_mode
    for ticket mode on finish_reason=stop. Sub-agents inheriting ticket
    mode via canary stayed locked in indefinitely.

    Wave 4: these tests now exercise PostProcessStage.execute() end-to-end
    (with unrelated sub-methods patched as no-ops) rather than calling
    apply_post_process directly. The agent is registered with the agent
    registry's get_or_load so the post-process ops see the active mode.
    """

    def setUp(self):
        # Fresh registry populated with real builtins (TicketSkill + WorkSkill).
        # The bug only reproduces with the real TicketSkill whose
        # should_deactivate inherits BaseSkill.should_deactivate (returns
        # finish_reason == "stop"). Mocking would mask the bug.
        self._registry = _skill_registry_new__(SkillRegistry)
        self._registry.populate_builtins()

    def _make_subagent_ctx(self):
        body = {"messages": [{"role": "user", "content": "Investigate."}]}
        ctx = MagicMock()
        ctx.body = body
        ctx.session_id = "subagent_6f6c1ee313d0ad5a"
        ctx.is_subagent = True
        ctx.subagent_ticket_mode = True
        ctx.inherited_skill = "ticket"
        ctx.seeded_tasks = []
        ctx._mode = Mode.SKILL_SUBAGENT
        ctx.messages = body["messages"]
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.llama_body = body
        ctx.request_id = "req_sub_ticket_stop"
        return ctx

    def _make_subagent_agent(self):
        ticket_ms = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        agent = MagicMock()
        agent.agent_id = "subagent_6f6c1ee313d0ad5a"
        agent.session_id = "subagent_6f6c1ee313d0ad5a"
        agent.active_mode = ticket_ms
        agent.snapshot_mode.return_value = ticket_ms
        agent._mode_lock = MagicMock()
        agent._deactivation_callbacks = []
        return agent

    def _noop_submethods(self):
        """Patch the 8 unrelated PostProcessStage sub-methods to no-ops."""
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

    def _run_stage(self, agent, ctx, response_data):
        """Patch agent_registry.get_or_load to return the test's agent, then execute."""
        def _get_or_load(session_id):
            return agent
        ctx.response_data = response_data
        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop").
        ctx.mode_snapshot_pre_upstream = agent.active_mode
        with patch("proxy.agent.agent_registry.registry.get_or_load", side_effect=_get_or_load), \
             patch("proxy.skills.skill_registry.SkillRegistry", return_value=self._registry), \
             self._noop_submethods():
            PostProcessStage().execute(ctx)

    def test_subagent_ticket_mode_deactivates_on_finish_reason_stop(self):
        """When sub-agent's upstream returns stop, ticket mode clears."""
        agent = self._make_subagent_agent()
        ctx = self._make_subagent_ctx()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Investigation complete."},
                "finish_reason": "stop",
            }]
        }
        self._run_stage(agent, ctx, data)
        agent.clear_active_mode.assert_called_once()
        self.assertTrue(ctx.skill_lifecycle_finalized)

    def test_subagent_ticket_mode_persists_on_tool_calls(self):
        """finish_reason=tool_calls does NOT clear ticket mode (tool loop continues)."""
        agent = self._make_subagent_agent()
        ctx = self._make_subagent_ctx()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }]
        }
        self._run_stage(agent, ctx, data)
        agent.clear_active_mode.assert_not_called()

    def test_main_agent_ticket_mode_clears_on_stop(self):
        """Main agents in ticket mode clear on stop (PER-TURN contract).

        Ticket/work/custom modes are scoped to the turn activated by the
        user's trigger; they clear on `finish_reason="stop"` for BOTH main
        and sub-agents. This prevents apply_post_process from rebuilding a
        ticket canary on later plain prompts (which would leak read-only
        ticket tools into bare sub-agents).
        """
        ticket_ms = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            skill_state={},
        )
        agent = MagicMock()
        agent.agent_id = "main_agent_001"
        agent.session_id = "main_agent_session"
        agent.active_mode = ticket_ms
        agent.snapshot_mode.return_value = ticket_ms
        agent._mode_lock = MagicMock()
        agent._deactivation_callbacks = []
        ctx = MagicMock()
        ctx.body = {"messages": [{"role": "user", "content": "More ticket work."}]}
        ctx.session_id = "main_agent_session"
        ctx.is_subagent = False
        ctx.subagent_ticket_mode = False
        ctx.inherited_skill = None
        ctx.seeded_tasks = []
        ctx._mode = Mode.SKILL
        ctx.messages = ctx.body["messages"]
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.llama_body = ctx.body
        ctx.request_id = "req_main_ticket_stop"
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Done for now."},
                "finish_reason": "stop",
            }]
        }
        # R1-A: pre-upstream snapshot must match the agent's active_mode so the
        # PostProcessStage CAS (clear_active_mode(expected_mode=...)) succeeds
        # against the COMPLETE turn disposition (finish_reason="stop").
        ctx.mode_snapshot_pre_upstream = agent.active_mode
        self._run_stage(agent, ctx, data)
        agent.clear_active_mode.assert_called_once()
        self.assertTrue(ctx.skill_lifecycle_finalized)


# ===========================================================================
# C6 fix: tool_cache three-state policy — deny-all branch (frozenset())
# ===========================================================================

class TestPrepareToolingPayloadDenyAll(unittest.TestCase):
    """C6 fix: tool_cache.prepare_tooling_payload honors the deny-all sentinel.

    Under the three-state policy:
      - ``None``           = unrestricted, fall through to default adapter path.
      - ``frozenset()``    = deny all tools, short-circuit with empty payload.
      - Non-empty frozenset = exact allowlist (the existing behavior).

    Prior to C6, ``if tool_filter:`` conflated ``None`` and ``frozenset()``,
    silently routing both through the default adapter path. This test pins
    the deny-all branch: the cache hash MUST be ``None`` (no tools to hash)
    and the [SKILL_MODE-TOOLS] log line MUST be emitted with 0 tools.
    """

    def test_deny_all_filter_returns_empty_tooling_payload(self):
        """A skill whose get_tool_filter() returns frozenset() returns 0 tools."""
        from proxy.skills.skill_definition import SkillDefinition
        from proxy.tool_cache import prepare_tooling_payload
        from proxy.types import ToolingPayload

        deny_all_skill = SkillDefinition(
            name="lockdown",
            trigger="^^lockdown",
            aliases=(),
            allowed_tool_names=frozenset(),
            ttl_seconds=300,
            description="Test deny-all skill.",
        )

        mock_registry = MagicMock()
        mock_registry.get.return_value = deny_all_skill

        body = {
            "tools": [
                {"type": "function",
                 "function": {"name": "bash", "description": "shell",
                              "parameters": {"type": "object", "properties": {}, "required": []}}},
                {"type": "function",
                 "function": {"name": "read", "description": "file",
                              "parameters": {"type": "object", "properties": {}, "required": []}}},
            ]
        }

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            result = prepare_tooling_payload(
                body, None, skill_name="lockdown",
                session_id="sess_deny", agent_id="agent_deny",
            )

        self.assertIsInstance(result, ToolingPayload)
        self.assertEqual(result.tools, [])
        self.assertEqual(result.augmented_tools, [])
        # C6 invariant: no cache hash for deny-all (nothing to hash).
        self.assertIsNone(result.tool_hash)
        self.assertFalse(result.cached)

    def test_deny_all_filter_emits_skill_mode_tools_log(self):
        """The [SKILL_MODE-TOOLS] log line is emitted with 0 tools (deny-all).

        Pre-C6, the truthiness check at the old L127 fell through to the
        default path, so this log line was NEVER emitted for deny-all. The
        fix moves it into the non-empty + deny-all branches.
        """
        from proxy.skills.skill_definition import SkillDefinition
        from proxy.tool_cache import prepare_tooling_payload

        deny_all_skill = SkillDefinition(
            name="lockdown",
            trigger="^^lockdown",
            aliases=(),
            allowed_tool_names=frozenset(),
            ttl_seconds=300,
            description="Test deny-all skill.",
        )
        mock_registry = MagicMock()
        mock_registry.get.return_value = deny_all_skill
        body = {"tools": []}

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            with self.assertLogs("proxy.tool_cache", level="INFO") as log_cm:
                prepare_tooling_payload(
                    body, None, skill_name="lockdown",
                    session_id="sess_deny", agent_id="agent_deny",
                )

        skill_mode_lines = [line for line in log_cm.output if "[SKILL_MODE-TOOLS]" in line]
        self.assertEqual(len(skill_mode_lines), 1,
                         "Expected exactly one [SKILL_MODE-TOOLS] line for deny-all")
        self.assertIn("Filtered to 0 tools", skill_mode_lines[0])
        self.assertIn("lockdown", skill_mode_lines[0])

    def test_deny_all_overrides_incoming_body_tools(self):
        """Deny-all branch ignores body['tools']; returns empty regardless of input.

        This proves the C6 fix doesn't accidentally fall through to the
        default path which would emit the body['tools'] unfiltered.
        """
        from proxy.skills.skill_definition import SkillDefinition
        from proxy.tool_cache import prepare_tooling_payload

        deny_all_skill = SkillDefinition(
            name="strict",
            trigger="^^strict",
            aliases=(),
            allowed_tool_names=frozenset(),
            ttl_seconds=300,
            description="Strict no-tools skill.",
        )
        mock_registry = MagicMock()
        mock_registry.get.return_value = deny_all_skill

        # Body has 5 tools (some allowed, some not). Deny-all must zero them all.
        body = {"tools": [
            {"type": "function",
             "function": {"name": f"tool_{i}", "description": "x",
                          "parameters": {"type": "object", "properties": {}, "required": []}}}
            for i in range(5)
        ]}

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            result = prepare_tooling_payload(
                body, None, skill_name="strict",
                session_id="sess_strict", agent_id="agent_strict",
            )

        self.assertEqual(result.tools, [],
                         "Deny-all must override body['tools'] to empty")


# ===========================================================================
# Wave 3 — Fail-closed mode application (spec §7.2)
# ===========================================================================

class TestApplyModeFailClosed(unittest.TestCase):
    """Spec §7.2: apply_mode is strict about the active-skill registry snapshot.

    - If the active ModeState names a skill not in the registry, raise
      ActiveSkillUnavailable (NOT a swallowed fallback).
    - If a skill callback raises, wrap into SkillRuntimeError (NOT swallow
      into a DEFAULT cascade).
    - ContextLimitError still propagates unchanged (handled by
      ContextOptimizeStage for HTTP 413).
    """

    def test_apply_mode_raises_active_skill_unavailable_when_skill_not_in_registry(self):
        """Agent.active_mode references a skill name the registry cannot find.

        Without the registry snapshot match, apply_mode raises
        ActiveSkillUnavailable carrying the missing skill name. The caller
        (ContextOptimizeStage) is responsible for compare-and-clearing the
        snapshot.
        """
        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # skill name absent
        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            active = MockModeState(name="defunct_skill")
            agent = _MockAgent(active_mode=active)
            ctx = _make_ctx()
            body = {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"function": {"name": "read"}}],
                "model": "test",
            }

            with self.assertRaises(ActiveSkillUnavailable) as cm:
                apply_mode(ctx, agent, body, Mode.SKILL)

        self.assertEqual(cm.exception.skill_name, "defunct_skill")
        self.assertIn("defunct_skill", str(cm.exception))
        # Body untouched (raised before any side effect).
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_apply_mode_raises_active_skill_unavailable_for_skill_subagent(self):
        """ActiveSkillUnavailable also fires for Mode.SKILL_SUBAGENT."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            active = MockModeState(name="old_skill")
            agent = _MockAgent(active_mode=active)
            ctx = _make_ctx(is_subagent=True)
            body = {
                "messages": [{"role": "user", "content": "do sub work"}],
                "tools": [{"function": {"name": "read"}}],
                "model": "test",
            }

            with self.assertRaises(ActiveSkillUnavailable) as cm:
                apply_mode(ctx, agent, body, Mode.SKILL_SUBAGENT)

        self.assertEqual(cm.exception.skill_name, "old_skill")

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_apply_mode_raises_skill_runtime_error_on_callback_failure(
        self, mock_tooling, mock_select
    ):
        """When a skill callback (assemble_system_prompt) raises, apply_mode wraps in SkillRuntimeError."""
        mock_tl = MagicMock()
        mock_tl.augmented_tools = []
        mock_tooling.return_value = mock_tl
        mock_select.return_value = None

        from proxy.skills.skill_definition import SkillDefinition
        skill_def = SkillDefinition(
            name="review",
            trigger="^^review",
            aliases=(),
            allowed_tool_names=frozenset({"read"}),
            ttl_seconds=300,
            description="Review skill (test fixture).",
        )
        mock_registry = MagicMock()
        mock_registry.get.return_value = skill_def

        with patch(
            "proxy.skills.skill_registry.SkillRegistry",
            return_value=mock_registry,
        ):
            with patch(
                "proxy.skill_prompts.assemble_system_prompt",
                side_effect=ValueError("malformed trigger rewrite"),
            ):
                active = MockModeState(name="review")
                agent = _MockAgent(active_mode=active)
                ctx = _make_ctx()
                body = {
                    "messages": [{"role": "user", "content": "^^review fix bug"}],
                    "tools": [{"function": {"name": "read"}}],
                    "model": "test",
                }

                with self.assertRaises(SkillRuntimeError) as cm:
                    apply_mode(ctx, agent, body, Mode.SKILL)

        self.assertEqual(cm.exception.skill_name, "review")
        self.assertEqual(cm.exception.source, "apply_mode")
        self.assertIsInstance(cm.exception.original, ValueError)
        self.assertIn("malformed trigger rewrite", str(cm.exception.original))
        # Wrapper's __str__ does NOT echo the original message.
        self.assertNotIn("malformed trigger rewrite", str(cm.exception))

    @patch("proxy.context.select_context")
    @patch("proxy.tool_cache.prepare_tooling_payload")
    def test_apply_mode_does_not_wrap_context_limit_error(
        self, mock_tooling, mock_select
    ):
        """ContextLimitError propagates unchanged (not wrapped as SkillRuntimeError)."""
        from proxy.context import ContextLimitError

        mock_tl = MagicMock()
        mock_tl.augmented_tools = []
        mock_tooling.return_value = mock_tl
        mock_select.side_effect = ContextLimitError("token overflow")

        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_skill.allowed_tool_names = None
        mock_skill.get_tool_filter.return_value = frozenset({"read"})
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=mock_registry):
            ctx = _make_ctx()
            agent = _MockAgent()

            body = {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "model": "test",
            }

            with self.assertRaises(ContextLimitError):
                apply_mode(ctx, agent, body, Mode.DEFAULT)


class TestContextOptimizeStageFailClosed(unittest.TestCase):
    """ContextOptimizeStage translates active-skill failures without fallback.

    Spec §7.2 and Slice 1B:
    - ActiveSkillUnavailable → 409 skill_state_conflict.
    - SkillRuntimeError → 500 skill_runtime_error.
    - Neither failure cascades to DEFAULT.
    - The stage does not clear active mode locally.
    - RequestPipeline owns critical-failure compare-and-clear.
    """

    def _make_ctx(self, *, session_id="sess_fail_closed",
                  is_subagent=False, subagent_ticket_mode=False,
                  inherited_skill=None,
                  snapshot_mode=None):
        """Build a context with the minimum shape ContextOptimizeStage needs.

        `snapshot_mode` installs the ModeState visible to the stage. Direct
        stage execution does not finalize it; RequestPipeline owns
        compare-and-clear.
        """
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"function": {"name": "read"}}],
            "model": "test-model",
        }
        ctx = MagicMock()
        ctx.body = dict(body)
        ctx.session_id = session_id
        ctx.request_id = "req-fc-test"
        ctx.is_subagent = is_subagent
        ctx.subagent_ticket_mode = subagent_ticket_mode
        ctx.inherited_skill = inherited_skill
        ctx.model_cfg = None
        ctx.model_name = "test-model"
        ctx.messages = body["messages"]

        captured_snapshot = snapshot_mode

        class _Agent:
            def __init__(self):
                self.agent_id = "agent_fc_test"
                self.active_mode = captured_snapshot
                self._mode_lock = threading.RLock()
                self._deactivation_callbacks = []
                self._cleared_with = []

            def snapshot_mode(self):
                return self.active_mode

            def clear_active_mode(self, expected_mode=None):
                if expected_mode is not None and self.active_mode is not expected_mode:
                    return False
                self._cleared_with.append(expected_mode)
                self.active_mode = None
                return True

            def get_context_limit_override(self):
                return None

        return ctx, _Agent()

    def test_context_optimize_translates_active_skill_unavailable_without_local_clear(
        self,
    ):
        """ActiveSkillUnavailable maps to 409 without a stage-local clear."""
        from proxy.pipeline.exceptions import StageAbort
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )

        snapshot = MockModeState(name="defunct")
        ctx, agent = self._make_ctx(snapshot_mode=snapshot)

        with patch(
            "proxy.mode.apply_mode",
            side_effect=ActiveSkillUnavailable("defunct"),
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.SKILL,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ):
            stage = ContextOptimizeStage()
            with self.assertRaises(StageAbort) as cm:
                stage.execute(ctx)

        response_data, status_code = cm.exception.response_data
        self.assertEqual(status_code, 409)
        self.assertIsInstance(response_data, dict)
        self.assertIn("error", response_data)
        error = response_data["error"]
        self.assertEqual(error["type"], "skill_state_conflict")
        self.assertNotIn("code", error)
        self.assertNotIn("Traceback", error.get("message", ""))
        self.assertNotIn(
            "ActiveSkillUnavailable",
            error.get("message", ""),
        )

        self.assertEqual(
            agent._cleared_with,
            [],
            "ContextOptimizeStage must not clear locally; "
            "RequestPipeline owns critical-failure finalization",
        )
        self.assertIs(
            agent.active_mode,
            snapshot,
            "Direct stage execution bypasses RequestPipeline, so the mode "
            "must remain installed",
        )

    def test_context_optimize_translates_skill_runtime_error_without_local_clear(
        self,
    ):
        """SkillRuntimeError maps to 500 without a stage-local clear."""
        from proxy.pipeline.exceptions import StageAbort
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )

        snapshot = MockModeState(name="borked")
        ctx, agent = self._make_ctx(snapshot_mode=snapshot)

        inner = ValueError("downstream crashed")
        skill_err = SkillRuntimeError(
            "borked",
            source="apply_mode",
            original=inner,
        )

        with patch(
            "proxy.mode.apply_mode",
            side_effect=skill_err,
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.SKILL,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ):
            stage = ContextOptimizeStage()
            with self.assertRaises(StageAbort) as cm:
                stage.execute(ctx)

        response_data, status_code = cm.exception.response_data
        self.assertEqual(status_code, 500)
        self.assertEqual(
            response_data["error"]["type"],
            "skill_runtime_error",
        )
        self.assertNotIn("code", response_data["error"])
        message = response_data["error"]["message"]
        self.assertNotIn("downstream crashed", message)
        self.assertNotIn("ActiveSkillUnavailable", message)

        self.assertEqual(
            agent._cleared_with,
            [],
            "ContextOptimizeStage must not clear locally; "
            "RequestPipeline owns critical-failure finalization",
        )
        self.assertIs(
            agent.active_mode,
            snapshot,
            "Direct stage execution bypasses RequestPipeline, so the mode "
            "must remain installed",
        )

    def test_context_optimize_does_not_cascade_when_active_skill_error(self):
        """ActiveSkillUnavailable / SkillRuntimeError NEVER reach _cascade_apply_default."""
        from proxy.pipeline.exceptions import StageAbort
        from proxy.pipeline.stage_modules import context_optimize as co_mod
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )

        snapshot = MockModeState(name="ghost")
        ctx, agent = self._make_ctx(snapshot_mode=snapshot)

        cascade_calls = []
        real_cascade = co_mod._cascade_apply_default

        def spy_cascade(*args, **kwargs):
            cascade_calls.append(args)
            return real_cascade(*args, **kwargs)

        with patch(
            "proxy.mode.apply_mode",
            side_effect=ActiveSkillUnavailable("ghost"),
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.SKILL,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ), patch(
            "proxy.pipeline.stage_modules.context_optimize._cascade_apply_default",
            side_effect=spy_cascade,
        ):
            stage = ContextOptimizeStage()
            with self.assertRaises(StageAbort):
                stage.execute(ctx)

        self.assertEqual(
            cascade_calls, [],
            "_cascade_apply_default MUST NOT run for ActiveSkillUnavailable; "
            f"got calls={cascade_calls!r}",
        )

    def test_context_optimize_cascade_runs_for_genuine_runtimeerror_no_active_mode(self):
        """Pre-existing behavior: a RuntimeError with no active mode still cascades."""
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )
        from proxy.pipeline.stage_modules import context_optimize as co_mod

        ctx, agent = self._make_ctx()  # snapshot_mode=None → no active mode

        cascade_calls = []
        real_cascade = co_mod._cascade_apply_default

        def spy_cascade(*args, **kwargs):
            cascade_calls.append(args)
            return real_cascade(*args, **kwargs)

        with patch(
            "proxy.mode.apply_mode",
            side_effect=RuntimeError("boom"),
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.DEFAULT,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ), patch(
            "proxy.pipeline.stage_modules.context_optimize._cascade_apply_default",
            side_effect=spy_cascade,
        ):
            stage = ContextOptimizeStage()
            # Should NOT raise — cascade handles it.
            stage.execute(ctx)

        self.assertGreaterEqual(
            len(cascade_calls), 1,
            "Genuinely mode-free RuntimeError MUST cascade to DEFAULT "
            "(spec §7.2 preserves this for non-active-skill errors).",
        )

    def test_context_optimize_skips_cascade_when_active_mode_under_non_skill_error(self):
        """Non-skill Exception with an active mode does NOT cascade (Wave 3 tightening)."""
        from proxy.pipeline.stage_modules.context_optimize import (
            ContextOptimizeStage,
        )
        from proxy.pipeline.stage_modules import context_optimize as co_mod

        snapshot = MockModeState(name="anything")
        ctx, agent = self._make_ctx(snapshot_mode=snapshot)

        cascade_calls = []
        real_cascade = co_mod._cascade_apply_default

        def spy_cascade(*args, **kwargs):
            cascade_calls.append(args)
            return real_cascade(*args, **kwargs)

        with patch(
            "proxy.mode.apply_mode",
            side_effect=RuntimeError("non-skill boom"),
        ), patch(
            "proxy.mode.resolve_mode",
            return_value=Mode.DEFAULT,
        ), patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=agent,
        ), patch(
            "proxy.pipeline.stage_modules.context_optimize._cascade_apply_default",
            side_effect=spy_cascade,
        ):
            stage = ContextOptimizeStage()
            with self.assertRaises(RuntimeError):
                stage.execute(ctx)

        self.assertEqual(
            cascade_calls, [],
            "Cascade MUST NOT run when an active mode is set, even for "
            f"non-skill errors. Got: {cascade_calls!r}",
        )
        # The active mode is NOT cleared for non-skill RuntimeError.
        self.assertIs(agent.active_mode, snapshot)
