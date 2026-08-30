"""Theory 4 tests: resolve_mode() must activate the ticket skill when
inherited_skill is set, even if subagent_ticket_mode is False.

This file is the canary-detection / mode-resolution test counterpart of
`tests/unit/test_mode_resolution.py`. It is run by the orchestrator via:

    PYTHONPATH=src python3 -m pytest tests/proxy/test_mode_resolution.py \\
        tests/proxy/test_canary.py -v -k "theory4"

Hypothesis (Theory 4): resolve_mode() only activates the ticket skill if
subagent_ticket_mode=True. If a canary in the first user message still
names the inherited skill ("ticket") but subagent_ticket_mode is somehow
False, the skill is never activated, _is_skill stays False,
prepare_tooling_payload is called without skill_name, and proxy_ticket is
never injected into the tool set.

The expected behaviour is: resolve_mode() must consult ctx.inherited_skill
INDEPENDENTLY of ctx.subagent_ticket_mode, and when inherited_skill is set
the agent must transition to SKILL_SUBAGENT (sub-agent) or SKILL (main)
mode with the named skill activated.
"""
import threading
import unittest
from unittest.mock import MagicMock, patch

from proxy.mode import Mode


# ---------------------------------------------------------------------------
# Mock helpers (mirrors of tests/unit/test_mode_resolution.py)
# ---------------------------------------------------------------------------


class _MockActiveMode:
    """Minimal ModeState-like object for tests."""
    def __init__(self, name="ticket", age_seconds=0):
        import time as _t
        self.name = name
        self.activated_at = _t.time() - age_seconds
        self.ttl_seconds = 300
        self.skill_state = {}
        self.allowed_tool_names = frozenset(
            {"read", "write", "bash", "proxy_ticket", "proxy_work"}
        )

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
        self._set_active_mode_calls = getattr(self, "_set_active_mode_calls", 0) + 1
        self.active_mode = mode_state
        return True

    def clear_active_mode(self, expected_mode=None):
        self._clear_active_mode_calls = getattr(self, "_clear_active_mode_calls", 0) + 1
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
        mode = getattr(self, "active_mode", None)
        return mode is not None and getattr(mode, "name", None) == "ticket"

    def snapshot_mode(self):
        return self.active_mode

    def snapshot_mode_and_ticket_ts(self):
        return self.active_mode, getattr(self, "_ticket_mode_ts", None)

    def get_mode_name(self):
        ms = self.active_mode
        return ms.name if ms else None

    def is_mode_expired(self):
        return False


def _make_ctx(*, is_subagent=False, subagent_ticket_mode=False,
              inherited_skill=None, body=None, session_id="sess_theory4"):
    """Build a minimal RequestContext-like namespace object.

    Critically: this is the Theory 4 scenario — subagent_ticket_mode=False
    but inherited_skill is set.
    """
    if body is None:
        body = {
            "messages": [{"role": "user", "content": "do a subagent task"}],
            "tools": [
                {"type": "function", "function": {
                    "name": "read", "description": "r",
                    "parameters": {"type": "object", "properties": {}}
                }},
                {"type": "function", "function": {
                    "name": "bash", "description": "b",
                    "parameters": {"type": "object", "properties": {}}
                }},
            ],
        }
    ctx = MagicMock()
    ctx.body = body
    ctx.session_id = session_id
    ctx.is_subagent = is_subagent
    ctx.subagent_ticket_mode = subagent_ticket_mode
    ctx.inherited_skill = inherited_skill
    ctx.workspace_name = ""
    ctx.seeded_tasks = []
    ctx._skill_activated = False
    ctx._mode = None
    ctx.model_cfg = MagicMock()
    ctx.messages = body.get("messages", [])
    return ctx


def _activate_ticket_side_effect(ag, skill, skill_ctx, ctx=None):
    """Side-effect that mimics proxy.agent.agent._activate_skill for ticket."""
    ag.active_mode = _MockActiveMode(name="ticket")
    if ctx is not None:
        ctx._skill_activated = True


# ---------------------------------------------------------------------------
# Theory 4 — Sub-agent with inherited_skill but NO subagent_ticket_mode
# ---------------------------------------------------------------------------


class TestTheory4ResolveModeSkillSubagent(unittest.TestCase):
    """Theory 4: resolve_mode() with is_subagent=True, subagent_ticket_mode=False,
    inherited_skill="ticket" must still activate the ticket skill and return
    Mode.SKILL_SUBAGENT.
    """

    @patch("proxy.skills.skill_registry.SkillRegistry")
    @patch("proxy.agent.agent._inherit_skill")
    def test_theory4_subagent_inherited_skill_no_ticket_mode_flag(
        self, mock_activate, mock_registry_cls
    ):
        """Sub-agent + inherited_skill="ticket" + subagent_ticket_mode=False
        → resolve_mode must return Mode.SKILL_SUBAGENT, NOT Mode.SUBAGENT.

        Reproduces the scenario the user describes in Theory 4:
        subagent_ticket_mode is False but a canary in the user message
        still names the inherited skill. The resolver must honour the
        canary's named skill.
        """
        mock_skill = MagicMock()
        mock_skill.name = "ticket"
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill
        mock_registry_cls.return_value = mock_registry

        mock_activate.side_effect = _activate_ticket_side_effect

        agent = _MockAgent(is_ticket_mode=False, agent_id="agent_theory4")
        ctx = _make_ctx(
            is_subagent=True,
            subagent_ticket_mode=False,   # <-- The Theory 4 condition
            inherited_skill="ticket",     # <-- Set by the canary
        )

        from proxy.mode import resolve_mode
        mode = resolve_mode(ctx, agent)

        # The agent's active_mode must be set to a ticket-skill ModeState,
        # proving _inherit_skill was called via the inherited_skill branch.
        self.assertIsNotNone(
            agent.active_mode,
            "REGRESSION (Theory 4): resolve_mode() did NOT activate the "
            "ticket skill when inherited_skill='ticket' but "
            "subagent_ticket_mode=False",
        )
        self.assertEqual(
            agent.active_mode.name, "ticket",
            "REGRESSION (Theory 4): active_mode was set, but to a "
            "non-ticket skill",
        )
        # _inherit_skill must have been called (not just left dangling).
        self.assertEqual(mock_activate.call_count, 1)
        # The resolved mode must be SKILL_SUBAGENT.
        self.assertEqual(
            mode, Mode.SKILL_SUBAGENT,
            f"REGRESSION (Theory 4): resolve_mode() returned {mode!r} "
            f"instead of Mode.SKILL_SUBAGENT",
        )


class TestTheory4ApplyModeInjectsProxyTicket(unittest.TestCase):
    """Theory 4 follow-up: apply_mode() must inject proxy_ticket for a
    SKILL_SUBAGENT mode produced by the inherited_skill path.
    """

    @patch("proxy.context.select_context", return_value=None)
    @patch("proxy.tool_cache.cache_tool_schema")
    @patch("proxy.tool_cache.get_cached_tool_schema", return_value=None)
    @patch("proxy.skills.skill_registry.SkillRegistry")
    @patch("proxy.agent.agent._inherit_skill")
    def test_theory4_resolve_then_apply_injects_proxy_ticket(
        self, mock_activate, mock_registry_cls, mock_get_cached, mock_cache,
        mock_select,
    ):
        """End-to-end: resolve_mode() then apply_mode() for a sub-agent with
        inherited_skill="ticket" and subagent_ticket_mode=False must end
        with proxy_ticket in body["tools"].
        """
        # Use the REAL TicketSkill so the tool filter is authentic.
        from skills.ticket_skill import TicketSkill
        real_skill = TicketSkill()

        mock_registry = MagicMock()
        mock_registry.get.return_value = real_skill
        mock_registry_cls.return_value = mock_registry

        agent = _MockAgent(is_ticket_mode=False, agent_id="agent_theory4_e2e")
        ctx = _make_ctx(
            is_subagent=True,
            subagent_ticket_mode=False,
            inherited_skill="ticket",
        )

        from proxy.mode import resolve_mode, apply_mode
        mode = resolve_mode(ctx, agent)
        self.assertEqual(
            mode, Mode.SKILL_SUBAGENT,
            "Pre-condition for apply_mode injection: mode must be SKILL_SUBAGENT",
        )

        # The skill is now active on the agent. apply_mode should pick this
        # up via agent.snapshot_mode().name == "ticket" and route
        # prepare_tooling_payload with skill_name="ticket".
        apply_mode(ctx, agent, ctx.body, mode)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in ctx.body.get("tools", [])
        ]
        self.assertIn(
            "proxy_ticket_list_tasks", tool_names,
            f"REGRESSION (Theory 4): apply_mode() did NOT inject "
            f"proxy_ticket_list_tasks for a sub-agent with "
            f"inherited_skill='ticket'. tools={tool_names}",
        )
        self.assertIn(
            "proxy_ticket_get_task", tool_names,
            f"REGRESSION (Theory 4): apply_mode() did NOT inject "
            f"proxy_ticket_get_task for a sub-agent with "
            f"inherited_skill='ticket'. tools={tool_names}",
        )
        # bash should be filtered out by the ticket skill's tool filter
        self.assertNotIn(
            "bash", tool_names,
            f"REGRESSION (Theory 4): bash should be filtered out in ticket "
            f"mode but it is present. tools={tool_names}",
        )

    @patch("proxy.context.select_context", return_value=None)
    @patch("proxy.tool_cache.cache_tool_schema")
    @patch("proxy.tool_cache.get_cached_tool_schema", return_value=None)
    @patch("proxy.skills.skill_registry.SkillRegistry")
    @patch("proxy.agent.agent._activate_skill")
    def test_theory4_main_agent_inherited_skill_no_ticket_mode_flag(
        self, mock_activate, mock_registry_cls, mock_get_cached, mock_cache,
        mock_select,
    ):
        """MAIN agent (not sub-agent) + inherited_skill="ticket" +
        subagent_ticket_mode=False (irrelevant for main, but present) →
        resolve_mode returns Mode.SKILL and apply_mode injects proxy_ticket.

        This is the symmetric case — main session, e.g. an operator message
        that names a skill in canary form.
        """
        from skills.ticket_skill import TicketSkill
        real_skill = TicketSkill()

        mock_registry = MagicMock()
        mock_registry.get.return_value = real_skill
        mock_registry_cls.return_value = mock_registry

        mock_activate.side_effect = _activate_ticket_side_effect

        agent = _MockAgent(is_ticket_mode=False, agent_id="agent_theory4_main")
        ctx = _make_ctx(
            is_subagent=False,
            subagent_ticket_mode=False,
            inherited_skill="ticket",
        )

        from proxy.mode import resolve_mode, apply_mode
        mode = resolve_mode(ctx, agent)
        self.assertEqual(mode, Mode.SKILL)
        self.assertEqual(agent.active_mode.name, "ticket")

        apply_mode(ctx, agent, ctx.body, mode)
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in ctx.body.get("tools", [])
        ]
        # Slice 8b: decomposed proxy_ticket_* tools (ticket MAIN sees 2 read-only).
        self.assertIn("proxy_ticket_list_tasks", tool_names)
        self.assertIn("proxy_ticket_get_task", tool_names)


if __name__ == "__main__":
    unittest.main()
