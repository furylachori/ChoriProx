"""US-017f: Deferred Execution After Planning Conversation.

The user story says: WHEN the user says "^^ticket go now", "^^ticket do it",
"^^ticket execute" after a planning conversation, THEN the agent MUST immediately
execute all task operations discussed.

Reality: No "go" / "execute" subcommand exists in TicketSkill._parse_ticket_args.
All imperative triggers ("go now", "do it", "execute") fall through to the generic
"plan" action, which defers to LLM ticket planning mode. The parser preserves the
FULL user text (leading verb is NOT stripped) so the LLM sees complete intent.

These tests document the parsing behavior and verify that "plan" does NOT
trigger direct execution (handle_direct returns None for plan).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from skills.ticket_skill import TicketSkill


@pytest.fixture
def skill():
    return TicketSkill()


@pytest.fixture
def ctx():
    c = MagicMock()
    c.session_id = "test-sess-017f"
    c.client_hint = "kilocode"
    c.agent_id = "agent_test"
    c.messages = []
    c.is_subagent = False
    c.active_skill_name = "ticket"
    c.raw_user_text = ""
    c.skill_state = {}
    return c


# ---------------------------------------------------------------------------
# parse_subcommand: imperative triggers all route to "plan"
# ---------------------------------------------------------------------------


class TestUS017fDeferredParseSubcommand:
    """US-017f: imperative triggers parse as 'plan', not as 'execute'."""

    @pytest.mark.parametrize("text, expected_arg", [
        ("^^ticket go now", "go now"),
        ("^^ticket do it", "do it"),
        ("^^ticket execute", "execute"),
        ("^^ticket go", "go"),
        ("^^ticket just do it", "just do it"),
    ])
    def test_imperative_triggers_parse_as_plan(self, skill, text, expected_arg):
        action, arg = skill.parse_subcommand(text)
        assert action == "plan", f"{text!r} should parse as 'plan', got {action!r}"
        assert arg == expected_arg


    def test_plan_action_does_not_trigger_direct_execution(self, skill, ctx):
        """handle_direct returns None for 'plan', deferring to skill activation."""
        from proxy import state
        state.active_workspaces[ctx.session_id] = "test-ws"
        result = skill.handle_direct(ctx, action="plan", arg="do the thing")
        assert result is None
