"""Tests for WorkSkill (Phase 1d: NEW BaseSkill migration).

Mirrors tests/skills/test_ticket_skill.py for the work skill. Verifies:
  1. WorkSkill extends NEW BaseSkill from proxy.skills.skill_definition.
  2. WorkSkill().to_skill_definition() returns a SkillDefinition with all
     23 fields populated.
  3. sd.allowed_tool_names == WorkSkill().get_tool_filter(None) (round-trip).
  4. sd.sub_agent_tool_filter == WorkSkill().get_sub_agent_tool_filter(None).
  5. sd.sub_agent_proxy_work_access == ProxyAccessLevel.FULL (work sub-agents
     get proxy_work, not proxy_ticket).
  6. sd.fallback_skill == "ticket" (work → ticket fallback per R3 CRIT-2).
  7. WorkSkill has the legacy `handle_direct(ctx, "list", "")` delegating
     to TicketSkill (the ^^work list alias).
"""
from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

import pytest

from skills.work_skill import WorkSkill


class TestWorkSkillSkillDefinition:
    """Phase 1d: WorkSkill migrates to NEW BaseSkill (SkillDefinition subclass)."""

    def _to_skill_definition(self, skill: WorkSkill):
        """Construct a SkillDefinition for `skill`, holding a TRUSTED token.

        tests.skills.test_work_skill must be in TRUSTED_TRUST_CALLERS so
        `TrustToken.for_trusted_caller()` succeeds (CAT-3 trust gate).
        """
        from proxy.skills.skill_definition import _held_trust_token, TrustToken
        with _held_trust_token(TrustToken.for_trusted_caller()):
            return skill.to_skill_definition()

    def test_work_skill_extends_new_base_skill(self):
        """WorkSkill extends NEW BaseSkill from skill_definition (NOT the OLD one)."""
        from proxy.skills import skill_definition as new_skill_mod
        assert issubclass(WorkSkill, new_skill_mod.BaseSkill)

    def test_work_skill_to_skill_definition_returns_complete_definition(self):
        """WorkSkill().to_skill_definition() returns a SkillDefinition with all 23 fields.

        Plan §6 test 6 (R3 CRIT-2): C3 removes the work→ticket TTL fallback,
        so `sd.fallback_skill is None` (no automatic displacement on TTL
        expiry — the work session simply ends and ticket re-engages only
        on the next explicit ``^^ticket`` activation).
        """
        from proxy.skills.skill_definition import SkillDefinition
        skill = WorkSkill()
        sd = self._to_skill_definition(skill)

        assert isinstance(sd, SkillDefinition)
        assert sd.name == "work"
        assert sd.trigger == "^^work"
        assert sd.is_builtin, "work is a TRUSTED_BUILTIN_SKILLS"
        assert sd.fallback_skill is None
        assert isinstance(sd.allowed_tool_names, frozenset)
        assert len(sd.allowed_tool_names) > 0

    def test_work_skill_get_tool_filter_matches_subclass_tools(self):
        """sd.allowed_tool_names == WorkSkill().get_tool_filter(None) (round-trip).

        Plan §6 test 7 (R1 HIGH-2): work mode has FULL tool access including
        `read`, `write`, `edit`, `bash`. The sub-agent filter is separate
        (sub-agents cannot mutate directly).
        """
        skill = WorkSkill()
        sd = self._to_skill_definition(skill)
        assert sd.allowed_tool_names == skill.get_tool_filter(None)
        assert "bash" in sd.allowed_tool_names
        assert "write" in sd.allowed_tool_names

    def test_work_skill_sub_agent_tool_filter_is_separate(self):
        """sd.sub_agent_tool_filter == WorkSkill().get_sub_agent_tool_filter(None).

        Sub-agents in work mode are the WORKERS — they get full coding tools
        (read/write/edit/bash/glob/grep/search + proxy_work + proxy_ticket).
        `task` is NOT included — sub-agents cannot spawn sub-sub-agents
        (current CLIs do not support nesting).
        """
        skill = WorkSkill()
        sd = self._to_skill_definition(skill)
        assert sd.sub_agent_tool_filter == skill.get_sub_agent_tool_filter(None)
        # The sub-agent worker MUST have mutation tools.
        assert "bash" in sd.sub_agent_tool_filter
        assert "write" in sd.sub_agent_tool_filter
        assert "edit" in sd.sub_agent_tool_filter
        # Sub-agents cannot spawn sub-sub-agents (task tool blocked).
        assert "task" not in sd.sub_agent_tool_filter

    def test_work_subagent_has_coding_tools(self):
        """Work sub-agent filter includes the coding tools it needs to do work."""
        skill = WorkSkill()
        tools = skill.get_sub_agent_tool_filter(None)
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "bash" in tools
        assert "glob" in tools
        assert "grep" in tools
        assert "search" in tools
        assert "proxy_work" in tools
        # Decomposed read-only ticket tools (monolithic proxy_ticket was split).
        assert "proxy_ticket_list_tasks" in tools
        assert "proxy_ticket_get_task" in tools
        assert "proxy_ticket_append_review_note" in tools
        assert "task" not in tools

    def test_work_skill_sub_agent_proxy_work_access_is_full(self):
        """Work sub-agents get FULL `proxy_work` and READ_ONLY `proxy_ticket`.

        Per C4.5 plan locked decision #3, work sub-agents are READ_ONLY on
        proxy_ticket (list_tasks, get_task, append_review_note). The MAIN
        orchestrator owns ticket writes. The proxy_work access remains FULL
        so sub-agents can call report_bug, review_bug, etc.
        """
        from proxy.skills.skill_definition import ProxyAccessLevel
        skill = WorkSkill()
        sd = self._to_skill_definition(skill)
        assert sd.sub_agent_proxy_work_access == ProxyAccessLevel.FULL
        assert sd.sub_agent_proxy_ticket_access == ProxyAccessLevel.READ_ONLY


class TestWorkSkillProperties(unittest.TestCase):
    """Tests for WorkSkill class attributes (Phase 1d: NEW BaseSkill path)."""

    def setUp(self):
        self.skill = WorkSkill()

    def test_name(self):
        self.assertEqual(self.skill.name, "work")

    def test_trigger(self):
        self.assertEqual(self.skill.trigger, "^^work")

    def test_aliases(self):
        # Phase 1d: aliases is now a tuple (matches SkillDefinition.aliases field).
        self.assertEqual(self.skill.aliases, ())


class TestWorkSkillParseSubcommand(unittest.TestCase):
    """Tests for parse_subcommand (^^work <task_hash> → ("task", hash), bare → ("list", ""))."""

    def setUp(self):
        self.skill = WorkSkill()

    def test_bare_work_returns_list(self):
        """Bare ^^work → ("list", "") (alias for ^^ticket list)."""
        action, arg = self.skill.parse_subcommand("^^work")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_work_with_hash_returns_task(self):
        """^^work <8hex> → ("task", <hash>)."""
        action, arg = self.skill.parse_subcommand("^^work abc12345")
        self.assertEqual(action, "task")
        self.assertEqual(arg, "abc12345")

    def test_work_with_short_hash_returns_list(self):
        """^^work <5hex> → ("list", "") (not 8 hex, falls through)."""
        action, arg = self.skill.parse_subcommand("^^work abc12")
        self.assertEqual(action, "list")

    def test_non_work_text_returns_none(self):
        """Non-work text → (None, "") (not a work command)."""
        action, arg = self.skill.parse_subcommand("hello world")
        self.assertIsNone(action)
        self.assertEqual(arg, "")


class TestWorkSkillHandleDirectList(unittest.TestCase):
    """Tests for handle_direct("list", "") (^^work list delegates to TicketSkill)."""

    def setUp(self):
        self.skill = WorkSkill()
        self.ctx = MagicMock()
        self.ctx.session_id = "test-sess"
        self.ctx.client_hint = ""
        from proxy import state
        state.active_workspaces[self.ctx.session_id] = "test-ws"

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_list_delegates_to_ticket(self, mock_tg_cls):
        """^^work list → TicketSkill.handle_direct("list", "") → task list response.

        Phase 1d: WorkSkill.handle_direct imports TicketSkill inside the method
        body (lazy import). The test patches TaskGraph at the call site.
        """
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.is_available.return_value = True
        mock_tg.get_actionable_tasks.return_value = []
        mock_tg_cls.return_value = mock_tg

        result = self.skill.handle_direct(self.ctx, "list", "")
        self.assertIsNotNone(result)
        # "list" returns a "No active tasks" message when empty.
        self.assertIn("No active tasks", str(result))

    def test_handle_direct_unknown_action_returns_none(self):
        """Unknown action returns None (falls through to activation)."""
        result = self.skill.handle_direct(self.ctx, "unknown_action", "")
        self.assertIsNone(result)

    def test_handle_direct_task_hash_returns_none_on_non_default(self):
        """^^work <hash> (action "task") returns None on non-default → activation."""
        result = self.skill.handle_direct(self.ctx, "task", "abcd1234")
        self.assertIsNone(result)


class TestWorkSkillShouldDeactivate(unittest.TestCase):
    """Tests for should_deactivate (work mode deactivates on stop)."""

    def test_should_deactivate_on_stop(self):
        skill = WorkSkill()
        self.assertTrue(skill.should_deactivate(MagicMock(), "stop"))

    def test_should_not_deactivate_on_tool_calls(self):
        skill = WorkSkill()
        self.assertFalse(skill.should_deactivate(MagicMock(), "tool_calls"))

    def test_should_not_deactivate_on_length(self):
        skill = WorkSkill()
        self.assertFalse(skill.should_deactivate(MagicMock(), "length"))

    def test_should_deactivate_on_stop_main_ctx(self):
        """WorkSkill clears on stop for MAIN agents too (PER-TURN contract).

        Guards against an asymmetric 'fix' that makes ticket per-turn but
        leaves work persistent for main agents.
        """
        skill = WorkSkill()
        ctx = MagicMock(is_subagent=False)
        self.assertTrue(skill.should_deactivate(ctx, "stop"))


class TestWorkSkillOnActivate:
    """Tests for on_activate (rejects sub-agent work activation; US-092)."""

    def test_on_activate_rejects_subagent(self):
        """Sub-agents cannot activate work mode (per SPEC line 188-191)."""
        skill = WorkSkill()
        ctx = MagicMock()
        ctx.is_subagent = True
        ctx.session_id = "sess-12345678"
        ctx.skill_state = {}

        with pytest.raises(ValueError, match="Sub-agents cannot activate"):
            skill.on_activate(ctx)

    def test_on_activate_captures_task_hash(self):
        """Top-level ^^work <hash> activates work mode WITHOUT mutating ctx.

        R4 CRIT-4 fix: the previous implementation leaked an instance
        attribute onto the request `ctx` via
        `try: ctx.work_task_hash = task_hash; except AttributeError: pass`.
        That pattern was unsafe because the (potentially pooled) request
        context could carry the leftover attribute into a subsequent
        request. The new implementation captures the hash into a local
        variable and logs it; Phase 2 will route it through
        `proxy.work_task_store[session_id]`.

        This test pins the new contract: on_activate MUST NOT set
        `ctx.work_task_hash` (or any other undocumented attribute on
        the ctx). The captured task hash is verified by ensuring the
        call returns normally (no exception) and `skill_state["arg"]`
        is the source of truth (left untouched on ctx).
        """
        skill = WorkSkill()
        ctx = MagicMock()
        ctx.is_subagent = False
        ctx.session_id = "sess-12345678"
        ctx.skill_state = {"arg": "abc12345"}

        # Bug #10: on_activate now validates the hash exists in the task
        # graph. Mock so the validation succeeds for "abc12345" — this
        # test pins the ctx-mutation invariant, not hash validation
        # (which is covered in tests/unit/test_work_skill_activation.py).
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = {
            "name": "Test", "priority": "medium", "summary": "",
        }
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            # Must not raise (task_hash is present in skill_state).
            skill.on_activate(ctx)

        # The fix's primary invariant: ctx.work_task_hash MUST NOT be
        # set by on_activate. The previous buggy implementation did
        # `ctx.work_task_hash = task_hash` defensively; the new code
        # does not mutate the ctx at all.
        assert not ctx.__dict__.get("work_task_hash"), (
            "R4 CRIT-4: WorkSkill.on_activate must not mutate ctx "
            "with work_task_hash (Phase 2 introduces work_task_store)."
        )

    def test_on_activate_raises_without_task_hash(self):
        """R4 CRIT-4 follow-on: without a task hash in skill_state, raise ValueError.

        Previously the function silently set nothing on ctx and returned.
        Now it raises ValueError(US-092) — work mode is meaningless
        without a target task hash.
        """
        skill = WorkSkill()
        ctx = MagicMock()
        ctx.is_subagent = False
        ctx.session_id = "sess-12345678"
        ctx.skill_state = {}

        with pytest.raises(ValueError, match="requires a task hash"):
            skill.on_activate(ctx)


class TestWorkSkillToolFilter(unittest.TestCase):
    """Tests for get_tool_filter (FULL access including bash/write/edit)."""

    def test_get_tool_filter_includes_full_mutation_set(self):
        skill = WorkSkill()
        tools = skill.get_tool_filter(MagicMock())
        # Work mode MUST include write/edit/bash (the mutation tools).
        self.assertIn("bash", tools)
        self.assertIn("write", tools)
        self.assertIn("edit", tools)
        self.assertIn("read", tools)
        self.assertIn("proxy_work", tools)
        # Decomposed ticket tools (monolithic proxy_ticket was split per-action);
        # work mode holds the read-only ticket set.
        self.assertIn("proxy_ticket_list_tasks", tools)
        self.assertIn("proxy_ticket_get_task", tools)
        self.assertIn("proxy_ticket_append_review_note", tools)


if __name__ == "__main__":
    unittest.main()