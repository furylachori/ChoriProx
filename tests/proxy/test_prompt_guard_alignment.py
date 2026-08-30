"""C2 (Commit 2): prompt/guard behavioral alignment tests.

After C2:
- Ticket MAIN prompt = TICKET_MAIN_PROMPT (orchestrator identity).
- Ticket SUB prompt = TICKET_SUB_PROMPT (worker identity).
- Work SUB prompt = WORK_SUB_PROMPT (no lifecycle actions).
- Work MAIN prompt = WORK_MANAGER_PROMPT (one-shot completion contract).
- skill_guard READ_ONLY allows {list_tasks, get_task, append_review_note}.
- `question` is registry-forbidden in all skills.

These tests pin the cross-cutting invariants at the BEHAVIORAL boundary —
through real hooks and real guards, not by inspecting constants in
isolation. Each test has a corresponding revert-test in C2 STEP 4.
"""
import unittest
from unittest.mock import MagicMock, patch

from proxy.skills.skill_definition import ModeState, SkillRequestContext


def _ctx(*, is_subagent: bool = False, skill_state: dict = None) -> SkillRequestContext:
    """Build a SkillRequestContext with a ModeState carrying skill_state.

    Skill data lives on ModeState.skill_state (not on SkillRequestContext).
    For tests that need to drive get_activation_rewrite's skill_state lookup,
    the ModeState.skill_state must be populated.
    """
    mode_state = None
    if skill_state is not None:
        mode_state = ModeState(
            name="ticket",
            activated_at=__import__("time").time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            skill_state=skill_state,
        )
    return SkillRequestContext(
        session_id=f"sess-c2-{is_subagent}",
        is_subagent=is_subagent,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint=None,
        body={},
        seeded_tasks=[],
        mode_state=mode_state,
    )


def _make_proxy_ctx_for_skill_guard():
    """Build a RequestContext-like MagicMock for skill_guard tests."""
    from proxy.tool_pipeline.tool_call import ToolCall
    ctx = MagicMock()
    ctx.session_id = "sess-c2-guard"
    ctx.is_subagent = True  # skill_guard is the sub-agent gate
    ctx.client_hint = ""
    ctx.tool_name = "proxy_ticket"
    return ctx


class TestTicketMainPromptNoSubAgentIdentity(unittest.TestCase):
    """C2 (a): MAIN ticket prompt does NOT say 'focused sub-agent'."""

    def test_main_prompt_no_sub_agent_identity(self):
        from skills.ticket_skill import TicketSkill
        from proxy.tool_parsing.ticket_prompts import TICKET_MAIN_PROMPT
        ctx = _ctx(is_subagent=False)
        prompt = TicketSkill().get_system_prompt(ctx)
        # Revert-test 4.1: revert the constant rename and this assertion fails.
        self.assertNotIn("focused sub-agent", prompt)
        self.assertIn("planning orchestrator", prompt.lower())
        self.assertIn("task", prompt.lower())
        self.assertIn("delegate", prompt.lower())
        # The prompt MUST be the new TICKET_MAIN_PROMPT constant.
        self.assertEqual(prompt, TICKET_MAIN_PROMPT)


class TestTicketActivationRewriteSaysTaskTool(unittest.TestCase):
    """C2 (b): get_activation_rewrite says 'task tool', NOT 'proxy_ticket tool'."""

    def test_activation_rewrite_says_task_tool(self):
        from skills.ticket_skill import TicketSkill
        ctx = _ctx(skill_state={"arg": "build auth module"})
        rewrite = TicketSkill().get_activation_rewrite(ctx)
        # Revert-test 4.2: revert line 894 to old text and this fails.
        self.assertIn("task tool", rewrite)
        self.assertNotIn("proxy_ticket tool", rewrite)
        # The arg must still be in the rewrite.
        self.assertIn("build auth module", rewrite)


class TestReadOnlySubAgentCanCallGetTask(unittest.TestCase):
    """C2 (c): READ_ONLY sub-agent can call proxy_ticket(action='get_task')."""

    @patch("proxy.skill_guard.is_trusted_skill", return_value=True)
    def test_read_only_can_call_get_task(self, mock_trusted):
        from proxy.tool_pipeline.tool_call import ToolCall
        from proxy.skills.skill_registry import SkillRegistry
        from proxy.skill_guard import check_sub_agent_proxy_access
        from proxy.skills.skill_definition import ProxyAccessLevel

        skill = MagicMock()
        skill.name = "ticket-ro-test"
        skill.sub_agent_proxy_ticket_access = ProxyAccessLevel.READ_ONLY
        skill.sub_agent_proxy_work_access = ProxyAccessLevel.NEVER
        skill.check_action = None

        registry = MagicMock(spec=SkillRegistry)
        registry.get.return_value = skill

        tool_call = ToolCall(
            id="t1",
            name="proxy_ticket",
            arguments={"action": "get_task", "short_hash": "abc123"},
            source="native",
            from_cache=False,
        )
        ctx = _make_proxy_ctx_for_skill_guard()

        result = check_sub_agent_proxy_access(tool_call, ctx, agent=MagicMock(), registry=registry)
        # Revert-test 4.4: revert to {"list_tasks", "append_review_note"} and this fails.
        self.assertIsNone(
            result,
            f"READ_ONLY skill should allow get_task; got error: {result!r}",
        )

    @patch("proxy.skill_guard.is_trusted_skill", return_value=True)
    def test_read_only_still_blocks_create_tasks(self, mock_trusted):
        """Regression: READ_ONLY boundary must still block create_tasks."""
        from proxy.tool_pipeline.tool_call import ToolCall
        from proxy.skills.skill_registry import SkillRegistry
        from proxy.skill_guard import check_sub_agent_proxy_access
        from proxy.skills.skill_definition import ProxyAccessLevel

        skill = MagicMock()
        skill.name = "ticket-ro-test"
        skill.sub_agent_proxy_ticket_access = ProxyAccessLevel.READ_ONLY
        skill.sub_agent_proxy_work_access = ProxyAccessLevel.NEVER
        skill.check_action = None

        registry = MagicMock(spec=SkillRegistry)
        registry.get.return_value = skill

        tool_call = ToolCall(
            id="t2",
            name="proxy_ticket",
            arguments={"action": "create_tasks", "tasks": [{"name": "x", "priority": "high"}]},
            source="native",
            from_cache=False,
        )
        ctx = _make_proxy_ctx_for_skill_guard()

        result = check_sub_agent_proxy_access(tool_call, ctx, agent=MagicMock(), registry=registry)
        self.assertIsNotNone(
            result,
            f"READ_ONLY skill must still block create_tasks; got: {result!r}",
        )
        self.assertIn("read-only", result.lower())


class TestQuestionRegistryForbiddenInAllSkills(unittest.TestCase):
    """C2 (d): `question` is registry-forbidden in all skills."""

    def test_question_not_in_ticket_skill_tools(self):
        from proxy.tool_pipeline.tool_registry import registry
        tools_ticket = registry.get_tools_for_skill("ticket")
        self.assertNotIn("question", tools_ticket)

    def test_question_not_in_work_skill_tools(self):
        from proxy.tool_pipeline.tool_registry import registry
        tools_work = registry.get_tools_for_skill("work")
        self.assertNotIn("question", tools_work)

    def test_question_registry_has_empty_available_in_skills(self):
        from proxy.tool_pipeline.tool_registry import registry
        # Registry-level enforcement: question must have empty available_in_skills.
        question_meta = registry._tools.get("question")
        self.assertIsNotNone(question_meta)
        self.assertEqual(question_meta.available_in_skills, frozenset())


class TestTicketSubFilterNoProxyWork(unittest.TestCase):
    """C2 (e): TicketSkill sub_agent_tool_filter does NOT include proxy_work."""

    def test_sub_filter_no_proxy_work(self):
        from skills.ticket_skill import TicketSkill
        ctx = _ctx(is_subagent=True)
        tool_filter = TicketSkill().get_sub_agent_tool_filter(ctx)
        # Revert-test 4.5: re-add "proxy_work" to the filter and this fails.
        self.assertNotIn(
            "proxy_work", tool_filter,
            f"ticket sub-agent filter must NOT include proxy_work; got {set(tool_filter)}",
        )
        # Sanity: decomposed proxy_ticket_* tools are still in (workers need them).
        # Slice 8b: legacy monolithic proxy_ticket is gone; ticket sub-agent
        # receives the FULL decomposed ticket surface.
        self.assertIn("proxy_ticket_list_tasks", tool_filter)
        self.assertIn("proxy_ticket_get_task", tool_filter)
        self.assertIn("proxy_ticket_create_tasks", tool_filter)
        self.assertIn("proxy_ticket_edit", tool_filter)


class TestWorkSubPromptNoLifecycleActions(unittest.TestCase):
    """C2 (f): WorkSkill sub-agent prompt does NOT instruct start_work or stop_work."""

    def test_work_sub_prompt_no_lifecycle_actions(self):
        from skills.work_skill import WorkSkill
        from proxy.tool_parsing.work_prompts import WORK_SUB_PROMPT
        # Test the constant directly (placeholder substitution needs task context).
        # Revert-test 4.3: revert get_subagent_prompt to return WORK_MANAGER_PROMPT
        # and this fails (WORK_MANAGER_PROMPT has "Call start_work" as step 1).
        self.assertIn("1. Do the actual work", WORK_SUB_PROMPT)
        self.assertNotIn("1. Call start_work", WORK_SUB_PROMPT)
        # The proxy_work actions that ARE allowed for subs must be mentioned.
        self.assertIn("report_bug", WORK_SUB_PROMPT)
        # The forbidden lifecycle actions must be marked as not allowed.
        self.assertIn("NOT allowed", WORK_SUB_PROMPT)

    def test_work_sub_prompt_via_hook(self):
        """End-to-end through WorkSkill.get_subagent_prompt — uses placeholder
        substitution with ad-hoc fallback when no task context."""
        from skills.work_skill import WorkSkill
        ctx = _ctx(is_subagent=True)
        prompt = WorkSkill().get_subagent_prompt(ctx)
        # The constant is rendered with placeholder substitution. Step 1 should
        # be the worker workflow ("Do the actual work"), not the orchestrator
        # workflow ("Call start_work").
        self.assertNotIn("Call start_work", prompt)
        self.assertIn("Do the actual work", prompt)


if __name__ == "__main__":
    unittest.main()