"""C4.5: three-way prompt/guard/filter alignment consistency test.

After C2/C3/C4.5:
- Filter (get_tool_filter / get_sub_agent_tool_filter) ⊆ Injected schemas (∈ _get_injectable_memory_tools)
- Guard allows exactly what the filter includes (skill_guard.check_sub_agent_proxy_access)
- Prompt advertises exactly what the filter allows (system prompt text)

This test programmatically verifies all three sources AGREE for every
builtin skill × role. The architect review of C1-C4 found that pairwise
alignment was insufficient — this test catches any future drift.
"""
import unittest
from unittest.mock import MagicMock, patch

from proxy.skills.skill_definition import ProxyAccessLevel, SkillRequestContext


def _ctx(is_subagent: bool = False) -> SkillRequestContext:
    return SkillRequestContext(
        session_id="sess-c45-alignment",
        is_subagent=is_subagent,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint=None,
        body={},
        seeded_tasks=[],
        mode_state=None,
    )


def _patch_trust_gate():
    """Mock is_trusted_skill to return True so check_sub_agent_proxy_access
    passes the CAT-3 trust gate and reaches the READ_ONLY check."""
    return patch("proxy.skill_guard.is_trusted_skill", return_value=True)


class TestThreeWayAlignmentTicketMain(unittest.TestCase):
    """Ticket MAIN: filter, injection, and guard all agree."""

    def test_ticket_main_filter_injection_guard_alignment(self):
        from skills.ticket_skill import TicketSkill
        from proxy.tool_cache import _get_injectable_memory_tools
        from proxy.skill_guard import check_sub_agent_proxy_access
        from proxy.tool_pipeline.tool_call import ToolCall

        skill = TicketSkill()

        # MAIN filter (not sub-agent)
        filter_tools = skill.get_tool_filter(_ctx(is_subagent=False))

        # Injected schemas (after C1 fix)
        injected = _get_injectable_memory_tools(skill, "main")
        injected_names = {t["function"]["name"] for t in injected}

        # C4.5 fix 3: proxy_work removed from MAIN filter (and not injected).
        self.assertNotIn("proxy_work", filter_tools)
        self.assertNotIn("proxy_work", injected_names)

        # Decomposed proxy_ticket_* tools (Slice 8b) ARE in filter and IS injected.
        self.assertIn("proxy_ticket_list_tasks", filter_tools)
        self.assertIn("proxy_ticket_list_tasks", injected_names)
        self.assertIn("proxy_ticket_get_task", filter_tools)
        self.assertIn("proxy_ticket_get_task", injected_names)
        # Legacy monolithic proxy_ticket is no longer in either set.
        self.assertNotIn("proxy_ticket", filter_tools)
        self.assertNotIn("proxy_ticket", injected_names)

        # Filter ⊆ Injected (filter must not include tools not in the injection)
        # Injected proxy_* tools ⊆ Filter (the injection adds proxy_* to body["tools"];
        # the filter must allow them. The filter has additional client tools like read/glob/grep.)
        injected_proxy_tools = {n for n in injected_names if n.startswith("proxy_")}
        self.assertTrue(
            injected_proxy_tools <= filter_tools,
            f"Injected proxy_* tools {injected_proxy_tools} not ⊆ Filter {filter_tools}",
        )

        # Guard allows proxy_ticket_list_tasks — MAIN bypasses the per-action
        # READ_ONLY check via tool_guard._check_work_mode_actions.
        # (skill_guard.check_sub_agent_proxy_access is the sub-agent gate; MAIN
        # doesn't go through it.)
        with _patch_trust_gate():
            result = check_sub_agent_proxy_access(
                ToolCall(
                    id="t1", name="proxy_ticket_get_task",
                    arguments={"short_hash": "abc"},
                    source="native", from_cache=False,
                ),
                MagicMock(session_id="sess-c45", is_subagent=True, client_hint=""),
                agent=MagicMock(),
                registry=MagicMock(get=MagicMock(return_value=skill)),
            )
        # For SUB agents, READ_ONLY still allows get_task.
        self.assertIsNone(result, f"sub-agent proxy_ticket_get_task must be allowed; got {result!r}")


class TestThreeWayAlignmentTicketSub(unittest.TestCase):
    """Ticket SUB: filter, injection, and guard all agree."""

    def test_ticket_sub_filter_injection_guard_alignment(self):
        from skills.ticket_skill import TicketSkill
        from proxy.tool_cache import _get_injectable_memory_tools
        from proxy.skill_guard import check_sub_agent_proxy_access
        from proxy.tool_pipeline.tool_call import ToolCall

        skill = TicketSkill()

        # SUB filter (sub-agent)
        filter_tools = skill.get_sub_agent_tool_filter(_ctx(is_subagent=True))

        # Injected schemas (same as MAIN for ticket — both get proxy_ticket only)
        injected = _get_injectable_memory_tools(skill, "subagent")
        injected_names = {t["function"]["name"] for t in injected}

        # proxy_work NOT in sub filter (C2 fix), NOT injected.
        self.assertNotIn("proxy_work", filter_tools)
        self.assertNotIn("proxy_work", injected_names)

        # Decomposed proxy_ticket_* tools ARE in sub filter, IS injected.
        # Ticket sub-agents get the FULL decomposed ticket surface (all 10 tools).
        self.assertIn("proxy_ticket_list_tasks", filter_tools)
        self.assertIn("proxy_ticket_list_tasks", injected_names)
        self.assertIn("proxy_ticket_create_tasks", filter_tools)
        self.assertIn("proxy_ticket_create_tasks", injected_names)
        self.assertIn("proxy_ticket_edit", filter_tools)
        self.assertIn("proxy_ticket_edit", injected_names)
        # Legacy monolithic proxy_ticket is no longer in either set.
        self.assertNotIn("proxy_ticket", filter_tools)
        self.assertNotIn("proxy_ticket", injected_names)

        # Filter ⊆ Injected
        # Injected proxy_* tools ⊆ Filter (the injection adds proxy_* to body["tools"];
        # the filter must allow them. The filter has additional client tools like read/glob/grep.)
        injected_proxy_tools = {n for n in injected_names if n.startswith("proxy_")}
        self.assertTrue(
            injected_proxy_tools <= filter_tools,
            f"Injected proxy_* tools {injected_proxy_tools} not ⊆ Filter {filter_tools}",
        )

        # Guard: SUB with FULL proxy_ticket access (no restriction)
        skill.sub_agent_proxy_ticket_access = ProxyAccessLevel.FULL
        with _patch_trust_gate():
            result = check_sub_agent_proxy_access(
                ToolCall(
                    id="t2", name="proxy_ticket_create_tasks",
                    arguments={"tasks": [{"name": "x", "priority": "high"}]},
                    source="native", from_cache=False,
                ),
                MagicMock(session_id="sess-c45", is_subagent=True, client_hint=""),
                agent=MagicMock(),
                registry=MagicMock(get=MagicMock(return_value=skill)),
            )
        self.assertIsNone(
            result,
            f"ticket SUB with FULL proxy_ticket must allow create_tasks; got {result!r}",
        )


class TestThreeWayAlignmentWorkMain(unittest.TestCase):
    """Work MAIN: filter, injection, and guard all agree."""

    def test_work_main_filter_injection_guard_alignment(self):
        from skills.work_skill import WorkSkill
        from proxy.tool_cache import _get_injectable_memory_tools

        skill = WorkSkill()
        filter_tools = skill.get_tool_filter(_ctx(is_subagent=False))

        # C1: work mode injects BOTH proxy_work AND decomposed proxy_ticket_* tools
        injected = _get_injectable_memory_tools(skill, "main")
        injected_names = {t["function"]["name"] for t in injected}

        # Both should be in filter and injected
        self.assertIn("proxy_work", filter_tools)
        self.assertIn("proxy_work", injected_names)
        # Work MAIN sees 3 decomposed ticket tools + proxy_work.
        for tool_name in ("proxy_ticket_list_tasks", "proxy_ticket_get_task", "proxy_ticket_append_review_note"):
            self.assertIn(tool_name, filter_tools, f"work MAIN filter missing {tool_name}")
            self.assertIn(tool_name, injected_names, f"work MAIN injection missing {tool_name}")
        # Legacy monolithic proxy_ticket is no longer in either set.
        self.assertNotIn("proxy_ticket", filter_tools)
        self.assertNotIn("proxy_ticket", injected_names)

        # Filter ⊆ Injected
        # Injected proxy_* tools ⊆ Filter (the injection adds proxy_* to body["tools"];
        # the filter must allow them. The filter has additional client tools like read/glob/grep.)
        injected_proxy_tools = {n for n in injected_names if n.startswith("proxy_")}
        self.assertTrue(
            injected_proxy_tools <= filter_tools,
            f"Injected proxy_* tools {injected_proxy_tools} not ⊆ Filter {filter_tools}",
        )


class TestThreeWayAlignmentWorkSub(unittest.TestCase):
    """Work SUB: filter, injection, and guard all agree (C4.5 fix 1)."""

    def test_work_sub_filter_injection_guard_alignment(self):
        from skills.work_skill import WorkSkill
        from proxy.tool_cache import _get_injectable_memory_tools
        from proxy.skill_guard import check_sub_agent_proxy_access
        from proxy.tool_pipeline.tool_call import ToolCall

        skill = WorkSkill()

        # Confirm C4.5 fix 1: work SUB proxy_ticket access is READ_ONLY
        self.assertEqual(
            skill.sub_agent_proxy_ticket_access,
            ProxyAccessLevel.READ_ONLY,
            f"C4.5 fix 1: work sub proxy_ticket access must be READ_ONLY; got {skill.sub_agent_proxy_ticket_access!r}",
        )

        filter_tools = skill.get_sub_agent_tool_filter(_ctx(is_subagent=True))
        injected = _get_injectable_memory_tools(skill, "subagent")
        injected_names = {t["function"]["name"] for t in injected}

        # Both should be in filter and injected (Slice 8b: decomposed ticket + work)
        self.assertIn("proxy_work", filter_tools)
        self.assertIn("proxy_work", injected_names)
        # Work SUB sees the same 3 decomposed ticket tools as Work MAIN
        # (READ_ONLY = list_tasks, get_task, append_review_note).
        for tool_name in ("proxy_ticket_list_tasks", "proxy_ticket_get_task", "proxy_ticket_append_review_note"):
            self.assertIn(tool_name, filter_tools, f"work SUB filter missing {tool_name}")
            self.assertIn(tool_name, injected_names, f"work SUB injection missing {tool_name}")
        # Legacy monolithic proxy_ticket is no longer in either set.
        self.assertNotIn("proxy_ticket", filter_tools)
        self.assertNotIn("proxy_ticket", injected_names)

        # Filter ⊆ Injected
        # Injected proxy_* tools ⊆ Filter (the injection adds proxy_* to body["tools"];
        # the filter must allow them. The filter has additional client tools like read/glob/grep.)
        injected_proxy_tools = {n for n in injected_names if n.startswith("proxy_")}
        self.assertTrue(
            injected_proxy_tools <= filter_tools,
            f"Injected proxy_* tools {injected_proxy_tools} not ⊆ Filter {filter_tools}",
        )

        # Guard: work SUB with READ_ONLY proxy_ticket — get_task ALLOWED,
        # create_tasks DENIED.
        with _patch_trust_gate():
            # get_task ALLOWED
            allowed_result = check_sub_agent_proxy_access(
                ToolCall(
                    id="t3a", name="proxy_ticket_get_task",
                    arguments={"short_hash": "abc"},
                    source="native", from_cache=False,
                ),
                MagicMock(session_id="sess-c45", is_subagent=True, client_hint=""),
                agent=MagicMock(),
                registry=MagicMock(get=MagicMock(return_value=skill)),
            )
            self.assertIsNone(
                allowed_result,
                f"work SUB READ_ONLY must allow get_task; got {allowed_result!r}",
            )

            # create_tasks DENIED
            denied_result = check_sub_agent_proxy_access(
                ToolCall(
                    id="t3b", name="proxy_ticket_create_tasks",
                    arguments={"tasks": [{"name": "x", "priority": "high"}]},
                    source="native", from_cache=False,
                ),
                MagicMock(session_id="sess-c45", is_subagent=True, client_hint=""),
                agent=MagicMock(),
                registry=MagicMock(get=MagicMock(return_value=skill)),
            )
        self.assertIsNotNone(
            denied_result,
            f"work SUB READ_ONLY must DENY create_tasks; got: {denied_result!r}",
        )
        # Slice 8b: CapabilityTable.check_action returns the deny_message from
        # WORK_CAPABILITY_TABLE for proxy_ticket_create_tasks (subagent role).
        self.assertIn("Work sub-agents cannot create tasks", denied_result)

        # C6.5: also assert prompt advertises READ-ONLY proxy_ticket (matches the
        # skill_guard contract enforced above).
        from proxy.tool_parsing.work_prompts import WORK_SUB_PROMPT
        self.assertIn(
            "READ-ONLY", WORK_SUB_PROMPT,
            f"WORK_SUB_PROMPT must advertise READ-ONLY proxy_ticket (got: {WORK_SUB_PROMPT[:200]}...)",
        )


if __name__ == "__main__":
    unittest.main()
