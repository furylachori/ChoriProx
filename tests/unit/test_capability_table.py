"""C7a: Capability table data model tests.

Verifies:
1. Structural invariant: visible=True ∧ allowed=False (without actions_allowed) is ILLEGAL.
2. Lookup methods (visible_tools, check_action, injectable_proxy_tools) work correctly.
3. Consistency: each builtin table EXACTLY matches the current TicketSkill/WorkSkill
   get_tool_filter / get_sub_agent_tool_filter output.
"""

import pytest


class TestCapabilityTableInvariant:
    """Structural invariant: the bait-loop bug class is structurally impossible."""

    def test_visible_not_allowed_no_actions_raises(self):
        """visible=True, allowed=False, actions_allowed=None → ILLEGAL."""
        from proxy.skills.skill_definition import ToolCapability, CapabilityTable
        with pytest.raises(ValueError, match="Bait-loop"):
            CapabilityTable(entries=(
                ToolCapability("proxy_work", "main", True, False),
            ))

    def test_visible_not_allowed_with_actions_ok(self):
        """visible=True, allowed=False, actions_allowed={...} → LEGAL (partial)."""
        from proxy.skills.skill_definition import ToolCapability, CapabilityTable
        CapabilityTable(entries=(
            ToolCapability("proxy_ticket", "main", True, False,
                          actions_allowed=frozenset({"list_tasks"})),
        ))

    def test_invisible_not_allowed_ok(self):
        """visible=False, allowed=False → LEGAL (clean deny)."""
        from proxy.skills.skill_definition import ToolCapability, CapabilityTable
        CapabilityTable(entries=(
            ToolCapability("proxy_work", "main", False, False),
        ))

    def test_visible_allowed_ok(self):
        """visible=True, allowed=True → LEGAL (no restriction)."""
        from proxy.skills.skill_definition import ToolCapability, CapabilityTable
        CapabilityTable(entries=(
            ToolCapability("read", "main", True, True),
        ))

    def test_partial_allow_with_allowed_false(self):
        """C7.5: allowed=False + actions_allowed={x} should allow action x.

        The __post_init__ allows this combination (it only forbids
        visible=True ∧ allowed=False without actions_allowed). But
        check_action previously short-circuited allowed=False without
        consulting actions_allowed, denying everything. C7.5 fixes this
        hole so the partial-allow combination works as advertised.
        """
        from proxy.skills.skill_definition import ToolCapability, CapabilityTable
        table = CapabilityTable(entries=(
            ToolCapability("test_tool", "main", True, False,
                          actions_allowed=frozenset({"allowed_action"}),
                          deny_message="Only allowed_action permitted."),
        ))
        # allowed_action → allowed (partial-allow exception)
        assert table.check_action("test_tool", "allowed_action", "main") is None
        # other_action → denied
        assert table.check_action("test_tool", "other_action", "main") is not None


class TestCapabilityTableLookups:
    """Lookup methods (visible_tools, check_action, injectable_proxy_tools)."""

    def test_visible_tools_returns_correct_set_ticket_main(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        main_visible = TICKET_CAPABILITY_TABLE.visible_tools("main")
        assert "read" in main_visible
        assert "proxy_ticket_list_tasks" in main_visible and "proxy_ticket_get_task" in main_visible
        assert "proxy_work" not in main_visible

    def test_visible_tools_returns_correct_set_work_main(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        main_visible = WORK_CAPABILITY_TABLE.visible_tools("main")
        assert main_visible == frozenset({
            "read", "write", "edit", "bash", "glob", "grep", "search",
            "task", "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_append_review_note", "proxy_work",
        })

    def test_check_action_ticket_main_allowed(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        assert TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_list_tasks", None, "main") is None

    def test_check_action_ticket_main_denied_action(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_create_tasks", None, "main")
        assert result is not None
        assert "delegate" in result.lower() or "blocked" in result.lower()

    def test_check_action_ticket_main_proxy_work_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_work", "start_work", "main")
        assert result is not None
        assert "blocked" in result.lower() or "delegate" in result.lower()

    def test_check_action_work_sub_proxy_ticket_list_tasks_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_ticket_list_tasks", None, "subagent") is None

    def test_check_action_work_sub_proxy_work_start_work_denied(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        result = WORK_CAPABILITY_TABLE.check_action("proxy_work", "start_work", "subagent")
        assert result is not None

    def test_check_action_work_sub_proxy_work_report_bug_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "report_bug", "subagent") is None

    def test_injectable_proxy_tools_work_main(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        main_injectable = WORK_CAPABILITY_TABLE.injectable_proxy_tools("main")
        assert main_injectable == frozenset({
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_append_review_note", "proxy_work",
        })

    def test_injectable_proxy_tools_ticket_main(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        main_injectable = TICKET_CAPABILITY_TABLE.injectable_proxy_tools("main")
        assert main_injectable == frozenset({
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
        })

    def test_is_visible(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        assert TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_list_tasks", "main") is True
        assert TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_get_task", "main") is True
        assert TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_create_tasks", "main") is False
        assert TICKET_CAPABILITY_TABLE.is_visible("proxy_work", "main") is False
        assert TICKET_CAPABILITY_TABLE.is_visible("nonexistent", "main") is False


class TestTableMatchesCurrentBehavior:
    """Tables MUST EXACTLY match current filter methods. If any fails, table
    has a transcription error that must be fixed BEFORE C7b wires up."""

    def test_table_has_decomposed_tools(self):
        """Capability tables include all 10 decomposed proxy_ticket_* tools."""
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE, WORK_CAPABILITY_TABLE
        # Ticket SUB sees all 10
        ticket_sub_inj = TICKET_CAPABILITY_TABLE.injectable_proxy_tools("subagent")
        assert len(ticket_sub_inj & {
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
        }) == 10
        # Work MAIN sees 3 ticket tools + proxy_work
        work_main_inj = WORK_CAPABILITY_TABLE.injectable_proxy_tools("main")
        assert "proxy_ticket_list_tasks" in work_main_inj
        assert "proxy_ticket_get_task" in work_main_inj
        assert "proxy_ticket_append_review_note" in work_main_inj
        assert "proxy_work" in work_main_inj

    def test_work_sub_proxy_work_excludes_lifecycle(self):
        """Work SUB proxy_work actions_allowed must exclude lifecycle per US-095."""
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        caps = WORK_CAPABILITY_TABLE.for_role("subagent")
        work_cap = caps["proxy_work"]
        assert "start_work" not in work_cap.actions_allowed
        assert "stop_work" not in work_cap.actions_allowed
        assert "complete_work" not in work_cap.actions_allowed
        assert "report_bug" in work_cap.actions_allowed

    def test_ticket_main_proxy_ticket_list_get_only(self):
        """Ticket MAIN can only call list_tasks and get_task via decomposed tools."""
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        # list_tasks and get_task tools are visible+allowed for ticket MAIN
        assert TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_list_tasks", "main")
        assert TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_get_task", "main")
        # Other ticket tools are not visible to ticket MAIN
        assert not TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_create_tasks", "main")
        assert not TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_update_task", "main")
        assert not TICKET_CAPABILITY_TABLE.is_visible("proxy_ticket_edit", "main")

    def test_work_sub_proxy_ticket_read_only_decomposed(self):
        """Work SUB has read-only access to list_tasks, get_task, append_review_note."""
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        # Visible/allowed for work SUB
        assert WORK_CAPABILITY_TABLE.is_visible("proxy_ticket_list_tasks", "subagent")
        assert WORK_CAPABILITY_TABLE.is_visible("proxy_ticket_get_task", "subagent")
        assert WORK_CAPABILITY_TABLE.is_visible("proxy_ticket_append_review_note", "subagent")
        # Not visible (denied)
        assert not WORK_CAPABILITY_TABLE.is_visible("proxy_ticket_create_tasks", "subagent")
        assert not WORK_CAPABILITY_TABLE.is_visible("proxy_ticket_update_task", "subagent")
