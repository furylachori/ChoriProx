"""C7b Phase 3: Guard parity tests.

For every skill role tool action that the current guards handle, verify
table.check_action produces the same verdict. These tests are written
BEFORE the guard changes (Phase 4) - they must pass BEFORE (proving the
table matches) and AFTER (proving the wiring preserves behavior).
"""


class TestGuardTableParity:
    """The builtin tables must produce the SAME verdicts as the current
    hardcoded guards for every test case. If any of these tests fails,
    the table has a transcription error that must be fixed BEFORE the
    guard changes (Phase 4)."""

    # --- Ticket MAIN proxy_ticket ---

    def test_ticket_main_proxy_ticket_list_tasks_allowed(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        assert TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_list_tasks", None, "main") is None

    def test_ticket_main_proxy_ticket_get_task_allowed(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        assert TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_get_task", None, "main") is None

    def test_ticket_main_proxy_ticket_create_tasks_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_create_tasks", None, "main")
        assert result is not None
        assert "read-only" in result.lower() or "delegate" in result.lower()

    def test_ticket_main_proxy_ticket_update_task_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_update_task", None, "main")
        assert result is not None

    def test_ticket_main_proxy_ticket_append_review_note_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        # Note: ticket MAIN proxy_ticket is MORE restrictive than work MAIN -
        # it allows ONLY {list_tasks, get_task}, not append_review_note.
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_append_review_note", None, "main")
        assert result is not None

    # --- Ticket MAIN proxy_work ---

    def test_ticket_main_proxy_work_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_work", "stop_work", "main")
        assert result is not None
        assert "blocked" in result.lower() or "delegate" in result.lower()

    def test_ticket_main_proxy_work_start_work_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_work", "start_work", "main")
        assert result is not None

    # --- Ticket SUB proxy_ticket (FULL - all 10 actions allowed) ---

    def test_ticket_sub_proxy_ticket_full(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        # SUB has FULL access - create_tasks allowed
        assert TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_create_tasks", None, "subagent") is None

    def test_ticket_sub_proxy_ticket_list_tasks_allowed(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        assert TICKET_CAPABILITY_TABLE.check_action("proxy_ticket_list_tasks", None, "subagent") is None

    # --- Ticket SUB proxy_work (denied - NEVER) ---

    def test_ticket_sub_proxy_work_denied(self):
        from proxy.skills.skill_definition import TICKET_CAPABILITY_TABLE
        result = TICKET_CAPABILITY_TABLE.check_action("proxy_work", "start_work", "subagent")
        assert result is not None

    # --- Work MAIN proxy_ticket (READ_ONLY: {list_tasks, get_task, append_review_note}) ---

    def test_work_main_proxy_ticket_list_tasks_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_ticket_list_tasks", None, "main") is None

    def test_work_main_proxy_ticket_get_task_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_ticket_get_task", None, "main") is None

    def test_work_main_proxy_ticket_append_review_note_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_ticket_append_review_note", None, "main") is None

    def test_work_main_proxy_ticket_create_tasks_denied(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        result = WORK_CAPABILITY_TABLE.check_action("proxy_ticket_create_tasks", None, "main")
        assert result is not None

    # --- Work MAIN proxy_work (FULL - all 7 actions allowed) ---

    def test_work_main_proxy_work_start_work_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "start_work", "main") is None

    def test_work_main_proxy_work_stop_work_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "stop_work", "main") is None

    def test_work_main_proxy_work_report_bug_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "report_bug", "main") is None

    # --- Work SUB proxy_work (US-095: lifecycle excluded, bug actions allowed) ---

    def test_work_sub_proxy_work_start_work_denied(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        result = WORK_CAPABILITY_TABLE.check_action("proxy_work", "start_work", "subagent")
        assert result is not None

    def test_work_sub_proxy_work_stop_work_denied(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        result = WORK_CAPABILITY_TABLE.check_action("proxy_work", "stop_work", "subagent")
        assert result is not None

    def test_work_sub_proxy_work_complete_work_denied(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        result = WORK_CAPABILITY_TABLE.check_action("proxy_work", "complete_work", "subagent")
        assert result is not None

    def test_work_sub_proxy_work_report_bug_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "report_bug", "subagent") is None

    def test_work_sub_proxy_work_review_bug_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "review_bug", "subagent") is None

    def test_work_sub_proxy_work_handoff_task_allowed(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_work", "handoff_task", "subagent") is None

    # --- Work SUB proxy_ticket (READ_ONLY same as Work MAIN) ---

    def test_work_sub_proxy_ticket_read_only(self):
        from proxy.skills.skill_definition import WORK_CAPABILITY_TABLE
        assert WORK_CAPABILITY_TABLE.check_action("proxy_ticket_list_tasks", None, "subagent") is None
        assert WORK_CAPABILITY_TABLE.check_action("proxy_ticket_create_tasks", None, "subagent") is not None

