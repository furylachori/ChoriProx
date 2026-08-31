"""Tests for TicketSkill (Phase B skill system)."""
import logging
import unittest
from unittest.mock import MagicMock, patch

from skills.ticket_skill import TicketSkill


class TestTicketSkillProperties(unittest.TestCase):
    """Tests for TicketSkill property accessors."""

    def setUp(self):
        self.skill = TicketSkill()

    def test_name(self):
        self.assertEqual(self.skill.name, "ticket")

    def test_trigger(self):
        self.assertEqual(self.skill.trigger, "^^ticket")

    def test_aliases(self):
        # Phase 1d: aliases is now a tuple (matches SkillDefinition.aliases field).
        self.assertEqual(self.skill.aliases, ())

    def test_activation(self):
        # Phase 1d: `activation` is no longer a BaseSkill attribute (it lived
        # on the OLD SkillActivation enum, which is kept as a 1a-compat export
        # but is not part of NEW BaseSkill's surface). The canonical ticket
        # activation mode is COMMAND; verify via SkillMeta instead.
        from skills.ticket_skill import SkillActivation
        meta = self.skill.get_meta()
        self.assertEqual(meta.activation, SkillActivation.COMMAND)

    def test_ttl_seconds(self):
        self.assertEqual(self.skill.ttl_seconds, 300)

    def test_meta(self):
        meta = self.skill.get_meta()
        self.assertEqual(meta.name, "ticket")
        self.assertEqual(meta.trigger, "^^ticket")
        self.assertEqual(meta.aliases, ())


class TestTicketSkillParseSubcommand(unittest.TestCase):
    """Tests for parse_subcommand."""

    def setUp(self):
        self.skill = TicketSkill()

    def test_bare_trigger(self):
        """Bare ^^ticket → list with empty arg."""
        action, arg = self.skill.parse_subcommand("^^ticket")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_bare_trigger_with_whitespace(self):
        """^^ticket with trailing whitespace → list with empty arg."""
        action, arg = self.skill.parse_subcommand("^^ticket   ")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_plan_with_description(self):
        """^^ticket <description> → plan with arg."""
        action, arg = self.skill.parse_subcommand("^^ticket fix the login bug")
        self.assertEqual(action, "plan")
        self.assertEqual(arg, "fix the login bug")

    def test_non_ticket_text(self):
        """Non-ticket text → None."""
        action, arg = self.skill.parse_subcommand("hello world")
        self.assertIsNone(action)


class TestTicketSkillSystemPrompt(unittest.TestCase):
    """Tests for system prompt generation."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()
        self.ctx.skill_state = {}
        self.ctx.raw_user_text = ""

    def test_get_system_prompt_returns_planner_prompt(self):
        prompt = self.skill.get_system_prompt(self.ctx)
        self.assertIn("task", prompt.lower())


class TestTicketSkillSubagentPrompt(unittest.TestCase):
    """Tests for subagent prompt generation."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()

    def test_get_subagent_prompt_returns_subagent_prompt(self):
        prompt = self.skill.get_subagent_prompt(self.ctx)
        self.assertIsNotNone(prompt)
        self.assertIn("sub-agent", prompt.lower())

    def test_get_canary(self):
        """Canary must be SKILL variant for ticket skill with inheritance state."""
        canary = self.skill.get_canary(self.ctx)
        self.assertTrue(canary.startswith("[[__SUBAGENT__:SKILL:ticket"))


class TestTicketSkillToolFilter(unittest.TestCase):
    """Tests for tool filtering."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()

    def test_get_tool_filter_returns_read_write_search(self):
        tools = self.skill.get_tool_filter(self.ctx)
        self.assertIn("read", tools)
        self.assertIn("glob", tools)
        self.assertIn("grep", tools)
        self.assertIn("search", tools)
        # Decomposed ticket tools (monolithic proxy_ticket was split per-action).
        self.assertIn("proxy_ticket_list_tasks", tools)
        self.assertIn("proxy_ticket_get_task", tools)

    def test_get_tool_filter_contains_task(self):
        """Ticket mode sub-agents should have task tool."""
        tools = self.skill.get_tool_filter(self.ctx)
        self.assertIn("task", tools)


class TestTicketSkillActivation(unittest.TestCase):
    """Tests for on_activate / on_deactivate."""

    def setUp(self):
        self.skill = TicketSkill()

    def test_on_activate_logs_and_does_not_set_mode(self):
        """on_activate is a logging-only hook.
        Setting active_mode is the responsibility of Agent.set_active_mode."""
        from proxy import state
        ctx = MagicMock()
        ctx.session_id = "sess-12345678"
        ctx.client_hint = ""
        ctx.skill_state = {}
        state.active_workspaces[ctx.session_id] = "test-ws"

        # Should complete without error AND emit the expected activation log.
        with self.assertLogs("skills.ticket", level="INFO") as cm:
            self.skill.on_activate(ctx)

        self.assertTrue(
            any("Ticket skill activated" in m for m in cm.output),
            f"expected 'Ticket skill activated' log line not found in {cm.output}",
        )

    def test_on_deactivate_completes_without_error(self):
        """on_deactivate handles cleanup without clearing the agent's active mode."""
        ctx = MagicMock()
        ctx.session_id = "sess-12345678"
        # Should complete without error and return None.
        result = self.skill.on_deactivate(ctx)
        self.assertIsNone(result)


class TestTicketSkillHandleDirectList(unittest.TestCase):
    """Tests for handle_direct (bare ^^ticket → list tasks)."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()
        self.ctx.session_id = "sess-123"
        self.ctx.client_hint = ""
        from proxy import state
        state.active_workspaces[self.ctx.session_id] = "test-ws"

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_lists_tasks(self, mock_tg_cls):
        """# US-017b: When tasks exist, returns formatted list response."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.is_available.return_value = True
        mock_tg.get_actionable_tasks.return_value = [
            {
                "uuid": "entity-abc123",
                "short_hash": "abc123",
                "name": "Fix login",
                "priority": "high",
                "status": "pending",
                "is_blocked": False,
                "attributes": {},
            },
        ]
        mock_tg_cls.return_value = mock_tg

        result = self.skill.handle_direct(self.ctx, "list", "")

        self.assertIsNotNone(result)
        self.assertIn("Fix login", str(result))

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_empty_tasks(self, mock_tg_cls):
        """When no tasks, returns empty response."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.is_available.return_value = True
        mock_tg.get_actionable_tasks.return_value = []
        mock_tg_cls.return_value = mock_tg

        result = self.skill.handle_direct(self.ctx, "list", "")

        self.assertIsNotNone(result)
        self.assertIn("No active tasks", str(result))

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_store_unavailable(self, mock_tg_cls):
        """When store unavailable, returns error response."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.is_available.return_value = False
        mock_tg_cls.return_value = mock_tg

        result = self.skill.handle_direct(self.ctx, "list", "")

        self.assertIsNotNone(result)
        self.assertIn("unavailable", str(result))


class TestTicketSkillParseSubcommandNew(unittest.TestCase):
    """Tests for new subcommand parsing (done, invalidate, show)."""

    def setUp(self):
        self.skill = TicketSkill()

    def test_parse_done_subcommand(self):
        """Done subcommand with hashes returns (done, <hashes>)."""
        action, arg = self.skill.parse_subcommand("^^ticket done abc12345, def45678")
        self.assertEqual(action, "done")
        self.assertEqual(arg, "abc12345, def45678")

    def test_parse_done_no_hashes(self):
        """Done with no hashes returns (list, "")."""
        action, arg = self.skill.parse_subcommand("^^ticket done ")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_parse_invalidate_subcommand(self):
        """Invalidate subcommand with hashes returns (invalidate, <hashes>)."""
        action, arg = self.skill.parse_subcommand("^^ticket invalidate abcdef01, 12345678")
        self.assertEqual(action, "invalidate")
        self.assertEqual(arg, "abcdef01, 12345678")

    def test_parse_invalidate_no_hashes(self):
        """Invalidate with no hashes returns (list, "")."""
        action, arg = self.skill.parse_subcommand("^^ticket invalidate ")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_parse_bare_hash_show(self):
        """Bare 8-char hash returns (show, <hash>)."""
        action, arg = self.skill.parse_subcommand("^^ticket abc12345")
        self.assertEqual(action, "show")
        self.assertEqual(arg, "abc12345")

    def test_parse_8char_hash_show(self):
        """8-char hex returns (show, <hash>)."""
        action, arg = self.skill.parse_subcommand("^^ticket abc123de")
        self.assertEqual(action, "show")
        self.assertEqual(arg, "abc123de")

    def test_parse_5char_hash_plan(self):
        """5-char string falls through to plan (returns (plan, <string>))."""
        action, arg = self.skill.parse_subcommand("^^ticket abc12")
        self.assertEqual(action, "plan")
        self.assertEqual(arg, "abc12")

    def test_parse_uppercase_hash_show(self):
        """^^ticket A1B2C3D4 (uppercase 8-char hex) → show."""
        action, arg = self.skill.parse_subcommand("^^ticket A1B2C3D4")
        self.assertEqual(action, "show")
        self.assertEqual(arg, "A1B2C3D4")

    def test_parse_7char_hash_plan(self):
        """^^ticket abc1234 (7 hex chars, not 8) → plan."""
        action, arg = self.skill.parse_subcommand("^^ticket abc1234")
        self.assertEqual(action, "plan")
        self.assertEqual(arg, "abc1234")

    def test_parse_9char_hash_plan(self):
        """^^ticket abc123def (9 hex chars, not 8) → plan."""
        action, arg = self.skill.parse_subcommand("^^ticket abc123def")
        self.assertEqual(action, "plan")
        self.assertEqual(arg, "abc123def")

    def test_parse_xml_only_arg_returns_list(self):
        """^^ticket <environment_details>only xml</environment_details> → list."""
        action, arg = self.skill.parse_subcommand(
            "^^ticket <environment_details>only xml</environment_details>"
        )
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_parse_slow_path_xml_wrapped_text(self):
        """Slow path: text wrapped in XML, trigger outside the tags."""
        action, arg = self.skill.parse_subcommand(
            "<environment_details>some xml content</environment_details>^^ticket abc12345"
        )
        self.assertEqual(action, "show")
        self.assertEqual(arg, "abc12345")

    def test_parse_bare_done_returns_list(self):
        """^^ticket done (bare subcommand, no hashes) → list."""
        action, arg = self.skill.parse_subcommand("^^ticket done")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_parse_bare_invalidate_returns_list(self):
        """^^ticket invalidate (bare subcommand, no hashes) → list."""
        action, arg = self.skill.parse_subcommand("^^ticket invalidate")
        self.assertEqual(action, "list")
        self.assertEqual(arg, "")

    def test_parse_6char_hash_show(self):
        """6-char hex no longer matches {8}; falls through to plan."""
        action, arg = self.skill.parse_subcommand("^^ticket abc123")
        self.assertEqual(action, "plan")
        self.assertEqual(arg, "abc123")

    def test_parse_13char_hash_plan(self):
        """13-char hex (one over max) falls through to plan."""
        action, arg = self.skill.parse_subcommand("^^ticket abc123def4567")
        self.assertEqual(action, "plan")
        self.assertEqual(arg, "abc123def4567")

    def test_parse_whitespace_padded_hash_show(self):
        """Whitespace-padded 8-char hash is stripped and matched as show."""
        action, arg = self.skill.parse_subcommand("^^ticket   abc12345  ")
        self.assertEqual(action, "show")
        self.assertEqual(arg, "abc12345")


class TestTicketSkillHandleDirect(unittest.TestCase):
    """Tests for handle_direct."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()
        self.ctx.session_id = "test-sess"
        self.ctx.client_hint = ""
        from proxy import state
        state.active_workspaces[self.ctx.session_id] = "test-ws"

    @patch("skills.ticket_skill._get_task_graph_singleton")
    def test_handle_direct_show_task(self, mock_get_tg):
        """Show <hash> returns task details dict."""
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = {
            "uuid": "entity-fullhash1234",
            "short_hash": "abc123",
            "name": "Fix login",
            "priority": "high",
            "status": "pending",
            "summary": "Authentication is broken",
            "planning_info": "Need OAuth",
            "review_notes": [{"ts": "2026-06-04T10:00:00Z", "note": "Urgent", "author": "test-sess"}],
            "created_at": "2026-05-27T10:00:00Z",
            "blockers": [],
            "is_blocked": False,
            "blocked_count": 0,
            "parent_short_hash": None,
            "parent_name": None,
        }
        mock_tg.get_child_tasks.return_value = []
        mock_get_tg.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "show", "abc123")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("Fix login", content)
        self.assertIn("abc123", content)

    @patch("skills.ticket_skill._get_task_graph_singleton")
    def test_handle_direct_show_ambiguous_hash(self, mock_get_tg):
        """Ambiguous hash raises AmbiguousHashError."""
        from memory.task_graph import AmbiguousHashError
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.side_effect = AmbiguousHashError(3)
        mock_get_tg.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "show", "abc")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("Ambiguous hash", content)
        self.assertIn("3 tasks match", content)

    @patch("skills.ticket_skill._get_task_graph_singleton")
    def test_handle_direct_show_not_found(self, mock_get_tg):
        """Non-existent hash returns error dict."""
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = None
        mock_get_tg.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "show", "zzzzzz")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("No task found", content)

    def test_handle_direct_plan_returns_none(self):
        """Plan action returns None (not a handle_direct case)."""
        result = self.skill.handle_direct(self.ctx, "plan", "build a thing")
        self.assertIsNone(result)

    def test_handle_direct_unknown_action_returns_none(self):
        """Unknown action returns None (fall through to activation)."""
        result = self.skill.handle_direct(self.ctx, "unknown_action", "")
        self.assertIsNone(result)


class TestTicketSkillHandleDirectBatch(unittest.TestCase):
    """Tests for handle_direct batch operations (done, invalidate)."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()
        self.ctx.session_id = "test-sess"
        from proxy import state
        state.active_workspaces[self.ctx.session_id] = "myws"

    def tearDown(self):
        from proxy import state
        state.active_workspaces.clear()

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_done_batch(self, mock_tg_cls):
        """Done <hashes> calls update_tasks_batch_status with plan→active."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc", "ok": True, "status": "complete", "changed": True},
            {"hash": "def", "ok": True, "status": "complete", "changed": False},
        ]
        mock_tg_cls.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "done", "abc, def")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("abc", content)
        self.assertIn("marked complete", content)

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_done_empty_hash_list(self, mock_tg_cls):
        """Done with no hashes returns empty ok result."""
        result = self.skill.handle_direct(self.ctx, "done", "")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("No valid hashes", content)

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_reopen_batch(self, mock_tg_cls):
        """Reopen <hashes> calls update_tasks_batch_status with active→plan."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc", "ok": True, "status": "in_progress", "changed": True},
            {"hash": "def", "ok": False, "status": "not_found", "changed": None},
        ]
        mock_tg_cls.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "reopen", "abc, def")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("abc", content)
        self.assertIn("in_progress", content)

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_invalidate_batch(self, mock_tg_cls):
        """Invalidate <hashes> calls update_tasks_batch_status with active→plan."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc", "ok": True, "status": "invalid", "changed": True},
            {"hash": "def", "ok": False, "status": "not_found", "changed": None},
        ]
        mock_tg_cls.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "invalidate", "abc, def")
        self.assertIsNotNone(result)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("abc", content)
        self.assertIn("invalid", content)

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_done_strips_and_lowercases_hashes(self, mock_tg_cls):
        """Hashes are stripped and lowercased before batch update."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc", "ok": True, "status": "complete", "changed": True},
        ]
        mock_tg_cls.return_value = mock_tg
        self.skill.handle_direct(self.ctx, "done", "  ABC  ,  def  ")
        call_hashes = mock_tg.update_tasks_batch_status.call_args[0][0]
        self.assertEqual(call_hashes, ["abc", "def"])

    @patch("memory.task_graph.TaskGraph")
    def test_handle_direct_done_ambiguous_hash(self, mock_tg_cls):
        """AmbiguousHashError in batch done returns error_response with 409."""
        from memory.task_graph import AmbiguousHashError
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.side_effect = AmbiguousHashError(5)
        mock_tg_cls.return_value = mock_tg
        result = self.skill.handle_direct(self.ctx, "done", "abc, def")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("_status_code"), 409)
        content = result["choices"][0]["message"]["content"]
        self.assertIn("Ambiguous", content)


class TestTicketSkillHandleDirectBatchExtended(unittest.TestCase):
    """Extended tests for handle_direct batch operations."""

    def setUp(self):
        self.skill = TicketSkill()
        self.ctx = MagicMock()
        self.ctx.session_id = "test-sess"
        from proxy import state
        state.active_workspaces[self.ctx.session_id] = "myws"

    def tearDown(self):
        from proxy import state
        state.active_workspaces.clear()
        state.recent_workspaces.clear()
        state.session_last_access.clear()

    def test_handle_direct_done_filters_empty_strings(self):
        """Empty strings between spaces are filtered out."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc", "ok": True, "status": "complete", "changed": True},
            {"hash": "def", "ok": True, "status": "complete", "changed": True},
        ]
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = self.skill.handle_direct(self.ctx, "done", "abc, , ,def")
        # Verify empty strings were filtered — only abc,def passed to batch
        call_args = mock_tg.update_tasks_batch_status.call_args
        hashes = call_args[0][0]  # first positional arg
        self.assertNotIn("", hashes)
        self.assertEqual(hashes, ["abc", "def"])

    def test_handle_direct_done_passes_workspace_id(self):
        """workspace_id is passed through to update_tasks_batch_status."""
        from proxy import state
        state.active_workspaces[self.ctx.session_id] = "my_project"
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc", "ok": True, "status": "complete", "changed": True},
        ]
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            self.skill.handle_direct(self.ctx, "done", "abc")
        call_args = mock_tg.update_tasks_batch_status.call_args
        # workspace_id may be positional or keyword
        all_args = call_args[0] + tuple(call_args[1].values())
        self.assertTrue(any("my_project" in str(a) for a in all_args))

    def test_handle_direct_done_single_hash(self):
        """Single hash still works (not just multiple)."""
        mock_tg = MagicMock()
        mock_tg.__enter__.return_value = mock_tg
        mock_tg.update_tasks_batch_status.return_value = [
            {"hash": "abc123", "ok": True, "status": "complete", "changed": True},
        ]
        with patch("memory.task_graph.TaskGraph", return_value=mock_tg):
            result = self.skill.handle_direct(self.ctx, "done", "abc123")
        self.assertTrue(mock_tg.update_tasks_batch_status.called)


class TestWorkspaceIsolation(unittest.TestCase):
    """Cross-cutting isolation tests — verify workspace scoping.
    
    Tasks in one workspace must not be visible in another.
    """

    def setUp(self):
        from proxy import state
        from skills.ticket_skill import TicketSkill
        state.active_workspaces.clear()
        state.recent_workspaces.clear()
        self.skill = TicketSkill()

    @patch("skills.ticket_skill._get_task_graph_singleton")
    def test_workspace_isolation_task_visibility(self, mock_get_tg):
        """Tasks in workspace A are not visible in workspace B."""
        from proxy import state
        state.active_workspaces["sess_a"] = "project_a"
        state.active_workspaces["sess_b"] = "project_b"
        workspace_id_a = "project_a"
        workspace_id_b = "project_b"

        # Create task in workspace A (mocked)
        mock_tg_a = MagicMock()
        mock_tg_a.is_available.return_value = True
        mock_tg_a.get_actionable_tasks.return_value = [
            {"uuid": "entity-abc123", "short_hash": "abc123", "name": "Task A", "status": "pending", "priority": "medium", "is_blocked": False, "attributes": {}},
        ]
        mock_tg_a.update_tasks_batch_status.return_value = []

        # Create task in workspace B (mocked)
        mock_tg_b = MagicMock()
        mock_tg_b.is_available.return_value = True
        mock_tg_b.get_actionable_tasks.return_value = [
            {"uuid": "entity-def456", "short_hash": "def456", "name": "Task B", "status": "pending", "priority": "medium", "is_blocked": False, "attributes": {}},
        ]

        # First call returns workspace A tasks
        mock_get_tg.return_value = mock_tg_a
        ctx_a = MagicMock()
        ctx_a.session_id = "sess_a"
        result_a = self.skill.handle_direct(ctx_a, "list", "")

        # Second call returns workspace B tasks
        mock_get_tg.return_value = mock_tg_b
        ctx_b = MagicMock()
        ctx_b.session_id = "sess_b"
        result_b = self.skill.handle_direct(ctx_b, "list", "")

        # Verify correct workspace_id was passed to each call
        mock_tg_a.get_actionable_tasks.assert_called()
        call_kwargs_a = mock_tg_a.get_actionable_tasks.call_args
        self.assertIn(workspace_id_a, str(call_kwargs_a))
        mock_tg_b.get_actionable_tasks.assert_called()
        call_kwargs_b = mock_tg_b.get_actionable_tasks.call_args
        self.assertIn(workspace_id_b, str(call_kwargs_b))
        # Verify workspace B task hash NOT in workspace A result
        self.assertNotIn("def456", str(result_a))


class TestTicketSkillActivationLogs:
    """Pytest-style log verification for ticket skill activation."""

    def test_on_activate_logs(self, caplog):
        from unittest.mock import MagicMock
        from proxy import state
        caplog.set_level(logging.INFO)
        skill = TicketSkill()
        ctx = MagicMock()
        ctx.session_id = "sess-12345678"
        ctx.client_hint = ""
        ctx.skill_state = {}
        state.active_workspaces[ctx.session_id] = "test-ws"

        skill.on_activate(ctx)

        assert "Ticket skill activated:" in caplog.text


class TestTicketSkillSkillDefinition:
    """Phase 1d: TicketSkill migrates to NEW BaseSkill (SkillDefinition subclass)."""

    def _to_skill_definition(self, skill: "TicketSkill"):
        """Construct a SkillDefinition for `skill`, holding a TRUSTED token.

        TicketSkill / WorkSkill are in TRUSTED_BUILTIN_SKILLS; constructing
        a SkillDefinition for them requires a held TRUSTED token on the
        current thread (CAT-3 trust gate, R5 HIGH fix 4).
        """
        from proxy.skills.skill_definition import _held_trust_token, TrustToken
        with _held_trust_token(TrustToken.for_trusted_caller()):
            return skill.to_skill_definition()

    def test_ticket_skill_extends_new_base_skill(self):
        """TicketSkill extends NEW BaseSkill from skill_definition (NOT the OLD one).

        Pinned by Phase 1d plan §6 test 1: `from proxy.skills.skill_registry
        import BaseSkill` raises ImportError (Option 1). The NEW BaseSkill
        lives in `proxy.skills.skill_definition`.
        """
        from proxy.skills import skill_definition as new_skill_mod
        assert issubclass(TicketSkill, new_skill_mod.BaseSkill)

    def test_ticket_skill_to_skill_definition_returns_complete_definition(self):
        """TicketSkill().to_skill_definition() returns a SkillDefinition with all 23 fields.

        Plan §6 test 2: `sd.name == "ticket"`, `sd.trigger == "^^ticket"`,
        `sd.is_builtin is True` (the read-only @property), and
        `sd.allowed_tool_names` is a non-empty frozenset.
        """
        from proxy.skills.skill_definition import SkillDefinition
        skill = TicketSkill()
        sd = self._to_skill_definition(skill)

        assert isinstance(sd, SkillDefinition)
        assert sd.name == "ticket"
        assert sd.trigger == "^^ticket"
        assert sd.is_builtin, "ticket is a TRUSTED_BUILTIN_SKILLS"
        assert isinstance(sd.allowed_tool_names, frozenset)
        assert len(sd.allowed_tool_names) > 0

    def test_ticket_skill_get_tool_filter_matches_subclass_tools(self):
        """sd.allowed_tool_names == TicketSkill().get_tool_filter(None) (round-trip).

        Plan §6 test 3 (R1 HIGH-2): the round-trip from subclass method →
        SkillDefinition field is identity-stable.
        """
        skill = TicketSkill()
        sd = self._to_skill_definition(skill)
        assert sd.allowed_tool_names == skill.get_tool_filter(None)

    def test_ticket_skill_sub_agent_proxy_ticket_access_is_full(self):
        """Ticket sub-agents get FULL `proxy_ticket`, NEVER `proxy_work`.

        Plan §6 test 4 (R1 CRIT-3): ticket-mode sub-agents may invoke
        `proxy_ticket` (the ticket-management tool) but NOT `proxy_work`.
        """
        from proxy.skills.skill_definition import ProxyAccessLevel
        skill = TicketSkill()
        sd = self._to_skill_definition(skill)
        assert sd.sub_agent_proxy_ticket_access == ProxyAccessLevel.FULL
        assert sd.sub_agent_proxy_work_access == ProxyAccessLevel.NEVER


class TestTicketSkillShouldDeactivate(unittest.TestCase):
    """TicketSkill.should_deactivate is PER-TURN (clears on stop for BOTH
    main and sub-agents).

    Regression guard: a prior implementation returned False for main agents,
    which leaked ticket mode (and read-only ticket tools via canary injection)
    into later plain-prompt sub-agents.
    """

    def test_main_agent_stop_clears(self):
        skill = TicketSkill()
        ctx = MagicMock(is_subagent=False)
        self.assertTrue(skill.should_deactivate(ctx, "stop"))

    def test_subagent_stop_clears(self):
        skill = TicketSkill()
        ctx = MagicMock(is_subagent=True)
        self.assertTrue(skill.should_deactivate(ctx, "stop"))

    def test_tool_calls_keeps(self):
        skill = TicketSkill()
        ctx = MagicMock(is_subagent=False)
        self.assertFalse(skill.should_deactivate(ctx, "tool_calls"))

    def test_length_keeps_mode_active(self):
        """finish_reason='length' must NOT clear ticket mode (truncated != complete turn).

        Under the tool-loop finish_reason fix, a truncated ('length') memory
        resolution now reaches the client instead of being clobbered to 'stop'.
        Ticket mode is PER-TURN and clears only on 'stop', so a truncated turn
        keeps the mode active for the user to continue. Pins the contract that
        should_deactivate only ever returns True for 'stop'.
        """
        skill = TicketSkill()
        ctx = MagicMock(is_subagent=False)
        self.assertFalse(skill.should_deactivate(ctx, "length"))


if __name__ == "__main__":
    unittest.main()
