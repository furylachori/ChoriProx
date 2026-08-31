"""Integration tests for WorkspaceSkill.

US-099: Workspace mode is NOT a real mode. The `^^workspace` command
set is implemented as atomic one-shot direct commands handled in
`handle_direct()`. There is no LLM invocation, no skill activation,
no system prompt, no tool filter. This file tests the real code path
(parse + handle_direct), not the dead-mode scaffold.

The TestAC4_* classes and TestWorkspaceToolFilter class that
previously lived here were testing the dead workspace-mode scaffold
(get_tool_filter, on_activate, on_deactivate, get_system_prompt).
They were removed in the US-099 cleanup.
"""
from __future__ import annotations

import logging
import threading
import time
from unittest.mock import patch

import pytest

from proxy import state
from proxy.agent.agent_registry import registry as agent_registry
from proxy.skills.skill_registry import SkillContext
from proxy.workspace_state import workspace_store
from skills.workspace import WorkspaceSkill


@pytest.fixture(autouse=True)
def reset_workspace_state():
    """Reset workspace state before each test.

    Also resets the workspace_store catalog cache and disables the
    Neo4j-driven recovery function for the duration of each test.

    Why: tests in this file set per-session state (active_workspaces,
    recent_workspaces) directly and do NOT rely on recovery. If the
    catalog cache or recovery is left from a previous test/run, it
    re-populates from Neo4j, which accumulates workspace names written
    by prior test runs (e.g. ``test_handle_direct_create_skips_typo_guard``
    persists ``lna-ui`` to Neo4j). That stale catalog poisons the
    typo-guard's "existing" set, so the typo guard never fires and
    ``test_handle_direct_switch_typo_blocked`` fails with
    "Switched to workspace: lna-ui" instead of "Did you mean: luna-ui?".
    """
    saved_recovery_fn = workspace_store._recovery_fn
    state.active_workspaces.clear()
    state.recent_workspaces.clear()
    # Reset the global catalog cache so get_global_catalog() returns
    # a clean ["default"] instead of whatever Neo4j has accumulated.
    workspace_store._catalog_cache = [state.DEFAULT_WORKSPACE]
    workspace_store._catalog_cache_ts = time.time()
    # Disable Neo4j recovery for the duration of the test so the
    # cold-session branch of get_active() doesn't re-fetch the stale
    # catalog from Neo4j and overwrite our clean cache.
    workspace_store._recovery_fn = None
    agent_registry._agents.clear()
    agent_registry._agents_by_id.clear()
    agent_registry._lru.clear()
    yield
    state.active_workspaces.clear()
    state.recent_workspaces.clear()
    workspace_store._catalog_cache = [state.DEFAULT_WORKSPACE]
    workspace_store._catalog_cache_ts = 0.0
    workspace_store._recovery_fn = saved_recovery_fn
    agent_registry._agents.clear()
    agent_registry._agents_by_id.clear()
    agent_registry._lru.clear()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    session_id = f"test_session_{time.time_ns()}"

    class MockAgent:
        def __init__(self, session_id):
            self.active_mode = None
            self._ticket_mode_ts = None
            self._deactivation_callbacks = []
            self._mode_lock = threading.RLock()
            self.session_id = session_id
            self.agent_id = f"agent_{session_id}"
            self.context_limit_override = None
            self._messages = []

        def set_active_mode(self, mode_state, expected_mode=None):
            self.active_mode = mode_state
            return True

        def clear_active_mode(self, expected_mode=None):
            self.active_mode = None
            self._ticket_mode_ts = None
            return True

        def clear_ticket_mode(self):
            """Legacy wrapper. Mirrors Agent.clear_ticket_mode()."""
            return self.clear_active_mode()

        def register_deactivation_callback(self, cb):
            self._deactivation_callbacks.append(cb)

        def snapshot_mode(self):
            return self.active_mode

        def snapshot_mode_and_ticket_ts(self):
            return self.active_mode, self._ticket_mode_ts

        def get_mode_name(self):
            return self.active_mode.name if self.active_mode else None

        def is_mode_expired(self):
            return False

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

        @property
        def is_ticket_mode(self):
            mode = self.active_mode
            if mode is None:
                return self._ticket_mode_ts is not None
            return getattr(mode, "name", None) == "ticket"

        @property
        def is_skill_mode(self):
            return self.active_mode is not None

        @property
        def is_workspace_mode(self):
            mode = self.active_mode
            if mode is None:
                return False
            return getattr(mode, "name", None) == "workspace"

        def set_context_limit_override(self, tokens):
            self.context_limit_override = tokens

        def get_context_limit_override(self):
            return self.context_limit_override

    mock_agent_obj = MockAgent(session_id)

    with patch('proxy.agent.agent_registry.registry.get_or_load', return_value=mock_agent_obj):
        agent = agent_registry.get_or_load(session_id)
    return agent


@pytest.fixture
def skill():
    """Create a WorkspaceSkill instance."""
    return WorkspaceSkill()


@pytest.fixture
def ctx(mock_agent, skill):
    """Create a SkillContext for testing."""
    return SkillContext(
        session_id=mock_agent.session_id,
        agent_id=mock_agent.agent_id,
        messages=[],
        is_subagent=False,
        raw_user_text="^^workspace test",
        active_workspace=state.DEFAULT_WORKSPACE,
    )


# =============================================================================
# Tool Filter Tests
# =============================================================================
# US-099: Workspace mode is NOT a real mode. There is no get_tool_filter
# to test — ^^workspace subcommands are atomic one-shot direct commands
# handled in handle_direct() (see TestWorkspaceSubcommands above).
# The TestWorkspaceToolFilter class was removed along with the dead
# workspace-mode scaffold.


# =============================================================================
# Parse Subcommand Tests
# =============================================================================

class TestWorkspaceParseSubcommand:
    """Test workspace subcommand parsing."""

    def test_parse_bare_trigger(self, skill):
        """Test parsing bare ^^workspace."""
        action, arg = skill.parse_subcommand("^^workspace")
        assert action == "status"
        assert arg == ""

    def test_parse_switch(self, skill):
        """Test parsing ^^workspace <name>."""
        action, arg = skill.parse_subcommand("^^workspace myws")
        assert action == "switch"
        assert arg == "myws"

    def test_parse_switch_with_framework_reminder(self, skill):
        """^^ws <name> + appended <system-reminder> block: name parses correctly.

        The framework block is stripped from the EPHEMERAL command-parse text
        (canary-safe; message data untouched). Regression for the crash where
        the opencode mode-change reminder became the workspace name."""
        action, arg = skill.parse_subcommand(
            "^^ws myname<system-reminder>\n"
            "Your operational mode has changed from plan to build.\n"
            "</system-reminder>"
        )
        assert action == "switch"
        assert arg == "myname"

    def test_parse_slow_path_reminder_leading_whitespace(self, skill):
        """Slow parse path: leading whitespace defeats the fast-path startswith
        check, so parse falls through to extract_command_text (workspace.py:160).
        The reminder must still be stripped on that path too.

        Verified slow-path trigger: _normalize_text (skill_registry.py:52) only
        strips zero-width chars + NFKC — it does NOT strip leading whitespace,
        so text.startswith(self.trigger) / startswith(alias) both fail and the
        fast path (workspace.py:141-154) is skipped. extract_command_text at
        line 160 then strips both the whitespace and the reminder block, and the
        re-check at line 166 re-resolves ^^ws. This is a distinct branch from
        test_parse_switch_with_framework_reminder (no leading ws -> fast path)."""
        action, arg = skill.parse_subcommand(
            "  ^^ws myname<system-reminder>mode change</system-reminder>"
        )
        assert action == "switch"
        assert arg == "myname"

    def test_parse_reminder_only_no_name_is_status(self, skill):
        """Bare trigger + reminder (no name) -> status, no invalid-name crash."""
        action, arg = skill.parse_subcommand(
            "^^ws<system-reminder>mode change</system-reminder>"
        )
        assert action == "status"
        assert arg == ""

    def test_parse_leave(self, skill):
        """Test parsing ^^workspace leave."""
        action, arg = skill.parse_subcommand("^^workspace leave")
        assert action == "leave"
        assert arg == ""

    def test_parse_recent(self, skill):
        """Test parsing ^^workspace recent."""
        action, arg = skill.parse_subcommand("^^workspace recent")
        assert action == "recent"

    def test_parse_list(self, skill):
        """Test parsing ^^workspace list."""
        action, arg = skill.parse_subcommand("^^workspace list")
        assert action == "list"

    def test_parse_help(self, skill):
        """Test parsing ^^workspace help."""
        action, arg = skill.parse_subcommand("^^workspace help")
        assert action == "help"

    @pytest.mark.integration
    def test_parse_ws_alias_integration(self, skill):
        """^^ws is recognized as an alias for ^^workspace."""
        action, arg = skill.parse_subcommand("^^ws")
        assert action == "status"
        assert arg == ""

    @pytest.mark.integration
    def test_parse_ws_alias_leave_integration(self, skill):
        """^^ws leave is parsed the same as ^^workspace leave."""
        action, arg = skill.parse_subcommand("^^ws leave")
        assert action == "leave"
        assert arg == ""


# =============================================================================
# HandleDirect Tests
# =============================================================================

class TestHandleDirect:
    """Tests for handle_direct (Phase 3)."""

    def test_handle_direct_status(self, skill, ctx):
        """US-072: handle_direct("status", "") returns current workspace."""
        state.active_workspaces[ctx.session_id] = "my_project"
        result = skill.handle_direct(ctx, "status", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Current workspace:" in content
        assert "my_project" in content

    def test_handle_direct_list(self, skill, ctx):
        """US-072: handle_direct("list", "") returns workspace list."""
        state.recent_workspaces[ctx.session_id] = ["ws1", "ws2"]
        state.active_workspaces[ctx.session_id] = "ws1"
        result = skill.handle_direct(ctx, "list", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Workspaces:" in content

    def test_handle_direct_recent(self, skill, ctx):
        """US-072: handle_direct("recent", "") returns MRU list."""
        state.recent_workspaces[ctx.session_id] = ["a", "b"]
        result = skill.handle_direct(ctx, "recent", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Recent workspaces:" in content

    def test_handle_direct_help(self, skill, ctx):
        """US-072: handle_direct("help", "") returns command reference."""
        result = skill.handle_direct(ctx, "help", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Workspace Commands" in content

    def test_handle_direct_leave_not_default(self, skill, ctx):
        """handle_direct("leave", "") when in non-default workspace → deactivates."""
        state.active_workspaces[ctx.session_id] = "some_project"
        result = skill.handle_direct(ctx, "leave", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Returned to default workspace" in content
        assert ctx.session_id not in state.active_workspaces

    def test_handle_direct_leave_already_default(self, skill, ctx):
        """handle_direct("leave", "") when already at default → no-op."""
        state.active_workspaces[ctx.session_id] = state.DEFAULT_WORKSPACE
        result = skill.handle_direct(ctx, "leave", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Already at default workspace" in content
        assert state.active_workspaces.get(ctx.session_id) == state.DEFAULT_WORKSPACE


# =============================================================================
# Concurrent access — workspace isolation across sessions
# =============================================================================

def test_workspace_isolation_across_sessions():
    """Different sessions cannot see each other's workspaces."""
    from proxy import state
    try:
        state.active_workspaces["sess-a"] = "project-a"
        state.active_workspaces["sess-b"] = "project-b"
        assert state.active_workspaces["sess-a"] == "project-a"
        assert state.active_workspaces["sess-b"] == "project-b"
    finally:
        state.active_workspaces.pop("sess-a", None)
        state.active_workspaces.pop("sess-b", None)



# =============================================================================
# Phase 4: ^^workspace <name> local mutation, ^^workspace create, ^^workspace forget
# =============================================================================

class TestWorkspaceParseSubcommandPhase4:
    """Parse-level tests for create/forget subcommands (Phase 4)."""

    def test_parse_create_no_name(self, skill):
        """^^workspace create (no name) parses to ('create', '')."""
        action, arg = skill.parse_subcommand("^^workspace create")
        assert action == "create"
        assert arg == ""

    def test_parse_create_with_name(self, skill):
        """^^workspace create <name> parses to ('create', <name>)."""
        action, arg = skill.parse_subcommand("^^workspace create luna-ui")
        assert action == "create"
        assert arg == "luna-ui"

    def test_parse_forget_no_name(self, skill):
        """^^workspace forget (no name) parses to ('forget', '')."""
        action, arg = skill.parse_subcommand("^^workspace forget")
        assert action == "forget"
        assert arg == ""

    def test_parse_forget_with_name(self, skill):
        """^^workspace forget <name> parses to ('forget', <name>)."""
        action, arg = skill.parse_subcommand("^^workspace forget luna-ui")
        assert action == "forget"
        assert arg == "luna-ui"

    @pytest.mark.integration
    def test_parse_ws_alias_create(self, skill):
        """^^ws create <name> parses the same as ^^workspace create <name>."""
        action, arg = skill.parse_subcommand("^^ws create luna-ui")
        assert action == "create"
        assert arg == "luna-ui"

    @pytest.mark.integration
    def test_parse_ws_alias_forget(self, skill):
        """^^ws forget <name> parses the same as ^^workspace forget <name>."""
        action, arg = skill.parse_subcommand("^^ws forget luna-ui")
        assert action == "forget"
        assert arg == "luna-ui"


class TestHandleDirectSwitchCreateForget:
    """Handler-level tests for switch/create/forget (Phase 4)."""

    def test_handle_direct_switch_unknown_workspace_returns_404(self, skill, ctx):
        """^^workspace brand-new with empty MRU returns 404 (switch no longer auto-creates)."""
        result = skill.handle_direct(ctx, "switch", "brand-new")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "does not exist" in content
        assert "brand-new" in content
        assert "^^workspace create" in content
        assert result["_status_code"] == 404
        assert state.active_workspaces.get(ctx.session_id) is None
        assert "brand-new" not in state.recent_workspaces.get(ctx.session_id, [])

    def test_handle_direct_switch_existing_workspace(self, skill, ctx):
        """^^workspace luna-ui with luna-ui in MRU switches to it."""
        state.recent_workspaces[ctx.session_id] = ["luna-ui"]
        state.active_workspaces[ctx.session_id] = state.DEFAULT_WORKSPACE
        result = skill.handle_direct(ctx, "switch", "luna-ui")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "luna-ui" in content
        assert "Switched" in content
        assert state.active_workspaces.get(ctx.session_id) == "luna-ui"

    def test_handle_direct_switch_already_active(self, skill, ctx):
        """US-068: ^^workspace luna-ui when active is luna-ui is idempotent."""
        state.active_workspaces[ctx.session_id] = "luna-ui"
        result = skill.handle_direct(ctx, "switch", "luna-ui")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Already in workspace" in content
        assert "luna-ui" in content

    def test_handle_direct_switch_invalid_name(self, skill, ctx):
        """US-067: ^^workspace with invalid characters returns 400 and does not mutate state."""
        result = skill.handle_direct(ctx, "switch", "bad name!")
        assert result is not None
        assert result["_status_code"] == 400
        content = result["choices"][0]["message"]["content"]
        assert "Invalid workspace name" in content
        assert state.active_workspaces.get(ctx.session_id) is None

    def test_handle_direct_switch_typo_blocked(self, skill, ctx):
        """US-071: ^^workspace lna-ui with luna-ui in MRU is blocked by typo guard."""
        state.recent_workspaces[ctx.session_id] = ["luna-ui"]
        result = skill.handle_direct(ctx, "switch", "lna-ui")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "lna-ui" in content
        assert "Did you mean" in content
        assert "luna-ui" in content
        assert "^^workspace create" in content
        # state NOT mutated
        assert "lna-ui" not in state.recent_workspaces.get(ctx.session_id, [])

    def test_handle_direct_switch_unknown_no_similar_returns_404(self, skill, ctx):
        """^^workspace totally-unique-xyz with empty MRU returns 404 (switch no longer auto-creates)."""
        result = skill.handle_direct(ctx, "switch", "totally-unique-xyz")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "does not exist" in content
        assert "totally-unique-xyz" in content
        assert "^^workspace create" in content
        assert result["_status_code"] == 404
        assert state.active_workspaces.get(ctx.session_id) is None

    def test_handle_direct_switch_no_arg_shows_status(self, skill, ctx):
        """^^workspace (no name) shows current workspace."""
        result = skill.handle_direct(ctx, "switch", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Current workspace" in content
        assert state.DEFAULT_WORKSPACE in content

    def test_handle_direct_switch_short_circuits_llm(self, skill, ctx):
        """US-066: ^^workspace luna-ui returns non-None and does NOT touch agent.active_mode.

        THE HEADLINE TEST: ^^workspace luna-ui returns non-None and does NOT touch agent.active_mode.
        """
        # Pre-populate MRU so typo guard doesn't block
        state.recent_workspaces[ctx.session_id] = ["luna-ui"]
        result = skill.handle_direct(ctx, "switch", "luna-ui")
        # Must return a non-None dict (signals dispatch to skip LLM)
        assert result is not None
        # Response is in command format
        assert "choices" in result
        assert "model" in result
        # agent.active_mode was NOT touched (no LLM activation, no skill mode)
        # In this test we don't have an agent, but the call returning non-None
        # is what dispatch.py uses to skip _activate_skill. Verify response.
        assert "luna-ui" in result["choices"][0]["message"]["content"]

    def test_handle_direct_create_valid(self, skill, ctx):
        """^^workspace create foo creates and switches."""
        result = skill.handle_direct(ctx, "create", "foo")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "foo" in content
        assert "Created" in content or "Switched" in content
        assert state.active_workspaces.get(ctx.session_id) == "foo"

    def test_handle_direct_create_no_arg(self, skill, ctx):
        """^^workspace create (no name) returns 400."""
        result = skill.handle_direct(ctx, "create", "")
        assert result is not None
        assert result["_status_code"] == 400
        content = result["choices"][0]["message"]["content"]
        assert "Usage" in content
        assert "create" in content

    def test_handle_direct_create_skips_typo_guard(self, skill, ctx):
        """US-069: ^^workspace create lna-ui with luna-ui in MRU still creates (bypasses guard)."""
        state.recent_workspaces[ctx.session_id] = ["luna-ui"]
        result = skill.handle_direct(ctx, "create", "lna-ui")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "lna-ui" in content
        assert "Did you mean" not in content
        assert state.active_workspaces.get(ctx.session_id) == "lna-ui"

    def test_handle_direct_create_reserved_name(self, skill, ctx):
        """US-067: ^^workspace create create returns 400 (reserved keyword)."""
        result = skill.handle_direct(ctx, "create", "create")
        assert result is not None
        assert result["_status_code"] == 400
        content = result["choices"][0]["message"]["content"]
        assert "Invalid workspace name" in content
        assert state.active_workspaces.get(ctx.session_id) is None

    def test_handle_direct_create_already_active(self, skill, ctx):
        """US-068: ^^workspace create foo when active is foo is idempotent."""
        state.active_workspaces[ctx.session_id] = "luna-ui"
        result = skill.handle_direct(ctx, "create", "luna-ui")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Already in workspace" in content
        assert "luna-ui" in content

    def test_handle_direct_forget_existing(self, skill, ctx):
        """^^workspace forget luna-ui with luna-ui in MRU and not active removes it."""
        state.recent_workspaces[ctx.session_id] = ["luna-ui", "other"]
        state.active_workspaces[ctx.session_id] = "other"
        result = skill.handle_direct(ctx, "forget", "luna-ui")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Forgot workspace" in content
        assert "luna-ui" in content
        assert "luna-ui" not in state.recent_workspaces.get(ctx.session_id, [])
        # Active workspace unchanged
        assert state.active_workspaces.get(ctx.session_id) == "other"

    def test_handle_direct_forget_not_in_mru(self, skill, ctx):
        """^^workspace forget nonexistent returns friendly not-found message."""
        result = skill.handle_direct(ctx, "forget", "nonexistent")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "not in your recent list" in content
        assert "nonexistent" in content

    def test_handle_direct_forget_active_workspace(self, skill, ctx):
        """^^workspace forget luna-ui when luna-ui is active returns 400."""
        state.active_workspaces[ctx.session_id] = "luna-ui"
        state.recent_workspaces[ctx.session_id] = ["luna-ui"]
        result = skill.handle_direct(ctx, "forget", "luna-ui")
        assert result is not None
        assert result["_status_code"] == 400
        content = result["choices"][0]["message"]["content"]
        assert "Cannot forget" in content
        assert "luna-ui" in content
        # state unchanged
        assert "luna-ui" in state.recent_workspaces.get(ctx.session_id, [])
        assert state.active_workspaces.get(ctx.session_id) == "luna-ui"

    def test_handle_direct_forget_no_arg(self, skill, ctx):
        """^^workspace forget (no name) returns 400."""
        result = skill.handle_direct(ctx, "forget", "")
        assert result is not None
        assert result["_status_code"] == 400
        content = result["choices"][0]["message"]["content"]
        assert "Usage" in content
        assert "forget" in content

    def test_handle_direct_forget_reserved_name(self, skill, ctx):
        """US-067: ^^workspace forget default returns 400 (reserved)."""
        result = skill.handle_direct(ctx, "forget", "default")
        assert result is not None
        assert result["_status_code"] == 400
        content = result["choices"][0]["message"]["content"]
        assert "Invalid workspace name" in content


class TestWorkspaceCleanupSubcommand:
    """Tests for `^^workspace cleanup` — delete empty workspaces from the catalog."""

    def test_parse_cleanup(self, skill):
        action, arg = skill.parse_subcommand("^^workspace cleanup")
        assert action == "cleanup"
        assert arg == ""

    def test_cleanup_reserved_name_blocked(self, skill, ctx):
        """`cleanup` is reserved — switch/create with that name is rejected."""
        result = skill.handle_direct(ctx, "switch", "cleanup")
        assert result is not None
        assert result["_status_code"] == 400

    def test_cleanup_deletes_empty_workspaces(self, skill, ctx):
        """Cleanup calls cleanup_empty_workspaces and reports the deleted count."""
        with patch(
            "memory.workspace_persistence.cleanup_empty_workspaces",
            return_value=(2, ["alpha", "beta"]),
        ) as mock_cleanup:
            result = skill.handle_direct(ctx, "cleanup", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "Cleaned up 2" in content
        assert "alpha" in content and "beta" in content
        mock_cleanup.assert_called_once()
        # default workspace is not in the protect list when active == default
        protect_arg = mock_cleanup.call_args[0][0]
        assert protect_arg == []

    def test_cleanup_protects_active_workspace(self, skill, ctx):
        """The active workspace is passed in the protect list."""
        state.active_workspaces[ctx.session_id] = "choriprox"
        with patch(
            "memory.workspace_persistence.cleanup_empty_workspaces",
            return_value=(1, ["foo"]),
        ) as mock_cleanup:
            result = skill.handle_direct(ctx, "cleanup", "")
        assert result is not None
        protect_arg = mock_cleanup.call_args[0][0]
        assert protect_arg == ["choriprox"]

    def test_cleanup_no_empty_workspaces(self, skill, ctx):
        """When nothing is eligible, report no workspaces to clean up."""
        with patch(
            "memory.workspace_persistence.cleanup_empty_workspaces",
            return_value=(0, []),
        ):
            result = skill.handle_direct(ctx, "cleanup", "")
        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "No empty workspaces" in content

    def test_cleanup_invalidates_catalog_cache(self, skill, ctx):
        """Cleanup clears the in-memory catalog cache so the next list is fresh."""
        import time as _t
        workspace_store._catalog_cache = ["stale"]
        workspace_store._catalog_cache_ts = _t.time()
        with patch(
            "memory.workspace_persistence.cleanup_empty_workspaces",
            return_value=(1, ["foo"]),
        ):
            skill.handle_direct(ctx, "cleanup", "")
        assert workspace_store._catalog_cache == []
        assert workspace_store._catalog_cache_ts == 0.0

    def test_cleanup_empty_workspaces_function(self):
        """Direct test of cleanup_empty_workspaces with a recording executor."""
        from memory.workspace_persistence import cleanup_empty_workspaces

        class _RecExecutor:
            def __init__(self):
                self.reads = []
                self.writes = []

            def execute_read(self, q, p=None):
                self.reads.append((q, p))
                # eligibility read returns two empty workspaces
                if "RETURN w.workspace_id AS name" in q:
                    return [{"name": "alpha"}, {"name": "beta"}]
                return []

            def execute_write(self, q, p=None):
                self.writes.append((q, p))
                return [{"deleted": 2}]

        with patch(
            "memory.workspace_persistence._get_executor",
            return_value=_RecExecutor(),
        ):
            count, names = cleanup_empty_workspaces(protect=["choriprox"])
        assert count == 2
        assert names == ["alpha", "beta"]

    def test_cleanup_empty_workspaces_none_when_no_neo4j(self):
        """Returns (0, []) when Neo4j is unavailable."""
        from memory.workspace_persistence import cleanup_empty_workspaces
        with patch(
            "memory.workspace_persistence._get_executor",
            return_value=None,
        ):
            count, names = cleanup_empty_workspaces(protect=[])
        assert (count, names) == (0, [])


class TestWorkspaceStateStoreForget:
    """Direct tests for the new WorkspaceStateStore.forget method."""

    def test_workspace_store_forget_returns_true_when_present(self):
        """US-070: forget returns True when workspace is in MRU."""
        import threading
        from proxy.workspace_state import WorkspaceStateStore
        store = WorkspaceStateStore(threading.RLock())
        store.activate("s1", "ws-a")
        assert store.forget("s1", "ws-a") is True
        assert "ws-a" not in store.get_recent("s1")

    def test_workspace_store_forget_returns_false_when_absent(self):
        """US-070: forget returns False when workspace is not in MRU."""
        import threading
        from proxy.workspace_state import WorkspaceStateStore
        store = WorkspaceStateStore(threading.RLock())
        assert store.forget("s1", "ws-a") is False

    def test_workspace_store_forget_does_not_touch_active(self):
        """US-070: forget removes from MRU but leaves _active unchanged."""
        import threading
        from proxy.workspace_state import WorkspaceStateStore
        store = WorkspaceStateStore(threading.RLock())
        store.activate("s1", "ws-a")
        store.activate("s1", "ws-b")
        # ws-b is active
        assert store.get_active("s1") == "ws-b"
        result = store.forget("s1", "ws-a")
        assert result is True
        # _active unchanged
        assert store.get_active("s1") == "ws-b"
        # ws-a gone from MRU
        assert "ws-a" not in store.get_recent("s1")


# =============================================================================
# R1-HIGH-2: WorkspaceSkill.to_skill_definition() must not raise ValueError
# =============================================================================

class TestWorkspaceSkillDefinition:
    """to_skill_definition() compatibility with NEW BaseSkill (R1-HIGH-2).

    The `aliases` field on SkillDefinition is validated against
    `^[a-z][a-z0-9_-]*$` (skill_definition.py:497). The trigger-prefix
    form ("^^ws") does NOT match that regex, so `aliases=("^^ws",)` would
    raise ValueError when WorkspaceSkill.to_skill_definition() is called.
    The class-level attribute is now `_parser_aliases` (NOT `aliases`), so
    the SkillDefinition-level validator sees the BaseSkill default
    `aliases=()` while the parser-only code paths still resolve the
    trigger prefix aliases.
    """

    def test_to_skill_definition_succeeds(self, skill, ctx):
        """US: WorkspaceSkill.to_skill_definition() should not raise."""
        from skills.workspace import WorkspaceSkill
        ws = WorkspaceSkill()
        # Must not raise ValueError from the alias regex
        sd = ws.to_skill_definition()
        assert sd.name == "workspace"
        # SkillDefinition-level aliases is empty (^^ws is stored separately)
        assert sd.aliases == ()

    def test_parser_aliases_separate_from_skill_definition_aliases(self):
        """The trigger-prefix alias ("^^ws") is NOT in SkillDefinition.aliases.

        The class-level `_parser_aliases` tuple holds the parser-only
        prefix aliases; they must NOT bleed into the SkillDefinition
        identity, which is validated against the regex.
        """
        from skills.workspace import WorkspaceSkill
        ws = WorkspaceSkill()
        sd = ws.to_skill_definition()
        # "^^ws" is the parser-only alias and must not appear in
        # SkillDefinition.aliases (would fail the regex validator).
        assert "^^ws" not in sd.aliases
        # But the parser itself still resolves the alias.
        assert "^^ws" in ws._parser_aliases

    def test_parse_subcommand_still_resolves_ws_alias(self):
        """The parser still resolves ^^ws after the rename to _parser_aliases.

        Pins that the parser-only code paths (parse_subcommand) read
        self._parser_aliases and continue to match the trigger prefix.
        """
        from skills.workspace import WorkspaceSkill
        ws = WorkspaceSkill()
        action, arg = ws.parse_subcommand("^^ws myws")
        assert action == "switch"
        assert arg == "myws"
