"""Unit tests for the restored `^^workspace` / `^^ws` commands (R2-Issue 3).

These tests exercise the COMMAND_REGISTRY dispatch path — the same path
that `try_command` (priority 4) uses in DispatchStage. They verify that
the commands are registered, that the alias works, and that the
underlying WorkspaceSkill.handle_direct is reached with the correct
action/arg. They do NOT spin up the Flask app; they call
`dispatch_command` directly with a `CommandContext`, mirroring
`tests/unit/test_commands_v1.py`.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from commands import (
    COMMAND_HELP,
    COMMAND_REGISTRY,
    CommandContext,
    dispatch_command,
)
from proxy import state
from proxy.workspace_state import workspace_store


@pytest.fixture(autouse=True)
def reset_workspace_state():
    """Reset workspace_store + state dicts between tests.

    Mirrors the reset fixture in tests/skills/test_workspace_skill.py so
    catalog cache and Neo4j recovery do not leak across tests. Without
    this, a prior test that persists a workspace (e.g. ``lna-ui``)
    poisons the global catalog and the typo guard / list output.
    """
    saved_recovery_fn = workspace_store._recovery_fn
    state.active_workspaces.clear()
    state.recent_workspaces.clear()
    workspace_store._catalog_cache = [state.DEFAULT_WORKSPACE]
    workspace_store._catalog_cache_ts = time.time()
    workspace_store._recovery_fn = None
    yield
    state.active_workspaces.clear()
    state.recent_workspaces.clear()
    workspace_store._catalog_cache = [state.DEFAULT_WORKSPACE]
    workspace_store._catalog_cache_ts = 0.0
    workspace_store._recovery_fn = saved_recovery_fn


def _content(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_workspace_command_registered():
    assert "^^workspace" in COMMAND_REGISTRY
    assert "^^workspace" in COMMAND_HELP


def test_ws_alias_registered():
    assert "^^ws" in COMMAND_REGISTRY
    assert "^^ws" in COMMAND_HELP


def test_ws_help_text_present():
    assert "workspace" in COMMAND_HELP["^^workspace"].lower()
    assert "workspace" in COMMAND_HELP["^^ws"].lower()


def test_help_lists_workspace_commands():
    ctx = CommandContext(raw_text="^^help", session_id="s1")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "^^workspace" in content
    assert "^^ws" in content


# ---------------------------------------------------------------------------
# Bare / status
# ---------------------------------------------------------------------------

def test_workspace_command_bare_returns_status():
    """^^workspace (bare) returns current workspace info (defaults to 'default')."""
    ctx = CommandContext(raw_text="^^workspace", session_id="sess-status")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Current workspace" in content
    assert state.DEFAULT_WORKSPACE in content


def test_workspace_command_status_action():
    ctx = CommandContext(raw_text="^^workspace status", session_id="sess-status-2")
    state.active_workspaces["sess-status-2"] = "my_project"
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Current workspace" in content
    assert "my_project" in content


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------

def test_ws_alias_works():
    """^^ws is an alias for ^^workspace — bare returns current workspace status."""
    ctx = CommandContext(raw_text="^^ws", session_id="sess-alias")
    state.active_workspaces["sess-alias"] = "feature-x"
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Current workspace" in content
    assert "feature-x" in content


def test_ws_alias_status_matches_workspace():
    """^^ws status and ^^workspace status produce identical output for a session."""
    session_id = "sess-alias-cmp"
    state.active_workspaces[session_id] = "shared-ws"
    resp_ws = dispatch_command(CommandContext(raw_text="^^ws status", session_id=session_id))
    resp_full = dispatch_command(CommandContext(raw_text="^^workspace status", session_id=session_id))
    assert _content(resp_ws) == _content(resp_full)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def test_workspace_command_create():
    ctx = CommandContext(raw_text="^^workspace create myproject", session_id="sess-create")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "myproject" in content
    assert "Created" in content or "Switched" in content
    assert state.active_workspaces.get("sess-create") == "myproject"


def test_ws_alias_create():
    ctx = CommandContext(raw_text="^^ws create via-alias", session_id="sess-create-alias")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "via-alias" in content
    assert state.active_workspaces.get("sess-create-alias") == "via-alias"


def test_workspace_command_create_no_arg_returns_usage():
    ctx = CommandContext(raw_text="^^workspace create", session_id="sess-create-noarg")
    resp = dispatch_command(ctx)
    assert resp["_status_code"] == 400
    content = _content(resp)
    assert "Usage" in content
    assert "create" in content


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------

def test_workspace_command_switch_existing():
    # NOTE: `switch` is NOT an explicit subcommand keyword — WorkspaceSkill
    # treats any unrecognized first token as the switch target. So
    # `^^workspace feature-x` switches (there is no `^^workspace switch <name>`).
    state.recent_workspaces["sess-switch"] = ["feature-x"]
    state.active_workspaces["sess-switch"] = state.DEFAULT_WORKSPACE
    ctx = CommandContext(raw_text="^^workspace feature-x", session_id="sess-switch")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Switched" in content
    assert "feature-x" in content
    assert state.active_workspaces.get("sess-switch") == "feature-x"


def test_workspace_command_switch_bare_name():
    """^^workspace feature-x (no 'switch' keyword) still switches."""
    state.recent_workspaces["sess-switch-bare"] = ["feature-y"]
    state.active_workspaces["sess-switch-bare"] = state.DEFAULT_WORKSPACE
    ctx = CommandContext(raw_text="^^workspace feature-y", session_id="sess-switch-bare")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "feature-y" in content
    assert state.active_workspaces.get("sess-switch-bare") == "feature-y"


def test_workspace_command_switch_unknown_returns_404():
    # No `switch` keyword: `^^workspace brand-new` attempts to switch to
    # workspace "brand-new" and returns 404 since it doesn't exist.
    ctx = CommandContext(raw_text="^^workspace brand-new", session_id="sess-switch-404")
    resp = dispatch_command(ctx)
    assert resp["_status_code"] == 404
    content = _content(resp)
    assert "does not exist" in content
    assert "brand-new" in content
    assert "^^workspace create" in content


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------

def test_workspace_unknown_action_is_treated_as_switch():
    """An unrecognized first token is NOT an error — it's a switch attempt.

    WorkspaceSkill._dispatch_subcommand maps any non-subcommand token to
    ("switch", <token>). So `^^workspace foobar` attempts to switch to a
    workspace named "foobar" and returns a 404 (does-not-exist) message,
    NOT a generic "unknown action" error. This matches the documented
    workspace command contract.
    """
    ctx = CommandContext(raw_text="^^workspace foobar", session_id="sess-unknown")
    resp = dispatch_command(ctx)
    # foobar is not a known subcommand → treated as switch target → 404
    assert resp["_status_code"] == 404
    content = _content(resp)
    assert "foobar" in content
    assert "does not exist" in content


def test_ws_unknown_action_treated_as_switch():
    ctx = CommandContext(raw_text="^^ws frobnicate", session_id="sess-unknown-alias")
    resp = dispatch_command(ctx)
    assert resp["_status_code"] == 404
    content = _content(resp)
    assert "frobnicate" in content


# ---------------------------------------------------------------------------
# list / recent / help / leave
# ---------------------------------------------------------------------------

def test_workspace_command_list():
    workspace_store._catalog_cache = ["default", "alpha", "beta"]
    workspace_store._catalog_cache_ts = time.time()
    state.active_workspaces["sess-list"] = "alpha"
    ctx = CommandContext(raw_text="^^workspace list", session_id="sess-list")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Workspaces:" in content
    assert "alpha" in content
    assert "beta" in content


def test_workspace_command_recent():
    state.recent_workspaces["sess-recent"] = ["ws-a", "ws-b"]
    state.active_workspaces["sess-recent"] = "ws-a"
    ctx = CommandContext(raw_text="^^workspace recent", session_id="sess-recent")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Recent workspaces" in content
    assert "ws-a" in content
    assert "ws-b" in content


def test_workspace_command_help():
    ctx = CommandContext(raw_text="^^workspace help", session_id="sess-help")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Workspace Commands" in content
    assert "^^workspace" in content


def test_workspace_command_leave():
    state.active_workspaces["sess-leave"] = "some_project"
    ctx = CommandContext(raw_text="^^workspace leave", session_id="sess-leave")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Returned to default workspace" in content
    assert "sess-leave" not in state.active_workspaces


def test_ws_alias_leave():
    state.active_workspaces["sess-leave-alias"] = "some_project"
    ctx = CommandContext(raw_text="^^ws leave", session_id="sess-leave-alias")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Returned to default workspace" in content


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------

def test_workspace_command_forget():
    state.recent_workspaces["sess-forget"] = ["luna-ui", "other"]
    state.active_workspaces["sess-forget"] = "other"
    ctx = CommandContext(raw_text="^^workspace forget luna-ui", session_id="sess-forget")
    resp = dispatch_command(ctx)
    content = _content(resp)
    assert "Forgot workspace" in content
    assert "luna-ui" in content
    assert "luna-ui" not in state.recent_workspaces.get("sess-forget", [])


# ---------------------------------------------------------------------------
# Bug #7: ^^ws / ^^workspace must thread client_hint into workspace_store
# ---------------------------------------------------------------------------


class TestClientHintThreading:
    """Bug #7: ``^^ws <name>`` dropped ``client_hint`` because ``CommandContext``
    had no such field and ``_build_skill_ctx`` hardcoded ``client_hint=None``.

    The fix threads ``RequestContext.client_hint`` → ``CommandContext.client_hint``
    → ``SkillRequestContext.client_hint`` → ``workspace_store.activate`` so the
    client-level ``_client_active`` dict is populated via the command path
    (not just session-level).
    """

    def test_command_context_has_client_hint_field(self):
        """CommandContext exposes a client_hint field defaulting to ''."""
        ctx = CommandContext(raw_text="^^ws foo", session_id="s1")
        assert ctx.client_hint == ""

        ctx_with_hint = CommandContext(
            raw_text="^^ws foo",
            session_id="s1",
            client_hint="client-abc",
        )
        assert ctx_with_hint.client_hint == "client-abc"

    def test_ws_switch_populates_client_active(self):
        """^^ws <existing> with client_hint set → workspace_store._client_active
        is populated (client-level), not just session-level."""
        # Seed a known workspace so the switch succeeds (not 404).
        state.recent_workspaces["sess-ws-hint"] = ["feature-x"]
        state.active_workspaces["sess-ws-hint"] = state.DEFAULT_WORKSPACE
        workspace_store._client_active.clear()
        try:
            ctx = CommandContext(
                raw_text="^^ws feature-x",
                session_id="sess-ws-hint",
                client_hint="client-1",
            )
            resp = dispatch_command(ctx)
            content = _content(resp)
            assert "Switched" in content
            # Session-level activation.
            assert state.active_workspaces.get("sess-ws-hint") == "feature-x"
            # Bug #7 contract: client-level activation MUST also fire.
            assert workspace_store._client_active.get("client-1") == "feature-x", (
                f"Expected _client_active['client-1']='feature-x'; "
                f"got {workspace_store._client_active!r}"
            )
        finally:
            workspace_store._client_active.clear()

    def test_workspace_switch_populates_client_active(self):
        """^^workspace <existing> with client_hint also populates _client_active."""
        state.recent_workspaces["sess-ws-full"] = ["feature-y"]
        state.active_workspaces["sess-ws-full"] = state.DEFAULT_WORKSPACE
        workspace_store._client_active.clear()
        try:
            ctx = CommandContext(
                raw_text="^^workspace feature-y",
                session_id="sess-ws-full",
                client_hint="client-2",
            )
            resp = dispatch_command(ctx)
            content = _content(resp)
            assert "Switched" in content
            assert workspace_store._client_active.get("client-2") == "feature-y"
        finally:
            workspace_store._client_active.clear()

    def test_ws_create_with_client_hint_populates_client_active(self):
        """`^^ws create <name>` with client_hint persists client-level state."""
        workspace_store._client_active.clear()
        try:
            ctx = CommandContext(
                raw_text="^^ws create proj-x",
                session_id="sess-create-hint",
                client_hint="client-3",
            )
            resp = dispatch_command(ctx)
            content = _content(resp)
            assert "proj-x" in content
            assert workspace_store._client_active.get("client-3") == "proj-x"
        finally:
            workspace_store._client_active.clear()

    def test_ws_without_client_hint_does_not_populate_client_active(self):
        """Default CommandContext (no client_hint) → only session-level activation.

        Regression guard: workspace_store.activate ignores falsy client_hint,
        so session-only is the original behavior and must not change.
        """
        state.recent_workspaces["sess-no-hint"] = ["feature-z"]
        state.active_workspaces["sess-no-hint"] = state.DEFAULT_WORKSPACE
        workspace_store._client_active.clear()
        try:
            ctx = CommandContext(
                raw_text="^^ws feature-z",
                session_id="sess-no-hint",
                # client_hint intentionally omitted (defaults to "").
            )
            dispatch_command(ctx)
            assert state.active_workspaces.get("sess-no-hint") == "feature-z"
            # No client-level activation.
            assert workspace_store._client_active == {}
        finally:
            workspace_store._client_active.clear()
