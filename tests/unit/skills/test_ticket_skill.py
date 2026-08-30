"""Regression tests for the ticket-mode workspace override bugfix.

These tests lock in the fix for the ECHO-H3/INDIA H-8 overcorrection: when
in ticket mode, ticket operations must derive their workspace scope from
``workspace_store.get_active(ctx.session_id)`` (the authoritative source
written by ``WorkspaceSkill.on_activate``), NOT from a stale ``DEFAULT_WORKSPACE``
fallback.

Each test seeds ``state.active_workspaces[session_id]`` to a non-default
workspace, calls the relevant handler, and asserts that the workspace_id
passed to ``TaskGraph`` contains the active workspace name.

The ``state.active_workspaces = workspace_store._active`` alias in
``state.py:158`` means writing to ``state.active_workspaces`` populates the
underlying store (no separate setup needed).

NOTE: ``_get_task_graph_singleton`` is imported into ``skills.ticket_skill``
at module level via ``from proxy.context import _get_task_graph_singleton``,
so the mock must be patched at the import site (``skills.ticket_skill``),
not at the definition site (``proxy.context``).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proxy import state
from proxy.skills.skill_registry import SkillContext
from skills.ticket_skill import TicketSkill


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def session_id():
    """Unique session id per test."""
    return "test-sess-wsfix"


@pytest.fixture(autouse=True)
def reset_workspace_state(session_id):
    """Reset the active_workspaces dict around each test."""
    state.active_workspaces.pop(session_id, None)
    yield
    state.active_workspaces.pop(session_id, None)


@pytest.fixture
def ctx(session_id):
    """SkillContext for tests."""
    return SkillContext(
        session_id=session_id,
        agent_id=f"agent_{session_id}",
        messages=[],
        is_subagent=False,
        active_skill_name="ticket",
        raw_user_text="^^ticket",
    )


@pytest.fixture
def mock_tg():
    """Mocked TaskGraph singleton with sensible defaults."""
    tg = MagicMock()
    tg.is_available.return_value = True
    tg.get_actionable_tasks.return_value = []
    tg.get_task_enriched.return_value = None
    tg.get_child_tasks.return_value = []
    tg.update_tasks_batch_status.return_value = []
    return tg


# ----------------------------------------------------------------------------
# Regression tests
# ----------------------------------------------------------------------------


class TestWorkspaceScopeDerivation:
    """Verify ticket-mode operations derive workspace from active_workspaces,
    not from DEFAULT_WORKSPACE."""

    def test_list_tasks_uses_active_workspace(self, ctx, mock_tg):
        """_list_tasks must pass luna-ui to TaskGraph, not DEFAULT_WORKSPACE."""
        state.active_workspaces[ctx.session_id] = "luna-ui"

        # The new contract: workspace_id is derived from
        # workspace_store.get_active(ctx.session_id) — the agent's mode
        # is no longer consulted. No _agent_registry patching required.
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill()._list_tasks(ctx)

        # Sanity: didn't error
        assert "error" not in (result or {}), result

        # Assertion: TaskGraph was called with workspace_id that includes luna-ui
        mock_tg.get_actionable_tasks.assert_called_once()
        call_kwargs = mock_tg.get_actionable_tasks.call_args[1]
        assert call_kwargs["workspace_id"] == "luna-ui", (
            f"expected luna-ui workspace, got {call_kwargs['workspace_id']!r}"
        )
        # Negative: NOT default
        assert "default" not in call_kwargs["workspace_id"]

    def test_show_task_uses_active_workspace(self, ctx, mock_tg):
        """_show_task must pass luna-ui to TaskGraph, not DEFAULT_WORKSPACE."""
        state.active_workspaces[ctx.session_id] = "luna-ui"

        # The new contract: workspace_id is derived from
        # workspace_store.get_active(ctx.session_id) — the agent's mode
        # is no longer consulted. No _agent_registry patching required.
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill()._show_task(ctx, "abcd1234")

        assert "error" not in (result or {}), result

        # get_task_enriched received luna-ui workspace_id
        enriched_call = mock_tg.get_task_enriched.call_args
        assert enriched_call[1]["workspace_id"] == "luna-ui", (
            f"get_task_enriched: expected luna-ui, got {enriched_call[1]['workspace_id']!r}"
        )

    def test_batch_status_update_uses_active_workspace(self, ctx, mock_tg):
        """_batch_status_update must pass luna-ui to TaskGraph, not DEFAULT_WORKSPACE."""
        state.active_workspaces[ctx.session_id] = "luna-ui"
        mock_tg.update_tasks_batch_status.return_value = []

        # The new contract: workspace_id is derived from
        # workspace_store.get_active(ctx.session_id) — the agent's mode
        # is no longer consulted. No _agent_registry patching required.
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill()._batch_status_update(ctx, "abcd1234", "complete")

        assert "error" not in (result or {}), result

        call_kwargs = mock_tg.update_tasks_batch_status.call_args[1]
        assert call_kwargs["workspace_id"] == "luna-ui", (
            f"expected luna-ui, got {call_kwargs['workspace_id']!r}"
        )

    def test_list_tasks_defaults_when_no_active_workspace(self, ctx, mock_tg):
        """When no active workspace is set, get_active() returns DEFAULT_WORKSPACE.

        Documents the fallback behavior — we don't want a KeyError if a
        session has never set an active workspace.
        """
        # Explicitly DO NOT set state.active_workspaces[ctx.session_id]
        with patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg):
            result = TicketSkill()._list_tasks(ctx)

        assert "error" not in (result or {}), result
        call_kwargs = mock_tg.get_actionable_tasks.call_args[1]
        # DEFAULT_WORKSPACE fallback
        assert call_kwargs["workspace_id"] == "default"


# ----------------------------------------------------------------------------
# B4 follow-up: get_bug_by_short_hash returns an infrastructure_error sentinel
# on transient DB failure. The callers in TicketSkill must check for this
# sentinel BEFORE calling bug.get(...) — otherwise the dict passes through
# the `if bug is None` guard and produces malformed output.
# ----------------------------------------------------------------------------


INFRA_SENTINEL = {
    "error": "infrastructure_error",
    "message": "Database temporarily unavailable",
}


def _patch_skill_dependencies(mock_tg, get_bug_return):
    """Patch the deps shared by _show_bug and _resolve_bug.

    Returns a list of patcher objects; caller is responsible for .stop().
    """
    return [
        patch("skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg),
        patch(
            "memory.bug_tracker.get_bug_by_short_hash",
            return_value=get_bug_return,
        ),
    ]


class TestShowBugInfrastructureError:
    """_show_bug (line ~515) must surface infrastructure_error as 503."""

    def test_returns_503_when_get_bug_returns_sentinel(self, ctx, mock_tg, session_id):
        state.active_workspaces[session_id] = "luna-ui"
        patchers = _patch_skill_dependencies(mock_tg, INFRA_SENTINEL)
        for p in patchers:
            p.start()
        try:
            result = TicketSkill()._show_bug(ctx, "abcd1234")
        finally:
            for p in patchers:
                p.stop()

        assert result is not None
        assert result.get("_status_code") == 503, result
        content = result["choices"][0]["message"]["content"]
        assert "Database temporarily unavailable" in content, result
        assert "Bug [" not in content, (
            "sentinel must not be rendered as a normal bug view"
        )

    def test_returns_404_when_bug_not_found(self, ctx, mock_tg, session_id):
        state.active_workspaces[session_id] = "luna-ui"
        patchers = _patch_skill_dependencies(mock_tg, None)
        for p in patchers:
            p.start()
        try:
            result = TicketSkill()._show_bug(ctx, "deadbeef")
        finally:
            for p in patchers:
                p.stop()

        assert result is not None
        assert result.get("_status_code") == 404, result

    def test_returns_503_when_get_bug_by_short_hash_raises_infrastructure_error(
        self, ctx, mock_tg, session_id,
    ):
        from memory.query_executor import InfrastructureError

        state.active_workspaces[session_id] = "luna-ui"
        mock_tg.is_available = MagicMock(return_value=True)
        mock_tg._executor.execute_read.side_effect = InfrastructureError("db down")
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = TicketSkill()._show_bug(ctx, "abcd1234")

        assert result is not None
        assert result.get("_status_code") == 503, result
        content = result["choices"][0]["message"]["content"]
        assert "Database temporarily unavailable" in content, result
        assert "No bug found" not in content, (
            "DB outage must not surface as a 404"
        )

    def test_returns_bug_view_when_normal_dict(self, ctx, mock_tg, session_id):
        state.active_workspaces[session_id] = "luna-ui"
        bug = {
            "uuid": "entity-x",
            "short_hash": "abcd1234",
            "status": "open",
            "severity": "high",
            "title": "Sample bug",
            "description": "desc",
            "reported_by_agent": "agent-1",
            "review_count": 0,
            "max_reviews": 3,
            "validated_at": None,
            "reviews": [],
        }
        patchers = _patch_skill_dependencies(mock_tg, bug)
        for p in patchers:
            p.start()
        try:
            result = TicketSkill()._show_bug(ctx, "abcd1234")
        finally:
            for p in patchers:
                p.stop()

        assert result is not None
        assert "error" not in result, result
        content = result["choices"][0]["message"]["content"]
        assert "Sample bug" in content, result
        assert "abcd1234" in content, result


class TestResolveBugInfrastructureError:
    """_resolve_bug (line ~582) must surface infrastructure_error as 503."""

    def test_returns_503_when_get_bug_returns_sentinel(self, ctx, mock_tg, session_id):
        state.active_workspaces[session_id] = "luna-ui"
        patchers = _patch_skill_dependencies(mock_tg, INFRA_SENTINEL)
        for p in patchers:
            p.start()
        try:
            result = TicketSkill()._resolve_bug(ctx, "abcd1234|fixed it", "approved")
        finally:
            for p in patchers:
                p.stop()

        assert result is not None
        assert result.get("_status_code") == 503, result
        content = result["choices"][0]["message"]["content"]
        assert "Database temporarily unavailable" in content, result
        assert "Task created" not in content, (
            "sentinel must not flow through to resolve_bug"
        )

    def test_returns_404_when_bug_not_found(self, ctx, mock_tg, session_id):
        state.active_workspaces[session_id] = "luna-ui"
        patchers = _patch_skill_dependencies(mock_tg, None)
        for p in patchers:
            p.start()
        try:
            result = TicketSkill()._resolve_bug(ctx, "deadbeef|reason", "dismissed")
        finally:
            for p in patchers:
                p.stop()

        assert result is not None
        assert result.get("_status_code") == 404, result


# ----------------------------------------------------------------------------
# Block ^^ticket / ^^work on the default workspace (HTTP 409 guard).
#
# The default workspace is read-only for ticket/work operations. handle_direct
# must reject every subcommand (including plan/unknown) with a 409 that guides
# the user to ^^ws create/list/help. plan/unknown previously returned None
# (→ activation); they now return the 409 envelope so activation never fires.
# ----------------------------------------------------------------------------


class TestTicketBlockedOnDefaultWorkspace:
    """Guard tests: handle_direct rejects on the default workspace.

    The autouse ``reset_workspace_state`` fixture pops the session but does
    NOT set it, so ``get_active`` falls back to ``DEFAULT_WORKSPACE`` — the
    cold-session path this class exercises.
    """
    # US-017j: ^^ticket/^^work reject with HTTP 409 on the default workspace.

    _EXPECTED_STATUS = 409
    _EXPECTED_HINT = "^^ws create"

    def test_list_blocked_on_default(self, ctx, mock_tg):
        """list returns 409 and never touches the TaskGraph."""
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "list", "")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result
        mock_tg.get_actionable_tasks.assert_not_called()

    def test_show_blocked_on_default(self, ctx, mock_tg):
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "show", "abcd1234")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_done_blocked_on_default(self, ctx, mock_tg):
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "done", "abcd1234")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_bug_list_blocked_on_default(self, ctx, mock_tg):
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "bug_list", "")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_reopen_blocked_on_default(self, ctx, mock_tg):
        """reopen returns 409 on default — locks the dead code path."""
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "reopen", "abcd1234")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_plan_blocked_on_default(self, ctx, mock_tg):
        """plan returns a 409 dict (NOT None) → activation never happens."""
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "plan", "do the thing")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_unknown_action_blocked_on_default(self, ctx, mock_tg):
        """Unknown action returns 409 dict (NOT None) — locks the new semantics."""
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "unknown_action", "")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_work_list_delegation_blocked_on_default(self, ctx, mock_tg):
        """^^work list delegates to TicketSkill and must also 409 on default."""
        from skills.work_skill import WorkSkill
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = WorkSkill().handle_direct(ctx, "list", "")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_work_hash_activation_blocked_on_default(self, ctx, mock_tg):
        """^^work <hash> (action "task") 409s on default — work mode never activates."""
        from skills.work_skill import WorkSkill
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = WorkSkill().handle_direct(ctx, "task", "abcd1234")
        assert result is not None
        assert result.get("_status_code") == self._EXPECTED_STATUS, result

    def test_message_guides_user_to_ws_help(self, ctx, mock_tg):
        """The 409 message must guide the user to ^^ws create/list/help."""
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg
        ):
            result = TicketSkill().handle_direct(ctx, "list", "")
        content = result["choices"][0]["message"]["content"]
        assert self._EXPECTED_HINT in content, content
        assert "^^ws list" in content, content
        assert "^^ws help" in content, content

    def test_ticket_on_activate_raises_on_default(self, ctx):
        """Defense-in-depth: on_activate raises SkillValidationError(409) on default."""
        import pytest
        from proxy.skills.skill_definition import SkillValidationError
        with pytest.raises(SkillValidationError) as excinfo:
            TicketSkill().on_activate(ctx)
        assert excinfo.value.http_status == 409
