"""End-to-end workspace scoping for `^^ticket` (list action).

Verifies the full dispatch path:
    RequestContext → SkillDispatcher.dispatch → TicketSkill._list_tasks

Specifically locks in the bug-1 scenario: after switching to a workspace
that contains tasks, `^^ticket` (bare list) must return those tasks; after
switching to a different workspace, those tasks must NOT appear.

The TaskGraph singleton is mocked so the test does not require Neo4j.
The WorkspaceSkill + workspace_store are REAL (in-memory) so the
switch→list flow is exercised end-to-end.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proxy import state
from proxy.skills.skill_definition import SkillRequestContext
from proxy.skill_dispatcher import SkillDispatcher
from proxy.skills.skill_registry import SkillRegistry
from proxy.types import RequestContext
from proxy.workspace_state import workspace_store, DEFAULT_WORKSPACE
from skills.ticket_skill import TicketSkill
from skills.workspace import WorkspaceSkill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_id():
    # Unique per test to avoid Neo4j recovery re-activating a workspace
    # persisted by a prior test (recovery is keyed on session_id).
    import time as _t
    return f"test-ws-scope-{int(_t.time() * 1_000_000) % 10_000_000}"


@pytest.fixture(autouse=True)
def disable_workspace_recovery():
    """Disable Neo4j recovery + persist so tests stay hermetic.

    Recovery is keyed on session_id and reads from Neo4j; without this a
    cold session would recover the workspace a prior test persisted,
    breaking test isolation. Persist is disabled so test workspaces are
    not written to the real Neo4j catalog.
    """
    saved_recovery = workspace_store._recovery_fn
    saved_persist = workspace_store._persist
    workspace_store._recovery_fn = None
    workspace_store._persist = None
    yield
    workspace_store._recovery_fn = saved_recovery
    workspace_store._persist = saved_persist


def _make_request_context(session_id: str, text: str) -> RequestContext:
    """Build a RequestContext mimicking parse_request() output for a main agent."""
    return RequestContext(
        body={"messages": [{"role": "user", "content": text}]},
        model_name="test-model",
        wants_stream=False,
        session_id=session_id,
        client_hint="",
        messages=[{"role": "user", "content": text}],
        remote_addr="",
    )


def _make_mock_tg(workspace_tasks: dict[str, list[dict]]) -> MagicMock:
    """Build a mock TaskGraph whose get_actionable_tasks returns tasks by workspace_id."""
    tg = MagicMock()
    tg.is_available.return_value = True

    def _get_actionable_tasks(status_filter=None, global_list=False, workspace_id=None, limit=500):
        return list(workspace_tasks.get(workspace_id, []))

    tg.get_actionable_tasks.side_effect = _get_actionable_tasks
    tg.get_store_status.return_value = {"available": True}
    return tg


def _dispatch_ticket_list(session_id: str, tg: MagicMock) -> dict:
    """Run the real SkillDispatcher.dispatch for a bare `^^ticket` and return the response dict."""
    ctx = _make_request_context(session_id, "^^ticket")
    registry = SkillRegistry()
    dispatcher = SkillDispatcher(registry)
    # Agent is not needed for handle_direct (list action does not activate mode).
    with patch("skills.ticket_skill._get_task_graph_singleton", return_value=tg):
        result = dispatcher.dispatch("^^ticket", ctx, agent=None)
    assert result.handled, f"expected handled=True, got {result}"
    assert result.response is not None
    return result.response


def _content(resp: dict) -> str:
    return resp["choices"][0]["message"]["content"]


def _switch_to(session_id: str, workspace: str) -> None:
    """Run a real `^^workspace create <workspace>` for the session."""
    skill = WorkspaceSkill()
    ctx = SkillRequestContext(
        session_id=session_id,
        is_subagent=False,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint="",
        body={},
    )
    resp = skill.handle_direct(ctx, "create", workspace)
    assert resp is not None, f"workspace switch to {workspace} returned None"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkspaceTaskScopingDispatch:
    """`^^ticket` list must be scoped by the active workspace via the full dispatch path."""

    def test_tasks_visible_after_switching_to_workspace(self, session_id):
        """Switch to 'choriprox' → `^^ticket` returns the tasks stored there."""
        tasks = {
            "choriprox": [
                {
                    "uuid": "entity-1",
                    "short_hash": "abc12345",
                    "name": "Fix proxy scoping",
                    "status": "pending",
                    "priority": "high",
                    "is_blocked": False,
                    "blocked_count": 0,
                    "blockers": [],
                    "parent_short_hash": None,
                    "parent_name": None,
                    "summary": "",
                    "tag": "",
                    "planning_info": None,
                }
            ],
            "default": [],
        }
        tg = _make_mock_tg(tasks)

        _switch_to(session_id, "choriprox")
        assert workspace_store.get_active(session_id) == "choriprox"

        resp = _dispatch_ticket_list(session_id, tg)
        content = _content(resp)
        assert "abc12345" in content, f"task not listed after switching to choriprox: {content!r}"
        assert "Fix proxy scoping" in content

    def test_tasks_not_visible_after_switching_away(self, session_id):
        """Task in 'choriprox' must NOT appear when active workspace is another (empty) workspace."""
        tasks = {
            "choriprox": [
                {
                    "uuid": "entity-1",
                    "short_hash": "abc12345",
                    "name": "Fix proxy scoping",
                    "status": "pending",
                    "priority": "high",
                    "is_blocked": False,
                    "blocked_count": 0,
                    "blockers": [],
                    "parent_short_hash": None,
                    "parent_name": None,
                    "summary": "",
                    "tag": "",
                    "planning_info": None,
                }
            ],
            "empty-ws": [],
        }
        tg = _make_mock_tg(tasks)

        # Switch to an empty non-default workspace — choriprox tasks must
        # not leak. (The default workspace is now rejected by the
        # handle_direct guard with HTTP 409, so this scoping test uses a
        # real empty workspace to preserve the isolation intent.)
        _switch_to(session_id, "empty-ws")
        assert workspace_store.get_active(session_id) == "empty-ws"
        resp = _dispatch_ticket_list(session_id, tg)
        content = _content(resp)
        assert "No active tasks" in content, f"expected no tasks in empty-ws: {content!r}"
        assert "abc12345" not in content

    def test_workspace_id_passed_to_task_graph(self, session_id):
        """The workspace_id argument passed to get_actionable_tasks must equal the active workspace."""
        tg = _make_mock_tg({"choriprox": [], "default": []})
        _switch_to(session_id, "choriprox")

        _dispatch_ticket_list(session_id, tg)

        assert tg.get_actionable_tasks.called
        kwargs = tg.get_actionable_tasks.call_args.kwargs
        assert kwargs.get("workspace_id") == "choriprox", (
            f"expected workspace_id='choriprox', got {kwargs.get('workspace_id')!r}"
        )

    def test_isolation_between_workspaces(self, session_id):
        """Switching between two workspaces shows only the active one's tasks."""
        tasks = {
            "choriprox": [
                {
                    "uuid": "e-cp", "short_hash": "cp000001", "name": "CP task",
                    "status": "pending", "priority": "medium", "is_blocked": False,
                    "blocked_count": 0, "blockers": [], "parent_short_hash": None,
                    "parent_name": None, "summary": "", "tag": "", "planning_info": None,
                }
            ],
            "luna-ui": [
                {
                    "uuid": "e-lu", "short_hash": "lu000001", "name": "LU task",
                    "status": "pending", "priority": "medium", "is_blocked": False,
                    "blocked_count": 0, "blockers": [], "parent_short_hash": None,
                    "parent_name": None, "summary": "", "tag": "", "planning_info": None,
                }
            ],
            "default": [],
        }
        tg = _make_mock_tg(tasks)

        _switch_to(session_id, "choriprox")
        resp = _dispatch_ticket_list(session_id, tg)
        assert "cp000001" in _content(resp)
        assert "lu000001" not in _content(resp)

        _switch_to(session_id, "luna-ui")
        resp = _dispatch_ticket_list(session_id, tg)
        assert "lu000001" in _content(resp)
        assert "cp000001" not in _content(resp)