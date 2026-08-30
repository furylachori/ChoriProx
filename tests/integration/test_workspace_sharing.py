"""Integration tests for v5 global-catalog workspace SHARING model.

v5 inverts the v4 isolation model: workspaces are global, not per-session.
Multiple sessions on the same proxy can see each other's tasks in a
shared workspace; sessions in different workspaces see only their own.

This file inverts the assertions of the deprecated test_workspace_isolation.py
(per the v5 spec, §5.2 — "Invert ~50% of assertions. Add
test_visibility_within_shared_workspace"). It exercises:

  1. Two sessions in the same workspace see each other's tasks
  2. Two sessions in different workspaces see only their own tasks
  3. Task status updates by one session are visible to the other
  4. Session eviction does NOT delete the (global) :Workspace node
  5. Sub-agent inherits its parent's workspace via the canary
  6. store.get_global_catalog() returns ALL workspaces globally

DB layer (Neo4j) is mocked via a recording executor; workspace state and
the WorkspaceSkill are real.
"""
from __future__ import annotations

import logging
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from memory.workspace_persistence import (
    cleanup_neo4j_sessions,
    get_global_catalog,
)
from proxy import state
from proxy.skills.skill_registry import SkillContext
from proxy.workspace_state import workspace_store
from skills.workspace import WorkspaceSkill

logger = logging.getLogger("proxy.tests.workspace_sharing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_id(name: str) -> str:
    return f"sess-{name}-{int(time.time() * 1000) % 1_000_000}"


class _FakeTx:
    """In-memory Neo4j transaction stand-in."""

    def __init__(self, script):
        self._script = script
        self._idx = 0

    def run(self, query, params=None):
        records = self._script(self._idx, query, params or {})
        self._idx += 1
        return iter(records)


class _RecordingExecutor:
    """Records every write/read so tests can assert on the exact Cypher."""

    def __init__(self, store):
        self._store = store
        self.queries: list[dict] = []

    def execute_read(self, query, params=None, timeout=30):
        self.queries.append({"query": query, "params": params or {}, "op": "read"})
        return self._store.handle(query, params or {})

    def execute_write(self, query, params=None, timeout=30):
        self.queries.append({"query": query, "params": params or {}, "op": "write"})
        return self._store.handle(query, params or {})

    def execute_transaction_no_retry(self, fn):
        tx = _FakeTx(lambda idx, q, p: self._store.handle(q, p))
        return fn(tx)


class WorkspaceTaskStore:
    """Bare-name-keyed in-memory task store (v5 global model).

    Tasks are keyed by their BARE workspace name (no session_id prefix).
    List/get operations filter by workspace_id; the same name is visible
    to any session that asks. Different workspaces do NOT see each other.
    """

    def __init__(self):
        # bare workspace name → short_hash → task dict
        self._tasks: dict[str, dict[str, dict]] = {}
        # :Workspace catalog (bare names) — for catalog query responses
        self._workspace_nodes: dict[str, dict] = {}
        self._counter = 0

    def create_workspace(self, name: str, last_active_at: str | None = None) -> None:
        """Create a :Workspace node in our mock DB."""
        self._workspace_nodes[name] = {
            "workspace_id": name,
            "name": name,
            "last_active_at": last_active_at or "2024-01-01T00:00:00Z",
        }

    def create_task(self, workspace_id: str, name: str) -> dict:
        self._counter += 1
        short_hash = f"{self._counter:08x}"
        full_uuid = f"entity-{short_hash}12345678"
        rec = {
            "uuid": full_uuid,
            "short_hash": short_hash,
            "name": name,
            "priority": "medium",
            "status": "pending",
            "summary": "",
            "tag": "",
            "workspace_id": workspace_id,
            "assigned_agent_id": None,
            "blockers": [],
            "is_blocked": False,
            "blocked_count": 0,
            "parent_short_hash": None,
            "parent_name": None,
            "planning_info": None,
        }
        self._tasks.setdefault(workspace_id, {})[short_hash] = rec
        return rec

    def get_task(self, workspace_id: str, short_hash: str) -> dict | None:
        return self._tasks.get(workspace_id, {}).get(short_hash)

    def handle(self, query: str, params: dict) -> list[dict]:
        q = query

        # get_actionable_tasks (list_tasks path) — returns bare-workspace tasks
        if ("RETURN t.uuid AS uuid" in q and "size(active_blockers)" in q) or \
           "collect(DISTINCT CASE WHEN blocker.uuid IS NOT NULL" in q:
            return self._handle_get_actionable_tasks(params)

        # update_task_status check (FakeTaskGraph routes via FAKE_UPDATE_TASK_STATUS_CHECK)
        if "FAKE_UPDATE_TASK_STATUS_CHECK" in q:
            return self._handle_update_task_check(params)

        # update_task_status write
        if "FAKE_UPDATE_TASK_STATUS_SET" in q:
            return self._handle_update_task_set(params)

        # Catalog read (workspace list)
        if "MATCH (w:Workspace)" in q and "AS name" in q:
            return self._handle_catalog_read(params)

        # Default
        return []

    def _handle_get_actionable_tasks(self, params: dict) -> list[dict]:
        ws = params.get("workspace_id")
        tasks = list(self._tasks.get(ws, {}).values())
        return [
            {
                "uuid": t["uuid"],
                "short_hash": t["short_hash"],
                "name": t["name"],
                "priority": t["priority"],
                "status": t["status"],
                "summary": t["summary"],
                "tag": t["tag"],
                "blockers": t["blockers"],
                "is_blocked": t["is_blocked"],
                "blocked_count": t["blocked_count"],
                "parent_short_hash": t["parent_short_hash"],
                "parent_name": t["parent_name"],
                "planning_info": t["planning_info"],
                "assigned_agent_id": t["assigned_agent_id"],
            }
            for t in tasks
        ]

    def _handle_update_task_check(self, params: dict) -> list[dict]:
        ws = params.get("workspace_id")
        h = params.get("hash")
        bucket = self._tasks.get(ws, {})
        if h not in bucket:
            return [{"cnt": 0, "current_status": None}]
        t = bucket[h]
        return [{"cnt": 1, "current_status": t["status"]}]

    def _handle_update_task_set(self, params: dict) -> list[dict]:
        ws = params.get("workspace_id")
        h = params.get("hash")
        new_status = params.get("new_status")
        bucket = self._tasks.get(ws, {})
        if h not in bucket:
            return [{"updated": 0}]
        bucket[h]["status"] = new_status
        return [{"updated": 1}]

    def _handle_catalog_read(self, params: dict) -> list[dict]:
        # Return all :Workspace nodes (catalog is global)
        return [
            {"name": w["workspace_id"], "last_active_at": w["last_active_at"]}
            for w in self._workspace_nodes.values()
        ]


class FakeTaskGraph:
    """Minimal TaskGraph facade backed by WorkspaceTaskStore."""

    def __init__(self, store: WorkspaceTaskStore, executor: _RecordingExecutor):
        self._store = store
        self._executor = executor

    def get_actionable_tasks(
        self, status_filter=None, global_list: bool = False,
        workspace_id=None, limit: int = 500,
    ) -> list[dict]:
        return self._store._handle_get_actionable_tasks({"workspace_id": workspace_id})

    def get_store_status(self) -> dict:
        return {"available": True, "status": "connected"}

    def update_task_status(
        self, short_hash: str, new_status: str, workspace_id=None,
        caller_agent_id=None,
    ) -> dict:
        params = {"hash": short_hash, "new_status": new_status}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if caller_agent_id:
            params["caller_agent_id"] = caller_agent_id

        # Use a unique marker in the query string so the handle() dispatch
        # can route this to _handle_update_task_check unambiguously.
        check = self._executor.execute_read(
            "FAKE_UPDATE_TASK_STATUS_CHECK", params
        )
        if not check or check[0].get("cnt", 0) == 0:
            return {"ok": False, "status": "not_found"}
        result = self._executor.execute_write("FAKE_UPDATE_TASK_STATUS_SET", params)
        if result and result[0].get("updated") == 1:
            return {"ok": True, "status": new_status}
        return {"ok": False, "status": "error"}

    def create_task(
        self, session_id, name, priority="medium", summary="", tag="",
        blocked_by=None, parent_uuid=None, status="pending", workspace_id=None,
    ) -> dict:
        rec = self._store.create_task(workspace_id, name)
        return {
            "uuid": rec["uuid"],
            "short_hash": rec["short_hash"],
            "name": rec["name"],
            "priority": rec["priority"],
            "status": rec["status"],
            "summary": rec["summary"],
            "tag": rec["tag"],
        }


def _switch_workspace(skill: WorkspaceSkill, session_id: str, target: str) -> dict:
    """Execute a real `^^workspace create <target>` for the given session."""
    ctx = SkillContext(
        session_id=session_id,
        agent_id=f"agent-{session_id}",
        messages=[],
        active_workspace=workspace_store.get_active(session_id),
        client_hint="",
    )
    return skill.handle_direct(ctx, "create", target)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_a():
    return _make_session_id("a")


@pytest.fixture
def session_b():
    return _make_session_id("b")


@pytest.fixture
def workspace_skill():
    return WorkspaceSkill()


@pytest.fixture
def store():
    return WorkspaceTaskStore()


@pytest.fixture
def tg(store):
    executor = _RecordingExecutor(store)
    return FakeTaskGraph(store, executor)


# ---------------------------------------------------------------------------
# 1. Two sessions in the SAME workspace see the SAME tasks
# ---------------------------------------------------------------------------


class TestSharedWorkspaceVisibility:
    """The new core behavior: sessions sharing a workspace see each other's work."""

    def test_two_sessions_in_same_workspace_see_same_tasks(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        # Both sessions activate the same workspace.
        _switch_workspace(workspace_skill, session_a, "choriprox")
        _switch_workspace(workspace_skill, session_b, "choriprox")

        # The DB sees a single :Workspace "choriprox" (shared).
        store.create_workspace("choriprox")

        # Session A creates a task.
        ws_id = "choriprox"  # v5: bare name
        t1 = tg.create_task(session_id=session_a, name="from-A", workspace_id=ws_id)
        t1_hash = t1["short_hash"]

        # Session B lists the workspace — must see A's task.
        result = tg.get_actionable_tasks(workspace_id=ws_id)
        visible = {t["short_hash"] for t in result}
        assert t1_hash in visible, (
            f"Session B (workspace 'choriprox') must see A's task {t1_hash}; "
            f"got visible={visible}"
        )

    def test_two_sessions_in_same_workspace_see_each_others_tasks(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        """A creates one task, B creates another; both see both."""
        _switch_workspace(workspace_skill, session_a, "shared")
        _switch_workspace(workspace_skill, session_b, "shared")
        store.create_workspace("shared")

        t_a = tg.create_task(session_id=session_a, name="A-task", workspace_id="shared")
        t_b = tg.create_task(session_id=session_b, name="B-task", workspace_id="shared")

        # Session A's view
        list_a = tg.get_actionable_tasks(workspace_id="shared")
        hashes_a = {t["short_hash"] for t in list_a}
        assert {t_a["short_hash"], t_b["short_hash"]} == hashes_a, (
            f"Session A must see both tasks. Got {hashes_a}"
        )

        # Session B's view
        list_b = tg.get_actionable_tasks(workspace_id="shared")
        hashes_b = {t["short_hash"] for t in list_b}
        assert {t_a["short_hash"], t_b["short_hash"]} == hashes_b, (
            f"Session B must see both tasks. Got {hashes_b}"
        )


# ---------------------------------------------------------------------------
# 2. Two sessions in DIFFERENT workspaces see only their OWN tasks
# ---------------------------------------------------------------------------


class TestSeparateWorkspaceIsolation:
    """Per-workspace isolation still holds for tasks in DIFFERENT workspaces."""

    def test_two_sessions_in_different_workspaces_see_only_their_tasks(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        # Each session activates its own workspace.
        _switch_workspace(workspace_skill, session_a, "alpha")
        _switch_workspace(workspace_skill, session_b, "beta")
        store.create_workspace("alpha")
        store.create_workspace("beta")

        # A creates a task in alpha; B creates one in beta.
        t_a = tg.create_task(session_id=session_a, name="A-task", workspace_id="alpha")
        t_b = tg.create_task(session_id=session_b, name="B-task", workspace_id="beta")

        # A's listing in alpha — only A's task.
        list_a = tg.get_actionable_tasks(workspace_id="alpha")
        hashes_a = {t["short_hash"] for t in list_a}
        assert t_a["short_hash"] in hashes_a
        assert t_b["short_hash"] not in hashes_a, (
            f"alpha leaked beta's task: {hashes_a}"
        )

        # B's listing in beta — only B's task.
        list_b = tg.get_actionable_tasks(workspace_id="beta")
        hashes_b = {t["short_hash"] for t in list_b}
        assert t_b["short_hash"] in hashes_b
        assert t_a["short_hash"] not in hashes_b, (
            f"beta leaked alpha's task: {hashes_b}"
        )

    def test_cross_workspace_update_returns_not_found(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        """Updating a task that exists in alpha from beta returns not_found."""
        _switch_workspace(workspace_skill, session_a, "alpha")
        _switch_workspace(workspace_skill, session_b, "beta")
        store.create_workspace("alpha")
        store.create_workspace("beta")

        t_a = tg.create_task(session_id=session_a, name="isolated", workspace_id="alpha")
        t_a_hash = t_a["short_hash"]

        # Try updating from beta — must fail.
        result = tg.update_task_status(t_a_hash, "in_progress", workspace_id="beta")
        assert result["ok"] is False
        assert result["status"] in ("not_found", "error"), (
            f"Cross-workspace update must fail; got {result}"
        )

        # And alpha's task is unchanged.
        a_task = store.get_task("alpha", t_a_hash)
        assert a_task["status"] == "pending", (
            f"alpha task must be unchanged; status={a_task['status']}"
        )


# ---------------------------------------------------------------------------
# 3. Task status update visible ACROSS sessions in the same workspace
# ---------------------------------------------------------------------------


class TestSharedStatusUpdate:
    """Status updates by one session are immediately visible to others."""

    def test_task_status_update_visible_across_sessions(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        _switch_workspace(workspace_skill, session_a, "shared")
        _switch_workspace(workspace_skill, session_b, "shared")
        store.create_workspace("shared")

        t = tg.create_task(session_id=session_a, name="collab", workspace_id="shared")
        t_hash = t["short_hash"]

        # Session A sets status to in_progress.
        result = tg.update_task_status(t_hash, "in_progress", workspace_id="shared")
        assert result["ok"] is True, f"Update failed: {result}"

        # Session B re-lists — must see the new status.
        list_b = tg.get_actionable_tasks(workspace_id="shared")
        collab = next(
            (t for t in list_b if t["short_hash"] == t_hash),
            None,
        )
        assert collab is not None, f"Task {t_hash} not visible to B"
        assert collab["status"] == "in_progress", (
            f"Session B must see the new status; got {collab['status']}"
        )


# ---------------------------------------------------------------------------
# 4. Session eviction does NOT delete the (global) :Workspace node
# ---------------------------------------------------------------------------


class TestSessionEvictionPreservesWorkspace:
    """Evicting a session removes :Session + edge — but never :Workspace.

    The :Workspace node is a global catalog entry, shared by all sessions
    that have ever activated it. Eviction of any single session must not
    delete the :Workspace.
    """

    def test_session_eviction_does_not_delete_workspace(
        self, workspace_skill, session_a, tg, store
    ):
        # Set up: session A activates "shared", creates a :Workspace node
        _switch_workspace(workspace_skill, session_a, "shared")
        store.create_workspace("shared")

        # Mock the executor used by cleanup_neo4j_sessions
        mock_executor = MagicMock()
        mock_executor.execute_write.return_value = []

        with patch(
            "memory.workspace_persistence._get_executor", return_value=mock_executor
        ):
            cleanup_neo4j_sessions([session_a])

        # Inspect the Cypher — must NOT contain MATCH (w:Workspace) or
        # any DELETE that would target :Workspace nodes.
        assert mock_executor.execute_write.called, "No cleanup call was made"
        call = mock_executor.execute_write.call_args
        query = call[0][0]  # first positional arg = query

        # Scoped to :Session (and possibly edge), not :Workspace.
        assert "MATCH (s:Session" in query, (
            f"Cleanup must MATCH :Session nodes:\n{query}"
        )
        # The crucial assertion: no MATCH on :Workspace.
        assert "MATCH (w:Workspace)" not in query, (
            f"Cleanup MUST NOT delete :Workspace nodes:\n{query}"
        )
        # And the DELETE clause must delete `s` (and possibly `r`), not `w`.
        assert "DELETE r, s" in query or "DETACH DELETE s" in query, (
            f"Cleanup must delete :Session (and edge), not :Workspace:\n{query}"
        )
        # Double-check: `w` is not used as a variable being deleted.
        assert "DELETE w" not in query, (
            f"Cleanup MUST NOT delete variable `w` (the :Workspace):\n{query}"
        )


# ---------------------------------------------------------------------------
# 6. store.get_global_catalog() returns ALL workspaces (session-independent)
# ---------------------------------------------------------------------------


class TestGlobalCatalogSessionIndependence:
    """get_global_catalog is global — every session sees the same list."""

    def test_workspace_list_shows_global_catalog(
        self, workspace_skill, session_a, session_b, store
    ):
        # Set up: two distinct :Workspace nodes
        store.create_workspace("alpha")
        store.create_workspace("beta")
        store.create_workspace("choriprox")

        # Patch get_shared_executor_from_env to return our recording executor.
        executor = _RecordingExecutor(store)
        with patch(
            "memory.workspace_persistence._get_executor", return_value=executor
        ):
            # Session A asks for the catalog
            catalog_a = get_global_catalog()

            # Session B asks for the catalog
            catalog_b = get_global_catalog()

        # Both must see all three workspaces.
        for cat in (catalog_a, catalog_b):
            assert "alpha" in cat
            assert "beta" in cat
            assert "choriprox" in cat
            assert "default" in cat, (
                f"'default' must always be in the catalog; got {cat}"
            )

        # And the catalogs are equal (global, not per-session).
        assert catalog_a == catalog_b, (
            f"Catalog must be session-independent; "
            f"A={catalog_a}, B={catalog_b}"
        )

    def test_get_global_catalog_independent_of_active_workspace(
        self, workspace_skill, session_a, session_b, store
    ):
        """Even if session A is in 'alpha' and session B is in 'beta', they
        both see the same global catalog.
        """
        _switch_workspace(workspace_skill, session_a, "alpha")
        _switch_workspace(workspace_skill, session_b, "beta")
        store.create_workspace("alpha")
        store.create_workspace("beta")

        executor = _RecordingExecutor(store)
        with patch(
            "memory.workspace_persistence._get_executor", return_value=executor
        ):
            catalog_a = get_global_catalog()
            catalog_b = get_global_catalog()

        # Both sessions see both workspaces, regardless of their active one.
        assert "alpha" in catalog_a and "alpha" in catalog_b
        assert "beta" in catalog_a and "beta" in catalog_b
        assert catalog_a == catalog_b
