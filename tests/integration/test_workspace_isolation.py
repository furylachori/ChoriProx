"""Integration tests for workspace isolation across operations.

Test 3.4: Workspace Isolation Across Operations

Verifies that workspace state acts as a complete isolation boundary:
  - Tasks in workspace A are invisible from workspace B
  - Bugs in workspace A are invisible from workspace B

The test exercises the full chain:
  WorkspaceSkill.handle_direct (^^workspace <name>)
    -> workspace_store.activate (per-session state mutation)
       -> ListTasksAction.execute / UpdateTaskAction.execute (workspace_id filter)
          -> tg.get_actionable_tasks / update_task_status (Cypher workspace_where)

DB layer (Neo4j) is mocked with script-driven fake executors; workspace
state and SkillContext are real.
"""
from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from proxy import state
from proxy.agent.agent_registry import registry as agent_registry
from proxy.skills.skill_registry import SkillContext
from proxy.tool_pipeline.actions.list_tasks import ListTasksAction
from proxy.tool_pipeline.actions.update_task import UpdateTaskAction
from proxy.workspace_state import workspace_store
from skills.workspace import WorkspaceSkill

from tests.mode_harness import create_test_agent

logger = logging.getLogger("proxy.tests.workspace_isolation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_id(name: str) -> str:
    """Stable per-test session id (avoids nanosecond collisions across fixtures)."""
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
    """Script-driven executor with state. Mirrors handoff_flow's _FakeTx pattern.

    Records every query and its params so tests can assert on the exact Cypher
    the production code emitted, and uses an in-memory task store keyed by
    (workspace_id, short_hash) to enforce workspace isolation at the DB level.
    """

    def __init__(self, store: "WorkspaceTaskStore"):
        self._store = store
        self.queries: list[dict] = []
        self._cursor = {"i": 0}

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
    """Workspace-keyed in-memory task/bug store.

    Simulates the Cypher workspace_where guard by filtering reads by
    workspace_id and rejecting writes that target a workspace other than
    the task's home workspace. The same short_hash is allowed to exist
    in multiple workspaces, but a list_tasks call with workspace_id=X
    must only see X's tasks.
    """

    def __init__(self):
        # workspace_id -> short_hash -> task dict
        self._tasks: dict[str, dict[str, dict]] = {}
        # workspace_id -> bug_short_hash -> bug dict
        self._bugs: dict[str, dict[str, dict]] = {}
        self._counter = 0

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

    def handle(self, query: str, params: dict) -> list[dict]:
        q = query

        # ---- get_actionable_tasks (list_tasks) ----
        if "collect(DISTINCT CASE WHEN blocker.uuid IS NOT NULL" in q or (
            "RETURN t.uuid AS uuid" in q and "size(active_blockers)" in q
        ):
            return self._handle_get_actionable_tasks(params)

        # ---- get_bug_with_reviews (review_bug/resolve_bug path) ----
        if "b.reported_by_agent AS reported_by_agent" in q and "b.avg_confidence" in q:
            return self._handle_get_bug(params)

        # ---- list_bugs ----
        if "MATCH (b:Bug)" in q and "b.reported_by_agent AS reported_by_agent" in q:
            return self._handle_list_bugs(params)

        # ---- update_task_status check (count + current_status) ----
        if "collect(t.status)[0] AS current_status" in q:
            return self._handle_update_task_check(params)

        # ---- update_task_status write ----
        if "SET t.status = $new_status" in q and "RETURN count(t) AS updated" in q:
            return self._handle_update_task_set(params)

        # ---- create_task write ----
        if "CREATE (t:Task" in q or "CREATE (t:Entity:Task" in q:
            return self._handle_create_task(params)

        # ---- create_bug lookup ----
        if "ws IS NOT NULL AS has_active_session" in q:
            return self._handle_bug_report_check(params)

        # ---- create_bug write ----
        if "CREATE (b:Bug" in q:
            return self._handle_bug_create(params)

        # ---- get_recent_review_notes (no-op for these tests) ----
        if "ReviewNote" in q and "recent_notes" in q:
            return []

        # Default: empty
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

    def _handle_list_bugs(self, params: dict) -> list[dict]:
        ws = params.get("workspace_id")
        bugs = list(self._bugs.get(ws, {}).values())
        return [
            {
                "uuid": b["uuid"],
                "short_hash": b["short_hash"],
                "title": b["title"],
                "severity": b["severity"],
                "status": b["status"],
                "review_count": b["review_count"],
                "max_reviews": b["max_reviews"],
                "created_at": b["created_at"],
                "reported_by_agent": b["reported_by_agent"],
            }
            for b in bugs
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

    def _handle_create_task(self, params: dict) -> list[dict]:
        # create_task runs via Cypher CREATE; the store-side population
        # happens in CreateTasksAction.execute() which calls the same
        # script. To keep tests simple, populate the store from the
        # explicit Python helper (create_task) — this branch is a
        # fallback.
        return []

    def _handle_bug_report_check(self, params: dict) -> list[dict]:
        return [{
            "assigned_to": "agent-1",
            "has_active_session": True,
        }]

    def _handle_bug_create(self, params: dict) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# Fake TaskGraph
# ---------------------------------------------------------------------------


class FakeTaskGraph:
    """TaskGraph-shaped facade backed by WorkspaceTaskStore.

    The production methods all delegate to self._executor. We expose
    the same methods the actions call (get_actionable_tasks,
    get_store_status, update_task_status, create_task).
    """

    def __init__(self, store: WorkspaceTaskStore, executor: _RecordingExecutor):
        self._store = store
        self._executor = executor

    def get_actionable_tasks(
        self, status_filter=None, global_list: bool = False,
        workspace_id=None, limit: int = 500,
    ) -> list[dict]:
        # Record the read so the test can assert what was queried
        self._executor.queries.append({
            "query": "get_actionable_tasks",
            "params": {"workspace_id": workspace_id, "limit": limit},
            "op": "read",
        })
        return self._store._handle_get_actionable_tasks({"workspace_id": workspace_id})

    def get_store_status(self) -> dict:
        return {"available": True, "status": "connected"}

    def update_task_status(
        self, short_hash: str, new_status: str, workspace_id=None,
        caller_agent_id=None,
    ) -> dict:
        params = {
            "hash": short_hash,
            "new_status": new_status,
            "expected_status": self._tasks_lookup(short_hash, workspace_id),
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        if caller_agent_id:
            params["caller_agent_id"] = caller_agent_id

        check = self._executor.execute_read(
            "update_task_status_check", params
        )
        if not check or check[0].get("cnt", 0) == 0:
            return {"ok": False, "status": "not_found"}
        if check[0].get("current_status") == new_status:
            return {"ok": True, "status": new_status, "changed": False}
        # SET path
        result = self._executor.execute_write(
            "update_task_status_set", params
        )
        if result and result[0].get("updated") == 1:
            # Mirror to store
            self._store._tasks.get(workspace_id, {}).get(short_hash, {})["status"] = new_status
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

    def _tasks_lookup(self, short_hash: str, workspace_id: str):
        bucket = self._store._tasks.get(workspace_id, {})
        t = bucket.get(short_hash)
        return t["status"] if t else None


# ---------------------------------------------------------------------------
# Workspace switch helper — drives the real `^^workspace` direct command
# ---------------------------------------------------------------------------


def _switch_workspace(skill: WorkspaceSkill, session_id: str, target: str) -> dict:
    """Execute a real `^^workspace <target>` direct command for the session.

    Uses `create` to ensure the workspace exists (auto-create via switch is
    rejected by the typo guard). Returns the parsed response dict.
    """
    ctx = SkillContext(
        session_id=session_id,
        agent_id=f"agent-{session_id}",
        messages=[],
        active_workspace=workspace_store.get_active(session_id),
        client_hint="",
    )
    # Use the explicit create subcommand to bypass typo guard and ensure
    # the workspace exists.
    result = skill.handle_direct(ctx, "create", target)
    return result


# ---------------------------------------------------------------------------
# Fixture
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


@pytest.fixture
def seed_agent(session_a):
    """Seed an agent for session_a so tool guard can resolve mode."""
    agent = create_test_agent(session_id=session_a, agent_id=f"agent-{session_a}")
    with agent_registry._lock:
        agent_registry._agents[session_a] = agent
    return agent


# ---------------------------------------------------------------------------
# Test 1.1: Tasks in workspace A invisible from workspace B
# ---------------------------------------------------------------------------


class TestWorkspaceTasksInvisible:
    """Tasks created in workspace A must not be visible from workspace B."""

    def test_create_in_a_then_list_from_b_returns_empty(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        # Step 1: switch session_a to workspace A
        _switch_workspace(workspace_skill, session_a, "alpha")

        # Step 2: create task T1 in workspace A
        ws_id_a = state.make_workspace_id("alpha")
        t1 = tg.create_task(
            session_id=session_a,
            name="task-1",
            workspace_id=ws_id_a,
        )
        t1_hash = t1["short_hash"]
        assert t1_hash in store._tasks[ws_id_a]

        # Step 3: switch session_b to workspace B
        _switch_workspace(workspace_skill, session_b, "beta")
        ws_id_b = state.make_workspace_id("beta")

        # Step 4: list_tasks from session_b's workspace — assert T1 absent
        result = ListTasksAction.execute(
            {}, session_b, tg, workspace_id=ws_id_b
        )
        visible_hashes = {t["short_hash"] for t in result["data"]["tasks"]}
        assert t1_hash not in visible_hashes, (
            f"Task {t1_hash} from workspace A leaked into workspace B's list: {visible_hashes}"
        )

    def test_update_from_b_returns_not_found(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        """update_task(short_hash=T1) from workspace B returns not_found."""
        # Set up: T1 exists only in workspace A
        _switch_workspace(workspace_skill, session_a, "alpha")
        ws_id_a = state.make_workspace_id("alpha")
        t1 = tg.create_task(session_id=session_a, name="isolated", workspace_id=ws_id_a)
        t1_hash = t1["short_hash"]

        # Switch session_b to workspace B
        _switch_workspace(workspace_skill, session_b, "beta")
        ws_id_b = state.make_workspace_id("beta")

        # Update from workspace B with T1's hash — must be not_found
        result = UpdateTaskAction.execute(
            {"short_hash": t1_hash, "new_status": "in_progress"},
            session_b, tg, workspace_id=ws_id_b,
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "not_found", (
            f"Expected not_found, got {result}"
        )

        # Verify T1 is unchanged in workspace A
        list_a = ListTasksAction.execute(
            {}, session_a, tg, workspace_id=ws_id_a
        )
        t1_after = next(t for t in list_a["data"]["tasks"] if t["short_hash"] == t1_hash)
        assert t1_after["status"] == "pending", (
            f"Task in A should be untouched, but status is {t1_after['status']}"
        )


# ---------------------------------------------------------------------------
# Test 1.2: Switch back to workspace A — tasks visible again
# ---------------------------------------------------------------------------


class TestWorkspaceRoundTrip:
    """A→B→A round trip — task visibility follows the active workspace."""

    def test_round_trip_visibility(
        self, workspace_skill, session_a, session_b, tg, store
    ):
        # Set up: T1 in workspace A
        _switch_workspace(workspace_skill, session_a, "alpha")
        ws_id_a = state.make_workspace_id("alpha")
        t1 = tg.create_task(session_id=session_a, name="round-trip", workspace_id=ws_id_a)
        t1_hash = t1["short_hash"]

        # Verify visible from A
        list_a_initial = ListTasksAction.execute(
            {}, session_a, tg, workspace_id=ws_id_a
        )
        assert any(t["short_hash"] == t1_hash for t in list_a_initial["data"]["tasks"])

        # Switch B to beta, verify T1 NOT visible
        _switch_workspace(workspace_skill, session_b, "beta")
        ws_id_b = state.make_workspace_id("beta")
        list_b = ListTasksAction.execute(
            {}, session_b, tg, workspace_id=ws_id_b
        )
        assert not any(t["short_hash"] == t1_hash for t in list_b["data"]["tasks"])

        # Switch session_a back to alpha (it's still in alpha, but force a round trip)
        _switch_workspace(workspace_skill, session_a, "beta")  # detour
        _switch_workspace(workspace_skill, session_a, "alpha")  # back

        # Verify T1 visible again from A
        list_a_after = ListTasksAction.execute(
            {}, session_a, tg, workspace_id=ws_id_a
        )
        assert any(t["short_hash"] == t1_hash for t in list_a_after["data"]["tasks"]), (
            "T1 should be visible again after switching back to workspace A"
        )


# ---------------------------------------------------------------------------
# Test 1.3: Bugs in workspace A invisible from workspace B
# ---------------------------------------------------------------------------


class TestWorkspaceBugsInvisible:
    """Bugs created in workspace A must not appear in workspace B's list_bugs."""

    def test_bug_in_a_not_in_b_listing(
        self, workspace_skill, session_a, session_b, store
    ):
        from memory.bug_tracker import list_bugs

        # Set up: T1 + B1 in workspace A
        _switch_workspace(workspace_skill, session_a, "alpha")
        ws_id_a = state.make_workspace_id("alpha")

        # Inject T1 and B1 directly into the store
        store.create_task(ws_id_a, "bug-host")
        b1_short = "deadbeef"
        store._bugs.setdefault(ws_id_a, {})[b1_short] = {
            "uuid": f"bug-{b1_short}-001",
            "short_hash": b1_short,
            "title": "leaky bug",
            "severity": "medium",
            "status": "open",
            "review_count": 0,
            "max_reviews": 2,
            "created_at": "2024-01-01T00:00:00Z",
            "reported_by_agent": "agent-1",
        }

        # Listing from A — bug visible
        bugs_a = list_bugs(tg=MagicMock(_executor=_RecordingExecutor(store)), workspace_id=ws_id_a)
        assert any(b["short_hash"] == b1_short for b in bugs_a), (
            f"B1 should be visible in workspace A: {bugs_a}"
        )

        # Switch session_b to beta
        _switch_workspace(workspace_skill, session_b, "beta")
        ws_id_b = state.make_workspace_id("beta")

        # Listing from B — bug NOT visible
        bugs_b = list_bugs(tg=MagicMock(_executor=_RecordingExecutor(store)), workspace_id=ws_id_b)
        assert not any(b["short_hash"] == b1_short for b in bugs_b), (
            f"B1 from workspace A leaked into workspace B's bug list: {bugs_b}"
        )

    def test_list_bugs_workspace_filter_enforced(
        self, workspace_skill, session_a, session_b, store
    ):
        """The list_bugs Cypher workspace_where guards the boundary."""
        from memory.bug_tracker import list_bugs

        # Inject B1 in A and B2 in B
        _switch_workspace(workspace_skill, session_a, "alpha")
        ws_id_a = state.make_workspace_id("alpha")
        store._bugs.setdefault(ws_id_a, {})["bugA1"] = {
            "uuid": "bugA1-001", "short_hash": "bugA1", "title": "alpha bug",
            "severity": "low", "status": "open", "review_count": 0,
            "max_reviews": 2, "created_at": "2024-01-01T00:00:00Z",
            "reported_by_agent": "agent-1",
        }

        _switch_workspace(workspace_skill, session_b, "beta")
        ws_id_b = state.make_workspace_id("beta")
        store._bugs.setdefault(ws_id_b, {})["bugB1"] = {
            "uuid": "bugB1-001", "short_hash": "bugB1", "title": "beta bug",
            "severity": "low", "status": "open", "review_count": 0,
            "max_reviews": 2, "created_at": "2024-01-02T00:00:00Z",
            "reported_by_agent": "agent-1",
        }

        # Listing from each workspace yields only its own bugs
        executor_a = _RecordingExecutor(store)
        executor_b = _RecordingExecutor(store)
        bugs_a = list_bugs(tg=MagicMock(_executor=executor_a), workspace_id=ws_id_a)
        bugs_b = list_bugs(tg=MagicMock(_executor=executor_b), workspace_id=ws_id_b)

        assert {b["short_hash"] for b in bugs_a} == {"bugA1"}
        assert {b["short_hash"] for b in bugs_b} == {"bugB1"}
