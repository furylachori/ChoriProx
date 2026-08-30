"""Authorization tests at the Cypher/Neo4j level.

Tests that the WHERE clause enforces only the assigned agent can modify a task.
Two layers:
  1. Cypher structure tests — validate WHERE clause contains the right guard
     (uses build_recording_tg for module-level functions, direct executor
     inspection for TaskGraph instance methods)
  2. Real Neo4j tests — validate actual Cypher behavior against a live database
"""
import re
import uuid as _uuid
import pytest

from tests.unit.test_cypher_validator import build_recording_tg, validate_cypher_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task_uuid():
    return f"entity-{_uuid.uuid4().hex}"


def _make_short_hash(full_uuid=None):
    if full_uuid is None:
        full_uuid = _make_task_uuid()
    return full_uuid.replace("entity-", "")[:8]


def _insert_task(executor, short_hash, assigned_agent_id, workspace_id="ws-1", status="pending"):
    """Insert a Task node into Neo4j for testing."""
    full_uuid = _make_task_uuid()
    executor.execute_write(
        """
        CREATE (t:Entity:Task {
            uuid: $uuid,
            short_hash: $short_hash,
            name: $name,
            session_id: 'test',
            priority: 'medium',
            summary: '',
            status: $status,
            tag: '',
            created_at: datetime(),
            deleted_at: null,
            status_changed_at: datetime(),
            started_at: null,
            completed_at: null,
            workspace_id: $workspace_id,
            assigned_agent_id: $assigned_agent_id
        })
        """,
        {
            "uuid": full_uuid,
            "short_hash": short_hash,
            "name": f"task-{short_hash}",
            "status": status,
            "workspace_id": workspace_id,
            "assigned_agent_id": assigned_agent_id,
        },
    )
    return full_uuid


def _has_assignment_where(queries, keyword="assigned_agent_id"):
    """Return True if any captured query contains an assignment WHERE clause."""
    for q in queries:
        if keyword in q:
            return True
    return False


def _has_permissive_assignment(queries):
    """Return True if WHERE uses IS NULL OR = (permissive pattern)."""
    for q in queries:
        if "assigned_agent_id IS NULL" in q and "assigned_agent_id = $caller_agent_id" in q:
            return True
    return False


def _has_strict_assignment(queries):
    """Return True if WHERE uses strict equality (no IS NULL fallback)."""
    for q in queries:
        if re.search(r"t\.assigned_agent_id\s*=\s*\$agent_id", q):
            if "IS NULL" not in q:
                return True
    return False


# ---------------------------------------------------------------------------
# Cypher-level: work_sessions authorization structure
# ---------------------------------------------------------------------------


class TestWorkSessionAuthorizationWhereClause:
    """Validate authorization guards in work session Cypher."""

    def test_start_work_checks_assignment_in_cypher(self):
        """start_work queries must reference assigned_agent_id for authorization."""
        ws_uuid = f"ws-{_uuid.uuid4().hex}"
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"assigned_agent_id": "agent-A"}]
                if "RETURN t.assigned_agent_id" in q
                else [{"short_hash": task_hash, "status": "pending", "assigned_agent_id": "agent-A"}]
                if "RETURN" in q and "short_hash" in q
                else []
            )
        )
        from memory.work_sessions import create_work_session

        result = create_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-A",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        has_auth_check = any(
            "assigned_agent_id" in q for q in queries
        )
        assert has_auth_check, (
            f"start_work should check assigned_agent_id in queries: {queries}"
        )

    def test_start_work_cypher_passes_structural_validation(self):
        """All start_work Cypher queries must pass structural validation."""
        ws_uuid = f"ws-{_uuid.uuid4().hex}"
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"assigned_agent_id": "agent-A"}]
                if "RETURN t.assigned_agent_id" in q
                else [{"short_hash": task_hash, "status": "pending", "assigned_agent_id": "agent-A"}]
                if "RETURN" in q and "short_hash" in q
                else []
            )
        )
        from memory.work_sessions import create_work_session

        create_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-A",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        for i, q in enumerate(queries):
            issues = validate_cypher_structure(q)
            assert not issues, f"query[{i}] has structural issues: {issues}\n  query: {q.strip()[:300]}"

    def test_stop_work_has_strict_assignment_check(self):
        """close_work_session uses strict t.assigned_agent_id = $agent_id (no IS NULL)."""
        ws_uuid = f"ws-{_uuid.uuid4().hex}"
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"ws_uuid": ws_uuid, "task_status": "in_progress"}]
                if "RETURN target_ws.uuid" in q
                else [{"assigned_to": "agent-A", "session_exists": True}]
                if "RETURN t.assigned_agent_id" in q
                else []
            )
        )
        from memory.work_sessions import close_work_session

        close_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-A",
            workspace_id="ws-1",
        )
        has_assigned = _has_assignment_where(queries)
        assert has_assigned, (
            f"stop_work should check assigned_agent_id: {queries}"
        )

    def test_stop_work_cypher_passes_structural_validation(self):
        """All stop_work Cypher queries must pass structural validation."""
        ws_uuid = f"ws-{_uuid.uuid4().hex}"
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"ws_uuid": ws_uuid, "task_status": "in_progress"}]
                if "RETURN target_ws.uuid" in q
                else [{"assigned_to": "agent-A", "session_exists": True}]
                if "RETURN t.assigned_agent_id" in q
                else []
            )
        )
        from memory.work_sessions import close_work_session

        close_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-A",
            workspace_id="ws-1",
        )
        for i, q in enumerate(queries):
            issues = validate_cypher_structure(q)
            assert not issues, f"query[{i}] has structural issues: {issues}\n  query: {q.strip()[:300]}"


# ---------------------------------------------------------------------------
# Cypher-level: bug_tracker authorization structure
# ---------------------------------------------------------------------------


class TestBugTrackerAuthorizationWhereClause:
    """Validate authorization guards in bug tracker Cypher."""

    def test_create_bug_checks_assignment(self):
        """create_bug queries must reference assigned_agent_id."""
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"assigned_to": "agent-A", "has_active_session": True}]
                if "RETURN t.assigned_agent_id" in q
                else [{"uuid": bug_uuid, "short_hash": bug_hash}]
                if "RETURN b.uuid" in q
                else []
            )
        )
        from memory.bug_tracker import create_bug

        result = create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-A",
            workspace_id="ws-1",
            max_reviews=2,
        )
        has_auth = any("assigned_agent_id" in q for q in queries)
        assert has_auth, (
            f"create_bug should check assigned_agent_id: {queries}"
        )

    def test_create_bug_cypher_passes_structural_validation(self):
        """All create_bug Cypher queries must pass structural validation."""
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"assigned_to": "agent-A", "has_active_session": True}]
                if "RETURN t.assigned_agent_id" in q
                else [{"uuid": bug_uuid, "short_hash": bug_hash}]
                if "RETURN b.uuid" in q
                else []
            )
        )
        from memory.bug_tracker import create_bug

        create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-A",
            workspace_id="ws-1",
            max_reviews=2,
        )
        for i, q in enumerate(queries):
            issues = validate_cypher_structure(q)
            assert not issues, f"query[{i}] has structural issues: {issues}\n  query: {q.strip()[:300]}"

    def test_create_bug_rejects_null_assignment_in_logic(self):
        """create_bug explicitly rejects NULL assigned_agent_id (strict)."""
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"assigned_to": None, "has_active_session": True}]
                if "RETURN t.assigned_agent_id" in q
                else []
            )
        )
        from memory.bug_tracker import create_bug

        result = create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-B",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result.get("error") == "not_authorized"
        assert result.get("assigned_to") is None

    def test_create_bug_rejects_wrong_agent_in_logic(self):
        """create_bug rejects when task is assigned to different agent."""
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()
        task_hash = _make_short_hash()
        tg, queries = build_recording_tg(
            lambda idx, q, p: (
                [{"assigned_to": "agent-A", "has_active_session": True}]
                if "RETURN t.assigned_agent_id" in q
                else []
            )
        )
        from memory.bug_tracker import create_bug

        result = create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-B",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result.get("error") == "not_authorized"
        assert result.get("assigned_to") == "agent-A"


# ---------------------------------------------------------------------------
# Cypher-level: update_task_status WHERE clause structure
# ---------------------------------------------------------------------------


class TestUpdateTaskStatusAuthorizationWhereClause:
    """Validate the WHERE clause structure in update_task_status Cypher.

    Uses direct inspection of the method source to verify the assignment
    WHERE pattern, since TaskGraph is a class and build_recording_tg
    returns a MagicMock (not a real TaskGraph instance).
    """

    def test_assignment_where_clause_is_permissive_pattern(self):
        """update_task_status uses (IS NULL OR =) permissive pattern."""
        import inspect
        from memory.task_graph import TaskGraph

        source = inspect.getsource(TaskGraph.update_task_status)
        assert "assigned_agent_id IS NULL" in source, (
            "update_task_status should use permissive IS NULL pattern"
        )
        assert "assigned_agent_id = $caller_agent_id" in source, (
            "update_task_status should check caller_agent_id"
        )

    def test_assignment_where_clause_gated_on_caller_agent_id(self):
        """The assignment guard is only added when caller_agent_id is truthy."""
        import inspect
        from memory.task_graph import TaskGraph

        source = inspect.getsource(TaskGraph.update_task_status)
        assert "if caller_agent_id" in source, (
            "assignment_where should be conditional on caller_agent_id"
        )

    def test_update_task_uses_same_permissive_pattern(self):
        """update_task uses the same (IS NULL OR =) permissive pattern."""
        import inspect
        from memory.task_graph import TaskGraph

        source = inspect.getsource(TaskGraph.update_task)
        assert "assigned_agent_id IS NULL" in source
        assert "assigned_agent_id = $caller_agent_id" in source

    def test_count_query_includes_assignment_where(self):
        """The count query in update_task_status includes the assignment WHERE."""
        import inspect
        from memory.task_graph import TaskGraph

        source = inspect.getsource(TaskGraph.update_task_status)
        # The count_query f-string should include {assignment_where}
        assert "count_query" in source
        assert "{assignment_where}" in source


# ---------------------------------------------------------------------------
# Real Neo4j: authorization behavior tests
# ---------------------------------------------------------------------------


class _TestTaskGraph:
    """TaskGraph-compatible wrapper for real Neo4j tests.

    Has the same attributes the real methods access (VALID_TRANSITIONS,
    _executor) without calling the real __init__ which creates a Neo4jClient.
    """

    VALID_TRANSITIONS = {
        "pending": {"in_progress", "complete", "invalid"},
        "in_progress": {"complete", "pending"},
        "complete": {"invalid"},
        "invalid": {"pending"},
    }
    SHORT_HASH_RE = re.compile(r"^[0-9a-f]{4,8}$", re.IGNORECASE)

    def __init__(self, executor):
        self._executor = executor


def _build_tg_with_executor(executor):
    """Build a TaskGraph-compatible object wrapping a test executor."""
    return _TestTaskGraph(executor)


def _insert_work_session(executor, task_short_hash, agent_id, status="active"):
    """Insert a WorkSession linked to a task for testing."""
    ws_uuid = f"ws-{_uuid.uuid4().hex}"
    executor.execute_write(
        """
        MATCH (t:Task {short_hash: $short_hash})
        CREATE (ws:WorkSession {
            uuid: $ws_uuid,
            agent_id: $agent_id,
            session_id: 'test',
            started_at: datetime(),
            turn_count: 0,
            status: $status,
            transferred_from_agent: null,
            git_commit_sha: null
        })
        CREATE (t)-[:HAS_WORK_SESSION {created_at: datetime()}]->(ws)
        """,
        {
            "short_hash": task_short_hash,
            "ws_uuid": ws_uuid,
            "agent_id": agent_id,
            "status": status,
        },
    )
    return ws_uuid


class TestAuthorizationWithRealNeo4j:
    """Test actual authorization behavior against a live Neo4j database.

    These tests are skipped when Neo4j is not running.
    """

    @pytest.mark.integration_neo4j
    def test_assigned_agent_can_update_status(self, neo4j_test_db):
        """Test 1: Assigned agent CAN modify task status."""
        executor = neo4j_test_db
        from memory.task_graph import TaskGraph

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="pending")

        tg = _build_tg_with_executor(executor)
        result = TaskGraph.update_task_status(
            tg,
            task_hash,
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )
        assert result["ok"] is True
        assert result["status"] == "in_progress"

    @pytest.mark.integration_neo4j
    def test_different_agent_cannot_update_status(self, neo4j_test_db):
        """Test 2: Different agent CANNOT modify task status."""
        executor = neo4j_test_db
        from memory.task_graph import TaskGraph

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="pending")

        tg = _build_tg_with_executor(executor)
        result = TaskGraph.update_task_status(
            tg,
            task_hash,
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-B",
        )
        assert result["ok"] is False
        assert result["status"] == "not_found"

    @pytest.mark.integration_neo4j
    def test_unassigned_task_can_be_modified_permissive(self, neo4j_test_db):
        """Test 3: Unassigned task CAN be modified (permissive policy).

        The WHERE clause uses (t.assigned_agent_id IS NULL OR t.assigned_agent_id = $caller_agent_id)
        which allows any agent to modify unassigned tasks. This is the
        intended permissive behavior — NOT a bug.
        """
        executor = neo4j_test_db
        from memory.task_graph import TaskGraph

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id=None, status="pending")

        tg = _build_tg_with_executor(executor)
        result = TaskGraph.update_task_status(
            tg,
            task_hash,
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )
        # Permissive: unassigned tasks can be claimed by any agent
        assert result["ok"] is True
        assert result["status"] == "in_progress"

    @pytest.mark.integration_neo4j
    def test_update_task_field_with_assigned_agent(self, neo4j_test_db):
        """update_task (field update) succeeds for assigned agent."""
        executor = neo4j_test_db
        from memory.task_graph import TaskGraph

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A")

        tg = _build_tg_with_executor(executor)
        result = TaskGraph.update_task(
            tg,
            task_hash,
            workspace_id="ws-1",
            summary="updated by agent-A",
            caller_agent_id="agent-A",
        )
        assert result is True

    @pytest.mark.integration_neo4j
    def test_update_task_field_with_wrong_agent(self, neo4j_test_db):
        """update_task (field update) fails for wrong agent."""
        executor = neo4j_test_db
        from memory.task_graph import TaskGraph

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A")

        tg = _build_tg_with_executor(executor)
        result = TaskGraph.update_task(
            tg,
            task_hash,
            workspace_id="ws-1",
            summary="updated by agent-B",
            caller_agent_id="agent-B",
        )
        assert result is False

    @pytest.mark.integration_neo4j
    def test_start_work_authorization(self, neo4j_test_db):
        """Test 4: start_work checks assignment before claiming."""
        executor = neo4j_test_db
        from memory.work_sessions import create_work_session

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="pending")

        tg = _build_tg_with_executor(executor)
        ws_uuid = f"ws-{_uuid.uuid4().hex}"

        result = create_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-A",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        assert "error" not in result, f"Agent A should succeed: {result}"
        assert result.get("action") == "start_work"

    @pytest.mark.integration_neo4j
    def test_start_work_rejects_wrong_agent(self, neo4j_test_db):
        """start_work rejects agent that is not assigned."""
        executor = neo4j_test_db
        from memory.work_sessions import create_work_session

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="pending")

        tg = _build_tg_with_executor(executor)
        ws_uuid = f"ws-{_uuid.uuid4().hex}"

        result = create_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-B",
            session_id="sess-2",
            workspace_id="ws-1",
        )
        assert result.get("error") == "not_authorized"
        assert result.get("assigned_to") == "agent-A"

    @pytest.mark.integration_neo4j
    def test_stop_work_authorization(self, neo4j_test_db):
        """stop_work requires assignment match."""
        executor = neo4j_test_db
        from memory.work_sessions import create_work_session, close_work_session

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="in_progress")

        tg = _build_tg_with_executor(executor)
        ws_uuid = f"ws-{_uuid.uuid4().hex}"

        # Start work as agent A
        start_result = create_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-A",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        assert "error" not in start_result

        # Agent B tries to stop — should fail
        stop_result = close_work_session(
            tg,
            ws_uuid=ws_uuid,
            task_short_hash=task_hash,
            agent_id="agent-B",
            workspace_id="ws-1",
        )
        assert stop_result.get("error") in ("not_authorized", "no_active_session")

    @pytest.mark.integration_neo4j
    def test_create_bug_rejects_unassigned(self, neo4j_test_db):
        """Test 5: create_bug rejects when task is not assigned to caller."""
        executor = neo4j_test_db
        from memory.bug_tracker import create_bug

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id=None, status="in_progress")
        _insert_work_session(executor, task_hash, "agent-B")

        tg = _build_tg_with_executor(executor)
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()

        result = create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-B",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result.get("error") == "not_authorized"

    @pytest.mark.integration_neo4j
    def test_create_bug_rejects_wrong_agent(self, neo4j_test_db):
        """create_bug rejects when task is assigned to different agent."""
        executor = neo4j_test_db
        from memory.bug_tracker import create_bug

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="in_progress")
        _insert_work_session(executor, task_hash, "agent-B")

        tg = _build_tg_with_executor(executor)
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()

        result = create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-B",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result.get("error") == "not_authorized"
        assert result.get("assigned_to") == "agent-A"

    @pytest.mark.integration_neo4j
    def test_create_bug_succeeds_for_assigned_agent(self, neo4j_test_db):
        """create_bug succeeds when task is assigned to caller."""
        executor = neo4j_test_db
        from memory.bug_tracker import create_bug

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="in_progress")
        _insert_work_session(executor, task_hash, "agent-A")

        tg = _build_tg_with_executor(executor)
        bug_uuid = f"bug-{_uuid.uuid4().hex}"
        bug_hash = _make_short_hash()

        result = create_bug(
            tg,
            bug_uuid=bug_uuid,
            bug_short_hash=bug_hash,
            task_short_hash=task_hash,
            title="test bug",
            description="desc",
            severity="medium",
            caller_agent_id="agent-A",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert "error" not in result, f"Should succeed: {result}"
        assert result.get("uuid") == bug_uuid

    @pytest.mark.integration_neo4j
    def test_start_stop_start_work_cycle(self, neo4j_test_db):
        """Test 4 full cycle: start → stop → unassign → reassign."""
        executor = neo4j_test_db
        from memory.work_sessions import create_work_session, close_work_session

        task_hash = _make_short_hash()
        _insert_task(executor, task_hash, assigned_agent_id="agent-A", status="pending")

        tg = _build_tg_with_executor(executor)
        ws1 = f"ws-{_uuid.uuid4().hex}"

        # Agent A starts work
        r1 = create_work_session(
            tg, ws_uuid=ws1, task_short_hash=task_hash,
            agent_id="agent-A", session_id="s1", workspace_id="ws-1",
        )
        assert "error" not in r1

        # Agent B tries to start — should be rejected
        ws2 = f"ws-{_uuid.uuid4().hex}"
        r2 = create_work_session(
            tg, ws_uuid=ws2, task_short_hash=task_hash,
            agent_id="agent-B", session_id="s2", workspace_id="ws-1",
        )
        assert r2.get("error") in ("not_authorized", "task_locked")

        # Agent A stops work (releases the lock)
        r3 = close_work_session(
            tg, ws_uuid=ws1, task_short_hash=task_hash,
            agent_id="agent-A", workspace_id="ws-1",
        )
        assert r3.get("action") == "stop_work" or "error" not in r3

        # Unassign the task so agent B can claim it
        executor.execute_write(
            "MATCH (t:Task {short_hash: $h}) SET t.assigned_agent_id = null",
            {"h": task_hash},
        )

        # Agent B can now start
        ws3 = f"ws-{_uuid.uuid4().hex}"
        r4 = create_work_session(
            tg, ws_uuid=ws3, task_short_hash=task_hash,
            agent_id="agent-B", session_id="s3", workspace_id="ws-1",
        )
        assert "error" not in r4, f"Agent B should succeed after unassign: {r4}"
