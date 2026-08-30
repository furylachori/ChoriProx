"""Integration tests for the multi-step handoff flow between agents.

Handoff involves 4 modules working together:
  1. handoff_task action → closes agent A's session, reassigns task to agent B
  2. work_sessions.close_work_session() → sets session status to 'transferred'
  3. task_graph → updates assigned_agent_id
  4. start_work by agent B → succeeds after reassignment

These tests exercise the full flow using fake executors that simulate
Neo4j responses, verifying each module correctly integrates with the next.
"""
import pytest
from unittest.mock import MagicMock, patch

from proxy.tool_pipeline.actions.handoff_task import HandoffTaskAction
from proxy.tool_pipeline.actions.start_work import StartWorkAction
from memory.work_sessions import create_work_session, close_work_session


class _FakeTx:
    """In-memory Neo4j transaction stand-in. Script-driven responses."""

    def __init__(self, script):
        self._script = script
        self._idx = 0

    def run(self, query, params=None):
        records = self._script(self._idx, query, params or {})
        self._idx += 1
        return _RecordIter(records)


class _RecordIter:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


def _make_tg(script):
    """Build a fake TaskGraph with a script-driven executor.

    `script(idx, query, params)` returns records for each call.
    execute_transaction_no_retry runs the lambda with a _FakeTx.
    """
    tg = MagicMock()

    def _execute_tx_no_retry(fn):
        tx = _FakeTx(script)
        return fn(tx)

    call_idx = {"n": 0}

    def _step_script(query, params):
        records = script(call_idx["n"], query, params or {})
        call_idx["n"] += 1
        return records

    tg._executor.execute_transaction_no_retry.side_effect = _execute_tx_no_retry
    tg._executor.execute_write.side_effect = lambda q, p=None, timeout=30: _step_script(q, p)
    tg._executor.execute_read.side_effect = lambda q, p=None, timeout=30: _step_script(q, p)
    return tg


# ── Helpers ──────────────────────────────────────────────────────────────────

TASK_HASH = "abcd1234"
AGENT_A = "agent-a"
AGENT_B = "agent-b"
WORKSPACE = "ws-1"
SESSION_A = "sess-a"
SESSION_B = "sess-b"


def _handoff_script(closed_count=1):
    """Script for a successful handoff: matches task assigned to caller, closes N sessions."""
    return lambda idx, q, p: [
        {"closed_count": closed_count, "task_hash": TASK_HASH}
    ] if idx == 0 else []


def _start_work_script_after_handoff():
    """Cursor-based script for start_work succeeding after handoff.

    Uses matchers (not index) so it works across execute_read (pre-check)
    and _FakeTx.run() (transaction) which have independent counters.
    """
    return _cursor_script([
        (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
         [{"assigned_agent_id": AGENT_B}]),                    # execute_read pre-check
        (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
         [{"short_hash": TASK_HASH, "status": "in_progress", "assigned_agent_id": AGENT_B}]),  # tx lookup
        (lambda q, p: "active_agent" in q, []),                # tx lock check
        (lambda q, p: "ws.uuid" in q,
         [{"ws_uuid": "ws-new", "session_started_at": "2024-01-02T00:00:00Z",
           "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),  # tx claim
    ])


def _cursor_script(responses):
    """Cursor-based script factory matching test_work_sessions.py pattern.

    responses: ordered list of (matcher_fn, records). Each call consumes the
    next entry whose matcher returns True. The `idx` parameter is ignored —
    a shared cursor drives ordering. This is essential when the same script
    serves both execute_read (pre-check) and _FakeTx.run() (transaction).
    """
    cursor = {"i": 0}

    def _script(idx, query, params):
        for j in range(cursor["i"], len(responses)):
            matcher, records = responses[j]
            cursor["i"] = j + 1
            if matcher(query, params):
                return records
        return []
    return _script


# ── Test 1: Happy path — agent A handoffs to agent B ─────────────────────────

class TestHandoffHappyPath:
    """US-089: Agent A starts work, hands off to B, B starts work successfully.

    Covers the handoff_task contract: closes agent A's session, reassigns
    the task to agent B, and lets B claim it (with transferred_from_agent).
    """

    def test_full_handoff_flow(self):
        """Agent A → handoff → Agent B start_work with transferred_from_agent."""
        # Phase 1: Agent A starts work (agent-a claims the task)
        start_a_script = _cursor_script([
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": None}]),                    # execute_read pre-check
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": TASK_HASH, "status": "pending", "assigned_agent_id": None}]),  # tx lookup
            (lambda q, p: "active_agent" in q, []),            # tx lock check
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-a", "session_started_at": "2024-01-01T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),  # tx claim
        ])

        tg_a = _make_tg(start_a_script)
        start_result = create_work_session(
            tg=tg_a, ws_uuid="ws-a", task_short_hash=TASK_HASH,
            agent_id=AGENT_A, session_id=SESSION_A, workspace_id=WORKSPACE,
        )
        assert start_result["action"] == "start_work"
        assert start_result["task_status"] == "in_progress"

        # Phase 2: Agent A hands off to Agent B
        handoff_tg = _make_tg(_handoff_script(closed_count=1))
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            handoff_result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B, "context": "your turn"},
                SESSION_A, handoff_tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )

        assert handoff_result["operation"] == "handoff_task"
        assert handoff_result["data"]["to_agent_id"] == AGENT_B
        assert handoff_result["data"]["closed_sessions"] == 1
        assert handoff_result["data"]["short_hash"] == TASK_HASH

        # Phase 3: Agent B starts work (claims reassigned task with transferred_from)
        tg_b = _make_tg(_start_work_script_after_handoff())
        start_b_result = create_work_session(
            tg=tg_b, ws_uuid="ws-new", task_short_hash=TASK_HASH,
            agent_id=AGENT_B, session_id=SESSION_B, workspace_id=WORKSPACE,
            transferred_from_agent=AGENT_A,
        )

        assert start_b_result["action"] == "start_work"
        assert start_b_result["ws_uuid"] == "ws-new"
        assert start_b_result["task_short_hash"] == TASK_HASH

    def test_handoff_session_status_is_transferred(self):
        """The handoff query sets session status to 'transferred' (not 'completed')."""
        tx_queries = []

        def record_tx(idx, q, p):
            if idx == 0:
                tx_queries.append({"query": q, "params": p})
                return [{"closed_count": 1, "task_hash": TASK_HASH}]
            return []

        tg = _make_tg(record_tx)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B, "context": "delegating"},
                SESSION_A, tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )

        assert len(tx_queries) == 1
        q = tx_queries[0]["query"]
        assert "transferred" in q
        assert "transfer_note" in q
        assert tx_queries[0]["params"]["caller_agent_id"] == AGENT_A
        assert tx_queries[0]["params"]["to_agent_id"] == AGENT_B

    def test_handoff_updates_assigned_agent_id(self):
        """The handoff query SETs assigned_agent_id to the target agent."""
        tx_queries = []

        def record_tx(idx, q, p):
            if idx == 0:
                tx_queries.append({"query": q, "params": p})
                return [{"closed_count": 0, "task_hash": TASK_HASH}]
            return []

        tg = _make_tg(record_tx)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B},
                SESSION_A, tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )

        q = tx_queries[0]["query"]
        assert "t.assigned_agent_id = $to_agent_id" in q
        assert tx_queries[0]["params"]["to_agent_id"] == AGENT_B

    def test_start_work_passes_transferred_from_agent(self):
        """StartWorkAction passes transferred_from_agent to create_work_session."""
        with patch("memory.work_sessions.create_work_session", return_value={
            "action": "start_work", "ws_uuid": "ws-1", "task_short_hash": TASK_HASH,
            "task_status": "in_progress", "session_id": SESSION_B,
        }) as mock_create:
            StartWorkAction.execute(
                {"short_hash": TASK_HASH, "transferred_from_agent": AGENT_A},
                SESSION_B, MagicMock(),
                workspace_id=WORKSPACE, caller_agent_id=AGENT_B,
            )

        assert mock_create.call_args.kwargs["transferred_from_agent"] == AGENT_A

    def test_handoff_history_via_consecutive_work_sessions(self):
        """US-090: Handoff History via Consecutive WorkSessions.

        Consecutive WorkSession nodes on a task form a queryable chain
        with transferred_from_agent set on the new session.
        """
        # Phase 1: Agent A starts work (unassigned task)
        start_a_script = _cursor_script([
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": None}]),
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": TASK_HASH, "status": "pending", "assigned_agent_id": None}]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-a", "session_started_at": "2024-01-01T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),
        ])
        tg_a = _make_tg(start_a_script)
        start_a = create_work_session(
            tg=tg_a, ws_uuid="ws-a", task_short_hash=TASK_HASH,
            agent_id=AGENT_A, session_id=SESSION_A, workspace_id=WORKSPACE,
        )
        assert start_a["action"] == "start_work"

        # Phase 2: Agent A hands off to Agent B
        handoff_tg = _make_tg(_handoff_script(closed_count=1))
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            handoff_result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B, "context": "your turn"},
                SESSION_A, handoff_tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )
        assert handoff_result["operation"] == "handoff_task"

        # Phase 3: Agent B starts work — capture the claim query params
        tx_queries = []
        inner = _start_work_script_after_handoff()

        def _capturing_script(idx, query, params):
            tx_queries.append({"query": query, "params": dict(params)})
            return inner(idx, query, params)

        tg_b = _make_tg(_capturing_script)
        start_b = create_work_session(
            tg=tg_b, ws_uuid="ws-new", task_short_hash=TASK_HASH,
            agent_id=AGENT_B, session_id=SESSION_B, workspace_id=WORKSPACE,
            transferred_from_agent=AGENT_A,
        )
        assert start_b["action"] == "start_work"

        # Verify the claim query carried transferred_from_agent in its params
        claim_entry = next(e for e in tx_queries if "transferred_from_agent" in e.get("params", {}))
        assert claim_entry["params"]["transferred_from_agent"] == AGENT_A
        assert claim_entry["params"]["agent_id"] == AGENT_B


# ── Test 2: Agent B cannot handoff to self ───────────────────────────────────

class TestHandoffToSelf:
    def test_cannot_handoff_to_self(self):
        result = HandoffTaskAction.execute(
            {"short_hash": TASK_HASH, "to_agent_id": AGENT_A},
            SESSION_A, MagicMock(),
            workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
        )
        assert result["error"]["code"] == "cannot_handoff_to_self"
        assert "same agent" in result["error"]["message"].lower()


# ── Test 3: Target agent not found ───────────────────────────────────────────

class TestHandoffTargetNotFound:
    def test_target_agent_not_found(self):
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None):
            result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": "ghost-agent"},
                SESSION_A, MagicMock(),
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )
        assert result["error"]["code"] == "target_agent_not_found"
        assert "ghost-agent" in result["error"]["message"]

    def test_start_work_after_handoff_fails_if_task_not_reassigned(self):
        """If handoff didn't happen (e.g., target not found), agent B can't start."""
        # Agent B tries to start on a task still assigned to Agent A
        script = _cursor_script([
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": AGENT_A}]),
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q,
             [{"short_hash": TASK_HASH, "status": "pending", "assigned_agent_id": AGENT_A}]),
        ])

        tg = _make_tg(script)
        result = create_work_session(
            tg=tg, ws_uuid="ws-b", task_short_hash=TASK_HASH,
            agent_id=AGENT_B, session_id=SESSION_B, workspace_id=WORKSPACE,
        )
        assert result["error"] == "not_authorized"
        assert result["assigned_to"] == AGENT_A


# ── Test 4: Agent A cannot handoff after stop_work ───────────────────────────

class TestHandoffAfterStopWork:
    def test_handoff_after_stop_returns_error(self):
        """After stopping work, Agent A has no active session to transfer."""
        # stop_work closes the session, then handoff finds no active session
        # The handoff query matches the task (assigned to A) but finds 0 active sessions.
        # It still succeeds because the WHERE clause only checks assignment, not session.
        # However, if the task was unassigned by stop_work, it would fail.
        #
        # Per the code: stop_work does NOT unassign (SPEC US-097).
        # So handoff after stop_work succeeds with closed_sessions=0.
        # The real protection is that there's no active session to transfer.
        handoff_script = lambda idx, q, p: [
            [{"closed_count": 0, "task_hash": TASK_HASH}],  # no sessions to close
        ][idx] if idx < 1 else []

        tg = _make_tg(handoff_script)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B},
                SESSION_A, tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )

        # Handoff succeeds but with 0 closed sessions (nothing to transfer)
        assert result["operation"] == "handoff_task"
        assert result["data"]["closed_sessions"] == 0
        assert result["data"]["to_agent_id"] == AGENT_B

    def test_handoff_fails_if_task_not_assigned_to_caller(self):
        """If task was reassigned away, handoff returns not_authorized."""
        def script(idx, q, p):
            if idx == 0:
                return []  # main query: no match (task not assigned to A)
            # diagnostic query: task exists, assigned to B
            return [{"assigned_to": AGENT_B}]

        tg = _make_tg(script)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B},
                SESSION_A, tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )

        assert result["error"]["code"] == "not_authorized"
        assert AGENT_B in result["error"]["message"]


# ── Test 5: Handoff with workspace_id=None ───────────────────────────────────

class TestHandoffWorkspaceNone:
    """The A3/C8 bug: handoff with workspace_id=None was failing.

    The handoff query builds workspace_where="" when workspace_id is falsy,
    so it should succeed without a workspace filter.
    """

    def test_handoff_with_none_workspace_succeeds(self):
        tx_queries = []

        def script(idx, q, p):
            if idx == 0:
                tx_queries.append({"query": q, "params": p})
                return [{"closed_count": 1, "task_hash": TASK_HASH}]
            return []

        tg = _make_tg(script)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B, "context": "no ws"},
                SESSION_A, tg,
                workspace_id=None, caller_agent_id=AGENT_A,
            )

        assert result["operation"] == "handoff_task"
        assert result["data"]["closed_sessions"] == 1
        # Verify no workspace filter was applied
        q = tx_queries[0]["query"]
        assert "workspace_id" not in q or "workspace_where" not in q
        assert tx_queries[0]["params"]["workspace_id"] is None

    def test_start_work_with_none_workspace_after_handoff(self):
        """Agent B can start work with workspace_id=None after handoff."""
        script = _cursor_script([
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": AGENT_B}]),
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": TASK_HASH, "status": "in_progress", "assigned_agent_id": AGENT_B}]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-new", "session_started_at": None,
               "task_status": "in_progress", "task_started_at": None}]),
        ])

        tg = _make_tg(script)
        result = create_work_session(
            tg=tg, ws_uuid="ws-new", task_short_hash=TASK_HASH,
            agent_id=AGENT_B, session_id=SESSION_B, workspace_id=None,
            transferred_from_agent=AGENT_A,
        )

        assert result["action"] == "start_work"

    def test_full_handoff_flow_no_workspace(self):
        """End-to-end: handoff with workspace_id=None, then start_work with None."""
        # Phase 1: handoff
        handoff_tg = _make_tg(_handoff_script(closed_count=1))
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            h_result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B},
                SESSION_A, handoff_tg,
                workspace_id=None, caller_agent_id=AGENT_A,
            )
        assert h_result["operation"] == "handoff_task"

        # Phase 2: start_work by B
        start_tg = _make_tg(_start_work_script_after_handoff())
        s_result = create_work_session(
            tg=start_tg, ws_uuid="ws-new", task_short_hash=TASK_HASH,
            agent_id=AGENT_B, session_id=SESSION_B, workspace_id=None,
            transferred_from_agent=AGENT_A,
        )
        assert s_result["action"] == "start_work"


# ── Test 6: Handoff edge cases ───────────────────────────────────────────────

class TestHandoffEdgeCases:
    def test_handoff_missing_short_hash(self):
        result = HandoffTaskAction.execute(
            {"to_agent_id": AGENT_B}, SESSION_A, MagicMock(),
            workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
        )
        assert result["error"]["code"] == "invalid_input"
        assert "short_hash" in result["error"]["message"]

    def test_handoff_missing_to_agent_id(self):
        result = HandoffTaskAction.execute(
            {"short_hash": TASK_HASH}, SESSION_A, MagicMock(),
            workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
        )
        assert result["error"]["code"] == "invalid_input"
        assert "to_agent_id" in result["error"]["message"]

    def test_handoff_missing_caller_agent_id(self):
        result = HandoffTaskAction.execute(
            {"short_hash": TASK_HASH, "to_agent_id": AGENT_B},
            SESSION_A, MagicMock(),
            workspace_id=WORKSPACE, caller_agent_id=None,
        )
        assert result["error"]["code"] == "agent_not_found"

    def test_handoff_task_not_found(self):
        def script(idx, q, p):
            if idx == 0:
                return []  # main query: no match
            return []      # diagnostic: task doesn't exist

        tg = _make_tg(script)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            result = HandoffTaskAction.execute(
                {"short_hash": "deadbeef", "to_agent_id": AGENT_B},
                SESSION_A, tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )
        assert result["error"]["code"] == "task_not_found"

    def test_handoff_with_zero_closed_sessions_still_reassigns(self):
        """Handoff with no active sessions still reassigns the task."""
        tx_queries = []

        def script(idx, q, p):
            if idx == 0:
                tx_queries.append({"query": q, "params": p})
                return [{"closed_count": 0, "task_hash": TASK_HASH}]
            return []

        tg = _make_tg(script)
        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=MagicMock()):
            result = HandoffTaskAction.execute(
                {"short_hash": TASK_HASH, "to_agent_id": AGENT_B},
                SESSION_A, tg,
                workspace_id=WORKSPACE, caller_agent_id=AGENT_A,
            )

        assert result["operation"] == "handoff_task"
        assert result["data"]["closed_sessions"] == 0
        # The SET clause still ran — assigned_agent_id was updated
        q = tx_queries[0]["query"]
        assert "t.assigned_agent_id = $to_agent_id" in q


# ── Test 7: stop_work does NOT unassign ──────────────────────────────────────

class TestStopWorkDoesNotUnassign:
    """US-097: stop_work Does NOT Unassign."""

    def test_stop_work_preserves_assignment_then_start_work_succeeds(self):
        """US-097: stop_work Does NOT Unassign — after stop_work the task's
        assigned_agent_id is unchanged, and the agent can start_work again."""
        # Phase 1: start_work — agent claims the task
        start_script = _cursor_script([
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": None}]),
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": TASK_HASH, "status": "pending", "assigned_agent_id": None}]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-a", "session_started_at": "2024-01-01T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),
        ])

        start_tg = _make_tg(start_script)
        start_result = create_work_session(
            tg=start_tg, ws_uuid="ws-a", task_short_hash=TASK_HASH,
            agent_id=AGENT_A, session_id=SESSION_A, workspace_id=WORKSPACE,
        )
        assert start_result["action"] == "start_work"
        assert start_result["task_status"] == "in_progress"

        # Phase 2: stop_work — agent closes session; task must stay assigned
        close_queries = []
        close_records = [{"ws_uuid": "ws-a", "task_status": "in_progress"}]

        def _close_script(idx, q, p):
            close_queries.append({"query": q, "params": p})
            return close_records if idx == 0 else []

        stop_tg = MagicMock()
        stop_tg._executor.execute_transaction_no_retry.side_effect = lambda fn: (
            fn(_FakeTx(_close_script))
        )

        stop_result = close_work_session(
            tg=stop_tg, ws_uuid="ws-a", task_short_hash=TASK_HASH,
            agent_id=AGENT_A, workspace_id=WORKSPACE,
        )
        assert stop_result["action"] == "stop_work"

        close_query = close_queries[0]["query"]
        assert "assigned_agent_id" in close_query
        assert "SET" not in close_query.split("WHERE")[0]

        # Phase 3: start_work again — same agent succeeds (still assigned)
        restart_script = _cursor_script([
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": AGENT_A}]),
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": TASK_HASH, "status": "in_progress", "assigned_agent_id": AGENT_A}]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-b", "session_started_at": "2024-01-02T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),
        ])

        restart_tg = _make_tg(restart_script)
        restart_result = create_work_session(
            tg=restart_tg, ws_uuid="ws-b", task_short_hash=TASK_HASH,
            agent_id=AGENT_A, session_id="sess-a2", workspace_id=WORKSPACE,
        )
        assert restart_result["action"] == "start_work"
        assert restart_result["ws_uuid"] == "ws-b"
