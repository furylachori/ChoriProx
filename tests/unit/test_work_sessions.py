"""Tests for memory.work_sessions — create/close/get_active/cleanup_stale helpers.

These tests use fake executors that record Cypher queries and return
scripted responses. They mirror the pattern in tests/unit/test_proxy_ticket_actions.py.
"""
import pytest
from unittest.mock import MagicMock

from memory.work_sessions import (
    create_work_session,
    close_work_session,
    get_active_sessions,
    cleanup_stale_sessions,
    STALE_SESSION_HOURS,
)


class _FakeTx:
    """In-memory Neo4j transaction stand-in. Each run() inspects the query
    and the params to return appropriate fake records."""

    def __init__(self, script):
        self._script = script
        self._idx = 0
        self.records = []

    def run(self, query, params=None):
        records = self._script(self._idx, query, params or {})
        self._idx += 1
        self.records.append({"query": query, "params": params, "records": records})
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

    `script` is a stateful callable (idx, query, params) -> records. Each
    execute_transaction_no_retry call gets its own counter via _FakeTx,
    so multiple tx.run() calls within one transaction step through
    the script in order. execute_write / execute_read use a fresh counter
    since they represent a single Cypher call.
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


def _script_factory(responses):
    """responses: ordered list of (matcher, records). Each tx.run() / single
    Cypher call consumes the next entry that matches. If none match, returns []."""
    cursor = {"i": 0}

    def _script(idx, query, params):
        for j in range(cursor["i"], len(responses)):
            matcher, records = responses[j]
            cursor["i"] = j + 1
            if matcher(query, params):
                return records
        return []
    return _script


class TestCreateWorkSessionHappyPath:
    def test_creates_session_for_unassigned_pending_task(self):
        """US-073: Agent claims a pending task: WorkSession node + HAS_WORK_SESSION + status transition."""
        responses = [
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q, [
                {"assigned_agent_id": None}
            ]),
            (lambda q, p: "assigned_agent_id" in q, [
                {"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": None}
            ]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q, [
                {
                    "ws_uuid": "ws-abc123",
                    "session_started_at": "2024-01-01T00:00:00Z",
                    "task_status": "in_progress",
                    "task_started_at": "2024-01-01T00:00:00Z",
                }
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_work_session(
            tg=tg,
            ws_uuid="ws-abc123",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        assert result["action"] == "start_work"
        assert result["ws_uuid"] == "ws-abc123"
        assert result["task_status"] == "in_progress"
        assert result["task_short_hash"] == "abcd1234"
        assert result["session_id"] == "sess-1"

    def test_accepts_no_workspace(self):
        """workspace_id=None is allowed (no WHERE filter)."""
        responses = [
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q, [
                {"assigned_agent_id": None}
            ]),
            (lambda q, p: "assigned_agent_id" in q, [
                {"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": None}
            ]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q, [
                {"ws_uuid": "ws-1", "session_started_at": None, "task_status": "in_progress", "task_started_at": None}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_work_session(
            tg=tg,
            ws_uuid="ws-1",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            session_id="sess-1",
            workspace_id=None,
        )
        assert result["action"] == "start_work"


class TestCreateWorkSessionErrorCases:
    def test_rejects_if_task_not_found(self):
        """No rows for the task → task_not_found."""
        responses = [
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q, []),
            (lambda q, p: "assigned_agent_id" in q, []),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_work_session(
            tg=tg,
            ws_uuid="ws-1",
            task_short_hash="deadbeef",
            agent_id="agent-1",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        assert result["error"] == "task_not_found"

    def test_rejects_if_task_assigned_to_different_agent(self):
        """Task already claimed by another agent → not_authorized."""
        responses = [
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q, [
                {"assigned_agent_id": "other-agent"}
            ]),
            (lambda q, p: "assigned_agent_id" in q, [
                {"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": "other-agent"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_work_session(
            tg=tg,
            ws_uuid="ws-1",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        assert result["error"] == "not_authorized"
        assert result["assigned_to"] == "other-agent"

    def test_rejects_if_active_session_from_different_agent_within_24h(self):
        """US-074: Active session by other agent < 24h old → task_locked."""
        responses = [
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q, [
                {"assigned_agent_id": None}
            ]),
            (lambda q, p: "assigned_agent_id" in q, [
                {"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": None}
            ]),
            (lambda q, p: "active_agent" in q, [{"active_agent": "other-agent"}]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_work_session(
            tg=tg,
            ws_uuid="ws-1",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            session_id="sess-1",
            workspace_id="ws-1",
        )
        assert result["error"] == "task_locked"
        assert result["active_agent"] == "other-agent"

    def test_empty_short_hash_returns_invalid_input(self):
        tg = MagicMock()
        result = create_work_session(
            tg=tg,
            ws_uuid="ws-1",
            task_short_hash="",
            agent_id="agent-1",
            session_id="sess-1",
        )
        assert result["error"] == "invalid_input"


class TestCloseWorkSession:
    def _lookup_tg(self, assigned_to=None):
        responses = [
            (lambda q, p: "assigned_to" in q, [{}] if assigned_to is not None else []),
        ]
        return _make_tg(_script_factory(responses))

    def test_close_work_session_happy_path(self):
        """US-077: Stop a session by ws_uuid. Atomic: assignment check + session close in one tx."""
        # New API: single execute_transaction_no_retry with tx callback.
        # The transaction runs the close_query (returns a record) and skips the
        # diagnostic check_query (since the close succeeded).
        close_records = [{"ws_uuid": "ws-abc", "task_status": "complete"}]
        tg = MagicMock()
        tg._executor.execute_transaction_no_retry.side_effect = lambda fn: (
            fn(_FakeTx(_script_factory([(lambda q, p: True, close_records)])))
        )

        result = close_work_session(
            tg=tg,
            ws_uuid="ws-abc",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            workspace_id="ws-1",
        )
        assert result["action"] == "stop_work"
        assert result["ws_uuid"] == "ws-abc"
        assert result["task_status"] == "complete"

    def test_close_work_session_rejects_if_ws_uuid_doesnt_match(self):
        """US-078/US-079: No active session matching ws_uuid → no_active_session."""
        # New API: close_query returns 0 rows, then diagnostic check_query runs.
        close_records = []
        check_records = [{"assigned_to": "agent-1", "session_exists": True}]
        tg = MagicMock()
        tg._executor.execute_transaction_no_retry.side_effect = lambda fn: (
            fn(_FakeTx(_script_factory([
                (lambda q, p: "FOREACH" in q, close_records),  # close query: empty
                (lambda q, p: "FOREACH" not in q, check_records),  # check query: found
            ])))
        )

        result = close_work_session(
            tg=tg,
            ws_uuid="ws-nope",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            workspace_id="ws-1",
        )
        assert result["error"] == "no_active_session"

    def test_close_work_session_with_new_status_complete(self):
        """new_status='complete' triggers task status transition."""
        close_records = [{"ws_uuid": "ws-abc", "task_status": "complete"}]
        tg = MagicMock()
        tg._executor.execute_transaction_no_retry.side_effect = lambda fn: (
            fn(_FakeTx(_script_factory([(lambda q, p: True, close_records)])))
        )

        result = close_work_session(
            tg=tg,
            ws_uuid="ws-abc",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            workspace_id="ws-1",
            new_status="complete",
        )
        assert result["task_status"] == "complete"

    def test_close_work_session_rejects_invalid_new_status(self):
        """Invalid new_status → invalid_input (no transaction opened)."""
        tg = MagicMock()
        result = close_work_session(
            tg=tg,
            ws_uuid="ws-1",
            task_short_hash="abcd1234",
            agent_id="agent-1",
            new_status="bogus",
        )
        assert result["error"] == "invalid_input"


class TestSameAgentMultipleSessions:
    """US-075: Same Agent Can Start Multiple Sessions."""

    def test_same_agent_can_start_two_sessions_on_same_task(self):
        """US-075: Same Agent Can Start Multiple Sessions — an agent can
        create two work sessions on the same task, yielding distinct ws_uuids."""
        responses = [
            # Call 1: pre-check (unassigned)
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": None}]),
            # Call 1: tx lookup (pending, unassigned)
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": None}]),
            # Call 1: tx active_agent (no lock)
            (lambda q, p: "active_agent" in q, []),
            # Call 1: tx claim
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-first", "session_started_at": "2024-01-01T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),
            # Call 2: pre-check (now assigned to agent-1)
            (lambda q, p: "assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": "agent-1"}]),
            # Call 2: tx lookup (in_progress, assigned to agent-1)
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": "abcd1234", "status": "in_progress", "assigned_agent_id": "agent-1"}]),
            # Call 2: tx active_agent (no lock)
            (lambda q, p: "active_agent" in q, []),
            # Call 2: tx claim
            (lambda q, p: "ws.uuid" in q,
             [{"ws_uuid": "ws-second", "session_started_at": "2024-01-02T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}]),
        ]
        tg = _make_tg(_script_factory(responses))

        result1 = create_work_session(
            tg=tg, ws_uuid="ws-first", task_short_hash="abcd1234",
            agent_id="agent-1", session_id="sess-1", workspace_id="ws-1",
        )
        result2 = create_work_session(
            tg=tg, ws_uuid="ws-second", task_short_hash="abcd1234",
            agent_id="agent-1", session_id="sess-2", workspace_id="ws-1",
        )

        assert result1["action"] == "start_work"
        assert result2["action"] == "start_work"
        assert result1["ws_uuid"] == "ws-first"
        assert result2["ws_uuid"] == "ws-second"
        assert result1["ws_uuid"] != result2["ws_uuid"]


class TestStaleSessionLazyCleanup:
    """US-080: Stale Session Lazy Cleanup in start_work."""

    def test_create_work_session_lazy_cleans_stale_session(self):
        """US-080: Stale Session Lazy Cleanup in start_work — a 25h-old
        active session is filtered out by the lock check (past 24h threshold)
        and the claim query's FOREACH clause marks it as 'abandoned'."""
        captured_queries = []

        responses = [
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q, [
                {"assigned_agent_id": None}
            ]),
            (lambda q, p: "assigned_agent_id" in q, [
                {"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": None}
            ]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "ws.uuid" in q, [
                {"ws_uuid": "ws-new", "session_started_at": "2024-01-02T00:00:00Z",
                 "task_status": "in_progress", "task_started_at": "2024-01-01T00:00:00Z"}
            ]),
        ]
        inner = _script_factory(responses)

        def capturing_script(idx, query, params):
            captured_queries.append({"query": query, "params": params or {}})
            return inner(idx, query, params)

        tg = _make_tg(capturing_script)

        result = create_work_session(
            tg=tg, ws_uuid="ws-new", task_short_hash="abcd1234",
            agent_id="agent-1", session_id="sess-new", workspace_id="ws-1",
        )

        assert result["action"] == "start_work"
        assert result["ws_uuid"] == "ws-new"
        assert result["task_status"] == "in_progress"

        # Verify the claim query contains the FOREACH stale-cleanup clause
        claim_queries = [q for q in captured_queries if "FOREACH" in q["query"]]
        assert len(claim_queries) >= 1
        claim_q = claim_queries[0]
        assert "abandoned" in claim_q["query"]
        assert claim_q["params"]["stale_hours"] == STALE_SESSION_HOURS


class TestGetActiveSessions:
    def test_returns_only_active_sessions(self):
        """get_active_sessions returns active WorkSession rows for the agent."""
        records = [
            {"uuid": "ws-1", "started_at": "2024-01-01T00:00:00Z", "session_id": "sess-1"},
            {"uuid": "ws-2", "started_at": "2024-01-02T00:00:00Z", "session_id": "sess-2"},
        ]
        tg = _make_tg(_script_factory([(lambda q, p: True, records)]))

        result = get_active_sessions(
            tg=tg,
            task_short_hash="abcd1234",
            agent_id="agent-1",
            workspace_id="ws-1",
        )
        assert len(result) == 2
        assert result[0]["uuid"] == "ws-1"
        assert result[1]["uuid"] == "ws-2"

    def test_returns_empty_on_no_workspace_match(self):
        """No rows → empty list."""
        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))

        result = get_active_sessions(
            tg=tg,
            task_short_hash="abcd1234",
            agent_id="agent-1",
            workspace_id="ws-1",
        )
        assert result == []

    def test_returns_empty_for_empty_short_hash(self):
        tg = MagicMock()
        result = get_active_sessions(
            tg=tg,
            task_short_hash="",
            agent_id="agent-1",
        )
        assert result == []


class TestCleanupStaleSessions:
    def test_marks_sessions_older_than_24h_as_abandoned(self):
        """Cleanup returns count of marked sessions."""
        tg = _make_tg(_script_factory([(lambda q, p: True, [{"marked": 3}])]))

        result = cleanup_stale_sessions(
            tg=tg,
            task_short_hash="abcd1234",
            agent_id="agent-1",
            workspace_id="ws-1",
        )
        assert result == 3
        # Confirm the Cypher passed STALE_SESSION_HOURS
        call_args = tg._executor.execute_write.call_args
        params = call_args[0][1]
        assert params["stale_hours"] == STALE_SESSION_HOURS
        assert params["stale_hours"] == 24

    def test_returns_zero_when_no_stale_sessions(self):
        """Cleanup of fresh sessions → marked=0."""
        tg = _make_tg(_script_factory([(lambda q, p: True, [{"marked": 0}])]))

        result = cleanup_stale_sessions(
            tg=tg,
            task_short_hash="abcd1234",
            agent_id="agent-1",
        )
        assert result == 0
