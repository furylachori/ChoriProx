"""Tests for the three new Phase 2 actions: start_work, stop_work, handoff_task.

These verify the action classes wire up the work_sessions module correctly
and handle edge cases (invalid input, missing agent_id, ambiguous hash).
"""
import pytest
from unittest.mock import patch, MagicMock

from memory.task_graph import AmbiguousHashError

from proxy.tool_pipeline.actions.start_work import StartWorkAction
from proxy.tool_pipeline.actions.stop_work import StopWorkAction
from proxy.tool_pipeline.actions.handoff_task import HandoffTaskAction


@pytest.fixture
def mock_tg():
    tg = MagicMock()
    tg.is_available.return_value = True
    return tg


class _RecordIter:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _FakeTx:
    def __init__(self, script):
        self._script = script
        self._idx = 0
        self.records = []

    def run(self, query, params=None):
        records = self._script(self._idx, query, params or {})
        self._idx += 1
        self.records.append({"query": query, "params": params or {}, "records": records})
        return _RecordIter(records)


def _cursor_script(responses):
    cursor = {"i": 0}

    def _script(idx, query, params):
        for j in range(cursor["i"], len(responses)):
            matcher, records = responses[j]
            cursor["i"] = j + 1
            if matcher(query, params):
                return records
        return []
    return _script


def _make_tg(script):
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


class TestStartWorkAction:
    def test_happy_path_calls_create_work_session(self, mock_tg):
        recorded_queries = []

        def capturing_script(idx, query, params):
            recorded_queries.append({"query": query, "params": dict(params or {})})
            return script(idx, query, params)

        script = _cursor_script([
            (lambda q, p: "t.assigned_agent_id AS assigned_agent_id" in q,
             [{"assigned_agent_id": None}]),
            (lambda q, p: "LIMIT 2" in q and "assigned_agent_id" in q and "ws.uuid" not in q,
             [{"short_hash": "abcd1234", "status": "pending", "assigned_agent_id": None}]),
            (lambda q, p: "active_agent" in q, []),
            (lambda q, p: "MERGE (ws:WorkSession" in q,
             [{"ws_uuid": "ws-real-1", "session_started_at": "2026-01-01T00:00:00Z",
               "task_status": "in_progress", "task_started_at": "2026-01-01T00:00:00Z"}]),
        ])

        tg = _make_tg(capturing_script)

        result = StartWorkAction.execute(
            {"short_hash": "abcd1234"},
            "sess-1",
            tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["operation"] == "start_work"
        assert result["status"] == "success"
        assert result["session_id"] == "sess-1"
        assert result["data"]["task_status"] == "in_progress"
        assert result["data"]["ws_uuid"].startswith("ws-")

        claim_query = next(q for q in recorded_queries if "MERGE (ws:WorkSession" in q["query"])
        assert claim_query["params"]["ws_uuid"].startswith("ws-")
        assert len(claim_query["params"]["ws_uuid"]) == len("ws-") + 32
        assert claim_query["params"]["agent_id"] == "agent-1"
        assert claim_query["params"]["session_id"] == "sess-1"
        assert claim_query["params"]["short_hash"] == "abcd1234"
        assert claim_query["params"]["workspace_id"] == "ws-1"
        assert claim_query["params"]["transferred_from_agent"] is None

    def test_missing_short_hash_returns_invalid_input(self, mock_tg):
        result = StartWorkAction.execute(
            {"short_hash": ""},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_input"
        assert "short_hash" in result["error"]["message"]

    def test_missing_caller_agent_id_returns_agent_not_found(self, mock_tg):
        result = StartWorkAction.execute(
            {"short_hash": "abcd1234"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id=None,
        )
        assert result["error"]["code"] == "agent_not_found"

    def test_ambiguous_hash_returns_ambiguous(self, mock_tg):
        with patch("memory.work_sessions.create_work_session",
                   side_effect=AmbiguousHashError(2)):
            result = StartWorkAction.execute(
                {"short_hash": "abcd1234"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "ambiguous_hash"

    def test_passes_transferred_from_agent(self, mock_tg):
        with patch("memory.work_sessions.create_work_session", return_value={"action": "start_work", "ws_uuid": "ws-1"}) as mock_create:
            StartWorkAction.execute(
                {"short_hash": "abcd1234", "transferred_from_agent": "agent-0"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert mock_create.call_args.kwargs["transferred_from_agent"] == "agent-0"


class TestStopWorkAction:
    def test_happy_path_calls_close_work_session(self, mock_tg):
        with patch("memory.work_sessions.close_work_session", return_value={
            "action": "stop_work",
            "ws_uuid": "ws-1",
            "task_status": "in_progress",
            "session_id": "sess-1",
        }) as mock_close:
            result = StopWorkAction.execute(
                {"short_hash": "abcd1234", "ws_uuid": "ws-1"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert result["operation"] == "stop_work"
        mock_close.assert_called_once()
        kwargs = mock_close.call_args.kwargs
        assert kwargs["agent_id"] == "agent-1"
        assert kwargs["ws_uuid"] == "ws-1"
        assert kwargs["new_status"] is None

    def test_missing_short_hash_returns_invalid_input(self, mock_tg):
        result = StopWorkAction.execute(
            {"ws_uuid": "ws-1"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_input"

    def test_missing_ws_uuid_returns_invalid_input(self, mock_tg):
        result = StopWorkAction.execute(
            {"short_hash": "abcd1234", "ws_uuid": ""},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_input"
        assert "ws_uuid" in result["error"]["message"]

    def test_missing_caller_agent_id_returns_agent_not_found(self, mock_tg):
        result = StopWorkAction.execute(
            {"short_hash": "abcd1234", "ws_uuid": "ws-1"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id=None,
        )
        assert result["error"]["code"] == "agent_not_found"

    def test_invalid_new_status_returns_invalid_input(self, mock_tg):
        result = StopWorkAction.execute(
            {"short_hash": "abcd1234", "ws_uuid": "ws-1", "new_status": "bogus"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_input"
        assert "bogus" in result["error"]["message"]

    def test_valid_new_status_passed_through(self, mock_tg):
        with patch("memory.work_sessions.close_work_session", return_value={"action": "stop_work"}) as mock_close:
            StopWorkAction.execute(
                {"short_hash": "abcd1234", "ws_uuid": "ws-1", "new_status": "complete"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert mock_close.call_args.kwargs["new_status"] == "complete"

    def test_git_commit_sha_passed_through(self, mock_tg):
        with patch("memory.work_sessions.close_work_session", return_value={"action": "stop_work"}) as mock_close:
            StopWorkAction.execute(
                {"short_hash": "abcd1234", "ws_uuid": "ws-1", "git_commit_sha": "abc123def"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert mock_close.call_args.kwargs["git_commit_sha"] == "abc123def"

    def test_ambiguous_hash_returns_ambiguous(self, mock_tg):
        with patch("memory.work_sessions.close_work_session",
                   side_effect=AmbiguousHashError(2)):
            result = StopWorkAction.execute(
                {"short_hash": "abcd1234", "ws_uuid": "ws-1"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "ambiguous_hash"


class TestHandoffTaskAction:
    def _patch_agent_registry(self, target=None):
        return patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=target,
        )

    def test_happy_path_reassigns_task(self, mock_tg):
        recorded_queries = []
        script = _cursor_script([
            (lambda q, p: True,
             [{"closed_count": 1, "task_hash": "abcd1234"}]),
        ])

        def capturing_script(idx, query, params):
            recorded_queries.append({"query": query, "params": dict(params or {})})
            return script(idx, query, params)

        tg = _make_tg(capturing_script)

        with self._patch_agent_registry(target=MagicMock()):
            result = HandoffTaskAction.execute(
                {"short_hash": "abcd1234", "to_agent_id": "agent-2", "context": "need help"},
                "sess-1",
                tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )

        assert result["operation"] == "handoff_task"
        assert result["data"]["short_hash"] == "abcd1234"
        assert result["data"]["to_agent_id"] == "agent-2"
        assert result["data"]["closed_sessions"] == 1
        assert result["session_id"] == "sess-1"

        assert len(recorded_queries) == 1
        handoff_q = recorded_queries[0]
        assert "t.assigned_agent_id = $to_agent_id" in handoff_q["query"]
        assert handoff_q["params"]["short_hash"] == "abcd1234"
        assert handoff_q["params"]["caller_agent_id"] == "agent-1"
        assert handoff_q["params"]["to_agent_id"] == "agent-2"
        assert handoff_q["params"]["context"] == "need help"
        assert handoff_q["params"]["workspace_id"] == "ws-1"

    def test_missing_short_hash_returns_invalid_input(self, mock_tg):
        result = HandoffTaskAction.execute(
            {"to_agent_id": "agent-2"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_input"
        assert "short_hash" in result["error"]["message"]

    def test_missing_to_agent_id_returns_invalid_input(self, mock_tg):
        result = HandoffTaskAction.execute(
            {"short_hash": "abcd1234"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_input"
        assert "to_agent_id" in result["error"]["message"]

    def test_missing_caller_agent_id_returns_agent_not_found(self, mock_tg):
        result = HandoffTaskAction.execute(
            {"short_hash": "abcd1234", "to_agent_id": "agent-2"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id=None,
        )
        assert result["error"]["code"] == "agent_not_found"

    def test_cannot_handoff_to_self(self, mock_tg):
        result = HandoffTaskAction.execute(
            {"short_hash": "abcd1234", "to_agent_id": "agent-1"},
            "sess-1",
            mock_tg,
            workspace_id="ws-1",
            caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "cannot_handoff_to_self"

    def test_target_agent_not_found(self, mock_tg):
        with self._patch_agent_registry(target=None):
            result = HandoffTaskAction.execute(
                {"short_hash": "abcd1234", "to_agent_id": "missing-agent"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "target_agent_not_found"
        assert "missing-agent" in result["error"]["message"]

    def test_ambiguous_hash_returns_ambiguous(self, mock_tg):
        # The new API: execute_transaction_no_retry with a lambda. The lambda
        # calls _handoff_task_tx(tx, ...). We mock execute_transaction_no_retry
        # to invoke the lambda and propagate the AmbiguousHashError.
        with self._patch_agent_registry(target=MagicMock()):
            def raise_ambiguous(fn):
                tx = MagicMock()
                with patch("proxy.tool_pipeline.actions.handoff_task._handoff_task_tx",
                           side_effect=AmbiguousHashError(2)):
                    return fn(tx)
            mock_tg._executor.execute_transaction_no_retry.side_effect = raise_ambiguous
            result = HandoffTaskAction.execute(
                {"short_hash": "abcd1234", "to_agent_id": "agent-2"},
                "sess-1",
                mock_tg,
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "ambiguous_hash"


class TestHandoffTaskTx:
    """Direct tests of the _handoff_task_tx helper to verify query shape."""

    def test_no_records_returns_task_not_found(self):
        from proxy.tool_pipeline.actions.handoff_task import _handoff_task_tx
        from unittest.mock import MagicMock
        # New API: _handoff_task_tx(tx, ...) — first query returns 0 rows,
        # diagnostic check_query returns 0 rows → task_not_found
        tx = MagicMock()
        tx.run.side_effect = [
            [],  # close query: 0 rows (no match)
            [],  # diagnostic check: 0 rows (task not found)
        ]
        # tx.run returns iterables; need to convert to lists
        tx.run.return_value = iter([])
        # Simulate list(tx.run(...))
        call_count = {"n": 0}
        def run_side_effect(q, params=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return iter([])
            return iter([])
        tx.run.side_effect = run_side_effect
        result = _handoff_task_tx(
            tx, "abcd1234", "agent-1", "agent-2", "ctx", "ws-1", "sess-1"
        )
        assert result["error"] == "task_not_found"

    def test_not_authorized_when_assigned_to_other(self):
        from proxy.tool_pipeline.actions.handoff_task import _handoff_task_tx
        from unittest.mock import MagicMock
        tx = MagicMock()
        call_count = {"n": 0}
        def run_side_effect(q, params=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # close_query: 0 rows
                return iter([])
            # diagnostic: task exists but assigned to other agent
            return iter([{"assigned_to": "agent-other"}])
        tx.run.side_effect = run_side_effect
        result = _handoff_task_tx(
            tx, "abcd1234", "agent-1", "agent-2", "ctx", "ws-1", "sess-1"
        )
        assert result["error"] == "not_authorized"
        assert result["assigned_to"] == "agent-other"

    def test_zero_closed_sessions_is_valid(self):
        from proxy.tool_pipeline.actions.handoff_task import _handoff_task_tx
        from unittest.mock import MagicMock
        tx = MagicMock()
        # close_query returns 1 row (the assignment matched, 0 sessions to close)
        tx.run.return_value = iter([
            {"closed_count": 0, "task_hash": "abcd1234"}
        ])

        result = _handoff_task_tx(
            tx, "abcd1234", "agent-1", "agent-2", None, "ws-1", "sess-1"
        )
        # _handoff_task_tx returns the raw inner dict; the action wraps it.
        # Inner success shape: {task_hash, closed_count}.
        assert result["task_hash"] == "abcd1234"
        assert result["closed_count"] == 0

    def test_n_closed_sessions_recorded(self):
        from proxy.tool_pipeline.actions.handoff_task import _handoff_task_tx
        from unittest.mock import MagicMock
        tx = MagicMock()
        # close_query returns 1 row with closed_count=3
        tx.run.return_value = iter([
            {"closed_count": 3, "task_hash": "abcd1234"}
        ])

        result = _handoff_task_tx(
            tx, "abcd1234", "agent-1", "agent-2", "ctx", "ws-1", "sess-1"
        )
        assert result["closed_count"] == 3

    def test_query_is_parameterized(self):
        from proxy.tool_pipeline.actions.handoff_task import _handoff_task_tx
        from unittest.mock import MagicMock
        tx = MagicMock()
        tx.run.return_value = iter([
            {"closed_count": 1, "task_hash": "abcd1234"}
        ])

        _handoff_task_tx(
            tx, "abcd1234", "agent-1", "agent-2", "ctx", "ws-1", "sess-1"
        )
        # Verify the Cypher was parameterized (no f-string interpolation of user data)
        call_args = tx.run.call_args
        params = call_args[0][1]
        assert params["short_hash"] == "abcd1234"
        assert params["caller_agent_id"] == "agent-1"
        assert params["to_agent_id"] == "agent-2"
        assert params["context"] == "ctx"
        assert params["workspace_id"] == "ws-1"
        query = call_args[0][0]
        assert "$short_hash" in query
        assert "$caller_agent_id" in query
        assert "$to_agent_id" in query
