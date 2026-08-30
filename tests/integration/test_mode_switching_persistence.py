"""Integration tests: mode switching preserves work sessions (US-093).

The proxy supports two skill modes — ``ticket`` and ``work`` — and an
operator (or sub-agent) may legitimately need to switch between them
mid-session. The contract under test:

  1. Starting work mode opens a work session (via ``StartWorkAction``).
  2. Switching to ticket mode does NOT close the work session — it lives
     in Neo4j, independent of the in-process ``active_mode`` pointer on
     the ``Agent``.
  3. ``get_active_sessions`` continues to return the live session after
     the mode switch.
  4. ``StopWorkAction`` succeeds when invoked from ticket mode — the
     work session is the source of truth, not the in-memory mode flag.
  5. The task remains assigned to the agent across the mode switch
     (assignment is on the ``Task`` node, not the mode).

These tests use the same ``StatefulScript`` + ``_RecordIter`` pattern as
``test_ticket_work_lifecycle.py`` so the action handlers run for real
against an in-memory database simulation. This keeps the test fast and
deterministic without touching Neo4j.

Note: action results are now wrapped in the canonical action envelope
(slice-1, spec §11). The success path places the payload under
``data`` and reports ``status="success"`` + ``error=None``. Older
assertions that read flat fields like ``result["ws_uuid"]`` now
dereference ``result["data"]["ws_uuid"]``; the success assertion
checks ``status == "success"`` instead of ``"error" not in result``
because ``error`` is always present (set to ``None`` on success).

References:
  - src/memory/work_sessions.py   create_work_session, close_work_session, get_active_sessions
  - src/proxy/tool_pipeline/actions/start_work.py   StartWorkAction
  - src/proxy/tool_pipeline/actions/stop_work.py    StopWorkAction
  - tests/mode_harness.py         activate_ticket_mode, activate_work_mode
  - tests/integration/test_ticket_work_lifecycle.py  StatefulScript pattern
"""
from __future__ import annotations

import uuid as _uuid_mod
from unittest.mock import MagicMock, patch

import pytest

from tests.mode_harness import (
    activate_ticket_mode,
    activate_work_mode,
    create_test_agent,
)
from proxy.tool_pipeline.actions.start_work import StartWorkAction
from proxy.tool_pipeline.actions.stop_work import StopWorkAction
from memory.work_sessions import get_active_sessions


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Stateful script — focused on work-session state, not the full lifecycle
# ---------------------------------------------------------------------------


class _RecordIter:
    """Minimal record iterator matching the Neo4j driver protocol."""

    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class _WorkSessionScript:
    """In-memory simulation of the Cypher queries for work-session CRUD.

    The full lifecycle test (test_ticket_work_lifecycle.py) handles task
    creation, bug tracking, reviews, etc. — far more than this US-093
    spec needs. We keep this script minimal: just tasks, assignments,
    and WorkSession nodes.
    """

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.work_sessions: dict[str, dict] = {}

    def __call__(self, idx: int, query: str, params: dict) -> list[dict]:
        q = query

        if "t.short_hash AS short_hash" in q and "t.status AS status" in q:
            return self._handle_ws_lookup(params)

        if "t.assigned_agent_id AS assigned_agent_id" in q:
            return self._handle_ws_precheck(params)

        if "MERGE (ws:WorkSession" in q:
            return self._handle_ws_create(params)

        if "target_ws.uuid AS ws_uuid" in q:
            return self._handle_stop_work(params)

        if "ws IS NOT NULL AS session_exists" in q:
            return self._handle_stop_check(params)

        if "ws.status = 'active'" in q and "RETURN ws.uuid" in q:
            return self._handle_get_active(params)

        return []

    def _handle_ws_precheck(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        task = self.tasks.get(short_hash)
        if not task:
            return []
        return [{"assigned_agent_id": task.get("assigned_agent_id")}]

    def _handle_ws_lookup(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        task = self.tasks.get(short_hash)
        if not task:
            return []
        return [{
            "short_hash": task["short_hash"],
            "status": task["status"],
            "assigned_agent_id": task.get("assigned_agent_id"),
        }]

    def _handle_ws_create(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        agent_id = params.get("agent_id", "")
        ws_uuid = params.get("ws_uuid", "")

        task = self.tasks.get(short_hash)
        if not task:
            return []

        task["assigned_agent_id"] = agent_id
        if task["status"] == "pending":
            task["status"] = "in_progress"

        self.work_sessions[ws_uuid] = {
            "uuid": ws_uuid,
            "task_short_hash": short_hash,
            "agent_id": agent_id,
            "status": "active",
            "session_id": params.get("session_id", ""),
            "started_at": "2026-01-01T00:00:00Z",
        }
        return [{
            "ws_uuid": ws_uuid,
            "session_started_at": self.work_sessions[ws_uuid]["started_at"],
            "task_status": task["status"],
            "task_started_at": "2026-01-01T00:00:00Z",
        }]

    def _handle_stop_work(self, params: dict) -> list[dict]:
        ws_uuid = params.get("ws_uuid", "")
        short_hash = params.get("short_hash", "")
        new_status = params.get("new_status")
        ws = self.work_sessions.get(ws_uuid)
        if not ws or ws["status"] != "active":
            return []
        task = self.tasks.get(short_hash)
        if not task:
            return []

        ws["status"] = "completed"
        if new_status in ("complete", "invalid"):
            task["status"] = new_status
        return [{"ws_uuid": ws_uuid, "task_status": task["status"]}]

    def _handle_stop_check(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        ws_uuid = params.get("ws_uuid", "")
        task = self.tasks.get(short_hash)
        if not task:
            return []
        ws = self.work_sessions.get(ws_uuid)
        return [{
            "assigned_to": task.get("assigned_agent_id"),
            "session_exists": ws is not None,
        }]

    def _handle_get_active(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        agent_id = params.get("agent_id", "")
        results = []
        for ws in self.work_sessions.values():
            if (
                ws["task_short_hash"] == short_hash
                and ws["agent_id"] == agent_id
                and ws["status"] == "active"
            ):
                results.append({
                    "uuid": ws["uuid"],
                    "started_at": ws.get("started_at", ""),
                    "session_id": ws.get("session_id", ""),
                })
        return results

    def seed_task(self, name: str, status: str = "pending", assigned_to=None) -> str:
        """Insert a task and return its short_hash."""
        full_uuid = f"entity-{_uuid_mod.uuid4().hex}"
        short_hash = full_uuid[-8:]
        self.tasks[short_hash] = {
            "uuid": full_uuid,
            "short_hash": short_hash,
            "name": name,
            "priority": "medium",
            "summary": "",
            "status": status,
            "assigned_agent_id": assigned_to,
            "session_id": "",
            "workspace_id": "ws-1",
            "deleted_at": None,
        }
        return short_hash


# ---------------------------------------------------------------------------
# Fixture: build a mock TaskGraph wired to the work-session script
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_mocks():
    """Wire proxy to a mock TaskGraph for work-session operations."""
    script = _WorkSessionScript()
    tg = MagicMock()

    def _execute_tx(fn):
        tx = MagicMock()
        tx.run.side_effect = lambda q, p=None: _RecordIter(script(0, q, p or {}))
        return fn(tx)

    call_idx = {"n": 0}

    def _exec(q, p=None, timeout=30):
        records = script(call_idx["n"], q, p or {})
        call_idx["n"] += 1
        return records

    tg._executor.execute_transaction_no_retry.side_effect = _execute_tx
    tg._executor.execute_write.side_effect = _exec
    tg._executor.execute_read.side_effect = _exec

    agent = create_test_agent()
    session_id = f"ws-test-{_uuid_mod.uuid4().hex[:8]}"

    p = patch("proxy.agent.agent_registry.registry.get_or_load", return_value=agent)
    p.start()

    yield tg, agent, session_id, script

    p.stop()


# ===========================================================================
# US-093: Mode switching preserves work sessions
# ===========================================================================


class TestModeSwitchingPreservesWorkSession:
    """Work mode → ticket mode → work mode lifecycle, with assertions that
    the work session survives the mode switches and remains closeable.
    """

    def test_work_session_survives_work_to_ticket_switch(self, ws_mocks):
        """A work session started under work mode is still active after
        switching to ticket mode (^^ticket plan).
        """
        tg, agent, session_id, script = ws_mocks

        # 1. Activate work mode and start a work session.
        activate_work_mode(agent, task_hash="placeholder", ttl=300)
        assert agent.is_skill_mode

        task_hash = script.seed_task("Implement feature X")
        result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        # Canonical envelope: success path has status="success" + error=None.
        # ``error`` is always present (set to None on success), so check
        # ``status`` instead of ``"error" not in result``.
        assert result.get("status") == "success", (
            f"start_work failed: status={result.get('status')!r} "
            f"error={result.get('error')!r}"
        )
        ws_uuid = result["data"]["ws_uuid"]
        assert script.work_sessions[ws_uuid]["status"] == "active"

        # 2. Switch to ticket mode (^^ticket plan).
        activate_ticket_mode(agent, subcommand="plan")
        assert agent.is_ticket_mode
        # Work mode name is gone — but the work session is alive in the
        # task graph, not in the in-process mode pointer.
        assert agent.get_mode_name() == "ticket"
        assert agent.snapshot_mode().name != "work"

        # 3. The work session is still queryable via get_active_sessions.
        active = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id=agent.agent_id,
            workspace_id="ws-1",
        )
        assert len(active) == 1, (
            f"Work session lost during mode switch. Found: {active!r}"
        )
        assert active[0]["uuid"] == ws_uuid

        # 4. The task is still assigned to the agent.
        assert script.tasks[task_hash]["assigned_agent_id"] == agent.agent_id

    def test_stop_work_blocked_from_ticket_mode(self, ws_mocks):
        """StopWorkAction invoked from ticket mode must be blocked (C3).

        Previously stop_work was callable from ticket mode (US-093 stated
        "ticket has full proxy_work"). C3 reverses this: stop_work requires
        WORK mode — ticket mode delegates stop_work to a work sub-agent
        instead of allowing the MAIN agent to close the session directly.
        """
        tg, agent, session_id, script = ws_mocks

        # Start work in work mode.
        activate_work_mode(agent, task_hash="seed", ttl=300)
        task_hash = script.seed_task("Closeable task")
        start_result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ws_uuid = start_result["data"]["ws_uuid"]

        # Switch to ticket mode.
        activate_ticket_mode(agent, subcommand="plan")
        assert agent.is_ticket_mode

        # Stop work from ticket mode is now BLOCKED (C3).
        result = StopWorkAction.execute(
            {"short_hash": task_hash, "ws_uuid": ws_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result.get("status") == "error", (
            f"stop_work from ticket mode should be blocked after C3; got: {result}"
        )
        error_blob = str(result.get("error", "")) + str(result.get("message", ""))
        assert (
            "blocked" in error_blob.lower()
            or "delegate" in error_blob.lower()
        ), f"expected blocked/delegate error, got: {result}"
        # Session must NOT have been closed — work session preserved.
        assert script.work_sessions[ws_uuid]["status"] != "completed"

    def test_ticket_to_work_to_ticket_round_trip(self, ws_mocks):
        """Full round-trip: ticket → work → ticket. Work session is
        observable from BOTH ticket-mode observations.
        """
        tg, agent, session_id, script = ws_mocks

        activate_ticket_mode(agent, subcommand="plan")
        task_hash = script.seed_task("Round-trip task")

        # Move into work mode.
        activate_work_mode(agent, task_hash=task_hash, ttl=300)
        start_result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ws_uuid = start_result["data"]["ws_uuid"]

        # Back to ticket mode.
        activate_ticket_mode(agent, subcommand="plan")
        active = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id=agent.agent_id,
            workspace_id="ws-1",
        )
        assert len(active) == 1
        assert active[0]["uuid"] == ws_uuid

        # Back to work mode (different sub-agent lifecycle).
        activate_work_mode(agent, task_hash=task_hash, ttl=300)
        active = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id=agent.agent_id,
            workspace_id="ws-1",
        )
        assert len(active) == 1
        assert active[0]["uuid"] == ws_uuid

        # Final stop from work mode.
        stop_result = StopWorkAction.execute(
            {"short_hash": task_hash, "ws_uuid": ws_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert stop_result.get("status") == "success", (
            f"stop_work failed: {stop_result}"
        )
        assert script.work_sessions[ws_uuid]["status"] == "completed"

    def test_work_session_independent_of_mode_ttl(self, ws_mocks):
        """A short-TTL work mode that expires mid-session must NOT
        invalidate the underlying work session. The session is a
        graph entity; the mode is a UI hint.
        """
        tg, agent, session_id, script = ws_mocks

        activate_work_mode(agent, task_hash="short-ttl", ttl=1)
        task_hash = script.seed_task("Task with short-TTL mode")
        start_result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ws_uuid = start_result["data"]["ws_uuid"]

        import time
        time.sleep(1.2)
        assert agent.is_mode_expired(), "Mode TTL should have elapsed"

        # Session is still active in the graph even though the mode
        # pointer has expired.
        active = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id=agent.agent_id,
            workspace_id="ws-1",
        )
        assert len(active) == 1
        assert active[0]["uuid"] == ws_uuid

    def test_get_active_sessions_returns_empty_for_unrelated_agent(self, ws_mocks):
        """Sanity: a different agent_id on get_active_sessions returns
        nothing — the mode switch must not leak sessions across agents.
        """
        tg, agent, session_id, script = ws_mocks

        activate_work_mode(agent, task_hash="isolation", ttl=300)
        task_hash = script.seed_task("Isolation test")
        start_result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ws_uuid = start_result["data"]["ws_uuid"]

        activate_ticket_mode(agent, subcommand="plan")

        active_other = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id="some-other-agent",
            workspace_id="ws-1",
        )
        assert active_other == []

        active_self = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id=agent.agent_id,
            workspace_id="ws-1",
        )
        assert len(active_self) == 1
        assert active_self[0]["uuid"] == ws_uuid

    def test_session_kept_after_mode_clear(self, ws_mocks):
        """Explicitly clearing the active mode does NOT close the
        session. Only StopWorkAction does.
        """
        tg, agent, session_id, script = ws_mocks

        activate_work_mode(agent, task_hash="explicit-clear", ttl=300)
        task_hash = script.seed_task("Explicit clear test")
        start_result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ws_uuid = start_result["data"]["ws_uuid"]

        # Clear the active mode (simulates a cleanup pass).
        cleared = agent.clear_active_mode()
        assert cleared is True
        assert agent.snapshot_mode() is None

        # Session is still alive.
        active = get_active_sessions(
            tg=tg,
            task_short_hash=task_hash,
            agent_id=agent.agent_id,
            workspace_id="ws-1",
        )
        assert len(active) == 1
        assert active[0]["uuid"] == ws_uuid

        # Stop work now.
        stop_result = StopWorkAction.execute(
            {"short_hash": task_hash, "ws_uuid": ws_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert stop_result.get("status") == "success", (
            f"stop_work failed: {stop_result}"
        )
        assert script.work_sessions[ws_uuid]["status"] == "completed"
