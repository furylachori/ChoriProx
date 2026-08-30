"""Integration tests: full Ticket → Work mode lifecycle.

These tests exercise the most complex cross-module flow in the system:
ticket mode → task creation → work mode → work session → bug report →
review → validate → resolve → task completion → auto-resolve.

Mocking strategy: DB-level only (Neo4j/SQLite). Action logic, dispatch,
and mode management run for real. A StatefulScript returns query-appropriate
responses and mutates state so each step sees the effects of prior steps.
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
from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
from proxy.tool_pipeline.actions.start_work import StartWorkAction
from proxy.tool_pipeline.actions.report_bug import ReportBugAction
from proxy.tool_pipeline.actions.review_bug import ReviewBugAction
from proxy.tool_pipeline.actions.validate_bug import ValidateBugAction
from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction
from proxy.tool_pipeline.actions.stop_work import StopWorkAction
from proxy.tool_pipeline.actions.update_task import UpdateTaskAction


# ---------------------------------------------------------------------------
# Stateful script — simulates Neo4j query responses with mutable state
# ---------------------------------------------------------------------------


class StatefulScript:
    """Simulate Neo4j query responses for the lifecycle flow.

    Returns canned responses keyed by query keywords, and mutates internal
    state so each step sees the effects of prior steps.

    Query matching order is most-specific-first to avoid false positives
    from overlapping patterns.
    """

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.work_sessions: dict[str, dict] = {}
        self.bugs: dict[str, dict] = {}
        self.reviews: dict[str, dict] = {}
        self._call_count = 0
        self.resolution_tasks: dict[str, str] = {}  # task_hash → bug_uuid

    def __call__(self, idx: int, query: str, params: dict) -> list[dict]:
        self._call_count += 1
        q = query

        # === Update task status: check existence ===
        if "collect(t.status)[0] AS current_status" in q:
            return self._handle_update_task_check(params)

        # === Update task status: set new status ===
        if "SET t.status = $new_status" in q and "RETURN count(t) AS updated" in q:
            return self._handle_update_task_set(params)

        # === Validate: review aggregation (most specific: has SET + avg aggregate) ===
        if "avg(r.confidence_score) AS avg_confidence" in q and "SET b.avg_confidence" in q:
            return self._handle_validate_aggregate(params)

        # === Bug resolution (SET b.status = CASE) ===
        if "SET b.status = CASE" in q and "b.validated_at IS NOT NULL" in q:
            return self._handle_resolve(params)

        # === get_bug_with_reviews: reviews query ===
        if "r.reviewer_agent_id AS reviewer_agent_id" in q and "HAS_REVIEW" in q and "RETURN r.uuid" in q:
            return self._handle_get_reviews(params)

        # === agent_has_reviewed ===
        if "count(r) > 0 AS has_reviewed" in q:
            return self._handle_agent_has_reviewed(params)

        # === get_bug_with_reviews: bug query (has reported_by_agent AS reported_by_agent) ===
        if "b.reported_by_agent AS reported_by_agent" in q and "b.avg_confidence AS avg_confidence" in q:
            return self._handle_get_bug(params)

        # === Review: CREATE (r:Review) ===
        if "CREATE (r:Review" in q:
            return self._handle_review_create(params)

        # === Bug: CREATE (b:Bug) ===
        if "CREATE (b:Bug" in q:
            return self._handle_bug_create(params)

        # === Bug: check task + active session ===
        if "ws IS NOT NULL AS has_active_session" in q:
            return self._handle_bug_report_check(params)

        # === Work session: create session + claim ===
        if "MERGE (ws:WorkSession" in q:
            return self._handle_ws_create(params)

        # === Work session: task lookup (inside tx: has short_hash + status + assigned) ===
        if "t.short_hash AS short_hash" in q and "t.status AS status" in q:
            return self._handle_ws_lookup(params)

        # === Status differentiation: submit_review failure check ===
        if "b.reported_by_agent AS reporter" in q and "b.max_reviews" in q:
            return self._handle_bug_status(params)

        # === Status differentiation: resolve failure check ===
        if "b.validated_at AS validated_at" in q and "b.status AS status" in q:
            return self._handle_resolve_status(params)

        # === Task creation (batch) — only reached via tx.run for batch_atomic ===
        if "CREATE (t:Entity:Task" in q:
            return self._handle_create_task(params)

        # === Stop work: close session ===
        if "target_ws.uuid AS ws_uuid" in q:
            return self._handle_stop_work(params)

        # === Stop work: differentiation check ===
        if "ws IS NOT NULL AS session_exists" in q:
            return self._handle_stop_check(params)

        # === Pre-check: assigned_agent_id only (from create_work_session pre-check) ===
        if "t.assigned_agent_id AS assigned_agent_id" in q:
            return self._handle_ws_precheck(params)

        return []

    # -- handlers --

    def _handle_create_task(self, params: dict) -> list[dict]:
        name = params.get("name", "")
        priority = params.get("priority", "medium")
        summary = params.get("summary", "")
        status = params.get("status", "pending")

        for existing in self.tasks.values():
            if existing["name"] == name:
                return [{"error": "duplicate_name"}]

        full_uuid = f"entity-{_uuid_mod.uuid4().hex}"
        short_hash = full_uuid[-8:]
        self.tasks[short_hash] = {
            "uuid": full_uuid,
            "short_hash": short_hash,
            "name": name,
            "priority": priority,
            "summary": summary,
            "status": status,
            "assigned_agent_id": None,
            "session_id": "",
            "workspace_id": "ws-1",
            "deleted_at": None,
        }
        return [self.tasks[short_hash]]

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
        }
        return [{
            "ws_uuid": ws_uuid,
            "session_started_at": "2026-01-01T00:00:00Z",
            "task_status": task["status"],
            "task_started_at": "2026-01-01T00:00:00Z",
        }]

    def _handle_bug_report_check(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        caller = params.get("caller_agent_id", "")
        task = self.tasks.get(short_hash)
        if not task:
            return []
        has_session = any(
            ws["task_short_hash"] == short_hash
            and ws["agent_id"] == caller
            and ws["status"] == "active"
            for ws in self.work_sessions.values()
        )
        return [{"assigned_to": task.get("assigned_agent_id"), "has_active_session": has_session}]

    def _handle_bug_create(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        bug_hash = params.get("bug_short_hash", "")
        self.bugs[bug_uuid] = {
            "uuid": bug_uuid,
            "short_hash": bug_hash,
            "title": params.get("title", ""),
            "description": params.get("description", ""),
            "severity": params.get("severity", "medium"),
            "status": "open",
            "max_reviews": params.get("max_reviews", 2),
            "review_count": 0,
            "reported_by_agent": params.get("caller_agent_id", ""),
            "validated_at": None,
            "workspace_id": params.get("workspace_id", "ws-1"),
            "avg_confidence": None,
            "valid_reviewers": None,
            "total_reviewers": None,
            "dismissal_reason": None,
        }
        return [{"uuid": bug_uuid, "short_hash": bug_hash}]

    def _handle_review_create(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        caller = params.get("caller_agent_id", "")
        bug = self.bugs.get(bug_uuid)
        if not bug or bug["status"] != "open":
            return []
        if bug["reported_by_agent"] == caller:
            return []
        if bug["review_count"] >= bug["max_reviews"]:
            return []
        for r in self.reviews.values():
            if r["bug_uuid"] == bug_uuid and r["reviewer_agent_id"] == caller:
                return []

        review_uuid = params.get("review_uuid", "")
        self.reviews[review_uuid] = {
            "uuid": review_uuid,
            "bug_uuid": bug_uuid,
            "reviewer_agent_id": caller,
            "confidence_score": params.get("confidence", 0.9),
            "verdict": params.get("verdict", "valid"),
            "proposal": params.get("proposal", ""),
        }
        bug["review_count"] += 1
        return [{"review_uuid": review_uuid, "review_count": bug["review_count"]}]

    def _handle_get_bug(self, params: dict) -> list[dict]:
        """Handle get_bug_with_reviews bug query."""
        bug_uuid = params.get("bug_uuid", "")
        bug = self.bugs.get(bug_uuid)
        if not bug:
            return []
        return [{
            "uuid": bug["uuid"],
            "short_hash": bug["short_hash"],
            "title": bug["title"],
            "description": bug["description"],
            "severity": bug["severity"],
            "status": bug["status"],
            "max_reviews": bug["max_reviews"],
            "review_count": bug["review_count"],
            "reported_by_agent": bug["reported_by_agent"],
            "avg_confidence": bug["avg_confidence"],
            "valid_reviewers": bug["valid_reviewers"],
            "total_reviewers": bug["total_reviewers"],
            "created_at": "2026-01-01T00:00:00Z",
            "validated_at": bug["validated_at"],
            "approved_at": None,
            "dismissed_at": None,
            "dismissal_reason": bug["dismissal_reason"],
        }]

    def _handle_get_reviews(self, params: dict) -> list[dict]:
        """Handle get_bug_with_reviews reviews query."""
        bug_uuid = params.get("bug_uuid", "")
        bug_reviews = [
            {
                "uuid": r["uuid"],
                "reviewer_agent_id": r["reviewer_agent_id"],
                "confidence_score": r["confidence_score"],
                "verdict": r["verdict"],
                "proposal": r["proposal"],
                "submitted_at": "2026-01-01T00:00:30Z",
            }
            for r in self.reviews.values()
            if r["bug_uuid"] == bug_uuid
        ]
        return bug_reviews

    def _handle_agent_has_reviewed(self, params: dict) -> list[dict]:
        """Handle agent_has_reviewed query."""
        bug_uuid = params.get("bug_uuid", "")
        caller = params.get("caller_agent_id", "")
        has = any(
            r["bug_uuid"] == bug_uuid and r["reviewer_agent_id"] == caller
            for r in self.reviews.values()
        )
        return [{"has_reviewed": has}]

    def _handle_validate_aggregate(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        caller = params.get("caller_agent_id", "")
        bug = self.bugs.get(bug_uuid)
        if not bug or bug["status"] != "open":
            return []
        if bug["reported_by_agent"] == caller:
            return []
        if bug["review_count"] < bug["max_reviews"]:
            return []
        if bug["validated_at"] is not None:
            return []

        bug_reviews = [r for r in self.reviews.values() if r["bug_uuid"] == bug_uuid]
        if any(r["reviewer_agent_id"] == caller for r in bug_reviews):
            return []

        avg_conf = sum(r["confidence_score"] for r in bug_reviews) / len(bug_reviews) if bug_reviews else 0
        valid_count = sum(1 for r in bug_reviews if r["verdict"] == "valid")

        bug["validated_at"] = "2026-01-01T00:01:00Z"
        bug["avg_confidence"] = avg_conf
        bug["valid_reviewers"] = valid_count
        bug["total_reviewers"] = len(bug_reviews)

        return [{
            "uuid": bug_uuid,
            "status": bug["status"],
            "avg_confidence": avg_conf,
            "valid_count": valid_count,
            "total": len(bug_reviews),
        }]

    def _handle_bug_status(self, params: dict) -> list[dict]:
        """Handle submit_review failure differentiation query."""
        bug_uuid = params.get("bug_uuid", "")
        bug = self.bugs.get(bug_uuid)
        if not bug:
            return []
        return [{
            "status": bug["status"],
            "review_count": bug["review_count"],
            "max_reviews": bug["max_reviews"],
            "reporter": bug["reported_by_agent"],
        }]

    def _handle_resolve(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        resolution = params.get("resolution", "")
        bug = self.bugs.get(bug_uuid)
        if not bug or bug["status"] != "open" or bug["validated_at"] is None:
            return []

        if resolution == "approved":
            bug["status"] = "approved"
            task_uuid = params.get("task_uuid", "")
            task_hash = params.get("task_short_hash", "")
            self.tasks[task_hash] = {
                "uuid": task_uuid,
                "short_hash": task_hash,
                "name": bug["title"],
                "priority": "medium",
                "summary": bug["description"],
                "status": "pending",
                "assigned_agent_id": None,
                "session_id": "",
                "workspace_id": "ws-1",
                "deleted_at": None,
            }
            self.resolution_tasks[task_hash] = bug_uuid
            return [{"uuid": bug_uuid, "status": "approved"}]
        else:
            bug["status"] = "dismissed"
            bug["dismissal_reason"] = params.get("reason")
            return [{"uuid": bug_uuid, "status": "dismissed"}]

    def _handle_resolve_status(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        bug = self.bugs.get(bug_uuid)
        if not bug:
            return []
        return [{"status": bug["status"], "validated_at": bug["validated_at"]}]

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

    def _handle_update_task_check(self, params: dict) -> list[dict]:
        """Handle update_task_status existence check."""
        short_hash = params.get("hash", "") or params.get("short_hash", "")
        task = self.tasks.get(short_hash)
        if not task:
            return [{"cnt": 0, "current_status": None}]
        return [{"cnt": 1, "current_status": task["status"]}]

    def _handle_update_task_set(self, params: dict) -> list[dict]:
        """Handle update_task_status SET + bug transition."""
        short_hash = params.get("hash", "") or params.get("short_hash", "")
        new_status = params.get("new_status", "")
        task = self.tasks.get(short_hash)
        if not task:
            return [{"updated": 0}]

        task["status"] = new_status

        # Bug auto-resolve/dismiss (mirrors task_graph.py OPTIONAL MATCH)
        if new_status == "complete":
            for bug in self.bugs.values():
                if self.resolution_tasks.get(short_hash) == bug["uuid"]:
                    bug["status"] = "resolved"
        elif new_status == "invalid":
            for bug in self.bugs.values():
                if self.resolution_tasks.get(short_hash) == bug["uuid"]:
                    bug["status"] = "dismissed"
                    bug["dismissal_reason"] = "task_invalidated"

        return [{"updated": 1}]


# ---------------------------------------------------------------------------
# Fixture: build a mock TaskGraph wired to a StatefulScript
# ---------------------------------------------------------------------------


class _RecordIter:
    """Minimal record iterator matching Neo4j driver protocol."""

    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


@pytest.fixture
def lifecycle_mocks():
    """Set up mock TaskGraph + agent registry for lifecycle tests."""
    script = StatefulScript()
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

    # CreateTasksAction calls tg.create_tasks_batch() directly
    def _create_tasks_batch(batch, workspace_id=None):
        results = []
        for task_dict in batch:
            result = script._handle_create_task(task_dict)
            results.extend(result)
        return results

    tg.create_tasks_batch.side_effect = _create_tasks_batch

    # UpdateTaskAction calls tg.update_task_status() directly
    def _update_task_status(hash_val, new_status, workspace_id=None, caller_agent_id=None):
        records = _exec(
            "MATCH (t:Task) WHERE t.short_hash = $hash "
            "RETURN count(t) AS cnt, collect(t.status)[0] AS current_status",
            {"hash": hash_val},
        )
        if not records or records[0].get("cnt", 0) == 0:
            return {"ok": False, "status": "not_found"}
        current = records[0]["current_status"]
        if current == new_status:
            return {"ok": True, "status": current, "changed": False}
        update_records = _exec(
            "MATCH (t:Task) WHERE t.short_hash = $hash "
            "SET t.status = $new_status RETURN count(t) AS updated",
            {"hash": hash_val, "new_status": new_status},
        )
        if update_records and update_records[0].get("updated", 0) == 1:
            return {"ok": True, "status": new_status}
        return {"ok": False, "status": current}

    tg.update_task_status.side_effect = _update_task_status

    agent = create_test_agent()
    session_id = f"lifecycle-{_uuid_mod.uuid4().hex[:8]}"

    p = patch("proxy.agent.agent_registry.registry.get_or_load", return_value=agent)
    p.start()

    yield tg, agent, session_id, script

    p.stop()


# =============================================================================
# Test 1: Happy path — full lifecycle (10 steps)
# =============================================================================


class TestFullLifecycle:
    """Happy path: ticket → create tasks → work → bug → review → validate →
    stop → resolve → update → verify all status transitions."""

    def test_full_ticket_work_lifecycle(self, lifecycle_mocks):
        tg, agent, session_id, script = lifecycle_mocks

        # ---- Step 1: ^^ticket plan → activate ticket mode ----
        activate_ticket_mode(agent, subcommand="plan")
        assert agent.is_ticket_mode

        # ---- Step 2: proxy_ticket create_tasks ----
        result = CreateTasksAction.execute(
            {"tasks": [{"name": "Implement feature X", "priority": "high", "summary": "Build the widget"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result.get("status") != "error", f"create_tasks failed: {result}"
        assert result["data"]["total_created"] == 1
        task_hash = result["data"]["created"][0]["short_hash"]
        assert script.tasks[task_hash]["status"] == "pending"

        # ---- Step 3: ^^work <hash> → activate work mode ----
        activate_work_mode(agent, task_hash=task_hash)
        assert agent.is_skill_mode
        snap = agent.snapshot_mode()
        assert snap.skill_state["task_hash"] == task_hash

        # ---- Step 4: proxy_work start_work ----
        result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result.get("status") != "error", f"start_work failed: {result}"
        assert result["operation"] == "start_work"
        ws_uuid = result["data"]["ws_uuid"]
        assert script.tasks[task_hash]["status"] == "in_progress"
        assert script.tasks[task_hash]["assigned_agent_id"] == agent.agent_id
        assert script.work_sessions[ws_uuid]["status"] == "active"

        # ---- Step 5: proxy_work report_bug ----
        result = ReportBugAction.execute(
            {
                "task_short_hash": task_hash,
                "title": "Widget crashes on empty input",
                "description": "The widget throws NPE when input is empty",
                "severity": "high",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result.get("status") != "error", f"report_bug failed: {result}"
        bug_uuid = result["data"]["bug_uuid"]
        assert script.bugs[bug_uuid]["status"] == "open"

        # ---- Step 6: proxy_work review_bug × 2 (different agents) ----
        for i, reviewer in enumerate(["reviewer-alpha", "reviewer-beta"]):
            result = ReviewBugAction.execute(
                {
                    "bug_uuid": bug_uuid,
                    "confidence": 0.9,
                    "verdict": "valid",
                    "proposal": f"Confirmed by reviewer {i + 1}",
                },
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )
            assert result.get("status") != "error", f"review_bug #{i + 1} failed: {result}"
            assert result["data"]["review_count"] == i + 1

        assert script.bugs[bug_uuid]["review_count"] == 2

        # ---- Step 7: proxy_work validate_bug ----
        result = ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator-agent",
        )
        assert result.get("status") != "error", f"validate_bug failed: {result}"
        assert result["operation"] == "validate_bug"
        assert script.bugs[bug_uuid]["validated_at"] is not None

        # ---- Step 8: proxy_work stop_work ----
        result = StopWorkAction.execute(
            {"short_hash": task_hash, "ws_uuid": ws_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result.get("status") != "error", f"stop_work failed: {result}"
        assert result["operation"] == "stop_work"
        assert script.work_sessions[ws_uuid]["status"] == "completed"

        # ---- Step 9: ^^ticket bug resolve approved → creates resolution task ----
        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="human-approver",
        )
        assert result.get("status") != "error", f"resolve_bug failed: {result}"
        assert result["data"]["bug_status"] == "approved"
        assert "task_short_hash" in result["data"]
        resolution_hash = result["data"]["task_short_hash"]
        assert script.tasks[resolution_hash]["status"] == "pending"

        # ---- Step 10: proxy_ticket update_task status=complete ----
        result = UpdateTaskAction.execute(
            {"short_hash": resolution_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result["ok"] is True, f"update_task failed: {result}"
        assert result["data"]["task_status"] == "complete"

        # ---- Final assertions ----
        assert script.tasks[task_hash]["status"] == "in_progress"
        assert script.tasks[task_hash]["assigned_agent_id"] == agent.agent_id
        assert script.bugs[bug_uuid]["status"] == "resolved"
        assert script.work_sessions[ws_uuid]["status"] == "completed"
        assert script.tasks[resolution_hash]["status"] == "complete"


# =============================================================================
# Test 2: Bug auto-resolve on task complete
# =============================================================================


class TestBugAutoResolve:
    """When a resolution task is completed, its linked bug auto-resolves."""

    def test_bug_auto_resolves_when_task_completes(self, lifecycle_mocks):
        """US-086: When a resolution task is completed, its linked bug auto-resolves."""
        tg, agent, session_id, script = lifecycle_mocks

        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Fix the bug", "priority": "high"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))

        activate_work_mode(agent, task_hash=task_hash)
        StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )

        ReportBugAction.execute(
            {"task_short_hash": task_hash, "title": "Bug", "description": "Desc", "severity": "medium"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))

        for reviewer in ["rev-1", "rev-2"]:
            ReviewBugAction.execute(
                {"bug_uuid": bug_uuid, "confidence": 0.8, "verdict": "valid", "proposal": "Confirmed"},
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )

        ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )

        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        res_hash = result["data"]["task_short_hash"]

        assert script.bugs[bug_uuid]["status"] == "approved"

        UpdateTaskAction.execute(
            {"short_hash": res_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )

        assert script.bugs[bug_uuid]["status"] == "resolved"


# =============================================================================
# Test 3: Bug auto-dismiss on task invalidation
# =============================================================================


class TestBugAutoDismiss:
    """When a resolution task is invalidated, its linked bug auto-dismisses."""

    def test_bug_auto_dismisses_when_task_invalidated(self, lifecycle_mocks):
        """US-087: When a resolution task is invalidated, its linked bug auto-dismisses."""
        tg, agent, session_id, script = lifecycle_mocks

        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Another fix", "priority": "medium"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))

        activate_work_mode(agent, task_hash=task_hash)
        StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )

        ReportBugAction.execute(
            {"task_short_hash": task_hash, "title": "Another bug", "description": "Desc", "severity": "low"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))

        for reviewer in ["rev-1", "rev-2"]:
            ReviewBugAction.execute(
                {"bug_uuid": bug_uuid, "confidence": 0.7, "verdict": "valid", "proposal": "Yep"},
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )

        ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )

        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        res_hash = result["data"]["task_short_hash"]

        UpdateTaskAction.execute(
            {"short_hash": res_hash, "new_status": "invalid"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )

        assert script.bugs[bug_uuid]["status"] == "dismissed"
        assert script.bugs[bug_uuid]["dismissal_reason"] == "task_invalidated"


# =============================================================================
# Test 4: Mode transitions preserve work session
# =============================================================================


class TestModeTransitions:
    """Switching modes preserves active work sessions and task assignment."""

    def test_mode_switch_preserves_work_session(self, lifecycle_mocks):
        tg, agent, session_id, script = lifecycle_mocks

        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Persistent task", "priority": "high"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))

        activate_work_mode(agent, task_hash=task_hash)
        result = StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ws_uuid = result["data"]["ws_uuid"]

        ReportBugAction.execute(
            {
                "task_short_hash": task_hash,
                "title": "Bug in work mode",
                "description": "Found during work",
                "severity": "medium",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))

        # Switch back to ticket mode (^^ticket plan)
        activate_ticket_mode(agent, subcommand="plan", task_hash=task_hash)
        assert agent.is_ticket_mode

        # Work session still active
        assert script.work_sessions[ws_uuid]["status"] == "active"

        # Task still assigned
        assert script.tasks[task_hash]["assigned_agent_id"] == agent.agent_id

        # Bug still exists
        assert script.bugs[bug_uuid]["status"] == "open"

        # Stop work from ticket mode is BLOCKED after C3 (was US-093 allowed).
        result = StopWorkAction.execute(
            {"short_hash": task_hash, "ws_uuid": ws_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert result.get("error") is not None, (
            f"stop_work from ticket mode should be blocked after C3; got: {result}"
        )
        error_msg = result["error"]["message"].lower()
        assert (
            "blocked" in error_msg
            or "delegate" in error_msg
        ), f"expected blocked/delegate error, got: {result}"
        # Session must remain active — work session preserved.
        assert script.work_sessions[ws_uuid]["status"] != "completed"


# =============================================================================
# Wave 11A: Additional bug-lifecycle tests (regression coverage).
#
# These complement the existing 4 lifecycle tests. They cover:
#   - Bug state machine ordering (open -> approved -> resolved is a chain;
#     once approved, dismiss is rejected).
#   - Review-count gating (validate_bug is blocked below max_reviews).
#   - Idempotency of resolve (second resolve returns the same final state).
#   - Multi-bug isolation (auto-resolve only fires for the linked task).
#   - Dismiss path stores reason, creates no task.
# =============================================================================


class TestBugStateMachine:
    """Pin down the legal state transitions of a bug.

    Per SPEC: open -> approved | dismissed (terminal states).
    Re-resolving an already-resolved bug must be a no-op.
    """

    def _drive_to_open_bug(
        self, lifecycle_mocks, title="State-machine bug", severity="high",
    ):
        tg, agent, session_id, script = lifecycle_mocks
        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": title, "priority": "high"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))
        activate_work_mode(agent, task_hash=task_hash)
        StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ReportBugAction.execute(
            {
                "task_short_hash": task_hash,
                "title": title,
                "description": "Desc",
                "severity": severity,
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))
        for reviewer in ["rev-1", "rev-2"]:
            ReviewBugAction.execute(
                {
                    "bug_uuid": bug_uuid,
                    "confidence": 0.9,
                    "verdict": "valid",
                    "proposal": "Confirmed",
                },
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )
        ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )
        return tg, agent, session_id, script, task_hash, bug_uuid

    def test_bug_transitions_open_to_approved_to_resolved(self, lifecycle_mocks):
        """The full bug state machine: open -> approved -> resolved.

        Validates the linear progression rather than just the endpoints.
        """
        tg, agent, session_id, script, task_hash, bug_uuid = (
            self._drive_to_open_bug(lifecycle_mocks)
        )
        assert script.bugs[bug_uuid]["status"] == "open"

        # Approve the bug
        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        assert result.get("status") != "error"
        assert script.bugs[bug_uuid]["status"] == "approved"
        res_hash = result["data"]["task_short_hash"]

        # Complete the resolution task -> bug -> resolved
        UpdateTaskAction.execute(
            {"short_hash": res_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert script.bugs[bug_uuid]["status"] == "resolved"

    def test_resolve_approved_bug_again_returns_error(self, lifecycle_mocks):
        """A bug already approved (terminal in the human-approval stage)
        must not be re-resolvable. The mock returns [] for a non-open bug,
        which the production resolve_bug maps to bug_not_open."""
        tg, agent, session_id, script, task_hash, bug_uuid = (
            self._drive_to_open_bug(lifecycle_mocks)
        )

        # First resolve: approved
        ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        assert script.bugs[bug_uuid]["status"] == "approved"

        # Second resolve attempt must be rejected
        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "dismissed"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        # Production returns error for non-open bug; mock returns no rows,
        # which the action maps to a structured error.
        assert result.get("error") is not None, (
            f"Second resolve must be rejected, got: {result}"
        )
        # Bug remains in approved state — never silently flipped to dismissed
        assert script.bugs[bug_uuid]["status"] == "approved"

    def test_dismiss_path_stores_reason_no_task(self, lifecycle_mocks):
        """Dismiss transitions open -> dismissed, records the reason, and
        creates no resolution task (per SPEC: only 'approved' creates a Task)."""
        tg, agent, session_id, script, task_hash, bug_uuid = (
            self._drive_to_open_bug(lifecycle_mocks, title="Will dismiss")
        )
        tasks_before = len(script.tasks)

        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "dismissed", "reason": "false positive"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        assert result.get("status") != "error"
        assert script.bugs[bug_uuid]["status"] == "dismissed"
        # No new task was created
        assert len(script.tasks) == tasks_before, (
            f"Dismiss should not create a task. Tasks: {list(script.tasks.keys())}"
        )
        # Resolve response surfaces the new status
        assert result["data"]["bug_status"] == "dismissed"

    def test_dismiss_does_not_trigger_auto_resolve(self, lifecycle_mocks):
        """A bug that is dismissed (no resolution task) must NOT be moved
        to 'resolved' by any task completion. The auto-resolve path in
        production only targets bugs that are 'approved' and linked to
        a now-complete Task via :RESOLVED_BY."""
        tg, agent, session_id, script, task_hash, bug_uuid = (
            self._drive_to_open_bug(lifecycle_mocks, title="Dismiss safety")
        )
        ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "dismissed"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        assert script.bugs[bug_uuid]["status"] == "dismissed"

        # Mark the original task as complete — must not flip the bug to resolved
        UpdateTaskAction.execute(
            {"short_hash": task_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert script.bugs[bug_uuid]["status"] == "dismissed", (
            "Dismissing a bug must put it in a terminal state — completing "
            "an unrelated task should not flip it to resolved."
        )


class TestBugReviewGating:
    """validate_bug must be blocked until review_count >= max_reviews."""

    def test_validate_with_partial_reviews_is_rejected(self, lifecycle_mocks):
        """One of two reviews is not enough — validate must return an error."""
        tg, agent, session_id, script = lifecycle_mocks

        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Need 2 reviews"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))
        activate_work_mode(agent, task_hash=task_hash)
        StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ReportBugAction.execute(
            {
                "task_short_hash": task_hash,
                "title": "Partial review",
                "description": "Only one review",
                "severity": "medium",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))

        # Only one review (max_reviews defaults to 2)
        ReviewBugAction.execute(
            {
                "bug_uuid": bug_uuid,
                "confidence": 0.9,
                "verdict": "valid",
                "proposal": "Confirmed by one",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id="rev-1",
        )
        assert script.bugs[bug_uuid]["review_count"] == 1

        # Validate is rejected
        result = ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )
        assert result.get("error") is not None, (
            f"Validate with partial reviews must error, got: {result}"
        )
        # Bug remains unvalidated
        assert script.bugs[bug_uuid]["validated_at"] is None


class TestMultiBugIsolation:
    """Auto-resolve must target only the bug linked to the resolution task.

    Regression guard: completing Task B must not resolve Bug A even if A is
    in 'approved' state waiting for its own resolution task.
    """

    def test_completing_unrelated_task_does_not_resolve_bug(self, lifecycle_mocks):
        tg, agent, session_id, script = lifecycle_mocks

        # === Bug A: open -> approved -> has resolution task A ===
        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Task A", "priority": "high"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_a_hash = next(iter(script.tasks))
        activate_work_mode(agent, task_hash=task_a_hash)
        StartWorkAction.execute(
            {"short_hash": task_a_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ReportBugAction.execute(
            {
                "task_short_hash": task_a_hash,
                "title": "Bug A",
                "description": "First",
                "severity": "high",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_a = next(iter(script.bugs))
        for reviewer in ["rev-1", "rev-2"]:
            ReviewBugAction.execute(
                {
                    "bug_uuid": bug_a,
                    "confidence": 0.9,
                    "verdict": "valid",
                    "proposal": "OK",
                },
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )
        ValidateBugAction.execute(
            {"bug_uuid": bug_a},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )
        result = ResolveBugAction.execute(
            {"bug_uuid": bug_a, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        res_task_a_hash = result["data"]["task_short_hash"]
        assert script.bugs[bug_a]["status"] == "approved"

        # === Now create a totally unrelated task B and complete it ===
        CreateTasksAction.execute(
            {"tasks": [{"name": "Task B", "priority": "low"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        # Pick the second created task (not task_a_hash, not res_task_a_hash)
        unrelated_hash = next(
            h for h in script.tasks
            if h not in (task_a_hash, res_task_a_hash)
        )
        UpdateTaskAction.execute(
            {"short_hash": unrelated_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )

        # Bug A must STILL be in approved state — its own resolution task
        # (res_task_a_hash) has not been completed yet.
        assert script.bugs[bug_a]["status"] == "approved", (
            f"Completing an unrelated task must not resolve Bug A. "
            f"Status: {script.bugs[bug_a]['status']}"
        )
        # Resolution task A is still pending
        assert script.tasks[res_task_a_hash]["status"] == "pending"

        # Now complete the actual resolution task — Bug A resolves
        UpdateTaskAction.execute(
            {"short_hash": res_task_a_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert script.bugs[bug_a]["status"] == "resolved"


class TestInvalidateTransitions:
    """When a resolution task is set to 'invalid', the linked bug is
    auto-dismissed with reason='task_invalidated'. Other transitions
    (in_progress, pending) must not trigger any bug state change.
    """

    def test_invalidating_resolution_task_records_dismissal_reason(
        self, lifecycle_mocks,
    ):
        tg, agent, session_id, script = lifecycle_mocks

        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Invalidate target"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))
        activate_work_mode(agent, task_hash=task_hash)
        StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ReportBugAction.execute(
            {
                "task_short_hash": task_hash,
                "title": "Will be invalidated",
                "description": "Desc",
                "severity": "low",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))
        for reviewer in ["rev-1", "rev-2"]:
            ReviewBugAction.execute(
                {
                    "bug_uuid": bug_uuid,
                    "confidence": 0.7,
                    "verdict": "valid",
                    "proposal": "Yep",
                },
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )
        ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )
        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        res_hash = result["data"]["task_short_hash"]

        UpdateTaskAction.execute(
            {"short_hash": res_hash, "new_status": "invalid"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )

        assert script.bugs[bug_uuid]["status"] == "dismissed"
        assert script.bugs[bug_uuid]["dismissal_reason"] == "task_invalidated"

    def test_reopening_resolution_task_does_not_unresolve_bug(self, lifecycle_mocks):
        """After a resolution task is complete and bug is 'resolved',
        reopening the task to 'in_progress' must not flip the bug back.
        The state transition is one-way once auto-resolve has fired."""
        tg, agent, session_id, script = lifecycle_mocks

        activate_ticket_mode(agent)
        CreateTasksAction.execute(
            {"tasks": [{"name": "Reopen target"}]},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        task_hash = next(iter(script.tasks))
        activate_work_mode(agent, task_hash=task_hash)
        StartWorkAction.execute(
            {"short_hash": task_hash},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        ReportBugAction.execute(
            {
                "task_short_hash": task_hash,
                "title": "Reopen bug",
                "description": "Desc",
                "severity": "medium",
            },
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        bug_uuid = next(iter(script.bugs))
        for reviewer in ["rev-1", "rev-2"]:
            ReviewBugAction.execute(
                {
                    "bug_uuid": bug_uuid,
                    "confidence": 0.8,
                    "verdict": "valid",
                    "proposal": "OK",
                },
                session_id, tg, workspace_id="ws-1", caller_agent_id=reviewer,
            )
        ValidateBugAction.execute(
            {"bug_uuid": bug_uuid},
            session_id, tg, workspace_id="ws-1", caller_agent_id="validator",
        )
        result = ResolveBugAction.execute(
            {"bug_uuid": bug_uuid, "resolution": "approved"},
            session_id, tg, workspace_id="ws-1", caller_agent_id="approver",
        )
        res_hash = result["data"]["task_short_hash"]

        # First: complete to trigger auto-resolve
        UpdateTaskAction.execute(
            {"short_hash": res_hash, "new_status": "complete"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        assert script.bugs[bug_uuid]["status"] == "resolved"

        # Now reopen — bug must NOT go back to 'approved'
        result2 = UpdateTaskAction.execute(
            {"short_hash": res_hash, "new_status": "in_progress"},
            session_id, tg, workspace_id="ws-1", caller_agent_id=agent.agent_id,
        )
        # The action may report ok or invalid_transition (state machine
        # guard may reject complete->in_progress). Either way, bug must
        # remain in 'resolved'.
        assert script.bugs[bug_uuid]["status"] == "resolved", (
            f"Bug must remain resolved after task reopen attempt. "
            f"Status: {script.bugs[bug_uuid]['status']}, action result: {result2}"
        )
