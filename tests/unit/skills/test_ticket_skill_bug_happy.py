"""US-096: Happy-path tests for ticket-skill bug subcommands.

These tests pin down the behavior of:

  - ``^^ticket bug list``                  → list open bugs for the workspace
  - ``^^ticket bug show <hash>``           → render one bug + its reviews
  - ``^^ticket bug resolve <hash> approved`` → create a resolution task
  - ``^^ticket bug dismiss <hash>``        → dismiss without creating a task

The action handlers run for real; the TaskGraph layer is replaced with a
``StatefulScript`` that returns canned records keyed by Cypher keywords
and mutates an in-memory store so cross-handler state is observable
(``list_bugs`` sees bugs created via ``create_bug``, etc.).

NOTE: These tests live under ``tests/unit/skills`` because they exercise
a single skill module's parse/handle pipeline. The DB layer is mocked.
"""
from __future__ import annotations

import secrets
import uuid as _uuid_mod
from unittest.mock import MagicMock, patch

import pytest

from proxy import state as _proxy_state
from proxy.skills.skill_registry import SkillContext
from skills.ticket_skill import TicketSkill


# ---------------------------------------------------------------------------
# Stateful script — simulates tg._executor at the Cypher boundary
# ---------------------------------------------------------------------------


class BugScript:
    """In-memory simulation of the bug-related Cypher queries.

    Mirrors the patterns established in test_ticket_work_lifecycle.py
    and test_workspace_isolation.py. The bug_tracker module calls into
    ``tg._executor.execute_read`` and ``tg._executor.execute_transaction_no_retry``
    — this script answers those calls based on query keywords.
    """

    def __init__(self) -> None:
        self.bugs: dict[str, dict] = {}
        self.reviews: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        self.queries: list[dict] = []

    def execute_read(self, query: str, params: dict) -> list[dict]:
        self.queries.append({"op": "read", "query": query, "params": params})
        q = query

        # ---- list_bugs: MATCH (b:Bug) ... ORDER BY created_at DESC ----
        if "MATCH (b:Bug)" in q and "ORDER BY b.created_at DESC" in q:
            return self._handle_list_bugs(params)

        # ---- get_bug_by_short_hash: lookup by short_hash, LIMIT 2 ----
        if "MATCH (b:Bug {short_hash: $short_hash})" in q:
            return self._handle_get_bug_by_hash(params)

        # ---- get_bug_with_reviews: bug query (reported_by_agent) ----
        if "b.reported_by_agent AS reported_by_agent" in q and "b.avg_confidence AS avg_confidence" in q:
            return self._handle_get_bug(params)

        # ---- get_bug_with_reviews: reviews query (r.uuid, r.reviewer_agent_id) ----
        if "MATCH (b:Bug {uuid: $bug_uuid})" in q and "r.reviewer_agent_id AS reviewer_agent_id" in q:
            return self._handle_get_reviews(params)

        # ---- create_bug: pre-check (has_active_session) ----
        if "ws IS NOT NULL AS has_active_session" in q:
            return [{"assigned_to": "agent-active", "has_active_session": True}]

        # ---- agent_has_reviewed ----
        if "count(r) > 0 AS has_reviewed" in q:
            return [{"has_reviewed": False}]

        # ---- Fallback: empty ----
        return []

    def execute_write(self, query: str, params: dict) -> list[dict]:
        self.queries.append({"op": "write", "query": query, "params": params})
        return []

    def execute_transaction_no_retry(self, fn) -> dict:
        """Run a transaction: capture queries inside the function.

        The production code uses ``execute_transaction_no_retry`` for
        atomic operations (create_bug, submit_review, validate_bug,
        resolve_bug). We hand the closure a fake ``tx`` whose ``run``
        method dispatches to ``execute_read``.
        """

        class _FakeTx:
            def __init__(self, script: "BugScript"):
                self._script = script

            def run(self, query: str, params: dict):
                if "CREATE (b:Bug" in query:
                    return iter(self._script._handle_create_bug_tx(params))
                if "CREATE (r:Review" in query:
                    return iter(self._script._handle_create_review_tx(params))
                if "avg(r.confidence_score) AS avg_confidence" in query and "SET b.avg_confidence" in query:
                    return iter(self._script._handle_validate_tx(params))
                if "SET b.status = CASE" in query and "b.validated_at IS NOT NULL" in query:
                    return iter(self._script._handle_resolve_tx(params))
                if "MATCH (b:Bug {uuid: $bug_uuid, status: 'open'})" in query and "ws IS NOT NULL AS has_active_session" in query:
                    return iter([{
                        "assigned_to": "agent-active",
                        "has_active_session": True,
                    }])
                if "MATCH (b:Bug {uuid: $bug_uuid, status: 'open'})" in query and "b.review_count >= b.max_reviews" in query:
                    return iter([])
                if "MATCH (b:Bug {uuid: $bug_uuid})" in query and "b.review_count AS review_count" in query:
                    return iter([{
                        "status": self._script.bugs.get(params.get("bug_uuid", ""), {}).get("status", "open"),
                        "review_count": self._script.bugs.get(params.get("bug_uuid", ""), {}).get("review_count", 0),
                        "max_reviews": self._script.bugs.get(params.get("bug_uuid", ""), {}).get("max_reviews", 2),
                        "reporter": self._script.bugs.get(params.get("bug_uuid", ""), {}).get("reported_by_agent", ""),
                    }])
                return iter([])

        return fn(_FakeTx(self))

    # -- handlers --

    def _handle_list_bugs(self, params: dict) -> list[dict]:
        statuses = params.get("statuses")
        out: list[dict] = []
        for b in self.bugs.values():
            if statuses and b.get("status") not in statuses:
                continue
            out.append({
                "uuid": b["uuid"],
                "short_hash": b["short_hash"],
                "title": b["title"],
                "severity": b["severity"],
                "status": b["status"],
                "review_count": b["review_count"],
                "max_reviews": b["max_reviews"],
                "created_at": b["created_at"],
                "reported_by_agent": b["reported_by_agent"],
            })
        return out

    def _handle_get_bug_by_hash(self, params: dict) -> list[dict]:
        short_hash = params.get("short_hash", "")
        for b in self.bugs.values():
            if b["short_hash"] == short_hash:
                return [{"uuid": b["uuid"]}]
        return []

    def _handle_get_bug(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        b = self.bugs.get(bug_uuid)
        if not b:
            return []
        return [{
            "uuid": b["uuid"],
            "short_hash": b["short_hash"],
            "title": b["title"],
            "description": b["description"],
            "severity": b["severity"],
            "status": b["status"],
            "max_reviews": b["max_reviews"],
            "review_count": b["review_count"],
            "reported_by_agent": b["reported_by_agent"],
            "avg_confidence": b.get("avg_confidence"),
            "valid_reviewers": b.get("valid_reviewers"),
            "total_reviewers": b.get("total_reviewers"),
            "created_at": b["created_at"],
            "validated_at": b.get("validated_at"),
            "approved_at": b.get("approved_at"),
            "dismissed_at": b.get("dismissed_at"),
            "dismissal_reason": b.get("dismissal_reason"),
        }]

    def _handle_get_reviews(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        out: list[dict] = []
        for r in self.reviews.values():
            if r["bug_uuid"] == bug_uuid:
                out.append({
                    "uuid": r["uuid"],
                    "reviewer_agent_id": r["reviewer_agent_id"],
                    "confidence_score": r["confidence_score"],
                    "verdict": r["verdict"],
                    "proposal": r["proposal"],
                    "submitted_at": r["submitted_at"],
                })
        return out

    def _handle_create_bug_tx(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        short_hash = params.get("bug_short_hash", "")
        if not bug_uuid:
            bug_uuid = f"bug-{_uuid_mod.uuid4().hex}"
        if not short_hash:
            short_hash = bug_uuid.replace("bug-", "")[:8]
        self.bugs[bug_uuid] = {
            "uuid": bug_uuid,
            "short_hash": short_hash,
            "title": params.get("title", ""),
            "description": params.get("description", ""),
            "severity": params.get("severity", "medium"),
            "status": "open",
            "max_reviews": int(params.get("max_reviews", 2)),
            "review_count": 0,
            "reported_by_agent": params.get("caller_agent_id", ""),
            "created_at": "2026-06-10T00:00:00Z",
            "validated_at": None,
            "avg_confidence": None,
            "valid_reviewers": None,
            "total_reviewers": None,
            "approved_at": None,
            "dismissed_at": None,
            "dismissal_reason": None,
        }
        return [{"uuid": bug_uuid, "short_hash": short_hash}]

    def _handle_create_review_tx(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        b = self.bugs.get(bug_uuid)
        if not b or b["status"] != "open":
            return []
        if b["review_count"] >= b["max_reviews"]:
            return []
        caller = params.get("caller_agent_id", "")
        if b["reported_by_agent"] == caller:
            return []
        for r in self.reviews.values():
            if r["bug_uuid"] == bug_uuid and r["reviewer_agent_id"] == caller:
                return []
        review_uuid = params.get("review_uuid", f"review-{_uuid_mod.uuid4().hex}")
        self.reviews[review_uuid] = {
            "uuid": review_uuid,
            "bug_uuid": bug_uuid,
            "reviewer_agent_id": caller,
            "confidence_score": float(params.get("confidence", 0.9)),
            "verdict": params.get("verdict", "valid"),
            "proposal": params.get("proposal", ""),
            "submitted_at": "2026-06-10T00:00:30Z",
        }
        b["review_count"] += 1
        return [{"review_uuid": review_uuid, "review_count": b["review_count"]}]

    def _handle_validate_tx(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        b = self.bugs.get(bug_uuid)
        if not b or b["status"] != "open":
            return []
        bug_reviews = [r for r in self.reviews.values() if r["bug_uuid"] == bug_uuid]
        if len(bug_reviews) < b["max_reviews"]:
            return []
        b["validated_at"] = "2026-06-10T00:01:00Z"
        b["avg_confidence"] = sum(r["confidence_score"] for r in bug_reviews) / len(bug_reviews)
        b["valid_reviewers"] = sum(1 for r in bug_reviews if r["verdict"] == "valid")
        b["total_reviewers"] = len(bug_reviews)
        return [{
            "uuid": bug_uuid,
            "status": b["status"],
            "avg_confidence": b["avg_confidence"],
            "valid_count": b["valid_reviewers"],
            "total": b["total_reviewers"],
        }]

    def _handle_resolve_tx(self, params: dict) -> list[dict]:
        bug_uuid = params.get("bug_uuid", "")
        b = self.bugs.get(bug_uuid)
        if not b or b["status"] != "open" or not b.get("validated_at"):
            return []
        resolution = params.get("resolution", "")
        if resolution == "approved":
            b["status"] = "approved"
            b["approved_at"] = "2026-06-10T00:02:00Z"
            task_short_hash = params.get("task_short_hash", "")
            task_uuid = params.get("task_uuid", "")
            self.tasks[task_short_hash] = {
                "uuid": task_uuid,
                "short_hash": task_short_hash,
                "name": b["title"],
                "summary": b["description"],
                "priority": "high" if b["severity"] in ("critical", "high") else "medium",
                "status": "pending",
                "workspace_id": params.get("workspace_id", ""),
            }
        else:
            b["status"] = "dismissed"
            b["dismissed_at"] = "2026-06-10T00:02:00Z"
            b["dismissal_reason"] = params.get("reason")
        return [{"uuid": bug_uuid, "status": b["status"]}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_id() -> str:
    return f"bug-cmd-{secrets.token_hex(4)}"


@pytest.fixture(autouse=True)
def _reset_workspace(session_id):
    _proxy_state.active_workspaces.pop(session_id, None)
    yield
    _proxy_state.active_workspaces.pop(session_id, None)


@pytest.fixture
def skill() -> TicketSkill:
    return TicketSkill()


@pytest.fixture
def script() -> BugScript:
    return BugScript()


@pytest.fixture
def mock_tg(script: BugScript) -> MagicMock:
    """TaskGraph with a wired BugScript executor."""
    tg = MagicMock()
    tg.is_available.return_value = True
    tg._executor = MagicMock()
    tg._executor.execute_read.side_effect = script.execute_read
    tg._executor.execute_write.side_effect = script.execute_write
    tg._executor.execute_transaction_no_retry.side_effect = script.execute_transaction_no_retry
    return tg


@pytest.fixture
def ctx(session_id: str) -> SkillContext:
    return SkillContext(
        session_id=session_id,
        agent_id=f"agent-{session_id}",
        messages=[],
        is_subagent=False,
        active_skill_name="ticket",
        raw_user_text="^^ticket",
    )


def _seed_bug(
    script: BugScript,
    *,
    short_hash: str | None = None,
    title: str = "Sample bug",
    severity: str = "high",
    status: str = "open",
    review_count: int = 0,
    max_reviews: int = 2,
    validated: bool = False,
    reviews: list[dict] | None = None,
) -> str:
    """Insert a bug + reviews directly into the script store. Returns bug_uuid."""
    bug_uuid = f"bug-{_uuid_mod.uuid4().hex}"
    if short_hash is None:
        short_hash = secrets.token_hex(4)
    script.bugs[bug_uuid] = {
        "uuid": bug_uuid,
        "short_hash": short_hash,
        "title": title,
        "description": f"Description for {title}",
        "severity": severity,
        "status": status,
        "max_reviews": max_reviews,
        "review_count": review_count,
        "reported_by_agent": "agent-reporter",
        "created_at": "2026-06-10T00:00:00Z",
        "validated_at": "2026-06-10T00:01:00Z" if validated else None,
        "avg_confidence": 0.9 if validated else None,
        "valid_reviewers": review_count if validated else None,
        "total_reviewers": review_count if validated else None,
        "approved_at": None,
        "dismissed_at": None,
        "dismissal_reason": None,
    }
    for r in reviews or []:
        review_uuid = r.get("uuid", f"review-{_uuid_mod.uuid4().hex}")
        script.reviews[review_uuid] = {
            "uuid": review_uuid,
            "bug_uuid": bug_uuid,
            "reviewer_agent_id": r.get("reviewer_agent_id", "agent-reviewer"),
            "confidence_score": r.get("confidence_score", 0.9),
            "verdict": r.get("verdict", "valid"),
            "proposal": r.get("proposal", "Confirmed"),
            "submitted_at": r.get("submitted_at", "2026-06-10T00:00:30Z"),
        }
    return bug_uuid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTicketBugList:
    """^^ticket bug list → returns formatted list of open bugs."""

    def test_bug_list_returns_open_bugs(self, skill, mock_tg, ctx, script, session_id):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_a = _seed_bug(script, title="Login broken", severity="high")
        bug_b = _seed_bug(script, title="Slow query", severity="medium")

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_list", "")

        assert result is not None
        assert "error" not in result, result
        content = result["choices"][0]["message"]["content"]
        assert "Open Bugs:" in content
        assert "Login broken" in content
        assert "Slow query" in content
        assert script.bugs[bug_a]["short_hash"] in content
        assert script.bugs[bug_b]["short_hash"] in content
        # Review counts are surfaced
        assert "0/2" in content

    def test_bug_list_empty_returns_no_bugs_message(
        self, skill, mock_tg, ctx, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_list", "")

        assert result is not None
        content = result["choices"][0]["message"]["content"]
        assert "No open bugs" in content

    def test_bug_list_via_parse_subcommand(self, skill, mock_tg, ctx, script, session_id):
        """parse_subcommand('^^ticket bug list') routes to handle_direct('bug_list', '')."""
        action, arg = skill.parse_subcommand("^^ticket bug list")
        assert action == "bug_list"
        assert arg == ""

        _proxy_state.active_workspaces[session_id] = "luna-ui"
        _seed_bug(script, title="Parsed bug")

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, action, arg)

        assert result is not None
        assert "Parsed bug" in result["choices"][0]["message"]["content"]

    def test_bug_list_excludes_resolved_bugs(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        """The status filter must exclude non-open bugs from the listing."""
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        _seed_bug(script, title="Open one", status="open")
        _seed_bug(script, title="Resolved one", status="resolved")

        # Capture what statuses the list_bugs call requested
        requested_statuses: list[list[str] | None] = []

        original_execute_read = script.execute_read

        def _capturing_execute_read(query, params):
            if "MATCH (b:Bug)" in query and "ORDER BY b.created_at DESC" in query:
                requested_statuses.append(params.get("statuses"))
            return original_execute_read(query, params)

        mock_tg._executor.execute_read.side_effect = _capturing_execute_read

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_list", "")

        # The status filter restricts to open
        assert any(s == ["open"] for s in requested_statuses), (
            f"Expected status_filter=['open'], got {requested_statuses!r}"
        )
        # Resolved bug is filtered out
        content = result["choices"][0]["message"]["content"]
        assert "Open one" in content
        assert "Resolved one" not in content


class TestTicketBugShow:
    """^^ticket bug show <hash> → returns full bug view with reviews."""

    def test_bug_show_returns_bug_with_reviews(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(
            script,
            title="Crash on submit",
            severity="critical",
            review_count=2,
            reviews=[
                {"reviewer_agent_id": "agent-1", "verdict": "valid", "confidence_score": 0.95,
                 "proposal": "Reproduced locally"},
                {"reviewer_agent_id": "agent-2", "verdict": "valid", "confidence_score": 0.85,
                 "proposal": "Confirmed in CI"},
            ],
        )
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_show", bug_hash)

        assert result is not None
        assert "error" not in result, result
        content = result["choices"][0]["message"]["content"]
        assert "Crash on submit" in content
        assert bug_hash in content
        assert "agent-1" in content
        assert "agent-2" in content
        assert "Reproduced locally" in content
        assert "Confirmed in CI" in content

    def test_bug_show_no_reviews(self, skill, mock_tg, ctx, script, session_id):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(script, title="Fresh bug", review_count=0)
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_show", bug_hash)

        assert result is not None
        assert "error" not in result
        content = result["choices"][0]["message"]["content"]
        assert "Fresh bug" in content
        assert "Reviews :" in content
        # The (no reviews) section is omitted when there are none
        # (production code only renders the section if reviews exist)
        assert "Reviews:" not in content or "0/2" in content

    def test_bug_show_not_found_returns_404(self, skill, mock_tg, ctx, session_id):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_show", "deadbeef")

        assert result is not None
        assert result.get("_status_code") == 404
        content = result["choices"][0]["message"]["content"]
        assert "No bug found" in content

    def test_bug_show_via_parse_subcommand(self, skill, mock_tg, ctx, script, session_id):
        """parse_subcommand('^^ticket bug show <hash>') routes correctly."""
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(script, title="Parse-routed bug")
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        action, arg = skill.parse_subcommand(f"^^ticket bug show {bug_hash}")
        assert action == "bug_show"
        assert arg == bug_hash

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, action, arg)

        assert result is not None
        assert "Parse-routed bug" in result["choices"][0]["message"]["content"]


class TestTicketBugResolve:
    """^^ticket bug resolve <hash> approved → creates a resolution task."""

    def test_bug_resolve_approved_creates_task(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(
            script,
            title="Resolve me",
            severity="high",
            review_count=2,
            validated=True,
            reviews=[
                {"reviewer_agent_id": "r1", "verdict": "valid", "confidence_score": 0.9},
                {"reviewer_agent_id": "r2", "verdict": "valid", "confidence_score": 0.95},
            ],
        )
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_resolve", bug_hash)

        assert result is not None
        assert "error" not in result, result
        content = result["choices"][0]["message"]["content"]
        assert "approved" in content
        assert "Task created" in content
        assert bug_hash in content
        # The bug moved to approved
        assert script.bugs[bug_uuid]["status"] == "approved"
        # A resolution task was created
        assert len(script.tasks) == 1
        created_task = next(iter(script.tasks.values()))
        assert created_task["name"] == "Resolve me"
        assert created_task["status"] == "pending"

    def test_bug_resolve_not_validated_returns_400(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(script, title="Unvalidated", validated=False, review_count=0)
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_resolve", bug_hash)

        assert result is not None
        assert result.get("_status_code") == 400
        content = result["choices"][0]["message"]["content"]
        assert "validated" in content.lower()
        # No task created
        assert script.tasks == {}

    def test_bug_resolve_not_found_returns_404(self, skill, mock_tg, ctx, session_id):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_resolve", "deadbeef")

        assert result is not None
        assert result.get("_status_code") == 404

    def test_bug_resolve_via_parse_subcommand_with_reason(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        """parse_subcommand('^^ticket bug resolve <hash>') routes to handle_direct('bug_resolve', '<hash>')."""
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(
            script, title="Routed resolve", severity="high",
            review_count=2, validated=True,
        )
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        action, arg = skill.parse_subcommand(f"^^ticket bug resolve {bug_hash}")
        assert action == "bug_resolve"
        assert arg == bug_hash

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, action, arg)

        assert result is not None
        assert "approved" in result["choices"][0]["message"]["content"]


class TestTicketBugDismiss:
    """^^ticket bug dismiss <hash> → dismisses without creating a task."""

    def test_bug_dismiss_no_task_created(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(
            script,
            title="Dismiss me",
            severity="low",
            review_count=2,
            validated=True,
            reviews=[
                {"reviewer_agent_id": "r1", "verdict": "valid", "confidence_score": 0.9},
                {"reviewer_agent_id": "r2", "verdict": "valid", "confidence_score": 0.85},
            ],
        )
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_dismiss", bug_hash)

        assert result is not None
        assert "error" not in result, result
        content = result["choices"][0]["message"]["content"]
        assert "dismissed" in content
        # NO task created on dismiss
        assert script.tasks == {}
        # Bug status updated
        assert script.bugs[bug_uuid]["status"] == "dismissed"

    def test_bug_dismiss_stores_reason(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(
            script, title="With reason", severity="medium",
            review_count=2, validated=True,
        )
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(
                ctx, "bug_dismiss", f"{bug_hash}|not a real issue",
            )

        assert result is not None
        assert "error" not in result, result
        # The dismissal_reason is captured for later ^^ticket bug show
        assert script.bugs[bug_uuid]["dismissal_reason"] == "not a real issue"

    def test_bug_dismiss_not_validated_returns_400(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(script, title="Unvalidated dismiss", validated=False)
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, "bug_dismiss", bug_hash)

        assert result is not None
        assert result.get("_status_code") == 400
        # No task created
        assert script.tasks == {}

    def test_bug_dismiss_via_parse_subcommand(
        self, skill, mock_tg, ctx, script, session_id,
    ):
        """parse_subcommand('^^ticket bug dismiss <hash>') routes correctly."""
        _proxy_state.active_workspaces[session_id] = "luna-ui"
        bug_uuid = _seed_bug(
            script, title="Routed dismiss", severity="low",
            review_count=2, validated=True,
        )
        bug_hash = script.bugs[bug_uuid]["short_hash"]

        action, arg = skill.parse_subcommand(f"^^ticket bug dismiss {bug_hash}")
        assert action == "bug_dismiss"
        assert arg == bug_hash

        with patch(
            "skills.ticket_skill._get_task_graph_singleton", return_value=mock_tg,
        ):
            result = skill.handle_direct(ctx, action, arg)

        assert result is not None
        assert "dismissed" in result["choices"][0]["message"]["content"]
        # No task
        assert script.tasks == {}
