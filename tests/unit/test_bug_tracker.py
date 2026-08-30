"""Tests for memory.bug_tracker — create/review/validate/resolve + read helpers.

These tests use fake executors that record Cypher queries and return
scripted responses. They mirror the pattern in tests/unit/test_work_sessions.py.
"""
from unittest.mock import MagicMock, patch

import pytest

from memory.bug_tracker import (
    agent_has_reviewed,
    create_bug,
    get_bug_by_short_hash,
    get_bug_with_reviews,
    list_bugs,
    resolve_bug,
    submit_review,
    validate_bug,
)
from memory.query_executor import InfrastructureError
from tests.unit.test_cypher_validator import (
    assert_queries_valid,
    build_recording_tg,
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


def _has_status_check(query):
    return "b.status" in query and "b.review_count" in query and "b.reported_by_agent" in query


def _has_resolve_status_check(query):
    return "b.status" in query and "b.validated_at" in query


# -----------------------------------------------------------------------------
# create_bug
# -----------------------------------------------------------------------------


class TestCreateBugHappyPath:
    def test_creates_bug_for_assigned_active_task(self):
        """US-081: Agent with active work session on an assigned task: Bug is created."""
        responses = [
            # First: lookup task + active session
            (lambda q, p: "assigned_to" in q and "has_active_session" in q, [
                {"assigned_to": "agent-1", "has_active_session": True}
            ]),
            # Second: create bug + FOUND_DURING
            (lambda q, p: "CREATE (b:Bug" in q, [
                {"uuid": "bug-abc", "short_hash": "aabbccdd"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_bug(
            tg=tg,
            bug_uuid="bug-abc",
            bug_short_hash="aabbccdd",
            task_short_hash="abcd1234",
            title="Foo",
            description="Bar",
            severity="high",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result["uuid"] == "bug-abc"
        assert result["short_hash"] == "aabbccdd"

    def test_creates_bug_for_unassigned_task(self):
        """Task with assigned_agent_id=NULL is NOT reportable per SPEC line 644.

        SPEC line 644: report_bug requires t.assigned_agent_id = $caller_agent_id.
        Unassigned tasks (NULL) must be rejected with not_authorized.
        """
        responses = [
            (lambda q, p: "assigned_to" in q and "has_active_session" in q, [
                {"assigned_to": None, "has_active_session": True}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="11223344",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="low",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result["error"] == "not_authorized"
        assert result["assigned_to"] is None

    def test_max_reviews_param_is_passed_to_cypher(self):
        """max_reviews is forwarded into the Cypher query params."""
        responses = [
            (lambda q, p: "assigned_to" in q, [
                {"assigned_to": "agent-1", "has_active_session": True}
            ]),
            (lambda q, p: "CREATE (b:Bug" in q, [
                {"uuid": "bug-1", "short_hash": "11223344"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))
        create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="11223344",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="medium",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=4,
        )
        # The second tx.run() call should have max_reviews=4 in params
        create_call = tg._executor.execute_transaction_no_retry.call_args
        # Inspect recorded params from the FakeTx
        # Easier: re-run with capture
        captured_params = []
        orig_run = _FakeTx.run
        captured = {}

        def _capture_run(self, query, params=None):
            res = orig_run(self, query, params)
            captured.setdefault("params", []).append(params or {})
            return res
        _FakeTx.run = _capture_run
        try:
            tg2 = _make_tg(_script_factory(responses))
            create_bug(
                tg=tg2,
                bug_uuid="bug-1",
                bug_short_hash="11223344",
                task_short_hash="abcd1234",
                title="T",
                description="D",
                severity="medium",
                caller_agent_id="agent-1",
                workspace_id="ws-1",
                max_reviews=4,
            )
            create_params = captured["params"][1]  # second tx.run = CREATE
        finally:
            _FakeTx.run = orig_run
        assert create_params["max_reviews"] == 4


class TestCreateBugErrorCases:
    def test_no_active_session(self):
        """Caller has an active session check fail → no_active_session."""
        responses = [
            (lambda q, p: "assigned_to" in q and "has_active_session" in q, [
                {"assigned_to": "agent-1", "has_active_session": False}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="11223344",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="low",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result["error"] == "no_active_session"

    def test_not_authorized(self):
        """Task assigned to different agent → not_authorized."""
        responses = [
            (lambda q, p: "assigned_to" in q and "has_active_session" in q, [
                {"assigned_to": "other-agent", "has_active_session": True}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="11223344",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="low",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result["error"] == "not_authorized"
        assert result["assigned_to"] == "other-agent"

    def test_task_not_found(self):
        """No rows for the task → task_not_found."""
        responses = [
            (lambda q, p: "assigned_to" in q and "has_active_session" in q, []),
        ]
        tg = _make_tg(_script_factory(responses))

        result = create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="11223344",
            task_short_hash="deadbeef",
            title="T",
            description="D",
            severity="low",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert result["error"] == "task_not_found"


# -----------------------------------------------------------------------------
# submit_review
# -----------------------------------------------------------------------------


class TestSubmitReviewHappyPath:
    def test_increments_review_count(self):
        """US-082: First review on a fresh bug: review_count → 1."""
        responses = [
            (lambda q, p: "CREATE (r:Review" in q, [
                {"review_uuid": "review-1", "review_count": 1}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix it",
            caller_agent_id="agent-2",
            workspace_id="ws-1",
        )
        assert result["review_uuid"] == "review-1"
        assert result["review_count"] == 1


class TestSubmitReviewErrorCases:
    def test_reviews_full(self):
        """US-082: WHERE review_count < max_reviews fails (max reached) → reviews_full."""
        responses = [
            (lambda q, p: "CREATE (r:Review" in q, []),  # main insert fails
            (lambda q, p: _has_status_check(q), [
                {"status": "open", "review_count": 2, "max_reviews": 2, "reporter": "agent-1"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-2",
            workspace_id="ws-1",
        )
        assert result["error"] == "max_reviews_reached"

    def test_cannot_review_own_bug(self):
        """Reporter attempts to review their own bug → cannot_review_own_bug."""
        responses = [
            (lambda q, p: "CREATE (r:Review" in q, []),
            (lambda q, p: _has_status_check(q), [
                {"status": "open", "review_count": 0, "max_reviews": 2, "reporter": "agent-1"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-1",  # reporter
            workspace_id="ws-1",
        )
        assert result["error"] == "cannot_review_own_bug"

    def test_already_reviewed(self):
        """Same agent already submitted a review → already_reviewed."""
        responses = [
            (lambda q, p: "CREATE (r:Review" in q, []),
            (lambda q, p: _has_status_check(q), [
                {"status": "open", "review_count": 1, "max_reviews": 2, "reporter": "agent-1"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-2",
            workspace_id="ws-1",
        )
        assert result["error"] == "already_reviewed"

    def test_bug_not_open(self):
        """Bug is approved/dismissed/resolved → bug_not_open."""
        responses = [
            (lambda q, p: "CREATE (r:Review" in q, []),
            (lambda q, p: _has_status_check(q), [
                {"status": "approved", "review_count": 1, "max_reviews": 2, "reporter": "agent-1"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-2",
            workspace_id="ws-1",
        )
        assert result["error"] == "bug_not_open"

    def test_bug_not_found(self):
        """Bug does not exist at all → bug_not_open."""
        responses = [
            (lambda q, p: "CREATE (r:Review" in q, []),
            (lambda q, p: _has_status_check(q), []),  # no bug
        ]
        tg = _make_tg(_script_factory(responses))

        result = submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="missing",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-2",
            workspace_id="ws-1",
        )
        assert result["error"] == "bug_not_open"


# -----------------------------------------------------------------------------
# validate_bug
# -----------------------------------------------------------------------------


def _has_validate_status_check(query):
    return "b.reported_by_agent" in query and "b.validated_at" in query and "collect(rev.reviewer_agent_id)" in query


class TestValidateBug:
    def test_aggregates_reviews(self):
        """Valid bug: returns avg_confidence, valid_count, total."""
        responses = [
            (lambda q, p: "SET b.avg_confidence" in q, [
                {
                    "uuid": "bug-1",
                    "status": "open",
                    "avg_confidence": 0.85,
                    "valid_count": 2,
                    "total": 2,
                }
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="verdict-agent",
            workspace_id="ws-1",
        )
        assert result["avg_confidence"] == 0.85
        assert result["valid_count"] == 2
        assert result["total"] == 2

    def test_rejects_caller_is_reporter(self):
        """US-083: Caller is the bug reporter → cannot_validate_own_bug."""
        responses = [
            (lambda q, p: "SET b.avg_confidence" in q, []),
            (lambda q, p: _has_validate_status_check(q), [
                {"status": "open", "review_count": 2, "max_reviews": 2,
                 "reporter": "agent-1", "validated_at": None, "reviewers": []}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="agent-1",  # reporter
            workspace_id="ws-1",
        )
        assert result["error"] == "cannot_validate_own_bug"

    def test_rejects_reviewer_cannot_validate(self):
        """US-083: Caller has reviewed the bug → reviewer_cannot_validate."""
        responses = [
            (lambda q, p: "SET b.avg_confidence" in q, []),
            (lambda q, p: _has_validate_status_check(q), [
                {"status": "open", "review_count": 2, "max_reviews": 2,
                 "reporter": "agent-1", "validated_at": None, "reviewers": ["agent-2"]}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="agent-2",  # already a reviewer
            workspace_id="ws-1",
        )
        assert result["error"] == "reviewer_cannot_validate"

    def test_rejects_reviews_incomplete(self):
        """review_count < max_reviews → reviews_incomplete."""
        responses = [
            (lambda q, p: "SET b.avg_confidence" in q, []),
            (lambda q, p: _has_validate_status_check(q), [
                {"status": "open", "review_count": 1, "max_reviews": 2,
                 "reporter": "agent-1", "validated_at": None, "reviewers": []}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="agent-3",
            workspace_id="ws-1",
        )
        assert result["error"] == "reviews_incomplete"

    def test_rejects_already_validated(self):
        """validated_at already set → already_validated."""
        responses = [
            (lambda q, p: "SET b.avg_confidence" in q, []),
            (lambda q, p: _has_validate_status_check(q), [
                {"status": "open", "review_count": 2, "max_reviews": 2,
                 "reporter": "agent-1", "validated_at": "2024-01-01T00:00:00Z", "reviewers": []}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="agent-3",
            workspace_id="ws-1",
        )
        assert result["error"] == "already_validated"


# -----------------------------------------------------------------------------
# resolve_bug
# -----------------------------------------------------------------------------


class TestResolveBug:
    def test_approved_creates_task(self):
        """US-084: resolution='approved': bug status → 'approved', task_short_hash returned."""
        responses = [
            (lambda q, p: "FOREACH" in q and "approved" in p.get("resolution", ""), [
                {"uuid": "bug-1", "status": "approved"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = resolve_bug(
            tg=tg,
            bug_uuid="bug-1",
            resolution="approved",
            reason="legit",
            caller_agent_id=None,
            workspace_id="ws-1",
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert result["status"] == "approved"
        assert result["task_short_hash"] == "aabbccdd"

    def test_dismissed_no_task(self):
        """US-085: resolution='dismissed': bug status → 'dismissed', no task_short_hash returned."""
        responses = [
            (lambda q, p: "FOREACH" in q, [
                {"uuid": "bug-1", "status": "dismissed"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = resolve_bug(
            tg=tg,
            bug_uuid="bug-1",
            resolution="dismissed",
            reason="not a bug",
            caller_agent_id=None,
            workspace_id="ws-1",
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert result["status"] == "dismissed"
        assert "task_short_hash" not in result

    def test_not_validated(self):
        """Bug has no validated_at → not_validated."""
        responses = [
            (lambda q, p: "FOREACH" in q, []),
            (lambda q, p: _has_resolve_status_check(q), [
                {"status": "open", "validated_at": None}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = resolve_bug(
            tg=tg,
            bug_uuid="bug-1",
            resolution="approved",
            reason=None,
            caller_agent_id=None,
            workspace_id="ws-1",
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert result["error"] == "not_validated"

    def test_bug_not_open(self):
        """Bug already approved/dismissed → bug_not_open."""
        responses = [
            (lambda q, p: "FOREACH" in q, []),
            (lambda q, p: _has_resolve_status_check(q), [
                {"status": "approved", "validated_at": "2024-01-01T00:00:00Z"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))

        result = resolve_bug(
            tg=tg,
            bug_uuid="bug-1",
            resolution="dismissed",
            reason=None,
            caller_agent_id=None,
            workspace_id="ws-1",
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert result["error"] == "bug_not_open"

    def test_bug_not_found(self):
        """Bug does not exist → bug_not_open."""
        responses = [
            (lambda q, p: "FOREACH" in q, []),
            (lambda q, p: _has_resolve_status_check(q), []),
        ]
        tg = _make_tg(_script_factory(responses))

        result = resolve_bug(
            tg=tg,
            bug_uuid="missing",
            resolution="dismissed",
            reason=None,
            caller_agent_id=None,
            workspace_id="ws-1",
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert result["error"] == "bug_not_open"


# -----------------------------------------------------------------------------
# B2 regression: Cypher structure of the 3 fixed queries must not have
# consecutive WHERE clauses. Caught by the structural validator in
# test_cypher_validator.py. The fake-executor pattern in the other tests
# silently ignores query syntax, so these tests close that gap.
# -----------------------------------------------------------------------------


class _EmptyScript:
    """Callable: returns [] for every (idx, query, params). Used to
    drive an action through the full Cypher path so we can capture
    every query the function emits, regardless of branch."""

    def __call__(self, idx, query, params):
        return []


_BUG_RECORD = {
    "uuid": "bug-1", "short_hash": "aabbccdd", "title": "T",
    "description": "D", "severity": "high", "status": "open",
    "max_reviews": 2, "review_count": 1, "reported_by_agent": "agent-1",
    "avg_confidence": None, "valid_reviewers": None, "total_reviewers": None,
    "created_at": "2024-01-01T00:00:00Z", "validated_at": None,
    "approved_at": None, "dismissed_at": None, "dismissal_reason": None,
}


class _BugPresentScript:
    """Callable: returns a bug record for get_bug_with_reviews' first
    execute_read (bug_query) so the function proceeds to its second
    execute_read (reviews_query). Reviews query returns []. Used to
    drive get_bug_with_reviews past its initial bug check."""

    def __call__(self, idx, query, params):
        if "reviewer_agent_id" in query:
            return []
        return [_BUG_RECORD]


def _fallback_status_queries(queries):
    """Return only the diagnostic fallback status_query from submit_review,
    validate_bug, and resolve_bug. The main query in each function also
    touches b.status / b.validated_at and contains a SET / CREATE write
    clause; the fallback is a pure read starting with the bare
    `MATCH (b:Bug {uuid: $bug_uuid})` (no `status: 'open'` filter)."""
    return [
        q for q in queries
        if "MATCH (b:Bug {uuid: $bug_uuid})" in q
        and "status: 'open'" not in q
    ]


class TestSubmitReviewQueryStructure:
    def test_submit_review_query_has_valid_cypher_structure(self):
        tg, queries = build_recording_tg(_EmptyScript())
        submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-2",
            workspace_id="ws-1",
        )
        assert len(queries) >= 1
        assert_queries_valid(queries)
        create_query = queries[0]
        assert "WHERE true" not in create_query

    def test_submit_review_with_no_workspace_uses_single_where(self):
        tg, queries = build_recording_tg(_EmptyScript())
        submit_review(
            tg=tg,
            review_uuid="review-1",
            bug_uuid="bug-1",
            confidence=0.9,
            verdict="valid",
            proposal="Fix",
            caller_agent_id="agent-2",
            workspace_id=None,
        )
        assert_queries_valid(queries)
        create_query = queries[0]
        assert "WHERE true" not in create_query


class TestValidateBugQueryStructure:
    def test_validate_bug_query_has_valid_cypher_structure(self):
        tg, queries = build_recording_tg(_EmptyScript())
        validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="verdict-agent",
            workspace_id="ws-1",
        )
        assert len(queries) >= 1
        assert_queries_valid(queries)
        validate_query = queries[0]
        assert "WHERE true" not in validate_query

    def test_validate_bug_with_no_workspace_uses_single_where(self):
        tg, queries = build_recording_tg(_EmptyScript())
        validate_bug(
            tg=tg,
            bug_uuid="bug-1",
            caller_agent_id="verdict-agent",
            workspace_id=None,
        )
        assert_queries_valid(queries)
        validate_query = queries[0]
        assert "WHERE true" not in validate_query


class TestResolveBugQueryStructure:
    def test_resolve_bug_query_has_valid_cypher_structure(self):
        tg, queries = build_recording_tg(_EmptyScript())
        resolve_bug(
            tg=tg,
            bug_uuid="bug-1",
            resolution="approved",
            reason="legit",
            caller_agent_id=None,
            workspace_id="ws-1",
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert len(queries) >= 1
        assert_queries_valid(queries)
        resolve_query = queries[0]
        assert "WHERE true" not in resolve_query

    def test_resolve_bug_with_no_workspace_uses_single_where(self):
        tg, queries = build_recording_tg(_EmptyScript())
        resolve_bug(
            tg=tg,
            bug_uuid="bug-1",
            resolution="dismissed",
            reason=None,
            caller_agent_id=None,
            workspace_id=None,
            task_uuid="entity-1",
            task_short_hash="aabbccdd",
        )
        assert_queries_valid(queries)
        resolve_query = queries[0]
        assert "WHERE true" not in resolve_query


class TestCreateBugQueryStructure:
    def test_create_bug_query_has_valid_cypher_structure(self):
        tg, queries = build_recording_tg(_EmptyScript())
        create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="aabbccdd",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="high",
            caller_agent_id="agent-1",
            workspace_id="ws-1",
            max_reviews=2,
        )
        assert len(queries) >= 1
        assert_queries_valid(queries)
        for q in queries:
            assert "WHERE true" not in q

    def test_create_bug_with_no_workspace_uses_single_where(self):
        tg, queries = build_recording_tg(_EmptyScript())
        create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="aabbccdd",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="low",
            caller_agent_id="agent-1",
            workspace_id=None,
            max_reviews=2,
        )
        assert_queries_valid(queries)
        for q in queries:
            assert "WHERE true" not in q

    def test_create_bug_with_default_workspace_uses_single_where(self):
        tg, queries = build_recording_tg(_EmptyScript())
        create_bug(
            tg=tg,
            bug_uuid="bug-1",
            bug_short_hash="aabbccdd",
            task_short_hash="abcd1234",
            title="T",
            description="D",
            severity="medium",
            caller_agent_id="agent-1",
            workspace_id="ws-1:default",
            max_reviews=2,
        )
        assert_queries_valid(queries)
        for q in queries:
            assert "WHERE true" not in q


# -----------------------------------------------------------------------------
# list_bugs / get_bug_with_reviews / agent_has_reviewed
# -----------------------------------------------------------------------------


class TestListBugs:
    def test_returns_listed_bugs(self):
        records = [
            {
                "uuid": "bug-1", "short_hash": "aabbccdd", "title": "T1",
                "severity": "high", "status": "open",
                "review_count": 1, "max_reviews": 2,
                "created_at": "2024-01-01T00:00:00Z", "reported_by_agent": "agent-1",
            },
            {
                "uuid": "bug-2", "short_hash": "eeff0011", "title": "T2",
                "severity": "low", "status": "open",
                "review_count": 0, "max_reviews": 2,
                "created_at": "2024-01-02T00:00:00Z", "reported_by_agent": "agent-2",
            },
        ]
        tg = _make_tg(_script_factory([(lambda q, p: True, records)]))
        result = list_bugs(tg, workspace_id="ws-1", status_filter=["open"], limit=100)
        assert len(result) == 2
        assert result[0]["short_hash"] == "aabbccdd"

    def test_returns_empty_on_no_workspace(self):
        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))
        result = list_bugs(tg, workspace_id="ws-1")
        assert result == []


class TestGetBugWithReviews:
    def test_returns_bug_with_reviews(self):
        responses = [
            (lambda q, p: "MATCH (b:Bug" in q and "RETURN b.uuid" in q and "valid_reviewers" in q, [
                {"uuid": "bug-1", "short_hash": "aabbccdd", "title": "T",
                 "description": "D", "severity": "high", "status": "open",
                 "max_reviews": 2, "review_count": 1, "reported_by_agent": "agent-1",
                 "avg_confidence": None, "valid_reviewers": None, "total_reviewers": None,
                 "created_at": "2024-01-01T00:00:00Z", "validated_at": None,
                 "approved_at": None, "dismissed_at": None, "dismissal_reason": None}
            ]),
            (lambda q, p: "MATCH (b:Bug" in q and "reviewer_agent_id" in q, [
                {"uuid": "review-1", "reviewer_agent_id": "agent-2",
                 "confidence_score": 0.9, "verdict": "valid",
                 "proposal": "Fix it", "submitted_at": "2024-01-02T00:00:00Z"}
            ]),
        ]
        tg = _make_tg(_script_factory(responses))
        bug = get_bug_with_reviews(tg, "bug-1", workspace_id="ws-1")
        assert bug is not None
        assert bug["short_hash"] == "aabbccdd"
        assert len(bug["reviews"]) == 1
        assert bug["reviews"][0]["verdict"] == "valid"

    def test_returns_none_when_bug_missing(self):
        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))
        bug = get_bug_with_reviews(tg, "missing", workspace_id="ws-1")
        assert bug is None


# -----------------------------------------------------------------------------
# Workspace scoping regression tests: fallback status_query + reviews_query
# must use _workspace_where (NOT exact property match) so that NULL-workspace
# bugs are visible when querying via a ":default" workspace.
# -----------------------------------------------------------------------------


class TestReviewsQueryWorkspaceScoping:
    """Regression: reviews_query must mirror bug_query's _workspace_where
    pattern. Exact-match against workspace_id silently dropped reviews for
    bugs that were created before workspace_id was set (NULL)."""

    def test_default_workspace_uses_or_null_clause(self):
        tg, queries = build_recording_tg(_BugPresentScript())
        get_bug_with_reviews(tg, "bug-1", workspace_id="ws-1:default")
        reviews_query = next(q for q in queries if "reviewer_agent_id" in q)
        assert "workspace_id: $workspace_id" not in reviews_query
        assert "b.workspace_id = $workspace_id" in reviews_query
        assert "$workspace_id" in reviews_query

    def test_set_workspace_uses_exact_match_clause(self):
        tg, queries = build_recording_tg(_BugPresentScript())
        get_bug_with_reviews(tg, "bug-1", workspace_id="ws-1")
        reviews_query = next(q for q in queries if "reviewer_agent_id" in q)
        assert "workspace_id: $workspace_id" not in reviews_query
        assert "IS NULL" not in reviews_query
        assert "b.workspace_id = $workspace_id" in reviews_query

    def test_none_workspace_omits_workspace_clause(self):
        tg, queries = build_recording_tg(_BugPresentScript())
        get_bug_with_reviews(tg, "bug-1", workspace_id=None)
        reviews_query = next(q for q in queries if "reviewer_agent_id" in q)
        assert "workspace_id: $workspace_id" not in reviews_query
        assert "IS NULL" not in reviews_query
        assert "b.workspace_id" not in reviews_query

    def test_reviews_query_must_not_contain_exact_property_match(self):
        """Hard guard: any reintroduction of the exact-match bug fails this."""
        tg, queries = build_recording_tg(_BugPresentScript())
        get_bug_with_reviews(tg, "bug-1", workspace_id="ws-1")
        reviews_query = next(q for q in queries if "reviewer_agent_id" in q)
        assert "uuid: $bug_uuid, workspace_id" not in reviews_query

    def test_returns_reviews_for_default_workspace_with_null_workspace_bug(self):
        """End-to-end: a bug with workspace_id=NULL must be returned when
        querying via the :default workspace."""
        bug_record = {
            "uuid": "bug-1", "short_hash": "aabbccdd", "title": "T",
            "description": "D", "severity": "high", "status": "open",
            "max_reviews": 2, "review_count": 1, "reported_by_agent": "agent-1",
            "avg_confidence": None, "valid_reviewers": None, "total_reviewers": None,
            "created_at": "2024-01-01T00:00:00Z", "validated_at": None,
            "approved_at": None, "dismissed_at": None, "dismissal_reason": None,
        }
        review_record = {
            "uuid": "review-1", "reviewer_agent_id": "agent-2",
            "confidence_score": 0.9, "verdict": "valid",
            "proposal": "Fix it", "submitted_at": "2024-01-02T00:00:00Z",
        }

        def script(idx, query, params):
            if "reviewer_agent_id" in query:
                return [review_record]
            return [bug_record]

        tg = _make_tg(script)
        bug = get_bug_with_reviews(tg, "bug-1", workspace_id="ws-1:default")
        assert bug is not None
        assert len(bug["reviews"]) == 1
        assert bug["reviews"][0]["verdict"] == "valid"


class TestFallbackStatusQueryWorkspaceScoping:
    """Regression: the diagnostic fallback status_query inside submit_review,
    validate_bug, and resolve_bug must use _workspace_where (NOT exact
    property match) so the user gets accurate error messages in the
    :default-workspace / NULL-bug edge case."""

    @pytest.mark.parametrize("action_fn,action_kwargs", [
        (
            "submit_review",
            {
                "review_uuid": "r-1", "confidence": 0.9, "verdict": "valid",
                "proposal": "Fix", "caller_agent_id": "agent-2",
            },
        ),
        (
            "validate_bug",
            {"caller_agent_id": "verdict-agent"},
        ),
        (
            "resolve_bug",
            {
                "resolution": "approved", "reason": "legit",
                "caller_agent_id": None, "task_uuid": "entity-1",
                "task_short_hash": "aabbccdd",
            },
        ),
    ])
    def test_default_workspace_uses_or_null_clause(self, action_fn, action_kwargs):
        from memory import bug_tracker
        fn = getattr(bug_tracker, action_fn)
        kwargs = {"tg": None, "bug_uuid": "bug-1", "workspace_id": "ws-1:default"}
        kwargs.update(action_kwargs)
        tg, queries = build_recording_tg(_EmptyScript())
        kwargs["tg"] = tg
        fn(**kwargs)
        status_queries = _fallback_status_queries(queries)
        assert status_queries, "no fallback status_query captured"
        for q in status_queries:
            assert "workspace_id: $workspace_id" not in q, (
                f"status_query still uses exact workspace_id match:\n{q}"
            )
            assert "b.workspace_id = $workspace_id" in q
            assert "$workspace_id" in q

    @pytest.mark.parametrize("action_fn,action_kwargs", [
        (
            "submit_review",
            {
                "review_uuid": "r-1", "confidence": 0.9, "verdict": "valid",
                "proposal": "Fix", "caller_agent_id": "agent-2",
            },
        ),
        (
            "validate_bug",
            {"caller_agent_id": "verdict-agent"},
        ),
        (
            "resolve_bug",
            {
                "resolution": "approved", "reason": "legit",
                "caller_agent_id": None, "task_uuid": "entity-1",
                "task_short_hash": "aabbccdd",
            },
        ),
    ])
    def test_set_workspace_uses_exact_match_clause(self, action_fn, action_kwargs):
        from memory import bug_tracker
        fn = getattr(bug_tracker, action_fn)
        kwargs = {"tg": None, "bug_uuid": "bug-1", "workspace_id": "ws-1"}
        kwargs.update(action_kwargs)
        tg, queries = build_recording_tg(_EmptyScript())
        kwargs["tg"] = tg
        fn(**kwargs)
        status_queries = _fallback_status_queries(queries)
        assert status_queries, "no fallback status_query captured"
        for q in status_queries:
            assert "workspace_id: $workspace_id" not in q
            assert "IS NULL" not in q
            assert "b.workspace_id = $workspace_id" in q


class TestGetBugByShortHash:
    def test_returns_bug_via_short_hash(self):
        responses = [
            (lambda q, p: "MATCH (b:Bug" in q and "RETURN b.uuid AS uuid" in q and "LIMIT 2" in q, [
                {"uuid": "bug-1"}
            ]),
            (lambda q, p: "MATCH (b:Bug" in q and "valid_reviewers" in q, [
                {"uuid": "bug-1", "short_hash": "aabbccdd", "title": "T",
                 "description": "D", "severity": "high", "status": "open",
                 "max_reviews": 2, "review_count": 0, "reported_by_agent": "agent-1",
                 "avg_confidence": None, "valid_reviewers": None, "total_reviewers": None,
                 "created_at": "2024-01-01T00:00:00Z", "validated_at": None,
                 "approved_at": None, "dismissed_at": None, "dismissal_reason": None}
            ]),
            (lambda q, p: "MATCH (b:Bug" in q and "reviewer_agent_id" in q, []),
        ]
        tg = _make_tg(_script_factory(responses))
        bug = get_bug_by_short_hash(tg, "aabbccdd", workspace_id="ws-1")
        assert bug is not None
        assert bug["uuid"] == "bug-1"

    def test_returns_none_for_empty_hash(self):
        tg = MagicMock()
        assert get_bug_by_short_hash(tg, "", workspace_id="ws-1") is None
        assert get_bug_by_short_hash(tg, None, workspace_id="ws-1") is None

    def test_returns_none_when_not_found(self):
        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))
        assert get_bug_by_short_hash(tg, "aabbccdd", workspace_id="ws-1") is None

    def test_returns_infrastructure_error_sentinel_on_initial_query_failure(self):
        tg = MagicMock()
        tg._executor.execute_read.side_effect = InfrastructureError("db down")
        result = get_bug_by_short_hash(tg, "aabbccdd", workspace_id="ws-1")
        assert result is not None
        assert result.get("error") == "infrastructure_error"
        assert result.get("message") == "Database temporarily unavailable"


class TestAgentHasReviewed:
    def test_returns_true_when_reviewed(self):
        tg = _make_tg(_script_factory([(lambda q, p: True, [{"has_reviewed": True}])]))
        assert agent_has_reviewed(tg, "bug-1", "agent-2") is True

    def test_returns_false_when_not_reviewed(self):
        tg = _make_tg(_script_factory([(lambda q, p: True, [{"has_reviewed": False}])]))
        assert agent_has_reviewed(tg, "bug-1", "agent-2") is False

    def test_returns_false_when_no_records(self):
        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))
        assert agent_has_reviewed(tg, "bug-1", "agent-2") is False


# -----------------------------------------------------------------------------
# InfrastructureError differentiation: get_bug_with_reviews + action layer
# -----------------------------------------------------------------------------


class TestGetBugWithReviewsInfrastructureError:
    def test_returns_sentinel_on_infrastructure_error(self):
        tg = MagicMock()
        tg._executor.execute_read.side_effect = InfrastructureError("db down")
        result = get_bug_with_reviews(tg, "bug-1", workspace_id="ws-1")
        assert result is not None
        assert result.get("error") == "infrastructure_error"
        assert "message" in result


class TestValidateBugActionInfrastructureError:
    def test_returns_infrastructure_error_when_db_down(self):
        from proxy.tool_pipeline.actions.validate_bug import ValidateBugAction

        tg = MagicMock()
        tg._executor.execute_read.side_effect = InfrastructureError("db down")
        result = ValidateBugAction.execute(
            arguments={"bug_uuid": "bug-1"},
            session_id="sess-1",
            tg=tg,
            workspace_id="ws-1",
            caller_agent_id="agent-2",
        )
        assert result["error"]["code"] == "infrastructure_error"
        assert "session_id" in result

    def test_still_returns_bug_not_open_when_bug_missing(self):
        from proxy.tool_pipeline.actions.validate_bug import ValidateBugAction

        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))
        result = ValidateBugAction.execute(
            arguments={"bug_uuid": "missing"},
            session_id="sess-1",
            tg=tg,
            workspace_id="ws-1",
            caller_agent_id="agent-2",
        )
        assert result["error"]["code"] == "bug_not_open"

    def test_returns_bug_not_open_for_non_open_bug(self):
        """US-094: validate_bug on Non-Open Bug.

        Action-layer pre-check rejects bugs whose status is not 'open'
        (e.g. 'approved') with bug_not_open before hitting the DB write.
        """
        from proxy.tool_pipeline.actions.validate_bug import ValidateBugAction

        approved_bug = {
            "uuid": "bug-1", "short_hash": "aabbccdd", "title": "T",
            "description": "D", "severity": "high", "status": "approved",
            "max_reviews": 2, "review_count": 2, "reported_by_agent": "agent-1",
            "avg_confidence": 0.9, "valid_reviewers": "2", "total_reviewers": "2",
            "created_at": "2024-01-01T00:00:00Z", "validated_at": "2024-01-12T00:00:00Z",
            "approved_at": None, "dismissed_at": None, "dismissal_reason": None,
        }
        tg = _make_tg(_script_factory([
            (lambda q, p: "reviewer_agent_id" in q, []),
            (lambda q, p: True, [approved_bug]),
        ]))
        result = ValidateBugAction.execute(
            arguments={"bug_uuid": "bug-1"},
            session_id="sess-1",
            tg=tg,
            workspace_id="ws-1",
            caller_agent_id="agent-3",
        )
        assert result["error"]["code"] == "bug_not_open"
        assert result["session_id"] == "sess-1"





class TestResolveBugActionInfrastructureError:
    def test_returns_infrastructure_error_when_db_down(self):
        from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction

        tg = MagicMock()
        tg._executor.execute_read.side_effect = InfrastructureError("db down")
        result = ResolveBugAction.execute(
            arguments={"bug_uuid": "bug-1", "resolution": "approved"},
            session_id="sess-1",
            tg=tg,
            workspace_id="ws-1",
            caller_agent_id=None,
        )
        assert result["error"]["code"] == "infrastructure_error"
        assert "session_id" in result

    def test_still_returns_bug_not_open_when_bug_missing(self):
        from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction

        tg = _make_tg(_script_factory([(lambda q, p: True, [])]))
        result = ResolveBugAction.execute(
            arguments={"bug_uuid": "missing", "resolution": "approved"},
            session_id="sess-1",
            tg=tg,
            workspace_id="ws-1",
            caller_agent_id=None,
        )
        assert result["error"]["code"] == "bug_not_open"


class TestResolveBugActionSubagentFailClosed:
    """C2: when get_or_load cannot verify the caller (returns None or raises),
    the action must fail closed — treat the caller as an untrusted sub-agent
    so the not_authorized gate applies. resolve_bug is a HUMAN-ONLY action."""

    _ARGS = {"bug_uuid": "bug-1", "resolution": "approved"}
    _BUG = {"status": "open", "validated_at": "2024-01-01T00:00:00Z"}
    _RESOLVE = {"status": "approved", "task_short_hash": "ts1hash1"}

    def test_unverifiable_agent_none_returns_not_authorized(self):
        """get_or_load returns None (SQLite degraded / evicted / unknown) — the
        real production fail-open path. Mock the bug lookup + resolve to succeed
        so a pass here pins the GATE, not a downstream cascade."""
        from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction

        with patch("proxy.agent.agent_registry.registry.get_or_load", return_value=None), \
                patch("memory.bug_tracker.get_bug_with_reviews") as mock_gbr, \
                patch("memory.bug_tracker.resolve_bug") as mock_rb:
            mock_gbr.return_value = self._BUG
            mock_rb.return_value = self._RESOLVE
            result = ResolveBugAction.execute(
                arguments=self._ARGS,
                session_id="sess-1",
                tg=MagicMock(),
                workspace_id="ws-1",
                caller_agent_id=None,
                is_subagent=False,
            )
            assert result["error"]["code"] == "not_authorized"
            assert "session_id" in result
            # Gate must short-circuit before the bug lookup / resolution.
            mock_gbr.assert_not_called()
            mock_rb.assert_not_called()

    def test_registry_lookup_raises_returns_not_authorized(self):
        """get_or_load raises (must not be silently swallowed to fail open)."""
        from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction

        with patch("proxy.agent.agent_registry.registry.get_or_load",
                   side_effect=RuntimeError("registry blew up")), \
                patch("memory.bug_tracker.get_bug_with_reviews") as mock_gbr, \
                patch("memory.bug_tracker.resolve_bug") as mock_rb:
            mock_gbr.return_value = self._BUG
            mock_rb.return_value = self._RESOLVE
            result = ResolveBugAction.execute(
                arguments=self._ARGS,
                session_id="sess-1",
                tg=MagicMock(),
                workspace_id="ws-1",
                caller_agent_id=None,
                is_subagent=False,
            )
            assert result["error"]["code"] == "not_authorized"
            mock_gbr.assert_not_called()
            mock_rb.assert_not_called()

    def test_verified_human_proceeds_past_gate(self):
        """Positive control: a verified human (is_subagent=False) must NOT be
        over-denied — the action proceeds and returns success."""
        from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction

        human = MagicMock()
        human.is_subagent = False
        with patch("proxy.agent.agent_registry.registry.get_or_load",
                   return_value=human), \
                patch("memory.bug_tracker.get_bug_with_reviews") as mock_gbr, \
                patch("memory.bug_tracker.resolve_bug") as mock_rb:
            mock_gbr.return_value = self._BUG
            mock_rb.return_value = self._RESOLVE
            result = ResolveBugAction.execute(
                arguments=self._ARGS,
                session_id="sess-1",
                tg=MagicMock(),
                workspace_id="ws-1",
                caller_agent_id=None,
                is_subagent=False,
            )
            assert result.get("status") != "error"
            assert result["operation"] == "resolve_bug"
            assert result["data"]["bug_status"] == "approved"
            assert result["data"]["bug_uuid"] == "bug-1"
