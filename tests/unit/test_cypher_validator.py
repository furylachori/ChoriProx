"""Reusable Cypher structural validator + capture-and-validate test helper.

Detects common syntax/structural issues in Cypher query strings without
requiring a live Neo4j instance. The regex-based checks catch a class of
bugs that the existing unit-test pattern (scripted fake executors that
return canned records) silently allows through, because the fake
executor never parses the query string — it just matches substrings.

Currently detects:
  - Two consecutive WHERE clauses without an intervening WITH (the
    exact bug that B2 introduced in submit_review / validate_bug /
    resolve_bug).
  - WHERE clauses that follow a SET, MERGE, or CREATE clause without a
    separating WITH. WHERE is only valid as part of MATCH / OPTIONAL
    MATCH / WITH; it cannot appear after a write clause.

The capture-and-validate helper runs a target function with a recording
fake executor, harvests every Cypher string the function emits, and
asserts each one passes validate_cypher_structure().

Future checks can be added to validate_cypher_structure() to cover
additional Cypher pitfalls (e.g. missing WITH between two write
subqueries, unbalanced parentheses in expressions, etc.).
"""
import re
from typing import Any, Callable, List
from unittest.mock import MagicMock


_CLAUSE_PATTERN = re.compile(
    r"\b(MATCH|OPTIONAL\s+MATCH|CREATE|MERGE|SET|DELETE|DETACH|REMOVE|WITH|UNWIND|RETURN|FOREACH|WHERE|CALL|LOAD\s+CSV|USE)\b",
    re.IGNORECASE,
)


def _strip_comments_and_strings(query: str) -> str:
    """Replace string literals and comments with blanks so keyword
    matching inside them is ignored. Cypher single-quoted strings can
    contain escaped quotes; this replacement is good enough for
    structural boundary detection."""
    out = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", query)
    out = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', out)
    out = re.sub(r"//[^\n]*", "", out)
    return out


def _split_clauses(query: str) -> List[str]:
    """Return the top-level Cypher clause headers in order."""
    return [
        re.sub(r"\s+", " ", m.group(0).upper()).strip()
        for m in _CLAUSE_PATTERN.finditer(query)
    ]


_WRITE_CLAUSES = {"CREATE", "MERGE", "SET", "DELETE", "REMOVE", "FOREACH"}


def validate_cypher_structure(query: str) -> List[str]:
    """Return a list of structural issue messages. Empty list = valid.

    This is a deliberately conservative check. It is NOT a full Cypher
    parser — it only catches the structural patterns that have actually
    bitten us. Adding more checks is encouraged.
    """
    issues: List[str] = []
    stripped = _strip_comments_and_strings(query)
    clauses = _split_clauses(stripped)
    if not clauses:
        return issues

    for i, cur in enumerate(clauses):
        if cur != "WHERE":
            continue
        if i == 0:
            continue
        prev = clauses[i - 1]
        if prev == "WHERE":
            issues.append(
                f"two consecutive WHERE clauses at positions {i-1} and {i}"
            )
            continue
        if prev in _WRITE_CLAUSES:
            issues.append(
                f"WHERE at position {i} follows write clause {prev} "
                f"without an intervening WITH"
            )
    return issues


class _RecordingTx:
    """In-memory tx whose run() records every query into the shared list."""

    def __init__(self, queries: List[str], records_by_query: Callable[[int, str, dict], list]):
        self._queries = queries
        self._records_by_query = records_by_query
        self._idx = 0

    def run(self, query, params=None):
        self._queries.append(query)
        records = self._records_by_query(self._idx, query, params or {})
        self._idx += 1
        return _RecordIter(records)


class _RecordIter:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


def build_recording_tg(records_by_query: Callable[[int, str, dict], list]
                       ) -> tuple[Any, List[str]]:
    """Build a fake TaskGraph that records every Cypher string issued.

    Returns (tg, queries) where `queries` is the list the executor
    appends to on every tx.run() / execute_read() / execute_write()
    call. The user inspects the list after the action runs.

    `records_by_query` is a callable (idx, query, params) -> list[dict]
    that returns the records the action should observe. This mirrors
    the existing _script_factory pattern in the test suite.
    """
    from unittest.mock import MagicMock

    tg = MagicMock()
    queries: List[str] = []

    def _execute_tx_no_retry(fn):
        tx = _RecordingTx(queries, records_by_query)
        return fn(tx)

    call_idx = {"n": 0}

    def _step_script(query, params):
        queries.append(query)
        records = records_by_query(call_idx["n"], query, params or {})
        call_idx["n"] += 1
        return records

    tg._executor.execute_transaction_no_retry.side_effect = _execute_tx_no_retry
    tg._executor.execute_write.side_effect = lambda q, p=None, timeout=30: _step_script(q, p)
    tg._executor.execute_read.side_effect = lambda q, p=None, timeout=30: _step_script(q, p)
    return tg, queries


def assert_queries_valid(queries: List[str]) -> None:
    """Assert every captured Cypher string passes structural validation.

    The assertion message lists every issue per query so a regression
    points directly at the offending file/function.
    """
    all_issues: List[str] = []
    for i, q in enumerate(queries):
        issues = validate_cypher_structure(q)
        for msg in issues:
            all_issues.append(f"query[{i}]: {msg}\n  query: {q.strip()[:200]}")
    assert not all_issues, (
        "Cypher structural validation failed:\n" + "\n".join(all_issues)
    )


# -----------------------------------------------------------------------------
# Self-tests for the validator
# -----------------------------------------------------------------------------


class TestValidateCypherStructure:
    def test_single_where_after_match_is_valid(self):
        q = "MATCH (n) WHERE n.x = 1 RETURN n"
        assert validate_cypher_structure(q) == []

    def test_consecutive_where_clauses_are_invalid(self):
        q = (
            "MATCH (b:Bug) "
            "WHERE true "
            "  AND b.workspace_id = $ws "
            "WHERE b.status = 'open' "
            "RETURN b"
        )
        issues = validate_cypher_structure(q)
        assert len(issues) == 1
        assert "consecutive WHERE" in issues[0]

    def test_where_after_with_is_valid(self):
        q = (
            "MATCH (b:Bug) "
            "WHERE b.status = 'open' "
            "WITH b "
            "WHERE b.x = 1 "
            "RETURN b"
        )
        assert validate_cypher_structure(q) == []

    def test_where_after_set_is_invalid(self):
        q = "MATCH (n) SET n.x = 1 WHERE n.y = 2 RETURN n"
        issues = validate_cypher_structure(q)
        assert len(issues) == 1

    def test_empty_query_has_no_issues(self):
        assert validate_cypher_structure("") == []

    def test_no_clauses_has_no_issues(self):
        assert validate_cypher_structure("RETURN 1") == []

    def test_where_inside_string_literal_is_ignored(self):
        q = "MATCH (n) WHERE n.x = 'WHERE something' RETURN n"
        assert validate_cypher_structure(q) == []

    def test_reproduces_b2_bug_in_submit_review(self):
        q = (
            "MATCH (b:Bug {uuid: $bug_uuid, status: 'open'}) "
            "WHERE true "
            "  AND b.workspace_id = $workspace_id "
            "WHERE b.review_count < b.max_reviews "
            "  AND b.reported_by_agent <> $caller_agent_id "
            "CREATE (r:Review {uuid: $review_uuid}) "
            "RETURN r"
        )
        issues = validate_cypher_structure(q)
        assert len(issues) == 1
        assert "WHERE" in issues[0]


# Authz clause that every MODIFY action on a Task must include.
# We accept the standard phrasing used in task_graph.py (the
# "t.assigned_agent_id IS NULL OR t.assigned_agent_id = $caller_agent_id"
# guard) AND the equivalent "= $caller_agent_id" phrasing for callers
# that already enforce assignment via the action layer.
_ASSIGNMENT_AUTHZ_RE = re.compile(
    r"t\.assigned_agent_id\s*=\s*\$caller_agent_id",
    re.IGNORECASE,
)


class TestCypherAuthorizationShape:
    """Verify that the generated Cypher for every MODIFY (write) action
    carries the assignment-guard authz clause:

        t.assigned_agent_id = $caller_agent_id

    The clause is a defense-in-depth boundary: even if the action layer
    forgets to check, the database refuses to mutate tasks the caller
    does not own.  An action that omits the clause would let any agent
    overwrite any task's status.
    """

    def test_update_task_status_emits_assignment_authz(self):
        """The canonical MODIFY action: update_task_status must include
        ``t.assigned_agent_id = $caller_agent_id`` in the WHERE clause.
        """
        from memory.task_graph import TaskGraph

        records_sequence = [
            [{"cnt": 1, "current_status": "pending"}],
            [{"updated": 1}],
        ]
        tg, queries = build_recording_tg(lambda i, q, p: records_sequence[i])

        TaskGraph.update_task_status(
            tg,
            "abcd1234",
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )

        assert queries, "update_task_status did not issue any queries"
        authz_queries = [q for q in queries if _ASSIGNMENT_AUTHZ_RE.search(q)]
        assert authz_queries, (
            "update_task_status did not emit an assignment-authz clause. "
            "Captured queries:\n" + "\n---\n".join(queries)
        )

    def test_update_task_status_omits_authz_when_no_caller_agent_id(self):
        """When caller_agent_id is None the action layer does not pass
        the param.  The authz clause must be omitted from the query —
        otherwise the parameterized query would still filter on
        $caller_agent_id (which is unset), rejecting every task.
        """
        from memory.task_graph import TaskGraph

        records_sequence = [
            [{"cnt": 1, "current_status": "pending"}],
            [{"updated": 1}],
        ]
        tg, queries = build_recording_tg(lambda i, q, p: records_sequence[i])

        TaskGraph.update_task_status(
            tg,
            "abcd1234",
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id=None,
        )

        authz_queries = [q for q in queries if _ASSIGNMENT_AUTHZ_RE.search(q)]
        assert not authz_queries, (
            "When caller_agent_id is None the authz clause must be "
            "omitted (no param is bound). Found it in:\n" +
            "\n---\n".join(authz_queries)
        )

    def test_modify_action_on_wrong_agent_returns_zero_rows(self):
        """Negative test for the authz boundary: when the executor
        returns zero rows, the action's "is the task owned by this
        caller?" check must reject the request with not_found — NOT
        silently update the row.

        Concretely: the count_query returns 0 records (the authz
        WHERE filtered out the task because the task belongs to
        another agent).  The action must surface this as a
        ``status: not_found`` response so the caller knows the
        write was rejected.
        """
        from memory.task_graph import TaskGraph

        records_sequence = [
            [],
        ]
        tg, queries = build_recording_tg(lambda i, q, p: records_sequence[i])

        result = TaskGraph.update_task_status(
            tg,
            "abcd1234",
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )

        assert queries, "update_task_status did not issue any queries"
        first_read = queries[0]
        assert _ASSIGNMENT_AUTHZ_RE.search(first_read), (
            "First read query must include assignment authz to defend "
            "against cross-agent writes. Query was:\n" + first_read
        )
        assert result["ok"] is False, (
            f"update_task_status must reject the request when the "
            f"executor returns 0 rows (authz filtered out the task). "
            f"Got result={result!r}"
        )
        assert result["status"] in {"not_found", "error"}, (
            f"Expected status in {{'not_found', 'error'}}, got "
            f"{result.get('status')!r}.  Action must not silently "
            f"report success when the authz check filtered out the task."
        )


class TestCypherAuthorizationExceptionList:
    """Pins the documented EXCEPTION list of actions whose authorization
    model does NOT use the standard ``t.assigned_agent_id = $caller_agent_id``
    clause.

    These actions have their own authz story (verdict-agent, human
    operator) and the assignment-check would be WRONG for them.  The
    test documents and pins which actions are on the exception list, so
    a future refactor that adds ``assigned_agent_id = $caller_agent_id``
    to a validate/resolve action — accidentally blocking the verdict
    agent or human operator — fails this test.
    """

    EXCEPTION_ACTIONS = {"validate_bug", "resolve_bug"}

    def test_exception_list_matches_documented_actions(self):
        """Pin the EXCEPTION_ACTIONS set.  Adding/removing entries is a
        security-relevant change and must be reviewed.
        """
        assert self.EXCEPTION_ACTIONS == {"validate_bug", "resolve_bug"}, (
            "Exception list changed — review whether the new entry "
            "really has its own authz model.  validate_bug and resolve_bug "
            "are excluded because validate_bug uses "
            "b.reported_by_agent <> $caller_agent_id (verdict agent must "
            "not be the reporter) and resolve_bug is a human-operator "
            "action that does not check agent identity at all."
        )

    def test_validate_bug_does_not_use_assigned_agent_id_authz(self):
        """The validate_bug Cypher must NOT include the
        ``t.assigned_agent_id = $caller_agent_id`` check — it uses
        a different authz model (verdict-agent must not be the reporter).
        """
        from memory import bug_tracker

        queries: List[str] = []
        tg = MagicMock()

        def _tx_step(fn):
            tx = MagicMock()
            captured = {"count": 0}

            def _run(query, params=None):
                queries.append(query)
                if captured["count"] == 0:
                    captured["count"] += 1
                    return iter([])
                return iter([{
                    "status": "open",
                    "review_count": 2,
                    "max_reviews": 2,
                    "reporter": "agent-X",
                    "validated_at": None,
                    "reviewers": ["agent-Y"],
                }])

            tx.run.side_effect = _run
            return fn(tx)

        tg._executor.execute_transaction_no_retry.side_effect = _tx_step

        result = bug_tracker.validate_bug(
            tg=tg,
            bug_uuid="bug-uuid-1",
            caller_agent_id="verdict-agent",
            workspace_id="ws-1",
        )

        assert queries, "validate_bug did not issue any queries"
        for q in queries:
            assert not _ASSIGNMENT_AUTHZ_RE.search(q), (
                "validate_bug must NOT use t.assigned_agent_id authz — "
                "it uses b.reported_by_agent <> $caller_agent_id. "
                "Query was:\n" + q
            )
        _ = result

    def test_resolve_bug_does_not_use_assigned_agent_id_authz(self):
        """The resolve_bug Cypher must NOT include the
        ``t.assigned_agent_id = $caller_agent_id`` check — it is a
        human-operator action that does not check agent identity.
        """
        from memory import bug_tracker

        queries: List[str] = []

        def _tx_step(fn):
            tx = MagicMock()
            captured = {"count": 0}

            def _run(query, params=None):
                queries.append(query)
                if captured["count"] == 0:
                    captured["count"] += 1
                    return iter([])
                return iter([{"status": "open", "validated_at": "2024-01-01T00:00:00Z"}])

            tx.run.side_effect = _run
            return fn(tx)

        tg = MagicMock()
        tg._executor.execute_transaction_no_retry.side_effect = _tx_step

        result = bug_tracker.resolve_bug(
            tg=tg,
            bug_uuid="bug-uuid-1",
            resolution="approved",
            reason=None,
            caller_agent_id="human-operator",
            workspace_id="ws-1",
            task_uuid="entity-task-uuid",
            task_short_hash="abcd1234",
        )

        assert queries, "resolve_bug did not issue any queries"
        for q in queries:
            assert not _ASSIGNMENT_AUTHZ_RE.search(q), (
                "resolve_bug must NOT use t.assigned_agent_id authz — "
                "it is a human-operator action. Query was:\n" + q
            )
        _ = result

