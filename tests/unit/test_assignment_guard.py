"""US-076: Cypher WHERE assignment enforcement on MODIFY actions.

Verifies the defense-in-depth boundary in the production Cypher:
when ``caller_agent_id`` is provided, every MODIFY action (UPDATE/SET
path) MUST inject

    AND (t.assigned_agent_id IS NULL OR t.assigned_agent_id = $caller_agent_id)

into the WHERE clause of its write query, and the
``update_task_status`` / ``update_task`` write MUST return a
``not_found`` style error when the row is filtered out by that clause
(no silent success).
"""
import pytest


from memory.task_graph import TaskGraph
from tests.unit.test_cypher_validator import build_recording_tg


def _count_zero_response(idx: int, query: str, params: dict) -> list:
    """A fake-executor response that returns count=0 for the initial
    count_query. This simulates the production DB returning zero rows
    because the assignment guard filtered out a task owned by a
    different agent.
    """
    return [{"cnt": 0, "current_status": None}]


class TestAssignmentGuardEnforcedOnUpdateTaskStatus:
    """US-076: update_task_status returns not_found when assignment guard
    filters out the row.  Without the guard the count query would return
    the row and the call would proceed silently.
    """

    def test_returns_not_found_when_assignment_filter_excludes_row(self):
        """When the WHERE clause filters out the row (caller_agent_id
        does not match assigned_agent_id), update_task_status MUST
        return ``{"ok": False, "status": "not_found"}`` rather than
        silently succeed.
        """
        tg, _queries = build_recording_tg(_count_zero_response)

        result = TaskGraph.update_task_status(
            tg,
            "abcd1234",
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )

        assert result["ok"] is False
        assert result["status"] == "not_found", (
            f"Expected not_found when assignment guard filters out the row, "
            f"got: {result!r}"
        )

    def test_emits_assignment_authz_clause_when_caller_agent_id_present(self):
        """The emitted Cypher MUST include the assignment-authz clause
        ``t.assigned_agent_id = $caller_agent_id`` (or the OR-with-NULL
        equivalent). This is the defense-in-depth boundary.
        """
        tg, queries = build_recording_tg(_count_zero_response)

        TaskGraph.update_task_status(
            tg,
            "abcd1234",
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )

        assert queries, "update_task_status did not issue any queries"
        authz_clause = (
            "t.assigned_agent_id IS NULL OR t.assigned_agent_id = $caller_agent_id"
        )
        assert any(authz_clause in q for q in queries), (
            "update_task_status did not emit the assignment-authz clause. "
            "Captured queries:\n" + "\n---\n".join(queries)
        )

    def test_emits_caller_agent_id_param(self):
        """The $caller_agent_id param MUST be present in the count_query
        params. Without it the parameterized query would still filter
        on $caller_agent_id (which is unset), rejecting every task.
        """
        captured_params: list[dict] = []

        def script(idx, query, params):
            captured_params.append(dict(params or {}))
            return [{"cnt": 0, "current_status": None}]

        tg, _queries = build_recording_tg(script)

        TaskGraph.update_task_status(
            tg,
            "abcd1234",
            "in_progress",
            workspace_id="ws-1",
            caller_agent_id="agent-A",
        )

        assert any(p.get("caller_agent_id") == "agent-A" for p in captured_params), (
            f"caller_agent_id not passed in params; captured: {captured_params!r}"
        )
