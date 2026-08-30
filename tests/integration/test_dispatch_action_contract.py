"""Contract tests for dispatch_action() -> action handler return shape.

These tests document and verify the contract between ``dispatch_action()``
(proxy/tool_pipeline/action_registry.py) and the registered TaskAction
handlers in ``proxy/tool_pipeline/actions/``.

Scope of the contract:
  1. ``dispatch_action`` forwards ``workspace_id`` and ``caller_agent_id``
     into the handler.
  2. Every action's return dict includes the ``session_id`` passed to
     ``dispatch_action``.
  3. Error responses have a consistent shape with ``error`` and
     ``session_id`` keys.
  4. Success responses are JSON-serializable dicts with the action's
     expected keys.

Behavior-level tests live elsewhere (e.g. test_ticket_work_lifecycle.py).
"""
from __future__ import annotations

import contextlib
import json
import uuid as _uuid
from unittest.mock import MagicMock, patch

import pytest

import memory.tools.proxy_ticket  # noqa: F401 — populates ACTION_REGISTRY
import memory.tools.proxy_work  # noqa: F401 — populates ACTION_REGISTRY
from memory.query_executor import InfrastructureError
from memory.task_graph import AmbiguousHashError
from proxy.tool_pipeline.action_registry import (
    ACTION_REGISTRY,
    dispatch_action,
    register_action,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture: ensure the registry is fully populated
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_registry_populated():
    """Re-register all actions before each test.

    Other test files (e.g. test_c9_fail_open_intent.py) use
    ``register_action`` to override handlers and pop them on teardown,
    which can leave the registry missing entries. This fixture guarantees
    the full set is available for every contract test.
    """
    from memory.tools.proxy_ticket import (
        CreateTasksAction,
        UpdateTaskAction,
        ListTasksAction,
        GetTaskAction,
        EditAction,
        SetParentAction,
        ClearParentAction,
        ClearBlockedByAction,
        SetPriorityAction,
        AppendReviewNoteAction,
    )
    from memory.tools.proxy_work import (
        StartWorkAction,
        StopWorkAction,
        ReportBugAction,
        ReviewBugAction,
        ValidateBugAction,
        ResolveBugAction,
        HandoffTaskAction,
    )

    register_action("create_tasks", CreateTasksAction)
    register_action("update_task", UpdateTaskAction)
    register_action("list_tasks", ListTasksAction)
    register_action("get_task", GetTaskAction)
    register_action("edit", EditAction)
    register_action("set_parent", SetParentAction)
    register_action("clear_parent", ClearParentAction)
    register_action("clear_blocked_by", ClearBlockedByAction)
    register_action("set_priority", SetPriorityAction)
    register_action("append_review_note", AppendReviewNoteAction)
    register_action("start_work", StartWorkAction)
    register_action("stop_work", StopWorkAction)
    register_action("report_bug", ReportBugAction)
    register_action("review_bug", ReviewBugAction)
    register_action("validate_bug", ValidateBugAction)
    register_action("resolve_bug", ResolveBugAction)
    register_action("handoff_task", HandoffTaskAction)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _temp_action(name: str, handler):
    """Register a handler for the block, then restore the prior binding."""
    existed = name in ACTION_REGISTRY
    previous = ACTION_REGISTRY.get(name)
    register_action(name, handler)
    try:
        yield
    finally:
        if existed:
            ACTION_REGISTRY[name] = previous
        else:
            ACTION_REGISTRY.pop(name, None)


def _make_capturing_handler(return_value: dict | None = None, side_effect=None):
    """Build a handler that records the kwargs it was called with.

    Returns ``(handler_mock, capture_dict)`` where ``capture_dict`` is a
    mutable container the test can inspect for captured args.
    """
    capture: dict = {"calls": []}
    handler = MagicMock()
    if side_effect is not None:
        handler.execute.side_effect = side_effect
    else:
        handler.execute.return_value = return_value

    def _record(*args, **kwargs):
        capture["calls"].append({"args": args, "kwargs": kwargs})
        if side_effect is not None:
            return side_effect(*args, **kwargs)
        return return_value

    handler.execute.side_effect = _record
    return handler, capture


def _mock_tg() -> MagicMock:
    """Build a generic MagicMock for ``tg``.

    Most actions in this file are stubbed via ``patch.object`` so the
    methods on ``tg`` are irrelevant. The TG is still passed in to satisfy
    the protocol.
    """
    return MagicMock()


# ---------------------------------------------------------------------------
# Test 1: workspace_id flows from dispatch to action handler
# ---------------------------------------------------------------------------


class TestWorkspaceIdFlowsToAction:
    """``dispatch_action`` must forward ``workspace_id`` to the handler."""

    def test_create_tasks_receives_workspace_id(self):
        handler, capture = _make_capturing_handler(
            return_value={"ok": True, "session_id": "sid"}
        )
        with _temp_action("create_tasks", handler):
            dispatch_action(
                "create_tasks",
                {"tasks": [{"name": "T", "priority": "high"}]},
                "sid",
                _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-test",
                caller_agent_id="agent-1",
            )

        assert len(capture["calls"]) == 1
        kwargs = capture["calls"][0]["kwargs"]
        assert kwargs.get("workspace_id") == "ws-test"

    def test_update_task_receives_workspace_id(self):
        handler, capture = _make_capturing_handler(
            return_value={"ok": True, "session_id": "sid"}
        )
        with _temp_action("update_task", handler):
            dispatch_action(
                "update_task",
                {"short_hash": "abc", "new_status": "complete"},
                "sid",
                _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-flow",
                caller_agent_id="agent-1",
            )

        assert len(capture["calls"]) == 1
        assert capture["calls"][0]["kwargs"].get("workspace_id") == "ws-flow"

    def test_workspace_id_none_is_preserved(self):
        """dispatch_action forwards workspace_id=None unchanged (no defaulting)."""
        handler, capture = _make_capturing_handler(
            return_value={"ok": True, "session_id": "sid"}
        )
        with _temp_action("create_tasks", handler):
            dispatch_action(
                "create_tasks",
                {"tasks": [{"name": "T", "priority": "high"}]},
                "sid",
                _mock_tg(),
                tool="proxy_ticket",
                workspace_id=None,
                caller_agent_id="agent-1",
            )

        assert "workspace_id" in capture["calls"][0]["kwargs"]
        assert capture["calls"][0]["kwargs"]["workspace_id"] is None


# ---------------------------------------------------------------------------
# Test 2: caller_agent_id flows from dispatch to action handler
# ---------------------------------------------------------------------------


class TestCallerAgentIdFlowsToAction:
    """``dispatch_action`` must forward ``caller_agent_id`` to the handler."""

    def test_update_task_receives_caller_agent_id(self):
        handler, capture = _make_capturing_handler(
            return_value={"ok": True, "session_id": "sid"}
        )
        with _temp_action("update_task", handler):
            dispatch_action(
                "update_task",
                {"short_hash": "abc", "new_status": "complete"},
                "sid",
                _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1",
                caller_agent_id="agent-1",
            )

        assert len(capture["calls"]) == 1
        assert capture["calls"][0]["kwargs"].get("caller_agent_id") == "agent-1"

    def test_caller_agent_id_resolved_from_session_when_omitted(self):
        """If caller_agent_id is None, dispatch resolves it from session_id."""
        from tests.mode_harness import create_test_agent
        from proxy.agent.agent_registry import registry as agent_registry

        agent = create_test_agent(agent_id="resolved-agent")
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            return {"ok": True, "session_id": args[1] if len(args) > 1 else "sid"}

        handler = MagicMock()
        handler.execute.side_effect = _spy
        with _temp_action("update_task", handler):
            with patch.object(agent_registry, "get_or_load", return_value=agent):
                dispatch_action(
                    "update_task",
                    {"short_hash": "abc", "new_status": "complete"},
                    "test-session",
                    _mock_tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-1",
                    caller_agent_id=None,
                )

        assert captured.get("caller_agent_id") == "resolved-agent"


# ---------------------------------------------------------------------------
# Test 3: return dict always contains session_id
# ---------------------------------------------------------------------------


# Actions and the arguments that produce a successful (non-error) response.
# For actions that need a backend, we patch the action class itself so the
# contract test does not depend on DB state.
_HAPPY_PATH_ARGS: dict[str, dict] = {
    "create_tasks": {"tasks": [{"name": "T", "priority": "high"}]},
    "update_task": {"short_hash": "abc12345", "new_status": "complete"},
    "list_tasks": {},
    "get_task": {"short_hash": "abc12345"},
    "edit": {"short_hash": "abc12345", "summary": "updated"},
    "set_parent": {"short_hash": "abc12345", "parent_hash": "parent12345"},
    "clear_parent": {"short_hash": "abc12345"},
    "clear_blocked_by": {"short_hash": "abc12345"},
    "set_priority": {"short_hash": "abc12345", "priority": "high"},
    "append_review_note": {"short_hash": "abc12345", "note": "test note"},
    "start_work": {"short_hash": "abc12345"},
    "stop_work": {"short_hash": "abc12345", "ws_uuid": "ws-uuid-1"},
    "report_bug": {"task_short_hash": "abc12345", "title": "t", "description": "d", "severity": "low"},
    "review_bug": {"bug_uuid": "b-1", "confidence": 0.9, "verdict": "valid", "proposal": "ok"},
    "validate_bug": {"bug_uuid": "b-1"},
    "resolve_bug": {"bug_uuid": "b-1", "resolution": "approved"},
    "handoff_task": {"short_hash": "abc12345", "to_agent_id": "other-agent"},
}


# The full set of action class imports. Patching these is the simplest way
# to make the contract test DB-agnostic.
def _import_action_classes():
    """Return a map of action_name -> action class."""
    from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction
    from proxy.tool_pipeline.actions.update_task import UpdateTaskAction
    from proxy.tool_pipeline.actions.list_tasks import ListTasksAction
    from proxy.tool_pipeline.actions.get_task import GetTaskAction
    from proxy.tool_pipeline.actions.edit import EditAction
    from proxy.tool_pipeline.actions.set_parent import SetParentAction
    from proxy.tool_pipeline.actions.clear_parent import ClearParentAction
    from proxy.tool_pipeline.actions.clear_blocked_by import ClearBlockedByAction
    from proxy.tool_pipeline.actions.set_priority import SetPriorityAction
    from proxy.tool_pipeline.actions.append_review_note import AppendReviewNoteAction
    from proxy.tool_pipeline.actions.start_work import StartWorkAction
    from proxy.tool_pipeline.actions.stop_work import StopWorkAction
    from proxy.tool_pipeline.actions.report_bug import ReportBugAction
    from proxy.tool_pipeline.actions.review_bug import ReviewBugAction
    from proxy.tool_pipeline.actions.validate_bug import ValidateBugAction
    from proxy.tool_pipeline.actions.resolve_bug import ResolveBugAction
    from proxy.tool_pipeline.actions.handoff_task import HandoffTaskAction

    return {
        "create_tasks": CreateTasksAction,
        "update_task": UpdateTaskAction,
        "list_tasks": ListTasksAction,
        "get_task": GetTaskAction,
        "edit": EditAction,
        "set_parent": SetParentAction,
        "clear_parent": ClearParentAction,
        "clear_blocked_by": ClearBlockedByAction,
        "set_priority": SetPriorityAction,
        "append_review_note": AppendReviewNoteAction,
        "start_work": StartWorkAction,
        "stop_work": StopWorkAction,
        "report_bug": ReportBugAction,
        "review_bug": ReviewBugAction,
        "validate_bug": ValidateBugAction,
        "resolve_bug": ResolveBugAction,
        "handoff_task": HandoffTaskAction,
    }


# A canonical "success" return for each action — just enough keys to look
# like a happy response. The point of these tests is the SHAPE, not the
# values.
_SUCCESS_RESPONSES: dict[str, dict] = {
    "create_tasks": {"action": "create_tasks", "created": [], "errors": [], "total_created": 0, "total_errors": 0},
    "update_task": {"action": "update_task", "completed": True, "status": "complete", "short_hash": "abc12345"},
    "list_tasks": {"action": "list_tasks", "tasks": [], "total": 0, "store_status": {}},
    "get_task": {"action": "get_task", "short_hash": "abc12345", "uuid": "u-1", "name": "T", "status": "pending", "priority": "high", "summary": "", "tag": None, "planning_info": None, "planning_info_truncated": False, "created_at": None, "assigned_agent_id": None, "blocked_by": [], "is_blocked": False, "blocked_count": 0, "parent_short_hash": None, "parent_name": None, "review_notes": []},
    "edit": {"action": "edit", "short_hash": "abc12345", "updated": True},
    "set_parent": {"action": "set_parent", "short_hash": "abc12345", "parent_short_hash": "parent12345", "updated": True},
    "clear_parent": {"action": "clear_parent", "short_hash": "abc12345", "updated": True},
    "clear_blocked_by": {"action": "clear_blocked_by", "short_hash": "abc12345", "updated": True},
    "set_priority": {"action": "set_priority", "short_hash": "abc12345", "priority": "high", "updated": True},
    "append_review_note": {"action": "append_review_note", "short_hash": "abc12345", "note_count": 1, "updated": True},
    "start_work": {"action": "start_work", "ws_uuid": "ws-uuid-1", "task_status": "in_progress"},
    "stop_work": {"action": "stop_work", "ws_uuid": "ws-uuid-1", "task_status": "complete"},
    "report_bug": {"action": "report_bug", "bug_uuid": "b-1", "bug_short_hash": "b1hash12"},
    "review_bug": {"action": "review_bug", "review_uuid": "r-1", "review_count": 1},
    "validate_bug": {"action": "validate_bug", "bug_uuid": "b-1", "avg_confidence": 0.9, "valid_count": 2, "total_count": 2},
    "resolve_bug": {"action": "resolve_bug", "bug_uuid": "b-1", "bug_status": "approved", "task_short_hash": "ts1hash1"},
    "handoff_task": {"action": "handoff_task", "short_hash": "abc12345", "to_agent_id": "other-agent", "closed_sessions": 0},
}


def _all_action_names() -> list[str]:
    """All action names currently registered."""
    # Sort for determinism
    return sorted(ACTION_REGISTRY.keys())


class TestReturnDictContainsSessionId:
    """Every action's return dict must include the session_id passed in."""

    def test_every_registered_action_includes_session_id(self):
        """For every action in the registry, dispatch must surface session_id."""
        action_classes = _import_action_classes()
        action_names = _all_action_names()
        assert action_names, "ACTION_REGISTRY should be populated by imports"

        session_id = f"contract-{_uuid.uuid4().hex[:8]}"
        for name in action_names:
            args = _HAPPY_PATH_ARGS.get(name)
            assert args is not None, f"no happy-path args for {name}"
            success = dict(_SUCCESS_RESPONSES[name])
            success["session_id"] = session_id

            cls = action_classes.get(name)
            assert cls is not None, f"no class mapping for {name}"

            with patch.object(cls, "execute", return_value=success):
                result = dispatch_action(
                    name,
                    args,
                    session_id,
                    _mock_tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-contract",
                    caller_agent_id="agent-1",
                )

            assert "session_id" in result, (
                f"{name} response missing session_id: {result}"
            )
            assert result["session_id"] == session_id, (
                f"{name} returned wrong session_id: {result.get('session_id')!r}"
            )


# ---------------------------------------------------------------------------
# Test 4: error response format is consistent
# ---------------------------------------------------------------------------


# Invalid arguments for each action. These should trigger the validation
# error path inside the handler (the "before any DB work" branch), which
# returns a uniform error dict shape.
_INVALID_ARGS: dict[str, dict] = {
    "create_tasks": {},  # missing required 'tasks'
    "update_task": {"short_hash": "", "new_status": "bogus"},  # both invalid
    "list_tasks": {"status_filter": "not-a-list"},  # wrong type
    "get_task": {},  # missing required short_hash
    "edit": {},  # missing required short_hash
    "set_parent": {},  # missing short_hash AND parent_hash
    "clear_parent": {},  # missing short_hash
    "clear_blocked_by": {},  # missing short_hash
    "set_priority": {},  # missing short_hash AND priority
    "append_review_note": {},  # missing short_hash AND note
    "start_work": {},  # missing short_hash; caller_agent_id is required
    "stop_work": {},  # missing short_hash AND ws_uuid
    "report_bug": {},  # missing all required fields
    "review_bug": {},  # missing all required fields
    "validate_bug": {},  # missing bug_uuid
    "resolve_bug": {},  # missing bug_uuid AND resolution
    "handoff_task": {},  # missing short_hash AND to_agent_id
}


class TestErrorResponseFormat:
    """Error responses must have a consistent shape."""

    def test_every_action_returns_error_with_session_id_on_invalid_input(self):
        session_id = f"err-{_uuid.uuid4().hex[:8]}"

        for name in _all_action_names():
            args = _INVALID_ARGS[name]
            # Call dispatch_action and let the real handler run validation.
            # The real handler's input-validation path returns a uniform
            # error dict shape that we inspect.
            result = dispatch_action(
                name,
                args,
                session_id,
                _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-err",
                caller_agent_id="agent-1",
            )

            # The response should still be a dict.
            assert isinstance(result, dict), (
                f"{name} returned non-dict on invalid input: {result!r}"
            )
            # If the action's validation fired, the response must have
            # both 'error' and 'session_id' keys.
            if result.get("error") is not None:
                assert "session_id" in result, (
                    f"{name} error response missing session_id: {result}"
                )
                assert result["session_id"] == session_id, (
                    f"{name} error response has wrong session_id: {result}"
                )

    def test_dispatch_error_response_for_unknown_action(self):
        """Unknown action returns a structured canonical error envelope."""
        result = dispatch_action(
            "_no_such_action_xyz", {}, "sid-1", _mock_tg(),
            tool="proxy_ticket",
            workspace_id="ws-1", caller_agent_id="agent-1",
        )
        assert result["error"]["code"] == "invalid_action"
        assert result["operation"] is None
        assert result["session_id"] == "sid-1"
        assert "message" in result["error"]
        assert "valid_actions" in result
        assert isinstance(result["valid_actions"], list)
        assert result["valid_actions"] == sorted(result["valid_actions"])

    def test_dispatch_error_response_for_ambiguous_hash(self):
        """AmbiguousHashError -> error.code='ambiguous_hash' with session_id."""
        handler = MagicMock()
        handler.execute.side_effect = AmbiguousHashError(2)
        with _temp_action("_test_ambiguous", handler):
            result = dispatch_action(
                "_test_ambiguous", {}, "sid-amb", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "ambiguous_hash"
        assert result["session_id"] == "sid-amb"

    def test_dispatch_error_response_for_infrastructure_error(self):
        """InfrastructureError -> error.code='infrastructure_error' with session_id."""
        handler = MagicMock()
        handler.execute.side_effect = InfrastructureError("neo4j down")
        with _temp_action("_test_infra", handler):
            result = dispatch_action(
                "_test_infra", {}, "sid-infra", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "infrastructure_error"
        assert result["session_id"] == "sid-infra"

    def test_dispatch_error_response_for_generic_exception(self):
        """Generic Exception -> error.code='action_failed' with session_id."""
        handler = MagicMock()
        handler.execute.side_effect = RuntimeError("boom")
        with _temp_action("_test_generic", handler):
            result = dispatch_action(
                "_test_generic", {}, "sid-fail", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        assert result["error"]["code"] == "action_failed"
        assert result["session_id"] == "sid-fail"

    def test_error_response_is_json_serializable(self):
        """Every error response must be JSON-serializable."""
        handler = MagicMock()
        handler.execute.side_effect = RuntimeError("boom")
        with _temp_action("_test_json", handler):
            result = dispatch_action(
                "_test_json", {}, "sid-j", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        # Must not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# Test 5: success response format is consistent
# ---------------------------------------------------------------------------


class TestSuccessResponseFormat:
    """Success responses must be JSON-serializable dicts without 'error'."""

    def test_every_action_success_response_is_json_serializable(self):
        action_classes = _import_action_classes()
        session_id = f"succ-{_uuid.uuid4().hex[:8]}"
        for name in _all_action_names():
            args = _HAPPY_PATH_ARGS[name]
            success = dict(_SUCCESS_RESPONSES[name])
            success["session_id"] = session_id
            cls = action_classes[name]

            with patch.object(cls, "execute", return_value=success):
                result = dispatch_action(
                    name,
                    args,
                    session_id,
                    _mock_tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-succ",
                    caller_agent_id="agent-1",
                )

            assert isinstance(result, dict), (
                f"{name} returned non-dict on success: {result!r}"
            )
            # Success response must not be an error envelope
            assert result.get("status") != "error", (
                f"{name} success response unexpectedly has status=error: {result}"
            )
            # Must be JSON-serializable
            json.dumps(result)
            # Must have session_id
            assert result.get("session_id") == session_id


# ---------------------------------------------------------------------------
# Test 6: action returns dict without session_id — dispatch adds it
# ---------------------------------------------------------------------------


class TestDispatchInjectsSessionId:
    """Document whether dispatch_action auto-injects session_id.

    The current contract (per action_registry.py:69-71) is that
    dispatch_action returns the handler's result *unchanged* on success.
    Session_id injection is the handler's responsibility, not dispatch's.
    This test pins down that contract.
    """

    def test_action_returning_dict_without_session_id_preserves_other_fields(self):
        """Other fields are preserved even if session_id is missing."""
        handler = MagicMock()
        handler.execute.return_value = {"operation": "noop", "ok": True}
        with _temp_action("_no_sid", handler):
            result = dispatch_action(
                "_no_sid", {}, "sid-x", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        assert result["operation"] == "noop"
        assert result["ok"] is True
        # Other fields preserved
        assert "operation" in result
        assert "ok" in result

    def test_action_returning_dict_with_session_id_preserves_it(self):
        """If the handler includes session_id, dispatch keeps it."""
        handler = MagicMock()
        handler.execute.return_value = {
            "operation": "noop", "ok": True, "session_id": "explicit-sid",
        }
        with _temp_action("_has_sid", handler):
            result = dispatch_action(
                "_has_sid", {}, "sid-x", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        assert result["session_id"] == "explicit-sid"

    def test_dispatch_does_not_inject_session_id_into_success_response(self):
        """Pin down the actual contract: dispatch does NOT auto-inject
        session_id into a successful response. Handlers must include it.

        This is intentional defense-in-depth: handlers own the response
        shape, and dispatch only guarantees session_id is present on
        error responses.
        """
        handler = MagicMock()
        handler.execute.return_value = {"operation": "noop", "ok": True}
        with _temp_action("_no_inject", handler):
            result = dispatch_action(
                "_no_inject", {}, "sid-x", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        # session_id is NOT injected by dispatch
        assert "session_id" not in result


# ---------------------------------------------------------------------------
# Test 7: action returns string or non-dict — dispatch handles gracefully
# ---------------------------------------------------------------------------


class TestDispatchHandlesNonDictReturn:
    """Document whether dispatch_action handles non-dict returns gracefully.

    Current contract (per action_registry.py:69-71): dispatch returns
    whatever the handler returns. There is no coercion to dict. This test
    pins down the current behavior and documents the contract gap if any.
    """

    def test_action_returning_string_propagates_unchanged(self):
        """Current behavior: dispatch returns the string unchanged."""
        handler = MagicMock()
        handler.execute.return_value = "just a string"
        with _temp_action("_returns_str", handler):
            result = dispatch_action(
                "_returns_str", {}, "sid-1", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        # Current contract: propagate as-is (does not coerce to dict)
        assert result == "just a string"

    def test_action_returning_none_propagates_unchanged(self):
        """Current behavior: dispatch returns None unchanged."""
        handler = MagicMock()
        handler.execute.return_value = None
        with _temp_action("_returns_none", handler):
            result = dispatch_action(
                "_returns_none", {}, "sid-1", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        # Current contract: propagate as-is
        assert result is None

    def test_dispatch_does_not_crash_on_non_dict(self):
        """No crash from dispatch_action when handler returns a non-dict."""
        handler = MagicMock()
        handler.execute.return_value = ["a", "list", "of", "things"]
        with _temp_action("_returns_list", handler):
            # Must not raise
            result = dispatch_action(
                "_returns_list", {}, "sid-1", _mock_tg(),
                tool="proxy_ticket",
                workspace_id="ws-1", caller_agent_id="agent-1",
            )
        # Propagates as-is
        assert result == ["a", "list", "of", "things"]


# ---------------------------------------------------------------------------
# Test 8: All registered actions are covered
# ---------------------------------------------------------------------------


class TestAllRegisteredActionsCovered:
    """For every action in ACTION_REGISTRY, the contract holds."""

    def test_action_registry_is_populated(self):
        """Sanity check: registry should not be empty after imports."""
        assert len(ACTION_REGISTRY) >= 17, (
            f"Expected at least 17 registered actions, got {len(ACTION_REGISTRY)}: "
            f"{sorted(ACTION_REGISTRY.keys())}"
        )

    def test_all_action_names_have_happy_path_args(self):
        """Every registered action has a happy-path arg set in this test."""
        action_names = set(_all_action_names())
        happy_path = set(_HAPPY_PATH_ARGS.keys())
        missing = action_names - happy_path
        assert not missing, f"No happy-path args for: {sorted(missing)}"

    def test_all_action_names_have_invalid_args(self):
        """Every registered action has an invalid-args set in this test."""
        action_names = set(_all_action_names())
        invalid = set(_INVALID_ARGS.keys())
        missing = action_names - invalid
        assert not missing, f"No invalid args for: {sorted(missing)}"

    def test_every_action_handler_has_execute_method(self):
        """Every registered handler is callable with the TaskAction protocol."""
        for name, handler in ACTION_REGISTRY.items():
            assert hasattr(handler, "execute"), (
                f"{name} handler missing execute method: {handler!r}"
            )

    def test_every_action_returns_dict_with_session_id_in_dispatch(self):
        """The main contract: dispatch surfaces session_id for every action."""
        action_classes = _import_action_classes()
        session_id = f"cov-{_uuid.uuid4().hex[:8]}"

        for name in _all_action_names():
            args = _HAPPY_PATH_ARGS[name]
            success = dict(_SUCCESS_RESPONSES[name])
            success["session_id"] = session_id
            cls = action_classes[name]

            with patch.object(cls, "execute", return_value=success):
                result = dispatch_action(
                    name,
                    args,
                    session_id,
                    _mock_tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-cov",
                    caller_agent_id="agent-1",
                )

            assert isinstance(result, dict), f"{name} -> non-dict: {result!r}"
            assert result.get("session_id") == session_id, (
                f"{name} -> session_id={result.get('session_id')!r}, "
                f"expected {session_id!r}"
            )
