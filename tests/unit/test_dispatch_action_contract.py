"""
Unit tests for ``dispatch_action``'s error-taxonomy contract.

``ConstraintViolation`` and ``QueryTimeoutError`` (defined in
``memory.query_executor``) are SIBLINGS of ``InfrastructureError`` — they all
subclass ``Exception`` directly. Before this fix they fell through
``dispatch_action``'s generic ``except Exception`` and were masked as
``action_failed``, losing the distinct signal these siblings carry:

  * ``ConstraintViolation`` — uniqueness/relationship constraint violations
    (not retryable). task_graph.py already emits ``constraint_violation``.
  * ``QueryTimeoutError`` — query exceeded the deadline (may be retryable).

These have LIVE propagation paths: e.g. ``EditAction`` →
``tg.get_task_by_short_hash`` reaches ``dispatch_action`` with these
exceptions (the action only catches ``AmbiguousHashError`` + raw neo4j
transient exceptions, NOT the mapped siblings).

The contract under test:

  1. ``dispatch_action`` surfaces ``ConstraintViolation`` as a canonical
    error envelope with ``error.code == "constraint_violation"`` (not
    ``action_failed``).
  2. ``dispatch_action`` surfaces ``QueryTimeoutError`` as a canonical
    error envelope with ``error.code == "query_timeout"`` (not
    ``action_failed``).
  3. Real-path: the REAL ``EditAction`` (not a mock handler) calls
     ``tg.get_task_by_short_hash``; when the executor raises
     ``QueryTimeoutError``, ``dispatch_action`` surfaces it as
     ``query_timeout`` — proving reachability through production action code.
"""
import contextlib
from unittest.mock import MagicMock

import pytest

from memory.query_executor import (
    ConstraintViolation,
    InfrastructureError,
    QueryTimeoutError,
)
from proxy.tool_pipeline.action_registry import (
    ACTION_REGISTRY,
    dispatch_action,
    register_action,
)
from proxy.tool_pipeline.actions.edit import EditAction

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers (mirror the test_c9_fail_open_intent.py pattern)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _temp_action(name, handler):
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


# ---------------------------------------------------------------------------
# ConstraintViolation is differentiated from generic Exception
# ---------------------------------------------------------------------------

class TestDispatchActionConstraintViolation:
    """``dispatch_action`` must surface ``ConstraintViolation`` as
    ``error.code == "constraint_violation"`` — not mask it as
    ``action_failed``."""

    def test_constraint_violation_returns_constraint_violation_payload(self):
        handler = MagicMock()
        handler.execute.side_effect = ConstraintViolation(
            "already exists with the same value"
        )
        with _temp_action("_contract_cv_handler", handler):
            result = dispatch_action(
                "_contract_cv_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "constraint_violation"
        assert result["error"]["code"] != "action_failed"
        assert result["status"] == "error"
        assert result["session_id"] == "sid"

    def test_constraint_violation_message_is_the_exception_text(self):
        handler = MagicMock()
        handler.execute.side_effect = ConstraintViolation("uniq boom")
        with _temp_action("_contract_cv_msg_handler", handler):
            result = dispatch_action(
                "_contract_cv_msg_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["message"] == "uniq boom"


# ---------------------------------------------------------------------------
# QueryTimeoutError is differentiated from generic Exception
# ---------------------------------------------------------------------------

class TestDispatchActionQueryTimeout:
    """``dispatch_action`` must surface ``QueryTimeoutError`` as
    ``error.code == "query_timeout"`` — not mask it as ``action_failed``."""

    def test_query_timeout_returns_query_timeout_payload(self):
        handler = MagicMock()
        handler.execute.side_effect = QueryTimeoutError("db query slow")
        with _temp_action("_contract_qt_handler", handler):
            result = dispatch_action(
                "_contract_qt_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "query_timeout"
        assert result["error"]["code"] != "action_failed"
        assert result["status"] == "error"
        assert result["session_id"] == "sid"

    def test_query_timeout_message_mentions_timed_out(self):
        handler = MagicMock()
        handler.execute.side_effect = QueryTimeoutError("deadline")
        with _temp_action("_contract_qt_msg_handler", handler):
            result = dispatch_action(
                "_contract_qt_msg_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert "timed out" in result["error"]["message"].lower()


# ---------------------------------------------------------------------------
# Real-path: the REAL EditAction propagates QueryTimeoutError to dispatch
# ---------------------------------------------------------------------------

class TestDispatchActionRealPathQueryTimeout:
    """Reachability proof: the production ``EditAction`` calls
    ``tg.get_task_by_short_hash`` (only ``AmbiguousHashError`` + raw neo4j
    transient exceptions are caught locally). When the executor raises
    ``QueryTimeoutError``, it must propagate to ``dispatch_action`` and surface
    as ``query_timeout`` — NOT be masked as ``action_failed``."""

    def test_real_edit_action_query_timeout_surfaces_via_dispatch(self):
        tg = MagicMock()
        tg.is_available.return_value = True
        # Real EditAction reaches get_task_by_short_hash when blocked_by is set.
        tg.get_task_by_short_hash.side_effect = QueryTimeoutError("db query slow")

        arguments = {
            "short_hash": "abc",
            "blocked_by": ["def"],
        }
        # Register the REAL EditAction (class, not a mock handler) so the
        # production code path is exercised — only `tg` is mocked.
        with _temp_action("edit", EditAction):
            result = dispatch_action(
                "edit", arguments, "sid", tg,
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "query_timeout"
        assert result["error"]["code"] != "action_failed"
        # Confirm the real action actually reached the executor call.
        tg.get_task_by_short_hash.assert_called_once()


# ---------------------------------------------------------------------------
# Regression guards: InfrastructureError + generic Exception stay stable
# ---------------------------------------------------------------------------

class TestDispatchActionTaxonomyRegressionGuards:
    """Ensure the new siblings did not disturb the existing handlers."""

    def test_infrastructure_error_still_returns_infrastructure_error(self):
        handler = MagicMock()
        handler.execute.side_effect = InfrastructureError("neo4j down")
        with _temp_action("_contract_infra_handler", handler):
            result = dispatch_action(
                "_contract_infra_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "infrastructure_error"

    def test_generic_exception_still_returns_action_failed(self):
        handler = MagicMock()
        handler.execute.side_effect = ValueError("boom")
        with _temp_action("_contract_generic_handler", handler):
            result = dispatch_action(
                "_contract_generic_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "action_failed"
