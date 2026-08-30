"""
Unit tests for the C9 fail-open intent fix.

C9 split the bare ``except Exception: pass`` in create_tasks.py:124-128 into:
  - ``except InfrastructureError: raise`` (fail-closed, batch rejected)
  - ``except Exception as e: ... pass`` (fail-open, task created unvalidated)

But the original ``raise`` was caught by ``dispatch_action``'s generic
``except Exception`` and converted to ``action_failed``, masking the
infrastructure_error signal. This test module verifies the corrected
behaviour:

  1. ``dispatch_action`` distinguishes ``InfrastructureError`` and returns
     a canonical error envelope with ``error.code == "infrastructure_error"``
     (not ``action_failed``).
  2. ``dispatch_action`` still converts a generic ``Exception`` to
     ``action_failed`` (the framework-level fallback).
  3. ``CreateTasksAction`` is fail-open for non-infrastructure errors from
     ``agent_registry.get_or_load`` — the task is still created.
  4. ``CreateTasksAction`` re-raises ``InfrastructureError`` so it reaches
     ``dispatch_action`` and surfaces as ``infrastructure_error``.
  5. The existing happy path through ``dispatch_action`` still works.
"""
import contextlib
import pytest
from unittest.mock import MagicMock, patch

from memory.query_executor import InfrastructureError
from memory.task_graph_common import AmbiguousHashError
from proxy.tool_pipeline.action_registry import (
    ACTION_REGISTRY,
    dispatch_action,
    register_action,
)
from proxy.tool_pipeline.actions.create_tasks import CreateTasksAction

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
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


def _mock_agent():
    """Return a mock agent whose ``agent_id`` is a stable string."""
    a = MagicMock()
    a.agent_id = "agent-1"
    return a


# ---------------------------------------------------------------------------
# dispatch_action: InfrastructureError is differentiated from generic Exception
# ---------------------------------------------------------------------------

class TestDispatchActionInfrastructureError:
    """``dispatch_action`` must surface ``InfrastructureError`` as
    ``error.code == "infrastructure_error"`` — not mask it as
    ``action_failed``."""

    def test_infrastructure_error_returns_infrastructure_error_payload(self):
        handler = MagicMock()
        handler.execute.side_effect = InfrastructureError("neo4j down")
        with _temp_action("_c9_infra_handler", handler):
            result = dispatch_action(
                "_c9_infra_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "infrastructure_error"
        assert result["status"] == "error"
        assert result["session_id"] == "sid"

    def test_generic_exception_still_returns_action_failed(self):
        handler = MagicMock()
        handler.execute.side_effect = ValueError("boom")
        with _temp_action("_c9_generic_handler", handler):
            result = dispatch_action(
                "_c9_generic_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "action_failed"
        assert result["session_id"] == "sid"

    def test_ambiguous_hash_still_returns_ambiguous(self):
        handler = MagicMock()
        handler.execute.side_effect = AmbiguousHashError(2)
        with _temp_action("_c9_ambiguous_handler", handler):
            result = dispatch_action(
                "_c9_ambiguous_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "ambiguous_hash"

    def test_happy_path_returns_handler_result_unchanged(self):
        handler = MagicMock()
        handler.execute.return_value = {"ok": True, "session_id": "sid"}
        with _temp_action("_c9_happy_handler", handler):
            result = dispatch_action(
                "_c9_happy_handler", {"x": 1}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result == {"ok": True, "session_id": "sid"}
        handler.execute.assert_called_once()

    def test_infra_error_message_mentions_database_unavailable(self):
        handler = MagicMock()
        handler.execute.side_effect = InfrastructureError("transient")
        with _temp_action("_c9_infra_msg_handler", handler):
            result = dispatch_action(
                "_c9_infra_msg_handler", {}, "sid", MagicMock(),
                tool="proxy_ticket", caller_agent_id="agent-1",
            )

        msg = result["error"]["message"].lower()
        assert "database" in msg or "unavailable" in msg


# ---------------------------------------------------------------------------
# CreateTasksAction: fail-open semantics for assign_to lookup
# ---------------------------------------------------------------------------

class TestCreateTasksAssignToFailOpen:
    """``create_tasks`` must remain fail-open for non-infrastructure errors
    raised by ``agent_registry.get_or_load`` and must re-raise
    ``InfrastructureError`` so the framework-level handler can decide."""

    @pytest.fixture
    def mock_tg(self):
        tg = MagicMock()
        tg.is_available.return_value = True
        tg.get_task_by_short_hash.return_value = None
        tg.update_task.return_value = True
        tg.create_tasks_batch.return_value = [
            {"short_hash": "t1_hash", "uuid": "entity-t1-00000001", "name": "T1", "priority": "high", "status": "pending"},
        ]
        return tg

    def test_non_infra_registry_error_returns_error_envelope(self, mock_tg):
        """Generic Exception in registry lookup → error envelope (fail-closed)."""
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=RuntimeError("registry blew up"),
        ):
            arguments = {
                "tasks": [
                    {"name": "T1", "priority": "high", "assign_to": "ghost-agent"},
                ]
            }
            result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result["status"] == "error"
        assert result["error"]["code"] == "assign_to_lookup_failed"
        assert "ghost-agent" in result["error"]["message"]
        mock_tg.create_tasks_batch.assert_not_called()

    def test_infrastructure_error_propagates_out_of_create_tasks(self, mock_tg):
        """InfrastructureError must re-raise (fail-closed at the framework)."""
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            side_effect=InfrastructureError("neo4j transient"),
        ):
            arguments = {
                "tasks": [
                    {"name": "T1", "priority": "high", "assign_to": "some-agent"},
                ]
            }
            with pytest.raises(InfrastructureError):
                CreateTasksAction.execute(arguments, "sid", mock_tg)

    def test_infrastructure_error_from_create_tasks_surfaces_as_infra_error_via_dispatch(self, mock_tg):
        """End-to-end: when CreateTasksAction re-raises, dispatch_action returns
        ``infrastructure_error`` — NOT ``action_failed`` (C9 regression guard)."""
        raising_handler = MagicMock()
        raising_handler.execute.side_effect = InfrastructureError("neo4j transient")
        with _temp_action("create_tasks", raising_handler):
            result = dispatch_action(
                "create_tasks", {"tasks": [{"name": "T1", "priority": "high"}]},
                "sid", mock_tg, tool="proxy_ticket", caller_agent_id="agent-1",
            )

        assert result["error"]["code"] == "infrastructure_error"
        assert result["error"]["code"] != "action_failed"

    def test_assign_to_not_found_still_returns_agent_not_found_error(self, mock_tg):
        """get_or_load returning None (not raising) → agent_not_found error."""
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=None,
        ):
            arguments = {
                "tasks": [
                    {"name": "T1", "priority": "high", "assign_to": "missing-agent"},
                ]
            }
            result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result["error"]["code"] == "assign_to_agent_not_found"
        mock_tg.create_tasks_batch.assert_not_called()

    def test_happy_path_with_valid_assign_to_still_creates_task(self, mock_tg):
        """No exception from registry → task is created (regression guard)."""
        fake_target = MagicMock()
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
            return_value=fake_target,
        ):
            arguments = {
                "tasks": [
                    {"name": "T1", "priority": "high", "assign_to": "real-agent"},
                ]
            }
            result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_tg.create_tasks_batch.assert_called_once()
        call_args = mock_tg.create_tasks_batch.call_args[0][0]
        assert call_args[0]["assign_to"] == "real-agent"

    def test_none_assign_to_skips_registry_lookup(self, mock_tg):
        """None assign_to must not call the registry (regression guard)."""
        with patch(
            "proxy.agent.agent_registry.registry.get_or_load",
        ) as mock_lookup:
            arguments = {
                "tasks": [
                    {"name": "T1", "priority": "high", "assign_to": None},
                ]
            }
            result = CreateTasksAction.execute(arguments, "sid", mock_tg)

        assert result.get("status") != "error"
        mock_lookup.assert_not_called()
        mock_tg.create_tasks_batch.assert_called_once()
        call_args = mock_tg.create_tasks_batch.call_args[0][0]
        assert call_args[0]["assign_to"] is None
