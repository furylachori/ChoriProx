"""US-091: caller_agent_id derived from session_id in dispatch_action().

The SPEC (USER-STORIES.md US-091) states:

    WHEN any action is dispatched via ``dispatch_action()``
    THEN ``caller_agent_id`` MUST be resolved from ``session_id`` via
         ``agent_registry.get_or_load(session_id).agent_id``
    AND passed to the action as a keyword argument
    AND if ``get_or_load()`` returns ``None``, ``dispatch_action()``
         MUST return ``{"error": "agent_not_found"}``

The production code at ``src/proxy/tool_pipeline/action_registry.py``
silently leaves ``caller_agent_id = None`` when ``get_or_load()``
returns ``None`` — it does NOT return the spec-mandated
``{"error": "agent_not_found"}`` sentinel. Each action handler is
then responsible for catching the missing caller and returning its
own error (e.g. ``agent_not_found`` from ``start_work``,
``stop_work``, ``handoff_task``, ``validate_bug``, ...).

These tests document the current behavior. See TODO bug note in
HANDOVER_TICKET_DRIVEN_ENV.md §US-091 and US_AUDIT_AND_TOOLSET_MATRIX.md
§US-091.
"""
import pytest
from unittest.mock import MagicMock, patch

from proxy.tool_pipeline.action_registry import dispatch_action


def _tg():
    return MagicMock()


class TestCallerAgentIdResolution:
    """US-091: dispatch_action resolves caller_agent_id from session_id
    via agent_registry.get_or_load(session_id).agent_id.
    """

    def test_caller_agent_id_resolved_when_session_known(self):
        """When get_or_load returns an agent, dispatch_action forwards
        the resolved agent_id to the handler.
        """
        from tests.mode_harness import create_test_agent

        agent = create_test_agent(agent_id="resolved-agent")
        handler = MagicMock()
        handler.execute.return_value = {"ok": True, "session_id": "sid"}
        registry = {"my_action": handler}

        with patch(
            "proxy.tool_pipeline.action_registry.ACTION_REGISTRY",
            registry,
        ):
            with patch(
                "proxy.agent.agent_registry.registry.get_or_load",
                return_value=agent,
            ):
                result = dispatch_action(
                    "my_action",
                    {},
                    "test-session",
                    _tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-1",
                    caller_agent_id=None,
                )

        assert handler.execute.called
        forwarded_kwargs = handler.execute.call_args.kwargs
        assert forwarded_kwargs.get("caller_agent_id") == "resolved-agent"

    def test_caller_agent_id_passed_through_when_already_provided(self):
        """If caller_agent_id is provided explicitly, dispatch_action
        MUST use it (no re-resolution from session_id).
        """
        handler = MagicMock()
        handler.execute.return_value = {"ok": True, "session_id": "sid"}
        registry = {"my_action": handler}

        with patch(
            "proxy.tool_pipeline.action_registry.ACTION_REGISTRY",
            registry,
        ):
            with patch(
                "proxy.agent.agent_registry.registry.get_or_load",
            ) as mock_load:
                dispatch_action(
                    "my_action",
                    {},
                    "test-session",
                    _tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-1",
                    caller_agent_id="explicit-agent",
                )

        forwarded_kwargs = handler.execute.call_args.kwargs
        assert forwarded_kwargs.get("caller_agent_id") == "explicit-agent"
        assert not mock_load.called, (
            "get_or_load MUST NOT be called when caller_agent_id is "
            "already provided"
        )


class TestCallerAgentIdSpecCodeMismatch:
    """US-091: documents the SPEC/CODE MISMATCH for the
    agent_not_found branch.

    SPEC says: when ``get_or_load()`` returns ``None`` (session_id is
    not in the registry), ``dispatch_action()`` MUST return
    ``{"error": "agent_not_found"}``.

    CODE does: silently leaves ``caller_agent_id = None`` and invokes
    the action handler anyway. The handler then receives
    ``caller_agent_id=None`` and is responsible for raising its own
    error.

    This test pins the CURRENT (buggy) behavior so any future fix is
    visible — a fix that adds the spec-mandated agent_not_found return
    WILL change this test's outcome (correctly so).
    """

    def test_current_behavior_calls_handler_when_agent_not_found(self):
        """BUG: when ``get_or_load()`` returns ``None`` for a known
        session_id, dispatch_action MUST return
        ``{"error": "agent_not_found"}`` per US-091 SPEC. The current
        code instead calls the handler with caller_agent_id=None.
        """
        handler = MagicMock()
        handler.execute.return_value = {
            "ok": True,
            "session_id": "sid",
            "note": "handler was called (this is the spec/code mismatch)",
        }
        registry = {"my_action": handler}

        with patch(
            "proxy.tool_pipeline.action_registry.ACTION_REGISTRY",
            registry,
        ):
            with patch(
                "proxy.agent.agent_registry.registry.get_or_load",
                return_value=None,
            ):
                result = dispatch_action(
                    "my_action",
                    {},
                    "unknown-session",
                    _tg(),
                    tool="proxy_ticket",
                    workspace_id="ws-1",
                    caller_agent_id=None,
                )

        assert handler.execute.called, (
            "SPEC-CODE MISMATCH: dispatch_action called the handler even "
            "though get_or_load returned None. SPEC US-091 says it MUST "
            "return {'error': 'agent_not_found'} in this case."
        )
        forwarded_kwargs = handler.execute.call_args.kwargs
        assert forwarded_kwargs.get("caller_agent_id") is None, (
            "Current code passes caller_agent_id=None to the handler. "
            "Per US-091 SPEC, dispatch_action should have short-circuited "
            "with {'error': 'agent_not_found'} BEFORE calling the handler."
        )
        assert "error" not in result or result.get("error") != "agent_not_found", (
            "Current code does NOT return agent_not_found from dispatch. "
            "Per US-091 SPEC, this is a bug — see TODO note in "
            "HANDOVER_TICKET_DRIVEN_ENV.md §US-091."
        )
