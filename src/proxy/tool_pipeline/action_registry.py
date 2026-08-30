"""Action registry for proxy_ticket — replaces the 438-line if/elif chain.

Each action is a small class implementing the TaskAction protocol.
dispatch_action() includes the ticket-mode gate and AmbiguousHashError handling
as defense-in-depth (mirrors proxy_ticket.py:66-72).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from memory.query_executor import ConstraintViolation, InfrastructureError, QueryTimeoutError

from proxy.tool_pipeline.action_envelope import error_envelope

logger = logging.getLogger("proxy.tool_pipeline")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TaskAction(Protocol):
    """Protocol that all task actions must satisfy."""

    def execute(self, arguments: dict, session_id: str, tg: Any, workspace_id: str = None, caller_agent_id: str = None) -> dict:
        """Execute the action. Returns a result dict with session_id."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ACTION_REGISTRY: dict[str, TaskAction] = {}


def register_action(name: str, handler: TaskAction) -> None:
    """Register a TaskAction handler under the given name."""
    ACTION_REGISTRY[name] = handler


def dispatch_action(name: str, arguments: dict, session_id: str, tg: Any, *, tool: str, registry: dict = None, workspace_id: str = None, caller_agent_id: str = None) -> dict:
    """Dispatch an action by name.

    Every return is a canonical envelope. Handlers MUST emit canonical
    envelopes on success (returned unchanged); the 6 exception paths below
    emit canonical envelopes directly (no normalization backstop).

    Args:
        workspace_id: Optional workspace scope for task queries.
        caller_agent_id: Optional agent ID of the caller. Resolved from session_id if not provided.
        tool: REQUIRED keyword-only. The calling tool name (e.g. "proxy_ticket"),
            used to populate the envelope's `tool` field on every code path.
    """
    from memory.task_graph import AmbiguousHashError

    if caller_agent_id is None and session_id:
        from proxy.agent.agent_registry import registry as agent_registry
        agent = agent_registry.get_or_load(session_id)
        if agent is not None:
            caller_agent_id = agent.agent_id

    reg = registry if registry is not None else ACTION_REGISTRY
    handler = reg.get(name)
    if not handler:
        return error_envelope(
            tool, None, "invalid_action",
            f"Unknown action: '{name}'. Valid actions: {', '.join(sorted(reg.keys()))}",
            session_id,
            valid_actions=sorted(reg.keys()),
        )

    # Authorization is already enforced by execute_ticket_tool() at proxy_ticket.py
    try:
        return handler.execute(arguments, session_id, tg, workspace_id=workspace_id, caller_agent_id=caller_agent_id)
    except AmbiguousHashError:
        return error_envelope(
            tool, name, "ambiguous_hash",
            "Multiple tasks match the given hash. Please be more specific.",
            session_id,
        )
    except InfrastructureError:
        logger.exception(f"Action '{name}' failed: infrastructure error")
        return error_envelope(
            tool, name, "infrastructure_error",
            f"Action '{name}' failed: database temporarily unavailable",
            session_id,
        )
    except QueryTimeoutError:
        logger.exception(f"Action '{name}' failed: database query timed out")
        return error_envelope(
            tool, name, "query_timeout",
            f"Action '{name}' failed: database query timed out",
            session_id,
        )
    except ConstraintViolation as e:
        return error_envelope(
            tool, name, "constraint_violation", str(e), session_id,
        )
    except Exception as e:
        logger.exception(f"Action '{name}' failed")
        return error_envelope(
            tool, name, "action_failed",
            f"Action '{name}' failed: internal error",
            session_id,
        )
