"""Mode harness — helpers for setting up Agent and ModeState in tests.

Provides factory functions for creating test-ready Agents, activating
skills (ticket/work), and building ModeState instances with controllable
clocks to eliminate time-dependent flakiness.
"""

import time
from unittest.mock import MagicMock

from proxy.agent.agent import Agent
from proxy.skills.skill_definition import ModeState


def create_test_agent(session_id="test-session", agent_id="agent-1"):
    """Create a fully-initialized Agent for testing.

    Returns an Agent instance. Does NOT touch the global AgentRegistry —
    the agent lives only in the caller's scope.
    """
    agent = Agent(session_id=session_id, agent_id=agent_id)
    return agent


def activate_ticket_mode(agent, subcommand="plan", task_hash=None, ttl=300):
    """Set up ticket mode on an agent.

    Installs a ModeState with name='ticket' and the given TTL.
    skill_state captures the subcommand and optional task_hash.
    """
    mode = ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=ttl,
        allowed_tool_names=frozenset([
            "read", "glob", "grep", "search", "proxy_ticket", "proxy_work",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
        ]),
        skill_state={
            "subcommand": subcommand,
            "task_hash": task_hash,
        },
    )
    agent.set_active_mode(mode)
    return mode


def activate_work_mode(agent, task_hash="abc123", ttl=300):
    """Set up work mode on an agent.

    Installs a ModeState with name='work' and the given TTL.
    """
    mode = ModeState(
        name="work",
        activated_at=time.time(),
        ttl_seconds=ttl,
        allowed_tool_names=frozenset([
            "read", "write", "edit", "bash", "glob", "grep", "search",
            "task", "proxy_work", "proxy_ticket",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
        ]),
        skill_state={"task_hash": task_hash},
    )
    agent.set_active_mode(mode)
    return mode


def create_test_mode_state(name, ttl=300, activated_at=None, **kwargs):
    """Create an ActiveMode (ModeState) with controllable clock.

    If activated_at is None, uses time.time() — pass a fixed float
    to make the mode deterministic (e.g., for expiry tests).
    """
    if activated_at is None:
        activated_at = time.time()
    defaults = {
        "name": name,
        "activated_at": activated_at,
        "ttl_seconds": ttl,
        "allowed_tool_names": None,
        "skill_state": {},
    }
    defaults.update(kwargs)
    return ModeState(**defaults)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_create_test_agent():
    agent = create_test_agent()
    assert agent.session_id == "test-session"
    assert agent.agent_id == "agent-1"
    assert agent.snapshot_mode() is None


def test_create_test_agent_default_ids():
    agent = create_test_agent()
    assert agent.session_id == "test-session"
    assert agent.agent_id == "agent-1"


def test_activate_ticket_mode():
    agent = create_test_agent()
    mode = activate_ticket_mode(agent, subcommand="plan", task_hash="abc123")
    assert mode.name == "ticket"
    assert agent.is_ticket_mode
    snap = agent.snapshot_mode()
    assert snap is not None
    assert snap.skill_state["subcommand"] == "plan"
    assert snap.skill_state["task_hash"] == "abc123"


def test_activate_work_mode():
    agent = create_test_agent()
    mode = activate_work_mode(agent, task_hash="deadbeef")
    assert mode.name == "work"
    assert agent.is_skill_mode
    snap = agent.snapshot_mode()
    assert snap.skill_state["task_hash"] == "deadbeef"


def test_create_test_mode_state_fixed_clock():
    fixed = 1700000000.0
    mode = create_test_mode_state("ticket", ttl=300, activated_at=fixed)
    assert mode.name == "ticket"
    assert mode.activated_at == fixed
    assert mode.ttl_seconds == 300


def test_create_test_mode_state_persistent():
    mode = create_test_mode_state("workspace", ttl=0)
    assert not mode.is_expired()


def test_ticket_mode_expiry():
    """Mode activated 600s ago with ttl=300 should be expired."""
    fixed = 1700000000.0
    mode = create_test_mode_state("ticket", ttl=300, activated_at=fixed - 600)
    assert mode.is_expired()


def test_clear_active_mode():
    agent = create_test_agent()
    activate_ticket_mode(agent)
    assert agent.is_ticket_mode
    agent.clear_active_mode()
    assert not agent.is_ticket_mode
    assert agent.snapshot_mode() is None
