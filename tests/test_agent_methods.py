"""Tests for Agent skill-mode methods (Batch A - A1).

Phase 3: ``set_ticket_mode`` was removed from ``Agent`` (zero production
callers). Tests now exercise ``set_active_mode`` with a ticket ``ModeState``
directly instead of going through the legacy wrapper.
"""
import time
import pytest
from unittest.mock import MagicMock, patch


class TestSetActiveMode:
    """Test agent.set_active_mode() method (ticket ModeState path)."""

    def test_set_active_mode_ticket_under_lock(self):
        """set_active_mode(ticket_mode_state) installs a ticket ModeState."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-session-001")
        mode_state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset({"read", "glob", "grep", "search", "proxy_ticket"}),
            skill_state={},
        )
        agent.set_active_mode(mode_state)
        assert agent.active_mode is mode_state
        assert agent.active_mode.name == "ticket"
        assert agent.active_mode.ttl_seconds == 300

    def test_set_active_mode_non_ticket_skill(self):
        """set_active_mode should set active_mode under lock."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-session-002")
        mode_state = ModeState(
            name="test-skill",
            activated_at=0.0,
            ttl_seconds=60,
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode_state)
        assert agent.active_mode is mode_state


class TestClearTicketMode:
    """Test agent.clear_ticket_mode() method."""

    def test_clear_ticket_mode_clears_mode(self):
        """clear_ticket_mode should clear active_mode."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-session-003")
        mode_state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset({"read", "glob", "grep", "search", "proxy_ticket"}),
            skill_state={},
        )
        agent.set_active_mode(mode_state)
        agent.clear_ticket_mode()
        assert agent.active_mode is None
