"""Tests for TTL enforcement (Batch B - B1, SEC-001)."""
import pytest
import time
from unittest.mock import MagicMock


class TestIsTicketModeTTL:
    """Test that is_ticket_mode respects TTL expiry."""

    def test_is_ticket_mode_false_when_mode_none(self):
        """is_ticket_mode returns False when active_mode is None."""
        from proxy.agent.agent import Agent
        agent = Agent(session_id="test-ttl-001")
        assert agent.is_ticket_mode is False

    def test_is_ticket_mode_false_after_ttl_expires(self):
        """is_ticket_mode returns False after TTL expires."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-ttl-002")
        mode = ModeState(
            name="ticket",
            activated_at=time.time() - 400,  # 400s ago
            ttl_seconds=300,  # 300s TTL — expired
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode)
        assert agent.is_ticket_mode is False

    def test_is_ticket_mode_true_before_ttl_expires(self):
        """is_ticket_mode returns True before TTL expires."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-ttl-003")
        mode = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode)
        assert agent.is_ticket_mode is True

    def test_is_ticket_mode_false_for_non_ticket_skill(self):
        """is_ticket_mode returns False for non-ticket skill."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-ttl-004")
        mode = ModeState(
            name="workspace",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode)
        assert agent.is_ticket_mode is False


class TestIsSkillModeTTL:
    """Test that is_skill_mode respects TTL expiry for any skill."""

    def test_is_skill_mode_false_when_mode_none(self):
        """is_skill_mode returns False when active_mode is None."""
        from proxy.agent.agent import Agent
        agent = Agent(session_id="test-skill-001")
        assert agent.is_skill_mode is False

    def test_is_skill_mode_false_after_ttl_expires(self):
        """is_skill_mode returns False after TTL expires."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-skill-002")
        mode = ModeState(
            name="workspace",
            activated_at=time.time() - 400,
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode)
        assert agent.is_skill_mode is False

    def test_is_skill_mode_true_before_ttl_expires(self):
        """is_skill_mode returns True before TTL expires."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        agent = Agent(session_id="test-skill-003")
        mode = ModeState(
            name="workspace",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode)
        assert agent.is_skill_mode is True
