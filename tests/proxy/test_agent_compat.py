"""Tests for Agent backward compatibility (Phase B skill system).
Tests against REAL Agent — not a stub.

Phase 3: ``Agent.set_ticket_mode`` was removed (zero production callers —
see docs/specs/consolidation-research/18-phase-3-final.md §H1). Tests now
build a ticket ``ModeState`` and pass it to ``set_active_mode`` directly.
"""
import pytest
import time
import threading


def _ticket_mode_state():
    """Build a canonical ticket ModeState matching the deleted wrapper's payload."""
    from proxy.skills.skill_definition import ModeState
    return ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=300,
        allowed_tool_names=frozenset({"read", "glob", "grep", "search", "proxy_ticket"}),
        skill_state={},
    )


class TestAgentModeMethods:
    """Test Agent.is_ticket_mode, set_active_mode, clear_active_mode, is_skill_mode."""

    def setup_method(self):
        from proxy.agent.agent import Agent
        self.Agent = Agent

    def test_is_ticket_mode_false_by_default(self):
        """is_ticket_mode returns False when no mode is active."""
        agent = self.Agent(session_id="compat-test-1")
        assert agent.is_ticket_mode is False

    def test_is_ticket_mode_true_after_set_active_mode_ticket(self):
        """set_active_mode(ticket_state) makes is_ticket_mode return True."""
        agent = self.Agent(session_id="compat-test-2")
        agent.set_active_mode(_ticket_mode_state())
        try:
            assert agent.is_ticket_mode is True
        finally:
            agent.clear_active_mode()

    def test_is_ticket_mode_false_after_clear_active_mode(self):
        """clear_active_mode resets is_ticket_mode to False."""
        agent = self.Agent(session_id="compat-test-3")
        agent.set_active_mode(_ticket_mode_state())
        agent.clear_active_mode()
        assert agent.is_ticket_mode is False

    def test_set_active_mode_ticket_creates_correct_mode_state(self):
        """set_active_mode(ticket_state) installs name='ticket', TTL=300."""
        agent = self.Agent(session_id="compat-test-4")
        ms = _ticket_mode_state()
        agent.set_active_mode(ms)
        try:
            assert agent.active_mode is ms
            assert agent.active_mode.name == "ticket"
            assert agent.active_mode.ttl_seconds == 300
        finally:
            agent.clear_active_mode()

    def test_is_skill_mode_true_for_ticket_skill(self):
        """is_skill_mode returns True for any active skill."""
        agent = self.Agent(session_id="compat-test-5")
        agent.set_active_mode(_ticket_mode_state())
        try:
            assert agent.is_skill_mode is True
        finally:
            agent.clear_active_mode()

    def test_is_skill_mode_false_when_no_mode(self):
        """is_skill_mode returns False when no mode is active."""
        agent = self.Agent(session_id="compat-test-6")
        assert agent.is_skill_mode is False


class TestModeStateCleanup:
    """Test that clear_active_mode properly cleans up."""

    def setup_method(self):
        from proxy.agent.agent import Agent
        self.Agent = Agent

    def test_clear_active_mode_sets_none(self):
        """clear_active_mode sets active_mode to None."""
        agent = self.Agent(session_id="compat-cleanup-1")
        agent.set_active_mode(_ticket_mode_state())
        agent.clear_active_mode()
        assert agent.active_mode is None


# ---------------------------------------------------------------------------
# Phase 3 regression tests
# ---------------------------------------------------------------------------


class TestPhase3Deletions:
    """Regression guards for Phase 3 deletions (see
    docs/specs/consolidation-research/18-phase-3-final.md).
    """

    def test_agent_spawn_method_removed(self):
        """Phase 3: AgentRegistry.spawn() was deleted (zero callers).

        Verify the method no longer exists on AgentRegistry. Any future
        re-introduction must be justified and accompanied by a real
        caller (no shims, no deprecation paths — see the §1 / §2
        no-shims policy).
        """
        from proxy.agent.agent_registry import AgentRegistry
        assert not hasattr(AgentRegistry, "spawn"), (
            "AgentRegistry.spawn() should have been deleted in Phase 3 "
            "(zero callers — see 18-phase-3-final.md §H1 / §4.2)"
        )

    def test_agent_set_ticket_mode_removed(self):
        """Phase 3: Agent.set_ticket_mode() was deleted (zero production callers).

        Verify the legacy wrapper no longer exists on Agent. Callers
        must use ``set_active_mode(ModeState(name='ticket', ...))``
        directly.
        """
        from proxy.agent.agent import Agent
        assert not hasattr(Agent, "set_ticket_mode"), (
            "Agent.set_ticket_mode() should have been deleted in Phase 3 "
            "(zero production callers — see 18-phase-3-final.md §H1)"
        )

    def test_memory_tool_definitions_is_disabled_placeholder(self):
        """Phase 3: MEMORY_TOOL_DEFINITIONS is intentionally empty (disabled).

        This is a feature-disabled pattern. Phase 4 will re-populate via
        the tool action registry (see
        docs/specs/consolidation-research/phase-4-tool-action-registry.md).
        The constant is KEPT (not deleted) so existing imports don't
        break and Phase 4 has a home for the schemas.
        """
        from memory.tools import MEMORY_TOOL_DEFINITIONS
        assert MEMORY_TOOL_DEFINITIONS == [], (
            f"MEMORY_TOOL_DEFINITIONS should remain empty (disabled). "
            f"Got: {MEMORY_TOOL_DEFINITIONS!r}"
        )

    def test_memory_tool_definitions_disabled_preserves_consumer_noop(self):
        """Phase 3: with MEMORY_TOOL_DEFINITIONS=[], the consumer loop
        at src/proxy/tool_cache.py:110-118 iterates an empty list and
        adds nothing to augmented_tools. The skill-mode tool list is
        therefore byte-identical to the no-MEMORY_TOOL_DEFINITIONS case.

        This pins the disabled-state contract: populating
        MEMORY_TOOL_DEFINITIONS in Phase 4 is the single switch that
        turns the loop on. Until then, behavior is preserved.
        """
        from memory.tools import MEMORY_TOOL_DEFINITIONS
        # Empty list → no items appended by the consumer loop.
        appended = []
        for canonical_tool in MEMORY_TOOL_DEFINITIONS:
            name = canonical_tool.get("function", {}).get("name", "")
            if name == "search":
                appended.append(canonical_tool)
        assert appended == [], (
            f"Disabled MEMORY_TOOL_DEFINITIONS must produce empty appends. "
            f"Got: {appended!r}"
        )


