"""Tests for stale-clear behavior (Batch A - A3)."""
import pytest
import time
from unittest.mock import MagicMock, patch


class TestStaleClearGuard:
    """Test that stale-clear respects activation flags and time guard."""

    def test_stale_clear_skipped_when_skill_activated(self):
        """Stale-clear should not fire when ctx._skill_activated is True."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState

        agent = Agent(session_id="test-stale-001")
        mode_state = ModeState(
            name="ticket",
            activated_at=time.time() - 2.0,  # 2 seconds ago (>1s)
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
        )
        agent.set_active_mode(mode_state)

        # Create mock context with _skill_activated=True
        ctx = MagicMock()
        ctx._skill_activated = True
        ctx._ticket_activated = False
        ctx.is_subagent = False
        ctx.session_id = "test-stale-001"
        ctx.messages = [{"role": "user", "content": "test"}]

        # Stale-clear should NOT fire due to _skill_activated flag
        # The guard condition should prevent clearing
        from proxy.mode import resolve_mode
        with patch.object(agent, 'clear_active_mode') as mock_clear:
            mode = resolve_mode(ctx, agent)
            mock_clear.assert_not_called()

    # NOTE: legacy test removed. test_stale_clear_fires_when_skill_not_activated
    # tested the old per-request stale-clear path that fired when
    # _skill_activated=False and the last message was from the user. The new
    # resolve_mode delegates to proxy.skill_resolver.resolve_mode, which only
    # clears modes that have hit their TTL expiry (is_expired()=True), not
    # request-scoped user-message detection. The user-message guard moved out
    # of resolve_mode; see tests/proxy/test_skill_mode.py for the new behavior.