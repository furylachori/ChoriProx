"""Tests for tool guard and canary validation (Batch B - B3, SEC-005/SEC-006)."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import threading


class TestEmptyAllowlistBlocksAll:
    """SEC-005: Empty frozenset means no tools allowed."""

    def test_empty_frozenset_blocks_all_tools(self):
        """Empty allowed_tool_names blocks all tools."""
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        import time
        
        agent = Agent(session_id="test-guard-001")
        mode = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),  # Empty — no tools allowed
        )
        agent.set_active_mode(mode)
        
        # Test the guard directly using MemoryToolGuard
        from proxy.tool_pipeline.tool_guard import MemoryToolGuard
        from proxy.tool_pipeline.tool_call import ToolCall
        
        guard = MemoryToolGuard()
        tool_call = ToolCall(
            id="test-001",
            name="task",
            arguments={"prompt": "test"}
        )
        
        # Create mock context with session_id
        ctx = MagicMock()
        ctx.session_id = "test-guard-001"
        
        # Patch the registry to return our agent
        with patch('proxy.agent.agent_registry.registry') as mock_reg:
            mock_reg.get_or_load.return_value = agent
            
            result = guard._check_mode_allowlist(tool_call, ctx)
            # Empty frozenset should block (result is error message)
            assert result is not None, "Empty frozenset should block all tools"
            assert "not allowed" in result


class TestUnregisteredSkillCanaryRejected:
    """SEC-006: Canary with unregistered skill name is rejected."""

    def test_unregistered_skill_canary_returns_none(self):
        """Canary for unregistered skill is rejected."""
        from proxy.canary import parse_canary

        # "ghost_skill" is not registered (and this has no sig:ts anyway)
        p = parse_canary("[[__SUBAGENT__:SKILL:ghost_skill]]")
        # Should be unverified with no skill_name
        assert p is None or not p.verified
        assert p is None or p.skill_name is None

    # NOTE: legacy test removed. test_registered_skill_canary_parses tested the
    # old behavior of the parser with a bare canary (no HMAC sig:ts).
    # The new canary format requires a 64-char hex signature and timestamp, and
    # HMAC verification is enforced. Bare canaries are unverified.
    # A proper "registered skill parses" test would need to use a valid signed
    # canary and patch verify_canary; that is out of scope for the legacy
    # fixup pass. See tests/proxy/test_skill_mode.py and test_skill_guard.py
    # for the new behavior coverage.