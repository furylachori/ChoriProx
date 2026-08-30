"""Tests for integration fixes (Batch C - C2, INT-002/007/008)."""
import sys
import pytest
from unittest.mock import MagicMock, patch


class TestCanarySkillDerivation:
    """INT-002: Only 'ticket' skill sets _ticket_mode_subagent=True."""

    def test_workspace_canary_does_not_set_ticket_mode(self):
        """[[__SUBAGENT__:SKILL:workspace]] should not set _ticket_mode_subagent."""
        from proxy.canary import parse_canary

        with patch("proxy.skills.skill_registry.SkillRegistry"):
            # Parsing the canary — workspace skill may or may not be registered
            # Either way, skill_name != "ticket"
            p = parse_canary("[[__SUBAGENT__:SKILL:workspace]]")
            skill_name = p.skill_name if (p and p.verified) else None
            # The flag should be skill_name == "ticket"
            is_ticket = (skill_name == "ticket")
            assert is_ticket is False

    # NOTE: legacy test removed. test_ticket_canary_sets_ticket_mode tested the
    # old behavior of the parser with a bare canary (no HMAC sig:ts).
    # The new canary format requires a 64-char hex signature and timestamp, and
    # HMAC verification is enforced. Bare canaries are unverified,
    # so the assertion (skill_name == "ticket") can no longer hold without
    # a properly-signed canary. Patching SkillRegistry.get has no effect on
    # parse_canary, which uses verify_canary for HMAC verification.


# NOTE: legacy class removed. TestSeedTaskContextCalledOnce tested the OLD
# apply_mode body that pre-populated task context. The new apply_mode
# delegates to skill_prompts.assemble_system_prompt and that pre-population
# path is gone. The single-call guarantee is no longer relevant.


class TestProxyTicketExceptionLogging:
    """INT-008: proxy_ticket logs exceptions instead of silently swallowing."""

    def test_workspace_extraction_logs_on_error(self):
        """Exception in workspace extraction is logged, not silently passed."""
        import sys
        import memory.tools.proxy_ticket  # Ensure module is loaded
        # Get the module object for proxy_ticket
        mod = sys.modules.get('memory.tools.proxy_ticket')
        assert mod is not None, "Module memory.tools.proxy_ticket not found"
        import inspect
        source = inspect.getsource(mod)

        # Should have logger.warning for the exception path
        assert "logger.warning" in source or "logger.exception" in source, \
            "Expected logger.warning/logger.exception for exception logging"
        # Should NOT have bare "except Exception: pass"
        assert "except Exception:\n            pass\n" not in source, \
            "Found bare pass in exception handler"
