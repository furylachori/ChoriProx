"""
Unit tests for apply_mode() basic behavior.

Tests apply_mode message handling and working-dir preservation.
"""

import pytest
from unittest.mock import patch, MagicMock

from proxy.mode import apply_mode, Mode


# ============================================================================
# Additional helper tests
# ============================================================================

class TestApplyModeBasic:
    """Basic tests for apply_mode function."""

    def _make_ctx(self, session_id="test-session", seeded_tasks=None):
        ctx = MagicMock()
        ctx.session_id = session_id
        ctx.seeded_tasks = seeded_tasks
        return ctx

    def _make_agent(self):
        agent = MagicMock()
        return agent

    def test_apply_mode_empty_messages(self):
        """Verify apply_mode handles empty messages."""
        body = {"messages": []}

        ctx = self._make_ctx()
        agent = self._make_agent()

        result = apply_mode(ctx, agent, body, Mode.DEFAULT)
        assert "messages" in result

    # NOTE: legacy test removed. The new apply_mode delegates to
    # skill_prompts.assemble_system_prompt which requires body["messages"];
    # the old "no messages key" tolerance no longer applies.

    def test_apply_mode_preserves_working_dir(self):
        """Verify apply_mode preserves working directory from original prompt."""
        body = {
            "messages": [
                {"role": "system", "content": "Working directory: /some/path\n\nSystem prompt"},
                {"role": "user", "content": "Hello"}
            ]
        }

        ctx = self._make_ctx()
        agent = self._make_agent()

        result = apply_mode(ctx, agent, body, Mode.DEFAULT)

        system_content = result["messages"][0]["content"]
        assert "/some/path" in system_content or "working" in system_content.lower()
