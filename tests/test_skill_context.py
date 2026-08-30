"""Tests for SkillContext factory (Batch C - C1, API-003/008/009)."""
import pytest
from unittest.mock import MagicMock


class TestFromRequestContext:
    """API-003: SkillContext.from_request_context factory method."""

    def test_from_request_context_populates_all_fields(self):
        """Factory populates all SkillContext fields from RequestContext."""
        from proxy.skills.skill_registry import SkillContext

        ctx = MagicMock()
        ctx.session_id = "test-session-123"
        ctx.is_subagent = True
        ctx.parent_agent_id = "parent-agent-456"
        ctx.inherited_skill = "ticket"
        ctx.working_directory = "/tmp/test"
        ctx.messages = [{"role": "user", "content": "hello"}]
        ctx.seeded_tasks = [{"description": "test task"}]
        ctx.active_workspace = "test-workspace"

        skill_ctx = SkillContext.from_request_context(ctx)

        assert skill_ctx.session_id == "test-session-123"
        assert skill_ctx.is_subagent is True
        assert skill_ctx.parent_agent_id == "parent-agent-456"
        assert skill_ctx.active_skill_name == "ticket"
        assert skill_ctx.working_directory == "/tmp/test"
        assert len(skill_ctx.messages) == 1
        assert len(skill_ctx.seeded_tasks) == 1
        assert skill_ctx.active_workspace == "test-workspace"

    def test_from_request_context_works_with_minimal_ctx(self):
        """Factory works with minimal RequestContext (only session_id)."""
        from proxy.skills.skill_registry import SkillContext

        ctx = MagicMock()
        ctx.session_id = "minimal-session"
        # All other attributes return MagicMock (truthy) via getattr
        ctx.is_subagent = False
        del ctx.messages  # Attribute doesn't exist
        ctx.seeded_tasks = []

        skill_ctx = SkillContext.from_request_context(ctx)

        assert skill_ctx.session_id == "minimal-session"
        assert skill_ctx.is_subagent is False

    def test_from_request_context_accepts_skill_state_override(self):
        """Factory accepts skill_state parameter to override defaults."""
        from proxy.skills.skill_registry import SkillContext

        ctx = MagicMock()
        ctx.session_id = "test-session"
        ctx.messages = []
        ctx.seeded_tasks = []

        skill_ctx = SkillContext.from_request_context(
            ctx,
            skill_state={"action": "inline", "arg": ""},
        )

        assert skill_ctx.skill_state == {"action": "inline", "arg": ""}

    def test_from_request_context_accepts_raw_user_text(self):
        """Factory accepts raw_user_text parameter."""
        from proxy.skills.skill_registry import SkillContext

        ctx = MagicMock()
        ctx.session_id = "test-session"
        ctx.messages = []
        ctx.seeded_tasks = []

        skill_ctx = SkillContext.from_request_context(
            ctx,
            raw_user_text="hello world",
        )

        assert skill_ctx.raw_user_text == "hello world"
