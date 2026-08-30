"""Unit tests for SkillContext class (Phase 1)."""

import unittest
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillContext:
    """Request-scoped state passed to all skill methods.

    This is the Phase 1 implementation for testing purposes.
    """

    session_id: str
    agent_id: str
    messages: list = field(default_factory=list)
    is_subagent: bool = False
    parent_agent_id: Optional[str] = None
    active_skill_name: Optional[str] = None
    skill_state: dict = field(default_factory=dict)
    working_directory: str = ""
    raw_user_text: str = ""
    seeded_tasks: list = field(default_factory=list)
    active_workspace: Optional[str] = None
    # Phase 1 additional fields
    workspace_name: str = ""
    user: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    request_id: Optional[str] = None

    @classmethod
    def from_request_context(cls, ctx: "RequestContext", agent: "Agent") -> "SkillContext":
        """Create SkillContext from a RequestContext and Agent.

        Shallow-copies messages (list copy, not deepcopy) for safety.
        """
        return cls(
            session_id=ctx.session_id,
            agent_id=agent.agent_id,
            messages=list(ctx.messages) if ctx.messages else [],
            workspace_name=getattr(ctx, "workspace_name", ""),
            user=getattr(ctx, "user", {}),
            metadata=getattr(ctx, "metadata", {}),
            request_id=getattr(ctx, "request_id", None),
            is_subagent=getattr(ctx, "is_subagent", False),
            skill_state=dict(getattr(ctx, "skill_state", {}) or {}),
            seeded_tasks=list(getattr(ctx, "seeded_tasks", []) or []),
        )


@dataclass
class RequestContext:
    """Minimal stub for testing - mimics the interface used by from_request_context."""

    session_id: str
    workspace_name: str = ""
    user: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    request_id: Optional[str] = None
    messages: list = field(default_factory=list)
    is_subagent: bool = False
    skill_state: dict = field(default_factory=dict)
    seeded_tasks: list = field(default_factory=list)


@dataclass
class Agent:
    """Minimal stub for testing."""

    agent_id: str


class MinimalSkill:
    """A minimal skill for testing skill_state propagation."""

    def __init__(self):
        self.captured_state = None

    def on_activate(self, ctx: SkillContext) -> None:
        self.captured_state = ctx.skill_state.get("foo")


class TestSkillContext(unittest.TestCase):
    """Tests for SkillContext."""

    def test_from_request_context(self):
        """Construct a minimal RequestContext stub (with workspace_name, user, metadata
        as simple dicts), pass it to SkillContext.from_request_context(), verify the
        returned ctx has correct workspace_name, user, and metadata.
        """
        req_ctx = RequestContext(
            session_id="test-session-123",
            workspace_name="my-workspace",
            user={"name": "testuser", "email": "test@example.com"},
            metadata={"source": "unit-test"},
            messages=[{"role": "user", "content": "Hello"}],
        )
        agent = Agent(agent_id="test-agent-1")

        ctx = SkillContext.from_request_context(req_ctx, agent)

        self.assertEqual(ctx.workspace_name, "my-workspace")
        self.assertEqual(ctx.user, {"name": "testuser", "email": "test@example.com"})
        self.assertEqual(ctx.metadata, {"source": "unit-test"})

    def test_skill_state_starts_empty(self):
        """New SkillContext has empty dict for skill_state."""
        ctx = SkillContext(
            session_id="test-session",
            agent_id="test-agent",
        )

        self.assertIsInstance(ctx.skill_state, dict)
        self.assertEqual(len(ctx.skill_state), 0)

    def test_skill_state_mutable(self):
        """skill_state dict can be assigned and persists."""
        ctx = SkillContext(
            session_id="test-session",
            agent_id="test-agent",
        )

        ctx.skill_state["foo"] = "bar"
        ctx.skill_state["count"] = 42

        self.assertEqual(ctx.skill_state["foo"], "bar")
        self.assertEqual(ctx.skill_state["count"], 42)
        self.assertEqual(len(ctx.skill_state), 2)

    def test_skill_state_propagates_to_skill(self):
        """Simulate the lifecycle: main request sets ctx.skill_state["foo"] = "bar",
        skill's on_activate reads it, verify the value is present.
        """
        req_ctx = RequestContext(
            session_id="test-session",
            messages=[{"role": "user", "content": "test"}],
        )
        agent = Agent(agent_id="test-agent")

        # Simulate main request setting skill_state
        ctx = SkillContext.from_request_context(req_ctx, agent)
        ctx.skill_state["foo"] = "bar"

        # Simulate skill's on_activate reading the state
        skill = MinimalSkill()
        skill.on_activate(ctx)

        self.assertEqual(skill.captured_state, "bar")

    def test_from_request_context_inherits_request_id(self):
        """If the stub RequestContext has a request_id, the SkillContext should expose it."""
        req_ctx = RequestContext(
            session_id="test-session",
            request_id="req-abc-123",
            messages=[],
        )
        agent = Agent(agent_id="test-agent")

        ctx = SkillContext.from_request_context(req_ctx, agent)

        self.assertEqual(ctx.request_id, "req-abc-123")

    def test_from_request_context_missing_fields(self):
        """If the stub RequestContext is missing optional fields (no metadata),
        it should not raise — use defaults.
        """
        req_ctx = RequestContext(
            session_id="test-session",
            messages=[{"role": "user", "content": "test"}],
        )
        agent = Agent(agent_id="test-agent")

        # Should not raise
        ctx = SkillContext.from_request_context(req_ctx, agent)

        self.assertEqual(ctx.session_id, "test-session")
        self.assertEqual(ctx.workspace_name, "")
        self.assertEqual(ctx.user, {})
        self.assertEqual(ctx.metadata, {})
        self.assertIsNone(ctx.request_id)


if __name__ == "__main__":
    unittest.main()
