"""Tests for the skill dispatch pipeline (try_skill_activation).

Verifies routing: parse_subcommand → handle_direct → on_activate,
error boundaries, and streaming paths.
"""
import logging
import time
import pytest
from unittest.mock import MagicMock, patch

from proxy.skills.skill_definition import BaseSkill
from proxy.skills.skill_registry import (
    SkillContext,
    SkillRegistry,
    SkillActivation,
    SkillMeta,
    reset_default_registry,
)
from proxy.pipeline.dispatch import try_skill_activation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(text: str, session_id: str = "test-sess", wants_stream: bool = False):
    """Build a minimal mock RequestContext for dispatch tests."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.wants_stream = wants_stream
    ctx.messages = [{"role": "user", "content": text}]
    ctx.is_subagent = False
    ctx.parent_agent_id = None
    ctx.inherited_skill = None
    ctx.working_directory = "/tmp"
    ctx.seeded_tasks = []
    ctx.active_workspace = None
    return ctx


def _make_features():
    """Build a minimal features object for dispatch tests."""
    return MagicMock()


def _reset_registry_singleton():
    """Reset the default registry so each test starts clean."""
    reset_default_registry()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def flask_app_context():
    """All dispatch tests MUST run inside a Flask application context."""
    from qwen_proxy import app
    with app.app_context():
        yield


@pytest.fixture
def dispatch_setup():
    """Shared setup for dispatch tests.

    Provides a clean SkillRegistry, features, and resets the registry
    before and after each test.
    """
    _reset_registry_singleton()
    from proxy.state import skill_rate_limiter
    skill_rate_limiter.reset()
    registry = SkillRegistry()
    registry.clear_all()
    features = _make_features()
    yield registry, features
    skill_rate_limiter.reset()
    _reset_registry_singleton()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHandleDirectRouting:
    """Verify handle_direct is called for all non-None actions."""

    def test_all_actions_call_handle_direct_first(self, dispatch_setup, caplog):
        caplog.set_level(logging.INFO)
        registry, features = dispatch_setup
        handle_called = []
        activate_called = []

        class TestSkill(BaseSkill):
            name = "test"
            trigger = "^^test"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("some_action", "some_arg")

            def handle_direct(self, ctx, action, arg):
                handle_called.append((action, arg))
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

            def on_activate(self, ctx):
                activate_called.append(True)

            def get_activation_rewrite(self, ctx):
                return "rewrite"

            def get_meta(self):
                return SkillMeta(name="test", trigger="^^test")

        registry.register(TestSkill())
        ctx = _make_ctx("^^test something")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert len(handle_called) == 1
        assert handle_called[0] == ("some_action", "some_arg")
        assert len(activate_called) == 0
        assert result.handled is True
        assert "[SKILL_MATCH]" in caplog.text

    def test_handle_direct_returns_dict_consumed(self, dispatch_setup, caplog):
        """When handle_direct returns a dict, the response is consumed."""
        caplog.set_level(logging.INFO)
        registry, features = dispatch_setup
        activate_called = []
        rewrite_called = []

        class TestSkill(BaseSkill):
            name = "test"
            trigger = "^^test"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("status", "")

            def handle_direct(self, ctx, action, arg):
                return {
                    "choices": [{"message": {"role": "assistant", "content": "direct response"}}],
                    "model": "skill",
                }

            def on_activate(self, ctx):
                activate_called.append(True)

            def get_activation_rewrite(self, ctx):
                rewrite_called.append(True)
                return "should not be used"

            def get_meta(self):
                return SkillMeta(name="test", trigger="^^test")

        registry.register(TestSkill())
        ctx = _make_ctx("^^test")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.rewritten_text is None
        response = result.response
        assert response["choices"][0]["message"]["content"] == "direct response"
        assert len(activate_called) == 0
        assert len(rewrite_called) == 0
        assert "[SKILL_HANDLED]" in caplog.text

    def test_handle_direct_returns_dict_streaming(self, dispatch_setup):
        """Streaming path: handle_direct returns dict → dispatch hands a plain
        dict back to the route; SSE framing is owned by app:_keepalive_stream
        (the dispatch path no longer builds a lazy Flask SSE Response — see
        ticket 199ece93 §4 for the threading rationale).
        """
        registry, features = dispatch_setup

        class TestSkill(BaseSkill):
            name = "test"
            trigger = "^^test"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("status", "")

            def handle_direct(self, ctx, action, arg):
                return {
                    "choices": [{"message": {"role": "assistant", "content": "streamed response"}}],
                    "model": "skill",
                }

            def get_meta(self):
                return SkillMeta(name="test", trigger="^^test")

        registry.register(TestSkill())
        ctx = _make_ctx("^^test", wants_stream=True)

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.response is not None
        # Plain dict back to the route; NOT a Flask SSE Response — the new
        # contract hands SSE framing to proxy.app:_keepalive_stream.
        assert isinstance(result.response, dict), (
            f"streaming dispatch must hand a plain dict to the route; "
            f"got {type(result.response).__name__}"
        )
        assert "choices" in result.response, (
            f"plain dict should preserve the skill response shape; "
            f"got {result.response!r}"
        )


class TestSubcommandRouting:
    """Verify specific subcommand routing patterns."""

    def test_hash_show_calls_handle_direct_with_hash(self, dispatch_setup):
        """^^ticket abc123 → handle_direct("show", "abc123")."""
        registry, features = dispatch_setup
        import re
        received_action = None
        received_arg = None

        class TestSkill(BaseSkill):
            name = "ticket"
            trigger = "^^ticket"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                stripped = text[len("^^ticket"):].strip()
                if re.fullmatch(r"[0-9a-fA-F]{6,12}", stripped):
                    return ("show", stripped)
                return (None, "")

            def handle_direct(self, ctx, action, arg):
                nonlocal received_action, received_arg
                received_action = action
                received_arg = arg
                return {"choices": [{"message": {"role": "assistant", "content": "task info"}}]}

            def get_meta(self):
                return SkillMeta(name="ticket", trigger="^^ticket")

        registry.register(TestSkill())
        ctx = _make_ctx("^^ticket abc123")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert received_action == "show"
        assert received_arg == "abc123"
        assert result.handled is True

    def test_batch_done_calls_handle_direct(self, dispatch_setup):
        """^^ticket done abc, def → handle_direct("done", "abc, def")."""
        registry, features = dispatch_setup
        received_action = None
        received_arg = None

        class TestSkill(BaseSkill):
            name = "ticket"
            trigger = "^^ticket"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                stripped = text[len("^^ticket"):].strip()
                if stripped.lower().startswith("done "):
                    return ("done", stripped[5:].strip())
                return (None, "")

            def handle_direct(self, ctx, action, arg):
                nonlocal received_action, received_arg
                received_action = action
                received_arg = arg
                return {"choices": [{"message": {"role": "assistant", "content": "batch done"}}]}

            def get_meta(self):
                return SkillMeta(name="ticket", trigger="^^ticket")

        registry.register(TestSkill())
        ctx = _make_ctx("^^ticket done abc, def")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert received_action == "done"
        assert received_arg == "abc, def"
        assert result.handled is True


class TestWorkspaceRouting:
    """Verify workspace-specific routing patterns."""

    def test_workspace_status_calls_handle_direct(self, dispatch_setup):
        """^^workspace → handle_direct("status", "")."""
        registry, features = dispatch_setup
        received_action = None
        received_arg = None

        class TestSkill(BaseSkill):
            name = "workspace"
            trigger = "^^workspace"
            activation = SkillActivation.COMMAND
            ttl_seconds = 0

            def parse_subcommand(self, text):
                stripped = text[len("^^workspace"):].strip()
                if not stripped:
                    return ("status", "")
                return ("status", "")

            def handle_direct(self, ctx, action, arg):
                nonlocal received_action, received_arg
                received_action = action
                received_arg = arg
                return {"choices": [{"message": {"role": "assistant", "content": "Current workspace: default"}}]}

            def get_meta(self):
                return SkillMeta(name="workspace", trigger="^^workspace")

        registry.register(TestSkill())
        ctx = _make_ctx("^^workspace")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert received_action == "status"
        assert received_arg == ""
        assert result.handled is True


class TestNegativeEdgeCases:
    """Verify dispatch returns None for empty/non-user/malformed messages."""

    def test_empty_messages_returns_none(self, dispatch_setup):
        """Dispatch returns None when messages list is empty."""
        registry, features = dispatch_setup
        from proxy.pipeline.dispatch import run_dispatch
        ctx = MagicMock()
        ctx.messages = []
        result = run_dispatch(ctx, features)
        assert result is None

    def test_system_message_returns_none(self, dispatch_setup):
        """Dispatch returns None for non-user messages."""
        registry, features = dispatch_setup
        from proxy.pipeline.dispatch import run_dispatch
        ctx = MagicMock()
        ctx.messages = [{"role": "system", "content": "You are a helper"}]
        result = run_dispatch(ctx, features)
        assert result is None

    def test_malformed_message_returns_none(self, dispatch_setup):
        """Dispatch returns None for malformed messages."""
        registry, features = dispatch_setup
        from proxy.pipeline.dispatch import run_dispatch
        ctx = MagicMock()
        ctx.messages = [{}]
        result = run_dispatch(ctx, features)
        assert result is None


class TestErrorBoundaries:
    """Verify error boundary behavior in dispatch."""

    def test_handle_direct_exception_returns_500(self, dispatch_setup):
        """Uncaught exception from handle_direct → 500."""
        registry, features = dispatch_setup

        class BrokenSkill(BaseSkill):
            name = "broken"
            trigger = "^^broken"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("action", "arg")

            def handle_direct(self, ctx, action, arg):
                raise RuntimeError("DB connection lost")

            def get_meta(self):
                return SkillMeta(name="broken", trigger="^^broken")

        registry.register(BrokenSkill())
        ctx = _make_ctx("^^broken something")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.status_code == 500
        assert "Command handling failed" in result.response["error"]["message"]

    def test_parse_subcommand_exception_returns_500(self, dispatch_setup, caplog):
        """Uncaught exception from parse_subcommand → 500."""
        registry, features = dispatch_setup

        class BrokenSkill(BaseSkill):
            name = "broken"
            trigger = "^^broken"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                raise ValueError("regex exploded")

            def get_meta(self):
                return SkillMeta(name="broken", trigger="^^broken")

        registry.register(BrokenSkill())
        ctx = _make_ctx("^^broken something")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.status_code == 500
        assert "Skill activation failed" in result.response["error"]["message"]
        assert "parse_subcommand failed" in caplog.text


def test_skill_rate_limit_enforced(caplog):
    """Skill activation rate limit returns 429 when exceeded."""
    from proxy.state import skill_rate_limiter
    from proxy.pipeline.dispatch import try_skill_activation
    from collections import deque
    import proxy.state as state
    state.active_workspaces["rate-test-sess"] = "test-ws"
    ctx = _make_ctx("^^ticket list")
    ctx.session_id = "rate-test-sess"
    # Fill rate limit past the 1000/min cap (raised from 10 in 872964c)
    skill_rate_limiter._entries["rate-test-sess"] = deque([time.time()] * 1001)
    features = MagicMock()
    try:
        result = try_skill_activation(ctx, features)
        assert result is not None
        assert result.status_code == 429
        assert "[SKILL_RATE_LIMIT]" in caplog.text
    finally:
        skill_rate_limiter._entries.pop("rate-test-sess", None)
        state.active_workspaces.pop("rate-test-sess", None)


def test_skill_rate_limit_ip_enforced(caplog):
    """IP-level rate limit returns 429 when exceeded (C-25 mitigation)."""
    from proxy.state import skill_ip_rate_limiter, skill_rate_limiter
    from proxy.pipeline.dispatch import try_skill_activation
    from collections import deque
    import proxy.state as state

    skill_ip_rate_limiter.reset()
    ctx = _make_ctx("^^ticket list")
    ctx.session_id = "ip-rate-test-sess"
    ctx.remote_addr = "10.0.0.99"

    state.active_workspaces["ip-rate-test-sess"] = "test-ws"
    # Fill IP rate limit past the 3000/min cap (raised from 30 in 872964c)
    skill_ip_rate_limiter._entries["10.0.0.99"] = deque([time.time()] * 3001)

    features = MagicMock()
    try:
        result = try_skill_activation(ctx, features)
        assert result is not None
        assert result.status_code == 429
        assert "[SKILL_RATE_LIMIT_IP]" in caplog.text
    finally:
        skill_ip_rate_limiter._entries.pop("10.0.0.99", None)
        skill_rate_limiter._entries.pop("ip-rate-test-sess", None)
        state.active_workspaces.pop("ip-rate-test-sess", None)


def test_ambiguous_hash_returns_409():
    """AmbiguousHashError from handle_direct → dispatch returns 409."""
    from memory.task_graph import AmbiguousHashError
    from proxy.pipeline.dispatch import try_skill_activation
    from proxy.state import skill_rate_limiter

    registry, features = SkillRegistry(), _make_features()
    _reset_registry_singleton()
    skill_rate_limiter.reset()

    class AmbiguousSkill(BaseSkill):
        name = "ticket"
        trigger = "^^ticket"
        activation = SkillActivation.COMMAND
        ttl_seconds = 300

        def parse_subcommand(self, text):
            return ("show", "abc")

        def handle_direct(self, ctx, action, arg):
            raise AmbiguousHashError(3)

        def get_meta(self):
            return SkillMeta(name="ticket", trigger="^^ticket")

    registry.clear_all()
    registry.register(AmbiguousSkill())
    ctx = _make_ctx("^^ticket abc")
    ctx.session_id = "ambig-test-sess"

    with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
        result = try_skill_activation(ctx, features)

    skill_rate_limiter.reset()
    _reset_registry_singleton()

    assert result.handled is True
    assert result.status_code == 409
    payload = result.response
    assert "Ambiguous" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# H-13: Missing branch coverage for dispatch
# ---------------------------------------------------------------------------


def test_agent_not_found_returns_500(dispatch_setup):
    """_get_agent returns None → 500 Skill activation failed."""
    registry, features = dispatch_setup

    class ActivatingSkill(BaseSkill):
        name = "test"
        trigger = "^^test"
        activation = SkillActivation.COMMAND
        ttl_seconds = 300

        def parse_subcommand(self, text):
            return ("plan", "do something")

        def handle_direct(self, ctx, action, arg):
            return None

        def get_meta(self):
            return SkillMeta(name="test", trigger="^^test")

    registry.register(ActivatingSkill())
    ctx = _make_ctx("^^test plan do something")

    with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
        with patch("proxy.pipeline.dispatch._get_agent", return_value=None):
            result = try_skill_activation(ctx, features)

    assert result.handled is True
    assert result.status_code == 500
    assert "Skill activation failed" in result.response["error"]["message"]


def test_malformed_choices_empty_list_non_streaming(dispatch_setup):
    """handle_direct returns dict with empty choices list — non-streaming path."""
    registry, features = dispatch_setup

    class MalformedSkill(BaseSkill):
        name = "test"
        trigger = "^^test"
        activation = SkillActivation.COMMAND
        ttl_seconds = 300

        def parse_subcommand(self, text):
            return ("status", "")

        def handle_direct(self, ctx, action, arg):
            return {"choices": [], "model": "skill"}

        def get_meta(self):
            return SkillMeta(name="test", trigger="^^test")

    registry.register(MalformedSkill())
    ctx = _make_ctx("^^test")

    with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
        result = try_skill_activation(ctx, features)

    assert result.handled is True
    assert result.response is not None
    # Non-streaming: the empty choices dict is passed through to jsonify
    json_data = result.response
    assert "choices" in json_data


# ---------------------------------------------------------------------------
# Ticket 2aa21136: malformed hash → plan mode fall-through
# ---------------------------------------------------------------------------


class TestMalformedHashFallThrough:
    """Verify that unknown double-caret tokens fall through to the LLM.

    Originally this class also asserted that malformed ticket hashes
    (e.g. ``^^ticket close trash``, ``^^ticket abc123``) entered plan
    mode. That contract was reverted in Phase 2B: malformed/unknown
    subcommands now FALL THROUGH to the LLM as ordinary user text
    (matches the user's stated contract: "^^ticket close trash" should
    reach the LLM as a regular message). The malformed-hash tests were
    removed; this class now only covers the double-caret-token fall-through
    path."""

    def test_unknown_double_caret_falls_through(self, dispatch_setup):
        registry, features = dispatch_setup
        ctx = _make_ctx("^^foo bar")
        result = try_skill_activation(ctx, features)
        assert result is None

    def test_garbage_after_double_caret_falls_through(self, dispatch_setup):
        registry, features = dispatch_setup
        ctx = _make_ctx("^^!@#$%")
        result = try_skill_activation(ctx, features)
        assert result is None

    def test_run_dispatch_returns_none_for_unknown_double_caret(self, dispatch_setup):
        from proxy.pipeline.dispatch import run_dispatch
        registry, features = dispatch_setup
        features.enhancer_enabled = False
        features.intercepts_commands = False
        ctx = _make_ctx("^^foo")
        result = run_dispatch(ctx, features)
        assert result is None
