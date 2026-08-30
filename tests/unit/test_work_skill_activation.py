"""Unit tests for the work skill activation fix.

Verifies that `SkillDispatcher.activate()` threads `skill_state` and
`raw_user_text` through to `skill.on_activate()` so that `WorkSkill`
can read the task hash from `skill_state["arg"]` (US-092). Also
verifies that dispatch error responses use `make_command_response()`
(OpenAI format with `choices`) instead of a bare `{"error": "string"}`.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from proxy.skill_dispatcher import SkillDispatcher
from proxy.skills.skill_definition import (
    SkillDefinition,
    SkillRequestContext,
    ModeState,
)
from proxy.skills.skill_registry import (
    SkillRegistry,
    _skill_registry_new__,
    reset_default_registry,
)
from skills.work_skill import WorkSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx() -> SimpleNamespace:
    """Build a minimal ctx-like object with the attributes the dispatcher reads."""
    return SimpleNamespace(
        session_id="s1",
        is_subagent=False,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint=None,
        body={},
        seeded_tasks=[],
        mode_state=None,
    )


def _make_agent() -> SimpleNamespace:
    """Build a minimal agent-like object with the methods the dispatcher calls."""
    return SimpleNamespace(
        snapshot_mode=lambda: None,
        set_active_mode=lambda ms: None,
    )


def _make_test_skill(on_activate=None, parse_subcommand=None, handle_direct=None) -> SkillDefinition:
    """Build a `SkillDefinition` with the provided callables wired in."""
    return SkillDefinition(
        name="test",
        trigger="^^test",
        aliases=(),
        parse_subcommand=parse_subcommand,
        handle_direct=handle_direct,
        on_activate=on_activate,
    )


def _make_work_skill_ctx(skill_state: dict) -> SkillRequestContext:
    """Build a `SkillRequestContext` for WorkSkill.on_activate tests."""
    ctx = SkillRequestContext(
        session_id="sess-work-act",
        is_subagent=False,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name="",
        client_hint=None,
        body={},
        seeded_tasks=[],
        mode_state=None,
    )
    ctx.skill_state = skill_state
    return ctx


@pytest.fixture(autouse=True)
def fresh_registry():
    """Each test gets a fresh registry (autouse)."""
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# 1. SkillDispatcher.activate threads skill_state to on_activate
# ---------------------------------------------------------------------------


class TestActivateThreadsSkillState:
    def test_activate_threads_skill_state_to_on_activate(self):
        """activate() MUST pass skill_state and raw_user_text through to on_activate,
        and set_active_mode MUST receive a ModeState whose skill_state contains
        {"action": "task", "arg": "abc12345"}."""
        captured = {}
        set_active_calls = []

        def fake_on_activate(skill_ctx: SkillRequestContext, mode_state: ModeState) -> None:
            captured["skill_state"] = getattr(skill_ctx, "skill_state", None)
            captured["raw_user_text"] = getattr(skill_ctx, "raw_user_text", None)

        skill = _make_test_skill(on_activate=fake_on_activate)
        ctx = _make_ctx()

        def _capture_set_active(mode_state):
            set_active_calls.append(mode_state)

        agent = SimpleNamespace(
            snapshot_mode=lambda: None,
            set_active_mode=_capture_set_active,
        )
        dispatcher = SkillDispatcher(_skill_registry_new__(SkillRegistry))

        ok = dispatcher.activate(
            skill,
            ctx,
            agent,
            skill_state={"action": "task", "arg": "abc12345"},
            raw_user_text="^^work abc12345",
        )

        assert ok is True
        assert captured["skill_state"] == {"action": "task", "arg": "abc12345"}
        assert captured["raw_user_text"] == "^^work abc12345"
        # Verify set_active_mode was called with a ModeState carrying the
        # skill_state (the bug was that skill_state was lost).
        assert len(set_active_calls) == 1
        ms = set_active_calls[0]
        assert isinstance(ms, ModeState)
        assert dict(ms.skill_state) == {"action": "task", "arg": "abc12345"}

    def test_activate_without_skill_state_defaults_to_empty(self):
        """activate() without skill_state MUST default on_activate's ctx to {}."""
        captured = {}

        def fake_on_activate(skill_ctx: SkillRequestContext, mode_state: ModeState) -> None:
            captured["skill_state"] = getattr(skill_ctx, "skill_state", None)

        skill = _make_test_skill(on_activate=fake_on_activate)
        ctx = _make_ctx()
        agent = _make_agent()
        dispatcher = SkillDispatcher(_skill_registry_new__(SkillRegistry))

        ok = dispatcher.activate(skill, ctx, agent)

        assert ok is True
        assert captured["skill_state"] == {}


# ---------------------------------------------------------------------------
# 2. WorkSkill.on_activate succeeds / raises with task hash
# ---------------------------------------------------------------------------


class TestWorkSkillOnActivate:
    def test_work_skill_on_activate_succeeds_with_task_hash(self):
        """on_activate MUST NOT raise when skill_state has a task hash."""
        from unittest.mock import patch, MagicMock

        ctx = _make_work_skill_ctx(skill_state={"action": "task", "arg": "abc12345"})
        mode_state = ModeState(
            name="work",
            activated_at=0.0,
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            sub_agent_tool_filter=frozenset(),
        )
        # Bug #10: on_activate now validates the hash exists in the task
        # graph. Mock so the validation succeeds regardless of live Neo4j.
        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = {
            "name": "Test", "priority": "medium", "summary": "",
        }
        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            WorkSkill().on_activate(ctx, mode_state)

    def test_work_skill_on_activate_raises_without_task_hash(self):
        """on_activate MUST raise ValueError(US-092) when skill_state has no arg."""
        ctx = _make_work_skill_ctx(skill_state={})
        mode_state = ModeState(
            name="work",
            activated_at=0.0,
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            sub_agent_tool_filter=frozenset(),
        )
        with pytest.raises(ValueError, match="task hash"):
            WorkSkill().on_activate(ctx, mode_state)


# ---------------------------------------------------------------------------
# 2b. Bug #10 — WorkSkill.on_activate validates the task hash exists
# ---------------------------------------------------------------------------


class TestWorkSkillOnActivateHashValidation:
    """Bug #10: ^^work <nonexistent hash> MUST be rejected at activation time.

    Previously on_activate read the hash and logged it but never validated
    existence — work mode activated for nonexistent tasks and the agent
    only discovered the error at start_work upstream. The fix validates via
    TaskGraph.get_task_enriched in on_activate and raises ValueError when
    the hash is absent.
    """

    def test_on_activate_rejects_nonexistent_hash(self):
        """A hash not in the task graph MUST raise ValueError."""
        from unittest.mock import patch, MagicMock

        ctx = _make_work_skill_ctx(skill_state={"action": "task", "arg": "deadbeef"})

        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = None  # task not found

        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            with pytest.raises(ValueError, match="No task found with hash 'deadbeef'"):
                WorkSkill().on_activate(ctx)

    def test_on_activate_accepts_existing_hash(self):
        """A hash that exists in the task graph MUST activate cleanly."""
        from unittest.mock import patch, MagicMock

        ctx = _make_work_skill_ctx(skill_state={"action": "task", "arg": "abc12345"})

        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.return_value = {
            "name": "Fix login",
            "priority": "high",
            "summary": "Auth bug.",
        }

        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            # Must NOT raise.
            WorkSkill().on_activate(ctx)

    def test_on_activate_fail_open_when_taskgraph_unavailable(self):
        """Bug #10 fail-open: when the task store is unavailable (e.g., Neo4j
        outage), on_activate MUST NOT block activation. The agent will hit
        the error at start_work instead — same as today.
        """
        from unittest.mock import patch, MagicMock

        ctx = _make_work_skill_ctx(skill_state={"action": "task", "arg": "anyhash1"})

        mock_tg = MagicMock()
        mock_tg.is_available.return_value = False  # store down

        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg):
            # Must NOT raise even though the hash is unresolvable.
            WorkSkill().on_activate(ctx)

    def test_on_activate_accepts_ambiguous_hash(self):
        """AmbiguousHashError means the hash EXISTS (matches multiple) —
        activation MUST proceed (start_work will surface the ambiguity).
        """
        from unittest.mock import patch, MagicMock
        from memory.task_graph import AmbiguousHashError

        ctx = _make_work_skill_ctx(skill_state={"action": "task", "arg": "abc12345"})

        mock_tg = MagicMock()
        mock_tg.is_available.return_value = True
        mock_tg.get_task_enriched.side_effect = AmbiguousHashError(3)

        with patch("proxy.context._get_task_graph_singleton", return_value=mock_tg), \
             patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            WorkSkill().on_activate(ctx)  # must not raise


# ---------------------------------------------------------------------------
# 3. Dispatch error response uses OpenAI format (make_command_response)
# ---------------------------------------------------------------------------


class TestDispatchErrorResponseOpenAIFormat:
    def test_dispatch_error_response_is_openai_format(self):
        """When a skill raises ValueError during dispatch, the response MUST be
        an OpenAI-style error object ({"error": {"message": ..., "type": ...}}),
        NOT a bare {"error": "string"} and NOT a choices-array.
        """
        from proxy.pipeline.dispatch import try_skill_activation

        ctx = SimpleNamespace(
            session_id="sess-dispatch-err",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint=None,
            body={"model": "test-model", "stream": False},
            seeded_tasks=[],
            mode_state=None,
            messages=[{"role": "user", "content": "^^work abc12345"}],
        )

        # Patch the dispatcher's dispatch path so SkillDispatcher.dispatch raises
        # ValueError (simulating WorkSkill.on_activate raising "requires a task
        # hash" or any other ValueError from a skill internal).
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "proxy.pipeline.dispatch._get_agent",
                lambda c: _make_agent(),
            )

            class _RaisingDispatcher:
                def __init__(self, registry, *a, **k):
                    pass

                def detect_command(self, text):
                    return SimpleNamespace(name="work")

                def dispatch(self, text, ctx, agent):
                    raise ValueError("simulated skill activation failure")

            mp.setattr(
                "proxy.skill_dispatcher.SkillDispatcher",
                _RaisingDispatcher,
            )

            # Make SkillRegistry() cheap (no builtin population needed for this
            # error-path test).
            mp.setattr(
                "proxy.skills.skill_registry.SkillRegistry",
                lambda *a, **k: MagicMock(),
            )

            result = try_skill_activation(ctx, features=SimpleNamespace())

        assert result is not None
        assert result.handled is True
        assert result.status_code == 500
        assert result.response is not None
        assert result.rewritten_text is None
        # The response is a plain OpenAI-style error dictionary carried by
        # DispatchResult.response; no Flask tuple or Response wrapping.
        body = result.response
        assert body == {
            "error": {
                "message": "Skill activation failed",
                "type": "skill_error",
            }
        }