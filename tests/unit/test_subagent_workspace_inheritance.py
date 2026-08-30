"""Unit tests for sub-agent workspace inheritance.

When a main agent in workspace "choriprox" spawns a sub-agent, the sub-agent
must inherit that workspace rather than fall back to "default". These tests
cover:
  - TicketSkill.get_inheritance_state includes workspace_name when non-default
  - WorkSkill.get_inheritance_state includes workspace_name when non-default
  - detect_subagent_ingress extracts workspace_name from the SKILL canary
  - parse_request activates the inherited workspace for the sub-agent session
  - parse_request rejects invalid workspace names
"""
from __future__ import annotations

import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy.skills.skill_definition import SkillRequestContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill_ctx(
    session_id: str = "sess-inherit-1",
    client_hint: str = "kilocode",
    workspace_name: str = "",
    skill_state: dict | None = None,
    work_task_hash: str = "",
) -> SkillRequestContext:
    ctx = SkillRequestContext(
        session_id=session_id,
        is_subagent=False,
        inherited_skill=None,
        subagent_ticket_mode=False,
        workspace_name=workspace_name,
        client_hint=client_hint,
        body={},
        seeded_tasks=[],
        mode_state=None,
    )
    ctx.skill_state = skill_state or {}
    ctx.work_task_hash = work_task_hash
    return ctx


# ---------------------------------------------------------------------------
# TicketSkill.get_inheritance_state
# ---------------------------------------------------------------------------


class TestTicketSkillInheritanceWorkspace:
    def test_includes_workspace_when_non_default(self):
        from skills.ticket_skill import TicketSkill

        with patch("proxy.workspace_state.workspace_store") as ws_mock, \
             patch("skills.ticket_skill._get_task_graph_singleton") as tg_factory:
            ws_mock.get_active.return_value = "choriprox"
            tg = MagicMock()
            tg.is_available.return_value = True
            tg.get_actionable_tasks.return_value = []
            tg.get_store_status.return_value = "ok"
            tg_factory.return_value = tg

            ctx = _make_skill_ctx()
            state = TicketSkill().get_inheritance_state(ctx)

            assert state is not None
            assert state.get("workspace_name") == "choriprox"

    def test_omits_workspace_when_default(self):
        from skills.ticket_skill import TicketSkill

        with patch("skills.ticket_skill.workspace_store") as ws_mock, \
             patch("skills.ticket_skill._get_task_graph_singleton") as tg_factory:
            ws_mock.get_active.return_value = "default"
            tg = MagicMock()
            tg.is_available.return_value = True
            tg.get_actionable_tasks.return_value = []
            tg.get_store_status.return_value = "ok"
            tg_factory.return_value = tg

            ctx = _make_skill_ctx()
            state = TicketSkill().get_inheritance_state(ctx)

            assert state is not None
            assert "workspace_name" not in state

    def test_includes_workspace_with_tasks(self):
        from skills.ticket_skill import TicketSkill

        with patch("proxy.workspace_state.workspace_store") as ws_mock, \
             patch("skills.ticket_skill._get_task_graph_singleton") as tg_factory:
            ws_mock.get_active.return_value = "choriprox"
            tg = MagicMock()
            tg.is_available.return_value = True
            tg.get_actionable_tasks.return_value = [
                {"short_hash": "abc12345", "name": "do thing", "priority": "high",
                 "status": "pending", "summary": "s", "tag": "", "blockers": [],
                 "is_blocked": False, "blocked_count": 0}
            ]
            tg.get_store_status.return_value = "ok"
            tg_factory.return_value = tg

            ctx = _make_skill_ctx()
            state = TicketSkill().get_inheritance_state(ctx)

            assert state is not None
            assert state.get("workspace_name") == "choriprox"
            assert len(state["tasks"]) == 1


# ---------------------------------------------------------------------------
# WorkSkill.get_inheritance_state
# ---------------------------------------------------------------------------


class TestWorkSkillInheritanceWorkspace:
    def test_includes_workspace_when_non_default(self):
        from skills.work_skill import WorkSkill

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.get_active.return_value = "choriprox"
            ctx = _make_skill_ctx(skill_state={"arg": "abc12345"})
            state = WorkSkill().get_inheritance_state(ctx)

            assert state is not None
            assert state.get("workspace_name") == "choriprox"
            assert state.get("task_hash") == "abc12345"

    def test_omits_workspace_when_default(self):
        from skills.work_skill import WorkSkill

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.get_active.return_value = "default"
            ctx = _make_skill_ctx(skill_state={"arg": "abc12345"})
            state = WorkSkill().get_inheritance_state(ctx)

            assert state is not None
            assert "workspace_name" not in state
            assert state.get("task_hash") == "abc12345"

    def test_includes_workspace_when_no_task_hash(self):
        """Empty task_hash but active non-default workspace → workspace_name emitted.

        This is the escalation path: ticket parent with subagent_mode="work"
        builds a work canary with no task_hash; the work sub-agent must still
        inherit the parent's active workspace.
        """
        from skills.work_skill import WorkSkill

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.get_active.return_value = "choriprox"
            ctx = _make_skill_ctx()
            state = WorkSkill().get_inheritance_state(ctx)

            assert state is not None
            assert state.get("workspace_name") == "choriprox"
            assert "task_hash" not in state


# ---------------------------------------------------------------------------
# detect_subagent_ingress SKILL branch extracts workspace_name
# ---------------------------------------------------------------------------


@pytest.fixture
def secret_unset(monkeypatch):
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    monkeypatch._secret_value = None  # noqa: SLF001
    return None


@pytest.fixture
def canary_module(monkeypatch, secret_unset):
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    import proxy.canary as mod
    importlib.reload(mod)
    return mod


def _build_skill_canary(canary_module, skill_name: str, state: dict) -> str:
    sig = canary_module.sign_canary(skill_name)
    state_str = ""
    if state:
        state_str = f"|{json.dumps(state)}"
    return f"[[__SUBAGENT__:SKILL:{skill_name}:{sig}{state_str}]]"


class TestSkillCanaryBranchWorkspace:
    def test_extracts_workspace_from_skill_canary(self, canary_module):
        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": "choriprox", "tasks": []}
        )
        body = {"messages": [{"role": "user", "content": canary + "\nDo work"}]}
        result = canary_module.detect_subagent_ingress(body)

        assert result.is_subagent is True
        assert result.inherited_skill == "ticket"
        assert result.workspace_name == "choriprox"

    def test_skill_canary_without_workspace_default_empty(self, canary_module):
        canary = _build_skill_canary(canary_module, "ticket", {"tasks": []})
        body = {"messages": [{"role": "user", "content": canary + "\nDo work"}]}
        result = canary_module.detect_subagent_ingress(body)

        assert result.is_subagent is True
        assert result.inherited_skill == "ticket"
        assert result.workspace_name == ""

    def test_skill_canary_no_state(self, canary_module):
        sig = canary_module.sign_canary("work")
        canary = f"[[__SUBAGENT__:SKILL:work:{sig}]]"
        body = {"messages": [{"role": "user", "content": canary + "\nDo work"}]}
        result = canary_module.detect_subagent_ingress(body)

        assert result.is_subagent is True
        assert result.inherited_skill == "work"
        assert result.workspace_name == ""


# ---------------------------------------------------------------------------
# parse_request activates inherited workspace
# ---------------------------------------------------------------------------


def _make_fake_request(content: str, session_id: str = "sess-inherit-2"):
    return SimpleNamespace(
        method="POST",
        path="/v1/chat/completions",
        remote_addr="127.0.0.1",
        headers={"X-Session-ID": session_id},
        get_json=lambda silent=False: {
            "model": "test-model",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        },
    )


class TestParseRequestActivatesWorkspace:
    """CP2B: the inherited-workspace activation was moved OUT of parse_request
    (now pure ingress) into proxy.owner_setup.activate_inherited_workspace,
    which runs on the owner execution path. parse_request still populates
    ctx.is_subagent / ctx.workspace_name; the activation is exercised at its
    real new call site.
    """

    def test_activates_inherited_workspace(self, monkeypatch, canary_module):
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": "choriprox", "tasks": []}
        )
        fake_request = _make_fake_request(canary + "\nDo work")

        set_active_calls = []
        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.set_active.side_effect = lambda sid, ws, **kw: set_active_calls.append((sid, ws, kw))
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        assert ctx.is_subagent is True
        assert ctx.workspace_name == "choriprox"
        assert len(set_active_calls) == 1
        assert set_active_calls[0][0] == ctx.session_id
        assert set_active_calls[0][1] == "choriprox"
        assert set_active_calls[0][2] == {}

    def test_rejects_invalid_workspace_name(self, monkeypatch, canary_module):
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": "bad name!", "tasks": []}
        )
        fake_request = _make_fake_request(canary + "\nDo work")

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.set_active.side_effect = AssertionError("set_active must NOT be called for invalid workspace")
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        assert ctx.workspace_name == "bad name!"
        ws_mock.set_active.assert_not_called()

    def test_rejects_too_long_workspace_name(self, monkeypatch, canary_module):
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        long_name = "a" * 65
        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": long_name, "tasks": []}
        )
        fake_request = _make_fake_request(canary + "\nDo work")

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.set_active.side_effect = AssertionError("set_active must NOT be called for too-long workspace")
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        ws_mock.set_active.assert_not_called()

    def test_rejects_default_workspace_name(self, monkeypatch, canary_module):
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": "default", "tasks": []}
        )
        fake_request = _make_fake_request(canary + "\nDo work")

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.set_active.side_effect = AssertionError("set_active must NOT be called for 'default'")
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        ws_mock.set_active.assert_not_called()

    def test_does_not_activate_for_non_subagent(self, monkeypatch, canary_module):
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        fake_request = _make_fake_request("plain request, no canary")

        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        assert ctx.is_subagent is False
        ws_mock.set_active.assert_not_called()

    def test_accepts_underscore_workspace_name(self, monkeypatch, canary_module):
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": "my_workspace", "tasks": []}
        )
        fake_request = _make_fake_request(canary + "\nDo work")

        set_active_calls = []
        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.set_active.side_effect = lambda sid, ws, **kw: set_active_calls.append((sid, ws, kw))
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        assert ctx.workspace_name == "my_workspace"
        assert len(set_active_calls) == 1
        assert set_active_calls[0][1] == "my_workspace"

    def test_parse_request_does_not_activate_workspace(self, monkeypatch, canary_module):
        """Revert-check pin (CP2B): parse_request ALONE must NOT touch the
        shared workspace_store — the activation was moved to owner_setup. If
        the deleted block is re-added to request_parse.py, the first assertion
        fails. A separate assertion confirms activate_inherited_workspace(ctx)
        DOES perform the activation.
        """
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        canary = _build_skill_canary(
            canary_module, "ticket", {"workspace_name": "choriprox", "tasks": []}
        )
        fake_request = _make_fake_request(canary + "\nDo work")

        set_active_calls = []
        with patch("proxy.workspace_state.workspace_store") as ws_mock:
            ws_mock.set_active.side_effect = lambda sid, ws, **kw: set_active_calls.append((sid, ws, kw))
            from proxy.pipeline.stages import parse_request
            ctx = parse_request(fake_request)

            # parse_request is pure ingress: no shared-state mutation.
            assert ctx.is_subagent is True
            assert ctx.workspace_name == "choriprox"
            ws_mock.set_active.assert_not_called()
            assert set_active_calls == []

            # The owner-only helper is where the activation actually happens.
            from proxy.owner_setup import activate_inherited_workspace
            activate_inherited_workspace(ctx)

        assert len(set_active_calls) == 1
        assert set_active_calls[0][0] == ctx.session_id
        assert set_active_calls[0][1] == "choriprox"


# ---------------------------------------------------------------------------
# DEFAULT-mode sub-agent workspace inheritance (BUG #6)
# ---------------------------------------------------------------------------


class TestDefaultModeWorkspaceCanary:
    """BUG #6: A DEFAULT-mode main agent that spawns a sub-agent must emit a
    WORKSPACE canary so the sub-agent inherits the parent's workspace.

    The existing tests in TestSkillCanaryBranchWorkspace cover the SKILL
    canary path; this class covers the DEFAULT-mode (no skill) path handled
    by the post-processing stage when no skill is active.
    """

    def test_apply_post_process_emits_workspace_canary_when_no_skill(self, monkeypatch):
        """No skill active + non-default workspace -> canary is WORKSPACE format.

        Wave 4: PostProcessStage.execute() replaces the direct apply_post_process
        call. The 8 unrelated sub-methods are patched as no-ops so only the
        canary-injection + lifecycle finalize paths run.
        """
        from proxy.mode import Mode as mode_mod
        from proxy.pipeline.post_process import PostProcessStage

        # Build a minimal agent (no active mode) and ctx.
        agent = MagicMock()
        agent.agent_id = "agent-1"
        agent.snapshot_mode.return_value = None  # no skill active
        agent.session_id = "sess-default-1"

        ctx = MagicMock()
        ctx.session_id = "sess-default-1"
        ctx.client_hint = "kilocode"
        ctx.messages = []
        ctx._mode = mode_mod.DEFAULT
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.swarm_policy = None
        ctx.llama_body = {"messages": []}
        ctx.request_id = "req_default_workspace_1"
        ctx.is_subagent = False

        data = {
            "choices": [{
                "message": {
                    "content": "done",
                    "tool_calls": [{
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": '{"prompt": "do the thing"}',
                        },
                    }],
                },
                "finish_reason": "stop",
            }]
        }
        ctx.response_data = data

        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        from contextlib import ExitStack
        def _noop(self, ctx):
            return None

        with patch("proxy.workspace_state.workspace_store") as ws_mock, \
             patch("proxy.skills.skill_registry.SkillRegistry") as reg_cls, \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=agent), \
             ExitStack() as stack:
            ws_mock.get_active.return_value = "choriprox"
            reg_cls.return_value.get.return_value = None  # no skill lookup needed
            stack.enter_context(patch.object(PostProcessStage, "_extract_minimax_reasoning", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_scan_write_results", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_sync_chat_history", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_persist_messages", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_populate_vfs_cache", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_capture_cache_metrics", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_handle_tool_interception", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_assign_model_name", _noop))
            result = PostProcessStage().execute(ctx)

        # Extract the injected canary from the task tool call prompt.
        tc = result["choices"][0]["message"]["tool_calls"][0]
        prompt = json.loads(tc["function"]["arguments"])["prompt"]
        assert prompt.startswith("[[__SUBAGENT__:WORKSPACE:choriprox:"), (
            f"Expected WORKSPACE canary for 'choriprox', got prompt={prompt!r}"
        )
        # Must NOT be the bare marker.
        assert not prompt.startswith("[[__SUBAGENT__]]\n"), (
            f"Bare canary emitted despite non-default workspace; prompt={prompt!r}"
        )

    def test_apply_post_process_emits_bare_canary_when_default_workspace(self, monkeypatch):
        """No skill active + default workspace -> bare [[__SUBAGENT__]] canary.

        Wave 4: PostProcessStage.execute() replaces the direct apply_post_process
        call. The 8 unrelated sub-methods are patched as no-ops so only the
        canary-injection + lifecycle finalize paths run.
        """
        from proxy.mode import Mode as mode_mod
        from proxy.pipeline.post_process import PostProcessStage

        agent = MagicMock()
        agent.agent_id = "agent-1"
        agent.snapshot_mode.return_value = None
        agent.session_id = "sess-default-2"

        ctx = MagicMock()
        ctx.session_id = "sess-default-2"
        ctx.client_hint = ""
        ctx.messages = []
        ctx._mode = mode_mod.DEFAULT
        ctx.model_route = MagicMock()
        ctx.model_route.is_remote = False
        ctx.model_route.model_cfg = None
        ctx.upstream = None
        ctx.swarm_policy = None
        ctx.llama_body = {"messages": []}
        ctx.request_id = "req_default_workspace_2"
        ctx.is_subagent = False

        data = {
            "choices": [{
                "message": {
                    "content": "done",
                    "tool_calls": [{
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "task",
                            "arguments": '{"prompt": "do the thing"}',
                        },
                    }],
                },
                "finish_reason": "stop",
            }]
        }
        ctx.response_data = data

        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        from contextlib import ExitStack
        def _noop(self, ctx):
            return None

        with patch("proxy.workspace_state.workspace_store") as ws_mock, \
             patch("proxy.skills.skill_registry.SkillRegistry") as reg_cls, \
             patch("proxy.agent.agent_registry.registry.get_or_load", return_value=agent), \
             ExitStack() as stack:
            ws_mock.get_active.return_value = "default"
            reg_cls.return_value.get.return_value = None
            stack.enter_context(patch.object(PostProcessStage, "_extract_minimax_reasoning", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_scan_write_results", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_sync_chat_history", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_persist_messages", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_populate_vfs_cache", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_capture_cache_metrics", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_handle_tool_interception", _noop))
            stack.enter_context(patch.object(PostProcessStage, "_assign_model_name", _noop))
            result = PostProcessStage().execute(ctx)

        tc = result["choices"][0]["message"]["tool_calls"][0]
        prompt = json.loads(tc["function"]["arguments"])["prompt"]
        assert prompt.startswith("[[__SUBAGENT__]]\n"), (
            f"Expected bare canary for default workspace, got prompt={prompt!r}"
        )

    def test_default_mode_workspace_canary_round_trips_through_detect_ingress(
        self, monkeypatch, canary_module
    ):
        """End-to-end: the WORKSPACE canary minted in apply_post_process is
        parsed by detect_subagent_ingress and sets workspace_name.
        """
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        # Mint a WORKSPACE canary the same way mode.py does.
        sig = canary_module.sign_canary("workspace:choriprox")
        canary = f"[[__SUBAGENT__:WORKSPACE:choriprox:{sig}]]"
        body = {
            "messages": [
                {"role": "user", "content": canary + "\nDo the task"}
            ]
        }

        result = canary_module.detect_subagent_ingress(body)

        assert result.is_subagent is True
        # WORKSPACE canary does NOT grant a skill mode — it only identifies
        # the workspace. Setting inherited_skill to the literal "workspace"
        # (a fake skill name) caused downstream skill_resolver to fail to
        # activate any mode, leaving the sub-agent with no active skill AND
        # proxy_ticket in its tool list. Now inherited_skill is None
        # (default mode) and workspace_name carries the workspace identity.
        assert result.inherited_skill is None
        assert result.workspace_name == "choriprox"
        # The canary prefix must be stripped from the message content.
        assert "Do the task" in body["messages"][0]["content"]
        assert "[[__SUBAGENT__" not in body["messages"][0]["content"]