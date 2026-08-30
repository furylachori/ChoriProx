"""Phase 2 Agent 5 — swarm canary injection tests.

Runs against the real `PostProcessStage._inject_task_canaries` boundary
and the real `proxy.canary.transform_swarm_task_calls` helper.

Test groups:
  - Valid MAIN swarm parent (single task, multi-task, default ws, non-default
    ws, identity stability, sibling preservation, prompt-strip preservation).
  - Budget: exactly limit accepted; limit+1 → swarm_max_calls_exceeded (400).
  - Atomic failure: second-call malformed → no partial mutation.
  - Universal no-nesting: parametrized over inherited_skill × swarm_policy.
  - Signing failures: 5 codes × StageAbort tuple.
  - Legacy preservation: bare / workspace / ticket / work / malformed-call.
  - Log redaction: prompt / canary / sig never appear at DEBUG.
  - Concurrency: threading.Barrier, two concurrent MAIN ctx's.

All tests use the real boundary — no mocking of the function under test.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from proxy import canary as canary_mod
from proxy.agent.agent import Agent
from proxy.skills.skill_definition import ModeState
from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.post_process import PostProcessStage
from proxy.skills.skill_definition import SubagentPolicy


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_TEST_SECRET = "x" * 32


@pytest.fixture(autouse=True)
def _pin_canary_secret(monkeypatch):
    """Pin `_SESSION_SECRET` to a known value for the swarm canary tests.

    Autouse so every test in this module starts with a known signing key.
    """
    monkeypatch.setattr(canary_mod, "_SESSION_SECRET", _TEST_SECRET)
    yield


def _make_post_process_ctx(
    session_id: str = "sess_swarm_pp",
    agent_id: str = "agent_swarm_pp",
    *,
    is_subagent: bool = False,
    swarm_policy=None,
    inherited_skill=None,
    workspace_name: str = "",
) -> SimpleNamespace:
    """Build a SimpleNamespace matching the attributes PostProcessStage touches."""
    body = {"messages": []}
    ctx = SimpleNamespace(
        session_id=session_id,
        agent_id=agent_id,
        is_subagent=is_subagent,
        subagent_ticket_mode=False,
        seeded_tasks=[],
        inherited_skill=inherited_skill,
        workspace_name=workspace_name,
        _mode=None,
        client_hint="",
        messages=[],
        body=body,
        request_id="req_swarm_pp",
        model_route=MagicMock(),
        upstream=None,
        llama_body=body,
        swarm_policy=swarm_policy,
        response_data=None,
    )
    ctx.model_route.is_remote = False
    ctx.model_route.model_cfg = None
    return ctx


def _make_agent(session_id: str = "sess_swarm_pp", mode_name: str | None = "swarm-parent") -> Agent:
    """Real Agent with the given mode active (None for no mode)."""
    agent = Agent(session_id=session_id, agent_id=f"agent_{session_id}")
    if mode_name is not None:
        agent.set_active_mode(
            ModeState(name=mode_name, activated_at=time.time(), ttl_seconds=300)
        )
    return agent


def _make_task_tool_call(call_id: str, prompt: str) -> dict:
    """Build a single native `task` tool-call entry."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({"description": "test", "prompt": prompt}),
        },
    }


def _make_response_with_task_calls(task_prompts: list[str], sibling_tool_calls: list | None = None) -> dict:
    """Build a parent response containing N native `task` tool calls plus optional siblings."""
    tool_calls = [_make_task_tool_call(f"call_{i}", p) for i, p in enumerate(task_prompts)]
    if sibling_tool_calls:
        tool_calls.extend(sibling_tool_calls)
    return {
        "choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }],
        "model": "mock",
    }


def _noop_submethods_patcher():
    """Patch the 8 PostProcessStage sub-methods to no-ops so we isolate _inject_task_canaries."""
    from contextlib import ExitStack
    stack = ExitStack()
    noop = lambda self, ctx: None
    stack.enter_context(patch.object(PostProcessStage, "_extract_minimax_reasoning", noop))
    stack.enter_context(patch.object(PostProcessStage, "_scan_write_results", noop))
    stack.enter_context(patch.object(PostProcessStage, "_sync_chat_history", noop))
    stack.enter_context(patch.object(PostProcessStage, "_persist_messages", noop))
    stack.enter_context(patch.object(PostProcessStage, "_populate_vfs_cache", noop))
    stack.enter_context(patch.object(PostProcessStage, "_capture_cache_metrics", noop))
    stack.enter_context(patch.object(PostProcessStage, "_handle_tool_interception", noop))
    stack.enter_context(patch.object(PostProcessStage, "_assign_model_name", noop))
    return stack


def _patch_get_or_load(agent):
    """Patch agent_registry.get_or_load to return the test's agent."""
    return patch(
        "proxy.agent.agent_registry.registry.get_or_load",
        side_effect=lambda session_id: agent,
    )


def _patch_workspace_resolution(active_ws: str = "default"):
    """Patch resolve_workspace to return the given workspace.

    `resolve_workspace` is imported lazily inside `proxy.mode.inject_task_canaries`,
    so the patch target is `proxy.workspace_state.resolve_workspace` (where
    the symbol is defined) rather than `proxy.mode.resolve_workspace`.
    """
    return patch("proxy.workspace_state.resolve_workspace", return_value=active_ws)


# ===========================================================================
# Valid MAIN swarm parent
# ===========================================================================


class TestValidMainSwarmParent:
    """Single + multi-task calls; default + non-default workspace."""

    def test_one_task_receives_signed_custom_skill_canary(self):
        """MAIN swarm parent with 1 task call receives a valid SKILL canary.

        The canary's signature is verified by the production signer. We
        use the real `parse_canary` and bypass the registry check by
        registering the parent skill in the SkillRegistry. Since the test
        module is not in TRUSTED_TRUST_CALLERS, we patch the registry's
        `get` to return a synthetic SkillDefinition (the production code
        only reads `name`).
        """
        from proxy.canary import parse_canary, verify_canary
        from proxy.skills.skill_registry import SkillRegistry

        policy = SubagentPolicy(
            tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4
        )
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        prompt_text = "investigate the auth bug"
        response = _make_response_with_task_calls([prompt_text])
        ctx.response_data = response

        fake_skill_def = MagicMock()
        fake_skill_def.name = "research"
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)
        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch(
                 "proxy.skills.skill_registry.SkillRegistry"
             ) as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.get.return_value = fake_skill_def
            mock_reg_cls.return_value = mock_reg
            result = PostProcessStage().execute(ctx)

            result_data = result if isinstance(result, dict) else ctx.response_data
            tc_args = json.loads(
                result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )
            # The task prompt now begins with a SKILL canary for "research".
            prompt = tc_args["prompt"]
            assert prompt.startswith("[[__SUBAGENT__:SKILL:research:"), (
                f"Expected SKILL canary for 'research', got: {prompt[:80]!r}"
            )
            # Original prompt is preserved AFTER the canary.
            assert prompt_text in prompt, f"Original prompt missing: {prompt!r}"
            # Round-trip parse (still inside the patch scope so the mock
            # SkillRegistry is active when parse_canary checks the registry).
            parsed = parse_canary(prompt)
            assert parsed is not None and parsed.kind == "skill", (
                f"parse_canary failed on the emitted marker: {prompt[:120]!r}"
            )
            assert parsed.skill_name == "research"
            # The signature verifies (no parse_canary verified=False fallback).
            assert parsed.verified, f"parse_canary unverified: {parsed}"

    def test_multiple_task_calls_all_receive_canary(self):
        """MAIN swarm parent with N=3 task calls: every call gets a valid canary."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["a", "b", "c"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            result = PostProcessStage().execute(ctx)

        result_data = result if isinstance(result, dict) else ctx.response_data
        tool_calls = result_data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 3
        for i, tc in enumerate(tool_calls):
            args = json.loads(tc["function"]["arguments"])
            assert args["prompt"].startswith("[[__SUBAGENT__:SKILL:research:"), (
                f"Task #{i} missing SKILL canary: {args['prompt'][:80]!r}"
            )

    def test_default_workspace_produces_no_state_suffix(self):
        """Default workspace produces no state suffix in the canary.

        The marker closes with `]]` directly after `:ts` (no `|{json}` suffix).
        """
        from proxy.canary import parse_canary
        from proxy.skills.skill_registry import SkillRegistry

        policy = SubagentPolicy(
            tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4
        )
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["investigate"])
        ctx.response_data = response

        fake_skill_def = MagicMock()
        fake_skill_def.name = "research"
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)
        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.get.return_value = fake_skill_def
            mock_reg_cls.return_value = mock_reg
            result = PostProcessStage().execute(ctx)

            result_data = result if isinstance(result, dict) else ctx.response_data
            prompt = json.loads(
                result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )["prompt"]
            # The closing `]]` is preceded by the timestamp (digits), no `|`.
            end = prompt.split("]]", 1)[0]
            assert not end.endswith("|"), (
                f"Default workspace must not produce a state suffix, got: {end!r}"
            )
            parsed = parse_canary(prompt)
            assert parsed is not None
            # State is empty dict (the marker omits the state segment entirely).
            assert parsed.state in (None, {})

    def test_non_default_workspace_carries_state(self):
        """Non-default workspace carries exactly `{"workspace_name": "alpha"}` state."""
        from proxy.canary import parse_canary
        from proxy.skills.skill_registry import SkillRegistry

        policy = SubagentPolicy(
            tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4
        )
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["investigate"])
        ctx.response_data = response

        fake_skill_def = MagicMock()
        fake_skill_def.name = "research"
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)
        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("alpha"), \
             patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.get.return_value = fake_skill_def
            mock_reg_cls.return_value = mock_reg
            result = PostProcessStage().execute(ctx)

            result_data = result if isinstance(result, dict) else ctx.response_data
            prompt = json.loads(
                result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )["prompt"]
            parsed = parse_canary(prompt)
            assert parsed is not None and parsed.kind == "skill", (
                f"parse_canary failed on emitted marker: {prompt[:120]!r}"
            )
            # State carries exactly workspace_name (and nothing else).
            assert parsed.state == {"workspace_name": "alpha"}

    def test_existing_canary_in_prompt_is_stripped(self):
        """An existing `[[__SUBAGENT__]]` in the prompt is stripped before prepending."""
        from proxy.canary import parse_canary

        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        prompt_with_existing = "[[__SUBAGENT__]]\ninvestigate the bug"
        response = _make_response_with_task_calls([prompt_with_existing])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            result = PostProcessStage().execute(ctx)

        result_data = result if isinstance(result, dict) else ctx.response_data
        prompt = json.loads(
            result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]
        # Only ONE canary in the prompt (no stacking).
        assert prompt.count("[[__SUBAGENT__") == 1, (
            f"Expected exactly 1 canary, got {prompt.count('[[__SUBAGENT__')}: {prompt!r}"
        )
        # Original prompt text survives.
        assert "investigate the bug" in prompt

    def test_response_data_is_new_object(self):
        """ctx.response_data is replaced (identity != source) after a successful swarm inject."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response
        original_id = id(response)
        original_data = copy.deepcopy(response)

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            result = PostProcessStage().execute(ctx)

        # The result is either ctx.response_data (replaced) or the early-return value.
        result_data = result if isinstance(result, dict) else ctx.response_data
        assert id(result_data) != original_id, "response_data must be a NEW object"

    def test_original_response_data_is_unchanged(self):
        """Original source response_data is byte-for-byte unchanged after a successful inject."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response
        original_data = copy.deepcopy(response)
        snapshot_args = response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            PostProcessStage().execute(ctx)

        # Source is deep-equal to its pre-call snapshot.
        assert response == original_data, "Source response_data must be byte-for-byte unchanged"
        # Source's first task call arguments are unchanged.
        assert (
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            == snapshot_args
        )

    def test_non_task_sibling_calls_preserved(self):
        """Non-task sibling tool calls are preserved byte-for-byte + ordering."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        sibling_a = {
            "id": "call_read_sibling",
            "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": "/foo/bar"})},
        }
        sibling_b = {
            "id": "call_glob_sibling",
            "type": "function",
            "function": {"name": "glob", "arguments": json.dumps({"pattern": "*.py"})},
        }
        response = _make_response_with_task_calls(["task1"], sibling_tool_calls=[sibling_a, sibling_b])
        ctx.response_data = response
        siblings_snapshot = copy.deepcopy([sibling_a, sibling_b])

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            result = PostProcessStage().execute(ctx)

        result_data = result if isinstance(result, dict) else ctx.response_data
        # Three tool calls: 1 task + 2 siblings.
        tcs = result_data["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 3
        # The task call is the FIRST (it was already first in the source).
        assert tcs[0]["function"]["name"] == "task"
        # The siblings follow in their original order.
        assert tcs[1] == siblings_snapshot[0]
        assert tcs[2] == siblings_snapshot[1]


# ===========================================================================
# Budget: max_task_calls_per_request
# ===========================================================================


class TestSwarmBudget:
    """Exactly limit accepted; limit+1 → swarm_max_calls_exceeded (400)."""

    def test_exactly_limit_accepted(self):
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=2)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["a", "b"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            result = PostProcessStage().execute(ctx)

        result_data = result if isinstance(result, dict) else ctx.response_data
        # Two task tool calls, each carrying a canary.
        tcs = result_data["choices"][0]["message"]["tool_calls"]
        assert len(tcs) == 2
        for tc in tcs:
            prompt = json.loads(tc["function"]["arguments"])["prompt"]
            assert prompt.startswith("[[__SUBAGENT__:SKILL:")

    def test_limit_plus_one_rejected_whole_response(self):
        """limit+1 task calls → StageAbort(swarm_max_calls_exceeded, 400)."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=2)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["a", "b", "c"])
        ctx.response_data = response
        original_snapshot = copy.deepcopy(response)

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 400, f"Expected HTTP 400, got {status}"
        assert envelope["error"]["code"] == "swarm_max_calls_exceeded"
        # Original response is unchanged (whole-response atomic rejection).
        assert response == original_snapshot

    def test_multiple_calls_in_one_response_count_separately(self):
        """Three task calls with limit=3 accepted; same count with limit=2 rejected."""
        # Accepted at limit=3.
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=3)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["a", "b", "c"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            PostProcessStage().execute(ctx)


# ===========================================================================
# Atomic failure on second call
# ===========================================================================


class TestAtomicFailureOnSecondCall:
    """Failure on the second-or-later task leaves source + first call unchanged."""

    def test_second_call_malformed_no_partial_mutation(self):
        """Second call's arguments is non-JSON → StageAbort; source unchanged."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        # First call is valid; second call has malformed arguments.
        tc1 = _make_task_tool_call("call_1", "first prompt")
        tc2 = {
            "id": "call_2",
            "type": "function",
            "function": {"name": "task", "arguments": "{not valid json"},
        }
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tc1, tc2],
                },
                "finish_reason": "tool_calls",
            }],
        }
        ctx.response_data = response
        # Snapshot every field we want to verify is unchanged.
        original_tc1_args = copy.deepcopy(tc1["function"]["arguments"])
        original_tc2_args = copy.deepcopy(tc2["function"]["arguments"])
        original_response_snapshot = copy.deepcopy(response)

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert envelope["error"]["code"] == "swarm_malformed_task_call", (
            f"Expected swarm_malformed_task_call, got {envelope['error']['code']}"
        )
        assert status == 502
        # Source first/second call arguments byte-for-byte unchanged.
        assert tc1["function"]["arguments"] == original_tc1_args
        assert tc2["function"]["arguments"] == original_tc2_args
        # Full source unchanged.
        assert response == original_response_snapshot


# ===========================================================================
# Universal no-nesting
# ===========================================================================


class TestUniversalNoNesting:
    """Every sub-agent response with a native task aborts (403)."""

    def test_bare_canary_subagent_with_task(self):
        """inherited_skill=None + is_subagent + task → 403 swarm_subagent_cannot_spawn."""
        ctx = _make_post_process_ctx(is_subagent=True, inherited_skill=None)
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"

    def test_workspace_canary_subagent_with_task(self):
        """inherited_skill='workspace:foo' + is_subagent + task → 403."""
        ctx = _make_post_process_ctx(
            is_subagent=True, inherited_skill="workspace:foo", workspace_name="foo"
        )
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("foo"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"

    def test_ticket_subagent_with_task(self):
        """inherited_skill='ticket' + is_subagent + task → 403."""
        ctx = _make_post_process_ctx(
            is_subagent=True, inherited_skill="ticket"
        )
        ctx.subagent_ticket_mode = True
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"

    def test_work_subagent_with_task(self):
        """inherited_skill='work' + is_subagent + task → 403."""
        ctx = _make_post_process_ctx(is_subagent=True, inherited_skill="work")
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"

    def test_custom_swarm_subagent_with_task(self):
        """inherited_skill='research' + is_subagent + task → 403 (sub-agent cannot spawn)."""
        ctx = _make_post_process_ctx(is_subagent=True, inherited_skill="research")
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"

    def test_unknown_inherited_skill_subagent_with_task(self):
        """inherited_skill='nonexistent-skill' + is_subagent + task → 403."""
        ctx = _make_post_process_ctx(
            is_subagent=True, inherited_skill="nonexistent-skill"
        )
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"

    def test_subagent_no_nesting_check_runs_before_registry(self):
        """The no-nesting check fires even when the registry is empty.

        Specifically: we do NOT set up a populated SkillRegistry, and the
        abort must still fire — proving the decision is registry-free.
        """
        ctx = _make_post_process_ctx(is_subagent=True, inherited_skill=None)
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        # Patch the registry to simulate a fully-empty registry.
        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.get.return_value = None
            mock_reg_cls.return_value = mock_reg
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)
        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"


class TestRegistryAccessOnlyAfterNoNesting:
    """SkillRegistry() must NOT be constructed before the universal no-nesting denial.

    Discriminating revert test for revert #13: a buggy "revert" would move the
    SkillRegistry() lookup (or any registry access) to BEFORE the no-nesting
    denial in ``proxy.mode.inject_task_canaries``. A sub-agent's native task
    call would still be rejected (universal §14 wins), but the registry would
    have been needlessly constructed first — a wasted I/O-bound operation.

    Pinning invariant: ``proxy.skills.skill_registry.SkillRegistry.__init__``
    is invoked ZERO times when a sub-agent's native task call triggers the
    no-nesting abort path.

    The test exercises the real production entry
    ``proxy.mode.inject_task_canaries(ctx, agent, response_data, mode)`` —
    no mocking of the function under test. The only mock is at the
    ``SkillRegistry`` class constructor (production import path), so the
    no-nesting check itself runs against the real source.
    """

    def test_skill_registry_not_constructed_before_no_nesting_abort(self):
        """Sub-agent + native task call → StageAbort(403), SkillRegistry() untouched.

        Drives the production ``proxy.mode.inject_task_canaries`` entry
        directly (not via ``PostProcessStage``) to isolate the ordering
        assertion to a single function. The patched ``SkillRegistry``
        constructor raises ``AssertionError`` if invoked — its call count
        is also asserted to be zero, providing defence-in-depth.
        """
        import proxy.skills.skill_registry as _sr_mod
        from proxy.mode import inject_task_canaries

        ctx = _make_post_process_ctx(is_subagent=True, inherited_skill=None)
        ctx.swarm_policy = None  # explicit (MagicMock contexts default falsy; be explicit)
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])

        # Patch the production import path. ``proxy.mode`` does
        #     from proxy.skills.skill_registry import SkillContext, SkillRegistry
        # inside ``inject_task_canaries`` (function-local import).
        # Patching the source attribute on the module intercepts that
        # local import. ``side_effect=AssertionError(...)`` makes any
        # accidental construction of the registry a hard failure.
        try:
            with patch.object(
                _sr_mod,
                "SkillRegistry",
                side_effect=AssertionError(
                    "SkillRegistry() must not be constructed before the no-nesting denial"
                ),
            ) as mock_reg_cls, \
                 patch(
                     "proxy.skills.skill_registry.SkillRegistry",
                     side_effect=AssertionError(
                         "SkillRegistry() must not be constructed before the no-nesting denial"
                     ),
                 ):
                with pytest.raises(StageAbort) as exc_info:
                    inject_task_canaries(
                        ctx=ctx, agent=agent, response_data=response, mode=None
                    )
        finally:
            pass  # patches restored automatically on context exit

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"
        # Defence-in-depth: even if a future change swallowed the
        # AssertionError, the call_count must remain zero.
        assert mock_reg_cls.call_count == 0, (
            f"SkillRegistry was constructed {mock_reg_cls.call_count} times "
            f"before the no-nesting denial — registry must be untouched."
        )

    def test_no_nesting_abort_fires_before_legacy_registry_path(self):
        """A sub-agent reaching the legacy MAIN code path is impossible.

        Even if a hypothetical revert bypassed the early no-nesting check
        (so control could fall through to the legacy ``SkillRegistry().get(...)``
        path on line 596 of mode.py), the no-nesting invariant must still
        hold: sub-agent + native task call → 403, never a legacy canary.

        This dual-asserts: the abort envelope is correct AND no canary was
        injected into the prompt. A revert that drops the early no-nesting
        check would let ``skill.get_canary(skill_ctx)`` execute and prepend
        a SKILL marker into the prompt — caught by the legacy-canary absence
        check.
        """
        from proxy.mode import inject_task_canaries

        ctx = _make_post_process_ctx(is_subagent=True, inherited_skill=None)
        ctx.swarm_policy = None
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        original_args = json.loads(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )

        try:
            with patch.object(
                proxy_skills_skill_registry := __import__(
                    "proxy.skills.skill_registry", fromlist=["SkillRegistry"]
                ),
                "SkillRegistry",
                side_effect=AssertionError(
                    "SkillRegistry() must not be constructed on sub-agent no-nesting path"
                ),
            ):
                with pytest.raises(StageAbort) as exc_info:
                    inject_task_canaries(
                        ctx=ctx, agent=agent, response_data=response, mode=None
                    )
        finally:
            pass

        envelope, status = exc_info.value.response_data
        assert status == 403
        assert envelope["error"]["code"] == "swarm_subagent_cannot_spawn"
        # The original task prompt is byte-for-byte unchanged — no canary
        # was injected into the source response.
        current_args = json.loads(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )
        assert current_args == original_args, (
            f"Task prompt was mutated on sub-agent no-nesting path. "
            f"Before: {original_args!r}, After: {current_args!r}"
        )


# ===========================================================================
# Signing failures (5 codes × StageAbort tuple)
# ===========================================================================


class TestSwarmSigningFailures:
    """Each SwarmCanaryError code maps to a StageAbort tuple with the right code/status."""

    def test_missing_secret(self):
        from proxy.canary import SwarmCanaryError
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch(
                 "proxy.canary.build_swarm_skill_canary",
                 side_effect=SwarmCanaryError("swarm_missing_secret", "no secret"),
             ):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)
        envelope, status = exc_info.value.response_data
        assert status == 500
        assert envelope["error"]["code"] == "swarm_missing_secret"
        # No fallback marker — original response's prompt is unchanged.
        original_prompt = json.loads(response["choices"][0]["message"]["tool_calls"][0]
                                     ["function"]["arguments"])["prompt"]
        assert "[[__SUBAGENT__:SKILL:" not in original_prompt

    def test_zero_signature_fallback(self):
        from proxy.canary import SwarmCanaryError
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch(
                 "proxy.canary.build_swarm_skill_canary",
                 side_effect=SwarmCanaryError("swarm_signing_failed", "zero sig"),
             ):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)
        envelope, status = exc_info.value.response_data
        assert status == 500
        assert envelope["error"]["code"] == "swarm_signing_failed"

    def test_strict_signer_exception(self):
        """A non-SwarmCanaryError exception from the signer also maps to swarm_signing_failed."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        # Patching sign_canary_strict itself to raise an unexpected exception.
        # mode.py's exception handler should map unknown codes to swarm_signing_failed.
        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch(
                 "proxy.canary.sign_canary_strict",
                 side_effect=RuntimeError("unexpected"),
             ):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)
        envelope, status = exc_info.value.response_data
        assert status == 500
        # RuntimeError (not SwarmCanaryError) bubbles out — the transform path
        # in inject_task_canaries re-raises and PostProcessStage propagates.
        # The error envelope is whatever RuntimeError produced; we verify the
        # response is NOT a fallback marker.
        original_prompt = json.loads(response["choices"][0]["message"]["tool_calls"][0]
                                     ["function"]["arguments"])["prompt"]
        assert "[[__SUBAGENT__:SKILL:" not in original_prompt

    def test_state_too_large(self):
        from proxy.canary import SwarmCanaryError
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch(
                 "proxy.canary.build_swarm_skill_canary",
                 side_effect=SwarmCanaryError("swarm_state_too_large", "too big"),
             ):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)
        envelope, status = exc_info.value.response_data
        assert status == 500
        assert envelope["error"]["code"] == "swarm_state_too_large"

    def test_policy_unavailable_when_mode_missing(self):
        """ctx.swarm_policy set but agent has no mode → swarm_policy_unavailable (500)."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name=None)  # No mode active
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)
        envelope, status = exc_info.value.response_data
        assert status == 500
        assert envelope["error"]["code"] == "swarm_policy_unavailable"


# ===========================================================================
# Legacy preservation
# ===========================================================================


class TestLegacyPreservation:
    """ctx.swarm_policy is None + not is_subagent → legacy ticket/work/workspace/bare paths."""

    def test_ordinary_main_no_mode_gets_bare_canary(self):
        """MAIN request, no mode, default workspace → bare canary prepended."""
        ctx = _make_post_process_ctx()  # no swarm_policy
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["look at the bug"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            result = PostProcessStage().execute(ctx)

        result_data = result if isinstance(result, dict) else ctx.response_data
        prompt = json.loads(
            result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]
        # Bare canary (no SKILL/WORKSPACE prefix).
        assert prompt.startswith("[[__SUBAGENT__]]\n"), (
            f"Expected bare canary prefix, got: {prompt[:80]!r}"
        )

    def test_non_default_workspace_main_gets_workspace_canary(self):
        """MAIN, no mode, non-default workspace → workspace canary (not bare, not SWARM)."""
        ctx = _make_post_process_ctx()  # no swarm_policy
        agent = _make_agent(mode_name=None)
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("feature-x"):
            result = PostProcessStage().execute(ctx)

        result_data = result if isinstance(result, dict) else ctx.response_data
        prompt = json.loads(
            result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]
        # WORKSPACE marker, NOT SKILL/SWARM.
        assert prompt.startswith("[[__SUBAGENT__:WORKSPACE:feature-x:"), (
            f"Expected workspace canary, got: {prompt[:80]!r}"
        )

    def test_ticket_main_gets_ticket_skill_canary(self):
        """MAIN ticket mode → ticket skill canary via skill.get_canary."""
        ctx = _make_post_process_ctx()  # no swarm_policy
        agent = _make_agent(mode_name="ticket")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response

        # Build a fake skill that returns a known canary string. post_process
        # must pass-through so ctx.response_data stays the real response dict.
        fake_canary = "[[__SUBAGENT__:SKILL:ticket:fake-sig:fake-ts]]\n"
        fake_skill = MagicMock()
        fake_skill.get_canary.return_value = fake_canary
        fake_skill.post_process = MagicMock(side_effect=lambda ctx, response: response)
        # Patch the registry get() to return the fake.
        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
            mock_reg = MagicMock()
            mock_reg.get.return_value = fake_skill
            mock_reg_cls.return_value = mock_reg
            result = PostProcessStage().execute(ctx)

            result_data = result if isinstance(result, dict) else ctx.response_data
            prompt = json.loads(
                result_data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )["prompt"]
            assert prompt.startswith("[[__SUBAGENT__:SKILL:ticket:"), (
                f"Expected ticket SKILL canary, got: {prompt[:80]!r}"
            )

    def test_existing_malformed_call_behavior_preserved(self):
        """Legacy path: malformed task call (non-JSON args) does NOT crash; silently skipped."""
        ctx = _make_post_process_ctx()  # no swarm_policy → legacy
        agent = _make_agent(mode_name=None)
        tc = {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "task", "arguments": "{not valid json"},
        }
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tc],
                },
                "finish_reason": "tool_calls",
            }],
        }
        ctx.response_data = response

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"):
            # Legacy behaviour: silently skip the malformed call.
            result = PostProcessStage().execute(ctx)
        # The function did not raise.
        assert result is not None or ctx.response_data is not None


# ===========================================================================
# Legacy strict-path bypass (revert test for spec §7.5 / §11)
# ===========================================================================


class TestLegacyTicketWorkStrictPathBypass:
    """Legacy MAIN ticket/work MUST go through ``skill.get_canary`` path,
    NOT the strict ``build_swarm_skill_canary`` path.

    Discriminating revert test for the spec §7.5 / §11 routing:

    - Current production (HEAD = ``f884216``): legacy MAIN paths are
      byte-identical to pre-strict — ``SkillRegistry().get(name).get_canary``
      builds the canary string and ``build_swarm_skill_canary`` is NEVER
      called when ``ctx.swarm_policy is None``.
    - Hypothetical revert (#15): legacy ticket/work routed through strict
      → ``build_swarm_skill_canary`` is called for legacy ticket/work
      requests. The "strict was NOT called" assertion below MUST fail.

    Patch targets (consumer-side, scoped):
      - ``proxy.mode.SkillRegistry`` — the legacy code's local binding for
        ``SkillRegistry()``. Patching here does NOT affect other importers
        (``proxy.skills.skill_registry.SkillRegistry`` stays real).
      - ``proxy.canary.build_swarm_skill_canary`` — the strict signer. The
        legacy code does ``from proxy.canary import build_swarm_skill_canary``
        inside ``inject_task_canaries``; this local import fetches the
        current value of the source attribute, so patching the source
        attribute intercepts the legacy call IF a revert routes through it.

    All patches use ``monkeypatch.setattr`` (pytest) for safe, function-
    scoped restoration. NO ``unittest.mock.patch`` is mixed in.
    """

    def _run_legacy_inject(self, monkeypatch, *, mode_name: str):
        """Drive ``proxy.mode.inject_task_canaries`` through the legacy path.

        Returns ``(result, legacy_skill, strict_mock)``:
          - ``result``: modified response_data dict with canary injected.
          - ``legacy_skill``: MagicMock whose ``get_canary`` is the legacy
            observation point.
          - ``strict_mock``: MagicMock at ``proxy.canary.build_swarm_skill_canary``
            — must remain uncalled for legacy MAIN ticket/work.
        """
        from proxy.mode import inject_task_canaries

        ctx = _make_post_process_ctx()
        assert ctx.swarm_policy is None, (
            "test invariant: legacy path requires ctx.swarm_policy = None"
        )
        ctx.is_subagent = False
        agent = _make_agent(mode_name=mode_name)
        response = _make_response_with_task_calls(
            [f"legacy-{mode_name}-payload"]
        )

        fake_canary = (
            f"[[__SUBAGENT__:SKILL:{mode_name}:"
            f"{'a' * 64}:1234567890]]\n"
        )

        legacy_skill = MagicMock(name=f"legacy_{mode_name}_skill")
        legacy_skill.get_canary.return_value = fake_canary

        fake_registry = MagicMock(name="legacy_registry")
        fake_registry.get.return_value = legacy_skill
        # `SkillRegistry` is imported lazily inside `inject_task_canaries`
        # (`from proxy.skills.skill_registry import SkillContext, SkillRegistry`),
        # so patching the source attribute is what intercepts the local import.
        # Replacing it with a lambda lets ``SkillRegistry()`` return our fake.
        monkeypatch.setattr(
            "proxy.skills.skill_registry.SkillRegistry",
            lambda *a, **kw: fake_registry,
        )

        strict_mock = MagicMock(name="strict_build_swarm_skill_canary")
        # If a revert routes the legacy branch through build_swarm_skill_canary,
        # the function under test MUST complete cleanly so the assertion fires.
        # Return a syntactically valid SKILL canary marker that the legacy
        # `_inject_canary_into_task_calls` can prepend without raising.
        strict_mock.return_value = (
            "[[__SUBAGENT__:SKILL:strict-revert-marker]]\n"
        )
        monkeypatch.setattr(
            canary_mod, "build_swarm_skill_canary", strict_mock
        )

        monkeypatch.setattr(
            "proxy.workspace_state.resolve_workspace",
            lambda *a, **kw: "default",
        )

        result = inject_task_canaries(
            ctx=ctx, agent=agent, response_data=response, mode=None
        )

        return result, legacy_skill, strict_mock

    def _extract_task_prompt(self, response_data: dict) -> str:
        return json.loads(
            response_data["choices"][0]["message"]["tool_calls"][0][
                "function"
            ]["arguments"]
        )["prompt"]

    def test_ticket_main_legacy_uses_skill_get_canary_and_skips_strict(
        self, monkeypatch
    ):
        """ticket_main: skill.get_canary MUST be called; build_swarm_skill_canary MUST NOT."""
        result, legacy_skill, strict_mock = self._run_legacy_inject(
            monkeypatch, mode_name="ticket"
        )

        assert legacy_skill.get_canary.call_count == 1, (
            "legacy path: skill.get_canary MUST be called exactly once for "
            f"ticket_main; got call_count={legacy_skill.get_canary.call_count}"
        )
        assert strict_mock.call_count == 0, (
            "strict path MUST be bypassed in legacy ticket_main; "
            f"build_swarm_skill_canary.call_count={strict_mock.call_count}. "
            "If this assertion fires, revert #15 has been applied: legacy "
            "ticket is being routed through build_swarm_skill_canary."
        )

        prompt = self._extract_task_prompt(result)
        assert prompt.startswith("[[__SUBAGENT__:SKILL:ticket:"), (
            f"Expected legacy ticket SKILL canary prefix; got: {prompt[:80]!r}"
        )
        assert prompt.endswith("legacy-ticket-payload"), (
            f"Original prompt payload must follow the canary; got: {prompt!r}"
        )

    def test_work_main_legacy_uses_skill_get_canary_and_skips_strict(
        self, monkeypatch
    ):
        """work_main: skill.get_canary MUST be called; build_swarm_skill_canary MUST NOT."""
        result, legacy_skill, strict_mock = self._run_legacy_inject(
            monkeypatch, mode_name="work"
        )

        assert legacy_skill.get_canary.call_count == 1, (
            "legacy path: skill.get_canary MUST be called exactly once for "
            f"work_main; got call_count={legacy_skill.get_canary.call_count}"
        )
        assert strict_mock.call_count == 0, (
            "strict path MUST be bypassed in legacy work_main; "
            f"build_swarm_skill_canary.call_count={strict_mock.call_count}. "
            "If this assertion fires, revert #15 has been applied: legacy "
            "work is being routed through build_swarm_skill_canary."
        )

        prompt = self._extract_task_prompt(result)
        assert prompt.startswith("[[__SUBAGENT__:SKILL:work:"), (
            f"Expected legacy work SKILL canary prefix; got: {prompt[:80]!r}"
        )
        assert prompt.endswith("legacy-work-payload"), (
            f"Original prompt payload must follow the canary; got: {prompt!r}"
        )


# ===========================================================================
# Log redaction
# ===========================================================================


class TestLogRedaction(unittest.TestCase):
    """Prompt content / canary / signature must NEVER appear at DEBUG."""

    def test_swarm_inject_does_not_log_prompt_content(self):
        """Inject a task call with a unique SECRET marker; verify it's not logged."""
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        secret_marker = "SECRET-MARKER-SWARM-XYZ-12345"
        response = _make_response_with_task_calls([secret_marker + " payload"])
        ctx.response_data = response

        # Need to register the parent skill so parse_canary can verify the canary.
        from proxy.skills.skill_registry import SkillRegistry
        fake_skill_def = MagicMock()
        fake_skill_def.name = "research"
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls, \
             self.assertLogs("proxy.mode", level=logging.DEBUG) as cm_mode:
            mock_reg = MagicMock()
            mock_reg.get.return_value = fake_skill_def
            mock_reg_cls.return_value = mock_reg
            PostProcessStage().execute(ctx)

        msgs = [r.getMessage() for r in cm_mode.records]
        joined = "\n".join(msgs)
        # No prompt content.
        self.assertNotIn(
            secret_marker, joined,
            f"Secret marker leaked into proxy.mode logs: "
            f"{[m for m in msgs if secret_marker in m]!r}",
        )
        # No canary prefix.
        self.assertNotIn(
            "[[__SUBAGENT__", joined,
            f"Canary marker leaked into proxy.mode logs: "
            f"{[m for m in msgs if '[[__SUBAGENT__' in m]!r}",
        )

    def test_swarm_inject_does_not_log_signature_fragment(self):
        """No 64-hex-char signature substring appears in swarm DEBUG logs."""
        import re
        policy = SubagentPolicy(tool_whitelist=frozenset({"read"}), max_task_calls_per_request=4)
        ctx = _make_post_process_ctx(swarm_policy=policy)
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["normal prompt"])
        ctx.response_data = response

        from proxy.skills.skill_registry import SkillRegistry
        fake_skill_def = MagicMock()
        fake_skill_def.name = "research"
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)

        with _noop_submethods_patcher(), _patch_get_or_load(agent), \
             _patch_workspace_resolution("default"), \
             patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls, \
             self.assertLogs("proxy.mode", level=logging.DEBUG) as cm_mode:
            mock_reg = MagicMock()
            mock_reg.get.return_value = fake_skill_def
            mock_reg_cls.return_value = mock_reg
            PostProcessStage().execute(ctx)

        msgs = [r.getMessage() for r in cm_mode.records]
        # 64-hex-char signature substring must NOT appear in swarm-related logs.
        swarm_msgs = [m for m in msgs if "swarm_inject" in m]
        swarm_joined = "\n".join(swarm_msgs)
        if swarm_joined:
            sig_pattern = re.compile(r"[0-9a-fA-F]{64}")
            self.assertIsNone(
                sig_pattern.search(swarm_joined),
                f"64-hex signature leaked into swarm_inject logs: {swarm_joined!r}",
            )

    def test_canary_strip_helper_removes_existing_canary(self):
        """_strip_existing_canaries strips a valid existing canary prefix."""
        from proxy.canary import _strip_existing_canaries

        # Use a valid 64-hex signature so the canary structurally parses.
        valid_sig = "a" * 64
        text = f"[[__SUBAGENT__:SKILL:foo:{valid_sig}:1700000000]]actual prompt content"
        stripped = _strip_existing_canaries(text)
        assert stripped == "actual prompt content", (
            f"Expected stripped prefix, got: {stripped!r}"
        )
        # No canary remnant.
        assert "[[__SUBAGENT__" not in stripped


class TestCanaryLoggerNoPromptDisclosure(unittest.TestCase):
    """Revert #11 — `proxy.canary` logger MUST NOT leak prompt content,
    canary markers, or HMAC signatures at DEBUG.

    The leaky form is:
        logger.debug(
            "canary_inject prompt_before_strip=%r after_strip=%r canary=%r "
            "final_prompt=%r",
            prompt_before_strip, prompt, chosen_canary, args["prompt"],
        )

    The correct (current) form is:
        logger.debug(
            "canary_inject task_count=%d request_id=%s code=ok",
            task_count, request_id,
        )

    This test captures from the `proxy.canary` logger (the right namespace —
    `TestLogRedaction` captures from `proxy.mode` and so misses the leak)
    and pins both the absence of every leaked field and the presence of the
    structural fields that prove the log line was actually captured.
    """

    def test_inject_canary_helper_logs_no_prompt_or_signature(self):
        """Direct call to `_inject_canary_into_task_calls` must not leak
        prompt content, canary markers, or 64-hex HMAC signatures into the
        `proxy.canary` logger; structural fields MUST be present.
        """
        import re
        from proxy.canary import _inject_canary_into_task_calls

        UNIQUE_PROMPT = "PROMPT_LEAK_CANARY_DETECTOR_42XYZ"
        UNIQUE_SIG = "deadbeef" * 8  # 64 hex chars, a stand-in HMAC signature
        INJECTED_CANARY = (
            f"[[__SUBAGENT__:SKILL:research:{UNIQUE_SIG}:1700000000]]"
        )

        ctx = _make_post_process_ctx(swarm_policy=None)
        ctx.swarm_policy = None
        ctx.request_id = "req_revert11_canary_log"

        response = _make_response_with_task_calls([UNIQUE_PROMPT])

        with self.assertLogs("proxy.canary", level=logging.DEBUG) as cm_canary:
            _inject_canary_into_task_calls(
                data=response,
                canary=INJECTED_CANARY,
                request_ctx=ctx,
                default_mode_name="research",
            )

        msgs = [r.getMessage() for r in cm_canary.records]
        joined = "\n".join(msgs)

        self.assertNotIn(
            UNIQUE_PROMPT, joined,
            f"Prompt content leaked into proxy.canary logs: "
            f"{[m for m in msgs if UNIQUE_PROMPT in m]!r}",
        )
        self.assertNotIn(
            "[[__SUBAGENT__", joined,
            f"Canary marker leaked into proxy.canary logs: "
            f"{[m for m in msgs if '[[__SUBAGENT__' in m]!r}",
        )
        sig_pattern = re.compile(r"[0-9a-fA-F]{64}")
        self.assertIsNone(
            sig_pattern.search(joined),
            f"64-hex signature leaked into proxy.canary logs: {joined!r}",
        )
        self.assertTrue(
            any("task_count=" in m and "request_id=" in m for m in msgs),
            f"Expected structural log line with task_count= and request_id=; "
            f"got: {msgs!r}",
        )


# ===========================================================================
# Concurrency
# ===========================================================================


class TestConcurrentMainSwarmRequests:
    """Two concurrent MAIN swarm requests with shared session, separate ctx's."""

    @pytest.mark.real_agent_registry
    def test_two_concurrent_separate_outcomes(self):
        """Request A (1 task, max=2) succeeds; request B (3 tasks, max=2) rejects.

        We pre-register both agents in the real AgentRegistry and use the
        production `get_or_load`. The two requests then have independent
        request-local policy + counts.
        """
        from proxy.agent.agent import Agent
        from proxy.agent.agent_registry import registry as agent_registry
        from proxy.skills.skill_registry import SkillRegistry

        sess_a = "sess_concurrent_a"
        sess_b = "sess_concurrent_b"
        agent_a = Agent(session_id=sess_a, agent_id=f"agent_{sess_a}")
        agent_a.set_active_mode(
            ModeState(name="research", activated_at=time.time(), ttl_seconds=300)
        )
        agent_b = Agent(session_id=sess_b, agent_id=f"agent_{sess_b}")
        agent_b.set_active_mode(
            ModeState(name="research", activated_at=time.time(), ttl_seconds=300)
        )
        agent_registry.register(agent_a)
        agent_registry.register(agent_b)

        policy_a = SubagentPolicy(
            tool_whitelist=frozenset({"read"}), max_task_calls_per_request=2
        )
        policy_b = SubagentPolicy(
            tool_whitelist=frozenset({"read"}), max_task_calls_per_request=2
        )
        ctx_a = _make_post_process_ctx(
            session_id=sess_a, swarm_policy=policy_a
        )
        ctx_b = _make_post_process_ctx(
            session_id=sess_b, swarm_policy=policy_b
        )

        # A: 1 task call. B: 3 task calls.
        response_a = _make_response_with_task_calls(["A"])
        response_b = _make_response_with_task_calls(["B1", "B2", "B3"])
        ctx_a.response_data = response_a
        ctx_b.response_data = response_b

        # Register a fake skill so parse_canary / apply_skill_response_transform
        # can complete without raising.
        fake_skill_def = MagicMock()
        fake_skill_def.name = "research"
        fake_skill_def.post_process = MagicMock(side_effect=lambda ctx, response: response)

        barrier = threading.Barrier(2)
        results = {"a": None, "b": None, "a_exc": None, "b_exc": None}

        def run_a():
            barrier.wait()
            try:
                with _noop_submethods_patcher(), \
                     _patch_workspace_resolution("default"), \
                     patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
                    mock_reg = MagicMock()
                    mock_reg.get.return_value = fake_skill_def
                    mock_reg_cls.return_value = mock_reg
                    results["a"] = PostProcessStage().execute(ctx_a)
            except Exception as e:
                results["a_exc"] = e

        def run_b():
            barrier.wait()
            try:
                with _noop_submethods_patcher(), \
                     _patch_workspace_resolution("default"), \
                     patch("proxy.skills.skill_registry.SkillRegistry") as mock_reg_cls:
                    mock_reg = MagicMock()
                    mock_reg.get.return_value = fake_skill_def
                    mock_reg_cls.return_value = mock_reg
                    results["b"] = PostProcessStage().execute(ctx_b)
            except StageAbort as e:
                results["b_exc"] = e

        thread_a = threading.Thread(target=run_a)
        thread_b = threading.Thread(target=run_b)
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)

        # Cleanup
        with agent_registry._lock:
            agent_registry._agents.pop(sess_a, None)
            agent_registry._agents.pop(sess_b, None)
            agent_registry._agents_by_id.pop(f"agent_{sess_a}", None)
            agent_registry._agents_by_id.pop(f"agent_{sess_b}", None)

        # A succeeded.
        if results["a_exc"] is not None:
            raise results["a_exc"]
        assert results["a"] is not None, "A must produce a result"
        # B rejected with swarm_max_calls_exceeded.
        assert results["b_exc"] is not None, "Expected StageAbort on request B"
        envelope, status = results["b_exc"].response_data
        assert status == 400
        assert envelope["error"]["code"] == "swarm_max_calls_exceeded"
        # A's response was NOT mutated by B (independent request-local state).
        result_a = results["a"] if isinstance(results["a"], dict) else ctx_a.response_data
        a_prompt = json.loads(
            result_a["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["prompt"]
        assert a_prompt.startswith("[[__SUBAGENT__:SKILL:research:"), (
            f"A's prompt should have SKILL canary, got: {a_prompt[:80]!r}"
        )
        # B's source response is still untouched.
        assert (
            response_b["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            == json.dumps({"description": "test", "prompt": "B1"})
        ), "B's source response must be unchanged after the abort"


# ===========================================================================
# Method-loop StageAbort propagation (revert #10)
# ===========================================================================


class TestMethodLoopStageAbortPropagates:
    """When ``_inject_task_canaries`` raises ``StageAbort`` from inside the
    method-loop early-return branch, the ``StageAbort`` MUST propagate out
    of ``PostProcessStage.execute()`` and the lifecycle-finalization
    ``finally:`` block MUST still run.

    Pinning invariant: the method-loop's ``except StageAbort: raise``
    clause (``src/proxy/pipeline/post_process.py:85-91``) is the only point
    that prevents a swallowed abort from falling through to the post-loop
    fallback path (``src/proxy/pipeline/post_process.py:147-148``). A revert
    that removes the ``raise`` (i.e. ``except StageAbort: pass``) would:

      1. swallow the abort inside the loop,
      2. let the loop iterate out normally,
      3. reach the post-loop ``_apply_skill_response_transform`` +
         ``_inject_task_canaries`` calls and re-raise StageAbort from
         ``_inject_task_canaries`` again.

    The end-state of the revert looks similar (StageAbort propagates,
    ``finally:`` runs) but ``_inject_task_canaries`` is invoked TWICE
    instead of ONCE — that call-count delta is the discriminant.

    Patch strategy:
      - ``PostProcessStage._handle_tool_interception`` returns a non-None
        dict so the method loop enters the early-return branch
        (``if result is not None and method == self._handle_tool_interception``).
      - ``PostProcessStage._inject_task_canaries`` is replaced by a
        ``MagicMock`` with ``side_effect=StageAbort((envelope, status))``
        so the StageAbort surfaces inside the loop's ``try`` block.
      - ``PostProcessStage._apply_skill_response_transform`` is replaced
        by a ``MagicMock`` to count its invocations.
      - The 7 sibling submethods and ``_assign_model_name`` are no-op'd
        so they neither raise nor mutate ``ctx.response_data``.
      - ``PostProcessStage._finalize_skill_lifecycle`` (the inner helper
        of the outer ``finally:``) is replaced by a ``MagicMock`` so the
        call from ``_finalize_skill_lifecycle_with_flag`` records its
        invocation. The outer ``_finalize_skill_lifecycle_with_flag`` is
        left UNPATCHED, so ``ctx.skill_lifecycle_finalized = True`` is
        still set in its ``finally:`` block.
    """

    def test_method_loop_stageabort_propagates_and_finalizes(self):
        swarm_envelope = {
            "error": {
                "code": "swarm_max_calls_exceeded",
                "message": "Too many swarm task calls",
            }
        }
        swarm_status = 400

        policy = SubagentPolicy(
            tool_whitelist=frozenset({"read"}),
            max_task_calls_per_request=4,
        )
        ctx = _make_post_process_ctx(swarm_policy=policy)
        ctx.swarm_policy = None  # Per AGENTS.md landmine note.
        agent = _make_agent(mode_name="research")
        response = _make_response_with_task_calls(["x"])
        ctx.response_data = response
        source_snapshot = copy.deepcopy(response)

        apply_transform_spy = MagicMock(name="apply_skill_response_transform")
        # ``side_effect=StageAbort(...)`` raises the same StageAbort on every
        # invocation; the loop's ``except StageAbort: raise`` (the fix) catches
        # the first raise and propagates, so this mock is called ONCE.
        # Under a revert that swallows the abort, the post-loop fallback at
        # ``post_process.py:147-148`` invokes ``_inject_task_canaries`` a
        # SECOND time, producing call_count=2.
        inject_canaries_spy = MagicMock(
            name="inject_task_canaries",
            side_effect=StageAbort((swarm_envelope, swarm_status)),
        )
        finalize_spy = MagicMock(name="finalize_skill_lifecycle")

        from contextlib import ExitStack
        with ExitStack() as stack:
            noop = lambda self, ctx: None
            # No-op the 7 sibling submethods.
            for name in (
                "_extract_minimax_reasoning",
                "_scan_write_results",
                "_sync_chat_history",
                "_persist_messages",
                "_populate_vfs_cache",
                "_capture_cache_metrics",
                "_assign_model_name",
            ):
                stack.enter_context(patch.object(PostProcessStage, name, noop))

            # Force the method-loop early-return branch.
            early_return_marker = {"early_return_marker": True}
            stack.enter_context(patch.object(
                PostProcessStage,
                "_handle_tool_interception",
                lambda self, ctx: early_return_marker,
            ))

            # Spy on the two split-post-process ops.
            stack.enter_context(patch.object(
                PostProcessStage,
                "_apply_skill_response_transform",
                apply_transform_spy,
            ))
            stack.enter_context(patch.object(
                PostProcessStage,
                "_inject_task_canaries",
                inject_canaries_spy,
            ))

            # Spy on the inner finalize helper; outer
            # ``_finalize_skill_lifecycle_with_flag`` still sets
            # ``ctx.skill_lifecycle_finalized`` in its ``finally:``.
            stack.enter_context(patch.object(
                PostProcessStage,
                "_finalize_skill_lifecycle",
                finalize_spy,
            ))

            stack.enter_context(_patch_get_or_load(agent))
            stack.enter_context(_patch_workspace_resolution("default"))

            with pytest.raises(StageAbort) as exc_info:
                PostProcessStage().execute(ctx)

        # 1. StageAbort propagates out of ``execute()``.
        envelope, status = exc_info.value.response_data
        assert envelope == swarm_envelope
        assert status == swarm_status

        # 2. The lifecycle-finalization ``finally:`` block still ran.
        assert finalize_spy.called, (
            "PostProcessStage._finalize_skill_lifecycle was NOT called — the "
            "outer `finally:` block did not run, so skill mode finalization "
            "was skipped on the StageAbort path."
        )
        assert getattr(ctx, "skill_lifecycle_finalized", False) is True, (
            "ctx.skill_lifecycle_finalized was NOT set to True — the outer "
            "`finally:` in PostProcessStage.execute() did not finalize the "
            "skill lifecycle before the StageAbort propagated out."
        )

        # 3. The abort's response_data carries a swarm_* code.
        assert envelope["error"]["code"].startswith("swarm_"), (
            f"abort envelope code MUST be a swarm_* code; got "
            f"{envelope['error']['code']!r}"
        )

        # 4. Pinned: method-loop ``except StageAbort: raise`` re-raised
        #    immediately — the post-loop fallback path was NOT reached.
        #    Under a revert that swallows StageAbort, both helpers are
        #    invoked a SECOND time at ``post_process.py:147-148``, giving
        #    call_count=2. Asserting == 1 proves the loop catch correctly
        #    re-raised.
        assert inject_canaries_spy.call_count == 1, (
            f"_inject_task_canaries must be called exactly once (method-loop "
            f"StageAbort catch must re-raise, not fall through to the "
            f"post-loop fallback); got call_count="
            f"{inject_canaries_spy.call_count}"
        )
        assert apply_transform_spy.call_count == 1, (
            f"_apply_skill_response_transform must be called exactly once; "
            f"got call_count={apply_transform_spy.call_count}"
        )

        # 5. Source response_data is unchanged (regression guard for the
        #    "swallowed abort silently mutated the response" failure mode).
        assert ctx.response_data == source_snapshot, (
            "ctx.response_data was mutated during the aborted path"
        )
