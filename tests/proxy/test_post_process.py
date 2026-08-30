"""Unit tests for the 3 post-process ops split per spec §7.5.

These are pure unit tests — they call the ops directly with controlled inputs.
End-to-end lifecycle tests are in `test_lifecycle_finalize.py`.

The 3 ops (all in `proxy.mode`):
- `apply_skill_response_transform` — first op, calls skill.post_process() if active.
- `inject_task_canaries` — second op, resolves skill/workspace/bare canary
  and inserts into task tool calls.
- `finalize_skill_mode` — third op, classifies turn disposition and CAS-clears
  the active mode when terminal (COMPLETE / FAILED).

Per spec §7.4 / §7.5 / §12 acceptance criteria:
- Apply_skill_response_transform: passthrough when no active mode; skill callback
  invoked when mode present; callback exception is swallowed (returns input).
- Inject_task_canaries: passthrough when no task tool calls; canary injected
  when a task tool call is present.
- Finalize_skill_mode: CONTINUE keeps mode alive; COMPLETE/FAILED clears via
  CAS. Stale expected_mode → CAS returns False → mode preserved.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from proxy.agent.agent import Agent
from proxy.skills.skill_definition import ModeState
from proxy.mode import (
    TurnDisposition,
    apply_skill_response_transform,
    classify_turn_response,
    finalize_skill_mode,
    inject_task_canaries,
)
from proxy.skills.skill_registry import get_default_registry, reset_default_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(session_id="sess_test", is_subagent=False, messages=None):
    """Build a minimal RequestContext-like namespace object.

    Matches the attribute names the 3 ops read on ctx (per the
    `proxy.mode` docstrings).
    """
    return SimpleNamespace(
        session_id=session_id,
        is_subagent=is_subagent,
        messages=messages or [],
        client_hint="",
    )


def _make_agent(session_id="sess_test"):
    """Construct a real Agent with active_mode=None."""
    return Agent(session_id=session_id, agent_id=f"agent_{session_id}")


def _make_mode_state(name="ticket", ttl_seconds=300, **kwargs):
    """Construct a real ModeState."""
    return ModeState(
        name=name,
        activated_at=time.time(),
        ttl_seconds=ttl_seconds,
        **kwargs,
    )


@pytest.fixture
def fresh_agent_registry():
    """Reset the skill registry before each test to avoid leak from prior tests.

    Required because some earlier tests in this layer install custom skills via
    `register_skill(...)` which persists in the singleton.
    """
    reset_default_registry()
    yield get_default_registry()
    reset_default_registry()


# ===========================================================================
# apply_skill_response_transform
# ===========================================================================


def test_apply_skill_response_transform_passthrough_when_no_mode(
    fresh_agent_registry,
):
    """When no active mode is set, the op returns response_data unchanged."""
    agent = _make_agent()
    ctx = _make_ctx()
    response_data = {
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ]
    }

    result = apply_skill_response_transform(
        ctx=ctx, agent=agent, response_data=response_data,
    )

    assert result is response_data, "Expected same dict back when no active mode"


def test_apply_skill_response_transform_calls_skill_post_process(
    fresh_agent_registry,
):
    """When a mode is active and the skill exists, skill.post_process() is called
    and its return value becomes the new response_data."""
    agent = _make_agent()
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)

    # Inject a stub skill into the registry that returns a transformed response.
    transformed = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "[TICKET] hello"},
                "finish_reason": "stop",
            }
        ]
    }
    stub_skill = MagicMock()
    stub_skill.post_process.return_value = transformed
    # Bypass the trust gate by replacing the existing ticket entry's post_process
    # in place — SkillRegistry.get(name) returns the SkillDefinition, which
    # carries the post_process callable. We use the registry's internals to swap.
    registry = fresh_agent_registry
    existing = registry.get("ticket")
    assert existing is not None, "ticket builtin should be registered after populate_builtins"
    # The stub_skill's post_process will be invoked by apply_skill_response_transform
    # — it reads the SkillDefinition's `post_process` callable. We need to
    # swap that callable on the EXISTING SkillDefinition via object.__setattr__
    # (SkillDefinition is frozen=True but a bypass is needed for testing).
    object.__setattr__(existing, "post_process", stub_skill.post_process)

    ctx = _make_ctx()
    response_data = {
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ]
    }

    result = apply_skill_response_transform(
        ctx=ctx, agent=agent, response_data=response_data,
    )

    stub_skill.post_process.assert_called_once()
    assert result is transformed, "Expected the post_process return value as new response_data"


def test_apply_skill_response_transform_swallows_callback_exception(
    fresh_agent_registry,
):
    """If skill.post_process() raises, return response_data unchanged."""
    agent = _make_agent()
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)

    exploding_skill = MagicMock()
    exploding_skill.post_process.side_effect = RuntimeError("skill exploded")
    registry = fresh_agent_registry
    existing = registry.get("ticket")
    object.__setattr__(existing, "post_process", exploding_skill.post_process)

    ctx = _make_ctx()
    response_data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    result = apply_skill_response_transform(
        ctx=ctx, agent=agent, response_data=response_data,
    )

    assert result is response_data, "Failed skill callback must NOT corrupt response_data"
    exploding_skill.post_process.assert_called_once()


def test_apply_skill_response_transform_passthrough_for_non_dict(
    fresh_agent_registry,
):
    """When response_data is not a dict, the op returns it unchanged."""
    agent = _make_agent()
    agent.set_active_mode(_make_mode_state(name="ticket"))
    ctx = _make_ctx()

    # String is not a dict — must be returned unchanged.
    result = apply_skill_response_transform(
        ctx=ctx, agent=agent, response_data="not a dict",
    )
    assert result == "not a dict"

    # None is not a dict — must be returned unchanged.
    result = apply_skill_response_transform(
        ctx=ctx, agent=agent, response_data=None,
    )
    assert result is None


# ===========================================================================
# inject_task_canaries
# ===========================================================================


def test_inject_task_canaries_passthrough_when_no_mode(
    fresh_agent_registry,
):
    """When no active mode is set, inject_task_canaries returns response_data
    with the canary injected for any task tool calls.

    Note: per spec §7.5, inject_task_canaries is the SECOND op and ALWAYS
    inserts a canary into task tool calls (skill-aware when a skill is
    active, workspace-aware when a non-default workspace is set, bare
    otherwise). It is NOT a passthrough — but it does not mutate
    response_data structure when there are no task tool calls.
    """
    agent = _make_agent()
    ctx = _make_ctx()
    response_data = {
        "choices": [
            {"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
        ]
    }

    result = inject_task_canaries(
        ctx=ctx, agent=agent, response_data=response_data, mode=None,
    )

    # No task tool calls → response_data structurally unchanged.
    assert result is not None
    assert "choices" in result
    # First choice's message.content is preserved.
    assert result["choices"][0]["message"]["content"] == "done"


def test_inject_task_canaries_injects_into_task_calls(
    fresh_agent_registry,
):
    """When a task tool call is present and a skill is active, the skill's
    canary is prepended to the task prompt."""
    agent = _make_agent()
    agent.set_active_mode(_make_mode_state(name="ticket"))

    # Stub the registry to return a skill with a known canary.
    fake_canary = "[[__SUBAGENT__:SKILL:ticket:fake-sig]]"
    stub_skill = MagicMock()
    stub_skill.get_canary.return_value = fake_canary
    registry = fresh_agent_registry
    existing = registry.get("ticket")
    object.__setattr__(existing, "get_canary", stub_skill.get_canary)

    ctx = _make_ctx()
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_task_001",
                            "type": "function",
                            "function": {
                                "name": "task",
                                "arguments": '{"description": "investigate", "prompt": "look at this"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    result = inject_task_canaries(
        ctx=ctx, agent=agent, response_data=response_data, mode=None,
    )

    stub_skill.get_canary.assert_called_once()
    # The task prompt should now START with the canary marker.
    task_args = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    import json
    parsed = json.loads(task_args)
    assert parsed["prompt"].startswith("[[__SUBAGENT__:SKILL:ticket:"), (
        f"Expected task prompt to start with skill canary, got: {parsed['prompt'][:80]!r}"
    )


def test_inject_task_canaries_passthrough_when_no_task_calls(
    fresh_agent_registry,
):
    """When the response has no task tool calls, no canary injection happens."""
    agent = _make_agent()
    agent.set_active_mode(_make_mode_state(name="ticket"))

    ctx = _make_ctx()
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Just text, no tool calls.",
                    "tool_calls": [
                        {
                            "id": "call_read_001",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ]
    }

    result = inject_task_canaries(
        ctx=ctx, agent=agent, response_data=response_data, mode=None,
    )

    # The non-task tool call's arguments must be unchanged.
    tc = result["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "read"
    assert tc["function"]["arguments"] == "{}"


# ===========================================================================
# finalize_skill_mode
# ===========================================================================


def test_finalize_skill_mode_continue_keeps_mode():
    """CONTINUE disposition (tool_calls / length / null+tool_calls):
    clear_active_mode is NOT called. Mode preserved."""
    agent = _make_agent()
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)
    ctx = SimpleNamespace(
        response_data={
            "choices": [
                {
                    "message": {"tool_calls": [{"id": "c1"}]},
                    "finish_reason": None,  # None + tool_calls = CONTINUE
                }
            ]
        },
    )

    result = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    assert result is False
    assert agent.snapshot_mode() is mode, "Mode must be preserved on CONTINUE"


def test_finalize_skill_mode_complete_clears_mode():
    """COMPLETE disposition (stop): clear_active_mode is called with expected_mode.
    Mode cleared."""
    agent = _make_agent()
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)
    ctx = SimpleNamespace(
        response_data={"choices": [{"finish_reason": "stop"}]},
    )

    result = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    assert result is True
    assert agent.snapshot_mode() is None


def test_finalize_skill_mode_failed_clears_mode():
    """FAILED disposition (content_filter): clear_active_mode is called.
    Mode cleared."""
    agent = _make_agent()
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)
    ctx = SimpleNamespace(
        response_data={"choices": [{"finish_reason": "content_filter"}]},
    )

    result = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    assert result is True
    assert agent.snapshot_mode() is None


def test_finalize_skill_mode_cas_preserves_concurrent_replacement():
    """If the agent's active_mode has been replaced since the snapshot was taken,
    finalize_skill_mode does NOT clear (CAS returns False).

    Per spec §7.5: the CAS in clear_active_mode(expected_mode=...) is what
    prevents a newer concurrent mode from being wiped. finalize_skill_mode
    delegates that CAS to clear_active_mode and propagates its return value.

    Note: the CAS only fires when expected_mode is non-None. With expected_mode=None,
    clear_active_mode always clears (the existing legacy "always clear" path).
    """
    agent = _make_agent()
    # Capture a stale snapshot — mode A was the active mode at request entry.
    stale_snapshot = _make_mode_state(name="ticket")

    # Then install a new mode (this is the concurrent replacement: a
    # sub-agent tool-loop or a canary-driven activation that arrived
    # AFTER the snapshot capture).
    new_mode = _make_mode_state(name="work")
    agent.set_active_mode(new_mode)

    ctx = SimpleNamespace(
        response_data={"choices": [{"finish_reason": "stop"}]},
    )

    # Finalize with the stale snapshot — CAS mismatch: agent.active_mode is
    # new_mode, NOT stale_snapshot. clear_active_mode returns False.
    cleared = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=stale_snapshot)

    assert cleared is False
    # The current (new) mode is preserved.
    assert agent.snapshot_mode() is new_mode


def test_finalize_skill_mode_returns_cas_false_even_on_failed():
    """CAS mismatch (concurrent replacement) propagates the False upward
    even when the disposition is FAILED."""
    agent = _make_agent()
    stale_snapshot = _make_mode_state(name="ticket")
    new_mode = _make_mode_state(name="work")
    agent.set_active_mode(new_mode)

    ctx = SimpleNamespace(
        response_data={"choices": [{"finish_reason": "content_filter"}]},
    )

    cleared = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=stale_snapshot)

    assert cleared is False
    assert agent.snapshot_mode() is new_mode


# ===========================================================================
# classify_turn_response (full matrix — complements test_turn_disposition.py
# at the apply_post_process level, parameterized)
# ===========================================================================


@pytest.mark.parametrize(
    "data,expected_disposition,should_clear",
    [
        # Success terminal — COMPLETE → clear
        ({"choices": [{"finish_reason": "stop"}]}, TurnDisposition.COMPLETE, True),
        # Failure terminal — FAILED → clear
        ({"choices": [{"finish_reason": "content_filter"}]}, TurnDisposition.FAILED, True),
        # Unknown reason — FAILED → clear
        ({"choices": [{"finish_reason": "garbage_xyz"}]}, TurnDisposition.FAILED, True),
        # Continuation — keep
        ({"choices": [{"finish_reason": "tool_calls"}]}, TurnDisposition.CONTINUE, False),
        ({"choices": [{"finish_reason": "length"}]}, TurnDisposition.CONTINUE, False),
        (
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {"tool_calls": [{"id": "c1"}]},
                    }
                ]
            },
            TurnDisposition.CONTINUE,
            False,
        ),
        # Malformed — FAILED → clear
        ({}, TurnDisposition.FAILED, True),
        ({"choices": []}, TurnDisposition.FAILED, True),
        # data=None: no turn completed (preflight / streaming-deferred) —
        # lifecycle finalizer returns False without classifying/clearing,
        # preserving the mode (spec §7.5 Wave 4 fix).
        (None, None, False),
    ],
)
def test_finalize_skill_mode_full_matrix(
    fresh_agent_registry, data, expected_disposition, should_clear
):
    """End-to-end: classify_turn_response + finalize_skill_mode produce the
    expected keep/clear outcome across the spec §7.4 matrix."""
    agent = _make_agent()
    mode = _make_mode_state(name="ticket")
    agent.set_active_mode(mode)
    ctx = SimpleNamespace(response_data=data)

    # Sanity: classify_turn_response returns the expected disposition.
    # Skipped when expected_disposition is None (data=None short-circuits
    # in finalize_skill_mode before classification).
    if expected_disposition is not None:
        assert classify_turn_response(data) is expected_disposition

    # Run finalize_skill_mode and check the outcome.
    cleared = finalize_skill_mode(ctx=ctx, agent=agent, expected_mode=mode)

    if should_clear:
        assert cleared is True
        assert agent.snapshot_mode() is None
    else:
        assert cleared is False
        assert agent.snapshot_mode() is mode