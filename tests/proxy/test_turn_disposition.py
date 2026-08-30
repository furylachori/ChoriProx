"""Unit tests for classify_turn_response per spec §7.4.

The TurnDisposition enum + classify_turn_response function in
`proxy.mode` is the central classifier for one LLM turn response.
It feeds `finalize_skill_mode` (spec §7.5) and the canonical lifecycle
callbacks on builtins (`BaseSkill.should_deactivate`) and JSON skills
(`custom_skill_loader._default_should_deactivate`).

Revert-test matrix (spec §10.3):
- Restore stop-only `should_deactivate` → `content_filter`/malformed
  cleanup tests fail.
- Remove the classifier → these tests fail (callers return UNKNOWN
  instead of the correct disposition).
- Remove the CONTINUE branch for tool_calls+None → tool-loop tests
  silently clear the mode mid-turn.
"""

import pytest

from proxy.mode import TurnDisposition, classify_turn_response


# --- Standard finish_reason -> disposition mapping ---


def test_classify_stop_is_complete():
    assert classify_turn_response(
        {"choices": [{"finish_reason": "stop"}]}
    ) == TurnDisposition.COMPLETE


def test_classify_content_filter_is_failed():
    assert classify_turn_response(
        {"choices": [{"finish_reason": "content_filter"}]}
    ) == TurnDisposition.FAILED


def test_classify_tool_calls_is_continue():
    assert classify_turn_response(
        {"choices": [{"finish_reason": "tool_calls"}]}
    ) == TurnDisposition.CONTINUE


def test_classify_length_is_continue():
    assert classify_turn_response(
        {"choices": [{"finish_reason": "length"}]}
    ) == TurnDisposition.CONTINUE


# --- Null finish_reason variants ---


def test_classify_null_finish_reason_with_tool_calls_is_continue():
    """Provider-compatible: None finish_reason + non-empty tool_calls
    is a continuation signal (the model is asking for tool execution).
    Spec §7.4 — must be CONTINUE."""
    data = {
        "choices": [
            {
                "finish_reason": None,
                "message": {"tool_calls": [{"id": "1", "type": "function"}]},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.CONTINUE


def test_classify_null_finish_reason_no_tool_calls_is_complete():
    """Null finish_reason without tool_calls = no termination signal
    but no continuation signal either. Default = COMPLETE so the mode
    is cleared on the slow path (defensive)."""
    data = {
        "choices": [
            {
                "finish_reason": None,
                "message": {"content": "hi"},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.COMPLETE


# --- Unknown finish_reason: defers to FAILED ---


def test_classify_unknown_finish_reason_is_failed():
    """Any non-null finish_reason we don't recognize is FAILED. The
    classifier's policy is conservative — better to clear the mode and
    log than silently retain it."""
    assert classify_turn_response(
        {"choices": [{"finish_reason": "what_is_this"}]}
    ) == TurnDisposition.FAILED


# --- Missing / empty choices: FAILED (malformed response) ---


def test_classify_missing_choices_is_failed():
    assert classify_turn_response({}) == TurnDisposition.FAILED
    assert classify_turn_response({"choices": []}) == TurnDisposition.FAILED


# --- Malformed response shape: FAILED ---


def test_classify_malformed_response_is_failed():
    """None / non-dict / wrong inner types all classify as FAILED.
    Note this is the path that triggers mode cleanup in production —
    any crash here means the mode survives a malformed response."""
    assert classify_turn_response(None) == TurnDisposition.FAILED
    assert classify_turn_response("not a dict") == TurnDisposition.FAILED
    assert classify_turn_response(42) == TurnDisposition.FAILED
    assert classify_turn_response([]) == TurnDisposition.FAILED
    assert classify_turn_response({"choices": [None]}) == TurnDisposition.FAILED
    assert classify_turn_response({"choices": ["not a dict"]}) == TurnDisposition.FAILED


def test_classify_missing_message_key_is_complete():
    """choices[0] present but no "message" key → finish_reason logic
    runs with default empty message. With finish_reason=stop → COMPLETE."""
    assert classify_turn_response(
        {"choices": [{"finish_reason": "stop"}]}
    ) == TurnDisposition.COMPLETE


# --- Empty tool_calls list with null finish_reason must NOT be CONTINUE ---


def test_classify_empty_tool_calls_with_null_finish_reason_is_not_continue():
    """Edge case: tool_calls is an empty list. The classifier must
    NOT treat this as CONTINUE (would silently retain the mode for a
    response that has no continuation signal). Falls through to the
    no-termination-signal COMPLETE branch."""
    data = {
        "choices": [
            {
                "finish_reason": None,
                "message": {"tool_calls": []},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.COMPLETE


# --- Tool-calls-first precedence (non-empty message.tool_calls) ---


def test_classify_stop_with_tool_calls_is_continue():
    """REVERT TEST: a non-empty message.tool_calls list drives CONTINUE
    even when finish_reason=="stop" (providers like MiniMax emit stop
    alongside tool_calls). Flips to COMPLETE if the tool-calls-first
    reorder is reverted."""
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"tool_calls": [{"id": "1", "type": "function"}]},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.CONTINUE


def test_classify_length_with_tool_calls_is_continue():
    """Precedence pin: tool_calls + length is still CONTINUE."""
    data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"tool_calls": [{"id": "1"}]},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.CONTINUE


def test_classify_content_filter_with_tool_calls_is_failed():
    """Safety carve-out pin: content_filter is fail-closed even when
    tool_calls are present — a provider-redacted completion must not
    drive a continuation."""
    data = {
        "choices": [
            {
                "finish_reason": "content_filter",
                "message": {"tool_calls": [{"id": "1"}]},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.FAILED


def test_classify_unknown_reason_with_tool_calls_is_continue():
    """Tool-calls-first for non-carve-out reasons: an unknown non-null
    finish_reason with tool_calls is a continuation."""
    data = {
        "choices": [
            {
                "finish_reason": "what_is_this",
                "message": {"tool_calls": [{"id": "1"}]},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.CONTINUE


def test_classify_stop_with_empty_tool_calls_is_complete():
    """Empty tool_calls list is NOT a continuation; stop → COMPLETE."""
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"tool_calls": []},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.COMPLETE


def test_classify_stop_with_string_tool_calls_is_complete():
    """Truthy non-list tool_calls (a string) is NOT a continuation;
    stop → COMPLETE."""
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"tool_calls": "yes"},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.COMPLETE


def test_classify_null_reason_with_dict_tool_calls_is_complete():
    """Non-list (dict) tool_calls is NOT a continuation; None → COMPLETE."""
    data = {
        "choices": [
            {
                "finish_reason": None,
                "message": {"tool_calls": {"id": "1"}},
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.COMPLETE


def test_classify_stop_with_null_message_is_complete():
    """Whole message is null (non-dict): the isinstance guard prevents an
    AttributeError; falls through to stop → COMPLETE."""
    data = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": None,
            }
        ]
    }
    assert classify_turn_response(data) == TurnDisposition.COMPLETE


def test_default_should_deactivate_content_filter():
    """Custom JSON skills' default lifecycle must clear on content_filter
    (spec §7.4: FAILED → clear). This is the revert test for "Restore
    stop-only should_deactivate" from spec §10.3."""
    from proxy.skills.custom_skill_loader import _default_should_deactivate
    # Patching out dependencies is not needed — _default_should_deactivate is pure.
    assert _default_should_deactivate(None, "content_filter", name="x") is True
    assert _default_should_deactivate(None, "stop", name="x") is True
    assert _default_should_deactivate(None, "tool_calls", name="x") is False
    assert _default_should_deactivate(None, "length", name="x") is False


def test_default_should_deactivate_unknown_finish_reason():
    """Unknown non-null finish_reason must clear (FAILED branch)."""
    from proxy.skills.custom_skill_loader import _default_should_deactivate
    assert _default_should_deactivate(None, "mystery_reason", name="x") is True


def test_default_should_deactivate_none_finish_reason():
    """Null finish_reason with the JSON-loader callback (which has no
    access to the response's tool_calls) must clear (COMPLETE
    default — no termination signal but also no continuation signal)."""
    from proxy.skills.custom_skill_loader import _default_should_deactivate
    assert _default_should_deactivate(None, None, name="x") is True


# --- BaseSkill.should_deactivate mirror ---


def test_base_skill_should_deactivate_matches_classifier():
    """`BaseSkill.should_deactivate` is the canonical lifecycle for
    builtin skills (TicketSkill, WorkSkill). The mapping must agree
    with `classify_turn_response` for the finish_reason-only cases
    the callback signature exposes."""
    from proxy.skills.skill_definition import BaseSkill
    base = BaseSkill()
    # should_deactivate only reads finish_reason; the ctx is not used
    # by the default implementation, so pass None.
    # tool_calls / length → keep alive (CONTINUE)
    assert base.should_deactivate(None, "tool_calls") is False
    assert base.should_deactivate(None, "length") is False
    # stop / content_filter / null / unknown → clear
    assert base.should_deactivate(None, "stop") is True
    assert base.should_deactivate(None, "content_filter") is True
    assert base.should_deactivate(None, None) is True
    assert base.should_deactivate(None, "mystery_reason") is True


# --- finalize_skill_mode integration ---


def test_finalize_skill_mode_returns_false_on_continue():
    """On CONTINUE, the active mode is preserved; finalize_skill_mode
    returns False without ever calling clear_active_mode."""
    from unittest.mock import MagicMock

    from proxy.mode import finalize_skill_mode

    agent = MagicMock()
    ctx = MagicMock()
    ctx.response_data = {"choices": [{"finish_reason": "tool_calls"}]}
    expected_mode = MagicMock(name="expected_mode")

    result = finalize_skill_mode(ctx, agent, expected_mode)

    assert result is False
    agent.clear_active_mode.assert_not_called()


def test_finalize_skill_mode_clears_on_complete():
    """On COMPLETE, finalize_skill_mode delegates to
    clear_active_mode(expected_mode=...) and returns its result."""
    from unittest.mock import MagicMock

    from proxy.mode import finalize_skill_mode

    agent = MagicMock()
    agent.clear_active_mode.return_value = True
    ctx = MagicMock()
    ctx.response_data = {"choices": [{"finish_reason": "stop"}]}
    expected_mode = MagicMock(name="expected_mode")

    result = finalize_skill_mode(ctx, agent, expected_mode)

    assert result is True
    agent.clear_active_mode.assert_called_once_with(expected_mode=expected_mode)


def test_finalize_skill_mode_clears_on_failed():
    """On FAILED (malformed response), finalize_skill_mode still
    delegates to clear_active_mode. The CAS in clear_active_mode
    is what protects against concurrent replacement."""
    from unittest.mock import MagicMock

    from proxy.mode import finalize_skill_mode

    agent = MagicMock()
    agent.clear_active_mode.return_value = True
    ctx = MagicMock()
    # Malformed payload that classifies as FAILED (spec §7.5). Note that
    # response_data=None is a SEPARATE case (no turn completed / preflight)
    # where finalize_skill_mode short-circuits and preserves the mode —
    # covered by test_finalize_skill_mode_returns_false_on_none_response_data.
    ctx.response_data = {}
    expected_mode = MagicMock(name="expected_mode")

    result = finalize_skill_mode(ctx, agent, expected_mode)

    assert result is True
    agent.clear_active_mode.assert_called_once_with(expected_mode=expected_mode)


def test_finalize_skill_mode_returns_cas_false():
    """If clear_active_mode returns False (concurrent replacement
    detected), finalize_skill_mode propagates the False upward so
    the caller can record the mode was preserved."""
    from unittest.mock import MagicMock

    from proxy.mode import finalize_skill_mode

    agent = MagicMock()
    agent.clear_active_mode.return_value = False
    ctx = MagicMock()
    ctx.response_data = {"choices": [{"finish_reason": "content_filter"}]}
    expected_mode = MagicMock(name="expected_mode")

    result = finalize_skill_mode(ctx, agent, expected_mode)

    assert result is False
    agent.clear_active_mode.assert_called_once_with(expected_mode=expected_mode)


def test_finalize_skill_mode_returns_false_on_none_response_data():
    """When ctx.response_data is None, no real turn completed (preflight,
    deferred/streaming upstream, command/skill short-circuit). There is
    nothing to classify, so finalize_skill_mode preserves the mode:
    returns False and never calls clear_active_mode."""
    from unittest.mock import MagicMock

    from proxy.mode import finalize_skill_mode

    agent = MagicMock()
    ctx = MagicMock()
    ctx.response_data = None
    expected_mode = MagicMock(name="expected_mode")

    result = finalize_skill_mode(ctx, agent, expected_mode)

    assert result is False
    agent.clear_active_mode.assert_not_called()


# --- Parameterised matrix (spec §7.4 policy table) ---


@pytest.mark.parametrize(
    "data,expected",
    [
        # Success terminal
        ({"choices": [{"finish_reason": "stop"}]}, TurnDisposition.COMPLETE),
        # Failure terminal
        ({"choices": [{"finish_reason": "content_filter"}]}, TurnDisposition.FAILED),
        # Continuation: explicit and implicit
        ({"choices": [{"finish_reason": "tool_calls"}]}, TurnDisposition.CONTINUE),
        ({"choices": [{"finish_reason": "length"}]}, TurnDisposition.CONTINUE),
        (
            {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {"tool_calls": [{"id": "1"}]},
                    }
                ]
            },
            TurnDisposition.CONTINUE,
        ),
        # Tool-calls-first precedence (non-empty message.tool_calls)
        (
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"tool_calls": [{"id": "1", "type": "function"}]},
                    }
                ]
            },
            TurnDisposition.CONTINUE,  # revert test
        ),
        (
            {
                "choices": [
                    {"finish_reason": "length", "message": {"tool_calls": [{"id": "1"}]}}
                ]
            },
            TurnDisposition.CONTINUE,
        ),
        (
            {
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"tool_calls": [{"id": "1"}]},
                    }
                ]
            },
            TurnDisposition.FAILED,
        ),
        (
            {
                "choices": [
                    {
                        "finish_reason": "what_is_this",
                        "message": {"tool_calls": [{"id": "1"}]},
                    }
                ]
            },
            TurnDisposition.CONTINUE,
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"tool_calls": []}}]},
            TurnDisposition.COMPLETE,
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": {"tool_calls": "yes"}}]},
            TurnDisposition.COMPLETE,
        ),
        (
            {"choices": [{"finish_reason": None, "message": {"tool_calls": {"id": "1"}}}]},
            TurnDisposition.COMPLETE,
        ),
        (
            {"choices": [{"finish_reason": "stop", "message": None}]},
            TurnDisposition.COMPLETE,
        ),
        # Unknown reason → FAILED
        ({"choices": [{"finish_reason": "weird"}]}, TurnDisposition.FAILED),
        # Malformed → FAILED
        ({}, TurnDisposition.FAILED),
        ({"choices": []}, TurnDisposition.FAILED),
        (None, TurnDisposition.FAILED),
        ("string", TurnDisposition.FAILED),
    ],
)
def test_classify_turn_response_full_matrix(data, expected):
    assert classify_turn_response(data) == expected
