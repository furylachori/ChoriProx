"""Tests for enforce_context_limits — exercise the REAL estimator path.

These tests do NOT patch enforce_context_limits (unlike the 16 sites in
test_pipeline_stages.py and the patched test_context_limit_error_returns_413
in test_pipeline_errors.py). They construct realistic bodies + a pinned
model_cfg and assert the tools-cost-separate 413 guard fires (or doesn't)
based on the UN-TRIMMABLE portion (tool schemas + system) alone.

Model math (pinned): context_limit=32000, tokens_per_char=0.25 (always, per
model_registry.get_context_config). raw_limit=32000 tokens => 128000 chars.
"""

import json
from types import SimpleNamespace

import pytest

from proxy.context import enforce_context_limits


# Pinned model_cfg. _get_raw_context_limit reads .context_limit directly;
# get_context_config reads .context_limit/.budget_ratio/.raw_exchange_count
# but tokens_per_char is always 0.25 regardless.
MODEL_CFG = SimpleNamespace(
    context_limit=32000,
    budget_ratio=0.75,
    raw_exchange_count=8,
)
RAW_LIMIT_TOKENS = 32000
TPC = 0.25
RAW_LIMIT_CHARS = RAW_LIMIT_TOKENS / TPC  # 128000


def _huge_tools(n=20, per_tool_chars=30000):
    """Build ~20 tool entries whose serialized JSON is ~600KB (>> raw limit)."""
    pad = "x" * per_tool_chars
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": pad,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(n)
    ]


def _huge_tools_char_count():
    return len(json.dumps(_huge_tools(), separators=(",", ":")))


# ---------------------------------------------------------------------------
# Test A: single message + huge tools -> 413
# ---------------------------------------------------------------------------


def test_A_single_message_huge_tools_raises():
    """1 user message + ~600KB of tools: un-trimmable portion alone exceeds
    raw_limit -> ValueError. Messages count is irrelevant to the guard."""
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": _huge_tools(),
    }
    # sanity: un-trimmable tools alone are way over the char-equivalent limit
    assert _huge_tools_char_count() * TPC > RAW_LIMIT_TOKENS
    with pytest.raises(ValueError, match="un-trimmable tool/system schemas"):
        enforce_context_limits(body, MODEL_CFG)


# ---------------------------------------------------------------------------
# Test B: multi-message + huge tools -> 413 (the case minimal-restore missed)
# ---------------------------------------------------------------------------


def test_B_multi_message_huge_tools_raises():
    """5 messages + same huge tools: still raises. This is the common multi-turn
    case the deleted single-message-only branch would have missed."""
    body = {
        "messages": [
            {"role": "user", "content": f"message number {i}"}
            for i in range(5)
        ],
        "tools": _huge_tools(),
    }
    with pytest.raises(ValueError, match="un-trimmable tool/system schemas"):
        enforce_context_limits(body, MODEL_CFG)


# ---------------------------------------------------------------------------
# Test C: multi-message + small tools + large TRIMMABLE messages -> NOT 413
# ---------------------------------------------------------------------------


def test_C_large_messages_small_tools_does_not_raise():
    """False-positive guard. Several messages whose total far exceeds raw_limit,
    but tools/system are small. Must NOT raise: select_context()/V2 will trim
    the messages later. The guard keys only on the un-trimmable portion."""
    # 6 messages of ~30KB each = ~180KB total >> 128000 char-equivalent limit,
    # but they are all TRIMMABLE.
    big = "y" * 30000
    body = {
        "messages": [
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
        ],
        "tools": _small_tools(),
        "system": "small system prompt",
    }
    # sanity: messages alone exceed the char-equivalent limit
    total_msg_chars = sum(len(m["content"]) for m in body["messages"])
    assert total_msg_chars * TPC > RAW_LIMIT_TOKENS
    # but un-trimmable portion is small
    untrimmable = len(json.dumps(body["tools"], separators=(",", ":"))) + len(
        body["system"]
    )
    assert untrimmable * TPC < RAW_LIMIT_TOKENS
    # must not raise
    result = enforce_context_limits(body, MODEL_CFG)
    assert result is None


def _small_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# ---------------------------------------------------------------------------
# Boundary: un-trimmable exactly at / just over limit
# ---------------------------------------------------------------------------


def test_untrimmable_just_under_limit_does_not_raise():
    """Un-trimmable portion just under raw_limit must pass (strict >)."""
    # craft tools JSON to land just under the char-equivalent limit
    target_chars = int(RAW_LIMIT_CHARS) - 100  # 100 chars of headroom
    # base overhead of a minimal tools list JSON
    base = json.dumps([{"x": "z"}], separators=(",", ":"))
    pad_len = max(0, target_chars - len(base) - 30)  # leave room for structure
    tools = [{"x": "y" * pad_len}]
    body = {"messages": [], "tools": tools}
    untrimmable_chars = len(json.dumps(tools, separators=(",", ":")))
    assert untrimmable_chars * TPC < RAW_LIMIT_TOKENS
    assert enforce_context_limits(body, MODEL_CFG) is None


def test_missing_tools_and_system_does_not_raise():
    """No tools, no system -> un-trimmable = 0 -> never raises."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert enforce_context_limits(body, MODEL_CFG) is None
