"""Tests for guard-blocked tool calls producing soft errors (not hard kills).

When MemoryToolGuard.check() blocks a tool call (e.g. proxy_ticket from a
bare sub-agent), the ToolPipeline now collects the error and injects it
into execute_fn as a synthetic {"error": ...} result. The LLM sees the
error as a tool result and can recover, instead of being hard-killed
(popped tool_calls + finish_reason=stop + early return).
"""
import json
from unittest.mock import MagicMock

import pytest

from proxy.tool_pipeline.tool_pipeline import (
    ToolPipeline,
    ToolContext,
    _stable_args_key,
)
from proxy.tool_pipeline.tool_call import ToolCall


def _make_ctx(session_id="sess-test", is_subagent=False, client_hint="", execute_fn=None):
    return ToolContext(
        session_id=session_id,
        body={"model": "test-model"},
        llama_body={"messages": [], "tools": [], "model": "test-model"},
        auth_headers={},
        original_model="test-model",
        upstream_url="http://upstream/v1",
        execute_fn=execute_fn,
        is_subagent=is_subagent,
        client_hint=client_hint,
    )


def _make_response_data(tool_calls):
    """Build a response_data with OpenAI-format tool_calls in the message."""
    native = []
    for name, args in tool_calls:
        native.append({
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": native,
            },
            "finish_reason": "tool_calls",
        }]
    }


def _mock_result(tool_calls=None, content=None, finish_reason="stop"):
    """Build a minimal result object compatible with _format()."""
    r = MagicMock()
    r.tool_calls = tool_calls
    r.content = content
    r.finish_reason = finish_reason
    r.messages_for_history = None
    return r


# ---------------------------------------------------------------------------
# Test 3: _stable_args_key strips injected keys
# ---------------------------------------------------------------------------

def test_stable_args_key_strips_injected_keys():
    key_with_injected = _stable_args_key(
        "proxy_ticket",
        {"action": "list_tasks", "session_id": "s1", "client_hint": "opencode"},
    )
    key_without = _stable_args_key(
        "proxy_ticket",
        {"action": "list_tasks"},
    )
    assert key_with_injected == key_without

    different_action = _stable_args_key(
        "proxy_ticket",
        {"action": "create_task"},
    )
    assert different_action != key_without


# ---------------------------------------------------------------------------
# Test 1: guard block produces soft error, not hard kill
# ---------------------------------------------------------------------------

def test_guard_block_produces_soft_error_not_hard_kill(monkeypatch):
    pipeline = ToolPipeline()

    def fake_check(tc, ctx):
        if tc.name == "proxy_ticket":
            return "proxy_ticket blocked by guard"
        return None
    monkeypatch.setattr(pipeline.guard, "check", fake_check)

    captured = {}

    def mock_execute(response_data, memory_calls, ctx, original_tool_calls_list=None, valid_params_by_tool=None):
        captured["called"] = True
        captured["blocked_errors"] = dict(pipeline._blocked_errors)
        captured["message_tool_calls"] = response_data["choices"][0]["message"].get("tool_calls")
        captured["finish_reason"] = response_data["choices"][0].get("finish_reason")
        return _mock_result()

    pipeline._execute_via_loop = mock_execute

    response_data = _make_response_data([("proxy_ticket", {"action": "list_tasks"})])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    # Flow continued to _execute_via_loop (guard did NOT return early)
    assert captured.get("called") is True
    # tool_calls NOT popped by the guard
    assert captured["message_tool_calls"] is not None
    assert len(captured["message_tool_calls"]) == 1
    # finish_reason NOT set to "stop" by the guard
    assert captured["finish_reason"] != "stop"
    # _blocked_errors populated with the error keyed by stable args
    assert len(captured["blocked_errors"]) == 1
    key = _stable_args_key("proxy_ticket", {"action": "list_tasks"})
    assert key in captured["blocked_errors"]
    assert captured["blocked_errors"][key] == "proxy_ticket blocked by guard"


# ---------------------------------------------------------------------------
# Test 4: guard block preserves tool_calls in message after full run()
# ---------------------------------------------------------------------------

def test_guard_block_preserves_tool_calls_in_message(monkeypatch):
    pipeline = ToolPipeline()

    def fake_check(tc, ctx):
        return "blocked by guard"
    monkeypatch.setattr(pipeline.guard, "check", fake_check)

    executed = {"called": False}

    def mock_execute(response_data, memory_calls, ctx, original_tool_calls_list=None, valid_params_by_tool=None):
        executed["called"] = True
        # Return a result that keeps tool_calls so _format() preserves them.
        return _mock_result(
            tool_calls=response_data["choices"][0]["message"]["tool_calls"],
            finish_reason="tool_calls",
        )

    pipeline._execute_via_loop = mock_execute

    response_data = _make_response_data([("proxy_ticket", {"action": "list_tasks"})])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    # Guard did NOT return early — _execute_via_loop was invoked
    assert executed["called"] is True
    message = response_data["choices"][0]["message"]
    # tool_calls still present in the final message (not popped by guard)
    assert message.get("tool_calls") is not None
    assert len(message["tool_calls"]) == 1
    # finish_reason reflects the result, not the guard's hard-kill "stop"
    assert response_data["choices"][0]["finish_reason"] != "stop"


# ---------------------------------------------------------------------------
# Test 5: non-blocked calls not affected
# ---------------------------------------------------------------------------

def test_non_blocked_calls_not_affected(monkeypatch):
    pipeline = ToolPipeline()

    def fake_check(tc, ctx):
        if tc.name == "proxy_ticket":
            return "proxy_ticket blocked"
        return None
    monkeypatch.setattr(pipeline.guard, "check", fake_check)

    captured = {}

    def mock_execute(response_data, memory_calls, ctx, original_tool_calls_list=None, valid_params_by_tool=None):
        captured["blocked_errors"] = dict(pipeline._blocked_errors)
        captured["memory_call_names"] = [tc.name for tc in memory_calls]
        return _mock_result()

    pipeline._execute_via_loop = mock_execute

    response_data = _make_response_data([
        ("search", {"query": "foo"}),
        ("proxy_ticket", {"action": "list_tasks"}),
    ])
    ctx = _make_ctx()

    pipeline.run(response_data, ctx)

    # Only the blocked tool's error is collected
    assert len(captured["blocked_errors"]) == 1
    assert _stable_args_key("proxy_ticket", {"action": "list_tasks"}) in captured["blocked_errors"]
    assert _stable_args_key("search", {"query": "foo"}) not in captured["blocked_errors"]
    # Both memory calls still flow through to _execute_via_loop
    assert "proxy_ticket" in captured["memory_call_names"]
    assert "search" in captured["memory_call_names"]


# ---------------------------------------------------------------------------
# Test 2: wrapped execute_fn returns error for blocked tool, passthrough otherwise
# ---------------------------------------------------------------------------

def test_guard_wrapped_execute_fn_returns_error_for_blocked_tool(monkeypatch):
    pipeline = ToolPipeline()
    pipeline._blocked_errors = {
        _stable_args_key("proxy_ticket", {"action": "list_tasks"}): "blocked by guard",
    }

    call_log = []

    def original_fn(name, arguments):
        call_log.append((name, arguments))
        return {"result": f"executed {name}"}

    captured = {}

    def fake_run_memory_tool_loop(**kwargs):
        captured["execute_fn"] = kwargs["execute_fn"]
        return _mock_result()

    monkeypatch.setattr("memory.tool_loop.run_memory_tool_loop", fake_run_memory_tool_loop)

    ctx = _make_ctx(execute_fn=original_fn)
    response_data = _make_response_data([("proxy_ticket", {"action": "list_tasks"})])
    memory_calls = [ToolCall(id="call_proxy_ticket", name="proxy_ticket", arguments={"action": "list_tasks"})]

    pipeline._execute_via_loop(response_data, memory_calls, ctx)

    wrapped_fn = captured["execute_fn"]

    # Blocked tool returns the synthetic error WITHOUT calling the original fn.
    # Args here include the proxy-injected keys (session_id, client_hint) which
    # run_memory_tool_loop would have added — the stable key must still match.
    result = wrapped_fn("proxy_ticket", {"action": "list_tasks", "session_id": "s1", "client_hint": "opencode"})
    assert result == {"error": "blocked by guard"}
    assert len(call_log) == 0

    # Non-blocked tool falls through to the original execute_fn.
    result2 = wrapped_fn("search", {"query": "foo"})
    assert result2 == {"result": "executed search"}
    assert len(call_log) == 1
    assert call_log[0][0] == "search"


# ---------------------------------------------------------------------------
# Test 6 (BATCH 2.1): end-to-end guard block with camelCase args (opencode)
# ---------------------------------------------------------------------------

def _opencode_proxy_ticket_schema():
    """A real proxy_ticket tool schema with `task_id` (snake_case) — the
    snake_case server side of a camelCase client (opencode). The client's
    LLM may emit `taskId` even though the server schema says `task_id`;
    the producer normalizes via valid_params, and the consumer must
    re-normalize so the key lookup matches."""
    return {
        "type": "function",
        "function": {
            "name": "proxy_ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "task_id": {"type": "string"},
                },
            },
        },
    }


def test_guard_block_matches_across_producer_and_consumer_with_camelcase_args(monkeypatch):
    """BATCH 2.1: producer keys _blocked_errors with NORMALIZED args
    (taskId -> task_id via _parse -> normalize_tool_arguments). Consumer
    (the wrapped execute_fn) was receiving RAW args from
    run_memory_tool_loop (which only does json.loads, no normalization),
    so for camelCase clients (opencode) the key lookup missed and the
    guard-blocked tool executed anyway. Fix: consumer re-normalizes
    via the same valid_params_by_tool. This test exercises the full
    path — real _parse, real normalize_tool_arguments, real guard — and
    asserts the synthetic error is returned (not the real close action).
    """
    pipeline = ToolPipeline()

    def fake_check(tc, ctx):
        if tc.name == "proxy_ticket":
            return "proxy_ticket blocked by guard (sub-agent cannot close)"
        return None
    monkeypatch.setattr(pipeline.guard, "check", fake_check)

    real_close_called = {"called": False}

    def original_fn(name, arguments):
        if name == "proxy_ticket" and arguments.get("action") == "close":
            real_close_called["called"] = True
        return {"result": f"executed {name}", "arguments": dict(arguments)}

    captured = {}

    def fake_run_memory_tool_loop(**kwargs):
        captured["execute_fn"] = kwargs["execute_fn"]
        return _mock_result()

    monkeypatch.setattr("memory.tool_loop.run_memory_tool_loop", fake_run_memory_tool_loop)

    response_data = _make_response_data([
        ("proxy_ticket", {"action": "close", "taskId": "abc"}),
    ])

    ctx = ToolContext(
        session_id="sess_b21",
        body={"model": "test-model"},
        llama_body={
            "messages": [],
            "tools": [_opencode_proxy_ticket_schema()],
            "model": "test-model",
        },
        auth_headers={},
        original_model="test-model",
        upstream_url="http://upstream/v1",
        execute_fn=original_fn,
        is_subagent=True,
        client_hint="opencode",
    )

    pipeline.run(response_data, ctx)

    assert captured.get("execute_fn") is not None, "run_memory_tool_loop was not invoked"
    wrapped_fn = captured["execute_fn"]

    # run_memory_tool_loop injects session_id + client_hint, AND passes the
    # raw (unnormalized) args. The wrapped fn must re-normalize taskId -> task_id
    # before computing the stable key, otherwise the lookup misses and the
    # guard-blocked action executes.
    result = wrapped_fn(
        "proxy_ticket",
        {"action": "close", "taskId": "abc", "session_id": "sess_b21", "client_hint": "opencode"},
    )
    assert result == {"error": "proxy_ticket blocked by guard (sub-agent cannot close)"}, (
        f"Guard-blocked close should have been short-circuited to synthetic "
        f"error. Got {result!r}. Pre-fix code would have returned the real "
        f"close result, executing the action anyway."
    )
    assert not real_close_called["called"], (
        "Real close action must NOT execute when guard blocks. "
        "Pre-fix code would have called original_fn for camelCase args."
    )

    # Idempotency: snake_case args (already normalized) still match.
    result2 = wrapped_fn(
        "proxy_ticket",
        {"action": "close", "task_id": "abc", "session_id": "sess_b21", "client_hint": "opencode"},
    )
    assert result2 == {"error": "proxy_ticket blocked by guard (sub-agent cannot close)"}
