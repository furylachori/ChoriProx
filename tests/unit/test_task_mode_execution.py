"""US-017d: Agents MUST Execute in Task Mode, Not Meta-Plan.

Verifies that when the LLM returns tool_calls, the pipeline dispatches
them for execution (task-mode behavior). When the LLM returns only text
(no tool_calls), the response passes through unchanged — no meta-planning
interception or forced dispatch.

The contract is enforced by the pipeline's tool-call detection logic in
handle_tool_interception → ToolPipeline.run(), not by the LLM itself.
"""
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

from proxy.pipeline.tool_interception import handle_tool_interception


def _make_response_with_tool_calls(tool_calls):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
    }


def _make_response_text_only(text):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
    }


def _make_openai_tool_call(name, arguments, call_id=None):
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _base_kwargs(**overrides):
    defaults = {
        "body": {"model": "test-model"},
        "session_id": "test-sess-017d",
        "upstream_url": "http://localhost:9999/v1",
        "tool_parser": None,
        "auth_headers": {"Authorization": "Bearer test"},
        "original_model": "test-model",
        "model_cfg": MagicMock(memory_enabled=True),
    }
    defaults.update(overrides)
    return defaults


def _mock_tool_loop_result(content="Done."):
    result = MagicMock()
    result.tool_calls = []
    result.content = content
    result.finish_reason = "stop"
    result.messages_for_history = []
    return result


class TestUS017dToolCallsTriggerDispatch:
    """US-017d: tool_calls in LLM response MUST be dispatched, not ignored."""

    def test_native_tool_calls_enter_pipeline(self):
        """proxy_ticket create_tasks triggers ToolPipeline dispatch."""
        tc = _make_openai_tool_call("proxy_ticket", {
            "action": "create_tasks",
            "tasks": [{"name": "Fix bug A", "priority": "high"}],
        })
        data = _make_response_with_tool_calls([tc])

        with patch("memory.tool_loop.run_memory_tool_loop", return_value=_mock_tool_loop_result()) as mock_loop, \
             patch("proxy.tool_pipeline.tool_guard.MemoryToolGuard.check", return_value=None):
            handle_tool_interception(data, **_base_kwargs())

        mock_loop.assert_called_once()

    def test_multiple_tool_calls_dispatched_in_single_pass(self):
        """Multiple tool_calls in one response are dispatched together."""
        tc1 = _make_openai_tool_call("proxy_ticket", {
            "action": "create_tasks",
            "tasks": [{"name": "Fix bug A"}],
        })
        tc2 = _make_openai_tool_call("proxy_ticket", {
            "action": "create_tasks",
            "tasks": [{"name": "Fix bug B"}],
        })
        data = _make_response_with_tool_calls([tc1, tc2])

        with patch("memory.tool_loop.run_memory_tool_loop", return_value=_mock_tool_loop_result("2 tasks created.")) as mock_loop, \
             patch("proxy.tool_pipeline.tool_guard.MemoryToolGuard.check", return_value=None):
            handle_tool_interception(data, **_base_kwargs())

        mock_loop.assert_called_once()
        call_args = mock_loop.call_args
        assert len(call_args[1]["tool_calls_list"]) == 2


class TestUS017dMetaPlanningNotDispatched:
    """US-017d: plain text without tool_calls MUST pass through unchanged."""

    def test_meta_planning_text_returns_unchanged(self):
        """'Should I create tasks?' is returned as-is, not dispatched."""
        meta_text = "Should I create tasks for this? Here's what I would do: ..."
        data = _make_response_text_only(meta_text)

        with patch("memory.tool_loop.run_memory_tool_loop") as mock_loop:
            result = handle_tool_interception(data, **_base_kwargs())

        mock_loop.assert_not_called()
        assert result["choices"][0]["message"]["content"] == meta_text
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_empty_content_no_dispatch(self):
        """Empty content with no tool_calls passes through without dispatch."""
        data = _make_response_text_only("")

        with patch("memory.tool_loop.run_memory_tool_loop") as mock_loop:
            result = handle_tool_interception(data, **_base_kwargs())

        mock_loop.assert_not_called()
        assert result["choices"][0]["finish_reason"] == "stop"


class TestUS017dWorkModeDispatch:
    """US-017d: work mode agents dispatch tool_calls identically."""

    def test_work_mode_proxy_work_dispatched(self):
        """proxy_work start_work in work mode triggers dispatch."""
        tc = _make_openai_tool_call("proxy_work", {
            "action": "start_work",
            "short_hash": "abc12345",
        })
        data = _make_response_with_tool_calls([tc])

        with patch("memory.tool_loop.run_memory_tool_loop", return_value=_mock_tool_loop_result("Work started.")) as mock_loop, \
             patch("proxy.tool_pipeline.tool_guard.MemoryToolGuard.check", return_value=None):
            handle_tool_interception(data, **_base_kwargs())

        mock_loop.assert_called_once()
