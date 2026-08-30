"""US-017g: Discover-Then-Execute with Vague Criteria.

Verifies that the tool loop supports chaining: a discovery tool call
(e.g. search) returns results that are fed to the LLM, which then
emits execute tool calls (e.g. proxy_ticket) on the discovered items.
The loop must terminate correctly after execution.
"""

import json
import threading
import pytest
from unittest.mock import patch, MagicMock


def _make_tool_call(name, arguments=None, call_id=None):
    return {
        "name": name,
        "arguments": arguments or {},
        "id": call_id or f"call_{id(arguments or {}) // 1000}_0",
    }


def _make_openai_tool_call(name, call_id=None):
    return {
        "id": call_id or f"call_{id(name) // 1000}_0",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _make_requery_response(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def _base_kwargs(**overrides):
    defaults = {
        "tool_calls_list": [],
        "working_messages": [{"role": "user", "content": "close everything related to auth"}],
        "tools": [],
        "model": "qwen3-coder",
        "session_id": "test-session",
        "llama_url": "http://localhost:8080/v1/chat/completions",
        "body": {"model": "qwen3-coder"},
        "llama_body": {
            "model": "qwen3-coder",
            "messages": [{"role": "user", "content": "close everything related to auth"}],
        },
        "chat_histories": {},
        "chat_histories_lock": threading.Lock(),
    }
    defaults.update(overrides)
    return defaults


class TestDiscoverThenExecute:
    """US-017g: search discovers tasks, then proxy_ticket closes them."""

    def test_search_then_ticket_chain(self):
        from memory.tool_loop import run_memory_tool_loop

        search_results = {
            "results": [
                {"uuid": "task-aaa", "name": "fix auth bug", "status": "open"},
                {"uuid": "task-bbb", "name": "auth refactor", "status": "open"},
            ]
        }
        ticket_result = {"success": True, "closed": ["task-aaa", "task-bbb"]}

        call_log = []

        def mock_execute(tool_name, tool_args):
            call_log.append((tool_name, tool_args))
            if tool_name == "search":
                return search_results
            if tool_name == "proxy_ticket":
                return ticket_result
            return {}

        requery_turn = [0]

        def mock_requery(**kwargs):
            requery_turn[0] += 1
            msgs = kwargs.get("working_messages", [])
            if requery_turn[0] == 1:
                close_tc = _make_tool_call(
                    "proxy_ticket",
                    {"action": "close", "task_uuids": ["task-aaa", "task-bbb"]},
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": "Found 2 tasks, closing them now.",
                    "tool_calls": [_make_openai_tool_call("proxy_ticket")],
                }
                msgs.append(assistant_msg)
                from proxy.types import RequeryResult
                return RequeryResult(
                    requery_failed=False,
                    requery_data={},
                    working_messages=msgs,
                    _raw_tool_id_map={},
                    new_tool_calls=[close_tc],
                    requery_content="Found 2 tasks, closing them now.",
                    should_continue=True,
                )
            else:
                from proxy.types import RequeryResult
                return RequeryResult(
                    requery_failed=False,
                    requery_data={},
                    working_messages=msgs,
                    _raw_tool_id_map={},
                    new_tool_calls=None,
                    requery_content="Closed 2 auth-related tasks.",
                    should_continue=False,
                )

        search_tc = _make_tool_call(
            "search", {"query": "auth", "session_id": "test-session"}
        )

        with patch("proxy.requery.requery_llm", side_effect=mock_requery):
            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[search_tc],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=MagicMock(return_value=[_make_openai_tool_call("proxy_ticket")]),
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        assert len(call_log) == 2
        assert call_log[0][0] == "search"
        assert call_log[1][0] == "proxy_ticket"
        assert call_log[1][1].get("task_uuids") == ["task-aaa", "task-bbb"]
        assert result.content == "Closed 2 auth-related tasks."
        assert result.finish_reason == "stop"


class TestDiscoverFindsNothing:
    """US-017g: when discovery returns no results, no execute calls happen."""

    def test_empty_search_terminates_without_execute(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": []})

        def mock_requery(**kwargs):
            from proxy.types import RequeryResult
            msgs = kwargs.get("working_messages", [])
            return RequeryResult(
                requery_failed=False,
                requery_data={},
                working_messages=msgs,
                _raw_tool_id_map={},
                new_tool_calls=None,
                requery_content="No matching tasks found.",
                should_continue=False,
            )

        search_tc = _make_tool_call(
            "search", {"query": "nonexistent", "session_id": "test-session"}
        )

        with patch("proxy.requery.requery_llm", side_effect=mock_requery):
            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[search_tc],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=MagicMock(return_value=[]),
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        mock_execute.assert_called_once()
        assert mock_execute.call_args[0][0] == "search"
        assert result.content == "No matching tasks found."
        assert result.finish_reason == "stop"
