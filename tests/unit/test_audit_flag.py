"""US-017h: Audit and Flag — Analysis With Action.

Verifies that the tool loop supports analysis → action chaining where
the analysis step is multi-criteria (e.g. evaluating staleness). The
LLM's analysis results from one iteration must be available to
subsequent tool calls so the agent can act on its findings.
"""

import copy
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


def _make_openai_tc(name, args, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)},
    }


def _base_working_messages(user_content, tool_calls=None):
    msgs = [{"role": "user", "content": user_content}]
    if tool_calls:
        msgs.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
    return msgs


def _base_kwargs(**overrides):
    defaults = {
        "tool_calls_list": [],
        "working_messages": [{"role": "user", "content": "review all open bugs and close the stale ones"}],
        "tools": [],
        "model": "qwen3-coder",
        "session_id": "test-session",
        "llama_url": "http://localhost:8080/v1/chat/completions",
        "body": {"model": "qwen3-coder"},
        "llama_body": {
            "model": "qwen3-coder",
            "messages": [{"role": "user", "content": "review all open bugs and close the stale ones"}],
        },
        "chat_histories": {},
        "chat_histories_lock": threading.Lock(),
    }
    defaults.update(overrides)
    return defaults


class TestAuditAnalysisThenAction:
    """US-017h: search finds bugs, LLM analyzes staleness, then dismisses stale ones."""

    def test_search_analyze_dismiss_chain(self):
        from memory.tool_loop import run_memory_tool_loop

        search_args = {"query": "open bugs", "session_id": "test-session"}
        bugs_found = {
            "results": [
                {"uuid": "bug-001", "title": "old login crash", "status": "open", "created_at": "2024-01-01T00:00:00Z"},
                {"uuid": "bug-002", "title": "recent UI glitch", "status": "open", "created_at": "2026-06-09T00:00:00Z"},
                {"uuid": "bug-003", "title": "stale timeout error", "status": "open", "created_at": "2024-03-15T00:00:00Z"},
            ]
        }
        dismiss_result = {"action": "resolve_bug", "bug_uuid": "bug-001", "bug_status": "dismissed"}
        dismiss_result_2 = {"action": "resolve_bug", "bug_uuid": "bug-003", "bug_status": "dismissed"}

        call_log = []

        def mock_execute(tool_name, tool_args):
            call_log.append((tool_name, tool_args))
            if tool_name == "search":
                return bugs_found
            if tool_name == "proxy_ticket":
                bug_id = tool_args.get("bug_uuid")
                if bug_id == "bug-001":
                    return dismiss_result
                if bug_id == "bug-003":
                    return dismiss_result_2
            return {}

        requery_turn = [0]

        def mock_requery(**kwargs):
            requery_turn[0] += 1
            msgs = kwargs.get("working_messages", [])
            if requery_turn[0] == 1:
                dismiss_tc_1 = _make_tool_call(
                    "proxy_ticket",
                    {"action": "resolve_bug", "bug_uuid": "bug-001", "resolution": "dismissed", "reason": "stale: over 1 year old"},
                )
                dismiss_tc_2 = _make_tool_call(
                    "proxy_ticket",
                    {"action": "resolve_bug", "bug_uuid": "bug-003", "resolution": "dismissed", "reason": "stale: over 1 year old"},
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": "Analyzed 3 bugs. Bug-001 and bug-003 are stale (>1 year). Bug-002 is recent. Dismissing stale ones.",
                    "tool_calls": [_make_openai_tool_call("proxy_ticket")],
                }
                msgs.append(assistant_msg)
                from proxy.types import RequeryResult
                return RequeryResult(
                    requery_failed=False,
                    requery_data={},
                    working_messages=msgs,
                    _raw_tool_id_map={},
                    new_tool_calls=[dismiss_tc_1, dismiss_tc_2],
                    requery_content="Analyzed 3 bugs. Bug-001 and bug-003 are stale (>1 year). Bug-002 is recent. Dismissing stale ones.",
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
                    requery_content="Dismissed 2 stale bugs (bug-001, bug-003). Kept bug-002 (recent).",
                    should_continue=False,
                )

        search_tc = _make_tool_call("search", search_args, call_id="call_search_0")
        assistant_tc = _make_openai_tc("search", search_args, "call_search_0")
        working = _base_working_messages(
            "review all open bugs and close the stale ones",
            tool_calls=[assistant_tc],
        )

        with patch("proxy.requery.requery_llm", side_effect=mock_requery):
            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[search_tc],
                    working_messages=working,
                    llama_body={"model": "qwen3-coder", "messages": working},
                    execute_fn=mock_execute,
                    make_tool_calls_fn=MagicMock(return_value=[_make_openai_tool_call("proxy_ticket")]),
                    parse_tools_fn=MagicMock(return_value=[]),
                    existing_tool_ids=["call_search_0"],
                )
            )

        assert len(call_log) == 3
        assert call_log[0][0] == "search"
        assert call_log[1][0] == "proxy_ticket"
        assert call_log[1][1].get("bug_uuid") == "bug-001"
        assert call_log[2][0] == "proxy_ticket"
        assert call_log[2][1].get("bug_uuid") == "bug-003"
        assert result.content == "Dismissed 2 stale bugs (bug-001, bug-003). Kept bug-002 (recent)."
        assert result.finish_reason == "stop"


class TestAnalysisResultsAvailableToNextIteration:
    """US-017h: analysis results from iteration N are in working_messages for iteration N+1."""

    def test_analysis_context_carries_forward(self):
        from memory.tool_loop import run_memory_tool_loop

        search_args = {"query": "stale bugs", "session_id": "test-session"}
        bugs_found = {
            "results": [
                {"uuid": "bug-100", "title": "ancient bug", "status": "open", "created_at": "2023-06-01T00:00:00Z"},
            ]
        }
        dismiss_result = {"action": "resolve_bug", "bug_uuid": "bug-100", "bug_status": "dismissed"}

        captured_working_msgs = []

        def mock_execute(tool_name, tool_args):
            if tool_name == "search":
                return bugs_found
            if tool_name == "proxy_ticket":
                return dismiss_result
            return {}

        requery_turn = [0]

        def mock_requery(**kwargs):
            requery_turn[0] += 1
            msgs = kwargs.get("working_messages", [])
            captured_working_msgs.append(copy.deepcopy(msgs))

            if requery_turn[0] == 1:
                dismiss_tc = _make_tool_call(
                    "proxy_ticket",
                    {"action": "resolve_bug", "bug_uuid": "bug-100", "resolution": "dismissed", "reason": "stale"},
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": "Found 1 stale bug. Dismissing it.",
                    "tool_calls": [_make_openai_tool_call("proxy_ticket")],
                }
                msgs.append(assistant_msg)
                from proxy.types import RequeryResult
                return RequeryResult(
                    requery_failed=False,
                    requery_data={},
                    working_messages=msgs,
                    _raw_tool_id_map={},
                    new_tool_calls=[dismiss_tc],
                    requery_content="Found 1 stale bug. Dismissing it.",
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
                    requery_content="Dismissed bug-100.",
                    should_continue=False,
                )

        search_tc = _make_tool_call("search", search_args, call_id="call_search_0")
        assistant_tc = _make_openai_tc("search", search_args, "call_search_0")
        working = _base_working_messages(
            "review all open bugs and close the stale ones",
            tool_calls=[assistant_tc],
        )

        with patch("proxy.requery.requery_llm", side_effect=mock_requery):
            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[search_tc],
                    working_messages=working,
                    llama_body={"model": "qwen3-coder", "messages": working},
                    execute_fn=mock_execute,
                    make_tool_calls_fn=MagicMock(return_value=[_make_openai_tool_call("proxy_ticket")]),
                    parse_tools_fn=MagicMock(return_value=[]),
                    existing_tool_ids=["call_search_0"],
                )
            )

        assert len(captured_working_msgs) == 2

        second_iter_msgs = captured_working_msgs[1]
        tool_result_msgs = [m for m in second_iter_msgs if m.get("role") == "tool"]
        assert len(tool_result_msgs) >= 1
        assert "bug-100" in tool_result_msgs[0]["content"]

        assistant_msgs = [m for m in second_iter_msgs if m.get("role") == "assistant" and m.get("content")]
        has_analysis = any("stale" in m["content"].lower() for m in assistant_msgs)
        assert has_analysis

        assert result.content == "Dismissed bug-100."
        assert result.finish_reason == "stop"
