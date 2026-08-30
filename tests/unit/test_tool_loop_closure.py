"""Slice 3 closure phase — unit tests.

Spec §13 reference where no US-XXX exists.
"""
import json
import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_tool_call(name, arguments=None, call_id=None):
    return {
        "name": name,
        "arguments": arguments or {},
        "id": call_id or "call_closure_test",
    }


def _make_openai_tool_call(name, call_id=None):
    return {
        "id": call_id or "call_closure_openai",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _make_requery_response(content="", tool_calls=None, finish_reason="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _loop_kwargs(**overrides):
    recall_tc = _make_tool_call("recall", {"semantic_targets": ["x"], "keywords": ["x"]})
    openai_recall = _make_openai_tool_call("recall")
    defaults = {
        "tool_calls_list": [recall_tc],
        "working_messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [openai_recall]},
        ],
        "tools": [],
        "model": "test-model",
        "session_id": "sess-closure-finish",
        "llama_url": "http://localhost:8080/v1/chat/completions",
        "body": {"model": "test-model"},
        "llama_body": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "chat_histories": {},
        "chat_histories_lock": threading.Lock(),
        "execute_fn": MagicMock(return_value={"results": []}),
        "make_tool_calls_fn": MagicMock(return_value=[_make_openai_tool_call("recall")]),
        "parse_tools_fn": MagicMock(return_value=[]),
    }
    defaults.update(overrides)
    return defaults


class TestToolLoopResultFields:
    def test_default_fields(self):
        from memory.tool_loop import (
            ToolLoopResult,
            TERMINATION_MODEL_COMPLETED,
        )
        r = ToolLoopResult()
        assert r.needs_closure is False
        assert r.termination_reason == TERMINATION_MODEL_COMPLETED
        assert r.closure_source is None

    def test_explicit_construction_round_trips(self):
        from memory.tool_loop import (
            ToolLoopResult,
            TERMINATION_REQUERY_FAILED,
        )
        r = ToolLoopResult(
            termination_reason=TERMINATION_REQUERY_FAILED,
            closure_source="fallback",
            execution_records=[{"tool": "x", "operation": "y", "status": "error"}],
        )
        assert r.termination_reason == TERMINATION_REQUERY_FAILED
        assert r.closure_source == "fallback"
        assert r.execution_records == [
            {"tool": "x", "operation": "y", "status": "error"}
        ]


class TestBracketLedgerDetection:
    def test_bracket_ledger_at_position_zero_detected(self):
        from memory.tool_loop import _is_bracket_ledger_content
        assert _is_bracket_ledger_content("[create_tasks: error] db timeout") is True

    def test_bracket_ledger_after_blank_line_detected(self):
        from memory.tool_loop import _is_bracket_ledger_content
        assert _is_bracket_ledger_content("\n[create_tasks: error] msg") is True

    def test_clean_prose_not_detected(self):
        from memory.tool_loop import _is_bracket_ledger_content
        assert _is_bracket_ledger_content("The task was created successfully.") is False

    def test_brackets_without_colon_not_detected(self):
        from memory.tool_loop import _is_bracket_ledger_content
        assert _is_bracket_ledger_content("[brackets without colon] prose") is False

    def test_empty_or_none_returns_false(self):
        from memory.tool_loop import _is_bracket_ledger_content
        assert _is_bracket_ledger_content(None) is False
        assert _is_bracket_ledger_content("") is False

    def test_full_observed_ledger_at_boundary(self):
        from memory.tool_loop import _is_bracket_ledger_content
        assert _is_bracket_ledger_content(
            "[create_tasks: success] hash=deadbeef name=Fix the bug"
        ) is True


class TestCompactExecutionLedger:
    def test_includes_created_hash_priority_name(self):
        from memory.tool_loop import build_compact_execution_ledger
        records = [
            {
                "tool": "proxy_ticket", "operation": "create_tasks",
                "status": "success", "iteration": 1, "attempt": 1,
                "error": None,
                "data": {"created": [{"short_hash": "abc12345",
                                     "priority": "high",
                                     "name": "T1"}]},
            },
        ]
        out = build_compact_execution_ledger(records)
        assert "Recorded 1 operation(s)" in out
        assert "hash=abc12345" in out
        assert "priority=high" in out
        assert "name=T1" in out

    def test_caps_characters_with_omission_marker(self):
        from memory.tool_loop import build_compact_execution_ledger
        records = [
            {
                "tool": "x",
                "operation": f"op{i}",
                "status": "success",
                "iteration": i,
                "attempt": 1,
                "error": None,
                "data": {"created": [
                    {"short_hash": f"deadbe{i:02d}", "priority": "high",
                     "name": f"task-with-a-long-name-{i}-suffix"}
                ]},
            }
            for i in range(50)
        ]
        out = build_compact_execution_ledger(records, max_chars=500)
        assert len(out) <= 500
        assert "…" in out
        assert "Recorded 50 operation(s)" in out

    def test_includes_attempt_code_message_for_failures(self):
        from memory.tool_loop import build_compact_execution_ledger
        records = [
            {
                "tool": "proxy_ticket", "operation": "create_tasks",
                "status": "error", "iteration": 1, "attempt": 2,
                "error": {
                    "code": "invalid_input",
                    "message": "create_tasks requires a non-empty tasks array",
                },
                "data": None,
            },
        ]
        out = build_compact_execution_ledger(records)
        assert "attempt=2" in out
        assert "code=invalid_input" in out
        assert "create_tasks requires a non-empty tasks array" in out

    def test_tolerates_legacy_recall_records(self):
        from memory.tool_loop import build_compact_execution_ledger
        records = [
            {
                "tool": "recall", "operation": "search",
                "status": "success", "iteration": 1, "attempt": 1,
                "error": None,
                "data": {"results": [], "total_matches": 0},
            },
        ]
        out = build_compact_execution_ledger(records)
        assert "recall:search:success" in out

    def test_tolerates_malformed_record(self):
        from memory.tool_loop import build_compact_execution_ledger
        records = [
            {"status": "error"},
            None,
            "not a dict",
            {"tool": "ok", "operation": "noop", "status": "success",
             "iteration": 1, "attempt": 1, "error": None, "data": None},
        ]
        out = build_compact_execution_ledger(records)
        assert "ok:noop:success" in out

    def test_empty_records_returns_marker(self):
        from memory.tool_loop import build_compact_execution_ledger
        assert build_compact_execution_ledger([]) == "(no operations recorded)"
        assert build_compact_execution_ledger(None) == "(no operations recorded)"


class TestDeterministicFallback:
    def _rec(self, status, tool="proxy_ticket", operation="create_tasks",
             attempt=1, error=None, data=None):
        return {
            "tool": tool, "operation": operation, "status": status,
            "iteration": 1, "attempt": attempt, "error": error, "data": data,
        }

    def test_errors_with_successes_is_partial(self):
        from memory.tool_loop import render_deterministic_fallback
        recs = [
            self._rec("success", data={"created": [
                {"short_hash": "abc12345", "priority": "high",
                 "name": "T1"}]}),
            self._rec("error", attempt=1, error={
                "code": "invalid_input",
                "message": "create_tasks requires a non-empty tasks array"}),
        ]
        out = render_deterministic_fallback(recs, "iteration_limit")
        assert out.startswith("PARTIAL")
        assert "hash=abc12345" in out
        assert "code=invalid_input" in out
        assert "create_tasks requires a non-empty tasks array" in out
        assert "Termination: iteration_limit" in out

    def test_errors_only_is_failed(self):
        from memory.tool_loop import render_deterministic_fallback
        out = render_deterministic_fallback(
            [self._rec("error", error={"code": "x", "message": "boom"})],
            "requery_failed",
        )
        assert out.startswith("FAILED")
        assert "boom" in out

    def test_successes_only_is_complete(self):
        from memory.tool_loop import render_deterministic_fallback
        out = render_deterministic_fallback(
            [self._rec("success", data={"created": [
                {"short_hash": "abc", "priority": "low", "name": "T"}]})],
            "model_completed",
        )
        assert out.startswith("COMPLETE")

    def test_no_records_is_failed(self):
        from memory.tool_loop import render_deterministic_fallback
        out = render_deterministic_fallback([], "internal_error")
        assert out.startswith("FAILED")
        assert "No confirmed operations" in out

    def test_non_create_success_is_listed_as_confirmed(self):
        from memory.tool_loop import render_deterministic_fallback
        out = render_deterministic_fallback(
            [self._rec("success", tool="recall", operation="search",
                       data={"results": [{"x": 1}], "total_matches": 1})],
            "iteration_limit",
        )
        assert out.startswith("COMPLETE")
        assert "recall search returned 1 result(s)" in out

    def test_oversized_report_is_capped_with_omission_marker(self):
        from memory.tool_loop import render_deterministic_fallback
        big_records = []
        for i in range(80):
            big_records.append(self._rec(
                "error", attempt=i + 1,
                error={"code": "tool_execution_failed",
                       "message": "x" * 100 + str(i)},
            ))
        out = render_deterministic_fallback(big_records, "iteration_limit")
        assert out.startswith("FAILED")
        assert "omitted" in out
        assert len(out) <= 2200

    def test_observed_error_error_success_is_partial_with_details(self):
        from memory.tool_loop import render_deterministic_fallback
        recs = [
            self._rec("error", attempt=1, error={
                "code": "invalid_input",
                "message": "create_tasks requires a non-empty tasks array"}),
            self._rec("error", attempt=2, error={
                "code": "invalid_input",
                "message": "create_tasks requires a non-empty tasks array"}),
            self._rec("success", data={"created": [
                {"short_hash": "cafef00d", "priority": "high",
                 "name": "Fix the bug"}]}),
        ]
        out = render_deterministic_fallback(recs, "requery_failed")
        assert out.startswith("PARTIAL")
        assert "hash=cafef00d" in out
        assert "name=Fix the bug" in out
        assert out.count("code=invalid_input") == 2
        assert out.count(
            "create_tasks requires a non-empty tasks array"
        ) == 2
        assert "No confirmed operations" not in out

    def test_failure_lines_use_prose_not_bracket_syntax(self):
        from memory.tool_loop import render_deterministic_fallback
        out = render_deterministic_fallback(
            [self._rec("error", attempt=1, error={
                "code": "invalid_input",
                "message": "create_tasks requires a non-empty tasks array",
            })],
            "requery_failed",
        )
        assert "proxy_ticket create_tasks failed (attempt 1) code=invalid_input" in out
        assert "reason: create_tasks requires a non-empty tasks array" in out
        assert "[proxy_ticket" not in out
        assert "[create_tasks" not in out


class TestCleanCloseSkipEligible:
    def test_clean_path_skips_closure(self):
        from memory.tool_loop import (
            _clean_close_skip_eligible,
            TERMINATION_MODEL_COMPLETED,
        )
        assert _clean_close_skip_eligible(
            content="Here is the answer.",
            termination_reason=TERMINATION_MODEL_COMPLETED,
            new_tool_calls=[],
        ) is True

    def test_bracket_ledger_content_does_not_skip(self):
        from memory.tool_loop import (
            _clean_close_skip_eligible,
            TERMINATION_MODEL_COMPLETED,
        )
        assert _clean_close_skip_eligible(
            content="[create_tasks: error] boom",
            termination_reason=TERMINATION_MODEL_COMPLETED,
            new_tool_calls=[],
        ) is False

    def test_empty_content_does_not_skip(self):
        from memory.tool_loop import (
            _clean_close_skip_eligible,
            TERMINATION_MODEL_COMPLETED,
        )
        assert _clean_close_skip_eligible(
            content="",
            termination_reason=TERMINATION_MODEL_COMPLETED,
            new_tool_calls=[],
        ) is False

    def test_iteration_limit_does_not_skip(self):
        from memory.tool_loop import (
            _clean_close_skip_eligible,
            TERMINATION_ITERATION_LIMIT,
        )
        assert _clean_close_skip_eligible(
            content="anything",
            termination_reason=TERMINATION_ITERATION_LIMIT,
            new_tool_calls=[],
        ) is False

    def test_tool_calls_present_does_not_skip(self):
        from memory.tool_loop import (
            _clean_close_skip_eligible,
            TERMINATION_MODEL_COMPLETED,
        )
        assert _clean_close_skip_eligible(
            content="anything",
            termination_reason=TERMINATION_MODEL_COMPLETED,
            new_tool_calls=[{"name": "x"}],
        ) is False

    def test_closing_bracket_after_separator_does_not_skip(self):
        from memory.tool_loop import (
            _clean_close_skip_eligible,
            TERMINATION_MODEL_COMPLETED,
        )
        assert _clean_close_skip_eligible(
            content="Task labeled [high] priority was created.",
            termination_reason=TERMINATION_MODEL_COMPLETED,
            new_tool_calls=[],
        ) is True


class TestSanitizeDanglingAssistantToolCalls:
    def test_trailing_assistant_with_tool_calls_stripped(self):
        from memory.tool_loop import _sanitize_closure_request_messages
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "x"}}]},
        ]
        out = _sanitize_closure_request_messages(messages)
        assert out == [{"role": "user", "content": "hi"}]

    def test_assistant_with_paired_tool_results_kept(self):
        from memory.tool_loop import _sanitize_closure_request_messages
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "x"}}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        ]
        out = _sanitize_closure_request_messages(messages)
        assert len(out) == 3
        assert out[1]["tool_calls"] == [
            {"id": "tc1", "type": "function",
             "function": {"name": "x"}}
        ]

    def test_empty_or_none_safe(self):
        from memory.tool_loop import _sanitize_closure_request_messages
        assert _sanitize_closure_request_messages([]) == []
        assert _sanitize_closure_request_messages(None) == []

    def test_deep_copies_inner_dicts(self):
        from memory.tool_loop import _sanitize_closure_request_messages
        original = [
            {"role": "user", "content": "hi",
             "tool_calls": [{"id": "tc1", "function": {"name": "x"}}]},
        ]
        out = _sanitize_closure_request_messages(original)
        out[0]["content"] = "mutated"
        out[0]["tool_calls"][0]["function"]["name"] = "mutated"
        assert original[0]["content"] == "hi"
        assert original[0]["tool_calls"][0]["function"]["name"] == "x"


class TestClosureDispatch:
    def test_observed_flow_closure_llm_success(self):
        from memory.tool_loop import run_memory_tool_loop

        closure_text = (
            "PARTIAL Confirmed operations:\n"
            "- created hash=abc12345 priority=high name=Fix the bug\n"
            "Errors:\n"
            "- [create_tasks:error] attempt 1 code=invalid_input "
            "create_tasks requires a non-empty tasks array\n"
        )

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [
                _make_requery_response(
                    content="", tool_calls=[{
                        "id": "tc_pt_1", "type": "function",
                        "function": {"name": "proxy_ticket",
                                     "arguments": json.dumps({
                                         "action": "create_tasks",
                                         "tasks": [{"name": "Fix the bug",
                                                    "priority": "high"}],
                                     })}}],
                    finish_reason="tool_calls"),
                {"choices": []},
                {"choices": [{
                    "message": {"role": "assistant",
                                "content": closure_text},
                    "finish_reason": "stop"}]},
            ]
            result = run_memory_tool_loop(
                **_loop_kwargs(
                    tool_calls_list=[
                        _make_tool_call("proxy_ticket",
                                         {"action": "create_tasks",
                                          "tasks": [{"name": "Fix the bug",
                                                     "priority": "high"}]},
                                         call_id="tc_init"),
                    ],
                )
            )

        assert not result.tool_calls
        content = result.content or ""
        assert content.startswith("PARTIAL")
        assert "hash=abc12345" in content
        assert "code=invalid_input" in content
        assert "create_tasks requires a non-empty tasks array" in content
        assert result.finish_reason == "stop"
        assert result.closure_source == "llm"
        assert result.needs_closure is False

    def test_closure_failure_uses_deterministic_fallback(self):
        from memory.tool_loop import run_memory_tool_loop, TERMINATION_REQUERY_FAILED

        def _execute_fn(name, args):
            return {
                "ok": False, "tool": name, "operation": "create_tasks",
                "status": "error", "data": None,
                "error": {
                    "code": "invalid_input",
                    "message": "create_tasks requires a non-empty tasks array",
                    "retryable": True, "category": "validation",
                },
                "session_id": "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [{}, {}]
            result = run_memory_tool_loop(
                **_loop_kwargs(execute_fn=_execute_fn)
            )

        assert result.termination_reason == TERMINATION_REQUERY_FAILED
        assert result.closure_source == "fallback"
        assert result.content is not None
        assert result.content.startswith("FAILED")
        assert "code=invalid_input" in result.content
        assert "create_tasks requires a non-empty tasks array" in result.content
        assert result.finish_reason == "stop"

    def test_closure_timeout_falls_back_to_deterministic(self):
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
            return {
                "ok": True, "tool": name, "operation": "create_tasks",
                "status": "success",
                "data": {
                    "created": [
                        {"short_hash": "abc12345", "priority": "high",
                         "name": "T1"},
                    ],
                    "errors": [], "wiring_errors": [],
                    "total_created": 1, "total_errors": 0,
                },
                "error": None, "session_id": "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [
                _make_requery_response(content="", tool_calls=[{
                    "id": "tc_init", "type": "function",
                    "function": {"name": "proxy_ticket",
                                 "arguments": json.dumps({
                                     "action": "create_tasks",
                                     "tasks": [{"name": "T1"}],
                                 })}}],
                    finish_reason="tool_calls"),
                TimeoutError("closure upstream timeout"),
            ]
            result = run_memory_tool_loop(
                **_loop_kwargs(execute_fn=_execute_fn)
            )

        assert result.closure_source == "fallback"
        content = result.content or ""
        assert content.startswith("COMPLETE")
        assert "hash=abc12345" in content
        assert result.finish_reason == "stop"

    def test_happy_path_skips_closure_no_extra_call(self):
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(
                content="The work is complete.",
            )
            result = run_memory_tool_loop(**_loop_kwargs(
                tool_calls_list=[
                    _make_tool_call("recall",
                                     {"semantic_targets": ["x"],
                                      "keywords": ["x"]},
                                     call_id="tc_init"),
                ],
            ))

        assert mock_upstream.call_count == 1
        assert not result.tool_calls
        assert result.content == "The work is complete."
        assert result.finish_reason == "stop"
        assert result.closure_source is None
        assert result.termination_reason == "model_completed"

    def test_raw_bracket_content_triggers_closure(self):
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [
                _make_requery_response(
                    content="[create_tasks: ok] done",
                ),
                {"choices": [{"message": {"content": ""}}]},
                {"choices": [{
                    "message": {"role": "assistant",
                                "content": "FAILED No confirmed operations."},
                    "finish_reason": "stop"}]},
            ]
            result = run_memory_tool_loop(**_loop_kwargs(
                tool_calls_list=[
                    _make_tool_call("recall",
                                     {"semantic_targets": ["x"],
                                      "keywords": ["x"]},
                                     call_id="tc_init"),
                ],
            ))

        assert not result.tool_calls
        content = result.content or ""
        assert not content.startswith("["), (
            f"raw bracket-ledger content must not surface; got {content!r}"
        )
        assert result.finish_reason == "stop"


class TestClosureFinalizePrefixValidation:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        from proxy.requery import closure_finalize
        self.closure_finalize = closure_finalize
        self.base_body = {"model": "m", "messages": []}
        self.base_kwargs = dict(
            working_messages=[{"role": "user", "content": "x"}],
            llama_url="http://x/v1",
            body=self.base_body,
            llama_body=self.base_body,
            model_cfg=None,
            auth_headers={},
            ledger="ledger",
            termination_reason="requery_failed",
        )

    def _run(self, content, tool_calls=None, message_extra=None,
             finish_reason="stop"):
        message = {"role": "assistant", "content": content}
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        if message_extra:
            message.update(message_extra)
        response = {"choices": [{"message": message, "finish_reason": finish_reason}]}
        with patch("proxy.requery.call_upstream", return_value=response):
            return self.closure_finalize(**self.base_kwargs)

    @pytest.mark.parametrize("content,expected_token", [
        ("COMPLETE All done.", "COMPLETE"),
        ("PARTIAL Created T1.", "PARTIAL"),
        ("FAILED System error.", "FAILED"),
        ("COMPLETE\nMulti line\ncontent.", "COMPLETE"),
        ("COMPLETE:Body text", "COMPLETE"),
    ])
    def test_accepts_valid_prefixes(self, content, expected_token):
        result = self._run(content)
        assert result.success is True
        assert result.content == content

    @pytest.mark.parametrize("content,reason", [
        ("COMPLETELY did the work.", "missing_prefix"),
        ("COMPLETE-ish did the work.", "missing_prefix"),
        ("COMPLETE ", "missing_prefix"),
        ("COMPLETE\n", "missing_prefix"),
        ("COMPLETE:", "missing_prefix"),
        ("COMPLETE :", "missing_prefix"),
        (" complete lower.", "leading_whitespace"),
        ("COMPLETE-Body text", "missing_prefix"),
        ("completely different", "missing_prefix"),
        ("   COMPLETE Leading whitespace.", "leading_whitespace"),
    ])
    def test_rejects_invalid_prefixes(self, content, reason):
        result = self._run(content)
        assert result.success is False
        assert result.failure_reason == reason

    def test_rejects_native_tool_call(self):
        result = self._run(
            "COMPLETE All done.",
            tool_calls=[{"id": "tc1", "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}],
        )
        assert result.success is False
        assert result.failure_reason == "tool_calls_native"

    def test_rejects_malformed_native_tool_call(self):
        result = self._run(
            "COMPLETE All done.",
            tool_calls="not-a-list",
        )
        assert result.success is False
        assert result.failure_reason == "tool_calls_native"

    def test_rejects_legacy_function_call(self):
        result = self._run(
            "COMPLETE All done.",
            message_extra={"function_call": {"name": "x"}},
        )
        assert result.success is False
        assert result.failure_reason == "tool_calls_native"

    def test_rejects_finish_reason_tool_calls(self):
        result = self._run(
            "COMPLETE All done.",
            finish_reason="tool_calls",
        )
        assert result.success is False
        assert result.failure_reason == "finish_reason_tool_calls"

    def test_rejects_legacy_function_call_finish_reason(self):
        result = self._run(
            "COMPLETE All done.",
            finish_reason="function_call",
        )
        assert result.success is False
        assert result.failure_reason == "finish_reason_tool_calls"

    def test_rejects_parsed_tool_call_when_parser_fails_open(self):
        fake_parse = lambda content: [{"name": "proxy_ticket",
                                       "arguments": {"action": "list_tasks"}}]
        with patch("proxy.requery.call_upstream") as mock_call:
            mock_call.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "PARTIAL <tool_call>x</tool_call>",
                    },
                    "finish_reason": "stop",
                }],
            }
            result = self.closure_finalize(
                **self.base_kwargs, parse_tools_fn=fake_parse,
            )
        assert result.success is False
        assert result.failure_reason == "tool_calls_parsed"

    def test_parser_exception_fails_closed(self):
        def bad_parse(content):
            raise RuntimeError("parser crashed")
        with patch("proxy.requery.call_upstream") as mock_call:
            mock_call.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "PARTIAL something",
                    },
                    "finish_reason": "stop",
                }],
            }
            result = self.closure_finalize(
                **self.base_kwargs, parse_tools_fn=bad_parse,
            )
        assert result.success is False
        assert result.failure_reason == "tool_calls_parsed"


class TestClosureFinalizeRequestShape:
    def test_request_body_strips_tool_enabling_fields(self):
        from proxy.requery import closure_finalize
        captured = {}

        def _fake_call(url, body, headers, wants_stream, timeout_cap=None):
            captured["body"] = body
            return {"choices": [{
                "message": {"role": "assistant",
                            "content": "COMPLETE All done."},
                "finish_reason": "stop"}]}

        with patch("proxy.requery.call_upstream",
                   side_effect=_fake_call):
            closure_finalize(
                working_messages=[
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": None,
                     "tool_calls": [{
                         "id": "tc1", "type": "function",
                         "function": {"name": "y",
                                      "arguments": "{}"}}]},
                ],
                llama_url="http://x/v1",
                body={"model": "m"},
                llama_body={"model": "m",
                            "messages": [{"role": "user",
                                          "content": "x"}]},
                model_cfg=None,
                auth_headers={},
                ledger="x",
                termination_reason="requery_failed",
            )

        body = captured["body"]
        for forbidden in ("tools", "tool_choice", "parallel_tool_calls",
                          "functions", "function_call"):
            assert forbidden not in body
        assert body.get("stream") is False
        last = body["messages"][-1]
        assert last["role"] == "user"
        assert "Termination reason: requery_failed" in last["content"]
        assert "Produce ONE" in last["content"]


class TestToolPipelineFormatUnresolvedDefense:
    def _real_result(self, **kwargs):
        from memory.tool_loop import (
            ToolLoopResult,
            TERMINATION_REQUERY_FAILED,
        )
        defaults = dict(
            content=None,
            tool_calls=[{
                "id": "tc1", "type": "function",
                "function": {"name": "x", "arguments": "{}"}}],
            finish_reason="tool_calls",
            messages_for_history=[],
            execution_records=[{
                "tool": "proxy_ticket", "operation": "create_tasks",
                "status": "error", "iteration": 1, "attempt": 1,
                "error": {"code": "x", "message": "boom"}, "data": None,
            }],
            needs_closure=False,
            termination_reason=TERMINATION_REQUERY_FAILED,
            closure_source=None,
        )
        defaults.update(kwargs)
        return ToolLoopResult(**defaults)

    def test_unresolved_closure_strips_tool_calls_and_emits_fallback(self):
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline
        pipeline = ToolPipeline()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc1", "type": "function",
                                "function": {"name": "x",
                                             "arguments": "{}"}}]},
                "finish_reason": "tool_calls",
            }]
        }
        result = self._real_result(needs_closure=True)
        from proxy.tool_pipeline.tool_pipeline import ToolContext
        ctx = ToolContext(
            session_id="", body={},
            llama_body={"messages": []}, auth_headers={},
            original_model="m", upstream_url="http://x/v1",
        )
        out = pipeline._format(data, result, ctx)
        assert isinstance(out, dict)
        assert out["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in out["choices"][0]["message"]
        content = out["choices"][0]["message"]["content"]
        assert content.startswith("FAILED")
        assert result.needs_closure is False
        assert result.closure_source == "fallback"

    def test_clean_closure_passes_through(self):
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline, ToolContext
        from memory.tool_loop import ToolLoopResult, TERMINATION_MODEL_COMPLETED
        pipeline = ToolPipeline()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }]
        }
        result = ToolLoopResult(
            content="COMPLETE All operations succeeded.",
            tool_calls=None,
            finish_reason="stop",
            messages_for_history=[],
            execution_records=[],
            needs_closure=False,
            termination_reason=TERMINATION_MODEL_COMPLETED,
            closure_source="llm",
        )
        ctx = ToolContext(
            session_id="", body={},
            llama_body={"messages": []}, auth_headers={},
            original_model="m", upstream_url="http://x/v1",
        )
        out = pipeline._format(data, result, ctx)
        assert out["choices"][0]["message"]["content"] == \
            "COMPLETE All operations succeeded."
        assert out["choices"][0]["finish_reason"] == "stop"
        assert result.closure_source == "llm"

    def test_needs_closure_identity_check_uses_is_not_truthiness(self):
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline, ToolContext
        pipeline = ToolPipeline()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }]
        }
        result = MagicMock()
        result.needs_closure = False
        result.tool_calls = None
        result.content = "ok"
        result.finish_reason = "stop"
        result.messages_for_history = []
        result.execution_records = []
        result.closure_source = None
        result.termination_reason = "model_completed"
        ctx = ToolContext(
            session_id="", body={},
            llama_body={"messages": []}, auth_headers={},
            original_model="m", upstream_url="http://x/v1",
        )
        out = pipeline._format(data, result, ctx)
        assert out["choices"][0]["message"]["content"] == "ok"
        assert out["choices"][0]["finish_reason"] == "stop"


class TestSyncHistoryDanglingPair:
    def test_chat_history_sync_rejects_dangling_pair(self):
        from proxy.tool_pipeline.tool_pipeline import (
            _assert_no_dangling_tool_pairs,
        )
        bad = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "x"}}]},
        ]
        with pytest.raises(RuntimeError, match="dangling"):
            _assert_no_dangling_tool_pairs(bad)

    def test_chat_history_sync_accepts_paired(self):
        from proxy.tool_pipeline.tool_pipeline import (
            _assert_no_dangling_tool_pairs,
        )
        ok = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function",
                 "function": {"name": "x"}}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "ok"},
        ]
        _assert_no_dangling_tool_pairs(ok)


class TestIterationLimitUnexecutedRecording:
    """Slice 4 (Spec §14 — Max-Iteration Contract): when the tool loop
    exits via iteration_limit, any memory tool call from the final
    iteration's requery was never executed and MUST be recorded with
    ``status='not_executed'`` so the closure ledger/fallback can mention it.
    """

    @staticmethod
    def _success_proxy_ticket(name, args):
        return {
            "ok": True,
            "tool": name,
            "operation": "create_tasks",
            "status": "success",
            "data": {
                "created": [
                    {"short_hash": "abc12345", "priority": "high", "name": "T1"}
                ],
                "errors": [],
                "wiring_errors": [],
                "total_created": 1,
                "total_errors": 0,
            },
            "error": None,
            "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
        }

    @staticmethod
    def _memory_parse(content):
        return [
            _make_tool_call(
                "proxy_ticket",
                {"action": "create_tasks",
                 "tasks": [{"name": "T1", "priority": "high"}]},
            )
        ]

    def test_text_plus_memory_tool_call_on_final_iteration_records_not_executed(self):
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [
                _make_requery_response(
                    content="thinking about it",
                    tool_calls=[],
                    finish_reason="tool_calls",
                ),
                _make_requery_response(
                    content="more thinking",
                    tool_calls=[],
                    finish_reason="tool_calls",
                ),
                {},
            ]

            result = run_memory_tool_loop(
                **_loop_kwargs(
                    execute_fn=self._success_proxy_ticket,
                    parse_tools_fn=self._memory_parse,
                    max_iterations=2,
                )
            )

        not_executed = [
            r for r in result.execution_records if r.get("status") == "not_executed"
        ]
        assert len(not_executed) == 1
        assert not_executed[0]["tool"] == "proxy_ticket"

        assert result.termination_reason == "iteration_limit"
        assert result.closure_source in ("llm", "fallback")
        if result.closure_source == "fallback":
            content = result.content or ""
            assert "not executed" in content.lower(), (
                f"fallback MUST surface the unexecuted op via _not_executed_prose; "
                f"got {content!r}"
            )
            assert "proxy_ticket" in content

    def test_only_memory_tools_recorded_not_passthrough(self):
        from memory.tool_loop import run_memory_tool_loop

        def _mixed_parse(content):
            return [
                _make_tool_call("proxy_ticket", {"action": "list_tasks"}),
                _make_tool_call("read_file", {"path": "/tmp/x"}),
            ]

        def _execute(name, args):
            return {
                "ok": True,
                "tool": name,
                "operation": "list_tasks",
                "status": "success",
                "data": {"tasks": []},
                "error": None,
                "session_id": "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [
                _make_requery_response(content="", tool_calls=[],
                                       finish_reason="tool_calls"),
                _make_requery_response(content="", tool_calls=[],
                                       finish_reason="tool_calls"),
                {},
            ]

            result = run_memory_tool_loop(
                **_loop_kwargs(
                    execute_fn=_execute,
                    parse_tools_fn=_mixed_parse,
                    max_iterations=2,
                )
            )

        not_executed = [
            r for r in result.execution_records if r.get("status") == "not_executed"
        ]
        assert len(not_executed) == 1
        assert not_executed[0]["tool"] == "proxy_ticket"
        not_executed_names = {r["tool"] for r in not_executed}
        assert "read_file" not in not_executed_names, (
            "passthrough calls are out of scope — only memory tools get "
            "not_executed records"
        )

    def test_not_executed_record_shape(self):
        from memory.tool_loop import run_memory_tool_loop

        def _memory_parse(content):
            return [
                _make_tool_call(
                    "proxy_ticket",
                    {"action": "create_tasks",
                     "tasks": [{"name": "X"}]},
                )
            ]

        def _execute(name, args):
            return {
                "ok": True,
                "tool": name,
                "operation": "create_tasks",
                "status": "success",
                "data": {"created": []},
                "error": None,
                "session_id": "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [
                _make_requery_response(content="", tool_calls=[],
                                       finish_reason="tool_calls"),
                _make_requery_response(content="", tool_calls=[],
                                       finish_reason="tool_calls"),
                {},
            ]

            result = run_memory_tool_loop(
                **_loop_kwargs(
                    execute_fn=_execute,
                    parse_tools_fn=_memory_parse,
                    max_iterations=2,
                )
            )

        not_executed = [
            r for r in result.execution_records if r.get("status") == "not_executed"
        ]
        assert len(not_executed) == 1
        rec = not_executed[0]
        assert rec["status"] == "not_executed"
        assert rec["tool"] == "proxy_ticket"
        assert rec["operation"] == "create_tasks"
        assert rec["tool_call_id"] is None
        assert rec["iteration"] == 3
        assert rec["attempt"] is None
        assert rec["data"] is None
        assert isinstance(rec["error"], dict)
        assert rec["error"]["code"] == "iteration_limit"
        assert rec["error"]["retryable"] is False
        assert rec["error"]["category"] == "infrastructure"

    def test_classify_outcome_with_not_executed(self):
        from memory.tool_loop import _classify_outcome

        def _rec(status):
            return {
                "tool": "t",
                "operation": "op",
                "status": status,
                "error": None,
                "data": None,
            }

        assert _classify_outcome([_rec("success"), _rec("not_executed")]) == "PARTIAL"
        assert _classify_outcome([_rec("not_executed")]) == "FAILED"
        assert _classify_outcome([_rec("success")]) == "COMPLETE"
        assert _classify_outcome([_rec("error"), _rec("not_executed")]) == "FAILED"
        assert _classify_outcome([_rec("not_executed"),
                                  _rec("not_executed")]) == "FAILED"
        assert _classify_outcome([_rec("partial"),
                                  _rec("not_executed")]) == "PARTIAL"
        assert _classify_outcome([_rec("success"), _rec("error"),
                                  _rec("not_executed")]) == "PARTIAL"


class TestNotExecutedProse:
    def test_uses_not_executed_wording(self):
        from memory.tool_loop import _not_executed_prose
        records = [
            {"tool": "proxy_ticket", "operation": "create_tasks",
             "status": "not_executed"},
        ]
        out = _not_executed_prose(records)
        assert len(out) == 1
        assert "not executed" in out[0]
        assert "failed" not in out[0]

    def test_includes_tool_and_operation_names(self):
        from memory.tool_loop import _not_executed_prose
        records = [
            {"tool": "proxy_work", "operation": "open_session",
             "status": "not_executed",
             "error": {"code": "iteration_limit",
                       "message": "Operation not executed — iteration limit reached"}},
        ]
        out = _not_executed_prose(records)
        assert "proxy_work" in out[0]
        assert "open_session" in out[0]
        assert "iteration limit reached" in out[0]


class TestDeterministicFallbackWithNotExecuted:
    def _success_rec(self, tool="proxy_ticket", operation="create_tasks"):
        return {
            "tool": tool, "operation": operation, "status": "success",
            "iteration": 1, "attempt": 1, "error": None,
            "data": {"created": [
                {"short_hash": "abc12345", "priority": "high", "name": "T1"}]},
        }

    def _not_executed_rec(self, tool="proxy_ticket", operation="list_tasks"):
        return {
            "tool": tool, "operation": operation, "status": "not_executed",
            "iteration": 3, "attempt": None,
            "error": {
                "code": "iteration_limit",
                "message": "Operation not executed — iteration limit reached",
                "retryable": False, "category": "infrastructure",
            },
            "data": None,
        }

    def _error_rec(self, tool="proxy_ticket", operation="update_task"):
        return {
            "tool": tool, "operation": operation, "status": "error",
            "iteration": 2, "attempt": 1,
            "error": {"code": "not_found", "message": "missing",
                      "retryable": False, "category": "validation"},
            "data": None,
        }

    def test_includes_not_executed_section(self):
        from memory.tool_loop import render_deterministic_fallback
        records = [self._success_rec(), self._not_executed_rec()]
        out = render_deterministic_fallback(records, "iteration_limit")
        assert "Not executed:" in out
        assert "proxy_ticket list_tasks not executed" in out
        assert "iteration limit reached" in out

    def test_section_appears_between_confirmed_and_errors(self):
        from memory.tool_loop import render_deterministic_fallback
        records = [
            self._success_rec(),
            self._not_executed_rec(),
            self._error_rec(),
        ]
        out = render_deterministic_fallback(records, "iteration_limit")
        confirmed_idx = out.find("Confirmed operations:")
        not_executed_idx = out.find("Not executed:")
        errors_idx = out.find("Errors:")
        assert confirmed_idx >= 0
        assert not_executed_idx >= 0
        assert errors_idx >= 0
        assert confirmed_idx < not_executed_idx < errors_idx, (
            "Not executed: section MUST appear between Confirmed operations: "
            "and Errors:"
        )

    def test_outcome_partial_when_success_and_not_executed(self):
        from memory.tool_loop import render_deterministic_fallback
        records = [self._success_rec(), self._not_executed_rec()]
        out = render_deterministic_fallback(records, "iteration_limit")
        assert out.startswith("PARTIAL"), (
            "success + not_executed must classify as PARTIAL"
        )


class TestSemanticRetryWiring:
    """Slice 5 (Spec §15 — Bounded Semantic Retry): ``run_memory_tool_loop``
    MUST consult ``SemanticRetryTracker`` before executing a memory tool
    call and break out of the while-loop (into the closure phase) when
    any call in the batch is rejected.

    The tracker is constructed inside the loop and is per-invocation —
    no shared state across requests. Sibling calls in the same batch
    MUST still execute when one is rejected (``continue`` to next call,
    not ``break`` out of the for-loop).
    """

    @staticmethod
    def _validation_error_envelope(name, args):
        return {
            "ok": False,
            "tool": name,
            "operation": "create_tasks",
            "status": "error",
            "data": None,
            "error": {
                "code": "invalid_input",
                "message": "create_tasks requires a non-empty tasks array",
                "retryable": True,
                "category": "validation",
            },
            "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
        }

    @staticmethod
    def _authorization_error_envelope(name, args):
        return {
            "ok": False,
            "tool": name,
            "operation": "create_tasks",
            "status": "error",
            "data": None,
            "error": {
                "code": "access_denied",
                "message": "ticket mode inactive",
                "retryable": False,
                "category": "authorization",
            },
            "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
        }

    @staticmethod
    def _success_list_tasks_envelope(name, args):
        return {
            "ok": True,
            "tool": name,
            "operation": "list_tasks",
            "status": "success",
            "data": {"tasks": []},
            "error": None,
            "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
        }

    def test_repeated_identical_validation_error_rejects_third_call(self):
        """Two consecutive identical validation failures exhaust the
        consecutive_identical budget. The 3rd call's ``check`` MUST
        return ``("validation_retry_exhausted", "validation")`` and the
        loop MUST skip execution (execute_fn called twice, not three
        times) and dispatch closure. The fallback report MUST mention
        the retry exhaustion."""
        from memory.tool_loop import (
            run_memory_tool_loop,
            TERMINATION_VALIDATION_RETRY_EXHAUSTED,
        )

        call_count = {"n": 0}

        def _execute(name, args):
            call_count["n"] += 1
            return self._validation_error_envelope(name, args)

        identical_args = {"action": "create_tasks", "tasks": []}
        identical_tc = _make_tool_call(
            "proxy_ticket", identical_args, call_id="tc_v1",
        )
        openai_identical = _make_openai_tool_call("proxy_ticket", call_id="tc_v1")

        def _repeat_parse(_content):
            return [_make_tool_call("proxy_ticket", identical_args, call_id="tc_v1")]

        iter1 = _make_requery_response(
            content="",
            tool_calls=[{
                "id": "tc_v1", "type": "function",
                "function": {
                    "name": "proxy_ticket",
                    "arguments": json.dumps(identical_args),
                },
            }],
            finish_reason="tool_calls",
        )
        iter2 = _make_requery_response(
            content="",
            tool_calls=[{
                "id": "tc_v1", "type": "function",
                "function": {
                    "name": "proxy_ticket",
                    "arguments": json.dumps(identical_args),
                },
            }],
            finish_reason="tool_calls",
        )
        closure_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "PARTIAL Operations summary:\n"
                        "- proxy_ticket create_tasks not executed "
                        "(Retry budget exhausted (validation))\n"
                    ),
                },
                "finish_reason": "stop",
            }],
        }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [iter1, iter2, closure_resp]
            result = run_memory_tool_loop(
                **_loop_kwargs(
                    tool_calls_list=[identical_tc],
                    working_messages=[
                        {"role": "user", "content": "do it"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [openai_identical]},
                    ],
                    execute_fn=_execute,
                    parse_tools_fn=_repeat_parse,
                )
            )

        assert call_count["n"] == 2, (
            "third identical validation call MUST be rejected before "
            f"execution; got {call_count['n']} execute_fn calls"
        )
        assert result.termination_reason == TERMINATION_VALIDATION_RETRY_EXHAUSTED, (
            f"termination_reason must be validation_retry_exhausted; "
            f"got {result.termination_reason!r}"
        )
        assert any(
            r.get("status") == "not_executed"
            and r.get("error", {}).get("code") == "validation_retry_exhausted"
            for r in result.execution_records
        ), (
            "execution_records MUST contain a not_executed entry whose "
            "error.code is validation_retry_exhausted"
        )
        assert result.content is not None
        assert "Retry budget exhausted" in result.content, (
            "closure fallback MUST surface the retry-exhaustion reason; "
            f"got {result.content!r}"
        )

    def test_authorization_error_zero_retries_immediate_closure(self):
        """An authorization failure MUST reject the next check
        immediately — no second execution. Closure dispatches."""
        from memory.tool_loop import (
            run_memory_tool_loop,
            TERMINATION_AUTHORIZATION_DENIED,
        )

        call_count = {"n": 0}

        def _execute(name, args):
            call_count["n"] += 1
            return self._authorization_error_envelope(name, args)

        identical_args = {"action": "create_tasks", "tasks": [{"name": "X"}]}
        identical_tc = _make_tool_call(
            "proxy_ticket", identical_args, call_id="tc_a1",
        )
        openai_identical = _make_openai_tool_call("proxy_ticket", call_id="tc_a1")

        def _repeat_parse(_content):
            return [_make_tool_call("proxy_ticket", identical_args, call_id="tc_a1")]

        iter1 = _make_requery_response(
            content="",
            tool_calls=[{
                "id": "tc_a1", "type": "function",
                "function": {
                    "name": "proxy_ticket",
                    "arguments": json.dumps(identical_args),
                },
            }],
            finish_reason="tool_calls",
        )
        closure_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "FAILED Operations summary:\n"
                        "- proxy_ticket create_tasks not executed "
                        "(Retry budget exhausted (authorization))\n"
                    ),
                },
                "finish_reason": "stop",
            }],
        }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [iter1, closure_resp]
            result = run_memory_tool_loop(
                **_loop_kwargs(
                    tool_calls_list=[identical_tc],
                    working_messages=[
                        {"role": "user", "content": "do it"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [openai_identical]},
                    ],
                    execute_fn=_execute,
                    parse_tools_fn=_repeat_parse,
                )
            )

        assert call_count["n"] == 1, (
            "authorization error MUST block the second execution; "
            f"got {call_count['n']} execute_fn calls"
        )
        assert result.termination_reason == TERMINATION_AUTHORIZATION_DENIED
        assert any(
            r.get("error", {}).get("code") == "authorization_no_retry"
            for r in result.execution_records
        ), (
            "execution_records MUST contain a not_executed entry with "
            "error.code == authorization_no_retry"
        )

    def test_materially_corrected_call_executes_despite_prior_failure(self):
        from memory.tool_loop import run_memory_tool_loop

        executed_args = []

        def _execute(name, args):
            executed_args.append(json.dumps(args, sort_keys=True, default=str))
            if not args.get("tasks"):
                return self._validation_error_envelope(name, args)
            return {
                "ok": True, "tool": name, "operation": "create_tasks",
                "status": "success",
                "data": {"created": [{"short_hash": "abc12345"}]},
                "error": None,
                "session_id": args.get("session_id", ""),
            }

        broken_args = {"action": "create_tasks", "tasks": []}
        corrected_args = {"action": "create_tasks", "tasks": [{"name": "T1"}]}
        broken_tc = _make_tool_call(
            "proxy_ticket", broken_args, call_id="tc_m1",
        )
        openai_broken = _make_openai_tool_call("proxy_ticket", call_id="tc_m1")

        iter1 = _make_requery_response(
            content="",
            tool_calls=[{
                "id": "tc_m2", "type": "function",
                "function": {
                    "name": "proxy_ticket",
                    "arguments": json.dumps(broken_args),
                },
            }],
            finish_reason="tool_calls",
        )
        iter2 = _make_requery_response(
            content="",
            tool_calls=[{
                "id": "tc_m3", "type": "function",
                "function": {
                    "name": "proxy_ticket",
                    "arguments": json.dumps(corrected_args),
                },
            }],
            finish_reason="tool_calls",
        )
        final = _make_requery_response(
            content="COMPLETE done.", finish_reason="stop",
        )

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [iter1, iter2, final]
            result = run_memory_tool_loop(
                **_loop_kwargs(
                    tool_calls_list=[broken_tc],
                    working_messages=[
                        {"role": "user", "content": "do it"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [openai_broken]},
                    ],
                    execute_fn=_execute,
                )
            )

        assert len(executed_args) == 3, (
            "both broken calls and the corrected call MUST execute — "
            f"material correction overrides two prior failures; got {len(executed_args)}"
        )
        assert result.termination_reason == "model_completed"

    def test_sibling_calls_in_batch_still_execute_when_one_is_rejected(self):
        """When one sibling in a batch is rejected by the retry tracker,
        the OTHER sibling in the same for-loop batch MUST still execute.
        Pins the ``continue`` (not ``break``) contract for rejected
        siblings."""
        from memory.tool_loop import (
            run_memory_tool_loop,
            TERMINATION_VALIDATION_RETRY_EXHAUSTED,
        )

        call_log = []

        def _execute(name, args):
            call_log.append(args.get("action"))
            if args.get("action") == "create_tasks":
                return self._validation_error_envelope(name, args)
            return self._success_list_tasks_envelope(name, args)

        a_args = {"action": "create_tasks", "tasks": []}
        b_args = {"action": "list_tasks"}
        a_tc = _make_tool_call("proxy_ticket", a_args, call_id="tc_sib_a")
        openai_a = _make_openai_tool_call("proxy_ticket", call_id="tc_sib_a")

        # iter1 requery: returns A again (so iter2 will see A with state).
        iter1 = _make_requery_response(
            content="",
            tool_calls=[{
                "id": "tc_sib_a", "type": "function",
                "function": {
                    "name": "proxy_ticket",
                    "arguments": json.dumps(a_args),
                },
            }],
            finish_reason="tool_calls",
        )
        # iter2 requery: returns BOTH A and B (siblings in iter3 batch).
        iter2 = _make_requery_response(
            content="",
            tool_calls=[
                {"id": "tc_sib_a", "type": "function",
                 "function": {"name": "proxy_ticket",
                              "arguments": json.dumps(a_args)}},
                {"id": "tc_sib_b", "type": "function",
                 "function": {"name": "proxy_ticket",
                              "arguments": json.dumps(b_args)}},
            ],
            finish_reason="tool_calls",
        )
        closure_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "PARTIAL Confirmed operations:\n"
                        "- proxy_ticket list_tasks returned 0 result(s)\n"
                        "Not executed:\n"
                        "- proxy_ticket create_tasks not executed "
                        "(Retry budget exhausted (validation))\n"
                    ),
                },
                "finish_reason": "stop",
            }],
        }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [iter1, iter2, closure_resp]
            result = run_memory_tool_loop(
                **_loop_kwargs(
                    tool_calls_list=[a_tc],
                    working_messages=[
                        {"role": "user", "content": "do it"},
                        {"role": "assistant", "content": None,
                         "tool_calls": [openai_a]},
                    ],
                    execute_fn=_execute,
                )
            )

        # A executed twice (iter1 and iter2), then rejected on iter3.
        # B executed once (iter3, despite A being rejected).
        a_count = call_log.count("create_tasks")
        b_count = call_log.count("list_tasks")
        assert a_count == 2, (
            f"create_tasks MUST execute exactly twice (iter1 + iter2); "
            f"got {call_log}"
        )
        assert b_count == 1, (
            f"list_tasks MUST execute exactly once in iter3 despite A's "
            f"rejection; sibling calls continue; got {call_log}"
        )
        assert result.termination_reason == TERMINATION_VALIDATION_RETRY_EXHAUSTED

    def test_not_executed_prose_uses_message_for_retry_codes(self):
        """Slice 5: ``_not_executed_prose`` MUST surface
        ``error.message`` (truncated) when ``error.code`` contains
        'retry' or 'denied'. Pins the new prose contract."""
        from memory.tool_loop import _not_executed_prose

        records = [
            {
                "tool": "proxy_ticket", "operation": "create_tasks",
                "status": "not_executed",
                "error": {
                    "code": "validation_retry_exhausted",
                    "message": "Retry budget exhausted (validation)",
                    "retryable": False, "category": "validation",
                },
            },
            {
                "tool": "proxy_work", "operation": "start_work",
                "status": "not_executed",
                "error": {
                    "code": "authorization_no_retry",
                    "message": "Retry budget exhausted (authorization)",
                    "retryable": False, "category": "authorization",
                },
            },
        ]
        out = _not_executed_prose(records)
        assert len(out) == 2
        assert "Retry budget exhausted (validation)" in out[0]
        assert "Retry budget exhausted (authorization)" in out[1]