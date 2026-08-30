"""Unit tests for memory tool interception loop.

All tests use mocked dependencies - NO real network calls or real LLM.
"""

import json
import time
import threading
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


def _make_tool_call(name, arguments=None, call_id=None):
    """Helper to build a parsed tool call dict."""
    return {
        "name": name,
        "arguments": arguments or {},
        "id": call_id or f"call_{int(time.time() * 1000000)}_0",
    }


def _make_openai_tool_call(name, call_id=None):
    """Helper to build an OpenAI-format tool call."""
    return {
        "id": call_id or f"call_{int(time.time() * 1000000)}_0",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _make_requery_response(content="", tool_calls=None, finish_reason="stop"):
    """Build a mock requery HTTP response."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish_reason}]}


def _base_kwargs(**overrides):
    """Build base kwargs for run_memory_tool_loop with sensible defaults."""
    defaults = {
        "tool_calls_list": [],
        "working_messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "model": "qwen3-coder",
        "session_id": "test-session",
        "llama_url": "http://localhost:8080/v1/chat/completions",
        "body": {"model": "qwen3-coder"},
        "llama_body": {
            "model": "qwen3-coder",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "chat_histories": {},
        "chat_histories_lock": threading.Lock(),
    }
    defaults.update(overrides)
    return defaults


class TestEmptyToolList:
    def test_empty_tool_list_passthrough(self):
        from memory.tool_loop import run_memory_tool_loop

        result = run_memory_tool_loop(**_base_kwargs(tool_calls_list=[]))

        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.content is None or result.content == ""


class TestSingleMemoryTool:
    def test_single_memory_tool_no_requery(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": [], "total_matches": 0})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[_make_tool_call("recall", {"semantic_targets": ["test"], "keywords": ["test"]})],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=MagicMock(return_value=[]),
            )
        )

        mock_execute.assert_called_once_with("recall", {"semantic_targets": ["test"], "keywords": ["test"], "session_id": "test-session"})
        assert result.finish_reason == "stop"
        assert result.content is not None or result.tool_calls is not None


class TestSessionIdInjectionPrecedence:
    """Regression: session_id injection guard operator precedence (slice 8c).

    The guard must evaluate as ``(A or B) and C`` — i.e. session_id is only
    injected when it is truthy. Without parentheses Python evaluates
    ``A or B and C`` as ``A or (B and C)``, so for search/recall/proxy_work
    (branch A) the guard was always True and session_id was injected even when
    it was falsy ("" / None).
    """

    def test_falsy_session_id_not_injected_for_search(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": [], "total": 0})

        run_memory_tool_loop(
            **_base_kwargs(
                session_id="",
                tool_calls_list=[_make_tool_call("search", {"query": "test"})],
                execute_fn=mock_execute,
                make_tool_calls_fn=MagicMock(return_value=[_make_openai_tool_call("search")]),
                parse_tools_fn=MagicMock(return_value=[]),
            )
        )

        # Falsy session_id must NOT be injected into tool_args.
        mock_execute.assert_called_once_with("search", {"query": "test"})

    def test_truthy_session_id_injected_for_search(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": [], "total": 0})

        run_memory_tool_loop(
            **_base_kwargs(
                session_id="sess-abc",
                tool_calls_list=[_make_tool_call("search", {"query": "test"})],
                execute_fn=mock_execute,
                make_tool_calls_fn=MagicMock(return_value=[_make_openai_tool_call("search")]),
                parse_tools_fn=MagicMock(return_value=[]),
            )
        )

        # Truthy session_id IS injected into tool_args.
        mock_execute.assert_called_once_with("search", {"query": "test", "session_id": "sess-abc"})


class TestMemoryThenPassthrough:
    def test_memory_then_passthrough_returns_tool_calls(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": []})
        passthrough_tc = _make_tool_call("read_file", {"path": "/tmp/test.txt"})
        openai_tc = _make_openai_tool_call("read_file")

        def mock_parse(content):
            return [passthrough_tc]

        mock_make = MagicMock(return_value=[openai_tc])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[_make_tool_call("recall", {"semantic_targets": ["test"], "keywords": ["test"]})],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=mock_parse,
            )
        )

        assert result.tool_calls is not None
        assert result.finish_reason == "tool_calls"


class TestMultiIterationMemoryChain:
    def test_multi_iteration_chain(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"saved": True})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        iteration = [0]

        def mock_parse(content):
            iteration[0] += 1
            if iteration[0] <= 2:
                return [_make_tool_call("recall", {"semantic_targets": ["info"], "keywords": ["info"]})]
            return []

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[_make_tool_call("recall", {"semantic_targets": ["info"], "keywords": ["info"]})],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=mock_parse,
            )
        )

        assert mock_execute.call_count >= 2


class TestMaxIterationsExhausted:
    def test_max_iterations_exhausted(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"saved": True})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        always_memory = lambda content: [
            _make_tool_call("recall", {"semantic_targets": ["x"], "keywords": ["x"]})
        ]

        with patch("proxy.requery.call_upstream") as mock_upstream:
            # C6 (6.1): capture the final requery's content. Pre-C6 this
            # content was DISCARDED at tool_loop.py:610. The sentinel string
            # below is preserved in messages_for_history under Slice 3 closure.
            mock_upstream.return_value = _make_requery_response(
                content="C6_FINAL_REQUERY_CONTENT",
                tool_calls=[],
                finish_reason="tool_calls",
            )

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[_make_tool_call("recall", {"semantic_targets": ["info"], "keywords": ["info"]})],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=always_memory,
                    max_iterations=2,
                )
            )

        assert mock_execute.call_count == 2
        # Slice 3 (mandatory closure): iteration_limit is a non-passthrough
        # terminal state that MUST go through the closure phase, so the raw
        # requery content is no longer surfaced as result.content. The C6
        # capture invariant still holds — the sentinel survives in
        # messages_for_history so future turns retain the pre-exhaustion
        # conversation state.
        assert result.content is not None
        assert "C6_FINAL_REQUERY_CONTENT" not in result.content, (
            "Slice 3: iteration_limit must dispatch closure — raw requery "
            "content must not leak as the final user-visible content. Got: "
            f"{result.content!r}"
        )
        history_text = json.dumps(result.messages_for_history, default=str)
        assert "C6_FINAL_REQUERY_CONTENT" in history_text, (
            "C6 (6.1) invariant: the final requery's content must be "
            "preserved in messages_for_history even after Slice 3 closure. "
            f"Got history={history_text!r}"
        )
        assert result.termination_reason == "iteration_limit"
        assert result.needs_closure is False
        assert result.closure_source in ("llm", "fallback")
        assert any(r.get("status") == "not_executed" for r in result.execution_records)


class TestBudgetTimeout:
    def test_budget_timeout_breaks_immediately(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"saved": True})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        time_values = [0.0, 100.0]

        with patch("memory.tool_loop.time") as mock_time:
            mock_time.time.side_effect = time_values

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["info"], "keywords": ["info"]})
                    ],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                    max_total_time=5.0,
                )
            )

        mock_execute.assert_not_called()


class TestRequeryHttpError:
    def test_requery_http_error_breaks_gracefully(self):
        from memory.tool_loop import run_memory_tool_loop
        import requests as req

        mock_execute = MagicMock(return_value={"saved": True})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        with patch("memory.tool_loop.requests.post") as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError("refused")

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["info"], "keywords": ["info"]})
                    ],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        assert mock_execute.call_count == 1


class TestToolExecutionException:
    def test_tool_execution_exception_returns_error(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(side_effect=RuntimeError("db down"))
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        with patch("memory.tool_loop.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_requery_response(content="done")
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["test"], "keywords": ["test"]})
                    ],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        assert mock_execute.call_count == 1


class TestToolResultTruncation:
    def test_tool_result_truncation_over_16kb(self):
        from memory.tool_loop import run_memory_tool_loop

        large_result = {"data": "x" * 20000}
        mock_execute = MagicMock(return_value=large_result)
        tool_call = _make_tool_call("recall", {"semantic_targets": ["test"], "keywords": ["test"]})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        # Include an assistant message with tool_calls so the tool result
        # is not stripped by _cleanup_orphaned_tool_results.
        messages_with_assistant = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        ]

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[tool_call],
                    working_messages=messages_with_assistant,
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        requery_call = mock_upstream.call_args
        tool_msgs = [
            m for m in requery_call[0][1]["messages"] if m.get("role") == "tool"
        ]
        assert len(tool_msgs) == 1
        assert "TRUNCATED" in tool_msgs[0]["content"]


class TestWorkingMessagesTrimmed:
    def test_working_messages_trimmed_over_42(self):
        from memory.tool_loop import run_memory_tool_loop

        many_messages = [{"role": "user", "content": f"msg{i}"} for i in range(50)]

        mock_execute = MagicMock(return_value={"results": []})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["test"], "keywords": ["test"]})
                    ],
                    working_messages=many_messages,
                    llama_body={
                        "model": "qwen3-coder",
                        "messages": many_messages,
                    },
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        requery_call = mock_upstream.call_args
        msgs = requery_call[0][1]["messages"]
        assert len(msgs) <= 42


class TestJsonParseFailureInArgs:
    def test_json_parse_failure_falls_back_to_empty_dict(self):
        """§17.1 — malformed JSON now produces canonical envelope, not empty dict."""
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"ok": True})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[
                    _make_tool_call("recall", "not valid json {{{")
                ],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=MagicMock(return_value=[]),
            )
        )

        # §17.1: execute_fn should NOT be called for malformed JSON
        mock_execute.assert_not_called()
        assert len(result.execution_records) > 0
        assert result.execution_records[0]["error"]["code"] == "malformed_arguments"


class TestNoChoicesInRequeryResponse:
    def test_no_choices_in_requery_response_breaks(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"saved": True})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        with patch("proxy.http_client.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")

            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["info"], "keywords": ["info"]})
                    ],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        assert mock_execute.call_count == 1


def _loop_kwargs(**overrides):
    """Build kwargs for the finish_reason regression tests.

    Mirrors test_tool_loop_timeout.py: includes an assistant message carrying
    the recall tool_call so the loop's tool_call_id map resolves, and patches
    proxy.requery.call_upstream to return the canned requery response.
    """
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
        "session_id": "sess-finish",
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


class TestFinishReasonContentOnlyMemoryResolution:
    """The core fix: run_memory_tool_loop must NOT clobber the LAST requery's
    upstream finish_reason when resolving memory tools to content-only output.

    Pre-fix, tool_loop.py:598 hardcoded `finish_reason="tool_calls" if
    final_tool_calls else "stop"`, so an upstream "length" (truncation)
    on a content-only resolution was silently mapped to "stop" — losing
    the client's retry/truncation signal.
    """

    def test_content_only_preserves_length_finish_reason(self):
        """upstream "length" + content-only memory resolution → "length"."""
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(
                content="here is the answer", finish_reason="length"
            )

            result = run_memory_tool_loop(**_loop_kwargs())

        assert result.finish_reason == "length", (
            "truncation signal must reach the client on a content-only "
            "memory resolution; pre-fix it was clobbered to 'stop'"
        )

    def test_content_only_preserves_stop_finish_reason(self):
        """upstream "stop" + content-only → "stop" (no regression)."""
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(
                content="here is the answer", finish_reason="stop"
            )

            result = run_memory_tool_loop(**_loop_kwargs())

        assert result.finish_reason == "stop"

    def test_content_only_tool_calls_upstream_maps_to_stop(self):
        """upstream "tool_calls" but native calls fail to parse (empty
        new_tool_calls), content-only → "stop".

        Pins the contradictory-tool_calls exclusion: there are no surviving
        tool calls to return to the client, so "tool_calls" would be a lie.
        """
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            # message carries malformed tool_calls (no "function" key) so
            # requery.py's native parser fails and new_tool_calls stays [].
            mock_upstream.return_value = _make_requery_response(
                content="answer text",
                tool_calls=[{"no_function_key": True}],
                finish_reason="tool_calls",
            )

            result = run_memory_tool_loop(**_loop_kwargs())

        assert result.finish_reason == "stop", (
            "a content-only result must never carry finish_reason='tool_calls' "
            "(no surviving tool calls) — pins the bounded-rule exclusion"
        )

    def test_content_only_omitted_finish_reason_maps_to_stop(self):
        """upstream omits finish_reason (None) → "stop".

        None must NOT clobber the (also None) init value, and None is not a
        TRUNCATION_REASONS member, so it maps to "stop".
        """
        from memory.tool_loop import run_memory_tool_loop

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(
                content="answer text", finish_reason=None
            )

            result = run_memory_tool_loop(**_loop_kwargs())

        assert result.finish_reason == "stop"

    def test_passthrough_tool_calls_with_length_upstream_returns_tool_calls(self):
        """upstream "length" + passthrough tool calls survive → "tool_calls".

        When passthrough tool calls exist, finish_reason is intentionally
        "tool_calls" regardless of the upstream truncation signal — the
        client must execute the tool calls to continue the turn. The
        truncation signal is intentionally hidden here (final_tool_calls wins).
        """
        from memory.tool_loop import run_memory_tool_loop

        passthrough_openai = _make_openai_tool_call("read_file")
        mock_make = MagicMock(return_value=[passthrough_openai])

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(
                content="read this file",
                tool_calls=[
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/x"}',
                        },
                    }
                ],
                finish_reason="length",
            )

            result = run_memory_tool_loop(**_loop_kwargs(make_tool_calls_fn=mock_make))

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None

    def test_multi_iteration_last_break_finish_reason_wins(self):
        """iter1 should_continue (memory tools, upstream "stop"), iter2
        content-only upstream "length" → final "length".

        The last iteration that breaks the loop provides the finish_reason.
        A prior iteration's "stop" must NOT override the final "length".
        """
        from memory.tool_loop import run_memory_tool_loop

        iter1 = _make_requery_response(
            content="",
            tool_calls=[
                {
                    "id": "call_recall2",
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "arguments": '{"semantic_targets":["y"],"keywords":["y"]}',
                    },
                }
            ],
            finish_reason="stop",
        )
        iter2 = _make_requery_response(
            content="final answer", finish_reason="length"
        )

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [iter1, iter2]

            result = run_memory_tool_loop(**_loop_kwargs())

        assert mock_upstream.call_count == 2
        assert result.finish_reason == "length", (
            "the LAST breaking iteration's upstream finish_reason must win"
        )

    def test_max_iterations_exit_returns_stop_not_stale(self):
        """all iterations should_continue (upstream returns "tool_calls"
        each time), exhaust max_iterations → "stop" (NOT "tool_calls").

        Pins two complementary invariants from C6 (6.1):
        1. CRITICAL reset fix: when the loop exhausts iterations via
           should_continue continues, a stale "tool_calls" must NOT leak
           into the final result — finish_reason resets to None and maps
           to "stop".
        2. Content capture: the FINAL requery's content is now RETURNED
           (not discarded as it was pre-C6) — see the strengthened
           assertions in TestMaxIterationsExhausted above for the full
           content pin. This test focuses on the finish_reason invariant;
           content coverage lives next door.
        """
        from memory.tool_loop import run_memory_tool_loop

        response = _make_requery_response(
            content="",
            tool_calls=[
                {
                    "id": "call_recall3",
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "arguments": '{"semantic_targets":["z"],"keywords":["z"]}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = response

            result = run_memory_tool_loop(**_loop_kwargs(max_iterations=2))

        assert result.finish_reason == "stop", (
            "on max-iterations exhaustion the discarded requery finish_reason "
            "must be reset to None and map to 'stop' — never a stale "
            "'tool_calls'"
        )


class TestSentinelSurvivesEndToEndWiring:
    """LOW arch-review-2 finding: pin that the trim helpers are WIRED into
    ``run_memory_tool_loop`` at ``tool_loop.py:510`` and ``:522``.

    ``test_tool_loop_trim_sentinels.py`` exercises the helpers in isolation
    via direct calls to ``_trim_working_messages_by_tokens`` and
    ``_enforce_working_messages_count_cap``. That proves the helpers work,
    but it does NOT prove the helpers are actually called at the production
    call sites. If someone un-wires the helper call (or refactors around
    it), the unit tests on the helpers stay green while production
    silently stops preserving ``_sub_agent_return`` sentinels.

    These tests call ``run_memory_tool_loop`` end-to-end with an
    overflowing buffer + trailing ``_sub_agent_return`` sentinel and assert
    the sentinel survives all the way to ``call_upstream``. The helpers
    ARE the production code (not dead code like ``MessageBuffer``), so the
    gap is narrower than the wave-4 landmine — but it is still a wiring
    gap, and this layer pins it.
    """

    def _build_buffer(self, big_content_len: int, sentinel_call_id: str):
        """Build a working_messages buffer with the given big content length
        plus a trailing ``_sub_agent_return`` sentinel and its paired
        assistant-with-tool-calls.

        Returns ``(working_messages, sentinel_call_id)``. The sentinel is
        the LAST message in the buffer (so it survives the newest-first
        walk), and the paired assistant-with-tool-calls is right before
        it so ``_cleanup_orphaned_tool_results`` does not orphan it.
        """
        big = "x" * big_content_len
        buf = []
        for i in range(80):
            role = "user" if i % 2 == 0 else "assistant"
            buf.append({"role": role, "content": f"msg{i} {big}"})
        # Paired assistant-with-tool-calls must be present so
        # _cleanup_orphaned_tool_results does not orphan the sentinel.
        buf.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": sentinel_call_id,
                    "type": "function",
                    "function": {"name": "sub-agent", "arguments": "{}"},
                }
            ],
        })
        # Sentinel is the LAST message (newest) so it is the first thing
        # the trim helper considers and never the drop target.
        buf.append({
            "role": "tool",
            "content": "sub-agent round-trip result",
            "_sub_agent_return": True,
            "tool_call_id": sentinel_call_id,
        })
        return buf

    def test_sentinel_survives_token_budget_trim_to_upstream(self):
        """Token-budget trim at ``tool_loop.py:510``: an overflowing buffer
        with a trailing ``_sub_agent_return`` sentinel preserves the
        sentinel end-to-end through ``run_memory_tool_loop`` →
        ``call_upstream``.

        Setup: 80 user/assistant messages each with ``1500 * 0.25 = 375``
        tokens (so total ≈ 30,000 > 19,500 budget), plus a paired
        assistant-with-tool-calls + sentinel = 82 messages. After the
        memory tool result is appended, the token-budget trim at
        ``tool_loop.py:510`` MUST run, and the sentinel MUST survive it.

        Two assertions, together pinning the wiring:
        - ``len(msgs) < 82`` proves the trim RAN (revert-test catches
          un-wiring; without the trim the buffer is uncapped at 82).
        - ``sentinel in msgs`` proves the trim PRESERVED the sentinel
          (the H2 invariant — a naive inline trim that drops oldest
          would silently lose the sentinel).
        """
        from memory.tool_loop import run_memory_tool_loop

        sentinel_call_id = "call_subagent_token_budget"
        many_messages = self._build_buffer(1500, sentinel_call_id)

        assert len(many_messages) == 82, (
            f"test setup invariant: got {len(many_messages)}, expected 82"
        )

        mock_execute = MagicMock(return_value={"results": []})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")

            run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call(
                            "recall",
                            {"semantic_targets": ["x"], "keywords": ["x"]},
                        )
                    ],
                    working_messages=many_messages,
                    llama_body={
                        "model": "qwen3-coder",
                        "messages": many_messages,
                    },
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        requery_call = mock_upstream.call_args
        assert requery_call is not None, (
            "call_upstream was not invoked — test setup did not enter the "
            "memory tool loop. End-to-end wiring test requires a real "
            "upstream call to validate the helper is reachable."
        )
        msgs = requery_call[0][1]["messages"]
        assert len(msgs) < 82, (
            "token-budget trim did not reduce the buffer. With 80 × 375-token "
            "messages (~30,000 tokens) vs. 19,500 budget, the trim at "
            "tool_loop.py:510 MUST drop messages. len(msgs)=82 means the "
            "trim helper is un-wired (revert-test failure)."
        )
        sentinels_in_upstream = [m for m in msgs if m.get("_sub_agent_return")]
        assert len(sentinels_in_upstream) == 1, (
            "_sub_agent_return sentinel was dropped by the token-budget trim "
            "at tool_loop.py:510. The trim helper preserved oldest-first "
            "instead of preserving the sentinel — this is the H2 arch-review-2 "
            "bug class. Surviving sentinels in upstream payload: "
            f"{len(sentinels_in_upstream)} (expected 1)."
        )
        assert (
            sentinels_in_upstream[0]["content"] == "sub-agent round-trip result"
        )
        assert sentinels_in_upstream[0]["tool_call_id"] == sentinel_call_id

    def test_sentinel_survives_count_cap_to_upstream(self):
        """Count-cap at ``tool_loop.py:522``: a buffer with > 69 messages
        including a trailing ``_sub_agent_return`` sentinel preserves the
        sentinel end-to-end through ``run_memory_tool_loop`` →
        ``call_upstream``.

        Setup: 80 cheap user/assistant messages + assistant-with-tool-calls
        + sentinel = 82 messages. After the memory tool result is appended
        the count is 83 > 69, so the count-cap at ``tool_loop.py:522``
        MUST run, and the sentinel MUST survive it.

        Two assertions, together pinning the wiring:
        - ``len(msgs) < 82`` proves the cap RAN (revert-test catches
          un-wiring; without the cap the buffer is uncapped at 82).
        - ``sentinel in msgs`` proves the cap PRESERVED the sentinel
          (the H1 invariant — a naive inline cap that drops oldest
          would silently lose the sentinel).
        """
        from memory.tool_loop import run_memory_tool_loop

        sentinel_call_id = "call_subagent_count_cap"
        # Cheap content so token-budget trim does NOT trigger — we want
        # to exercise the count-cap path exclusively.
        buf = []
        for i in range(80):
            role = "user" if i % 2 == 0 else "assistant"
            buf.append({"role": role, "content": f"msg{i}"})
        buf.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": sentinel_call_id,
                    "type": "function",
                    "function": {"name": "sub-agent", "arguments": "{}"},
                }
            ],
        })
        buf.append({
            "role": "tool",
            "content": "sub-agent round-trip result",
            "_sub_agent_return": True,
            "tool_call_id": sentinel_call_id,
        })

        assert len(buf) == 82, f"test setup invariant: got {len(buf)}, expected 82"

        mock_execute = MagicMock(return_value={"results": []})
        mock_make = MagicMock(return_value=[_make_openai_tool_call("recall")])

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")

            run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call(
                            "recall",
                            {"semantic_targets": ["x"], "keywords": ["x"]},
                        )
                    ],
                    working_messages=buf,
                    llama_body={
                        "model": "qwen3-coder",
                        "messages": buf,
                    },
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=MagicMock(return_value=[]),
                )
            )

        requery_call = mock_upstream.call_args
        assert requery_call is not None, (
            "call_upstream was not invoked — test setup did not enter the "
            "memory tool loop. End-to-end wiring test requires a real "
            "upstream call to validate the helper is reachable."
        )
        msgs = requery_call[0][1]["messages"]
        assert len(msgs) < 82, (
            "count-cap did not reduce the buffer. With 82 input messages + 1 "
            "tool result (83 total) vs. 69 cap, the cap at tool_loop.py:522 "
            "MUST drop messages. len(msgs)=82 means the count-cap helper is "
            "un-wired (revert-test failure)."
        )
        sentinels_in_upstream = [m for m in msgs if m.get("_sub_agent_return")]
        assert len(sentinels_in_upstream) == 1, (
            "_sub_agent_return sentinel was dropped by the count-cap at "
            "tool_loop.py:522. The cap helper dropped oldest-first instead of "
            "preserving the sentinel — this is the H1 arch-review-2 bug "
            "class. Surviving sentinels in upstream payload: "
            f"{len(sentinels_in_upstream)} (expected 1)."
        )
        assert (
            sentinels_in_upstream[0]["content"] == "sub-agent round-trip result"
        )
        assert sentinels_in_upstream[0]["tool_call_id"] == sentinel_call_id


class TestExhaustionReturnsFinalContent:
    """C6 (6.1): at max_iterations, the final requery's content is captured
    and returned — pre-C6 this content was DISCARDED at tool_loop.py:610.

    Pre-C6 the exhausting iteration's requery output (the final answer
    the LLM produced right before the loop gave up) was thrown away —
    result.content fell back to the stale pre-loop content (None/""),
    and the client received an empty response with finish_reason="stop".

    C6 (6.1) fixes this by promoting ``requery_content`` (captured inside
    the loop) to ``content`` in the exhaustion branch, so the final
    iteration's answer survives. This test class pins that invariant
    end-to-end through ``run_memory_tool_loop`` — a revert-test catches
    any future regression that re-discards the final requery output.
    """

    def test_exhaustion_returns_last_requery_content(self):
        """5 iterations with max_iterations=5: result.content must be the
        FINAL requery's content (the 5th iteration's "FINAL ANSWER FROM
        ITERATION 5"), NOT the stale pre-loop content.

        Each iteration's requery returns a unique content sentinel
        ("FINAL ANSWER FROM ITERATION N") plus the same recall tool_call,
        which drives should_continue=True so the loop iterates. After
        iteration 5 (where iteration+=1 makes iteration==5), the
        ``iteration >= max_iterations`` branch at tool_loop.py:603 fires
        and ``content = requery_content`` — which MUST equal the 5th
        iteration's content.

        Revert-test: revert the C6 capture (replace ``content = requery_content``
        with ``content = None``) and this FAILS — result.content would be
        None instead of "FINAL ANSWER FROM ITERATION 5".
        """
        from memory.tool_loop import run_memory_tool_loop

        def _recall_requery(iteration):
            return _make_requery_response(
                content=f"FINAL ANSWER FROM ITERATION {iteration}",
                tool_calls=[
                    {
                        "id": f"call_recall_iter{iteration}",
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "arguments": '{"semantic_targets":["x"],"keywords":["x"]}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        side_effects = [_recall_requery(i) for i in range(1, 6)]

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = side_effects

            result = run_memory_tool_loop(**_loop_kwargs(max_iterations=5))

        # Slice 3: 5 requery iterations + 1 closure attempt = 6 total.
        assert mock_upstream.call_count == 6, (
            f"expected 5 requery iterations + 1 closure attempt = 6, "
            f"got {mock_upstream.call_count}"
        )

        # C6 (6.1): finish_reason resets to "stop" on exhaustion
        # (companion to test_max_iterations_exit_returns_stop_not_stale).
        assert result.finish_reason == "stop", (
            f"C6 (6.1): on max_iterations exhaustion finish_reason must "
            f"be 'stop', got {result.finish_reason!r}"
        )

        # Slice 3: iteration_limit is a non-passthrough terminal state that
        # MUST dispatch closure. The deterministic fallback is used here
        # (closure call would have made one additional upstream request,
        # but mock_upstream.side_effect only contains the 5 iteration
        # responses; the 6th call would have raised StopIteration → the
        # closure wrapper catches and falls back).
        assert result.termination_reason == "iteration_limit", (
            f"Slice 3: iteration_limit terminal reason must be set, "
            f"got {result.termination_reason!r}"
        )
        assert result.needs_closure is False, (
            "Slice 3: needs_closure must be resolved (False) after the "
            "closure helper runs."
        )
        assert result.closure_source in ("llm", "fallback"), (
            f"closure_source must be 'llm' or 'fallback', got "
            f"{result.closure_source!r}"
        )
        assert result.content is not None
        assert "FINAL ANSWER FROM ITERATION 5" not in result.content, (
            "Slice 3: the raw requery content must NOT leak as the final "
            "user-visible content on iteration_limit. Got: "
            f"{result.content!r}"
        )
        assert any(r.get("status") == "not_executed" for r in result.execution_records)
        # Defensive: C6 invariant survives via messages_for_history.
        history_text = json.dumps(result.messages_for_history, default=str)
        assert "FINAL ANSWER FROM ITERATION 5" in history_text, (
            "C6 (6.1) invariant: the final requery's content must be "
            "preserved in messages_for_history under Slice 3 closure."
        )

        # Defensive: no earlier iteration's content leaked through the
        # closure report (defensive against renderer bugs).
        for i in range(1, 5):
            assert f"FINAL ANSWER FROM ITERATION {i}" not in result.content, (
                f"Slice 3: an earlier iteration's content "
                f"({i}) leaked through the closure report — the FINAL "
                f"iteration's content must not be referenced."
            )


class TestFallbackActionable:
    """C6 (6.2): the generic fallback AND the raw-JSON dump name the failing
    action(s) — pre-C6 the client received a vague "I encountered an issue"
    placeholder with no actionable signal about which tool failed.

    Slice 3 update: requery-failed branches now go through the closure
    phase; the deterministic fallback renderer carries the action name,
    status, error code, and message into the FAILED report so the
    actionable signal still reaches the client. The C6 invariants now
    pin the closure report rather than the raw-JSON dump.
    """

    def test_fallback_includes_action_name(self):
        """Requery fails (mock returns ``{}`` → ``requery_failed=True``) +
        no passthrough calls + raw_results contains an error envelope with
        ``operation="create_tasks"`` → the returned content MUST contain
        ``"create_tasks"`` (the failing action name), NOT a vague
        "I encountered an issue" placeholder.

        Revert-test: revert the closure fallback rendering at
        render_deterministic_fallback / _details_for_failed and this
        FAILS — the client receives no signal about which action
        failed.
        """
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
            return {
                "ok": False,
                "tool": name,
                "operation": "create_tasks",
                "status": "error",
                "data": None,
                "error": {
                    "code": "tool_execution_failed",
                    "message": "db timeout",
                    "retryable": False,
                    "category": "internal",
                },
                "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = {}
            result = run_memory_tool_loop(
                **_loop_kwargs(execute_fn=_execute_fn)
            )

        assert result.finish_reason == "stop", (
            f"fallback should map to finish_reason='stop', got "
            f"{result.finish_reason!r}"
        )
        assert result.content is not None
        assert "create_tasks" in result.content, (
            f"Slice 3: closure fallback MUST name the failing action "
            f"('create_tasks') — got {result.content!r}"
        )
        assert "I encountered an issue" not in result.content
        assert result.termination_reason == "requery_failed"
        assert result.closure_source == "fallback"
        assert result.needs_closure is False

    def test_error_envelope_renders_message_not_code(self):
        """Slice 3: requery-failed → closure phase → deterministic fallback.
        The fallback MUST surface the actionable error.message, the
        error.code, the attempt number, and the operation name — but
        NOT the raw ``{'code': ..., ...}`` dict-stringify bug from
        pre-Slice 2.
        """
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
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

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = {}
            result = run_memory_tool_loop(
                **_loop_kwargs(execute_fn=_execute_fn)
            )

        assert result.termination_reason == "requery_failed"
        assert result.closure_source == "fallback"
        assert result.content is not None
        assert "create_tasks requires a non-empty tasks array" in result.content, (
            f"closure fallback must surface actionable error.message, "
            f"got {result.content!r}"
        )
        assert "create_tasks" in result.content
        assert "invalid_input" in result.content, (
            f"closure fallback must surface error.code, got {result.content!r}"
        )
        # Pre-Slice 2 dict-stringify bug must not reappear.
        assert "{'code'" not in result.content
        assert "FAILED" in result.content[:20], (
            f"prefix must be a FAILED token at position 0, got "
            f"{result.content!r}"
        )

    def test_partial_envelope_renders_partial_status(self):
        """A partial envelope drives the closure fallback's PARTIAL branch.

        _classify_outcome() returns PARTIAL when at least one record has
        status='partial'. The renderer prefixes the report with PARTIAL
        and includes confirmed operations / errors sections.
        """
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
            return {
                "ok": True,
                "tool": name,
                "operation": "create_tasks",
                "status": "partial",
                "data": {
                    "created": [],
                    "errors": [{"name": "x", "error": "validation", "message": "name must be unique"}],
                    "wiring_errors": [],
                    "total_created": 0,
                    "total_errors": 1,
                },
                "error": None,
                "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = {}
            result = run_memory_tool_loop(
                **_loop_kwargs(execute_fn=_execute_fn)
            )

        assert result.termination_reason == "requery_failed"
        assert result.closure_source == "fallback"
        assert result.content is not None
        assert result.content.startswith("PARTIAL"), (
            f"closure fallback must classify partial-only records as "
            f"PARTIAL with the prefix at character zero, got {result.content!r}"
        )
        assert "Termination: requery_failed" in result.content

    def test_max_iterations_summary_renders_error_message(self):
        """Slice 3: max-iterations → closure phase → deterministic fallback.
        The fallback surfaces the failing action, attempt number,
        error.code, and the actionable error.message — even though the
        previous Site 1 bullet-list summary is replaced by the closure
        report.
        """
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
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

        always_memory = lambda content: [
            _make_tool_call("recall", {"semantic_targets": ["x"], "keywords": ["x"]})
        ]

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(
                content="",
                tool_calls=[],
                finish_reason="tool_calls",
            )

            result = run_memory_tool_loop(
                **_loop_kwargs(
                    execute_fn=_execute_fn,
                    parse_tools_fn=always_memory,
                    max_iterations=2,
                )
            )

        assert result.termination_reason == "iteration_limit"
        assert result.closure_source == "fallback"
        assert result.content is not None
        assert result.content.startswith("FAILED"), (
            f"closure fallback must prefix with FAILED at position 0 on "
            f"iteration_limit when all records errored, got "
            f"{result.content!r}"
        )
        assert "create_tasks requires a non-empty tasks array" in result.content, (
            f"closure fallback must surface actionable error.message, "
            f"got {result.content!r}"
        )
        assert "invalid_input" in result.content, (
            f"closure fallback must surface error.code, got {result.content!r}"
        )
        assert "{'code'" not in result.content
        assert "Termination: iteration_limit" in result.content

    def test_site3_record_precedence_when_helper_fails(self):
        """Slice 3: requery_failed → closure → deterministic fallback surfaces
        actionable error.code/message from the canonical execution record.
        """
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
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

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = {}
            result = run_memory_tool_loop(
                **_loop_kwargs(execute_fn=_execute_fn)
            )

        assert result.content is not None
        assert "create_tasks" in result.content, (
            f"closure fallback must name the failing action from "
            f"execution_records, got {result.content!r}"
        )
        assert result.content.startswith("FAILED"), (
            f"closure fallback must begin with the FAILED prefix token at "
            f"character zero, got {result.content!r}"
        )
        assert "invalid_input" in result.content, (
            f"closure fallback must surface error.code, got {result.content!r}"
        )
        assert "create_tasks requires a non-empty tasks array" in result.content
        assert result.termination_reason == "requery_failed"
        assert result.closure_source == "fallback"
        assert result.needs_closure is False
        assert "I encountered an issue" not in result.content
        assert "Tool loop failed after processing" not in result.content, (
            f"the pre-Slice-3 message must be replaced by the "
            f"closure fallback report, got {result.content!r}"
        )


class TestNoOrphanToolResults:
    """C6 (6.3): when ``tool_call_id`` lookup fails (no match in
    ``original_tc_id_map`` because the new tool_call's args differ from
    the original assistant message's args), the result is NOT appended
    to ``working_messages`` as an orphan tool-role message.

    Pre-C6 the code path generated a synthetic ``mem_xxx`` tool_call_id
    and appended the result to ``working_messages`` anyway. The orphan
    message would either be stripped by ``_cleanup_orphaned_tool_results``
    (wasted work) or, if cleanup was bypassed, trigger a strict-upstream
    400 ("tool_call_id not found").

    C6 (6.3) fixes this by SKIPPING the append when lookup misses and
    logging a warning. The result is still preserved in ``raw_results``
    for the fallback summary path. This test class pins the skip.

    Note: ``_loop_kwargs()``'s default ``working_messages`` contains an
    assistant message whose recall tool_call uses empty ``arguments="{}"``
    — which naturally mismatches the new tool_call's
    ``{"semantic_targets": ["x"], "keywords": ["x"]}``. So the default
    ``_loop_kwargs()`` setup already forces a lookup miss on iteration 1,
    and the C6 fix is what prevents the orphan from being appended.
    """

    def test_synthetic_id_results_skipped(self):
        """§17.2 — tool-call ID mismatch now preserves feedback via synthetic pair.

        Pre-§17.2: orphan tool-role messages were silently dropped (C6 skip).
        Post-§17.2: a synthetic assistant/tool message pair is created so the LLM
        receives error feedback.
        """
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": [{"text": "data"}]})

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")
            run_memory_tool_loop(**_loop_kwargs(execute_fn=mock_execute))

        mock_execute.assert_called_once()

        requery_call = mock_upstream.call_args
        assert requery_call is not None, (
            "test setup invariant: call_upstream was not invoked — "
            "expected exactly one requery after the memory tool execution"
        )
        requery_msgs = requery_call[0][1]["messages"]
        tool_msgs_in_requery = [m for m in requery_msgs if m.get("role") == "tool"]
        assert len(tool_msgs_in_requery) == 1, (
            f"§17.2: exactly one tool-role message expected in requery; "
            f"found {len(tool_msgs_in_requery)}"
        )
        assert tool_msgs_in_requery[0]["tool_call_id"].startswith("synth_"), (
            "§17.2: synthetic tool_call_id should use 'synth_' prefix"
        )

        all_ids = [
            m.get("tool_call_id")
            for m in requery_msgs
            if m.get("role") == "tool" and m.get("tool_call_id")
        ]
        assert all(id_.startswith("synth_") for id_ in all_ids), (
            f"§17.2: all tool_call_ids should be synthetic; got {all_ids}"
        )


class TestExecutionRecords:
    """Spec §9: ``run_memory_tool_loop`` captures a structured execution
    record per memory-tool call on ``ToolLoopResult.execution_records``.

    Each entry has shape::

        {
            "tool": "<memory-tool-name>",
            "operation": "<envelope.operation or None>",
            "tool_call_id": "<matched id or None>",
            "iteration": <1-based int>,
            "attempt": <1-based int, keyed on (tool, operation)>,
            "status": "success" | "partial" | "error",
            "error": <envelope.error dict or None>,
            "data": <envelope.data dict or None>,
        }

    ``raw_results`` continues to carry the raw envelope; the structured
    record is the canonical observability surface for telemetry and for
    the Slice 2 detail-preference formatter.
    """

    @staticmethod
    def _success_envelope(tool, args):
        return {
            "ok": True,
            "tool": tool,
            "operation": "search",
            "status": "success",
            "data": {"results": [], "total_matches": 0},
            "error": None,
            "session_id": args.get("session_id", "") if isinstance(args, dict) else "",
        }

    def test_execution_records_captured_for_single_call(self):
        """One memory-tool call → one record with all 8 canonical keys,
        status='success', error=None, data equal to the envelope's data dict."""
        from memory.tool_loop import run_memory_tool_loop

        captured = []

        def _execute_fn(name, args):
            env = self._success_envelope(name, args)
            captured.append(env)
            return env

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")
            result = run_memory_tool_loop(**_loop_kwargs(execute_fn=_execute_fn))

        assert len(result.execution_records) == 1, (
            f"expected exactly 1 execution record, got {len(result.execution_records)}"
        )
        rec = result.execution_records[0]
        # All 8 canonical keys are present.
        expected_keys = {
            "tool", "operation", "tool_call_id", "iteration",
            "attempt", "status", "error", "data",
        }
        assert expected_keys.issubset(rec.keys()), (
            f"missing keys: {expected_keys - set(rec.keys())}; record={rec}"
        )
        assert rec["tool"] == "recall"
        assert rec["operation"] == "search"
        assert rec["iteration"] == 1
        assert rec["attempt"] == 1
        assert rec["status"] == "success"
        assert rec["error"] is None
        assert rec["data"] == {"results": [], "total_matches": 0}

    def test_execution_records_attempt_increments_for_repeat_operation(self):
        """Two iterations of the same (tool, operation) pair → attempt 1, then 2.

        Drives multi-iteration by having the first requery return a recall
        tool_call (``should_continue=True``); the second returns nothing
        (``should_continue=False``). ``execute_fn`` returns the same
        canonical envelope on both calls so the (tool, operation) key
        matches and the attempt counter increments.
        """
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
            return self._success_envelope(name, args)

        iter1 = _make_requery_response(
            content="",
            tool_calls=[
                {
                    "id": "call_recall_attempt2",
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "arguments": '{"semantic_targets":["y"],"keywords":["y"]}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )
        iter2 = _make_requery_response(content="done")

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.side_effect = [iter1, iter2]
            result = run_memory_tool_loop(**_loop_kwargs(execute_fn=_execute_fn))

        assert len(result.execution_records) == 2, (
            f"expected 2 execution records (one per iteration), "
            f"got {len(result.execution_records)}: {result.execution_records}"
        )
        rec1, rec2 = result.execution_records
        assert rec1["attempt"] == 1, f"first call attempt must be 1; got {rec1['attempt']}"
        assert rec2["attempt"] == 2, f"second call attempt must be 2; got {rec2['attempt']}"
        # Both records share the same (tool, operation) identity.
        assert rec1["tool"] == rec2["tool"] == "recall"
        assert rec1["operation"] == rec2["operation"] == "search"
        # Iteration counter is 1-based per loop pass.
        assert rec1["iteration"] == 1
        assert rec2["iteration"] == 2

    def test_execution_records_record_error_envelopes(self):
        """An error envelope is captured with status='error', error as the
        full error dict, data=None — preserves the canonical shape for
        downstream telemetry without losing the structured error code."""
        from memory.tool_loop import run_memory_tool_loop

        def _execute_fn(name, args):
            return {
                "ok": False,
                "tool": name,
                "operation": "search",
                "status": "error",
                "data": None,
                "error": {
                    "code": "infrastructure_error",
                    "message": "Database temporarily unavailable",
                    "retryable": True,
                    "category": "infrastructure",
                },
                "session_id": "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")
            result = run_memory_tool_loop(**_loop_kwargs(execute_fn=_execute_fn))

        assert len(result.execution_records) == 1
        rec = result.execution_records[0]
        assert rec["status"] == "error", (
            f"error envelope must be recorded with status='error'; got {rec['status']!r}"
        )
        assert isinstance(rec["error"], dict), (
            f"error must be the structured dict, not a string; got {rec['error']!r}"
        )
        assert rec["error"]["code"] == "infrastructure_error"
        assert rec["data"] is None

    def test_execution_records_record_partial_envelopes(self):
        """A partial envelope (e.g. create_tasks with per-task errors) is
        captured with status='partial' and data preserved verbatim — the
        per-task errors must remain reachable via ``data["errors"]`` so
        Slice 2's detail-preference formatter can surface them."""
        from memory.tool_loop import run_memory_tool_loop

        partial_data = {
            "created": [{"short_hash": "abc12345", "name": "T1"}],
            "errors": [{"name": "T2", "error": "duplicate_name"}],
            "wiring_errors": [],
            "total_created": 1,
            "total_errors": 1,
        }

        def _execute_fn(name, args):
            return {
                "ok": True,
                "tool": name,
                "operation": "create_tasks",
                "status": "partial",
                "data": partial_data,
                "error": None,
                "session_id": "",
            }

        with patch("proxy.requery.call_upstream") as mock_upstream:
            mock_upstream.return_value = _make_requery_response(content="done")
            result = run_memory_tool_loop(**_loop_kwargs(execute_fn=_execute_fn))

        assert len(result.execution_records) == 1
        rec = result.execution_records[0]
        assert rec["status"] == "partial", (
            f"partial envelope must be recorded with status='partial'; got {rec['status']!r}"
        )
        assert rec["error"] is None, (
            "partial envelope has no top-level error; per-task errors live in data.errors"
        )
        assert rec["data"] == partial_data
        assert rec["data"]["errors"], (
            "data.errors must be non-empty for a partial create_tasks envelope"
        )
        assert rec["data"]["errors"][0]["error"] == "duplicate_name"


class TestMalformedJsonEnvelope:
    """§17.1 — malformed argument JSON produces malformed_arguments envelope, not {}."""

    def test_json_decode_error_produces_envelope(self):
        from memory.tool_loop import run_memory_tool_loop
        mock_execute = MagicMock()
        mock_make = MagicMock(return_value=[])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[
                    _make_tool_call("proxy_ticket", "NOT VALID JSON {{{", call_id="call_1"),
                ],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=MagicMock(return_value=[]),
                working_messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "proxy_ticket", "arguments": "NOT VALID JSON {{{"}}
                    ]},
                ],
            )
        )

        mock_execute.assert_not_called()

        assert len(result.execution_records) > 0
        err_rec = result.execution_records[0]
        assert err_rec["status"] == "error"
        assert err_rec["error"]["code"] == "malformed_arguments"
        assert err_rec["error"]["retryable"] is True

    def test_malformed_json_does_not_become_empty_args(self):
        from memory.tool_loop import run_memory_tool_loop
        mock_execute = MagicMock()
        mock_make = MagicMock(return_value=[])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[
                    _make_tool_call("proxy_ticket", "invalid json", call_id="call_2"),
                ],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=MagicMock(return_value=[]),
                working_messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_2", "type": "function", "function": {"name": "proxy_ticket", "arguments": "invalid json"}}
                    ]},
                ],
            )
        )

        mock_execute.assert_not_called()
        assert len(result.execution_records) > 0
        assert result.execution_records[0]["error"]["code"] == "malformed_arguments"


class TestIdMismatchFeedbackPreserved:
    """§17.2 — tool-call ID mismatch preserves feedback in working_messages."""

    def test_id_mismatch_result_preserved(self):
        from memory.tool_loop import run_memory_tool_loop
        mock_execute = MagicMock(return_value={
            "operation": "list_tasks",
            "status": "error",
            "error": {"code": "not_found", "message": "task not found", "retryable": False, "category": "validation"},
        })
        mock_make = MagicMock(return_value=[])
        mock_parse = MagicMock(return_value=[])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[
                    _make_tool_call("recall", {"semantic_targets": ["x"]}, call_id="no_match"),
                ],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=mock_parse,
                working_messages=[
                    {"role": "user", "content": "hello"},
                ],
            )
        )

        assert len(result.execution_records) > 0
        assert result.execution_records[0]["status"] == "error"
        # §17.2: the result must be in messages_for_history (not silently dropped)
        tool_msgs = [m for m in result.messages_for_history if m.get("role") == "tool"]
        assert len(tool_msgs) > 0, (
            "§17.2: tool result must be preserved in messages_for_history; "
            "the old skip path silently dropped it"
        )

    def test_id_mismatch_result_not_silently_dropped(self):
        from memory.tool_loop import run_memory_tool_loop
        mock_execute = MagicMock(return_value={
            "operation": "list_tasks",
            "status": "error",
            "error": {"code": "access_denied", "message": "permission denied"},
        })
        mock_make = MagicMock(return_value=[])
        mock_parse = MagicMock(return_value=[])

        result = run_memory_tool_loop(
            **_base_kwargs(
                tool_calls_list=[
                    _make_tool_call("proxy_ticket", {"action": "update_task", "short_hash": "abc"}, call_id="orphan_id"),
                ],
                execute_fn=mock_execute,
                make_tool_calls_fn=mock_make,
                parse_tools_fn=mock_parse,
                working_messages=[
                    {"role": "user", "content": "list tasks"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "different_id", "type": "function", "function": {"name": "proxy_ticket", "arguments": '{"action": "list_tasks"}'}}
                    ]},
                ],
            )
        )

        assert len(result.execution_records) > 0
        rec = result.execution_records[0]
        assert rec["tool"] == "proxy_ticket"
        assert rec["status"] == "error"
        # §17.2: the result must be in messages_for_history (not silently dropped)
        tool_msgs = [m for m in result.messages_for_history if m.get("role") == "tool"]
        assert len(tool_msgs) > 0, (
            "§17.2: tool result must be preserved in messages_for_history; "
            "the old skip path silently dropped it"
        )


class TestExceptionPathMarksFailure:
    """§17.3 — unexpected loop exceptions mark failure explicitly."""

    def test_exception_path_sets_internal_error(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": [], "total_matches": 0})
        mock_make = MagicMock(return_value=[])
        mock_parse = MagicMock(return_value=[])

        with patch("proxy.requery.requery_llm", side_effect=RuntimeError("requery boom")):
            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["x"]}, call_id="call_exp"),
                    ],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=mock_parse,
                    working_messages=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "Let me search for that.",
                         "tool_calls": [
                             {"id": "call_exp", "type": "function",
                              "function": {"name": "recall", "arguments": '{"semantic_targets": ["x"]}'}}
                         ]},
                    ],
                )
            )

        assert result.termination_reason == "internal_error"
        assert result.content is not None
        # §17.3: _requery_failed ensures history is not corrupted.
        # Without _requery_failed=True, the sanitized assistant content
        # would leak into messages_for_history.
        assert result.messages_for_history == [], (
            "§17.3: messages_for_history must be empty when _requery_failed=True "
            "to prevent stale/corrupt history from reaching the client"
        )

    def test_exception_path_closure_runs(self):
        from memory.tool_loop import run_memory_tool_loop

        mock_execute = MagicMock(return_value={"results": [], "total_matches": 0})
        mock_make = MagicMock(return_value=[])
        mock_parse = MagicMock(return_value=[])

        with patch("proxy.requery.requery_llm", side_effect=RuntimeError("requery boom")):
            result = run_memory_tool_loop(
                **_base_kwargs(
                    tool_calls_list=[
                        _make_tool_call("recall", {"semantic_targets": ["x"]}, call_id="call_exp2"),
                    ],
                    execute_fn=mock_execute,
                    make_tool_calls_fn=mock_make,
                    parse_tools_fn=mock_parse,
                    working_messages=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "Let me search for that.",
                         "tool_calls": [
                             {"id": "call_exp2", "type": "function",
                              "function": {"name": "recall", "arguments": '{"semantic_targets": ["x"]}'}}
                         ]},
                    ],
                )
            )

        assert result.content is not None
        assert "FAILED" in result.content or "PARTIAL" in result.content or "COMPLETE" in result.content
