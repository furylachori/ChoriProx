"""Unit tests for src/proxy/upstream_sse.py.

Tests call reconstruct_openai_sse or the public accumulator directly.
The parser is never mocked. Payload dicts are built with helpers; tool-call
fragments are extracted to variables to avoid nested-call brace miscounting.

NOTE: these exercise the upstream_sse reconstruction PARSER. The parser IS
wired into the production request path (see `proxy/pipeline/stage_modules/
upstream.py:101`: `call_upstream(..., wants_stream=bool(ctx.wants_stream))`).
This is the production SSE reconstruction path for stream=true requests; the
per-line feed contracts here are what the production accumulator enforces.
"""

from __future__ import annotations

import inspect
import json

import pytest

from proxy.upstream_sse import (
    DEFAULT_UPSTREAM_STREAM_LIMITS,
    OpenAIStreamAccumulator,
    UpstreamStreamError,
    UpstreamStreamLimitError,
    UpstreamStreamLimits,
    UpstreamStreamProtocolError,
    reconstruct_openai_sse,
)


_MISSING = object()


def _tc(index, cid=_MISSING, name=_MISSING, arguments=_MISSING):
    """One tool_call fragment.

    Uses _MISSING sentinel to distinguish absent (not in JSON), None
    (JSON null), and "" (JSON empty string) for the optional fields.
    """
    f = {}
    if name is not _MISSING:
        f["name"] = name
    if arguments is not _MISSING:
        f["arguments"] = arguments
    out = {"index": index, "type": "function", "function": f}
    if cid is not _MISSING:
        out["id"] = cid
    return out


def _choice(role=_MISSING, content=_MISSING, reasoning=_MISSING, reasoning_content=_MISSING,
            tool_calls=_MISSING, finish=_MISSING, index=0):
    """One choice with explicit integer index (default 0).

    Default role is _MISSING (absent). Pass role=None for explicit null,
    role="assistant" for explicit string, role="user" for non-assistant.
    """
    delta = {}
    if role is not _MISSING:
        delta["role"] = role
    if content is not _MISSING:
        delta["content"] = content
    if reasoning is not _MISSING:
        delta["reasoning"] = reasoning
    if reasoning_content is not _MISSING:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not _MISSING:
        delta["tool_calls"] = tool_calls
    choice = {"index": index, "delta": delta}
    if finish is not _MISSING:
        choice["finish_reason"] = finish
    return choice


def _data(d):
    return b"data: " + json.dumps(d).encode("utf-8")


def _event_bytes(d):
    """Return the assembled_bytes length for a JSON event payload."""
    return len(json.dumps(d).encode("utf-8"))


def _text_response(text="hello"):
    return [
        _data({"choices": [_choice(role="assistant")]}),
        b"",
        _data({"choices": [_choice(content=text)]}),
        b"",
        _data({"choices": [_choice(finish="stop")]}),
        b"",
        b"data: [DONE]",
        b"",
    ]


# ===========================================================================
# Framing
# ===========================================================================


class TestFraming:
    def test_ignores_comments_and_blank_events(self):
        lines = [b": keepalive", b"", b": another comment", b"", *_text_response()]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "hello"

    def test_accepts_data_with_and_without_space(self):
        lines = [
            b"data:" + json.dumps({"choices": [_choice(role="assistant")]}).encode(),
            b"",
            _data({"choices": [_choice(content="a")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "a"

    def test_accepts_crlf_normalized_lines(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}) + b"\r",
            b"\r",
            _data({"choices": [_choice(content="hi")]}) + b"\r",
            b"\r",
            _data({"choices": [_choice(finish="stop")]}) + b"\r",
            b"\r",
            b"data: [DONE]\r",
            b"\r",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "hi"

    def test_joins_multiple_data_lines_per_event(self):
        json_text = json.dumps({"choices": [_choice(role="assistant")]})
        split = json_text.index(":") + 1
        lines = [
            b"data:" + json_text[:split].encode(),
            b"data:" + json_text[split:].encode(),
            b"",
            _data({"choices": [_choice(content="z")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "z"

    def test_accepts_utf8_bom_only_at_stream_start(self):
        lines = [
            b"\xef\xbb\xbf" + _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="ok")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "ok"

    def test_ignores_event_id_retry_and_unknown_fields(self):
        lines = [
            b"event: message",
            b"id: evt-1",
            b"retry: 3000",
            b"unknown: value",
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "x"

    def test_rejects_invalid_utf8(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b"data: " + b"\xff\xfe")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_utf8"

    def test_rejects_malformed_json(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b"data: {not valid json")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_json"

    def test_rejects_non_object_json(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b"data: [1,2,3]")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_json"

    def test_done_completes_terminal_response(self):
        result = reconstruct_openai_sse(_text_response("done via done"))
        assert result["choices"][0]["message"]["content"] == "done via done"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_clean_eof_completes_terminal_response(self):
        lines = [
            _data({"id": "chatcmpl-y", "choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="eof-complete")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "eof-complete"

    def test_rejects_done_before_terminal_state(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "incomplete_response"

    def test_rejects_eof_before_finish_reason(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="no finish")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "incomplete_response"

    def test_rejects_duplicate_done(self):
        lines = [*_text_response(), b"data: [DONE]", b""]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_event"

    def test_rejects_data_after_done(self):
        lines = [*_text_response(), _data({"choices": [_choice(content="after done")]}), b""]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_event"

    def test_rejects_empty_data_after_done(self):
        lines = [*_text_response(), b"data:", b""]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_event"

    def test_rejects_whitespace_data_after_done(self):
        lines = [*_text_response(), b"data:   ", b""]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_event"

    def test_rejects_malformed_data_after_done(self):
        lines = [*_text_response(), b"data: {not-json", b""]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_event"

    def test_allows_comments_after_done(self):
        lines = [*_text_response(), b": keepalive", b""]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "hello"


# ===========================================================================
# Content and reasoning
# ===========================================================================


class TestContentAndReasoning:
    def test_reconstructs_fragmented_content_in_order(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="Hel")]}),
            b"",
            _data({"choices": [_choice(content="lo ")]}),
            b"",
            _data({"choices": [_choice(content="world")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "Hello world"

    def test_empty_and_null_content_do_not_erase_prior_content(self):
        empty_c = _choice()
        empty_c["delta"]["content"] = ""
        null_c = _choice()
        null_c["delta"]["content"] = None
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="keep")]}),
            b"",
            _data({"choices": [empty_c]}),
            b"",
            _data({"choices": [null_c]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "keep"

    def test_preserves_split_minimax_think_markup_verbatim(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="<think>Hel")]}),
            b"",
            _data({"choices": [_choice(content="lo</think>")]}),
            b"",
            _data({"choices": [_choice(content=" world")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "<think>Hello</think> world"

    def test_reconstructs_reasoning_content_independently(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(reasoning_content="rc1")]}),
            b"",
            _data({"choices": [_choice(reasoning_content="rc2")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "rc1rc2"
        assert "reasoning" not in msg

    def test_reconstructs_reasoning_alias_and_populates_reasoning_content(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(reasoning="r1")]}),
            b"",
            _data({"choices": [_choice(reasoning="r2")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        msg = result["choices"][0]["message"]
        assert msg["reasoning"] == "r1r2"
        assert msg["reasoning_content"] == "r1r2"

    def test_preserves_both_reasoning_fields_independently(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(reasoning="R", reasoning_content="RC")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        msg = result["choices"][0]["message"]
        assert msg["reasoning"] == "R"
        assert msg["reasoning_content"] == "RC"

    def test_rejects_non_string_content_or_reasoning(self):
        for field in ("content", "reasoning", "reasoning_content"):
            base = _choice()
            base["delta"][field] = 123
            lines = [
                _data({"choices": [_choice(role="assistant")]}),
                b"",
                _data({"choices": [base]}),
                b"",
                _data({"choices": [_choice(finish="stop")]}),
                b"",
            ]
            with pytest.raises(UpstreamStreamProtocolError) as excinfo:
                reconstruct_openai_sse(lines)
            assert excinfo.value.code == "malformed_chunk"


# ===========================================================================
# Choices and finish reasons
# ===========================================================================


class TestChoicesAndFinishReasons:
    def test_rejects_nonzero_choice_index(self):
        c = _choice(role="assistant")
        c["index"] = 1
        lines = [
            _data({"choices": [c]}),
            b"",
            _data({"choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "unsupported_choice"

    def test_rejects_multiple_choices(self):
        lines = [
            _data({"choices": [_choice(role="assistant"), _choice(content="two")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "unsupported_choice"

    def test_rejects_missing_or_boolean_choice_index(self):
        c1 = _choice(role="assistant")
        del c1["index"]
        c2 = _choice()
        c2["index"] = True
        for bad in (c1, c2):
            lines = [
                _data({"choices": [bad]}),
                b"",
                _data({"choices": [_choice(content="x")]}),
                b"",
                _data({"choices": [_choice(finish="stop")]}),
                b"",
            ]
            with pytest.raises(UpstreamStreamProtocolError) as excinfo:
                reconstruct_openai_sse(lines)
            assert excinfo.value.code == "unsupported_choice"

    @pytest.mark.parametrize("reason", [
        "stop",
        "tool_calls",
        "length",
        "content_filter",
        "provider_custom_reason",
    ])
    def test_preserves_finish_reason_matrix(self, reason):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish=reason)]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["finish_reason"] == reason

    def test_rejects_conflicting_finish_reasons(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            _data({"choices": [_choice(finish="length")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_chunk"

    def test_same_chunk_content_and_finish_reason(self):
        same_chunk = _choice(content="last token", finish="stop")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [same_chunk]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["content"] == "last token"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_rejects_semantic_delta_after_finish_reason(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            _data({"choices": [_choice(content="after stop")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "post_terminal_delta"

    def test_rejects_non_assistant_role(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [{"index": 0, "delta": {"role": "user"}}]}))
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_chunk"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False

    def test_rejects_non_string_role(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [{"index": 0, "delta": {"role": 123}}]}))
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_chunk"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False

    def test_defaults_missing_role_to_assistant(self):
        lines = [
            _data({"choices": [_choice(content="hi")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["choices"][0]["message"]["role"] == "assistant"

    def test_post_terminal_content_rejected(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="a", finish="stop")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(_data({"choices": [_choice(content="after stop")]}))
            acc.feed_line(b"")
        assert excinfo.value.code == "post_terminal_delta"
        assert str(excinfo.value) == "content after finish reason"

    def test_post_terminal_reasoning_content_rejected(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="a", finish="stop")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(_data({"choices": [_choice(reasoning_content="after stop")]}))
            acc.feed_line(b"")
        assert excinfo.value.code == "post_terminal_delta"

    def test_post_terminal_reasoning_rejected(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="a", finish="stop")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(_data({"choices": [_choice(reasoning="after stop")]}))
            acc.feed_line(b"")
        assert excinfo.value.code == "post_terminal_delta"

    def test_post_terminal_tool_calls_rejected(self):
        frag = _tc(0, "c-1", "f", "{}")
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="a", finish="stop")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(_data({"choices": [_choice(tool_calls=[frag])]}))
            acc.feed_line(b"")
        assert excinfo.value.code == "post_terminal_delta"

    def test_post_terminal_benign_role_assistant_accepted(self):
        # MiniMax repeats role:"assistant" (and content:"") in its terminal /
        # duplicate-terminal chunks. A benign, non-conflicting post-terminal
        # role is a no-op, NOT a protocol error.
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="a", finish="stop")]}))
        acc.feed_line(b"")
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="")]}))
        acc.feed_line(b"")
        result = acc.finish()
        assert result["choices"][0]["message"]["content"] == "a"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_post_terminal_invalid_role_still_rejected(self):
        # A post-terminal role that is NOT "assistant" is still rejected.
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="a", finish="stop")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(_data({"choices": [_choice(role="user")]}))
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_chunk"


# ===========================================================================
# Tool calls
# ===========================================================================


class TestToolCalls:
    def test_reconstructs_fragmented_function_name_and_arguments(self):
        frag1 = _tc(0, "call-1", "read_", '{"p')
        frag2 = _tc(0, name="file", arguments='":"/tmp"}')
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag1])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag2])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == '{"p":"/tmp"}'
        assert tc["id"] == "call-1"
        assert tc["type"] == "function"
        assert "index" not in tc

    def test_reconstructs_interleaved_tool_calls_by_explicit_index(self):
        b_first = _tc(1, "c2", "B", "{")
        a_first = _tc(0, "c1", "A", "{")
        close_a = _tc(0, arguments="}")
        close_b = _tc(1, arguments="}")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[b_first])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[a_first])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[close_a, close_b])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "A"
        assert tool_calls[0]["id"] == "c1"
        assert tool_calls[0]["function"]["arguments"] == "{}"
        assert tool_calls[1]["function"]["name"] == "B"
        assert tool_calls[1]["id"] == "c2"
        assert tool_calls[1]["function"]["arguments"] == "{}"

    def test_preserves_exact_argument_string_without_parsing(self):
        frag1 = _tc(0, "c", "f", '{"incomplete')
        frag2 = _tc(0, arguments='"}')
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag1])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag2])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        arguments = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert arguments == '{"incomplete"}'

    def test_rejects_conflicting_tool_call_id(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[_tc(0, "id-a", "f")])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[_tc(0, "id-b")])]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_chunk"

    def test_rejects_invalid_tool_call_index(self):
        for bad in [-1, True, "x", 1.5]:
            bad_frag = {"index": bad, "id": "c", "type": "function", "function": {"name": "f"}}
            lines = [
                _data({"choices": [_choice(role="assistant")]}),
                b"",
                _data({"choices": [_choice(tool_calls=[bad_frag])]}),
                b"",
            ]
            with pytest.raises(UpstreamStreamProtocolError) as excinfo:
                reconstruct_openai_sse(lines)
            assert excinfo.value.code == "malformed_chunk"

    def test_rejects_incomplete_tool_call(self):
        frag = _tc(0, "c")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "incomplete_tool_call"

    def test_does_not_synthesize_missing_tool_call_id(self):
        frag = _tc(0, name="f", arguments="{}")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "incomplete_tool_call"

    def test_rejects_tool_call_with_missing_arguments(self):
        frag = _tc(0, "call-1", "read")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "incomplete_tool_call"

    def test_accepts_explicit_empty_tool_call_arguments(self):
        frag = _tc(0, "call-1", "read", arguments="")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["function"]["arguments"] == ""

    def test_null_arguments_do_not_complete_tool_call(self):
        frag = _tc(0, "call-1", "read", arguments=None)
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "incomplete_tool_call"

    def test_accepts_single_sparse_tool_call_index(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=2,
        )
        frag = _tc(7, "call-7", "read", arguments="{}")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag])]}),
            b"",
            _data({"choices": [_choice(finish="tool_calls")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines, limits=limits)
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call-7"
        assert tool_calls[0]["function"]["name"] == "read"
        assert tool_calls[0]["function"]["arguments"] == "{}"

    def test_tool_call_limit_counts_distinct_sparse_indexes(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=2,
        )
        frag_2 = _tc(2, "c-2", "f2", arguments="{}")
        frag_7 = _tc(7, "c-7", "f7", arguments="{}")
        frag_100 = _tc(100, "c-100", "f100", arguments="{}")
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag_2])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag_7])]}),
            b"",
            _data({"choices": [_choice(tool_calls=[frag_100])]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            reconstruct_openai_sse(lines, limits=limits)
        assert excinfo.value.code == "limit_exceeded"

    def test_invalid_tool_fragment_does_not_create_or_partially_update_call(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        bad_frag = {
            "index": 0,
            "id": "c-0",
            "type": "function",
            "function": {"name": "f0", "arguments": 123},
        }
        acc.feed_line(_data({"choices": [_choice(tool_calls=[bad_frag])]}))
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_chunk"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert acc._tool_calls == {}
        assert acc._tool_argument_bytes == 0
        assert acc._response_bytes == 0

    def test_tool_argument_limit_rejection_discards_whole_fragment_event(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=3, max_tool_calls=10,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        frag = _tc(0, "c-0", "f0", arguments="abcd")
        acc.feed_line(_data({"choices": [_choice(tool_calls=[frag])]}))
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "limit_exceeded"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert acc._tool_calls == {}
        assert acc._tool_argument_bytes == 0
        assert acc._response_bytes == 0

    def test_index_only_tool_fragment_is_retained_and_rejected_as_incomplete(self):
        lines = [
            _data({
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0}
                            ]
                        },
                    }
                ]
            }),
            b"",
            _data({
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ]
            }),
            b"",
        ]

        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)

        assert excinfo.value.code == "incomplete_tool_call"
        assert excinfo.value.semantic_started is True

    def test_index_only_tool_fragment_can_be_completed_by_later_fragment(self):
        first = {
            "index": 0,
        }
        second = {
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": "{}",
            },
        }

        lines = [
            _data({
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [first]},
                    }
                ]
            }),
            b"",
            _data({
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [second]},
                    }
                ]
            }),
            b"",
            _data({
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ]
            }),
            b"",
        ]

        result = reconstruct_openai_sse(lines)
        tool_calls = result["choices"][0]["message"]["tool_calls"]

        assert tool_calls == [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": "{}",
                },
            }
        ]

    def test_first_event_content_category_limit_rejection_preserves_parent(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=3, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"choices": [_choice(content="abcd")]}))
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "limit_exceeded"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert acc._content_parts == []
        assert acc._content_bytes == 0
        assert acc._response_bytes == 0

    def test_first_event_reasoning_category_limit_rejection_preserves_parent(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=3,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"choices": [_choice(reasoning_content="abcd")]}))
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "limit_exceeded"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert acc._reasoning_content_parts == []
        assert acc._reasoning_bytes == 0
        assert acc._response_bytes == 0


# ===========================================================================
# Usage and metadata
# ===========================================================================


class TestUsageAndMetadata:
    def test_preserves_usage_on_terminal_chunk(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="hi")]}),
            b"",
            _data({"choices": [_choice(finish="stop")],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}

    def test_preserves_usage_only_chunk_after_finish_reason(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
            _data({"choices": [], "usage": {"prompt_tokens": 10}}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["usage"] == {"prompt_tokens": 10}

    def test_preserves_nested_prompt_cache_usage(self):
        usage = {
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens": 20,
        }
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(finish="stop")], "usage": usage}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["usage"] == usage

    def test_value_equal_usage_repeat_does_not_conflict(self):
        usage = {"prompt_tokens": 5, "completion_tokens": 3}
        lines = [
            _data({"choices": [_choice(role="assistant")], "usage": usage}),
            b"",
            _data({"choices": [_choice(content="x")], "usage": usage}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["usage"] == usage

    def test_rejects_conflicting_usage(self):
        lines = [
            _data({"choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(finish="stop")], "usage": {"prompt_tokens": 1}}),
            b"",
            _data({"choices": [], "usage": {"prompt_tokens": 2}}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "conflicting_metadata"

    def test_preserves_stable_top_level_metadata(self):
        lines = [
            _data({"id": "chatcmpl-abc", "created": 1720000000, "model": "m",
                   "system_fingerprint": "fp",
                   "choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["id"] == "chatcmpl-abc"
        assert result["created"] == 1720000000
        assert result["model"] == "m"
        assert result["system_fingerprint"] == "fp"
        assert result["object"] == "chat.completion"

    def test_preserves_bounded_unknown_top_level_metadata(self):
        lines = [
            _data({"base_resp": {"status": 0}, "choices": [_choice(role="assistant")]}),
            b"",
            _data({"choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["base_resp"] == {"status": 0}

    def test_value_equal_metadata_repeat_does_not_conflict(self):
        lines = [
            _data({"id": "abc", "model": "m", "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "abc", "model": "m", "choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["id"] == "abc"
        assert result["model"] == "m"

    def test_rejects_conflicting_metadata(self):
        lines = [
            _data({"id": "a", "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "b", "choices": [_choice(content="x")]}),
            b"",
            _data({"choices": [_choice(finish="stop")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "conflicting_metadata"

    def test_rejects_top_level_provider_error_without_leaking_payload(self):
        lines = [
            _data({"error": {"message": "super secret credential leak", "code": 500}}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "malformed_event"
        assert "super secret" not in str(excinfo.value)
        assert "credential" not in str(excinfo.value)

    def test_malformed_choice_event_does_not_retain_metadata_or_usage(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"id": "id-a",
                              "usage": {"prompt_tokens": 1},
                              "choices": [{"index": 0, "delta": 123}]}))
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_chunk"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert acc._response_bytes == 0
        assert acc._top == {}
        assert acc._usage is None
        assert acc._usage_present is False
        assert acc._has_choice is False
        assert acc._content_parts == []
        assert acc._tool_calls == {}


# ===========================================================================
# Semantic-start state
# ===========================================================================


class TestSemanticStartState:
    def test_comments_and_blank_events_do_not_start_semantics(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b": keepalive")
        acc.feed_line(b"")
        acc.feed_line(b": another")
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.semantic_started is False

    def test_role_starts_semantics_before_later_failure(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant")]}))
        acc.feed_line(b"")
        acc.feed_line(b"data: [DONE]")
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"data: [DONE]")
            acc.feed_line(b"")
        assert excinfo.value.semantic_started is True

    def test_content_starts_semantics_before_later_failure(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(content="hi")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.semantic_started is True

    def test_reasoning_starts_semantics_before_later_failure(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(reasoning_content="rc")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.semantic_started is True

    def test_tool_fragment_starts_semantics_before_later_failure(self):
        frag = _tc(0, "c", "f")
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(tool_calls=[frag])]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.semantic_started is True

    def test_finish_reason_starts_semantics_before_later_failure(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(finish="stop")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(_data({"choices": [_choice(content="after")]}))
            acc.feed_line(b"")
        assert excinfo.value.semantic_started is True

    def test_usage_starts_semantics_before_later_failure(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [],
                             "usage": {"prompt_tokens": 1}}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.semantic_started is True

    def test_retained_metadata_starts_semantics_before_later_failure(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"id": "xyz", "choices": []}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.semantic_started is True

    def test_role_only_event_starts_semantics(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [_choice(role="assistant")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.code == "incomplete_response"
        assert acc.semantic_started is True

    def test_malformed_first_event_reports_semantic_started_false(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b"data: {not json")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.semantic_started is False

    def test_empty_delta_does_not_start_semantics(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [{"index": 0, "delta": {}}]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.code == "incomplete_response"
        assert excinfo.value.semantic_started is False

    def test_malformed_delta_reports_semantic_started_false(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [{"index": 0, "delta": 123}]}))
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "malformed_chunk"
        assert excinfo.value.semantic_started is False

    def test_first_empty_content_starts_semantics(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"choices": [{"index": 0, "delta": {"content": ""}}]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.finish()
        assert excinfo.value.code == "incomplete_response"
        assert excinfo.value.semantic_started is True

    def test_rejected_event_preserves_prior_semantic_started(self):
        acc = OpenAIStreamAccumulator()

        first = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "accepted"},
                }
            ]
        }
        acc.feed_line(_data(first))
        acc.feed_line(b"")

        assert acc.semantic_started is True
        prior_response_bytes = acc._response_bytes
        prior_content = list(acc._content_parts)

        acc.feed_line(
            b'data: {"choices":[{"index":0,'
            b'"delta":{"reasoning":"\\ud800"}}]}'
        )
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")

        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is True
        assert acc.semantic_started is True
        assert acc._response_bytes == prior_response_bytes
        assert acc._content_parts == prior_content
        assert acc._reasoning_parts == []

    def test_parent_semantic_state_precedes_retained_state_publication(self):
        observations = []

        class ObservingAccumulator(OpenAIStreamAccumulator):
            def __setattr__(self, name, value):
                if (
                    name == "_content_parts"
                    and "_semantic_started" in self.__dict__
                ):
                    observations.append(self._semantic_started)
                super().__setattr__(name, value)

        acc = ObservingAccumulator()
        observations.clear()

        acc.feed_line(_data({
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "accepted"},
                }
            ]
        }))
        acc.feed_line(b"")

        assert observations == [True]
        assert acc.semantic_started is True
        assert acc._content_parts == ["accepted"]


# ===========================================================================
# Limits
# ===========================================================================


class TestLimits:
    def test_line_byte_limit_boundary(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10, max_event_bytes=100, max_response_bytes=1000,
            max_content_bytes=1000, max_reasoning_bytes=1000,
            max_tool_argument_bytes=1000, max_tool_calls=4,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(b"X" * 10)
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            OpenAIStreamAccumulator(limits=limits).feed_line(b"X" * 11)
        assert excinfo.value.code == "limit_exceeded"

    def test_event_byte_limit_boundary(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=200, max_event_bytes=200, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=4,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(b"data: " + b"X" * 100)
        acc.feed_line(b"data: " + b"X" * 99)
        assert acc._pending_bytes == 200
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc.feed_line(b"data: " + b"X" * 1)
        assert excinfo.value.code == "limit_exceeded"

    def test_allows_event_limit_greater_than_response_limit(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=200, max_event_bytes=200, max_response_bytes=20,
            max_content_bytes=20, max_reasoning_bytes=20,
            max_tool_argument_bytes=20, max_tool_calls=2,
        )
        assert limits.max_event_bytes > limits.max_response_bytes
        assert limits.max_line_bytes <= limits.max_event_bytes

    def test_response_limit_counts_complete_semantic_event_bytes(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        event_payload = {"id": "chatcmpl-x", "choices": [_choice(role="assistant")]}
        expected = _event_bytes(event_payload)
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data(event_payload))
        acc.feed_line(b"")
        assert acc._response_bytes == expected

    def test_response_limit_aggregates_multiple_semantic_events(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        e1 = {"id": "id-1", "choices": [_choice(role="assistant")]}
        e2 = {"choices": [_choice(content="hi")]}
        e3 = {"choices": [_choice(finish="stop")]}
        expected = _event_bytes(e1) + _event_bytes(e2) + _event_bytes(e3)
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data(e1))
        acc.feed_line(b"")
        acc.feed_line(_data(e2))
        acc.feed_line(b"")
        acc.feed_line(_data(e3))
        acc.feed_line(b"")
        assert acc._response_bytes == expected

    def test_response_limit_counts_ignored_nested_semantic_event_fields(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        event_payload = {
            "id": "id-x",
            "ignored_field": "should be counted",
            "nested_ignored": {"k": "v"},
            "choices": [_choice(role="assistant")],
        }
        expected = _event_bytes(event_payload)
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data(event_payload))
        acc.feed_line(b"")
        assert acc._response_bytes == expected

    def test_response_limit_rejection_does_not_charge_partial(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000, max_response_bytes=80,
            max_content_bytes=80, max_reasoning_bytes=80,
            max_tool_argument_bytes=80, max_tool_calls=10,
        )
        e1 = {"id": "id-1", "choices": [_choice(role="assistant")]}
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data(e1))
        acc.feed_line(b"")
        before = acc._response_bytes
        big_event = {
            "id": "id-1",
            "long_padding": "X" * 100,
            "choices": [_choice(finish="stop")],
        }
        acc.feed_line(_data(big_event))
        with pytest.raises(UpstreamStreamLimitError):
            acc.feed_line(b"")
        assert acc._response_bytes == before

    def test_response_limit_counts_repeat_identical_semantic_events(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        e1 = {"id": "id-1", "choices": [_choice(role="assistant")]}
        e2 = {"id": "id-1", "choices": [_choice(role="assistant")]}
        cost1 = _event_bytes(e1)
        cost2 = _event_bytes(e2)
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data(e1))
        acc.feed_line(b"")
        acc.feed_line(_data(e2))
        acc.feed_line(b"")
        assert acc._response_bytes == cost1 + cost2

        tight = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000,
            max_response_bytes=cost1 + cost2 - 1,
            max_content_bytes=cost1 + cost2 - 1,
            max_reasoning_bytes=cost1 + cost2 - 1,
            max_tool_argument_bytes=cost1 + cost2 - 1,
            max_tool_calls=10,
        )
        acc2 = OpenAIStreamAccumulator(limits=tight)
        acc2.feed_line(_data(e1))
        acc2.feed_line(b"")
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc2.feed_line(_data(e2))
            acc2.feed_line(b"")
        assert excinfo.value.code == "limit_exceeded"
        assert acc2._response_bytes == cost1
        assert acc2._top == {"id": "id-1"}

    def test_tool_argument_event_counts_full_json_bytes(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=1000, max_event_bytes=1000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=10,
        )
        frag = _tc(0, "c", "f", arguments="{}")
        event_payload = {"choices": [_choice(tool_calls=[frag])]}
        expected = _event_bytes(event_payload)
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data(event_payload))
        acc.feed_line(b"")
        assert acc._response_bytes == expected

    def test_content_byte_limit_counts_utf8_bytes(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=5, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=4,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"choices": [_choice(role="assistant", content="aa中")]}))
        acc.feed_line(b"")
        assert acc._content_bytes == 5
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc2 = OpenAIStreamAccumulator(limits=limits)
            acc2.feed_line(_data({"choices": [_choice(role="assistant", content="aa中a")]}))
            acc2.feed_line(b"")
        assert excinfo.value.code == "limit_exceeded"
        assert len("aa中a") == 4
        assert len("aa中a".encode("utf-8")) == 6

    def test_reasoning_byte_limit_is_aggregate(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=6,
            max_tool_argument_bytes=10000, max_tool_calls=4,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"choices": [_choice(role="assistant", reasoning_content="abc")]}))
        acc.feed_line(b"")
        acc.feed_line(_data({"choices": [_choice(reasoning="def")]}))
        acc.feed_line(b"")
        assert acc._reasoning_bytes == 6
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc2 = OpenAIStreamAccumulator(limits=limits)
            acc2.feed_line(_data({"choices": [_choice(role="assistant", reasoning_content="abc")]}))
            acc2.feed_line(b"")
            acc2.feed_line(_data({"choices": [_choice(reasoning="defg")]}))
            acc2.feed_line(b"")
            acc2.finish()
        assert excinfo.value.code == "limit_exceeded"

    def test_tool_argument_byte_limit_is_aggregate(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=6, max_tool_calls=4,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        frag1 = _tc(0, "c", "f", "abc")
        frag2 = _tc(0, arguments="def")
        acc.feed_line(_data({"choices": [_choice(role="assistant", tool_calls=[frag1])]}))
        acc.feed_line(b"")
        acc.feed_line(_data({"choices": [_choice(tool_calls=[frag2])]}))
        acc.feed_line(b"")
        assert acc._tool_argument_bytes == 6
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc2 = OpenAIStreamAccumulator(limits=limits)
            f1 = _tc(0, "c", "f", "abc")
            f2 = _tc(0, arguments="defg")
            acc2.feed_line(_data({"choices": [_choice(role="assistant", tool_calls=[f1])]}))
            acc2.feed_line(b"")
            acc2.feed_line(_data({"choices": [_choice(tool_calls=[f2])]}))
            acc2.feed_line(b"")
            acc2.finish()
        assert excinfo.value.code == "limit_exceeded"

    def test_tool_call_count_limit_boundary(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=10000,
            max_tool_argument_bytes=10000, max_tool_calls=2,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"choices": [_choice(role="assistant", tool_calls=[
            _tc(0, "a", "f", "{}"),
            _tc(1, "b", "g", "{}"),
        ])]}))
        acc.feed_line(b"")
        assert len(acc._tool_calls) == 2
        with pytest.raises(UpstreamStreamLimitError) as excinfo:
            acc2 = OpenAIStreamAccumulator(limits=limits)
            acc2.feed_line(_data({"choices": [_choice(role="assistant", tool_calls=[
                _tc(0, "a", "f", "{}"),
                _tc(1, "b", "g", "{}"),
                _tc(2, "c", "h", "{}"),
            ])]}))
            acc2.feed_line(b"")
            acc2.finish()
        assert excinfo.value.code == "limit_exceeded"

    def test_reasoning_limit_failure_does_not_append_rejected_fragment(self):
        limits = UpstreamStreamLimits(
            max_line_bytes=10000, max_event_bytes=10000, max_response_bytes=10000,
            max_content_bytes=10000, max_reasoning_bytes=3,
            max_tool_argument_bytes=10000, max_tool_calls=2,
        )
        acc = OpenAIStreamAccumulator(limits=limits)
        acc.feed_line(_data({"choices": [_choice(role="assistant", reasoning_content="abc")]}))
        acc.feed_line(b"")
        with pytest.raises(UpstreamStreamLimitError):
            acc.feed_line(_data({"choices": [_choice(reasoning="d")]}))
            acc.feed_line(b"")
        assert acc._reasoning_content_parts == ["abc"]
        assert acc._reasoning_parts == []
        assert acc._reasoning_bytes == 3
        assert acc.semantic_started is True


# ===========================================================================
# Recursive lone-surrogate (escaped JSON)
# ===========================================================================


class TestLoneSurrogate:
    def test_lone_surrogate_content_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b'data: {"choices":[{"index":0,"delta":{"content":"\\ud800"}}]}')
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert "\\ud800" not in str(excinfo.value)
        assert "\\ud800" not in repr(excinfo.value)

    def test_lone_surrogate_reasoning_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b'data: {"choices":[{"index":0,"delta":{"reasoning":"\\ud800"}}]}')
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert "\\ud800" not in str(excinfo.value)
        assert "\\ud800" not in repr(excinfo.value)

    def test_lone_surrogate_tool_argument_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
            b'{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"\\ud800"}}'
            b']}}]}'
        )
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert "\\ud800" not in str(excinfo.value)
        assert "\\ud800" not in repr(excinfo.value)

    def test_lone_surrogate_in_metadata_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b'data: {"id":"bad\\ud800value","choices":[]}')
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False
        assert "\\ud800" not in str(excinfo.value)
        assert "\\ud800" not in repr(excinfo.value)

    def test_lone_surrogate_in_nested_metadata_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(b'data: {"base_resp":{"status":0,"k":"bad\\ud800v"},"choices":[]}')
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False

    def test_lone_surrogate_in_usage_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(
            b'data: {"choices":[],'
            b'"usage":{"prompt_tokens":1,"k":"bad\\ud800v"}}'
        )
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False

    def test_lone_surrogate_in_nested_usage_raises_typed_protocol_error(self):
        acc = OpenAIStreamAccumulator()
        acc.feed_line(
            b'data: {"choices":[],'
            b'"usage":{"prompt_tokens":1,"details":{"k":"bad\\ud800v"}}}'
        )
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            acc.feed_line(b"")
        assert excinfo.value.code == "invalid_unicode"
        assert excinfo.value.semantic_started is False
        assert acc.semantic_started is False


# ===========================================================================
# Limit constructor validation
# ===========================================================================


class TestLimitValidation:
    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10", None])
    def test_rejects_non_positive_integer_scalar(self, bad):
        with pytest.raises(ValueError):
            UpstreamStreamLimits(max_line_bytes=bad)

    def test_rejects_line_greater_than_event(self):
        with pytest.raises(ValueError):
            UpstreamStreamLimits(
                max_line_bytes=2000, max_event_bytes=1000, max_response_bytes=10000,
                max_content_bytes=1000, max_reasoning_bytes=1000,
                max_tool_argument_bytes=1000, max_tool_calls=2,
            )

    def test_rejects_content_greater_than_response(self):
        with pytest.raises(ValueError):
            UpstreamStreamLimits(
                max_line_bytes=10, max_event_bytes=20, max_response_bytes=100,
                max_content_bytes=200, max_reasoning_bytes=50,
                max_tool_argument_bytes=50, max_tool_calls=2,
            )

    def test_rejects_reasoning_greater_than_response(self):
        with pytest.raises(ValueError):
            UpstreamStreamLimits(
                max_line_bytes=10, max_event_bytes=20, max_response_bytes=100,
                max_content_bytes=50, max_reasoning_bytes=200,
                max_tool_argument_bytes=50, max_tool_calls=2,
            )

    def test_rejects_tool_argument_greater_than_response(self):
        with pytest.raises(ValueError):
            UpstreamStreamLimits(
                max_line_bytes=10, max_event_bytes=20, max_response_bytes=100,
                max_content_bytes=50, max_reasoning_bytes=50,
                max_tool_argument_bytes=200, max_tool_calls=2,
            )


# ===========================================================================
# Isolation
# ===========================================================================


class TestIsolation:
    def test_parser_calls_do_not_share_state(self):
        r1 = reconstruct_openai_sse(_text_response("first"))
        r2 = reconstruct_openai_sse(_text_response("second"))
        assert r1["choices"][0]["message"]["content"] == "first"
        assert r2["choices"][0]["message"]["content"] == "second"

    def test_default_limits_are_immutable(self):
        with pytest.raises((AttributeError, TypeError)):
            DEFAULT_UPSTREAM_STREAM_LIMITS.max_line_bytes = 5  # type: ignore[misc]

    def test_public_parser_defaults_use_canonical_limits_instance(self):
        reconstruct_default = inspect.signature(
            reconstruct_openai_sse
        ).parameters["limits"].default

        accumulator_default = inspect.signature(
            OpenAIStreamAccumulator
        ).parameters["limits"].default

        assert reconstruct_default is DEFAULT_UPSTREAM_STREAM_LIMITS
        assert accumulator_default is DEFAULT_UPSTREAM_STREAM_LIMITS


# ---------------------------------------------------------------------------
# Real MiniMax regression (ground truth captured via PROXY_CAPTURE_UPSTREAM_SSE)
# ---------------------------------------------------------------------------


class TestMiniMaxRealStreamRegression:
    """Pins the exact real-MiniMax quirks that crashed the strict parser
    (commit 3c28a63 -> revert af5bea0). Bytes captured from MiniMax-M3 via
    PROXY_CAPTURE_UPSTREAM_SSE. Quirks: role in EVERY delta (incl. terminal),
    a DUPLICATE terminal chunk carrying role + content:"", a tool call split
    across two chunks by index, `reasoning` duplicated with `content`, and NO
    [DONE] (clean EOF). Revert-test: restoring the strict post-terminal
    rejection makes this raise 'role after finish reason'.
    """

    def _minimax_lines(self):
        return [
            b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"<think>\\nThe","role":"assistant","reasoning":"The"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
            b'data: {"id":"x","choices":[{"index":0,"delta":{"content":" plan","role":"assistant","reasoning":" plan"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
            b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"\\n</think>\\n\\n</think>","role":"assistant"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
            b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"id":"call_abc","type":"function","function":{"name":"bash","arguments":""},"index":0}]}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
            b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"command\\":\\"echo \\\\\\"Hello\\\\\\"\\"}"},"index":0}]}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
            b'data: {"id":"x","choices":[{"finish_reason":"tool_calls","index":0,"delta":{"content":"","role":"assistant"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
            b'data: {"id":"x","choices":[{"finish_reason":"tool_calls","index":0,"delta":{"content":"","role":"assistant"}}],"model":"MiniMax-M3","object":"chat.completion.chunk","usage":null,"service_tier":"standard"}',
            b"",
        ]

    def test_minimax_stream_reconstructs_without_error(self):
        result = reconstruct_openai_sse(self._minimax_lines())
        choice = result["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        msg = choice["message"]
        # content preserves the <think> block verbatim (incl. doubled </think>)
        assert msg["content"] == "<think>\nThe plan\n</think>\n\n</think>"
        # single tool call, reconstructed across the two fragment chunks by index
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "bash"
        assert tc["function"]["arguments"] == '{"command":"echo \\"Hello\\""}'

    def test_minimax_nonempty_content_after_finish_still_rejected(self):
        """The relaxation is surgical: a NON-empty content fragment after the
        terminal finish is still a protocol error (real corruption)."""
        lines = [
            b'data: {"choices":[{"index":0,"delta":{"content":"hi","role":"assistant"},"finish_reason":"stop"}]}',
            b"",
            b'data: {"choices":[{"index":0,"delta":{"content":"MORE AFTER FINISH"}}]}',
            b"",
        ]
        with pytest.raises(UpstreamStreamError) as ei:
            reconstruct_openai_sse(lines)
        assert ei.value.code == "post_terminal_delta"


# ===========================================================================
# Created-drift tolerance (GLM via ollama.com increments `created` per chunk)
# ===========================================================================


class TestCreatedDrift:
    """GLM via ollama.com re-emits `created` with a fresh wall-clock-second
    Unix timestamp per chunk. Strict equality on `created` falsely flags a
    valid completion as conflicting_metadata → 502. This relaxation is
    scoped to `created` ONLY — `id`, `model`, `system_fingerprint`, and
    arbitrary unknown top-level keys still raise on conflict (the
    spliced-completion detector for those fields must remain).
    """

    def test_created_drifting_chunks_succeed_first_value_wins(self):
        """GLM via ollama.com increments `created` per chunk; first value must win."""
        lines = [
            _data({"id": "x", "model": "m", "created": 1720000000,
                   "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000001,
                   "choices": [_choice(content="Hi")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000002,
                   "choices": [_choice(finish="stop")]}),
            b"",
            b"data: [DONE]",
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["created"] == 1720000000  # FIRST value wins
        assert result["choices"][0]["message"]["content"] == "Hi"

    def test_created_drift_revert_test_raises_under_strict(self):
        """Pin: pre-fix strict equality raises conflicting_metadata on created drift.
        This is the revert-test: the same lines succeed under the fix and raise here
        if the relaxation is reverted.
        """
        lines = [
            _data({"id": "x", "model": "m", "created": 1720000000,
                   "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000001,
                   "choices": [_choice(content="Hi")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000002,
                   "choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["created"] == 1720000000

    def test_id_drift_still_rejects(self):
        """Scope guard: relaxing `created` must NOT relax `id`.
        Drifting id is the spliced-completion data-integrity detector; it must still raise."""
        lines = [
            _data({"id": "a", "model": "m", "created": 1, "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "b", "model": "m", "created": 2, "choices": [_choice(content="x")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "conflicting_metadata"

    def test_model_drift_still_rejects(self):
        """Scope guard: relaxing `created` must NOT relax `model`."""
        lines = [
            _data({"id": "x", "model": "m1", "created": 1, "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "x", "model": "m2", "created": 1, "choices": [_choice(content="x")]}),
            b"",
        ]
        with pytest.raises(UpstreamStreamProtocolError) as excinfo:
            reconstruct_openai_sse(lines)
        assert excinfo.value.code == "conflicting_metadata"

    def test_created_out_of_order_first_seen_wins(self):
        """First-seen value wins, NOT max/last-wins."""
        lines = [
            _data({"id": "x", "model": "m", "created": 1720000002,
                   "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000000,
                   "choices": [_choice(content="x")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000001,
                   "choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["created"] == 1720000002  # first-seen wins

    def test_created_late_adoption(self):
        """First chunk has no created; the first chunk that carries one wins."""
        lines = [
            _data({"id": "x", "model": "m", "choices": [_choice(role="assistant")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000099,
                   "choices": [_choice(content="x")]}),
            b"",
            _data({"id": "x", "model": "m", "created": 1720000100,
                   "choices": [_choice(finish="stop")]}),
            b"",
        ]
        result = reconstruct_openai_sse(lines)
        assert result["created"] == 1720000099  # first-SEEN value wins

    def test_full_stream_completeness_under_drift(self):
        """Drive the full finish() lifecycle through OpenAIStreamAccumulator
        (not reconstruct_openai_sse) to exercise semantic_started +
        has_choice + finish_reason invariants under created drift.
        """
        acc = OpenAIStreamAccumulator()
        acc.feed_line(_data({"id": "x", "model": "m", "created": 1, "choices": [_choice(role="assistant")]}))
        acc.feed_line(b"")
        acc.feed_line(_data({"id": "x", "model": "m", "created": 2, "choices": [_choice(content="Hi")]}))
        acc.feed_line(b"")
        acc.feed_line(_data({"id": "x", "model": "m", "created": 3, "choices": [_choice(finish="stop")]}))
        acc.feed_line(b"")
        acc.feed_line(b"data: [DONE]")
        acc.feed_line(b"")
        result = acc.finish()
        assert result["created"] == 1
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["message"]["content"] == "Hi"


# ---------------------------------------------------------------------------
# GLM-ollama.com regression (synthetic reproduction of captured quirks;
# original capture files purged for the open-source release)
# ---------------------------------------------------------------------------


class TestGLMOllamaRealStreamRegression:
    """Pins the GLM-ollama.com quirk that crashed the strict parser:
    `created` increments per chunk. Synthetic bytes reproduce the shape of
    the original GLM-5.2:cloud captures (https://ollama.com/v1/chat/completions).
    Revert-test: restoring strict `created` equality makes both streams raise
    'conflicting_metadata'.
    """

    def _glm_text_lines(self):
        return [
            b'data: {"id":"chatcmpl-259","object":"chat.completion.chunk","created":1784841550,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi! What"}}]}',
            b"",
            b'data: {"id":"chatcmpl-259","object":"chat.completion.chunk","created":1784841551,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"content":" would you like to do next?"}}]}',
            b"",
            b'data: {"id":"chatcmpl-259","object":"chat.completion.chunk","created":1784841551,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            b"",
            b"data: [DONE]",
            b"",
        ]

    def _glm_tool_call_lines(self):
        return [
            # reasoning_content co-emitted with content in the SAME delta
            b'data: {"id":"chatcmpl-240","object":"chat.completion.chunk","created":1784834756,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Reading example.txt to check the","content":"Continuing D2 implementation"}}]}',
            b"",
            b'data: {"id":"chatcmpl-240","object":"chat.completion.chunk","created":1784834758,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"reasoning_content":" header before editing."}}]}',
            b"",
            # tool call opens
            b'data: {"id":"chatcmpl-240","object":"chat.completion.chunk","created":1784834759,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_dmhoy0fx","type":"function","function":{"name":"read","arguments":""}}]}}]}',
            b"",
            # arguments fragmented across two chunks
            b'data: {"id":"chatcmpl-240","object":"chat.completion.chunk","created":1784834759,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"filePath\\":\\"/home/dev/projects/sa"}}]}}]}',
            b"",
            b'data: {"id":"chatcmpl-240","object":"chat.completion.chunk","created":1784834759,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"mple-app/example.txt\\",\\"limit\\":10}"}}]}}]}',
            b"",
            b'data: {"id":"chatcmpl-240","object":"chat.completion.chunk","created":1784834759,"model":"glm-5.2:cloud","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            b"",
        ]

    def test_glm_text_stream_reconstructs_without_error(self):
        """GLM text stream with per-chunk `created` drift reconstructs cleanly."""
        result = reconstruct_openai_sse(self._glm_text_lines())
        # canonical top-level metadata survives
        assert result["id"] == "chatcmpl-259"
        assert result["model"] == "glm-5.2:cloud"
        assert result["system_fingerprint"] == "fp_ollama"
        # first-seen `created` wins (1784841550), not the terminal value (1784841551)
        assert result["created"] == 1784841550
        assert result["choices"][0]["message"]["content"] == "Hi! What would you like to do next?"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_glm_tool_call_stream_reconstructs_without_error(self):
        """GLM tool-call stream with per-chunk `created` drift and
        reasoning+content co-emission reconstructs cleanly.
        """
        result = reconstruct_openai_sse(self._glm_tool_call_lines())
        # canonical top-level metadata survives
        assert result["id"] == "chatcmpl-240"
        assert result["model"] == "glm-5.2:cloud"
        assert result["system_fingerprint"] == "fp_ollama"
        # first-seen `created` wins (1784834756), not the terminal value (1784834759)
        assert result["created"] == 1784834756
        msg = result["choices"][0]["message"]
        assert msg["content"].startswith("Continuing D2 implementation")
        # reasoning text was emitted alongside content and reconstructed
        assert "example.txt" in msg.get("reasoning_content", "")
        # single tool call with read function
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = msg["tool_calls"]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "call_dmhoy0fx"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "read"
        # arguments reassembled across fragments and decoded to a JSON object
        import json as _json
        args = _json.loads(tc["function"]["arguments"])
        assert args["filePath"].endswith("example.txt")
