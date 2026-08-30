"""Regression tests for ToolPipeline._format() finish_reason override (site 2).

Commit 40e6ee1 (M3-4) applied the truncation-guard
(`result.finish_reason not in TRUNCATION_REASONS`) at TWO sites in
src/proxy/tool_pipeline/tool_pipeline.py:

- Site 1 (line 267-272): passthrough path, covered by
  tests/integration/test_finish_reason_consistency_integration.py (via
  Flask request → pipeline.run → site 1).
- Site 2 (line 594-599): _format() in the memory-tool-loop path,
  UNTESTED. Every integration test uses passthrough tools (read_file)
  which hit site 1, never site 2.

These tests exercise site 2 directly by constructing the inputs to
_format() and asserting the resulting data["choices"][0]["finish_reason"].

Boundary cases covered:

1. finish_reason="length"          + tool_calls → preserved as "length"
                                    (truncation signal — client continues)
2. finish_reason="content_filter"  + tool_calls → preserved (refusal signal)
3. finish_reason="stop"            + tool_calls → overridden to "tool_calls"
                                    (spec happy path)
4. finish_reason=None              + tool_calls → finish_reason in `data`
                                    is left untouched (documents the
                                    `is not None` guard — pre-fix it
                                    would have set None/"tool_calls"
                                    inconsistently across sites).

If a future revert removes the truncation guard from site 2 (or
inverts the order of the `is not None`/`not in TRUNCATION_REASONS`
checks), these tests will catch it immediately.
"""
from unittest.mock import MagicMock

from proxy.tool_pipeline.tool_pipeline import (
    ToolContext,
    ToolPipeline,
    TRUNCATION_REASONS,
)


def _make_ctx(session_id=""):
    """Empty session_id skips the chat_histories side-effect in _format()."""
    return ToolContext(
        session_id=session_id,
        body={"model": "test-model"},
        llama_body={"messages": [], "tools": [], "model": "test-model"},
        auth_headers={},
        original_model="test-model",
        upstream_url="http://upstream/v1",
    )


def _make_data(initial_finish_reason="stop"):
    """Build a minimal OpenAI-style response_data dict.

    The message has tool_calls already present so _format() enters the
    `if message.get("tool_calls"):` branch (the truncated-guard path).
    """
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "recall", "arguments": '{"q": "x"}'},
                    }
                ],
            },
            "finish_reason": initial_finish_reason,
        }]
    }


def _mock_result(tool_calls=None, content=None, finish_reason=None,
                 messages_for_history=None):
    """Build a minimal result object compatible with _format()."""
    r = MagicMock()
    r.tool_calls = tool_calls
    r.content = content
    r.finish_reason = finish_reason
    r.messages_for_history = messages_for_history
    return r


class TestFormatFinishReasonTruncationGuard:
    """Site 2 of the M3-4 finish_reason truncation guard."""

    def test_length_with_tool_calls_preserves_truncation_signal(self):
        """finish_reason='length' + tool_calls → MUST stay 'length'.

        The client uses 'length' to decide whether to continue the
        turn (the LLM ran out of tokens mid-call). Overriding to
        'tool_calls' would silently drop the truncation signal and
        the client would stop the loop early, losing the rest of
        the tool execution.
        """
        pipeline = ToolPipeline()
        data = _make_data(initial_finish_reason="length")
        result = _mock_result(
            tool_calls=data["choices"][0]["message"]["tool_calls"],
            finish_reason="length",
        )
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "length", (
            "truncation signal must NOT be overridden to 'tool_calls' — "
            "the client uses it to decide whether to continue"
        )

    def test_content_filter_with_tool_calls_preserves_refusal_signal(self):
        """finish_reason='content_filter' + tool_calls → MUST stay
        'content_filter'. Same rationale as 'length': the upstream
        refused the request for safety reasons and that signal must
        reach the client unchanged.
        """
        pipeline = ToolPipeline()
        data = _make_data(initial_finish_reason="content_filter")
        result = _mock_result(
            tool_calls=data["choices"][0]["message"]["tool_calls"],
            finish_reason="content_filter",
        )
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "content_filter", (
            "content_filter refusal signal must be preserved when "
            "tool_calls are present"
        )

    def test_stop_with_tool_calls_overrides_to_tool_calls(self):
        """finish_reason='stop' + tool_calls → 'tool_calls' (spec happy
        path). Per OpenAI spec, when the assistant emits tool_calls,
        finish_reason must be 'tool_calls' regardless of the upstream
        completion signal (as long as it's not a truncation/refusal).
        """
        pipeline = ToolPipeline()
        data = _make_data(initial_finish_reason="stop")
        result = _mock_result(
            tool_calls=data["choices"][0]["message"]["tool_calls"],
            finish_reason="stop",
        )
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "tool_calls", (
            "non-truncation, non-refusal finish_reason must be "
            "overridden to 'tool_calls' when tool_calls are present"
        )

    def test_none_with_tool_calls_leaves_finish_reason_untouched(self):
        """finish_reason=None + tool_calls → data['choices'][0]
        ['finish_reason'] is NOT mutated.

        Documents the `and result.finish_reason is not None` guard in
        _format (line 596). Without the guard, pre-fix code would have
        overridden 'stop' to 'tool_calls' even when the upstream
        returned no signal, creating site-1/site-2 inconsistency.
        A future revert that drops the `is not None` check would
        start mutating finish_reason here and this test would fail.
        """
        pipeline = ToolPipeline()
        data = _make_data(initial_finish_reason="stop")
        result = _mock_result(
            tool_calls=data["choices"][0]["message"]["tool_calls"],
            finish_reason=None,
        )
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "stop", (
            "finish_reason=None must not mutate data; the `is not None` "
            "guard preserves any prior value (e.g. 'stop' from the "
            "passthrough site). Pre-fix, this would have become "
            "'tool_calls' inconsistently with site 1."
        )

    def test_truncation_reasons_constant_matches_openai_spec(self):
        """TRUNCATION_REASONS must be exactly {'length', 'content_filter'}
        — pin the contract so an accidental rename (e.g. 'len',
        'max_tokens') is caught. Both site 1 and site 2 share this
        constant; a divergence would re-introduce the M3-4 bug.
        """
        assert TRUNCATION_REASONS == frozenset({"length", "content_filter"})


class TestFormatFinishReasonNoToolCalls:
    """When result.tool_calls is empty/None, _format() falls through to
    the else branch (line 598-599) and assigns result.finish_reason
    directly. These tests document that branch's behavior so a future
    edit to the if/else structure doesn't accidentally swap the
    override rule.
    """

    def test_no_tool_calls_uses_result_finish_reason_verbatim(self):
        """Empty tool_calls → finish_reason in data equals
        result.finish_reason directly (no override)."""
        pipeline = ToolPipeline()
        data = _make_data(initial_finish_reason="length")
        empty_tcs = []
        result = _mock_result(
            tool_calls=empty_tcs,
            finish_reason="length",
        )
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "length"

    def test_no_tool_calls_assigns_stop_when_upstream_stop(self):
        """Empty tool_calls + finish_reason='stop' → data stays 'stop'."""
        pipeline = ToolPipeline()
        data = _make_data(initial_finish_reason="stop")
        result = _mock_result(tool_calls=[], finish_reason="stop")
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "stop"


class TestFormatMemoryLoopPathFinishReasonLive:
    """Proves the 40e6ee1 site-2 guard is now LIVE for the memory-tool-loop
    path (not just unreachable).

    Pre-fix, run_memory_tool_loop clobbered its ToolLoopResult.finish_reason
    to "stop" for every content-only memory resolution, so even though
    _format() had the truncation guard, result.finish_reason was never a
    TRUNCATION_REASONS member — the guard was unreachable. Now that the tool
    loop preserves the upstream "length" signal, a content-only "length"
    result flows through _format() unchanged to the client.
    """

    def test_memory_loop_length_result_flows_through_format(self):
        """A ToolLoopResult-shaped result with finish_reason='length' and NO
        tool_calls passes through _format() as 'length' (the guard is live)."""
        pipeline = ToolPipeline()
        # No tool_calls on the message OR the result — content-only memory
        # resolution path.
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "memory answer"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = _mock_result(
            tool_calls=None,
            content="memory answer",
            finish_reason="length",
        )
        ctx = _make_ctx()

        pipeline._format(data, result, ctx)

        assert data["choices"][0]["finish_reason"] == "length", (
            "the memory-loop path must now propagate a 'length' finish_reason "
            "through _format() to the client — the site-2 guard is live"
        )


class TestFormatClosureEnforcement:
    """Slice 3 closure enforcement on the _format path. These tests use
    real ToolLoopResult instances — no MagicMock closures — to pin the
    defense-in-depth behavior on the tool-pipeline boundary."""

    def _real_result(self, **kwargs):
        from memory.tool_loop import (
            ToolLoopResult,
            TERMINATION_REQUERY_FAILED,
        )
        defaults = dict(
            content=None,
            tool_calls=None,
            finish_reason="tool_calls",
            messages_for_history=[],
            execution_records=[{
                "tool": "proxy_ticket",
                "operation": "create_tasks",
                "status": "error",
                "iteration": 1,
                "attempt": 1,
                "error": {"code": "tool_execution_failed", "message": "boom"},
                "data": None,
            }],
            needs_closure=False,
            termination_reason=TERMINATION_REQUERY_FAILED,
            closure_source=None,
        )
        defaults.update(kwargs)
        return ToolLoopResult(**defaults)

    def test_unresolved_needs_closure_strips_tool_calls_and_emits_fallback(self):
        """A ToolLoopResult with needs_closure=True (defense-in-depth
        case) must NEVER raise into ToolPipeline.run's catch-all. The
        _format layer strips the raw tool_calls and renders the
        deterministic fallback via the lazy helper import — even if
        the upstream tool_loop layer failed to resolve closure."""
        pipeline = ToolPipeline()
        openai_tool_call = {
            "id": "tc1", "type": "function",
            "function": {"name": "x", "arguments": "{}"},
        }
        data = {
            "choices": [{
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [openai_tool_call],
                },
                "finish_reason": "tool_calls",
            }]
        }
        result = self._real_result(
            tool_calls=[openai_tool_call],
            needs_closure=True,
        )
        ctx = _make_ctx()

        out = pipeline._format(data, result, ctx)

        assert isinstance(out, dict)
        assert out["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in out["choices"][0]["message"]
        content = out["choices"][0]["message"]["content"]
        assert content, "deterministic fallback content must be non-empty"
        assert content.startswith("FAILED"), (
            f"unresolved closure must emit FAILED prefix at position 0, "
            f"got {content!r}"
        )
        assert "boom" in content
        assert result.needs_closure is False
        assert result.closure_source == "fallback"

    def test_resolved_closure_with_empty_content_replaced_by_fallback(self):
        """A dedicated closure result with empty/non-substantive content
        must be rejected — _format replaces it via the deterministic
        fallback renderer."""
        pipeline = ToolPipeline()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }]
        }
        result = self._real_result(
            content="",
            finish_reason="stop",
            needs_closure=False,
            termination_reason="model_completed",
            closure_source="llm",
        )
        ctx = _make_ctx()

        out = pipeline._format(data, result, ctx)

        content = out["choices"][0]["message"]["content"]
        assert content, "empty closure content must be replaced"
        assert content.startswith("FAILED")
        assert result.closure_source == "fallback"

    def test_clean_closure_passes_through(self):
        """A properly resolved LLM closure with substantive content
        passes through unchanged."""
        pipeline = ToolPipeline()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }]
        }
        result = self._real_result(
            content="COMPLETE All operations succeeded.",
            finish_reason="stop",
            needs_closure=False,
            termination_reason="model_completed",
            closure_source="llm",
            execution_records=[{
                "tool": "list_tasks",
                "operation": "list_tasks",
                "status": "success",
                "iteration": 1,
                "attempt": 1,
                "error": None,
                "data": None,
            }],
        )
        ctx = _make_ctx()

        out = pipeline._format(data, result, ctx)

        assert out["choices"][0]["message"]["content"] == "COMPLETE All operations succeeded."
        assert out["choices"][0]["finish_reason"] == "stop"
        assert result.closure_source == "llm"
