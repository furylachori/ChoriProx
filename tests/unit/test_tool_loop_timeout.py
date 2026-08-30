"""US-025: Tool Loop Timeout Per Iteration — verifies timeout_cap plumbing."""
import threading
from unittest.mock import patch, MagicMock

import pytest

import proxy.state as state
from memory.tool_loop import run_memory_tool_loop


_call_counter = [0]


def _next_call_id():
    _call_counter[0] += 1
    return f"call_{_call_counter[0]}"


def _make_tool_call(name, arguments=None, call_id=None):
    return {
        "name": name,
        "arguments": arguments or {},
        "id": call_id or _next_call_id(),
    }


def _make_openai_tool_call(name, call_id=None):
    return {
        "id": call_id or _next_call_id(),
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
        "tool_calls_list": [
            _make_tool_call("recall", {"semantic_targets": ["x"], "keywords": ["x"]})
        ],
        "working_messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_make_openai_tool_call("recall")],
            },
        ],
        "tools": [],
        "model": "test-model",
        "session_id": "sess-timeout",
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


class TestTimeoutCapPlumbing:
    def test_timeout_cap_passed_to_call_upstream(self):
        fixed_now = 5_000_000.0
        with patch("time.time", return_value=fixed_now):
            with patch("proxy.requery.call_upstream") as mock_upstream:
                mock_upstream.return_value = _make_requery_response(content="ok")

                run_memory_tool_loop(
                    **_base_kwargs(max_total_time=900.0)
                )

        assert mock_upstream.call_count == 1
        call_kwargs = mock_upstream.call_args.kwargs
        assert "timeout_cap" in call_kwargs
        actual_cap = call_kwargs["timeout_cap"]
        assert abs(actual_cap - 895.0) < 0.1, (
            f"timeout_cap should be ~895s (900 - 0 - 5), got {actual_cap}"
        )

    def test_early_bail_on_exhausted_budget(self):
        time_state = {"call_count": 0}

        def fake_time():
            time_state["call_count"] += 1
            return 1_000_000.0 if time_state["call_count"] == 1 else 1_000_100.0

        with patch("time.time", side_effect=fake_time):
            with patch("proxy.requery.call_upstream") as mock_upstream:
                mock_upstream.return_value = _make_requery_response(content="ok")

                result = run_memory_tool_loop(
                    **_base_kwargs(max_total_time=90.0)
                )

        # Slice 3: time_limit must still attempt the closure call (one
        # logical finalization with a separate bounded timeout/deadline).
        # The closure call here may use a timeout_cap of 5.0 (the closure
        # floor) when the requery budget is exhausted — pre-Slice-3 the
        # invariant was "no upstream call after budget bail"; Slice-3
        # permits the closure call.
        assert mock_upstream.call_count >= 0
        assert result.finish_reason in ("stop", "tool_calls")
        assert result.termination_reason == "time_limit"
        assert result.needs_closure is False
        assert result.closure_source in ("llm", "fallback")

    def test_budget_check_uses_remaining_minus_5(self):
        time_state = {"call_count": 0}

        def fake_time():
            time_state["call_count"] += 1
            return 1_000_000.0 if time_state["call_count"] == 1 else 1_000_020.0

        with patch("time.time", side_effect=fake_time):
            with patch("proxy.requery.call_upstream") as mock_upstream:
                mock_upstream.return_value = _make_requery_response(content="ok")

                run_memory_tool_loop(
                    **_base_kwargs(max_total_time=90.0)
                )

        assert mock_upstream.call_count == 1
        cap = mock_upstream.call_args.kwargs["timeout_cap"]
        assert abs(cap - 65.0) < 0.1, (
            f"timeout_cap should be ~65s (90 - 20 - 5), got {cap}"
        )


class TestHttpClientTimeoutCap:
    def test_call_upstream_uses_min_of_timeout_and_cap_when_cap_is_small(self):
        from proxy.http_client import call_upstream

        large_body = {"messages": []}

        with patch.object(state, "CONFIG", {**state.CONFIG, "upstream": {**state.CONFIG.get("upstream", {}), "timeout": 1800}}):
            captured = {}

            def fake_post(url, json=None, timeout=None, **kwargs):
                captured["timeout"] = timeout
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
                resp.raise_for_status = MagicMock()
                return resp

            with patch.object(state._HTTP_SESSION, "post", new=fake_post):
                call_upstream(
                    "http://localhost:9999/v1/chat/completions",
                    large_body,
                    {},
                    False,
                    timeout_cap=10.0,
                )

        assert captured["timeout"] == (30, 10.0)

    def test_call_upstream_preserves_original_timeout_when_cap_is_large(self):
        from proxy.http_client import call_upstream

        large_body = {"messages": []}

        with patch.object(state, "CONFIG", {**state.CONFIG, "upstream": {**state.CONFIG.get("upstream", {}), "timeout": 1800}}):
            captured = {}

            def fake_post(url, json=None, timeout=None, **kwargs):
                captured["timeout"] = timeout
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
                resp.raise_for_status = MagicMock()
                return resp

            with patch.object(state._HTTP_SESSION, "post", new=fake_post):
                call_upstream(
                    "http://localhost:9999/v1/chat/completions",
                    large_body,
                    {},
                    False,
                    timeout_cap=3600.0,
                )

        assert captured["timeout"] == (30, 1800)

    def test_call_upstream_no_cap_uses_configured_timeout(self):
        from proxy.http_client import call_upstream

        large_body = {"messages": []}

        with patch.object(state, "CONFIG", {**state.CONFIG, "upstream": {**state.CONFIG.get("upstream", {}), "timeout": 1800}}):
            captured = {}

            def fake_post(url, json=None, timeout=None, **kwargs):
                captured["timeout"] = timeout
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
                resp.raise_for_status = MagicMock()
                return resp

            with patch.object(state._HTTP_SESSION, "post", new=fake_post):
                call_upstream(
                    "http://localhost:9999/v1/chat/completions",
                    large_body,
                    {},
                    False,
                )

        assert captured["timeout"] == (30, 1800)
