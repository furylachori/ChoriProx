"""CP3-3: focused unit tests for ``_materialize_artifact`` + ``_freeze_stream_identity``.

The completion worker (NOT delivery) renders the terminal SSE artifact exactly
ONCE, off the request thread, with frozen id/created. These tests pin:

  1. render happens exactly once, in a ``completion-worker-*`` pool thread;
  2. freeze stamps missing id/created so replay is byte-stable;
  3. materialize is byte-stable when id/created are already present;
  4. materialize captures no Flask request context (no jsonify);
  5. stage_abort uses ``_unwrap_stage_abort`` (not the jsonify path);
  6. a materialize exception still lets the worker publish (event set).
"""
import itertools
import threading
import types

import pytest

import proxy.app as app
import proxy.completion_coordinator as cc
import proxy.streaming as streaming


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinator_state():
    """Snapshot/restore coordinator module globals so tests don't leak into the
    shared singleton executor (mirrors test_completion_coordinator.py)."""
    orig_sem, orig_pending, orig_shutdown = cc._semaphore, cc._pending, cc._shutdown
    yield
    cc._semaphore, cc._pending, cc._shutdown = orig_sem, orig_pending, orig_shutdown


def _ctx():
    return types.SimpleNamespace(
        request_id="r",
        session_id="s",
        upstream=types.SimpleNamespace(original_model="m"),
    )


def _success_result(with_identity=True):
    result = {
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
    }
    if with_identity:
        result["id"] = "chatcmpl-fixed"
        result["created"] = 1234567890
    return result


def _success_outcome(with_identity=True):
    return {
        "result": _success_result(with_identity=with_identity),
        "stage_abort": None,
        "backend_error": None,
        "error": None,
    }


def _stage_abort_outcome():
    return {
        "result": None,
        "stage_abort": {
            "id": "chatcmpl-abort",
            "created": 7,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "aborted"},
                    "finish_reason": "stop",
                }
            ],
        },
        "backend_error": None,
        "error": None,
    }


def _backend_error_outcome():
    from proxy.types import BackendError

    return {
        "result": None,
        "stage_abort": None,
        "backend_error": BackendError(502, {"error": {"message": "upstream boom"}}),
        "error": None,
    }


def _error_outcome():
    return {
        "result": None,
        "stage_abort": None,
        "backend_error": None,
        "error": RuntimeError("boom"),
    }


# ---------------------------------------------------------------------------
# 1. Render happens exactly once, in a completion-worker pool thread.
# ---------------------------------------------------------------------------


def test_materialize_once_in_worker_thread(coordinator_state):
    calls = []
    real_stream_response = streaming.stream_response

    def _spy(message, original_data, original_model=None):
        calls.append(threading.current_thread().name)
        yield from real_stream_response(message, original_data, original_model)

    streaming.stream_response = _spy
    try:
        cc._semaphore = threading.Semaphore(4)
        cc._pending = 0
        cc._shutdown = False

        class _P:
            def run(self, ctx):
                return _success_result(with_identity=True)

        ctx = _ctx()
        record = cc.admit(_P(), ctx, materialize=app._materialize_artifact).record
        assert record.completion_event.wait(timeout=5)

        chunks = list(app._keepalive_stream(ctx, record))
    finally:
        streaming.stream_response = real_stream_response

    # Rendered EXACTLY once — in the worker thread. Delivery re-emits the
    # frozen artifact verbatim (no second render).
    assert len(calls) == 1
    assert calls[0].startswith("completion-worker-")
    assert "".join(chunks).count("data: [DONE]\n\n") == 1


# ---------------------------------------------------------------------------
# 2. Freeze makes missing id/created deterministic (structural detector).
# ---------------------------------------------------------------------------


def test_freeze_makes_missing_id_created_deterministic(monkeypatch):
    # An incrementing time source: if the freeze were reverted, stream_response's
    # int(time.time()) fallback would fire and produce a DIFFERENT id/created on
    # each call, breaking the byte-stability assertion below.
    counter = itertools.count(1000)
    monkeypatch.setattr(streaming.time, "time", lambda: next(counter))

    outcome = _success_outcome(with_identity=False)
    assert "id" not in outcome["result"]
    assert "created" not in outcome["result"]

    art1 = app._materialize_artifact(outcome, _ctx())

    # STRUCTURAL freeze detector: only the freeze writes id/created back onto
    # the result dict; stream_response never writes them back into original_data.
    assert "id" in outcome["result"]
    assert "created" in outcome["result"]

    art2 = app._materialize_artifact(outcome, _ctx())

    # Frozen → identical bytes across renders.
    assert art1.terminal_chunks == art2.terminal_chunks


# ---------------------------------------------------------------------------
# 3. Positive control: id/created already present → byte-stable.
# ---------------------------------------------------------------------------


def test_materialize_with_id_created_is_byte_stable():
    outcome = _success_outcome(with_identity=True)
    art1 = app._materialize_artifact(outcome, _ctx())
    art2 = app._materialize_artifact(outcome, _ctx())
    assert art1.terminal_chunks == art2.terminal_chunks
    assert art1.terminal_chunks[-1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 4. Materialize runs off the request thread with NO Flask context.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome_factory",
    [
        _success_outcome,
        _stage_abort_outcome,
        _backend_error_outcome,
        _error_outcome,
    ],
)
def test_materialize_runs_off_request_thread_no_context(monkeypatch, outcome_factory):
    def _boom(*a, **k):
        raise AssertionError("jsonify must not be called during materialize")

    monkeypatch.setattr("flask.jsonify", _boom)

    results = {}

    def _worker():
        try:
            results["art"] = app._materialize_artifact(outcome_factory(), _ctx())
        except BaseException as exc:  # noqa: BLE001 — capture for the assertion
            results["exc"] = exc

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=5)

    assert "exc" not in results, f"materialize raised off-thread: {results.get('exc')!r}"
    assert results["art"].terminal_chunks[-1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 5. stage_abort uses _unwrap_stage_abort (not _response_from_stage_abort/jsonify).
# ---------------------------------------------------------------------------


def test_materialize_stage_abort_uses_unwrap_not_jsonify(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("stage_abort must use _unwrap_stage_abort, not jsonify")

    monkeypatch.setattr("flask.jsonify", _boom)

    results = {}

    def _worker():
        try:
            results["art"] = app._materialize_artifact(_stage_abort_outcome(), _ctx())
        except BaseException as exc:  # noqa: BLE001
            results["exc"] = exc

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=5)

    assert "exc" not in results, f"stage_abort materialize failed off-thread: {results.get('exc')!r}"
    art = results["art"]
    assert art.logical_status == "stage_abort"
    assert art.terminal_chunks.count("data: [DONE]\n\n") == 1


# ---------------------------------------------------------------------------
# 6. Worker sets the completion event even if materialize raises.
# ---------------------------------------------------------------------------


def test_worker_sets_event_even_if_materialize_raises(coordinator_state, monkeypatch):
    def _raising(outcome, ctx):
        raise RuntimeError("render boom")

    monkeypatch.setattr(app, "_materialize_artifact", _raising)

    cc._semaphore = threading.Semaphore(4)
    cc._pending = 0
    cc._shutdown = False

    class _P:
        def run(self, ctx):
            return _success_result(with_identity=True)

    ctx = _ctx()
    record = cc.admit(_P(), ctx, materialize=app._materialize_artifact).record

    # Event STILL set (delivery never hangs), but artifact stayed None.
    assert record.completion_event.wait(timeout=5) is True
    assert record.artifact is None


# ---------------------------------------------------------------------------
# 7. CP4-3: has_tool_calls detector in _materialize_artifact.
# ---------------------------------------------------------------------------


def _outcome_with_choices(choices):
    """Build a success outcome with arbitrary choices list."""
    return {
        "result": {
            "object": "chat.completion",
            "id": "chatcmpl-tc",
            "created": 1234567890,
            "model": "m",
            "choices": choices,
        },
        "stage_abort": None,
        "backend_error": None,
        "error": None,
    }


class TestHasToolCallsDetector:
    """CP4-3: _materialize_artifact sets has_tool_calls=True when any choice
    contains tool_calls, function_call, or a tool_calls/function_call
    finish_reason. This gates the replay safety check (is_safe_for_replay).
    """

    def test_tool_calls_in_message(self):
        """has_tool_calls=True when choices[0].message.tool_calls is populated."""
        outcome = _outcome_with_choices([
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ])
        art = app._materialize_artifact(outcome, _ctx())
        assert art.has_tool_calls is True

    def test_function_call_in_message(self):
        """has_tool_calls=True when choices[0].message.function_call is populated."""
        outcome = _outcome_with_choices([
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "get_weather", "arguments": "{}"},
                },
                "finish_reason": "function_call",
            }
        ])
        art = app._materialize_artifact(outcome, _ctx())
        assert art.has_tool_calls is True

    def test_finish_reason_tool_calls_without_message_tool_calls(self):
        """has_tool_calls=True when finish_reason=='tool_calls' even without
        tool_calls key in message (pins the finish_reason branch)."""
        outcome = _outcome_with_choices([
            {
                "index": 0,
                "message": {"role": "assistant", "content": "thinking..."},
                "finish_reason": "tool_calls",
            }
        ])
        art = app._materialize_artifact(outcome, _ctx())
        assert art.has_tool_calls is True

    def test_second_choice_has_tool_calls(self):
        """has_tool_calls=True when choices[1] has tool_calls but choices[0]
        is clean (pins the all-choices loop, not just choices[0])."""
        outcome = _outcome_with_choices([
            {
                "index": 0,
                "message": {"role": "assistant", "content": "clean"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ])
        art = app._materialize_artifact(outcome, _ctx())
        assert art.has_tool_calls is True

    def test_stop_content_no_tool_calls(self):
        """has_tool_calls=False for a normal stop/content response."""
        outcome = _success_outcome(with_identity=True)
        art = app._materialize_artifact(outcome, _ctx())
        assert art.has_tool_calls is False
