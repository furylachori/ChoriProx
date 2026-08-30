"""CP3-2: the Flask delivery generator (`_keepalive_stream`) is a PURE
SUBSCRIBER to a per-request CompletionRecord.

It waits on ``record.completion_event`` (never ``future.result()`` /
``cancel()`` / ``join()`` / ``add_done_callback``) and reads the published
``record.outcome``. These tests fabricate a CompletionRecord directly (no
admit / executor) so the subscriber contract is exercised in isolation.
"""
import ast
import inspect
import threading
import types

import pytest

import proxy.app as app
from proxy.completion_coordinator import CompletionArtifact
from proxy.types import BackendError


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ScriptedEvent:
    """Deterministic Event stand-in: ``wait`` returns False ``false_count``
    times, then True forever. No real sleeping."""

    def __init__(self, false_count):
        self._remaining = false_count

    def wait(self, timeout=None):
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True

    def is_set(self):
        return self._remaining == 0


def _record(*, completion_event, outcome=None, future=None, artifact=None):
    return types.SimpleNamespace(
        completion_event=completion_event,
        outcome=outcome,
        future=future,
        artifact=artifact,
    )


def _ctx():
    return types.SimpleNamespace(
        request_id="r",
        session_id="s",
        upstream=types.SimpleNamespace(original_model="m"),
    )


def _success_outcome():
    return {
        "result": {
            "id": "chatcmpl-x",
            "created": 123,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        },
        "stage_abort": None,
        "backend_error": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# 1. AST proof: delivery never touches the Future result/cancel/join API.
# ---------------------------------------------------------------------------


def test_keepalive_is_pure_subscriber_ast():
    source = inspect.getsource(app._keepalive_stream)
    tree = ast.parse(source)

    attr_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    call_func_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    forbidden_attrs = {"result", "cancel", "join", "add_done_callback"}
    assert forbidden_attrs.isdisjoint(attr_names), (
        f"_keepalive_stream must not touch the Future API; found "
        f"{forbidden_attrs & attr_names}"
    )

    forbidden_calls = {"clear_active_mode", "set_active_mode", "submit_sync"}
    assert forbidden_calls.isdisjoint(attr_names | call_func_names), (
        f"_keepalive_stream must not invoke lifecycle/side-effect calls; found "
        f"{forbidden_calls & (attr_names | call_func_names)}"
    )

    # CP3-3: delivery must NOT render — all terminal SSE is materialized in
    # the worker. Assert the render functions never appear in delivery.
    forbidden_renderers = {
        "_sse_from_response_data",
        "stream_response",
        "_materialize_artifact",
        "_unwrap_stage_abort",
        "_map_exception_to_envelope",
    }
    assert forbidden_renderers.isdisjoint(attr_names | call_func_names), (
        f"_keepalive_stream must not render terminal SSE (CP3-3); found "
        f"{forbidden_renderers & (attr_names | call_func_names)}"
    )

    # Sanity: the permitted event-observation surface is actually present.
    assert "wait" in attr_names
    assert "done" in attr_names
    assert "is_set" in attr_names


# ---------------------------------------------------------------------------
# 2. Heartbeat: N keepalives then exactly one terminal + one [DONE].
# ---------------------------------------------------------------------------


def test_heartbeat_emits_N_keepalives_then_terminal():
    artifact = CompletionArtifact(
        terminal_chunks=(
            'data: {"id": "chatcmpl-x"}\n\n',
            "data: [DONE]\n\n",
        ),
        logical_status="ok",
        original_request_id="r",
    )
    record = _record(
        completion_event=_ScriptedEvent(3),
        outcome=_success_outcome(),
        future=None,
        artifact=artifact,
    )
    chunks = list(app._keepalive_stream(_ctx(), record))

    assert chunks[0] == ": connected\n\n"
    assert chunks.count(": keepalive\n\n") == 3
    # The 3 keepalives must precede the terminal payload.
    done_index = chunks.index("data: [DONE]\n\n")
    assert chunks[1:4] == [": keepalive\n\n"] * 3
    assert done_index > 3
    # The artifact's terminal_chunks appear verbatim after the keepalives.
    assert chunks[4:] == list(artifact.terminal_chunks)
    assert chunks.count("data: [DONE]\n\n") == 1


# ---------------------------------------------------------------------------
# 3. Each of the 4 outcome branches materializes exactly one [DONE].
#    RE-TARGETED to _materialize_artifact (CP3-3): the exactly-one-[DONE]-per-
#    branch invariant now lives in the worker-side renderer. Asserting it
#    against a hand-built artifact in delivery would be tautological.
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    "outcome_factory,expected_status",
    [
        (_stage_abort_outcome, "stage_abort"),
        (_backend_error_outcome, "backend_error"),
        (_error_outcome, "error"),
        (_success_outcome, "ok"),
    ],
)
def test_each_outcome_branch_emits_exactly_one_done(outcome_factory, expected_status):
    artifact = app._materialize_artifact(outcome_factory(), _ctx())

    assert artifact.logical_status == expected_status
    assert artifact.terminal_chunks.count("data: [DONE]\n\n") == 1


# ---------------------------------------------------------------------------
# 4. Cancelled-worker backstop: event never fires, future.done() is True.
# ---------------------------------------------------------------------------


class _NeverFiresEvent:
    """``wait`` always False, ``is_set`` always False — models a worker that
    was cancelled before publishing (the event will never be set)."""

    def wait(self, timeout=None):
        return False

    def is_set(self):
        return False


def test_cancelled_worker_backstop_terminates():
    record = _record(
        completion_event=_NeverFiresEvent(),
        outcome=None,
        future=types.SimpleNamespace(done=lambda: True),
        artifact=None,
    )
    chunks = list(app._keepalive_stream(_ctx(), record))

    assert chunks[0] == ": connected\n\n"
    assert chunks.count("data: [DONE]\n\n") == 1
    # The canonical cancelled-worker error is surfaced.
    assert any("worker_cancelled" in c for c in chunks)


# ---------------------------------------------------------------------------
# 5. Delivery emits the worker-materialized artifact verbatim (anti-append).
# ---------------------------------------------------------------------------


def test_delivery_emits_artifact_verbatim_no_append():
    art = app._materialize_artifact(_success_outcome(), _ctx())
    record = _record(
        completion_event=_ScriptedEvent(0),
        outcome=_success_outcome(),
        future=None,
        artifact=art,
    )
    chunks = list(app._keepalive_stream(_ctx(), record))

    # Non-comment chunks (those not starting with ":") must equal the
    # artifact's terminal_chunks EXACTLY — delivery neither renders nor
    # appends anything.
    delivered = [c for c in chunks if not c.startswith(":")]
    assert delivered == list(art.terminal_chunks)
    assert chunks.count("data: [DONE]\n\n") == 1


# ---------------------------------------------------------------------------
# 6. Delivery emits a canonical error when the artifact is None (render fail).
# ---------------------------------------------------------------------------


def test_delivery_emits_canonical_error_when_artifact_none():
    record = _record(
        completion_event=_ScriptedEvent(0),
        outcome=_success_outcome(),
        future=None,
        artifact=None,
    )
    chunks = list(app._keepalive_stream(_ctx(), record))

    assert chunks.count("data: [DONE]\n\n") == 1
    assert any(
        "materialize_failed" in c or "proxy_error" in c for c in chunks
    )
