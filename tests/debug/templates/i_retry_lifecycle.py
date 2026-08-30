#!/usr/bin/env python3
"""Template I — full Flask retry lifecycle through debug_session.

Pins retry/Recovery-After behavior across the FULL production call-stack:

    DebugSession.chat
    -> Flask POST /v1/chat/completions
    -> RequestPipeline
    -> UpstreamStage.execute
    -> proxy.http_client.call_upstream(..., wants_stream=False)
    -> dynamic localhost JSON mock backend
    -> 429 Retry-After handling
    -> second upstream POST
    -> final Flask response

Two subcases validate independent Retry-After branches end-to-end:

  * Subcase A — NEGATIVE Retry-After (-5). Production parser raises on negative
    and falls back to exponential backoff. The harness's deep-copy of the
    canned dict (post-fix) guarantees the caller-owned fixture is preserved
    across the 429 + recovery round-trip.

  * Subcase B — ZERO Retry-After (0). The retry path triggers with delay=0;
    `time.sleep(0)` is a no-op so the second attempt fires immediately. Both
    attempts must preserve structurally equal outbound messages and the
    caller-owned canned dict (with its `_retry_after=0` control key) is
    unmutated.

This is JSON transport — it does NOT activate upstream SSE.
Production callers continue to pass wants_stream=False.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/i_retry_lifecycle.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session


def main():
    from proxy import http_client

    # Save the entire upstream config; restore in `finally`.
    original_upstream_config = copy.deepcopy(http_client.CONFIG.get("upstream", {}))

    # Tighten retry windows to sub-second so the test exits within the 30s
    # subprocess timeout. Production retry policy is 2s/30s, owned by unit
    # tests; we only need to verify the harness wires call_upstream +
    # Retry-After fallback correctly. The exact sleep durations are NOT
    # asserted here.
    try:
        http_client.CONFIG.setdefault("upstream", {}).update(
            {
                "retry_max_attempts": 3,
                "retry_base_delay": 0.01,
                "retry_max_delay": 0.02,
            }
        )

        messages = [
            {
                "role": "user",
                "content": "RETRY_INPUT_MARKER immutable outbound content",
            }
        ]

        # ------------------------------------------------------------------
        # Subcase A — negative Retry-After
        # ------------------------------------------------------------------
        negative_error = {
            "_status": 429,
            "_body": {"error": "rate limited"},
            "_retry_after": -5,
        }
        negative_success = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "NEGATIVE_RETRY_SECOND_RESPONSE_MARKER",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        negative_error_snapshot = copy.deepcopy(negative_error)
        negative_success_snapshot = copy.deepcopy(negative_success)

        with debug_session(
            canned_responses=[negative_error, negative_success],
            models=["kimi"],
        ) as c:
            response = c.chat(
                messages=copy.deepcopy(messages),
                model="kimi",
                client_hint="retry-negative",
            )

            captured = c.captured_requests
            final_status = response.status_code
            final_content = response.content or ""
            captured_count = len(captured)

        # Per-step assertions. The first 4 are basic; the last 5 pin the
        # retry lifecycle specifically.
        _emit(
            1,
            "A: final status 200",
            final_status == 200,
            f"got {final_status}",
        )
        _emit(
            2,
            "A: success marker in body",
            "NEGATIVE_RETRY_SECOND_RESPONSE_MARKER" in final_content,
            f"content prefix={final_content[:120]!r}",
        )
        _emit(
            3,
            "A: no ValueError surfaced",
            "ValueError" not in final_content,
            f"body prefix={final_content[:120]!r}",
        )
        _emit(
            4,
            "A: not 5xx",
            final_status != 500,
            f"got {final_status}",
        )
        _emit(
            5,
            "A: exactly two upstream requests captured",
            captured_count == 2,
            f"got {captured_count}",
        )
        # ``prepare_upstream_request`` rewrites ``model`` to the
        # ``backend_model`` registered in the model registry (the wire
        # form the upstream actually receives). For "kimi" this is
        # ``opencode-go/kimi-k2.6`` per `_make_proxy_models_config`.
        # The invariant is that BOTH attempts pin the SAME backend model
        # string (NOT that they round-trip the public model name).
        backend_model_0 = captured[0]["body"].get("model")
        backend_model_1 = captured[1]["body"].get("model")
        _emit(
            6,
            "A: both captured bodies pin same backend model",
            backend_model_0 == backend_model_1,
            f"models={(backend_model_0, backend_model_1)}",
        )
        try:
            msgs_0 = captured[0]["body"]["messages"]
            msgs_1 = captured[1]["body"]["messages"]
        except (KeyError, IndexError) as exc:
            msgs_0 = msgs_1 = None
            _emit(
                7,
                "A: both captured bodies have messages",
                False,
                f"extract failed: {exc}",
            )
        else:
            _emit(
                7,
                "A: both captured bodies have messages",
                True,
                f"counts={ (len(msgs_0), len(msgs_1)) }",
            )
        # Outbound messages structurally equal (chat-completions canonical form):
        # same number of messages, every role and content text identical.
        if isinstance(msgs_0, list) and isinstance(msgs_1, list) and len(msgs_0) == len(msgs_1):
            structurally_equal = all(
                m0.get("role") == m1.get("role")
                and (m0.get("content") or "") == (m1.get("content") or "")
                for m0, m1 in zip(msgs_0, msgs_1)
            )
            _emit(
                8,
                "A: outbound messages structurally equal across attempts",
                structurally_equal,
                f"equal={structurally_equal}",
            )
        else:
            _emit(
                8,
                "A: outbound messages structurally equal across attempts",
                False,
                f"len mismatch or non-list: {type(msgs_0).__name__}, {type(msgs_1).__name__}",
            )
        all_text = " ".join(
            (m.get("content") or "") if isinstance(m, dict) else ""
            for body in captured
            for m in body["body"]["messages"]
        )
        _emit(
            9,
            "A: RETRY_INPUT_MARKER appears in both attempts",
            "RETRY_INPUT_MARKER immutable outbound content" in all_text,
            f"all_text prefix={all_text[:120]!r}",
        )
        _emit(
            10,
            "A: caller-owned negative_error unchanged",
            negative_error == negative_error_snapshot,
            f"eq={negative_error == negative_error_snapshot}",
        )
        _emit(
            11,
            "A: caller-owned negative_success unchanged",
            negative_success == negative_success_snapshot,
            f"eq={negative_success == negative_success_snapshot}",
        )

        # ------------------------------------------------------------------
        # Subcase B — zero Retry-After
        # ------------------------------------------------------------------
        zero_error = {
            "_status": 429,
            "_body": {"error": "rate limited"},
            "_retry_after": 0,
        }
        zero_success = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ZERO_RETRY_SECOND_RESPONSE_MARKER",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        zero_error_snapshot = copy.deepcopy(zero_error)
        zero_success_snapshot = copy.deepcopy(zero_success)

        with debug_session(
            canned_responses=[zero_error, zero_success],
            models=["kimi"],
        ) as c:
            response = c.chat(
                messages=copy.deepcopy(messages),
                model="kimi",
                client_hint="retry-zero",
            )
            captured = c.captured_requests
            final_status = response.status_code
            final_content = response.content or ""
            captured_count = len(captured)

        _emit(
            12,
            "B: final status 200",
            final_status == 200,
            f"got {final_status}",
        )
        _emit(
            13,
            "B: success marker in body",
            "ZERO_RETRY_SECOND_RESPONSE_MARKER" in final_content,
            f"content prefix={final_content[:120]!r}",
        )
        _emit(
            14,
            "B: exactly two upstream requests captured",
            captured_count == 2,
            f"got {captured_count}",
        )
        if captured_count == 2:
            try:
                msgs_0 = captured[0]["body"]["messages"]
                msgs_1 = captured[1]["body"]["messages"]
            except (KeyError, IndexError):
                msgs_0 = msgs_1 = None
            structurally_equal = (
                isinstance(msgs_0, list)
                and isinstance(msgs_1, list)
                and len(msgs_0) == len(msgs_1)
                and all(
                    m0.get("role") == m1.get("role")
                    and (m0.get("content") or "") == (m1.get("content") or "")
                    for m0, m1 in zip(msgs_0, msgs_1)
                )
            )
            _emit(
                15,
                "B: outbound messages structurally equal across attempts",
                structurally_equal,
                f"equal={structurally_equal}",
            )
        else:
            _emit(
                15,
                "B: outbound messages structurally equal across attempts",
                False,
                "captured_count != 2",
            )
        _emit(
            16,
            "B: caller-owned zero_error unchanged",
            zero_error == zero_error_snapshot,
            f"eq={zero_error == zero_error_snapshot}",
        )
        _emit(
            17,
            "B: caller-owned zero_success unchanged",
            zero_success == zero_success_snapshot,
            f"eq={zero_success == zero_success_snapshot}",
        )

    finally:
        # Restore the production upstream config so we don't leak tightened
        # retry windows into other scenarios that may run in the same
        # process. debug_session spawns each template in its own
        # subprocess, so this is belt-and-suspenders.
        http_client.CONFIG["upstream"] = original_upstream_config


if __name__ == "__main__":
    try:
        main()
    finally:
        _finish()
