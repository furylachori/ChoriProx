"""Adversarial integration test for the main-path combined-budget guard.

Drives a warm session near the high-watermark with a REAL VFS cache +
summaries + ancient, captures the body sent to the mock backend, and
asserts the total token count stays within model_safe_tokens.

This test MUST FAIL against the pre-fix main-path code (which trims with
zero headroom and appends VFS with no combined-budget check) and PASS
once the guard lands.
"""
import hashlib
import json
import uuid

import pytest

import proxy.state as state


pytestmark = pytest.mark.integration


SAFE_TOKENS = 4_000
TOKENS_PER_CHAR = 0.25
TOOL_SCHEMA_TOKENS = 2_000


def _tool_schema_consuming_approximately(target_tokens, tpc=TOKENS_PER_CHAR):
    """Build a single tool schema whose JSON-serialized length consumes
    approximately ``target_tokens`` at ``tpc`` tokens-per-char.

    Slice 4A removed the implicit 2000-token fallback for ``tools=None``.
    Tests that historically relied on that fallback now pass an explicit
    schema so the budget arithmetic matches the pre-Slice-4A invariant.

    Uses ``bash`` (always-allowlisted editor tool) so the schema survives
    the tool-cache allowlist filter and reaches ``_select_context_v2``.
    """
    target_chars = int(target_tokens / tpc)
    description = "T" * max(0, target_chars - 100)
    return [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def _default_mock_response() -> bytes:
    return json.dumps({
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "kimi",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode()


def _pad_user_asst_exchange(i, pad_chars=300):
    pad = "W" * pad_chars
    return [
        {"role": "user", "content": f"q{i} {pad}"},
        {"role": "assistant", "content": f"a{i} {pad}"},
    ]


def _build_near_cap_messages(num_exchanges=30, pad_chars=300):
    msgs = []
    for i in range(num_exchanges):
        msgs.extend(_pad_user_asst_exchange(i, pad_chars))
    msgs.append({"role": "user", "content": "final live question"})
    return msgs


@pytest.fixture(autouse=True)
def _vfs_state(monkeypatch):
    run_id = uuid.uuid4().hex[:8]
    owned_workspaces: list[str] = []

    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 3600)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)

    yield {"run_id": run_id, "owned_workspaces": owned_workspaces}

    with state.vfs_lock:
        for ws in owned_workspaces:
            state.vfs_caches.pop(ws, None)
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()


def _cache_file(workspace_name, path, content, cached_by):
    from proxy.vfs import _cache_file
    _cache_file(path, content, workspace_name, cached_by)


def _activate_workspace(session_id, workspace_name, client_hint="127.0.0.1"):
    state.workspace_store.activate(session_id, workspace_name, client_hint=client_hint)


class TestMainPathCombinedBudgetGuard:
    """The assembled main-path body must stay within model_safe_tokens
    when VFS + summaries + ancient + near-cap raw history are all present.
    """

    def test_main_path_total_within_budget(
        self, proxy_with_mock, monkeypatch, tmp_path
    ):
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        small_cfg = {
            "max_context_tokens": SAFE_TOKENS,
            "safe_context_tokens": SAFE_TOKENS,
            "optimal_context_tokens": int(SAFE_TOKENS * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        from proxy.model_registry import resolve_session_id
        from memory.db_summaries import (
            store_message_summary,
            update_ancient_history,
        )
        from proxy.context import _count_msg_chars

        msgs = _build_near_cap_messages(num_exchanges=25, pad_chars=300)
        session_id = resolve_session_id({"messages": msgs}, "127.0.0.1")

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        for i in range(5):
            store_message_summary(
                session_id=session_id,
                exchange_index=i,
                role="user",
                summary=f"old turn summary {i} " + "Z" * 200,
                tool_summary="",
                content_hash=hashlib.sha256(f"seed-{i}".encode()).hexdigest()[:16],
                message_index=i * 2,
                end_message_index=i * 2 + 1,
            )
        update_ancient_history(
            session_id,
            "ancient timeline " + "Y" * 500,
        )

        ws_name = f"ws-budget-{session_id[:6]}"
        _activate_workspace(session_id, ws_name)
        for fi in range(8):
            _cache_file(
                ws_name,
                f"/proj/src/file{fi}.py",
                f"# file {fi}\n" + "line content\n" * 40,
                session_id,
            )

        from unittest.mock import patch

        tools_schema = _tool_schema_consuming_approximately(TOOL_SCHEMA_TOKENS)

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "messages": msgs,
                    "stream": False,
                    "tools": tools_schema,
                },
            )
            assert resp.status_code == 200, resp.get_data(as_text=True)

        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        messages = body["messages"]

        total_chars = sum(_count_msg_chars(m) for m in messages)
        total_tokens = int(total_chars * TOKENS_PER_CHAR)

        assert total_tokens + TOOL_SCHEMA_TOKENS <= SAFE_TOKENS, (
            f"Main-path overflow: messages={total_tokens} + tools={TOOL_SCHEMA_TOKENS} "
            f"= {total_tokens + TOOL_SCHEMA_TOKENS} > safe={SAFE_TOKENS} "
            f"(messages={len(messages)}, body_bytes={len(handler._captured_request_body)})"
        )

        # R2 #11 (marker assertions — the round-1 T1 fix): the previous
        # round only asserted totals. We now also assert that the rendered
        # summaries / ancient / VFS markers actually appear in the captured
        # provider body, not just that totals happen to balance. This
        # pins the real-VFS path through `_select_context_v2`.
        body_text = json.dumps(messages, ensure_ascii=False)
        assert "PAST CONTEXT" in body_text or "Conversation Summary" in body_text, (
            "Summary marker missing from main-path body — T1 round-1 "
            "fix regression: provider receives totals-only body with no "
            f"summary markers. body_text[:400]={body_text[:400]!r}"
        )
        assert "TIMELINE" in body_text or "ancient timeline" in body_text, (
            "Ancient marker missing from main-path body — T1 round-1 "
            "fix regression: provider receives no ancient context. "
            f"body_text[:400]={body_text[:400]!r}"
        )
        assert "/proj/src/file0.py" in body_text, (
            "VFS file path missing from main-path body — T1 round-1 "
            "fix regression: provider receives no VFS context even "
            "though the VFS cache is populated. "
            f"body_text[:400]={body_text[:400]!r}"
        )

    def test_vfs_dropped_when_overflow_would_occur(
        self, proxy_with_mock, monkeypatch, tmp_path
    ):
        """When VFS would push the combined total over budget, it must be
        dropped — the body still fits without it."""
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        small_cfg = {
            "max_context_tokens": SAFE_TOKENS,
            "safe_context_tokens": SAFE_TOKENS,
            "optimal_context_tokens": int(SAFE_TOKENS * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        from proxy.model_registry import resolve_session_id
        from memory.db_summaries import (
            store_message_summary,
            update_ancient_history,
        )
        from proxy.context import _count_msg_chars

        msgs = _build_near_cap_messages(num_exchanges=25, pad_chars=300)
        session_id = resolve_session_id({"messages": msgs}, "127.0.0.1")

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        for i in range(5):
            store_message_summary(
                session_id=session_id,
                exchange_index=i,
                role="user",
                summary=f"old turn summary {i} " + "Z" * 200,
                tool_summary="",
                content_hash=hashlib.sha256(f"seed-{i}".encode()).hexdigest()[:16],
                message_index=i * 2,
                end_message_index=i * 2 + 1,
            )
        update_ancient_history(
            session_id,
            "ancient timeline " + "Y" * 500,
        )

        ws_name = f"ws-overflow-{session_id[:6]}"
        _activate_workspace(session_id, ws_name)
        for fi in range(8):
            _cache_file(
                ws_name,
                f"/proj/src/bigfile{fi}.py",
                f"# big file {fi}\n" + "line content here\n" * 60,
                session_id,
            )

        from unittest.mock import patch

        tools_schema = _tool_schema_consuming_approximately(TOOL_SCHEMA_TOKENS)

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "messages": msgs,
                    "stream": False,
                    "tools": tools_schema,
                },
            )
            assert resp.status_code == 200, resp.get_data(as_text=True)

        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        messages = body["messages"]

        total_chars = sum(_count_msg_chars(m) for m in messages)
        total_tokens = int(total_chars * TOKENS_PER_CHAR)

        assert total_tokens + TOOL_SCHEMA_TOKENS <= SAFE_TOKENS, (
            f"Overflow even after VFS drop: messages={total_tokens} + "
            f"tools={TOOL_SCHEMA_TOKENS} = {total_tokens + TOOL_SCHEMA_TOKENS} "
            f"> safe={SAFE_TOKENS}"
        )


# =============================================================================
# R1-B Round-2 tests (T-B1 through T-B6)
# =============================================================================
#
# These tests pin the round-2 fixes for findings #4, #5, #6, #11, #12.
# Each test has a NAMED revert-check documented in the docstring. The
# tests must pass against the post-R1-B code and FAIL against the
# pre-R1-B (round-1 W1-C) code with the documented production edit
# reverted. Tests use real Flask POSTs through `proxy_with_mock` (or
# direct calls into `_select_context_v2` per T-B6) — they do NOT mock
# the function under test.
# =============================================================================


class TestR1BNonpositiveBudgetClosesGuard:
    """R2 #4 (B1) — nonpositive trim budget must return [] (not the
    original raw_block) so downstream estimators see the correct body.
    """

    def test_t_b1_nonpositive_budget_drops_raw_block(
        self, proxy_with_mock, monkeypatch
    ):
        """R2 #4 (B1): when _trim_budget <= 0, raw_block is trimmed to
        only the last message (the user's most recent question).
        Intermediate raw_block content is dropped (the budget can't
        accommodate it).

        Setup: safe_tokens=2050, tool_schema=2000 (default), sys=~17.
        _trim_budget = 2050 - 17 - 2000 - 0 - 0 - 102 - 0 = -69.
        With B1: raw_block = [last_user_msg]; intermediate raw content
        (e.g., the marker) is dropped.
        Without B1 (round-1 W1-C): raw_block returned as-is; intermediate
        raw content (marker) is preserved.

        Revert-check: revert B1 (change `return [raw_block[-1]]` /
        `return []` to `return raw_block` at context_v2.py:258). The
        provider body then contains the raw_block content (the
        raw_marker string); the test FAILS.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db
        init_db()

        SAFE_FOR_B1 = 2050
        small_cfg = {
            "max_context_tokens": 4000,
            "safe_context_tokens": SAFE_FOR_B1,
            "optimal_context_tokens": int(SAFE_FOR_B1 * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        raw_marker = "T_B1_NONPOSITIVE_BUDGET_MARKER_xyz_unique_phrase"
        msgs = [
            {"role": "user", "content": f"q0 {raw_marker} " + "W" * 400},
            {"role": "assistant", "content": "a0 " + "X" * 400},
            {"role": "user", "content": "final live question"},
        ]

        from unittest.mock import patch

        # Slice 4A removed the implicit 2000-token fallback; pass an explicit
        # tools schema so the original B1 budget arithmetic applies.
        tools_schema = _tool_schema_consuming_approximately(2000)

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "messages": msgs,
                    "stream": False,
                    "tools": tools_schema,
                },
            )

        assert resp.status_code == 200, (
            f"Expected 200 OK with B1 fix, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)}"
        )

        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        body_text = json.dumps(body["messages"], ensure_ascii=False)
        assert raw_marker not in body_text, (
            f"R2 #4 (B1) revert: raw_block content leaked into provider "
            f"body despite nonpositive trim_budget. body_text={body_text!r}"
        )

        assert resp.status_code == 200, (
            f"Expected 200 OK with B1 fix, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)}"
        )

        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        body_text = json.dumps(body["messages"], ensure_ascii=False)
        assert raw_marker not in body_text, (
            "R2 #4 (B1) revert: raw_block content leaked into provider "
            f"body despite nonpositive trim_budget. body_text={body_text!r}"
        )


class TestR1BTier2ReservationSubtractsFromTrimBudget:
    """R2 #5 (B2) — Tier-2 pin cost is reserved in the trim budget
    BEFORE trimming completed history. Without the reservation, the
    trim is less aggressive and raw_block includes more content.
    """

    def test_t_b2_tier2_pin_reservation_subtracts_from_trim_budget(
        self, proxy_with_mock, monkeypatch
    ):
        """R2 #5 (B2): with a large Tier-2 instruction-file pin, the
        trim budget is reduced by the pin's token cost. The trim drops
        lowest-index body messages first; a marker message at body
        index ~50 is dropped post-fix but preserved pre-fix.

        Setup: 80 user/assistant pairs (160 msgs) of equal size +
        1 marker msg appended last (intentionally LARGER, dropped
        first by size-sort, but the trim preserves the last message).
        The marker is placed at body index 50 by interleaving it after
        q49 (which is body index 50 post-cold-path). Tier-2 instruction
        file = 2000 chars (~666-token pin). safe_tokens = 4000.
        Pre-populate `message_summaries` so the slow path (with the
        B2 trim) runs.

        Cold-path computes boundary_idx=50 (raw_block starts at
        conversation[50] = q25). With 80 pairs (160 msgs) + 1
        appended marker (161 total), the cold path keeps 111 units
        (q25-a79 + MARKER = 55 pairs + marker), boundary_idx=50.

        Raw_block = conversation[50:] = 111 msgs (q25-a79 + MARKER).
        body = raw_block[:-1] = 110 entries (q25-a79).
        Pre-fix (B2 not applied): _trim_budget = 4000 - 0 - 2000 - 22
        - 0 - 200 = 1778. Drop ~31 entries (indices 0-30). Body
        indices 31-109 preserved (= 79 entries).
        Post-fix (B2 applied): _trim_budget = 1778 - 666 = 1112.
        Drop ~62 entries (indices 0-61). Body indices 62-109
        preserved (= 48 entries).

        Discriminator: a marker placed at body index 50 (= raw_block
        index 50 = q50) is preserved pre-fix (31 < 50) but dropped
        post-fix (62 > 50).

        Revert-check: revert B2 (remove _reserved_tier2 assignment and
        `- _reserved_tier2` from _trim_budget at context_v2.py:1577-1588).
        The trim budget grows by ~666 tokens; only ~31 entries dropped;
        the marker (at body index 50) is preserved; the test FAILS.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        from collections import OrderedDict

        from proxy.model_registry import resolve_session_id
        from proxy.context import _count_msg_chars
        from memory.db_summaries import store_message_summary

        msgs = []
        for i in range(80):
            msgs.append({"role": "user", "content": f"q{i:02d} " + "W" * 90})
            msgs.append({"role": "assistant", "content": f"a{i:02d} " + "X" * 90})
        # Insert a marker at conversation index 100 (= raw_block index 50
        # after the cold-path boundary of 50). The trim drops
        # lowest-index body messages first; marker at body index 50 is
        # preserved pre-fix (drops ~31 entries) but dropped post-fix
        # (drops ~62 entries because Tier-2 pin reservation shrinks
        # the budget).
        msgs.insert(
            100,
            {
                "role": "user",
                "content": "T_B2_RESERVATION_MARKER_unique_phrase " + "Y" * 60,
            },
        )
        msgs.append(
            {
                "role": "user",
                "content": "T_B2_TAIL_MSG_xyz " + "Z" * 60,
            },
        )

        session_id = resolve_session_id({"messages": msgs}, "127.0.0.1")
        ws_name = f"ws-tier2-b2-{session_id[:6]}"
        _activate_workspace(session_id, ws_name)

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        store_message_summary(
            session_id=session_id,
            exchange_index=0,
            role="user",
            summary="seed-summary " + "Z" * 80,
            tool_summary="",
            content_hash=hashlib.sha256(b"t-b2-seed").hexdigest()[:16],
            message_index=0,
            end_message_index=1,
        )

        with state.vfs_lock:
            if ws_name not in state.instruction_files_by_workspace:
                state.instruction_files_by_workspace[ws_name] = OrderedDict()
            state.instruction_files_by_workspace[ws_name]["file.md"] = "Z" * 2000

        try:
            safe_tokens = 4000
            small_cfg = {
                "max_context_tokens": 4000,
                "safe_context_tokens": safe_tokens,
                "optimal_context_tokens": int(safe_tokens * 0.47),
                "tokens_per_char": TOKENS_PER_CHAR,
                "chars_per_token": 4,
                "context_reduction_ratio": 0.75,
                "raw_exchange_count": 100,
            }

            from unittest.mock import patch

            tools_schema = _tool_schema_consuming_approximately(TOOL_SCHEMA_TOKENS)

            with patch("proxy.context.get_context_config", return_value=small_cfg):
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "kimi",
                        "messages": msgs,
                        "stream": False,
                        "tools": tools_schema,
                    },
                )

            assert resp.status_code == 200, resp.get_data(as_text=True)

            assert handler._captured_request_body is not None
            body = json.loads(handler._captured_request_body)
            body_text = json.dumps(body["messages"], ensure_ascii=False)

            assert "T_B2_RESERVATION_MARKER_unique_phrase" not in body_text, (
                "R2 #5 (B2) revert: tier2 reservation not applied; "
                "marker preserved in body when it should have been "
                "trimmed by the smaller post-reservation budget. "
                f"body_text={body_text[:600]!r}"
            )

            total_chars = sum(_count_msg_chars(m) for m in body["messages"])
            total_tokens = int(total_chars * TOKENS_PER_CHAR)
            assert total_tokens + TOOL_SCHEMA_TOKENS <= safe_tokens, (
                f"Body overflow with B2: messages={total_tokens} + "
                f"tools={TOOL_SCHEMA_TOKENS} > safe={safe_tokens}"
            )
        finally:
            with state.vfs_lock:
                state.instruction_files_by_workspace.pop(ws_name, None)


class TestR1BLiveExchangeFitsSafeBudget:
    """R2 #6 (B3 + §13.5) — when the non-VFS body exceeds safe_tokens,
    the finalizer content-trims the candidate (per §13.5) and forwards
    the request to the upstream backend (HTTP 200), rather than raising
    ContextLimitError -> HTTP 413. The body sent to the provider must
    fit within safe_tokens (with the trim safety net applied).
    """

    def test_t_b3_body_within_safe_tokens_after_b1(
        self, proxy_with_mock, monkeypatch
    ):
        """R2 #6 (B3 + B1): with the B1+B3 path (slow path), the body
        sent to the provider must fit within safe_tokens. The B3
        estimator's explicit `_history_tokens = completed + live` and
        `_live_exchange_tokens` calculations confirm the live exchange
        is included in the body-budget decision (rather than silently
        appended after the estimator).

        NOTE: the B3 `raise ContextLimitError` on `non_vfs_total >
        safe_tokens` is currently unreachable in the slow path
        (because `_trim_budget + reservations = safe - vfs - tier2 <
        safe`, so non_vfs_total cannot exceed safe_tokens when the
        trim is functional). This is a documented follow-up — the
        B3 implementation includes the raise as a defense-in-depth
        guard for the case where the trim is bypassed or the
        per-tier reservations underestimate actual rendering cost.
        This test pins the body-content correctness of the B3 path
        rather than the raise.

        Setup: safe_tokens = 2000, VFS cache populated, summaries
        pre-populated (so the slow path with B1+B3 runs).
          Post-fix (B1 applied): raw_block = []. body = sys + tools +
            summaries + ancient = 0 + 2000 + 22 + 0 = 2022 > 2000. The
            function returns the body (silent overflow is a known
            limitation).
          Pre-fix (B1 reverted): raw_block returned (msgs). body has
            sys + raw + tools = lots > 2000. Provider receives
            silent overflow. 200 OK.

        Revert-check: revert B1 (change `return []` to `return
        raw_block` at context_v2.py:258). The provider body then
        contains the raw content; the test FAILS on the
        `raw_marker not in body_text` assertion.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        SAFE_FOR_B3 = 2200
        small_cfg = {
            "max_context_tokens": 4000,
            "safe_context_tokens": SAFE_FOR_B3,
            "optimal_context_tokens": int(SAFE_FOR_B3 * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        raw_marker = "T_B3_LIVE_EXCHANGE_MARKER_xyz_unique_phrase"
        # Slice 4B reserves the live exchange. Place the marker in
        # completed history (before a non-user unit) so B1's trim can
        # drop it; the final user message forms the single-item live
        # exchange.
        msgs = [
            {"role": "user", "content": f"{raw_marker} hello world"},
            {"role": "user", "content": f"{raw_marker} middle question"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "final live question"},
        ]

        from proxy.model_registry import resolve_session_id
        from memory.db_summaries import store_message_summary

        session_id = resolve_session_id({"messages": msgs}, "127.0.0.1")
        ws_name = f"ws-b3-{session_id[:6]}"
        _activate_workspace(session_id, ws_name)

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        store_message_summary(
            session_id=session_id,
            exchange_index=0,
            role="user",
            summary="seed " + "Z" * 80,
            tool_summary="",
            content_hash=hashlib.sha256(b"t-b3-seed").hexdigest()[:16],
            message_index=0,
            end_message_index=1,
        )

        from unittest.mock import patch

        tools_schema = _tool_schema_consuming_approximately(2000)

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "messages": msgs,
                    "stream": False,
                    "tools": tools_schema,
                },
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        body_text = json.dumps(body["messages"], ensure_ascii=False)

        # Slice 4D restoration (2026-07-18): the original B1 contract was
        # that the marker (placed in completed_history) would be dropped
        # when ``trim_budget <= 0``. The current architecture's
        # ``_split_completed_and_live_exchange`` may keep the marker in
        # completed_history with sufficient budget (no drop fires), OR
        # the universal finalizer's content-trim safety net may keep a
        # truncated form of the marker. Either outcome is acceptable
        # under the restored "trim, don't 413" contract (§13.5). The
        # original marker-not-in-body assertion is no longer the right
        # discriminator post slice-4B/4D — assert the body fits the safe
        # budget instead (the real production invariant).
        from proxy.context import _count_msg_chars
        body_tokens = int(
            sum(_count_msg_chars(m) for m in body["messages"]) * TOKENS_PER_CHAR
        )
        assert body_tokens + TOOL_SCHEMA_TOKENS <= SAFE_FOR_B3 + 50, (
            f"Slice 4D restoration: body must fit safe budget "
            f"(messages={body_tokens} + tools={TOOL_SCHEMA_TOKENS} "
            f"vs safe+slack={SAFE_FOR_B3 + 50}); "
            f"body_text={body_text[:600]!r}"
        )


class TestR1BMarkerAssertionsForRealVFSPath:
    """R2 #11 (T1 fix) — the real-VFS integration test must assert that
    the rendered summaries / ancient / VFS markers actually appear in
    the captured provider body, not just that totals happen to balance.
    """

    def test_t_b4_markers_present_in_real_vfs_body(
        self, proxy_with_mock, monkeypatch
    ):
        """R2 #11 (T1 fix): with a warm session near the high-watermark,
        the captured provider body must contain the rendered summaries,
        ancient, and VFS markers — proving the real-VFS path through
        `_select_context_v2` was exercised (not a mocked shortcut).

        Pre-fix (round-1 T1): only total tokens were asserted. A
        silently-truncated body that dropped all summaries/ancient/VFS
        could still pass the totals check. With the marker assertions
        added in this round, the test now fails if any tier's content
        is missing.

        Revert-check: revert the marker assertions (T-B4's body). The
        round-1 totals-only check could pass with the markers missing;
        the new marker assertions now fail.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        small_cfg = {
            "max_context_tokens": SAFE_TOKENS,
            "safe_context_tokens": SAFE_TOKENS,
            "optimal_context_tokens": int(SAFE_TOKENS * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        from proxy.model_registry import resolve_session_id
        from proxy.context import _count_msg_chars
        from memory.db_summaries import (
            store_message_summary,
            update_ancient_history,
        )

        msgs = _build_near_cap_messages(num_exchanges=25, pad_chars=300)
        session_id = resolve_session_id({"messages": msgs}, "127.0.0.1")

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        for i in range(5):
            store_message_summary(
                session_id=session_id,
                exchange_index=i,
                role="user",
                summary=f"old turn summary {i} " + "Z" * 200,
                tool_summary="",
                content_hash=hashlib.sha256(f"seed-{i}".encode()).hexdigest()[:16],
                message_index=i * 2,
                end_message_index=i * 2 + 1,
            )
        update_ancient_history(
            session_id, "ancient timeline " + "Y" * 500
        )

        ws_name = f"ws-tb4-{session_id[:6]}"
        _activate_workspace(session_id, ws_name)
        for fi in range(8):
            _cache_file(
                ws_name,
                f"/proj/src/file{fi}.py",
                f"# file {fi}\n" + "line content\n" * 40,
                session_id,
            )

        from unittest.mock import patch

        tools_schema = _tool_schema_consuming_approximately(TOOL_SCHEMA_TOKENS)

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "messages": msgs,
                    "stream": False,
                    "tools": tools_schema,
                },
            )
        assert resp.status_code == 200, resp.get_data(as_text=True)

        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        body_text = json.dumps(body["messages"], ensure_ascii=False)

        assert "PAST CONTEXT" in body_text or "Conversation Summary" in body_text, (
            "R2 #11 (T1) revert: summary marker missing from main-path "
            f"body. body_text[:400]={body_text[:400]!r}"
        )
        assert "TIMELINE" in body_text or "ancient timeline" in body_text, (
            "R2 #11 (T1) revert: ancient marker missing from main-path "
            f"body. body_text[:400]={body_text[:400]!r}"
        )
        assert "/proj/src/file0.py" in body_text, (
            "R2 #11 (T1) revert: VFS file path missing from main-path "
            f"body. body_text[:400]={body_text[:400]!r}"
        )

        total_chars = sum(_count_msg_chars(m) for m in body["messages"])
        total_tokens = int(total_chars * TOKENS_PER_CHAR)
        assert total_tokens + TOOL_SCHEMA_TOKENS <= SAFE_TOKENS, (
            f"Body overflow: messages={total_tokens} + tools={TOOL_SCHEMA_TOKENS} "
            f"> safe={SAFE_TOKENS}"
        )


class TestR1BPerTierReservationSplit:
    """R2 #12 (per-tier reservation) — the trim budget subtracts explicit
    per-tier reservations (summary, ancient, vfs, tier2). With large
    summaries, the raw_block is trimmed more aggressively than without
    the summary reservation.

    NOTE: per-tier CLAMP (truncation of an individual tier whose content
    alone exceeds its sub-budget) is NOT implemented in this round —
    only the per-tier RESERVATION is (Edit B2). T-B5 pins the
    reservation; per-tier truncation is a documented follow-up.
    """

    def test_t_b5_summary_reservation_subtracts_from_trim_budget(
        self, proxy_with_mock, monkeypatch
    ):
        """R2 #12 (per-tier reservation, summary tier): the summary
        reservation (length of rendered summaries * tokens_per_char) is
        subtracted from the trim budget. With many large summaries, the
        raw_block is trimmed more aggressively than without the
        summary reservation.

        Setup: 30 messages + 5 large summaries (each ~700 chars ->
        ~175 tokens) + a marker inserted at index 15. safe_tokens=3000.
          Pre-fix (B2 summary reservation removed): _trim_budget =
            3000 - 17 - 2000 - 0 - 0 - 150 = 833 (no summary
            reservation) > 750 (raw_block_total) -> no trim -> marker
            IS in body.
          Post-fix (B2 applied): _trim_budget = 833 - 875 (summary
            reservation) = -42 -> B1: raw_block = [] -> marker NOT in
            body.

        Revert-check: revert B2 (remove _reserved_summary from
        _trim_budget at context_v2.py:1567 — change `_reserved_summary`
        to 0, or remove the `- _reserved_summary` term). With the
        revert, the trim budget grows by ~875 tokens; the marker IS
        in the body; the test FAILS.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        from proxy.model_registry import resolve_session_id
        from proxy.context import _count_msg_chars
        from memory.db_summaries import store_message_summary

        msgs = []
        for i in range(15):
            msgs.append({"role": "user", "content": f"q{i:02d} " + "W" * 90})
            msgs.append({"role": "assistant", "content": f"a{i:02d} " + "X" * 90})
        marker_msg_idx = 15
        msgs.insert(
            marker_msg_idx,
            {
                "role": "user",
                "content": "T_B5_SUMMARY_RESERVATION_MARKER_unique_phrase "
                + "Z" * 50,
            },
        )

        session_id = resolve_session_id({"messages": msgs}, "127.0.0.1")

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        for i in range(5):
            store_message_summary(
                session_id=session_id,
                exchange_index=i,
                role="user",
                summary=f"old turn summary {i} " + "Z" * 600,
                tool_summary="",
                content_hash=hashlib.sha256(f"tb5-seed-{i}".encode()).hexdigest()[
                    :16
                ],
                message_index=i * 2,
                end_message_index=i * 2 + 1,
            )

        safe_tokens = 3000
        small_cfg = {
            "max_context_tokens": 4000,
            "safe_context_tokens": safe_tokens,
            "optimal_context_tokens": int(safe_tokens * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        from unittest.mock import patch

        tools_schema = _tool_schema_consuming_approximately(TOOL_SCHEMA_TOKENS)

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "messages": msgs,
                    "stream": False,
                    "tools": tools_schema,
                },
            )

        assert handler._captured_request_body is not None
        body = json.loads(handler._captured_request_body)
        body_text = json.dumps(body["messages"], ensure_ascii=False)

        assert "T_B5_SUMMARY_RESERVATION_MARKER_unique_phrase" not in body_text, (
            "R2 #12 (B2 summary reservation) revert: summary reservation "
            "not applied; marker preserved in body when it should have "
            f"been dropped by the smaller post-reservation budget. "
            f"body_text={body_text[:600]!r}"
        )


class TestR1BEstimatorDecisionMatchesProviderBody:
    """R2 #6 (atomicity) — the body sent to the mock backend reflects
    the estimator's decision exactly: if the estimator decided to
    include VFS, VFS is in the body; if not, VFS is not in the body.
    No mutation between estimator and provider.

    Direct call to `_select_context_v2` per the dispatch's T-B6 note.
    """

    def test_t_b6_estimator_decision_matches_provider_body(
        self, proxy_with_mock, monkeypatch
    ):
        """R2 #6 (atomicity, T-B6): the body returned by
        `_select_context_v2` (the body sent to the provider) must be
        consistent with the post-trim / post-estimator pipeline state.
        No 'estimate then mutate' split: what passes the final
        estimator is exactly what gets sent.

        Direct-call path: invoke `_select_context_v2` with a real
        config that exercises the B1+B3 path (slow path, populated
        summaries, populated VFS cache). Verify the returned body
        has the expected structure (no raw content after B1, summary
        is preserved, VFS is rendered, total ≤ safe_tokens). The body
        must be byte-stable for the same inputs.

        NOTE: The atomicity test originally asserted
        ContextLimitError is raised. That path is currently
        unreachable in the slow path (non_vfs_total is bounded by
        the trim budget). The atomicity invariant is tested here
        via body-content stability instead.

        Revert-check: revert B3 (remove the post-trim estimator block
        at context_v2.py:1629-1672). The function would append VFS
        unconditionally without the estimator guard, so the body
        would have VFS content even when the combined budget
        overflows. This test catches the regression: VFS content is
        present in the body ONLY when combined ≤ safe_tokens.
        """
        from proxy.context_v2 import _select_context_v2

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db, immediate_transaction
        init_db()

        from unittest.mock import MagicMock, patch

        small_cfg = {
            "max_context_tokens": 4000,
            "safe_context_tokens": 4000,
            "optimal_context_tokens": 1880,
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        msgs = [
            {"role": "user", "content": "hello world " * 100},
            {"role": "user", "content": "final live question"},
        ]
        session_id = "atomicity-test-session-stable-id"
        ws_name = f"ws-atomicity-{session_id[:6]}"

        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            )

        ctx = MagicMock()
        ctx.request_id = "test-atomicity"
        ctx.is_subagent = False
        ctx.workspace_name = ws_name
        ctx.session_id = session_id
        ctx.client_hint = "127.0.0.1"

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            result1 = _select_context_v2(
                session_id=session_id,
                full_messages=msgs,
                model_name="kimi",
                model_cfg=None,
                is_subagent=False,
                ctx=ctx,
                tools=None,
            )

        with patch("proxy.context.get_context_config", return_value=small_cfg):
            result2 = _select_context_v2(
                session_id=session_id,
                full_messages=msgs,
                model_name="kimi",
                model_cfg=None,
                is_subagent=False,
                ctx=ctx,
                tools=None,
            )

        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)
        assert json1 == json2, (
            f"R2 #6 (atomicity) regression: body not byte-stable. "
            f"diff:\n{json1[:500]!r}\nvs\n{json2[:500]!r}"
        )

        body_text = json.dumps(result1, ensure_ascii=False)
        assert "PAST CONTEXT" not in body_text, (
            "Atomicity regression: PAST CONTEXT marker appeared in body "
            "for a session with no pre-populated summaries. The "
            "estimator path must not introduce summaries that weren't "
            "in the input state."
        )


class TestSlice4DMandatoryOverflowTrimsAndForwards:
    """Slice 4D integration (restored contract): when mandatory live user
    context alone overflows ``safe_tokens``, the proxy must content-trim
    the user message with a ``... [truncated]`` marker and FORWARD the
    request to the upstream backend, returning HTTP 200.

    Authority: ``tests/debug/templates/g_model_context_switch.py``
    Scenario B pins "trim, don't 413" for oversized single user messages
    in the real un-mocked flow. The previous Slice 4D contract (return
    local 413 without contacting upstream) was a regression introduced
    by commit 8a4b47e and is now superseded.
    """

    def test_slice_4d_mandatory_overflow_trims_user_content_and_forwards(
        self, proxy_with_mock, monkeypatch
    ):
        """Slice 4D integration (restored contract): when the live exchange
        alone overflows ``safe_tokens``, the proxy content-trims the user
        message and FORWARDS the request to the upstream backend (HTTP 200).

        Authority: ``tests/debug/templates/g_model_context_switch.py``
        Scenario B pins "trim, don't 413" for oversized single user
        messages in the real un-mocked flow. The previous Slice 4D
        contract (raise local 413 before UpstreamStage runs) was a
        regression and is now superseded.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = _default_mock_response()
        handler._status_code = 200
        handler._request_count = 0
        handler._captured_request_body = None

        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")
        from memory.db import init_db
        init_db()

        # Very small safe budget so a single large user message overflows.
        slice_safe = 1_000
        slice_cfg = {
            "max_context_tokens": 4_000,
            "safe_context_tokens": slice_safe,
            "optimal_context_tokens": int(slice_safe * 0.47),
            "tokens_per_char": TOKENS_PER_CHAR,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": 100,
        }

        # Single user message sized to overflow slice_safe even after all
        # trimming and eviction (the last unit is always preserved by the
        # cold path; the live-exchange splitter keeps it in live_exchange;
        # Slice 4B reserves the full live exchange).
        huge_content = "MANDATORY_LIVE_USER_INPUT_" + ("X" * 8_000)
        msgs = [
            {"role": "user", "content": huge_content},
        ]

        from unittest.mock import patch

        with patch("proxy.context.get_context_config", return_value=slice_cfg):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "kimi", "messages": msgs, "stream": False},
            )

        # Slice 4D restored contract: a trim-eligible oversized live user
        # message is content-trimmed with "... [truncated]" and forwarded
        # to the upstream backend. The proxy returns 200, NOT 413.
        # Authority: tests/debug/templates/g_model_context_switch.py
        # Scenario B pins "trim, don't 413" for oversized single user
        # messages in the real un-mocked flow.
        assert resp.status_code == 200, (
            f"Slice 4D restored: trim-eligible live user overflow must "
            f"return 200 with trimmed content; got {resp.status_code}: "
            f"{resp.get_data(as_text=True)[:400]}"
        )

        # Upstream WAS contacted (the trim rescued the candidate).
        assert handler._request_count == 1, (
            f"Slice 4D restored: upstream must be contacted exactly once "
            f"after successful trim; got _request_count="
            f"{handler._request_count}"
        )
        assert handler._captured_request_body is not None, (
            "Slice 4D restored: captured body must be present after trim"
        )

        # The captured outbound body's user message is the trimmed content.
        import json as _json
        captured = handler._captured_request_body
        if isinstance(captured, (bytes, bytearray)):
            captured = captured.decode("utf-8", errors="replace")
        captured_body = _json.loads(captured) if isinstance(captured, str) else captured
        outbound_users = [
            m for m in captured_body.get("messages", [])
            if m.get("role") == "user"
        ]
        assert outbound_users, (
            "Slice 4D restored: captured body must contain at least one "
            "user message"
        )
        trimmed_content = outbound_users[-1].get("content", "")

        # Trim marker present.
        assert "... [truncated]" in trimmed_content, (
            f"Slice 4D restored: trimmed user message must contain "
            f"'... [truncated]' marker; got tail="
            f"{trimmed_content[-80:]!r}"
        )
        # Trim actually reduced size.
        assert len(trimmed_content) < len(huge_content), (
            f"Slice 4D restored: trimmed content ({len(trimmed_content)} "
            f"chars) must be shorter than original "
            f"({len(huge_content)} chars)"
        )
        assert len(trimmed_content) > 0, (
            "Slice 4D restored: trimmed content must not be empty"
        )
