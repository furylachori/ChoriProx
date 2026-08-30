#!/usr/bin/env python3
"""Template K — summary-budget truncation via captured upstream body.

Pins the Context V2 summary-budget decisions end-to-end through debug_session.

The scenario seeds a session with 20 long summaries whose RENDERED row cost
(per-zone: `[msg#{msg_idx}] {role}: {summary}`) far exceeds the production
budget. Crucially, the persisted ``rendered_*_tokens`` columns are set to 1
to defeat a known regression where the budget path used the persisted
estimates instead of recomputing the actual rendered cost.

The setup guarantees:
  * the 15% summary budget cap is hit, AND
  * stale persisted fields do NOT cause the budget to under-count, AND
  * the kept summaries form a NEWEST-CONTIGUOUS SUFFIX (the budget
    drops oldest-first).

Asserted AFTER the full Flask chat:
  * response.status_code == 200
  * exactly one captured upstream request
  * the captured request body contains the current request marker
  * exactly one ``[PAST CONTEXT`` system message is present
  * the summary block contains at least one ``SUMMARY_*_MARKER_`` entry
  * SUMMARY_00_MARKER is absent (oldest summary dropped)
  * SUMMARY_19_MARKER is present (newest summary retained)
  * the retained markers are ascending
  * the retained markers form a CONTIGUOUS SUFFIX of the [0..19] range
  * the sum of the kept rows' actual rendered cost <= summary_budget
  * the pre-truncation actual rendered cost WAS > summary_budget
  * the complete captured prompt envelope (all messages) satisfies
    ``int(sum(len(content) for content in captured_messages) * tokens_per_char)
       <= safe_tokens``
  * no pool checkout remains

Production call-stack:
    real SQLite store_messages_batch / store_message_summary
    -> DebugSession.chat
    -> Flask /v1/chat/completions
    -> ContextOptimizeStage
    -> apply_mode / select_context
    -> _select_context_v2
    -> get_rich_summaries
    -> _apply_summary_budget
    -> _summary_token_cost
    -> ProviderCompatStage
    -> UpstreamStage
    -> captured dynamic-localhost request body

The debug_session default no-ops ``memory.sync.submit_sync`` and
``memory.sync.submit_ancient_compress``; this suppresses the SOTA side effect
that would otherwise fire from ``_apply_summary_budget`` after it trims the
budget. We assert the budget DECISION (what was kept) was correct, but DO NOT
require the ancient-compress dispatch to fire.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/k_summary_budget.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session


def _expected_render_cost(message_index, role, summary, tool_summary, tokens_per_char):
    """Recompute the kept-row cost per the canonical production shape.

    Mirrors ``proxy.context_v2._summary_token_cost`` in zone 1 only (our
    message_indexes 0..38 are all < ZONE1_FULL_MIN=50):

        text = f"[msg#{msg_idx}] {role}: {summary} {tool_summary}".strip()
        cost = int(len(text) * tokens_per_char)

    We do NOT call ``_summary_token_cost``; the template derives expected
    costs INDEPENDENTLY so a production regression cannot tautologically
    pass. Caller asserts ``actual_kept_cost <= summary_budget`` after the
    chat.
    """
    text = f"[msg#{message_index}] {role}: {summary} {tool_summary}".strip()
    return int(len(text) * tokens_per_char)


def _parse_retained_marker_numbers(summary_block):
    """Extract the ``NN`` from each ``SUMMARY_NN_MARKER_`` occurrence."""
    out: list[int] = []
    needle = "_MARKER_"
    pos = 0
    while True:
        idx = summary_block.find("SUMMARY_", pos)
        if idx < 0:
            break
        # Walk past the digits
        start = idx + len("SUMMARY_")
        end = start
        while end < len(summary_block) and summary_block[end].isdigit():
            end += 1
        if (
            end > start
            and end + len(needle) <= len(summary_block)
            and summary_block[end : end + len(needle)] == needle
        ):
            try:
                out.append(int(summary_block[start:end]))
            except ValueError:
                pass
        pos = end if end > pos else pos + 1
    return out


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "summary-budget.db")

        # Enter debug_session FIRST (not after seeding) so the production
        # ``_select_context_v2`` path that the request will hit uses the
        # same model_cfg resolution and the same model_registry.
        with debug_session(
            db_path=db_path,
            models=["qwen3-coder"],
            canned_responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "SUMMARY_BUDGET_REPLY_OK",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ],
        ) as c:
            from memory.db import store_messages_batch
            from memory.db_summaries import store_message_summary
            import memory.db_pool as db_pool
            from proxy.context import get_context_config

            # Configuration: qwen3-coder -> context_limit=32000, ratio=0.75
            # => safe_context_tokens=24000, summary_budget=24000*0.15=3600.
            cfg = get_context_config("qwen3-coder")
            tokens_per_char = cfg["tokens_per_char"]
            safe_tokens = cfg["safe_context_tokens"]
            summary_budget = int(safe_tokens * 0.15)

            # Define exact incoming request data; derive session_id from
            # the SAME messages + client_hint that c.chat() will send so
            # the proxy's content-derived session_id matches the seeded
            # session.
            client_hint = "summary-budget-client"
            incoming_messages = [
                {
                    "role": "user",
                    "content": "SUMMARY_BUDGET_CURRENT_REQUEST_MARKER",
                }
            ]
            session_id = c.session_id_for(incoming_messages, client_hint)

            # Seed 30 exchanges (60 messages alternating user/assistant,
            # ~500 chars each). Use unique markers per message so
            # Context V2 sees real stored state.
            seed_messages = []
            for i in range(30):
                user_content = (
                    f"SEED_USER_MSG_{i:02d}_MARKER_"
                    + ("abcdef" * 80)  # 480 chars
                    + f" [u{i:02d}]"
                )
                asst_content = (
                    f"SEED_ASST_MSG_{i:02d}_MARKER_"
                    + ("ghijkl" * 80)  # 480 chars
                    + f" [a{i:02d}]"
                )
                seed_messages.append({"role": "user", "content": user_content})
                seed_messages.append(
                    {"role": "assistant", "content": asst_content}
                )
            stored = store_messages_batch(session_id, seed_messages, start_index=0)
            _emit(
                1,
                "setup: messages stored",
                stored == len(seed_messages),
                f"stored={stored}, expected={len(seed_messages)}",
            )

            # Store exactly 20 summaries at exchange indexes 0..19. Each
            # summary is 1000+ chars so that the rendered row cost blows
            # the budget for ALL 20.
            for i in range(20):
                ok = store_message_summary(
                    session_id=session_id,
                    exchange_index=i,
                    role="user",
                    summary=f"SUMMARY_{i:02d}_MARKER_" + ("x" * 1000),
                    tool_summary="",
                    content_hash=f"summary-budget-hash-{i}",
                    model_used="qwen3-coder",
                    token_estimate=1,
                    end_message_index=i * 2 + 1,
                    message_index=i * 2,
                )
                if not ok:
                    break
            _emit(
                2,
                "setup: 20 summaries stored",
                ok is True,
                f"final store_message_summary returned {ok!r}",
            )

            # Mislead the persisted rendered_*_tokens columns to 1 — this
            # is the regression that R4 (revert-test) reintroduces. Setup
            # only — production uses `_summary_token_cost`, not these
            # columns.
            #
            # ``db_pool.connection()`` uses deferred transactions, not
            # autocommit. The UPDATE is invisible to other connections
            # until the transaction commits. Use
            # ``immediate_transaction`` so the COMMIT is explicit and
            # visible to the verification query that follows.
            from memory.db_pool import immediate_transaction

            with immediate_transaction(pooled=True) as conn:
                conn.execute(
                    "UPDATE message_summaries "
                    "SET rendered_full_tokens = 1, "
                    "    rendered_action_tokens = 1, "
                    "    rendered_narrative_tokens = 1 "
                    "WHERE session_id = ?",
                    (session_id,),
                )
            with db_pool.connection() as conn:
                row_count = conn.execute(
                    "SELECT COUNT(*) FROM message_summaries "
                    "WHERE session_id = ? AND rendered_full_tokens = 1",
                    (session_id,),
                ).fetchone()[0]
            _emit(
                3,
                "setup: all 20 persisted rendered fields forced to 1",
                row_count == 20,
                f"row_count={row_count}",
            )

            # Reset checkout — pool must be clean before the chat.
            assert db_pool._pool._checked_out == {}, "checkout leaked before chat"

            # Verify pre-truncation actual cost > budget. Use the canonical
            # text-rendering formula for EACH persisted summary.
            with db_pool.connection() as conn:
                rows = conn.execute(
                    "SELECT message_index, role, summary, tool_summary "
                    "FROM message_summaries WHERE session_id = ? "
                    "ORDER BY exchange_index",
                    (session_id,),
                ).fetchall()
            pre_costs = [
                _expected_render_cost(
                    row["message_index"],
                    row["role"],
                    row["summary"],
                    row["tool_summary"] or "",
                    tokens_per_char,
                )
                for row in rows
            ]
            pre_total = sum(pre_costs)
            _emit(
                4,
                "setup: pre-truncation actual cost > summary_budget",
                pre_total > summary_budget,
                f"pre_total={pre_total}, budget={summary_budget}, "
                f"costs={pre_costs}",
            )
            _emit(
                5,
                "setup: safe_tokens matches expected 24000",
                safe_tokens == 24000,
                f"safe_tokens={safe_tokens}",
            )
            _emit(
                6,
                "setup: summary_budget > 0",
                summary_budget > 0,
                f"summary_budget={summary_budget}",
            )

            # Send the chat — captures the FULL upstream request body.
            response = c.chat(
                messages=incoming_messages,
                model="qwen3-coder",
                client_hint=client_hint,
            )
            _emit(
                7,
                "chat: status 200",
                response.status_code == 200,
                f"got {response.status_code}",
            )
            _emit(
                8,
                "chat: success marker in body",
                "SUMMARY_BUDGET_REPLY_OK" in response.content,
                f"content prefix={response.content[:120]!r}",
            )
            captured = c.captured_requests
            _emit(
                9,
                "chat: exactly one captured upstream request",
                len(captured) == 1,
                f"got {len(captured)}",
            )
            if not captured:
                _finish()
                return

            outbound_body = captured[-1]["body"]
            outbound_messages = outbound_body.get("messages", []) or []

            # The current request marker must be in the captured outbound.
            all_text_lower = "\n".join(
                str(m.get("content") or "") for m in outbound_messages
            )
            _emit(
                10,
                "outbound: current request marker preserved",
                "SUMMARY_BUDGET_CURRENT_REQUEST_MARKER" in all_text_lower,
                "marker must round-trip into the captured upstream body",
            )

            # Exactly one system message with [PAST CONTEXT prefix.
            summary_blocks = [
                m
                for m in outbound_messages
                if m.get("role") == "system"
                and isinstance(m.get("content"), str)
                and "[PAST CONTEXT" in (m.get("content") or "")
            ]
            _emit(
                11,
                "summary: exactly one [PAST CONTEXT system message",
                len(summary_blocks) == 1,
                f"count={len(summary_blocks)}",
            )
            if not summary_blocks:
                _finish()
                return
            summary_block = summary_blocks[0]["content"]

            _emit(
                12,
                "summary: at least one SUMMARY_*_MARKER_* entry retained",
                "SUMMARY_" in summary_block and "_MARKER_" in summary_block,
                f"len(summary_block)={len(summary_block)}",
            )

            retained = _parse_retained_marker_numbers(summary_block)
            _emit(
                13,
                "summary: SUMMARY_00_MARKER_ is absent (oldest dropped)",
                not any(n == 0 for n in retained),
                f"retained={retained}",
            )
            _emit(
                14,
                "summary: SUMMARY_19_MARKER_ is present (newest kept)",
                any(n == 19 for n in retained),
                f"retained={retained}",
            )
            _emit(
                15,
                "summary: retained marker numbers are ascending",
                retained == sorted(retained),
                f"retained={retained}",
            )
            if retained:
                # Contiguous suffix of [0..19]: {N..19} for some N>=1.
                n_min, n_max = retained[0], retained[-1]
                expected_len = 19 - n_min + 1
                _emit(
                    16,
                    "summary: retained markers form contiguous suffix of [0..19]",
                    len(retained) == expected_len,
                    f"retained={retained} expected_len={expected_len}",
                )

                # Independent cost computation for the kept indexes.
                kept_costs = [
                    pre_costs[n]
                    for n in retained
                    if 0 <= n < len(pre_costs)
                ]
                kept_actual = sum(kept_costs)
                _emit(
                    17,
                    "summary: actual kept-row cost <= summary_budget",
                    kept_actual <= summary_budget,
                    f"kept_actual={kept_actual}, budget={summary_budget}, "
                    f"retained={retained}",
                )

            # Pre-truncation assertion: the FULL pre-trim cost was
            # strictly greater than the budget (confirming the budget
            # did have to drop something).
            _emit(
                18,
                "summary: pre-truncation actual cost > summary_budget",
                pre_total > summary_budget,
                f"pre_total={pre_total}, budget={summary_budget}",
            )

            # Captured envelope independent check: ALL messages in the
            # captured body, including the summary block, must fit
            # within safe_tokens. Provider-compat wrapping overhead may
            # add some chars, so the assertion is documented as a
            # soft diagnostic — but a *massive* overflow is a bug.
            total_chars = sum(
                len(m.get("content") or "")
                for m in outbound_messages
                if m.get("role") != "tool"
            )
            envelope_tokens = int(total_chars * tokens_per_char)
            _emit(
                19,
                "envelope: complete captured messages <= safe_tokens",
                envelope_tokens <= safe_tokens,
                f"envelope_tokens={envelope_tokens}, safe_tokens={safe_tokens}",
            )

            # Diagnostic: wrapper-inclusive block length. NOT asserted as
            # <= summary_budget; the budget caps the kept rendered rows
            # only, while the wrapper is part of _format_summaries (not
            # _apply_summary_budget).
            block_diag = int(len(summary_block) * tokens_per_char)
            _emit(
                21,
                "diagnostic: summary_block_tokens (informational only)",
                True,
                f"summary_block_tokens_diagnostic={block_diag} "
                f"summary_budget={summary_budget}",
            )

            # No checkout leak.
            _emit(
                20,
                "checkout: no pool checkout after chat",
                db_pool._pool._checked_out == {},
                f"checked_out={list(db_pool._pool._checked_out)}",
            )

    _finish()


if __name__ == "__main__":
    main()
