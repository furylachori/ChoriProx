"""End-to-end integration test for V2 chain-hash cache stability.

Per docs/v2-chain-hash-redesign.md §11:
- 3 consecutive requests with PROXY_DUMP_REQUESTS=1
- Verify: dump 1, 2, 3 have identical summaries block sha256
- Verify: dump 1, 2, 3 differ only in last_user message
- Verify: upstream no_cache < 500 tokens (target: < 200)
- Verify: ancient block byte-stable across warm turns 1→2
- Verify: ancient block refreshes on turn 3 after chain_state invalidation
  + DB write (F4 contract)

Exercises the full Flask request flow: parse_request → CleanupStage →
DispatchStage → RouteResolutionStage → SecurityStage → ContextOptimizeStage
→ ProviderCompatStage → UpstreamStage → PostProcessStage, capturing dumps
written by http_client._dump_request_payload (PROXY_DUMP_REQUESTS=1) to
validate that V2 produces a cache-stable prefix across consecutive turns.

Approach (Option B — full Flask + mock HTTP backend via proxy_with_mock):
- Configure V2 with a small safe_context_tokens (2000) so that a moderate
  conversation engages V2 (passthrough threshold 0.95 × safe = 1900 tokens).
- Seed SQLite with per-exchange summaries + ancient_history for the
  content-derived session_id so the cold-path on turn 1 populates in-memory
  state.
- Send 3 sequential POSTs, each appending 2 messages (assistant reply + new
  user question). The conversation grows monotonically; raw_block stays
  under HIGH_WATERMARK so turns 2 and 3 stay on the warm path.
- Between turns 2 and 3, write a NEW ancient to DB and invalidate chain
  state; turn 3 must serve the NEW ancient (F4 fix verification).
- Read dumps, extract the summaries/ancient blocks (rendered as user
  messages after MiniMaxAdapter's system→user conversion), and verify the
  cache-stability properties.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


CHARS_PER_MSG = 200
PAD = "X" * CHARS_PER_MSG

SAFE_TOKENS = 2_000
LOW_WATERMARK = int(SAFE_TOKENS * 0.65)
HIGH_WATERMARK = int(SAFE_TOKENS * 0.95)
TOKENS_PER_CHAR = 0.25


def _make_turn_messages(turn_number, base_exchanges=20, max_turns=3):
    """Build a growing user/assistant conversation that ends with [assistant, user].

    Models the real client flow: each turn adds an assistant reply and a new
    user question, so the conversation always ends in [assistant, user]. After
    V2's cold-path trim (raw_block = conversation[boundary_idx:]), the LAST
    user is the freshest question (last_user) and raw_block[:-1] ends with
    the assistant reply — never a user message — so it is never adjacent to
    the post-MiniMax-converted-to-user summaries block.
    """
    messages = []
    for i in range(base_exchanges):
        messages.append({"role": "user", "content": f"q{i} {PAD}"})
        messages.append({"role": "assistant", "content": f"a{i} {PAD}"})
    messages.append({"role": "assistant", "content": f"a{base_exchanges} {PAD}"})
    messages.append({"role": "user", "content": f"final question turn 1 {PAD[:20]}"})

    for t in range(2, turn_number + 1):
        messages.append({"role": "assistant", "content": f"a_t{t}_reply {PAD}"})
        messages.append({"role": "user", "content": f"final question turn {t} {PAD[:20]}"})

    return messages


def _extract_marker_block(dump_messages, marker):
    """Return the message whose content contains ``marker`` (after MiniMax)."""
    for m in dump_messages:
        c = m.get("content")
        if isinstance(c, str) and marker in c:
            return m
    return None


def _extract_summaries_block(dump_messages):
    """The message containing [PAST CONTEXT. MiniMaxAdapter merges this with
    adjacent user messages (e.g. the last raw_block user message, or the
    ancient block), so the matched message may also contain [TIMELINE — ]
    and trailing raw-block content. The cache-stability contract cares about
    whether THIS combined block is byte-identical across turns."""
    return _extract_marker_block(dump_messages, "[PAST CONTEXT")


def _extract_ancient_block(dump_messages):
    """The message containing [TIMELINE. After MiniMaxAdapter merge with the
    summaries block, this is usually the same message as the summaries block.
    Returned separately only when it lives in its own message."""
    summaries = _extract_summaries_block(dump_messages)
    if summaries is not None and "[TIMELINE" in summaries.get("content", ""):
        return summaries
    return _extract_marker_block(dump_messages, "[TIMELINE")


def _extract_barrier(dump_messages):
    """The barrier is an assistant message whose content is exactly a single
    space (BARRIER_CONTENT). Cannot reuse _extract_marker_block here because
    substring-matching a single space would match every message containing
    whitespace, so search directly with an exact-match check."""
    for m in dump_messages:
        if m.get("role") == "assistant" and m.get("content") == " ":
            return m
    return None


def _hash_block(block):
    if block is None:
        return None
    content = block.get("content", "")
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _last_user_content(dump_messages):
    """The final user-role message in the dump (last_user after V2+MiniMax)."""
    for m in reversed(dump_messages):
        if m.get("role") == "user":
            return m.get("content")
    return None


def _no_cache_tokens(last_user_content, tokens_per_char=TOKENS_PER_CHAR):
    if not isinstance(last_user_content, str):
        return int(len(str(last_user_content)) * tokens_per_char)
    return int(len(last_user_content) * tokens_per_char)


def _assert_dump_block_isolation(dumps, summaries_blocks, ancient_blocks,
                                  barriers, last_users, per_dump_metrics):
    """Verify the V2 cache-stability assertions from §11 of the design doc.

    Contract: summaries block is byte-stable across consecutive warm turns
    (turns 1/2/3 share the same summaries content). Ancient is byte-stable
    across the warm turns 1→2 (no cache invalidation), but MUST refresh on
    turn 3 after the test invalidates chain_state (F4 fix).

    Returns ``(cache_stable, no_cache_ok, target_ok, blockers)`` tuple so
    callers can produce the §11 report table.
    """
    blockers = []
    for i, block in enumerate(summaries_blocks):
        if block is None:
            blockers.append(f"Turn {i+1}: summaries block missing")

    h_summaries = [_hash_block(b) for b in summaries_blocks]
    h_ancient = [_hash_block(b) for b in ancient_blocks]

    cache_stable = (
        h_summaries[0] is not None
        and h_summaries[0] == h_summaries[1] == h_summaries[2]
    )
    if not cache_stable:
        blockers.append(
            f"Summaries NOT stable across turns 1/2/3: {h_summaries}"
        )

    for i, b in enumerate(barriers):
        if b is None:
            blockers.append(f"Turn {i+1}: barrier message missing")

    differ_ok = (
        last_users[0] != last_users[1]
        and last_users[1] != last_users[2]
        and last_users[0] != last_users[2]
    )
    if not differ_ok:
        blockers.append(
            f"last_user must differ across all 3 turns: "
            f"{[str(u)[:40] if u else None for u in last_users]}"
        )

    no_cache_tokens = [_no_cache_tokens(lu) for lu in last_users]
    no_cache_ok = all(t < 500 for t in no_cache_tokens)
    if not no_cache_ok:
        blockers.append(
            f"no_cache last_user >= 500 tokens: {no_cache_tokens}"
        )

    target_ok = (no_cache_tokens[1] < 200) or (no_cache_tokens[2] < 200)

    for i in range(3):
        per_dump_metrics.append({
            "turn": i + 1,
            "message_count": dumps[i].get("message_count"),
            "body_bytes": dumps[i].get("body_bytes"),
            "messages_by_role": dumps[i].get("messages_by_role"),
            "first_msg_sha256": (dumps[i].get("first_msg_sha256") or "")[:16],
            "last_msg_sha256": (dumps[i].get("last_msg_sha256") or "")[:16],
            "body_sha256": (dumps[i].get("body_sha256") or "")[:16],
            "summaries_sha256": h_summaries[i],
            "ancient_sha256": h_ancient[i],
            "last_user_first80": (last_users[i] or "")[:80],
            "last_user_tokens_est": no_cache_tokens[i],
        })

    return cache_stable, no_cache_ok, target_ok, blockers


class TestV2CacheStability:
    """3-turn growing conversation: summaries/ancient stable, last_user differs."""

    @pytest.fixture
    def dump_dir(self, tmp_path):
        d = tmp_path / "dumps"
        d.mkdir()
        return d

    def test_v2_cache_stability_across_3_consecutive_turns(
        self, proxy_with_mock, monkeypatch, dump_dir
    ):
        """3 sequential POSTs through Flask → V2 → MiniMax → mock upstream.

        Asserts the §11 cache-stability contract: dumps 1, 2, 3 have
        identical summaries block sha256, differ only in last_user, and
        no_cache (last_user) stays under 500 tokens (target < 200).

        Ancient contract (F4 fix): byte-stable across warm turns 1→2
        (no invalidation), but MUST refresh on turn 3 after the test
        writes new ancient to DB and calls invalidate_chain_state_for_session.
        """
        client, handler, port = proxy_with_mock
        handler._response_body = json.dumps({
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "minimax/MiniMax-M2.7",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()

        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_REQUESTS_DIR", str(dump_dir))

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

        seeded_session_id = resolve_session_id(
            {"messages": _make_turn_messages(1)}, "127.0.0.1"
        )
        with immediate_transaction(pooled=True) as conn:
            conn.execute(
                "DELETE FROM message_summaries WHERE session_id = ?",
                (seeded_session_id,),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (seeded_session_id,),
            )

        for i in range(3):
            store_message_summary(
                session_id=seeded_session_id,
                exchange_index=i,
                role="user",
                summary=f"summary of old turn {i}",
                tool_summary="",
                content_hash=hashlib.sha256(f"seed-{i}".encode()).hexdigest()[:16],
                message_index=i * 2,
                end_message_index=i * 2 + 1,
            )
        update_ancient_history(
            seeded_session_id, "ancient timeline text"
        )

        from unittest.mock import patch
        import proxy.state as state
        state._chain_state.clear()

        NEW_ANCIENT_MARKER = "ANCIENT_REFRESHED_AFTER_INVALIDATION_ZZZ_99999"

        with patch("proxy.context.get_context_config", return_value=small_cfg), \
             patch("proxy.vfs._build_vfs_context", return_value=None), \
             patch("proxy.vfs._build_vfs_inherited_context", return_value=None):
            for turn in (1, 2):
                msgs = _make_turn_messages(turn)
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "mini", "messages": msgs, "stream": False},
                )
                assert resp.status_code == 200, (
                    f"Turn {turn} returned {resp.status_code}: "
                    f"{resp.get_data(as_text=True)[:200]}"
                )

            update_ancient_history(seeded_session_id, NEW_ANCIENT_MARKER)
            from proxy.state import invalidate_chain_state_for_session
            invalidate_chain_state_for_session(seeded_session_id)

            msgs = _make_turn_messages(3)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "mini", "messages": msgs, "stream": False},
            )
            assert resp.status_code == 200, (
                f"Turn 3 returned {resp.status_code}: "
                f"{resp.get_data(as_text=True)[:200]}"
            )

        dump_files = sorted(dump_dir.glob("*.json"))
        assert len(dump_files) == 3, (
            f"Expected 3 dumps, found {len(dump_files)}: "
            f"{[f.name for f in dump_files]}"
        )

        def _read_timestamp(p):
            with open(p) as fp:
                return json.load(fp)["timestamp"]

        ordered_files = sorted(dump_files, key=_read_timestamp)

        dumps = []
        for f in ordered_files:
            with open(f) as fp:
                dumps.append(json.load(fp))

        summaries_blocks = [
            _extract_summaries_block(d["request_body"]["messages"]) for d in dumps
        ]
        ancient_blocks = [
            _extract_ancient_block(d["request_body"]["messages"]) for d in dumps
        ]
        barriers = [_extract_barrier(d["request_body"]["messages"]) for d in dumps]
        last_users = [_last_user_content(d["request_body"]["messages"]) for d in dumps]
        h_ancient = [_hash_block(b) for b in ancient_blocks]

        per_dump_metrics = []
        cache_stable, no_cache_ok, target_ok, blockers = _assert_dump_block_isolation(
            dumps, summaries_blocks, ancient_blocks, barriers, last_users,
            per_dump_metrics,
        )

        for m in per_dump_metrics:
            print(json.dumps(m))

        if dumps[0].get("first_msg_sha256") == dumps[1].get("first_msg_sha256"):
            print("[note] first_msg_sha256 stable between turns 1 and 2 (system prompt)")
        if dumps[1].get("first_msg_sha256") == dumps[2].get("first_msg_sha256"):
            print("[note] first_msg_sha256 stable between turns 2 and 3 (system prompt)")

        for table in ("message_summaries", "conversation_summaries"):
            from memory.db import connection
            with connection(pooled=True) as conn:
                conn.execute(
                    f"DELETE FROM {table} WHERE session_id = ?",
                    (seeded_session_id,),
                )

        for f in dump_dir.glob("*.json"):
            f.unlink()

        assert cache_stable, (
            "V2 cache stability FAILED — summaries block changed between "
            "turns 1/2/3. Blockers: " + "; ".join(blockers)
        )
        assert no_cache_ok, (
            f"V2 no_cache > 500 tokens. Per-turn: "
            f"{[m['last_user_tokens_est'] for m in per_dump_metrics]}. "
            f"Blockers: {'; '.join(blockers)}"
        )
        assert target_ok, (
            f"Target no_cache < 200 not met. Per-turn: "
            f"{[m['last_user_tokens_est'] for m in per_dump_metrics]}"
        )
        assert (
            dumps[1]["last_msg_sha256"] != dumps[2]["last_msg_sha256"]
        ), "Dumps 2 and 3 must differ at the last message (last_user changes per turn)"

        assert h_ancient[0] == h_ancient[1], (
            f"Ancient must be byte-stable across warm turns 1→2 (no invalidation). "
            f"h_ancient={h_ancient}"
        )
        assert h_ancient[1] != h_ancient[2], (
            f"Ancient MUST refresh on turn 3 after invalidate_chain_state_for_session "
            f"+ DB write. F4 fix contract violated. h_ancient={h_ancient}"
        )
        ancient_turn3_content = ancient_blocks[2].get("content", "") if ancient_blocks[2] else ""
        assert NEW_ANCIENT_MARKER in ancient_turn3_content, (
            f"Turn 3 must surface NEW ancient written between turns 2 and 3. "
            f"Got ancient content (first 500 chars): {ancient_turn3_content[:500]!r}"
        )
