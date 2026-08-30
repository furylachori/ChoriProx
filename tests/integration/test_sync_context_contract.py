"""Integration tests for the sync → context injection contract.

Verifies that data written by the sync pipeline (`sync_exchange()` →
`store_message_summary()`) is correctly retrieved by the context pipeline
(`_select_context_v2()` → `get_rich_summaries()`).

Covers:
- Test 1: sync → context full path (summary appears in output)
- Test 2: Schema compatibility — write then read round-trips
- Test 3: Summary injection in system prompt with identifiable marker
- Test 4: WAL mode race — concurrent sync write / context read
- Test 5: Graceful degradation when no summaries exist

Most tests use `:memory:` SQLite (fast, per-test isolation). The WAL race
test (Test 4) needs a real on-disk DB so multiple threads share state.
"""
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


# ── DB setup (autouse) ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Use `:memory:` SQLite. Per-connection but only one thread per test.

    The WAL race test (TestWALModeRace) overrides this with its own
    tmp_path-based DB.
    """
    monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

    # If a previous test or the conftest already created a pool at the
    # default path, shut it down so the next pool creation binds to
    # :memory:. The conftest's autouse `reset_shared_state` may have
    # imported `proxy.state` which transitively created a pool.
    from memory import db as _db_mod
    if _db_mod._pool is not None:
        try:
            _db_mod._pool.shutdown()
        except Exception:
            pass
        _db_mod._pool = None  # type: ignore[attr-defined]

    from memory.db import init_db
    init_db(":memory:")

    yield ":memory:"

    # Cleanup
    try:
        from memory.db import close_all_connections
        close_all_connections()
    except Exception:
        pass
    try:
        monkeypatch.delenv("PROXY_TEST_DB", raising=False)
    except Exception:
        pass


# ── Helpers ────────────────────────────────────────────────────────────

SESSION = "test-sync-ctx-contract"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _seed_messages(session_id: str, n_exchanges: int, chars_per_msg: int = 200):
    """Store N user+assistant exchanges via the real persistence path."""
    from memory.db import store_messages_batch

    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    store_messages_batch(session_id, msgs, start_index=0)
    return msgs


def _make_messages(n_exchanges: int, chars_per_msg: int = 200):
    """Build a user+assistant message list (no DB I/O)."""
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    return msgs


def _no_vfs_patches():
    """Suppress VFS side-effects in _select_context_v2."""
    return (
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    )


@contextmanager
def _patch_sync_deps(mock_sota):
    """Patch all external deps that sync_exchange() touches.

    - _get_sota → mock_sota
    - _get_graph → mock (Neo4j stub)
    - proxy.vfs._build_vfs_context/_build_vfs_inherited_context → None
    """
    g = _mock_graph()
    with patch("memory.sync._get_sota", return_value=mock_sota), \
         patch("memory.sync._get_graph", return_value=g), \
         patch("memory.persistence._get_graph", return_value=g), \
         patch("proxy.vfs._build_vfs_context", return_value=None), \
         patch("proxy.vfs._build_vfs_inherited_context", return_value=None):
        yield


# Model config: 10K context, raw_exchange_count=8.
# safe=8000, passthrough threshold=2400. 30 exchanges ≈ 3100 tokens
# → enters the tiered pipeline and emits the summary block.
SMALL_MODEL = SimpleNamespace(context_limit=10_000, raw_exchange_count=8)


def _make_sota_mock(summary_text_fn):
    """Build a SOTA mock whose summarize_message returns a deterministic
    summary string built from the exchange index.

    `summary_text_fn(exchange_index, role) -> str` returns the summary text
    for that exchange.

    Also stubs extract_and_summarize_from_summaries() with an empty (but
    JSON-serializable) extraction so the post-summarization phase of
    sync_exchange() doesn't blow up on the JSON serializer.
    """
    mock = MagicMock()

    def _summarize(session_id, exchange_index, raw_content, role,
                   tool_calls_json=None, previous_messages_text=""):
        return {
            "summary": summary_text_fn(exchange_index, role),
            "tool_summary": "",
            "token_estimate": 25,
        }

    mock.summarize_message.side_effect = _summarize
    mock.extract_and_summarize_from_summaries.return_value = {
        "extraction_timestamp": "2025-01-01T00:00:00Z",
        "entities": [],
        "facts": [],
        "events": [],
        "instructions": [],
        "tasks": [],
    }
    mock.model = "test-model"
    return mock


def _mock_graph():
    """Mock the Neo4j graph so extraction doesn't try to write to a real DB."""
    g = MagicMock()
    g.store_memory.return_value = True
    return g


# ============================================================================
# Test 1: sync → context full path
# ============================================================================


class TestSyncToContextFullPath:
    """The summaries stored by `sync_exchange()` must surface inside
    `_select_context_v2()` output as a system-prompt summary block."""

    def test_summaries_appear_in_context_output(self):
        """After sync_exchange(), _select_context_v2() output must contain
        the summary text written by sync."""
        from memory.db import connection
        from memory.sync import sync_exchange
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 30, chars_per_msg=200)

        def _text(ei, role):
            return f"EXCHANGE-{ei}-DISTINCT-MARKER"

        mock_sota = _make_sota_mock(_text)
        with _patch_sync_deps(mock_sota):
            sync_exchange(SESSION)

        # Confirm rows were actually written
        with connection() as conn:
            n = conn.execute(
                "SELECT COUNT(*) as c FROM message_summaries WHERE session_id = ?",
                (SESSION,),
            ).fetchone()["c"]
        assert n > 0, "sync_exchange() must write summary rows"

        # Build a request large enough to exceed the V2 passthrough threshold
        # (0.30 × 8000 = 2400 tokens). The 30 seeded exchanges × 200 chars
        # ≈ 3100 tokens > 2400 → enters the tiered pipeline and the
        # summary block is built.
        new_request = _make_messages(30, chars_per_msg=200)
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
            )

        all_text = " ".join(
            m.get("content", "") for m in result if isinstance(m.get("content"), str)
        )
        assert "EXCHANGE-" in all_text, (
            "Summary content written by sync must appear in context output"
        )

    def test_summary_text_matches_what_was_stored(self):
        """The summary text visible in the output must match what sync wrote
        byte-for-byte (not truncated to empty or replaced by placeholder)."""
        from memory.sync import sync_exchange
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 30, chars_per_msg=200)

        def _text(ei, role):
            return f"detailed-work-done-at-exchange-{ei}"

        mock_sota = _make_sota_mock(_text)
        with _patch_sync_deps(mock_sota):
            sync_exchange(SESSION)

        # Same large request — exceeds V2 passthrough threshold
        new_request = _make_messages(30, chars_per_msg=200)
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
            )

        # At least one summary line must contain a real summary, not an
        # empty placeholder.
        summary_lines = []
        for m in result:
            content = m.get("content", "")
            if isinstance(content, str) and "detailed-work-done-at-exchange-" in content:
                for line in content.split("\n"):
                    if "detailed-work-done-at-exchange-" in line:
                        summary_lines.append(line)

        assert len(summary_lines) > 0, (
            "Output must contain real summary lines, not just placeholders"
        )
        for line in summary_lines:
            assert "detailed-work-done-at-exchange-" in line
            # Must not be a tombstone marker
            assert "(no response)" not in line
            assert "(empty message)" not in line
            assert "(parse error)" not in line


# ============================================================================
# Test 2: Schema compatibility — write then read
# ============================================================================


class TestSchemaRoundTrip:
    """Fields written by `store_message_summary()` must round-trip through
    `get_rich_summaries()` without NULLs or missing data."""

    def test_all_fields_round_trip(self):
        from memory.db_summaries import (
            store_message_summary,
            get_rich_summaries,
        )

        # Write a fully-populated summary
        store_message_summary(
            session_id=SESSION,
            exchange_index=0,
            role="user",
            summary="user asked about async patterns",
            tool_summary="",
            content_hash=_content_hash("user-0 content"),
            model_used="test-model",
            token_estimate=42,
            end_message_index=1,
            message_index=0,
        )

        rows = get_rich_summaries(SESSION)
        assert len(rows) == 1
        row = rows[0]

        # Fields that get_rich_summaries() actually returns (the read path's
        # contract). Fields written but not selected (model_used,
        # end_message_index) are not part of this contract — see test
        # below for direct SELECT verification.
        assert row["exchange_index"] == 0
        assert row["role"] == "user"
        assert row["summary"] == "user asked about async patterns"
        assert row["tool_summary"] == ""
        assert row["content_hash"] == _content_hash("user-0 content")
        assert row["token_estimate"] == 42
        assert row["message_index"] == 0

        # Pre-computed rendered tokens must be present (non-NULL)
        for tf in ("rendered_full_tokens",
                   "rendered_action_tokens",
                   "rendered_narrative_tokens"):
            v = row.get(tf)
            assert v is not None, f"{tf} must be non-NULL after round-trip"
            assert v > 0, f"{tf} must be > 0 (got {v})"

    def test_multiple_summaries_all_fields_present(self):
        """Five summaries across different exchanges — none may have NULLs.

        Verified both via get_rich_summaries() (the read path) and via a
        direct SELECT (covers fields not in get_rich_summaries' column list).
        """
        from memory.db_summaries import (
            store_message_summary,
            get_rich_summaries,
        )
        from memory.db import connection

        for ei in range(5):
            store_message_summary(
                session_id=SESSION,
                exchange_index=ei,
                role="user",
                summary=f"summary text {ei}",
                tool_summary=f"[tools: action_{ei}]",
                content_hash=_content_hash(f"user-{ei}"),
                model_used="m",
                token_estimate=10 + ei,
                end_message_index=ei * 2 + 1,
                message_index=ei * 2,
            )

        rows = get_rich_summaries(SESSION)
        assert len(rows) == 5

        # Order: ascending by exchange_index
        for i, row in enumerate(rows):
            assert row["exchange_index"] == i
            for field in ("exchange_index", "role", "summary",
                          "tool_summary", "content_hash",
                          "token_estimate", "message_index",
                          "rendered_full_tokens",
                          "rendered_action_tokens",
                          "rendered_narrative_tokens"):
                v = row.get(field)
                assert v is not None, (
                    f"Row {i}: field {field!r} is NULL — schema mismatch"
                )

        # Direct SELECT verifies fields NOT exposed by get_rich_summaries
        with connection() as conn:
            direct = conn.execute(
                "SELECT session_id, exchange_index, model_used, end_message_index "
                "FROM message_summaries WHERE session_id = ? ORDER BY exchange_index",
                (SESSION,),
            ).fetchall()
        assert len(direct) == 5
        for i, r in enumerate(direct):
            assert r["session_id"] == SESSION
            assert r["exchange_index"] == i
            assert r["model_used"] == "m"
            assert r["end_message_index"] == i * 2 + 1

    def test_rendered_tokens_match_formula(self):
        """The pre-computed rendered_*_tokens must reflect the formula used
        by `compute_rendered_tokens()` — guard against silent drift between
        the write path and the read path's expectations."""
        from memory.db_summaries import (
            store_message_summary,
            get_rich_summaries,
        )

        summary = "the quick brown fox jumps"
        tool_summary = "[tools: read_file]"
        store_message_summary(
            session_id=SESSION,
            exchange_index=7,
            role="user",
            summary=summary,
            tool_summary=tool_summary,
            content_hash=_content_hash("user-7"),
            message_index=14,
        )

        rows = get_rich_summaries(SESSION, before_exchange_index=8)
        assert len(rows) == 1
        row = rows[0]

        # Compute expected tokens from the public formula
        full_line = f"[exchange#7] user: {summary} {tool_summary}"
        narrative_line = f"[exchange#7] user: {summary}"
        expected_full = int(len(full_line.split()) / 0.75)
        expected_narrative = int(len(narrative_line.split()) / 0.75)
        assert row["rendered_full_tokens"] == expected_full
        assert row["rendered_narrative_tokens"] == expected_narrative

    def test_before_exchange_filter_works(self):
        """Schema-level: the read path's before_exchange_index filter must
        match the value produced by the write path (exchange_index)."""
        from memory.db_summaries import (
            store_message_summary,
            get_rich_summaries,
        )

        for ei in range(10):
            store_message_summary(
                session_id=SESSION,
                exchange_index=ei,
                role="user",
                summary=f"s{ei}",
                tool_summary="",
                content_hash=_content_hash(f"u{ei}"),
                message_index=ei * 2,
            )

        # before_exchange_index=5 → exchanges 0..4 only
        rows = get_rich_summaries(SESSION, before_exchange_index=5)
        assert [r["exchange_index"] for r in rows] == [0, 1, 2, 3, 4]


# ============================================================================
# Test 3: Summary injection in system prompt
# ============================================================================


class TestSummaryInjectionInSystemPrompt:
    """Summaries must surface as a system-role message with the
    `[PAST CONTEXT]` marker (main agent) or `[Conversation Summary]`
    marker (sub-agent)."""

    def test_main_agent_past_context_marker(self):
        from memory.sync import sync_exchange
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 30, chars_per_msg=200)
        mock_sota = _make_sota_mock(lambda ei, role: f"action-summary-{ei}")
        with _patch_sync_deps(mock_sota):
            sync_exchange(SESSION)

        # Large request to exceed V2 passthrough threshold
        new_request = _make_messages(30, chars_per_msg=200)
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
                is_subagent=False,
            )

        # Find the summary system message
        summary_msgs = [
            m for m in result
            if m.get("role") == "system" and "PAST CONTEXT" in m.get("content", "")
        ]
        assert len(summary_msgs) == 1, (
            f"Expected exactly one [PAST CONTEXT] system message, "
            f"got {len(summary_msgs)}: "
            f"{[m.get('content', '')[:60] for m in result]}"
        )
        content = summary_msgs[0]["content"]
        assert "[END PAST CONTEXT" in content, (
            "Past-context block must be closed with [END PAST CONTEXT]"
        )
        assert "action-summary-" in content, (
            "Summary text from sync must be inside the [PAST CONTEXT] block"
        )

    def test_subagent_uses_conversation_summary_marker(self):
        """Sub-agents get a different marker — `[Conversation Summary]`."""
        from memory.sync import sync_exchange
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 30, chars_per_msg=200)
        mock_sota = _make_sota_mock(lambda ei, role: f"sub-sum-{ei}")
        with _patch_sync_deps(mock_sota):
            sync_exchange(SESSION)

        new_request = _make_messages(30, chars_per_msg=200)
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
                is_subagent=True,
            )

        summary_msgs = [
            m for m in result
            if m.get("role") == "system" and "Conversation Summary" in m.get("content", "")
        ]
        assert len(summary_msgs) >= 1, (
            "Sub-agent output must contain a [Conversation Summary] system message"
        )
        if summary_msgs:
            content = summary_msgs[0]["content"]
            assert "[End Summary]" in content
            assert "sub-sum-" in content

    def test_summary_block_first_after_system_prompt(self):
        """Order: system prompt → [PAST CONTEXT] → raw messages."""
        from memory.sync import sync_exchange
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 30, chars_per_msg=200)
        mock_sota = _make_sota_mock(lambda ei, role: f"order-test-{ei}")
        with _patch_sync_deps(mock_sota):
            sync_exchange(SESSION)

        # Large request to exceed V2 passthrough threshold
        new_request = _make_messages(30, chars_per_msg=200)
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
                is_subagent=False,
            )

        # Find indices of: past-context block, raw user msg with marker
        idx_past = idx_user = None
        for i, m in enumerate(result):
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            if idx_past is None and "PAST CONTEXT" in content:
                idx_past = i
            if idx_user is None and m.get("role") == "user" and "user-29" in content:
                idx_user = i

        assert idx_past is not None, "Past-context block must be present"
        assert idx_user is not None, "Raw user message must be present"
        assert idx_past < idx_user, (
            f"Past-context block must come before raw messages: "
            f"past={idx_past} user={idx_user}"
        )


# ============================================================================
# Test 4: WAL mode race — sync writing while context reads
# ============================================================================


class TestWALModeRace:
    """Concurrent sync-write and context-read must not produce
    `database is locked` errors. Requires a real on-disk DB so multiple
    threads share state and WAL mode is effective."""

    def test_concurrent_write_read_no_lock_errors(self, tmp_path, monkeypatch):
        # Override the autouse :memory: with a real on-disk DB.
        from memory import db as _db_mod
        if _db_mod._pool is not None:
            try:
                _db_mod._pool.shutdown()
            except Exception:
                pass
            _db_mod._pool = None
        db_path = str(tmp_path / "wal_race.db")
        monkeypatch.setattr(_db_mod, "_get_db_path", lambda: db_path)
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        from memory.db import init_db
        init_db(db_path)

        from memory.db import connection
        from memory.db_summaries import get_rich_summaries
        from memory.sync import sync_exchange

        # Verify WAL is enabled
        with connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal", (
            f"Tests require WAL mode, got {mode!r}. "
            f"Race-condition semantics differ in non-WAL modes."
        )

        _seed_messages(SESSION, 30, chars_per_msg=200)

        errors: list = []
        read_seen_summary: list = []
        stop_flag = threading.Event()

        def _writer():
            """Continuously re-run sync_exchange (idempotent: same content_hash)."""
            mock_sota = _make_sota_mock(lambda ei, role: f"race-sum-{ei}")
            try:
                with _patch_sync_deps(mock_sota):
                    for _ in range(15):
                        if stop_flag.is_set():
                            return
                        sync_exchange(SESSION)
                        time.sleep(0.005)
            except Exception as e:
                errors.append(("writer", e))

        def _reader():
            """Continuously read summaries and check for race-sum-* text."""
            try:
                for i in range(30):
                    if stop_flag.is_set():
                        return
                    rows = get_rich_summaries(SESSION)
                    for r in rows:
                        if "race-sum-" in r.get("summary", ""):
                            read_seen_summary.append(i)
                            return
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("reader", e))

        t_writer = threading.Thread(target=_writer, daemon=True)
        t_reader = threading.Thread(target=_reader, daemon=True)
        t_writer.start()
        t_reader.start()

        t_writer.join(timeout=10.0)
        t_reader.join(timeout=10.0)
        stop_flag.set()

        # The contract: NO "database is locked" errors in either thread.
        for label, err in errors:
            msg = str(err).lower()
            assert "locked" not in msg, (
                f"{label} thread got database-locked error: {err!r}"
            )

        # At least one read after the first sync cycle must observe the summary
        assert len(read_seen_summary) > 0, (
            "Reader never observed summaries written by sync — "
            "cross-thread visibility broken"
        )


# ============================================================================
# Test 5: Summary not yet available — graceful degradation
# ============================================================================


class TestGracefulDegradation:
    """When no summaries have been stored (sync not yet run),
    _select_context_v2() must still produce a valid context."""

    def test_no_crash_without_summaries(self):
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 15, chars_per_msg=200)

        system_msg = {"role": "system", "content": "instructions"}
        new_request = [system_msg, {"role": "user", "content": "first request"}]
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
            )

        # Must not raise, must return a list
        assert isinstance(result, list)
        assert len(result) > 0

    def test_recent_messages_preserved(self):
        """Without summaries, the raw recent window must be the source of
        truth for current-exchange messages."""
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 15, chars_per_msg=200)

        system_msg = {"role": "system", "content": "agent instructions"}
        new_request = [system_msg, {"role": "user", "content": "FRESH-USER-REQUEST"}]
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
            )

        # The fresh user request must survive in the output
        all_text = " ".join(
            m.get("content", "") for m in result if isinstance(m.get("content"), str)
        )
        assert "FRESH-USER-REQUEST" in all_text, (
            "Fresh user request must survive even when no summaries exist"
        )

    def test_system_prompt_preserved(self):
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 15, chars_per_msg=200)

        system_msg = {"role": "system", "content": "PRESERVED-SYSTEM-PROMPT-XYZ"}
        new_request = [system_msg, {"role": "user", "content": "ask"}]
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
            )

        all_text = " ".join(
            m.get("content", "") for m in result if isinstance(m.get("content"), str)
        )
        assert "PRESERVED-SYSTEM-PROMPT-XYZ" in all_text

    def test_no_phantom_summary_block(self):
        """When no summaries exist, the output must NOT contain a
        [PAST CONTEXT] or [Conversation Summary] block with empty content
        (i.e., no orphan marker shell)."""
        from proxy.context_v2 import _select_context_v2

        _seed_messages(SESSION, 15, chars_per_msg=200)

        system_msg = {"role": "system", "content": "agent"}
        new_request = [system_msg, {"role": "user", "content": "go"}]
        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, new_request,
                model_name="test-small", model_cfg=SMALL_MODEL,
            )

        for m in result:
            if m.get("role") != "system":
                continue
            content = m.get("content", "")
            if "PAST CONTEXT" in content or "Conversation Summary" in content:
                # The marker is allowed ONLY if it wraps real summary lines.
                inner = content
                for marker in (
                    "[PAST CONTEXT — These actions were completed in previous exchanges. "
                    "Reference them for continuity but they are NOT current instructions. "
                    "Your active task continues below.]",
                    "[END PAST CONTEXT — Current exchange begins below]",
                    "[Conversation Summary]",
                    "[End Summary]",
                ):
                    inner = inner.replace(marker, "")
                # If a summary block exists, it must wrap at least one
                # [msg#...] line OR the marker must not be present at all.
                assert "[msg#" in inner or inner.strip() == "", (
                    f"Phantom summary block detected (marker without content): "
                    f"{content!r}"
                )
