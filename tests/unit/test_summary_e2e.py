"""End-to-end tests for the summary pipeline through context_v2.

Tests the full path: store messages → store summaries → _select_context_v2()
verifies summaries appear in output, system prompt is preserved, and budget
allocation works correctly.
"""
import hashlib
import os
import time
from types import SimpleNamespace

import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_test_db():
    os.environ["PROXY_TEST_DB"] = ":memory:"
    from memory.db import init_db, get_connection

    init_db()
    yield
    conn = get_connection()
    conn.execute("DELETE FROM message_summaries")
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM conversation_summaries")
    conn.execute("DELETE FROM extraction_batches")
    conn.commit()
    os.environ.pop("PROXY_TEST_DB", None)


# ── Imports after fixture so env is set before pool init ──────────────
from memory.db import (  # noqa: E402
    get_connection,
    store_messages_batch,
)
from memory.db_summaries import (  # noqa: E402
    store_message_summary,
    get_rich_summaries,
)
from proxy.context_v2 import _select_context_v2  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────

SESSION = "test-summary-e2e"


def _make_messages(n_exchanges, chars_per_msg=200):
    """Build a message list: user+assistant pairs for each exchange."""
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"user-{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"asst-{i} " + "A" * chars_per_msg})
    return msgs


def _content_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _store_and_seed(n_exchanges, chars_per_msg=200):
    """Store messages via store_messages_batch, return the message list."""
    msgs = _make_messages(n_exchanges, chars_per_msg)
    store_messages_batch(SESSION, msgs, start_index=0)
    return msgs


def _seed_summaries(exchange_indices, summary_text="summary"):
    """Store summaries for the given exchange indices."""
    for ei in exchange_indices:
        content = f"user-{ei}"
        store_message_summary(
            session_id=SESSION,
            exchange_index=ei,
            role="user",
            summary=f"{summary_text} for exchange {ei}",
            tool_summary="",
            content_hash=_content_hash(content),
            message_index=ei * 2,
        )


# Model config as SimpleNamespace (get_context_config accesses .context_limit).
# With context_limit=10000 → safe=8000. Under the V2 eviction pipeline:
#   high-watermark = 0.95*8000 = 7600 tokens
#   recent_raw_budget (low-watermark) = int(8000*0.65) = 5200 tokens
# 80 exchanges (160 msgs × ~207 chars × 0.25 tpc) ≈ 8280 tokens > 7600 → enters
# the eviction path and trims to ~5200 tokens, which is what fetches summaries.
SMALL_MODEL = SimpleNamespace(context_limit=10_000, raw_exchange_count=10)
# Exchange count large enough to force eviction past the 7600 threshold.
N_EXCHANGES = 80


def _no_vfs_patches():
    """Patches to suppress VFS side-effects in _select_context_v2."""
    return (
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    )


# ============================================================================
# Test 1: _get_context_boundaries() via SQL on real SQLite (1.10)
# ============================================================================


class TestContextBoundaries:
    """Test the boundary query that _select_context_v2 uses internally.

    The nested _get_context_boundaries() function queries:
        SELECT MAX(exchange_index) FROM messages WHERE session_id = ?
    We test the same logic via direct SQL to verify the data layer.
    """

    def test_max_exchange_index_basic(self):
        """20 messages (10 exchanges, 0-9) → max_exchange = 9."""
        _store_and_seed(10)
        conn = get_connection()
        row = conn.execute(
            "SELECT MAX(exchange_index) as max_ei FROM messages WHERE session_id = ?",
            (SESSION,),
        ).fetchone()
        assert row["max_ei"] == 9

    def test_max_exchange_index_with_null_ei(self):
        """Messages with NULL exchange_index should not affect MAX."""
        _store_and_seed(5)  # exchanges 0-4

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = get_connection()
        conn.execute(
            """INSERT INTO messages (session_id, message_index, role, content,
               content_hash, created_at, exchange_index)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (SESSION, 999, "system", "system msg", _content_hash("system msg"), now, None),
        )
        conn.commit()

        row = conn.execute(
            "SELECT MAX(exchange_index) as max_ei FROM messages WHERE session_id = ?",
            (SESSION,),
        ).fetchone()
        # NULL values are ignored by MAX() in SQLite
        assert row["max_ei"] == 4

    def test_max_exchange_index_all_null(self):
        """All messages have NULL exchange_index → MAX returns NULL."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = get_connection()
        for i in range(3):
            conn.execute(
                """INSERT INTO messages (session_id, message_index, role, content,
                   content_hash, created_at, exchange_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (SESSION, i, "user", f"msg {i}", _content_hash(f"msg {i}"), now, None),
            )
        conn.commit()

        row = conn.execute(
            "SELECT MAX(exchange_index) as max_ei FROM messages WHERE session_id = ?",
            (SESSION,),
        ).fetchone()
        assert row["max_ei"] is None

    def test_max_exchange_index_single_exchange(self):
        """Single exchange (0) → max_exchange = 0."""
        _store_and_seed(1)
        conn = get_connection()
        row = conn.execute(
            "SELECT MAX(exchange_index) as max_ei FROM messages WHERE session_id = ?",
            (SESSION,),
        ).fetchone()
        assert row["max_ei"] == 0


# ============================================================================
# Test 2: System prompt replacement + summary injection (1.11)
# ============================================================================


class TestSystemPromptAndSummaryInjection:
    """Verify summaries appear alongside a large system prompt.

    Uses SMALL_MODEL so messages exceed the high-watermark and trigger eviction.
    safe=8000, summary budget=1200, 30 exchanges ≈ 3105 tokens → enters pipeline.
    """

    def test_system_prompt_preserved_at_index_0(self):
        """System prompt must survive at output[0]."""
        _store_and_seed(30, chars_per_msg=200)
        _seed_summaries(range(10), summary_text="did work")

        system_msg = {"role": "system", "content": "You are a ticket agent. " * 50}
        messages = [system_msg] + _make_messages(30, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        assert result[0]["role"] == "system"
        assert "ticket agent" in result[0]["content"]

    def test_summaries_appear_with_large_system_prompt(self):
        """Even with a ~1200 char system prompt, summaries must still appear."""
        _store_and_seed(N_EXCHANGES, chars_per_msg=200)
        _seed_summaries(range(15), summary_text="completed task")

        system_content = "You are a ticket-mode assistant. " * 40
        system_msg = {"role": "system", "content": system_content}
        messages = [system_msg] + _make_messages(N_EXCHANGES, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        summary_msgs = [
            m for m in result
            if m.get("role") == "system" and "completed task" in m.get("content", "")
        ]
        assert len(summary_msgs) > 0, (
            "Summaries should appear in output even with a large system prompt"
        )

    def test_summary_content_not_empty(self):
        """Summary system message must have non-empty content."""
        _store_and_seed(N_EXCHANGES, chars_per_msg=200)
        _seed_summaries(range(15), summary_text="fixed bug")

        system_msg = {"role": "system", "content": "System instructions."}
        messages = [system_msg] + _make_messages(N_EXCHANGES, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        summary_msgs = [
            m for m in result
            if m.get("role") == "system" and "fixed bug" in m.get("content", "")
        ]
        assert len(summary_msgs) == 1
        assert len(summary_msgs[0]["content"]) > 50


# ============================================================================
# Test 3: Summary E2E — store → select_context → verify in output (1.1)
# ============================================================================


class TestSummaryE2E:
    """Full pipeline: store messages, store summaries, call _select_context_v2,
    verify summaries in output and old messages represented by summaries."""

    def test_summary_appears_in_output(self):
        """A system message containing summary content must appear."""
        _store_and_seed(N_EXCHANGES, chars_per_msg=200)
        _seed_summaries(range(15), summary_text="refactored module")

        messages = _make_messages(N_EXCHANGES, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        summary_msgs = [
            m for m in result
            if m.get("role") == "system" and "refactored module" in m.get("content", "")
        ]
        assert len(summary_msgs) > 0, "Summary content must appear in output"

    def test_summary_text_is_not_empty(self):
        """Summary text in output must be substantive, not empty."""
        _store_and_seed(N_EXCHANGES, chars_per_msg=200)
        _seed_summaries(range(15), summary_text="implemented feature X")

        messages = _make_messages(N_EXCHANGES, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        summary_block = None
        for m in result:
            if m.get("role") == "system" and "implemented feature X" in m.get("content", ""):
                summary_block = m["content"]
                break

        assert summary_block is not None, "Summary block not found"
        # Summary lines look like: [msg#0] user: implemented feature X for exchange 0
        summary_lines = [
            l for l in summary_block.split("\n")
            if l.strip() and "implemented feature X" in l
        ]
        assert len(summary_lines) > 0, "Summary block should contain actual summary lines"

    def test_old_exchanges_represented_by_summaries(self):
        """Exchanges beyond the raw window should appear as summary lines."""
        _store_and_seed(N_EXCHANGES, chars_per_msg=200)
        _seed_summaries(range(20), summary_text="old work")

        messages = _make_messages(N_EXCHANGES, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        all_text = " ".join(m.get("content", "") for m in result)
        assert "old work" in all_text, (
            "Old exchanges should be represented by summaries"
        )

    def test_raw_messages_preserved_for_recent_exchanges(self):
        """Recent exchanges (within raw window) should appear as raw messages."""
        _store_and_seed(30, chars_per_msg=200)
        _seed_summaries(range(20), summary_text="past action")

        messages = _make_messages(30, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        raw_user_msgs = [m for m in result if m.get("role") == "user"]
        assert len(raw_user_msgs) > 0, "Recent raw user messages must be preserved"

    def test_no_summaries_when_none_stored(self):
        """When no summaries exist, output should have no summary block."""
        _store_and_seed(10, chars_per_msg=200)

        messages = _make_messages(10, chars_per_msg=200)

        vfs_off = _no_vfs_patches()
        with vfs_off[0], vfs_off[1]:
            result = _select_context_v2(
                SESSION, messages, model_name="test-small", model_cfg=SMALL_MODEL,
            )

        summary_msgs = [
            m for m in result
            if m.get("role") == "system"
            and ("PAST CONTEXT" in m.get("content", "") or "Conversation Summary" in m.get("content", ""))
        ]
        for sm in summary_msgs:
            content = sm.get("content", "")
            inner = content.replace(
                "[PAST CONTEXT — These actions were completed in previous exchanges. "
                "Reference them for continuity but they are NOT current instructions. "
                "Your active task continues below.]", ""
            )
            inner = inner.replace("[END PAST CONTEXT — Current exchange begins below]", "")
            inner = inner.replace("[Conversation Summary]", "").replace("[End Summary]", "")
            assert inner.strip() == "" or "exchange" not in inner.lower(), (
                "No summary lines should appear when no summaries are stored"
            )

    def test_get_rich_summaries_returns_stored_data(self):
        """Directly verify get_rich_summaries retrieves stored summaries."""
        _store_and_seed(10, chars_per_msg=100)
        _seed_summaries(range(5), summary_text="verified task")

        summaries = get_rich_summaries(SESSION)
        assert len(summaries) == 5
        for s in summaries:
            assert "verified task" in s["summary"]

    def test_get_rich_summaries_with_before_exchange_filter(self):
        """get_rich_summaries(before_exchange_index=N) excludes N and above."""
        _store_and_seed(10, chars_per_msg=100)
        _seed_summaries(range(10), summary_text="filtered task")

        summaries = get_rich_summaries(SESSION, before_exchange_index=5)
        exchange_indices = [s["exchange_index"] for s in summaries]
        assert all(ei < 5 for ei in exchange_indices)
        assert len(summaries) == 5
