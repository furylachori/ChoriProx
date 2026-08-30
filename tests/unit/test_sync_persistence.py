"""Tests for sync pipeline persistence and exchange identification.

Covers:
- Test 1 (1.2): submit_sync() / sync_exchange() creates SQLite summary rows
- Test 2 (1.3): _identify_uncached_exchanges() with real SQLite
- Test 3 (1.4): exchange_index assignment during message persistence
"""

import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_test_db():
    os.environ["PROXY_TEST_DB"] = ":memory:"
    from memory.db import init_db

    init_db()
    yield
    os.environ.pop("PROXY_TEST_DB", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_exchange(session_id, exchange_idx, user_content, assistant_content):
    """Store a user+assistant pair with proper exchange_index."""
    from memory.db import store_message
    user_hash = hashlib.sha256(user_content.encode()).hexdigest()[:16]
    # user message at message_index = exchange_idx * 2
    store_message(session_id, exchange_idx * 2, "user", content=user_content, content_hash=user_hash)
    # assistant message at message_index = exchange_idx * 2 + 1
    store_message(session_id, exchange_idx * 2 + 1, "assistant", content=assistant_content)


def _seed_exchanges(session_id, count, content_prefix="User message"):
    """Store N user+assistant exchanges."""
    for i in range(count):
        _store_exchange(session_id, i, f"{content_prefix} {i} with enough content", f"Assistant reply {i}")


def _compute_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# Test 1 (1.2): sync_exchange() creates SQLite summary rows
# ============================================================================


class TestSyncExchangePersistence:
    """sync_exchange() must persist per-exchange summaries to message_summaries."""

    def test_sync_exchange_creates_summary_rows(self):
        """sync_exchange() with mocked SOTA stores summary rows in message_summaries."""
        session_id = "test-sync-persist-1"
        _seed_exchanges(session_id, 3)

        from memory.db import get_connection

        mock_sota = MagicMock()
        mock_sota.model = "test-model"
        mock_sota.summarize_message.return_value = {
            "summary": "User asked about Python async patterns and got a detailed explanation.",
            "tool_summary": "",
            "token_estimate": 42,
        }

        with patch("memory.sync._get_sota", return_value=mock_sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        conn = get_connection()
        rows = conn.execute(
            "SELECT session_id, exchange_index, summary, content_hash FROM message_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        assert len(rows) > 0, "sync_exchange should have stored at least one summary row"
        for row in rows:
            assert row["session_id"] == session_id
            assert row["exchange_index"] is not None
            assert row["summary"] is not None
            assert len(row["summary"]) > 0
            assert row["content_hash"] is not None

    def test_sync_exchange_summary_has_correct_session_id(self):
        """Summary rows must reference the correct session_id."""
        session_id = "test-sync-session-match"
        _seed_exchanges(session_id, 2)

        mock_sota = MagicMock()
        mock_sota.model = "test-model"
        mock_sota.summarize_message.return_value = {
            "summary": "A meaningful summary of the exchange that is long enough.",
            "tool_summary": "",
            "token_estimate": 30,
        }

        with patch("memory.sync._get_sota", return_value=mock_sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        from memory.db import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT session_id, exchange_index FROM message_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        assert len(rows) >= 1
        for row in rows:
            assert row["session_id"] == session_id

    def test_sync_exchange_stores_content_hash(self):
        """Summary rows must have a content_hash matching the user message."""
        session_id = "test-sync-hash"
        user_content = "Tell me about the history of computing machines and their impact"
        _store_exchange(session_id, 0, user_content, "Computers evolved from abacus to modern CPUs.")
        expected_hash = _compute_hash(user_content)

        mock_sota = MagicMock()
        mock_sota.model = "test-model"
        mock_sota.summarize_message.return_value = {
            "summary": "User inquired about computing history and received a comprehensive overview.",
            "tool_summary": "",
            "token_estimate": 25,
        }

        with patch("memory.sync._get_sota", return_value=mock_sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        from memory.db import get_connection
        conn = get_connection()
        row = conn.execute(
            "SELECT content_hash FROM message_summaries WHERE session_id = ? AND exchange_index = 0",
            (session_id,),
        ).fetchone()

        assert row is not None
        assert row["content_hash"] == expected_hash


# ============================================================================
# Test 2 (1.3): _identify_uncached_exchanges() with real SQLite
# ============================================================================


class TestIdentifyUncachedExchanges:
    """_identify_uncached_exchanges() must correctly filter cached vs uncached."""

    def test_returns_only_uncached_exchanges(self):
        """Exchanges with existing summaries are skipped; uncached ones are returned."""
        session_id = "test-uncached-filter"
        _seed_exchanges(session_id, 10)

        from memory.db import store_message_summary

        # Cache summaries for exchanges 0-4
        for i in range(5):
            content = f"User message {i} with enough content"
            h = _compute_hash(content)
            store_message_summary(session_id, i, "user", f"Summary for exchange {i}", "", h)

        from memory.sync import _identify_uncached_exchanges

        # start_idx=0, end_idx=19 (10 exchanges × 2 messages each)
        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 19)

        uncached_indices = [exchange_idx for _, exchange_idx, _ in uncached]
        assert sorted(uncached_indices) == [5, 6, 7, 8, 9]

    def test_short_messages_skipped(self):
        """Exchanges where user message < 20 chars are skipped."""
        session_id = "test-short-msg"
        _store_exchange(session_id, 0, "hi", "Hello!")
        _store_exchange(session_id, 1, "This is a long enough user message for testing.", "Got it.")

        from memory.sync import _identify_uncached_exchanges

        # message indices: exchange 0 -> msgs 0,1; exchange 1 -> msgs 2,3
        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 3)

        uncached_indices = [exchange_idx for _, exchange_idx, _ in uncached]
        assert 0 not in uncached_indices, "Short message exchange should be skipped"
        assert 1 in uncached_indices, "Long message exchange should be included"

    def test_null_exchange_index_skipped(self):
        """Messages with NULL exchange_index are grouped and skipped."""
        session_id = "test-null-ei"
        from memory.db import store_message

        # Store a message with a valid exchange_index
        store_message(session_id, 0, "user", content="This message has a valid exchange index set")
        # Insert a message with NULL exchange_index by directly manipulating the DB
        from memory.db import immediate_transaction
        with immediate_transaction() as conn:
            conn.execute(
                """INSERT INTO messages (session_id, message_index, role, content, content_hash, created_at, exchange_index)
                   VALUES (?, ?, ?, ?, ?, datetime('now'), NULL)""",
                (session_id, 99, "user", "This message has NULL exchange_index", "null-ei-hash"),
            )

        from memory.sync import _identify_uncached_exchanges

        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 99)

        # Should only include exchange 0 (the one with valid exchange_index)
        uncached_indices = [exchange_idx for _, exchange_idx, _ in uncached]
        assert 0 in uncached_indices
        # NULL exchange_index rows are excluded; there should be exactly 1 uncached exchange
        assert len(uncached) == 1

    def test_all_cached_returns_empty(self):
        """When all exchanges are cached, return empty uncached list."""
        session_id = "test-all-cached"
        _seed_exchanges(session_id, 3)

        from memory.db import store_message_summary

        for i in range(3):
            content = f"User message {i} with enough content"
            h = _compute_hash(content)
            store_message_summary(session_id, i, "user", f"Summary {i}", "", h)

        from memory.sync import _identify_uncached_exchanges

        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 5)
        assert uncached == []

    def test_no_messages_returns_empty(self):
        """When no messages exist in range, return empty uncached list."""
        session_id = "test-no-messages"

        from memory.sync import _identify_uncached_exchanges

        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 99)
        assert uncached == []
        assert all_rows == []

    def test_all_rows_snapshot_overwritten_by_summary_lookup(self):
        """REGRESSION GUARD: _identify_uncached_exchanges() must NOT
        overwrite the all_rows message snapshot with the summary-lookup
        result.

        Previously the inner block at line 320 reassigned `rows` from
        the original message snapshot to the message_summaries lookup
        result. The documented contract says all_rows is the pre-fetched
        message snapshot for the range. M5-4 captured the original
        rows in `initial_rows` so the snapshot survives the inner
        reassignment.
        """
        session_id = "test-all-rows"
        _seed_exchanges(session_id, 3)

        from memory.sync import _identify_uncached_exchanges

        # With long messages (>20 chars), lookup_keys is non-empty →
        # the inner block used to reassign `rows` to the summary lookup
        # result. M5-4 ensures the message snapshot is preserved.
        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 5)
        assert len(uncached) == 3  # uncached exchanges are correct
        # The all_rows snapshot must be the original message snapshot,
        # NOT the message_summaries lookup result.
        assert len(all_rows) == 6, (
            f"all_rows was the summary-lookup result (empty), not the "
            f"6-row message snapshot. Length: {len(all_rows)}"
        )

    def test_all_rows_preserved_when_no_lookup(self):
        """When all exchanges have short messages (< 20 chars), lookup_keys is
        empty so `rows` is never reassigned — all_rows is the correct snapshot."""
        session_id = "test-all-rows-nolookup"
        # Store short messages (< 20 chars) so they are skipped by exchange filter
        from memory.db import store_message
        store_message(session_id, 0, "user", content="hi there")
        store_message(session_id, 1, "assistant", content="hello back")
        store_message(session_id, 2, "user", content="short")
        store_message(session_id, 3, "assistant", content="reply")

        from memory.sync import _identify_uncached_exchanges

        uncached, all_rows = _identify_uncached_exchanges(session_id, 0, 3)
        assert len(uncached) == 0  # all skipped (short messages)
        # all_rows preserved since no lookup occurred
        assert len(all_rows) == 4


# ============================================================================
# Test 3 (1.4): exchange_index assignment during persistence
# ============================================================================


class TestExchangeIndexAssignment:
    """Messages stored via store_messages_batch() must have correct exchange_index."""

    def test_sequential_exchange_indices(self):
        """User messages increment exchange_index; assistant messages share preceding user's index."""
        session_id = "test-ei-assign"
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "How do I install it?"},
            {"role": "assistant", "content": "Use your package manager."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "Sure, here is hello world."},
        ]

        from memory.db import store_messages_batch, get_connection

        store_messages_batch(session_id, messages, 0)

        conn = get_connection()
        rows = conn.execute(
            "SELECT message_index, role, exchange_index FROM messages WHERE session_id = ? ORDER BY message_index",
            (session_id,),
        ).fetchall()

        assert len(rows) == 6

        # User messages should have exchange_index 0, 1, 2
        user_rows = [r for r in rows if r["role"] == "user"]
        user_eis = [r["exchange_index"] for r in user_rows]
        assert user_eis == [0, 1, 2], f"User exchange_indices should be sequential: {user_eis}"

        # Assistant messages should share their preceding user's exchange_index
        assistant_rows = [r for r in rows if r["role"] == "assistant"]
        assistant_eis = [r["exchange_index"] for r in assistant_rows]
        assert assistant_eis == [0, 1, 2], f"Assistant exchange_indices should match users: {assistant_eis}"

    def test_no_null_exchange_index(self):
        """All stored messages must have non-NULL exchange_index."""
        session_id = "test-no-null-ei"
        messages = [
            {"role": "user", "content": "First question about databases"},
            {"role": "assistant", "content": "Here is the answer about databases."},
            {"role": "user", "content": "Follow-up question about indexing"},
            {"role": "assistant", "content": "Indexes improve query performance."},
            {"role": "user", "content": "Final question about normalization"},
            {"role": "assistant", "content": "Normalization reduces data redundancy."},
        ]

        from memory.db import store_messages_batch, get_connection

        store_messages_batch(session_id, messages, 0)

        conn = get_connection()
        rows = conn.execute(
            "SELECT exchange_index FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        for row in rows:
            assert row["exchange_index"] is not None, "exchange_index must not be NULL"

    def test_exchange_index_continues_across_batches(self):
        """Second batch continues exchange_index from where first batch left off."""
        session_id = "test-ei-continuation"
        batch1 = [
            {"role": "user", "content": "First batch question"},
            {"role": "assistant", "content": "First batch answer"},
        ]
        batch2 = [
            {"role": "user", "content": "Second batch question"},
            {"role": "assistant", "content": "Second batch answer"},
        ]

        from memory.db import store_messages_batch, get_connection

        store_messages_batch(session_id, batch1, 0)
        store_messages_batch(session_id, batch2, 2)

        conn = get_connection()
        rows = conn.execute(
            "SELECT message_index, role, exchange_index FROM messages WHERE session_id = ? ORDER BY message_index",
            (session_id,),
        ).fetchall()

        assert rows[0]["exchange_index"] == 0  # user msg 0
        assert rows[1]["exchange_index"] == 0  # assistant msg 1
        assert rows[2]["exchange_index"] == 1  # user msg 2
        assert rows[3]["exchange_index"] == 1  # assistant msg 3

    def test_single_message_store_assigns_exchange_index(self):
        """store_message() also assigns exchange_index correctly."""
        session_id = "test-single-msg-ei"

        from memory.db import store_message, get_connection

        store_message(session_id, 0, "user", content="A user message for testing exchange index")
        store_message(session_id, 1, "assistant", content="An assistant response for testing")
        store_message(session_id, 2, "user", content="Another user message for exchange index testing")

        conn = get_connection()
        rows = conn.execute(
            "SELECT message_index, role, exchange_index FROM messages WHERE session_id = ? ORDER BY message_index",
            (session_id,),
        ).fetchall()

        assert rows[0]["exchange_index"] == 0  # first user
        assert rows[1]["exchange_index"] == 0  # assistant
        assert rows[2]["exchange_index"] == 1  # second user

    def test_five_user_five_assistant_sequential(self):
        """5 user messages and 5 assistant responses get sequential exchange indices 0-4."""
        session_id = "test-5u5a"
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"User question number {i} about a technical topic"})
            messages.append({"role": "assistant", "content": f"Assistant answer number {i} with details"})

        from memory.db import store_messages_batch, get_connection

        store_messages_batch(session_id, messages, 0)

        conn = get_connection()
        rows = conn.execute(
            "SELECT message_index, role, exchange_index FROM messages WHERE session_id = ? ORDER BY message_index",
            (session_id,),
        ).fetchall()

        assert len(rows) == 10

        for i in range(5):
            user_row = rows[i * 2]
            asst_row = rows[i * 2 + 1]
            assert user_row["role"] == "user"
            assert asst_row["role"] == "assistant"
            assert user_row["exchange_index"] == i
            assert asst_row["exchange_index"] == i
