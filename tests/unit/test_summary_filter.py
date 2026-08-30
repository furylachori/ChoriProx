"""Tests for get_rich_summaries() filter and _maybe_compress_summaries()."""

import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest

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


from memory.db_summaries import (
    get_rich_summaries,
    store_message_summary,
    get_ancient_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary_text(word_count: int) -> str:
    """Generate a summary string with approximately *word_count* words."""
    return " ".join(["word"] * word_count)


def _store_n_summaries(session_id: str, n: int, word_count: int = 10):
    """Store *n* summaries with exchange_index 0..n-1."""
    for i in range(n):
        store_message_summary(
            session_id=session_id,
            exchange_index=i,
            role="user" if i % 2 == 0 else "assistant",
            summary=_make_summary_text(word_count),
            tool_summary="",
            content_hash=hashlib.sha256(f"hash-{i}".encode()).hexdigest()[:16],
        )


# ============================================================================
# Test 1: get_rich_summaries() before_exchange_index filter
# ============================================================================


class TestGetRichSummariesFilter:
    """Test the before_exchange_index filter on get_rich_summaries()."""

    def test_before_exchange_index_returns_only_older(self):
        """before_exchange_index=12 returns exchanges 0-11 only."""
        sid = "test-filter-older"
        _store_n_summaries(sid, 20)

        results = get_rich_summaries(sid, before_exchange_index=12)

        assert len(results) == 12
        returned_indices = [r["exchange_index"] for r in results]
        assert returned_indices == list(range(12))

    def test_before_exchange_index_zero_returns_empty(self):
        """before_exchange_index=0 returns nothing (no index < 0)."""
        sid = "test-filter-zero"
        _store_n_summaries(sid, 20)

        results = get_rich_summaries(sid, before_exchange_index=0)

        assert results == []

    def test_before_exchange_index_none_returns_all(self):
        """before_exchange_index=None returns all summaries."""
        sid = "test-filter-none"
        _store_n_summaries(sid, 20)

        results = get_rich_summaries(sid, before_exchange_index=None)

        assert len(results) == 20
        returned_indices = [r["exchange_index"] for r in results]
        assert returned_indices == list(range(20))

    def test_before_exchange_index_larger_than_count(self):
        """before_exchange_index larger than stored count returns all."""
        sid = "test-filter-large"
        _store_n_summaries(sid, 5)

        results = get_rich_summaries(sid, before_exchange_index=100)

        assert len(results) == 5

    def test_results_ordered_by_exchange_index(self):
        """Results are ordered by exchange_index ascending."""
        sid = "test-filter-order"
        _store_n_summaries(sid, 10)

        results = get_rich_summaries(sid, before_exchange_index=8)

        indices = [r["exchange_index"] for r in results]
        assert indices == sorted(indices)

    def test_empty_session_returns_empty(self):
        """No summaries stored → empty list."""
        results = get_rich_summaries("nonexistent-session", before_exchange_index=5)
        assert results == []


# ============================================================================
# Test 2: _maybe_compress_summaries() with small-window model
# ============================================================================


class TestMaybeCompressSummaries:
    """Test _maybe_compress_summaries compression trigger and eviction."""

    def _store_large_summaries(self, session_id: str, count: int = 25, word_count: int = 220):
        """Store summaries sized to produce ~300 rendered_narrative_tokens each.

        rendered_narrative_tokens = int(len(narrative_line.split()) / 0.75)
        narrative_line = "[exchange#{i}] {role}: {summary}"  → ~3 prefix words
        So ~220 words in summary → (223 / 0.75) ≈ 297 tokens ≈ 300.
        """
        for i in range(count):
            store_message_summary(
                session_id=session_id,
                exchange_index=i,
                role="user" if i % 2 == 0 else "assistant",
                summary=_make_summary_text(word_count),
                tool_summary="",
                content_hash=hashlib.sha256(f"hash-{i}".encode()).hexdigest()[:16],
            )

    def test_compression_triggers_when_threshold_exceeded(self):
        """25 summaries × ~300 tokens = ~7500 > threshold 1500 → compression fires."""
        from memory.sync import _maybe_compress_summaries

        sid = "test-compress-trigger"
        self._store_large_summaries(sid, count=25, word_count=220)

        mock_sota = MagicMock()
        mock_sota.compress_ancient_history.return_value = "compressed ancient history text"

        with patch("memory.sync._get_sota", return_value=mock_sota):
            _maybe_compress_summaries(sid, safe_context_tokens=8000)

        # SOTA compress API was called
        mock_sota.compress_ancient_history.assert_called_once()

        # Ancient history was stored
        ancient = get_ancient_history(sid)
        assert ancient == "compressed ancient history text"

    def test_evicted_summaries_deleted(self):
        """Summaries chosen for eviction are removed from message_summaries."""
        from memory.sync import _maybe_compress_summaries
        from memory.db import get_connection

        sid = "test-compress-evict"
        self._store_large_summaries(sid, count=25, word_count=220)

        mock_sota = MagicMock()
        mock_sota.compress_ancient_history.return_value = "compressed"

        with patch("memory.sync._get_sota", return_value=mock_sota):
            _maybe_compress_summaries(sid, safe_context_tokens=8000)

        # evict_count = max(10, 25 // 5) = max(10, 5) = 10
        # Summaries 0-9 should be evicted
        remaining = get_rich_summaries(sid)
        remaining_indices = [r["exchange_index"] for r in remaining]

        # The first 10 (indices 0-9) should be gone
        for i in range(10):
            assert i not in remaining_indices, f"exchange_index {i} should have been evicted"

        # The rest (indices 10-24) should remain
        assert len(remaining) == 15

    def test_compression_does_not_fire_when_under_threshold(self):
        """Few small summaries → no compression."""
        from memory.sync import _maybe_compress_summaries

        sid = "test-no-compress"
        # 5 summaries × ~30 tokens = ~150 tokens, well under 1500 threshold
        _store_n_summaries(sid, 5, word_count=10)

        mock_sota = MagicMock()

        with patch("memory.sync._get_sota", return_value=mock_sota):
            _maybe_compress_summaries(sid, safe_context_tokens=8000)

        mock_sota.compress_ancient_history.assert_not_called()
        assert get_ancient_history(sid) == ""

    def test_compression_skipped_on_sota_error(self):
        """SOTA CompressionError → summaries kept, ancient history unchanged."""
        from memory.sync import _maybe_compress_summaries
        from memory.sota_client import CompressionError

        sid = "test-compress-error"
        self._store_large_summaries(sid, count=25, word_count=220)

        mock_sota = MagicMock()
        mock_sota.compress_ancient_history.side_effect = CompressionError("api down")

        with patch("memory.sync._get_sota", return_value=mock_sota):
            _maybe_compress_summaries(sid, safe_context_tokens=8000)

        # No ancient history stored
        assert get_ancient_history(sid) == ""

        # All summaries still present (none evicted)
        remaining = get_rich_summaries(sid)
        assert len(remaining) == 25

    def test_compression_appends_to_existing_ancient(self):
        """Existing ancient_history is passed to SOTA for incremental compression."""
        from memory.sync import _maybe_compress_summaries
        from memory.db_summaries import update_ancient_history

        sid = "test-compress-append"
        update_ancient_history(sid, "previous ancient context")
        self._store_large_summaries(sid, count=25, word_count=220)

        mock_sota = MagicMock()
        mock_sota.compress_ancient_history.return_value = "updated ancient context"

        with patch("memory.sync._get_sota", return_value=mock_sota):
            _maybe_compress_summaries(sid, safe_context_tokens=8000)

        # SOTA received existing ancient history
        call_args = mock_sota.compress_ancient_history.call_args
        assert call_args[0][1] == "previous ancient context"  # existing_ancient param

        # Updated value stored
        assert get_ancient_history(sid) == "updated ancient context"
