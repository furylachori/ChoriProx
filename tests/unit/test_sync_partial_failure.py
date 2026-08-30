"""M5-4: regression test for data loss in _summarize_uncached_exchanges.

Bug: previously, the summarize phase advanced the cursor to end_exchange
on partial success (new_count > 0). failed_exchanges above the cursor
were lost — _identify_uncached_exchanges would no longer see them, the
next sync cycle would never retry them. Permanent data loss.

Fix: cursor now advances only to the highest CONTIGUOUS successful
exchange. The first failed exchange blocks the cursor. On the next
sync cycle, _identify_uncached_exchanges re-encounters the failed
exchanges (because the cursor hasn't crossed them) and retries them.
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


def _store_exchange(session_id, exchange_idx, user_content, assistant_content):
    from memory.db import store_message
    user_hash = hashlib.sha256(user_content.encode()).hexdigest()[:16]
    store_message(session_id, exchange_idx * 2, "user", content=user_content, content_hash=user_hash)
    store_message(session_id, exchange_idx * 2 + 1, "assistant", content=assistant_content)


def _seed_exchanges(session_id, count, content_prefix="User message"):
    for i in range(count):
        _store_exchange(
            session_id, i, f"{content_prefix} {i} with enough content", f"Assistant reply {i}"
        )


def _make_sota_with_failure_at(fail_indices, model="test-model"):
    """Build a mock SOTA whose summarize_message raises SOTAAPIError
    on the configured fail_indices (counted from the start of the
    processed batch)."""
    sota = MagicMock()
    sota.model = model
    counter = {"n": 0}

    def fake_summarize(session_id, exchange_idx, *args, **kwargs):
        counter["n"] += 1
        if counter["n"] - 1 in fail_indices:
            from memory.sota_channel import SOTAAPIError
            raise SOTAAPIError("simulated SOTA failure", status_code=500)
        return {
            "summary": f"Summary for exchange {exchange_idx} from mock SOTA",
            "tool_summary": "",
            "token_estimate": 30,
        }

    sota.summarize_message.side_effect = fake_summarize
    return sota


def _cursor(session_id):
    from memory.db import get_connection
    row = get_connection().execute(
        "SELECT last_exchange_index FROM conversation_summaries "
        "WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row["last_exchange_index"] if row else 0


class TestCursorAdvanceOnPartialFailure:
    """M5-4: cursor must NOT cross failed exchanges."""

    def test_first_failure_blocks_cursor_advancing_past_it(self):
        """Exchange 2 fails; cursor must stop at exchange 1 (the highest contiguous success)."""
        session_id = "test-m5-4-block"
        _seed_exchanges(session_id, 5)

        sota = _make_sota_with_failure_at(fail_indices={2})

        with patch("memory.sync._get_sota", return_value=sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        # Cursor must NOT be 4 (end_exchange) — that would lose exchange 2.
        # Cursor must be 1 — the highest contiguous success starting at 0.
        assert _cursor(session_id) == 1, (
            f"Expected cursor=1 (highest contiguous success before failure at #2), "
            f"got {_cursor(session_id)}. The previous bug would have set cursor=4."
        )

    def test_failed_exchanges_are_retried_on_next_run(self):
        """A failed exchange must be re-encountered and processed on the next sync run.

        Cursor advance is gated on BOTH successful summarization AND
        successful extraction. Since extraction is mocked to fail in
        this test (it's not the focus), the cursor stays at 1 across
        both runs — but exchange 2 must be summarized on the second run
        because the cursor never crossed it.
        """
        session_id = "test-m5-4-retry"
        _seed_exchanges(session_id, 5)

        sota_fail_first = _make_sota_with_failure_at(fail_indices={2})
        # Disable extraction phase so the cursor doesn't advance to end_exchange
        # in the post-extraction step (line 848). The test is about
        # summarization cursor (line 734), not the post-extraction one.
        sota_fail_first.extract_and_summarize_from_summaries.side_effect = Exception(
            "test: skip extraction"
        )

        with patch("memory.sync._get_sota", return_value=sota_fail_first):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        # First run: cursor stops at 1 (summarization cursor).
        assert _cursor(session_id) == 1

        # Verify exchange 2 was NOT stored in message_summaries.
        from memory.db import get_connection
        rows = get_connection().execute(
            "SELECT exchange_index FROM message_summaries "
            "WHERE session_id = ? ORDER BY exchange_index",
            (session_id,),
        ).fetchall()
        stored_indices = {r["exchange_index"] for r in rows}
        assert 2 not in stored_indices, (
            f"Exchange 2 should NOT be stored (SOTA failed for it). "
            f"Stored: {sorted(stored_indices)}"
        )

        # Second run: a fresh SOTA mock that succeeds on every call.
        # Extraction still mocked-out (not the focus of this test).
        sota_succeed_all = _make_sota_with_failure_at(fail_indices=set())
        sota_succeed_all.extract_and_summarize_from_summaries.side_effect = Exception(
            "test: skip extraction"
        )
        with patch("memory.sync._get_sota", return_value=sota_succeed_all):
            sync_exchange(session_id)

        # After the retry, exchange 2 (previously blocked) must now be
        # stored. The summarization phase picks it up because the cursor
        # never crossed it (set to 1 in run 1, not advanced since).
        rows = get_connection().execute(
            "SELECT exchange_index FROM message_summaries "
            "WHERE session_id = ? ORDER BY exchange_index",
            (session_id,),
        ).fetchall()
        stored_indices = {r["exchange_index"] for r in rows}
        assert 2 in stored_indices, (
            f"After retry, exchange 2 (the previously failed one) "
            f"should be stored. Stored: {sorted(stored_indices)}"
        )

    def test_all_failures_dont_advance_cursor(self):
        """If every SOTA call fails, the cursor must NOT advance."""
        session_id = "test-m5-4-all-fail"
        _seed_exchanges(session_id, 3)

        sota = _make_sota_with_failure_at(fail_indices={0, 1, 2})

        with patch("memory.sync._get_sota", return_value=sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        # No summaries stored. Cursor did not advance.
        assert _cursor(session_id) == 0, (
            f"Cursor should be 0 (no successes), got {_cursor(session_id)}"
        )
        from memory.db import get_connection
        rows = get_connection().execute(
            "SELECT COUNT(*) AS n FROM message_summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert rows["n"] == 0

    def test_late_failure_does_not_undo_earlier_successes(self):
        """Failures late in the batch don't erase earlier successful summaries."""
        session_id = "test-m5-4-late"
        _seed_exchanges(session_id, 5)

        # Exchange 4 (last) fails. Exchanges 0..3 succeed.
        sota = _make_sota_with_failure_at(fail_indices={4})

        with patch("memory.sync._get_sota", return_value=sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        # Cursor advances to 3 (highest contiguous success before failure at #4).
        assert _cursor(session_id) == 3, (
            f"Expected cursor=3 (highest contiguous success), got {_cursor(session_id)}"
        )

        from memory.db import get_connection
        rows = get_connection().execute(
            "SELECT exchange_index FROM message_summaries "
            "WHERE session_id = ? ORDER BY exchange_index",
            (session_id,),
        ).fetchall()
        stored_indices = {r["exchange_index"] for r in rows}
        assert stored_indices == {0, 1, 2, 3}, (
            f"Expected 0..3 stored, got: {sorted(stored_indices)}"
        )

    def test_first_run_failure_does_not_lose_subsequent_exchanges(self):
        """The old bug: a single failure in the middle causes the cursor to
        jump to end_exchange, losing every exchange above the failure.

        This test reproduces the OLD bug behavior to confirm the fix
        prevents it. Before the fix, cursor would equal 4 (end_exchange).
        After the fix, cursor equals 1."""
        session_id = "test-m5-4-no-data-loss"
        _seed_exchanges(session_id, 5)

        sota = _make_sota_with_failure_at(fail_indices={2})

        with patch("memory.sync._get_sota", return_value=sota):
            from memory.sync import sync_exchange
            sync_exchange(session_id)

        # The old buggy behavior would set cursor=4 (= end_exchange). After
        # the fix, cursor is the highest contiguous success before the
        # failure, which is 1.
        old_buggy_cursor = 4
        assert _cursor(session_id) != old_buggy_cursor, (
            "REGRESSION: cursor advanced past the failed exchange (data loss bug)."
        )
        assert _cursor(session_id) == 1