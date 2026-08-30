"""US-F1: Cursor lifecycle — advance on fully-cached windows, monotonic writes."""
import hashlib
import threading
import time

import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Per-test fresh DB."""
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)
    from memory.db import init_db
    init_db(db_path)
    yield db_path


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _seed_user_assistant_exchange(session_id, exchange_idx, msg_idx, user_content, reply="reply"):
    """Insert a user+assistant exchange into messages table."""
    now = _now()
    from memory.db_pool import connection
    with connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, exchange_index, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, msg_idx, "user", user_content, exchange_idx, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, exchange_index, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, msg_idx + 1, "assistant", reply, exchange_idx, now),
        )
        conn.commit()


def _seed_summary(session_id, exchange_idx, content_hash, summary="cached summary"):
    """Insert a matching summary for dedup."""
    now = _now()
    from memory.db_pool import connection
    with connection() as conn:
        conn.execute(
            "INSERT INTO message_summaries "
            "(session_id, exchange_index, role, summary, content_hash, model_used, "
            " token_estimate, end_message_index, message_index, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, exchange_idx, "user", summary, content_hash, "test-model",
             10, 0, 0, now, now),
        )
        conn.commit()


def _compute_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _cursor(session_id):
    from memory.db_summaries import get_last_summarized_exchange_index
    return get_last_summarized_exchange_index(session_id)


def _enable_extraction_path(monkeypatch, batch_id="batch_test"):
    """Patch the extraction+persistence helpers so sync_exchange completes the path."""
    monkeypatch.setattr("memory.db.get_rich_summaries", lambda *a, **kw: [
        {"message_index": 1, "exchange_index": 1, "summary": "x", "rendered_narrative_tokens": 10},
        {"message_index": 2, "exchange_index": 2, "summary": "y", "rendered_narrative_tokens": 10},
        {"message_index": 3, "exchange_index": 3, "summary": "z", "rendered_narrative_tokens": 10},
    ])
    monkeypatch.setattr("memory.db.get_summary_age", lambda *a, **kw: None)
    monkeypatch.setattr("memory.db.get_messages_in_range", lambda *a, **kw: [
        {"role": "user", "content": "x", "message_index": 1},
    ])
    monkeypatch.setattr("memory.db.get_extraction_batch", lambda *a, **kw: None)
    monkeypatch.setattr("memory.db.record_extraction_batch", lambda *a, **kw: batch_id)
    monkeypatch.setattr("memory.batch_manager._set_batch_processing", lambda *a, **kw: (True, False))
    monkeypatch.setattr("memory.batch_manager._claim_batch_for_extraction", lambda *a, **kw: True)
    monkeypatch.setattr("memory.batch_manager._update_batch_extraction", lambda *a, **kw: True)
    monkeypatch.setattr("memory.batch_manager._update_batch_status", lambda *a, **kw: True)
    monkeypatch.setattr("memory.batch_manager._update_batch_status_if", lambda *a, **kw: True)
    monkeypatch.setattr("memory.sync._store_entities_and_facts", lambda *a, **kw: {"errors": []})
    monkeypatch.setattr("memory.sync._get_sota", lambda: _mock_sota_client())


def _mock_sota_client():
    class _MockSOTA:
        def summarize_message(self, *a, **kw):
            return {
                "summary": "summary text",
                "tool_summary": "",
                "token_estimate": 10,
            }

        def extract_and_summarize_from_summaries(self, *a, **kw):
            return {"entities": [], "facts": []}

        model = "test-sota-model"

    return _MockSOTA()


class TestCursorLifecycle:
    def test_all_dedup_window_advances_cursor(self, monkeypatch):
        """Seed 10 exchanges with matching summaries; bootstrap cursor at -1;
        run sync_exchange; assert cursor advances to 9 (end of batch)."""
        from memory.sync import sync_exchange

        session_id = "test-cursor-all-dedup"

        for i in range(10):
            content = f"unique user content for exchange {i} {i*1000}"
            _seed_user_assistant_exchange(session_id, i, i * 2, content)
            _seed_summary(session_id, i, _compute_hash(content))

        _enable_extraction_path(monkeypatch)

        sync_exchange(session_id)

        assert _cursor(session_id) == 9, (
            f"cursor must advance to 9 after all-dedup window; got {_cursor(session_id)}"
        )

    def test_model_switch_advances_past_dedup_prefix(self, monkeypatch):
        """Pre-populate cursor=9; sync_exchange still advances past dedup'd prefix.

        Post-C3 the cursor is session-scoped — there's no per-model cursor to
        inherit from. This test now verifies that re-running sync_exchange
        advances the session cursor over the remaining dedup'd window
        (exchanges 10..14).
        """
        from memory.db_summaries import set_last_summarized_exchange_index
        from memory.sync import sync_exchange

        session_id = "test-cursor-session-scoped"

        for i in range(15):
            content = f"shared exchange {i} {i*1000}"
            _seed_user_assistant_exchange(session_id, i, i * 2, content)
            _seed_summary(session_id, i, _compute_hash(content))

        set_last_summarized_exchange_index(session_id, 9)

        _enable_extraction_path(monkeypatch)

        sync_exchange(session_id)

        assert _cursor(session_id) == 14, (
            f"session cursor must advance to 14 (end of 15 exchanges) after dedup'd prefix; "
            f"got {_cursor(session_id)}"
        )

    def test_all_short_messages_window_advances_cursor(self, monkeypatch):
        """10 user messages <20 chars → all skipped-but-accounted (sync.py:288).
        Cursor still advances."""
        from memory.sync import sync_exchange

        session_id = "test-cursor-all-short"

        for i in range(10):
            _seed_user_assistant_exchange(session_id, i, i * 2, "ok")

        _enable_extraction_path(monkeypatch)

        sync_exchange(session_id)

        assert _cursor(session_id) == 9, (
            f"cursor must advance over short-message window; got {_cursor(session_id)}"
        )

    def test_monotonic_upsert_with_barrier(self):
        """Two threads + Barrier write cursor 5 and 10. Final value MUST be 10."""
        from memory.db_summaries import (
            set_last_summarized_exchange_index,
            get_last_summarized_exchange_index,
        )

        session_id = "test-monotonic"

        for _ in range(20):
            barrier = threading.Barrier(2)
            results = []

            def write(idx):
                barrier.wait()
                set_last_summarized_exchange_index(session_id, idx)
                results.append(get_last_summarized_exchange_index(session_id))

            t1 = threading.Thread(target=write, args=(5,))
            t2 = threading.Thread(target=write, args=(10,))
            t1.start(); t2.start()
            t1.join(); t2.join()

            final = get_last_summarized_exchange_index(session_id)
            assert final == 10, (
                f"monotonic upsert violated: cursor must end at 10 (max(5,10)), got {final}. "
                f"Per-write results: {results}"
            )

    def test_single_exchange_summarized_advances_cursor(self, monkeypatch):
        """Seed exactly 1 exchange (user + assistant); run sync_exchange;
        cursor must advance to 0. The old `new_cursor > start_exchange`
        comparison was FALSE for a single new exchange because
        start_exchange == new_cursor."""
        from memory.sync import sync_exchange

        session_id = "test-cursor-single-exchange"

        long_user_content = "this is a long enough user message to pass the short-message filter"
        _seed_user_assistant_exchange(session_id, 0, 0, long_user_content, reply="hi back from the assistant")

        _enable_extraction_path(monkeypatch)

        sync_exchange(session_id)

        assert _cursor(session_id) == 0, (
            f"single-exchange cursor must advance to 0 (the only exchange index); "
            f"got {_cursor(session_id)}"
        )

    