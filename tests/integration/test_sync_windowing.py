"""M3-1 (P1.3): _determine_sync_range must read the summarization cursor from DB.

Pre-fix: last_synced = -1 was hardcoded, so every sync cycle re-processed
exchanges 0..9 regardless of session size. For any session with >10
exchanges, exchanges 10+ were never summarized (DATA LOSS).

This test seeds the cursor at exchange 9, inserts 15 exchanges, and
asserts the next call returns start_exchange=10 (NOT 0).
"""

import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)

    from memory.db import init_db

    init_db(db_path)
    yield db_path


def _seed_messages(session_id, count, start_message_index=0):
    """Insert `count` messages into the DB with consecutive exchange_index."""
    import time
    from memory.db import init_db, connection
    init_db()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with connection() as conn:
        for i in range(count):
            exchange = i
            msg_idx = start_message_index + i * 2
            conn.execute(
                "INSERT INTO messages (session_id, message_index, role, content, exchange_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, msg_idx, "user", f"msg {i}", exchange, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, message_index, role, content, exchange_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, msg_idx + 1, "assistant", f"reply {i}", exchange, now),
            )
        conn.commit()


def test_determine_sync_range_reads_cursor_not_minus_one():
    """Cursor=9 → next start_exchange=10 (NOT 0)."""
    import memory.db as db_module
    from memory.db_summaries import set_last_summarized_exchange_index
    from memory.sync import _determine_sync_range

    session_id = "test-windowing-cursor-9"
    model_name = "test-model-windowing-9"

    db_module.init_db()
    _seed_messages(session_id, count=15)
    set_last_summarized_exchange_index(session_id, 9)

    result = _determine_sync_range(session_id)

    assert result is not None, "should return a range when exchanges 10..14 exist"
    assert result["start_exchange"] == 10, (
        f"start_exchange must be 10 (cursor=9 → 9+1), got {result['start_exchange']}. "
        f"Pre-fix code hardcoded last_synced=-1, returning start_exchange=0 and re-processing 0..9."
    )
    assert result["end_exchange"] == 14, (
        f"end_exchange should be 14 (max 15-1), got {result['end_exchange']}"
    )


def test_determine_sync_range_first_run_bootstrap():
    """No cursor row → bootstrap from 0, start_exchange=0."""
    import memory.db as db_module
    from memory.sync import _determine_sync_range

    session_id = "test-windowing-bootstrap"

    db_module.init_db()
    _seed_messages(session_id, count=5)

    result = _determine_sync_range(session_id)

    assert result is not None
    assert result["start_exchange"] == 0
    assert result["end_exchange"] == 4


def test_determine_sync_range_no_messages_returns_none():
    """Empty session → no exchanges to process → None."""
    import memory.db as db_module
    from memory.sync import _determine_sync_range

    session_id = "test-windowing-empty"

    db_module.init_db()
    result = _determine_sync_range(session_id)

    assert result is None


def test_determine_sync_range_cursor_at_max_returns_none():
    """Cursor equals max exchange index → no more work to do."""
    import memory.db as db_module
    from memory.db_summaries import set_last_summarized_exchange_index
    from memory.sync import _determine_sync_range

    session_id = "test-windowing-caught-up"

    db_module.init_db()
    _seed_messages(session_id, count=5)
    set_last_summarized_exchange_index(session_id, 4)

    result = _determine_sync_range(session_id)
    assert result is None, "cursor=max means nothing new to process"


def test_determine_sync_range_respects_batch_window():
    """With cursor=0 and 25 exchanges, returns first 10 (batch size)."""
    import memory.db as db_module
    from memory.sync import _determine_sync_range

    session_id = "test-windowing-batch-25"

    db_module.init_db()
    _seed_messages(session_id, count=25)

    result = _determine_sync_range(session_id)

    assert result is not None
    assert result["start_exchange"] == 0
    assert result["end_exchange"] == 9
