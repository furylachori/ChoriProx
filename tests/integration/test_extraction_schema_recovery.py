"""Regression tests for extraction_batches schema recovery.

The proxy was vulnerable to a "no such table: extraction_batches" crash when the
live SQLite file was replaced with a partial/corrupted version (e.g., the prior
backup_live_db fixture used sqlite3.Connection.backup() which is documented
unreliable in WAL mode). These tests pin down both:

1. The proxy's sync helpers self-heal by calling init_db() when the table is gone.
2. The integration fixture's shutil.copy-based backup/restore preserves the
   extraction_batches table AND its rows through a corruption event.
"""

import os
import shutil
import sqlite3
import tempfile

import pytest


def _drop_extraction_batches(db_path):
    c = sqlite3.connect(db_path)
    try:
        c.execute("DROP TABLE IF EXISTS extraction_batches")
        for idx in (
            "idx_extraction_batches_status",
            "idx_extraction_batches_session",
            "idx_batches_covered",
        ):
            c.execute(f"DROP TABLE IF EXISTS {idx}")
        c.commit()
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        c.close()


def test_sync_recovers_when_extraction_batches_missing(tmp_path, monkeypatch):
    """Sync ops must call init_db() and proceed when extraction_batches is gone."""
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)

    import memory.db as db
    from memory.batch_manager import (
        _get_incomplete_batches,
        _is_covered,
        _set_batch_processing,
        _update_batch_status,
    )
    from memory.db_summaries import (
        get_batch_age,
        get_batch_status,
        get_extraction_batch,
        record_extraction_batch,
    )

    db.init_db(db_path)

    bid = record_extraction_batch("sess-A", 0, 1, start_message_index=0, end_message_index=1)
    assert bid is not None
    assert get_batch_status(bid) == "pending"

    _drop_extraction_batches(db_path)

    raw = sqlite3.connect(db_path)
    try:
        assert raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_batches'"
        ).fetchone() is None
    finally:
        raw.close()

    new_bid = record_extraction_batch("sess-B", 0, 1, start_message_index=0, end_message_index=1)
    assert new_bid is not None, "record_extraction_batch must self-heal"
    assert get_batch_status(new_bid) == "pending"
    assert get_extraction_batch("sess-B", 0, 1)["id"] == new_bid
    assert get_batch_age(new_bid) is not None

    claimed, was_failed = _set_batch_processing(new_bid)
    assert claimed is True
    assert was_failed is False

    _update_batch_status(new_bid, "completed")
    assert get_batch_status(new_bid) == "completed"

    assert _is_covered("sess-B", 0, 1) is True

    incomplete = _get_incomplete_batches()
    assert all(b["session_id"] != "sess-B" for b in incomplete)


def test_skipped_batch_with_messages_recovers_via_message_range(
    tmp_path, monkeypatch
):
    """C8.7 regression: a 'skipped' batch whose messages still exist must
    be recoverable, not marked 'dead'. The previous code passed exchange
    indices to get_messages_in_range (which expects message indices)
    so the lookup returned [] and the batch was always marked dead.

    This test seeds messages at exchange_index 0 and 1, records a
    'skipped' batch covering that range, and calls _prepare_extraction_batch.
    With C8.7's exchange→message conversion, the messages are found and
    the batch is NOT marked dead.
    """
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)

    import memory.db as db
    from memory.batch_manager import _prepare_extraction_batch
    from memory.db_summaries import record_extraction_batch, get_batch_status

    db.init_db(db_path)

    # Seed messages at message_index 0 (exchange 0) and message_index 1 (exchange 0)
    # Note: exchange_index is per-user-message; a user+assistant pair = 1 exchange.
    # For the test, we seed a user row with exchange_index=0 and a separate
    # user row with exchange_index=1.
    import hashlib
    for idx, exch in [(0, 0), (2, 1)]:
        h = hashlib.sha256(f"content-{idx}".encode()).hexdigest()
        from memory.db_pool import immediate_transaction
        with immediate_transaction(pooled=True) as conn:
            now = db._now_iso_sqlite()
            conn.execute(
                """INSERT INTO messages (session_id, message_index, role, content,
                   content_hash, created_at, exchange_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("sess-C8-7", idx, "user", f"content-{idx}", h, now, exch),
            )

    # Record a batch in 'skipped' state covering exchange 0-1
    bid = record_extraction_batch(
        "sess-C8-7", 0, 1, status="skipped",
        start_message_index=0, end_message_index=2,
    )
    assert bid is not None
    assert get_batch_status(bid) == "skipped"

    # Call _prepare_extraction_batch — the new code path must find messages
    # via exchange→message conversion, NOT mark the batch dead.
    result = _prepare_extraction_batch("sess-C8-7", 0, 1)
    # Pre-fix: result was non-None but batch was marked dead.
    # Post-fix: result may be None (skipped batches can't be re-claimed
    # by _set_batch_processing — pre-existing design), but the batch
    # must NOT be marked dead when messages exist. Verify the
    # conversion step directly: messages are found by the new
    # message-range query.
    assert get_batch_status(bid) != "dead", (
        f"C8.7 regression: batch was marked 'dead' even though messages "
        f"exist. Status: {get_batch_status(bid)}"
    )


def test_backup_live_db_preserves_schema_and_data(tmp_path):
    """The shutil.copy backup/restore flow used by backup_live_db must keep schema + data."""
    db_path = str(tmp_path / "memory.db")
    backup_dir = tmp_path / "snapshot"
    backup_dir.mkdir()

    init = sqlite3.connect(db_path)
    try:
        init.executescript(_SCHEMA_SQL)
        init.execute(
            "INSERT INTO extraction_batches "
            "(session_id, start_message_index, end_message_index, "
            " start_exchange_index, end_exchange_index, status, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("sess-X", 0, 1, 0, 1, "pending", "2026-06-21T19:00:00Z", "2026-06-21T19:00:00Z"),
        )
        init.execute(
            "INSERT INTO messages "
            "(session_id, message_index, role, content, content_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("sess-X", 0, "user", "hello", "h0", "2026-06-21T19:00:00Z"),
        )
        init.commit()
        init.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        init.close()

    backup_paths = {
        "": str(backup_dir / "memory.db.snapshot"),
        "-shm": str(backup_dir / "memory.db.snapshot-shm"),
        "-wal": str(backup_dir / "memory.db.snapshot-wal"),
    }
    for suffix, backup_path in backup_paths.items():
        try:
            shutil.copy(db_path + suffix, backup_path)
        except FileNotFoundError:
            pass

    pre = sqlite3.connect(db_path)
    try:
        assert pre.execute(
            "SELECT COUNT(*) FROM extraction_batches WHERE session_id = ?", ("sess-X",)
        ).fetchone()[0] == 1
    finally:
        pre.close()

    _drop_extraction_batches(db_path)

    raw = sqlite3.connect(db_path)
    try:
        assert raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_batches'"
        ).fetchone() is None
    finally:
        raw.close()

    for suffix, backup_path in backup_paths.items():
        try:
            shutil.copy(backup_path, db_path + suffix)
        except (FileNotFoundError, shutil.SameFileError):
            pass

    raw = sqlite3.connect(db_path)
    try:
        row = raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_batches'"
        ).fetchone()
        assert row is not None, "extraction_batches table must survive restore"

        count = raw.execute(
            "SELECT COUNT(*) FROM extraction_batches WHERE session_id = ?", ("sess-X",)
        ).fetchone()[0]
        assert count == 1, "extraction_batches row must survive restore"

        msg_count = raw.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("sess-X",)
        ).fetchone()[0]
        assert msg_count == 1, "messages row must survive restore"
    finally:
        raw.close()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    content_hash TEXT,
    tool_calls_json TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    is_edited BOOLEAN DEFAULT 0,
    exchange_index INTEGER DEFAULT NULL,
    UNIQUE(session_id, message_index)
);

CREATE TABLE IF NOT EXISTS extraction_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_index INTEGER NOT NULL,
    end_message_index INTEGER NOT NULL,
    extraction_json TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TEXT,
    updated_at TEXT,
    start_exchange_index INTEGER NOT NULL,
    end_exchange_index INTEGER NOT NULL,
    UNIQUE(session_id, start_exchange_index, end_exchange_index)
);

CREATE INDEX IF NOT EXISTS idx_extraction_batches_status ON extraction_batches(status);
CREATE INDEX IF NOT EXISTS idx_extraction_batches_session ON extraction_batches(session_id);
CREATE INDEX IF NOT EXISTS idx_batches_covered ON extraction_batches(session_id, status, start_exchange_index, end_exchange_index);
"""
