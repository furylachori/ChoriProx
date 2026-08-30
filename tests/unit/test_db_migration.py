"""Tests for V2 chain-hash DB migrations, backfill, and store-time computation.

Covers docs/v2-chain-hash-redesign.md §3 (schema), §5 (backfill), §12 step 4/5/10:
- _migrate_add_chain_hash / _migrate_recreate_conversation_summaries / _migrate_add_indexes
- _backfill_chain_hash (per-session sequential, chunk 333, skip system, BYTE-IDENTITY)
- store_messages_batch / store_message chain_hash computation
- db_summaries model_name parameter (7 functions)
"""

import json
import sqlite3

import pytest

pytestmark = pytest.mark.unit


def _base_schema_sql():
    """Return the base SCHEMA_SQL without importing the full db module chain.

    Post-C3: conversation_summaries is session-scoped (no model_name column,
    session_id is the primary key). raw_start_message_index is part of the
    base schema now (was previously added by the V1→V2 migration).
    """
    return """
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
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        session_id TEXT NOT NULL PRIMARY KEY,
        summary TEXT,
        summary_tokens INTEGER,
        exchange_count INTEGER DEFAULT 0,
        last_exchange_index INTEGER DEFAULT -1,
        last_exchange_start INTEGER,
        last_message_index INTEGER,
        raw_start_message_index INTEGER DEFAULT -1,
        ancient_history TEXT,
        created_at TEXT,
        updated_at TEXT
    );
    """


def _make_raw_conn():
    """Create a raw in-memory SQLite connection with the base (pre-V2) schema.

    Post-C3: the conversation_summaries table is session-scoped (session_id PK,
    no model_name column). The migration _migrate_recreate_conversation_summaries
    is a no-op on this schema (model_name already absent).
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_base_schema_sql())
    return conn


def _make_v1_conn():
    """Create a raw in-memory SQLite connection with the V1 multi-model schema.

    Pre-C3 V1 schema: model_name column present, PK(session_id, model_name).
    The migration should collapse multi-model rows into session-scoped rows.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
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
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            session_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            summary TEXT,
            summary_tokens INTEGER DEFAULT 0,
            exchange_count INTEGER DEFAULT 0,
            last_exchange_index INTEGER DEFAULT -1,
            last_exchange_start INTEGER,
            last_message_index INTEGER,
            raw_start_message_index INTEGER DEFAULT -1,
            ancient_history TEXT,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (session_id, model_name)
        );
    """)
    return conn


def _make_migrated_conn():
    """Create a raw connection with base schema + all V2 migrations applied."""
    from memory.db import (
        _migrate_add_chain_hash,
        _migrate_recreate_conversation_summaries,
        _migrate_add_indexes,
    )
    conn = _make_raw_conn()
    _migrate_add_chain_hash(conn)
    _migrate_recreate_conversation_summaries(conn)
    _migrate_add_indexes(conn)
    return conn


def _insert_message(conn, session_id, message_index, role, content,
                    tool_calls_json=None, tool_call_id=None):
    conn.execute(
        """INSERT INTO messages (session_id, message_index, role, content,
           content_hash, tool_calls_json, tool_call_id, created_at, exchange_index)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, message_index, role, content,
         f"hash_{message_index}", tool_calls_json, tool_call_id,
         "2026-01-01T00:00:00Z", message_index),
    )


def _conv_msgs(role_user_alternating, count, system_prefix=True):
    """Build a list of conversation message dicts."""
    msgs = []
    if system_prefix:
        msgs.append({"role": "system", "content": "sys"})
    for i in range(count):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"})
    return msgs


# ============================================================================
# Migration idempotency
# ============================================================================


class TestMigrationIdempotency:
    def test_add_chain_hash_twice(self):
        from memory.db import _migrate_add_chain_hash
        conn = _make_raw_conn()
        _migrate_add_chain_hash(conn)
        _migrate_add_chain_hash(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        assert "chain_hash" in cols

    def test_recreate_conversation_summaries_twice(self):
        from memory.db import _migrate_recreate_conversation_summaries
        conn = _make_raw_conn()
        _migrate_recreate_conversation_summaries(conn)
        _migrate_recreate_conversation_summaries(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "model_name" not in cols
        assert "raw_start_message_index" in cols

    def test_v1_to_session_scoped_collapse(self):
        """V1 schema (model_name present) → migration collapses to session-scoped PK."""
        from memory.db import _migrate_recreate_conversation_summaries
        conn = _make_v1_conn()
        now = "2026-01-01T00:00:00Z"
        for mn, lei, ah in [("m-A", 40, "ancient-A"), ("m-B", 7, "ancient-B"), ("m-C", 25, "ancient-C")]:
            conn.execute(
                "INSERT INTO conversation_summaries (session_id, model_name, last_exchange_index, "
                "ancient_history, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("s", mn, lei, ah, now, now),
            )
        conn.commit()

        _migrate_recreate_conversation_summaries(conn)

        rows = conn.execute(
            "SELECT session_id, last_exchange_index, ancient_history FROM conversation_summaries "
            "WHERE session_id = 's'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["last_exchange_index"] == 40
        assert rows[0]["ancient_history"] == "ancient-A"

        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "model_name" not in cols

    def test_add_indexes_twice(self):
        from memory.db import _migrate_recreate_conversation_summaries, _migrate_add_indexes
        conn = _make_raw_conn()
        _migrate_recreate_conversation_summaries(conn)
        _migrate_add_indexes(conn)
        _migrate_add_indexes(conn)

    def test_columns_and_indexes_exist_after_migration(self):
        from memory.db import (
            _migrate_add_chain_hash,
            _migrate_recreate_conversation_summaries,
            _migrate_add_indexes,
        )
        conn = _make_raw_conn()
        _migrate_add_chain_hash(conn)
        _migrate_recreate_conversation_summaries(conn)
        _migrate_add_indexes(conn)

        msg_cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        assert "chain_hash" in msg_cols

        cs_cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "model_name" not in cs_cols
        assert "raw_start_message_index" in cs_cols

        indexes = {r[1] for r in conn.execute("PRAGMA index_list(conversation_summaries)").fetchall()}
        assert "idx_conv_summaries_boundary" in indexes

        msg_indexes = {r[1] for r in conn.execute("PRAGMA index_list(messages)").fetchall()}
        assert "idx_messages_session_exchange" in msg_indexes


# ============================================================================
# Backfill
# ============================================================================


class TestBackfill:
    def test_backfill_populates_non_system_messages(self):
        from memory.db import _backfill_chain_hash
        conn = _make_migrated_conn()
        for i in range(5):
            _insert_message(conn, "s1", i, "user" if i % 2 == 0 else "assistant", f"msg-{i}")
        conn.commit()
        _backfill_chain_hash(conn)

        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        hashes = [r[0] for r in rows]
        assert all(h is not None for h in hashes)
        assert len(set(hashes)) == 5

    def test_backfill_system_messages_get_null(self):
        from memory.db import _backfill_chain_hash
        conn = _make_migrated_conn()
        _insert_message(conn, "s1", 0, "system", "sys prompt")
        _insert_message(conn, "s1", 1, "user", "hello")
        _insert_message(conn, "s1", 2, "assistant", "hi")
        conn.commit()
        _backfill_chain_hash(conn)

        rows = conn.execute(
            "SELECT role, chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        assert rows[0][1] is None
        assert rows[1][1] is not None
        assert rows[2][1] is not None

    def test_backfill_correctly_chains(self):
        from memory.db import _backfill_chain_hash
        conn = _make_migrated_conn()
        for i in range(4):
            _insert_message(conn, "s1", i, "user", f"msg-{i}")
        conn.commit()
        _backfill_chain_hash(conn)

        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        h = [r[0] for r in rows]
        assert h[0] != h[1]
        assert h[1] != h[2]

    def test_backfill_handles_system_prefix_gap(self):
        from memory.db import _backfill_chain_hash
        from proxy.chain_hash import _message_hash
        conn = _make_migrated_conn()
        _insert_message(conn, "s1", 0, "system", "sys")
        _insert_message(conn, "s1", 1, "user", "first real")
        conn.commit()
        _backfill_chain_hash(conn)

        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        assert rows[0][0] is None
        expected = _message_hash({"role": "user", "content": "first real", "tool_calls": None})
        assert rows[1][0] == expected

    def test_backfill_chunks_at_333(self):
        from memory.db import _backfill_chain_hash
        conn = _make_migrated_conn()
        count = 500
        for i in range(count):
            _insert_message(conn, "s1", i, "user", f"msg-{i}")
        conn.commit()
        _backfill_chain_hash(conn)

        null_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND chain_hash IS NULL", ("s1",)
        ).fetchone()[0]
        assert null_count == 0

        total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("s1",)
        ).fetchone()[0]
        assert total == count

    def test_backfill_byte_identity_matches_compute_chain(self):
        from memory.db import _backfill_chain_hash
        from proxy.chain_hash import compute_chain
        conn = _make_migrated_conn()

        conv = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "answer", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "search", "arguments": '{"q":"x"}'}}
            ]},
        ]
        for i, m in enumerate(conv):
            tcj = json.dumps(m["tool_calls"]) if m.get("tool_calls") else None
            _insert_message(conn, "s1", i, m["role"], m["content"], tool_calls_json=tcj)
        conn.commit()
        _backfill_chain_hash(conn)

        non_system = [m for m in conv if m["role"] != "system"]
        expected_chain, _ = compute_chain(non_system)

        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? AND role != 'system' ORDER BY message_index",
            ("s1",),
        ).fetchall()
        db_chain = [r[0] for r in rows]
        assert db_chain == expected_chain


# ============================================================================
# store_messages_batch chain_hash
# ============================================================================


class TestStoreMessagesBatchChainHash:
    def test_batch_computes_chain_hash(self):
        from memory.db import store_messages_batch
        conn = _make_migrated_conn()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "q2"},
        ]
        store_messages_batch("s1", msgs, 0, conn=conn)
        conn.commit()

        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        hashes = [r[0] for r in rows]
        assert all(h is not None for h in hashes)
        assert len(set(hashes)) == 3

    def test_batch_system_messages_get_null(self):
        from memory.db import store_messages_batch
        conn = _make_migrated_conn()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        store_messages_batch("s1", msgs, 0, conn=conn)
        conn.commit()

        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        assert rows[0][0] is None
        assert rows[1][0] is not None

    def test_batch_first_non_system_seeds_with_msg_hash(self):
        from memory.db import store_messages_batch
        from proxy.chain_hash import _message_hash
        conn = _make_migrated_conn()
        msgs = [{"role": "user", "content": "seed"}]
        store_messages_batch("s1", msgs, 0, conn=conn)
        conn.commit()

        row = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? AND message_index = 0", ("s1",)
        ).fetchone()
        expected = _message_hash({"role": "user", "content": "seed", "tool_calls": None})
        assert row[0] == expected

    def test_batch_failure_does_not_block_store(self, monkeypatch):
        from memory.db import store_messages_batch

        def _raise(msg):
            raise RuntimeError("boom")

        monkeypatch.setattr("proxy.chain_hash._message_hash", _raise)
        conn = _make_migrated_conn()
        msgs = [{"role": "user", "content": "hello"}]
        result = store_messages_batch("s1", msgs, 0, conn=conn)
        conn.commit()
        assert result == 1

        row = conn.execute(
            "SELECT content, chain_hash FROM messages WHERE session_id = ? AND message_index = 0", ("s1",)
        ).fetchone()
        assert row["content"] == "hello"
        assert row["chain_hash"] is None


# ============================================================================
# store_message chain_hash (single-message path)
# ============================================================================


class TestStoreMessageChainHash:
    def test_store_message_via_pooled_db(self):
        """Use the pooled :memory: DB path (store_message uses immediate_transaction)."""
        import os
        os.environ["PROXY_TEST_DB"] = ":memory:"
        from memory.db import init_db, store_message, get_connection
        init_db()
        store_message("s-sm", 0, "user", content="hello")
        store_message("s-sm", 1, "assistant", content="hi")
        conn = get_connection()
        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s-sm",)
        ).fetchall()
        assert rows[0][0] is not None
        assert rows[1][0] is not None
        assert rows[0][0] != rows[1][0]
        conn.execute("DELETE FROM messages")
        conn.commit()
        os.environ.pop("PROXY_TEST_DB", None)

    def test_store_message_lowercase_role_persistence(self):
        """MAJOR 5: role stored as lowercase so DB chain_hash matches in-memory chain."""
        import os
        os.environ["PROXY_TEST_DB"] = ":memory:"
        from memory.db import init_db, store_message, get_connection
        init_db()
        store_message("s-lc", 0, "User", content="hello")
        conn = get_connection()
        row = conn.execute(
            "SELECT role FROM messages WHERE session_id = ? AND message_index = 0", ("s-lc",)
        ).fetchone()
        assert row["role"] == "user"
        conn.execute("DELETE FROM messages")
        conn.commit()
        os.environ.pop("PROXY_TEST_DB", None)


# ============================================================================
# db_summaries model_name parameter
# ============================================================================


@pytest.fixture
def pooled_db():
    """Pooled :memory: DB with full init_db (runs all migrations)."""
    import os
    os.environ["PROXY_TEST_DB"] = ":memory:"
    from memory.db import init_db, get_connection
    init_db()
    yield
    conn = get_connection()
    conn.execute("DELETE FROM conversation_summaries")
    conn.execute("DELETE FROM messages")
    conn.commit()
    os.environ.pop("PROXY_TEST_DB", None)


class TestDbSummariesSessionScope:
    def test_get_set_last_summarized_exchange_index(self, pooled_db):
        from memory.db_summaries import (
            set_last_summarized_exchange_index,
            get_last_summarized_exchange_index,
            update_exchange_metadata,
        )
        assert get_last_summarized_exchange_index("s1") == -1
        update_exchange_metadata("s1", 0, 0)
        set_last_summarized_exchange_index("s1", 5)
        assert get_last_summarized_exchange_index("s1") == 5
        assert get_last_summarized_exchange_index("s2") == -1

    def test_update_exchange_metadata_session(self, pooled_db):
        from memory.db_summaries import update_exchange_metadata, get_exchange_metadata
        update_exchange_metadata("s1", 3, 10)
        assert get_exchange_metadata("s1") == (3, 10)
        assert get_exchange_metadata("s2") == (0, 0)

        update_exchange_metadata("s1", 5, 20)
        assert get_exchange_metadata("s1") == (5, 20)

    def test_get_ancient_history_by_session(self, pooled_db):
        from memory.db_summaries import update_ancient_history, get_ancient_history
        update_ancient_history("s1", "ancient A")
        update_ancient_history("s2", "ancient B")
        assert get_ancient_history("s1") == "ancient A"
        assert get_ancient_history("s2") == "ancient B"
        assert get_ancient_history("s3") == ""

    def test_two_sessions_independent_state(self, pooled_db):
        from memory.db_summaries import (
            update_ancient_history,
            get_ancient_history,
            update_exchange_metadata,
            get_exchange_metadata,
            get_summary_age,
        )
        update_exchange_metadata("s1", 10, 100)
        update_ancient_history("s1", "old context A")
        update_exchange_metadata("s2", 20, 200)
        update_ancient_history("s2", "old context B")

        assert get_exchange_metadata("s1") == (10, 100)
        assert get_exchange_metadata("s2") == (20, 200)
        assert get_ancient_history("s1") == "old context A"
        assert get_ancient_history("s2") == "old context B"
        assert get_summary_age("s1") is not None
        assert get_summary_age("s3") is None
