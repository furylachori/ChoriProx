"""Unit tests for V2 round-4 patch: boundary persistence, queue bounds, dead code removal.

Covers:
- test_post_process_boundary_persists: full V2 → post_process → DB write flow
- test_migration_preserves_created_at_column: created_at NOT dropped by migration
- test_ancient_compress_queue_full_returns_none: bounded queue cap
- test_chain_to_position_not_in_state: dead field removed from canonical state
- test_chain_to_position_not_returned_from_load: dead field removed from _load_state_from_db
- test_schemasql_matches_post_migration: SCHEMA_SQL reflects post-migration schema
- test_chained_state_max_entries_env_override: env var override
- test_ancient_capped_at_max_chars: HIGH #3 — 100K char cap
- test_subagent_log_message_updated: HIGH #9 — log text
"""
import logging
import os
import sqlite3
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Migration: created_at/updated_at preservation
# ---------------------------------------------------------------------------


def _base_schema_sql():
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
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_base_schema_sql())
    return conn


class TestMigrationPreservesCreatedAt:
    """CRITICAL #1: post_process.py writes (created_at, updated_at) on every
    boundary INSERT. Migration must NOT drop these columns."""

    def test_migration_keeps_created_at_column(self):
        from memory.db import _migrate_recreate_conversation_summaries
        conn = _make_raw_conn()
        _migrate_recreate_conversation_summaries(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "created_at" in cols, "created_at must be preserved (post_process.py depends on it)"
        assert "updated_at" in cols, "updated_at must be preserved (post_process.py depends on it)"

    def test_migration_idempotent_with_created_at(self):
        from memory.db import _migrate_recreate_conversation_summaries
        conn = _make_raw_conn()
        _migrate_recreate_conversation_summaries(conn)
        _migrate_recreate_conversation_summaries(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "created_at" in cols
        assert "updated_at" in cols
        assert "model_name" not in cols

    def test_post_process_insert_against_migrated_schema(self):
        """Full chain: pre-V2 schema → migrate → INSERT raw_start_message_index
        with created_at/updated_at (mirrors post_process.py)."""
        from memory.db import _migrate_recreate_conversation_summaries
        conn = _make_raw_conn()
        _migrate_recreate_conversation_summaries(conn)
        now = "2026-06-21T00:00:00Z"
        conn.execute(
            """INSERT INTO conversation_summaries
               (session_id, raw_start_message_index, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
               raw_start_message_index = excluded.raw_start_message_index,
               updated_at = excluded.updated_at""",
            ("s1", 42, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT raw_start_message_index, created_at, updated_at FROM conversation_summaries "
            "WHERE session_id = ?",
            ("s1",),
        ).fetchone()
        assert row["raw_start_message_index"] == 42
        assert row["created_at"] == now
        assert row["updated_at"] == now

    def test_post_process_exchange_metadata_insert_against_migrated_schema(self):
        """Mirrors post_process.py exchange metadata INSERT (session-scoped)."""
        from memory.db import _migrate_recreate_conversation_summaries
        conn = _make_raw_conn()
        _migrate_recreate_conversation_summaries(conn)
        now = "2026-06-21T00:00:00Z"
        conn.execute(
            """INSERT INTO conversation_summaries
               (session_id, exchange_count, last_exchange_start,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
               exchange_count = excluded.exchange_count,
               last_exchange_start = excluded.last_exchange_start,
               updated_at = excluded.updated_at""",
            ("s1", 5, 10, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT exchange_count, last_exchange_start FROM conversation_summaries "
            "WHERE session_id = ?",
            ("s1",),
        ).fetchone()
        assert row["exchange_count"] == 5
        assert row["last_exchange_start"] == 10


class TestSchemaSqlMatchesPostMigration:
    """CRITICAL #2/#7: SCHEMA_SQL must reflect the post-migration reality."""

    def test_schemasql_conversation_summaries_columns(self):
        from memory.db import SCHEMA_SQL
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "session_id" in cols
        assert "model_name" not in cols
        assert "raw_start_message_index" in cols
        assert "ancient_history" in cols
        assert "created_at" in cols
        assert "updated_at" in cols


# ---------------------------------------------------------------------------
# State: chain_to_position removed (CRITICAL #5)
# ---------------------------------------------------------------------------


class TestChainToPositionRemoved:
    """CRITICAL #5: chain_to_position is dead memory; 53MB worst-case waste."""

    def test_new_chain_state_entry_omits_chain_to_position(self):
        import proxy.state as state
        entry = state.new_chain_state_entry()
        assert "chain_to_position" not in entry

    def test_load_state_from_db_omits_chain_to_position(self):
        """Direct test against _load_state_from_db mock connection."""
        from proxy.context_v2 import _load_state_from_db
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE messages (
                session_id TEXT, message_index INTEGER, role TEXT, content TEXT,
                tool_calls_json TEXT, tool_call_id TEXT, chain_hash TEXT
            );
            CREATE TABLE conversation_summaries (
                session_id TEXT NOT NULL, model_name TEXT NOT NULL,
                raw_start_message_index INTEGER DEFAULT -1,
                PRIMARY KEY (session_id, model_name)
            );
        """)
        for i in range(3):
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s1", i, "user", f"m{i}", None, None, f"h{i}"),
            )
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
            ("s1", "m1", 2),
        )
        conn.commit()

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        with patch("memory.db.connection", return_value=cm):
            with patch("memory.db.get_rich_summaries", return_value=[]):
                with patch("memory.db.get_ancient_history", return_value=""):
                    result = _load_state_from_db("s1", "m1")
        assert result is not None
        assert "chain_to_position" not in result
        assert result["boundary_idx"] == 2
        assert result["last_chain"] == ["h0", "h1", "h2"]
        assert result["boundary_chain"] == "h1"


# ---------------------------------------------------------------------------
# Ancient compress queue bound (CRITICAL #6)
# ---------------------------------------------------------------------------


class TestAncientCompressQueueBound:
    """CRITICAL #6: queue was unbounded (2GB+ observed). Now capped at 100."""

    def test_queue_full_returns_none(self, monkeypatch):
        import memory.sync as sync
        sync._ancient_compress_pending = sync._ANCIENT_COMPRESS_MAX_PENDING
        result = sync.submit_ancient_compress("s1", [{"summary": "x"}])
        assert result is None, "Queue full must return None (not dispatch)"
        sync._ancient_compress_pending = 0

    def test_queue_below_cap_dispatches(self):
        import memory.sync as sync
        sync._ancient_compress_pending = 0
        future = sync.submit_ancient_compress("s1", [{"summary": "x"}])
        try:
            assert future is not None
        finally:
            future.result(timeout=10)
        assert sync._ancient_compress_pending == 0

    def test_empty_payload_returns_none_without_inc(self):
        import memory.sync as sync
        sync._ancient_compress_pending = 0
        result = sync.submit_ancient_compress("s1", [])
        assert result is None
        assert sync._ancient_compress_pending == 0

    def test_pending_counter_decrements_after_worker(self):
        import memory.sync as sync
        sync._ancient_compress_pending = 0
        future = sync.submit_ancient_compress("s1", [{"summary": "x"}])
        future.result(timeout=10)
        assert sync._ancient_compress_pending == 0


# ---------------------------------------------------------------------------
# V2 ancient history cap (HIGH #3)
# ---------------------------------------------------------------------------


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content=""):
    return {"role": "assistant", "content": content}


def _system(content):
    return {"role": "system", "content": content}


def _budget_config(safe=10_000):
    return {
        "max_context_tokens": 40_000,
        "tokens_per_char": 0.25,
        "safe_context_tokens": safe,
        "raw_exchange_count": 100,
    }


def _make_ctx(request_id="test-req"):
    return SimpleNamespace(
        request_id=request_id,
        v2_boundary_idx=None,
        v2_boundary_chain=None,
    )


def _mock_conn(min_mi=None, raw_start=None, chain_rows=None):
    def execute_side_effect(query, *args, **kwargs):
        q = str(query).upper()
        result = MagicMock()
        if "MIN(MESSAGE_INDEX)" in q:
            result.fetchone.return_value = {"mi": min_mi}
        elif "RAW_START_MESSAGE_INDEX" in q:
            if raw_start is not None:
                result.fetchone.return_value = {"raw_start_message_index": raw_start}
            else:
                result.fetchone.return_value = None
        elif "CHAIN_HASH" in q:
            result.fetchall.return_value = chain_rows or []
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


class TestAncientCap:
    """D5: ancient cap is per-model (V2_ANCIENT_MAX_RATIO × model_safe_tokens / tokens_per_char)."""

    def test_ancient_capped_per_model(self):
        from proxy.context_v2 import _evict_with_cap, _ancient_max_chars, V2_ANCIENT_MAX_RATIO
        big_ancient = "X" * 200_000
        new_chain = ["h" + str(i) for i in range(50)]
        ctx = _make_ctx()
        cfg = _budget_config(safe=10_000)
        result = _evict_with_cap(
            raw_block=[],
            boundary_idx=0,
            session_id="s1",
            model_name="m1",
            raw_tokens=10_000,
            low_watermark=6500,
            high_watermark=9500,
            ctx=ctx,
            new_chain=new_chain,
            tokens_per_char=0.25,
            snap_summaries=[],
            snap_ancient=big_ancient,
            model_safe_tokens=10_000,
        )
        cap = _ancient_max_chars(10_000, 0.25)
        assert len(result.ancient) <= cap + 200, (
            f"Ancient must be capped at per-model cap {cap}, got len={len(result.ancient)}"
        )
        assert "[...truncated" in result.ancient

    def test_ancient_under_cap_passes_through(self):
        from proxy.context_v2 import _evict_with_cap, _ancient_max_chars
        cap = _ancient_max_chars(model_safe_tokens=10_000, tokens_per_char=0.25)
        small_ancient = "X" * max(1, cap // 2)
        new_chain = ["h" + str(i) for i in range(50)]
        ctx = _make_ctx()
        result = _evict_with_cap(
            raw_block=[],
            boundary_idx=0,
            session_id="s1",
            model_name="m1",
            raw_tokens=10_000,
            low_watermark=6500,
            high_watermark=9500,
            ctx=ctx,
            new_chain=new_chain,
            tokens_per_char=0.25,
            snap_summaries=[],
            snap_ancient=small_ancient,
            model_safe_tokens=10_000,
        )
        assert result.ancient == small_ancient, (
            f"ancient under per-model cap should pass through; "
            f"small_len={len(small_ancient)}, cap={cap}, got_len={len(result.ancient)}"
        )


# ---------------------------------------------------------------------------
# State config (HIGH #7)
# ---------------------------------------------------------------------------


class TestChainStateMaxEntriesEnvOverride:
    """HIGH #7: CHAIN_STATE_MAX_ENTRIES must be env-configurable.

    Verifies the env var name `PROXY_V2_CHAIN_STATE_MAX_ENTRIES` is read at
    module load time. Uses subprocess to test the env var override since
    reloading proxy.state in-process breaks test isolation.
    """

    def test_default_is_100(self):
        import proxy.state as state
        assert state.CHAIN_STATE_MAX_ENTRIES == 100

    def test_env_override_via_subprocess(self):
        """Run a subprocess with the env var set, verify the value propagates."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={"PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "42", "PATH": os.environ.get("PATH", "")},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "42" in result.stdout

    def test_zero_clamped_via_subprocess(self):
        """PROXY_V2_CHAIN_STATE_MAX_ENTRIES=0 clamps to 1."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={"PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "0", "PATH": os.environ.get("PATH", "")},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "1" in result.stdout


# ---------------------------------------------------------------------------
# Backfill isolation (CRITICAL #4)
# ---------------------------------------------------------------------------


class TestBackfillPerSessionIsolation:
    """CRITICAL #4: a corrupted session must not abort the entire backfill."""

    def _make_migrated_conn(self):
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

    def test_backfill_continues_after_corrupt_session(self):
        """Corrupted tool_calls_json in s1 must NOT stop backfill for s2."""
        from memory.db import _backfill_chain_hash

        conn = self._make_migrated_conn()
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 0, "user", "hello", "h0", "not valid json {{{", None, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s2", 0, "user", "goodbye", "h1", None, None, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

        _backfill_chain_hash(conn)

        s1 = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ?", ("s1",)
        ).fetchone()
        s2 = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ?", ("s2",)
        ).fetchone()
        assert s1[0] is None, "Corrupted s1 must remain unset (not crash)"
        assert s2[0] is not None, "Good s2 must be backfilled even after s1 fails"

    def test_backfill_one_session_isolated(self):
        """_backfill_one_session works on a clean session."""
        from memory.db import _backfill_one_session

        conn = self._make_migrated_conn()
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 0, "user", "a", "h0", None, None, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 1, "assistant", "b", "h1", None, None, "2026-01-01T00:00:00Z"),
        )
        conn.commit()

        count = _backfill_one_session(conn, "s1")
        assert count == 2
        rows = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? ORDER BY message_index", ("s1",)
        ).fetchall()
        assert all(r[0] is not None for r in rows)
