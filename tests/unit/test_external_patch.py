"""Unit tests for external-review V2 patches.

Covers CRITICAL #1, CRITICAL #2, HIGH #1, HIGH #2, HIGH #4, HIGH #5 from
the external review brief.
"""
import logging
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _system(content="sys"):
    return {"role": "system", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content=""):
    return {"role": "assistant", "content": content}


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


# ---------------------------------------------------------------------------
# CRITICAL #1: _load_state_from_db uses chain position, not SQL message_index
# ---------------------------------------------------------------------------


class TestLoadStateWithSystemMessage:
    """CRITICAL #1: _load_state_from_db must look up boundary_chain via chain
    position (non-system 0-indexed), not messages.message_index (SQL column
    which includes system at 0). Otherwise conversation[K] is looked up as
    SQL message_index K-1 and returns the wrong message.

    With system@0, user@1, assistant@2 (chain hashes h_user, h_asst), and
    raw_start=1 (conversation position 1 → first non-system is user, boundary
    is right after user → boundary_chain should be h_user).
    """

    def _build_session(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE messages (
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                content_hash TEXT,
                tool_calls_json TEXT,
                tool_call_id TEXT,
                chain_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                is_edited BOOLEAN DEFAULT 0,
                exchange_index INTEGER DEFAULT NULL,
                UNIQUE(session_id, message_index)
            );
            CREATE TABLE conversation_summaries (
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                summary TEXT,
                summary_tokens INTEGER,
                exchange_count INTEGER DEFAULT 0,
                last_exchange_start INTEGER DEFAULT 0,
                last_message_index INTEGER DEFAULT -1,
                raw_start_message_index INTEGER DEFAULT -1,
                ancient_history TEXT,
                last_exchange_index INTEGER DEFAULT -1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, model_name)
            );
        """)
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "chain_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", 0, "system", "sys", None, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "chain_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", 1, "user", "u1", "h_user", "2026-01-01T00:00:01Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "chain_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", 2, "assistant", "a1", "h_asst", "2026-01-01T00:00:02Z"),
        )
        conn.execute(
            "INSERT INTO conversation_summaries "
            "(session_id, model_name, raw_start_message_index, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("s1", "m1", 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        return conn

    def test_boundary_chain_resolves_to_user(self):
        """raw_start=1 (conv position 1, first non-system = user). boundary
        is right after user (raw block = [assistant]). boundary_chain MUST
        be h_user (the LAST summarized message), not h_asst.

        Old buggy code: looked up SQL message_index == 0, but user is at
        SQL message_index 1 (system is at 0). Old code returned None →
        cold path. After fix: returns h_user correctly.
        """
        from proxy.context_v2 import _load_state_from_db

        conn = self._build_session()

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        with patch("memory.db.connection", return_value=cm):
            with patch("memory.db.get_rich_summaries", return_value=[]):
                with patch("memory.db.get_ancient_history", return_value=""):
                    result = _load_state_from_db("s1", "m1")

        assert result is not None, (
            "Restart recovery must succeed when system@0 is present; "
            "old code returned None because it looked up SQL message_index 0 "
            "but user is at SQL message_index 1"
        )
        assert result["boundary_idx"] == 1
        assert result["boundary_chain"] == "h_user", (
            f"boundary_chain must be h_user (last summarized message); got "
            f"{result['boundary_chain']!r}. The fix looks up chain position "
            f"directly, not via SQL message_index."
        )
        assert result["last_chain"] == ["h_user", "h_asst"]

    def test_boundary_chain_resolves_to_assistant_when_raw_starts_at_2(self):
        """raw_start=2 (conv position 2, first non-system after user+asst).
        Boundary is right after assistant (raw block = empty / next message).
        boundary_chain MUST be h_asst, NOT h_user.

        Old buggy code: looked up SQL message_index == 1 (user). Old code
        returned h_user (the WRONG chain entry). After fix: returns h_asst.
        """
        from proxy.context_v2 import _load_state_from_db

        conn = self._build_session()
        conn.execute(
            "UPDATE conversation_summaries SET raw_start_message_index = 2 "
            "WHERE session_id = ? AND model_name = ?",
            ("s1", "m1"),
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
        assert result["boundary_idx"] == 2
        assert result["boundary_chain"] == "h_asst", (
            f"boundary_chain must be h_asst (last summarized message); got "
            f"{result['boundary_chain']!r}. Old buggy code returned h_user "
            f"because it looked up SQL message_index 1 (which is user, not "
            f"the message at conversation position 2-1=1)."
        )


# ---------------------------------------------------------------------------
# CRITICAL #2: DB-load path gates against concurrent populator
# ---------------------------------------------------------------------------


class TestLoadStateDbRace:
    """CRITICAL #2: After _load_state_from_db, the code must NOT clobber an
    in-memory state entry populated by a concurrent thread. The cold-init
    path at lines 967-969 has this gate; the DB-load path did not.

    When existing.get("version", 0) > 0, the existing in-memory state wins
    and the DB-loaded state is discarded.
    """

    def test_db_load_does_not_overwrite_concurrent_population(self):
        import proxy.state as state
        from proxy.context_v2 import _select_context_v2
        from contextlib import ExitStack

        from proxy.chain_hash import compute_chain

        state._chain_state.clear()
        key = ("sess-race", "modelA")

        pad = "X" * 8000
        msgs = [_system("sys"), _user(f"u1 {pad}"), _assistant(f"a1 {pad}")]
        cfg = _budget_config(safe=10_000)
        ctx = _make_ctx("race-test")

        conv = msgs[1:]
        forced_chain, _ = compute_chain(conv)

        concurrent_entry = state.new_chain_state_entry()
        concurrent_entry["last_chain"] = list(forced_chain)
        concurrent_entry["boundary_idx"] = 1
        concurrent_entry["boundary_chain"] = forced_chain[0]
        concurrent_entry["summaries"] = [{"message_index": 1, "role": "user",
                                          "summary": "concurrent", "tool_summary": ""}]
        concurrent_entry["ancient"] = "concurrent ancient"
        concurrent_entry["version"] = 5

        def _populate_during_load(session_id, model_name, ctx=None,
                             model_safe_tokens=None, tokens_per_char=None):
            with state._chain_state_lock:
                state._chain_state[key] = concurrent_entry
            db_entry = state.new_chain_state_entry()
            db_entry["last_chain"] = ["h_db"]
            db_entry["boundary_idx"] = 0
            db_entry["boundary_chain"] = None
            db_entry["summaries"] = []
            db_entry["ancient"] = ""
            db_entry["version"] = 0
            return db_entry

        conn_cm = MagicMock()
        conn_cm.__enter__ = MagicMock(return_value=MagicMock())
        conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_rich_summaries", return_value=[]))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                side_effect=_populate_during_load,
            ))
            _select_context_v2("sess-race", msgs, model_name="modelA", ctx=ctx)

        with state._chain_state_lock:
            surviving = state._chain_state.get(key)
        assert surviving is not None
        assert surviving.get("version", 0) > 0, (
            "In-memory state populated by concurrent thread must NOT be "
            "overwritten by DB-loaded version=0 entry"
        )
        assert surviving["boundary_idx"] == 1, (
            f"Concurrent entry's boundary must survive (idx=1), got "
            f"{surviving['boundary_idx']}. The DB-load path overwrote the "
            f"in-memory state instead of deferring to it."
        )
        assert surviving["summaries"] and surviving["summaries"][0]["summary"] == "concurrent", (
            "Concurrent summaries must survive"
        )

        state._chain_state.clear()


# ---------------------------------------------------------------------------
# HIGH #1: _evict_with_cap converts SQL message_index to conversation position
# ---------------------------------------------------------------------------


class TestEvictWithCapHandlesSystemMessage:
    """HIGH #1: max_evictable_boundary must be a conversation position
    (non-system 0-indexed), not the SQL message_index column (which
    includes system at 0). With system@0, SQL message_index 13 corresponds
    to conversation position 12.

    Old code: max_evictable_boundary = min_mi (SQL) = 13.
    New code: max_evictable_boundary = position (conv) = 12.

    Returning 13 from the old code drops the first non-summarized message
    (the assistant message), which is wrong.
    """

    def _build_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE messages (
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                content_hash TEXT,
                tool_calls_json TEXT,
                tool_call_id TEXT,
                chain_hash TEXT,
                created_at TEXT NOT NULL,
                exchange_index INTEGER DEFAULT NULL,
                UNIQUE(session_id, message_index)
            );
        """)
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "chain_hash, exchange_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s1", 0, "system", "sys", None, 0, "2026-01-01T00:00:00Z"),
        )
        for i in range(1, 20):
            role = "user" if i % 2 == 1 else "assistant"
            conn.execute(
                "INSERT INTO messages (session_id, message_index, role, content, "
                "chain_hash, exchange_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s1", i, role, f"m{i}", f"h{i}", i // 4, "2026-01-01T00:00:00Z"),
            )
        conn.commit()
        return conn

    def test_max_evictable_boundary_uses_conversation_position(self):
        from proxy.context_v2 import _evict_with_cap
        from proxy.chain_hash import compute_chain

        conn = self._build_db()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)

        last_exchange = 2
        min_sql_mi = conn.execute(
            "SELECT MIN(message_index) AS mi FROM messages "
            "WHERE session_id = ? AND exchange_index > ?",
            ("s1", last_exchange),
        ).fetchone()["mi"]
        expected_position = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? "
            "AND message_index < ? AND (role IS NULL OR role != 'system')",
            ("s1", min_sql_mi),
        ).fetchone()[0]

        ctx = _make_ctx()
        new_chain = list(compute_chain([_user(f"u{i}") for i in range(15)]))
        raw_block = [_user(f"raw{i}") for i in range(20)]

        with patch("memory.db.connection", return_value=cm):
            result = _evict_with_cap(
                raw_block=raw_block,
                boundary_idx=0,
                session_id="s1",
                model_name="m1",
                raw_tokens=8_000,
                low_watermark=6_500,
                high_watermark=9_500,
                ctx=ctx,
                new_chain=new_chain,
                tokens_per_char=0.25,
                snap_summaries=[],
                snap_ancient="",
                model_safe_tokens=10_000,
            )

        assert result.boundary_idx <= expected_position, (
            f"max_evictable_boundary must clamp to conversation position "
            f"{expected_position}, not SQL message_index {min_sql_mi}. "
            f"Old code returned {min_sql_mi} which drops the first "
            f"non-summarized message. Got boundary_idx={result.boundary_idx}."
        )
        assert result.skipped_due_to_cap is True, (
            "When proposed boundary exceeds last-summarized exchange, "
            "eviction must be skipped (cap reached) instead of force-evicting"
        )


# ---------------------------------------------------------------------------
# HIGH #2: Cold-init CAS-loser does not persist stale boundary
# ---------------------------------------------------------------------------


class TestColdInitCasLoserDoesNotPersist:
    """HIGH #2: When the cold-init CAS check fails (a concurrent thread
    populated _chain_state with version > 0 during our computation), the
    losing thread must set ctx.v2_boundary_idx = None and ctx.v2_boundary_chain
    = None, NOT persist its cold-computed values.

    Old code: fell through and called ctx.v2_boundary_idx = boundary_idx
    with the losing thread's cold-computed values, overwriting the
    winner's boundary.
    """

    def test_cold_init_loser_does_not_overwrite_ctx_boundary(self):
        import proxy.state as state
        from proxy.context_v2 import _select_context_v2
        from contextlib import ExitStack

        state._chain_state.clear()
        key = ("sess-cold-loser", "modelA")

        winner_entry = state.new_chain_state_entry()
        winner_entry["last_chain"] = ["h_winner"]
        winner_entry["boundary_idx"] = 77
        winner_entry["boundary_chain"] = "h_winner_77"
        winner_entry["summaries"] = [{"message_index": 1, "role": "user",
                                      "summary": "winner", "tool_summary": ""}]
        winner_entry["ancient"] = "winner ancient"
        winner_entry["version"] = 1

        def _populate_winner_during_cold_compute(*a, **kw):
            with state._chain_state_lock:
                state._chain_state[key] = winner_entry
            return []

        pad = "X" * 8000
        msgs = [_system("sys"), _user(f"u1 {pad}"), _assistant(f"a1 {pad}")]
        cfg = _budget_config(safe=10_000)
        ctx = _make_ctx("cold-loser")

        conn_cm = MagicMock()
        conn_cm.__enter__ = MagicMock(return_value=MagicMock())
        conn_cm.__exit__ = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch("proxy.context.get_context_config", return_value=cfg))
            stack.enter_context(patch("memory.db.get_ancient_history", return_value=""))
            stack.enter_context(patch("proxy.vfs._build_vfs_context", return_value=None))
            stack.enter_context(patch("proxy.vfs._build_vfs_inherited_context", return_value=None))
            stack.enter_context(patch("memory.db.connection", return_value=conn_cm))
            stack.enter_context(patch(
                "memory.db_summaries.get_last_summarized_exchange_index",
                return_value=-1,
            ))
            stack.enter_context(patch("memory.sync.submit_ancient_compress", return_value=None))
            stack.enter_context(patch(
                "proxy.context_v2._load_state_from_db",
                return_value=None,
            ))
            stack.enter_context(patch(
                "memory.db.get_rich_summaries",
                side_effect=_populate_winner_during_cold_compute,
            ))

            _select_context_v2("sess-cold-loser", msgs, model_name="modelA", ctx=ctx)

        assert ctx.v2_boundary_idx is None, (
            f"Cold-init CAS-loser must NOT persist its boundary to ctx; "
            f"the winner's boundary must win. Got "
            f"ctx.v2_boundary_idx={ctx.v2_boundary_idx}."
        )
        assert ctx.v2_boundary_chain is None, (
            f"Cold-init CAS-loser must NOT persist its boundary chain to ctx. "
            f"Got ctx.v2_boundary_chain={ctx.v2_boundary_chain}."
        )

        with state._chain_state_lock:
            surviving = state._chain_state.get(key)
        assert surviving is not None
        assert surviving["boundary_idx"] == 77, (
            f"Winner's boundary (77) must remain. Got {surviving['boundary_idx']}"
        )
        assert surviving["boundary_chain"] == "h_winner_77"

        state._chain_state.clear()


# ---------------------------------------------------------------------------
# HIGH #4: PROXY_V2_CHAIN_STATE_MAX_ENTRIES env validation
# ---------------------------------------------------------------------------


class TestChainStateMaxEntriesEnvValidation:
    """HIGH #4: invalid PROXY_V2_CHAIN_STATE_MAX_ENTRIES (typos like 'banana')
    must NOT crash module import. Must fall back to safe default (100)."""

    def test_banana_falls_back_to_100(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={**os.environ,
                 "PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "banana",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "100" in result.stdout, (
            f"Invalid PROXY_V2_CHAIN_STATE_MAX_ENTRIES=banana must fall back to "
            f"default 100, got stdout={result.stdout!r}"
        )

    def test_empty_falls_back_to_100(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={**os.environ,
                 "PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "100" in result.stdout

    def test_zero_clamps_to_1(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={**os.environ,
                 "PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "0",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "1" in result.stdout

    def test_negative_clamps_to_1(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={**os.environ,
                 "PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "-5",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "1" in result.stdout

    def test_valid_value_used(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.state; "
             "print(proxy.state.CHAIN_STATE_MAX_ENTRIES)"],
            env={**os.environ,
                 "PROXY_V2_CHAIN_STATE_MAX_ENTRIES": "250",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "250" in result.stdout


# ---------------------------------------------------------------------------
# HIGH #5: per-row try/except in _backfill_one_session
# ---------------------------------------------------------------------------


class TestBackfillCorruptRowDoesNotAbortSession:
    """HIGH #5: A single corrupted row (bad json in tool_calls_json) inside a
    session must NOT abort the entire session's backfill. The fix wraps the
    per-row body in try/except and sets that row's chain_hash to NULL while
    continuing to process subsequent rows."""

    def _make_migrated_conn(self):
        from memory.db import (
            _migrate_add_chain_hash,
            _migrate_recreate_conversation_summaries,
            _migrate_add_indexes,
        )
        from tests.unit.test_v2_round4_patch import _make_raw_conn
        conn = _make_raw_conn()
        _migrate_add_chain_hash(conn)
        _migrate_recreate_conversation_summaries(conn)
        _migrate_add_indexes(conn)
        return conn

    def test_single_corrupt_row_continues_session(self):
        from memory.db import _backfill_one_session

        conn = self._make_migrated_conn()
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 0, "user", "first", "h0", None, None, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 1, "assistant", "corrupt", "h1",
             "not valid json {{{", None, "2026-01-01T00:00:01Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 2, "user", "third", "h2", None, None, "2026-01-01T00:00:02Z"),
        )
        conn.commit()

        count = _backfill_one_session(conn, "s1")

        assert count == 3, "All 3 rows must be processed (corrupt row does not abort)"
        rows = conn.execute(
            "SELECT message_index, chain_hash FROM messages WHERE session_id = ? "
            "ORDER BY message_index",
            ("s1",),
        ).fetchall()
        hashes_by_idx = {r["message_index"]: r["chain_hash"] for r in rows}
        assert hashes_by_idx[0] is not None, "First row must be backfilled"
        assert hashes_by_idx[1] is None, (
            "Corrupt row must have chain_hash=NULL (not the chain hash of "
            "garbage), so downstream chain validation can detect the gap"
        )
        assert hashes_by_idx[2] is not None, (
            "Third row (after corrupt row) must be backfilled; the fix "
            "allows the session backfill to continue past the corrupt row"
        )

    def test_session_after_corrupt_still_chains(self):
        """Sanity: after the fix, the chain resumes from prev_chain computed
        before the corrupt row (so third row's chain_hash depends on first
        row's chain_hash, not None)."""
        from memory.db import _backfill_one_session

        conn = self._make_migrated_conn()
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 0, "user", "first", "h0", None, None, "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 1, "assistant", "corrupt", "h1",
             "broken", None, "2026-01-01T00:00:01Z"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "content_hash, tool_calls_json, tool_call_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", 2, "user", "third", "h2", None, None, "2026-01-01T00:00:02Z"),
        )
        conn.commit()

        _backfill_one_session(conn, "s1")
        third_hash = conn.execute(
            "SELECT chain_hash FROM messages WHERE session_id = ? AND message_index = 2",
            ("s1",),
        ).fetchone()["chain_hash"]

        assert third_hash is not None
        assert len(third_hash) == 16, (
            f"Third row's chain_hash must be a 16-char hex (computed from "
            f"prev_chain of row 0, since row 1 was corrupt), got {third_hash!r}"
        )


# ---------------------------------------------------------------------------
# MEDIUM #4: last_user role guard
# ---------------------------------------------------------------------------


class TestLastUserRoleGuard:
    """MEDIUM #4: warn only for retained assistant tails with no live user.

    The current live-exchange contract starts at the most recent retained user
    and includes every later assistant/tool message. An assistant-ending block
    warns only when the retained block has no user-led live exchange.
    """

    def _run_with_assistant_end(self, caplog, *, boundary_idx):
        cfg = _budget_config(safe=300)
        padding = "X" * 200
        msgs = [
            _system("sys"),
            _user(f"u1 {padding}"),
            _assistant(f"a1 {padding}"),
            _user(f"u2 {padding}"),
            _assistant(f"a2 {padding}"),
        ]
        conversation = msgs[1:]

        from proxy.context_v2 import _select_context_v2
        from contextlib import ExitStack

        ctx = _make_ctx("last-user-guard")
        conn_cm = MagicMock()
        conn_cm.__enter__ = MagicMock(return_value=MagicMock())
        conn_cm.__exit__ = MagicMock(return_value=False)

        from proxy.chain_hash import compute_chain
        full_chain, _chain_to_position = compute_chain(conversation)
        forced_chain = list(full_chain)
        boundary_chain = (
            forced_chain[boundary_idx - 1]
            if boundary_idx > 0
            else None
        )
        db_entry = {
            "last_chain": forced_chain,
            "boundary_idx": boundary_idx,
            "boundary_chain": boundary_chain,
            "summaries": [],
            "ancient": "",
            "version": 0,
        }

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "proxy.context.get_context_config",
                    return_value=cfg,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_rich_summaries",
                    return_value=[],
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.get_ancient_history",
                    return_value="",
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_context",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "proxy.vfs._build_vfs_inherited_context",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db.connection",
                    return_value=conn_cm,
                )
            )
            stack.enter_context(
                patch(
                    "memory.db_summaries."
                    "get_last_summarized_exchange_index",
                    return_value=-1,
                )
            )
            stack.enter_context(
                patch(
                    "memory.sync.submit_ancient_compress",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "proxy.context_v2._load_state_from_db",
                    return_value=db_entry,
                )
            )
            result = _select_context_v2(
                "sess-guard",
                msgs,
                model_name="m1",
                ctx=ctx,
            )

        return result

    def test_warning_logged_when_raw_block_ends_with_assistant(self, caplog):
        """A retained assistant-only tail logs the no-user warning."""
        import logging

        with caplog.at_level(
            logging.WARNING,
            logger="proxy.context_v2",
        ):
            result = self._run_with_assistant_end(
                caplog,
                boundary_idx=3,
            )

        warning_records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and "ends with assistant" in record.message
        ]
        assert len(warning_records) == 1, (
            "retained assistant-only tail must log exactly one warning"
        )
        assert [message.get("role") for message in result] == [
            "system",
            "assistant",
        ]
        assert result[-1]["content"] == f"a2 {'X' * 200}"

    def test_messages_preserved_when_raw_block_ends_with_assistant(
        self,
        caplog,
    ):
        """A user-led live suffix preserves messages without a no-user warning."""
        import logging

        with caplog.at_level(
            logging.WARNING,
            logger="proxy.context_v2",
        ):
            result = self._run_with_assistant_end(
                caplog,
                boundary_idx=0,
            )

        warning_records = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and "ends with assistant" in record.message
        ]
        assert warning_records == []

        roles = [message.get("role") for message in result]
        assert roles == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert result[1]["content"] == f"u1 {'X' * 200}"
        assert result[2]["content"] == f"a1 {'X' * 200}"
        assert result[3]["content"] == f"u2 {'X' * 200}"
        assert result[4]["content"] == f"a2 {'X' * 200}"
