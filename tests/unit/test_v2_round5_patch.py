"""Unit tests for V2 round-5 patch.

Covers the four HIGH issues from round 5 review:

1. test_chain_query_includes_prefix_for_boundary — _load_state_from_db must
   fetch the prefix (message_index < raw_start) when raw_start sits inside
   the prefix zone. The old code's DESC LIMIT 10000 truncated chain lost
   the prefix for sessions > 10000 messages, causing boundary_chain to
   reference the WRONG chain entry (positional index).

2. test_env_var_validation_falls_back_to_default — invalid V2_* env vars
   (typos like "banana") must NOT crash import; they must fall back to
   the safe defaults. clamp_min/clamp_max also enforced.

3. test_ancient_compress_counter_recovers_from_submit_failure — when
   _exec_submit_sync raises (pool shutdown), the _ancient_compress_pending
   counter must be decremented. Without this, counter leaks up to
   _ANCIENT_COMPRESS_MAX_PENDING and all future submits return None.

4. test_post_process_boundary_persists_independent_of_exchange_count —
   the V2 boundary INSERT must run even when the exchange_count INSERT
   raises. The old code wrapped both INSERTs in one try block, so a
   failure in the first skipped the second.
"""
import importlib
import logging
import os
import sqlite3
import subprocess
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


pytestmark = pytest.mark.unit


def _user(content):
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# HIGH 1: chain query includes prefix for boundary lookup
# ---------------------------------------------------------------------------


class TestChainQueryPrefixForBoundary:
    """HIGH 1: _load_state_from_db must fetch prefix rows when raw_start
    sits inside the prefix zone (small raw_start, recent tail dominated by
    later messages). The previous DESC LIMIT N query dropped the prefix
    for sessions > N messages, causing boundary_chain to point at the
    wrong position.

    This test simulates a session with 10001 messages where raw_start=42,
    ensuring the prefix (indices 0..41) is loaded so the boundary can be
    located correctly.
    """

    def _build_session(self, total_messages, raw_start):
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                UNIQUE(session_id, model_name)
            );
        """)
        for i in range(total_messages):
            conn.execute(
                "INSERT INTO messages "
                "(session_id, message_index, role, content, chain_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("s1", i, "user", f"m{i}", f"h{i}", "2026-01-01T00:00:00Z"),
            )
        conn.execute(
            "INSERT INTO conversation_summaries "
            "(session_id, model_name, raw_start_message_index, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s1", "m1", raw_start, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        return conn

    def test_prefix_loaded_for_boundary_in_prefix_zone(self):
        """Session has 10001 messages. raw_start=42 (inside prefix). The
        DESC LIMIT 10000 query alone returns messages 1..10000 reversed,
        missing index 0 entirely. boundary_chain must still resolve to
        h[41] (message_index 41 = raw_start - 1).
        """
        from proxy.context_v2 import _load_state_from_db

        conn = self._build_session(total_messages=10001, raw_start=42)

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        with patch("memory.db.connection", return_value=cm):
            with patch("memory.db.get_rich_summaries", return_value=[]):
                with patch("memory.db.get_ancient_history", return_value=""):
                    result = _load_state_from_db("s1", "m1")

        assert result is not None, (
            "Restart recovery must succeed when raw_start is inside the prefix; "
            "old code returned None due to boundary_chain being looked up by "
            "positional index into a truncated list"
        )
        assert result["boundary_idx"] == 42
        assert result["boundary_chain"] == "h41"

    def test_recent_tail_dominates_chain(self):
        """Session has 200 messages, raw_start=180. boundary_chain must
        resolve to h[179] even though 180 is far past the prefix zone
        but still loaded by the recent tail."""
        from proxy.context_v2 import _load_state_from_db

        conn = self._build_session(total_messages=200, raw_start=180)

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        with patch("memory.db.connection", return_value=cm):
            with patch("memory.db.get_rich_summaries", return_value=[]):
                with patch("memory.db.get_ancient_history", return_value=""):
                    result = _load_state_from_db("s1", "m1")

        assert result is not None
        assert result["boundary_idx"] == 180
        assert result["boundary_chain"] == "h179"

    def test_boundary_at_zero(self):
        """raw_start=0 means no boundary chain (start of conversation).
        boundary_chain must be None."""
        from proxy.context_v2 import _load_state_from_db

        conn = self._build_session(total_messages=10, raw_start=0)

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=conn)
        cm.__exit__ = MagicMock(return_value=False)
        with patch("memory.db.connection", return_value=cm):
            with patch("memory.db.get_rich_summaries", return_value=[]):
                with patch("memory.db.get_ancient_history", return_value=""):
                    result = _load_state_from_db("s1", "m1")

        assert result is not None
        assert result["boundary_idx"] == 0
        assert result["boundary_chain"] is None


# ---------------------------------------------------------------------------
# HIGH 2: V2 env var validation falls back to defaults
# ---------------------------------------------------------------------------


class TestEnvVarValidation:
    """HIGH 2: invalid V2_* env vars (typos like 'banana') must NOT crash
    module import. They must fall back to safe defaults and log a warning.
    clamp_min/clamp_max also enforced.
    """

    def test_invalid_int_falls_back(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.context_v2 as m; "
             "print(m.V2_ANCIENT_MAX_RATIO)"],
            env={**os.environ,
                 "V2_ANCIENT_MAX_RATIO": "banana",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "0.1" in result.stdout, (
            f"Invalid V2_ANCIENT_MAX_RATIO=banana must fall back to default 0.1, "
            f"got stdout={result.stdout!r}"
        )

    def test_invalid_float_falls_back(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.context_v2 as m; "
             "print(m.V2_LOW_WATERMARK_RATIO)"],
            env={**os.environ,
                 "V2_LOW_WATERMARK_RATIO": "not-a-number",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "0.65" in result.stdout

    def test_empty_string_falls_back(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.context_v2 as m; "
             "print(m.V2_CHAIN_QUERY_LIMIT)"],
            env={**os.environ,
                 "V2_CHAIN_QUERY_LIMIT": "",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "10000" in result.stdout

    def test_clamp_min_enforced(self):
        """V2_CHAIN_QUERY_LIMIT below clamp_min (100) must fall back to default."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.context_v2 as m; "
             "print(m.V2_CHAIN_QUERY_LIMIT)"],
            env={**os.environ,
                 "V2_CHAIN_QUERY_LIMIT": "10",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "10000" in result.stdout, (
            f"V2_CHAIN_QUERY_LIMIT=10 must be clamped; got {result.stdout!r}"
        )

    def test_clamp_max_enforced_for_ratios(self):
        """Ratio > 1.0 is nonsensical; must fall back to default."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.context_v2 as m; "
             "print(m.V2_LOW_WATERMARK_RATIO)"],
            env={**os.environ,
                 "V2_LOW_WATERMARK_RATIO": "5.0",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "0.65" in result.stdout

    def test_valid_value_used(self):
        """Sanity check: a valid value IS used (not always the default)."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src'); "
             "import proxy.context_v2 as m; "
             "print(m.V2_ANCIENT_MAX_RATIO)"],
            env={**os.environ,
                 "V2_ANCIENT_MAX_RATIO": "0.25",
                 "PYTHONPATH": "src"},
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr}"
        assert "0.25" in result.stdout


# ---------------------------------------------------------------------------
# HIGH 3: _ancient_compress_pending counter recovers from submit failure
# ---------------------------------------------------------------------------


class TestAncientCompressCounterRecovery:
    """HIGH 3: when _exec_submit_sync raises (pool shutdown, exhaustion),
    _ancient_compress_pending must be decremented. Without this, the
    counter leaks until it hits _ANCIENT_COMPRESS_MAX_PENDING and all
    future submits return None.
    """

    def test_counter_decrements_on_submit_runtime_error(self, monkeypatch):
        import memory.sync as sync

        sync._ancient_compress_pending = 0

        def _raise(*a, **kw):
            raise RuntimeError("pool shutdown")

        monkeypatch.setattr(sync, "_exec_submit_summary", _raise)

        result = sync.submit_ancient_compress("s1", [{"summary": "x"}])
        assert result is None, "submit must return None when pool raises"
        assert sync._ancient_compress_pending == 0, (
            f"Counter must be decremented on submit failure, got "
            f"{sync._ancient_compress_pending}"
        )

        sync._ancient_compress_pending = 0

    def test_counter_decrements_on_submit_value_error(self, monkeypatch):
        """Other Exception subclasses (e.g. ValueError) must also decrement."""
        import memory.sync as sync

        sync._ancient_compress_pending = 0

        def _raise(*a, **kw):
            raise ValueError("bad arg")

        monkeypatch.setattr(sync, "_exec_submit_summary", _raise)

        result = sync.submit_ancient_compress("s1", [{"summary": "x"}])
        assert result is None
        assert sync._ancient_compress_pending == 0

        sync._ancient_compress_pending = 0

    def test_counter_decrements_after_worker_completes(self):
        """Sanity: normal happy-path still decrements after worker runs."""
        import memory.sync as sync

        sync._ancient_compress_pending = 0
        future = sync.submit_ancient_compress("s1", [{"summary": "x"}])
        future.result(timeout=10)
        assert sync._ancient_compress_pending == 0

    def test_repeated_failures_do_not_leak(self, monkeypatch):
        """50 consecutive submit failures must NOT push counter above 0."""
        import memory.sync as sync

        sync._ancient_compress_pending = 0

        def _raise(*a, **kw):
            raise RuntimeError("pool shutdown")

        monkeypatch.setattr(sync, "_exec_submit_summary", _raise)

        for _ in range(50):
            result = sync.submit_ancient_compress("s1", [{"summary": "x"}])
            assert result is None

        assert sync._ancient_compress_pending == 0, (
            f"After 50 submit failures, counter must remain 0 (no leak), "
            f"got {sync._ancient_compress_pending}"
        )

        sync._ancient_compress_pending = 0


# ---------------------------------------------------------------------------
# HIGH 4: V2 boundary persists independent of exchange_count failure
# ---------------------------------------------------------------------------


class TestBoundaryIndependentOfExchangeCount:
    """HIGH 4: the V2 boundary INSERT (raw_start_message_index) must run
    even when the exchange_count INSERT fails. The old code wrapped both
    INSERTs in a single try block, so a failure in the first skipped the
    second — silent data loss for restart recovery.
    """

    def test_boundary_persists_when_exchange_count_fails(self, monkeypatch):
        """Patch the exchange_count INSERT to raise, and verify the V2
        boundary INSERT still runs."""
        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

        from memory.db import init_db, get_connection, immediate_transaction
        import contextlib

        init_db()

        ctx = SimpleNamespace(
            session_id="s1",
            model_name="m1",
            body={"model": "m1", "messages": [_user("u1"), _user("u2")]},
            messages_original=None,
            v2_boundary_idx=77,
            v2_boundary_chain="boundary77",
            model_route=SimpleNamespace(
                is_remote=True,
                model_cfg=SimpleNamespace(
                    backend_model="test", memory_enabled=True, ignore_subagents=False,
                ),
            ),
            is_subagent=False,
        )

        state = {"saw_exchange_count_insert": False, "saw_boundary_insert": False}

        class _ConnProxy:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args, **kwargs):
                s = str(sql).lower()
                if "exchange_count" in s and "insert" in s:
                    state["saw_exchange_count_insert"] = True
                    raise RuntimeError("simulated exchange_count INSERT failure")
                if "raw_start_message_index" in s and "insert" in s:
                    state["saw_boundary_insert"] = True
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        @contextlib.contextmanager
        def _patched_immediate_transaction():
            conn = get_connection()
            proxy = _ConnProxy(conn)
            try:
                proxy.execute("BEGIN IMMEDIATE")
                yield proxy
                proxy.execute("COMMIT")
            except Exception:
                try:
                    proxy.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        stage = __import__("proxy.pipeline.post_process", fromlist=["PostProcessStage"]).PostProcessStage()
        with ExitStack() as stack:
            stack.enter_context(patch("memory.sync.submit_sync"))
            stack.enter_context(patch("memory.db.store_messages_batch"))
            stack.enter_context(patch("memory.db.touch_session"))
            stack.enter_context(patch("proxy.session.touch_session"))
            stack.enter_context(patch("memory.db.immediate_transaction", _patched_immediate_transaction))
            stage._persist_messages(ctx)

        assert state["saw_exchange_count_insert"], (
            "Test setup error: exchange_count INSERT was not attempted"
        )
        assert state["saw_boundary_insert"], (
            "V2 boundary INSERT was NOT attempted after exchange_count failed; "
            "HIGH 4 fix did not take effect"
        )

        conn = get_connection()
        row = conn.execute(
            "SELECT raw_start_message_index, exchange_count FROM conversation_summaries "
            "WHERE session_id = ?",
            ("s1",),
        ).fetchone()
        assert row is not None, "V2 boundary must persist even when exchange_count fails"
        assert row["raw_start_message_index"] == 77, (
            f"V2 boundary must be 77 (independent INSERT path), got {row['raw_start_message_index']}"
        )
        conn.execute("DELETE FROM conversation_summaries")
        conn.execute("DELETE FROM messages")
        conn.commit()
        monkeypatch.delenv("PROXY_TEST_DB", raising=False)

    def test_boundary_persists_when_exchange_count_zero(self, monkeypatch):
        """When new_exchanges==0, exchange_count branch is skipped entirely.
        V2 boundary INSERT must still run."""
        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

        from memory.db import init_db, get_connection
        init_db()

        ctx = SimpleNamespace(
            session_id="s1",
            model_name="m1",
            body={"model": "m1", "messages": [
                {"role": "assistant", "content": "no user messages"},
            ]},
            messages_original=None,
            v2_boundary_idx=33,
            v2_boundary_chain="boundary33",
            model_route=SimpleNamespace(
                is_remote=True,
                model_cfg=SimpleNamespace(
                    backend_model="test", memory_enabled=True, ignore_subagents=False,
                ),
            ),
            is_subagent=False,
        )

        stage = __import__("proxy.pipeline.post_process", fromlist=["PostProcessStage"]).PostProcessStage()
        with ExitStack() as stack:
            stack.enter_context(patch("memory.sync.submit_sync"))
            stack.enter_context(patch("memory.db.store_messages_batch"))
            stack.enter_context(patch("memory.db.touch_session"))
            stack.enter_context(patch("proxy.session.touch_session"))
            stage._persist_messages(ctx)

        conn = get_connection()
        row = conn.execute(
            "SELECT raw_start_message_index FROM conversation_summaries "
            "WHERE session_id = ?",
            ("s1",),
        ).fetchone()
        assert row is not None
        assert row["raw_start_message_index"] == 33
        conn.execute("DELETE FROM conversation_summaries")
        conn.execute("DELETE FROM messages")
        conn.commit()
        monkeypatch.delenv("PROXY_TEST_DB", raising=False)
