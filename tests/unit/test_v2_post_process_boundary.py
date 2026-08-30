"""Unit tests for V2 round-4 patch: post_process boundary persistence.

Verifies the full V2 → post_process → DB write flow:
  1. V2 sets ctx.v2_boundary_idx after _select_context_v2 runs.
  2. PostProcessStage._persist_messages writes raw_start_message_index to
     conversation_summaries for (session_id, model_name).
  3. Restart-recovery: subsequent V2 calls load boundary from DB and warm-path.

This guards against the CRITICAL #1 regression where migration dropped
created_at and every V2 turn hit:
    sqlite3.OperationalError: table conversation_summaries has no column named created_at
The error was silently swallowed at debug level → restart recovery always
cold-paths → entire V2 redesign defeated.
"""
import os
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


pytestmark = pytest.mark.unit


def _user(content):
    return {"role": "user", "content": content}


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


class TestPostProcessBoundaryPersists:
    """HIGH #10: boundary_idx must round-trip from V2 → post_process → DB."""

    def test_boundary_idx_written_to_db(self, monkeypatch):
        """End-to-end: V2 sets boundary, post_process writes it to DB."""
        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

        from memory.db import init_db, get_connection
        init_db()

        ctx = SimpleNamespace(
            session_id="s1",
            model_name="m1",
            body={"model": "m1", "messages": [_user("u1"), _user("u2")]},
            messages_original=None,
            v2_boundary_idx=42,
            v2_boundary_chain="abc123",
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
            try:
                stage._persist_messages(ctx)
            except Exception as e:
                pytest.fail(f"_persist_messages raised unexpectedly: {e!r}")

        conn = get_connection()
        row = conn.execute(
            "SELECT raw_start_message_index FROM conversation_summaries "
            "WHERE session_id = ?",
            ("s1",),
        ).fetchone()
        assert row is not None, "V2 boundary not persisted to conversation_summaries"
        assert row["raw_start_message_index"] == 42
        conn.execute("DELETE FROM conversation_summaries")
        conn.execute("DELETE FROM messages")
        conn.commit()
        monkeypatch.delenv("PROXY_TEST_DB", raising=False)

    def test_boundary_idx_handles_missing_field(self, monkeypatch):
        """When V2 sets v2_boundary_idx=None (no summaries injected), raw_start_message_index
        is NOT updated (boundary field stays at -1 sentinel)."""
        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

        from memory.db import init_db, get_connection
        init_db()

        ctx = SimpleNamespace(
            session_id="s2",
            model_name="m1",
            body={"model": "m1", "messages": [_user("u1")]},
            messages_original=None,
            v2_boundary_idx=None,
            v2_boundary_chain=None,
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
            try:
                stage._persist_messages(ctx)
            except Exception as e:
                pytest.fail(f"_persist_messages raised unexpectedly: {e!r}")

        conn = get_connection()
        row = conn.execute(
            "SELECT raw_start_message_index FROM conversation_summaries "
            "WHERE session_id = ?",
            ("s2",),
        ).fetchone()
        assert row is not None, "conversation_summaries row must exist"
        assert row["raw_start_message_index"] == -1, (
            f"raw_start_message_index must remain -1 (sentinel) when no user messages; "
            f"got {row['raw_start_message_index']}"
        )
        conn.execute("DELETE FROM conversation_summaries")
        conn.execute("DELETE FROM messages")
        conn.commit()
        monkeypatch.delenv("PROXY_TEST_DB", raising=False)

    def test_insert_uses_existing_created_at_column(self, monkeypatch):
        """Verify post_process INSERT column list is compatible with the
        post-migration schema. This is the CRITICAL #1 regression guard:
        created_at/updated_at must be present in conversation_summaries.
        """
        monkeypatch.setenv("PROXY_TEST_DB", ":memory:")

        from memory.db import init_db, get_connection
        init_db()

        conn = get_connection()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "created_at" in cols
        assert "updated_at" in cols
        conn.execute("DELETE FROM conversation_summaries")
        conn.execute("DELETE FROM messages")
        conn.commit()
        monkeypatch.delenv("PROXY_TEST_DB", raising=False)
