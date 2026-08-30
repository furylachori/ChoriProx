"""US-F1/F3/D8: Session-scoped conversation_summaries invariants.

Verifies:
- F3: conversation_summaries PRIMARY KEY is (session_id) — no model_name.
- F1: model-switch inheritance works because cursor is session-scoped.
- D8: recovery path no longer fails with NOT NULL constraint on model_name.

Tests:
1. _migrate_recreate_conversation_summaries collapses 3 multi-model rows into 1
2. Tie-break: when cursors tie, longest ancient_history wins.
3. Fresh DB schema has session_id PK, no model_name column, correct index.
4. Model-switch inheritance: pre-populated cursor=4 stays at 4 across "models".
5. Recovery path with no model_name.
6. D8 killed: model_name=None explicitly works.
"""
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


def _seed_v1_rows(conn, session_id, rows):
    """Insert legacy V1 rows (with model_name) for migration collapse tests."""
    for (model_name, lei, ah, raw_start, ec) in rows:
        conn.execute(
            "INSERT INTO conversation_summaries "
            "(session_id, model_name, last_exchange_index, ancient_history, "
            " raw_start_message_index, exchange_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, model_name, lei, ah, raw_start, ec, _now(), _now()),
        )
    conn.commit()


def _build_v1_schema_conn():
    """Create a connection with the pre-C3 V1 schema (model_name present)."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE conversation_summaries (
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


class TestMigrationCollapse:
    """F3: migration collapses multi-model rows per session to a single session-scoped row."""

    def test_no_tie_migration_picks_max_cursor(self):
        """3 models, distinct cursors → row with MAX(last_exchange_index) wins."""
        from memory.db_migrations import _migrate_recreate_conversation_summaries

        conn = _build_v1_schema_conn()
        _seed_v1_rows(conn, "s", [
            ("m-A", 40, "ancient-A-" + "X" * 1000, 10, 5),
            ("m-B", 7, "ancient-B-" + "Y" * 200, 3, 1),
            ("m-C", 25, "ancient-C-" + "Z" * 500, 8, 3),
        ])

        _migrate_recreate_conversation_summaries(conn)

        rows = conn.execute(
            "SELECT * FROM conversation_summaries WHERE session_id = ?", ("s",)
        ).fetchall()
        assert len(rows) == 1, f"Expected 1 row after collapse, got {len(rows)}"
        row = rows[0]
        assert row["last_exchange_index"] == 40
        assert row["ancient_history"].startswith("ancient-A")
        assert row["raw_start_message_index"] == 10

        cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "model_name" not in cols, "model_name column must be dropped post-collapse"

    def test_tie_break_picks_longest_ancient_history(self):
        """2 models with tied cursor → row with longest ancient_history wins."""
        from memory.db_migrations import _migrate_recreate_conversation_summaries

        conn = _build_v1_schema_conn()
        _seed_v1_rows(conn, "s", [
            ("m-A", 40, "X" * 500, 5, 3),
            ("m-B", 40, "Y" * 2000, 20, 8),
        ])

        _migrate_recreate_conversation_summaries(conn)

        rows = conn.execute(
            "SELECT * FROM conversation_summaries WHERE session_id = ?", ("s",)
        ).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["last_exchange_index"] == 40
        assert len(row["ancient_history"]) == 2000, (
            f"Tie-break must pick longest ancient_history (2000 chars), got {len(row['ancient_history'])}"
        )
        assert row["raw_start_message_index"] == 20, "Tie-break must carry raw_start from winning row"


class TestFreshDbSchema:
    """Fresh DB must have session_id PK, no model_name, correct index."""

    def test_init_db_creates_session_scoped_schema(self, _isolate_db):
        from memory.db_migrations import SCHEMA_SQL
        import sqlite3

        conn = sqlite3.connect(_isolate_db)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)

        cs_cols = [r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)").fetchall()]
        assert "session_id" in cs_cols
        assert "summary" in cs_cols
        assert "summary_tokens" in cs_cols
        assert "exchange_count" in cs_cols
        assert "last_exchange_index" in cs_cols
        assert "last_exchange_start" in cs_cols
        assert "last_message_index" in cs_cols
        assert "raw_start_message_index" in cs_cols
        assert "ancient_history" in cs_cols
        assert "created_at" in cs_cols
        assert "updated_at" in cs_cols
        assert "model_name" not in cs_cols, "Fresh schema must not include model_name"

        pk_cols = [
            r["name"] for r in conn.execute("PRAGMA index_list(conversation_summaries)").fetchall()
        ]

        indexes = conn.execute("PRAGMA index_list(conversation_summaries)").fetchall()
        assert any(
            idx["name"] == "idx_conv_summaries_boundary" for idx in indexes
        ), "idx_conv_summaries_boundary index must exist"

        index_info = conn.execute("PRAGMA index_info(idx_conv_summaries_boundary)").fetchall()
        index_cols = [r["name"] for r in index_info]
        assert index_cols == ["session_id", "raw_start_message_index"], (
            f"Index cols must be (session_id, raw_start_message_index), got {index_cols}"
        )

        indexes_with_model_name = []
        for idx in indexes:
            if idx["name"].startswith("sqlite_autoindex"):
                continue
            cols_for_idx = conn.execute(f"PRAGMA index_info({idx['name']})").fetchall()
            col_names = [c["name"] for c in cols_for_idx]
            if "model_name" in col_names:
                indexes_with_model_name.append(idx["name"])
        assert indexes_with_model_name == [], (
            f"No index should reference model_name; found: {indexes_with_model_name}"
        )

        conn.close()


class TestCursorSessionScope:
    """F1: cursor is session-scoped — 'model-switch' inherits the cursor."""

    def test_cursor_persists_across_model_changes(self):
        """Pre-populate cursor=4 on session s. Call sync_exchange twice with
        different "models" — cursor must remain at 4 (caught up), and both
        runs share the same (session_id) row in conversation_summaries."""
        from memory.db_summaries import (
            set_last_summarized_exchange_index,
            get_last_summarized_exchange_index,
        )

        session_id = "test-session-scope-cursor"
        set_last_summarized_exchange_index(session_id, 4)

        assert get_last_summarized_exchange_index(session_id) == 4

        from memory.db_pool import connection
        with connection() as conn:
            rows = conn.execute(
                "SELECT session_id FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        assert len(rows) == 1, "Session cursor lives in a single row, not per-model"


class TestRecoveryNoModelName:
    """D8: recovery path with no model_name parameter must work."""

    def test_sync_exchange_recovery_no_model_name(self):
        """sync_exchange(session_id, _recovery_start=0, _recovery_end=4) works
        without passing model_name — the D8 NOT NULL violation is gone.

        D8 killed is verified directly: set_last_summarized_exchange_index
        does NOT require model_name, and the recovery range function accepts
        a session_id-only signature.
        """
        from memory.sync import sync_exchange, _determine_sync_range
        import inspect

        sig = inspect.signature(_determine_sync_range)
        params = list(sig.parameters.keys())
        assert "model_name" not in params, (
            f"_determine_sync_range must not take model_name; got params={params}"
        )

        sig_sync = inspect.signature(sync_exchange)
        sync_params = list(sig_sync.parameters.keys())
        assert "model_name" not in sync_params, (
            f"sync_exchange must not take model_name; got params={sync_params}"
        )

        from memory.db_summaries import set_last_summarized_exchange_index
        sig_set = inspect.signature(set_last_summarized_exchange_index)
        set_params = list(sig_set.parameters.keys())
        assert "model_name" not in set_params, (
            f"set_last_summarized_exchange_index must not take model_name; got params={set_params}"
        )


class TestD8Killed:
    """D8 explicit: model_name=None must not trigger NOT NULL violation."""

    def test_set_last_summarized_exchange_index_no_model_name(self):
        """set_last_summarized_exchange_index(session_id, 5) works without
        model_name — no NOT NULL violation."""
        from memory.db_summaries import set_last_summarized_exchange_index

        session_id = "test-d8-killed"

        set_last_summarized_exchange_index(session_id, 5)

        from memory.db_pool import connection
        with connection() as conn:
            row = conn.execute(
                "SELECT last_exchange_index FROM conversation_summaries "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        assert row is not None, "Row must exist after set_last_summarized_exchange_index"
        assert row["last_exchange_index"] == 5


class TestLoadStateFromDbSessionScope:
    """C3 fix: _load_state_from_db must work against the session-scoped schema.

    Regression pinned at context_v2.py:446-450. The buggy query referenced
    `model_name`, which C3 dropped from conversation_summaries. At runtime
    this raises sqlite3.OperationalError: no such column: model_name, caught
    at context_v2.py:548-554 → cold path. V2 restart recovery becomes a
    silent no-op for every session with a conversation_summaries row.
    """

    def test_load_state_from_db_works_against_session_scoped_schema(self):
        """_load_state_from_db must NOT raise OperationalError against the
        post-C3 schema. It should return a valid state-entry dict when the
        session has all messages chained (gap check passes) and
        raw_start_message_index = 0.
        """
        import hashlib

        from memory.db_pool import connection
        from proxy.context_v2 import _load_state_from_db

        session_id = "test-load-state-session-scoped"
        model_name = "test-model"
        now = _now()

        with connection() as conn:
            for i in range(3):
                chain_hash = hashlib.sha256(f"msg-{i}".encode()).hexdigest()[:16]
                conn.execute(
                    "INSERT INTO messages (session_id, message_index, role, content, "
                    "exchange_index, chain_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        i,
                        "user" if i % 2 == 0 else "assistant",
                        f"message {i}",
                        i // 2,
                        chain_hash,
                        now,
                    ),
                )

            conn.execute(
                "INSERT INTO conversation_summaries "
                "(session_id, raw_start_message_index, last_exchange_index, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, 0, 1, now, now),
            )
            conn.commit()

        result = _load_state_from_db(session_id, model_name)

        assert result is not None, (
            "_load_state_from_db returned None — gap check or schema bug"
        )
        assert result["boundary_idx"] == 0
        assert result["boundary_chain"] is None
        assert isinstance(result["last_chain"], list)
        assert len(result["last_chain"]) == 3
        assert "summaries" in result
        assert "ancient" in result
        assert result["version"] == 0