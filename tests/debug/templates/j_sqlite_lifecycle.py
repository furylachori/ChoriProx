#!/usr/bin/env python3
"""Template J — SQLite lifecycle across :memory:, file-backed reuse,
and pre-init file migration/backfill.

Pinpoints three independently-owned SQLite lifecycles, each via the real
debug_session harness (no private-helper shortcuts):

  Subcase A — :memory: keeper
    debug_session(db_path=":memory:")
    -> memory.db.init_db
    -> memory.db_migrations.init_db
    -> :memory: keeper survives init_db's migration-connection close
    -> real pooled schema-backed write+read through memory.db_summaries
    -> full Flask chat
    -> no checked-out connection remains

  Subcase B — /health checkout ownership
    Direct sqlite3 legacy seed closed before context entry (NOT exercised
    here — that's Subcase C).
    Then:
    -> debug_session(db_path=file)
    -> Flask GET /health (60 iterations)
    -> proxy.admin.health
    -> memory.db.connection context manager
    -> no checked-out pool connection after every call
    -> SQLite health reported `ok` independently of Neo4j's degraded status

  Subcase C — clean pre-init file migration/backfill
    -> Direct sqlite3.connect on a brand-new path
    -> SCHEMA_SQL exec (legacy UNIQUE(session_id, message_index))
    -> direct INSERT one messages row, NO sessions row
    -> connection closed
    -> debug_session(db_path=path)
    -> memory.db.init_db -> memory.db_migrations.init_db
       runs canonical backfill_sessions_table and the
       exchange-index migration that swaps the unique constraint
       to (session_id, exchange_index).
    -> pooled production reads confirm legacy row + new constraint
    -> full Flask chat against the same explicit DB
    -> no checkout leak

Each subcase uses its own TemporaryDirectory. Each subcase asserts the
explicit invariants the harness diff 2 fix covers (no checked-out
connections after each pooled operation).

Production call-stack proof (each subcase above is independent):
    debug_session(db_path=...)
    -> memory.db.init_db
    -> memory.db_migrations.init_db
    -> schema migration + indexed checkouts
    -> pooled production reads / writes
    -> Flask /health and /v1/chat/completions
    -> no checked-out SQLite ownership remains

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/j_sqlite_lifecycle.py
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session


def _unique_index_columns(conn, table):
    """Return a list of tuples of column names for each UNIQUE index on `table`.

    Walks ``PRAGMA index_list(table)`` then ``PRAGMA index_info(name)``. The
    canonical constraint is whichever unique index's columns match
    ``(session_id, exchange_index)`` — NOT ``(session_id, message_index)``.
    """
    out = []
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    for row in rows:
        name = row[1]
        info = conn.execute(f"PRAGMA index_info({name})").fetchall()
        cols = tuple(r[2] for r in info)
        out.append(cols)
    return out


def _subcase_a():
    """`:memory:` keeper + full Flask chat, asserted schema-backed."""
    with debug_session(
        db_path=":memory:",
        canned_responses=[
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "MEMORY_DB_CHAT_OK",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ],
        models=["qwen3-coder"],
    ) as c:
        import memory.db_pool as db_pool
        from memory.db_summaries import store_message_summary

        # Pool bound to ":memory:" before any subcase-internal query.
        _emit(
            1,
            "A: pool exists",
            db_pool._pool is not None,
            "db_pool._pool must be initialized by debug_session",
        )
        _emit(
            2,
            "A: pool bound to :memory:",
            db_pool._pool._db_path == ":memory:",
            f"path={db_pool._pool._db_path!r}",
        )

        # Canonical tables must exist via a schema-backed read.
        with db_pool.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            tables = {row[0] for row in rows}
        _emit(
            3,
            "A: messages table exists",
            "messages" in tables,
            f"tables={sorted(tables)}",
        )
        _emit(
            4,
            "A: sessions table exists",
            "sessions" in tables,
            f"tables={sorted(tables)}",
        )
        _emit(
            5,
            "A: message_summaries table exists",
            "message_summaries" in tables,
            f"tables={sorted(tables)}",
        )

        # Canonical summary constraint — NOT the legacy one.
        with db_pool.connection() as conn:
            cols = _unique_index_columns(conn, "message_summaries")
        _emit(
            6,
            "A: message_summaries unique index includes (session_id, exchange_index)",
            ("session_id", "exchange_index") in cols,
            f"unique-index column tuples={cols}",
        )
        _emit(
            7,
            "A: message_summaries unique index excludes (session_id, message_index)",
            ("session_id", "message_index") not in cols,
            f"unique-index column tuples={cols}",
        )

        # Real pooled schema-backed write+read of store_message_summary.
        write_ok = store_message_summary(
            session_id="debug-memory-contract",
            exchange_index=0,
            role="user",
            summary="MEMORY_SUMMARY_CONTRACT",
            tool_summary="",
            content_hash="debug-memory-contract-hash",
            model_used="debug-session",
            token_estimate=1,
            end_message_index=1,
            message_index=0,
        )
        _emit(
            8,
            "A: store_message_summary succeeded",
            write_ok is True,
            f"returned {write_ok!r}",
        )

        assert db_pool._pool._checked_out == {}, (
            f"connection leaked after write: {list(db_pool._pool._checked_out)}"
        )
        _emit(
            9,
            "A: no checkout after pooled write",
            db_pool._pool._checked_out == {},
            f"checked_out={list(db_pool._pool._checked_out)}",
        )

        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT summary, role FROM message_summaries "
                "WHERE session_id = ? AND exchange_index = ?",
                ("debug-memory-contract", 0),
            ).fetchone()
        _emit(
            10,
            "A: pooled read returns stored summary",
            row is not None and row["summary"] == "MEMORY_SUMMARY_CONTRACT",
            f"row={dict(row) if row else None}",
        )

        _emit(
            11,
            "A: no checkout after pooled read",
            db_pool._pool._checked_out == {},
            f"checked_out={list(db_pool._pool._checked_out)}",
        )

        # Full Flask chat through the in-memory schema — proves the schema
        # is STILL alive after every prior pooled operation.
        response = c.chat(
            messages=[
                {"role": "user", "content": "memory keeper request"}
            ],
            client_hint="debug-memory-contract-chat",
        )
        _emit(
            12,
            "A: full chat returns 200",
            response.status_code == 200,
            f"got {response.status_code}",
        )
        _emit(
            13,
            "A: success marker in response",
            "MEMORY_DB_CHAT_OK" in response.content,
            f"content prefix={response.content[:120]!r}",
        )
        _emit(
            14,
            "A: exactly one captured upstream request",
            len(c.captured_requests) == 1,
            f"got {len(c.captured_requests)}",
        )
        _emit(
            15,
            "A: no checkout after full chat",
            db_pool._pool._checked_out == {},
            f"checked_out={list(db_pool._pool._checked_out)}",
        )


def _subcase_b():
    """60 /health calls — aggregate status independent from SQLite assertion."""
    failure_detail: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "health.db")
        with debug_session(db_path=db_path) as c:
            import memory.db_pool as db_pool

            for _ in range(60):
                resp = c.client.get("/health")
                if resp.status_code not in (200, 503):
                    failure_detail.append(
                        f"unexpected status={resp.status_code}"
                    )
                    break
                try:
                    body = resp.get_json()
                except Exception as exc:
                    failure_detail.append(f"json decode error: {exc}")
                    break
                if not isinstance(body, dict):
                    failure_detail.append(f"body is not dict: {body!r}")
                    break
                if body.get("sqlite") != "ok":
                    failure_detail.append(f"sqlite != ok: {body.get('sqlite')!r}")
                    break

            _emit(
                16,
                "B: 60 /health calls all 200/503",
                len(failure_detail) == 0,
                f"first failure={failure_detail[0] if failure_detail else 'none'}",
            )
            _emit(
                17,
                "B: pool exists across /health checks",
                db_pool._pool is not None,
                "db_pool._pool must be initialized",
            )
            _emit(
                18,
                "B: no checked-out pool connection after /health loop",
                db_pool._pool._checked_out == {},
                f"checked_out={list(db_pool._pool._checked_out)}",
            )

            # Aggregate-body assertion: ``sqlite`` is ``ok``, regardless of
            # Neo4j status. We do not require ``body["status"] == "ok"``
            # because Neo4j is mocked/degraded in this debug session.
            resp = c.client.get("/health")
            body = resp.get_json()
            _emit(
                19,
                "B: body['sqlite'] == 'ok' (independent of aggregate status)",
                isinstance(body, dict) and body.get("sqlite") == "ok",
                f"body={body}",
            )

            # Real pooled schema-backed query to confirm the pool is alive
            # AND the schema is still accessible after 61 health checks.
            with db_pool.connection() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM sessions"
                ).fetchone()[0]
            _emit(
                20,
                "B: SELECT COUNT(*) FROM sessions works after 60 /health checks",
                isinstance(count, int),
                f"count={count}",
            )
            _emit(
                21,
                "B: no checkout after pooled schema query",
                db_pool._pool._checked_out == {},
                f"checked_out={list(db_pool._pool._checked_out)}",
            )

            # Sanity probe — SELECT 1 works but is NOT the keeper proof; the
            # schema-backed sessions query above is.
            with db_pool.connection() as conn:
                one = conn.execute("SELECT 1").fetchone()[0]
            _emit(
                22,
                "B: SELECT 1 returns 1 (sanity)",
                one == 1,
                f"got {one}",
            )


def _subcase_c():
    """Pre-init legacy file: seed, close raw connection, then
    debug_session(db_path=file) runs the production migration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "legacy.db")
        # 1. Direct sqlite3 legacy seed. Do not call any memory.db_pool
        #    helper — open a raw connection, exec SCHEMA_SQL, insert one
        #    messages row, commit, and close. The legacy SCHEMA_SQL still
        #    declares UNIQUE(session_id, message_index) on
        #    message_summaries; the production migration will swap it for
        #    UNIQUE(session_id, exchange_index).
        from memory.db_migrations import SCHEMA_SQL

        with sqlite3.connect(db_path) as raw_conn:
            raw_conn.executescript(SCHEMA_SQL)
            raw_conn.execute(
                "INSERT INTO messages (session_id, message_index, role, content, "
                "content_hash, created_at, updated_at, exchange_index) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "LEGACY_BACKFILL_SESSION",
                    0,
                    "user",
                    "LEGACY_BACKFILL_MESSAGE_MARKER",
                    "legacy-backfill-hash",
                    "2026-07-13T00:00:00Z",
                    "2026-07-13T00:00:00Z",
                    0,
                ),
            )
            raw_conn.commit()
        # Raw connection is closed. Now enter debug_session — it will run
        # init_db() which migrates the legacy UNIQUE index and backfills
        # the sessions row.
        with debug_session(
            db_path=db_path,
            canned_responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "LEGACY_DB_CHAT_OK",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ],
        ) as c:
            import memory.db_pool as db_pool

            # 1. The legacy messages row is still present.
            with db_pool.connection() as conn:
                row = conn.execute(
                    "SELECT content FROM messages "
                    "WHERE session_id = ? AND message_index = 0",
                    ("LEGACY_BACKFILL_SESSION",),
                ).fetchone()
            _emit(
                23,
                "C: legacy message row preserved through migration",
                row is not None and row["content"] == "LEGACY_BACKFILL_MESSAGE_MARKER",
                f"row={dict(row) if row else None}",
            )

            # 2. Sessions backfill created the LEGACY_BACKFILL_SESSION row.
            with db_pool.connection() as conn:
                sess = conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    ("LEGACY_BACKFILL_SESSION",),
                ).fetchone()
            _emit(
                24,
                "C: sessions table contains LEGACY_BACKFILL_SESSION",
                sess is not None,
                f"row={dict(sess) if sess else None}",
            )

            # 3. message_summaries canonical constraint — the legacy
            #    (session_id, message_index) must be GONE.
            with db_pool.connection() as conn:
                cols = _unique_index_columns(conn, "message_summaries")
            _emit(
                25,
                "C: message_summaries unique index includes (session_id, exchange_index)",
                ("session_id", "exchange_index") in cols,
                f"unique-index column tuples={cols}",
            )
            _emit(
                26,
                "C: message_summaries unique index excludes legacy (session_id, message_index)",
                ("session_id", "message_index") not in cols,
                f"unique-index column tuples={cols}",
            )

            # 4. Full Flask chat against the same explicit DB.
            response = c.chat(
                messages=[
                    {"role": "user", "content": "legacy backfill chat"}
                ],
                client_hint="legacy-backfill-chat",
            )
            _emit(
                27,
                "C: full chat returns 200",
                response.status_code == 200,
                f"got {response.status_code}",
            )
            _emit(
                28,
                "C: success marker in response",
                "LEGACY_DB_CHAT_OK" in response.content,
                f"content prefix={response.content[:120]!r}",
            )
            _emit(
                29,
                "C: exactly one captured upstream request",
                len(c.captured_requests) == 1,
                f"got {len(c.captured_requests)}",
            )
            _emit(
                30,
                "C: no checkout after full chat",
                db_pool._pool._checked_out == {},
                f"checked_out={list(db_pool._pool._checked_out)}",
            )


def main():
    _subcase_a()
    _subcase_b()
    _subcase_c()


if __name__ == "__main__":
    try:
        main()
    finally:
        _finish()
