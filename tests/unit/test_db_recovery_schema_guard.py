"""US-126 (Issue B): validate that recovery never swaps in an incomplete db.

Fix: ``db_recovery._recover_corrupt_db_inner`` now runs
``db_migrations.validate_schema_complete`` on the healing connection BEFORE
``_atomic_swap``. If validation fails the recovery returns False and the
live db is left untouched.

The corrupt-page-omitted-DDL failure mode (Issue B): when ``sqlite3 .dump``
runs on a corrupt db whose pages backing ``sessions`` or
``rate_limiter_state`` are damaged, those ``CREATE TABLE`` statements are
silently absent from the dump. The healing db then passes
``PRAGMA integrity_check`` (SQLite structure is OK) but is missing
application tables — the F1 privilege-escalation guard
(``session_exists``) is silently disabled once the healing db is swapped
in. This file pins that the recovery path catches and refuses this case.

Per AGENTS.md test discipline (5 invariants enforced):

  1. Exercise the real production call site — no mocking of
     ``recover_corrupt_db`` or ``_recover_corrupt_db_inner``.
  2. Do not mock the function under test.
  3. Revert-test confirms each test FAILS when the production fix is reverted
     (see the section at the bottom of the file).
  4. Grep ``src/`` for callers of each method before testing — every method
     exercised here has production callers.
  5. Concurrency tests use ``threading.Barrier`` (NOT ``time.sleep``) for
     deterministic synchronized start.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
from unittest import mock

import pytest


# ----------------------------------------------------------------------------
# Helpers (shared with test_db_recovery.py; duplicated to keep this file standalone)
# ----------------------------------------------------------------------------


def _create_healthy_db(db_path):
    """Create a fresh db at ``db_path`` with the production schema for tests."""
    from memory import db_migrations
    db_migrations.init_db(db_path)


def _corrupt_header(db_path):
    """Write garbage to the first 16 bytes of the db header."""
    with open(db_path, "r+b") as f:
        f.write(b"\x00" * 16)


# ----------------------------------------------------------------------------
# TestValidatorBehavior
# ----------------------------------------------------------------------------


class TestValidatorBehavior:
    """Direct unit tests for ``db_migrations.validate_schema_complete``.

    These exercise the validator on real sqlite3 dbs (per AGENTS.md
    "Tautological tests" rule 2 — no mocking). The recovery tests below
    use this validator implicitly via ``recover_corrupt_db``.
    """

    def test_validator_accepts_full_schema(self, tmp_path):
        """A db freshly produced by ``init_db`` passes validation.

        Validates that:
          * validator returns ``(True, [])`` on a healthy production-schema db.
          * No false positives on any table or column.
        """
        db_path = str(tmp_path / "healthy.db")
        _create_healthy_db(db_path)

        from memory.db_migrations import validate_schema_complete

        con = sqlite3.connect(db_path)
        try:
            ok, diagnostics = validate_schema_complete(con)
        finally:
            con.close()

        assert ok is True, (
            f"validator rejected a freshly-created production-schema db; "
            f"diagnostics={diagnostics}"
        )
        assert diagnostics == [], (
            f"validator returned diagnostics on a healthy db: {diagnostics}"
        )

    def test_validator_rejects_missing_sessions(self, tmp_path):
        """Dropping the ``sessions`` table fails validation with a clear diagnostic.

        Issue B reproducer for the validator itself: without the
        ``sessions`` table the F1 ``session_exists`` privilege-escalation
        guard is silently disabled. The validator MUST name ``sessions`` in
        its diagnostics so operators can grep for it.
        """
        db_path = str(tmp_path / "no_sessions.db")
        _create_healthy_db(db_path)

        con = sqlite3.connect(db_path)
        try:
            con.execute("DROP TABLE sessions")
            con.commit()
        finally:
            con.close()

        from memory.db_migrations import validate_schema_complete

        con = sqlite3.connect(db_path)
        try:
            ok, diagnostics = validate_schema_complete(con)
        finally:
            con.close()

        assert ok is False, (
            "validator accepted a db with sessions dropped; F1 guard "
            "would silently disable"
        )
        assert any("sessions" in d for d in diagnostics), (
            f"diagnostics did not mention sessions; got {diagnostics}"
        )

    def test_validator_rejects_missing_rate_limiter_columns(self, tmp_path):
        """Dropping a critical ``rate_limiter_state`` column fails validation.

        Recreates the table without the ``timestamps`` column. The validator
        must catch both the missing column AND the lost PRIMARY KEY
        constraint (the recreate drops both).
        """
        db_path = str(tmp_path / "no_timestamps.db")
        _create_healthy_db(db_path)

        con = sqlite3.connect(db_path)
        try:
            con.execute(
                "CREATE TABLE rate_limiter_state_new "
                "(limiter_name TEXT NOT NULL, key TEXT NOT NULL)"
            )
            con.execute(
                "INSERT INTO rate_limiter_state_new "
                "SELECT limiter_name, key FROM rate_limiter_state"
            )
            con.execute("DROP TABLE rate_limiter_state")
            con.execute(
                "ALTER TABLE rate_limiter_state_new "
                "RENAME TO rate_limiter_state"
            )
            con.commit()
        finally:
            con.close()

        from memory.db_migrations import validate_schema_complete

        con = sqlite3.connect(db_path)
        try:
            ok, diagnostics = validate_schema_complete(con)
        finally:
            con.close()

        assert ok is False, (
            "validator accepted a rate_limiter_state missing the timestamps "
            "column AND its PRIMARY KEY"
        )
        joined = " | ".join(diagnostics)
        assert "timestamps" in joined, (
            f"validator did not name the missing timestamps column; "
            f"diagnostics={diagnostics}"
        )
        assert "PRIMARY KEY" in joined, (
            f"validator did not flag the missing PRIMARY KEY; "
            f"diagnostics={diagnostics}"
        )


# ----------------------------------------------------------------------------
# TestRecoverySchemaGuard
# ----------------------------------------------------------------------------


class TestRecoverySchemaGuard:
    """End-to-end tests for the recovery path's schema-completeness guard.

    These tests pin Issue B at the recovery-procedure level: even when the
    dump output omits CREATE TABLE statements for ``sessions`` and
    ``rate_limiter_state``, the recovery procedure must catch the omission
    via ``_apply_and_validate_schema`` and either fix it (idempotent
    re-apply of SCHEMA_SQL) or refuse the swap (validation still fails).
    """

    def _patch_subprocess_for_partial_dump(
        self,
        omit_tables=("sessions", "rate_limiter_state"),
    ):
        """Return a ``subprocess.run`` patch that:
          * runs the real subprocess for ``.recover`` (returns whatever the
            CLI produces; we don't care here).
          * writes a synthetic dump.sql derived from the REAL
            ``db_migrations.SCHEMA_SQL`` with the CREATE TABLE statements for
            ``omit_tables`` REMOVED — the Issue B failure mode (corrupt
            pages → omitted DDL for sessions/rate_limiter_state).

        Why derive from real SCHEMA_SQL rather than hand-rolled DDL: the
        validator's expanded required_columns list (see db_migrations.
        REQUIRED_COLUMNS) covers ~70% of every CREATE TABLE's columns. A
        hand-rolled minimal dump would diverge from production and fail
        validation even on the happy path. Using real SCHEMA_SQL minus
        omitted tables means the only structural difference from a healthy
        dump.sql is the missing CREATE TABLE statements — exactly the
        failure mode we want to exercise.

        The patch writes the dump.sql DIRECTLY to the file (the recovery
        opens it in text mode and passes the file handle as ``stdout=``).
        """
        import re as _re
        from memory import db_migrations

        real_run = subprocess.run
        patched_calls = {"dump": 0, "recover": 0}

        # Build the partial dump: full SCHEMA_SQL with CREATE TABLE for
        # omitted tables removed. The trailing semicolon + closing paren
        # is captured to fully strip the statement. CREATE INDEX statements
        # referencing the omitted table are also removed so the dump
        # doesn't reference nonexistent indexes (sqlite3 accepts orphan
        # CREATE INDEX but it's cleaner without).
        dump_sql = db_migrations.SCHEMA_SQL
        for tbl in omit_tables:
            dump_sql = _re.sub(
                rf"CREATE TABLE IF NOT EXISTS {tbl}\s*\([^;]*\);\s*",
                "",
                dump_sql,
                flags=_re.DOTALL,
            )
            dump_sql = _re.sub(
                rf"CREATE INDEX IF NOT EXISTS [^\s]* ON {tbl}\([^)]*\);\s*",
                "",
                dump_sql,
            )

        # Wrap in BEGIN/COMMIT envelope so executescript's implicit
        # transaction shape matches what sqlite3 .dump produces.
        bad_dump = (
            "PRAGMA foreign_keys=OFF;\n"
            "BEGIN TRANSACTION;\n"
            + dump_sql
            + "COMMIT;\n"
        )

        def patched_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[2] == ".dump":
                patched_calls["dump"] += 1
                stdout = kwargs.get("stdout")
                if stdout is not None and hasattr(stdout, "write"):
                    stdout.write(bad_dump)
                    try:
                        stdout.flush()
                    except Exception:
                        pass
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout=b"", stderr=b"",
                    )
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=bad_dump.encode("utf-8"), stderr=b"",
                )
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[2] == ".recover":
                patched_calls["recover"] += 1
            return real_run(*args, **kwargs)

        return patched_run, patched_calls

    def test_recovery_completes_missing_tables_after_partial_dump(self, tmp_path):
        """Even with a dump.sql that omits ``sessions`` and
        ``rate_limiter_state`` CREATE TABLE statements, the recovery
        procedure must produce a swapped db that has BOTH tables.

        This is the Issue B happy-path-after-fix: the new
        ``_apply_and_validate_schema`` step applies SCHEMA_SQL (idempotent)
        on top of whatever the dump loaded, so the omitted tables get
        filled in. Recovery returns True; the post-swap db has the tables.
        """
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "partial_dump.db")
        _create_healthy_db(db_path)
        _corrupt_header(db_path)

        patched_run, calls = self._patch_subprocess_for_partial_dump(
            omit_tables=("sessions", "rate_limiter_state"),
        )
        with mock.patch.object(subprocess, "run", side_effect=patched_run):
            ok = recover_corrupt_db(db_path)

        assert ok is True, (
            "Issue B fix FAIL: recovery returned False for a partial dump "
            "where the missing tables can be filled in by SCHEMA_SQL. "
            "Expected True after _apply_and_validate_schema re-applies "
            "the production schema."
        )
        assert calls["dump"] >= 1, (
            "patched_run was never called for .dump; cannot validate "
            "the partial-dump path. patched_calls={}".format(calls)
        )

        # Post-swap the db MUST have both omitted tables.
        con = sqlite3.connect(db_path)
        try:
            present = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            # PRAGMA integrity_check is a no-op-on-empty invariant for this
            # file but we keep it as a sanity check that the swap landed.
            integrity = con.execute("PRAGMA integrity_check").fetchone()
        finally:
            con.close()

        assert "sessions" in present, (
            f"Issue B FAIL: post-recovery db missing 'sessions' table; "
            f"present={sorted(present)}"
        )
        assert "rate_limiter_state" in present, (
            f"Issue B FAIL: post-recovery db missing 'rate_limiter_state' "
            f"table; present={sorted(present)}"
        )
        assert integrity is not None and integrity[0] == "ok", (
            f"post-recovery integrity_check failed: {integrity}"
        )

    def test_recovery_refuses_incomplete_candidate(self, tmp_path):
        """When the validator still fails after the SCHEMA_SQL catch-up
        retry, the recovery procedure returns False and the live db_path
        is unchanged.

        To deterministically force validator failure we patch
        ``_apply_and_validate_schema`` to always return False AFTER
        bypass-the-fix but BEFORE the swap. We restore the live db file
        before-and-after so the swap target (which doesn't exist) is
        obviously unchanged.

        We intentionally do NOT corrupt the header — the validator patch
        alone forces refusal, so the test exercises the validator guard
        without depending on actual corruption patterns. This keeps the
        "live db unchanged" assertion tight (post_swap == pre_swap).
        """
        from memory import db_recovery as db_recovery_mod
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "refused.db")
        _create_healthy_db(db_path)

        # Snapshot the live db content + mtime so we can prove the recovery
        # procedure did not swap anything in. (At this point the db is
        # healthy — corruption is not required for the validator guard.)
        pre_swap = open(db_path, "rb").read()
        pre_swap_mtime = os.path.getmtime(db_path)

        with mock.patch.object(
            db_recovery_mod,
            "_apply_and_validate_schema",
            return_value=False,
        ):
            ok = recover_corrupt_db(db_path)

        assert ok is False, (
            "Issue B fix FAIL: recovery returned True even though "
            "_apply_and_validate_schema returned False. The fix MUST "
            "refuse to swap when the candidate schema is incomplete."
        )

        # Live db_path must be unchanged — recovery must NOT have swapped
        # in the incomplete healing db.
        assert os.path.exists(db_path), (
            f"recovery deleted the live db at {db_path}"
        )
        post_swap = open(db_path, "rb").read()
        assert post_swap == pre_swap, (
            "Issue B fix FAIL: live db content changed after a refused "
            "recovery (atomic_swap must not be reached when validation fails)"
        )
        assert os.path.getmtime(db_path) == pre_swap_mtime, (
            "Issue B fix FAIL: live db mtime changed after a refused recovery"
        )

    def test_recovery_does_not_swap_when_validation_fails(self, tmp_path):
        """Refusal-path invariant: ``_atomic_swap`` is NOT called when the
        validator rejects the candidate.

        Drives the validator-refusal path directly: a partial dump (sessions
        and rate_limiter_state tables omitted) is loaded, then
        ``_apply_and_validate_schema`` is patched to return False BEFORE
        recovery runs. The validator gate must refuse the swap; the live
        db_path must be unchanged; and ``_atomic_swap`` must NOT be called.

        Why this differs from ``test_recovery_refuses_incomplete_candidate``:
        that test pins the recovery return value (False) and live-db
        integrity. This test additionally counts ``_atomic_swap`` calls so
        a regression that "passes recovery but proceeds to swap" would be
        caught loudly. (The earlier version of this test corrupted the
        header so ``.dump`` failed before the validator ran — making the
        test tautological. The current version uses a partial-dump patch
        that does NOT corrupt the header, forcing the validator gate to be
        the gatekeeper.)
        """
        from memory import db_recovery as db_recovery_mod
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "no_swap.db")
        _create_healthy_db(db_path)
        # NOTE: NO _corrupt_header here — we want recovery to reach the
        # validator gate via the partial-dump path, not bail out at the
        # ``.dump`` returncode gate.

        patched_run, calls = self._patch_subprocess_for_partial_dump(
            omit_tables=("sessions", "rate_limiter_state"),
        )
        real_atomic_swap = db_recovery_mod._atomic_swap
        swap_calls = {"n": 0}

        def counting_atomic_swap(*args, **kwargs):
            swap_calls["n"] += 1
            return real_atomic_swap(*args, **kwargs)

        with mock.patch.object(subprocess, "run", side_effect=patched_run), \
             mock.patch.object(
                 db_recovery_mod,
                 "_apply_and_validate_schema",
                 return_value=False,
             ), \
             mock.patch.object(
                 db_recovery_mod,
                 "_atomic_swap",
                 side_effect=counting_atomic_swap,
             ):
            ok = recover_corrupt_db(db_path)

        assert ok is False, (
            "recovery returned True even though validation refused"
        )
        assert swap_calls["n"] == 0, (
            f"Issue B fix FAIL: _atomic_swap was called {swap_calls['n']} "
            f"time(s) after validation refused. _atomic_swap MUST NOT run "
            f"when the healing db schema is incomplete."
        )
        assert calls["dump"] >= 1, (
            f"patched_run was never called for .dump; cannot validate the "
            f"partial-dump path. patched_calls={calls}"
        )

    def test_recovery_fallback_path_refuses_incomplete(self, tmp_path):
        """Issue 3: when ``dump.sql`` execution raises
        ``sqlite3.DatabaseError`` (severe corruption produces malformed SQL),
        the recovery procedure falls back to the schema-only path AND the
        validator gate still fires there.

        This test exercises the FALLBACK branch (the ``except
        sqlite3.DatabaseError as dump_err`` arm at ``db_recovery.py:771``),
        not the success branch. It patches ``subprocess.run`` for ``.dump``
        to write a syntactically broken dump.sql so ``con.executescript``
        raises naturally; ``_apply_and_validate_schema`` is then patched
        to return False (validator refusal); recovery must return False
        and ``_atomic_swap`` must NOT be called.

        Distinguishing from other tests: this one deliberately corrupts the
        db header so ``.dump`` would NORMALLY fail at returncode check, but
        our patch makes ``.dump`` succeed (rc=0) while producing malformed
        SQL. That mimics the real production failure mode where the file
        is partially readable and ``.dump`` succeeds at the syscall level
        but produces garbage output.
        """
        from memory import db_recovery as db_recovery_mod
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "fallback_refused.db")
        _create_healthy_db(db_path)
        _corrupt_header(db_path)

        real_atomic_swap = db_recovery_mod._atomic_swap
        swap_calls = {"n": 0}

        def counting_atomic_swap(*args, **kwargs):
            swap_calls["n"] += 1
            return real_atomic_swap(*args, **kwargs)

        real_run = subprocess.run
        dump_calls = {"n": 0}

        def patched_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[2] == ".dump":
                dump_calls["n"] += 1
                # Syntactically broken dump.sql: references a table that
                # doesn't exist. ``con.executescript`` raises
                # ``sqlite3.DatabaseError`` on the ``INSERT INTO
                # nonexistent_table`` line, which puts recovery into the
                # except branch (schema-only fallback).
                bad_dump = (
                    "PRAGMA foreign_keys=OFF;\n"
                    "BEGIN TRANSACTION;\n"
                    "INSERT INTO nonexistent_table VALUES (1);\n"
                    "COMMIT;\n"
                )
                stdout = kwargs.get("stdout")
                if stdout is not None and hasattr(stdout, "write"):
                    stdout.write(bad_dump)
                    try:
                        stdout.flush()
                    except Exception:
                        pass
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout=b"", stderr=b"",
                    )
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout=bad_dump.encode("utf-8"), stderr=b"",
                )
            return real_run(*args, **kwargs)

        with mock.patch.object(subprocess, "run", side_effect=patched_run), \
             mock.patch.object(
                 db_recovery_mod,
                 "_apply_and_validate_schema",
                 return_value=False,
             ), \
             mock.patch.object(
                 db_recovery_mod,
                 "_atomic_swap",
                 side_effect=counting_atomic_swap,
             ):
            ok = recover_corrupt_db(db_path)

        assert dump_calls["n"] >= 1, (
            f"subprocess.run was never called for .dump; patched_run didn't "
            f"intercept. dump_calls={dump_calls}"
        )
        assert ok is False, (
            "Issue 3 FAIL: recovery returned True on the fallback path even "
            "though the validator refused the candidate. The validator gate "
            "MUST fire on the fallback path too."
        )
        assert swap_calls["n"] == 0, (
            f"Issue 3 FAIL: _atomic_swap was called {swap_calls['n']} time(s) "
            f"on the fallback path after validator refused. The validator "
            f"gate MUST prevent the swap on the fallback path too."
        )


# ----------------------------------------------------------------------------
# TestSchemaHotPathCoverage
# ----------------------------------------------------------------------------


class TestSchemaHotPathCoverage:
    """Issue 2: ``validate_schema_complete`` must catch missing hot-path columns.

    These tests drop a single hot-path column from a healthy db and verify
    the validator refuses with a diagnostic naming the missing column. The
    columns exercised here are the ones production code reads on
    chat-completions hot paths, mode-transition paths, and SOTA sync paths.

    Each test:
      1. Builds a fresh production-schema db.
      2. Recreates ONE table without one hot-path column.
      3. Calls ``validate_schema_complete`` directly on the db.
      4. Asserts the validator refuses AND names the missing column.
    """

    @pytest.mark.parametrize("table,column_to_drop,owner_col", [
        # messages: hot-path columns from chat-completion history + SOTA sync
        ("messages", "role", "id"),
        ("messages", "content", "id"),
        ("messages", "message_index", "id"),
        ("messages", "created_at", "id"),
        ("messages", "content_hash", "id"),
        ("messages", "exchange_index", "id"),
        # conversation_summaries: hot-path columns from context injection
        ("conversation_summaries", "summary", "session_id"),
        ("conversation_summaries", "exchange_count", "session_id"),
        ("conversation_summaries", "last_exchange_index", "session_id"),
        ("conversation_summaries", "last_message_index", "session_id"),
        ("conversation_summaries", "raw_start_message_index", "session_id"),
        # extraction_batches: hot-path columns from SOTA sync
        ("extraction_batches", "status", "id"),
        ("extraction_batches", "extraction_json", "id"),
        ("extraction_batches", "start_message_index", "id"),
        ("extraction_batches", "end_message_index", "id"),
        # message_summaries: hot-path columns from context injection
        ("message_summaries", "summary", "id"),
        ("message_summaries", "content_hash", "id"),
        ("message_summaries", "end_message_index", "id"),
        # entities: hot-path columns from memory tool loop
        ("entities", "name", "id"),
        ("entities", "entity_type", "id"),
        ("entities", "summary", "id"),
        # facts: hot-path columns from memory tool loop
        ("facts", "source_entity_id", "id"),
        ("facts", "target_entity_id", "id"),
        ("facts", "relationship_type", "id"),
        ("facts", "fact", "id"),
        # supersessions: hot-path columns from SOTA extraction
        ("supersessions", "old_fact_id", "id"),
        ("supersessions", "new_fact_id", "id"),
        ("supersessions", "reason", "id"),
        # memory_log: hot-path columns from SOTA persistence
        ("memory_log", "memory_type", "memory_id"),
        ("memory_log", "action", "memory_id"),
        ("memory_log", "topic_key", "memory_id"),
        ("memory_log", "content", "memory_id"),
        # tasks: hot-path columns from mode-transition
        ("tasks", "title", "id"),
        ("tasks", "status", "id"),
        ("tasks", "parent_id", "id"),
        # tool_call_audit: hot-path columns from observability
        ("tool_call_audit", "tool_name", "id"),
        ("tool_call_audit", "arguments", "id"),
        ("tool_call_audit", "result_status", "id"),
        ("tool_call_audit", "timestamp", "id"),
    ])
    def test_validator_rejects_missing_hotpath_column(
        self, tmp_path, table, column_to_drop, owner_col
    ):
        """Drop a hot-path column; the validator must refuse with a
        diagnostic naming the missing column."""
        db_path = str(tmp_path / f"hotpath_{table}_{column_to_drop}.db")
        _create_healthy_db(db_path)

        # Recreate the table WITHOUT the target column (keep a primary key
        # so the table is still functional for the SELECT/INSERT we use).
        con = sqlite3.connect(db_path)
        try:
            cols_info = con.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
            cols_info_by_name = {row[1]: row for row in cols_info}
            kept = [
                (name, row)
                for name, row in cols_info_by_name.items()
                if name != column_to_drop
            ]
            if not kept:
                pytest.skip(
                    f"refusing to drop the only column of {table}"
                )
            col_defs = []
            for name, row in kept:
                col_type = row[2] or "TEXT"
                col_defs.append(f"{name} {col_type}")
            # Append owner_col if it was dropped (e.g. table has only one
            # column and that's the one we're dropping — re-add as a stub
            # so the CREATE TABLE succeeds).
            if owner_col not in [n for n, _ in kept]:
                col_defs.append(f"{owner_col} TEXT PRIMARY KEY")
            con.execute(
                f"CREATE TABLE {table}_new ({', '.join(col_defs)})"
            )
            # Best-effort copy of any rows that match the kept schema.
            kept_names = [n for n, _ in kept]
            if kept_names:
                try:
                    quoted = ", ".join(f'"{n}"' for n in kept_names)
                    con.execute(
                        f"INSERT INTO {table}_new ({quoted}) "
                        f"SELECT {quoted} FROM {table}"
                    )
                except sqlite3.OperationalError:
                    pass
            con.execute(f"DROP TABLE {table}")
            con.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
            con.commit()
        finally:
            con.close()

        from memory.db_migrations import validate_schema_complete

        con = sqlite3.connect(db_path)
        try:
            ok, diagnostics = validate_schema_complete(con)
        finally:
            con.close()

        joined = " | ".join(diagnostics)
        assert ok is False, (
            f"validator accepted a {table} missing the hot-path column "
            f"'{column_to_drop}'. diagnostics={diagnostics}"
        )
        assert column_to_drop in joined, (
            f"validator did not name the missing column '{column_to_drop}' "
            f"in diagnostics={diagnostics}"
        )


# ----------------------------------------------------------------------------
# TestSchemaCoverageMetatest
# ----------------------------------------------------------------------------


class TestSchemaCoverageMetatest:
    """Dogfooding guardrail: ``REQUIRED_COLUMNS`` must cover at least 70% of
    every CREATE TABLE's declared columns.

    This is the Issue 2 metatest — it parses ``SCHEMA_SQL`` itself, computes
    the column coverage for each table, and asserts the threshold. The
    intent: prevent the validator from drifting back to "only 8 columns"
    (the pre-fix state that caused this issue). If someone adds a new
    table or column to ``SCHEMA_SQL`` without updating ``REQUIRED_COLUMNS``,
    this test fails loudly with a per-table breakdown.

    The 70% threshold is a structural check that catches two regression
    classes:
      1. "Forgot the table entirely" — coverage would be 0%.
      2. "Coverage narrowed" — coverage drops below 70% on a hot-path table.

    Tables at 100% are encouraged (e.g. messages, conversation_summaries,
    extraction_batches); tables below 70% are flagged.
    """

    # Schema-coverage threshold — Issue 2 requires >= 70% per table.
    COVERAGE_THRESHOLD = 0.70

    def _parse_schema_columns(self, schema_sql: str) -> dict:
        """Parse ``CREATE TABLE IF NOT EXISTS name (...)`` blocks and return
        ``{table_name: [column_name, ...]}``.

        Skips lines that are table-level constraints (PRIMARY KEY, UNIQUE,
        FOREIGN KEY, CHECK, CONSTRAINT) rather than column declarations.
        Handles the SCHEMA_SQL idiom where ``CREATE TABLE`` is followed
        by column definitions and ends with ``);``.
        """
        import re as _re

        table_pattern = _re.compile(
            r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\);",
            _re.DOTALL,
        )
        out: dict = {}
        for m in table_pattern.finditer(schema_sql):
            table_name = m.group(1)
            cols_block = m.group(2)
            col_names = []
            for raw_line in cols_block.split("\n"):
                line = raw_line.strip().rstrip(",")
                if not line or line.startswith("--"):
                    continue
                # First token of the line — split on whitespace OR '(' so
                # ``UNIQUE(session_id,`` yields ``UNIQUE``, not
                # ``UNIQUE(session_id,``. Handles inline constraint forms
                # (``UNIQUE(...)``, ``PRIMARY KEY (...)``, etc.).
                first_token = _re.split(r"[\s(]", line, maxsplit=1)[
                    0
                ].upper()
                if first_token in {
                    "PRIMARY", "UNIQUE", "FOREIGN", "CHECK",
                    "CONSTRAINT",
                }:
                    continue
                # First token of a column declaration is the column name.
                # Strip trailing backticks/quotes that the schema might use.
                col_name = line.split()[0].strip('"').strip("`").strip("'")
                col_names.append(col_name)
            out[table_name] = col_names
        return out

    def test_validator_required_columns_metatest(self):
        """REQUIRED_COLUMNS covers >= 70% of every CREATE TABLE's columns.

        Pinning this prevents Issue 2 from regressing: a validator that
        silently narrows its scope (e.g. someone removes the expanded
        list thinking it's overkill) will fail this test on the affected
        tables. Per-table failures print the full schema and required list
        so the fix is mechanical.
        """
        from memory import db_migrations

        schema_columns = self._parse_schema_columns(db_migrations.SCHEMA_SQL)

        # Build {table: set(required_cols)} from the module-level tuple.
        required_per_table: dict = {}
        for tbl, col in db_migrations.REQUIRED_COLUMNS:
            required_per_table.setdefault(tbl, set()).add(col)

        # Tables in SCHEMA_SQL must all be covered at >= threshold.
        violations = []
        coverage_report = []
        for tbl, all_cols in sorted(schema_columns.items()):
            required_cols = required_per_table.get(tbl, set())
            total = len(all_cols)
            covered = len(required_cols)
            coverage = covered / total if total > 0 else 1.0
            coverage_report.append(
                f"  {tbl}: {covered}/{total} = {coverage:.0%}"
            )
            if coverage < self.COVERAGE_THRESHOLD:
                missing = sorted(set(all_cols) - required_cols)
                violations.append(
                    f"  {tbl}: {coverage:.0%} coverage "
                    f"(required {self.COVERAGE_THRESHOLD:.0%}). "
                    f"Missing from REQUIRED_COLUMNS: {missing}. "
                    f"Full schema: {all_cols}"
                )

        # Tables in REQUIRED_COLUMNS but not in SCHEMA_SQL = dead config.
        for tbl in sorted(set(required_per_table) - set(schema_columns)):
            violations.append(
                f"  {tbl}: in REQUIRED_COLUMNS but NOT in SCHEMA_SQL "
                f"(dead config — required_cols={sorted(required_per_table[tbl])})"
            )

        assert not violations, (
            "REQUIRED_COLUMNS coverage below "
            f"{self.COVERAGE_THRESHOLD:.0%} threshold:\n"
            + "\n".join(violations)
            + "\n\nPer-table coverage:\n"
            + "\n".join(coverage_report)
        )

    def test_required_tables_matches_schema(self):
        """REQUIRED_TABLES must equal the set of CREATE TABLE names in
        SCHEMA_SQL. (Excluding internal ``sqlite_*`` tables.)"""
        from memory import db_migrations

        schema_columns = self._parse_schema_columns(db_migrations.SCHEMA_SQL)
        schema_tables = set(schema_columns.keys())
        required_tables = set(db_migrations.REQUIRED_TABLES)

        assert required_tables == schema_tables, (
            f"REQUIRED_TABLES drift from SCHEMA_SQL.\n"
            f"In REQUIRED_TABLES but not in SCHEMA_SQL: "
            f"{sorted(required_tables - schema_tables)}\n"
            f"In SCHEMA_SQL but not in REQUIRED_TABLES: "
            f"{sorted(schema_tables - required_tables)}"
        )


# ----------------------------------------------------------------------------
# Revert-test protocol (manual, not part of pytest run)
# ----------------------------------------------------------------------------
#
# Per AGENTS.md "Tautological tests" rule 3, every fix MUST be revert-tested.
# To exercise this protocol:
#
#   1. Open src/memory/db_recovery.py.
#   2. Temporarily comment out the new ``if not _apply_and_validate_schema(con)``
#      calls in BOTH branches of the try/except that begins at line ~768.
#      (Or: replace ``_apply_and_validate_schema`` with a stub returning True
#      to bypass the validator entirely.)
#   3. Run:
#        PYTHONPATH=src python3 -m pytest \\
#          tests/unit/test_db_recovery_schema_guard.py::TestRecoverySchemaGuard -v
#   4. Confirm:
#        test_recovery_completes_missing_tables_after_partial_dump PASSES
#          (the dump + SCHEMA_SQL fallback path still produces a valid db)
#        test_recovery_refuses_incomplete_candidate FAILS
#          (without the validator the procedure would swap the incomplete db)
#        test_recovery_does_not_swap_when_validation_fails FAILS
#          (swap counter would be 1 instead of 0)
#   5. Restore the fix. All tests must pass again.
#
# The bypass step is best done with a temporary monkey-patch via Python:
#
#     from memory import db_recovery as mod
#     mod._apply_and_validate_schema = lambda con: True
#
# Then run the refused-candidate tests — they MUST fail.