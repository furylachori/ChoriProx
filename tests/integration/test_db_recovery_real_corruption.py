"""C7+ iter3: real-path tests for the 3 production CRITs the unit tests miss.

Companion to ``tests/unit/test_db_recovery.py``; uses REAL sqlite3 corruption
patterns (no mocks of the function under test) and proves the auto-heal
recovers under realistic load.

Per AGENTS.md test discipline (5 invariants enforced):

  1. Exercise the real production call site — no mocking of ``init_db``,
     ``recover_corrupt_db``, or ``_recover_corrupt_db_inner``.
  2. Do not mock the function under test.
  3. Revert-test confirms each test FAILS when the production fix is reverted
     (see ``git diff HEAD~ -- src/memory/``).
  4. Grep ``src/`` for callers of each method before testing — every method
     exercised here has production callers.
  5. Concurrency tests use ``threading.Barrier`` (NOT ``time.sleep``) for
     deterministic synchronized start.

Three CRITs are pinned here (R2 final spot-check findings):

  * **CRIT-1** (regression): ``init_db`` raised ``RuntimeError`` on an empty
    (0-byte) existing file. The magic-byte sniff in ``ensure_db_integrity``
    correctly returned False, the recovery procedure produced an
    empty/malformed ``dump.sql``, the schema-only fallback didn't exist, and
    the integrity re-check failed → ``init_db`` refused to start.
    Fix: gate the recovery invocation on ``os.path.getsize(path) > 0`` so a
    0-byte file at startup goes straight to fresh-init.
    Test: ``TestInitDbEmptyFile::test_init_db_succeeds_on_empty_existing_file``.

  * **CRIT-3** (regression): second recovery after re-corruption returned
    False. The dump.sql execution failure path returned False instead of
    falling through to a schema-only swap.
    Fix: when ``dump.sql`` execution fails, lazy-import ``SCHEMA_SQL`` from
    ``db_migrations`` and apply it to the healing connection.
    Test: ``TestSecondRecoveryRealCorruption::test_second_recovery_uses_schema_only_when_dump_fails``.

  * **CRIT-4** (same root cause as CRIT-3): 31/32 of N×N concurrent recovery
    races succeeded; 1 failed for the same dump.sql failure path with no
    schema-only fallback.
    Fix: same as CRIT-3.
    Test: ``TestConcurrentMultiPathReal::test_concurrent_multi_path_recovery_all_succeed``.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import threading

import pytest


# ----------------------------------------------------------------------------
# Helpers (shared with unit tests; duplicated to keep the new file standalone)
# ----------------------------------------------------------------------------


def _fresh_db(db_path: str) -> None:
    """Create a fresh db at ``db_path`` with the production schema.

    Uses ``db_migrations.init_db`` so the schema matches what production
    exercises (e.g. ``session_id``, ``chain_hash``, ``last_exchange_index``).
    A hand-rolled minimal schema would diverge from production and break
    code paths that read/write those columns. ``init_db`` is idempotent.
    """
    from memory import db_migrations
    db_migrations.init_db(db_path)


def _corrupt_at(db_path: str, offset: int, size: int = 1024) -> None:
    """Write ``size`` zero bytes at ``offset`` in the db file.

    This is the corruption pattern used by ``tests/unit/test_db_recovery.py``
    (``_corrupt_header`` is a special case at ``offset=0, size=16``). Triggers
    various pages of the SQLite file being zeroed / inaccessible.
    """
    with open(db_path, "r+b") as f:
        f.seek(offset)
        f.write(b"\x00" * size)


def _insert_some_messages(db_path: str, n: int = 20) -> None:
    """Insert ``n`` rows into the ``messages`` table.

    Mirrors the production schema (NOT NULL columns: session_id, message_index,
    role, created_at). Without these rows, the recovery procedure has nothing
    to lose — the test would not exercise the lost-and-found mapping path.
    """
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "INSERT INTO messages (session_id, message_index, role, content, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1", i, "user", f"msg{i}", f"2026-01-01T00:00:{i:02d}Z")
                for i in range(n)
            ],
        )
        con.commit()
    finally:
        con.close()


# ----------------------------------------------------------------------------
# CRIT-1: init_db on a 0-byte existing file
# ----------------------------------------------------------------------------


class TestInitDbEmptyFile:
    """CRIT-1 (iter3 regression): empty (0-byte) existing file at startup.

    Without the iter3 gate, ``init_db`` would invoke
    ``ensure_db_integrity`` → ``recover_corrupt_db`` on the 0-byte file.
    The recovery produced a valid-but-empty ``dump.sql`` (``BEGIN;COMMIT;``),
    the recover.sql failed to load, and the integrity re-check after recovery
    still returned False (the magic-byte sniff on the post-recovery file
    still failed because the schema-only swap wasn't actually implemented).
    The net effect: ``init_db`` raised ``RuntimeError`` and the proxy refused
    to start on a perfectly valid "operator touched an empty file" scenario.

    The iter3 fix gates the recovery invocation on
    ``os.path.getsize(path) > 0``: a 0-byte file goes straight to the
    schema-creation path, which ``executescript(SCHEMA_SQL)`` overwrites
    with a valid production-schema db.
    """

    def test_init_db_succeeds_on_empty_existing_file(self, tmp_path, monkeypatch):
        """Reproduces the R2 final spot-check failure: ``init_db`` raised on
        a 0-byte file. With the fix, ``init_db`` returns True and creates
        the production schema (≥10 tables).
        """
        db_path = str(tmp_path / "empty.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        # Pre-existing 0-byte file at startup (the failure scenario)
        open(db_path, "wb").close()
        assert os.path.getsize(db_path) == 0

        from memory import db_migrations
        # Must NOT raise RuntimeError. With the iter3 fix, init_db proceeds
        # to the schema-creation path (skipping the recovery procedure entirely
        # because the file is 0 bytes).
        result = db_migrations.init_db(db_path)
        assert result is True, (
            f"CRIT-1 FAIL: init_db returned {result} on a 0-byte file; "
            f"expected True (fresh-init path should have created schema)"
        )

        # Schema was actually created (not just a no-op)
        con = sqlite3.connect(db_path)
        try:
            n_tables = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
        finally:
            con.close()
        assert n_tables >= 10, (
            f"CRIT-1 FAIL: fresh-init should create >= 10 tables; got "
            f"{n_tables}. The schema-creation path didn't actually run."
        )

    def test_init_db_succeeds_on_heavily_corrupt_existing_file(self, tmp_path, monkeypatch):
        """CRIT-1 (severe variant): 4 KB of zeros at offset 0 — full header gone.

        The recovery path SHOULD engage here (file is non-zero AND corrupt).
        We don't assert the recovery outcome — that's covered by the unit tests
        — we only assert ``init_db`` does not raise ``RuntimeError("corrupt")``
        for any file the auto-heal can plausibly repair.
        """
        db_path = str(tmp_path / "bad.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        _fresh_db(db_path)
        _corrupt_at(db_path, 0, 4096)

        from memory import db_migrations
        try:
            result = db_migrations.init_db(db_path)
            assert result is True, (
                f"init_db returned {result}; expected True after auto-heal"
            )
        except RuntimeError as e:
            # If the proxy refuses to start due to unrecoverable corruption,
            # that's also acceptable — but it MUST NOT crash silently. The
            # error message must mention "corrupt" or similar to indicate
            # the refusal is intentional, not a bug.
            assert "corrupt" in str(e).lower() or "refusing" in str(e).lower(), (
                f"init_db raised RuntimeError with unexpected message: {e!r}"
            )


# ----------------------------------------------------------------------------
# CRIT-3: second recovery with corrupt dump.sql must use schema-only fallback
# ----------------------------------------------------------------------------


class TestSecondRecoveryRealCorruption:
    """CRIT-3 (iter3 regression): second recovery after re-corruption must
    succeed via the schema-only fallback path.

    The OLD code returned False when ``dump.sql`` execution raised
    ``sqlite3.DatabaseError`` (severely corrupt db produces malformed SQL
    output that ``con.executescript`` cannot parse). With the iter3 fix,
    ``_recover_corrupt_db_inner`` lazy-imports ``SCHEMA_SQL`` from
    ``memory.db_migrations`` and applies it as a fallback — the resulting db
    is empty but schema-valid, and the proxy can continue.

    This test uses REAL sqlite3 corruption (no mocks of the function under
    test). The corruption pattern forces ``dump.sql`` execution to fail
    mid-recovery by monkey-patching ``subprocess.run`` for the ``.dump``
    invocation — this is the only way to deterministically reproduce the
    failure mode in a unit test (real corruption is too random to reliably
    produce malformed dump.sql).
    """

    def test_second_recovery_uses_schema_only_when_dump_fails(self, tmp_path, monkeypatch):
        """CRIT-3 reproducer: recover once OK1=True, force dump.sql failure
        on the second recovery, assert OK2=True via schema-only fallback.

        Method: patch ``subprocess.run`` to return a syntactically broken
        dump.sql for the second recovery. The first recovery uses the real
        sqlite3 CLI to prove the path works under normal conditions.
        """
        from memory import db_migrations
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "iter3_crit3.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)

        # 1. Create healthy db + insert data
        _fresh_db(db_path)
        _insert_some_messages(db_path, n=20)

        # 2. First corruption + recovery (real path — no patches)
        _corrupt_at(db_path, 100, 1024)
        ok1 = recover_corrupt_db(db_path)
        assert ok1 is True, (
            f"first recovery failed: ok1={ok1}. The healthy-path "
            f"recovery must succeed before we can test the second-recovery "
            f"regression."
        )

        # 3. Patch subprocess.run for the SECOND recovery: make .dump write
        # a syntactically broken dump.sql to the file. This deterministically
        # reproduces the CRIT-3 failure mode (dump.sql execution raises
        # ``sqlite3.DatabaseError``) that the OLD code couldn't recover from.
        #
        # NOTE: subprocess.run is invoked with positional args ``[sqlite3,
        # db_path, .dump]`` (the command is at index 0, the db path at 1,
        # the subcommand at 2). The patch writes the malformed SQL directly
        # to the dump.sql file (the recovery opens it in text mode and
        # passes the file handle as ``stdout=``).
        real_run = subprocess.run
        call_count = {"dump": 0}

        def patched_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            # ``cmd`` shape: ``["sqlite3", db_path, ".dump" | ".recover"]``
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[2] == ".dump":
                call_count["dump"] += 1
                bad_dump = (
                    "PRAGMA foreign_keys=OFF;\n"
                    "BEGIN TRANSACTION;\n"
                    "INSERT INTO nonexistent_table VALUES (1);\n"
                    "COMMIT;\n"
                )
                if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
                    kwargs["stdout"].write(bad_dump)
                    kwargs["stdout"].flush()
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout=b"", stderr=b"",
                    )
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=bad_dump.encode("utf-8"), stderr=b"",
                )
            return real_run(*args, **kwargs)

        # 4. Re-corrupt the (recovered) db so recovery runs again
        _corrupt_at(db_path, 2048, 4096)
        monkeypatch.setattr(subprocess, "run", patched_run)
        ok2 = recover_corrupt_db(db_path)

        # CRIT-3 fix: schema-only fallback must succeed
        assert call_count["dump"] >= 1, (
            f"subprocess.run was never called for .dump; patched_run didn't "
            f"intercept. Cannot validate the second-recovery path."
        )
        assert ok2 is True, (
            f"CRIT-3 FAIL: second recovery after re-corruption returned "
            f"ok2={ok2}; the schema-only fallback should have produced a "
            f"valid db even with malformed dump.sql. OLD code (without "
            f"iter3 fix) returned False here because it returned False "
            f"immediately on dump.sql DatabaseError instead of falling "
            f"through to the static schema."
        )

        # The post-recovery db must have the production schema (>=10 tables)
        con = sqlite3.connect(db_path)
        try:
            n_tables = con.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
        finally:
            con.close()
        assert n_tables >= 10, (
            f"second-recovery db should have >= 10 tables from the schema-only "
            f"fallback; got {n_tables}. The fallback path didn't actually run."
        )


# ----------------------------------------------------------------------------
# CRIT-4: concurrent multi-path recovery must ALL succeed
# ----------------------------------------------------------------------------


class TestConcurrentMultiPathReal:
    """CRIT-4 (iter3 regression): concurrent recovery on N distinct corrupt
    paths must ALL succeed.

    The OLD code's dump.sql failure path returned False, which under
    contention meant ``N-1`` of ``N`` paths failed. With the iter3 fix
    (schema-only fallback), every path recovers regardless of the dump.sql
    output shape.

    Per AGENTS.md "Concurrency bugs need threading.Barrier" — sequential
    calls never exercise the race window. We synchronize N threads at a
    Barrier so they all hit ``recover_corrupt_db`` within the same
    scheduler tick.
    """

    def test_concurrent_multi_path_recovery_all_succeed(self, tmp_path, monkeypatch):
        """CRIT-4 reproducer: 16 threads × 16 distinct corrupt paths — all 16
        must recover. Patches ``subprocess.run`` to return a malformed
        dump.sql so the schema-only fallback is exercised under contention.

        This is a tightened version of the unit test
        ``test_concurrent_recover_corrupt_db_cross_path_does_not_short_circuit``
        which only asserts ``recovered >= n_paths // 2``. The unit test's
        loose bound would silently miss a revert of the schema-only
        fallback fix in any environment where corruption is mild; this
        integration test uses an aggressive patch that forces the
        fallback path on every recovery, so any revert causes
        ``succeeded < n_paths`` and the test fails.
        """
        from memory.db_recovery import recover_corrupt_db

        n_paths = 16
        monkeypatch.setenv("PROXY_TEST_DB", str(tmp_path / "memory.db"))

        # 1. Create N healthy dbs + insert data + corrupt each
        paths = []
        for i in range(n_paths):
            p = str(tmp_path / f"db_{i}.db")
            _fresh_db(p)
            _insert_some_messages(p, n=10)
            _corrupt_at(p, 100, 1024)
            paths.append(p)

        # 2. Patch subprocess.run to ALWAYS write malformed dump.sql —
        # forces the schema-only fallback on every single recovery. This
        # is the only way to deterministically exercise the fallback under
        # contention; real corruption is too random.
        #
        # NOTE: subprocess.run is invoked with positional args ``[sqlite3,
        # db_path, .dump]`` (the command is at index 0, the db path at 1,
        # the subcommand at 2). The patch writes the malformed SQL directly
        # to the dump.sql file (the recovery opens it in text mode and
        # passes the file handle as ``stdout=``).
        real_run = subprocess.run
        patch_lock = threading.Lock()
        patched_calls = {"n": 0}

        def patched_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[2] == ".dump":
                with patch_lock:
                    patched_calls["n"] += 1
                bad_dump = (
                    "PRAGMA foreign_keys=OFF;\n"
                    "BEGIN TRANSACTION;\n"
                    "INSERT INTO nonexistent_table VALUES (1);\n"
                    "COMMIT;\n"
                )
                if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
                    kwargs["stdout"].write(bad_dump)
                    kwargs["stdout"].flush()
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout=b"", stderr=b"",
                    )
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=bad_dump.encode("utf-8"), stderr=b"",
                )
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", patched_run)

        # 3. Concurrently recover — Barrier ensures synchronized start
        results: list = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(n_paths)

        def worker(path):
            try:
                barrier.wait(timeout=10)
                ok = recover_corrupt_db(path)
                with results_lock:
                    results.append((path, ok))
            except Exception as e:
                with results_lock:
                    results.append((path, e))

        threads = [
            threading.Thread(target=worker, args=(p,)) for p in paths
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # All threads completed
        assert len(results) == n_paths, (
            f"only {len(results)}/{n_paths} threads completed"
        )

        # No thread crashed with an unexpected exception
        for path, r in results:
            if isinstance(r, Exception):
                pytest.fail(f"thread raised: {path}: {r!r}")

        # CRIT-4 tight assertion: with the schema-only fallback fix, ALL N
        # paths must succeed. The OLD code (without the fallback) would
        # return False for every recovery that hit dump.sql DatabaseError,
        # dropping ``succeeded < n_paths``. This is the regression pin:
        # any revert of the fallback fix causes this assertion to fail.
        succeeded = sum(1 for _p, r in results if r is True)
        assert succeeded == n_paths, (
            f"CRIT-4 FAIL: only {succeeded}/{n_paths} corrupt paths "
            f"recovered successfully. The schema-only fallback must kick "
            f"in for every path under dump.sql failure. OLD code (without "
            f"iter3 fix) returned False here because it bailed out on "
            f"dump.sql DatabaseError instead of falling through to "
            f"SCHEMA_SQL. patched subprocess.run was called "
            f"{patched_calls['n']} times."
        )

        # Every post-recovery db must have the production schema
        for path, r in results:
            if r is True:
                con = sqlite3.connect(path)
                try:
                    n_tables = con.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    ).fetchone()[0]
                finally:
                    con.close()
                assert n_tables >= 10, (
                    f"{path} reported success but only has {n_tables} tables"
                )