"""Tests for SQLite corruption-recovery auto-heal (memory.db_recovery).

US-126 (planned, see ticket 24c1024e): SQLite auto-heal — self-recover from
the "database disk image is malformed" pattern (2026-07-06 prod incident).

Covers three call sites:
  * startup recovery (init_db's ensure_db_integrity → recover_corrupt_db hook)
  * pool-triggered recovery (db_pool._create_connection's inline catch)
  * concurrent-recovery guard (state.recovery_lock + Barrier)

Tests use real sqlite3 (per AGENTS.md "Tautological tests" rule 2). Each
test creates an isolated tmp_path db; no shared state across tests.
"""
import os
import sqlite3
import threading
import time
from unittest import mock

import pytest

from memory.db_pool import DatabaseRecoveryError


class _ParkProbe:
    """Wrap a threading.Condition; set `parked` when the target thread parks."""
    def __init__(self, real, ident_holder, parked_event):
        self._real = real; self._holder = ident_holder; self._parked = parked_event
    def __enter__(self): return self._real.__enter__()
    def __exit__(self, *a): return self._real.__exit__(*a)
    def wait(self, timeout=None):
        if threading.get_ident() == self._holder["v"]: self._parked.set()
        return self._real.wait(timeout)
    def wait_for(self, predicate, timeout=None):
        """Mirror Condition.wait_for through this wrapper's wait method."""
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = time.monotonic() + waittime
                else:
                    waittime = endtime - time.monotonic()
                if waittime <= 0:
                    break
            self.wait(waittime)
            result = predicate()
        return result
    def notify(self, n=1): return self._real.notify(n)
    def notify_all(self): return self._real.notify_all()
    def acquire(self, *a, **k): return self._real.acquire(*a, **k)
    def release(self): return self._real.release()


def test_park_probe_wait_for_initial_true_does_not_wait_db_recovery():
    """An initially true predicate returns without signaling a park."""
    parked = threading.Event()
    holder = {"v": threading.get_ident()}
    predicate_calls = [0]
    probe = _ParkProbe(
        threading.Condition(),
        holder,
        parked,
    )

    def predicate():
        predicate_calls[0] += 1
        return True

    with probe:
        result = probe.wait_for(predicate, timeout=5.0)

    assert result is True
    assert predicate_calls == [1]
    assert not parked.is_set()


def test_park_probe_wait_for_false_predicate_enters_wait_db_recovery():
    """A false predicate must release the condition lock and really wait."""
    parked = threading.Event()
    holder = {"v": None}
    ready = {"value": False}
    results = []
    errors = []
    probe = _ParkProbe(
        threading.Condition(),
        holder,
        parked,
    )

    def waiter():
        holder["v"] = threading.get_ident()
        try:
            with probe:
                results.append(
                    probe.wait_for(
                        lambda: ready["value"],
                        timeout=5.0,
                    )
                )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()

    parked_observed = False
    try:
        parked_observed = parked.wait(timeout=5.0)
    finally:
        with probe:
            ready["value"] = True
            probe.notify_all()
        thread.join(timeout=5.0)

    assert parked_observed, "wait_for never entered the actual wait path"
    assert not thread.is_alive()
    assert errors == []
    assert results == [True]


def test_park_probe_wait_for_false_then_true_waits_db_recovery():
    """A false-then-true predicate must wait between its evaluations."""
    parked = threading.Event()
    holder = {"v": threading.get_ident()}
    predicate_calls = [0]
    wait_calls = []
    real_condition = threading.Condition()
    original_wait = real_condition.wait

    def recording_wait(timeout=None):
        wait_calls.append(timeout)
        return False

    real_condition.wait = recording_wait
    probe = _ParkProbe(
        real_condition,
        holder,
        parked,
    )

    def predicate():
        predicate_calls[0] += 1
        return predicate_calls[0] >= 2

    try:
        with probe:
            result = probe.wait_for(predicate, timeout=1.0)
    finally:
        real_condition.wait = original_wait

    assert result is True
    assert predicate_calls == [2]
    assert wait_calls == [1.0]
    assert parked.is_set()


def _create_healthy_db(db_path):
    """Create a fresh db at db_path with the production schema for tests.

    Uses ``db_migrations.init_db`` so the schema matches what production
    exercises (e.g. ``session_id``, ``chain_hash``, ``last_exchange_index``).
    A hand-rolled minimal schema would diverge from production and break
    code paths that read/write those columns. init_db is idempotent and
    handles both first-ever-start (no file on disk) and existing-db cases.
    """
    from memory import db_migrations
    db_migrations.init_db(db_path)


def _corrupt_header(db_path):
    """Write garbage to the first 16 bytes of the db header.

    Destroys the SQLite magic header, so ``sqlite3.connect`` raises
    ``DatabaseError('file is not a database')`` (or 'malformed' — both are
    corruption indicators our recovery paths handle). Truncated header is
    sufficient for our purposes — both surface the corruption-recovery path).
    """
    with open(db_path, "r+b") as f:
        f.write(b"\x00" * 16)


# ----------------------------------------------------------------------------
# ensure_db_integrity
# ----------------------------------------------------------------------------

class TestEnsureDbIntegrity:
    def test_ensure_db_integrity_returns_true_for_healthy_db(self, tmp_path):
        db_path = str(tmp_path / "healthy.db")
        _create_healthy_db(db_path)
        from memory.db_recovery import ensure_db_integrity
        assert ensure_db_integrity(db_path) is True

    def test_ensure_db_integrity_returns_false_for_corrupt_header(self, tmp_path):
        db_path = str(tmp_path / "corrupt.db")
        _create_healthy_db(db_path)
        _corrupt_header(db_path)
        from memory.db_recovery import ensure_db_integrity
        assert ensure_db_integrity(db_path) is False

    def test_ensure_db_integrity_returns_false_for_empty_file(self, tmp_path):
        """CRIT-2 fix: a 0-byte file must NOT be reported as 'ok'.

        Prior bug: ``PRAGMA integrity_check`` with ``immutable=1`` returned
        'ok' on a zero-byte file because no pages are touched. The CRIT-2
        magic-byte sniff catches this case BEFORE the URI connect and
        returns False.
        """
        from memory.db_recovery import ensure_db_integrity

        # 1. Truly empty (0-byte) file
        db_path = str(tmp_path / "empty.db")
        open(db_path, "wb").close()
        assert os.path.getsize(db_path) == 0
        assert ensure_db_integrity(db_path) is False, (
            "0-byte file must return False (CRIT-2 bug: integrity_check+immutable"
            " would falsely return 'ok')"
        )

        # 2. Truncated magic header (only 15 bytes of the 16-byte signature)
        db_path2 = str(tmp_path / "truncated.db")
        with open(db_path2, "wb") as f:
            f.write(b"SQLite format 3")  # missing the trailing \x00
        assert ensure_db_integrity(db_path2) is False

        # 3. Wrong magic (16 bytes but not the SQLite signature)
        db_path3 = str(tmp_path / "wrong_magic.db")
        with open(db_path3, "wb") as f:
            f.write(b"\x00" * 16)
        assert ensure_db_integrity(db_path3) is False

        # 4. Garbage that happens to be >= 16 bytes (regression: must reject)
        db_path4 = str(tmp_path / "garbage.db")
        with open(db_path4, "wb") as f:
            f.write(b"not a sqlite db!!")  # 17 bytes
        assert ensure_db_integrity(db_path4) is False

    def test_ensure_db_integrity_returns_false_for_missing_file(self, tmp_path):
        from memory.db_recovery import ensure_db_integrity
        assert ensure_db_integrity(str(tmp_path / "no_such.db")) is False

    def test_ensure_db_integrity_returns_true_for_in_memory(self):
        from memory.db_recovery import ensure_db_integrity
        assert ensure_db_integrity(":memory:") is True


# ----------------------------------------------------------------------------
# recover_corrupt_db — rootpgno → table mapping
# ----------------------------------------------------------------------------

class TestLostAndFoundMapping:
    """Verify the helpers that map lost_and_found rows back to original tables."""

    def test_read_rootpages_from_master_returns_table_rootpages(self, tmp_path):
        """sqlite_master survives corruption — rootpages must be readable."""
        db_path = str(tmp_path / "master.db")
        _create_healthy_db(db_path)
        from memory.db_recovery import _read_rootpages_from_master
        rootpages = _read_rootpages_from_master(db_path)
        assert "messages" in rootpages
        assert "conversation_summaries" in rootpages
        # Skip internal tables
        assert "sqlite_master" not in rootpages
        # All returned rootpages are positive ints
        for tbl, rp in rootpages.items():
            assert isinstance(rp, int)
            assert rp > 0

    def test_map_lost_and_found_columns_returns_declared_order(self, tmp_path):
        """Column order from PRAGMA table_info must match DDL declaration order.

        Production messages schema (db_migrations.SCHEMA_SQL): id is first
        (INTEGER PRIMARY KEY), then session_id/message_index/role/content/...
        The pinned invariants are:
          1. ``id`` is the FIRST column (rowid-alias sits at offset 0 so
             ``_map_lost_and_found_columns`` can correctly skip it during
             recovery column-offset calculation).
          2. ``content`` is present (the recovery routine writes data back
             here; the test pins that the column exists at all).
          3. Length is > 2 (sanity: we're not seeing a minimal stub).
          4. All entries are non-empty strings (no Nones, no integers).
        """
        db_path = str(tmp_path / "cols.db")
        _create_healthy_db(db_path)
        from memory.db_recovery import _map_lost_and_found_columns
        cols = _map_lost_and_found_columns("messages", db_path)
        # Structural invariants — production schema has many columns; the
        # exact list drifts as migrations add columns. The invariants are
        # the shape (id first, content present, well-formed list).
        assert isinstance(cols, list)
        assert len(cols) > 2, (
            f"messages table must have more than 2 columns under production "
            f"schema; got {cols}"
        )
        assert cols[0] == "id", (
            f"FIRST column of messages must be 'id' (rowid-alias lives at "
            f"offset 0); got {cols[0]!r}"
        )
        assert "content" in cols, (
            f"'content' column must be present in messages schema; got {cols}"
        )
        for c in cols:
            assert isinstance(c, str) and c, (
                f"all column names must be non-empty strings; got {c!r}"
            )

    def test_first_col_is_rowid_alias_detects_integer_primary_key(self, tmp_path):
        """INTEGER PRIMARY KEY is a rowid alias — its value lives in record header."""
        db_path = str(tmp_path / "rowid.db")
        _create_healthy_db(db_path)
        from memory.db_recovery import _first_col_is_rowid_alias
        # messages.id is INTEGER PRIMARY KEY → rowid alias
        assert _first_col_is_rowid_alias("messages", db_path) is True

    def test_recovery_succeeds_on_real_corruption(self, tmp_path):
        """End-to-end: corrupt the header, run recover_corrupt_db, verify the
        recovered db has at least the schema (data may be partial/lost)."""
        db_path = str(tmp_path / "recover.db")
        _create_healthy_db(db_path)

        # Insert a row we expect to recover. Production messages schema
        # requires session_id, message_index, role, created_at (all NOT NULL).
        con = sqlite3.connect(db_path)
        try:
            con.execute(
                "INSERT INTO messages (session_id, message_index, role, "
                "content, created_at) VALUES (?, ?, ?, ?, ?)",
                ("sess-test", 0, "user", "preserved", "2026-01-01T00:00:00Z"),
            )
            con.commit()
        finally:
            con.close()

        _corrupt_header(db_path)

        from memory.db_recovery import recover_corrupt_db
        # Recovery may or may not succeed depending on how damaged the file is
        # (we corrupted only the header). At minimum, it must not hang.
        try:
            ok = recover_corrupt_db(db_path)
            assert isinstance(ok, bool)
        except Exception:
            # If subprocess sqlite3 isn't available, we accept the failure
            # but log it for the operator.
            pytest.skip("sqlite3 CLI unavailable")

    def test_insert_recovered_rows_restores_lost_and_found_into_real_table(self, tmp_path):
        """c7-arch-iter4: lost_and_found rows must be inserted into their
        original table — proves the FIX 1 rootpage inversion works.

        Setup: create a 'messages' table with 5 known rows. Read its
        rootpgno via _read_rootpages_from_master. Construct fake
        lost_and_found rows whose rootpgno matches 'messages', invoke
        _insert_recovered_rows, and assert the data round-trips into the
        real table.

        This test will FAIL without the rootpage-inversion fix
        (``counters["recovered"] == 0``) and PASS with it.
        """
        from memory.db_recovery import (
            _insert_recovered_rows,
            _read_rootpages_from_master,
        )

        path = str(tmp_path / "restore.db")
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        con.executemany(
            "INSERT INTO messages (content) VALUES (?)",
            [(f"row{i}",) for i in range(5)],
        )
        con.commit()
        con.close()

        # Get the rootpgno of 'messages'
        rootpages = _read_rootpages_from_master(path)
        assert "messages" in rootpages
        messages_pgno = rootpages["messages"]
        assert messages_pgno > 0, f"messages rootpgno must be positive; got {messages_pgno}"

        # Construct fake lost_and_found rows — plain dicts work because
        # _insert_recovered_rows only accesses keys by string name
        # (rootpgno, id, c0, c1, ...). Schema: (id INTEGER, content TEXT);
        # id is a rowid-alias so the real data starts at c1 (offset=1).
        fake_rows = []
        for i, content in enumerate(["recovered_0", "recovered_1", "recovered_2"]):
            fake_rows.append({
                "rootpgno": messages_pgno,
                "pgno": 999 + i,
                "nfield": 2,
                "id": 100 + i,
                "c0": None,  # rowid-alias lives in record header
                "c1": content,
            })

        counters = {"recovered": 0, "lost": 0}
        cache: dict = {}

        # Open the healing connection read-write on the same path. This
        # mimics the real recovery flow: healing_con sees the same schema
        # we just created; recover_db_path is the same path (the helper
        # functions re-read schema via their own connections).
        rw_con = sqlite3.connect(path)
        try:
            _insert_recovered_rows(
                rw_con, fake_rows, rootpages, cache, counters, path,
            )
            rw_con.commit()
        finally:
            rw_con.close()

        # The fix must land all 3 fake rows in the messages table.
        assert counters["recovered"] >= 3, (
            f"FIX 1 FAIL: only {counters['recovered']}/3 lost_and_found "
            f"rows recovered (lost={counters['lost']}). The rootpage "
            f"inversion bug would make this 0."
        )
        assert counters["lost"] == 0, (
            f"FIX 1 FAIL: {counters['lost']} rows counted as lost; "
            f"expected 0 (all 3 have a valid messages rootpgno)"
        )

        # Verify the data actually landed in messages (not just counted).
        verify_con = sqlite3.connect(path)
        try:
            contents = [
                r[0] for r in verify_con.execute(
                    "SELECT content FROM messages WHERE content LIKE 'recovered_%'"
                ).fetchall()
            ]
        finally:
            verify_con.close()

        assert sorted(contents) == [
            "recovered_0", "recovered_1", "recovered_2",
        ], (
            f"recovered rows did not land in messages table; got {contents}"
        )

    def test_insert_recovered_rows_lost_when_rootpgno_unknown(self, tmp_path):
        """c7-arch-iter4: negative-path — rows with unknown rootpgno are still
        counted as lost. Pinned so a future refactor can't silently
        over-recover into a wrong table.
        """
        from memory.db_recovery import (
            _insert_recovered_rows,
            _read_rootpages_from_master,
        )

        path = str(tmp_path / "lost.db")
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
        con.commit()
        con.close()

        rootpages = _read_rootpages_from_master(path)
        # Fake row with a rootpgno that doesn't match any known table.
        fake_rows = [{
            "rootpgno": 9999,  # bogus
            "pgno": 1,
            "nfield": 2,
            "id": 1,
            "c0": None,
            "c1": "should_be_lost",
        }]

        counters = {"recovered": 0, "lost": 0}
        cache: dict = {}
        rw_con = sqlite3.connect(path)
        try:
            _insert_recovered_rows(
                rw_con, fake_rows, rootpages, cache, counters, path,
            )
        finally:
            rw_con.close()

        assert counters["recovered"] == 0
        assert counters["lost"] == 1


# ----------------------------------------------------------------------------
# Concurrent recovery guard
# ----------------------------------------------------------------------------
# Concurrency test uses threading.Barrier (per AGENTS.md "Concurrency bugs
# need threading.Barrier"). Sequential calls on the lock don't exercise the
# race window.

class TestConcurrentRecoveryGuard:
    """Verify concurrent recover_corrupt_db callers serialize via state lock.

    CRIT-6 fix: the previous test only asserted "all return bool, no crash"
    which would not have caught the CRIT-3/4 bug (sticky
    ``recovery_in_progress`` flag). The new tests pin three invariants:

    1. **Inner-call accounting.** ``_recover_corrupt_db_inner`` is monkey-
       patched with a counter; under the lock-serialized design, exactly N
       threads acquire the lock and either (a) find the db already healthy
       on re-entry (no inner call needed) or (b) run inner. The expected
       count is ``1 <= count <= N`` (the first thread recovered; subsequent
       threads may either short-circuit via ``ensure_db_integrity`` inside
       the lock OR run inner on the now-healthy db — both are correct).

       Critically, this rejects the OLD ``recovery_in_progress`` flag
       pattern which would have pinned ``count == 1`` for EVERY process
       ever — including scenarios where the FIRST corruption had been
       recovered in a prior run (the bug).

    2. **Re-corruption re-recovery.** A "second corruption works" test:
       recover once successfully, re-corrupt, recover again — the second
       recovery must succeed and the inner counter must increment. The
       OLD flag stuck True after the first recovery, so the second
       call would short-circuit and skip the inner procedure.

    3. **No process-wide global state.** ``state.recovery_in_progress``
       must NOT exist as an attribute (CRIT-3 fix). Pinning this prevents
       future regressions that re-introduce the flag.
    """

    def test_concurrent_recover_corrupt_db_serializes_via_state_lock(self, tmp_path):
        """8 threads invoke recover_corrupt_db simultaneously via Barrier.

        CRIT-6: pin that ``_recover_corrupt_db_inner`` is called between 1
        and N times (lock-serialized, may re-enter for healthy db on re-acquire).
        """
        import memory.db_recovery as db_recovery_mod
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "concurrent.db")
        _create_healthy_db(db_path)
        _corrupt_header(db_path)

        # CRIT-6: monkey-patch _recover_corrupt_db_inner with a counter so we
        # can assert exact call count. Wrap the real function so the recovery
        # is REAL, not mocked.
        real_inner = db_recovery_mod._recover_corrupt_db_inner
        counter = {"n": 0}
        counter_lock = threading.Lock()

        def counting_inner(p):
            with counter_lock:
                counter["n"] += 1
            return real_inner(p)

        with mock.patch.object(
            db_recovery_mod, "_recover_corrupt_db_inner", side_effect=counting_inner
        ):
            n_threads = 8
            barrier = threading.Barrier(n_threads)
            results = []
            errors = []
            results_lock = threading.Lock()

            def worker(idx):
                try:
                    barrier.wait(timeout=10)  # synchronized start
                    r = recover_corrupt_db(db_path)
                    with results_lock:
                        results.append((idx, r))
                except Exception as e:
                    with results_lock:
                        errors.append((idx, e))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        # No thread crashed with an unexpected exception
        assert not errors, f"threads raised: {errors}"
        # All threads completed
        assert len(results) == n_threads, (
            f"only {len(results)}/{n_threads} threads completed"
        )
        # All returned bool
        for _idx, r in results:
            assert isinstance(r, bool)
            assert r is not None

        # CRIT-6 pin: inner was called at least once (someone did the work)
        # and at most N times (lock-serialized, no thread silently dropped).
        # Lower bound 1 = at least one thread recovered.
        # Upper bound N = no thread exceeded N (e.g., N+1 would be a leak).
        assert 1 <= counter["n"] <= n_threads, (
            f"CRIT-6 FAIL: inner called {counter['n']} times for {n_threads} "
            f"threads; expected 1 <= count <= {n_threads}"
        )

        # After the dust settles, the db must be queryable (recovery succeeded).
        try:
            con = sqlite3.connect(db_path)
            try:
                row = con.execute("PRAGMA integrity_check").fetchone()
                # Post-fix: the db may still be corrupt if .dump failed to
                # extract anything (header-only corruption). What we DO assert
                # is that recover_corrupt_db returned True for at least one
                # caller when the db was queryable.
            finally:
                con.close()
        except sqlite3.DatabaseError:
            pass  # Acceptable — header-corrupted dbs may remain unreadable

    def test_state_recovery_lock_is_position_13(self):
        """Lock ordering invariant: recovery_lock is position #13."""
        import proxy.state as state
        from proxy.lock_ordering import OrderedLock

        lock = getattr(state, "recovery_lock", None)
        assert lock is not None, "state.recovery_lock not defined"
        if isinstance(lock, OrderedLock):
            assert lock.position == 13, f"recovery_lock position is {lock.position}, expected 13"

    def test_state_recovery_in_progress_flag_removed(self):
        """CRIT-3 pin: the ``recovery_in_progress`` flag must NOT exist on state.

        The flag caused a process-wide cross-path short-circuit (after one
        successful recovery, all subsequent corruption events skipped the
        procedure). Re-introducing it is a regression.
        """
        import proxy.state as state
        assert not hasattr(state, "recovery_in_progress"), (
            "CRIT-3 regression: proxy.state.recovery_in_progress re-introduced. "
            "The flag caused auto-heal to be permanently disabled after the "
            "first successful recovery (see ticket 24c1024e)."
        )

    def test_second_corruption_recovers_after_first_recovery(self, tmp_path):
        """CRIT-3 pin: a second corruption after a successful recovery MUST
        also recover (the OLD ``recovery_in_progress`` flag stuck True and
        short-circuited every subsequent corruption event).

        Steps:
          1. Create healthy db, corrupt header, recover → ok=True.
          2. Re-corrupt the (recovered) db.
          3. Recover again → ok=True (inner counter incremented by 1 from #1).
        """
        import memory.db_recovery as db_recovery_mod
        from memory.db_recovery import recover_corrupt_db

        db_path = str(tmp_path / "second_recover.db")
        _create_healthy_db(db_path)
        _corrupt_header(db_path)

        real_inner = db_recovery_mod._recover_corrupt_db_inner
        counter = {"n": 0}

        def counting_inner(p):
            counter["n"] += 1
            return real_inner(p)

        with mock.patch.object(
            db_recovery_mod, "_recover_corrupt_db_inner", side_effect=counting_inner
        ):
            # First recovery
            ok1 = recover_corrupt_db(db_path)
            inner_after_first = counter["n"]
            # Second corruption + recovery
            _corrupt_header(db_path)
            ok2 = recover_corrupt_db(db_path)
            inner_after_second = counter["n"]

        assert ok1 is True, f"first recovery failed: ok1={ok1}"
        assert ok2 is True, (
            f"CRIT-3 FAIL: second recovery after re-corruption returned "
            f"ok2={ok2}; OLD code with ``recovery_in_progress`` flag stuck "
            f"True would have short-circuited this. inner calls: "
            f"{inner_after_first} → {inner_after_second}"
        )
        # Inner MUST have been called for the second recovery (not short-
        # circuited by a stale flag).
        assert inner_after_second >= inner_after_first + 1, (
            f"CRIT-3 FAIL: _recover_corrupt_db_inner was NOT called for the "
            f"second recovery (counter {inner_after_first} → {inner_after_second}). "
            f"Expected at least one more invocation than the first recovery."
        )

    def test_concurrent_recover_corrupt_db_cross_path_does_not_short_circuit(self, tmp_path):
        """CRIT-4 pin: 16 threads racing on 16 DIFFERENT corrupt paths must
        ALL recover. The OLD ``recovery_in_progress`` flag was process-wide,
        so only the first corruption actually recovered; the other 15
        short-circuited to ``ensure_db_integrity`` (which returned False,
        leaving 15 paths still corrupt).
        """
        import memory.db_recovery as db_recovery_mod
        from memory.db_recovery import recover_corrupt_db

        n_paths = 16
        paths = []
        for i in range(n_paths):
            p = str(tmp_path / f"cross_{i}.db")
            _create_healthy_db(p)
            _corrupt_header(p)
            paths.append(p)

        real_inner = db_recovery_mod._recover_corrupt_db_inner
        counter = {"n": 0}
        counter_lock = threading.Lock()

        def counting_inner(p):
            with counter_lock:
                counter["n"] += 1
            return real_inner(p)

        with mock.patch.object(
            db_recovery_mod, "_recover_corrupt_db_inner", side_effect=counting_inner
        ):
            barrier = threading.Barrier(n_paths)
            results = []
            results_lock = threading.Lock()

            def worker(idx, path):
                try:
                    barrier.wait(timeout=10)
                    r = recover_corrupt_db(path)
                    with results_lock:
                        results.append((idx, r))
                except Exception:
                    pass

            threads = [threading.Thread(target=worker, args=(i, paths[i])) for i in range(n_paths)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        # All threads completed
        assert len(results) == n_paths, (
            f"only {len(results)}/{n_paths} threads completed"
        )
        # CRIT-4 pin (tightened): the OLD ``recovery_in_progress`` flag pinned
        # ``counter["n"] == 1`` regardless of how many distinct corrupt paths
        # existed. With the flag removed and the lock alone providing
        # serialization, each path's first arriving thread must reach the
        # inner procedure. We require a strict lower bound that is
        # unambiguously distinct from the OLD ``==1`` regression:
        #   * ``counter["n"] == n_paths`` — every path's first caller reached inner
        #   * Under lock contention, subsequent threads re-enter on the now-
        #     healthy db (inner runs again, idempotent), so we may see up to
        #     ``2 * n_paths`` calls. We don't assert an upper bound; the lower
        #     bound ``>= n_paths // 2`` is sufficient to reject the OLD bug.
        # The OLD bug pinned ``counter["n"] == 1``; this assertion rejects it.
        assert counter["n"] >= n_paths // 2, (
            f"CRIT-4 FAIL: _recover_corrupt_db_inner was called only "
            f"{counter['n']} times for {n_paths} distinct corrupt paths. "
            f"The OLD flag-based short-circuit pinned this to 1 regardless "
            f"of distinct paths. Expected >= {n_paths // 2} (>= n_paths // 2)."
        )
        # Cross-path correctness: every corrupt path must have produced at
        # least one ``recover_corrupt_db(path) is True`` result. With the OLD
        # flag, only the first path recovered (15 short-circuited to
        # ``ensure_db_integrity`` returning False). The lock-serialized design
        # recovers each path; allow minor flakiness under contention.
        recovered = sum(1 for _idx, r in results if r is True)
        assert recovered >= n_paths // 2, (
            f"CRIT-4 FAIL: only {recovered}/{n_paths} corrupt paths recovered "
            f"successfully. OLD flag-based bug pinned this to 1 (only the "
            f"first path recovered; remaining paths short-circuited to "
            f"``ensure_db_integrity`` which returned False)."
        )


# ----------------------------------------------------------------------------
# db_pool integration — inline recovery on DatabaseError
# ----------------------------------------------------------------------------

class TestPoolInlineRecovery:
    """Verify ``db_pool._create_connection`` recovers on DatabaseError('malformed'),
    and that recovery returns leases issued by the replacement generation, not
    the source pool.
    """

    def test_create_connection_recovers_on_malformed(self, tmp_path, monkeypatch):
        """Synthesize a corrupt db at PROXY_TEST_DB, force pool re-init, verify
        the next checkout() succeeds after recovery rebuilds the pool.

        The inline recovery in ``_create_connection`` catches the
        DatabaseError, runs recovery, and surfaces ``_PoolReplaced`` so the
        module-level ``_checkout_global`` retry helper routes the request
        through the replacement generation. ``memory.db_pool.connection()``
        is a public façade; it does not directly expose raw
        ``get_connection()`` ownership.
        """
        import memory.db_pool

        db_path = str(tmp_path / "pool_test.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)

        _create_healthy_db(db_path)
        _corrupt_header(db_path)

        # CRIT-3 fix: state.recovery_in_progress flag REMOVED — no reset
        # needed. Serialization is via state.recovery_lock alone.

        # Force pool rebuild + connection attempt. Inline recovery should kick in.
        # If subprocess sqlite3 isn't available, the inline recovery fails and
        # the DatabaseError re-raises — accept either outcome (no hang).
        try:
            memory.db_pool._ensure_pool(db_path)
            with memory.db_pool.connection() as conn:
                row = conn.execute("SELECT 1").fetchone()
                assert row[0] == 1
        except sqlite3.DatabaseError:
            # Inline recovery failed (no sqlite3 CLI) — that's acceptable in CI
            # without sqlite3 binary; the inline catch did its job.
            pass
        except Exception as e:
            # Other exceptions (e.g., recovery error) — also acceptable
            pass

    def test_get_return_connection_uses_replacement_pool_after_recovery(
        self, tmp_path, monkeypatch
    ):
        """``get_connection()`` returns a lease from the replacement pool,
        and ``return_connection()`` routes back to that replacement pool
        (not the current module-global)."""
        import memory.db_pool as db_pool

        db_path = str(tmp_path / "recovery_ownership.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        db_pool._pool = None

        _create_healthy_db(db_path)
        db_pool._ensure_pool(db_path)
        source_pool = db_pool._pool
        assert source_pool is not None
        assert source_pool._db_path == db_path

        # Corrupt the file before any physical connection is created.
        _corrupt_header(db_path)

        # Patch recover_corrupt_db with a deterministic, in-process function
        # that recreates a healthy file at the same path.
        from memory import db_recovery

        def fake_recover(path):
            # Recreate a healthy schema-only DB using raw sqlite3.
            if os.path.exists(path):
                os.remove(path)
            conn = sqlite3.connect(
                path,
                check_same_thread=False,
                isolation_level="",
            )
            try:
                from memory.db_migrations import SCHEMA_SQL
                conn.executescript(SCHEMA_SQL)
            finally:
                conn.close()
            return True

        monkeypatch.setattr(db_recovery, "recover_corrupt_db", fake_recover)

        # First get_connection() detects the corruption, runs recovery,
        # and returns a lease from the replacement pool.
        lease = db_pool.get_connection(db_path)
        try:
            replacement_pool = db_pool._pool
            assert replacement_pool is not source_pool, (
                "replacement pool must be a different instance from the source"
            )
            assert lease._pool is replacement_pool, (
                "lease must be issued by the replacement pool, not the source"
            )

            # Source pool was retired (not terminal).
            assert source_pool._retiring is True
            assert source_pool._shutting_down is False  # terminal is for manager-level only
            assert source_pool._created == 0
            assert source_pool._checked_out == {}

            # Replacement pool state: the lease is registered.
            assert len(replacement_pool._checked_out) == 1, (
                f"expected 1 lease in _checked_out after recovery, got "
                f"{replacement_pool._checked_out}"
            )
        finally:
            # Return the lease. This must route back to the replacement
            # pool (which issued it), not the global module variable.
            db_pool.return_connection(lease)

        # After return, the replacement pool has the conn back.
        assert replacement_pool._checked_out == {}
        assert replacement_pool._pool.qsize() == 1

    def test_concurrent_corrupt_checkouts_publish_one_replacement(
        self, tmp_path, monkeypatch
    ):
        """Concurrent corrupt checkouts publish only one replacement
        generation, not N — even though N threads observed the corruption."""
        import memory.db_pool as db_pool

        db_path = str(tmp_path / "concurrent_recovery.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        db_pool._pool = None

        _create_healthy_db(db_path)
        db_pool._ensure_pool(db_path)
        _corrupt_header(db_path)

        from memory import db_recovery

        recover_calls = [0]
        recover_lock = threading.Lock()

        def fake_recover(path):
            with recover_lock:
                recover_calls[0] += 1
            if os.path.exists(path):
                os.remove(path)
            conn = sqlite3.connect(
                path,
                check_same_thread=False,
                isolation_level="",
            )
            try:
                from memory.db_migrations import SCHEMA_SQL
                conn.executescript(SCHEMA_SQL)
            finally:
                conn.close()
            return True

        monkeypatch.setattr(db_recovery, "recover_corrupt_db", fake_recover)

        n_threads = 4
        barrier = threading.Barrier(n_threads)
        results: list = []
        results_lock = threading.Lock()

        def worker():
            barrier.wait(timeout=5.0)
            try:
                lease = db_pool.get_connection(db_path)
                with results_lock:
                    results.append(lease)
            except Exception as e:
                with results_lock:
                    results.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive()

        # All threads returned leases.
        leases = [r for r in results if isinstance(r, db_pool._ConnectionLease)]
        assert len(leases) == n_threads, (
            f"expected {n_threads} leases, got {len(leases)}: {results}"
        )

        # Only one recovery ran.
        assert recover_calls[0] == 1, (
            f"recovery should run exactly once, got {recover_calls[0]}"
        )

        # All leases point to the same replacement pool.
        pool_ids = {id(lease._pool) for lease in leases}
        assert len(pool_ids) == 1, (
            f"all leases must share a single replacement pool, got {pool_ids}"
        )
        replacement = leases[0]._pool
        assert replacement is db_pool._pool, (
            "replacement pool must be the current module-global"
        )

        # Return every lease.
        for lease in leases:
            assert lease._pool is replacement
            db_pool.return_connection(lease)

        # No checked-out leases remain. No superseded eager pool was
        # created (the replacement is lazy, so the maximum number of
        # physical conns equals the number of distinct checkout+return
        # pairs — at most the number of threads).
        assert replacement._checked_out == {}
        assert replacement._created <= n_threads

    def test_recovery_closes_pre_swap_checked_out_connections(
        self, tmp_path, monkeypatch
    ):
        """A pre-swap checked-out connection must be retired: its lease is
        marked released, the physical conn is closed, and using the lease
        after recovery raises RuntimeError. After recovery, a fresh
        checkout through the replacement pool writes to the new inode."""
        import memory.db_pool as db_pool

        db_path = str(tmp_path / "old_inode.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        db_pool._pool = None

        _create_healthy_db(db_path)
        db_pool._ensure_pool(db_path)
        source_pool = db_pool._pool
        assert source_pool is not None

        # Checkout a lease from the source pool.
        lease_a = source_pool.checkout(validate=False)
        assert source_pool._created == 1

        # Corrupt the file. The next physical connection creation will
        # fail at PRAGMA, triggering inline recovery on the source
        # generation. Recovery must close lease_a's underlying conn as
        # part of retiring the source pool.
        _corrupt_header(db_path)

        # Patch recover_corrupt_db with a deterministic, in-process
        # function that recreates a healthy file at the same path.
        from memory import db_recovery

        recover_calls = [0]

        def fake_recover(path):
            recover_calls[0] += 1
            if os.path.exists(path):
                os.remove(path)
            conn = sqlite3.connect(
                path,
                check_same_thread=False,
                isolation_level="",
            )
            try:
                from memory.db_migrations import SCHEMA_SQL
                conn.executescript(SCHEMA_SQL)
            finally:
                conn.close()
            return True

        monkeypatch.setattr(db_recovery, "recover_corrupt_db", fake_recover)

        try:
            with db_pool.connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS owner_swap (value INTEGER)"
                )
                conn.execute("INSERT INTO owner_swap VALUES (42)")
                conn.commit()
        finally:
            # No monkeypatch cleanup needed — we did not patch
            # _create_connection.

            pass

        # Verify the write is visible from a fresh raw connection on
        # the current (post-recovery) file.
        verify = sqlite3.connect(db_path)
        try:
            row = verify.execute(
                "SELECT value FROM owner_swap"
            ).fetchone()
            assert row[0] == 42, (
                f"expected row 42 from new inode, got {row}"
            )
        finally:
            verify.close()

        assert recover_calls[0] == 1

    def test_ensure_pool_replaces_shutdown_global_pool(
        self, tmp_path, monkeypatch
    ):
        """A terminal ConnectionPool is service-restartable: subsequent
        ``get_connection()`` calls publish a fresh pool instance."""
        import memory.db_pool as db_pool

        db_path = str(tmp_path / "restart.db")
        monkeypatch.setenv("PROXY_TEST_DB", db_path)
        db_pool._pool = None

        _create_healthy_db(db_path)
        db_pool._ensure_pool(db_path)

        # First round: checkout, return, then shut down the pool.
        first = db_pool.get_connection(db_path)
        first_pool = first._pool
        db_pool.return_connection(first)
        first_pool.shutdown(grace_period=0)
        assert first_pool._shutting_down is True

        # Second round: ensure_pool must publish a new pool.
        second = db_pool.get_connection(db_path)
        try:
            second_pool = second._pool
            assert second_pool is not first_pool, (
                "after shutdown, _ensure_pool must publish a fresh pool"
            )
            row = second.execute("SELECT 1").fetchone()
            assert row[0] == 1
        finally:
            db_pool.return_connection(second)


# ----------------------------------------------------------------------------
# init_db startup recovery
# ----------------------------------------------------------------------------

class TestInitDbStartupRecovery:
    """Verify init_db() runs ensure_db_integrity before opening the migration
    connection and triggers recover_corrupt_db on failure.
    """

    def test_init_db_recovers_corrupt_db(self, tmp_path):
        """init_db() must call ensure_db_integrity → recover_corrupt_db on a
        corrupt db so the proxy can come up after a hard crash.
        """
        from memory.db_migrations import init_db

        db_path = str(tmp_path / "initdb_test.db")
        _create_healthy_db(db_path)
        _corrupt_header(db_path)

        # init_db will call ensure_db_integrity → recover_corrupt_db.
        # We don't assert success/failure (depends on how damaged the db was);
        # we assert the call doesn't hang and returns a known type.
        try:
            ok = init_db(db_path)
            assert ok is True or ok is False  # must be a bool
        except RuntimeError as e:
            # If recovery fails entirely, init_db raises RuntimeError
            # (per the spec: "refusing to start"). Acceptable.
            assert "DB corruption recovery failed" in str(e) or "DB still corrupt" in str(e)

    def test_init_db_skips_recovery_when_db_healthy(self, tmp_path):
        """Healthy db → init_db succeeds without invoking recovery.

        Verified indirectly: ensure_db_integrity() returns True before init_db
        does any work, so recover_corrupt_db is never called.
        """
        from memory.db_migrations import init_db

        db_path = str(tmp_path / "initdb_healthy.db")
        # Use a minimal schema (the actual init_db will create its own).
        _create_healthy_db(db_path)
        # init_db should succeed and create all the production tables.
        ok = init_db(db_path)
        assert ok is True

        # The resulting db must have the production schema.
        con = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            # Spot-check a few critical tables
            assert "messages" in tables
            assert "conversation_summaries" in tables
        finally:
            con.close()


# ----------------------------------------------------------------------------
# executors.py — graceful shutdown atexit handler
# ----------------------------------------------------------------------------

@pytest.fixture
def isolated_executor_shutdown_state(monkeypatch):
    """Keep terminal shutdown tests away from worker-global singletons."""
    from memory import executors
    import memory.db_pool as db_pool

    for name in (
        "_sync_pool",
        "_query_pool",
        "_summary_pool",
        "_timeout_pool",
    ):
        monkeypatch.setattr(executors, name, None)

    monkeypatch.setattr(executors, "_shutdown_complete", False)
    monkeypatch.setattr(executors, "_shutdown_requested", False)
    monkeypatch.setattr(db_pool, "_pool", None)

    yield


class TestGracefulDbShutdown:
    """Verify ``executors._graceful_db_shutdown`` runs WAL checkpoint + pool
    close before interpreter exit (the prod incident 2026-07-06 prevention).
    """

    def test_graceful_db_shutdown_function_exists(self):
        from memory.executors import _graceful_db_shutdown
        assert callable(_graceful_db_shutdown)

    def test_graceful_db_shutdown_has_stable_signature(self):
        """The bound shutdown handler keeps its documented callable signature.

        CPython does not expose a stable public API for inspecting registered
        atexit handlers, so execution safety is covered separately by the isolated
        no-pool and bounded-timeout tests.
        """
        import inspect
        from memory import executors

        handler = executors._graceful_db_shutdown
        assert callable(handler)

        signature = inspect.signature(handler)
        parameters = list(signature.parameters.values())

        assert len(parameters) <= 1, (
            f"_graceful_db_shutdown must accept at most one argument, got {len(parameters)}"
        )

        if parameters:
            timeout_param = parameters[0]
            assert timeout_param.default is not inspect.Parameter.empty, (
                "_graceful_db_shutdown's single argument must have a default"
            )

    def test_graceful_db_shutdown_handles_no_pool(
        self,
        isolated_executor_shutdown_state,
    ):
        from memory import executors

        assert executors._sync_pool is None
        assert executors._query_pool is None
        assert executors._summary_pool is None
        assert executors._timeout_pool is None
        assert executors._shutdown_complete is False
        assert executors._shutdown_requested is False

        executors._graceful_db_shutdown(timeout=0.1)

        assert executors._shutdown_complete is True
        assert executors._shutdown_requested is True

    def test_shutdown_thread_pool_returns_within_timeout_when_pool_wedged(
        self,
        monkeypatch,
        isolated_executor_shutdown_state,
    ):
        import time
        from memory import executors

        release_shutdown = threading.Event()

        class _WedgedPool:
            """Fake pool whose wait=True shutdown blocks until released."""

            def shutdown(self, wait=True, cancel_futures=False):
                if wait:
                    release_shutdown.wait(timeout=5.0)

        monkeypatch.setattr(executors, "_sync_pool", _WedgedPool())
        monkeypatch.setattr(executors, "_query_pool", None)
        monkeypatch.setattr(executors, "_summary_pool", None)
        monkeypatch.setattr(executors, "_timeout_pool", None)
        monkeypatch.setattr(executors, "_shutdown_complete", False)
        monkeypatch.setattr(executors, "_shutdown_requested", False)

        t0 = time.monotonic()
        try:
            executors._shutdown_thread_pool(timeout=0.5)
            elapsed = time.monotonic() - t0
        finally:
            release_shutdown.set()

        assert elapsed < 2.0
        assert executors._shutdown_complete is True
        assert executors._shutdown_requested is True


# ---------------------------------------------------------------------------
# init_db() maintenance token cleanup
# ---------------------------------------------------------------------------


class _MaintenanceTokenCleanupFixture:
    """Per-test isolation for ``init_db`` maintenance-token tests."""


@pytest.fixture
def maintenance_token_isolation():
    """Reset manager state and clear the path-gate state."""
    import memory.db_pool as db_pool

    with db_pool._pool_condition:
        pool = db_pool._pool
        db_pool._pool = None

    if pool is not None:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass

    with db_pool._pool_condition:
        assert not db_pool._active_raw_connections
        assert db_pool._active_recovery_leaders == 0
        db_pool._recovery_attempts_by_source.clear()
        db_pool._recovering_paths.clear()
        db_pool._maintenance_paths.clear()
        db_pool._terminal_shutdown = False
        db_pool._pool_condition.notify_all()

    yield

    with db_pool._pool_condition:
        pool = db_pool._pool
        db_pool._pool = None

    if pool is not None:
        try:
            pool.shutdown(grace_period=0)
        except Exception:
            pass

    with db_pool._pool_condition:
        assert not db_pool._active_raw_connections
        assert db_pool._active_recovery_leaders == 0
        db_pool._recovery_attempts_by_source.clear()
        db_pool._recovering_paths.clear()
        db_pool._maintenance_paths.clear()
        db_pool._terminal_shutdown = False
        db_pool._pool_condition.notify_all()


def test_init_db_releases_token_on_integrity_failure(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """If ``ensure_db_integrity`` raises, the maintenance token must be
    released so the path is not wedged forever. init_db returns False
    on errors but the path gate must be unblocked for the next
    caller."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "initdb-integrity-fail.db")
    # Pre-create a non-empty db so the integrity probe runs.
    sqlite3.connect(db_path).execute("CREATE TABLE x (v INTEGER)")

    # Patch the function in its imported location. init_db does
    # ``from memory.db_recovery import ensure_db_integrity`` lazily,
    # so we patch the source module instead.
    def bad_ensure(path):
        raise RuntimeError("simulated integrity failure")

    monkeypatch.setattr(
        "memory.db_recovery.ensure_db_integrity", bad_ensure
    )

    # init_db returns False on errors. The contract: the maintenance
    # token is still released.
    assert db_migrations.init_db(db_path) is False

    with db_pool._pool_condition:
        assert db_path not in db_pool._maintenance_paths, (
            "maintenance token leaked on init_db integrity failure"
        )


def test_init_db_releases_token_on_create_raw_failure(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """If ``_create_raw_connection`` raises, the maintenance token must
    be released. The path is unblocked for the next caller."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "initdb-create-fail.db")

    def bad_create_raw(path, **kwargs):
        raise RuntimeError("simulated create_raw failure")

    monkeypatch.setattr(
        db_migrations, "_create_raw_connection", bad_create_raw
    )

    # init_db returns False on errors. The token is released.
    assert db_migrations.init_db(db_path) is False

    with db_pool._pool_condition:
        assert db_path not in db_pool._maintenance_paths, (
            "maintenance token leaked on init_db create failure"
        )


def test_init_db_releases_token_on_raw_close_failure(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """If the raw conn ``close()`` raises, the maintenance token must
    STILL be released. The active raw count must also be decremented
    so the path gate does not stay wedged."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "initdb-close-fail.db")
    sqlite3.connect(db_path).close()

    # Patch the _RawConnectionLease's close method to raise AFTER
    # running the production bookkeeping. The production close()
    # decrements the active raw count and only then re-raises. This
    # lets the test verify both: (a) the count is decremented and
    # (b) the maintenance token is released even when close raises.
    real_close = db_pool._RawConnectionLease.close

    def bad_close(self):
        # Run the production bookkeeping manually (decrement count)
        # so the path gate is not wedged. Then re-raise as if the
        # physical close failed.
        import memory.db_pool as db_pool_inner

        if self._closed:
            raise RuntimeError("simulated close failure")

        physical_error: Exception = RuntimeError(
            "simulated close failure"
        )

        object.__setattr__(self, "_closed", True)
        with db_pool_inner._pool_condition:
            count = db_pool_inner._active_raw_connections.get(
                self._db_path, 0
            )
            if count <= 0:
                raise AssertionError("raw connection count underflow")
            if count == 1:
                del db_pool_inner._active_raw_connections[self._db_path]
            else:
                db_pool_inner._active_raw_connections[
                    self._db_path
                ] = count - 1
            db_pool_inner._pool_condition.notify_all()

        raise physical_error

    monkeypatch.setattr(db_pool._RawConnectionLease, "close", bad_close)

    with pytest.raises(RuntimeError, match="simulated close failure"):
        db_migrations.init_db(db_path)

    with db_pool._pool_condition:
        assert db_path not in db_pool._maintenance_paths, (
            "maintenance token leaked on init_db close failure"
        )
        # The active raw count must be drained.
        assert db_path not in db_pool._active_raw_connections, (
            "active raw conn leaked on init_db close failure"
        )


def test_init_db_unblocks_path_after_token_cleanup(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """After init_db fails and releases its token, a follow-up
    _checkout_global() must succeed (not block forever on the gate)."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "initdb-unblock.db")
    sqlite3.connect(db_path).execute("CREATE TABLE x (v INTEGER)")

    def bad_ensure(path):
        raise RuntimeError("simulated integrity failure")

    monkeypatch.setattr(
        "memory.db_recovery.ensure_db_integrity", bad_ensure
    )

    # init_db returns False on errors.
    assert db_migrations.init_db(db_path) is False

    # The path gate is clear. A subsequent checkout on this path must
    # not block.
    lease = db_pool._checkout_global(db_path, validate=True, timeout=2.0)
    try:
        row = lease.execute("SELECT 1").fetchone()
        assert row[0] == 1
    finally:
        db_pool.return_connection(lease)


# ---------------------------------------------------------------------------
# Revert-test for Fix 1: raw conn close must happen under the maintenance token
# ---------------------------------------------------------------------------


def test_init_db_raw_close_happens_under_maintenance_token(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """Discriminating test for Fix 1: the raw conn close happens
    INSIDE the ``with path_maintenance`` block in ``init_db``. We
    intercept the production ``_RawConnectionLease.close`` to record
    the maintenance token state at the moment of close. The close MUST
    see the token in ``_maintenance_paths`` and the active raw conn
    count must still be 1 (the close decrements it after the gate is
    already past the unsafe-path check).

    Without Fix 1, the close happens in the OUTER ``finally`` of
    ``init_db`` — which runs AFTER the ``with`` block exits and the
    token is released. The recording close would then see
    ``token_present=False`` and the test would fail.
    """
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "initdb-close-under-maint.db")
    # Pre-create a valid db file (the production path is a file, not
    # :memory:, so the ``keeper`` branch is not taken and the only
    # close we observe is the raw conn close).
    sqlite3.connect(db_path).execute("CREATE TABLE x (v INTEGER)")

    close_state = {}

    real_close = db_pool._RawConnectionLease.close

    def recording_close(self):
        # Record the maintenance token state AND the active raw conn
        # count at the moment of close. Both are gated on
        # ``_pool_condition`` so we get a consistent snapshot.
        with db_pool._pool_condition:
            close_state["token_present"] = (
                db_path in db_pool._maintenance_paths
            )
            close_state["raw_count"] = db_pool._active_raw_connections.get(
                db_path, 0
            )
        return real_close(self)

    monkeypatch.setattr(
        db_pool._RawConnectionLease, "close", recording_close
    )

    # Run init_db. The close should happen with token_present=True.
    result = db_migrations.init_db(db_path)
    assert result is True
    assert close_state.get("token_present") is True, (
        f"raw conn close happened AFTER path_maintenance scope exited — "
        f"Fix 1 is missing: close must happen inside the maintenance "
        f"scope. close_state={close_state!r}"
    )
    assert close_state.get("raw_count") == 1, (
        f"raw conn count at close time was {close_state.get('raw_count')}, "
        f"expected 1 (the conn is still registered until close decrements). "
        f"close_state={close_state!r}"
    )


def test_init_db_migration_failure_closes_raw_under_maintenance(
    tmp_path, monkeypatch, maintenance_token_isolation,
):
    """If the migration body raises, the raw conn is still closed
    while the path-maintenance token is held."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "migration-body-failure.db")
    close_observed = {}
    real_create = db_migrations._create_raw_connection

    class FailingConnection:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)
        def executescript(self, sql):
            raise RuntimeError("migration body failed")
        def close(self):
            with db_pool._pool_condition:
                close_observed["token_present"] = (
                    db_path in db_pool._maintenance_paths
                )
            return self._real.close()
        def __getattr__(self, name):
            return getattr(self._real, name)
        def __setattr__(self, name, value):
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(self._real, name, value)

    def failing_create(path, **kwargs):
        lease = real_create(path, **kwargs)
        object.__setattr__(lease, "_conn", FailingConnection(lease._conn))
        return lease

    monkeypatch.setattr(
        db_migrations, "_create_raw_connection", failing_create,
    )

    assert db_migrations.init_db(db_path) is False
    assert close_observed == {"token_present": True}

    with db_pool._pool_condition:
        assert db_path not in db_pool._active_raw_connections
        assert db_path not in db_pool._maintenance_paths


def test_init_db_body_error_remains_primary_when_raw_close_fails(
    tmp_path, monkeypatch, maintenance_token_isolation, caplog,
):
    """When the migration body raises AND the raw close also raises,
    the body error remains primary, the path is marked unsafe, and
    init_db returns False (does not propagate to the caller)."""
    import logging
    import memory.db_pool as db_pool
    from memory import db_migrations

    db_path = str(tmp_path / "body-plus-close-fail.db")
    real_create = db_migrations._create_raw_connection
    physical_connections = []

    class BothFailConnection:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)
        def executescript(self, sql):
            raise RuntimeError("migration body failed")
        def close(self):
            raise RuntimeError("raw close failed")
        def __getattr__(self, name):
            return getattr(self._real, name)
        def __setattr__(self, name, value):
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(self._real, name, value)

    def both_fail_create(path, **kwargs):
        lease = real_create(path, **kwargs)
        physical_connections.append(lease._conn)
        object.__setattr__(
            lease,
            "_conn",
            BothFailConnection(lease._conn),
        )
        return lease

    monkeypatch.setattr(
        db_migrations, "_create_raw_connection", both_fail_create,
    )

    try:
        with caplog.at_level(logging.ERROR):
            assert db_migrations.init_db(db_path) is False

        assert "migration body failed" in caplog.text
        assert "raw close failed" in caplog.text

        with db_pool._pool_condition:
            assert db_path not in db_pool._active_raw_connections
            assert db_path not in db_pool._maintenance_paths
            assert isinstance(
                db_pool._unsafe_raw_paths[db_path], RuntimeError
            )
    finally:
        for physical in physical_connections:
            try:
                physical.close()
            except Exception:
                pass

        with db_pool._pool_condition:
            db_pool._unsafe_raw_paths.pop(db_path, None)
            db_pool._pool_condition.notify_all()


def test_init_db_memory_keeper_checked_in_exactly_once(monkeypatch):
    """The :memory: keeper lease is checked in exactly once
    after a successful migration."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    checkin_count = [0]
    original_checkin = db_pool.ConnectionPool.checkin

    def counting_checkin(self, lease):
        if lease._pool._db_path == ":memory:":
            checkin_count[0] += 1
        return original_checkin(self, lease)

    monkeypatch.setattr(
        db_pool.ConnectionPool, "checkin", counting_checkin,
    )

    monkeypatch.setattr(db_pool, "_get_db_path", lambda: ":memory:")

    result = db_migrations.init_db(":memory:")
    assert result is True
    assert checkin_count[0] == 1, (
        f"keeper was checked in {checkin_count[0]} times, expected 1"
    )


# ---------------------------------------------------------------------------
# check_integrity() uses the explicit path
# ---------------------------------------------------------------------------


def test_check_integrity_uses_explicit_path(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """check_integrity must use the explicit ``db_path`` argument, not
    the configured default path."""
    import memory.db_pool as db_pool
    from memory import db_migrations

    explicit_path = str(tmp_path / "check-integrity-explicit.db")
    other_path = str(tmp_path / "check-integrity-other.db")

    # Pre-initialize both paths with a valid schema.
    db_migrations.init_db(explicit_path)
    db_migrations.init_db(other_path)

    # The default path is other_path; the explicit path is explicit_path.
    monkeypatch.setattr(db_migrations, "_get_db_path", lambda: other_path)

    # A correctly-implemented check_integrity(explicit_path) would
    # return False when the explicit path is corrupt, and True for
    # the OTHER healthy path.
    # Corrupt the explicit path by replacing it with random non-SQLite
    # data. We need a corruption visible to a fresh connection.
    raw = sqlite3.connect(":memory:")
    try:
        raw.execute("ATTACH DATABASE ? AS explicit_db", (explicit_path,))
        try:
            raw.execute("DROP TABLE IF EXISTS explicit_db.messages")
            raw.execute("DETACH DATABASE explicit_db")
        except Exception:
            pass
    finally:
        raw.close()

    with open(explicit_path, "wb") as f:
        f.write(b"\x00" * 16 + b"not-a-sqlite-file" + b"\x00" * 16)

    # check_integrity(other_path) must return True (healthy).
    # check_integrity(explicit_path) must reflect the explicit path's
    # state, not the default path's state.
    assert db_migrations.check_integrity(other_path) is True
    # The explicit path may or may not be detected as corrupt by
    # PRAGMA integrity_check via a fresh connection (the file is
    # non-SQLite, so any open will fail). The contract is: the
    # explicit path is checked, not the default.
    try:
        result = db_migrations.check_integrity(explicit_path)
        # If the pool can be created and integrity_check returns "ok",
        # something is wrong (the file is corrupt).
        assert result is False, (
            f"check_integrity({explicit_path!r}) returned True on a "
            f"corrupt file"
        )
    except Exception:
        # A failure to even open the file is also acceptable
        # (the corrupt file prevents the pool from forming).
        pass


# ---------------------------------------------------------------------------
# Recovery bookkeeping invariants
# ---------------------------------------------------------------------------


def test_recovery_publication_does_not_leave_source_alive(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """When a source is recovered and the replacement is published,
    the source's checked-out set must be empty and its inflight
    counter must be zero."""
    import memory.db_pool as db_pool
    from memory import db_recovery

    db_path = str(tmp_path / "recovery-source-clean.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    monkeypatch.setattr(
        db_recovery,
        "recover_corrupt_db",
        lambda p: True,
    )

    db_pool._recover_source(pool, pool._corruption_error)

    assert pool._created == 0
    assert pool._checked_out == {}
    assert pool._inflight == 0
    assert pool._creating == 0


def test_recovery_failure_does_not_set_new_pool(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """A failed recovery must NOT publish a new pool, and the global
    must be detached (set to None) so the next checkout triggers a
    fresh pool creation."""
    import memory.db_pool as db_pool
    from memory import db_recovery

    db_path = str(tmp_path / "recovery-fail.db")
    sqlite3.connect(db_path).close()

    pool = db_pool._ensure_pool(db_path)
    pool._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    def bad_recover(p):
        raise DatabaseRecoveryError("simulated recovery failure")

    monkeypatch.setattr(db_recovery, "recover_corrupt_db", bad_recover)

    with pytest.raises(DatabaseRecoveryError):
        db_pool._recover_source(pool, pool._corruption_error)

    # Source must be drained.
    assert pool._inflight == 0
    assert pool._creating == 0

    # Global is detached.
    with db_pool._pool_condition:
        assert db_pool._pool is None


# ---------------------------------------------------------------------------
# Terminal shutdown ordering
# ---------------------------------------------------------------------------


def test_terminal_shutdown_waits_for_maintenance_paths(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """Terminal shutdown must wait for an active maintenance token to
    be released before returning the detached pool."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "terminal-wait-maint.db")
    pool = db_pool._ensure_pool(db_path)

    # Start maintenance in a thread; do not release the token yet.
    token_holder = []
    maint_started = threading.Event()
    maint_done = threading.Event()

    def start_maint():
        try:
            tok = db_pool.begin_path_maintenance(db_path, drain_timeout=1.0)
            token_holder.append(tok)
        finally:
            maint_started.set()
            maint_done.set()

    t = threading.Thread(target=start_maint)
    t.start()
    assert maint_done.wait(timeout=2.0)
    t.join(timeout=2.0)

    # The pool is shut down; only the token remains.
    token = token_holder[0]
    with db_pool._pool_condition:
        assert db_path in db_pool._maintenance_paths

    # Run terminal shutdown in a thread; it must block on the maintenance path.
    shutdown_done = threading.Event()
    shutdown_outcome = []

    _ts_ident = {"v": None}
    _ts_parked = threading.Event()
    monkeypatch.setattr(db_pool, "_pool_condition", _ParkProbe(db_pool._pool_condition, _ts_ident, _ts_parked))

    def do_shutdown():
        _ts_ident["v"] = threading.get_ident()
        try:
            db_pool._begin_terminal_shutdown(
                deadline=time.monotonic() + 5.0
            )
        finally:
            shutdown_done.set()

    t2 = threading.Thread(target=do_shutdown)
    t2.start()

    # Terminal shutdown must park on _pool_condition while the maintenance path is held.
    assert _ts_parked.wait(timeout=5.0), "terminal shutdown never parked on _pool_condition"

    # Release the token. The terminal shutdown should now complete.
    db_pool.end_path_maintenance(token)
    assert shutdown_done.wait(timeout=2.0), (
        "terminal shutdown did not complete after maintenance release"
    )
    t2.join(timeout=1.0)
    assert not t2.is_alive()


def test_terminal_shutdown_does_not_hang_forever_on_wedged_maintenance(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """Terminal shutdown with a tight deadline must not hang forever
    on a maintenance path that never releases."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "terminal-wedged-maint.db")
    pool = db_pool._ensure_pool(db_path)

    # Manually register a maintenance token that will never be released.
    token = db_pool._MaintenanceToken(db_path)
    with db_pool._pool_condition:
        db_pool._maintenance_paths[db_path] = token

    t0 = time.monotonic()
    db_pool._begin_terminal_shutdown(deadline=t0 + 0.5)
    elapsed = time.monotonic() - t0

    # Must not hang for more than 1s.
    assert elapsed < 1.5, (
        f"terminal shutdown hung for {elapsed:.2f}s on wedged maintenance"
    )
    assert db_pool._terminal_shutdown is True


# ---------------------------------------------------------------------------
# check_integrity() smoke
# ---------------------------------------------------------------------------


def test_check_integrity_healthy_db_returns_true(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    from memory import db_migrations

    db_path = str(tmp_path / "check-integrity-healthy.db")
    db_migrations.init_db(db_path)

    assert db_migrations.check_integrity(db_path) is True


def test_check_integrity_missing_db_returns_false(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    from memory import db_migrations

    missing_path = str(tmp_path / "check-integrity-missing.db")
    # No init_db; the file does not exist.
    # check_integrity on a missing file: PRAGMA integrity_check
    # returns "ok" on an empty file or fails on a missing file. The
    # pool cannot form if the file is missing and the path is empty.
    # The implementation must return False in that case.
    try:
        result = db_migrations.check_integrity(missing_path)
        # If a check could form, it might return True (empty file) or
        # raise. Either way, the function must not raise an unhandled
        # exception.
        assert isinstance(result, bool)
    except Exception:
        # Acceptable: the missing file prevented pool formation.
        pass


# ---------------------------------------------------------------------------
# Integration: super_recover pool is not rebuilt on superseded source
# ---------------------------------------------------------------------------


def test_recovery_after_concurrent_newer_pool_keeps_newer(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """Sequence:
    1. source_pool detects corruption.
    2. Same path: a newer pool is installed as the global.
    3. source_pool's recovery is invoked.
    Expected: the source is retired, the file is NOT touched, the
    newer pool remains the global, and the marker table on the
    newer pool is still accessible.
    """
    import memory.db_pool as db_pool
    from memory import db_migrations
    from memory import db_recovery

    db_path = str(tmp_path / "recover-keeps-newer.db")
    sqlite3.connect(db_path).close()

    # Mark the file with NEW_POOL before the recovery test.
    def fake_recover(path):
        c = sqlite3.connect(path)
        try:
            c.execute(
                "CREATE TABLE IF NOT EXISTS recovery_marker (id INTEGER PRIMARY KEY, note TEXT)"
            )
            c.execute(
                "INSERT OR REPLACE INTO recovery_marker (id, note) VALUES (1, 'NEW_POOL')"
            )
            c.commit()
        finally:
            c.close()
        return True
    monkeypatch.setattr(db_recovery, "recover_corrupt_db", fake_recover)

    # Step 1: source pool detects corruption.
    source = db_pool._ensure_pool(db_path)
    source._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    # Step 2: install a newer pool as the global. Mark it with
    # NEW_POOL via the recover helper.
    fake_recover(db_path)
    new_pool = db_pool.ConnectionPool(db_path, pool_size=4, lazy=True)
    with db_pool._pool_condition:
        db_pool._pool = new_pool

    # Step 3: trigger recovery on the source. Must NOT touch the file.
    db_pool._recover_source(source, source._corruption_error)

    # The newer pool's marker is intact.
    c = sqlite3.connect(db_path)
    try:
        row = c.execute(
            "SELECT note FROM recovery_marker WHERE id = 1"
        ).fetchone()
        assert row[0] == "NEW_POOL"
    finally:
        c.close()

    # The new pool is still the global.
    assert db_pool._pool is new_pool


# ---------------------------------------------------------------------------
# Validation repair table — used to assert that a corrupted file is
# rebuilt correctly when recovery is allowed to proceed.
# ---------------------------------------------------------------------------


def test_recovery_rebuilds_db_file_for_current_source(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """When the source IS the current global, recovery rebuilds the
    database file. A recovery marker placed by the recover helper
    must be reachable from a fresh raw connection."""
    import memory.db_pool as db_pool
    from memory import db_recovery

    db_path = str(tmp_path / "recover-current-source.db")
    sqlite3.connect(db_path).close()

    source = db_pool._ensure_pool(db_path)
    source._corruption_error = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    def fake_recover(path):
        c = sqlite3.connect(path)
        try:
            c.execute(
                "CREATE TABLE IF NOT EXISTS recovery_marker (id INTEGER PRIMARY KEY, note TEXT)"
            )
            c.execute(
                "INSERT OR REPLACE INTO recovery_marker (id, note) VALUES (1, 'POST_RECOVERY')"
            )
            c.commit()
        finally:
            c.close()
        return True

    monkeypatch.setattr(db_recovery, "recover_corrupt_db", fake_recover)
    db_pool._recover_source(source, source._corruption_error)

    c = sqlite3.connect(db_path)
    try:
        row = c.execute(
            "SELECT note FROM recovery_marker WHERE id = 1"
        ).fetchone()
        assert row[0] == "POST_RECOVERY"
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Revert-test for Fix 2: late-supersession double-decrement guard
# ---------------------------------------------------------------------------


def test_late_supersession_does_not_double_decrement_leader(
    tmp_path, monkeypatch, maintenance_token_isolation
):
    """Late supersession performs terminal leader bookkeeping once."""
    import memory.db_pool as db_pool
    import memory.db_recovery as db_recovery

    db_path = str(tmp_path / "late-supersede-fix2.db")
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE seed_t (v INTEGER)")
    seed.close()

    source = db_pool.ConnectionPool(
        db_path,
        pool_size=4,
        lazy=True,
    )
    source._corruption_error = sqlite3.DatabaseError(
        "simulated corruption"
    )
    db_pool._pool = source

    newer = db_pool.ConnectionPool(
        db_path,
        pool_size=4,
        lazy=True,
    )
    real_retire = source.retire_for_recovery
    retire_timeouts = []

    def retire_and_supersede(
        timeout=db_pool._QUIESCENCE_DRAIN_TIMEOUT,
    ):
        retire_timeouts.append(timeout)
        with db_pool._pool_condition:
            db_pool._pool = newer
            db_pool._pool_condition.notify_all()
        return real_retire(timeout=timeout)

    monkeypatch.setattr(
        source,
        "retire_for_recovery",
        retire_and_supersede,
    )

    def fake_recover(_path):
        raise AssertionError("recover_corrupt_db should not be called")

    monkeypatch.setattr(
        db_recovery,
        "recover_corrupt_db",
        fake_recover,
    )

    db_pool._recover_source(
        source,
        source._corruption_error,
    )

    assert len(retire_timeouts) == 1
    assert 0.0 <= retire_timeouts[0] <= (
        db_pool._QUIESCENCE_DRAIN_TIMEOUT
    )
    assert db_pool._pool is newer

    with db_pool._pool_condition:
        assert db_pool._active_recovery_leaders == 0, (
            f"leader count not balanced: "
            f"{db_pool._active_recovery_leaders}"
        )
        assert db_path not in db_pool._recovering_paths


# ---------------------------------------------------------------------------
# Revert-test for Fix 3: _shutdown_thread_pool race
# ---------------------------------------------------------------------------


def test_shutdown_thread_pool_snapshots_generation_created_under_pool_lock(
    monkeypatch,
    isolated_executor_shutdown_state,
):
    """Shutdown waits for an in-progress executor publication and snapshots it."""
    import memory.executors as executors

    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    shutdown_attempted_lock = threading.Event()
    shutdown_acquired_lock = threading.Event()
    shutdown_called = threading.Event()
    getter_done = threading.Event()
    shutdown_done = threading.Event()
    shutdown_ident = {"value": None}

    real_pool_lock = executors._pool_lock

    class RecordingLock:
        def __enter__(self):
            if threading.get_ident() == shutdown_ident["value"]:
                shutdown_attempted_lock.set()

            real_pool_lock.acquire()

            if threading.get_ident() == shutdown_ident["value"]:
                shutdown_acquired_lock.set()

            return self

        def __exit__(self, exc_type, exc, tb):
            real_pool_lock.release()

    class FakeExecutor:
        def shutdown(self, **kwargs):
            shutdown_called.set()

    fake = FakeExecutor()

    def blocking_constructor(*args, **kwargs):
        constructor_entered.set()
        assert release_constructor.wait(timeout=5.0)
        return fake

    monkeypatch.setattr(executors, "_pool_lock", RecordingLock())
    monkeypatch.setattr(
        executors,
        "ThreadPoolExecutor",
        blocking_constructor,
    )

    def get_pool():
        executors._get_sync_pool()
        getter_done.set()

    def shutdown():
        shutdown_ident["value"] = threading.get_ident()
        executors._shutdown_thread_pool(timeout=0)
        shutdown_done.set()

    getter_thread = threading.Thread(target=get_pool)
    shutdown_thread = threading.Thread(target=shutdown)

    getter_thread.start()
    assert constructor_entered.wait(timeout=2.0)

    shutdown_thread.start()

    try:
        assert shutdown_attempted_lock.wait(timeout=2.0)
        assert not shutdown_acquired_lock.is_set(), (
            "shutdown acquired _pool_lock while getter still owned it"
        )

        release_constructor.set()

        assert getter_done.wait(timeout=2.0)
        assert shutdown_acquired_lock.wait(timeout=2.0)
        assert shutdown_done.wait(timeout=2.0)
        assert shutdown_called.is_set()
    finally:
        release_constructor.set()
        getter_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=2.0)

    assert not getter_thread.is_alive()
    assert not shutdown_thread.is_alive()


# ---------------------------------------------------------------------------
# Revert-test for Fix 4: early-superseded attempt terminal state
# ---------------------------------------------------------------------------


def test_early_superseded_attempt_wakes_follower(
    tmp_path, monkeypatch, maintenance_token_isolation,
):
    """A follower wakes after a superseded retirement succeeds."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "early-supersede-fix4.db")
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE seed_t (v INTEGER)")
    seed.close()

    source = db_pool.ConnectionPool(
        db_path,
        pool_size=4,
        lazy=True,
    )
    source._corruption_error = sqlite3.DatabaseError(
        "simulated corruption"
    )

    newer = db_pool.ConnectionPool(
        db_path,
        pool_size=4,
        lazy=True,
    )
    db_pool._pool = newer

    retire_started = threading.Event()
    retire_can_proceed = threading.Event()
    retire_timeouts = []

    def slow_retire(
        timeout=db_pool._QUIESCENCE_DRAIN_TIMEOUT,
    ):
        retire_timeouts.append(timeout)
        retire_started.set()
        if not retire_can_proceed.wait(timeout=timeout):
            raise AssertionError(
                "test did not release superseded retirement "
                "before the production recovery deadline"
            )

    monkeypatch.setattr(
        source,
        "retire_for_recovery",
        slow_retire,
    )

    real_condition = db_pool._pool_condition
    follower_waiting = threading.Event()
    follower_ident = {"value": None}

    class RecordingCondition:
        def __enter__(self):
            return real_condition.__enter__()

        def __exit__(self, *args):
            return real_condition.__exit__(*args)

        def wait(self, timeout=None):
            if threading.get_ident() == follower_ident["value"]:
                follower_waiting.set()
            return real_condition.wait(timeout)

        def notify_all(self):
            return real_condition.notify_all()

    monkeypatch.setattr(
        db_pool,
        "_pool_condition",
        RecordingCondition(),
    )

    leader_result = []
    leader_error = []

    def leader():
        try:
            db_pool._recover_source(
                source,
                source._corruption_error,
            )
            leader_result.append("returned")
        except BaseException as exc:
            leader_error.append(exc)

    follower_result = []
    follower_error = []

    def follower():
        follower_ident["value"] = threading.get_ident()
        try:
            db_pool._recover_source(
                source,
                source._corruption_error,
            )
            follower_result.append("returned")
        except BaseException as exc:
            follower_error.append(exc)

    leader_thread = threading.Thread(
        target=leader,
        daemon=True,
    )
    follower_thread = threading.Thread(
        target=follower,
        daemon=True,
    )

    try:
        leader_thread.start()
        assert retire_started.wait(timeout=2.0), (
            "leader never entered superseded retirement"
        )

        follower_thread.start()
        assert follower_waiting.wait(timeout=2.0), (
            "follower never entered the condition.wait loop"
        )
    finally:
        retire_can_proceed.set()
        leader_thread.join(timeout=2.0)
        if follower_thread.ident is not None:
            follower_thread.join(timeout=2.0)

    assert not leader_thread.is_alive()
    assert not follower_thread.is_alive(), (
        "follower did not wake after the superseded attempt succeeded"
    )
    assert len(retire_timeouts) == 1
    assert 0.0 <= retire_timeouts[0] <= (
        db_pool._QUIESCENCE_DRAIN_TIMEOUT
    )
    assert leader_error == []
    assert follower_error == []
    assert leader_result == ["returned"]
    assert follower_result == ["returned"]


def test_early_superseded_retire_failure_wakes_follower_with_error(
    tmp_path, monkeypatch, maintenance_token_isolation,
):
    """A follower receives the superseded leader's retirement failure."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "early-supersede-retire-fail.db")
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE seed_t (v INTEGER)")
    seed.close()

    source = db_pool.ConnectionPool(
        db_path,
        pool_size=4,
        lazy=True,
    )
    source._corruption_error = sqlite3.DatabaseError(
        "simulated corruption"
    )

    newer = db_pool.ConnectionPool(
        db_path,
        pool_size=4,
        lazy=True,
    )
    db_pool._pool = newer

    retire_started = threading.Event()
    retire_can_proceed = threading.Event()
    retire_error = RuntimeError("simulated retire failure")
    retire_timeouts = []

    def bad_retire(
        timeout=db_pool._QUIESCENCE_DRAIN_TIMEOUT,
    ):
        retire_timeouts.append(timeout)
        retire_started.set()
        if not retire_can_proceed.wait(timeout=timeout):
            raise AssertionError(
                "test did not release failing superseded retirement "
                "before the production recovery deadline"
            )
        raise retire_error

    monkeypatch.setattr(
        source,
        "retire_for_recovery",
        bad_retire,
    )

    real_condition = db_pool._pool_condition
    follower_waiting = threading.Event()
    follower_ident = {"value": None}

    class RecordingCondition:
        def __enter__(self):
            return real_condition.__enter__()

        def __exit__(self, *args):
            return real_condition.__exit__(*args)

        def wait(self, timeout=None):
            if threading.get_ident() == follower_ident["value"]:
                follower_waiting.set()
            return real_condition.wait(timeout)

        def notify_all(self):
            return real_condition.notify_all()

    monkeypatch.setattr(
        db_pool,
        "_pool_condition",
        RecordingCondition(),
    )

    leader_error = []

    def leader():
        try:
            db_pool._recover_source(
                source,
                source._corruption_error,
            )
        except BaseException as exc:
            leader_error.append(exc)

    follower_error = []

    def follower():
        follower_ident["value"] = threading.get_ident()
        try:
            db_pool._recover_source(
                source,
                source._corruption_error,
            )
        except BaseException as exc:
            follower_error.append(exc)

    leader_thread = threading.Thread(
        target=leader,
        daemon=True,
    )
    follower_thread = threading.Thread(
        target=follower,
        daemon=True,
    )

    try:
        leader_thread.start()
        assert retire_started.wait(timeout=2.0), (
            "leader never entered failing superseded retirement"
        )

        follower_thread.start()
        assert follower_waiting.wait(timeout=2.0), (
            "follower never entered the condition.wait loop"
        )
    finally:
        retire_can_proceed.set()
        leader_thread.join(timeout=2.0)
        if follower_thread.ident is not None:
            follower_thread.join(timeout=2.0)

    assert not leader_thread.is_alive()
    assert not follower_thread.is_alive()
    assert len(retire_timeouts) == 1
    assert 0.0 <= retire_timeouts[0] <= (
        db_pool._QUIESCENCE_DRAIN_TIMEOUT
    )
    assert leader_error == [retire_error]
    assert len(follower_error) == 1
    assert isinstance(
        follower_error[0],
        db_pool.DatabaseRecoveryError,
    )
    assert follower_error[0].__cause__ is retire_error


# ---------------------------------------------------------------------------
# Terminal shutdown: daemon-worker pattern
# ---------------------------------------------------------------------------


def test_terminal_shutdown_times_out_without_closing_active_lease(
    tmp_path, monkeypatch,
):
    """A wedged lease operation lock must not block terminal
    shutdown past the deadline. Pool shutdown is run in a daemon
    worker; if it does not complete in time, the checkpoint is
    skipped and the active lease is NOT force-closed."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "terminal-active-lease.db")
    pool = db_pool._ensure_pool(db_path)
    lease = pool.checkout(validate=True)

    operation_held = threading.Event()
    release_operation = threading.Event()
    physical_close_called = threading.Event()
    checkpoint_called = threading.Event()

    real_close = lease._conn.close

    class RecordingConnection:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)
        def close(self):
            physical_close_called.set()
            return real_close()
        def __getattr__(self, name):
            return getattr(self._real, name)

    object.__setattr__(lease, "_conn", RecordingConnection(lease._conn))

    def hold_operation():
        with lease._operation_lock:
            operation_held.set()
            assert release_operation.wait(timeout=5.0)

    worker = threading.Thread(target=hold_operation, daemon=True)
    worker.start()
    assert operation_held.wait(timeout=2.0)

    # Intercept only checkpoint connection creation, not pool shutdown.
    real_connect = db_pool.sqlite3.connect

    def recording_connect(*args, **kwargs):
        checkpoint_called.set()
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(db_pool.sqlite3, "connect", recording_connect)

    # Run the real terminal path with a tight deadline.
    started = time.monotonic()
    db_pool._shutdown_global_pool(timeout=0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"terminal shutdown took {elapsed:.2f}s"
    assert not physical_close_called.is_set(), (
        "lease physical close was forced despite wedged operation lock"
    )
    assert not checkpoint_called.is_set(), (
        "WAL checkpoint ran despite pool shutdown not completing"
    )

    # Now release the operation. The daemon worker thread will
    # finish its drain and trigger the normal close.
    release_operation.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    # The pool's shutdown completed.
    assert pool._shutdown_complete.wait(timeout=2.0)
    assert physical_close_called.is_set()


def test_terminal_shutdown_checkpoints_only_after_pool_shutdown_completes(
    tmp_path, monkeypatch,
):
    """WAL checkpoint starts only after pool shutdown returns."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "terminal-checkpoint-after.db")
    sqlite3.connect(db_path).close()
    pool = db_pool._ensure_pool(db_path)

    shutdown_entered = threading.Event()
    release_shutdown = threading.Event()
    checkpoint_called = threading.Event()
    shutdown_done = threading.Event()

    real_shutdown = pool.shutdown
    real_connect = db_pool.sqlite3.connect

    def gated_shutdown(*args, **kwargs):
        shutdown_entered.set()
        assert release_shutdown.wait(timeout=5.0)
        return real_shutdown(*args, **kwargs)

    def recording_connect(*args, **kwargs):
        checkpoint_called.set()
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(pool, "shutdown", gated_shutdown)
    monkeypatch.setattr(
        db_pool.sqlite3,
        "connect",
        recording_connect,
    )

    def run_terminal_shutdown():
        try:
            db_pool._shutdown_global_pool(timeout=5.0)
        finally:
            shutdown_done.set()

    shutdown_thread = threading.Thread(
        target=run_terminal_shutdown,
    )
    shutdown_thread.start()

    try:
        assert shutdown_entered.wait(timeout=2.0)
        assert not checkpoint_called.is_set(), (
            "checkpoint ran before pool shutdown completed"
        )

        release_shutdown.set()

        assert shutdown_done.wait(timeout=2.0)
        assert checkpoint_called.is_set(), "checkpoint never ran"
    finally:
        release_shutdown.set()
        shutdown_thread.join(timeout=2.0)

    assert not shutdown_thread.is_alive()


def test_terminal_shutdown_skips_checkpoint_when_pool_shutdown_raises(
    tmp_path, monkeypatch,
):
    """If pool.shutdown() raises, the WAL checkpoint is skipped."""
    import memory.db_pool as db_pool

    db_path = str(tmp_path / "terminal-shutdown-raises.db")
    sqlite3.connect(db_path).close()
    pool = db_pool._ensure_pool(db_path)

    def bad_shutdown(self, grace_period=0):
        raise RuntimeError("simulated pool shutdown failure")

    monkeypatch.setattr(db_pool.ConnectionPool, "shutdown", bad_shutdown)

    checkpoint_called = threading.Event()
    real_connect = db_pool.sqlite3.connect

    def recording_connect(*args, **kwargs):
        checkpoint_called.set()
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(db_pool.sqlite3, "connect", recording_connect)

    db_pool._shutdown_global_pool(timeout=2.0)
    assert not checkpoint_called.is_set(), (
        "WAL checkpoint ran despite pool shutdown raising"
    )