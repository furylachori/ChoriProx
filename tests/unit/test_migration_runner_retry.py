"""Auto-retry behavior for ``MigrationRunner`` failed migrations.

Bug being pinned here: ``get_current_version()`` used to return
``max(m.version)`` over ALL ``_Migration`` markers regardless of status.
When ``up()`` raised, the runner wrote ``_Migration {version: N, status:
'failed'}`` and subsequent boots saw ``current == N``, so ``run_pending()``
(which only runs ``version > current``) permanently skipped the failed
migration until someone manually deleted the marker node.

The fix makes ``get_current_version()`` return the MAX version whose status
is ``'completed'``. A failed marker at version N no longer advances the
cursor, so ``run_pending()`` naturally retries it. All migrations are
idempotent by contract, so retrying is safe.

These tests pin:

1. ``get_current_version()`` ignores ``status: 'failed'`` markers and returns
   the last ``'completed'`` version (0 when nothing is completed).
2. ``run_pending()`` retries a previously-failed migration: ``up()`` runs
   again, a WARNING is logged, and the single marker node for that version is
   flipped from ``'failed'`` to ``'completed'`` (one node per version, no
   duplicate is created).
3. Revert-test: with the pre-fix ``get_current_version`` (MAX regardless of
   status) the retry never happens — proving the tests above discriminate the
   fix rather than passing tautologically.

The fake executor below simulates real Neo4j marker semantics (MERGE = one
node per version; MATCH/SET mutates the existing node) so the production
``run_pending()`` code path runs unmodified.
"""
import unittest
from unittest.mock import MagicMock

from memory.migrations.runner import MigrationRunner


class _StubMigration:
    def __init__(self, version=1, description="stub", fail=False):
        self.version = version
        self.description = description
        self.fail = fail
        self.up_calls = 0

    def up(self, executor):
        self.up_calls += 1
        if self.fail:
            raise RuntimeError(f"migration {self.version} boom")


class _FakeExecutor:
    """Stateful stand-in for QueryExecutor.

    Tracks ``_Migration`` markers as ``{version: {status, error,
    completed_at}}`` and routes the runner's Cypher by query substring,
    mirroring how the real runner talks to Neo4j:

    - ``max(m.version)`` read  → MAX version among ``status == 'completed'``
      (this is the production query shape after the fix).
    - ``status: 'failed'`` read → count of failed markers for ``params['v']``.
    - ``MATCH (t:Task)`` read  → task count (B-12 guard input).
    - ``MigrationLock`` write  → always ``{"status": "running"}`` (single
      worker, so the lock never blocks).
    - ``MERGE (m:_Migration ...)`` write → set status ``'running'`` (creates
      the node if absent; never a second node for the same version).
    - ``SET m.status = 'completed'`` write → flip the existing node to
      ``'completed'`` and clear its error.
    - ``SET m.status = 'failed'`` write → flip the existing node to
      ``'failed'`` and record the error.
    """

    def __init__(self, markers=None, task_count=0):
        # {version: {"status": str, "error": str|None, "completed_at": bool}}
        self.markers = dict(markers or {})
        self.task_count = task_count
        self.writes = []  # log of (query, params) for write assertions

    # -- read routing -------------------------------------------------------
    def execute_read(self, query, params=None, **kwargs):
        params = params or {}
        if "max(m.version)" in query:
            if "status = 'completed'" in query:
                versions = [
                    v for v, s in self.markers.items()
                    if s["status"] == "completed"
                ]
            else:
                # Old (buggy) query shape: MAX over all markers regardless of
                # status. Honored so the revert-test reflects production.
                versions = list(self.markers)
            return [{"version": max(versions) if versions else None}]
        if "status: 'failed'" in query:
            node = self.markers.get(params.get("v"))
            c = 1 if node and node["status"] == "failed" else 0
            return [{"c": c}]
        if "MATCH (t:Task)" in query:
            return [{"c": self.task_count}]
        return []

    # -- write routing ------------------------------------------------------
    def execute_write(self, query, params=None, **kwargs):
        params = params or {}
        self.writes.append((query, params))
        if "MigrationLock" in query:
            return [{"status": "running"}]
        if "MERGE (m:_Migration" in query:
            v = params.get("v")
            node = self.markers.setdefault(
                v, {"status": None, "error": None, "completed_at": False}
            )
            node["status"] = "running"
            return []
        if "SET m.status = 'completed'" in query:
            v = params.get("v")
            node = self.markers.setdefault(
                v, {"status": None, "error": None, "completed_at": False}
            )
            node["status"] = "completed"
            node["error"] = None
            node["completed_at"] = True
            return []
        if "SET m.status = 'failed'" in query:
            v = params.get("v")
            node = self.markers.setdefault(
                v, {"status": None, "error": None, "completed_at": False}
            )
            node["status"] = "failed"
            node["error"] = params.get("err")
            return []
        return []


class TestGetCurrentVersionIgnoresFailed(unittest.TestCase):
    """get_current_version() must return the last COMPLETED version."""

    def test_failed_marker_above_completed_is_ignored(self):
        """failed@5 + completed@3 → current == 3 (not 5)."""
        executor = _FakeExecutor(markers={
            5: {"status": "failed", "error": "boom", "completed_at": False},
            3: {"status": "completed", "error": None, "completed_at": True},
        })
        runner = MigrationRunner(executor)
        self.assertEqual(runner.get_current_version(), 3)

    def test_no_completed_markers_returns_zero(self):
        """Only a failed marker → current == 0 (empty-DB behavior preserved)."""
        executor = _FakeExecutor(markers={
            5: {"status": "failed", "error": "boom", "completed_at": False},
        })
        runner = MigrationRunner(executor)
        self.assertEqual(runner.get_current_version(), 0)

    def test_no_markers_returns_zero(self):
        executor = _FakeExecutor(markers={})
        runner = MigrationRunner(executor)
        self.assertEqual(runner.get_current_version(), 0)

    def test_max_completed_wins(self):
        """completed@2 + completed@7 + failed@9 → current == 7."""
        executor = _FakeExecutor(markers={
            2: {"status": "completed", "error": None, "completed_at": True},
            7: {"status": "completed", "error": None, "completed_at": True},
            9: {"status": "failed", "error": "x", "completed_at": False},
        })
        runner = MigrationRunner(executor)
        self.assertEqual(runner.get_current_version(), 7)


class TestRunPendingRetriesFailed(unittest.TestCase):
    """run_pending() must re-run a failed migration and heal its marker."""

    def _seed_failed(self, version):
        return {version: {"status": "failed", "error": "earlier boom",
                          "completed_at": False}}

    def test_retries_failed_migration_and_completes(self):
        executor = _FakeExecutor(markers=self._seed_failed(5), task_count=0)
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=5, description="retry me")
        runner.register(migration)

        applied = runner.run_pending()

        # up() ran again despite the pre-existing failed marker.
        self.assertEqual(migration.up_calls, 1)
        self.assertEqual(applied, [5])
        # The single marker node flipped failed → completed (one node/version).
        self.assertEqual(executor.markers[5]["status"], "completed")
        self.assertTrue(executor.markers[5]["completed_at"])
        # Stale error from the prior failure is cleared.
        self.assertIsNone(executor.markers[5]["error"])
        # No duplicate marker node was created for version 5.
        self.assertEqual(
            [v for v in executor.markers if v == 5], [5]
        )

    def test_retry_logs_warning(self):
        executor = _FakeExecutor(markers=self._seed_failed(5), task_count=0)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=5, description="retry me"))

        with self.assertLogs("memory.migrations.runner", level="WARNING") as cm:
            runner.run_pending()

        retry_msgs = [m for m in cm.output if "Retrying previously-failed" in m]
        self.assertEqual(len(retry_msgs), 1)
        self.assertIn("5", retry_msgs[0])

    def test_no_warning_for_fresh_migration(self):
        """A migration with no failed marker runs without the retry WARNING."""
        executor = _FakeExecutor(markers={}, task_count=0)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=1, description="fresh"))

        with self.assertLogs("memory.migrations.runner", level="INFO") as cm:
            runner.run_pending()

        self.assertFalse(
            any("Retrying previously-failed" in m for m in cm.output),
            f"unexpected retry warning: {cm.output!r}",
        )

    def test_completed_migration_is_not_rerun(self):
        """A completed marker still short-circuits (version <= current)."""
        executor = _FakeExecutor(markers={
            5: {"status": "completed", "error": None, "completed_at": True},
        })
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=5)
        runner.register(migration)

        applied = runner.run_pending()

        self.assertEqual(applied, [])
        self.assertEqual(migration.up_calls, 0)

    def test_failed_migration_still_writes_failed_on_repeat_failure(self):
        """If the retry also fails, the marker stays 'failed' and error updates."""
        executor = _FakeExecutor(markers=self._seed_failed(5), task_count=0)
        runner = MigrationRunner(executor)
        runner.register(_StubMigration(version=5, fail=True))

        with self.assertRaises(RuntimeError):
            runner.run_pending()

        self.assertEqual(executor.markers[5]["status"], "failed")
        self.assertIn("boom", executor.markers[5]["error"])


class TestRevertGuard(unittest.TestCase):
    """Revert-test: the pre-fix get_current_version (MAX regardless of status)
    must cause the retry to be skipped — proving the retry test discriminates
    the fix. This simulates reverting the one-line query change without
    touching the production file."""

    def test_old_behavior_skips_failed_migration(self):
        executor = _FakeExecutor(markers={
            5: {"status": "failed", "error": "boom", "completed_at": False},
        })
        runner = MigrationRunner(executor)
        migration = _StubMigration(version=5)
        runner.register(migration)

        # Simulate the OLD (buggy) get_current_version: MAX over all markers
        # regardless of status. The fake executor's max() read filters on
        # 'completed', so emulate the old all-status MAX directly from state.
        def old_all_status_max():
            return max(executor.markers) if executor.markers else 0

        runner.get_current_version = old_all_status_max

        applied = runner.run_pending()

        # Under the old behavior current == 5, so version 5 is skipped forever.
        self.assertEqual(applied, [])
        self.assertEqual(migration.up_calls, 0)
        self.assertEqual(executor.markers[5]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
