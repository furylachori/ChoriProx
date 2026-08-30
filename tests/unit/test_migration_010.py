"""Tests for migration 010: StripWorkspacePrefix.

The v5 workspace rework replaces compound `session_id:name`
workspace_ids with bare `name` workspace_ids. Migration 010 is responsible
for the in-place rewrite of existing Task and Bug data, plus a follow-up
phase that backfills NULL workspace_ids to 'default', re-keys
:Workspace nodes, and deletes orphaned compound :Workspace nodes.

These tests pin down the Cypher the migration emits, using a mock
executor that records every `execute_write` call. Real Cypher strings
are used so the tests catch typos and structural drift, but the
executor returns scripted values for `migrated` / `backfilled` /
`rekeyed` / `deleted` so the migration can drive its own loops.
"""
import importlib
import inspect

import pytest

from memory.migrations.base import Migration
from memory.migrations.runner import MigrationRunner


migration_module = importlib.import_module(
    "memory.migrations.010_strip_workspace_prefix"
)
StripWorkspacePrefix = migration_module.StripWorkspacePrefix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingExecutor:
    """Mock executor that records every write call and returns scripted results.

    The first N `execute_write` results are taken from `results` (in order);
    subsequent calls return `[]`. Tests can also inspect `calls` to assert
    on the exact Cypher string and params.
    """

    def __init__(self, results=None):
        self.calls: list[dict] = []
        self._results = list(results or [])

    def execute_write(self, query, params=None, timeout=30):
        self.calls.append({"query": query, "params": params or {}})
        if self._results:
            return self._results.pop(0)
        return []

    def execute_read(self, query, params=None, timeout=30):
        self.calls.append({"query": query, "params": params or {}, "op": "read"})
        return []


def _all_queries(executor) -> list[str]:
    """Convenience: just the Cypher strings from a RecordingExecutor."""
    return [c["query"] for c in executor.calls]


def _first_query_containing(executor, needle: str) -> str:
    """Return the first call whose query contains `needle` (case-insensitive)."""
    for c in executor.calls:
        if needle.lower() in c["query"].lower():
            return c["query"]
    raise AssertionError(f"No call with query containing {needle!r}")


def _all_queries_containing(executor, needle: str) -> list[str]:
    return [c["query"] for c in executor.calls if needle.lower() in c["query"].lower()]


@pytest.fixture
def migration():
    return StripWorkspacePrefix()


@pytest.fixture
def executor():
    # Default: every call returns [{...}] with no rows affected.
    return _RecordingExecutor(results=[[{"migrated": 0}], [{"migrated": 0}],
                                       [{"backfilled": 0}], [{"backfilled": 0}],
                                       [{"rekeyed": 0}], [{"deleted": 0}]])


# ---------------------------------------------------------------------------
# Test class — migration shape
# ---------------------------------------------------------------------------


class TestMigration010Import:
    def test_can_import_migration_module(self):
        mod = importlib.import_module("memory.migrations.010_strip_workspace_prefix")
        assert mod is not None

    def test_migration_class_exported(self):
        mod = importlib.import_module("memory.migrations.010_strip_workspace_prefix")
        assert hasattr(mod, "StripWorkspacePrefix")

    def test_migration_version_is_ten(self):
        mod = importlib.import_module("memory.migrations.010_strip_workspace_prefix")
        cls = mod.StripWorkspacePrefix
        assert cls.version == 10

    def test_migration_inherits_from_migration_base(self):
        mod = importlib.import_module("memory.migrations.010_strip_workspace_prefix")
        cls = mod.StripWorkspacePrefix
        assert issubclass(cls, Migration)

    def test_migration_description_mentions_strip(self):
        mod = importlib.import_module("memory.migrations.010_strip_workspace_prefix")
        cls = mod.StripWorkspacePrefix
        assert "strip" in cls.description.lower()

    def test_migration_registered_in_package_init(self):
        from memory.migrations import StripWorkspacePrefix as Reg
        assert Reg is not None
        assert Reg.version == 10


class TestMigration010Registration:
    def test_migration_can_be_registered_with_runner(self):
        class FakeExecutor:
            pass

        runner = MigrationRunner(FakeExecutor())
        runner.register(StripWorkspacePrefix())
        assert any(m.version == 10 for m in runner._migrations)

    def test_migration_sorts_after_009(self):
        mod_009 = importlib.import_module("memory.migrations.009_workspace_backfill")
        WorkspaceBackfill = mod_009.WorkspaceBackfill

        class FakeExecutor:
            pass

        runner = MigrationRunner(FakeExecutor())
        runner.register(StripWorkspacePrefix())
        runner.register(WorkspaceBackfill())
        versions = [m.version for m in runner._migrations]
        assert versions == sorted(versions)
        assert 9 in versions
        assert 10 in versions


# ---------------------------------------------------------------------------
# Test class — Phase 1: strip Task prefixes
# ---------------------------------------------------------------------------


class TestMigration010StripTaskPrefixes:
    def test_010_strips_task_prefixes(self, migration, executor):
        """Phase 1: strip compound prefix from :Task.workspace_id, paginated."""
        # First call returns 5 rows (one full BATCH_SIZE=5000 page would be the
        # signal to keep going; 5 means we're done after this single page).
        executor._results.insert(0, [{"migrated": 5}])
        # Pre-load other phases with empty results.
        for _ in range(5):
            executor._results.append([{k: 0} for k in ("migrated", "backfilled",
                                                       "backfilled", "rekeyed",
                                                       "deleted")])

        migration.up(executor)

        queries = _all_queries(executor)
        # Find the Task-strip query
        task_query = next(
            (q for q in queries
             if "MATCH (t:Task)" in q and "SET t.workspace_id" in q
             and "parts" in q),
            None,
        )
        assert task_query is not None, (
            "No Task-strip query found. Saw:\n" + "\n--\n".join(queries)
        )
        # The SET clause must use parts[-1] (the bare name).
        assert "SET t.workspace_id = parts[-1]" in task_query, (
            f"Expected SET t.workspace_id = parts[-1] in:\n{task_query}"
        )
        # The WHERE clause must filter on CONTAINS ':' (idempotency guard).
        assert "CONTAINS ':'" in task_query, (
            f"Expected 'CONTAINS \\':\\'' in WHERE clause of:\n{task_query}"
        )

    def test_010_uses_pagination(self, migration, executor):
        """Phase 1 must paginate with SKIP/LIMIT and the third partial page
        breaks the loop (12K tasks → 3 pages of 5000, last page has < 5000).
        """
        # Three Phase-1 calls: full 5000, full 5000, partial 2000 (< BATCH_SIZE).
        # Then one Phase-2 result (not used here, but the queue may carry it).
        executor._results = [
            [{"migrated": 5000}],  # first Phase-1 page
            [{"migrated": 5000}],  # second Phase-1 page
            [{"migrated": 2000}],  # third Phase-1 page — partial, break
            [{"migrated": 0}],     # Phase 2
            [{"backfilled": 0}],   # Phase 3
            [{"backfilled": 0}],   # Phase 3b
            [{"rekeyed": 0}],      # Phase 4
            [{"deleted": 0}],      # Phase 5
        ]

        migration.up(executor)

        # Inspect the Phase-1 call params: skip should advance by 5000 each step.
        phase1_calls = [
            c for c in executor.calls
            if "MATCH (t:Task)" in c["query"] and "parts" in c["query"]
        ]
        skips = [c["params"].get("skip") for c in phase1_calls]
        # Three Phase-1 calls means we see skips 0, 5000, 10000.
        assert skips == [0, 5000, 10000], (
            f"Expected Phase-1 pagination [0, 5000, 10000], got {skips}"
        )
        # The third page returned < 5000 → loop must break (no fourth Phase-1 call).
        assert len(phase1_calls) == 3, (
            f"Phase-1 loop did not break on partial page (calls={len(phase1_calls)})"
        )


class TestMigration010StripBugPrefixes:
    def test_010_strips_bug_prefixes(self, migration, executor):
        """Phase 2: strip compound prefix from :Bug.workspace_id."""
        # Make Phase 1 a no-op (5 calls, all with 0 rows).
        executor._results = [
            [{"migrated": 0}],     # Phase 1 (one page, 0 rows → break)
            [{"migrated": 0}],     # Phase 2
            [{"backfilled": 0}],   # Phase 3
            [{"backfilled": 0}],   # Phase 3b
            [{"rekeyed": 0}],      # Phase 4
            [{"deleted": 0}],      # Phase 5
        ]
        migration.up(executor)

        queries = _all_queries(executor)
        bug_query = next(
            (q for q in queries
             if "MATCH (b:Bug)" in q and "SET b.workspace_id" in q
             and "parts" in q),
            None,
        )
        assert bug_query is not None, (
            "No Bug-strip query found. Saw:\n" + "\n--\n".join(queries)
        )
        assert "SET b.workspace_id = parts[-1]" in bug_query, (
            f"Expected SET b.workspace_id = parts[-1] in:\n{bug_query}"
        )
        assert "CONTAINS ':'" in bug_query, (
            f"Expected 'CONTAINS \\':\\'' in WHERE clause of:\n{bug_query}"
        )


class TestMigration010NullHandling:
    def test_010_handles_null_workspace_id(self, migration, executor):
        """The Phase-1 / Phase-2 WHERE must include IS NOT NULL so we don't
        try to split() a NULL string.
        """
        migration.up(executor)
        queries = _all_queries(executor)

        # Phase 1 (Task) and Phase 2 (Bug) queries must both guard on IS NOT NULL.
        task_strip = next(
            (q for q in queries
             if "MATCH (t:Task)" in q and "SET t.workspace_id" in q
             and "parts" in q),
            None,
        )
        bug_strip = next(
            (q for q in queries
             if "MATCH (b:Bug)" in q and "SET b.workspace_id" in q
             and "parts" in q),
            None,
        )
        assert task_strip is not None
        assert bug_strip is not None
        assert "IS NOT NULL" in task_strip, (
            f"Task strip query must guard on IS NOT NULL:\n{task_strip}"
        )
        assert "IS NOT NULL" in bug_strip, (
            f"Bug strip query must guard on IS NOT NULL:\n{bug_strip}"
        )


class TestMigration010AlreadyBare:
    def test_010_handles_already_bare(self, migration, executor):
        """The CONTAINS ':' filter ensures already-bare workspace_ids are
        skipped — this is the idempotency guard.
        """
        migration.up(executor)
        queries = _all_queries(executor)

        task_strip = next(
            (q for q in queries
             if "MATCH (t:Task)" in q and "parts" in q),
            None,
        )
        bug_strip = next(
            (q for q in queries
             if "MATCH (b:Bug)" in q and "parts" in q),
            None,
        )
        assert task_strip is not None
        assert bug_strip is not None
        assert "CONTAINS ':'" in task_strip, (
            f"Task strip query must filter on CONTAINS ':' for idempotency:\n"
            f"{task_strip}"
        )
        assert "CONTAINS ':'" in bug_strip, (
            f"Bug strip query must filter on CONTAINS ':' for idempotency:\n"
            f"{bug_strip}"
        )


class TestMigration010MultipleColons:
    def test_010_handles_multiple_colons(self, migration, executor):
        """parts[-1] (not parts[0]) makes the migration robust to multiple
        colons in a workspace_id, defensive against pathological inputs.
        """
        migration.up(executor)
        queries = _all_queries(executor)

        for label, prefix in (("Task", "t:"), ("Bug", "b:")):
            strip = next(
                (q for q in queries
                 if f"MATCH ({prefix[0]}:{label})" in q
                 and "parts" in q),
                None,
            )
            assert strip is not None, f"No {label} strip query found"
            assert "parts[-1]" in strip, (
                f"{label} strip query must use parts[-1] (last segment):\n"
                f"{strip}"
            )
            assert "parts[0]" not in strip, (
                f"{label} strip query must NOT use parts[0] (would pick the "
                f"session_id, not the bare name):\n{strip}"
            )


class TestMigration010BackfillNull:
    def test_010_backfills_null_to_default(self, migration, executor):
        """Phase 3 must have a query that SETs NULL workspace_id to 'default'."""
        migration.up(executor)
        queries = _all_queries(executor)

        # Look for both Task and Bug backfill queries: each must touch
        # NULL workspace_id and SET to 'default'.
        task_backfill = next(
            (q for q in queries
             if "MATCH (t:Task)" in q
             and "t.workspace_id IS NULL" in q
             and "SET t.workspace_id = 'default'" in q),
            None,
        )
        bug_backfill = next(
            (q for q in queries
             if "MATCH (b:Bug)" in q
             and "b.workspace_id IS NULL" in q
             and "SET b.workspace_id = 'default'" in q),
            None,
        )
        assert task_backfill is not None, (
            "No Task NULL backfill query found. Saw:\n"
            + "\n--\n".join(queries)
        )
        assert bug_backfill is not None, (
            "No Bug NULL backfill query found. Saw:\n"
            + "\n--\n".join(queries)
        )


class TestMigration010RekeyWorkspace:
    def test_010_rekeys_workspace_nodes(self, migration, executor):
        """Phase 4 must:
          (a) find :Workspace nodes with compound workspace_id,
          (b) re-key them to the bare name,
          (c) drop the now-redundant session_id property.
        """
        migration.up(executor)
        queries = _all_queries(executor)

        rekey = next(
            (q for q in queries
             if "MATCH (w:Workspace)" in q
             and "w.workspace_id CONTAINS ':'" in q
             and "SET w.workspace_id" in q),
            None,
        )
        assert rekey is not None, (
            "No :Workspace rekey query found. Saw:\n"
            + "\n--\n".join(queries)
        )
        # Must drop the session_id property.
        assert "REMOVE w.session_id" in rekey, (
            f"Rekey query must REMOVE w.session_id:\n{rekey}"
        )


class TestMigration010DeleteOrphans:
    def test_010_deletes_orphans(self, migration, executor):
        """Phase 5: detach-delete compound :Workspace nodes that have no
        remaining Task/Bug references.
        """
        migration.up(executor)
        queries = _all_queries(executor)

        orphan = next(
            (q for q in queries
             if "MATCH (w:Workspace)" in q
             and "w.workspace_id CONTAINS ':'" in q
             and "DETACH DELETE" in q),
            None,
        )
        assert orphan is not None, (
            "No orphan :Workspace delete query found. Saw:\n"
            + "\n--\n".join(queries)
        )
        # Scope: must be compound IDs only (not all :Workspace nodes).
        assert "CONTAINS ':'" in orphan, (
            f"Orphan delete must be scoped to compound IDs:\n{orphan}"
        )
        # Must check that no remaining Task/Bug reference the node.
        assert "Task" in orphan or "t:Task" in orphan, (
            f"Orphan delete should check for Task references:\n{orphan}"
        )


class TestMigration010Idempotent:
    def test_010_idempotent(self, migration, executor):
        """Re-running the migration on already-migrated data must be a no-op.

        We model "already migrated" by having the Phase-1 SELECT return 0
        rows on the second run — that breaks the paginated loop on the
        very first call.
        """
        # First run: one Phase-1 page with 0 rows → break, then 0-rows for
        # all other phases.
        first = _RecordingExecutor(results=[
            [{"migrated": 0}],
            [{"migrated": 0}],
            [{"backfilled": 0}],
            [{"backfilled": 0}],
            [{"rekeyed": 0}],
            [{"deleted": 0}],
        ])
        migration.up(first)
        first_call_count = len(first.calls)

        # Second run: also 0 rows. Re-running is a no-op.
        second = _RecordingExecutor(results=[
            [{"migrated": 0}],
            [{"migrated": 0}],
            [{"backfilled": 0}],
            [{"backfilled": 0}],
            [{"rekeyed": 0}],
            [{"deleted": 0}],
        ])
        migration.up(second)
        second_call_count = len(second.calls)

        # Same number of total write calls on both runs (idempotent).
        assert first_call_count == second_call_count, (
            f"Idempotency violated: first run made {first_call_count} calls, "
            f"second run made {second_call_count}"
        )

        # And on the second run, no SET is invoked for the strip phases
        # because the WHERE already excludes bare values.
        for c in second.calls:
            if "parts" in c["query"]:
                assert "CONTAINS ':'" in c["query"], (
                    "Strip query lost its CONTAINS ':' idempotency filter:\n"
                    f"{c['query']}"
                )


class TestMigration010SoftDeletedHandling:
    def test_010_soft_deleted_handling(self, migration, executor):
        """The migration must NOT filter by deleted_at.

        Soft-deleted tasks (deleted_at IS NOT NULL) must also be migrated.
        A v4 → v5 data rewrite should not lose track of soft-deleted rows.
        """
        migration.up(executor)
        queries = _all_queries(executor)

        for label, prefix in (("Task", "t:"), ("Bug", "b:")):
            strip = next(
                (q for q in queries
                 if f"MATCH ({prefix[0]}:{label})" in q
                 and "parts" in q),
                None,
            )
            assert strip is not None, f"No {label} strip query found"
            assert "deleted_at" not in strip, (
                f"{label} strip query must NOT filter by deleted_at "
                f"(soft-deleted rows should also be migrated):\n{strip}"
            )
