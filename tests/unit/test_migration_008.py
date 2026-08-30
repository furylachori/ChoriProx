"""Tests for migration 008: Bug tracking graph."""
import importlib
import inspect

import pytest

from memory.migrations.base import Migration
from memory.migrations.runner import MigrationRunner


class TestMigration008Import:
    def test_can_import_migration_module(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        assert mod is not None

    def test_migration_class_exported(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        assert hasattr(mod, "BugTrackingMigration")

    def test_migration_version_is_eight(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        cls = mod.BugTrackingMigration
        assert cls.version == 8

    def test_migration_inherits_from_migration_base(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        cls = mod.BugTrackingMigration
        assert issubclass(cls, Migration)

    def test_migration_description_mentions_bug(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        cls = mod.BugTrackingMigration
        assert "bug" in cls.description.lower()

    def test_migration_registered_in_package_init(self):
        from memory.migrations import BugTrackingMigration
        assert BugTrackingMigration is not None
        assert BugTrackingMigration.version == 8


class TestMigration008Up:
    def test_up_executes_five_statements(self):
        """up() must issue exactly five write calls: 3 constraints + 2 indexes."""
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        migration = mod.BugTrackingMigration()
        migration.up(FakeExecutor())

        assert len(calls) == 5

    def test_up_creates_bug_uuid_constraint(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        mod.BugTrackingMigration().up(FakeExecutor())
        bug_uuid_q = next(q for q in calls if "bug_uuid_unique" in q)
        assert "CREATE CONSTRAINT bug_uuid_unique IF NOT EXISTS" in bug_uuid_q
        assert "FOR (b:Bug) REQUIRE b.uuid IS UNIQUE" in bug_uuid_q

    def test_up_creates_bug_short_hash_workspace_constraint(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        mod.BugTrackingMigration().up(FakeExecutor())
        composite_q = next(q for q in calls if "bug_short_hash_workspace_unique" in q)
        assert "FOR (b:Bug) REQUIRE (b.short_hash, b.workspace_id) IS UNIQUE" in composite_q

    def test_up_creates_review_uuid_constraint(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        mod.BugTrackingMigration().up(FakeExecutor())
        review_q = next(q for q in calls if "review_uuid_unique" in q)
        assert "FOR (r:Review) REQUIRE r.uuid IS UNIQUE" in review_q

    def test_up_creates_status_index(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        mod.BugTrackingMigration().up(FakeExecutor())
        status_idx_q = next(q for q in calls if "bug_status" in q and "INDEX" in q)
        assert "FOR (b:Bug) ON (b.status)" in status_idx_q

    def test_up_creates_workspace_status_index(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        calls = []

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                calls.append(query)
                return []

        mod.BugTrackingMigration().up(FakeExecutor())
        ws_idx_q = next(q for q in calls if "bug_workspace_status" in q)
        assert "FOR (b:Bug) ON (b.workspace_id, b.status)" in ws_idx_q

    def test_up_is_idempotent_via_if_not_exists(self):
        """All statements include IF NOT EXISTS so re-running is safe."""
        mod = importlib.import_module("memory.migrations.008_bug_tracking")
        migration = mod.BugTrackingMigration()
        src = inspect.getsource(migration.up)
        # 3 constraints + 2 indexes = 5 IF NOT EXISTS clauses
        assert src.count("IF NOT EXISTS") == 5

    def test_up_propagates_infrastructure_error(self):
        """A failing write must not be silently swallowed by up()."""
        from memory.query_executor import InfrastructureError

        mod = importlib.import_module("memory.migrations.008_bug_tracking")

        class FakeExecutor:
            def execute_write(self, query, params=None, timeout=30):
                raise InfrastructureError("simulated failure")

        migration = mod.BugTrackingMigration()
        with pytest.raises(InfrastructureError):
            migration.up(FakeExecutor())


class TestMigration008Registration:
    def test_migration_can_be_registered_with_runner(self):
        mod = importlib.import_module("memory.migrations.008_bug_tracking")

        class FakeExecutor:
            pass

        runner = MigrationRunner(FakeExecutor())
        runner.register(mod.BugTrackingMigration())
        assert any(m.version == 8 for m in runner._migrations)

    def test_migration_sorts_after_007(self):
        import importlib
        mod_007 = importlib.import_module("memory.migrations.007_work_sessions")
        mod_008 = importlib.import_module("memory.migrations.008_bug_tracking")
        WorkSessionMigration = mod_007.WorkSessionMigration
        BugTrackingMigration = mod_008.BugTrackingMigration

        class FakeExecutor:
            pass

        runner = MigrationRunner(FakeExecutor())
        runner.register(WorkSessionMigration())
        runner.register(BugTrackingMigration())
        versions = [m.version for m in runner._migrations]
        assert versions == sorted(versions)
        assert 7 in versions
        assert 8 in versions
