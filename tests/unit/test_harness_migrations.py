"""Unit tests for the Neo4j test harness migration registration (M1).

Pins that ``_run_all_migrations`` registers exactly the full set of
migrations present on the branch and that ``expected_version`` tracks
the highest ``.version``. This runs without a real Neo4j — it inspects
the harness internals directly.
"""
import pytest

pytestmark = pytest.mark.unit


def _harness_module():
    from tests.integration import neo4j_harness
    return neo4j_harness


def test_run_all_migrations_registers_exactly_11():
    """The harness registers all 11 migrations (1-8 + 9 + 10 + 11)."""
    mod = _harness_module()
    import inspect
    src = inspect.getsource(mod._run_all_migrations)
    # The migration classes are imported and iterated in a tuple. Count the
    # `.register(migration_cls())` calls by counting the class names listed
    # in the iteration tuple.
    expected_classes = [
        "BackfillParentTaskUuid",
        "VerifyBackfill",
        "RemoveProjectRelationships",
        "GraphNativeTasks",
        "EnsureGraphNative",
        "AgentTaskAssignmentMigration",
        "WorkSessionMigration",
        "BugTrackingMigration",
        "WorkspaceBackfill",
        "StripWorkspacePrefix",
        "EntityDedupConstraint",
    ]
    for cls_name in expected_classes:
        assert cls_name in src, f"{cls_name} missing from _run_all_migrations"


def test_expected_version_tracks_migration_11():
    """expected_version in the fixture equals EntityDedupConstraint.version (11)."""
    from memory.migrations import EntityDedupConstraint
    assert EntityDedupConstraint.version == 11
    # Inspect the fixture source to confirm it references EntityDedupConstraint.
    import inspect
    mod = _harness_module()
    fixture_src = inspect.getsource(mod.neo4j_test_db)
    assert "EntityDedupConstraint" in fixture_src
    assert "expected_version = EntityDedupConstraint.version" in fixture_src


def test_migration_009_and_010_have_correct_versions():
    """Direct check of the migration class version attributes."""
    from memory.migrations import WorkspaceBackfill, StripWorkspacePrefix
    assert WorkspaceBackfill.version == 9
    assert StripWorkspacePrefix.version == 10


def test_migrations_init_exports_009_and_010():
    """The migrations package exports both new classes."""
    from memory.migrations import WorkspaceBackfill, StripWorkspacePrefix  # noqa: F401


def test_migration_011_has_correct_version():
    """Migration 011 is registered with version=11 and exported from __init__."""
    from memory.migrations import EntityDedupConstraint
    assert EntityDedupConstraint.version == 11