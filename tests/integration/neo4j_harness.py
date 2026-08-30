"""Neo4j test harness — provides isolated test databases with full schema.

Each test that opts in gets a UNIQUE database on an Enterprise Edition server
(the throwaway test container, bolt://localhost:7688). All migrations are
applied on setup and the schema version is verified before yielding.

POLICY — the harness NEVER wipes a shared database:
    * Production (:7687) is un-wipeable by any code path, any env var, any
      caller. Fixtures skip/mock rather than touch it.
    * The Community Edition "wipe the shared DB" fallback has been REMOVED.
      There is no opt-in flag — nothing an agent or human can set to enable a
      destructive wipe. If per-test isolation is unavailable (Community
      Edition, or Docker test container down), the test SKIPS.
    * Tests default to bolt://localhost:7688 (docker-compose.test.yml). If that
      container is not running, Neo4j tests skip automatically — they never
      fall back to production.

Fixtures:
    neo4j_test_db  — function-scoped, yields a QueryExecutor-compatible object
    real_or_mock_neo4j — returns real executor if Neo4j is up, MagicMock otherwise
"""

import logging
import os
from unittest.mock import MagicMock

import pytest

from tests.integration.neo4j_harness_utils import safe_db_name as _safe_db_name

logger = logging.getLogger(__name__)

# ── No-wipe policy ─────────────────────────────────────────────────────────
# This harness used to gate a destructive "wipe the shared Community DB"
# fallback behind an I_AM_AN_IDIOT_ASK_A_HUMAN=yes env var. That flag was a
# foot-gun: agents and humans set it without thinking and then wiped
# production (hundreds of wipes recorded in backups/). The flag is GONE.
# _clear_all_data now refuses UNCONDITIONALLY — there is nothing to set,
# nothing to forget, no judgement call. Isolation comes from per-test
# Enterprise databases; when isolation is unavailable, tests SKIP.
_SKIP_MSG = (
    "Neo4j test isolation is unavailable (need the Enterprise test container "
    "on bolt://localhost:7688 — start it with "
    "`docker compose -f docker-compose.test.yml up -d`). The test harness "
    "REFUSES to wipe any shared database and SKIPS instead. There is no "
    "override flag."
)


def _neo4j_available() -> bool:
    """Check if Neo4j is reachable without touching the shared driver singleton."""
    try:
        from neo4j import GraphDatabase, Query as Neo4jQuery
    except ImportError:
        return False
    bolt_url = _get_bolt_url()
    auth = _get_auth()
    driver = GraphDatabase.driver(bolt_url, auth=auth)
    try:
        session = driver.session()
        try:
            result = session.run(Neo4jQuery("RETURN 1", timeout=2.0))
            result.consume()
            return True
        finally:
            session.close()
    except Exception:
        return False
    finally:
        driver.close()


def _get_bolt_url() -> str:
    # Default to the ISOLATED test container (:7688), never production (:7687).
    # The Makefile also sets this explicitly; defaulting here means a bare
    # `pytest tests/integration` still targets the throwaway test DB, not prod.
    return os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7688")


def _is_prod_bolt_url(bolt_url: str) -> bool:
    """True if a bolt URL points at the PRODUCTION graph (port 7687).

    Tests must never run destructive operations against production. The
    production DB lives on :7687 (docker-compose.yml); the throwaway test DB
    lives on :7688 (docker-compose.test.yml). Detecting :7687 lets the harness
    refuse to wipe production even if the wipe gate is explicitly opened.

    Robust against trailing slashes, paths, query strings, IPv6 bracket hosts,
    out-of-range ports (``:76870`` must NOT match), and a missing port (bolt
    defaults to 7687, so ``bolt://prod-host`` IS production — fail-safe).
    """
    from urllib.parse import urlsplit
    netloc = urlsplit(bolt_url).netloc
    # Strip any path/query that urlsplit left attached, then isolate the port
    # after the host (IPv6 hosts are wrapped in [...]).
    hostpart = netloc.split("/", 1)[0].split("?", 1)[0]
    if hostpart.endswith("]"):  # IPv6 bracket host with no port
        port_str = ""
    elif ":" in hostpart:
        port_str = hostpart.rsplit(":", 1)[-1]
    else:
        port_str = ""
    if not port_str:
        # No explicit port → bolt default 7687 → treat as production (fail safe).
        return True
    return port_str == "7687"


def _get_auth() -> tuple:
    auth_str = os.environ.get("NEO4J_AUTH", "neo4j:password")
    return tuple(auth_str.split(":", 1)) if ":" in auth_str else ("neo4j", "password")


class _TestQueryExecutor:
    """Lightweight QueryExecutor-compatible wrapper around a Neo4j driver.

    Production ``QueryExecutor`` uses a global singleton driver, which cannot
    target a per-test database.  This wrapper holds its own driver so it can
    connect to any database.
    """

    def __init__(self, driver, database: str):
        self._driver = driver
        self._database = database

    def execute_read(self, query, params=None, timeout=30):
        session = self._driver.session(database=self._database)
        try:
            from neo4j import Query as Neo4jQuery
            result = session.run(Neo4jQuery(query, timeout=timeout), params or {})
            records = [dict(r) for r in result]
            result.consume()
            return records
        finally:
            session.close()

    def execute_write(self, query, params=None, timeout=30):
        session = self._driver.session(database=self._database)
        try:
            from neo4j import Query as Neo4jQuery
            result = session.run(Neo4jQuery(query, timeout=timeout), params or {})
            records = [dict(r) for r in result]
            result.consume()
            return records
        finally:
            session.close()

    def execute_write_no_retry(self, query, params=None, timeout=30):
        return self.execute_write(query, params, timeout)

    def execute_transaction(self, fn):
        session = self._driver.session(database=self._database)
        try:
            tx = session.begin_transaction()
            try:
                result = fn(tx)
                tx.commit()
                return result
            except Exception:
                tx.rollback()
                raise
        finally:
            session.close()

    def execute_transaction_no_retry(self, fn):
        return self.execute_transaction(fn)

    def is_available(self) -> bool:
        try:
            session = self._driver.session(database=self._database)
            try:
                from neo4j import Query as Neo4jQuery
                result = session.run(Neo4jQuery("RETURN 1", timeout=2.0))
                result.consume()
                return True
            finally:
                session.close()
        except Exception:
            return False


def _create_db(driver, db_name: str) -> bool:
    """Create a database and wait for it to come online.

    Returns True on success, False if unsupported (Community Edition).
    Enterprise Neo4j creates databases asynchronously — CREATE OR REPLACE
    returns immediately but the DB may not be 'online' yet, causing
    DatabaseNotFound races. We poll SHOW DATABASES until currentStatus='online'.
    """
    import time as _time
    session = driver.session(database="system")
    try:
        from neo4j import Query as Neo4jQuery
        result = session.run(Neo4jQuery(f"CREATE OR REPLACE DATABASE `{db_name}`", timeout=30))
        result.consume()
    except Exception as exc:
        logger.debug("CREATE DATABASE failed (likely Community Edition): %s", exc)
        return False
    finally:
        session.close()

    # Wait for the DB to come online (Enterprise creates asynchronously).
    for _ in range(30):
        _time.sleep(0.5)
        session = driver.session(database="system")
        try:
            row = session.run(
                Neo4jQuery("SHOW DATABASE $name", timeout=5),
                {"name": db_name},
            ).single()
        finally:
            session.close()
        if row and row.get("currentStatus") == "online":
            return True
    logger.warning(
        "Database %s did not come online within 15s — returning False "
        "(tests using this DB will be skipped)", db_name,
    )
    return False  # not online → caller skips the test (safe)


def _drop_db(driver, db_name: str) -> None:
    session = driver.session(database="system")
    try:
        from neo4j import Query as Neo4jQuery
        result = session.run(Neo4jQuery(f"DROP DATABASE `{db_name}` IF EXISTS", timeout=30))
        result.consume()
    except Exception as exc:
        logger.warning("DROP DATABASE %s failed: %s", db_name, exc)
    finally:
        session.close()


def _clear_all_data(driver, database: str) -> None:
    """Refuse to wipe a shared database. UNCONDITIONAL — no flag, no override.

    This function used to run ``MATCH (n) DETACH DELETE n`` on the shared
    Community Edition database, gated behind I_AM_AN_IDIOT_ASK_A_HUMAN=yes.
    That gate was removed because it was a foot-gun (agents set the flag and
    wiped production). The harness now isolates tests into per-test Enterprise
    databases and NEVER wipes a shared DB. This function is retained as a
    safe no-op so that any residual call path cannot destroy data.
    """
    logger.warning(
        "_clear_all_data refused: the test harness NEVER wipes a shared "
        "database (target %r, bolt URL %r). Tests isolate via per-test "
        "Enterprise databases on the test container "
        "(PROXY_NEO4J_BOLT_URL=bolt://localhost:7688). There is no override "
        "flag. NOT wiping.",
        database, _get_bolt_url(),
    )
    return


def _run_all_migrations(executor: _TestQueryExecutor) -> int:
    """Run every registered migration and return the final schema version."""
    from memory.migrations.runner import MigrationRunner
    from memory.migrations import (
        BackfillParentTaskUuid,
        VerifyBackfill,
        RemoveProjectRelationships,
        GraphNativeTasks,
        EnsureGraphNative,
        AgentTaskAssignmentMigration,
        WorkSessionMigration,
        BugTrackingMigration,
        WorkspaceBackfill,
        StripWorkspacePrefix,
        EntityDedupConstraint,
    )

    runner = MigrationRunner(executor)
    for migration_cls in (
        BackfillParentTaskUuid,
        VerifyBackfill,
        RemoveProjectRelationships,
        GraphNativeTasks,
        EnsureGraphNative,
        AgentTaskAssignmentMigration,
        WorkSessionMigration,
        BugTrackingMigration,
        WorkspaceBackfill,
        StripWorkspacePrefix,
        EntityDedupConstraint,
    ):
        runner.register(migration_cls())

    runner.run_pending()
    return runner.get_current_version()


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def neo4j_test_db(request):
    """Opt-in fixture: isolated Neo4j database with full schema.

    Creates a unique database per test (Neo4j 5.x Enterprise) or cleans the
    default database (Community fallback).  Runs all migrations and verifies
    the schema version before yielding.

    Yields a ``_TestQueryExecutor`` (same interface as ``QueryExecutor``).
    Teardown drops the database or cleans data, even on failure/skip.

    Usage::

        @pytest.mark.integration_neo4j
        def test_something(neo4j_test_db):
            executor = neo4j_test_db
            rows = executor.execute_read("MATCH (n) RETURN count(n) AS cnt")
    """
    if not _neo4j_available():
        pytest.skip("Neo4j is not running — skipping integration_neo4j test")
    if _is_prod_bolt_url(_get_bolt_url()):
        pytest.skip(
            "PROXY_NEO4J_BOLT_URL targets PRODUCTION (:7687). The test harness "
            "refuses to run DDL/migrations against prod. Use the isolated "
            "container: PROXY_NEO4J_BOLT_URL=bolt://localhost:7688 "
            "NEO4J_AUTH=neo4j:testpassword"
        )
    from neo4j import GraphDatabase, Query as Neo4jQuery

    bolt_url = _get_bolt_url()
    auth = _get_auth()
    test_id = getattr(request.node, "nodeid", request.node.name)
    db_name = _safe_db_name(test_id)
    created_db = False

    driver = GraphDatabase.driver(bolt_url, auth=auth)
    try:
        created_db = _create_db(driver, db_name)
        if not created_db:
            # Community Edition (or no CREATE DATABASE support) cannot isolate
            # this test. We refuse to wipe the shared DB, so SKIP.
            driver.close()
            pytest.skip(_SKIP_MSG)

        executor = _TestQueryExecutor(driver, database=db_name)
        final_version = _run_all_migrations(executor)

        from memory.migrations import EntityDedupConstraint
        expected_version = EntityDedupConstraint.version
        assert final_version == expected_version, (
            f"Schema version mismatch: expected {expected_version}, got {final_version}"
        )

        yield executor
    finally:
        if created_db:
            _drop_db(driver, db_name)
        driver.close()


@pytest.fixture
def real_or_mock_neo4j(request):
    """Opt-in fixture: real executor if Neo4j is up, MagicMock otherwise.

    When Neo4j is available, creates an isolated test database with full
    schema (same as ``neo4j_test_db``).  When it is not, returns a
    ``MagicMock`` with sensible defaults so tests can run without a database.

    Usage::

        def test_something(real_or_mock_neo4j):
            executor = real_or_mock_neo4j
            executor.execute_read.return_value = [{"cnt": 0}]  # mock path
            # or, when Neo4j is running, real queries work directly
    """
    if not _neo4j_available():
        mock = MagicMock()
        mock.execute_read.return_value = []
        mock.execute_write.return_value = []
        mock.is_available.return_value = False
        yield mock
        return

    if _is_prod_bolt_url(_get_bolt_url()):
        # Never create/drop databases on production — fall back to a mock so
        # the test runs without touching prod (same contract as Neo4j-down).
        mock = MagicMock()
        mock.execute_read.return_value = []
        mock.execute_write.return_value = []
        mock.is_available.return_value = False
        yield mock
        return

    # Neo4j is available — create an isolated database (same logic as neo4j_test_db)
    from neo4j import GraphDatabase

    bolt_url = _get_bolt_url()
    auth = _get_auth()
    test_id = getattr(request.node, "nodeid", request.node.name)
    db_name = _safe_db_name(test_id)
    created_db = False

    driver = GraphDatabase.driver(bolt_url, auth=auth)
    try:
        created_db = _create_db(driver, db_name)
        if not created_db:
            # No isolation available — fall back to a mock rather than wipe the
            # shared DB (this fixture's contract is real-executor-or-mock).
            driver.close()
            mock = MagicMock()
            mock.execute_read.return_value = []
            mock.execute_write.return_value = []
            mock.is_available.return_value = False
            yield mock
            return

        executor = _TestQueryExecutor(driver, database=db_name)
        _run_all_migrations(executor)

        yield executor
    finally:
        if created_db:
            _drop_db(driver, db_name)
        driver.close()
