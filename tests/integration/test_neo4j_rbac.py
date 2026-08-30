"""Integration tests for the Neo4j RBAC rule (B-11).

The defensive rule is in ``docker/neo4j-prod-init.cypher``:

    DENY DELETE ON GRAPH * NODE Task TO admin, architect, editor, publisher, reader;

This rule is the strongest defence against the agent-via-bash data-loss
vector: even if an LLM agent discovers the prod password and runs

    cypher-shell -u neo4j -p <password> "MATCH (t:Task) DETACH DELETE t"

the DENY rule blocks it — :Task nodes cannot be hard-deleted by any role.

The proxy's normal operations (MERGE/SET/CREATE/REMOVE) are unaffected.

NOTE: Requires a live Neo4j Enterprise container with the RBAC rule applied
(via ``bash scripts/neo4j-apply-rbac.sh``). The test is SKIPPED by default
to avoid touching the production container — to run it explicitly:

    PROXY_NEO4J_BOLT_URL=bolt://localhost:7688 \\
    NEO4J_AUTH=neo4j:testpassword \\
    RBAC_TEST_ALLOW=yes \\
    PYTHONPATH=src pytest tests/integration/test_neo4j_rbac.py -v

KNOWN GAPS (B-7): only ``:Task`` is protected. Other labels
(``:Workspace``, ``:Session``, ``:Bug``, ``:ReviewNote``) are NOT protected
by this rule — a future hardening pass should extend the deny rule to
those labels too. This test suite documents that gap by checking the
inverse (i.e. that the deny rule does NOT block deletion of non-Task
labels, which is the current expected behavior).
"""
import os

import pytest


_RBAC_TEST_ALLOWED = os.environ.get("RBAC_TEST_ALLOW") == "yes"


pytestmark = pytest.mark.integration_neo4j


def _is_skipped_by_default() -> bool:
    return not _RBAC_TEST_ALLOWED


_SKIP_REASON = (
    "RBAC tests are skipped by default to avoid touching the production "
    "Neo4j container. Set RBAC_TEST_ALLOW=yes to opt in. This requires a "
    "live Enterprise container with the deny rule loaded — run "
    "scripts/neo4j-apply-rbac.sh against the test container first."
)


def _neo4j_alive():
    try:
        from neo4j import GraphDatabase
        bolt_url = os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7688")
        auth_str = os.environ.get("NEO4J_AUTH", "neo4j:password")
        auth = tuple(auth_str.split(":", 1)) if ":" in auth_str else ("neo4j", "password")
        driver = GraphDatabase.driver(bolt_url, auth=auth)
        try:
            with driver.session() as s:
                s.run("RETURN 1").consume()
            return True
        finally:
            driver.close()
    except Exception:
        return False


@pytest.fixture
def neo4j_driver():
    """Yield a Neo4j driver.

    Skips unless ``RBAC_TEST_ALLOW=yes`` is set AND a live Neo4j is reachable.
    Never runs against production (:7687) — these tests run DETACH DELETE on
    probe nodes, so they are confined to the isolated test container.
    """
    if _is_skipped_by_default():
        pytest.skip(_SKIP_REASON)
    from tests.integration.neo4j_harness import _is_prod_bolt_url
    bolt_url = os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7688")
    if _is_prod_bolt_url(bolt_url):
        pytest.skip(
            "RBAC tests refuse to run against production (:7687) — they run "
            "DETACH DELETE on probe nodes. Use the test container :7688."
        )
    if not _neo4j_alive():
        pytest.skip("Neo4j is not running")
    from neo4j import GraphDatabase
    auth_str = os.environ.get("NEO4J_AUTH", "neo4j:password")
    auth = tuple(auth_str.split(":", 1)) if ":" in auth_str else ("neo4j", "password")
    driver = GraphDatabase.driver(bolt_url, auth=auth)
    try:
        yield driver
    finally:
        driver.close()


def _has_deny_rule(neo4j_driver, action: str, node_label: str) -> bool:
    """Check SHOW PRIVILEGES for a DENY rule on the given node label and action.

    Neo4j Enterprise uses the internal action name ``delete_element`` and
    the segment form ``NODE(Task)`` (uppercase). This helper normalises
    both so the test reads naturally as ``("delete", "Task")``.
    """
    action_aliases = {
        "delete": "delete_element",
        "create": "create_element",
    }
    target_action = action_aliases.get(action.lower(), action)
    target_segment = f"NODE({node_label})".upper()
    with neo4j_driver.session() as s:
        result = s.run("SHOW PRIVILEGES")
        rows = [dict(r) for r in result]
    for row in rows:
        if (
            row.get("access") == "DENIED"
            and row.get("action", "").lower() == target_action.lower()
            and str(row.get("segment", "")).upper() == target_segment
        ):
            return True
    return False


class TestRbacRuleLoaded:
    """The deny rule must be loaded into the running Neo4j container."""

    def test_deny_delete_on_task_is_loaded(self, neo4j_driver):
        assert _has_deny_rule(
            neo4j_driver, "delete", "Task"
        ), (
            "DENY DELETE on :Task not present in SHOW PRIVILEGES. "
            "Run scripts/neo4j-apply-rbac.sh against the test container."
        )


class TestRbacBlocksDeleteOnTask:
    """The deny rule actually prevents ``MATCH (t:Task) DETACH DELETE t``."""

    def test_delete_task_raises_for_neo4j_user(self, neo4j_driver):
        marker_uuid = "rbac-blocks-delete-test"
        with neo4j_driver.session() as s:
            try:
                s.run("MERGE (t:Task {uuid: $u}) SET t.name = 'probe'", u=marker_uuid)
            except Exception:
                pass
            with pytest.raises(Exception) as exc_info:
                s.run("MATCH (t:Task {uuid: $u}) DETACH DELETE t", u=marker_uuid)
            assert "not allowed" in str(exc_info.value).lower() or "denied" in str(exc_info.value).lower(), (
                f"expected access denied; got: {exc_info.value!r}"
            )


class TestRbacDoesNotBlockOtherOps:
    """The deny rule does NOT block MERGE / SET on :Task (the proxy's normal path)."""

    def test_merge_task_works(self, neo4j_driver):
        marker_uuid = "rbac-merge-test"
        with neo4j_driver.session() as s:
            s.run("MERGE (t:Task {uuid: $u}) SET t.name = 'ok'", u=marker_uuid)
            row = s.run(
                "MATCH (t:Task {uuid: $u}) RETURN t.name AS name", u=marker_uuid
            ).single()
            assert row is not None
            assert row["name"] == "ok"
            try:
                s.run("MATCH (t:Task {uuid: $u}) DETACH DELETE t", u=marker_uuid)
            except Exception:
                pass

    def test_set_property_on_task_works(self, neo4j_driver):
        marker_uuid = "rbac-set-prop-test"
        with neo4j_driver.session() as s:
            s.run(
                "MERGE (t:Task {uuid: $u}) SET t.status = 'pending'",
                u=marker_uuid,
            )
            row = s.run(
                "MATCH (t:Task {uuid: $u}) RETURN t.status AS status", u=marker_uuid
            ).single()
            assert row is not None
            assert row["status"] == "pending"
            try:
                s.run("MATCH (t:Task {uuid: $u}) DETACH DELETE t", u=marker_uuid)
            except Exception:
                pass


class TestRbacGapDocumented:
    """B-7: non-Task labels are NOT protected. This test documents the gap."""

    def test_workspace_deletion_is_not_blocked(self, neo4j_driver):
        """``DENY DELETE`` only covers ``:Task``. ``:Workspace`` is unprotected.

        This is the B-7 gap. A future hardening pass should extend the rule
        to ``:Workspace`` (and ideally to ``:Session``, ``:Bug``,
        ``:ReviewNote``). For now, this test documents the gap by
        asserting that :Workspace CAN be deleted — that is, the rule
        does NOT block it.
        """
        marker_uuid = "rbac-gap-workspace-test"
        with neo4j_driver.session() as s:
            s.run("MERGE (w:Workspace {uuid: $u}) SET w.name = 'probe'", u=marker_uuid)
            s.run("MATCH (w:Workspace {uuid: $u}) DETACH DELETE w", u=marker_uuid)
            row = s.run(
                "MATCH (w:Workspace {uuid: $u}) RETURN count(w) AS n", u=marker_uuid
            ).single()
            assert row is not None
            assert row["n"] == 0, (
                ":Workspace should be deletable (B-7 gap) — DELETE was not blocked"
            )


def test_suite_is_skipped_by_default():
    """The whole module is opt-in: when ``RBAC_TEST_ALLOW`` is unset, tests skip.

    This is a sanity check: it documents and pins the opt-in behaviour so a
    future change cannot accidentally run these tests against production
    data.
    """
    if _is_skipped_by_default():
        return
    assert _neo4j_alive(), (
        "RBAC_TEST_ALLOW=yes but Neo4j is not reachable; either unset the "
        "env var or start the test container"
    )
