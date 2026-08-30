"""Regression tests: the Neo4j test harness must NEVER wipe a shared database.

The harness used to gate a destructive ``MATCH (n) DETACH DELETE n`` behind the
I_AM_AN_IDIOT_ASK_A_HUMAN=yes env var. That flag was a foot-gun — agents and
humans set it without thinking and wiped production (hundreds of wipes recorded
in backups/). The flag has been REMOVED and the wipe is now impossible:

  1. NO-WIPE — _clear_all_data refuses UNCONDITIONALLY. There is no flag, no
     override, no judgement call. It never opens a session, never runs Cypher.
  2. PRODUCTION GUARD — fixtures skip/mock when the bolt URL targets :7687, so
     no CREATE/DROP DATABASE or wipe ever runs against prod.
  3. ISOLATION DEFAULT — tests default to the throwaway container on :7688; if
     it is down, Neo4j tests skip rather than fall back to production.

These tests pin that contract so a future "fix the skips" change cannot
re-introduce the data-loss bug (commit 2753fbe removed skip markers and caused
3 production data losses in one day).
"""
from __future__ import annotations

from unittest.mock import MagicMock
import importlib

import pytest


def test_clear_all_data_refuses_unconditionally(monkeypatch):
    """_clear_all_data never wipes — no session, no Cypher, regardless of target."""
    monkeypatch.setenv("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7688")
    import tests.integration.neo4j_harness as harness
    importlib.reload(harness)
    driver = MagicMock()
    harness._clear_all_data(driver, "neo4j")
    driver.session.assert_not_called()


def test_legacy_idiot_flag_is_a_noop(monkeypatch):
    """Setting the old I_AM_AN_IDIOT_ASK_A_HUMAN=yes changes NOTHING.

    The flag no longer exists in the harness. Even if an agent exports it and
    points at a non-prod DB, _clear_all_data must still refuse — proving the
    destructive path is gone, not just gated.
    """
    monkeypatch.setenv("I_AM_AN_IDIOT_ASK_A_HUMAN", "yes")
    monkeypatch.setenv("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7688")
    import tests.integration.neo4j_harness as harness
    importlib.reload(harness)
    assert not hasattr(harness, "_WIPE_REAL_DB_OK"), "the wipe flag must be gone"
    assert not hasattr(harness, "_backup_before_wipe"), "the wipe backup must be gone"
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value = session
    harness._clear_all_data(driver, "neo4j")
    driver.session.assert_not_called()
    cypher_calls = [c for c in session.run.call_args_list if c.args]
    assert not any("DETACH DELETE" in str(c.args[0]) for c in cypher_calls), \
        f"_clear_all_data must never run DETACH DELETE, got: {cypher_calls}"


def test_clear_all_data_refuses_prod(monkeypatch):
    """Production (:7687) is never wiped (subsumed by the unconditional refusal)."""
    monkeypatch.setenv("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")
    import tests.integration.neo4j_harness as harness
    importlib.reload(harness)
    driver = MagicMock()
    harness._clear_all_data(driver, "neo4j")
    driver.session.assert_not_called()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("bolt://localhost:7687", True),
        ("bolt://127.0.0.1:7687", True),
        ("bolt://localhost:7688", False),
        ("bolt://localhost:76870", False),  # must NOT false-positive on :76870
        ("bolt://prod-host", True),  # no port → bolt default 7687 → prod (fail-safe)
        ("bolt://[::1]:7687", True),
        ("bolt://[::1]:7688", False),
        ("neo4j://prod-host:7687", True),
        ("bolt://localhost:7687/", True),  # trailing slash
        ("bolt://localhost:7687/db", True),  # path
        ("bolt://localhost:7687?timeout=5", True),  # query string
        ("bolt://localhost:7688/", False),
        ("", True),  # empty → no port → fail-safe prod
    ],
)
def test_is_prod_bolt_url(url, expected):
    """_is_prod_bolt_url is the safety discriminator — pin its contract."""
    import tests.integration.neo4j_harness as harness
    assert harness._is_prod_bolt_url(url) is expected


class _FakeNode:
    nodeid = "tests/integration/test_fake.py::test_fake"
    name = "test_fake"


class _FakeRequest:
    node = _FakeNode()


def test_neo4j_test_db_skips_on_prod(monkeypatch):
    """neo4j_test_db refuses to run any DDL/migrations against prod (:7687).

    Pins no-wipe policy layer #2: with a prod bolt URL the fixture must skip
    BEFORE opening a driver — so CREATE/DROP DATABASE never touches prod.
    """
    monkeypatch.setenv("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")
    import tests.integration.neo4j_harness as harness
    importlib.reload(harness)
    monkeypatch.setattr(harness, "_neo4j_available", lambda: True)
    opened = {"driver": False}
    monkeypatch.setattr(
        "neo4j.GraphDatabase.driver",
        lambda *a, **k: opened.__setitem__("driver", True),
    )
    gen = harness.neo4j_test_db.__wrapped__(_FakeRequest())
    with pytest.raises(pytest.skip.Exception):
        next(gen)
    assert opened["driver"] is False, "no driver may be opened against prod"


def test_real_or_mock_neo4j_mocks_on_prod(monkeypatch):
    """real_or_mock_neo4j falls back to a MagicMock on prod (:7687), never a real driver."""
    monkeypatch.setenv("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")
    import tests.integration.neo4j_harness as harness
    importlib.reload(harness)
    monkeypatch.setattr(harness, "_neo4j_available", lambda: True)
    opened = {"driver": False}
    monkeypatch.setattr(
        "neo4j.GraphDatabase.driver",
        lambda *a, **k: opened.__setitem__("driver", True),
    )
    gen = harness.real_or_mock_neo4j.__wrapped__(_FakeRequest())
    executor = next(gen)
    assert isinstance(executor, MagicMock), "prod must yield a mock, not a real executor"
    assert opened["driver"] is False, "no driver may be opened against prod"
