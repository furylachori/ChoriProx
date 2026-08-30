"""Tests for Neo4j driver singleton invariant (US-045).

Verifies that multiple QueryExecutor instances with the same bolt URL
share a single underlying Neo4j driver, preventing connection pool exhaustion.

These tests require a real Neo4j instance (or at minimum, a real neo4j driver).
Mark with @pytest.mark.integration_neo4j to run only when Neo4j is available.
"""

import os

import pytest

from memory.query_executor import (
    QueryExecutor,
    get_shared_driver,
    reset_shared_driver,
    close_shared_driver,
)

_BOLT_URL = os.environ.get("PROXY_NEO4J_BOLT_URL", "bolt://localhost:7687")
_AUTH = tuple(os.environ.get("NEO4J_AUTH", "neo4j:password").split(":", 1))


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset driver singleton before and after each test."""
    close_shared_driver()
    reset_shared_driver()
    yield
    close_shared_driver()
    reset_shared_driver()


@pytest.mark.integration_neo4j
def test_neo4j_driver_singleton_same_url():
    """US-045: Multiple QueryExecutor instances with the same bolt URL MUST share a single driver."""
    bolt_url = _BOLT_URL
    auth = _AUTH

    exec1 = QueryExecutor(bolt_url, auth)
    exec2 = QueryExecutor(bolt_url, auth)

    assert exec1._driver is exec2._driver, (
        "Two QueryExecutor instances with the same bolt URL must share the same driver object. "
        "This prevents connection pool exhaustion."
    )


@pytest.mark.integration_neo4j
def test_neo4j_driver_different_urls_get_different_drivers():
    """Different bolt URLs MUST get different drivers."""
    auth = _AUTH

    driver_a = get_shared_driver("bolt://host-a:7687", auth)
    # Reset to allow a new driver with a different URL
    reset_shared_driver()
    driver_b = get_shared_driver("bolt://host-b:7687", auth)

    assert driver_a is not driver_b, (
        "Different bolt URLs must produce different driver instances."
    )


@pytest.mark.integration_neo4j
def test_neo4j_driver_singleton_same_auth_tuple():
    """Same URL + same auth tuple → same driver."""
    bolt_url = _BOLT_URL
    auth = _AUTH

    driver1 = get_shared_driver(bolt_url, auth)
    driver2 = get_shared_driver(bolt_url, auth)

    assert driver1 is driver2, (
        "get_shared_driver called twice with identical args must return the same object."
    )


@pytest.mark.integration_neo4j
def test_neo4j_driver_returns_none_for_missing_url():
    """get_shared_driver returns None when bolt_url is None."""
    driver = get_shared_driver(None, _AUTH)
    assert driver is None


@pytest.mark.integration_neo4j
def test_neo4j_driver_returns_none_for_missing_auth():
    """get_shared_driver returns None when auth is None."""
    driver = get_shared_driver(_BOLT_URL, None)
    assert driver is None


@pytest.mark.integration_neo4j
def test_query_executor_shares_driver_after_reset():
    """After reset, a new QueryExecutor gets a fresh driver (not the old one)."""
    bolt_url = _BOLT_URL
    auth = _AUTH

    exec1 = QueryExecutor(bolt_url, auth)
    old_driver = exec1._driver

    reset_shared_driver()

    exec2 = QueryExecutor(bolt_url, auth)
    # After reset, a NEW driver is created (different object)
    assert exec2._driver is not old_driver, (
        "After reset_shared_driver, a new driver must be created."
    )
    # But both exec2's driver should be the same as a fresh get_shared_driver call
    driver_check = get_shared_driver(bolt_url, auth)
    assert exec2._driver is driver_check
