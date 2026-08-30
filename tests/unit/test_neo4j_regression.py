"""Regression tests for Neo4j driver singleton pattern.

These tests prevent regression of Bug #8: QueryExecutor creating multiple
driver instances instead of reusing a shared singleton.
"""
import ast
import os
from unittest.mock import Mock, patch

import pytest

from memory.query_executor import (
    QueryExecutor,
    close_shared_driver,
    reset_shared_driver,
)


class TestNeo4jDriverRegression:
    """Regression tests to prevent driver singleton bugs."""

    def setup_method(self):
        """Reset singleton state before each test."""
        reset_shared_driver()

    def teardown_method(self):
        """Reset singleton state after each test."""
        reset_shared_driver()

    def test_no_direct_GraphDatabase_driver_calls(self):
        """AST scan: no direct GraphDatabase.driver() calls outside query_executor.py."""
        src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        src_dir = os.path.abspath(src_dir)

        violations = []

        for root, dirs, files in os.walk(src_dir):
            for filename in files:
                if not filename.endswith(".py"):
                    continue

                filepath = os.path.join(root, filename)
                relpath = os.path.relpath(filepath, src_dir)

                if relpath == os.path.join("memory", "query_executor.py"):
                    continue

                try:
                    with open(filepath, "r") as f:
                        code = f.read()

                    tree = ast.parse(code)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute):
                                if node.func.attr == "driver":
                                    if isinstance(node.func.value, ast.Name):
                                        if node.func.value.id == "GraphDatabase":
                                            violations.append(
                                                f"{relpath}:{node.lineno}"
                                            )
                except Exception:
                    pass

        assert not violations, (
            f"Direct GraphDatabase.driver() calls found outside query_executor.py:\n"
            + "\n".join(violations)
        )

    def test_single_neo4j_driver(self):
        """Verify that only one Neo4j driver is created across multiple instantiations."""
        with patch("memory.query_executor.GraphDatabase.driver") as mock_driver:
            mock_driver.return_value = Mock()

            executor1 = QueryExecutor("bolt://localhost:7687", ("neo4j", "password"))
            executor2 = QueryExecutor("bolt://localhost:7687", ("neo4j", "password"))

            assert mock_driver.call_count == 1, (
                f"Expected 1 driver creation, got {mock_driver.call_count}"
            )

    def test_close_is_idempotent_post_refactor(self):
        """close_shared_driver() is idempotent after singleton refactor."""
        with patch("memory.query_executor.GraphDatabase") as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver

            executor = QueryExecutor("bolt://localhost:7687", ("neo4j", "test"))

            close_shared_driver()
            close_shared_driver()
            close_shared_driver()

            assert mock_driver.close.call_count == 1
