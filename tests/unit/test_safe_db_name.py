"""Invariant tests for ``safe_db_name`` (B-9: function behind 3 data losses).

The ``safe_db_name`` helper converts pytest node IDs into valid, unique Neo4j
database names.  A bug in uniqueness — sibling tests colliding on the same
sanitized prefix — caused CREATE OR REPLACE to destroy concurrent test
databases, taking production workspaces/tasks with them.  These tests pin the
invariants that prevent that class of regression:

1. Determinism — same input always yields the same output.
2. Uniqueness across siblings — N distinct inputs yield N distinct names.
3. Uniqueness across inputs that sanitize to the same canonical form
   (``test_aaa`` vs ``test-aaa``) — preserved by hashing the RAW input.
4. Length cap — every output fits the Neo4j ``<= 63`` database-name limit.
5. Valid format — every output matches the Neo4j naming rules
   (``[a-zA-Z][a-zA-Z0-9.-]{2,62}``).

Note on idempotency: ``f(f(x)) == f(x)`` is NOT a property of this function
and CANNOT be one without sacrificing uniqueness across inputs that sanitize
to the same canonical form (e.g. ``test_aaa`` and ``test-aaa`` would collide
if the digest were computed from the sanitized form).  The function is
intentionally non-idempotent; the closest property is determinism.
"""
from __future__ import annotations

import re
import unittest

from tests.integration.neo4j_harness_utils import safe_db_name


class TestSafeDbNameInvariants(unittest.TestCase):
    """Pin the invariants that prevent the safe-db-name data-loss class."""

    NEO4J_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9.-]{2,62}$")

    def test_uniqueness_within_class(self):
        """Seven sibling tests produce seven distinct DB names."""
        siblings = [
            safe_db_name(f"tests/integration/test_x.py::TestClass::test_{i}")
            for i in range(7)
        ]
        self.assertEqual(
            len(set(siblings)),
            7,
            f"sibling tests must produce distinct DB names; got duplicates in {siblings!r}",
        )

    def test_length_under_neo4j_limit(self):
        """Output length must be <= 63 for any input length."""
        cases = [
            "x",
            "tests/" + "a" * 500 + "/test_x",
            "tests/" + "z" * 2000,
            "/" * 1000,
            "tests/integration/test_very_long_path_name_with_many_underscores_and_dots.py::TestClassWithLongName::test_method_with_a_long_name",
            "",
        ]
        for case in cases:
            name = safe_db_name(case)
            self.assertLessEqual(
                len(name),
                63,
                f"name too long ({len(name)} chars) for {case[:50]!r}: got {name!r}",
            )

    def test_empty_input_produces_valid_name(self):
        """Empty input still yields a name that matches Neo4j naming rules."""
        name = safe_db_name("")
        self.assertTrue(
            self.NEO4J_NAME_RE.match(name),
            f"invalid name from empty input: {name!r}",
        )
        self.assertLessEqual(len(name), 63)

    def test_unicode_and_special_chars_handled(self):
        """Unicode, spaces, brackets, quotes, and injection-style inputs are sanitized."""
        cases = [
            "tests/тест_файл.py",
            "tests/file with spaces.py::test_method",
            "tests/file[brackets].py",
            "tests/file'quote.py",
            "tests/../../etc/passwd",
            "tests/file;DROP TABLE tasks;--",
            "tests/file`backtick`.py",
            "tests/file\"doublequote\".py",
        ]
        for case in cases:
            name = safe_db_name(case)
            self.assertTrue(
                self.NEO4J_NAME_RE.match(name),
                f"invalid name for {case!r}: {name!r}",
            )

    def test_starts_with_letter(self):
        """First character must be a letter (Neo4j database-name requirement)."""
        cases = [
            "",
            "1leading_digit",
            "_leading_underscore",
            "-leading_dash",
            ".leading_dot",
            "9abc",
            "0",
            "---",
            "...",
        ]
        for case in cases:
            name = safe_db_name(case)
            self.assertTrue(
                name and name[0].isalpha(),
                f"name does not start with a letter for {case!r}: {name!r}",
            )

    def test_uniqueness_with_same_prefix_diff_suffix(self):
        """Inputs that truncate to the same 53-char prefix still produce distinct names."""
        base = "a" * 60 + "_test_"
        cases = [base + f"variant_{i}" for i in range(5)]
        names = [safe_db_name(case) for case in cases]
        self.assertEqual(
            len(set(names)),
            len(cases),
            f"same-prefix inputs must produce distinct DB names; got {names!r}",
        )

    def test_deterministic(self):
        """Same input always yields the same output (no hidden randomness/state)."""
        sample = "tests/integration/test_neo4j_harness_smoke.py::TestX::test_y"
        first = safe_db_name(sample)
        for _ in range(5):
            self.assertEqual(safe_db_name(sample), first)

    def test_format_is_lowercase_alnum_dots_dashes(self):
        """Every output character belongs to the Neo4j-allowed charset."""
        cases = [
            "PlainName",
            "MIXED_case_NAME",
            "tests/UPPER/lower.PY",
            "with spaces",
            "with_underscores-and-dashes.and.dots",
            "123_numeric_start",
        ]
        for case in cases:
            name = safe_db_name(case)
            self.assertTrue(
                self.NEO4J_NAME_RE.match(name),
                f"name {name!r} (from {case!r}) does not match Neo4j naming rules",
            )

    def test_distinguishes_inputs_that_sanitize_to_same_canonical_form(self):
        """``test_aaa`` and ``test-aaa`` must NOT collide.

        The digest is computed from the raw input precisely so that inputs
        which sanitize to the same canonical form still produce distinct
        database names.  This test pins that property — it is the reason
        idempotency (``f(f(x)) == f(x)``) cannot be added without sacrificing
        uniqueness.
        """
        a = safe_db_name("test_aaa")
        b = safe_db_name("test-aaa")
        self.assertNotEqual(
            a,
            b,
            f"sanitization-equivalent inputs must produce distinct DB names; "
            f"both produced {a!r}",
        )


if __name__ == "__main__":
    unittest.main()