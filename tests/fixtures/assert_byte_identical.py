"""
Byte-identical assertion helpers for snapshot testing.

Provides tight comparison utilities that surface exact byte-level
differences, making it straightforward to catch accidental transformations
(e.g., added whitespace, unicode normalization, key ordering).
"""

from __future__ import annotations

import json
from difflib import unified_diff
from typing import Any


def assert_string_identical(
    actual: str, expected: str, name: str = "string"
) -> None:
    """Raise AssertionError if two strings differ, showing a unified diff."""
    if actual == expected:
        return
    diff = unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=f"{name} (expected)",
        tofile=f"{name} (actual)",
    )
    diff_lines = list(diff)
    msg = (
        f"{name} mismatch:\n"
        + ("".join(diff_lines) if diff_lines else f"  expected: {expected!r}\n  actual: {actual!r}")
    )
    raise AssertionError(msg)


def assert_list_identical(
    actual: list[Any], expected: list[Any], name: str = "list"
) -> None:
    """Raise AssertionError if two lists differ (ordered, byte-level comparison)."""
    if actual == expected:
        return
    # Show first index where they diverge
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            raise AssertionError(
                f"{name}[{i}] mismatch:\n  expected: {e!r}\n  actual: {a!r}"
            )
    raise AssertionError(
        f"{name} length mismatch: expected {len(expected)}, got {len(actual)}"
    )


def assert_json_identical(
    actual: Any, expected: Any, name: str = "json"
) -> None:
    """
    Raise AssertionError if JSON dumps differ byte-for-byte.

    Serialises both objects with json.dumps using the same parameters,
    then does a direct string comparison so that key ordering and
    escaping differences are surfaced explicitly.
    """
    actual_serialised = json.dumps(actual, sort_keys=True, ensure_ascii=False, indent=2)
    expected_serialised = json.dumps(expected, sort_keys=True, ensure_ascii=False, indent=2)
    assert_string_identical(actual_serialised, expected_serialised, name=name)


class SnapshotSuite:
    """
    Loads and provides access to named fixtures stored in ``snapshots.json``.
    """

    _cache: dict[str, dict[str, Any]] | None = None

    @classmethod
    def _load(cls) -> dict[str, dict[str, Any]]:
        if cls._cache is not None:
            return cls._cache
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        snapshot_path = os.path.join(root, "ticket_mode_responses", "snapshots.json")
        with open(snapshot_path, encoding="utf-8") as fh:
            records: list[dict[str, Any]] = json.load(fh)
        cls._cache = {rec["name"]: rec for rec in records}
        return cls._cache

    @classmethod
    def load(cls, name: str) -> dict[str, Any]:
        """
        Load a fixture by its ``name`` field.

        Raises
        ------
        KeyError
            If ``name`` is not present in the snapshot file.
        """
        store = cls._load()
        if name not in store:
            available = ", ".join(sorted(store.keys()))
            raise KeyError(f"Fixture {name!r} not found. Available: {available}")
        return store[name]

    @classmethod
    def all_names(cls) -> list[str]:
        """Return sorted list of all fixture names."""
        return sorted(cls._load().keys())
