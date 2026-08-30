"""Unit tests for subagent_request_builder._detect_return_path.

Revert-test contract: each new branch must fail when removed.
"""
import pytest

from tests.fixtures.subagent_request_builder import (
    _detect_return_path,
    make_subagent_request,
    _TRUNC_MARKER,
)


class TestDetectReturnPathFallbackTrimmed:
    """The §13.5 trim safety net can rescue what was previously the
    'fallback' raise path. The heuristic must identify this as 'fallback'
    (not 'trimmed-fits' or 'unknown') regardless of trim aggressiveness.

    These tests are REVERT-TESTS: each test fails if the corresponding
    detection clause is removed from _detect_return_path.
    """

    def _user(self, content):
        return {"role": "user", "content": content}

    def test_verbatim_first_user_is_fallback(self):
        """Single non-system message, no trim marker → fallback."""
        original = [self._user("hello world")]
        result = [self._user("hello world")]
        assert _detect_return_path(original, result) == "fallback"

    def test_trimmed_first_user_with_marker_is_fallback(self):
        """§13.5 trim safety net: single user message trimmed with marker → fallback.

        Realistic case: keep_chars >= 50 (production budget math makes this
        the floor), but if the trim is severe enough to leave < 50 chars of
        overlap, the prefix-startswith check catches it.
        """
        original = [self._user("X" * 89_000)]
        trimmed = "X" * 87_000 + _TRUNC_MARKER
        result = [self._user(trimmed)]
        assert _detect_return_path(original, result) == "fallback"

    def test_extreme_trim_short_prefix_marker_is_fallback(self):
        """Even keep_chars < 50 + marker is detected via prefix-startswith."""
        original = [self._user("X" * 89_000)]
        trimmed = "X" * 10 + _TRUNC_MARKER
        result = [self._user(trimmed)]
        assert _detect_return_path(original, result) == "fallback"

    def test_trimmed_with_marker_in_multi_unit_path_is_trimmed_fits(self):
        """Multiple non-system messages, one has marker → trimmed-fits (not fallback)."""
        original = [self._user("hello"), {"role": "assistant", "content": "B" * 200}]
        result = [
            self._user("hello"),
            {"role": "assistant", "content": "B" * 30 + _TRUNC_MARKER},
        ]
        assert _detect_return_path(original, result) == "trimmed-fits"