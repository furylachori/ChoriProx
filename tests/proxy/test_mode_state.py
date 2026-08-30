"""Phase 1 unit tests for ModeState.

Tests the ModeState dataclass for TTL expiry, persistence,
skill_state round-tripping, and equality semantics.
"""
import time
import unittest
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional
from unittest.mock import patch

from proxy.skills.skill_definition import ModeState
from proxy.mode import Mode


class TestModeStateTTL(unittest.TestCase):
    """Tests for ModeState TTL (time-to-live) behavior."""

    def test_mode_state_ttl_expires(self):
        """Create ModeState with TTL=10s, advance virtual clock past TTL, verify is_expired() returns True."""
        fake_now = [1_000_000.0]
        with patch("time.time", side_effect=lambda: fake_now[0]):
            state = ModeState(
                name="ticket",
                activated_at=time.time(),
                ttl_seconds=10,
                allowed_tool_names=frozenset(),
                skill_state={},
            )
            fake_now[0] += 11
            self.assertTrue(state.is_expired())

    def test_mode_state_ttl_not_yet_expired(self):
        """Create ModeState with TTL=300s, immediately check is_expired(), should be False."""
        state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            skill_state={},
        )
        self.assertFalse(state.is_expired())

    def test_mode_state_persistent_never_expires(self):
        """Create ModeState with persistent=True (no TTL), is_expired() always False."""
        # ttl_seconds=0 means persistent (never expires)
        state = ModeState(
            name="workspace",
            activated_at=time.time(),
            ttl_seconds=0,
            allowed_tool_names=frozenset(),
            skill_state={},
        )
        self.assertFalse(state.is_expired())

    def test_mode_state_refresh_clears_expiry(self):
        """Fresh ModeState with future TTL is not expired (replaces refresh() check).

        1a removed refresh() (R1-HIGH-4 YAGNI). Fresh ModeState is
        not expired; this pins the same invariant without refresh().
        """
        state = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            skill_state={},
        )
        self.assertFalse(state.is_expired())


class TestModeStateSkillState(unittest.TestCase):
    """Tests for ModeState skill_state persistence."""

    def test_mode_state_skill_state_persists(self):
        """Create ModeState with skill_state={"key": "val"}, verify skill_state is round-tripped."""
        initial_state = {"key": "val", "nested": {"a": 1, "b": 2}}
        state = ModeState(
            name="skill",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            skill_state=initial_state,
        )
        self.assertEqual(state.skill_state, initial_state)
        # 1a: skill_state is a MappingProxyType (read-only view of the
        # original dict). The values are still accessible.
        self.assertIsInstance(state.skill_state, MappingProxyType)


class TestModeStateEquality(unittest.TestCase):
    """Tests for ModeState equality semantics."""

    def test_mode_state_equality(self):
        """Two ModeState instances with same name/name_prefix/TTL are equal."""
        now = time.time()
        state1 = ModeState(
            name="ticket",
            activated_at=now,
            ttl_seconds=300,
            allowed_tool_names=frozenset(["tool1"]),
            skill_state={"key": "val"},
        )
        state2 = ModeState(
            name="ticket",
            activated_at=now,
            ttl_seconds=300,
            allowed_tool_names=frozenset(["tool1"]),
            skill_state={"key": "val"},
        )
        self.assertEqual(state1, state2)

    def test_mode_state_inequality(self):
        """Two ModeState instances with different names are not equal."""
        now = time.time()
        state1 = ModeState(
            name="ticket",
            activated_at=now,
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            skill_state={},
        )
        state2 = ModeState(
            name="workspace",
            activated_at=now,
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
            skill_state={},
        )
        self.assertNotEqual(state1, state2)


class TestModeEnum(unittest.TestCase):
    """Tests for Mode enum values."""

    def test_mode_enum_values(self):
        """Verify Mode.SKILL exists."""
        self.assertIsNotNone(Mode.SKILL)


if __name__ == "__main__":
    unittest.main()
