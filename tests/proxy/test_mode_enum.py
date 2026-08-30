"""Phase 5 unit test for Mode enum.

Verifies that Mode.SKILL and Mode.SKILL_SUBAGENT exist and are distinct.
"""
import unittest

from proxy.mode import Mode


class TestModeEnumValues(unittest.TestCase):
    """Tests for Mode enum values."""

    def test_mode_skill_exists(self):
        """Verify Mode.SKILL exists."""
        self.assertIsNotNone(Mode.SKILL)

    def test_mode_skill_subagent_exists(self):
        """Verify Mode.SKILL_SUBAGENT exists."""
        self.assertIsNotNone(Mode.SKILL_SUBAGENT)

    def test_mode_skill_is_not_subagent(self):
        """Mode.SKILL and Mode.SKILL_SUBAGENT are distinct."""
        self.assertIsNot(Mode.SKILL, Mode.SKILL_SUBAGENT)


if __name__ == "__main__":
    unittest.main()
