"""Unit tests for SkillRegistry.

Phase 1d: migrated to NEW `BaseSkill` from `proxy.skills.skill_definition`.
The OLD `BaseSkill` was removed from `proxy.skills.skill_registry` in 1d;
the `to_skill_definition` shim is gone; `SkillRegistrationError` lives in
`proxy.skills.skill_definition`.
"""
import os
import threading
import unittest
from typing import Optional
from unittest.mock import MagicMock

import pytest

from proxy.skills.skill_definition import (
    BaseSkill,
    SkillRegistrationError,
    SkillRequestContext,
)
from proxy.skills.skill_registry import (
    SkillContext,  # OLD context — kept for downstream consumers
    SkillMeta,
    SkillRegistry,
    reset_default_registry,
)


class DummySkill(BaseSkill):
    """Minimal test double for Skill protocol (NEW BaseSkill path).

    Phase 1d: extends NEW `BaseSkill`. Only `name` and `trigger` are
    required by the NEW BaseSkill validator; the OLD-specific methods
    (get_inheritance_state, get_activation_rewrite, get_tool_filter) are
    dropped because they are not part of the NEW BaseSkill interface.
    The approval-flow methods (has_pending_approval, is_approval_message,
    execute_approval, clear_pending_proposal) were removed in Wave 4.1
    as dead code; they were no-ops kept for back-compat with tests that
    no longer exist.
    """

    def __init__(
        self,
        name: str,
        trigger: Optional[str] = None,
        aliases=None,
    ):
        # NEW BaseSkill's __setattr__ allows setting name/trigger only
        # before _definition is cached. Use object.__setattr__ to bypass
        # the override (the test only constructs skills; to_skill_definition
        # is not invoked here).
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "trigger", trigger or f"^^{name}")
        object.__setattr__(self, "aliases", tuple(aliases) if aliases else ())

    # NEW BaseSkill already provides defaults for these methods, but
    # tests rely on `register()` (OLD path) calling `skill.name`,
    # `skill.trigger`, `skill.aliases`. The class attrs above cover that.

    def get_meta(self) -> SkillMeta:
        return SkillMeta(
            name=self.name,
            trigger=self.trigger,
            aliases=self.aliases,
        )


class TestSkillRegistry(unittest.TestCase):
    """Tests for SkillRegistry class."""

    def setUp(self) -> None:
        """Reset the default registry before each test.

        Phase 1b: replace the 1a-era reset_singleton() + CHORI_SKILL_REGISTRY_RESET=1
        env-var pattern with the unconditional reset_default_registry().

        ``SkillRegistry`` is ``@dataclass(frozen=True)`` so direct attribute
        assignment (e.g. ``registry._skills = ...``) raises
        ``FrozenInstanceError``. Use the ``clear_all()`` method, which already
        uses ``object.__setattr__`` internally to bypass the frozen guard.
        """
        reset_default_registry()
        registry = SkillRegistry()
        registry.clear_all()

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clear the singleton instance again
        reset_default_registry()

    def test_singleton_isolation(self) -> None:
        """Verify SkillRegistry is a singleton: state persists within same instance."""
        registry = SkillRegistry()

        # Multiple calls return the same instance
        registry2 = SkillRegistry()
        self.assertIs(registry, registry2)

        # State persists across calls. Phase 1a: ``_skills`` is a read-only
        # ``MappingProxyType``; use the public ``register()`` API instead of
        # writing directly to the internal dict. The skill name is not in
        # ``TRUSTED_BUILTIN_SKILLS`` so the trust gate does not fire.
        registry.register(DummySkill("test_persist"))
        registry3 = SkillRegistry()
        self.assertIsNotNone(registry3.get("test_persist"))

    def test_register_skill(self) -> None:
        """Register a skill instance, retrieve it by name, verify metadata."""
        registry = SkillRegistry()
        skill = DummySkill("test_skill", trigger="^^test", aliases=["^^alias"])

        registry.register(skill)

        # Retrieve by name
        retrieved = registry.get("test_skill")
        self.assertIs(retrieved, skill)

        # Verify metadata (no auto-registered skills)
        meta_list = registry.list_skills()
        self.assertGreaterEqual(len(meta_list), 1)
        meta = next(m for m in meta_list if m.name == "test_skill")
        self.assertEqual(meta.trigger, "^^test")
        # Phase 1b: meta is a raw skill (BaseSkill subclass), which stores
        # aliases as a tuple (NEW BaseSkill contract). Normalize to list for comparison.
        self.assertEqual(list(meta.aliases), ["^^alias"])

    def test_register_duplicate_raises(self) -> None:
        """Registering a skill with an existing name raises SkillRegistrationError.

        Phase 1a: duplicate-name rejection now raises
        ``SkillRegistrationError`` (a ``PermissionError`` subclass), not
        ``ValueError``. The message still starts with "Duplicate skill name".
        """
        registry = SkillRegistry()
        skill1 = DummySkill("duplicate_name")
        skill2 = DummySkill("duplicate_name")

        registry.register(skill1)

        with self.assertRaises(SkillRegistrationError) as ctx:
            registry.register(skill2)
        self.assertIn("Duplicate skill name", str(ctx.exception))

    def test_get_by_trigger(self) -> None:
        """Register a skill with trigger, call get_by_trigger, verify returns skill."""
        registry = SkillRegistry()
        skill = DummySkill("test_ws_trigger", trigger="!ws")

        registry.register(skill)

        retrieved = registry.get_by_trigger("!ws")
        self.assertIs(retrieved, skill)

    def test_get_by_trigger_alias(self) -> None:
        """Register a skill with alias, call get_by_trigger, verify returns skill."""
        registry = SkillRegistry()
        skill = DummySkill("test_alias", aliases=["!test", "!t"])
        registry.register(skill)

        retrieved = registry.get_by_trigger("!test")
        self.assertIs(retrieved, skill)

        retrieved2 = registry.get_by_trigger("!t")
        self.assertIs(retrieved2, skill)

    def test_get_by_trigger_no_match(self) -> None:
        """Unregistered trigger returns None."""
        registry = SkillRegistry()
        skill = DummySkill("some_skill", trigger="^^existing")

        registry.register(skill)

        result = registry.get_by_trigger("^^nonexistent")
        self.assertIsNone(result)

    def test_get_by_trigger_alias_no_match(self) -> None:
        """Unregistered alias returns None."""
        registry = SkillRegistry()
        skill = DummySkill("some_skill", aliases=["^^real"])

        registry.register(skill)

        result = registry.get_by_trigger("^^notregistered")
        self.assertIsNone(result)

    def test_list_skills(self) -> None:
        """After registering N skills, list_skills returns all N."""
        registry = SkillRegistry()

        skills = [
            DummySkill("skill1", trigger="^^s1"),
            DummySkill("skill2", trigger="^^s2"),
            DummySkill("skill3", trigger="^^s3"),
        ]

        for skill in skills:
            registry.register(skill)

        meta_list = registry.list_skills()
        self.assertEqual(len(meta_list), 3)
        names = {meta.name for meta in meta_list}
        self.assertEqual(names, {"skill1", "skill2", "skill3"})

    def test_concurrent_skill_registration(self) -> None:
        """Verify thread-safe skill registration with concurrent activations."""
        from proxy.skills.skill_registry import SkillRegistry

        registry = SkillRegistry()
        errors = []
        lock = threading.Lock()

        class CountedSkill(BaseSkill):
            def __init__(self, name):
                object.__setattr__(self, "name", name)
                object.__setattr__(self, "trigger", f"^^cnt_{name}")
                object.__setattr__(self, "aliases", ())

        def register_and_get(name):
            try:
                skill = CountedSkill(name)
                registry.register(skill)
                result = registry.get(name)
                if result is None:
                    errors.append(f"Failed to retrieve {name}")
            except Exception as e:
                with lock:
                    errors.append(f"Exception for {name}: {e}")

        threads = [
            threading.Thread(target=register_and_get, args=(f"concurrent_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent registration errors: {errors}")
        # Verify all registered skills are retrievable
        for i in range(5):
            name = f"concurrent_{i}"
            self.assertIsNotNone(registry.get(name), f"Skill {name} not found")

    def test_list_skills_empty(self) -> None:
        """Registry should start empty (no auto-registration)."""
        registry = SkillRegistry()

        meta_list = registry.list_skills()
        self.assertEqual(len(meta_list), 0)


class TestBaseSkillDefaults(unittest.TestCase):
    """Verify NEW BaseSkill default implementations don't crash new subclasses."""

    def test_get_system_prompt_default_returns_empty_string(self) -> None:
        """A minimal NEW BaseSkill subclass returns '' from get_system_prompt()."""

        class MinimalSkill(BaseSkill):
            name: str = "minimal"
            trigger: str = "^^minimal"

        skill = MinimalSkill()
        ctx = MagicMock()

        self.assertEqual(skill.get_system_prompt(ctx), "")

    def test_get_tool_filter_default_returns_none(self) -> None:
        """A minimal NEW BaseSkill subclass returns ``None`` from get_tool_filter().

        Wave 1 (spec §6) three-state policy: ``None`` is the
        "unrestricted" sentinel. The BaseSkill default semantic is
        "unrestricted unless overridden". An empty ``frozenset()``
        from a subclass is now the explicit "deny all" sentinel
        (fail-closed per spec §4.4).
        """

        class MinimalSkill(BaseSkill):
            name: str = "minimal"
            trigger: str = "^^minimal"

        skill = MinimalSkill()
        ctx = MagicMock()

        self.assertIsNone(skill.get_tool_filter(ctx))


class TestOldBaseSkillRemoved(unittest.TestCase):
    """Phase 1d: verify OLD `BaseSkill` and `to_skill_definition` shim are gone."""

    def test_old_base_skill_is_not_importable(self) -> None:
        """The OLD `BaseSkill` was removed in Phase 1d.

        The canonical `BaseSkill` is now in `proxy.skills.skill_definition`.
        Importing the OLD name from `proxy.skills.skill_registry` MUST raise
        `ImportError`. This pins the migration boundary so a future
        regression that re-adds the OLD class is caught immediately.
        """
        with self.assertRaises(ImportError):
            from proxy.skills.skill_registry import BaseSkill  # noqa: F401

    def test_skill_registry_does_not_have_to_skill_definition_shim(self) -> None:
        """The 1b-era `to_skill_definition` shim was removed in Phase 1d.

        Pinning test: importing any name from `proxy.skills.skill_registry`
        must NOT bring back the OLD shim. The shim lived on the OLD
        `BaseSkill`; since the class is gone, the method is gone too.
        """
        # Importing the module must not define `BaseSkill` (which carried the shim).
        import proxy.skills.skill_registry as sr_module
        self.assertFalse(hasattr(sr_module, "BaseSkill"))


def test_get_nonexistent_skill_returns_none():
    """SkillRegistry.get() returns None for unknown skill names."""
    from proxy.skills.skill_registry import SkillRegistry
    registry = SkillRegistry()
    assert registry.get("nonexistent_skill_xyz") is None


if __name__ == "__main__":
    unittest.main()