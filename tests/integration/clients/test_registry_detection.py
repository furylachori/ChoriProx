"""Registry detection integration tests.

Verify ClientRegistry.detect() works with real filesystem directories.
"""

import pytest
import os

from clients.registry import ClientRegistry
from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter
from clients.interface import DetectionError, RegistryError


class TestRegistryDetection:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global singleton after every test to prevent leakage."""
        yield
        ClientRegistry.reset()

    @pytest.fixture
    def isolated_registry(self):
        """Return a fresh registry instance that bypasses the singleton."""
        return ClientRegistry._for_testing()

    def test_detect_finds_kilocode_config_dir(self, tmp_path, isolated_registry):
        # Setup: create .kilo directory
        kilo_dir = tmp_path / ".kilo"
        kilo_dir.mkdir()

        isolated_registry.register(KiloCodeAdapter())
        isolated_registry.register(OpenCodeAdapter())

        adapter = isolated_registry.detect(str(tmp_path))
        assert adapter.name == "kilocode"
        assert isolated_registry._active == "kilocode"

    def test_detect_finds_opencode_config_dir(self, tmp_path, isolated_registry):
        # Setup: create .opencode directory (no .kilo)
        opencode_dir = tmp_path / ".opencode"
        opencode_dir.mkdir()

        isolated_registry.register(KiloCodeAdapter())
        isolated_registry.register(OpenCodeAdapter())

        adapter = isolated_registry.detect(str(tmp_path))
        assert adapter.name == "opencode"
        assert isolated_registry._active == "opencode"

    def test_detect_prefers_first_match_order(self, tmp_path, isolated_registry):
        # Setup: create both .kilo and .opencode
        (tmp_path / ".kilo").mkdir()
        (tmp_path / ".opencode").mkdir()

        # Register opencode first
        isolated_registry.register(OpenCodeAdapter())
        isolated_registry.register(KiloCodeAdapter())

        adapter = isolated_registry.detect(str(tmp_path))
        # First registered wins
        assert adapter.name == "opencode"

    def test_detect_raises_when_no_config_dir(self, tmp_path, isolated_registry):
        isolated_registry.register(KiloCodeAdapter())
        isolated_registry.register(OpenCodeAdapter())

        with pytest.raises(DetectionError) as exc_info:
            isolated_registry.detect(str(tmp_path))

        assert "No client config directory found" in str(exc_info.value)

    def test_detect_raises_on_empty_project_root(self, isolated_registry):
        isolated_registry.register(KiloCodeAdapter())

        with pytest.raises(DetectionError) as exc_info:
            isolated_registry.detect("")

        assert "project_root must be a non-empty string" in str(exc_info.value)

    def test_detect_sets_active_client(self, tmp_path, isolated_registry):
        (tmp_path / ".kilo").mkdir()
        isolated_registry.register(KiloCodeAdapter())

        isolated_registry.detect(str(tmp_path))
        active = isolated_registry.active()
        assert active.name == "kilocode"

    def test_active_raises_before_detect(self, isolated_registry):
        isolated_registry.register(KiloCodeAdapter())

        with pytest.raises(RegistryError) as exc_info:
            isolated_registry.active()

        assert "No active client set" in str(exc_info.value)

    def test_detect_with_real_nested_path(self, tmp_path, isolated_registry):
        nested = tmp_path / "projects" / "foo"
        nested.mkdir(parents=True)
        (nested / ".opencode").mkdir()

        isolated_registry.register(OpenCodeAdapter())
        adapter = isolated_registry.detect(str(nested))
        assert adapter.name == "opencode"

    def test_detect_is_case_sensitive(self, tmp_path, isolated_registry):
        # Setup: .Kilo (uppercase K) should NOT match .kilo on case-sensitive filesystems
        (tmp_path / ".Kilo").mkdir()
        isolated_registry.register(KiloCodeAdapter())

        # macOS default APFS is case-insensitive, so .Kilo matches .kilo there.
        # Only assert DetectionError on case-sensitive filesystems.
        if os.path.isdir(os.path.join(str(tmp_path), ".kilo")):
            pytest.skip("Filesystem is case-insensitive; .Kilo matches .kilo")

        with pytest.raises(DetectionError):
            isolated_registry.detect(str(tmp_path))
