"""Fixtures for src/clients/ unit tests."""

import json
import os
import pytest
from unittest.mock import MagicMock

from clients.interface import (
    ConfigLoader,
    CapabilitySet,
)
from clients.orchestrator_config import HookCapabilities
from clients.registry import ClientRegistry
from clients.model_registry import ModelRegistry, ToolParserRegistry
from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter


@pytest.fixture(autouse=True)
def reset_client_registry():
    """Reset the ClientRegistry singleton before and after every test."""
    ClientRegistry.reset()
    yield
    ClientRegistry.reset()


@pytest.fixture
def registry():
    """Return a fresh isolated ClientRegistry instance (not the singleton)."""
    return ClientRegistry._for_testing()


@pytest.fixture
def mock_loader():
    """Return a MagicMock configured as a ConfigLoader."""
    loader = MagicMock(spec=ConfigLoader)
    loader.path_join = os.path.join
    return loader


@pytest.fixture
def kilocode_adapter():
    """Return a KiloCodeAdapter instance."""
    return KiloCodeAdapter()


@pytest.fixture
def opencode_adapter():
    """Return an OpenCodeAdapter instance."""
    return OpenCodeAdapter()


@pytest.fixture
def default_capabilities():
    """Return a default CapabilitySet with properly constructed hooks."""
    return CapabilitySet(hooks=HookCapabilities())


@pytest.fixture
def kilocode_project(tmp_path):
    """Return a temporary project root containing a .kilo agent config."""
    project = tmp_path / "kilocode_proj"
    agent_dir = project / ".kilo" / "agent"
    agent_dir.mkdir(parents=True)
    config = {"name": "test-agent", "model": "gpt-4", "temperature": 0.5}
    (agent_dir / "agent.json").write_text(json.dumps(config), encoding="utf-8")
    return str(project)


@pytest.fixture
def opencode_project(tmp_path):
    """Return a temporary project root containing a .opencode markdown config."""
    project = tmp_path / "opencode_proj"
    agent_dir = project / ".opencode"
    agent_dir.mkdir(parents=True)
    frontmatter = (
        "---\n"
        "name: test-agent\n"
        "model: claude-3\n"
        "temperature: 0.7\n"
        "---\n\n"
        "# Agent config\n"
    )
    (agent_dir / "agent.md").write_text(frontmatter, encoding="utf-8")
    return str(project)


@pytest.fixture
def model_registry(tmp_path):
    """Return a ModelRegistry backed by an empty temp directory."""
    return ModelRegistry(base_dir=str(tmp_path))


@pytest.fixture
def parser_registry():
    """Return a fresh ToolParserRegistry."""
    return ToolParserRegistry()


@pytest.fixture
def registered_kilocode(registry, kilocode_adapter):
    """Register KiloCodeAdapter in the isolated registry."""
    registry.register(kilocode_adapter)
    return registry


@pytest.fixture
def registered_opencode(registry, opencode_adapter):
    """Register OpenCodeAdapter in the isolated registry."""
    registry.register(opencode_adapter)
    return registry
