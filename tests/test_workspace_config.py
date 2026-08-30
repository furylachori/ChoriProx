"""Tests for workspace configuration fixes (Batch C - C3, CFG-001/002/005/006/008).

Note: proxy_workspace was removed in the workspace cleanup (US-099). The
workspace skill now handles all workspace operations via handle_direct
in a single local state mutation path.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestWorkspaceNamePattern:
    """CFG-001: Workspace names should reject underscores."""

    def test_workspace_name_rejects_underscore(self):
        """'my_workspace' should be rejected."""
        from skills.workspace import _is_valid_workspace_name
        assert _is_valid_workspace_name("my_workspace") is False

    def test_workspace_name_accepts_hyphen(self):
        """'my-workspace' should be accepted."""
        from skills.workspace import _is_valid_workspace_name
        assert _is_valid_workspace_name("my-workspace") is True

    def test_workspace_name_rejects_reserved(self):
        """Reserved names should be rejected."""
        from skills.workspace import _is_valid_workspace_name
        for name in ["leave", "default", "recent", "help", "list", "status"]:
            assert _is_valid_workspace_name(name) is False, f"{name} should be reserved"


class TestLowercaseNormalization:
    """CFG-005: Workspace name is lowercased before validation."""

    def test_uppercase_workspace_accepted(self):
        """'My-Workspace' should be accepted and normalized."""
        from skills.workspace import _is_valid_workspace_name
        # Should accept because .lower() is applied before validation
        assert _is_valid_workspace_name("My-Workspace") is True


class TestModeStateTypeAnnotation:
    """CFG-008: ModeState.allowed_tool_names is frozenset[str]."""

    def test_mode_state_has_frozenset_type(self):
        """ModeState.allowed_tool_names should be typed as frozenset[str]."""
        from proxy.skills.skill_definition import ModeState
        import inspect

        sig = inspect.signature(ModeState)
        param = sig.parameters.get('allowed_tool_names')
        assert param is not None
        annotation = str(param.annotation)
        assert 'frozenset' in annotation
