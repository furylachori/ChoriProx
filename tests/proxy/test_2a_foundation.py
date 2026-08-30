"""Tests for Phase 2A foundation: ModeState re-export, boot assertion.

These tests pin the contracts that 2A introduced:
1. proxy.state.ModeState is identical to proxy.agent.types.ModeState (re-export)
2. assert_session_secret_configured raises in production without secret
3. skill_definition.py imports ModeState from proxy.state (the new chain)

# Phase 2B: skill dispatcher is always on (no flag).
"""
import os

import pytest


def test_mode_state_identity_across_three_import_paths():
    """proxy.state.ModeState is proxy.agent.types.ModeState is proxy.skills.skill_definition.ModeState.

    The re-export in proxy.state must be the SAME class object (not a copy).
    """
    from proxy.state import ModeState as MS_State
    from proxy.agent.types import ModeState as MS_Types
    from proxy.skills.skill_definition import ModeState as MSD

    assert MS_State is MS_Types, (
        "proxy.state.ModeState must be the same class object as proxy.agent.types.ModeState"
    )
    assert MS_State is MSD, (
        "proxy.state.ModeState must be the same class object as proxy.skills.skill_definition.ModeState"
    )


def test_assert_session_secret_configured_raises_in_prod_without_secret():
    """RuntimeError raised when PROXY_ENV=production and PROXY_SESSION_SECRET=''.

    In dev mode, no error. With secret set, no error.
    """
    from proxy.assertions import assert_session_secret_configured

    original_env = os.environ.get("PROXY_ENV")
    original_secret = os.environ.get("PROXY_SESSION_SECRET")

    try:
        # Production without secret: raises
        os.environ["PROXY_ENV"] = "production"
        os.environ.pop("PROXY_SESSION_SECRET", None)
        with pytest.raises(RuntimeError) as exc_info:
            assert_session_secret_configured()
        assert "PROXY_SESSION_SECRET" in str(exc_info.value)

        # Production with secret: no raise
        os.environ["PROXY_ENV"] = "production"
        os.environ["PROXY_SESSION_SECRET"] = "test-secret-12345"
        assert_session_secret_configured()  # should not raise

        # Dev without secret: no raise
        os.environ["PROXY_ENV"] = "dev"
        os.environ.pop("PROXY_SESSION_SECRET", None)
        assert_session_secret_configured()  # should not raise

        # Unset PROXY_ENV: defaults to production behavior
        os.environ.pop("PROXY_ENV", None)
        os.environ.pop("PROXY_SESSION_SECRET", None)
        with pytest.raises(RuntimeError):
            assert_session_secret_configured()
    finally:
        if original_env is not None:
            os.environ["PROXY_ENV"] = original_env
        else:
            os.environ.pop("PROXY_ENV", None)
        if original_secret is not None:
            os.environ["PROXY_SESSION_SECRET"] = original_secret
        else:
            os.environ.pop("PROXY_SESSION_SECRET", None)


def test_skill_definition_imports_mode_state_from_proxy_state():
    """skill_definition.py line 5 is `from proxy.state import ModeState` (Phase 2A chain)."""
    import inspect
    import proxy.skills.skill_definition as sd_module

    source = inspect.getsource(sd_module)
    first_lines = "\n".join(source.split("\n")[:10])
    assert "from proxy.state import ModeState" in first_lines, (
        f"skill_definition.py should import ModeState from proxy.state. "
        f"First 10 lines:\n{first_lines}"
    )
    # Also verify the old import is gone
    assert "from proxy.agent.types import ModeState" not in first_lines, (
        "skill_definition.py should NO LONGER import ModeState directly from proxy.agent.types"
    )
