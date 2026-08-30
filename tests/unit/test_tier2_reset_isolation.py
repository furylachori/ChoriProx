"""Spec §9.2 — reset isolation: _reset_all_shared_state clears Tier-2 state."""
import pytest


def test_reset_all_shared_state_clears_tier2():
    """Populate the Tier-2 map, invoke the canonical reset, assert cleared."""
    import proxy.state as state
    from proxy.state import record_instruction_file

    # Populate
    with state.vfs_lock:
        record_instruction_file("ws-isolation", "path.md", "content")
    assert len(state.instruction_files_by_workspace) == 1

    # Invoke the canonical reset (integration path)
    from tests.integration.conftest import _reset_all_shared_state

    _reset_all_shared_state()

    # Assert cleared
    assert len(state.instruction_files_by_workspace) == 0
    assert len(state._last_scanned_by_tool_call) == 0
