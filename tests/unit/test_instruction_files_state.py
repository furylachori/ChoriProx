"""Unit tests for Tier-2 instruction file state (state.py).

This state is module-level and the root conftest does NOT reset it yet —
every test MUST clean up via the `_reset_instruction_files` fixture.
"""
import pytest

import proxy.state as state
from proxy.state import (
    record_instruction_file,
    get_instruction_files,
    instruction_files_by_workspace,
    INSTRUCTION_FILES_MAX_PER_WORKSPACE,
    INSTRUCTION_FILES_MAX_WORKSPACES,
    DEFAULT_WORKSPACE,
)


@pytest.fixture(autouse=True)
def _reset_instruction_files():
    """Clean module-level state before and after each test."""
    with state.vfs_lock:
        instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()
    yield
    with state.vfs_lock:
        instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()


class TestInnerCap:
    def test_64_distinct_all_present(self):
        with state.vfs_lock:
            for i in range(64):
                record_instruction_file("ws", f"path/{i}.md", f"content-{i}")
        result = get_instruction_files("ws")
        assert len(result) == 64
        for i in range(64):
            assert result[f"path/{i}.md"] == f"content-{i}"

    def test_65th_evicts_first_inserted(self):
        with state.vfs_lock:
            for i in range(65):
                record_instruction_file("ws", f"path/{i}.md", f"content-{i}")
        result = get_instruction_files("ws")
        assert len(result) == 64
        assert "path/0.md" not in result
        assert result["path/64.md"] == "content-64"
        # The first-inserted (path/0.md) is gone; path/1..64 remain
        for i in range(1, 65):
            assert f"path/{i}.md" in result


class TestRefreshOnResight:
    def test_refresh_updates_content_but_not_position(self):
        with state.vfs_lock:
            record_instruction_file("ws", "A.md", "v1")
            record_instruction_file("ws", "B.md", "v1")
            record_instruction_file("ws", "C.md", "v1")

            # Refresh A with new content
            result = record_instruction_file("ws", "A.md", "v2")
            assert result == "updated"

            # Position unchanged: A still first
            key = "ws"
            keys_list = list(instruction_files_by_workspace[key].keys())
            assert keys_list == ["A.md", "B.md", "C.md"]
            # Content updated
            assert instruction_files_by_workspace[key]["A.md"] == "v2"

    def test_refresh_same_content_returns_same(self):
        with state.vfs_lock:
            record_instruction_file("ws", "A.md", "v1")
            record_instruction_file("ws", "B.md", "v1")
            result = record_instruction_file("ws", "A.md", "v1")
            assert result == "same"


class TestOuterCap:
    def test_129_workspaces_evicts_first(self):
        for i in range(129):
            with state.vfs_lock:
                record_instruction_file(f"ws-{i}", "file.md", f"content-{i}")
        with state.vfs_lock:
            assert len(instruction_files_by_workspace) == 128
            # First workspace dropped
            assert "ws-0" not in instruction_files_by_workspace
            # ws-1..128 present
            assert "ws-1" in instruction_files_by_workspace
            assert "ws-128" in instruction_files_by_workspace


class TestDefaultWorkspaceIsolation:
    def test_different_sessions_isolated(self):
        with state.vfs_lock:
            record_instruction_file(DEFAULT_WORKSPACE, "shared.md", "alpha", session_id="sess-A")
            record_instruction_files_ = record_instruction_file(
                DEFAULT_WORKSPACE, "shared.md", "beta", session_id="sess-B"
            )

        result_a = get_instruction_files(DEFAULT_WORKSPACE, session_id="sess-A")
        result_b = get_instruction_files(DEFAULT_WORKSPACE, session_id="sess-B")
        assert result_a["shared.md"] == "alpha"
        assert result_b["shared.md"] == "beta"
        assert result_a is not result_b


class TestReturnValueSemantics:
    def test_returns_new_on_first_sight(self):
        with state.vfs_lock:
            assert record_instruction_file("ws", "f.md", "c1") == "new"

    def test_returns_updated_on_changed_content(self):
        with state.vfs_lock:
            record_instruction_file("ws", "f.md", "c1")
            assert record_instruction_file("ws", "f.md", "c2") == "updated"

    def test_returns_same_on_identical_content(self):
        with state.vfs_lock:
            record_instruction_file("ws", "f.md", "c1")
            assert record_instruction_file("ws", "f.md", "c1") == "same"


class TestGetReturnsCopy:
    def test_mutation_does_not_affect_module_state(self):
        with state.vfs_lock:
            record_instruction_file("ws", "f.md", "original")
        result = get_instruction_files("ws")
        result["f.md"] = "mutated"
        result2 = get_instruction_files("ws")
        assert result2["f.md"] == "original"

    def test_missing_workspace_returns_empty_dict(self):
        result = get_instruction_files("nonexistent-workspace-xyz")
        assert result == {}
