"""Smoke tests for _emit_tier2_pin: populated map inserts a pin, empty no-ops."""
import sys
import types

import pytest

from proxy.context_v2 import _emit_tier2_pin


@pytest.fixture(autouse=True)
def _reset_instruction_map():
    import proxy.state as state
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
    yield
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()


def _make_ctx(workspace_name=None):
    return types.SimpleNamespace(workspace_name=workspace_name, request_id="r-smoke")


class TestEmitTier2Pin:
    def test_empty_map_returns_input_unchanged(self):
        msgs = [{"role": "system", "content": "hi"}, {"role": "user", "content": "q"}]
        result = _emit_tier2_pin(msgs, "sess-1")
        assert result is msgs
        assert result == msgs

    def test_populated_map_inserts_pin_after_first_system(self):
        import proxy.state as state
        with state.vfs_lock:
            state.record_instruction_file("ws-x", "/a/AGENTS.md", "rule alpha")
        ctx = _make_ctx(workspace_name="ws-x")
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        result = _emit_tier2_pin(msgs, "sess-1", ctx=ctx)
        assert result is not msgs
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "sys"
        assert result[1]["role"] == "system"
        assert "CLIENT_SUB_INSTRUCTIONS" in result[1]["content"]
        assert "rule alpha" in result[1]["content"]
        assert result[2] == {"role": "user", "content": "q"}

    def test_no_system_message_inserts_at_index_0(self):
        import proxy.state as state
        with state.vfs_lock:
            state.record_instruction_file("ws-x", "/a/AGENTS.md", "rule beta")
        ctx = _make_ctx(workspace_name="ws-x")
        msgs = [{"role": "user", "content": "q"}]
        result = _emit_tier2_pin(msgs, "sess-1", ctx=ctx)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "CLIENT_SUB_INSTRUCTIONS" in result[0]["content"]

    def test_uses_ctx_workspace_when_set(self):
        import proxy.state as state
        with state.vfs_lock:
            state.record_instruction_file("ws-ctx", "/a/AGENTS.md", "from ctx")
            state.record_instruction_file("ws-other", "/a/AGENTS.md", "from other")
        ctx = _make_ctx(workspace_name="ws-ctx")
        result = _emit_tier2_pin([{"role": "user", "content": "q"}], "sess-1", ctx=ctx)
        assert "from ctx" in result[0]["content"]
        assert "from other" not in result[0]["content"]

    def test_input_list_not_mutated(self):
        import proxy.state as state
        with state.vfs_lock:
            state.record_instruction_file("ws-x", "/a/AGENTS.md", "rule gamma")
        ctx = _make_ctx(workspace_name="ws-x")
        msgs = [{"role": "system", "content": "sys"}]
        original_len = len(msgs)
        _emit_tier2_pin(msgs, "sess-1", ctx=ctx)
        assert len(msgs) == original_len
        assert msgs == [{"role": "system", "content": "sys"}]
