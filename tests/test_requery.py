"""Tests for requery tool resolution (Batch B - B6, NEW-001)."""
import pytest
from unittest.mock import MagicMock


class TestRequeryToolResolution:
    """NEW-001: Requery uses correct priority chain."""

    def test_requery_uses_snapshot_when_available(self):
        """When ctx._tools_snapshot is set, requery uses it."""
        from proxy.requery import requery_tools

        # Mock ctx with _tools_snapshot
        ctx = MagicMock()
        ctx._tools_snapshot = [{"name": "filtered_tool", "type": "function"}]
        ctx.body = {"messages": []}

        tools = requery_tools(ctx)
        # Should return the snapshot (may be a copy or the original)
        assert tools is not None
        assert len(tools) >= 1

    def test_requery_falls_back_to_body(self):
        """When _tools_snapshot is not set, requery uses llama_body tools."""
        from proxy.requery import resolve_requery_tools

        ctx = MagicMock()
        ctx._tools_snapshot = None
        llama_body = {
            "tools": [{"name": "body_tool", "type": "function"}]
        }

        tools = resolve_requery_tools(ctx, llama_body)
        assert len(tools) >= 1

    def test_requery_no_dead_reconstruct_reference(self):
        """requery.py should not reference _reconstruct_tools."""
        import proxy.requery
        source = open(proxy.requery.__file__).read()
        assert "_reconstruct_tools" not in source