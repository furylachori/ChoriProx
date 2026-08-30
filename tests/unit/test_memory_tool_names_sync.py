"""Verify memory tool names are consistent across all consumers.

After Tier 3 Phase 3, MEMORY_TOOL_NAMES literal was deleted from
src/memory/tools.py. All consumers go through ToolRegistry.get_memory_tool_names().

This test ensures the ToolRegistry and ProxyFeatures stay in sync.
"""

from clients.orchestrator_config import ProxyFeatures
from proxy.tool_pipeline.tool_registry import registry


class TestMemoryToolNamesSync:
    def test_proxy_features_memory_tool_names_matches_registry(self):
        """ProxyFeatures.memory_tool_names must match ToolRegistry.get_memory_tool_names()."""
        assert ProxyFeatures.memory_tool_names == frozenset(registry.get_memory_tool_names())

    def test_registry_memory_tool_names_contains_all_memory_tools(self):
        """ToolRegistry.get_memory_tool_names() must contain all 14 memory tools."""
        expected = {
            "recall", "search", "proxy_ticket", "proxy_work",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
        }
        assert registry.get_memory_tool_names() == expected
        assert ProxyFeatures.memory_tool_names == frozenset(expected)
