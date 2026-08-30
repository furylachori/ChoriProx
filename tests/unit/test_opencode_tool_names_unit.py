"""
Unit tests for ToolRegistry as single source of truth for tool names.
After Tier 3 Phase 3, the DEFAULT_TOOL_NAMES and MEMORY_TOOL_NAMES
literals were deleted. All consumers go through ToolRegistry.

Validates the tool names used by the proxy's adapter-aware tool filtering.
"""

import pytest

from proxy.tool_pipeline.tool_registry import registry

# Memory tools (from ToolRegistry.get_memory_tool_names)
MEMORY_TOOLS = {
    "recall",
    "search",
    "proxy_ticket",
    "proxy_work",
    "proxy_ticket_list_tasks",
    "proxy_ticket_get_task",
    "proxy_ticket_create_tasks",
    "proxy_ticket_update_task",
    "proxy_ticket_edit",
    "proxy_ticket_set_parent",
    "proxy_ticket_clear_parent",
    "proxy_ticket_clear_blocked_by",
    "proxy_ticket_set_priority",
    "proxy_ticket_append_review_note",
}


class TestDefaultTools:
    """Tests for ToolRegistry.get_default_tools()."""

    def test_has_twelve_tools(self):
        """get_default_tools() should have 12 tools (was 13 in the old flat list; 'question' moved to PASSTHROUGH)."""
        assert len(registry.get_default_tools()) == 12

    def test_contains_core_editor_tools(self):
        """get_default_tools() should contain the core editor tools."""
        core = {"bash", "read", "glob", "grep", "edit", "write"}
        assert core.issubset(registry.get_default_tools())

    def test_contains_web_tools(self):
        """get_default_tools() should contain webfetch and MiniMax_web_search."""
        web = {"webfetch", "MiniMax_web_search"}
        assert web.issubset(registry.get_default_tools())

    def test_contains_task_tool(self):
        """get_default_tools() must include 'task' for sub-agent support."""
        assert "task" in registry.get_default_tools()

    def test_contains_suggest_tool(self):
        """get_default_tools() should contain 'suggest'."""
        assert "suggest" in registry.get_default_tools()

    def test_contains_recall_and_search(self):
        """get_default_tools() should contain recall and search (subset of memory tools)."""
        assert {"recall", "search"}.issubset(registry.get_default_tools())

    def test_no_passthrough_tools(self):
        """get_default_tools() should NOT include PASSTHROUGH tools (question, ask, todowrite, etc.)."""
        passthrough = {"question", "ask", "todowrite", "new_task", "TaskCreateTool"}
        assert not (passthrough & registry.get_default_tools())

    def test_returns_set(self):
        """get_default_tools() should return a set for O(1) lookup."""
        assert isinstance(registry.get_default_tools(), set)


class TestMemoryToolNamesSync:
    """Tests for memory tool names consistency."""

    def test_memory_tool_names_matches_expected(self):
        """get_memory_tool_names() must match the expected 4 tools."""
        assert registry.get_memory_tool_names() == MEMORY_TOOLS

    def test_adapter_tool_names_include_task(self):
        """Adapters should return 'task' in their tool set."""
        from clients.kilocode import KiloCodeAdapter
        from clients.opencode import OpenCodeAdapter

        kilo = KiloCodeAdapter()
        opencode = OpenCodeAdapter()

        assert "task" in kilo.get_tool_names()
        assert "task" in opencode.get_tool_names()

    def test_adapter_tool_names_superset_of_default(self):
        """Adapter tool names should be a superset of the registry's default tools."""
        from clients.kilocode import KiloCodeAdapter

        adapter_tools = KiloCodeAdapter().get_tool_names()
        assert registry.get_default_tools().issubset(adapter_tools)
