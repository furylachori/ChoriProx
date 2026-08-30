"""Slice 8a — dispatch routing tests for decomposed ticket tools."""

import pytest
from unittest.mock import patch, MagicMock

from memory.tools.dispatch import execute_memory_tool, TICKET_TOOL_OPERATIONS


class TestTicketToolDispatch:
    """Verify dispatch routing for each of the 10 new tool names."""

    def test_all_10_names_route_to_execute_ticket_tool(self):
        """Each decomposed tool name dispatches via execute_ticket_tool with correct operation."""
        for tool_name, expected_op in TICKET_TOOL_OPERATIONS.items():
            with patch("memory.tools.proxy_ticket.execute_ticket_tool") as mock_exec:
                mock_exec.return_value = {"ok": True, "operation": expected_op}
                result = execute_memory_tool(tool_name, {"some": "args"})
                mock_exec.assert_called_once()
                call_args = mock_exec.call_args
                assert call_args[0][0] == expected_op
                assert result == {"ok": True, "operation": expected_op}

    def test_old_proxy_ticket_returns_unknown_error(self):
        """The old monolithic proxy_ticket name is no longer recognized."""
        result = execute_memory_tool("proxy_ticket", {"action": "list_tasks"})
        assert result["error"]["code"] == "unknown_memory_tool"

    def test_unknown_tool_returns_error(self):
        """Unknown memory tool returns error_envelope."""
        result = execute_memory_tool("nonexistent_tool", {})
        assert result["error"]["code"] == "unknown_memory_tool"


class TestDispatchRevertCheck:
    """Revert tests — confirm routing fails without the mapping."""

    def test_remove_mapping_dispatches_to_unknown(self):
        """Without TICKET_TOOL_OPERATIONS routing, new tool → unknown_memory_tool."""
        result = execute_memory_tool("proxy_ticket_list_tasks", {})
        assert result.get("error", {}).get("code") != "unknown_memory_tool"
