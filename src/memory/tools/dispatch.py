"""Memory tool dispatcher — routes tool calls to the appropriate handler."""
import logging

from memory.query_executor import InfrastructureError

logger = logging.getLogger("memory.tools")

TICKET_TOOL_OPERATIONS = {
    "proxy_ticket_list_tasks": "list_tasks",
    "proxy_ticket_get_task": "get_task",
    "proxy_ticket_create_tasks": "create_tasks",
    "proxy_ticket_update_task": "update_task",
    "proxy_ticket_edit": "edit",
    "proxy_ticket_set_parent": "set_parent",
    "proxy_ticket_clear_parent": "clear_parent",
    "proxy_ticket_clear_blocked_by": "clear_blocked_by",
    "proxy_ticket_set_priority": "set_priority",
    "proxy_ticket_append_review_note": "append_review_note",
}


def execute_memory_tool(name, arguments):
    """Dispatch a memory tool call to the appropriate handler."""
    logger.info(
        "Executing memory tool '%s' with args: %s", name, list(arguments.keys())
    )
    try:
        if name == "recall":
            from memory.tools.recall import recall
            return recall(
                queries=arguments.get("queries", []),
                keywords=arguments.get("keywords", []),
                limit=arguments.get("limit", 10),
                session_id=arguments.get("session_id"),
            )
        elif name == "search":
            from memory.tools.search import search
            return search(
                keywords=arguments.get("keywords", []),
                match_all=arguments.get("match_all", False),
                session_id=arguments.get("session_id"),
                time_start=arguments.get("time_start"),
                time_end=arguments.get("time_end"),
                limit=arguments.get("limit", 10),
            )
        elif name in TICKET_TOOL_OPERATIONS:
            from memory.tools.proxy_ticket import execute_ticket_tool
            operation = TICKET_TOOL_OPERATIONS[name]
            return execute_ticket_tool(
                operation, arguments,
                session_id=arguments.get("session_id"),
                client_hint=arguments.get("client_hint", ""),
            )
        elif name == "proxy_work":
            from memory.tools.proxy_work import execute_proxy_work
            return execute_proxy_work(
                arguments,
                session_id=arguments.get("session_id"),
                client_hint=arguments.get("client_hint", ""),
            )
        else:
            logger.warning("Unknown memory tool: %s", name)
            from proxy.tool_pipeline.action_envelope import error_envelope
            return error_envelope(
                "unknown", None, "unknown_memory_tool",
                f"Unknown memory tool: {name}", "",
            )
    except InfrastructureError:
        raise
    except Exception as e:
        logger.error("Memory tool '%s' failed: %s", name, e)
        from proxy.tool_pipeline.action_envelope import error_envelope
        return error_envelope(
            name, None, "tool_execution_failed",
            str(e), "",
        )
