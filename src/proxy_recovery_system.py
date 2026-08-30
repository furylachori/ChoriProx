"""Proxy recovery system — tool description optimization.

Sole live function: `apply_proxy_recovery_system(body, session_id)` which
walks `body["tools"]` and rewrites tool schemas that lack `parameters`
into the optimized schemas defined in `tool_optimizer.py`. Memory tools
and tools that already carry a full `parameters` schema pass through
unchanged. Called from `proxy.pipeline.stages.ProviderCompatStage`.
"""

import logging

from tool_optimizer import generate_optimized_tool_description


# Configure logging
logger = logging.getLogger("proxy_recovery")
logger.setLevel(logging.INFO)


def apply_proxy_recovery_system(body: dict, session_id: str) -> dict:
    """Optimize tool descriptions in request body."""
    if "tools" in body:
        optimized_tools = []
        for tool in body["tools"]:
            tool_name = tool.get("function", {}).get("name", tool.get("name", ""))
            # Only optimize tools that DON'T already have full definitions
            # (memory tools and tools with parameters should pass through unchanged)
            if tool_name and "parameters" not in tool.get("function", {}):
                optimized = generate_optimized_tool_description(tool_name)
                optimized_tools.append(optimized)
            else:
                optimized_tools.append(tool)

        if optimized_tools:
            body["tools"] = optimized_tools

    return body
