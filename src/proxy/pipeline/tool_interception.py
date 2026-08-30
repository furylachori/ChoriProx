"""Tool interception — parse tool calls from upstream responses and run memory tool loop."""

import json
import logging
import re
from typing import Any

from proxy import state as _state

logger = logging.getLogger("proxy.pipeline.tool_interception")


def handle_tool_interception(
    data: dict,
    body: dict,
    session_id: str,
    upstream_url: str,
    tool_parser,
    auth_headers: dict,
    original_model: str,
    model_cfg=None,
    llama_body: dict = None,
    execute_fn=None,
    parse_tools_fn=None,
    max_iterations: int = 5,
    max_total_time: float = 900.0,
    max_result_size: int = 16 * 1024,
    max_calls_per_iteration: int = _state.MAX_CALLS_PER_ITERATION,
    is_subagent: bool = False,
    subagent_ticket_mode: bool = False,
    unattended: bool = False,
    client_hint: str = "",
) -> dict:
    """Parse tool calls from response and run memory tool loop if needed.

    Delegates to ToolPipeline.run() for tool execution.
    """
    if llama_body is None:
        llama_body = body

    if "choices" not in data or not data["choices"]:
        return data

    message = data["choices"][0]["message"]
    content = message.get("content", "") or ""

    reasoning = message.get("reasoning", "") or ""
    has_dsml = bool(reasoning) and (
        ("｜" in reasoning and "tool_calls" in reasoning)
        or ("‹" in reasoning and "›" in reasoning and "invoke" in reasoning)
        or ("<tool_call>" in reasoning)
    )
    parse_target = reasoning if has_dsml else content

    dsml_block = None
    if has_dsml:
        dsml_match = re.search(
            r"<｜DSML｜tool_calls>.*?</｜DSML｜tool_calls>", reasoning, re.DOTALL
        )
        if dsml_match:
            dsml_block = dsml_match.group(0)
        try:
            from memory.sync import _cache_reasoning
            from proxy.chat_history import chat_histories, chat_histories_lock
            with chat_histories_lock:
                msg_count = len(chat_histories.get(session_id, []))
            _cache_reasoning(session_id, msg_count, message.get("reasoning", ""))
        except Exception:
            pass

    native_tool_calls = message.get("tool_calls")
    if isinstance(native_tool_calls, list) and native_tool_calls:
        tool_calls_list = []
        for tc in native_tool_calls:
            try:
                func = tc["function"]
                arguments = func["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_calls_list.append({"name": func["name"], "arguments": arguments})
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                logger.warning("Failed to parse native tool call for session=%s: %s", session_id, e)
    else:
        from proxy.tool_parsing import parse_all_tools_from_content
        tool_calls_list = parse_all_tools_from_content(parse_target)

    if not tool_calls_list:
        return data

    from proxy.tool_pipeline.tool_pipeline import ToolPipeline, ToolContext

    # Extract incoming tool names for dynamic alias resolution.
    incoming_tool_names = None
    if isinstance(body.get("tools"), list):
        incoming_tool_names = {
            t.get("function", {}).get("name", t.get("name", ""))
            for t in body["tools"]
            if t.get("function", {}).get("name") or t.get("name")
        }

    ctx = ToolContext(
        session_id=session_id,
        body=body,
        llama_body=llama_body,
        auth_headers=auth_headers,
        original_model=original_model,
        upstream_url=upstream_url,
        model_cfg=model_cfg,
        execute_fn=execute_fn,
        parse_tools_fn=parse_tools_fn,
        max_iterations=max_iterations,
        max_total_time=max_total_time,
        max_result_size=max_result_size,
        max_calls_per_iteration=max_calls_per_iteration,
        tool_parser=tool_parser,
        is_subagent=is_subagent,
        subagent_ticket_mode=subagent_ticket_mode,
        unattended=unattended,
        tool_calls_list=tool_calls_list,
        client_hint=client_hint,
        incoming_tool_names=incoming_tool_names,
    )

    pipeline = ToolPipeline()
    return pipeline.run(data, ctx)
