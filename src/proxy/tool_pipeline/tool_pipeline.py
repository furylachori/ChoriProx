"""Unified tool lifecycle: parse → classify → execute → format.

Phase 6: ToolPipeline wraps, not replaces, the existing run_memory_tool_loop().
handle_tool_interception() delegates to ToolPipeline.run(), which internally
calls run_memory_tool_loop() with MessageBuffer instead of raw list.
"""
import json
import logging
import re
import time
import uuid
import functools
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, List, Tuple

from proxy.tool_pipeline.tool_call import ToolCall
from proxy.tool_pipeline.tool_guard import MemoryToolGuard
from proxy.tool_pipeline.tool_registry import ToolRegistry, registry as default_registry
from proxy.tool_pipeline.action_registry import dispatch_action

logger = logging.getLogger("proxy.tool_pipeline")


_INJECTED_KEYS = frozenset({"session_id", "client_hint"})

TRUNCATION_REASONS = frozenset({"length", "content_filter"})


def _stable_args_key(name, arguments):
    """Build a stable lookup key for tool args, ignoring proxy-injected keys.

    run_memory_tool_loop injects session_id and client_hint into proxy_* args
    before calling execute_fn. These keys must be stripped so guard-time and
    execution-time keys match.
    """
    clean = {k: v for k, v in (arguments or {}).items() if k not in _INJECTED_KEYS}
    return (name, json.dumps(clean, sort_keys=True, default=str))


@dataclass
class ToolContext:
    """Context passed to ToolPipeline for tool execution.
    
    Thin value object constructed at handle_tool_interception() boundary.
    Contains all params that ToolPipeline needs from handle_tool_interception().
    
    Phase 7: These fields get merged directly into RequestContext.
    """
    session_id: str
    body: dict
    llama_body: dict
    auth_headers: dict
    original_model: str
    upstream_url: str
    model_cfg: Any = None
    execute_fn: Callable = None
    parse_tools_fn: Callable = None
    max_iterations: int = 5
    max_total_time: float = 90.0
    max_result_size: int = 16 * 1024
    max_calls_per_iteration: int = 50
    tool_parser: Any = None
    is_subagent: bool = False
    subagent_ticket_mode: bool = False
    unattended: bool = False  # True if no human partner is available
    tool_calls_list: Optional[List[dict]] = None
    client_hint: str = ""
    incoming_tool_names: Optional[set] = None


class ToolPipeline:
    """Unified parse → classify → execute → format lifecycle.
    
    Error handling strategy:
    - Parse failure: return response_data unchanged (no tool calls → normal response)
    - Guard rejection: return error in tool_calls format (LLM sees the error)
    - Execution failure: catch per-tool, return partial results (some succeed, some fail)
    - Format failure: return response_data with error injected into content
    
    All errors are logged at WARNING level with session_id for debugging.
    No exceptions escape to the caller — ToolPipeline is a catch-all boundary.
    """

    def __init__(
        self, 
        registry: Optional[ToolRegistry] = None,
        action_registry: Optional[dict] = None,
    ):
        self.registry = registry or default_registry
        self.action_registry = action_registry or {}
        self.guard = MemoryToolGuard()
        self._cached_reads = {}  # file_path → cached content (for VFS read interception)
        self._blocked_errors = {}  # (tool_name, stable_args) → error message (soft-error path)

    def run(self, response_data: dict, ctx: ToolContext) -> dict:
        """Main entry point. Called by handle_tool_interception() as delegate.
        
        Args:
            response_data: The LLM response dict from upstream
            ctx: ToolContext with session, auth, config, and callback functions
        
        Returns:
            Modified response_data with tool results injected, or unchanged
            if no memory tools were called or execution failed.
        """
        try:
            from proxy.tool_parsing import build_valid_params_by_tool, extract_file_path
            vpbt = build_valid_params_by_tool(ctx.llama_body.get("tools", []))
            tool_calls = self._parse(response_data, ctx.incoming_tool_names, valid_params_by_tool=vpbt)
            if not tool_calls:
                logger.debug(f"ToolPipeline: No tool calls parsed for session={ctx.session_id}")
                return response_data

            # Build original (unnormalized) tool_calls_list for run_memory_tool_loop
            # _parse() already returns ToolCall objects with normalized arguments,
            # but run_memory_tool_loop calls make_tool_calls() internally which
            # also normalizes — so we reconstruct from the original response to
            # avoid double-normalization.
            message = response_data["choices"][0].get("message", {})
            if ctx.tool_calls_list is not None:
                original_tool_calls_list = ctx.tool_calls_list
            else:
                native_tool_calls = message.get("tool_calls")
                if isinstance(native_tool_calls, list) and native_tool_calls:
                    original_tool_calls_list = []
                    for tc in native_tool_calls:
                        try:
                            func = tc["function"]
                            args = func["arguments"]
                            if isinstance(args, str):
                                args = json.loads(args)
                            original_tool_calls_list.append({"name": func["name"], "arguments": args})
                        except (KeyError, TypeError, json.JSONDecodeError):
                            pass
                else:
                    original_tool_calls_list = [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]

            logger.info(
                f"ToolPipeline: Parsed {len(tool_calls)} tool call(s) for session={ctx.session_id}: "
                f"{[tc.name for tc in tool_calls]}"
            )

            memory_calls, passthrough_calls = self._classify(tool_calls)

            # ★ VFS read interception — check cache for read tool calls.
            # M3-2: cache hit is served (the read already happened in a
            # prior request, so the read-once guard is bypassed via
            # `from_cache=True` on the tool call annotation).
            self._cached_reads = {}
            remaining_passthrough = []
            # Iterate ONLY passthrough (the default registry puts `read`
            # in passthrough). When a custom registry classifies `read`
            # as MEMORY, the read tool call stays in memory_calls and
            # is also checked via a second pass below.
            for tc in list(passthrough_calls):
                name = tc.name if hasattr(tc, 'name') else tc.get("name", "")

                # ★ Unattended question check — return graceful error instead of blocking
                # Block if: unattended=True OR is_subagent=True (sub-agents are always unattended)
                if name == "question" and (getattr(ctx, 'unattended', False) or getattr(ctx, 'is_subagent', False)):
                    logger.warning(
                        f"ToolPipeline: question tool blocked in unattended mode for session={ctx.session_id}"
                    )
                    message["content"] = (
                        "No human partner is available. Proceed with the most conservative "
                        "interpretation of the user's intent. If truly unresolvable, return PARTIAL "
                        "with a one-line explanation."
                    )
                    message.pop("tool_calls", None)
                    response_data["choices"][0]["finish_reason"] = "stop"
                    return response_data

                if name == "read":
                    cached_content = self._check_vfs_read_cache(tc, ctx.session_id, getattr(ctx, 'is_subagent', False), getattr(ctx, 'client_hint', ''))
                    if cached_content is not None:
                        # Cache hit — store for delivery, move to memory_calls.
                        # M3-2: mark the tool call `from_cache=True` so the
                        # read-once guard in tool_guard.py:251 does NOT
                        # increment the read count and does NOT block
                        # the read. The cache means the read already
                        # happened in a prior request and was already
                        # counted then.
                        args = tc.arguments if hasattr(tc, 'arguments') else tc.get("arguments", {})
                        file_path = extract_file_path(args)
                        if file_path:
                            import os
                            normalized = os.path.normpath(os.path.realpath(file_path))
                            self._cached_reads[normalized] = cached_content
                            # Annotate the tool call so the guard sees from_cache=True.
                            # The original `tc` is a frozen dataclass; replace
                            # creates a new instance with from_cache=True.
                            tc = replace(tc, from_cache=True)
                            memory_calls.append(tc)
                            # Remove from passthrough: the read is now served as a memory call.
                            logger.info(f"ToolPipeline: VFS cache hit served for read: {normalized}")
                            continue
                remaining_passthrough.append(tc)
            passthrough_calls = remaining_passthrough

            # M3-2 second pass: also check MEMORY-classified reads for cache hits.
            # The first pass only iterates passthrough, so a custom registry that
            # classifies `read` as MEMORY (some tests do this) needs its own pass.
            # We mutate a local copy of memory_calls so the subsequent guard loop
            # sees the annotated-from_cache tool calls.
            annotated_memory = []
            for tc in memory_calls:
                name = tc.name if hasattr(tc, 'name') else tc.get("name", "")
                if name == "read" and not getattr(tc, "from_cache", False):
                    cached_content = self._check_vfs_read_cache(tc, ctx.session_id, getattr(ctx, 'is_subagent', False), getattr(ctx, 'client_hint', ''))
                    if cached_content is not None:
                        args = tc.arguments if hasattr(tc, 'arguments') else tc.get("arguments", {})
                        file_path = extract_file_path(args)
                        if file_path:
                            import os
                            normalized = os.path.normpath(os.path.realpath(file_path))
                            self._cached_reads[normalized] = cached_content
                            tc = replace(tc, from_cache=True)
                            logger.info(f"ToolPipeline: VFS cache hit served for read (memory pass): {normalized}")
                annotated_memory.append(tc)
            memory_calls = annotated_memory

            # Security guard — collect blocked errors as soft errors.
            # Soft-error path: instead of returning early (which pops tool_calls
            # and forces finish_reason="stop"), we record the error keyed by
            # stable (name, args) and let _execute_via_loop run. The wrapped
            # execute_fn then returns {"error": <msg>} for blocked calls so
            # the LLM sees the rejection as a tool result and can recover.
            self._blocked_errors = {}
            for tc in list(memory_calls):
                error = self.guard.check(tc, ctx)
                if error:
                    logger.warning(
                        f"ToolPipeline: Guard blocked {tc.name}: {error}"
                    )
                    self._blocked_errors[_stable_args_key(tc.name, tc.arguments)] = error

            # Filter out guard-blocked tool calls BEFORE the VFS write-through.
            # M3-1: a guard-blocked write still updates VFS via the original
            # `_write_through_cache(tool_calls, ...)` call, but the client never
            # executes it (the LLM sees a soft error). The cache then serves
            # bogus content to the next sub-agent on cache hit — cross-agent
            # cache poisoning. Filter to memory_calls minus blocked (option A
            # from the planning_info: filter at call site, leaves
            # _write_through_cache unchanged as a pure internal helper).
            blocked_keys = set(self._blocked_errors.keys())
            non_blocked_calls = [
                tc for tc in tool_calls
                if _stable_args_key(tc.name, tc.arguments) not in blocked_keys
            ]
            self._write_through_cache(non_blocked_calls, ctx.session_id, getattr(ctx, 'is_subagent', False), getattr(ctx, 'client_hint', ''))

            if not memory_calls:
                logger.debug(f"ToolPipeline: No memory tools to execute, all passthrough for session={ctx.session_id}")
                # Rebuild response_data with normalized arguments so the caller
                # sees the canonical key names (e.g. "file_path" not "path").
                from proxy.tool_parsing import make_tool_calls
                message["tool_calls"] = make_tool_calls(original_tool_calls_list, valid_params_by_tool=vpbt)
                
                # Strip OpenAI-style tool_call blocks from content for passthrough.
                clean_content = message.get("content", "") or ""
                if clean_content and "tool_call" in clean_content:
                    clean_content = re.sub(
                        r'<tool_call>.*?</tool_call>',
                        '', clean_content, flags=re.DOTALL
                    )
                    clean_content = clean_content.strip() or None
                    message["content"] = clean_content

                if message.get("tool_calls"):
                    _fr = response_data["choices"][0].get("finish_reason")
                    if _fr is not None and _fr not in TRUNCATION_REASONS:
                        response_data["choices"][0]["finish_reason"] = "tool_calls"
                return response_data

            result = self._execute_via_loop(response_data, memory_calls, ctx, original_tool_calls_list, valid_params_by_tool=vpbt)

            return self._format(response_data, result, ctx)

        except Exception as e:
            logger.error(
                f"ToolPipeline: Unhandled exception in run() for session={ctx.session_id}: {e}",
                exc_info=True
            )
            return response_data
        finally:
            self._cached_reads = {}

    def _parse(self, response_data: dict, valid_names: Optional[set] = None, valid_params_by_tool=None) -> List[ToolCall]:
        """Extract ToolCall objects from LLM response.
        
        Algorithm:
        1. Check if response contains "tool_calls" key (OpenAI format).
            - If present: iterate tool_calls, parse each function's arguments
              from JSON string to dict, create ToolCall(id=..., name=..., 
              arguments=..., source="llm").
        2. If no "tool_calls" key, check if content contains DSML/YAML tool
            blocks (proxy-native format).
            - Use parse_yaml_tool_blocks(content) from tool_parsing.py
            - Create ToolCall objects with source="parsed"
        3. Return empty list if neither format found.
        
        Args:
            response_data: The LLM response dict from upstream.
            valid_names: Set of tool names the client sent in the request.
                Used for dynamic alias resolution (e.g., read_file → read
                when the client only knows about 'read').
        """
        from proxy.tool_parsing import normalize_tool_arguments, canonicalize_tool_name

        tool_calls: List[ToolCall] = []

        if "choices" not in response_data or not response_data["choices"]:
            return []

        message = response_data["choices"][0].get("message", {})
        if not message:
            return []

        native_tool_calls = message.get("tool_calls")
        if isinstance(native_tool_calls, list) and native_tool_calls:
            for tc in native_tool_calls:
                try:
                    name = tc.get("function", {}).get("name", "")
                    if not name or not name.strip():
                        logger.warning("Skipping tool call with empty name")
                        continue
                    name = canonicalize_tool_name(name, valid_names)
                    tool_call = ToolCall.from_openai(tc)
                    tool_call = ToolCall(
                        id=tool_call.id,
                        name=name,
                        arguments=normalize_tool_arguments(name, tool_call.arguments, valid_params=(valid_params_by_tool or {}).get(name)),
                        source=tool_call.source,
                    )
                    tool_calls.append(tool_call)
                except (KeyError, TypeError, json.JSONDecodeError) as e:
                    logger.warning(f"ToolPipeline: Failed to parse native tool call: {e}")
                    continue
            
            if tool_calls:
                logger.debug(f"ToolPipeline: Parsed {len(tool_calls)} native tool calls")
                return tool_calls

        content = message.get("content", "") or ""
        reasoning = message.get("reasoning", "") or ""
        
        has_dsml = bool(reasoning) and (
            ("｜" in reasoning and "tool_calls" in reasoning)
            or ("‹" in reasoning and "›" in reasoning and "invoke" in reasoning)
            or ("<tool_call>" in reasoning)
        )
        parse_target = reasoning if has_dsml else content

        if parse_target:
            try:
                from proxy.tool_parsing import parse_all_tools_from_content
                parsed = parse_all_tools_from_content(parse_target)
                
                for tc_dict in parsed:
                    name = tc_dict.get("name")
                    if name and name.strip():
                        name = canonicalize_tool_name(name, valid_names)
                        args = tc_dict.get("arguments", {})
                        args = normalize_tool_arguments(name, args, valid_params=(valid_params_by_tool or {}).get(name))
                        tool_call = ToolCall.from_parsed(name, args)
                        tool_calls.append(tool_call)
                
                if tool_calls:
                    logger.debug(f"ToolPipeline: Parsed {len(tool_calls)} tool calls from content")
                    
            except Exception as e:
                logger.warning(f"ToolPipeline: Failed to parse tool blocks from content: {e}")

        return tool_calls

    def _classify(self, tool_calls: List[ToolCall]) -> Tuple[List[ToolCall], List[ToolCall]]:
        """Split into memory calls vs passthrough.
        
        Uses ToolRegistry.get_memory_tool_names() to determine which tools
        are intercepted and executed locally vs passed through to the client.
        """
        memory_names = self.registry.get_memory_tool_names()
        memory = [tc for tc in tool_calls if tc.name in memory_names]
        passthrough = [tc for tc in tool_calls if tc.name not in memory_names]
        
        if passthrough:
            logger.debug(
                f"ToolPipeline: {len(passthrough)} passthrough tool(s): "
                f"{[tc.name for tc in passthrough]}"
            )
        
        return memory, passthrough

    def _execute_via_loop(self, response_data: dict, memory_calls: List[ToolCall], ctx: ToolContext, original_tool_calls_list: list = None, valid_params_by_tool=None) -> Any:
        """Execute memory tools by delegating to run_memory_tool_loop().
        
        In Phase 6, we DON'T rewrite the loop - we wrap it. The loop function
        signature is unchanged, but we pass a MessageBuffer for working_messages
        instead of a raw list.
        
        However, since run_memory_tool_loop expects a list (not MessageBuffer),
        in Phase 6 we just call it with the existing params. Phase 7 will 
        refactor the loop to accept Agent/MessageBuffer.
        
        This method mirrors handle_tool_interception() lines 789-850.
        """
        from proxy.chat_history import chat_histories, chat_histories_lock
        from proxy.state import _HTTP_SESSION
        from proxy.tool_parsing import make_tool_calls, extract_file_path, normalize_tool_arguments
        from memory.tool_loop import run_memory_tool_loop

        try:
            # Build memory-only tool_calls_list for the assistant message in working_messages.
            # original_tool_calls_list (all tools) is passed to run_memory_tool_loop so it
            # can detect non-memory tools and return them in the result without needing
            # to re-parse from the response data.
            tool_calls_list = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in memory_calls
            ]

            from proxy.agent.message_buffer import MessageBuffer
            working_buffer = MessageBuffer(messages=[dict(msg) for msg in ctx.llama_body.get("messages", [])])
            
            original_ids = None
            message = response_data["choices"][0].get("message", {})
            native_tool_calls = message.get("tool_calls")
            if isinstance(native_tool_calls, list) and native_tool_calls:
                original_ids = [tc.get("id") for tc in native_tool_calls]

            content = message.get("content", "") or ""
            reasoning = message.get("reasoning", "") or ""
            clean_content = content.strip() or None
            
            has_dsml = bool(reasoning) and (
                ("｜" in reasoning and "tool_calls" in reasoning)
                or ("‹" in reasoning and "›" in reasoning and "invoke" in reasoning)
                or ("<tool_call>" in reasoning)
            )

            dsml_block = None
            if has_dsml:
                dsml_match = re.search(
                    r"<｜DSML｜tool_calls>.*?</｜DSML｜tool_calls>", reasoning, re.DOTALL
                )

                if dsml_match:
                    dsml_block = dsml_match.group(0)
            
            if has_dsml and dsml_block:
                remaining_reasoning = reasoning.replace(dsml_block, "").strip()
                if remaining_reasoning:
                    clean_content = remaining_reasoning

            if clean_content and "tool_call" in clean_content:
                clean_content = re.sub(
                    r'<tool_call>.*?</tool_call>',
                    '', clean_content, flags=re.DOTALL
                )
                clean_content = clean_content.strip() or None

            assistant_msg = {
                "role": "assistant",
                "content": clean_content,
            }
            assistant_msg["tool_calls"] = make_tool_calls(tool_calls_list, existing_ids=original_ids, valid_params_by_tool=valid_params_by_tool)

            # Attach provenance metadata for debug API
            try:
                from proxy.types import MessageProvenance
                assistant_msg["_provenance"] = MessageProvenance(
                    agent_id=getattr(ctx, 'session_id', 'unknown'),
                    message_type="assistant",
                    stage="pipeline",
                ).__dict__
            except Exception as e:
                logger.debug(f"ToolPipeline: MessageProvenance attachment failed (non-fatal): {e}")

            working_buffer.append(assistant_msg)

            execute_fn = ctx.execute_fn
            if execute_fn is None:
                from memory.tools import execute_memory_tool
                execute_fn = execute_memory_tool

            # Wrap execute_fn to intercept cached reads
            if self._cached_reads:
                original_execute_fn = execute_fn
                def wrapped_execute_fn(name, arguments):
                    if name == "read":
                        file_path = extract_file_path(arguments)
                        if file_path:
                            import os
                            normalized = os.path.normpath(os.path.realpath(file_path))
                            if normalized in self._cached_reads:
                                cached_content = self._cached_reads.pop(normalized)
                                logger.info(f"ToolPipeline: Serving cached read for {normalized}")
                                return {"content": cached_content, "cached": True}
                    return original_execute_fn(name, arguments)
                execute_fn = wrapped_execute_fn

            # Wrap execute_fn to return synthetic errors for guard-blocked calls.
            # run_memory_tool_loop injects session_id/client_hint into args before
            # invoking execute_fn, so we strip those keys via _stable_args_key.
            # SECURITY: also re-normalize args via the same valid_params_by_tool
            # the producer used (see _parse() at line 304-331). The producer keys
            # _blocked_errors with NORMALIZED args; run_memory_tool_loop passes
            # RAW args (it only json.loads string args — no normalization), so
            # for clients that send camelCase (opencode) or other degraded
            # forms (kilocode), the consumer must normalize before keying or
            # the lookup misses and the guard-blocked tool executes anyway.
            if self._blocked_errors:
                original_execute_fn = execute_fn
                blocked_keys = dict(self._blocked_errors)
                vpbt_for_blocked = valid_params_by_tool
                def blocked_wrapped_execute_fn(name, arguments):
                    if name in (vpbt_for_blocked or {}):
                        arguments = normalize_tool_arguments(
                            name, arguments, valid_params=vpbt_for_blocked[name],
                        )
                    key = _stable_args_key(name, arguments)
                    if key in blocked_keys:
                        return {"error": blocked_keys[key]}
                    return original_execute_fn(name, arguments)
                execute_fn = blocked_wrapped_execute_fn

            parse_tools_fn = ctx.parse_tools_fn
            if parse_tools_fn is None:
                if ctx.tool_parser:
                    parse_tools_fn = ctx.tool_parser.parse
                else:
                    from proxy.tool_parsing import parse_all_tools_from_content
                    parse_tools_fn = parse_all_tools_from_content

            result = run_memory_tool_loop(
                tool_calls_list=original_tool_calls_list,
                working_messages=working_buffer.to_list(),
                tools=ctx.llama_body.get("tools", []),
                model=ctx.llama_body.get("model", ctx.body.get("model", "default")),
                session_id=ctx.session_id,
                client_hint=ctx.client_hint,
                llama_url=ctx.upstream_url,
                body=ctx.body,
                llama_body=ctx.llama_body,
                chat_histories=chat_histories,
                chat_histories_lock=chat_histories_lock,
                execute_fn=execute_fn,
                make_tool_calls_fn=functools.partial(make_tool_calls, valid_params_by_tool=valid_params_by_tool),
                existing_tool_ids=original_ids,
                parse_tools_fn=parse_tools_fn,
                http_session=_HTTP_SESSION,
                max_iterations=ctx.max_iterations,
                max_total_time=ctx.max_total_time,
                max_result_size=ctx.max_result_size,
                max_calls_per_iteration=ctx.max_calls_per_iteration,
                auth_headers=ctx.auth_headers,
                model_cfg=ctx.model_cfg,
            )

            return result

        except Exception as e:
            logger.error(
                f"ToolPipeline: Error in _execute_via_loop for session={ctx.session_id}: {e}",
                exc_info=True
            )
            raise

    def _format(self, data: dict, result: Any, ctx: ToolContext) -> dict:
        from proxy.chat_history import chat_histories, chat_histories_lock
        from proxy.session import touch_session

        if getattr(result, "needs_closure", False) is True:
            fallback_content = self._emergency_closure_fallback(result)
            data["choices"][0]["message"].pop("tool_calls", None)
            data["choices"][0]["message"]["content"] = fallback_content
            data["choices"][0]["finish_reason"] = "stop"
            try:
                object.__setattr__(result, "tool_calls", None)
                object.__setattr__(result, "content", fallback_content)
                object.__setattr__(result, "needs_closure", False)
                object.__setattr__(result, "closure_source", "fallback")
            except Exception:
                pass
            self._sync_history(ctx, result)
            return data

        message = data["choices"][0]["message"]

        if hasattr(result, "tool_calls") and result.tool_calls:
            message["tool_calls"] = result.tool_calls
        else:
            message.pop("tool_calls", None)

        if hasattr(result, "content") and result.content is not None:
            content_value = result.content
            if not content_value or not str(content_value).strip():
                content_value = self._emergency_closure_fallback(result)
                try:
                    object.__setattr__(result, "content", content_value)
                    object.__setattr__(result, "closure_source", "fallback")
                except Exception:
                    pass
            message["content"] = content_value

        if hasattr(result, "finish_reason"):
            if message.get("tool_calls"):
                if result.finish_reason is not None and result.finish_reason not in TRUNCATION_REASONS:
                    data["choices"][0]["finish_reason"] = "tool_calls"
            else:
                data["choices"][0]["finish_reason"] = result.finish_reason

        self._sync_history(ctx, result)
        return data

    @staticmethod
    def _emergency_closure_fallback(result: Any) -> str:
        from memory.tool_loop import render_deterministic_fallback
        try:
            content = render_deterministic_fallback(
                getattr(result, "execution_records", []) or [],
                getattr(result, "termination_reason", "") or "",
            )
        except Exception:
            content = (
                "FAILED The tool loop terminated before a completion "
                "report could be rendered."
            )
        if not content.startswith(("COMPLETE", "PARTIAL", "FAILED")):
            content = "FAILED " + content
        return content

    def _sync_history(self, ctx: ToolContext, result: Any) -> None:
        if not ctx.session_id:
            return
        if not hasattr(result, "messages_for_history") or not result.messages_for_history:
            return
        import copy as _copy
        from proxy.chat_history import chat_histories, chat_histories_lock
        from proxy.session import touch_session
        try:
            sanitized = [
                _copy.deepcopy(m) for m in result.messages_for_history
                if isinstance(m, dict)
            ]
            _assert_no_dangling_tool_pairs(sanitized)
            with chat_histories_lock:
                if ctx.session_id not in chat_histories:
                    chat_histories[ctx.session_id] = []
                chat_histories[ctx.session_id].extend(sanitized)
                touch_session(ctx.session_id)
                logger.debug(
                    f"Synced {len(sanitized)} messages to chat_histories"
                )
        except Exception as e:
            logger.warning(f"Failed to sync messages to history: {e}")


    def _write_through_cache(self, tool_calls, session_id, is_subagent=False, client_hint: str = ""):
        """Update VFS cache for write/edit/bash tool calls BEFORE passing to client.
        
        Processes bash FIRST (invalidate_all), then write, then edit.
        This prevents bash invalidation from wiping writes in the same batch.
        """
        from proxy.vfs import _vfs_active_for_context, _cache_file_write_through, _cache_file_edit_through
        import proxy.state as state
        import os
        from proxy.tool_parsing import extract_file_path
        
        if not _vfs_active_for_context(is_subagent):
            return
        
        workspace_id = self._resolve_workspace_id(session_id, client_hint)
        
        bash_calls = []
        write_calls = []
        edit_calls = []
        
        for tc in tool_calls:
            name = tc.name if hasattr(tc, 'name') else tc.get("name", "")
            if name == "bash":
                bash_calls.append(tc)
            elif name == "write":
                write_calls.append(tc)
            elif name == "edit":
                edit_calls.append(tc)
        
        for tc in bash_calls:
            with state.vfs_lock:
                cache = state.vfs_caches.get(workspace_id)
                if cache:
                    cache.invalidate_all()
        
        for tc in write_calls:
            args = tc.arguments if hasattr(tc, 'arguments') else tc.get("arguments", {})
            file_path = extract_file_path(args)
            if not file_path:
                continue
            
            from proxy.tool_pipeline.tool_limits import FILE_BLOCKLIST_PATTERNS
            normalized_lower = file_path.lower()
            if any(p in normalized_lower for p in FILE_BLOCKLIST_PATTERNS):
                continue
            
            normalized = os.path.normpath(os.path.realpath(file_path))
            content = args.get("content", "")
            if content:
                tool_call_id = tc.id if hasattr(tc, 'id') else tc.get("id")
                _cache_file_write_through(normalized, content, workspace_id, session_id, tool_call_id)
        
        for tc in edit_calls:
            args = tc.arguments if hasattr(tc, 'arguments') else tc.get("arguments", {})
            file_path = extract_file_path(args)
            if not file_path:
                continue
            
            from proxy.tool_pipeline.tool_limits import FILE_BLOCKLIST_PATTERNS
            normalized_lower = file_path.lower()
            if any(p in normalized_lower for p in FILE_BLOCKLIST_PATTERNS):
                continue
            
            normalized = os.path.normpath(os.path.realpath(file_path))
            old_str = args.get("old_string") or args.get("old_str") or args.get("oldString") or ""
            new_str = args.get("new_string") or args.get("new_str") or args.get("newString") or ""
            replace_all = args.get("replaceAll", False) or args.get("replace_all", False)
            if old_str and new_str:
                tool_call_id = tc.id if hasattr(tc, 'id') else tc.get("id")
                _cache_file_edit_through(normalized, old_str, new_str, workspace_id, session_id, replace_all, tool_call_id)

    def _check_vfs_read_cache(self, tool_call, session_id, is_subagent=False, client_hint: str = ""):
        """Check VFS cache for read tool calls. Returns cached content or None.
        
        Returns None on: VFS disabled, not a read tool, blocklisted path,
        partial read (offset/limit), cache miss, or expired entry.
        """
        from proxy.vfs import _vfs_active_for_context, _is_expired
        import proxy.state as state
        import os
        from proxy.tool_parsing import extract_file_path
        
        if not _vfs_active_for_context(is_subagent):
            return None
        
        name = tool_call.name if hasattr(tool_call, 'name') else tool_call.get("name", "")
        if name != "read":
            return None
        
        args = tool_call.arguments if hasattr(tool_call, 'arguments') else tool_call.get("arguments", {})
        file_path = extract_file_path(args)
        if not file_path:
            return None
        
        from proxy.tool_pipeline.tool_limits import FILE_BLOCKLIST_PATTERNS
        normalized_lower = file_path.lower()
        if any(p in normalized_lower for p in FILE_BLOCKLIST_PATTERNS):
            return None
        
        offset = args.get("offset")
        limit = args.get("limit")
        if offset is not None or limit is not None:
            return None
        
        normalized = os.path.normpath(os.path.realpath(file_path))
        workspace_id = self._resolve_workspace_id(session_id, client_hint)
        
        entry_content = None
        entry_verified = True
        with state.vfs_lock:
            cache = state.vfs_caches.get(workspace_id)
            if cache is None:
                state.vfs_cache_misses += 1
                return None
            entry = cache.get(normalized)
            if entry and not _is_expired(entry, state.VFS_TTL_SECONDS):
                state.vfs_cache_hits += 1
                cache.record_reader(normalized, session_id)
                entry_content = entry.content
                entry_verified = entry.verified
            else:
                state.vfs_cache_misses += 1
                return None
        
        # Build content with warnings (outside lock)
        content = entry_content
        if not entry_verified:
            content += "\n\n[VFS WARNING: This content was written by another agent but the write has not been confirmed. It may not match the file on disk.]"
        
        # Phase 3: Conflict detection — warn about recent writes by other agents
        from proxy.vfs import detect_write_conflicts
        conflicts = detect_write_conflicts(normalized, workspace_id, session_id)
        if conflicts:
            state.vfs_write_conflicts += len(conflicts)
            MAX_CONFLICT_LINES = 5
            conflicts.sort(key=lambda c: c.timestamp, reverse=True)
            conflict_lines = []
            for c in conflicts[:MAX_CONFLICT_LINES]:
                age = time.time() - c.timestamp
                age_str = f"{int(age)}s" if age < 60 else f"{int(age/60)}m"
                conflict_lines.append(
                    f"  - {c.operation} by {c.agent_id[:12]} ({age_str} ago, hash:{c.new_hash[:8]})"
                )
            if len(conflicts) > MAX_CONFLICT_LINES:
                conflict_lines.append(f"  ... and {len(conflicts) - MAX_CONFLICT_LINES} more")
            content += (
                f"\n\n[VFS CONFLICT WARNING: This file was modified by another agent "
                f"within the last 2 minutes:\n"
                + "\n".join(conflict_lines)
                + "\nConsider re-reading from disk to get the latest version.]"
            )
        
        return content

    def _resolve_workspace_id(self, session_id, client_hint: str = ""):
        """Resolve workspace ID from session_id.

        v5: workspaces are global. Returns the session's active workspace
        name (bare), inheriting from the parent agent tree if needed.
        """
        from proxy.workspace_state import resolve_inherited_workspace
        workspace_id = resolve_inherited_workspace(session_id, client_hint=client_hint)

        return workspace_id


def _assert_no_dangling_tool_pairs(messages: list) -> None:
    valid_ids = set()
    present_ids = set()
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            for tc in m["tool_calls"]:
                if isinstance(tc, dict) and tc.get("id"):
                    valid_ids.add(tc["id"])
        elif role == "tool" and m.get("tool_call_id"):
            present_ids.add(m["tool_call_id"])
    dangling = valid_ids - present_ids
    if dangling:
        raise RuntimeError(
            f"chat history contains dangling assistant tool_call(s) "
            f"with no paired tool result: {sorted(dangling)}"
        )
