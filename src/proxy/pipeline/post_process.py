"""Post-processing stage for pipeline responses.

Extracted from stages.py for modularity. Handles tool interception,
message persistence, chat history sync, and skill post-processing.
"""

import copy
import hashlib
import json
import logging
import re

from proxy.feature_flags import CACHE_METRICS_ENABLED
from proxy.openai_compat import (
    build_upstream_protocol_error,
    chat_completion_shape_error,
)
from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.tool_interception import handle_tool_interception
from proxy.tool_parsing.normalize import extract_file_path
from proxy.utils import _now_iso_sqlite

logger = logging.getLogger("proxy.pipeline.stages")


class PostProcessStage:
    """Tool interception, requery loop, sync.

    Decomposed into private methods, each handling one concern:
    1. _extract_minimax_reasoning — extract <think> blocks from MiniMax responses
    2. _scan_write_results — confirm/rollback VFS snapshots for write/edit results
    3. _sync_chat_history — sync messages to ChatHistoryCache
    4. _persist_messages — store messages to SQLite (touch_session, store_messages_batch)
    5. _populate_vfs_cache — cache file contents from read tool results
    6. _handle_tool_interception — parse and execute tool calls via ToolPipeline
    7. _assign_model_name — set data["model"]

    After the 8 sub-methods complete, three skill-related ops run (spec §7.5):
    8. _apply_skill_response_transform — call skill.post_process()
    9. _inject_task_canaries — insert skill/workspace/bare canary into task tool calls
    10. _finalize_skill_lifecycle — classify the turn disposition and clear mode if terminal

    Lifecycle finalization runs in a `finally` block so it ALWAYS runs on
    every exit path (normal response, empty choices, malformed response,
    tool interception early return, synchronization error, StageAbort,
    skill transform exception, upstream error passed to post-processing).
    """

    critical = False

    def execute(self, ctx):
        route = getattr(ctx, "model_route", None)
        data = getattr(ctx, "response_data", None)

        # Consume the expected_mode snapshot that ContextOptimizeStage
        # captured via agent.snapshot_mode() IMMEDIATELY AFTER apply_mode
        # succeeded — i.e. BEFORE the upstream call. This is the CAS
        # expected_mode: comparing against the pre-upstream snapshot lets
        # finalize_skill_mode detect a concurrent replacement (Request B
        # installing a newer mode while Request A's completion thread was
        # finishing upstream) and no-op the clear so B's mode survives.
        # Re-snapshotting here (after the upstream call) would read B's
        # newer mode and CAS-clear it — the exact clobber the contract in
        # proxy/mode.py:353-370 forbids.
        #
        # Fall back to None (not a live agent / preflight short-circuit /
        # cascade fallback): finalize_skill_mode treats None as "preserve
        # the mode" unchanged.
        #
        # hasattr(type(ctx), ...) — NOT hasattr(ctx, ...) and NOT getattr(ctx,
        # ..., None) — because on a MagicMock ctx the latter two auto-create a
        # truthy sentinel attribute that would poison the identity CAS in
        # clear_active_mode. Checking the CLASS bypasses MagicMock instance
        # attribute auto-vivification: MagicMock's class has no such attribute,
        # while RequestContext declares it as a dataclass field.
        expected_mode = (
            ctx.mode_snapshot_pre_upstream
            if hasattr(type(ctx), "mode_snapshot_pre_upstream")
            else None
        )

        if data is None:
            self._finalize_skill_lifecycle_with_flag(ctx, expected_mode)
            return None

        try:
            # R2 #10 (F8 duplicate classifier): consolidate the two shape
            # checks. The single-source semantic validator is
            # `chat_completion_shape_error` — it returns a stable reason
            # for malformed shapes and returns None for both valid
            # completions AND provider-error envelopes (the dict-error
            # signal detected below). For everything malformed we replace
            # with the canonical upstream_protocol_error envelope and
            # abort 502.
            shape_error = chat_completion_shape_error(data)
            if shape_error is not None:
                logger.error(
                    "Backend returned malformed chat completion: %s",
                    shape_error,
                    extra={
                        "request_id": getattr(ctx, "request_id", "?"),
                        "session_id": getattr(ctx, "session_id", "?"),
                    },
                )
                ctx.response_data = build_upstream_protocol_error()
                raise StageAbort((ctx.response_data, 502))

            # Dict-error short-circuit (different signal from the shape
            # check above): the upstream returned a provider error envelope
            # (object-valued error, no choices). Preserve the ORIGINAL body
            # verbatim so the client sees the upstream's actual error
            # message instead of the canonical upstream_protocol_error.
            # chat_completion_shape_error above returns None for this shape,
            # so it falls through here. No `isinstance(data, dict)` guard
            # needed — the shape validator already guaranteed `data` is a
            # dict when it returned None (non-dict triggers
            # "response is not an object" above).
            if (
                isinstance(data.get("error"), dict)
                and not data.get("choices")
            ):
                raise StageAbort((data, 502))

            methods = [
                self._extract_minimax_reasoning,
                self._scan_write_results,
                self._sync_chat_history,
                self._persist_messages,
                self._populate_vfs_cache,
                self._capture_cache_metrics,
                self._handle_tool_interception,
                self._assign_model_name,
            ]
            for method in methods:
                try:
                    result = method(ctx)
                    if result is not None and method == self._handle_tool_interception:
                        # Tool interception early-return: still need to finalize
                        # the skill lifecycle in the finally block.
                        self._apply_skill_response_transform(ctx)
                        self._inject_task_canaries(ctx)
                        return result
                except StageAbort:
                    # Swarm-policy abort or other fail-closed abort from any
                    # sub-method must propagate so the Flask boundary emits
                    # the canonical error envelope with the correct status.
                    # The lifecycle-finalization `finally:` block at the end
                    # of execute() still runs (it's not part of this try/except).
                    raise
                except Exception as e:
                    logger.error("PostProcessStage.%s failed: %s", method.__name__, e, exc_info=True)

            try:
                from memory.sync import submit_sync

                finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
                # A content_filter turn is still a COMPLETE turn for memory
                # extraction purposes — the prior code returned "stop" here in
                # every case (tool_loop clobbered finish_reason), so sync always
                # fired. Preserve that intent now that content_filter can flow
                # on the content-only memory resolution path.
                is_final_response = finish_reason in ("stop", "length", "content_filter")
                ignore_subagents = (
                    route.model_cfg and getattr(route.model_cfg, "ignore_subagents", False)
                )
                model_cfg = route.model_cfg if route else None
                memory_enabled = model_cfg and getattr(model_cfg, "memory_enabled", True)

                if is_final_response and not (ctx.is_subagent and ignore_subagents) and ctx.session_id:
                    if memory_enabled:
                        future = submit_sync(ctx.session_id)
                        if future is not None:
                            logger.info(
                                "Submitted sync for session %s",
                                ctx.session_id,
                                extra={"request_id": getattr(ctx, "request_id", "?")} if hasattr(ctx, "request_id") else {},
                            )
                        else:
                            logger.info(
                                "Sync deferred (debounced) for session %s",
                                ctx.session_id,
                                extra={"request_id": getattr(ctx, "request_id", "?")} if hasattr(ctx, "request_id") else {},
                            )
            except ImportError:
                logger.exception("PostProcessStage: Failed to import submit_sync — memory sync DISABLED")
            except Exception:
                logger.warning(
                    "submit_sync failed (non-fatal) for session=%s",
                    ctx.session_id,
                    exc_info=True,
                )

            self._apply_skill_response_transform(ctx)
            self._inject_task_canaries(ctx)

            data = ctx.response_data
            return data
        finally:
            self._finalize_skill_lifecycle_with_flag(ctx, expected_mode)

    def _finalize_skill_lifecycle_with_flag(self, ctx, expected_mode):
        """Single entry point for the finally block + early-return cleanup.

        Calls proxy.mode.finalize_skill_mode (Agent 1's helper) and sets
        ctx.skill_lifecycle_finalized = True. Wraps the finalize call in
        try/except so a misbehaving skill lifecycle never raises out of
        the finally block (which would mask the original return value).
        """
        try:
            self._finalize_skill_lifecycle(ctx, expected_mode)
        except Exception as e:
            logger.warning("PostProcessStage: finalize_skill_lifecycle failed: %s", e)
        finally:
            ctx.skill_lifecycle_finalized = True

    def _extract_minimax_reasoning(self, ctx):
        """Extract <think> blocks from MiniMax model responses."""
        route = ctx.model_route
        data = ctx.response_data

        if not (route.model_cfg and "minimax" in (route.model_cfg.backend_model or "").lower()):
            return

        from proxy.utils import extract_text_content

        choices = data.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return
        
        msg = choices[0].get("message", {})
        content = msg.get("content", "") or ""
        content = extract_text_content(content)
        think_match = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
        if think_match:
            msg["reasoning_content"] = think_match.group(1).strip()
            msg["content"] = think_match.group(2).strip()
            try:
                from memory.sync import _cache_reasoning
                from proxy.chat_history import chat_histories, chat_histories_lock
                with chat_histories_lock:
                    msg_count = len(chat_histories.get(ctx.session_id, []))
                _cache_reasoning(ctx.session_id, msg_count, msg["reasoning_content"])
            except Exception:
                pass

    def _sync_chat_history(self, ctx):
        """Sync response messages to ChatHistoryCache."""
        route = ctx.model_route

        if not (route.is_remote and ctx.session_id):
            return

        from proxy.chat_history import chat_histories, chat_histories_lock

        original_messages = (
            ctx.messages_original
            if ctx.messages_original is not None
            else ctx.body.get("messages", [])
        )
        
        with chat_histories_lock:
            if ctx.session_id in chat_histories:
                existing = chat_histories[ctx.session_id]
                seen = set()
                for em in existing:
                    content = em.get("content") or ""
                    if isinstance(content, list):
                        content = json.dumps(content, sort_keys=True)
                    fp = (em.get("role"), hashlib.sha256(content.encode()).digest())
                    seen.add(fp)
                new_msgs = []
                for m in original_messages:
                    content = m.get("content") or ""
                    if isinstance(content, list):
                        content = json.dumps(content, sort_keys=True)
                    fp = (m.get("role"), hashlib.sha256(content.encode()).digest())
                    if fp not in seen:
                        new_msgs.append(m)
                chat_histories[ctx.session_id] = existing + new_msgs
            else:
                chat_histories[ctx.session_id] = copy.deepcopy(original_messages)

    def _persist_messages(self, ctx):
        """Store messages to SQLite for session history."""
        route = ctx.model_route

        if not (route.is_remote and ctx.session_id):
            return

        from proxy.session import touch_session
        touch_session(ctx.session_id)
        from memory.db import touch_session as db_touch_session
        db_touch_session(ctx.session_id)

        try:
            from memory.db import store_messages_batch, immediate_transaction
            import time

            original_messages = (
                ctx.messages_original
                if ctx.messages_original is not None
                else ctx.body.get("messages", [])
            )
            if original_messages:
                with immediate_transaction() as conn:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(message_index), -1) AS latest_idx FROM messages WHERE session_id = ?",
                        (ctx.session_id,)
                    ).fetchone()
                    last_stored = row["latest_idx"] if row else -1
                    new_start = last_stored + 1
                    if new_start < len(original_messages):
                        new_msgs = original_messages[new_start:]
                        store_messages_batch(ctx.session_id, new_msgs, new_start, conn=conn)
                        new_exchanges = sum(
                            1 for m in original_messages[new_start:] if m.get("role") == "user"
                        )
                        if new_exchanges > 0:
                            try:
                                prev_row = conn.execute(
                                    "SELECT COALESCE(exchange_count, 0) FROM conversation_summaries "
                                    "WHERE session_id = ?",
                                    (ctx.session_id,)
                                ).fetchone()
                                prev_count = prev_row[0] if prev_row else 0
                                new_user_indices = [
                                    new_start + i for i, m in enumerate(new_msgs) if m.get("role") == "user"
                                ]
                                last_user_idx = max(new_user_indices) if new_user_indices else new_start
                                now = _now_iso_sqlite()
                                conn.execute(
                                    """INSERT INTO conversation_summaries
                                       (session_id, exchange_count, last_exchange_start,
                                        created_at, updated_at)
                                       VALUES (?, ?, ?, ?, ?)
                                       ON CONFLICT(session_id) DO UPDATE SET
                                       exchange_count = excluded.exchange_count,
                                       last_exchange_start = excluded.last_exchange_start,
                                       updated_at = excluded.updated_at""",
                                    (ctx.session_id, prev_count + new_exchanges,
                                     last_user_idx, now, now),
                                )
                            except Exception:
                                logger.warning(
                                    "exchange_count INSERT failed (non-fatal) for session=%s",
                                    ctx.session_id, exc_info=True,
                                )
                        v2_boundary_idx = getattr(ctx, "v2_boundary_idx", None)
                        if v2_boundary_idx is not None:
                            try:
                                # raw_start_message_index is a 0-indexed position in the
                                # CONVERSATION (non-system message list), NOT messages.message_index
                                # (which includes system at index 0). Off-by-one trap if system
                                # messages exist.
                                now = _now_iso_sqlite()
                                conn.execute(
                                    """INSERT INTO conversation_summaries
                                       (session_id, raw_start_message_index,
                                        created_at, updated_at)
                                       VALUES (?, ?, ?, ?)
                                       ON CONFLICT(session_id) DO UPDATE SET
                                       raw_start_message_index = excluded.raw_start_message_index,
                                       updated_at = excluded.updated_at""",
                                    (ctx.session_id, v2_boundary_idx, now, now),
                                )
                            except Exception:
                                logger.warning(
                                    "v2_boundary_idx persist failed (non-fatal) "
                                    "for session=%s boundary=%s",
                                    ctx.session_id, v2_boundary_idx, exc_info=True,
                                )
                    else:
                        logger.debug(
                            "No new messages to persist: new_start=%d total_messages=%d session_id=%s",
                            new_start, len(original_messages), ctx.session_id,
                        )
        except Exception:
            logger.warning(
                "store_messages_batch failed (non-fatal) for session=%s",
                ctx.session_id, exc_info=True,
            )

    def _capture_cache_metrics(self, ctx):
        """Record prompt cache metrics from upstream response."""
        if not CACHE_METRICS_ENABLED:
            return

        data = getattr(ctx, "response_data", None)
        if data is None or not isinstance(data, dict):
            return

        route = getattr(ctx, "model_route", None)
        if route is None:
            return

        tokens = _extract_cache_tokens(data)
        cache_read = tokens.get("cache_read")
        cache_write = tokens.get("cache_write")
        no_cache = tokens.get("no_cache")
        total = tokens.get("total")

        backend_model = None
        route = getattr(ctx, "model_route", None)
        if route and getattr(route, "model_cfg", None):
            backend_model = getattr(route.model_cfg, "backend_model", None)
        model_key = getattr(ctx, "model_name", "") or ""
        provider = _detect_provider(backend_model, model_key)

        model = model_key or "unknown"

        from proxy.metrics import record_prompt_cache_metrics
        record_prompt_cache_metrics(
            model=model,
            provider=provider,
            cache_read_tokens=cache_read,
            no_cache_tokens=no_cache,
        )

        # Skip INFO log for non-reporting backends (cache_read is None)
        if cache_read is not None:
            req_id = getattr(ctx, "request_id", "?")
            logger.info(
                "[%s] cache_metrics: model=%s provider=%s total=%s cache_read=%s cache_write=%s no_cache=%s",
                req_id, model, provider,
                total if total is not None else "null",
                cache_read,
                cache_write if cache_write is not None else "null",
                no_cache if no_cache is not None else "null",
            )

    def _handle_tool_interception(self, ctx):
        """Parse tool calls and execute via ToolPipeline (black-box sub-pipeline).

        Modifies ctx.response_data in place. Returns None to continue pipeline.
        """
        route = ctx.model_route
        upstream = ctx.upstream
        llama_body = getattr(ctx, "llama_body", ctx.body)

        memory_enabled = (
            getattr(route.model_cfg, "memory_enabled", True) if route.model_cfg else True
        )
        if not memory_enabled:
            return

        features = getattr(ctx, "_features", None)
        max_iter = features.max_tool_iterations if features else 5
        max_time = float(features.tool_loop_timeout) if features else 90.0

        ctx.response_data = handle_tool_interception(
            ctx.response_data,
            llama_body,
            ctx.session_id,
            upstream.url,
            route.tool_parser,
            upstream.headers,
            upstream.original_model,
            route.model_cfg,
            max_iterations=max_iter,
            max_total_time=max_time,
            is_subagent=getattr(ctx, "is_subagent", False),
            subagent_ticket_mode=getattr(ctx, "subagent_ticket_mode", False),
            unattended=getattr(ctx, "unattended", False),
            client_hint=getattr(ctx, "client_hint", ""),
        )

    def _assign_model_name(self, ctx):
        """Set model name in response data."""
        upstream = getattr(ctx, "upstream", None)
        if upstream is not None:
            ctx.response_data["model"] = upstream.original_model

    def _apply_skill_response_transform(self, ctx):
        """First post-process op (spec §7.5): skill.post_process() callback."""
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            from proxy.mode import Mode, apply_skill_response_transform

            agent = agent_registry.get_or_load(ctx.session_id)
            if agent is None:
                logger.warning(
                    "PostProcessStage: agent registry returned None for session %s",
                    ctx.session_id,
                )
                return
            ctx.response_data = apply_skill_response_transform(
                ctx=ctx,
                agent=agent,
                response_data=ctx.response_data,
            )
        except Exception as e:
            logger.warning("PostProcessStage: apply_skill_response_transform failed: %s", e)

    def _inject_task_canaries(self, ctx):
        """Second post-process op (spec §7.5): insert canary into task tool calls.

        Propagates :class:`StageAbort` from `inject_task_canaries` (swarm
        rejection paths raise StageAbort — they MUST reach the Flask shell
        so the canonical envelope is emitted). Legacy unexpected exceptions
        are logged + swallowed per existing behaviour.
        """
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            from proxy.mode import Mode, inject_task_canaries

            agent = agent_registry.get_or_load(ctx.session_id)
            if agent is None:
                logger.warning(
                    "PostProcessStage: agent registry returned None for session %s",
                    ctx.session_id,
                )
                return
            mode = getattr(ctx, "_mode", Mode.DEFAULT)
            ctx.response_data = inject_task_canaries(
                ctx=ctx,
                agent=agent,
                response_data=ctx.response_data,
                mode=mode,
            )
        except StageAbort:
            # Swarm-policy abort or other fail-closed abort — propagate.
            raise
        except Exception as e:
            logger.warning("PostProcessStage: inject_task_canaries failed: %s", e)

    def _finalize_skill_lifecycle(self, ctx, expected_mode):
        """Third post-process op (spec §7.5): classify + compare-and-clear.

        Runs in execute()'s `finally` block so it ALWAYS runs on every exit
        path (normal response, empty choices, malformed response, tool
        interception early return, StageAbort, sync error, transform error).

        Delegates to proxy.mode.finalize_skill_mode (Agent 1's helper).
        The CAS in agent.clear_active_mode(expected_mode=...) prevents a
        concurrent mode replacement from being clobbered.
        """
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            from proxy.mode import finalize_skill_mode

            agent = agent_registry.get_or_load(ctx.session_id)
            if agent is None:
                logger.warning(
                    "PostProcessStage: agent registry returned None for session %s",
                    ctx.session_id,
                )
                return
            finalize_skill_mode(
                ctx=ctx,
                agent=agent,
                expected_mode=expected_mode,
            )
        except Exception as e:
            logger.warning("PostProcessStage: finalize_skill_lifecycle failed: %s", e)

    def _scan_write_results(self, ctx):
        """Scan incoming messages for write/edit tool results and confirm or rollback.
        
        MUST run BEFORE _sync_chat_history() so the historical message filter
        can distinguish new messages from historical ones (REG-004 fix).
        """
        import proxy.state as state
        from proxy.vfs import (
            _vfs_active_for_context, _looks_like_failure,
            _rollback_snapshot, _rollback_stale_snapshots,
        )
        
        if not state.VFS_ENABLED:
            return
        
        workspace_id = self._resolve_vfs_workspace(ctx)
        _rollback_stale_snapshots(workspace_id)
        
        if not _vfs_active_for_context(getattr(ctx, 'is_subagent', False)):
            return
        
        messages = ctx.messages_original or ctx.body.get("messages", [])
        
        from proxy.chat_history import chat_histories, chat_histories_lock
        with chat_histories_lock:
            historical_count = len(chat_histories.get(ctx.session_id, []))
        new_messages = messages[historical_count:] if historical_count < len(messages) else messages
        
        call_index = {}
        for msg in new_messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                if name not in ("write", "edit"):
                    continue
                import json
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                file_path = extract_file_path(args)
                if file_path:
                    import os
                    normalized = os.path.normpath(os.path.realpath(file_path))
                    call_index[tc.get("id")] = (name, normalized)
        
        for msg in new_messages:
            if msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id not in call_index:
                continue
            
            tool_name, normalized_path = call_index[tool_call_id]
            result_content = msg.get("content", "")
            
            with state.vfs_lock:
                snapshot = state.vfs_rollback_snapshots.pop(tool_call_id, None)
            
            if _looks_like_failure(result_content):
                if snapshot:
                    _rollback_snapshot(snapshot)
                    logger.warning(
                        "VFS: Rolled back %s for %s — failure detected: %.100s",
                        tool_name, normalized_path, result_content,
                    )
                    # Remove phantom write record from log
                    with state.vfs_lock:
                        state.vfs_write_log[:] = [
                            r for r in state.vfs_write_log
                            if not (r.path == normalized_path
                                    and r.workspace_id == workspace_id
                                    and r.agent_id == ctx.session_id
                                    and r.new_hash == snapshot.new_hash
                                    and abs(r.timestamp - snapshot.created_at) < 2.0)
                        ]
                else:
                    with state.vfs_lock:
                        cache = state.vfs_caches.get(workspace_id)
                        if cache:
                            cache.invalidate(normalized_path)
            else:
                if snapshot:
                    pass
                with state.vfs_lock:
                    cache = state.vfs_caches.get(workspace_id)
                    if cache:
                        entry = cache.get(normalized_path)
                        if entry and not entry.verified:
                            from proxy.vfs import VFSEntry
                            confirmed = VFSEntry(
                                path=entry.path,
                                content=entry.content,
                                content_hash=entry.content_hash,
                                size_bytes=entry.size_bytes,
                                cached_at=entry.cached_at,
                                cached_by=entry.cached_by,
                                workspace_id=entry.workspace_id,
                                verified=True,
                            )
                            cache.put(confirmed)

    def _populate_vfs_cache(self, ctx):
        """Scan original_messages for read tool results and cache file contents.
        
        Runs AFTER _sync_chat_history() and BEFORE _handle_tool_interception().
        """
        import proxy.state as state
        from proxy.vfs import _vfs_active_for_context, _cache_file
        
        if not _vfs_active_for_context(getattr(ctx, 'is_subagent', False)):
            return
        
        messages = ctx.messages_original or ctx.body.get("messages", [])
        workspace_id = self._resolve_vfs_workspace(ctx)
        
        call_index = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                if name != "read":
                    continue
                import json
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                call_index[tc.get("id")] = args
        
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id not in call_index:
                continue
            
            args = call_index[tool_call_id]
            file_path = extract_file_path(args)
            if not file_path:
                continue
            
            if args.get("offset") is not None or args.get("limit") is not None:
                continue
            
            content = msg.get("content", "")
            if not content:
                continue
            
            from proxy.tool_pipeline.tool_limits import FILE_BLOCKLIST_PATTERNS
            normalized_lower = file_path.lower()
            if any(p in normalized_lower for p in FILE_BLOCKLIST_PATTERNS):
                continue
            
            _cache_file(file_path, content, workspace_id, ctx.session_id)

    def _resolve_vfs_workspace(self, ctx):
        """Resolve workspace ID for VFS operations.

        v5: workspaces are global. Use the session's actual active workspace
        (bare name), inheriting from the parent agent tree if needed.
        """
        from proxy.vfs import _vfs_active_for_context
        from proxy.workspace_state import resolve_inherited_workspace
        session_id = getattr(ctx, 'session_id', '')
        workspace_id = resolve_inherited_workspace(
            session_id, client_hint=getattr(ctx, 'client_hint', '')
        )

        return workspace_id


# ── Module-level helpers for prompt cache metrics ──────────────────────
# Multi-alias fallback chains: try every known field name per concept.
# First match wins. Provider-agnostic — adding a new provider's naming
# convention is just adding an alias path.
_READ_ALIASES = [
    # cache_read
    ("cache_read", "prompt_cache_hit_tokens"),                    # DeepSeek (flat)
    ("cache_read", "prompt_tokens_details", "cached_tokens"),     # MiniMax, Qwen, MiMo (nested)
    ("cache_read", "cached_tokens"),                               # flat fallback
    ("cache_read", "cache_read_input_tokens"),                     # future: Anthropic compat
    ("cache_read", "input_tokens_details", "cached_tokens"),      # future: Responses API
    # cache_write — explicit cache creation tokens
    ("cache_write", "cache_creation_input_tokens"),                # Qwen explicit (flat)
    ("cache_write", "prompt_tokens_details", "cache_creation_input_tokens"),
    ("cache_write", "cache_write_input_tokens"),                  # future: Anthropic compat
]


def _read_nested(usage: dict, *keys: str):
    """Traverse a dict by a sequence of keys. Returns None on missing key or non-dict."""
    current = usage
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_cache_tokens(data: dict) -> dict:
    """Extract V3-shaped cache token counts from an upstream response dict.
    
    Uses multi-alias fallback (not per-provider if/else). Returns dict with
    keys: total, cache_read, cache_write, no_cache. Missing values are None.
    
    no_cache is COMPUTED as total - cache_read - cache_write, not read from
    the response. Some providers (Qwen, MiMo) don't report miss separately.
    """
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {"total": None, "cache_read": None, "cache_write": None, "no_cache": None}

    cache_read = cache_write = None

    # Multi-alias fallback
    for key, *path in _READ_ALIASES:
        val = _read_nested(usage, *path)
        if val is not None and isinstance(val, (int, float)):
            clamped = max(0, int(val))
            if int(val) < 0:
                logger.warning("cache_metrics: negative %s=%s — clamping to 0", key, int(val))
            if key == "cache_read" and cache_read is None:
                cache_read = clamped
            elif key == "cache_write" and cache_write is None:
                cache_write = clamped

    # Total prompt tokens — first prompt_tokens, then input_tokens
    total = usage.get("prompt_tokens")
    if not isinstance(total, (int, float)):
        total = usage.get("input_tokens")
    total = int(total) if isinstance(total, (int, float)) else None

    # Compute no_cache = total - cache_read - cache_write
    # Only computed when cache_read is present (provider reported cache data).
    # For non-reporting backends, no_cache stays None so Prometheus counters
    # are not inflated with bogus data.
    no_cache = None
    if total is not None and cache_read is not None:
        raw = total
        raw -= cache_read
        if cache_write is not None:
            raw -= cache_write
        if raw < 0:
            logger.warning(
                "cache_metrics: negative no_cache=%d — cache_read=%s total=%s",
                raw, cache_read, total,
            )
        no_cache = max(0, raw)

    return {"total": total, "cache_read": cache_read, "cache_write": cache_write, "no_cache": no_cache}


# Manual overrides for model keys whose backend_model doesn't contain a
# recognizable provider substring (e.g., ark-code-latest for deepseek).
_PROVIDER_OVERRIDES = {
    "deepseek-v4-pro-ark": "deepseek",
    "qwen3-coder": "qwen",
}


def _detect_provider(backend_model: str = "", model_key: str = "") -> str:
    """Derive provider name from backend_model (substring) + model_key (override).
    
    Returns a short provider string for observability labels. This does NOT
    affect request routing — that is handled by model_registry independently.
    """
    # Override: exact model_key match for models with opaque backend_model
    if model_key in _PROVIDER_OVERRIDES:
        return _PROVIDER_OVERRIDES[model_key]

    bm = (backend_model or "").lower()
    if "deepseek" in bm:
        return "deepseek"
    if "minimax" in bm:
        return "minimax"
    if "qwen" in bm:
        return "qwen"
    if "mimo" in bm:
        return "mimo"
    if "kimi" in bm:
        return "kimi"

    return "other"
