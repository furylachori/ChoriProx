"""pytest-free debug harness for the ChoriProx proxy.

Extracted from `tests/integration/conftest.py:proxy_with_mock` +
`tests/integration/conftest.py:mock_backend_server` so the same lifecycle +
interaction is available outside pytest (IPython, scratch scripts).

Public API:
    debug_session(...)  → contextmanager that yields DebugSession
    DebugSession        → chat, send_as_subagent, activate_ticket,
                          inspect_agent, get_chat_history,
                          get_tool_counts, set_tool_limit, dump_captured,
                          reset, time_travel, sign_canary, capture_canary,
                          rate_limit_status, last_tool_call, canned, state,
                          ensure_agent, extract_task_prompts,
                          extract_swarm_canary
    ProxyResponse       → thin wrapper around Flask test client response

Architecture mirrors the conftest fixtures exactly — one implementation,
zero drift. `_reset_all_shared_state()` reuses the conftest reset (with the
spec additions: escalation_rate_limiter, ToolCallTracker._instances,
canary_verify_cache, _chain_state, vfs_caches, vfs_write_log).

`debug_session()` patches `memory.sync.submit_sync` to a no-op, so
harness scenarios do not enqueue SOTA synchronization work. Runnable
templates still use `tests/debug/templates/_helpers.py::_finish()` to
convert accumulated checks into deterministic process exit status.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from unittest.mock import MagicMock, patch

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProxyResponse
# ---------------------------------------------------------------------------


@dataclass
class ProxyResponse:
    """Thin wrapper around Flask test client response."""

    status_code: int
    data: bytes
    _json: Optional[dict] = field(default=None, repr=False)

    @property
    def content(self) -> str:
        """Decoded response body."""
        return self.data.decode("utf-8", errors="replace")

    def json(self) -> Optional[dict]:
        """Parsed JSON body, or None if not JSON."""
        if self._json is not None:
            return self._json
        try:
            return json.loads(self.data)
        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Mock backend (closure factory, not class state)
# ---------------------------------------------------------------------------


def _make_handler_class(responses: list, captured: list) -> type:
    """Factory that creates a handler class sharing state across connections.

    `responses`: FIFO queue of canned response dicts consumed one per POST.
    `captured`: shared list each handler appends its parsed request body to.

    Also exposes class-level `_captured_request_body`, `_captured_headers`,
    `_response_body`, `_status_code`, `_request_count` for backwards compat
    with `tests/integration/conftest.py:MockBackendHandler` so existing
    tests that reach into `handler._captured_headers` keep working.

    The closure-based design is intentional: HTTPServer never passes kwargs
    to the handler, so a `__init__(self, *, responses=None)` design would
    break — the handler would be constructed without the canned queue.
    """
    _queue = deque(responses or [])

    class _Handler(BaseHTTPRequestHandler):
        # Class-level state mirrors tests/integration/conftest.py:MockBackendHandler.
        # New code should prefer `c.captured_requests`; legacy code can still
        # reach into these attributes after grabbing the class via
        # `c._handler_class`.
        _captured_request_body = None
        _captured_headers = None
        _response_body = b'{"choices":[{"message":{"role":"assistant","content":"default"},"finish_reason":"stop"}]}'
        _status_code = 200
        _request_count = 0

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}

            # Update shared list (preferred API)
            captured.append({"body": body, "headers": dict(self.headers)})
            # Also update class-level attrs (legacy API)
            type(self)._captured_request_body = raw
            type(self)._captured_headers = dict(self.headers)
            type(self)._request_count += 1

            if _queue:
                # Deep-copy the canned dict BEFORE popping control keys
                # (`_status`, `_body`, `_retry_after`). The caller may reuse
                # the same dict across retry subcases; only the detached
                # copy may lose control keys. Hidden control keys (`_status`,
                # `_body`, `_retry_after`) are NEVER serialized into the
                # body — only `resp["_body"]` (or the dict itself when there
                # is no `_body`) reaches `json.dumps`.
                resp = copy.deepcopy(_queue.popleft())
                status = resp.pop("_status", 200)
                body = resp.pop("_body", resp)
                retry_after = resp.pop("_retry_after", None)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                if retry_after is not None:
                    self.send_header("Retry-After", str(retry_after))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                # Use class-level _response_body + _status_code for backwards compat
                self.send_response(self._status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(self._response_body)))
                self.end_headers()
                self.wfile.write(self._response_body)

        def log_message(self, fmt, *args):
            pass

    return _Handler


# ---------------------------------------------------------------------------
# Model registry config
# ---------------------------------------------------------------------------


_DEFAULT_MODELS = ["kimi", "mini", "qwen3-coder"]


def _make_proxy_models_config(backend_url: str, models: list[str]) -> dict:
    """Build the proxy_models.json config pointing all named models at the mock backend."""
    model_configs = {
        "kimi": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 262144,
            "name": "ChoriProx Kimi K2.6",
            "api_key_env": "TEST_API_KEY",
            "backend_model": "opencode-go/kimi-k2.6",
        },
        "mini": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 204800,
            "name": "ChoriProx MiniMax M2.7",
            "api_key_env": "TEST_API_KEY",
            "backend_model": "minimax/MiniMax-M2.7",
        },
        "qwen3-coder": {
            "backend": backend_url,
            "tool_parser": "xml",
            "context_limit": 32000,
            "name": "ChoriProx Qwen3 Coder",
        },
        "deepseek": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 65536,
            "backend_model": "deepseek-v4-pro",
            "api_key_env": "TEST_API_KEY",
        },
        "mimo": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 32000,
            "backend_model": "mimo-7b",
            "api_key_env": "TEST_API_KEY",
        },
        "glm-5.2": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 1000000,
            "name": "ChoriProx GLM 5.2",
            "api_key_env": "TEST_API_KEY",
            "backend_model": "glm-5.2:cloud",
        },
    }
    selected = {k: v for k, v in model_configs.items() if k in models}
    return {
        "models": selected,
        "default": {
            "backend": backend_url,
            "tool_parser": "openai",
            "context_limit": 32000,
        },
    }


# ---------------------------------------------------------------------------
# Shared state reset
# ---------------------------------------------------------------------------


def _reset_all_shared_state() -> None:
    """Reset all shared in-memory state to prevent cross-session contamination.

    Mirrors `tests/integration/conftest.py:_reset_all_shared_state` with the
    spec additions (escalation_rate_limiter, ToolCallTracker._instances,
    canary_verify_cache, _chain_state, vfs_caches, vfs_write_log,
    vfs_rollback_snapshots).
    """
    # Rate limiters — delegate to the single-source-of-truth helper in
    # proxy.state. Pattern matches tests/integration/conftest.py and
    # tests/conftest.py:reset_rate_limiters — all three sites call
    # state.reset_all_rate_limiters(), so adding a new limiter auto-
    # propagates to every test/devtools reset path.
    try:
        import proxy.state as state
        state.reset_all_rate_limiters()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (rate limiters): %s", e)

    # Chat histories + LRU
    try:
        from proxy.chat_history import chat_histories, chat_histories_lock, _chat_histories_lru

        with chat_histories_lock:
            chat_histories.clear()
            _chat_histories_lru.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (chat histories): %s", e)

    # Chat summaries — use module reference to avoid stale local binding
    try:
        import proxy.state as state

        with state.chat_summaries_lock:
            state.chat_summaries.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (chat summaries): %s", e)

    # Workspace state — use module reference to avoid stale local binding
    try:
        import proxy.state as state

        with state.workspace_store._lock:
            state.workspace_store._active.clear()
            state.workspace_store._recent.clear()
            state.workspace_store._last_access.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (workspace): %s", e)

    # Tool cache — use module reference to avoid stale local binding
    try:
        import proxy.state as state

        with state.tool_cache_lock:
            state.tool_cache.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (tool cache): %s", e)

    # Model config cache — use module reference to avoid stale local binding
    try:
        import proxy.state as state

        state.model_config_cache.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (model config cache): %s", e)

    # Agent registry
    try:
        from proxy.agent.agent_registry import registry

        with registry._lock:
            registry._agents.clear()
            registry._agents_by_id.clear()
            registry._lru.clear()
            registry._children.clear()
            registry._in_flight_requests.clear()
            registry._gol_query_log.clear()
            registry._gol_call_count = 0
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (agent registry): %s", e)

    # Counters (must use module reference for ints)
    try:
        import proxy.state as state

        state.request_counter = 0
        state.tool_cache_size = 0
        state.tool_cache_hits = 0
        state.tool_cache_misses = 0
        state.param_modifications = 0
        state.param_drops = 0
        state.param_clamps = 0
        state.param_injects = 0
        state.param_mod_details.clear()
        state.param_mod_log_timestamps.clear()
        state._last_cleanup_time = 0.0
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (counters): %s", e)

    # Neo4j circuit breaker — use module reference to avoid stale local binding
    try:
        import proxy.state as state

        state.neo4j_circuit_breaker._failure_count = 0
        state.neo4j_circuit_breaker._state = "closed"
        state.neo4j_circuit_breaker._last_failure_time = None
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (neo4j circuit breaker): %s", e)

    # ToolCallTracker instances (per-session)
    try:
        from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker

        with ToolCallTracker._instances_lock:
            ToolCallTracker._instances.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (ToolCallTracker): %s", e)

    # Canary verification cache (stale canary from session A must not leak to B)
    try:
        import proxy.state as state

        with state.canary_verify_cache_lock:
            state.canary_verify_cache.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (canary cache): %s", e)

    # V2 chain-hash state per (session_id, model_name)
    try:
        import proxy.state as state

        with state._chain_state_lock:
            state._chain_state.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (chain state): %s", e)

    # VFS caches + write log + rollback snapshots (must hold vfs_lock — all
    # production code acquires it before touching these collections)
    try:
        import proxy.state as state

        with state.vfs_lock:
            state.vfs_caches.clear()
            state.vfs_write_log.clear()
            state.vfs_rollback_snapshots.clear()
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()
    except (ImportError, AttributeError) as e:
        logger.debug("reset partial (VFS): %s", e)


# ---------------------------------------------------------------------------
# DebugSession
# ---------------------------------------------------------------------------


class DebugSession:
    """Interactive proxy debug session backed by app.test_client()."""

    def __init__(
        self,
        client,
        mock_port: int,
        captured_bodies: list,
        handler_class: type,
    ):
        self._client = client
        self._mock_port = mock_port
        self._captured_bodies = captured_bodies
        self._handler_class = handler_class
        self._inbound: list[dict] = []
        self._history: list[dict] = []

        # Wrap client.post so dump_captured() can see what the test sent.
        # Keep a reference to the original so the wrapper calls through.
        self._orig_post = client.post

        def _recording_post(*args, **kwargs):
            method = "POST"
            path = args[0] if args else kwargs.get("path", "/")
            json_body = kwargs.get("json")
            data = kwargs.get("data")
            headers = dict(kwargs.get("headers") or {})
            try:
                body_repr = json.loads(data) if isinstance(data, (str, bytes)) else json_body
            except (json.JSONDecodeError, TypeError):
                body_repr = data
            self._inbound.append(
                {"method": method, "path": path, "body": body_repr, "headers": headers}
            )
            return self._orig_post(*args, **kwargs)

        self._client.post = _recording_post

    # ----- properties -----

    @property
    def client(self):
        """Flask test client. `client.environ_base['HTTP_X_SESSION_CLIENT']` pin."""
        return self._client

    @property
    def mock_port(self) -> int:
        """TCP port the mock backend is bound to."""
        return self._mock_port

    @property
    def captured_requests(self) -> list[dict]:
        """Outbound bodies the proxy sent to the mock backend."""
        return list(self._captured_bodies)

    # ----- chat -----

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        model: str = "kimi",
        client_hint: Optional[str] = None,
        stream: bool = False,
        extra_body: Optional[dict] = None,
    ) -> ProxyResponse:
        """Send a chat request via app.test_client(). Returns parsed response.

        Optional `client_hint` is written to `HTTP_X_SESSION_CLIENT` so the
        proxy derives a deterministic session_id (else it's whatever Flask's
        test client defaults to).

        Optional `extra_body` merges control fields (`temperature`, `top_p`,
        `max_tokens`, `tool_choice`, etc.) into the request body. Keys in
        `extra_body` overwrite the defaults from `messages`/`model`/`stream`
        if they collide.
        """
        if client_hint:
            self._client.environ_base["HTTP_X_SESSION_CLIENT"] = client_hint

        body: dict = {"messages": messages, "model": model, "stream": stream}
        if tools is not None:
            body["tools"] = tools
        if extra_body:
            body.update(extra_body)

        # Call through the wrapped `client.post` so dump_captured() records
        # this request in the inbound list. The wrapper calls the original
        # _HTTP_SESSION.post under the hood — no recursion.
        resp = self._client.post("/v1/chat/completions", json=body)
        parsed = None
        try:
            parsed = resp.get_json()
        except (json.JSONDecodeError, AttributeError, TypeError):
            parsed = None
        return ProxyResponse(status_code=resp.status_code, data=resp.data, _json=parsed)

    def turn(
        self,
        content: str,
        tools: Optional[list[dict]] = None,
        model: str = "kimi",
        client_hint: Optional[str] = None,
        stream: bool = False,
        extra_body: Optional[dict] = None,
    ) -> ProxyResponse:
        """Send a user message, automatically accumulating conversation history.

        Appends {"role": "user", "content": content} to internal history,
        sends the full history via chat(), then appends the assistant's response
        to history. Returns the ProxyResponse.

        Use this for multi-turn conversations without manually building messages[].
        Use c.chat(messages=[...]) for precise control over the message array.
        """
        if stream:
            raise ValueError(
                "turn() does not support stream=True — streaming responses cannot be "
                "parsed into history. Use chat(messages=[...], stream=True) directly."
            )
        self._history.append({"role": "user", "content": content})
        resp = self.chat(
            messages=list(self._history),  # copy to prevent mutation
            tools=tools,
            model=model,
            client_hint=client_hint,
            stream=stream,
            extra_body=extra_body,
        )
        # Append assistant response to history for next turn
        try:
            body = resp.json()
            if body and "choices" in body and body["choices"]:
                msg = body["choices"][0].get("message", {})
                if msg:
                    self._history.append(msg)
        except (TypeError, KeyError, IndexError):
            pass
        return resp

    @property
    def history(self) -> list[dict]:
        """Current accumulated conversation history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear accumulated conversation history without resetting proxy state."""
        self._history.clear()

    # ----- sub-agent spawn -----

    def send_as_subagent(
        self,
        canary: str,
        subagent_prompt: str,
        subagent_messages: list[dict],
        client_hint: Optional[str] = None,
        model: str = "kimi",
        tools: Optional[list[dict]] = None,
    ) -> tuple[ProxyResponse, str]:
        """Send a turn as a sub-agent with the canary embedded as the first user message.

        Returns (ProxyResponse, derived_session_id) where derived_session_id is the
        content-derived session_id the proxy assigned to this sub-agent turn.

        The canary MUST be at position-0 of the first user message (proxy's
        canary parser requires loc.start == 0). This method builds that message
        internally; callers must NOT pre-embed the canary in `subagent_messages`.

        Optional ``tools`` is forwarded verbatim to ``chat()`` so the proxy's
        sub-agent tool filter (``SkillDefinition.subagents.tool_whitelist``) has
        something to constrain. Without ``tools=`` the proxy's ``select_tools``
        sees an empty client-tool set and the sub-agent's filtered set is also
        empty — not a failure, but the operator can't verify the whitelist is
        being applied.
        """
        from proxy.model_registry import resolve_session_id

        # Embed canary + subagent_prompt as the first user message — proxy
        # requires canary at position 0 of the message content.
        combined_first = f"{canary}\n{subagent_prompt}" if subagent_prompt else canary
        # Drop any message that already contains the canary to prevent duplication.
        messages_with_canary = [{"role": "user", "content": combined_first}] + [
            m for m in subagent_messages
            if not (m.get("role") == "user" and canary in m.get("content", ""))
        ]
        sid = resolve_session_id({"messages": messages_with_canary}, client_hint or "")
        resp = self.chat(
            messages=messages_with_canary,
            model=model,
            client_hint=client_hint,
            tools=tools,
        )
        return resp, sid

    # ----- agent pre-seed (custom-skill first-request helper) -----

    def ensure_agent(self, session_id: str) -> None:
        """Ensure an Agent is registered for ``session_id``.

        Required for the FIRST request to a custom-skill trigger in a debug
        session. The pipeline's ``_get_agent(ctx)`` looks up the agent from
        SQLite, but the debug harness never initializes SQLite state before
        the first request — so the agent load returns ``None`` and the skill
        dispatcher reports "activation failed".

        Pre-registering a fresh ``Agent(session_id=session_id)`` lets the
        skill dispatcher find the skill, dispatch, and install the mode
        cleanly. Subsequent requests in the same session don't need this —
        the agent gets persisted by post-processing and reloaded on demand.

        Example:

            parent_sid = c.session_id_for(
                [{'role':'user','content':'^^research do something'}],
                client_hint='parent'
            )
            c.ensure_agent(parent_sid)
            r = c.chat(messages=[...], client_hint='parent')
        """
        if not session_id:
            raise ValueError("session_id must be non-empty")
        from proxy.agent.agent import Agent
        from proxy.agent.agent_registry import registry
        from proxy.skill_lifecycle import register_lifecycle_callbacks

        # Idempotency: check the public lookup first (do NOT touch _agents
        # directly — registry.register handles duplicate session IDs).
        if registry.get_or_load(session_id) is not None:
            return
        agent = Agent(session_id=session_id)
        register_lifecycle_callbacks(agent)
        registry.register(agent)

    # ----- skill activation -----

    # Default fallback tool sets — kept for offline mode when the skill classes
    # can't be imported. Mirrors TicketSkill.get_tool_filter() /
    # WorkSkill.get_tool_filter() at the time this was authored.
    _FALLBACK_TICKET_TOOLS = frozenset(
        {"read", "glob", "grep", "search", "proxy_ticket_list_tasks", "proxy_ticket_get_task", "proxy_work", "task", "webfetch"}
    )
    _FALLBACK_WORK_TOOLS = frozenset(
        {
            "read",
            "write",
            "edit",
            "bash",
            "glob",
            "grep",
            "search",
            "task",
            "proxy_work",
            "proxy_ticket_list_tasks",
            "proxy_ticket_get_task",
            "proxy_ticket_append_review_note",
        }
    )

    @staticmethod
    def _get_skill_tool_filter(skill_class_path: str) -> Optional[frozenset]:
        """Lazy-import a skill class and call get_tool_filter(None). None on failure."""
        try:
            import importlib

            module_path, class_name = skill_class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return frozenset(cls().get_tool_filter(None))
        except (ImportError, AttributeError, TypeError):
            return None

    def activate_ticket(
        self,
        session_id: str,
        workspace: Optional[str] = None,
        task_hash: Optional[str] = None,
        ttl: int = 300,
        subcommand: str = "plan",
    ) -> None:
        """Pre-seed agent.active_mode by registering an Agent + setting ModeState.

        IMPORTANT: session_id MUST match the proxy's content-derived session_id exactly.
        Use c.session_id_for(messages, client_hint) to compute it before calling.

        Example:
            sid = c.session_id_for([{"role":"user","content":"hi"}], client_hint="alice")
            c.activate_ticket(sid, workspace="choriprox")

        Pattern from tests/mode_harness.py:25-44. Constructs a fresh Agent
        (no SQLite history needed), registers it, then sets ticket mode.
        Registers skill lifecycle callbacks so on_deactivate fires when the
        proxy later clears the mode (e.g. finish_reason=stop post-processing).

        `workspace` is set separately via workspace_store.set_active() —
        it is NOT part of skill_state (the skill mode and the workspace
        selection are orthogonal per `proxy.workspace_state`).
        """
        if not session_id:
            raise ValueError("session_id must be non-empty")

        from proxy.agent.agent import Agent
        from proxy.agent.agent_registry import registry
        from proxy.skills.skill_definition import ModeState
        from types import MappingProxyType

        with registry._lock:
            existing = registry._agents.get(session_id)
        if existing is not None:
            agent = existing
        else:
            agent = Agent(session_id=session_id, agent_id=session_id)
            registry.register(agent)

        try:
            from proxy.skill_lifecycle import register_lifecycle_callbacks

            register_lifecycle_callbacks(agent)
        except (ImportError, AttributeError):
            pass  # Lifecycle callbacks are optional; mode still clears correctly

        tool_names = (
            self._get_skill_tool_filter("skills.ticket_skill.TicketSkill")
            or self._FALLBACK_TICKET_TOOLS
        )

        skill_state = {"action": subcommand, "arg": task_hash or ""}

        mode = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=ttl,
            allowed_tool_names=tool_names,
            skill_state=MappingProxyType(skill_state),
        )
        agent.set_active_mode(mode)

        if workspace:
            import proxy.state as state

            state.workspace_store.set_active(session_id, workspace)

    def activate_work(self, session_id: str, task_hash: str, ttl: int = 300) -> None:
        """Pre-seed agent.active_mode to a work ModeState.

        IMPORTANT: session_id MUST match the proxy's content-derived session_id exactly.
        Use c.session_id_for(messages, client_hint) to compute it before calling.

        skill_state uses the real dispatcher key format: {"action": "do", "arg": task_hash}.
        """
        if not session_id:
            raise ValueError("session_id must be non-empty")

        from proxy.agent.agent import Agent
        from proxy.agent.agent_registry import registry
        from proxy.skills.skill_definition import ModeState
        from types import MappingProxyType

        with registry._lock:
            existing = registry._agents.get(session_id)
        if existing is not None:
            agent = existing
        else:
            agent = Agent(session_id=session_id, agent_id=session_id)
            registry.register(agent)

        try:
            from proxy.skill_lifecycle import register_lifecycle_callbacks

            register_lifecycle_callbacks(agent)
        except (ImportError, AttributeError):
            pass

        tool_names = (
            self._get_skill_tool_filter("skills.work_skill.WorkSkill")
            or self._FALLBACK_WORK_TOOLS
        )

        mode = ModeState(
            name="work",
            activated_at=time.time(),
            ttl_seconds=ttl,
            allowed_tool_names=tool_names,
            skill_state=MappingProxyType({"action": "do", "arg": task_hash}),
        )
        agent.set_active_mode(mode)

    def activate_workspace(self, session_id: str, workspace: str) -> None:
        """workspace_store.set_active() — NOT a skill mode."""
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not workspace:
            raise ValueError("workspace must be non-empty")
        import proxy.state as state

        state.workspace_store.set_active(session_id, workspace)

    # ----- introspection -----

    def inspect_agent(self, session_id: str) -> Optional[dict]:
        """Return ModeState snapshot for session_id, or None if unknown.

        Checks the registry's _agents dict first (fast path for pre-registered
        agents) then falls back to get_or_load() for auto-registered agents.
        The peek is intentional: pytest's _reset_all_shared_state() clears
        _agents BEFORE the test starts, but agents registered via
        activate_ticket() during the test land in _agents and are visible
        there even when get_or_load() may not return the same instance
        (e.g., due to SQLite persistence paths that don't apply in
        debug_session's tmp DB).
        """
        from proxy.agent.agent_registry import registry

        with registry._lock:
            agent = registry._agents.get(session_id)
        if agent is None:
            agent = registry.get_or_load(session_id)
        if agent is None:
            return None
        snap = agent.snapshot_mode()
        if snap is None:
            return None
        # snap is a frozen ModeState dataclass; convert to a JSON-able dict
        return {
            "name": snap.name,
            "activated_at": snap.activated_at,
            "ttl_seconds": snap.ttl_seconds,
            "allowed_tool_names": list(snap.allowed_tool_names) if snap.allowed_tool_names else None,
            "sub_agent_tool_filter": list(snap.sub_agent_tool_filter),
            "skill_state": dict(snap.skill_state),
        }

    def session_id_for(
        self,
        messages: list[dict],
        client_hint: str = "",
    ) -> str:
        """Compute the content-derived session_id the proxy will assign.

        Mirrors `proxy.model_registry.resolve_session_id` — the proxy uses
        sha256("|".join(first 10 user messages) + ":" + client_hint)[:16].
        Useful for pre-seeding agent state so the proxy's lookup hits.

        Example:
            sid = c.session_id_for([{"role":"user","content":"list tasks"}],
                                    client_hint="sess-1")
            c.activate_ticket(sid, workspace="choriprox")
            r = c.chat(messages=[...], client_hint="sess-1")
        """
        from proxy.model_registry import resolve_session_id

        return resolve_session_id({"messages": messages}, client_hint)

    def get_chat_history(self, session_id: str) -> list[dict]:
        """Read proxy.chat_history.chat_histories[session_id] (raw messages)."""
        from proxy.chat_history import chat_histories

        entry = dict.get(chat_histories, session_id)
        return list(entry) if entry is not None else []

    def get_tool_counts(self, session_id: str) -> dict:
        """Return ToolCallTracker.get_instance(session_id).status_summary().

        Returns {} if no tracker exists yet (lazy — the tracker is created
        on first call, so a fresh session returns {} instead of allocating
        one).
        """
        from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker

        tracker = ToolCallTracker._instances.get(session_id)
        if tracker is None:
            return {}
        return tracker.status_summary()

    def set_tool_limit(self, field: str, n: int) -> None:
        """Mutate the tool-call limit for existing trackers in-place.

        Calls ``ToolLimits.set_limit(field, n)`` on every currently-registered
        tracker. Does NOT propagate to future ``get_instance()`` calls — each
        new ``ToolCallTracker`` constructs a fresh ``ToolLimits`` with its own
        empty ``_overrides`` dict, ignoring any prior sentinel state.
        """
        from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
        from proxy.tool_pipeline.tool_limits import ToolLimits

        # ToolLimits() creates a fresh empty _overrides per instance — sentinel
        # does not affect future trackers.
        sentinel = ToolLimits()
        sentinel.set_limit(field, n)
        # Update any existing trackers so the override applies now.
        with ToolCallTracker._instances_lock:
            for tracker in ToolCallTracker._instances.values():
                tracker._limits.set_limit(field, n)

    # ----- diagnostics -----

    def dump_captured(self) -> dict:
        """Return {inbound, outbound} dicts of captured request/response traffic."""
        return {
            "inbound": list(self._inbound),
            "outbound": list(self._captured_bodies),
        }

    def reset(self) -> None:
        """Reset all proxy state singletons (rate limiters, registries, etc)."""
        _reset_all_shared_state()
        self._inbound.clear()
        self._captured_bodies.clear()
        self._history.clear()

    # ----- time travel -----

    @contextmanager
    def time_travel(self, seconds: float):
        """Context manager that patches time.time() and time.monotonic() to advance by `seconds`.

        Use for TTL expiry tests: with c.time_travel(360): # 6 minutes
        """
        import time as time_mod

        original_time = time_mod.time
        original_mono = time_mod.monotonic
        offset = seconds

        def fake_time():
            return original_time() + offset

        def fake_mono():
            return original_mono() + offset

        with patch.object(time_mod, "time", side_effect=fake_time):
            with patch.object(time_mod, "monotonic", side_effect=fake_mono):
                yield

    # ----- canary helpers -----

    def sign_canary(self, skill_name: str = "ticket", state_str: str = "") -> str:
        """Sign a fresh canary using proxy.canary.sign_canary.

        Returns the wrapped marker `[[__SUBAGENT__:SKILL:<name>:<sig>:<ts>]]`
        ready to embed at position-0 of a sub-agent's first user message.
        """
        try:
            from proxy.canary import sign_canary

            sig_ts = sign_canary(skill_name, state_str)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Cannot sign canary: {e}")
        return f"[[__SUBAGENT__:SKILL:{skill_name}:{sig_ts}]]"

    def capture_canary(self) -> Optional[str]:
        """Extract the canary from the most recent captured outbound body.

        Returns the first line of the last message's content (canaries are
        always emitted at position 0), or None if nothing was captured or
        the body has no messages.
        """
        if not self._captured_bodies:
            return None
        try:
            msgs = self._captured_bodies[-1]["body"].get("messages", [])
            if not msgs:
                return None
            content = msgs[-1].get("content", "")
            if isinstance(content, str):
                return content.split("\n")[0] if content else None
        except (KeyError, TypeError, IndexError):
            pass
        return None

    # ----- swarm helpers (custom research swarms) -----

    def extract_task_prompts(self, response) -> list[str]:
        """Return the child prompt from every native ``task`` call in a response.

        Parses ``response.json()``, selects ``tool_calls`` entries whose
        ``function.name == "task"``, JSON-decodes ``arguments`` and returns
        the ``prompt`` value from each. Malformed / non-task / non-JSON
        entries are skipped. Returns an empty list when no native task
        calls are present (or response has no choices / isn't JSON).
        """
        try:
            data = response.json()
        except (TypeError, AttributeError):
            return []
        if not isinstance(data, dict):
            return []
        try:
            choices = data.get("choices") or []
            if not choices:
                return []
            message = choices[0].get("message") or {}
            tc_list = message.get("tool_calls") or []
        except (IndexError, KeyError, TypeError):
            return []
        if not isinstance(tc_list, list):
            return []
        out: list[str] = []
        for tc in tc_list:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict) or fn.get("name") != "task":
                continue
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(args, dict):
                continue
            prompt = args.get("prompt", "")
            if isinstance(prompt, str):
                out.append(prompt)
        return out

    def extract_swarm_canary(self, response) -> Optional[str]:
        """Return the canary MARKER prefix from the first swarm-signed task call.

        Uses ``extract_task_prompts`` to materialize each post-injection
        prompt (which the proxy has already prepended with a SKILL canary
        marker at post-process time) and ``proxy.canary.parse_canary`` to
        locate the marker. The returned prefix is the slice of the task
        prompt that represents the canary marker WITHOUT the child
        instruction text that follows it.

        Returns None when no native task call carries a parseable SKILL
        canary. Does NOT use ``capture_canary()`` — that helper reads the
        wrong field (canaries live inside ``arguments.prompt`` of ``task``
        calls, not message content).
        """
        try:
            from proxy.canary import parse_canary
        except (ImportError, AttributeError):
            return None
        for prompt in self.extract_task_prompts(response):
            if not isinstance(prompt, str):
                continue
            parsed = parse_canary(prompt)
            if parsed is None or parsed.kind != "skill":
                continue
            remaining = parsed.remaining
            if remaining and prompt.endswith(remaining):
                marker = prompt[: len(prompt) - len(remaining)]
            elif not remaining:
                marker = prompt
            else:
                marker = prompt
            if marker and (marker != prompt or parsed.verified):
                return marker
        return None

    # ----- rate limiter introspection -----

    def rate_limit_status(self, limiter_name: str) -> dict:
        """Return {entries_count, max_requests, window_seconds} for a named rate limiter.

        Valid names are auto-discovered from `proxy.state` — any `*_rate_limiter`
        attribute that is a `RateLimiter` instance. The name lookup uses the
        `RateLimiter._name` field (e.g. `enhancer`), NOT the attribute name
        (`enhancer_rate_limiter`), to match the public rate-limit API
        (`record_rate_limiter_check(self._name)` keys). This keeps the
        harness in sync as new limiters are added to state.py.
        """
        import proxy.state as state
        from proxy.rate_limiter import RateLimiter

        limiters = {
            getattr(state, name)._name: getattr(state, name)
            for name in dir(state)
            if name.endswith("_rate_limiter")
            and isinstance(getattr(state, name), RateLimiter)
        }
        if limiter_name not in limiters:
            raise ValueError(
                f"Unknown limiter {limiter_name!r}. "
                f"Valid: {sorted(limiters)}"
            )
        rl = limiters[limiter_name]
        return {
            "entries_count": len(rl._entries),
            "max_requests": rl._max,
            "window_seconds": rl._window,
        }

    # ----- tool call history -----

    def last_tool_call(self, name: Optional[str] = None) -> Optional[dict]:
        """Parses `turn()` history (not `chat()` — see Gotcha #4).

        Walks history backwards looking for an assistant message with
        tool_calls. Returns {name, arguments, id} for the first match
        (optionally filtered by name), or None if no tool calls have
        been observed.
        """
        if not self._history:
            return None
        for msg in reversed(self._history):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {})
                tc_name = fn.get("name")
                if name is None or tc_name == name:
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = fn.get("arguments", "{}")
                    return {"name": tc_name, "arguments": args, "id": tc.get("id")}
        return None

    # ----- module shortcuts -----

    @property
    def state(self):
        """Direct reference to proxy.state module. Saves an import in reproducers."""
        import proxy.state as state

        return state

    @property
    def canned(self):
        """Builder namespace for canned response dicts.

        Examples:
            c.canned.stop("hi")
            c.canned.tool_call("read", {"filePath": "/x"})
            c.canned.length("truncated...")
        """
        return _CannedResponseBuilder()


# ---------------------------------------------------------------------------
# Canned response builder
# ---------------------------------------------------------------------------


class _CannedResponseBuilder:
    """Builder for canned response dicts consumed by the mock backend.

    Use:
        c.canned.stop("hi")
        c.canned.tool_call("read", {"x": 1})
        c.canned.length("truncated...")
    """

    def stop(self, content: str) -> dict:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }

    def tool_call(self, name: str, arguments: dict, call_id: str = "call_1") -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    def task_call(
        self,
        prompt: str,
        *,
        description: str = "research",
        call_id: str = "call_task_1",
    ) -> dict:
        """Build a response with one native ``task`` tool call.

        ``prompt`` is the child prompt (without canary — the proxy injects
        the canary at post-process time). ``description`` populates the
        ``description`` argument (purely informational). ``finish_reason``
        is ``"tool_calls"``.
        """
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "task",
                                    "arguments": json.dumps(
                                        {"prompt": prompt, "description": description},
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    def task_calls(self, prompts: list[str]) -> dict:
        """Build a response with one native ``task`` call per prompt.

        Raises ``ValueError`` when ``prompts`` is empty. Call IDs are
        ``call_task_1``..``call_task_N`` and the order matches ``prompts``.
        """
        if not prompts:
            raise ValueError("task_calls requires at least one prompt")
        tool_calls = [
            {
                "id": f"call_task_{i + 1}",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps(
                        {"prompt": p, "description": "research"},
                        ensure_ascii=False,
                    ),
                },
            }
            for i, p in enumerate(prompts)
        ]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    def length(self, content: str) -> dict:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "length",
                }
            ]
        }

    def error(self, status: int, body: dict, retry_after: Optional[int] = None) -> dict:
        """Build a non-200 canned response. The mock backend pops the `_`-prefixed
        keys before serializing — they signal HTTP status / Retry-After to the
        handler, not body content.
        """
        d = {"_status": status, "_body": body}
        if retry_after is not None:
            d["_retry_after"] = retry_after
        return d

    def server_error(self, status: int = 503) -> dict:
        """Shortcut for `error(status, {"error": "upstream down"})`."""
        return {"_status": status, "_body": {"error": "upstream down"}}


# ---------------------------------------------------------------------------
# debug_session() context manager
# ---------------------------------------------------------------------------


@contextmanager
def debug_session(
    canned_responses: Optional[list[dict]] = None,
    db_path: Optional[str] = None,
    secret: str = "",
    api_key: str = "",
    neo4j_disabled: bool = True,
    vfs_enabled: bool = False,
    log_level: str = "DEBUG",
    log_format: str = "json",
    models: Optional[list[str]] = None,
    *,
    skills_dir: Optional[str] = None,
    enable_custom_skill_subagents: bool = False,
):
    """Context manager that wires the proxy to a mock backend.

    Sets required env vars BEFORE importing qwen_proxy. Spawns a mock
    HTTPServer on a dynamic port. Writes a tmp_path/proxy_models.json
    pointing at the mock. Monkeypatches _model_registry, _HTTP_SESSION,
    USE_ADAPTERS, and (optionally) Neo4jClient. Yields a DebugSession
    bound to app.test_client().

    On exit: shutdown server, restore env, restore monkeypatched globals,
    clear all rate limiters and chat history singletons.

    Optional ``skills_dir`` (explicit custom-skill directory) and
    ``enable_custom_skill_subagents`` (kill-switch for the ``task`` tool in
    custom parents) configure the runtime registry for the session:

    - ``skills_dir is None`` → preserve the caller's ``PROXY_SKILLS_DIR``
      (don't touch it).
    - ``skills_dir == ""`` → set ``PROXY_SKILLS_DIR=""`` (explicitly disable).
    - non-empty ``skills_dir`` → set ``PROXY_SKILLS_DIR`` to that value.
    - ``enable_custom_skill_subagents is True`` → set
      ``PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS=1``.
    - explicit ``skills_dir`` BUT ``enable_custom_skill_subagents is False``
      → set ``PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS=0`` (deterministic).

    When ``skills_dir`` is explicitly supplied (not None), the registry is
    rebuilt BEFORE yield so the custom skills in that directory are loaded
    into a fresh singleton. On teardown the registry is reset BEFORE env
    vars are restored, so the post-teardown process starts clean.

    Startup failure: when the custom-skill loader rejects the configuration
    (invalid schema, duplicate name, denied tool, etc.), the raised
    ``SkillRegistryConfigurationError`` propagates out of the ``with``
    block's entry — no session is yielded. The ``finally`` block still runs
    (env restored, partial patches torn down).
    """
    # ── 1. Env var setup (BEFORE import) ───────────────────────────────
    saved_env: dict = {}
    env_overrides = [
        ("PROXY_ENV", "dev"),
        ("PROXY_SESSION_SECRET", secret or os.environ.get("PROXY_SESSION_SECRET", "")),
        ("PROXY_API_KEY", api_key),
        ("PROXY_VFS_ENABLED", "0" if not vfs_enabled else "1"),
        ("PROXY_LOG_FORMAT", log_format),
        ("PROXY_DEBUG", "1" if log_level == "DEBUG" else ""),
        ("TEST_API_KEY", "sk-test-secret-key-12345"),  # mock backend API key
    ]

    # Track whether we touched the two new env keys so we only restore
    # what we actually mutated (saved_env captures pre-mutation values).
    has_explicit_skills_dir = skills_dir is not None
    if has_explicit_skills_dir:
        saved_env["PROXY_SKILLS_DIR"] = os.environ.get("PROXY_SKILLS_DIR")
        if skills_dir == "":
            os.environ["PROXY_SKILLS_DIR"] = ""
        else:
            os.environ["PROXY_SKILLS_DIR"] = skills_dir
        saved_env["PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS"] = os.environ.get(
            "PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS"
        )
        if enable_custom_skill_subagents:
            os.environ["PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS"] = "1"
        else:
            os.environ["PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS"] = "0"

    _tmpdir = None
    server = None
    patches: list = []
    session = None
    try:
        for key, value in env_overrides:
            saved_env[key] = os.environ.get(key)
            os.environ[key] = value

        _tmpdir = tempfile.mkdtemp(prefix="debug_session_")
        _db_path = db_path or os.path.join(_tmpdir, "memory.db")

        # Save caller's PROXY_TEST_DB BEFORE overriding it so teardown
        # restores the exact pre-session value (absent or set).
        saved_env["PROXY_TEST_DB"] = os.environ.get("PROXY_TEST_DB")

        # Ensure the test DB schema exists BEFORE the proxy imports.
        # ``init_db()`` reads ``PROXY_TEST_DB``, so the env override has
        # to be live when we call it. Without the schema the proxy's
        # rate-limiters / chat-history / post-process hit SQLite tables
        # that don't exist — some calls deadlock on a never-created
        # table (no try/except around the read).
        try:
            os.environ["PROXY_TEST_DB"] = _db_path
            from memory.db import init_db
            init_db()
        except Exception as _init_err:
            logger.debug("debug_session: pre-init of SQLite failed: %s", _init_err)
        _proxy_models_path = os.path.join(_tmpdir, "proxy_models.json")
        # NOTE: do NOT re-save PROXY_TEST_DB here — saved_env already holds
        # the caller's original.
        os.environ["PROXY_TEST_DB"] = _db_path
        # ── 2. Spawn mock backend on dynamic port ───────────────────────
        _captured_bodies: list = []
        handler_cls = _make_handler_class(canned_responses or [], _captured_bodies)
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        # ── 3. Write proxy_models.json ──────────────────────────────────
        backend_url = f"http://127.0.0.1:{port}/v1"
        cfg = _make_proxy_models_config(backend_url, models or _DEFAULT_MODELS)
        with open(_proxy_models_path, "w") as f:
            json.dump(cfg, f)

        # ── 4. Patch Neo4jClient BEFORE first instantiation ─────────────
        # Modules are already in sys.modules by the time debug_session()
        # is called — post-import patching works because Neo4jClient is
        # only looked up via attribute access at instantiation sites.
        #
        # Mock submit_sync AND submit_ancient_compress to no-ops
        # UNCONDITIONALLY (regardless of ``neo4j_disabled``). The
        # non-daemon _sync_pool workers and the bounded
        # _summary_pool workers must not fire SOTA dispatch during any
        # scenario — even ones with real Neo4j (the templates don't
        # speak to SOTA, only to the dynamic localhost mock backend).
        # Mirrors the autouse `mock_sync_submit` fixture in
        # tests/conftest.py. Kept BEFORE the proxy import so the
        # patches apply before the first post-process call.
        #
        # Two callers route here: the public sync scheduler
        # (`memory.sync.submit_sync`) and the ancient-compress fallback
        # dispatched from `_apply_summary_budget` (which returns
        # dropped summary rows to SOTA for re-compression). Both must
        # be quiet — otherwise a deterministic scenario would leak real
        # network traffic and produce non-deterministic exit codes.
        patches.append(patch("memory.sync.submit_sync", MagicMock(return_value=None)))
        patches.append(
            patch(
                "memory.sync.submit_ancient_compress",
                MagicMock(return_value=None),
            )
        )

        if neo4j_disabled:
            patches.append(patch("memory.graph.Neo4jClient", MagicMock()))
            patches.append(patch("memory.task_graph.Neo4jClient", MagicMock()))
            # Mock init_db to a no-op so the proxy's startup migration
            # code path doesn't try to connect to Neo4j. Mirrors the
            # autouse `mock_startup_migrations` fixture in tests/conftest.py.
            patches.append(patch("memory.sync.init_db", MagicMock(return_value=None)))
            # workspace_persistence has its own _get_executor() and the
            # shared query_executor entry point — patch both to return None
            # so _persist_workspace / recover_workspaces early-return as
            # no-ops (the production code path handles executor is None).
            patches.append(
                patch(
                    "memory.query_executor.get_shared_executor_from_env",
                    MagicMock(return_value=None),
                )
            )
            patches.append(
                patch(
                    "memory.workspace_persistence._get_executor",
                    MagicMock(return_value=None),
                )
            )

        for p in patches:
            p.start()

        # ── 5. Import proxy + wire registry ────────────────────────────
        import qwen_proxy
        from clients.model_registry import ModelRegistry

        reg = ModelRegistry(config_path="proxy_models.json", base_dir=_tmpdir)
        # String-based patch() — references look up the attribute on each
        # access so subsequent monkeypatch.setattr calls don't bypass.
        new_patch_start = len(patches)
        patches.append(patch("qwen_proxy._model_registry", reg))
        patches.append(patch("proxy.app._model_registry", reg))
        patches.append(patch("proxy.model_registry._model_registry", reg))
        patches.append(patch("qwen_proxy.USE_ADAPTERS", True))
        patches.append(patch("proxy.state._HTTP_SESSION", requests.Session()))
        patches.append(patch("qwen_proxy._HTTP_SESSION", requests.Session()))
        for p in patches[new_patch_start:]:
            p.start()

        qwen_proxy.app.config["TESTING"] = True

        # ── 5b. Rebuild registry for explicit custom-skill directory ────
        # When the caller supplies an explicit skills_dir, the registry
        # singleton must be torn down and rebuilt so the directory's custom
        # skills are published to the running proxy. Same call pattern as
        # Wave 2 eager-init (proxy/app.py:_eager_init_registry).
        if has_explicit_skills_dir:
            from proxy.skills.skill_registry import (
                get_default_registry,
                reset_default_registry,
            )
            reset_default_registry()
            get_default_registry()

        # ── 6. Yield session ───────────────────────────────────────────
        session = DebugSession(
            client=qwen_proxy.app.test_client(),
            mock_port=port,
            captured_bodies=_captured_bodies,
            handler_class=handler_cls,
        )
        yield session
    finally:
        # ── Teardown ────────────────────────────────────────────────────
        if server is not None:
            server.shutdown()
            server.server_close()
        for p in reversed(patches):
            try:
                p.stop()
            except RuntimeError:
                pass
        # If we rebuilt the registry for an explicit skills_dir, reset it
        # BEFORE restoring env so any registry references captured inside
        # the session don't leak into the post-teardown process state.
        if has_explicit_skills_dir:
            try:
                from proxy.skills.skill_registry import reset_default_registry
                reset_default_registry()
            except (ImportError, AttributeError):
                pass
        _reset_all_shared_state()
        if _tmpdir:
            shutil.rmtree(_tmpdir, ignore_errors=True)
        # Restore env (only the keys we touched)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
