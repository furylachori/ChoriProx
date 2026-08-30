"""Provider compatibility registry.

Adapters are matched by backend model name substring. Each adapter applies
provider-specific parameter mutations (temperature clamping, role conversion,
reasoning_content injection, etc.) and logs each modification via
_track_param_modification().
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Protocol

logger = logging.getLogger("proxy.pipeline")

# ---------------------------------------------------------------------------
# Replicated from proxy.app (lines 207-270) so provider_compat can be used
# without importing the entire app module.
# ---------------------------------------------------------------------------
import time as _time
import threading as _threading

try:
    from proxy.state import (
        state,
        PROXY_TRACE,
        param_mod_lock,
        param_mod_log_lock,
        param_modifications,
        param_injects,
        param_clamps,
        param_drops,
        param_mod_details,
        PARAM_MOD_LOG_INTERVAL,
        param_mod_log_timestamps,
    )
except ImportError:
    # Fallback stubs when running outside app context (tests, etc.)
    class _FakeState:
        param_modifications = 0
        param_injects = 0
        param_clamps = 0
        param_drops = 0
        param_mod_details: dict = {}
    state = _FakeState()
    PROXY_TRACE = False
    param_mod_lock = _threading.Lock()
    param_mod_log_lock = _threading.Lock()
    param_modifications = 0
    param_injects = 0
    param_clamps = 0
    param_drops = 0
    param_mod_details = {}
    PARAM_MOD_LOG_INTERVAL = 60.0
    param_mod_log_timestamps: dict[str, float] = {}


def _track_param_modification(
    backend: str,
    param_name: str,
    action: str,
    original_value: Any = None,
    new_value: Any = None,
    msg_index: int | None = None,
) -> None:
    """Track and log a parameter modification with rate limiting.

    Replicated from proxy.app:207 so this module is self-contained.
    """
    with param_mod_lock:
        state.param_modifications += 1  # type: ignore[attr-defined]
        if action == "inject":
            state.param_injects += 1  # type: ignore[attr-defined]
        elif action == "clamp":
            state.param_clamps += 1  # type: ignore[attr-defined]
        elif action == "drop":
            state.param_drops += 1  # type: ignore[attr-defined]

        # H-2: Cap param_mod_details to prevent unbounded memory growth.
        # Evict oldest half when the dict exceeds MAX_PARAM_MOD_DETAILS entries.
        MAX_PARAM_MOD_DETAILS = 1000
        if len(state.param_mod_details) > MAX_PARAM_MOD_DETAILS:  # type: ignore[attr-defined]
            keys = list(state.param_mod_details.keys())[:MAX_PARAM_MOD_DETAILS // 2]  # type: ignore[attr-defined]
            for k in keys:
                del state.param_mod_details[k]  # type: ignore[attr-defined]

        detail_key = f"{backend}:{param_name}:{action}"
        state.param_mod_details[detail_key] = (  # type: ignore[attr-defined]
            state.param_mod_details.get(detail_key, 0) + 1  # type: ignore[attr-defined]
        )

    now = _time.time()
    log_key = f"{backend}:{param_name}:{action}"
    should_log = False

    with param_mod_log_lock:
        last_log = param_mod_log_timestamps.get(log_key, 0)
        if now - last_log >= PARAM_MOD_LOG_INTERVAL:
            param_mod_log_timestamps[log_key] = now
            should_log = True

    if PROXY_TRACE or should_log:
        if action == "clamp":
            msg = (
                f"[provider_compat] {backend}: clamped {param_name} "
                f"from {original_value!r} to {new_value!r}"
            )
        elif action == "inject":
            where = f"messages[{msg_index}]" if msg_index is not None else "message"
            msg = f"[provider_compat] {backend}: injected {param_name} in {where}"
        elif action == "drop":
            msg = (
                f"[provider_compat] {backend}: dropped {param_name} "
                f"(value: {original_value!r})"
            )
        else:
            msg = f"[provider_compat] {backend}: {action} on {param_name}"
        logger.debug(msg)


# ---------------------------------------------------------------------------
# ProviderAdapter protocol + registry
# ---------------------------------------------------------------------------

class ProviderAdapter(Protocol):
    """Protocol that all provider adapters must satisfy."""

    @staticmethod
    def can_handle(backend: str) -> bool:
        """Return True if this adapter applies to the given backend model name."""
        ...

    @staticmethod
    def apply(body: dict, model_cfg: Any) -> dict:
        """Apply provider-specific mutations to the request body. Returns body."""
        ...


_REGISTRY: list[type[ProviderAdapter]] = []


def register_adapter(adapter: type[ProviderAdapter]) -> None:
    """Register a ProviderAdapter class in the global registry."""
    _REGISTRY.append(adapter)


def _get_adapter(backend: str) -> type[ProviderAdapter] | None:
    backend = (backend or "").lower()
    for adapter_cls in _REGISTRY:
        if adapter_cls.can_handle(backend):
            return adapter_cls
    return None


def apply_provider_compat(body: dict, model_cfg: Any) -> dict:
    """Apply all matching provider adapters to the request body.

    Deep-copies messages before mutation to prevent leaking provider-specific
    changes back into ctx.body (which is later used for message storage).
    """
    if not model_cfg:
        return body

    body = copy.deepcopy(body)

    backend = (model_cfg.backend_model or "").lower()
    adapter_cls = _get_adapter(backend)
    if adapter_cls:
        body = adapter_cls.apply(body, model_cfg)

    return body


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------

class DeepSeekAdapter:
    """DeepSeek: inject reasoning_content on ALL assistant messages (thinking mode)."""

    @staticmethod
    def can_handle(backend: str) -> bool:
        return "deepseek" in backend

    @staticmethod
    def apply(body: dict, model_cfg: Any) -> dict:
        for idx, msg in enumerate(body.get("messages", [])):
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                msg["reasoning_content"] = " "
                _track_param_modification(
                    backend="deepseek",
                    param_name="reasoning_content",
                    action="inject",
                    msg_index=idx,
                )
        return body


class MiMoAdapter:
    """MiMo: identical to DeepSeek — inject reasoning_content on ALL assistant messages."""

    @staticmethod
    def can_handle(backend: str) -> bool:
        return "mimo" in backend

    @staticmethod
    def apply(body: dict, model_cfg: Any) -> dict:
        for idx, msg in enumerate(body.get("messages", [])):
            if msg.get("role") == "assistant" and "reasoning_content" not in msg:
                msg["reasoning_content"] = " "
                _track_param_modification(
                    backend="mimo",
                    param_name="reasoning_content",
                    action="inject",
                    msg_index=idx,
                )
        return body


class KimiAdapter:
    """Kimi/Moonshot: inject reasoning_content on tool-call messages + clamp temperature to [0,1]."""

    @staticmethod
    def can_handle(backend: str) -> bool:
        return "kimi" in backend

    @staticmethod
    def apply(body: dict, model_cfg: Any) -> dict:
        # Inject reasoning_content on assistant messages with tool_calls
        for idx, msg in enumerate(body.get("messages", [])):
            if (
                msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and "reasoning_content" not in msg
            ):
                msg["reasoning_content"] = " "
                _track_param_modification(
                    backend="kimi",
                    param_name="reasoning_content",
                    action="inject",
                    msg_index=idx,
                )
        # Clamp temperature to [0, 1]
        if "temperature" in body and body["temperature"] is not None:
            original_temp = body["temperature"]
            new_temp = max(0.0, min(1.0, original_temp))
            if new_temp != original_temp:
                body["temperature"] = new_temp
                _track_param_modification(
                    backend="kimi",
                    param_name="temperature",
                    action="clamp",
                    original_value=original_temp,
                    new_value=new_temp,
                )
        return body


class MiniMaxAdapter:
    """MiniMax: clamp temperature to (0,1] + convert system role to user."""

    # V2 chain-hash: this adapter merges consecutive role=user messages
    # in apply() (lines 295-311). The proxy inserts a synthetic assistant
    # barrier before last_user to break the merge for the per-turn cache
    # block. Other adapters (OpenAI, Anthropic) do not merge — default False.
    merges_consecutive_users: bool = True

    @staticmethod
    def can_handle(backend: str) -> bool:
        return "minimax" in backend

    @staticmethod
    def apply(body: dict, model_cfg: Any) -> dict:
        # Clamp temperature to (0, 1]
        if "temperature" in body and body["temperature"] is not None:
            original_temp = body["temperature"]
            new_temp = max(0.01, min(1.0, original_temp))
            if new_temp != original_temp:
                body["temperature"] = new_temp
                _track_param_modification(
                    backend="minimax",
                    param_name="temperature",
                    action="clamp",
                    original_value=original_temp,
                    new_value=new_temp,
                )

        # Convert system role to user (MiniMax does not support system role).
        # Track converted messages so the consecutive-user merger skips them;
        # each system-converted-to-user message must remain distinct so
        # ContextOptimizeStage doesn't lose the user's second system guidance
        # into the auto-injected TASK_PAYLOAD wrapper.
        converted_from_system = []
        for msg in body.get("messages", []):
            if msg.get("role") == "system":
                msg["role"] = "user"
                converted_from_system.append(id(msg))
                _track_param_modification(
                    backend="minimax",
                    param_name="system_role→user",
                    action="convert",
                )

        # Merge consecutive user messages ONLY when neither was converted from
        # system. Consecutive converted messages stay distinct so each
        # system-converted-to-user message survives as a separate user message.
        # Non-converted consecutive user messages still merge (MiniMax errors
        # on consecutive user turns).
        msgs = body.get("messages", [])
        if msgs:
            merged = []
            merge_count = 0
            for m in msgs:
                prev = merged[-1] if merged else None
                prev_is_user = prev is not None and prev.get("role") == "user"
                cur_is_user = m.get("role") == "user"
                prev_is_converted = prev is not None and id(prev) in converted_from_system
                cur_is_converted = id(m) in converted_from_system
                if prev_is_user and cur_is_user and not prev_is_converted and not cur_is_converted:
                    prev_content = prev.get("content")
                    prev_content = prev_content if isinstance(prev_content, str) else str(prev_content or "")
                    cur_content = m.get("content")
                    cur_content = cur_content if isinstance(cur_content, str) else str(cur_content or "")
                    prev["content"] = prev_content + "\n\n" + cur_content
                    merge_count += 1
                else:
                    merged.append(m)
            if merge_count:
                body["messages"] = merged
                _track_param_modification(backend="minimax", param_name="consecutive_user_merge", action="merge")

            # Insert synthetic assistant barriers between consecutive user
            # messages (MiniMax rejects back-to-back user turns). The merge
            # above already collapsed every [non-conv][non-conv] run, so any
            # remaining user/user adjacency needs a barrier — including the
            # [non-conv user][conv-from-system user] case produced by VFS +
            # conversation_block ending in user.
            barrier_inserted = 0
            barrier_msgs = []
            for i, m in enumerate(merged):
                if (m.get("role") == "user"
                        and barrier_msgs
                        and barrier_msgs[-1].get("role") == "user"):
                    barrier_msgs.append({
                        "role": "assistant",
                        "content": " ",
                    })
                    barrier_inserted += 1
                barrier_msgs.append(m)
            if barrier_inserted:
                body["messages"] = barrier_msgs
                _track_param_modification(
                    backend="minimax",
                    param_name="consecutive_user_barrier",
                    action="insert",
                )
        return body


# ---------------------------------------------------------------------------
# Register built-in adapters at import time
# ---------------------------------------------------------------------------
register_adapter(DeepSeekAdapter)
register_adapter(MiMoAdapter)
register_adapter(KimiAdapter)
register_adapter(MiniMaxAdapter)
