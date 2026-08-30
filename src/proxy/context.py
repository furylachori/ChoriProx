"""Token estimation and context optimization."""

import logging
import json
import os
import sys
import threading

from proxy.utils import extract_text_content
from proxy.state import (
    chat_histories,
    chat_summaries,
    chat_histories_lock,
    chat_summaries_lock,
)

logger = logging.getLogger(__name__)


class ContextLimitError(Exception):
    """Raised when assembled context exceeds the model's safe token budget.

    Distinct from ValueError so callers can distinguish context-limit failures
    from validation or skill-activation errors.
    """
    pass


ENABLE_TOKEN_CALCULATION = True
MAX_MESSAGE_CONTENT_SIZE = 100 * 1024  # 100KB per message
MAX_TOTAL_CONTENT_SIZE = 500 * 1024  # 500KB total

_model_registry = {}


def _get_raw_context_limit(model_cfg) -> int:
    """Get the raw model context limit (not the safe/80% limit)."""
    if model_cfg and hasattr(model_cfg, 'context_limit') and model_cfg.context_limit:
        return model_cfg.context_limit
    try:
        from proxy.model_registry import _model_registry
        return 128000
    except Exception:
        return 128000


def resolve_effective_context_limit(model_cfg, override):
    """Resolve the effective context limit from a reduce-only override.

    Reduce-only: the effective budget can never exceed the real model window.
    Returns the real model limit when no override is set, the override when the
    real window is unknown, and min(override, real) otherwise.
    """
    real = getattr(model_cfg, "context_limit", None) if model_cfg else None
    if override is None:
        return real
    if not real:
        return override
    return min(override, real)


def enforce_context_limits(body: dict, model_cfg) -> None:
    """Pre-trim safety net for oversized un-trimmable requests.

    Messages are trimmable downstream: ``select_context()`` / the V2 pipeline
    pair-aware-trim and cap message history to fit the safe budget. Tool and
    system-prompt schemas are NOT trimmed by those stages — they pass through
    verbatim. ``calculate_tokens_from_request_body`` caps only the *message*
    portion of its estimate (and the message total) and then adds tools/system
    AFTER that capping, so an oversized-tool request can produce an estimate
    that exceeds the raw limit with nothing downstream able to trim it.

    This guard rejects such requests early. It computes the UN-TRIMMABLE
    portion (tool schemas + system prompt) and raises ``ValueError`` if that
    portion alone exceeds the model's raw context limit. The ``ValueError``
    propagates to ``ContextOptimizeStage``, which catches it and raises
    ``StageAbort(413)`` (no upstream round-trip is burned).

    Note: ``ContextLimitError`` (this module) exists as a distinct signal but
    is currently never raised anywhere in src/. Wiring it through the
    estimator is a follow-up, not addressed here.
    """
    # Tools and system prompt are un-trimmable: select_context()/V2 trim only
    # the message history, so if these schemas alone exceed the raw limit the
    # request can never fit upstream. Reject before burning an upstream call.
    try:
        config = get_context_config(model_cfg=model_cfg)
        tpc = config.get("tokens_per_char", 0.25)
    except Exception:
        tpc = 0.25

    untrimmable_chars = len(
        json.dumps(body.get("tools") or [], separators=(",", ":"))
    ) + len((body.get("system") or ""))
    untrimmable_tokens = untrimmable_chars * tpc
    raw_limit = _get_raw_context_limit(model_cfg)
    if untrimmable_tokens > raw_limit:
        raise ValueError(
            f"Request too large: un-trimmable tool/system schemas "
            f"({untrimmable_tokens} tokens) exceed model context limit of {raw_limit}"
        )


_singleton_init_lock = threading.Lock()
_task_graph_singleton = None


def _get_task_graph_singleton():
    """Get or create a shared TaskGraph instance (singleton pattern)."""
    global _task_graph_singleton
    if _task_graph_singleton is None:
        with _singleton_init_lock:
            if _task_graph_singleton is None:
                from memory.task_graph import TaskGraph
                _task_graph_singleton = TaskGraph()
    return _task_graph_singleton


def get_context_config(model_name=None, model_cfg=None, context_limit_override=None):
    """Get context configuration for the given model.

    Delegates to model_registry.get_context_config to ensure consistent caching.
    """
    from proxy.model_registry import get_context_config as registry_get_config

    return registry_get_config(model_name, model_cfg, context_limit_override=context_limit_override)


def get_safe_context_limit(model_name=None, model_cfg=None):
    """Get safe context limits for the given model.

    Delegates to model_registry.get_safe_context_limit to ensure consistent caching.
    """
    from proxy.model_registry import get_safe_context_limit as registry_get_limit

    return registry_get_limit(model_name, model_cfg)


def calculate_tokens_from_text(text, tokens_per_char=None):
    """Calculate approximate token count from text."""
    if not ENABLE_TOKEN_CALCULATION:
        return 0  # Disabled
    if isinstance(text, list):
        text = " ".join(item.get("text", "") for item in text if isinstance(item, dict))
    if not isinstance(text, str):
        return 0
    if not text:
        return 0
    if tokens_per_char is None:
        config = get_context_config()
        tokens_per_char = config["tokens_per_char"]
    return len(text) * tokens_per_char


def calculate_tokens_from_request_body(body, model_name=None):
    """Calculate approximate token count from complete request body including tools."""
    if not ENABLE_TOKEN_CALCULATION:
        return 0

    try:
        config = get_context_config(model_name)
        if config:
            safe_tokens = config.get("safe_context_tokens", 0)
            tpc = config.get("tokens_per_char", 0)
            if safe_tokens > 0 and tpc > 0:
                max_message_size = max(int((safe_tokens * 0.10) / tpc), 512)
                max_total_size = max(int((safe_tokens * 0.50) / tpc), 2048)
            else:
                max_message_size = MAX_MESSAGE_CONTENT_SIZE
                max_total_size = MAX_TOTAL_CONTENT_SIZE
        else:
            max_message_size = MAX_MESSAGE_CONTENT_SIZE
            max_total_size = MAX_TOTAL_CONTENT_SIZE
    except Exception:
        logger.warning("Failed to get context config for model %s, using fallback limits", model_name, exc_info=True)
        config = None
        max_message_size = MAX_MESSAGE_CONTENT_SIZE
        max_total_size = MAX_TOTAL_CONTENT_SIZE

    total_len = 0

    if "messages" in body:
        for msg in body["messages"]:
            content = msg.get("content", "")
            content = extract_text_content(content)
            if content:
                content_len = len(content)
                if content_len > max_message_size:
                    logger.info(
                        "Message content exceeds per-message limit (%d > %d), "
                        "capping estimate — select_context() will truncate",
                        content_len, max_message_size,
                    )
                    total_len += max_message_size
                    continue
                total_len += content_len + 1

    if total_len > max_total_size:
        logger.info(
            "Total content exceeds soft limit (%d > %d), "
            "capping estimate — select_context() will trim",
            total_len, max_total_size,
        )
        total_len = max_total_size

    if "tools" in body:
        tools = body["tools"]
        if isinstance(tools, list):
            total_len += len(json.dumps(tools, separators=(",", ":")))
        elif isinstance(tools, dict) and tools.get("cached"):
            total_len += len(f"cached_tools_{tools.get('hash', '')}")

    if "system" in body:
        total_len += len(body["system"]) + 1

    tpc = config["tokens_per_char"] if config else 0.25
    return total_len * tpc


def calculate_text_from_tokens(token_count, model_name=None):
    """Calculate approximate text length from token count."""
    if not ENABLE_TOKEN_CALCULATION:
        return 0  # Disabled
    config = get_context_config(model_name)
    return int(token_count * config["chars_per_token"])





def _group_message_units(messages: list) -> list:
    """Group a flat list of messages into atomic units for pair-aware trimming.

    A "unit" is a list of consecutive messages that must stay together when
    trimming. Rules:
    - A `role: user` message is a unit of length 1.
    - A `role: assistant` message with no `tool_calls` is a unit of length 1.
    - A `role: assistant` message WITH `tool_calls` is the head of a unit that
      includes all consecutive following `role: tool` messages (its results).
    - A `role: tool` message with no preceding assistant-with-tool-calls in the
      current scan (malformed input) is a unit of length 1, marked for orphan
      handling downstream.
    - Anything else (unknown role) is a unit of length 1.

    Returns: list of lists (each inner list is a unit, in original order).
    """
    units = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if (
            role == "assistant"
            and isinstance(m.get("tool_calls"), list)
            and m.get("tool_calls")
        ):
            unit = [m]
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                unit.append(messages[j])
                j += 1
            units.append(unit)
            i = j
        else:
            units.append([m])
            i += 1
    return units


def _count_msg_chars(msg):
    """Return character count of a single message.

    Accounts for: string content, list content (text items), tool_calls
    function name+arguments, and reasoning_content/reasoning fields.
    Used by V2 context selector; does NOT account for tool-role message
    fields (tool_call_id, name) — parity with existing _msg_len closure.
    """
    total = 0
    content = msg.get("content", "")
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        total += sum(len(item.get("text", "")) for item in content if isinstance(item, dict))
    for tc in (msg.get("tool_calls") or []):
        func = tc.get("function", {})
        total += len(func.get("name", ""))
        total += len(func.get("arguments", ""))
    for field in ("reasoning_content", "reasoning"):
        val = msg.get(field)
        if isinstance(val, str):
            total += len(val)
    return total


def select_context(session_id, full_messages, model_name=None, model_cfg=None,
                   is_subagent=False, ctx=None, tools=None):
    """Select optimal context using the three-tier pipeline.

    V1 (monolithic summary) was removed — V2 is the only context path.
    `ctx` is forwarded so V2 can record chain-hash boundary fields.
    `tools` (optional) is the tool schema list from the request body, used to
    compute accurate tool_schema_tokens (rather than the old fixed 2000).
    """
    logger.info("Context pipeline active: three-tier (V2)")
    from proxy.context_v2 import _select_context_v2
    return _select_context_v2(
        session_id, full_messages, model_name, model_cfg,
        is_subagent=is_subagent, ctx=ctx, tools=tools,
    )






