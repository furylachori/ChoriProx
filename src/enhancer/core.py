"""Enhancer core: context assembly, SOTA call, response formatting."""

import json
import logging
import time
from sqlite3 import OperationalError

from proxy.openai_compat import build_openai_response

from memory.db import connection, get_summary, get_last_messages
from enhancer.prompts import (
    ENHANCEMENT_SYSTEM_PROMPT,
    build_enhancement_prompt,
    sanitize_message,
)

logger = logging.getLogger("enhancer.core")

_MAX_RECENT = 3
_TRUNCATE_AFTER_ESCAPE = 1000  # chars, after escaping (Fix #6)


def _get_recent_messages(session_id: str) -> list[dict]:
    """Get recent messages with busy_timeout=3000 and fail-closed on OperationalError (Fix #3, #4).

    All messages are sanitized at load time (Fix #5, #7, #8).
    Truncation happens AFTER escape (Fix #6).
    """
    try:
        with connection(pooled=True) as conn:
            conn.execute("PRAGMA busy_timeout = 3000")
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY message_index DESC LIMIT ?",
                (session_id, _MAX_RECENT),
            ).fetchall()
    except OperationalError:
        # Fix #3: Fail closed on DB contention — raise, don't silently skip
        raise ValueError("System busy. Please try again later.")

    messages = []
    for row in reversed(rows):
        content = row["content"] or ""
        # Fix #5: sanitize at load time (includes NFKC + escape + control-char strip)
        # Fix #7: NFKC normalization inside sanitize_message
        # Fix #8: single escape — only here, never again
        content = sanitize_message(content)
        # Fix #6: truncate AFTER escaping
        content = content[:_TRUNCATE_AFTER_ESCAPE]
        messages.append({"role": row["role"], "content": content})
    return messages


def build_context(session_id: str, raw_prompt: str) -> tuple[str, str]:
    """Assemble session summary + recent messages + raw prompt into SOTA request.

    Returns (system_prompt, user_prompt) ready for SOTA call.
    """
    # Get summary (existing db layer — fails gracefully)
    summary = get_summary(session_id) or ""
    summary = sanitize_message(summary)[:_TRUNCATE_AFTER_ESCAPE]

    # Get recent messages (sanitized + truncated inside)
    recent = _get_recent_messages(session_id)

    # Build the enhancement prompt (Fix #4: plain-text boundary markers)
    # Fix #8: raw_prompt already sanitized by caller, no double escape
    user_prompt = build_enhancement_prompt(summary, recent, raw_prompt)

    return ENHANCEMENT_SYSTEM_PROMPT, user_prompt


def call_sota(
    system_prompt: str,
    user_prompt: str,
    *,
    timeout: float = 15.0,
    max_tokens: int = 512,
) -> str:
    """Call SOTA provider to rewrite the prompt.

    Routes through clients.sota_client.call_sota_extraction so enhancer/
    does not import memory/ directly. The previous code called
    `clients.get_client()` with no args, raising TypeError that was
    swallowed and re-raised as RuntimeError("SOTA client unavailable"),
    which the dispatch keyword filter did not catch — every ^^imp
    invocation returned HTTP 500.

    Refs: docs/REVIEW_2026_06_28/FINAL_REVIEW.md M7-A1 H-1.
    """
    from clients.sota_client import call_sota_extraction

    return call_sota_extraction(
        user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=max_tokens,
        session_id="enhancer",
    )


def format_response(enhanced: str) -> dict:
    """Format enhanced prompt as synthetic OpenAI chat.completion response."""
    ts = int(time.time())
    return build_openai_response(f"**Enhanced prompt**\n\n{enhanced}\n\n_Sent via ^^imp_", model="enhancer", response_id=f"chatcmpl-enhancer-{ts}", created=ts)


def format_error(message: str) -> dict:
    """Format an error as synthetic OpenAI chat.completion response."""
    ts = int(time.time())
    return build_openai_response(message, model="enhancer", response_id=f"chatcmpl-enhancer-{ts}", created=ts)
