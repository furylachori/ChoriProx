"""Enhancer package — ^^imp trigger detection and dispatch."""

import re

from enhancer.core import build_context, call_sota, format_response, format_error, sanitize_message

# Fix #5: \A matches absolute start of string, not after newlines
_RAW_IMP_TRIGGER = re.compile(r'\A\s*\^{2}imp(?:\s+|$)', re.IGNORECASE)

# M6 #4: zero-width spaces and BOMs that users may paste from chat
# clients (Slack, iMessage, etc.) shouldn't break ^^imp detection.
_ZWSP_CHARS = ("\u200b", "\u2060", "\ufeff")


def _normalize_text(text: str) -> str:
    """Strip leading zero-width/word-joiner/BOM chars from command input."""
    if not text:
        return text
    i = 0
    while i < len(text) and text[i] in _ZWSP_CHARS:
        i += 1
    return text[i:]


def is_imp_trigger(text: str) -> bool:
    """True if the message text starts with ^^imp (after stripping ZWSP/BOM)."""
    if not text:
        return False
    return _RAW_IMP_TRIGGER.search(_normalize_text(text)) is not None


def extract_raw_prompt(text: str) -> str:
    """Extract raw prompt from message after ^^imp trigger.

    Strips the trigger and leading/trailing whitespace.
    """
    normalized = _normalize_text(text)
    match = _RAW_IMP_TRIGGER.search(normalized)
    if not match:
        return ""
    raw = normalized[match.end():].strip()
    # Fix #7 + #8: sanitize once at extraction time
    return sanitize_message(raw)


def dispatch_imp(session_id: str, text: str) -> dict:
    """Handle ^^imp command: build context, call SOTA, format response.

    Raises specific SOTA errors for caller to handle (Fix #1).
    """
    raw_prompt = extract_raw_prompt(text)
    if not raw_prompt:
        return format_error("No prompt to enhance. Usage: ^^imp <your prompt>")

    system_prompt, user_prompt = build_context(session_id, raw_prompt)

    try:
        enhanced = call_sota(system_prompt, user_prompt)
    except Exception:
        # Let specific errors bubble up to app.py handler (Fix #1)
        raise

    if not enhanced:
        return format_error("Enhancer returned empty response. Sending original.")

    return format_response(enhanced)
