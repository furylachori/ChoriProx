"""Shared utility functions for the proxy package."""

import time
from datetime import datetime, timezone


def safe_content(content) -> str:
    """Extract text from content field (string, list, dict, or None).

    Identical implementation extracted from both tool_loop.py and tools.py
    (which had character-for-identical copies).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def extract_text_content(content) -> str:
    """Extract text from OpenAI content field.

    Canonical implementation is context.py's version, which is the most complete:
    - Handles None → ""
    - Handles dict → .get("text", "")
    - Handles image_url items in lists (skips them)
    - Handles bare str items in lists (uses str(item))
    - Falls through to str(content) for any remaining type

    Behavioral note: auth.py and model_registry.py previously returned ""
    for falsy non-string content (e.g. extract_text_content(0) → "").
    The canonical version returns "0" (no falsy guard — str(0) → "0").
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # image_url items have no text — mark as placeholder
                    parts.append("[image]")
            else:
                # bare str or other items in list
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def _now_iso_sqlite() -> str:
    """SQLite-friendly UTC timestamp (no microseconds, 'Z' suffix)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_iso() -> str:
    """ISO 8601 UTC timestamp with microseconds. Use for Neo4j/HTTP."""
    return datetime.now(timezone.utc).isoformat()
