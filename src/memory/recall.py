"""
Memory recall and context injection filtering.

Controls which memories are injected into LLM context based on
type, confidence, and content safety checks.
"""

import logging
from html import escape
from memory.security import (
    can_auto_inject,
    filter_memory_content,
    should_drop_memory,
)

logger = logging.getLogger("memory.recall")


def filter_memories_for_context(
    memories: list, context_budget_pct: float = 0.0
) -> list:
    """Filter memories for safe context injection.

    Only 'fact' and 'event' types pass through. Content is checked for
    dangerous patterns. Low-confidence memories dropped when context budget high.
    """
    safe_memories = []

    for memory in memories:
        memory_type = memory.get("memory_type", "fact")

        # Type gate: only auto-inject types
        if not can_auto_inject(memory_type):
            continue

        # Confidence gate
        if should_drop_memory(memory, context_budget_pct):
            logger.info(
                f"Dropping low-confidence memory: {memory.get('id', 'unknown')}"
            )
            continue

        # Content safety gate
        content = memory.get("content", "")
        is_safe, threats = filter_memory_content(content)
        if not is_safe:
            logger.warning(f"Blocking unsafe memory: {threats}")
            continue

        safe_memories.append(memory)

    return safe_memories


def format_memories_for_injection(memories: list) -> str:
    """Format filtered memories as a system message for context injection."""
    if not memories:
        return ""

    lines = ["--- RELEVANT MEMORIES ---"]
    for mem in memories:
        mem_type = mem.get("memory_type", "fact")
        content = mem.get("content", "")
        confidence = mem.get("confidence", "medium")
        safe_content = content.replace('--', '\u2012\u2012')
        safe_content = escape(safe_content, quote=True)
        lines.append(f"[{mem_type}|{confidence}] {safe_content}")
    lines.append("--- END MEMORIES ---")

    return "\n".join(lines)
