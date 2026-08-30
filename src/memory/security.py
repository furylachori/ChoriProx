"""
Memory poisoning defense for the memory system.

Type-based injection control: only 'fact' and 'event' memory types
are auto-injected into context. Rule-based content filter blocks
dangerous patterns before context injection.
"""

import re
import logging

import proxy.state as _state

logger = logging.getLogger("memory.security")

# Memory types that can be auto-injected into context
AUTO_INJECT_TYPES = {"fact", "event"}

# Memory types that require explicit tool calls only
TOOL_ONLY_TYPES = {"instruction", "task"}

# Content filter patterns — block before context injection
BLOCK_PATTERNS = [
    # Imperative verbs attempting to override behavior
    (re.compile(r"\b(always|never|must|should)\b", re.I), "imperative_override"),
    # Tool name injection
    (re.compile(r"\b(?<!_)bash(?!_)\b", re.I), "tool_injection"),
    # Meta-instructions
    (
        re.compile(r"before\s+responding|first\s+call|remember\s+that", re.I),
        "meta_instruction",
    ),
    # Credential patterns
    (re.compile(r"(password|secret|token|key)\s*:", re.I), "credential_leak"),
]

# Confidence threshold — drop low-confidence memories when context exceeds this
# fraction. Defaults to the proxy-wide CONTEXT_BUDGET_RATIO (env var), else 0.75.
LOW_CONFIDENCE_THRESHOLD = getattr(_state, "CONTEXT_BUDGET_RATIO", 0.75)


def can_auto_inject(memory_type: str) -> bool:
    """Check if a memory type can be auto-injected into context."""
    return memory_type in AUTO_INJECT_TYPES


def filter_memory_content(content: str) -> tuple:
    """Filter memory content for dangerous patterns.

    Returns (is_safe, detected_threats) where detected_threats is a list of
    threat type strings. is_safe is True if no threats detected.
    """
    if not content or not isinstance(content, str):
        return True, []

    threats = []
    for pattern, threat_type in BLOCK_PATTERNS:
        if pattern.search(content):
            threats.append(threat_type)

    if threats:
        logger.warning(f"Memory content filter blocked: {threats}")

    return len(threats) == 0, threats


def should_drop_memory(memory: dict, context_budget_pct: float) -> bool:
    """Determine if a memory should be dropped based on confidence and context budget."""
    if context_budget_pct < LOW_CONFIDENCE_THRESHOLD:
        return False

    confidence = memory.get("confidence", "medium")
    return confidence == "low"
