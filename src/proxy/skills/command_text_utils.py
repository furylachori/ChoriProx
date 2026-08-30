"""Shared utilities for extracting clean command text from CLI-wrapped messages.

AI-coding clients append framework XML blocks to user messages (e.g.
<environment_details> with cwd/git/time, <system-reminder> with mode-change
notices). These are NOT user content, but they must NOT be stripped from the
message data — their env context is load-bearing and flows verbatim to the
upstream LLM (the proxy cannot reconstruct cwd/git/time).

These helpers strip the known framework tags from the EPHEMERAL command-parse
text only (a local derived for ^^ command parsing), so commands like
``^^ws <name>`` parse cleanly when a trailing framework block is present.
Message data (``ctx.messages`` / ``ctx.body``) is never mutated by this layer.

Canary-safe: only the EXPLICIT tag names in ``_FRAMEWORK_TAGS`` are stripped —
never generic ``<tag>`` matching, which would recurse into canary text and wipe
sub-agent mode detection.

Uses a state-machine parser (not regex) to correctly handle nested same-tag XML.
"""
import hashlib
import logging
import traceback

logger = logging.getLogger("proxy.skills.command_text_utils")

# Known client-framework tags injected into user messages. Stripped from the
# ephemeral command-parse text ONLY. Add a tag here when a client starts
# emitting a new framework block that corrupts ^^ command parsing — do NOT
# broaden to generic <tag> matching (canary wipe risk).
_FRAMEWORK_TAGS: tuple[tuple[str, str], ...] = (
    ("<environment_details>", "</environment_details>"),
    ("<system-reminder>", "</system-reminder>"),
)


def strip_environment_details(text: str) -> str:
    """Remove known client-framework XML blocks from EPHEMERAL command-parse text.

    Strips every tag in ``_FRAMEWORK_TAGS`` (currently ``<environment_details>``
    and ``<system-reminder>``) so ``^^`` commands parse cleanly — e.g.
    ``^^ws myname<system-reminder>...</system-reminder>`` resolves to name
    ``myname`` instead of the reminder block.

    IMPORTANT — this is EPHEMERAL:
    - Operates on a local string derived for command parsing, NEVER on message
      data. ``ctx.messages`` / ``ctx.body`` are untouched; the framework blocks
      (with their cwd/git/time context) flow verbatim to the upstream LLM.
    - Canary-safe: only the explicit tag names below are stripped, never generic
      ``<tag>`` matching.

    State-machine parser (not regex) handles:
    - Paired tags: ``<tag>...</tag>`` -> removed
    - Nested same-tag: ``<tag>foo<tag>bar</tag></tag>`` -> removed (all levels)
    - Orphaned opening tag: ``<tag>...`` (no close) -> removed to end-of-string
    - DOES NOT handle orphaned closing tags — preserved as plain text + WARNING.

    Tag matching is CASE-SENSITIVE (only lowercase tags match).
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        matched = None
        for open_tag, close_tag in _FRAMEWORK_TAGS:
            if text[i:i + len(open_tag)] == open_tag:
                matched = (open_tag, close_tag)
                break

        if matched:
            open_tag, close_tag = matched
            depth = 1
            j = i + len(open_tag)
            while j < n and depth > 0:
                if text[j:j + len(open_tag)] == open_tag:
                    depth += 1
                    j += len(open_tag)
                elif text[j:j + len(close_tag)] == close_tag:
                    depth -= 1
                    j += len(close_tag)
                else:
                    j += 1
            i = j  # skip the block (matched close or end-of-string)
        else:
            result.append(text[i])
            i += 1

    cleaned = "".join(result).strip()

    # Defensive: detect orphaned closing tags of any known framework tag. Should
    # NEVER happen with well-formed client output — if it does, either the client
    # is sending malformed data or our pipeline corrupted the message upstream.
    # Log at WARNING; handled gracefully (preserved as text), not a crash.
    orphaned = [close for _open, close in _FRAMEWORK_TAGS if close in cleaned]
    if orphaned:
        orig_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
        cleaned_hash = hashlib.sha256(cleaned.encode("utf-8", errors="replace")).hexdigest()[:16]
        logger.warning(
            "[FRAMEWORK-TAG-ORPHAN] Orphaned closing tag(s) %s detected after stripping.\n"
            "  ORIGINAL length: %d  sha256[:16]: %s\n"
            "  CLEANED length: %d  sha256[:16]: %s\n"
            "  CALL STACK:\n%s",
            orphaned,
            len(text), orig_hash,
            len(cleaned), cleaned_hash,
            "".join(traceback.format_stack()),
        )

    return cleaned


def extract_command_text(raw_text: str) -> str:
    """Extract clean command text from a potentially CLI-wrapped message.

    Convenience wrapper: strips whitespace + known framework XML tags.
    Use as the first step in parse_subcommand.

    Args:
        raw_text: The raw user message text

    Returns:
        Clean text ready for command parsing
    """
    return strip_environment_details(raw_text.strip())
