"""Conversation identity hash chain for the V2 context pipeline.

Implements §2 of docs/v2-chain-hash-redesign.md. Pure functions — no shared
state, no locks, no RequestContext. The canonicalizer is the SINGLE source of
truth: the in-memory chain (compute_chain) and the DB-stored chain_hash
produce byte-identical values, so restart recovery loads the chain the runtime
would have produced.

Content normalization mirrors memory.db._normalize_content (db.py:869-888)
exactly, because that function is applied at store time (db.py:920, 1159) and
the DB content column therefore holds the normalized form. Any divergence
here would make DB-stored chain_hash differ from the runtime chain for
multimodal (list) content, breaking restart recovery.
"""

import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def _canonical(msg):
    """Canonicalize a message dict for hashing.

    Single source of truth for both the DB-stored chain_hash and the in-memory
    chain. Content normalization mirrors memory.db._normalize_content exactly.

    - role: lowercased (M11) so "User"/"user"/"USER" hash identically.
    - content: None -> "" (M12); list -> joined text (only str parts and
      {type:"text"} dict parts survive, matching _normalize_content; non-text
      parts are dropped); other non-str -> str().
    - tool_calls: restricted to OpenAI-spec fields {id, type,
      function.{name, arguments}} (M10); arguments string -> dict parsed when
      valid JSON.
    - tool_call_id: intentionally dropped (M13) — clients regenerate it.

    Returns f"{role}\\x00{content}\\x00{tool_calls}" (NUL-separated, deterministic).
    """
    role = (msg.get("role", "") or "").lower()

    content = msg.get("content")
    if content is None:
        content = ""
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        content = "\n".join(parts)
    elif not isinstance(content, str):
        content = str(content)

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        canonical_calls = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                logger.warning("_canonical: skipping non-dict tool_call entry (%s)", type(tc).__name__)
                continue
            func = tc.get("function", tc)
            if not isinstance(func, dict):
                func = {}
            args = func.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            canonical_calls.append({
                "id": tc.get("id", ""),
                "type": tc.get("type", ""),
                "function": {"name": func.get("name", ""), "arguments": args},
            })
        tool_calls = json.dumps(canonical_calls, sort_keys=True, separators=(",", ":"))
    else:
        tool_calls = ""

    return f"{role}\x00{content}\x00{tool_calls}"


def _message_hash(msg):
    """sha256(_canonical(msg)).hexdigest()[:16]."""
    return hashlib.sha256(_canonical(msg).encode("utf-8")).hexdigest()[:16]


def compute_chain(messages):
    """Compute cumulative chain hash for conversation messages.

    Skips messages with role='system' (system prompt changes every turn).

    Returns (chain_list, chain_to_position):
    - chain_list: 16-char hex cumulative hashes, one per non-system message.
      chain_list[0] = msg_hash of the first non-system message;
      chain_list[i] = sha256(f"{chain_list[i-1]}:{msg_hash[i]}")[:16].
    - chain_to_position: {chain_hash: position_in_non_system_list} for O(1)
      boundary validation.

    Empty input or all-system input returns ([], {}).
    """
    if not messages:
        return ([], {})
    chain_list = []
    chain_to_position = {}
    prev = None
    for msg in messages:
        if (msg.get("role", "") or "").lower() == "system":
            continue
        msg_hash = _message_hash(msg)
        if prev is None:
            chain_hash = msg_hash
        else:
            chain_hash = hashlib.sha256(f"{prev}:{msg_hash}".encode("utf-8")).hexdigest()[:16]
        chain_list.append(chain_hash)
        chain_to_position[chain_hash] = len(chain_list) - 1
        prev = chain_hash
    return (chain_list, chain_to_position)


def find_fork_point(new_chain, old_chain):
    """Find first index where chains diverge.

    Returns:
    - 0 if old_chain is empty (cold path).
    - min(len(new_chain), len(old_chain)) if no divergence (full match or append).
    - First index where new_chain[i] != old_chain[i] otherwise.
    """
    if not old_chain:
        return 0
    limit = min(len(new_chain), len(old_chain))
    for i in range(limit):
        if new_chain[i] != old_chain[i]:
            return i
    return limit
