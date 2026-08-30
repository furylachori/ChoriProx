# Task Planning v2 — Phase 1: Context Assembly

> **Depends on:** Nothing. First component to implement.
> **Output:** `src/planning/task_context.py`
> **Next:** [Phase 2 — Read-Only Tool Suite](./phase-2-tools.md)

---

## 4. Phase 1: Context Assembly (`task_context.py`)

### 4.1 Design Principles

1. **Reuse existing DB accessors** — call `get_summary(session_id)` and `get_last_messages(session_id, n=50)` directly. Do NOT generate a new summary. *(Note: `n=50` is the argument passed by `assemble_context()`; the underlying DB function may have a different default.)*
2. **Token budget is hard-capped** — never exceed the computed available input tokens.
3. **Truncate oldest-first** — preserve the most recent messages because they are most relevant to the current planning request.
4. **Minimum retention** — always keep at least 5 recent messages even if they slightly exceed budget (within 10% overage tolerance).

### 4.2 Token Budget Model

The planning model is `step-3.5-flash-2603` with a **256K context window**.

```
Context window:           256,000 tokens
Usable input ratio:       × 0.80
                          ─────────────
Gross input budget:       204,800 tokens

Reserves:
  System prompt:          ~   500 tokens
  Tools schema:           ~ 1,500 tokens
  Response budget:            4,096 tokens
  Safety margin:          ~   500 tokens
                          ─────────────
Total reserves:           ~ 6,600 tokens

Available input tokens:   204,800 − 6,600 ≈ 198,200 tokens
```

**Concrete constant:**

```python
# src/planning/task_context.py

# NOTE: TOKEN_RATIO is a heuristic. If tiktoken becomes available, replace
# this with exact encoding (cl100k_base).
TOKEN_RATIO = 2.8  # conservative underestimate for safety margin

# Verify with provider documentation. step-3.5-flash-2603 claims 256K but
# some "flash" model variants ship at 128K. Use env override if needed.
MODEL_CONTEXT_WINDOW = int(os.environ.get("PLANNER_CTX_WINDOW", "256000"))
USABLE_INPUT_RATIO = 0.80
SYSTEM_PROMPT_RESERVE = 500
TOOLS_SCHEMA_RESERVE = 1_500
RESPONSE_BUDGET = 4_096
SAFETY_MARGIN = 500
TOTAL_RESERVES = SYSTEM_PROMPT_RESERVE + TOOLS_SCHEMA_RESERVE + RESPONSE_BUDGET + SAFETY_MARGIN
AVAILABLE_INPUT_TOKENS = int(MODEL_CONTEXT_WINDOW * USABLE_INPUT_RATIO) - TOTAL_RESERVES
# => 198,200


def estimate_tokens(text: str) -> int:
    """Character-length heuristic. Conservative (underestimates density)."""
    return max(1, int(len(text) / TOKEN_RATIO))
```

### 4.3 Context Assembly Algorithm

```python
# src/planning/task_context.py

import os
import json
import logging

from memory.db import get_summary, get_last_messages

logger = logging.getLogger("planning.task_context")


def assemble_context(
    session_id: str,
    user_description: str,
    working_dir: str,
    max_messages: int = 50,
) -> dict:
    """Build the messages array and metadata for the planning loop.

    Returns:
        {
            "system_prompt": str,
            "messages": list[dict],   # ready for SOTAChannel.ask_multi()
            "budget_used_tokens": int,
            "budget_total_tokens": int,
            "summary_included": bool,
            "message_count": int,
        }
    """
    budget = AVAILABLE_INPUT_TOKENS

    # 1. Fixed-cost overhead
    system_prompt = build_planning_system_prompt(working_dir)
    system_cost = estimate_tokens(system_prompt)
    user_cost = estimate_tokens(user_description)

    # Pre-check: truncate user_description if it alone would exhaust budget
    if user_cost > budget - SYSTEM_PROMPT_RESERVE:
        logger.warning("User description exceeds token budget; truncating.")
        max_chars = int((budget - SYSTEM_PROMPT_RESERVE) * TOKEN_RATIO)
        user_description = user_description[:max_chars] + "\n...[truncated]"
        user_cost = estimate_tokens(user_description)

    # 2. Fetch existing summary (zero generation cost)
    summary = get_summary(session_id) or ""
    summary_cost = estimate_tokens(summary) if summary else 0

    # 3. Fetch raw messages from DB (ordered ASC by message_index)
    raw_messages = get_last_messages(session_id, n=max_messages)

    # Truncate extremely long individual messages before cost estimation
    MAX_MSG_CONTENT_CHARS = 28_000  # ~10K tokens at 2.8 chars/token
    for m in raw_messages:
        content = m.get("content") or ""
        if len(content) > MAX_MSG_CONTENT_CHARS:
            m["content"] = content[:MAX_MSG_CONTENT_CHARS] + "\n...[truncated]"

    # 4. Compute per-message costs
    msg_costs = []
    for m in raw_messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        text = f"[{m.get('message_index', '?')}] {role}: {content}"
        msg_costs.append({"msg": m, "cost": estimate_tokens(text)})

    # 5. Budget allocation: summary first, then newest messages
    remaining = budget - system_cost - user_cost

    included_summary = ""
    if remaining > 0 and summary and summary_cost <= remaining:
        included_summary = summary
        remaining -= summary_cost
    elif remaining > 0 and summary:
        # Truncate summary to fit
        chars = int(remaining * TOKEN_RATIO)
        included_summary = summary[:chars] + "\n...[truncated]"
        remaining = 0

    # 6. Greedily include newest messages until budget exhausted
    included_msgs = []
    for item in reversed(msg_costs):  # newest first
        if item["cost"] <= remaining:
            included_msgs.insert(0, item["msg"])
            remaining -= item["cost"]
        else:
            break

    # 7. Minimum retention guard: keep at least 5 messages
    MIN_MESSAGES = 5
    if len(included_msgs) < MIN_MESSAGES and len(raw_messages) >= MIN_MESSAGES:
        # Re-run without summary truncation, just take last 5 raw
        included_msgs = raw_messages[-MIN_MESSAGES:]
        included_summary = ""
        remaining = budget - system_cost - user_cost - sum(
            estimate_tokens(f"[{m.get('message_index', '?')}] {m.get('role', '')}: {m.get('content', '')}")
            for m in included_msgs
        )

    # 8. Build final messages
    messages = [{"role": "system", "content": system_prompt}]
    if included_summary:
        messages.append({
            "role": "system",
            "content": f"Session summary:\n{included_summary}",
        })
    for m in included_msgs:
        msg = {
            "role": m.get("role", "user"),
            "content": m.get("content") or "",
        }
        tc_json = m.get("tool_calls_json")
        if tc_json:
            try:
                msg["tool_calls"] = json.loads(tc_json) if isinstance(tc_json, str) else tc_json
            except json.JSONDecodeError:
                pass
        tc_id = m.get("tool_call_id")
        if tc_id:
            msg["tool_call_id"] = tc_id
        messages.append(msg)
    messages.append({"role": "user", "content": user_description})

    budget_used = budget - remaining

    return {
        "system_prompt": system_prompt,
        "messages": messages,
        "budget_used_tokens": budget_used,
        "budget_total_tokens": budget,
        "summary_included": bool(included_summary),
        "message_count": len(included_msgs),
    }
```

### 4.4 Truncation Strategy Summary

| Priority | Content | Truncation Rule |
|----------|---------|-----------------|
| 1 | System prompt | Never truncated |
| 2 | User description | Never truncated (it's the trigger) |
| 3 | Session summary | Truncated tail if too long |
| 4 | Recent messages | Oldest dropped first; minimum 5 kept |
| 5 | Message content | Individual messages truncated only if a single message exceeds ~28KB (rare) |
