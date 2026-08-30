# Component: Recovery System

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/05-recovery-system.md)

Multi-layer defense against malformed tool calls and infinite retry loops. Layer 1 (tool
execution plan detection) is archived dead code — only Layers 2–4 are active.

## Layer 2 — Malformed Call Detection & Auto-Fix

Detects known failure patterns via regex and applies safe fixes before escalation:

- **Missing `description`** — auto-generates a 3–5 word description from tool name
  (`auto_fix_missing_description()` in `tool_optimizer.py`)
- **Wrong tool name** (e.g. `execute_command` → `bash`) — suggests correct name
- **Missing required field** — clarification prompt to LLM

Pattern table lives in `PROBLEM_PATTERNS` dict. Auto-fix is conservative: only
`description` is patched; other fields get clarification prompts.

## Layer 3 — Progressive Escalation (SessionRetryManager)

`SessionRetryManager` tracks retries **per session** (not per tool) with bounded state:

- **`MAX_RETRIES = 3`** per session; `MAX_PATTERN_RETRIES = 2` for stuck-pattern detection
- **Escalation levels**: `LEVEL_1_CLARIFICATION` → `LEVEL_2_EXAMPLES` → `LEVEL_3_FALLBACK`
- **Stuck-pattern detection**: hashes tool call (ignoring `description`) — same hash 3×
  triggers `STUCK_PATTERN_DETECTED` hard stop
- **Session continuation**: on proxy restart, sessions with >10 messages start at retry count 1
- **Thread-safe**: `threading.RLock` guards all state mutations
- **Cleanup**: `clear_session()` removes retry + pattern state on session end

## Layer 4 — Optimized Tool Descriptions

Reduces tool description token cost from ~8,400 to ~1,000 tokens (88% reduction):

- **Optimized descriptions**: each tool gets a minimal OpenAI-format schema (~100 tokens)
  via `generate_optimized_tool_description()`
- **Delivery strategy** (`ContextAwareToolDelivery`):
  - First request → all tools with optimized descriptions
  - Subsequent requests → minimal references (`{"name": tool, "available": true}`)
- **Static schemas**: tool definitions come from Kilo CLI provider; no runtime schema
  changes, no cache invalidation needed
