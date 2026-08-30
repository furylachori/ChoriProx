# Component: Context Budget

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/03-context-optimization.md)

**Goal**: Eliminate 4.5+ minute LLM timeouts by enforcing a strict 19,500 token context limit within llama's 32k window.

## Token Budget Model (19,500 total)

| Component | Percentage | Characters | Purpose |
|-----------|------------|------------|---------|
| System Prompt + Tools | ~15% | ~2,925 | Fixed overhead — system instructions + available tools |
| Immediate Messages | 8% | ~1,560 | Recent conversation for continuity — complete messages only |
| Fixed Summary | 7% | ~1,365 | Compressed conversation history — strict size enforcement |
| Buffer/Safety | 70% | ~13,650 | Work space for LLM responses |

## select_context()

The `select_context()` function is called in the main proxy route before forwarding to llama-server. The system prompt at index 0 is always preserved. Optimization applies only to non-system messages. Configuration values: `max_context_tokens: 32000`, `safe_context_tokens: 19500`, `optimal_context_tokens: 15000`, `tokens_per_char: 0.25`.

**Immediate messages** (8% / 1,560 tokens): Recent messages selected from the end of the conversation, including only complete messages that fit within the limit. Falls back to including at least the last message even if it exceeds the budget. No partial message truncation.

**Fixed summary** (7% / 1,365 tokens): Compressed conversation history stored per session. Compression triggers at 70% threshold. Focuses on key decisions, constraints, and requirements; excludes tool execution details and code blocks.

## Memory Injection

Memories injected into context use the format `[type|confidence] content`, providing confidence scores so the LLM can weigh evidence. Memory types: fact, event, instruction, task. Only memory tools are injected into context; passthrough tools are discarded.

## Summary Injection

Summaries stored in SQLite, read via read-through cache. Fixed size limit of 1,365 tokens enforced. Token limit enforcement uses priority-based truncation: system prompt is never truncated, immediate messages are truncated from oldest first, then summary is compressed/truncated.
