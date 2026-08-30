# Block: Context Optimization

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 4. Context Optimization Implementation

**File:** qwen_proxy.py:562-680 (select_context function)

### 4.1 select_context(session_id, full_messages, model_name)

Entry point called at qwen_proxy.py:1328 before sending to LLM.

**Behavior:**
1. Oversized last message (>1200 tokens) -> single message mode: [system_prompt, last_message]
2. Normal mode: system prompt + summary injection + memory injection + recent messages

**Budget:** 1,200 token limit for non-system messages.

### 4.2 Summary Injection (qwen_proxy.py:611-617)

Appends conversation summary to system prompt if <=300 tokens:

```python
if summary and summary_tokens <= 300 and system_prompt:
    summary_content = f"\n\nCONVERSATION SUMMARY:\n{summary}\n\n---\nCONTINUE CONVERSATION:"
    system_prompt["content"] += summary_content
```

### 4.3 Memory Injection (qwen_proxy.py:619-658)

Recalls relevant memories from Neo4j and injects as separate system message.
200-token budget. Graceful degradation on failure.

```python
MEMORY_BUDGET = 200
recalled = graph.recall_for_context(session_id, last_content, timeout_ms=200)
safe_memories = filter_memories_for_context(recalled, context_budget_pct=0.0)
memory_text = format_memories_for_injection(safe_memories)
# Inject if within budget
```

### 4.4 Recent Message Selection (qwen_proxy.py:660-680)

Adds messages in reverse order (newest first) until remaining_tokens exhausted.
Always preserves system prompt at index 0.
