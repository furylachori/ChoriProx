# Block: Tool Loop

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 8. Tool Loop Extraction

**File:** src/memory/tool_loop.py (329 lines)

### 8.1 ToolLoopResult Dataclass

```python
@dataclass
class ToolLoopResult:
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    finish_reason: str = "stop"
    messages_for_history: list = field(default_factory=list)
```

### 8.2 run_memory_tool_loop Signature

```python
def run_memory_tool_loop(
    tool_calls_list: list,
    working_messages: list,
    tools: list,
    model: str,
    session_id: Optional[str],
    llama_url: str,
    body: dict,
    llama_body: dict,
    chat_histories: dict,
    chat_histories_lock: threading.Lock,
    execute_fn: Optional[Callable] = None,
    make_tool_calls_fn: Optional[Callable] = None,
    parse_tools_fn: Optional[Callable] = None,
    max_iterations: int = 3,
    max_total_time: float = 90.0,
    max_result_size: int = 16 * 1024,
    max_calls_per_iteration: int = 10,
) -> ToolLoopResult:
```

### 8.3 Behavior

1. Separates memory tool calls from passthrough calls
2. Executes memory tools locally via execute_fn
3. Appends results to working_messages
4. Trims messages if over 80% of 26k token budget (or 42-message fallback)
5. Re-queries LLM with updated messages
6. Parses new tool calls from response
7. Loops until no memory tools remain, max iterations hit, or budget exceeded
8. Returns ToolLoopResult with content, tool_calls, finish_reason, messages_for_history

Transaction-safe: rolls back working_messages on error (tool_loop.py:304).
