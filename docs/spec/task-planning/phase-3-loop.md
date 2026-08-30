# Task Planning v2 — Phase 3: Planning Tool Loop

> **Depends on:** [Phase 1 — Context Assembly](./phase-1-context.md) + [Phase 2 — Tool Suite](./phase-2-tools.md)
> **Output:** `src/planning/task_loop.py`
> **Next:** [Phases 4-5 — Channel & Integration](./phases-4-5-integration.md)

---

## 6. Phase 3: Planning Tool Loop (`task_loop.py`)

The planning loop follows the pattern proven in `src/memory/tool_loop.py` but is strictly read-only and simpler because it does not need to split memory vs. passthrough tool calls.

### 6.1 Design

```python
# src/planning/task_loop.py

import json
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

from memory.sota_channel import SOTAChannel, SOTAResult
from planning.task_tools import PLANNING_TOOLS, execute_readonly_tool

logger = logging.getLogger("planning.task_loop")

MAX_ITERATIONS = 5
MAX_TOTAL_TIME_SECONDS = 90.0
MAX_RESULT_CHARS = 8_000  # per tool result message


@dataclass
class PlanningLoopResult:
    raw: dict
    parsed: list
    elapsed_ms: float
    iterations: int
    tool_calls_made: int
    finish_reason: str  # "add_tasks" | "add_tasks_forced" | "timeout" | "max_iterations" | "no_tool_call"


def run_planning_loop(
    sota: SOTAChannel,
    messages: list[dict],
    working_dir: str,
    session_id: str,
    max_iterations: int = MAX_ITERATIONS,
    max_total_time: float = MAX_TOTAL_TIME_SECONDS,
    force_add_tasks_on_miss: bool = False,
) -> PlanningLoopResult:
    """Run the multi-turn read-only planning loop.

    On each iteration:
      1. Call SOTA with tool_choice="auto" and PLANNING_TOOLS.
      2. If model calls read-only tools, execute them and append results.
      3. If model calls add_tasks, parse and return.
      4. If model returns plain text, stop and return error (model refused).

    After the final tool result, if the model still hasn't called add_tasks:
      - If force_add_tasks_on_miss=True, run ONE more iteration with
        tool_choice={"type": "function", "function": {"name": "add_tasks"}}.
      - Otherwise stop and return finish_reason="max_iterations".
    """
    start_time = time.monotonic()
    iterations = 0
    tool_calls_made = 0
    working_messages = [dict(m) for m in messages]

    # Lightweight rolling budget guard
    TOKEN_RATIO = 2.8  # match task_context.py
    MAX_CONTEXT_TOKENS = int(os.environ.get("PLANNER_CTX_WINDOW", "256000"))
    TOKEN_THRESHOLD = int(MAX_CONTEXT_TOKENS * 0.80)  # 80% hard stop

    def _estimate_msg_tokens(msgs):
        return sum(max(1, int(len(json.dumps(m, default=str)) / TOKEN_RATIO)) for m in msgs)

    while iterations < max_iterations:
        elapsed = time.monotonic() - start_time
        if elapsed > max_total_time:
            logger.warning("[PLAN] Timeout after %.1fs", elapsed)
            return _build_timeout_result(working_messages, start_time, iterations, tool_calls_made)

        # Rolling token guard: evict oldest tool results if projected overflow
        current_est = _estimate_msg_tokens(working_messages)
        if current_est > TOKEN_THRESHOLD:
            logger.warning("[PLAN] Token estimate %d > threshold %d; evicting oldest tool results", current_est, TOKEN_THRESHOLD)
            # Remove oldest tool messages until under threshold
            while current_est > TOKEN_THRESHOLD and len(working_messages) > 2:
                for i, m in enumerate(working_messages):
                    if m.get("role") == "tool":
                        working_messages.pop(i)
                        current_est = _estimate_msg_tokens(working_messages)
                        break
                else:
                    break

        # Iteration call with auto tool choice
        result: SOTAResult = sota.ask_multi(
            messages=working_messages,
            session_id=session_id,
            tools=PLANNING_TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=4096,
        )

        msg = result.raw if isinstance(result.raw, dict) else {}

        tool_calls = msg.get("tool_calls", [])
        content = msg.get("content", "")

        # OpenAI requires the assistant message with tool_calls to precede every tool result
        if tool_calls:
            working_messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls,
            })

        if not tool_calls:
            # Model did not call any tools — usually means it refused or produced raw text
            logger.warning("[PLAN] Iteration %d: no tool calls (content=%d chars)", iterations + 1, len(content))
            return PlanningLoopResult(
                raw={"content": content},
                parsed=[],
                elapsed_ms=(time.monotonic() - start_time) * 1000,
                iterations=iterations + 1,
                tool_calls_made=tool_calls_made,
                finish_reason="no_tool_call",
            )

        # Partition tool calls
        readonly_calls = []
        add_tasks_calls = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name == "add_tasks":
                add_tasks_calls.append(tc)
            else:
                readonly_calls.append(tc)

        if add_tasks_calls:
            # Parse the first add_tasks call and return
            args = add_tasks_calls[0].get("function", {}).get("arguments") or add_tasks_calls[0].get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            parsed = _parse_add_tasks(args)
            return PlanningLoopResult(
                raw=args,
                parsed=parsed,
                elapsed_ms=(time.monotonic() - start_time) * 1000,
                iterations=iterations + 1,
                tool_calls_made=tool_calls_made,
                finish_reason="add_tasks",
            )

        if not readonly_calls:
            # Should not happen, but guard
            break

        # Execute read-only tools and append results
        for tc in readonly_calls:
            tool_calls_made += 1
            func = tc.get("function", {})
            name = func.get("name") or tc.get("name", "")
            args = func.get("arguments") or tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            exec_result = execute_readonly_tool(name, args, working_dir)
            result_text = json.dumps(exec_result, default=str)
            if len(result_text) > MAX_RESULT_CHARS:
                result_text = result_text[:MAX_RESULT_CHARS] + f"\n...[truncated {len(result_text)} chars]"

            working_messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"plan_{tool_calls_made}"),
                "name": name,
                "content": result_text,
            })

        iterations += 1

    # Exhausted iterations without add_tasks
    if force_add_tasks_on_miss:
        logger.info("[PLAN] Forcing add_tasks after %d iterations", iterations)
        # Rolling token guard before forced add_tasks call too
        current_est = _estimate_msg_tokens(working_messages)
        if current_est > TOKEN_THRESHOLD:
            logger.warning("[PLAN] Token estimate %d > threshold %d; evicting oldest tool results (forced)", current_est, TOKEN_THRESHOLD)
            while current_est > TOKEN_THRESHOLD and len(working_messages) > 2:
                for i, m in enumerate(working_messages):
                    if m.get("role") == "tool":
                        working_messages.pop(i)
                        current_est = _estimate_msg_tokens(working_messages)
                        break
                else:
                    break

        result = sota.ask_multi(
            messages=working_messages,
            session_id=session_id,
            tools=PLANNING_TOOLS,
            tool_choice={"type": "function", "function": {"name": "add_tasks"}},
            temperature=0.3,
            max_tokens=4096,
        )
        msg = result.raw if isinstance(result.raw, dict) else {}
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name == "add_tasks":
                args = tc.get("function", {}).get("arguments") or tc.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                parsed = _parse_add_tasks(args)
                return PlanningLoopResult(
                    raw=args,
                    parsed=parsed,
                    elapsed_ms=(time.monotonic() - start_time) * 1000,
                    iterations=iterations + 1,
                    tool_calls_made=tool_calls_made,
                    finish_reason="add_tasks_forced",
                )

    return PlanningLoopResult(
        raw={},
        parsed=[],
        elapsed_ms=(time.monotonic() - start_time) * 1000,
        iterations=iterations,
        tool_calls_made=tool_calls_made,
        finish_reason="max_iterations",
    )


def _parse_add_tasks(arguments: dict) -> list[dict]:
    """Validate and normalize add_tasks arguments. Raises SOTAParseError on invalid input."""
    from memory.sota_channel import SOTAParseError
    tasks = arguments.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise SOTAParseError("No tasks array in tool call result")
    parsed = []
    for t in tasks:
        if not isinstance(t, dict):
            raise SOTAParseError("Task item must be a dict")
        name = str(t.get("name", "")).strip()
        priority = str(t.get("priority", "")).lower()
        if not name:
            raise SOTAParseError("Task name required and must be non-empty string")
        if priority not in {"high", "medium", "low"}:
            raise SOTAParseError(f"Invalid priority '{priority}', must be high/medium/low")
        parsed.append({
            "name": name,
            "priority": priority,
            "summary": str(t.get("summary", "")),
            "tag": t.get("tag"),
        })
    return parsed


def _build_timeout_result(working_messages, start_time, iterations, tool_calls_made):
    return PlanningLoopResult(
        raw={},
        parsed=[],
        elapsed_ms=(time.monotonic() - start_time) * 1000,
        iterations=iterations,
        tool_calls_made=tool_calls_made,
        finish_reason="timeout",
    )
```

### 6.2 Message Evolution Across Turns

**Turn 0 (initial):**

```
system:     TASK_PLANNER_PROMPT + working_dir
system:     Session summary: <...>    (optional)
user:       [msg#0] user: ...
assistant:  [msg#1] assistant: ...
...
user:       ^^task Refactor auth module to use JWT
```

**Turn 1 — model calls `file_glob`:**

```
assistant:  {"role":"assistant","content":"","tool_calls":[{"name":"file_glob","arguments":{"pattern":"src/**/auth*.py"}}]}
tool:       {"role":"tool","tool_call_id":"...","name":"file_glob","content":"{\"pattern\":\"src/**/auth*.py\",\"matches\":[\"src/auth/core.py\",\"src/auth/jwt.py\"],\"truncated\":false}"}
```

**Turn 2 — model calls `read_file`:**

```
assistant:  {"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/auth/jwt.py\"}"}}]}
tool:       {"role":"tool","tool_call_id":"call_abc123","name":"read_file","content":"{\"path\":\"src/auth/jwt.py\",\"offset\":1,\"lines_returned\":42,\"content\":\"...\"}"}
```

**Turn 3 — model calls `add_tasks`:**

```
assistant:  {"role":"assistant","content":"","tool_calls":[{"id":"call_def456","type":"function","function":{"name":"add_tasks","arguments":"{\"tasks\":[...]}"}}]}
```
→ Loop terminates, tasks parsed and returned.

### 6.3 Convergence Safeguards

| Safeguard | Value | Behavior |
|-----------|-------|----------|
| `max_iterations` | 5 | Hard stop after 5 assistant turns (excluding the final `add_tasks`). |
| `max_total_time` | 90s | Abort loop if cumulative time exceeds 90s. |
| Read-only allowlist | `PLANNING_TOOLS` | Any tool name not in this list is rejected by `execute_readonly_tool()`. |
| Result size cap | 8,000 chars | Truncate individual tool results before appending to messages. |
| Path containment | Working directory only | Files outside the project are unreadable. |
| Secret pattern block | `.env`, `*.key`, `*secret*` | Blocked at the tool executor level. |

---
