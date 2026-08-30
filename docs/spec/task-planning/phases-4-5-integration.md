# Task Planning v2 — Phases 4-5: Channel Extension & Integration

> **Depends on:** All prior phases ([1](./phase-1-context.md) → [2](./phase-2-tools.md) → [3](./phase-3-loop.md))
> **Outputs:** `src/memory/sota_channel.py` (modify), `src/commands/task_commands.py` (modify)
> **Next:** Return to [Overview](../task-planning-v2.md) for acceptance criteria

---

## 7. Phase 4: SOTAChannel Extension (`sota_channel.py`)

### 7.1 New Method: `ask_multi()`

```python
# src/memory/sota_channel.py

from typing import Optional, Union

class SOTAChannel:
    # ... existing __init__ and ask() ...

    def ask_multi(
        self,
        messages: list[dict],
        session_id: str = "",
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> SOTAResult:
        """Multi-turn SOTA call with an explicit message history.

        Unlike ask(), which builds a hardcoded 2-message array,
        ask_multi() accepts the full message list, allowing the caller
        to evolve the conversation across tool turns.

        Parameters:
            messages: Full message array (system + user + assistant + tool).
            session_id: For logging and circuit-breaker context.
            tools: Override the instance tools for this call only.
            tool_choice: Override for this call only. Accepts:
                - "auto" → model decides when to call tools (default)
                - "none" → disable tools
                - {"type": "function", "function": {"name": "add_tasks"}} → force a tool
            temperature: Override instance default.
            max_tokens: Override instance default.
            reasoning_effort: Override instance default.

        Returns:
            SOTAResult where `raw` is the parsed message dict:
                {"content": str|None, "tool_calls": list|None}
        """
        if not messages:
            raise ValueError("messages cannot be empty")
        if not any(m.get("role") == "system" for m in messages):
            raise ValueError("messages must contain a system message")

        # Validate only the last user message, not tool results
        last_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_content = m.get("content", "")
                break
        _validate_input(last_user_content)

        actual_tools = tools if tools is not None else self._tools
        actual_tool_choice = tool_choice if tool_choice is not None else self._tool_choice
        actual_temp = temperature if temperature is not None else self._temperature
        actual_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        actual_reasoning = reasoning_effort if reasoning_effort is not None else self._reasoning_effort

        return self._call_sota_with_retries(
            messages=messages,
            tools=actual_tools,
            tool_choice=actual_tool_choice,
            temperature=actual_temp,
            max_tokens=actual_max_tokens,
            reasoning_effort=actual_reasoning,
            session_id=session_id,
        )

    def ask(self, prompt: str, session_id: str = "") -> SOTAResult:
        """Single-shot wrapper around ask_multi(). Maintains backward compatibility."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.ask_multi(
            messages=messages,
            session_id=session_id,
            tools=self._tools,
            tool_choice=self._tool_choice,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            reasoning_effort=self._reasoning_effort,
        )

    def _call_sota_with_retries(
        self,
        messages: list[dict],
        tools: Optional[list],
        tool_choice: Optional[Union[str, dict]],
        temperature: Optional[float],
        max_tokens: int,
        reasoning_effort: Optional[str],
        session_id: str,
    ) -> SOTAResult:
        """Shared SOTA call with retry, circuit-breaker, and response cleaning."""
        from memory._sota_http import get_sota_client
        from memory._sota_rate_limiter import claim_slot as _claim_slot, _SOTA_RETRY_DELAY as _sota_retry_delay

        if not _sota_breaker.can_execute():
            raise CircuitBreakerOpen(
                f"[sota] Circuit is {_sota_breaker.state.value}. Try again after 120s."
            )

        client = get_sota_client()
        _elapsed = 0.0
        usage = None

        with _claim_slot():
            for attempt in range(5):
                try:
                    _call_start = time.monotonic()
                    extra_body = {"reasoning_effort": reasoning_effort}

                    response = client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body=extra_body,
                        tools=tools if tools else None,
                        tool_choice=tool_choice if tool_choice else None,
                    )

                    _elapsed = time.monotonic() - _call_start

                    if not response.choices or response.choices[0].message is None:
                        raise SOTAParseError(
                            f"SOTA response missing choices or message (attempt {attempt + 1}/5)"
                        )

                    message = response.choices[0].message
                    cleaned_content = _clean_sota_response(message.content or "")

                    raw = {
                        "content": cleaned_content,
                        "tool_calls": [],
                    }

                    if message.tool_calls:
                        for tc in message.tool_calls:
                            raw["tool_calls"].append({
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            })

                    parsed = self._parse_fn(raw) if self._parse_fn else raw
                    _sota_breaker.on_success()
                    usage = getattr(response, "usage", None) if hasattr(response, "usage") else None
                    break

                except SOTAParseError:
                    logger.warning("[sota] Parse error (attempt %d/5)", attempt + 1)
                    if attempt == 4:
                        raise
                    time.sleep(_sota_retry_delay * (2 ** attempt) + random.uniform(0, 1))

                except (OpenAITimeout, OpenAIAPIError) as e:
                    status = getattr(e, "status_code", None)
                    logger.warning("[sota] API error (attempt %d/5): %s", attempt + 1, e)
                    if status == 429:
                        retry_after = parse_retry_after(
                            getattr(getattr(e, "response", None), "headers", {}).get("Retry-After"),
                            default=2.5
                        )
                        time.sleep(retry_after)
                        continue
                    if status in (502, 503, 504):
                        sleep = _sota_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(sleep)
                        if attempt == 4:
                            _sota_breaker.on_failure()
                            raise SOTAAPIError(f"Server error {status} after 5 attempts", status_code=status)
                        continue
                    if isinstance(e, OpenAITimeout):
                        if attempt == 4:
                            _sota_breaker.on_failure()
                            raise SOTATimeoutError("API call timed out")
                        time.sleep(_sota_retry_delay)
                        continue
                    if attempt == 4:
                        _sota_breaker.on_failure()
                        raise SOTAAPIError(str(e), status_code=status)
                    time.sleep(_sota_retry_delay)

        return SOTAResult(
            raw=raw,
            parsed=parsed,
            elapsed_ms=_elapsed * 1000,
            attempts=attempt + 1,
            model=self._model,
            usage=usage,
        )
```

### 7.2 `ask()` Becomes a Thin Wrapper

```python
    def ask(self, prompt: str, session_id: str = "") -> SOTAResult:
        """Single-shot wrapper around ask_multi(). Maintains backward compatibility."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.ask_multi(
            messages=messages,
            session_id=session_id,
            tools=self._tools,
            tool_choice=self._tool_choice,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            reasoning_effort=self._reasoning_effort,
        )
```

### 7.3 Per-Call `tool_choice` Override Explained

SOTA APIs (OpenAI-compatible) support three values for `tool_choice`:

| Value | Behavior | Use in Planner |
|-------|----------|----------------|
| `"auto"` | Model decides whether to call a tool or respond with text. | **Used on every planning iteration.** Lets the model explore via file reads, then decide when it's ready to emit `add_tasks`. |
| `"none"` | Model must not call tools. | Not used in planner. Useful for testing or content-only fallback. |
| `{"type": "function", "function": {"name": "add_tasks"}}` | Model **must** call `add_tasks`. | **Optional fallback** after the final tool result if the model still hasn't emitted `add_tasks` and `force_add_tasks_on_miss=True`. |

The planner's default is `auto` on every iteration. The `force_add_tasks_on_miss` flag (default `False`) controls whether one final forced iteration is allowed. This flag is intentionally conservative: if the model refused to plan after exploring, we should surface that to the user rather than coerce a low-quality plan.

---

## 8. Phase 5: Integration (`task_commands.py`)

### 8.1 Updated `plan_tasks()` Flow

```python
# src/commands/task_commands.py

import re
import time
import logging
import threading
from commands import register, make_command_response, CommandContext
from proxy.session_state import store_proposal, PendingProposal, ProposedTask
from memory.sota_channel import SOTAChannel, SOTAParseError, SOTATimeoutError, SOTAAPIError, CircuitBreakerOpen
from memory.sota_channel import SOTAMode

from planning.task_context import assemble_context
from planning.task_loop import run_planning_loop, PlanningLoopResult

logger = logging.getLogger(__name__)

# Extract working directory from main proxy system prompt (regex fallback)
_WORKING_DIR_RE = re.compile(r"Working directory:\s*(.+)", re.I)


def _extract_working_dir(system_prompt_text: str) -> str:
    m = _WORKING_DIR_RE.search(system_prompt_text or "")
    return m.group(1).strip() if m else "/path/to/choriprox"


def _build_planning_system_prompt(working_dir: str) -> str:
    return f"""You are a task planner for a software engineering assistant.

Working directory: {working_dir}

Given a description of work and optional session context, break it into actionable, concrete tasks. You may use the provided read-only tools to inspect the codebase, chat history, or external resources before finalizing your plan.

Rules:
- Each task should be small enough to complete in one focused session.
- Use priority levels: high (urgent/blocking), medium (standard), low (nice-to-have).
- Tag tasks that can be done in parallel with the same alphabetic tag (A, B, C...).
- Tasks with different tags are independent; same tag = same dependency group.
- Do NOT ask clarifying questions — make reasonable assumptions and propose directly.
- Include a 1-2 sentence summary for each task.
- Task names should be short and descriptive (max ~60 chars).

Tool usage guidelines:
- Use file_glob and read_file to understand the current codebase state.
- Use code_search to find relevant existing code.
- Use web_fetch for external documentation or APIs.
- Use recall/search to check prior session memory (available when memory tools are enabled).
- Call add_tasks ONLY when you have enough information to produce a concrete plan.
- If the request is ambiguous and you cannot find relevant files, still propose your best guess.
"""


TASK_PLANNER_PROMPT = _build_planning_system_prompt("/path/to/choriprox")

# ADD_TASKS_TOOL schema remains identical to current implementation
# (moved to task_tools.py for cohesion)


@dataclass
class _ProposedTask:
    name: str
    priority: str
    summary: str = ""
    tag: Optional[str] = None


_task_planner = None
_planner_lock = threading.Lock()


def _get_planner() -> SOTAChannel:
    global _task_planner
    if _task_planner is None:
        with _planner_lock:
            if _task_planner is None:
                _task_planner = SOTAChannel(
                    mode=SOTAMode.PLANNING,
                    system_prompt=TASK_PLANNER_PROMPT,
                    tools=None,          # tools passed per-call via ask_multi()
                    tool_choice=None,    # overridden per-call
                    temperature=0.3,
                    max_tokens=4096,
                    parse_fn=None,       # parsing handled by task_loop
                )
    return _task_planner


def plan_tasks(session_id: str, description: str, system_prompt_hint: str = "") -> dict:
    """Call SOTA to plan tasks for a user description, store proposal in session state."""
    if not description or not description.strip():
        return make_command_response("Usage: ^^task <description of work>")

    working_dir = _extract_working_dir(system_prompt_hint)

    try:
        # Phase 1: assemble context
        ctx = assemble_context(
            session_id=session_id,
            user_description=description,
            working_dir=working_dir,
            max_messages=50,
        )

        # Phase 2: run multi-turn planning loop
        planner = _get_planner()
        loop_result: PlanningLoopResult = run_planning_loop(
            sota=planner,
            messages=ctx["messages"],
            working_dir=working_dir,
            session_id=session_id,
            max_iterations=5,
            max_total_time=90.0,
            force_add_tasks_on_miss=False,
        )

        if loop_result.finish_reason not in {"add_tasks", "add_tasks_forced"}:
            logger.warning(
                "^^task planning did not converge for session %s: finish_reason=%s",
                session_id, loop_result.finish_reason
            )
            return make_command_response(
                "I wasn't able to build a solid plan from the available context. "
                "Try rephrasing or providing more details."
            )

        proposals = [_ProposedTask(**t) for t in loop_result.parsed]

    except (SOTATimeoutError, SOTAAPIError, SOTAParseError, ValueError, CircuitBreakerOpen) as e:
        logger.warning("^^task SOTA failure for session %s: %s", session_id, e)
        return make_command_response("Planning unavailable, try again later.")

    # M7: Deduplicate within batch by normalized name
    seen = set()
    deduped = []
    for pt in proposals:
        norm = pt.name.lower().strip()
        if norm in seen:
            logger.warning("Dropping duplicate task name '%s' in batch for session %s", pt.name, session_id)
            continue
        seen.add(norm)
        deduped.append(pt)

    # Store in session state for approval
    pending = PendingProposal(
        tasks=[
            ProposedTask(name=p.name, priority=p.priority, summary=p.summary, tag=p.tag)
            for p in deduped
        ],
        created_at=time.time(),
        last_accessed_at=time.time(),
    )
    store_proposal(session_id, pending)

    # Format response to user
    if not deduped:
        return make_command_response("The planner didn't produce valid tasks. Try rephrasing.")

    lines = ["Proposed Tasks:"]
    for idx, t in enumerate(deduped):
        tag = t.tag if t.tag else chr(65 + idx % 26)
        lines.append(f" [{tag}] {t.priority.upper()} — {t.name}")
        if t.summary:
            lines.append(f"      Summary: {t.summary}")
    lines.append("─────")
    lines.append("Reply 'yes' or 'looks good' to create these tasks,")
    lines.append("or describe changes needed.")
    return make_command_response("\n".join(lines))
```

### 8.2 Caller Update Required

The proxy layer that invokes `plan_tasks()` must pass the current system prompt (or a snippet containing the working directory line) so `_extract_working_dir()` can resolve paths correctly. If not provided, it falls back to `/path/to/choriprox`.

### 8.3 Planning Subpackage Import Map

| Importing Module | Imported Symbol | Source Module | Rationale |
|------------------|-----------------|---------------|-----------|
| `task_commands.py` | `assemble_context` | `planning.task_context` | Build token-budgeted message list for planner |
| `task_commands.py` | `run_planning_loop`, `PlanningLoopResult` | `planning.task_loop` | Execute multi-turn planning loop |
| `task_loop.py` | `SOTAChannel`, `SOTAResult` | `memory.sota_channel` | Issue SOTA calls with per-call `tool_choice` |
| `task_loop.py` | `PLANNING_TOOLS`, `execute_readonly_tool` | `planning.task_tools` | Tool schemas and read-only execution sandbox |
| `task_context.py` | `get_summary`, `get_last_messages` | *(DB layer, e.g. `memory.db`)* | Fetch session history for context assembly |
| `task_tools.py` | `MEMORY_TOOL_DEFINITIONS` | `memory.tools` | Re-export `recall`/`search` tool schemas when enabled |

**Removed in this revision:** `from planning.task_tools import ADD_TASKS_TOOL` in `task_commands.py` — the schema is now bundled in `PLANNING_TOOLS` and passed per-call by `task_loop.py`, making the direct import redundant.

---

## 13. Appendix A: Token Budget Worksheet

### 13.1 Step-by-Step Calculation

| Step | Formula | Result |
|------|---------|--------|
| Context window | — | 256,000 tokens |
| Usable input ratio | 80% | 204,800 tokens |
| System prompt reserve | ~500 | −500 |
| Tools schema reserve | ~1,500 | −1,500 |
| Response budget | 4,096 | −4,096 |
| Safety margin | ~500 | −500 |
| **Available input tokens** | **204,800 − 6,600** | **198,200 tokens** |

### 13.2 What 198,200 Tokens Buys (Approximate)

```
System prompt:              ~   500 tokens
Session summary:            ~ 2,000 tokens  (7,000 chars @ 2.8)
Recent messages (50 msg):   ~15,000 tokens  (avg 1,050 chars/msg @ 2.8)
User description:           ~   500 tokens  (1,750 chars @ 2.8)
                             ─────────────
Typical total:                ~18,000 tokens

Remaining budget for tool results across 4 read calls:
  ~95,800 − 18,000 = ~77,800 tokens
  At 8,000 chars max per result → ~2,857 tokens per result
  Can fit ~27 full tool results comfortably.
```

### 13.3 Overage Scenarios

| Scenario | Behavior |
|----------|----------|
| Summary alone exceeds budget | Summary is truncated to fit; messages may be reduced to min 5. |
| One message is extremely long | Its content is truncated at ~28,000 chars (8,000 tokens) before inclusion. |
| Total context slightly over | Tolerate up to 10% overage before aggressive truncation kicks in. |
