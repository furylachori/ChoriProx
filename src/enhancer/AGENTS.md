# `src/enhancer/AGENTS.md` — `^^imp` (DEAD CODE PATH)

> The `^^imp` infrastructure exists but **the feature is currently dead.**
> `ProxyFeatures.enhancer_enabled` does NOT exist as a field on the
> dataclass, so `dispatch.py:80`'s gate `getattr(features, "enhancer_enabled", False)`
> always returns False. Every `^^imp` invocation falls through to the
> next dispatcher (command / skill).

## Module Purpose

Owns the `^^imp` prompt-enhancer trigger — if it were live. Currently:
- `enhancer.is_imp_trigger(text)` detects the trigger.
- `enhancer.dispatch_imp(session_id, text)` would build context, call SOTA,
  format the response.
- BUT: the `try_enhancer` gate in `dispatch.py` always returns `None` (no
  `enhancer_enabled` field on `ProxyFeatures`).

## Files in this module

| File | LOC | Purpose |
|---|---|---|
| `__init__.py` | 66 | `is_imp_trigger(text)`, `extract_raw_prompt(text)`, `dispatch_imp(session_id, text)` + zero-width-stripping helpers |
| `core.py` | 110 | `build_context`, `call_sota`, `format_response`, `format_error` — the live path IF the gate ever opens |
| `prompts.py` | 58 | `ENHANCEMENT_SYSTEM_PROMPT`, `build_enhancement_prompt`, `sanitize_message` |

## Public API

```python
from enhancer import is_imp_trigger, dispatch_imp

# Detection
if is_imp_trigger(user_text):
    response = dispatch_imp(session_id, user_text)
```

### `is_imp_trigger(text)` — zero-width-stripping + regex

```python
_RAW_IMP_TRIGGER = re.compile(r'\A\s*\^{2}imp(?:\s+|$)', re.IGNORECASE)
_ZWSP_CHARS = ("\u200b", "\u2060", "\ufeff")  # BOM, ZWSP, word joiner

def _normalize_text(text):
    """Strip leading zero-width/word-joiner/BOM chars."""
    if not text:
        return text
    i = 0
    while i < len(text) and text[i] in _ZWSP_CHARS:
        i += 1
    return text[i:]

def is_imp_trigger(text):
    if not text:
        return False
    return _RAW_IMP_TRIGGER.search(_normalize_text(text)) is not None
```

`\A` matches absolute start (NOT after newlines). M6 #4: zero-width
spaces and BOMs from chat clients (Slack, iMessage) don't break detection.

### `extract_raw_prompt(text)`

```python
def extract_raw_prompt(text):
    normalized = _normalize_text(text)
    match = _RAW_IMP_TRIGGER.search(normalized)
    if not match:
        return ""
    raw = normalized[match.end():].strip()
    return sanitize_message(raw)  # NFKC + escape + control-char strip
```

### `dispatch_imp(session_id, text)` (line 46)

```python
def dispatch_imp(session_id, text):
    raw_prompt = extract_raw_prompt(text)
    if not raw_prompt:
        return format_error("No prompt to enhance. Usage: ^^imp <your prompt>")

    system_prompt, user_prompt = build_context(session_id, raw_prompt)

    try:
        enhanced = call_sota(system_prompt, user_prompt)
    except Exception:
        raise  # let specific errors bubble up to app.py handler

    if not enhanced:
        return format_error("Enhancer returned empty response. Sending original.")

    return format_response(enhanced)
```

### `call_sota(...)` (core.py:72)

Routes through `clients.sota_client.call_sota_extraction` — the layering
firewall. Default 15s timeout, 512 max tokens, temperature 0.2.

### `build_context(session_id, raw_prompt)` (core.py:53)

Assembles:
- Summary from `memory.db.get_summary`
- 3 recent messages from `memory.db.get_last_messages`
- `ENHANCEMENT_SYSTEM_PROMPT` + `build_enhancement_prompt(...)`

All inputs are sanitized via `sanitize_message` (NFKC + escape +
control-char strip). Fail-closed on `OperationalError` (raises
`ValueError("System busy. Please try again later.")`).

## Why this is dead code

`src/proxy/pipeline/dispatch.py:80` calls:

```python
enhanced = try_enhancer(ctx, features)
if enhanced is not None:
    return enhanced
```

`try_enhancer` checks:

```python
def try_enhancer(ctx, features):
    if not getattr(features, "enhancer_enabled", False):
        return None
    # ... enhancer code path ...
```

But `ProxyFeatures` (in `clients/orchestrator_config.py:28`) has no
`enhancer_enabled` field:

```python
@dataclass(frozen=True)
class ProxyFeatures:
    version: int = 1
    intercepts_memory: bool = True
    runs_tool_loop: bool = True
    injects_context: bool = True
    replaces_system_prompt: bool = True
    memory_tool_names: frozenset = _memory_tool_names_default
    max_tool_iterations: int = 15
    ticket_mode_enabled: bool = True
    tool_loop_timeout: int = 90
    intercepts_commands: bool = True
    task_commands_enabled: bool = True
    # NO enhancer_enabled field
```

So `getattr(features, "enhancer_enabled", False)` always returns `False`,
and `try_enhancer` always returns `None`. The `^^imp` trigger falls
through to the command / skill dispatcher.

## Rate limiter (configured but unused)

`enhancer_rate_limiter = RateLimiter(max_requests=500, window_seconds=60, name="enhancer", persist=True)`
defined in `proxy/state.py:132`. Auto-discovered by
`reset_all_rate_limiters()`. **Never checked in dispatch.py** because the
gate always returns False.

## Key Invariants

1. **`try_enhancer` always returns None in production** — `getattr(features, "enhancer_enabled", False)` is always False.
2. **`is_imp_trigger` strips zero-width characters** before regex match. Bare `^^imp` (no args) matches; `^^imp <text>` matches.
3. **`dispatch_imp` calls `sanitize_message` at extraction time** (single escape — no double-escape downstream).
4. **`call_sota` routes through `clients.sota_client.call_sota_extraction`** — the layering firewall. Never reach into `memory/sota_*` directly.
5. **`ENHANCEMENT_SYSTEM_PROMPT` + `build_enhancement_prompt`** are in `prompts.py`. Sanitization is per-message, not per-batch.

## Landmines

- **The feature is DEAD.** Adding `enhancer_enabled` to `ProxyFeatures` and setting it to True would re-enable `^^imp`. Do NOT do this casually — it's a prompt-rewriting feature that the proxy has been running without.
- **`call_sota` failure is silent** — `dispatch_imp` re-raises (line 60), letting `app.py` handler convert to 500. The previous code (before commit `M7-A1`) called `clients.get_client()` with no args and caught `TypeError`, which silently swallowed → every `^^imp` returned 500 with `RuntimeError("SOTA client unavailable")`.
- **`_get_recent_messages` truncates AFTER escape** (line 48 of core.py). 1000 chars post-escape. Pre-escape length doesn't bound.
- **`_TRUNCATE_AFTER_ESCAPE = 1000`** (line 20 of core.py) — applies to summary + each message + raw_prompt.
- **`OperationalError` fail-closed** (line 36 of core.py) — SQLite busy raises `ValueError("System busy. Please try again later.")`. The dispatcher propagates this as a 500.
- **Zero-width char detection only at start** — `_normalize_text` strips leading ZWSP/BOM only. Mid-message ZWSP is NOT stripped (would corrupt canary text).
- **Use case sensitivity from `re.IGNORECASE`** — `^^imp` matches but so does `^^IMP` / `^^Imp`. This is intentional.

## Code Conventions

```python
# CORRECT — detect ^^imp
from enhancer import is_imp_trigger, dispatch_imp
if is_imp_trigger(user_text):
    response = dispatch_imp(session_id, user_text)

# WRONG — reach into memory/* for SOTA directly
from memory.sota_channel import SOTAChannel  # layering violation
channel = SOTAChannel(mode=SOTAMode.EXTRACTION, ...)
channel.ask(prompt)

# CORRECT — use the layering firewall
from clients.sota_client import call_sota_extraction
result = call_sota_extraction(prompt, system_prompt=sp, session_id="my_session")

# CORRECT — sanitize at extraction time (single escape)
raw = extract_raw_prompt(text)  # already sanitized
```

## External Dependencies

**Imports from:**
- `proxy.openai_compat.build_openai_response` — synthetic response builder.
- `memory.db.connection`, `memory.db.get_summary`, `memory.db.get_last_messages` — context assembly.
- `clients.sota_client.call_sota_extraction` — SOTA layering firewall.
- `enhancer.prompts.ENHANCEMENT_SYSTEM_PROMPT`, `enhancer.prompts.build_enhancement_prompt`, `enhancer.prompts.sanitize_message`.

**Imported by:**
- `src/proxy/pipeline/dispatch.py` — `try_enhancer` gate (line 80).
- Tests under `tests/unit/` (mostly negative tests since the feature is dead).

## Cross-References

- `src/clients/AGENTS.md` — `clients/sota_client.py` is the layering firewall.
- `src/proxy/pipeline/dispatch.py` — `try_enhancer` gate that always returns None.
- `src/proxy/AGENTS.md` — `enhancer_rate_limiter` (configured but unused).
- `src/commands/AGENTS.md` — `^^imp` is NOT a command (skipped in `parse_command`); it's a separate dispatcher.

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
