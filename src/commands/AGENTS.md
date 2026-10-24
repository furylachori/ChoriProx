# `src/commands/AGENTS.md` — `^^` Command System

> Owns the `^^` command dispatcher. Four built-in commands: `^^help`,
> `^^status`, `^^context`, `^^workspace` / `^^ws`. The `^^ticket` / `^^work`
> triggers are SKILLS, not commands — they go through the SkillRegistry + SkillDispatcher.

## Module Purpose

Provides:
- `COMMAND_PREFIX = "^^"` constant
- `COMMAND_REGISTRY` dict + `@register` decorator
- `parse_command`, `dispatch_command`, `make_command_response`
- `CommandContext` dataclass
- Built-in commands: `^^help`, `^^status`, `^^context`, `^^workspace` (via `workspace_commands.py`), `^^ws` (alias)

## Files in this module

| File | LOC | Purpose |
|---|---|---|
| `__init__.py` | 261 | Public façade — `COMMAND_PREFIX`, `COMMAND_REGISTRY`, `COMMAND_HELP`, `CommandContext`, `register`, `is_command`, `parse_command`, `make_command_response`, `dispatch_command`, `cmd_help`, `cmd_status`, `cmd_context` |
| `task_commands.py` | 19 | `is_ticket_trigger(text)` + `is_work_trigger(text)` predicates (used by TicketSkill/WorkSkill `parse_subcommand`) |
| `workspace_commands.py` | 109 | `cmd_workspace` + `cmd_workspace_alias` — dispatches to `WorkspaceSkill.handle_direct` |

## Public API (`commands/__init__.py`)

```python
COMMAND_PREFIX = "^^"
COMMAND_REGISTRY: dict[str, Callable] = {}
COMMAND_HELP: dict[str, str] = {}

@dataclass
class CommandContext:
    raw_text: str
    session_id: str
    client_hint: str = ""  # Bug #7 — threaded from RequestContext

def register(name: str, help_text: str = "") -> Callable: ...   # decorator (line 39)
def is_command(text: str) -> bool: ...                          # line 52
def parse_command(text: str) -> tuple[str, list[str]]: ...      # line 57
def make_command_response(content: str, model: str = "skill") -> dict: ...  # line 66
def dispatch_command(ctx: CommandContext) -> dict: ...          # line 72
```

### `dispatch_command` (line 72)

```python
def dispatch_command(ctx: CommandContext) -> dict:
    name, args = parse_command(ctx.raw_text)
    with _registry_lock:
        handler = COMMAND_REGISTRY.get(name)
    if handler is None:
        return make_command_response(
            f"Unknown command: {name}. Type ^^help for available commands.",
            model="^^command",
        )
    return handler(ctx, args)
```

### `parse_command` (line 57)

Lowercases the command name for case-insensitive matching. Returns `(name, args_list)`.

### `make_command_response` (line 66)

Builds a synthetic OpenAI chat completion response via
`proxy.openai_compat.build_openai_response`. Sets `model="skill"` by default
(or `"^^command"` for built-in commands).

## Built-in commands

### `^^help` (line 89)

Lists commands and skills per spec §8.1 in a dual-section layout:

```text
Available Commands:

Commands:
- ^^help     — List commands and skills
- ^^status   — Show proxy and custom-skills status
- ^^workspace — Manage workspace context
  aliases: ^^ws

Skills:
- ^^ticket — Create and manage tasks [builtin]
- ^^work   — Execute a selected task [builtin]
- ^^review — Review code [custom]
  aliases: ^^audit
```

**Critical invariants:**

- Skills section is sorted alphabetically by canonical trigger
  (`SkillDefinition.trigger`).
- `[builtin]` tag for trusted builtins (ticket, work — read via
  `SkillDefinition.is_builtin`).
- `[custom]` tag for JSON-loaded custom skills.
- Aliases render as a 2-space-indented sub-line `aliases: ^^audit`.
  Omitted when `SkillDefinition.aliases` is empty.
- `^^ticket` is NOT in `COMMAND_HELP` (the inline duplicate was REMOVED
  per spec §8.1). It appears ONLY in the Skills section.
- `^^workspace` is in `COMMAND_HELP` (it's a command, not a registered
  skill — `WorkspaceSkill` is not registered in `SkillRegistry`).
  It appears ONLY in the Commands section.
- The Skills section is OMITTED when `SkillRegistry.list_skills()`
  returns `[]` — do not render an empty `Skills:` header.
- The Commands section is ALWAYS rendered, even when `COMMAND_REGISTRY`
  is empty (failsafe — never show an empty `^^help` output).
- Both sections share the `Available Commands:\n` banner (legacy format
  preserved for clients that grep the first line).

**Implementation outline** (cmd_help at line 90):

```python
@register("^^help", "List commands and skills")
def cmd_help(ctx, args):
    lines = ["Available Commands:\n"]
    lines.append("Commands:")
    from commands import COMMAND_REGISTRY, COMMAND_HELP
    with _registry_lock:
        command_items = sorted(COMMAND_REGISTRY.items())
    for name, _handler in command_items:
        # ^^help + ^^status + ^^workspace + ^^ws render here
        desc = COMMAND_HELP.get(name, "")
        lines.append(f"- {name} — {desc}")

    from proxy.skills.skill_registry import get_default_registry
    skills = get_default_registry().list_skills()
    if skills:
        lines.append("")
        lines.append("Skills:")
        for sd in sorted(skills, key=lambda s: s.trigger):
            tag = "[builtin]" if sd.is_builtin else "[custom]"
            lines.append(f"- {sd.trigger} — {sd.description} {tag}")
            if sd.aliases:
                lines.append(f"  aliases: {', '.join(sd.aliases)}")

    return make_command_response("\n".join(lines), model="^^command")
```

The implementation reads from `SkillRegistry.list_skills()` —
NOT from `COMMAND_HELP` — so the Skills section is driven by
the live registry, not a static help dict. See
`src/proxy/skills/AGENTS.md` §"^^help registry discovery" for the full
contract and the cross-section discipline rules.

### `^^status` (line 132)

Shows proxy health and stats:
- Models loaded (from `_model_registry`)
- Total requests (from `proxy.state.request_counter`)
- Sessions in memory (from `chat_histories`)
- Tool schemas cached (size + hits + misses + hit rate)
- Log lines dropped (from `log_overflow_count`)

```python
@register("^^status", "Show proxy health and stats")
def cmd_status(ctx, args):
    from proxy.state import request_counter, tool_cache_size, tool_cache_hits, ...
    # ... build lines list
    return make_command_response("\n".join(lines), model="^^command")
```

#### Custom Skills section in `^^status` (Wave 2)

`cmd_status` reads `registry.custom_skills_status()` and appends a
`Custom Skills:` block with status, directory, discovered count, loaded
names, and restart requirement. Detailed per-file issues belong in
`/health` and startup logs, not in `^^status`.

Example output (spec §8.2):

```text
Custom Skills:
- Status: ok
- Loading: enabled
- Directory: /etc/choriprox/skills
- Discovered: 3
- Loaded: 3 (build, plan, review)
- Restart: required for changes to take effect
```

### `^^context` (registered in `__init__.py`)

Sets or reports a per-session, reduce-only context-budget override.

```python
@register("^^context", "Set the context budget the proxy enforces (in thousands of tokens)")
def cmd_context(ctx, args):
    # ^^context <N>  — set override to N*1000 tokens (200 <= N <= 1000)
    # ^^context       — report current override or "model default"
```

**Contract:**
- `^^context <N>` where `200 <= N <= 1000`, integer — sets the session's requested
  context budget to `N * 1000` tokens. Returns a confirmation. No upstream round-trip.
- `^^context` (no arg) — reports the current requested budget, or "model default" if unset.
- Rejects non-integer, out-of-range (`< 200` or `> 1000`), and `>1 arg` with a helpful error.

**Reduce-only clamp:** the raw requested tokens are stored on the Agent
(`Agent.context_limit_override`). The effective budget is
`min(override, model_cfg.context_limit)`, computed at READ time by
`proxy.context.resolve_effective_context_limit`. The command runs before route
resolution so it cannot know the model window — the clamp is deferred.

**No inheritance:** sub-agents always get `context_limit_override = None` from
`ContextOptimizeStage` (the override is never copied via canary or sub-agent
spawning). The override dies on Agent LRU eviction / restart (in-memory only).

**Agent load:** `cmd_context` loads the agent inline with lazy imports
(`from proxy.agent.agent_registry import registry as agent_registry`) — do NOT
import `proxy.pipeline.dispatch` here (circular). On a miss it calls
`memory.db.touch_session(session_id)` then retries `get_or_load`, matching the
auto-register pattern used elsewhere.

### `^^ticket` / `^^work` (NOT commands — they're skills)

`^^ticket` and `^^work` are registered `SkillDefinition`s, not commands.
They are NOT in `COMMAND_REGISTRY` and (as of Wave 5 / spec §8.1) they
are NOT in `COMMAND_HELP` either — they appear ONLY in the Skills section
of `^^help` via `SkillRegistry.list_skills()`. Before Wave 5,
`COMMAND_HELP["^^ticket"]` was set inline (line 27); that line was
REMOVED to eliminate the duplication mandated out by spec §8.1.

The `^^ticket` / `^^work` triggers are detected by `is_ticket_trigger(text)`
/ `is_work_trigger(text)` in `task_commands.py:8-17`:

```python
def is_ticket_trigger(text: str) -> bool:
    t = text.strip()
    return t == "^^ticket" or t.startswith("^^ticket ")

def is_work_trigger(text: str) -> bool:
    t = text.strip()
    return t == "^^work" or t.startswith("^^work ")
```

These predicates are called from `TicketSkill.parse_subcommand` (line 133)
and `WorkSkill.parse_subcommand` (line 103).

### `^^workspace` / `^^ws` (registered as commands via `workspace_commands.py`)

`WorkspaceSkill` is **NOT in the SkillRegistry** (R2-Issue 3, option (b) —
workspace mode was never built). The `^^workspace` command is registered
in `COMMAND_REGISTRY` and delegates to `WorkspaceSkill.handle_direct`.

`commands/workspace_commands.py`:
- `cmd_workspace(ctx, args)` — registered via `@register("^^workspace", _HELP_TEXT)`.
- `cmd_workspace_alias(ctx, args)` — registered via `@register("^^ws", _HELP_TEXT)`.

Both delegate to `_dispatch_workspace(ctx, args)` which:
1. Reconstructs the trigger text via `WorkspaceSkill.parse_subcommand`
2. Builds a minimal `SkillRequestContext` (with `session_id` + `client_hint`)
3. Calls `WorkspaceSkill.handle_direct(skill_ctx, action, arg)`

Imported at the bottom of `commands/__init__.py:262`:
```python
from commands import workspace_commands as _workspace_commands  # noqa: E402,F401
```

This import is intentional — the `@register` decorators run as a side
effect, adding `^^workspace` + `^^ws` to `COMMAND_REGISTRY` /
`COMMAND_HELP`.

## Rate limiting

`command_rate_limiter = RateLimiter(max_requests=500, window_seconds=60, name="command", persist=True)`
defined in `proxy/state.py:131`. Check before dispatch in `app.py`:

```python
from proxy.state import command_rate_limiter
result = command_rate_limiter.check(session_id)
if not result.allowed:
    return jsonify({"error": "rate_limited", "retry_after": int(result.retry_after)}), 429
```

## Key Invariants

1. **`COMMAND_PREFIX = "^^"`** is the only command prefix. Skills use `^^<name>` triggers which dispatch through `SkillRegistry` instead.
2. **`COMMAND_REGISTRY` is module-level** with `_registry_lock = threading.RLock()` (line 26). All registrations are locked.
3. **`parse_command` lowercases** for case-insensitive matching.
4. **Duplicate registration warns + overwrites** (line 43-45):
   ```python
   warnings.warn(f"Command '{name}' is already registered. Previous handler will be overwritten.", UserWarning, stacklevel=2)
   ```
5. **`^^ticket` / `^^work` are NOT in `COMMAND_REGISTRY` AND NOT in `COMMAND_HELP` (Wave 5)** — they appear ONLY in `^^help` Skills section via `SkillRegistry.list_skills()`. Spec §8.1 explicitly removed the prior `COMMAND_HELP["^^ticket"]` duplicate. `^^help`, `^^status`, and `^^context` are the command entries in `COMMAND_REGISTRY` from this module (plus `^^workspace`/`^^ws` via `workspace_commands.py`).
6. **`^^workspace` IS in `COMMAND_REGISTRY` AND in `COMMAND_HELP`** because `WorkspaceSkill` is not a registered skill. It appears in the Commands section of `^^help`, never in the Skills section. Workspace mode was never built (US-066).
7. **`make_command_response` uses `proxy.openai_compat.build_openai_response`** — synthetic OpenAI response shape.
8. **`CommandContext.client_hint`** is threaded from `RequestContext.client_hint` at dispatch (Bug #7 fix) so `^^ws <name>` populates `workspace_store._client_active` rather than only session-level state.

## Landmines

- **Duplicate registration** — `@register` decorator overwrites without raising. The `UserWarning` is the only signal.
- **Thread safety** — `_registry_lock` protects `COMMAND_REGISTRY` / `COMMAND_HELP`. Forgetting to lock during read is a race.
- **`is_command` requires more than just `^^`** — `text.startswith(COMMAND_PREFIX) and len(text) > len(COMMAND_PREFIX)`. Bare `^^` doesn't trigger.
- **`^^ticket` is NOT in `COMMAND_HELP` AND NOT in `COMMAND_REGISTRY` (Wave 5)** — spec §8.1 removed the prior inline `COMMAND_HELP["^^ticket"]`. `^^help` discovers `^^ticket` from `SkillRegistry.list_skills()` (Skills section, `[builtin]` tag); `dispatch_command` returns "Unknown command: ^^ticket" if invoked via the command path (which never happens in production — `app.py` routes `^^ticket` to the skill dispatcher first).
- **`workspace_commands.py` import side effect** — the `from commands import workspace_commands` at the bottom of `__init__.py:262` registers `^^workspace` + `^^ws` on first import. If you import `commands.task_commands` before `commands`, you won't see those commands registered.
- **`raw_text` reconstruction** — `cmd_workspace` uses `ctx.raw_text` (the original trigger text) to call `WorkspaceSkill.parse_subcommand`, preserving the `^^workspace` vs `^^ws` alias.
- **`TaskCommands` is predicates only** — `task_commands.py:19` is `is_ticket_trigger` / `is_work_trigger`. The actual command handling is in `TicketSkill.parse_subcommand` / `WorkSkill.parse_subcommand`.

## Code Conventions

```python
# CORRECT — register a new command
from commands import register, make_command_response, CommandContext

@register("^^mycommand", "Description for ^^help")
def cmd_mycommand(ctx: CommandContext, args: list[str]) -> dict:
    return make_command_response("output text", model="^^command")

# CORRECT — use thread-safe dispatch
from commands import dispatch_command
ctx = CommandContext(raw_text=text, session_id=sid, client_hint=client_hint)
response = dispatch_command(ctx)

# WRONG — direct COMMAND_REGISTRY access bypassing the lock
COMMAND_REGISTRY["^^help"] = my_handler  # race with concurrent dispatch

# CORRECT — check rate limit BEFORE dispatch
from proxy.state import command_rate_limiter
result = command_rate_limiter.check(session_id)
if not result.allowed:
    return jsonify({"error": "rate_limited", "retry_after": int(result.retry_after)}), 429
```

## External Dependencies

**Imports from:**
- `proxy.openai_compat.build_openai_response` — synthetic response builder.
- `proxy.state.command_rate_limiter` — rate limiting (used by `app.py`).
- `proxy.state.request_counter`, `tool_cache_size`, `tool_cache_hits`, etc. — status command stats.
- `proxy.skills.skill_definition.SkillRequestContext` — workspace command context.
- `skills.workspace.WorkspaceSkill` — workspace command dispatcher.

**Imported by:**
- `src/proxy/app.py` — `_require_api_key` check + command dispatch.
- `src/proxy/pipeline/dispatch.py` — `try_command` priority handler.
- `src/skills/ticket_skill.py` / `work_skill.py` — `is_ticket_trigger` / `is_work_trigger` predicates.
- Tests under `tests/unit/test_commands_v1.py`, `tests/proxy/`.

## Cross-References

- `src/skills/AGENTS.md` — TicketSkill / WorkSkill / WorkspaceSkill handle the actual logic.
- `src/proxy/skills/AGENTS.md` — SkillRegistry registers TicketSkill + WorkSkill but NOT WorkspaceSkill.
- `src/proxy/AGENTS.md` — `command_rate_limiter` is part of the 8-limiter pool.
- `src/enhancer/AGENTS.md` — `^^imp` is a separate dispatcher (DEAD CODE — always returns None).

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
