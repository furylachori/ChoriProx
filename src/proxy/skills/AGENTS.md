# `src/proxy/skills/AGENTS.md` — Skill Infrastructure

> Owns the SkillRegistry singleton, the SkillDefinition/BaseSkill/TrustToken
> types, the TrustToken gate, and the custom JSON skills loader. The
> concrete skill classes (TicketSkill/WorkSkill/WorkspaceSkill) live in
> `src/skills/AGENTS.md`.

## Module Purpose

Provides the runtime machinery for skills: registry, trust gate, dataclass
types for skill definitions, the base class, the JSON-file loader for
user-defined custom skills, and safe I/O for skill-related files.

## Files in this module

| File | Purpose | Key lines |
|---|---|---|
| `__init__.py` | Re-exports `SkillRegistry`, `get_default_registry`, `reset_default_registry`, `_default_registry`, `_held_trust_token` | |
| `skill_registry.py` | `SkillRegistry` (frozen dataclass, singleton via `__new__` shim) + `get_default_registry` + `populate_builtins` + `reset_default_registry` + `custom_skills_status` + `SkillContext` (legacy) + `SkillMeta` + `TRUSTED_TRUST_CALLERS` + `_ZERO_WIDTH_RE` + `_normalize_text` + `error_response` + `SkillActivation` (688 LOC); delegates JSON loading to `custom_skill_loader.load_custom_skills` | line 201 `_default_registry` singleton; line 227 `_registry_init_lock`; line 233 `_skill_registry_new__` recursion-break; line 250 thin `__new__` shim; line 262 `SkillRegistry` class; line 329 `custom_skills_status`; line 399 `populate_builtins`; line 428 `populate_builtins` calls `custom_skill_loader.load_custom_skills`; line 613 `get_default_registry`; line 646 `reset_default_registry` |
| `skill_definition.py` | `SkillDefinition` (frozen dataclass, now includes the REQUIRED `description` field), `BaseSkill` ABC, `SkillRequestContext`, `ModeState` (re-exported), `ProxyAccessLevel`, `TrustToken`, `_TRUSTED_CALL_SITES`, `_is_exempt_from_callsite_check`, `_held_trust_token`, `TRUSTED_BUILTIN_SKILLS`, `is_trusted_skill`, `SkillRegistrationError`, `SkillValidationError`, `ToolCapability`, `CapabilityTable` (1438 LOC) | line 66 `_TRUSTED_CALL_SITES`; line 71 `_is_exempt_from_callsite_check`; line 85 `TrustToken`; line 287 `TRUSTED_BUILTIN_SKILLS`; line 290 `is_trusted_skill`; line 310 `SkillRegistrationError`; line 320 `SkillValidationError`; line 343 `ProxyAccessLevel`; line 363 `_held_trust_token`; line 388 `SkillRequestContext`; line 457 `ToolCapability`; line 473 `CapabilityTable`; line 672 `SkillDefinition` (note: REQUIRED `description` field); line 940 `_skill_definition_setattr`; line 967 `BaseSkill` |
| `path_utils.py` | `get_skills_dir() -> Optional[Path]` + `read_contained_regular_file(root, relative_path, *, max_bytes) -> str` + `SkillPathError(OSError)` — descriptor-based safe I/O chokepoint for skill files (replaces the prior `resolve_skill_path` + reopen pattern); 226 LOC | module-level `SkillPathError(OSError)` line 43; `get_skills_dir` line 60; `read_contained_regular_file` line 86 |
| `command_text_utils.py` | `strip_environment_details(text)` + `extract_command_text(raw_text)` — ephemeral CLI-framework tag stripping for `^^` command parsing | line 30 `_FRAMEWORK_TAGS` |
| `custom_skill_loader.py` | Strict-schema JSON loader for custom skills — `ParsedCustomSkill`, `SkillLoadIssue`, `CustomSkillsStatus`, `SkillRegistryConfigurationError`, `LoadResult`, `ResolvedSkillsDir`, `resolve_skills_dir`, `discover_custom_skill_files`, `parse_custom_skill_file`, `load_custom_skills`, `build_custom_skill_definition`, default-callback factories (`_default_*`); exposes `PERMANENTLY_DENIED_FOR_CUSTOM = frozenset({"proxy_ticket", "proxy_work"}); CONDITIONALLY_ADMITTED_FOR_CUSTOM = frozenset({"task"})`; 1663 LOC | line 55 `PERMANENTLY_DENIED_FOR_CUSTOM`; line 102 `ParsedCustomSkill`; line 121 `SkillLoadIssue`; line 136 `CustomSkillsStatus`; line 155 `SkillRegistryConfigurationError`; line 173 `LoadResult`; line 187 `ResolvedSkillsDir`; line 200 `resolve_skills_dir`; line 922 `discover_custom_skill_files`; line 989 `parse_custom_skill_file`; line 1302 `load_custom_skills`; line 1591 `build_custom_skill_definition` |

## SkillRegistry — singleton with double-checked locking

```python
# Public API
from proxy.skills.skill_registry import (
    SkillRegistry,            # dataclass — instances always via get_default_registry
    get_default_registry,     # function — returns the module singleton
    reset_default_registry,   # function — clears the singleton (test-only)
    SkillContext,             # legacy Phase 1a context (use SkillRequestContext for new code)
    SkillMeta,                # public metadata for legacy consumers
    error_response,           # error envelope helper
    SkillActivation,          # enum: COMMAND | AUTO | INHERITED
)
```

The singleton lives at module level (`_default_registry` line 201).
`get_default_registry()` uses double-checked locking on `_registry_init_lock`
(line 227) and constructs via `_skill_registry_new__` (line 233) which
bypasses the thin `__new__` shim (line 250) to avoid infinite re-entry.

`SkillRegistry.__init__` is a guard: if `_initialized` is True, no-op
(line 290). Otherwise calls `__post_init__` (line 308) which sets
`_skills = MappingProxyType(self._inner)` and `_initialized = True`.

`reset_default_registry()` (line 646) — test-only. Clears the OLD registry's
internal state before nulling the global ref to prevent split-brain (FIX-4).

## `populate_builtins(*, trusted=True)` — startup contract

Called once at startup by `get_default_registry()`. Idempotently registers
`TicketSkill` and `WorkSkill` (NOT `WorkspaceSkill` — see R2-Issue 3, option
(b)). Returns the count of NEW registrations. JSON custom skills are
loaded by delegating to `custom_skill_loader.load_custom_skills` (atomic
publication — see [Custom JSON skills loader](#custom-json-skills-loader--srcproxskillscustom_skill_loaderpy)).

```python
count = 0
for factory_class in (TicketSkill, WorkSkill):
    factory = factory_class()
    if factory.name in self._inner:
        continue
    with _held_trust_token(TrustToken.for_trusted_caller()):
        sd = factory.to_skill_definition()
    self.register_skill(sd, trusted_registrant=TrustToken.for_trusted_caller())
    count += 1
# Custom JSON skills (Wave 1) — delegated to the extracted loader module.
resolved = resolve_skills_dir()
parsed_skills, issues, status = load_custom_skills(
    resolved.root, tool_catalog=tool_registry, source=resolved.source
)
object.__setattr__(self, "custom_status", status)
if issues:
    raise SkillRegistryConfigurationError(issues=tuple(issues), message=...)
# Atomic publication: build every definition, then register the batch.
sds = tuple(build_custom_skill_definition(p) for p in parsed_skills)
self._register_validated_batch(sds)
```

## TrustToken — confused-deputy defense

`TrustToken` is the unforgeable marker for `register_skill(..., trusted_registrant=...)`.
The only way to obtain a TRUSTED instance is `TrustToken.for_trusted_caller()`,
which inspects the call stack to verify the caller's module `__name__` is in
`TRUSTED_TRUST_CALLERS` AND (for production callers) the immediate caller's
function name (`co_name`) is in `_TRUSTED_CALL_SITES`.

### `TRUSTED_TRUST_CALLERS` (skill_registry.py:35-49)

```python
TRUSTED_TRUST_CALLERS: frozenset[str] = frozenset({
    "__main__",
    "proxy.agent.agent_registry",
    "proxy.skills.skill_registry",
    "tests.test_skill_dispatch",
    "tests.test_skill_definition",
    "tests.skills.test_skill_registry",
    "tests.skills.test_ticket_skill",
    "tests.skills.test_work_skill",
    "tests.unit.test_mode_resolution",
    "tests.proxy.test_skill_mode",
    "tests.proxy.test_canary_repro",
    "tests.proxy.test_skill_registry",
    "tests.proxy.test_trust_gate",
    "tests.integration.test_skill_dispatch_integration",
})
```

Adding a module here is a **security-relevant change** that MUST be reviewed.

### `_TRUSTED_CALL_SITES` (skill_definition.py:66-68)

```python
_TRUSTED_CALL_SITES: frozenset[str] = frozenset({
    "populate_builtins",
})
```

For production callers: the immediate function name MUST be in this set.
Wrappers inside a trusted module that mint tokens on behalf of an untrusted
caller are blocked.

Tests are exempt via `_is_exempt_from_callsite_check` (line 71) — they are
already trusted via `TRUSTED_TRUST_CALLERS`.

### `TRUSTED_BUILTIN_SKILLS = frozenset({"ticket", "work"})`

The set of skill names that are considered "trusted builtin" for `proxy_*`
access control. **Modifying this set requires BOTH a code change AND a PR
review** (spec §14 row "Trusted builtin set"). `workspace` is REMOVED from
the set (R2-Issue 3, option (b)) — workspace routing is parser-only, not a
registered skill.

`is_trusted_skill(name)` (skill_definition.py:290) is the PUBLIC lookup —
security-critical call sites MUST import it explicitly.

### `SkillRegistrationError(PermissionError)`

Raised when a skill registration is denied by the CAT-3 trust-gate.
Subclass of `PermissionError` so existing `except PermissionError` blocks
continue to work.

## `SkillDefinition` — frozen dataclass (26 fields)

Per the internal skill-system specification (§3.5). Splits
data (identity, tool filters, access levels, TTL) from behavior (10
callable fields for prompts, lifecycle, dispatch, canary).

`frozen=True` blocks whole-attribute reassignment but does NOT block the
`is_builtin` @property or `__post_init__`.

### Data fields (16)

- `name`, `trigger`, `aliases`, `description`, `ttl_seconds`, `fallback_skill`, `fire_and_forget`
- `allowed_tool_names`, `sub_agent_tool_filter`
- `sub_agent_proxy_ticket_access: ProxyAccessLevel`, `sub_agent_proxy_work_access: ProxyAccessLevel`
- `call_limits: dict`, `check_action: Callable | None`, `blocked_tools_for_subagent: frozenset`
- `capability_table: Optional[CapabilityTable]`, `subagents: Optional[SubagentPolicy]`

### Behavior fields (10 callable, all optional)

- `get_system_prompt`, `get_subagent_prompt`
- `parse_subcommand`, `handle_direct`, `post_process`, `should_deactivate`
- `get_canary`, `get_activation_rewrite`
- `on_activate`, `on_deactivate`

`BaseSkill.to_skill_definition()` reads `getattr(self, '<field>', None)` for
each field. The cached `SkillDefinition` returned from `BaseSkill` is
immutable — tests verify this (spec §5 test 17).

### `ProxyAccessLevel` enum

```python
NEVER     = "never"         # proxy_* not in tool filter
READ_ONLY = "read_only"     # list_tasks, get_task, append_review_note only
FULL      = "full"          # all actions
```

Per-skill access level for `proxy_*` tools. Resolved at runtime by
`check_sub_agent_proxy_access` (Phase 1c) to decide whether a sub-agent in
a given skill can call `proxy_ticket` or `proxy_work`.

## `BaseSkill` — ABC

Provides `to_skill_definition()` that builds a `SkillDefinition` from
class attributes + optional callable methods. The Phase 1d rewrite
collapses legacy `BaseSkill` and NEW `BaseSkill` into a single class.

## Custom JSON skills loader — `src/proxy/skills/custom_skill_loader.py`

The loader is **extracted** from `skill_registry.py` into its own module.
`SkillRegistry.populate_builtins()` (line 399) calls
`custom_skill_loader.load_custom_skills()` (line 1302) at startup, after
registering the Python builtins. **Hot-reload: NO.** Restart the proxy to
pick up changes.

### Strict schema (whole-skill rejection)

The loader rejects an entire skill on **any** of these failures:
unknown top-level key, missing required key, malformed list entry
(non-string element), duplicate name within the same configuration,
filename/name mismatch, filename/trigger-name mismatch, tool entry in
`tool_whitelist_denied`, file too large, prompt file too large, decode error,
non-UTF-8 input, BOM in input, invalid name regex, invalid trigger regex,
duplicate alias, alias without `^^` prefix, alias conflicting with another
skill's trigger/alias, TTL out of range.

Failure of any one skill is **not** a partial success — the loader raises
`SkillRegistryConfigurationError` carrying a tuple of `SkillLoadIssue`
objects describing every problem found. **No skills are published.** This is
atomic startup: either every valid skill lands in the registry or none do.

### Required JSON fields

```json
{
  "name": "<skill_name>",                    // ^[a-z][a-z0-9_-]*$
  "trigger": "^^<name>",                     // ^\\^\\^[a-z][a-z0-9_-]*$
  "description": "<1..160 chars>",           // REQUIRED; metadata only (not stored)
  "prompt": "...",                           // OR prompt_file (exactly one)
  "prompt_file": "<rel_path>",               // alternative to prompt
  "tool_whitelist": ["bash", "read", ...]    // three-state: see below
}
```

### Optional fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `aliases` | list[str] (each starts with `^^`) | `[]` | Up to 16. Must not collide with another skill's `trigger` or `alias`. |
| `ttl_seconds` | int (1..86400) | `300` | Mode TTL after activation. |

### Rejected fields (whole skill rejected)

The loader **rejects** any of these as a top-level key (a single unknown
key fails the skill — see `_check_unknown_keys` at line 882). Listed
verbatim because each is a previous-version trap:

- `fire_and_forget`, `fallback_skill`
- `subagent_tool_filter`, `subagent_proxy_ticket_access`, `subagent_proxy_work_access`
- `subagent_prompt`, `subagent_prompt_file`
- `on_activate`, `on_deactivate`
- `handle_direct`, `parse_subcommand`, `post_process`, `should_deactivate`
- `get_canary`, `get_activation_rewrite`, `get_subagent_prompt`
- `check_action`, `call_limits`, `blocked_tools_for_subagent`
- `enabled`, `tools`, `vars`, `metadata`

`description` is **REQUIRED** at the top level. Nested unknown keys
(`description_metadata`, etc.) are rejected. To customize behavior beyond
what JSON supports, define a Python `BaseSkill` subclass and register via
`registry.register_skill()`.

### Custom skill parent tool admission (spec internal documentation §0.1)

The parent tool admission decision splits into two frozen constants:

```python
PERMANENTLY_DENIED_FOR_CUSTOM = frozenset({"proxy_ticket", "proxy_work"})
CONDITIONALLY_ADMITTED_FOR_CUSTOM = frozenset({"task"})
```

`proxy_ticket` and `proxy_work` are **permanently denied**; `task` is
provisionally admitted **only** when the same JSON object declares a valid
nested `subagents` block and the deployment kill switch
`PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS=1` is set plus `PROXY_SESSION_SECRET` is
present. Sub-agents remain universally forbidden from using `task`
(universal no-nesting rule). Provisional admission becomes final only after
the whole `subagents` policy, deployment flag, signing secret, identity,
and full custom configuration validate — there is no second whitelist parser,
alternate denylist, or runtime-only exception.

### `SubagentPolicy` (skill_definition.py:249) and `SkillDefinition.subagents` (skill_definition.py:775)

`SubagentPolicy` is a frozen dataclass representing a swarm-capable
custom skill's per-request sub-agent spawning policy (spec §7.1):

```python
@dataclass(frozen=True)
class SubagentPolicy:
    tool_whitelist: frozenset[str]
    max_task_calls_per_request: int
```

Constructed **only** by the custom-skill loader; carried into
`SkillDefinition.subagents: Optional[SubagentPolicy]` and referenced per-request
via `RequestContext.swarm_policy`. Runtime MUST NOT reopen the JSON file,
reparse a raw dict, or independently validate the nested schema. The child
tool policy is `SkillDefinition.subagents.tool_whitelist` exactly (an empty
`frozenset()` means "prompt-only child / deny-all"; distinct from "no policy
configured"). `task`, `question`, `proxy_ticket`, and `proxy_work` are
**always** rejected from a child's `tool_whitelist`
(`SWARM_CHILD_FORBIDDEN`, `swarm_child_forbidden_tool` code). Unknown tools
(`swarm_unknown_tool`), duplicates (`swarm_duplicate_tool`), and
`max_task_calls_per_request` outside `[1, 16]` are startup rejects.

### Three-state tool policy (`tool_whitelist`)

| Value | Meaning |
|---|---|
| Field **omitted** (`None`) | No tool filter — LLM sees the union of all client-allowed tools. WARNING logged at load. |
| `[]` (empty list) | LLM sees **zero** tools. Use to expose a pure-text/pure-routing skill. |
| Non-empty list | LLM sees exactly those tools (subject to `PERMANENTLY_DENIED_FOR_CUSTOM` filtering). |

The three states are distinct. The dataclass field `allowed_tool_names`
can be `None`, `frozenset()`, or a non-empty `frozenset` — see
`SkillDefinition` (line 672) and `_check_tool_whitelist` (line 491).

### Strict UTF-8 + BOM rejection

`_decode_strict_utf8` (line 896) rejects input that:
- contains a UTF-8 BOM (U+FEFF) at the start of the file
- fails strict `utf-8` decode (no replacement-character fallback)
- exceeds the per-file size bound

The decoder raises `SkillLoadIssue` with `kind="json"` or `kind="prompt"`
so the operator knows which file failed.

### Size bounds

| Resource | Bound | Enforced by |
|---|---|---|
| JSON file size | 64 KiB | `_safe_read_bytes` (line 954) |
| Prompt file size | 256 KiB | `read_contained_regular_file` (`path_utils.py:86`) |
| Files per directory | 256 | `discover_custom_skill_files` (line 922) |
| Aliases per skill | 16 | `_check_aliases` (line 417) |
| Tools per skill | 64 | `_check_tool_whitelist` (line 491) |
| Description length | 1..160 chars | `_check_description` (line 307) |
| TTL | 1..86400 seconds | `_check_ttl` (line 594) |

### Reserved names

`ticket` and `work` cannot be overridden by JSON. The trust gate in
`register_skill()` rejects `SkillDefinition`s with these names from
untrusted registrants (`TRUSTED_BUILTIN_SKILLS`, line 287).

### Duplicate handling

- **Duplicate name** (two files with the same `name`): the loader
  accumulates all issues and raises `SkillRegistryConfigurationError` —
  **no skill is published** (atomic startup). See `_check_name` (line 355).
- **Duplicate trigger** (two files with different `name` but the same
  `trigger`): same — atomic failure.
- **Alias colliding with another skill's `trigger` or `alias`**: same.

### `PROXY_SKILLS_DIR` semantics (Wave 1, 2026-07)

`get_skills_dir() -> Optional[Path]` (`path_utils.py:60`) returns the path
or `None` to indicate "custom loading explicitly disabled". The test is
**membership in `os.environ`**, not truthiness:

| Environment state | Return value | Use |
|---|---|---|
| Variable absent | `<repo-root>/skills` | Default skills dir |
| `PROXY_SKILLS_DIR=""` (empty string) | `None` | Explicitly disabled |
| `PROXY_SKILLS_DIR="/abs/path"` | `Path("/abs/path")` verbatim | Override to absolute path |
| `PROXY_SKILLS_DIR="rel/path"` | `<repo-root>/rel/path` | Override relative to repo root |

The empty-string case is the **only** way to disable custom loading without
unsetting the variable. The absent case falls through to the default.
Downstream callers MUST handle `None` (skip custom loading cleanly).

### Default behavior callbacks installed by the loader

`build_custom_skill_definition` (line 1591) wires the following defaults.
Ordinary custom skills are direct-dispatch and cannot spawn sub-agents.
Swarm-capable custom skills retain direct command activation but may emit
native `task` calls under their validated request-local `SubagentPolicy`.

| Field | Default behavior |
|---|---|
| `parse_subcommand(text, name)` | Returns `("__direct__", text)` (`_default_parse_subcommand`, line 1518). |
| `handle_direct(ctx, mode, action)` | Returns `None` (`_default_handle_direct`, line 1534). |
| `post_process(ctx, response)` | Passthrough — returns `response` unchanged (`_default_post_process`, line 1538). |
| `should_deactivate(ctx, finish_reason)` | Returns `True` when the request's terminal disposition is `COMPLETE` or `FAILED` (`_default_should_deactivate`, line 1542). |
| `get_canary(ctx)` | Returns `None` (`_default_get_canary`, line 1565). |
| `get_activation_rewrite(ctx)` | Returns `"User request: <args>"` when the activation carries args, or `""` for a bare trigger (`_default_get_activation_rewrite`, line 1569). |
| `get_system_prompt(ctx)` | Returns the (post-`$KEY`/`$ARG`) prompt text (`_default_get_system_prompt`, line 1510). |
| `get_subagent_prompt(ctx)` | Returns the prompt text (line 1514). Ordinary custom skills do not spawn sub-agents. Swarm-capable custom skills do — the callback is wired for both type completeness and the swarm-parent path. |

### Quick start: adding a new custom skill

```bash
# 1. Create skills/<name>.json
cat > skills/my-skill.json <<'EOF'
{
  "name": "my-skill",
  "trigger": "^^my-skill",
  "description": "Short (≤160 chars) operator-facing description.",
  "prompt": "What the LLM should do when invoked.",
  "tool_whitelist": ["bash", "read", "glob", "grep"]
}
EOF

# 2. Restart the proxy — there is no hot-reload.

# 3. Verify with ^^my-skill test in the orchestrator.
```

### Restart-only contract

Custom skills are loaded once at proxy startup, inside
`SkillRegistry.populate_builtins()` → `custom_skill_loader.load_custom_skills()`.
To reload after editing, **restart the proxy.** Hot-reload is not in scope
(Track B from the custom-skills plan; deferred).

### Eager startup wiring (Wave 2)

`_eager_init_registry()` in `src/proxy/app.py` constructs the singleton
at process boot when called explicitly:

- CLI: `qwen_proxy.py:__main__` calls `_eager_init_registry()` BEFORE
  `init_db()`. On `SkillRegistryConfigurationError`, the CLI logs the
  diagnostic list and calls `sys.exit(1)` (operator-actionable failure
  mode per spec §5.1).
- WSGI (gunicorn `--preload`): the master process's app import triggers
  `populate_builtins()` under `_registry_init_lock`. Workers fork after.
- WSGI (without `--preload`): first request handler triggers
  construction lazily (current behavior).
- Tests: do NOT rely on import-time eager init. Use
  `monkeypatch.setenv("PROXY_SKILLS_DIR", tmp_path)` BEFORE the first
  `get_default_registry()` call, then `reset_default_registry()` between
  tests as needed.

`_registry_init_lock` (this module, line 227) is a module-local
serialization gate. It is NOT in the 15-position lock hierarchy
(`proxy/state.py:294-315`) because it is a singleton-init lock acquired
BEFORE any per-instance or per-request lock. Acquired under no other
lock; held during `populate_builtins()` file I/O. `Lock()` (not RLock)
because `populate_builtins()` is called once per process lifetime.

### Public status surface

`SkillRegistry.custom_skills_status()` (skill_registry.py:329) returns a
`CustomSkillsStatus` (custom_skill_loader.py:136) snapshot describing the
configured directory, the loaded skills, and any issues — useful for
`^^status` and debug endpoints. It does NOT mutate the registry.

`SkillRegistry.list_skills()` (skill_registry.py:386) returns a list of
every registered `SkillDefinition` — used by `^^help` (Wave 5) to render
the Skills section. Aliases are read from `SkillDefinition.aliases`
(`tuple[str, ...]`, ordered, possibly empty).

### Public discoverability surface

The two public read-only methods drive every operator-facing surface:

| Method | Used by | Returns |
|---|---|---|
| `custom_skills_status()` | `^^status` Custom Skills block, `/health` `custom_skills` block, startup logs | `CustomSkillsStatus` snapshot |
| `list_skills()` | `^^help` Skills section | `list[SkillDefinition]` |

Both are guarded by `SkillRegistry._lock` (per-instance RLock) and never
mutate the registry. `list_skills()` exposes `SkillDefinition.is_builtin`
which is how `^^help` differentiates `[builtin]` vs `[custom]` labels.

## Standalone CLI validator (spec §8.4)

`python -m proxy.skills.validate_custom_skills [--dir <path>] [--format json]`

A standalone operator-facing validator. Shares `custom_skill_loader.load_custom_skills`
with the runtime loader — **no duplicated schema logic**. Schema drift
between runtime and validator is impossible by construction.

### Exit codes

- `0` — all parsed OK
- `1` — validation failure (one or more `SkillLoadIssue`)
- `2` — invocation or directory-access failure (spec §5.1 fatal — missing
  arg, unreadable directory, `PermissionError`, etc.)

### JSON output (stdout, `--format json`)

```json
{
  "status": "ok|invalid|fatal|disabled",
  "directory": "/etc/choriprox/skills",
  "source": "env|default|arg|disabled",
  "state": "ok|degraded|disabled|error",
  "discovered": 3,
  "loaded": 3,
  "loaded_names": ["build", "plan", "review"],
  "valid": [{"file": str, "name": str, "trigger": str}],
  "invalid": [{"file": str, "code": str, "message": str}],
  "issues": [...]
}
```

The `valid`/`invalid` arrays are sufficient for CI gating. `issues` mirrors
the structured per-file diagnostics consumed by `/health` and startup logs.

### Use cases

```bash
# CI gating — fail the build on any invalid skill
python -m proxy.skills.validate_custom_skills --dir skills/ || exit 1

# Production debugging — list only the invalid files
python -m proxy.skills.validate_custom_skills --dir /etc/choriprox/skills | jq .invalid

# Migration validation before deploy
python -m proxy.skills.validate_custom_skills --dir ./examples/skills --format json | jq '.state'
```

### Argument precedence

- `--dir <path>` (flag) overrides `PROXY_SKILLS_DIR` env var.
- No `--dir` and `PROXY_SKILLS_DIR` unset: falls through to
  `<repo>/skills` (per spec §5.2; healthy-disabled when missing).
- `PROXY_SKILLS_DIR=""` (empty string) explicitly disables custom-skill
  loading (exit `0`, `status="disabled"`).

The validator reuses `custom_skill_loader.resolve_skills_dir()` so the precedence
table above is the SAME as the runtime loader's — operators never need to
mental-model two different config-resolution paths.

## ^^help registry discovery (spec §8.1)

`^^help` enumerates both the `COMMAND_REGISTRY` (commands) and the
`SkillRegistry` (skills) — see `src/commands/__init__.py:cmd_help`.

### Output format

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
- ^^review — Review code for regressions and missing tests [custom]
  aliases: ^^audit
```

### Critical invariants

- Skills section is sorted alphabetically by canonical trigger
  (`SkillDefinition.trigger`).
- `[builtin]` tag for trusted builtins (ticket, work — read from
  `SkillDefinition.is_builtin`).
- `[custom]` tag for JSON-loaded custom skills.
- Aliases render as a 2-space-indented sub-line `aliases: ^^audit` —
  omitted entirely when `SkillDefinition.aliases` is empty.
- `^^ticket` is a SKILL (not a command) and appears ONLY in the Skills
  section. The inline `COMMAND_HELP["^^ticket"]` entry was removed per
  spec §8.1 ("^^ticket is no longer duplicated in COMMAND_HELP").
- `^^workspace` is a COMMAND (not a registered skill) and appears ONLY in
  the Commands section. `WorkspaceSkill` is not registered in
  `SkillRegistry` (US-066).
- The Skills section header is OMITTED when `list_skills()` returns
  `[]` — do not render an empty `Skills:` block.
- The Commands section header is ALWAYS rendered, even when `COMMAND_REGISTRY`
  is empty (failsafe — never show an empty `^^help`).
- Both sections are wrapped under a single `Available Commands:\n` banner,
  matching the legacy header format.

### Cross-section discipline

- A trigger MUST appear in exactly one section. The Commands section is
  for entries in `COMMAND_REGISTRY`; the Skills section is for entries
  in `registry.list_skills()`. A trigger that appears in both is a bug —
  check `SkillRegistry.populate_builtins` and `commands/__init__.py` at
  the same time.
- Description text comes from `SkillDefinition.description` (skills) or
  `COMMAND_HELP[name]` (commands). The line-1 line is the trigger; the
  line-2 `aliases:` sub-line only renders for skills with non-empty
  `aliases`.

## `SkillRequestContext` (skill_definition.py:388)

Per-request skill data. Built once per request from `RequestContext`.
The pipeline does NOT pass `RequestContext` into skill callables — it
converts to `SkillRequestContext` so the skill's call surface is decoupled.

`from_request_context(ctx)` (line 417) uses `getattr` with defaults so the
helper is robust to the pipeline adding new fields.

Note: there are TWO context classes — the OLD `SkillContext` in
`skill_registry.py` (line 84, ~109 LOC) used by legacy code paths, and
`SkillRequestContext` here. New code MUST use `SkillRequestContext`.

## Key Invariants

1. **`SkillRegistry` is a singleton.** Always use `get_default_registry()`.
2. **`SkillDefinition` is frozen.** Whole-attribute reassignment raises `FrozenInstanceError`. `@property` for `is_builtin` works because properties aren't field assignments.
3. **`TRUSTED_BUILTIN_SKILLS = {"ticket", "work"}`.** Modifying requires PR review.
4. **TrustToken cannot be self-asserted.** `TrustToken(_verified)` constructor is private. Only `TrustToken.for_trusted_caller()` returns a TRUSTED instance.
5. **Custom skills are direct-dispatch with conditional spawning.** Ordinary custom skills dispatch directly. Swarm-capable custom skills additionally emit signed sub-agent canaries on `task` tool calls (`Path B` in `proxy/mode.py:581`); sub-agents carry the configured child policy.
6. **Atomic custom-skills publication.** `load_custom_skills` validates the entire configuration first; a single invalid skill raises `SkillRegistryConfigurationError` and publishes nothing. Use `custom_skills_status()` (skill_registry.py:329) to inspect the configured state without mutating the registry.
7. **`_skills = MappingProxyType(self._inner)`.** Rebuilt on every registration (CRIT R2-4 — frozen=True blocks plain `self._skills = ...` so use `object.__setattr__`).
8. **Hot-reload NOT supported.** Restart the proxy to pick up JSON changes.
9. **`_held_trust_token` thread-local.** `SkillDefinition.__post_init__` checks it to determine whether a trusted name is being constructed.
10. **`read_contained_regular_file` opens via fd, not via re-resolved path.** The leaf is opened with `O_RDONLY | O_CLOEXEC | O_NOFOLLOW` and security decisions (regular-file check, size check, BOM check, UTF-8 decode) all run on the descriptor; never on a re-resolved pathname.
11. **`description` is REQUIRED on `SkillDefinition`.** The loader's `_check_description` (custom_skill_loader.py:307) rejects skills without it.
12. **Eager-init wiring constructs the singleton before the first request.** CLI exits 1 on configuration error (Wave 2); WSGI fails the worker.

## Landmines

- **Adding to `TRUSTED_TRUST_CALLERS`** — security-relevant. PR review required.
- **Adding to `_TRUSTED_CALL_SITES`** — same. Each new production entry-point function that mints tokens must be enumerated.
- **Re-registration of builtins** — `register_skill()` rejects `TRUSTED_BUILTIN_SKILLS` names with untrusted registrants. The trust gate runs BEFORE the duplicate check (CRIT R1-2).
- **Builtin protection** — overwriting a non-`SkillDefinition` entry (legacy builtin registered via `register()`) is rejected (CRIT R1-3).
- **Three-state `tool_whitelist`** — `None` (omitted) means "no filter, all client tools visible"; `frozenset()` (empty list) means "zero tools visible"; non-empty frozenset means "exactly these tools". The old "empty list = allow all" Phase 1 semantics are GONE. Custom skills must declare an explicit list to lock down tool access.
- **`PERMANENTLY_DENIED_FOR_CUSTOM` (`proxy_ticket`, `proxy_work`)** — these tools are permanently denied to JSON skills at loader time AND enforced at runtime by the CAT-3 trust gate. See `_check_tool_whitelist` (custom_skill_loader.py:491).
- **`CONDITIONALLY_ADMITTED_FOR_CUSTOM` (`task`)** — admitted only on a valid swarm-parent (valid `subagents` block + flag `"1"` + signing secret). An ordinary custom skill that lists `task` is rejected at startup.
- **Unknown JSON key** — the loader rejects the WHOLE skill on any unrecognized top-level key (`_check_unknown_keys`, custom_skill_loader.py:882). No partial publication; atomic startup fails the entire custom configuration if any one skill is invalid. This is intentional: silent field stripping is what created Phase 1's drift.
- **Reserved builtin names** — `ticket`/`work` cannot be overridden via JSON. The trust gate rejects re-registration.
- **Alias without `^^` prefix** — the whole skill is skipped.
- **`prompt_file` outside skills dir** — `read_contained_regular_file` raises `SkillPathError` (containment check via path validation + O_NOFOLLOW descriptor open).
- **`SkillPathError` vs `SkillRegistrationError`** — `SkillPathError` is an `OSError`; `SkillRegistrationError` is a `PermissionError`. They are separate hierarchies. Do not raise one in place of the other. `SkillRegistryConfigurationError` is yet a third hierarchy (`RuntimeError`).
- **`parents[2]` in path_utils** — points at `src/`, not repo root. Must be `parents[3]`.
- **`SkillContext` (legacy) vs `SkillRequestContext`** — they coexist. New code uses the latter.
- **Worker wrapper confusion-deputy** — `_TRUSTED_CALL_SITES` blocks wrappers from minting tokens on behalf of untrusted callers.
- **BOM in JSON or prompt file** — `_decode_strict_utf8` rejects U+FEFF-prefixed input outright. Some Windows editors add BOM silently; strip it before deploy.

## Code Conventions

```python
# CORRECT — get the singleton
from proxy.skills.skill_registry import get_default_registry
registry = get_default_registry()
skill_def = registry.get_skill("ticket")

# CORRECT — construct a builtin (only in trusted callers)
from proxy.skills.skill_definition import TrustToken, _held_trust_token
from skills.ticket_skill import TicketSkill
with _held_trust_token(TrustToken.for_trusted_caller()):
    factory = TicketSkill()
    sd = factory.to_skill_definition()
registry.register_skill(sd, trusted_registrant=TrustToken.for_trusted_caller())

# WRONG — self-asserting trust
TrustToken(_verified=True)  # private constructor; raises TypeError on bool
registry.register_skill(sd, trusted_registrant=True)  # bool→TrustToken TypeError

# WRONG — bypassing the singleton
SkillRegistry()  # returns the existing singleton; do not create new instances

# CORRECT — safely read a custom skill file under PROXY_SKILLS_DIR
from proxy.skills.path_utils import (
    get_skills_dir,
    read_contained_regular_file,
)

skills_dir = get_skills_dir()  # Path | None (None if PROXY_SKILLS_DIR="")
if skills_dir is not None:
    prompt_text = read_contained_regular_file(
        skills_dir, data["prompt_file"], max_bytes=256 * 1024
    )

# CORRECT — inspect custom-skills status without mutating the registry
from proxy.skills.skill_registry import get_default_registry
from proxy.skills.custom_skill_loader import CustomSkillsStatus
registry = get_default_registry()
status: CustomSkillsStatus = registry.custom_skills_status()  # read-only snapshot

# CORRECT — load custom skills from a directory and report issues
from proxy.skills.custom_skill_loader import (
    load_custom_skills,
    SkillRegistryConfigurationError,
)
try:
    parsed_skills, issues, status = load_custom_skills(registry)
except SkillRegistryConfigurationError as exc:
    for issue in exc.issues:
        logger.warning("custom_skill_issue: %s", issue)
    raise  # atomic startup: do NOT continue with partial publication
```

### `SkillPathError(OSError)`

Defined in `path_utils.py`. Raised on any filesystem failure inside the
safe-read helper. Subclasses `OSError` (NOT `PermissionError`) so it does
NOT collide with `SkillRegistrationError`. Catch them separately. Neither
is the same as `SkillRegistryConfigurationError` (a `RuntimeError`) raised
by the atomic-publication path in `custom_skill_loader.load_custom_skills`.

## External Dependencies

**Imports from:**
- `proxy.state` — `ModeState` re-export (from `proxy.agent.types`).
- `skills.ticket_skill`, `skills.work_skill` — concrete skill classes for `populate_builtins` (lazy at line 399 via `get_default_registry` → `populate_builtins`).
- `proxy.skills.custom_skill_loader` — `load_custom_skills`, `build_custom_skill_definition`, exception/issue types (`SkillRegistryConfigurationError`, `SkillLoadIssue`, `CustomSkillsStatus`).
- `proxy.agent.agent_registry` — `agent_registry.get_or_load` for `SkillContext.from_request_context`.

**Imported by:**
- `src/proxy/pipeline/dispatch.py` — `SkillDispatcher` calls `registry.get_by_trigger(...)`.
- `src/proxy/pipeline/post_process.py` — `_skill_post_process` reads `ModeState`.
- `src/proxy/pipeline/motivator_stage.py` — telemetry.
- `src/proxy/agent/agent.py` — `_activate_skill` / `_inherit_skill`.
- `src/skills/ticket_skill.py`, `src/skills/work_skill.py` — extend `BaseSkill`.
- `src/proxy/skill_resolver.py` — `resolve_mode` reads skill def.
- Tests under `tests/proxy/`, `tests/skills/`, `tests/unit/`.

## Cross-References

- `src/skills/AGENTS.md` — TicketSkill, WorkSkill, WorkspaceSkill (the concrete classes).
- `src/proxy/agent/AGENTS.md` — `Agent.set_active_mode` is what `_activate_skill` / `_inherit_skill` call.
- `src/proxy/AGENTS.md` — the proxy shell owns `populate_builtins` startup contract.
- `src/proxy/canary.py` (module docstring) — `parse_canary` reads `SkillDefinition` for canary format.
- `tests/AGENTS.md` (planned) — `tests/proxy/test_trust_gate.py` for the trust gate regression.