# `src/proxy/tool_parsing/AGENTS.md` — Tool Argument Normalization (THE LANDMINE ZONE)

> **WARNING:** `normalize.py` has been broken 5+ times the SAME way. Read the
> invariants below BEFORE editing. The two regression tests
> `tests/unit/test_argument_normalization_unit.py` (must stay GREEN with
> `valid_params=None`) and `tests/unit/test_argument_normalization_schema.py`
> (schema-driven case) are the canonical regression set.

## Module Purpose

Reconciles LLM-degraded tool/argument names back to the names the client
actually sent in the incoming request. Also provides the 7-extractor parser
and the system-prompt templates for ticket/work modes.

## Files in this module

| File | Purpose |
|---|---|
| `normalize.py` | **THE LANDMINE** — `normalize_tool_arguments`, `canonicalize_tool_name`, `make_tool_calls`, `build_valid_params_by_tool`, `extract_file_path`, `_PROXY_INTERNAL_CONTROL_KEYS`, `TOOL_NAME_ALIASES` (372 LOC) |
| `parsers.py` | 7-extractor parser (`parse_all_tools_from_content`): YAML → DSML → XML `<tools>` → `<function=name>` → Markdown ```json``` → bare JSON → native `tool_calls` |
| `prompts.py` | `OUR_SYSTEM_PROMPT` (the main proxy system prompt — duplicated name, see landmines) |
| `ticket_prompts.py` | `TICKET_MAIN_PROMPT`, `TICKET_SUB_PROMPT`, `TICKET_SHARED_REFERENCE` (split by role in C2; instructs MAIN ticket agents to delegate via the `task` tool, sub-agents to use `proxy_ticket`) |
| `work_prompts.py` | `WORK_MANAGER_PROMPT` |

## THE LANDMINE — Case Folding (broken 5+ times)

`normalize.py` heals LLM-degraded tool/arg names back to the names the
client actually sent. This has been broken the SAME way 5+ times: someone
adds a case-insensitive / `.lower()` fallback or hardcodes a snake_case
canonical target, and opencode's camelCase args (`filePath`) get silently
rewritten to snake_case (`file_path`) → the client rejects the tool call
with `SchemaError: Missing key ["filePath"]`. `read`/`edit`/`write` then
fail for opencode.

### THE INVARIANTS (all must hold)

1. **Schema-driven.** `normalize_tool_arguments(name, args, valid_params=...)` reconciles against `valid_params` — the client's real param names from `build_valid_params_by_tool(ctx.llama_body["tools"])`. **Resolver order is FIXED:** exact schema match → fuzzy (casing/separator) → synonym alias (then reconciled against schema) → passthrough unchanged. Do not reorder.

2. **Never case-fold without the schema.** Any new key resolution MUST go through `_resolve()`, which checks `valid_params` BEFORE any `.lower()`. A bare `mappings.get(key.lower())` that ignores the schema IS the recurring bug.

3. **Alias targets are intermediate, not final.** `param_mappings` targets are snake_case tokens (e.g. `file_path`) that get fuzzy-resolved to the real schema name at runtime. Do NOT "fix" them to camelCase (breaks kilocode); do NOT assume a target casing.

4. **`valid_params=None` is byte-identical legacy behavior**, reserved for internally-executed memory tools (`recall`, `search`, …) which run locally in snake_case and never round-trip to the client. It is NOT a "default client" path.

5. **Path extraction uses `extract_file_path()`** (`file_path` | `filePath` | `path`). Never `.get("file_path")` on normalized args — every such site has broken opencode (VFS cache, read blocklist/read-once guard, write/edit tracking). Grep `src/` for new occurrences before merging.

6. **No per-client casing tables.** The opencode/kilocode `normalize_tool_args` adapter methods are vestigial/dead (never called in `src/`). Do not wire them up — the schema path auto-adapts to any client. If you ever see `normalize_tool_args`/`denormalize_tool_args` actually called by a client, you've reintroduced the casing-table antipattern. These methods exist ONLY as no-op stubs and tests pin this invariant at `tests/unit/clients/test_canary_preservation.py`.

### When editing `normalize.py` or adding a tool to `param_mappings`:

- Keep `valid_params=None` byte-identical (run `tests/unit/test_argument_normalization_unit.py` — must stay green unchanged).
- Add a schema-driven case to `tests/unit/test_argument_normalization_schema.py` (exact-preserve + heal + idempotency).
- Convert any new `.get("file_path")` site on normalized args to `extract_file_path()`.
- If you add a new importer of `proxy.tool_parsing` from inside `proxy/tool_pipeline/`, import from the leaf `proxy.tool_parsing.normalize` — NOT the package — to avoid a circular import (`tool_parsing.__init__` → `prompts` → `tool_pipeline` → `tool_guard`).

## Key Public API

### `build_valid_params_by_tool(tools_schema)`

Map `{tool_name: frozenset(real_param_names)}` from an OpenAI `tools[]` array.
Returns `{}` for missing/malformed input. Tolerates tools whose `parameters`
is absent or has no `properties`. Called at the top of `tool_pipeline.py:run()`
and `tool_parsing/__init__.py`.

### `canonicalize_tool_name(name, valid_names=None)`

4-step resolver (in order):
1. Already in `valid_names` → return as-is.
2. In static `TOOL_NAME_ALIASES` → return mapped name (LLM hallucinations: `todo_write` → `todowrite`, `execute_code` → `bash`, etc.).
3. Try stripping common suffixes (`_file`, `_search`, …) and check against `valid_names`.
4. Fallback: return `name` unchanged.

### `normalize_tool_arguments(tool_name, arguments, valid_params=None)`

The main normalizer. Three phases inside `_resolve()`:
1. **Exact schema match** — if `key in valid_params`, return `key` unchanged.
2. **Synonym alias** — check `mappings[key]` (snake_case tokens) — if target is `_PROXY_INTERNAL_CONTROL_KEYS` (`subagent_mode`, `subagent_type`), return target unchanged (proxy-internal).
3. **Fuzzy schema match** — `_fuzzy(key)` against `valid_params`, return real schema param name.
4. **Alias → schema** — if mapped target is in `valid_params`, return it; else fuzzy-resolve target.

Always canonicalizes `tool_name` first via `canonicalize_tool_name(tool_name, set(param_mappings.keys()))`.

### `extract_file_path(args)`

Tries `file_path` (snake_case) → `filePath` (opencode camelCase) → `path`
(kilocode internal). Returns `""` if none present or `args` is not a dict.

USE THIS instead of `.get("file_path")` on normalized args. Every such site
in `src/proxy/tool_pipeline/` and `src/memory/tools/` has been broken opencode.

### `make_tool_calls(tool_calls_list, existing_ids=None, valid_params_by_tool=None)`

Builds the OpenAI `tool_calls` array for the client response.
**ALWAYS pass `valid_params_by_tool`** — otherwise arg names are not
reconciled to the client schema and opencode rejects with `SchemaError`.

### `provide_defaults_for_optional_params(tool_name, args, valid_params=None)`

Provides defaults: `bash.description = "shell command"`,
`task.subagent_type = "general"`, `grep.path = "."`.

### `_PROXY_INTERNAL_CONTROL_KEYS = frozenset({"subagent_mode", "subagent_type"})`

Proxy-injected control keys that are NEVER in the client tool schema.
Their paraphrase aliases (`mode` → `subagent_mode`, `type` → `subagent_type`)
MUST apply unconditionally regardless of `valid_params` so `canary.py:_resolve_task_canary`
(which reads `args.get("subagent_mode")`) sees the normalized value.

## Parsers (`parsers.py`)

`parse_all_tools_from_content(content)` — the 7-extractor parser used by
`tool_pipeline.py:357` for embedded tool-call formats when native `tool_calls`
is absent. Order:

1. **YAML** — `parse_yaml_tool_blocks`
2. **DSML** — `|` fullwidth pipe + `tool_calls` (kilocode Chinese)
3. **XML `<tools>...</tools>`**
4. **XML `<function=name...>`**
5. **Markdown ```json code blocks```**
6. **Bare JSON via `_extract_balanced_json`**
7. **Native `tool_calls`** (handled separately in `_parse`)

Also handles `<tool_call>` (Hermes format) and `‹...›` (chevron delimiters).

## Key Invariants

1. **`valid_params=None` is the legacy byte-identical path.** Reserved for `recall`/`search`/`create_tasks`/etc. that run locally in snake_case. NOT a "default client" path.
2. **Resolver order is FIXED.** Exact → fuzzy → synonym → passthrough. Reordering breaks case 2 (the `.lower()` bug).
3. **`_resolve` checks `valid_params` BEFORE any `.lower()`.** The bare `mappings.get(key.lower())` antipattern silently rewrites `filePath` → `file_path`.
4. **`extract_file_path()` reads from raw args.** Works regardless of casing because opencode may pass `filePath` while LLM-degraded args pass `path`. Always use it for path-sensitive ops (VFS cache, read-once, write/edit tracking).
5. **`make_tool_calls(..., valid_params_by_tool=...)` is the ONLY way to build client-facing tool_calls.** Hand-constructing one misses argument reconciliation.
6. **`build_valid_params_by_tool` is the SINGLE source of `valid_params_by_tool`.** Build it ONCE at the top of `ToolPipeline.run()` (or in the equivalent handler), pass it down through all normalization calls.

## Landmines

- **Case folding without schema** — see invariants above. The bug class has caused 5+ regressions.
- **Hardcoding snake_case canonical target** — e.g., someone "fixing" `param_mappings["read"]["filePath"] = "file_path"` to look like the resolved output. This breaks kilocode which DOES send snake_case `file_path`. The mapping is INTERMEDIATE.
- **Hardcoding camelCase canonical target** — the inverse. Breaks kilocode. The schema auto-resolves either way at runtime.
- **Skipping `make_tool_calls` and hand-constructing the array** — guarantees a schema-mismatch for opencode.
- **`.get("file_path")` on normalized args** — `filePath` is opencode's case. `extract_file_path()` handles both. Every site has broken opencode.
- **`valid_params=None` ≠ "default client".** It's a legacy byte-identical path for local memory tools. Don't add client-arg handling there.
- **`proxy.tool_parsing.__init__.py` circular import.** When importing from inside `proxy/tool_pipeline/`, use `from proxy.tool_parsing.normalize import ...` (the leaf), NOT `from proxy.tool_parsing import ...` (the package).
- **`parsers.py:211` dead regex** (`json_pattern`) — pre-Phase-3 fallback. Don't add new code paths through it; the 7-extractor covers all live formats.
- **Duplicate `OUR_SYSTEM_PROMPT` definition** — exists in both `proxy/tool_parsing/prompts.py` AND `proxy/skill_prompts.py`. The `tool_parsing` one is canonical; `skill_prompts.py` is the legacy alias.
- **`detect_tool_mode` duplicated branches** — `parsers.py` has TWO switch-on-shape branches that do the same thing (pre-Phase-3 refactor artifact).
- **Fullwidth pipe `｜` in DSML** — kilocode uses Unicode `｜` (U+FF5C), not ASCII `|`. The `has_dsml` check (tool_pipeline.py:347) and `parsers.py` MUST check both.

## Code Conventions

```python
# CORRECT — schema-driven
from proxy.tool_parsing import (
    build_valid_params_by_tool, extract_file_path,
    normalize_tool_arguments, make_tool_calls,
)
vpbt = build_valid_params_by_tool(ctx.llama_body.get("tools", []))
args = normalize_tool_arguments(name, args, valid_params=vpbt.get(name))
path = extract_file_path(args)  # never .get("file_path")
response["tool_calls"] = make_tool_calls(tool_calls_list, valid_params_by_tool=vpbt)

# CORRECT — memory tool internal (legacy snake_case)
args = normalize_tool_arguments("create_tasks", arguments, valid_params=None)
# byte-identical legacy behavior — preserve exactly

# WRONG — case-folding without schema
# mappings.get(key.lower())  # never do this without valid_params check

# WRONG — hardcoded canonical target
# param_mappings["read"]["filePath"] = "file_path"  # already snake_case in mapping; runtime resolves
```

## External Dependencies

**Imports from:**
- (internal stdlib only: `json`, `logging`, `re`, `uuid`).

**Imported by:**
- `proxy.tool_pipeline.tool_pipeline` — `_parse`, `make_tool_calls`, `extract_file_path`, `build_valid_params_by_tool`, `normalize_tool_arguments`.
- `proxy.tool_pipeline.tool_guard` — `extract_file_path` (for `read` blocklist).
- `memory.tools.proxy_ticket`, `memory.tools.proxy_work` — `extract_file_path` for write/edit tracking.
- `memory.tools.schemas` — tool definitions.
- Tests under `tests/unit/test_argument_normalization_unit.py`,
  `tests/unit/test_argument_normalization_schema.py`,
  `tests/unit/clients/test_canary_preservation.py`.

## Cross-References

- `src/proxy/tool_pipeline/AGENTS.md` — `ToolPipeline` + `MemoryToolGuard` use these functions.
- `src/clients/AGENTS.md` — the dead per-client casing-table antipattern lives there. `normalize_tool_args`/`denormalize_tool_args` are stubs.
- `src/memory/tools/AGENTS.md` — `execute_memory_tool` uses these for memory-tool calls (legacy `valid_params=None`).
- `tests/AGENTS.md` (planned) — the `argument_normalization_*` test files are the canonical regression set.

## Repository artifact policy

Reviewers MUST flag and request rework when a PR commits a non-code artifact without an explicit justification in that originating PR identifying the file and its required purpose. Passing tests or calling a file "evidence" does not waive this rule.

Generated evidence, logs, traces, screenshots, recordings, dumps, and reports belong in the originating PR or issue, NEVER in the repository. Store bulky artifacts in Google Drive and link them from that PR or issue. Do not commit those artifacts as documentation or fixtures to evade this policy.

Source code and tests belong in the repository. Any minimal fixture, documentation, or configuration genuinely required by the implementation must have an explicit file-and-purpose justification in the originating PR; retain only what is required. If that justification is absent, request rework rather than accepting the artifact.

This AGENTS.md policy update is explicitly requested by the repository owner.
