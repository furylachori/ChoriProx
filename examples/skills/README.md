# ChoriProx Custom Skills — Examples

> For the full operator guide, see the internal documentation.

Starting-point examples for operators who want to write their own custom
skills. Each file in this directory is a minimal, valid JSON skill definition
that can be copied into your configured skills directory and edited to fit
your workflow.

## Trust model reminder

These examples are **administrator-owned, code-equivalent configuration for
the main agent** (per the custom-skills spec, §2). They are NOT user-uploaded
extensions, sandboxed plugins, or hot-reloadable modules. Granting a tool
through `tool_whitelist` (including high-impact tools like `bash`, `write`,
`edit`, and `webfetch`) is equivalent to granting the model that capability
on the client side.

**Do NOT place untrusted user-uploaded JSON here.** Treat these files like
source code: review every change, restrict write access to the deployment
administrator, and audit `tool_whitelist` grants with the same care you would
give a `sudoers` entry.

## Quick start

1. Copy an example file to your configured skills directory. The default
   skills directory is `<repo>/skills` when `PROXY_SKILLS_DIR` is unset and
   the directory exists; set `PROXY_SKILLS_DIR` to override.
2. Edit the JSON — change `name`, `trigger`, `description`, `prompt`, and
   `tool_whitelist` to match what you want the skill to do. Keep the file
   name in sync with the `name` field (the filename stem must equal `name`).
3. Restart the proxy. There is no hot-reload — JSON files are read once at
   process startup.
4. Verify the skill loaded: invoke `^^<name>` from your AI client, or check
   the registry status with `^^status`. If the registry rejects the file at
   startup, the proxy will fail closed and the diagnostic will tell you
   which file and which field caused the failure.

## Schema reference

The canonical JSON schema is defined in the internal documentation (§4 of the
custom-skills contract). Every file in this directory conforms to that
schema. The canonical shape:

```json
{
  "name": "review",
  "trigger": "^^review",
  "description": "Review code for correctness, regressions, and missing tests.",
  "prompt": "Review the requested code ...",
  "tool_whitelist": ["read", "glob", "grep"],
  "aliases": ["^^audit"],
  "ttl_seconds": 300
}
```

Fields:

- `name` — must match `^[a-z][a-z0-9_-]*$` and must equal the filename stem.
- `trigger` — must equal `^^<name>`.
- `description` — non-empty string, ≤ 160 characters.
- `prompt` OR `prompt_file` — exactly one of the two is required.
- `tool_whitelist` — required, non-empty list of canonical tool names.
- `aliases` — optional list of additional triggers, each matching `^^<name>`.
- `ttl_seconds` — optional integer between 1 and 86,400; defaults to 300.

Unknown top-level keys, reserved names (`ticket`, `work`), and reserved
triggers (`^^help`, `^^status`, `^^workspace`, `^^ws`) cause the file to be
rejected at startup. There is no fail-open fallback.

## Tool whitelist reminder

- **Empty list = deny all tools** (NOT allow all). A skill with an empty
  whitelist cannot invoke any tool.
- `proxy_ticket`, `proxy_work`, and `task` are **always rejected** for custom
  skills, regardless of what the author wrote. These tools are reserved for
  trusted built-in skills.
- Every entry in `tool_whitelist` must match a canonical tool name registered
  in the `ToolRegistry`. Misspelled or unknown entries cause the entire file
  to be rejected at startup.

## Permissions recommendation

The skills directory is admin-owned code-equivalent configuration. Treat it
like a configuration directory under root control:

| Resource | Recommended mode |
|---|---|
| Skills directory | `0750` (owner=rwx, group=rx, other=none) |
| Individual JSON files | `0640` (owner=rw, group=r, other=none) |
| Owner | the deployment administrator account |
| Proxy process | runs as a non-privileged user that **reads** the directory |

The proxy process must have read access. It should never have write access.
The directory should not be world-readable.

## Environment notes

- `PROXY_SKILLS_DIR=""` (empty string) — explicitly disables custom-skill
  loading. The proxy runs with built-in skills only.
- `PROXY_SKILLS_DIR` unset — the default is `<repo>/skills` if that path
  exists; otherwise no custom skills load. With no files in the default
  directory, no examples auto-load.
- `PROXY_SKILLS_DIR=/some/path` — absolute override. The directory must
  exist and be readable, or the proxy fails to start.
- `PROXY_SKILLS_DIR=rel/path` — relative to the repo root.

**Restart required.** There is no hot-reload and no `^^reload` command. Any
change to a JSON file, or to `PROXY_SKILLS_DIR`, takes effect on the next
proxy startup. See the internal documentation (§5.3) for the rationale (registry generation IDs,
atomic snapshot swap, active-mode invalidation, canary-cache invalidation,
in-flight draining — all of these must be solved before hot reload can be
added safely).

## Per-file purpose

| File | Trigger | Purpose | Tools |
|---|---|---|---|
| `build.json` | `^^build` | Builds the project. Detects the build system from project files (Makefile, package.json, pyproject.toml, etc.) and runs the appropriate build command. | `bash`, `read`, `glob`, `grep` |
| `plan.json` | `^^plan` | Creates an execution plan. Breaks the work into phases, identifies dependencies, estimates effort, and outputs a structured plan for review. Does not execute the plan. | `read`, `glob`, `grep`, `webfetch` |
| `test.json` | `^^test` (alias `^^t`) | Runs the project's test suite. If the user does not specify which tests, asks first. Uses the appropriate test runner (pytest, npm test, etc.) and reports results. | `bash`, `read`, `glob`, `grep` |