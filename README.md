# ChoriProx

> LLM proxy with memory, context optimization, tool interception, and provider compatibility.

[![Tests](https://img.shields.io/badge/tests-passing-green)]()
[![Python](https://img.shields.io/badge/python-≥3.9-blue)]()
[![Branch](https://img.shields.io/badge/branch-main-orange)]()

## What It Does

ChoriProx sits between AI coding clients (KiloCode, OpenCode, Cursor) and upstream LLM backends. It intercepts every request to add:

- **Multi-backend routing** — 20 models across 10 backends via [`proxy_models.json`](src/proxy_models.json)
- **Context optimization** — V2 three-tier pipeline (ancient summaries + raw window + rich summaries)
- **Memory system** — SQLite + Neo4j knowledge graph with background SOTA extraction
- **Tool interception** — Memory tools executed locally inside proxy, re-queried to LLM
- **Virtual Filesystem Layer** — Workspace-scoped file caching with write-through and rollback
- **Provider compatibility** — Automatic patches for DeepSeek, Kimi, MiniMax, MiMo
- **Sub-agent detection** — Canary-based filtering to prevent memory pollution
- **Streaming SSE** — Buffered upstream, re-streamed as OpenAI-compatible SSE
- **Circuit breaker** — Neo4j operations isolated; proxy continues when graph DB is down

For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick Start

```bash
# 1. Start Neo4j
neo4j start

# 2. Copy config
cp proxy_config.example.json proxy_config.json

# 3. Start proxy (minimal)
PYTHONPATH=src NEO4J_AUTH="neo4j:password" python3 src/qwen_proxy.py

# 4. Or use start.sh with all API keys
./start.sh
```

Proxy listens on **port 6666**. Point your client at `http://localhost:6666/v1/chat/completions`.

## Key Features

### Multi-Backend Routing
Select any model defined in [`proxy_models.json`](src/proxy_models.json). Each entry specifies backend URL, tool parser, context limit, and API key. See [CONFIGURATION.md](CONFIGURATION.md#config-files) for the model registry format.

### Context Optimization (V2)
Three-tier pipeline with dynamic token budgeting:
- **5% ancient** — compressed narrative timeline for old conversations
- **80% raw** — most recent exchanges preserved verbatim
- **15% summaries** — rich per-message summaries with tool tracking

The three-tier pipeline is standard. See [ARCHITECTURE.md](ARCHITECTURE.md#pipeline-stages) for the full pipeline.

### Memory System
- **SQLite** — Messages, summaries, entities, facts, task state (12 tables)
- **Neo4j** — Entity/fact knowledge graph with supersession tracking
- **SOTA extraction** — Background pipeline: per-message summarization → entity extraction → ancient history compression
- **4 memory tools** — `recall`, `search`, `proxy_ticket`, and `proxy_work` (all intercepted and executed inside the proxy)

See [ARCHITECTURE.md](ARCHITECTURE.md#storage) for the complete storage schema.

### Virtual Filesystem Layer (VFS)
Workspace-scoped LRU cache for file contents, enabling cross-agent file sharing:
- **Read caching** — Cache hits served instantly without disk I/O
- **Write-through** — Cache updated before client executes, with rollback snapshots
- **Rollback on failure** — Heuristic failure detection + 60s timeout → automatic rollback
- **Context injection** — Sub-agents see parent's recent file activity
- **Scope control** — `PROXY_VFS_SCOPE=all|subs|main`
- **Cache-safe main-agent placement** — Main VFS appears after completed
  retained history and immediately before the entire live exchange,
  preserving tool-call atomicity and the provider cache prefix.

See internal documentation for the full cache-safe tail placement,
provider-role, and cache-prefix guarantees.

See [CONFIGURATION.md](CONFIGURATION.md#virtual-filesystem-layer-vfs) for all VFS env vars.

### Tool Interception
Memory tools (`recall`, `search`, `proxy_ticket`, `proxy_work`) are intercepted and executed locally inside the proxy. Results are re-queried to the LLM before returning to the client. Tool loop capped at 69 messages.

### Commands
Inline `^^` prefix commands for proxy control:
- `^^help` — List available commands
- `^^status` — Proxy health (models, requests, sessions, tool cache)
- `^^imp <prompt>` — Send prompt to SOTA provider for enhancement
- `^^ticket <description>` — Enter task-planning mode

All commands rate-limited to 500 per 60s window per session (`command_rate_limiter` in `src/proxy/state.py`).

## Custom JSON skills

ChoriProx supports custom LLM activation skills defined as JSON files
in a directory referenced by `PROXY_SKILLS_DIR`. Each skill defines a
trigger (`^^<name>`), a description, a system prompt, and a tool whitelist.

```bash
# 1. Copy an example from examples/skills/ to your config directory
cp examples/skills/build.json /etc/choriprox/skills/

# 2. Set the env var
export PROXY_SKILLS_DIR=/etc/choriprox/skills/

# 3. Restart the proxy. JSON is loaded once at startup.
PYTHONPATH=src python3 src/qwen_proxy.py

# 4. Verify
curl -X POST http://localhost:6666/v1/chat/completions -d '{
  "model": "test",
  "messages": [{"role":"user","content":"^^build"}]
}'
```

See internal documentation for the full operator guide and canonical
schema.

**Trust boundary:** custom JSON files are administrator-owned
configuration. The proxy reads them but never writes. Until user roles
exist, every API caller who can invoke an enabled custom skill can
exercise its configured capabilities.

### Swarm-capable custom skills (kill switch)

A custom JSON skill may declare a nested `subagents` block to spawn
bounded parallel sub-agents. The capability is **disabled by default**
and gated by a deployment-wide kill switch:

- `PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS` — must be set to exactly `"1"`
  for ANY swarm-capable custom skill to load. Unset, empty, or any other
  value causes swarm-capable skills to be rejected at startup with a
  `swarm_*` code (see internal documentation §3).
- `PROXY_SESSION_SECRET` — **REQUIRED** whenever one or more
  swarm-capable skills are configured (signed swarm canaries depend on
  it). Without the secret, swarm-capable skills are rejected at startup
  with `swarm_missing_secret`.

This is **not** admin authentication — it is defense-in-depth deployment
configuration. The flag is read once at startup; there is no per-request
override. See internal documentation for the full contract and
`docs/feature-flags.md` for the operator reference.

## Testing

### Running the full test suite

The default test suite is fully self-contained — no external services required. To run it:

```bash
CHORI_SKILL_REGISTRY_RESET=1 PROXY_ENV=dev PYTHONPATH=src python3 -m pytest tests/ -k "not e2e_real"
```

### Optional: Real-LLM smoke tests (`tests/e2e_real/`)

There are 8 additional tests in `tests/e2e_real/` that exercise the proxy through a **real LLM** (not mocked). These skip automatically when no `llama-server` is reachable at `http://localhost:7777`.

To run them locally:

1. Start `llama-server` on port 7777 (e.g., `llama-server -m model.gguf --port 7777`)
2. Start the ChoriProx proxy on port 6666 (`./start.sh 6666`)
3. Run:

```bash
CHORI_SKILL_REGISTRY_RESET=1 PROXY_ENV=dev PYTHONPATH=src python3 -m pytest tests/e2e_real/ -v
```

These tests are **not** required for CI to pass — they exist as a pre-merge smoke check for environments with `llama-server` available. See `tests/e2e_real/conftest.py` for the skip fixture.

### Other test invocations

```bash
# Unit tests
PYTHONPATH=src python3 -m pytest tests/unit -n 4 --dist=loadfile -v

# Integration tests
PYTHONPATH=src python3 -m pytest tests/integration -n 4 --dist=loadfile -v

# All tests (unit + integration + proxy + root; excludes e2e_real)
make test

# Coverage (90% clients / 63% overall gates)
make coverage
```

## Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Request flow, directory map, tools, routes, storage, pipeline, locks |
| [CONFIGURATION.md](CONFIGURATION.md) | All env vars, config files, feature flags, rate limiters, tool limits |
| [USER-STORIES.md](USER-STORIES.md) | Behavioral contracts, including VFS cache-safe tail placement |
| [AGENTS.md](AGENTS.md) | Agent operating context, code conventions, known landmines |
| [docs/](docs/) | Specs, ADRs, implementation plans, prompt designs |

## Security Status

**Development tool — not for production use.**

- Binds to `0.0.0.0` (all interfaces) by default
- Authentication via `PROXY_API_KEY` (fail-closed when unset, localhost exempt)
- Malicious content detection **blocks** requests matching injection patterns with 400 error
- Session tokens use HMAC signing via `PROXY_SESSION_SECRET`

See [USER-STORIES.md](USER-STORIES.md) US-051/US-052 for auth behavioral contracts.

## Known Limitations

- `recall` and `search` memory tools are **stubs** (return empty results); `proxy_ticket` and `proxy_work` are fully implemented
- Streaming uses upstream SSE reconstruction (proxy consumes upstream SSE via `OpenAIStreamAccumulator`, reconstructs the full response, then re-streams to client); tool re-queries stay non-stream
- Client adapter layer is partial — core routing does not go through `clients/registry.py`
- Phase-stable KV cache prefix: not yet designed

## Project Structure

```
src/
├── qwen_proxy.py          # Entry shim
├── proxy_models.json      # Model registry (20 models, 10 backends)
├── config.py              # Config loader
├── proxy/                 # Core proxy
│   ├── app.py             # Flask routes
│   ├── state.py           # Shared mutable state
│   ├── pipeline/          # Request pipeline stages
│   ├── tool_pipeline/     # Tool parsing + execution
│   ├── agent/             # Agent lifecycle
│   └── vfs.py             # Virtual Filesystem Layer
├── memory/                # Memory system
│   ├── db.py              # SQLite (12 tables)
│   ├── graph.py           # Neo4j client
│   ├── sync.py            # SOTA extraction
│   ├── tool_loop.py       # Tool interception
│   ├── tools/             # Memory tool implementations
│   └── migrations/        # Neo4j schema versioning
├── skills/                # Skill implementations
├── clients/               # Client adapters
├── commands/              # ^^command system
└── enhancer/              # ^^imp enhancement
```

For the complete directory map with all 201 modules, see [ARCHITECTURE.md](ARCHITECTURE.md#directory-map).

Testing
