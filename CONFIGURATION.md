# ChoriProx Configuration Reference

> Last verified: 2026-06-25 | Branch: feature/custom_skills

## Environment Variables

### Core Proxy

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PROXY_API_KEY` | `""` | No | API key for request authentication. Fail-closed when unset — all endpoints except `/v1/models`, `/v2/models`, and `/health` reject non-localhost requests with 500. Localhost connections bypass auth by default (see `PROXY_DISABLE_LOCALHOST_BYPASS`). |
| `PROXY_DISABLE_LOCALHOST_BYPASS` | `""` | No | Disable localhost auth bypass. Set to `"1"`, `"true"`, or `"yes"` (case-insensitive) to require `PROXY_API_KEY` even for `127.0.0.1` / `::1` requests. **Recommended when deploying behind a reverse proxy** to prevent unauthenticated access. A startup warning is emitted if bypass is active and `PROXY_API_KEY` is set. |
| `PROXY_SESSION_SECRET` | `""` | No | HMAC secret for session tokens (`X-Session-Token` header) and canary signing. When unset, tokens are ignored and canary verification accepts any signature (fail-safe). A startup warning is emitted. **WARNING: Do not set until clients support token bootstrapping — it will cause 401 deadlock.** |
| `PROXY_DEBUG` | unset | No | Enable DEBUG logging level when set to any value. |
| `PROXY_TRACE` | `"0"` | No | Enable detailed request tracing (upstream URL, body, headers) when set to `"1"`. |
| `PROXY_LOG_FORMAT` | `"text"` | No | Log format: `"text"` (standard `asctime - name - level - message`) or `"json"` (structured JSON via `JSONFormatter`). |
| `PROXY_TRUSTED_PROXIES` | `""` | No | Comma-separated trusted proxy CIDR networks for `X-Forwarded-For` validation. Invalid entries are logged and ignored. |
| `PROXY_TEST_DB` | unset | No | Override SQLite DB path. Use `":memory:"` for in-memory shared cache (tests). When unset, defaults to `data/memory.db` relative to project root. |

### Neo4j

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `NEO4J_AUTH` | `"neo4j:password"` | Yes | Neo4j credentials in `user:password` format (colon separator, NOT slash). |
| `PROXY_NEO4J_BOLT_URL` | `"bolt://localhost:7687"` | No | Neo4j bolt connection URL. |

### SOTA (Memory Extraction)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PROXY_SOTA_API_KEY` | `""` | Yes | API key for SOTA extraction LLM backend. |
| `PROXY_SOTA_ENDPOINT` | `"https://api.stepfun.ai/step_plan/v1"` | Yes | SOTA LLM API base URL. |
| `PROXY_SOTA_MODEL` | `"step-3.5-flash-2603"` | Yes | Default SOTA model name. |
| `PROXY_SOTA_TIMEOUT` | `"180"` | No | SOTA request timeout in seconds. Floored at 180 in the OpenAI client. |
| `PROXY_SOTA_MODEL_EXTRACTION` | `"step-3.5-flash-2603"` | No | Model for extraction mode. |
| `PROXY_SOTA_MODEL_SUMMARIZATION` | `"step-3.5-flash-2603"` | No | Model for summarization mode. |
| `PROXY_SOTA_MODEL_COMPRESSION` | `"step-3.5-flash-2603"` | No | Model for compression mode. |
| `PROXY_SOTA_MODEL_PLANNING` | `"step-3.5-flash-2603"` | No | Model for planning mode. |
| `SOTA_MODEL_CTX` | `"256000"` | No | SOTA model context window in characters. Used to compute `SOTA_MAX_INPUT_CHARS = ctx * 0.60 * 4`. |

### Context Optimization

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `V2_PASSTHROUGH_THRESHOLD` | `"0.30"` | No | Fraction of safe budget below which V2 returns full history unchanged. |
| `PROXY_MESSAGE_RETENTION_DAYS` | `"7"` | No | Days to retain messages in SQLite before cleanup. Messages older than this are purged periodically. |

### Virtual Filesystem Layer (VFS)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PROXY_VFS_ENABLED` | `"1"` | No | Enable VFS. Set to `"0"` to disable without code changes. |
| `PROXY_VFS_TTL` | `"60"` | No | Cache entry TTL in seconds. |
| `PROXY_VFS_MAX_ENTRIES` | `"500"` | No | Max cached files per workspace. |
| `PROXY_VFS_MAX_ENTRY_BYTES` | `"1048576"` | No | Max bytes per cached entry (default 1 MB). |
| `PROXY_VFS_MAX_WORKSPACES` | `"20"` | No | Max workspace-scoped caches before LRU eviction. |
| `PROXY_VFS_SCOPE` | `"all"` | No | Agent-type scoping: `"all"`, `"subs"` (sub-agents only), `"main"` (main agent only). Invalid values default to `"all"` with a warning. |

### Observability & Debug

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PROXY_METRICS_ENABLED` | `""` | No | Enable Prometheus metrics. Set to `"1"`, `"true"`, or `"yes"` (case-insensitive). Requires `prometheus_client` package. |
| `PROXY_DEBUG_LOCK_ORDERING` | `""` | No | Enable runtime lock-ordering enforcement. Set to `"1"` to replace all hierarchy locks with `OrderedLock` that raises `RuntimeError` on ordering violations. |
| `PARAM_MOD_LOG_INTERVAL` | `"60"` | No | Rate limit interval (seconds) for parameter modification log messages. |

### Agent & Session

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `CHAT_HISTORIES_MAX_ENTRIES` | `"100"` | No | Max entries in chat history LRU cache. Evicted sessions reload from SQLite on next access. |
| `AGENT_REGISTRY_MAX_AGENTS` | `"100"` | No | Max agents in registry before LRU eviction. Keep high to prevent enumeration-based eviction attacks. |
| `PROXY_MOTIVATOR_ENABLED` | `"1"` | No | Enable sub-agent motivator reminders. Default ON; set to `"0"` to disable injection of context-usage-based advisories (50%/75%/90%) into sub-agent prompts. |
| `PROXY_MOTIVATOR_MAX_TURNS` | `"30"` | No | Legacy — thresholds now fire on context-usage %, not turn count. Kept for backward compat. |
| `PROXY_MOTIVATOR_CONTEXT_WINDOW_CHARS` | `"4000000"` | No | Default context window in chars (~1M tokens) for motivator scaling when no model_cfg override is available. |

### Admin & Auth

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `CONFIG_AUTH_PASSWORD` | `""` | No | Password for debug endpoint Basic auth (`admin:<password>`). When unset, debug endpoints (`/v1/debug/agents*`) return 403 (fail-closed). |

### Model & Adapter

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PROXY_USE_ADAPTERS` | `"1"` | No | Enable model adapter system. Set to `"0"` to disable client detection and model registry. |
| `PROJECT_ROOT` | `os.getcwd()` | No | Project root directory for path resolution. Used by client detection to identify the active coding client. |
| `ANTHROPIC_API_KEY` | `""` | No | API key for Anthropic client detection during adapter resolution. Used in `clients/interface.py`. |

### Task System

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `TASK_BATCH_MAX_SIZE` | `"100"` | No | Max tasks per batch status update in `update_tasks_batch_status()`. |

## Custom JSON Skills

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_SKILLS_DIR` | `<repo>/skills` | Directory containing JSON-defined custom skills. See [docs/skills/custom-json-skills.md](docs/skills/custom-json-skills.md) for the full operator guide. |

**Restart required:** Custom JSON files are loaded once at process startup.
Edit and restart.

**Validation:** Run `python -m proxy.skills.validate_custom_skills --dir /path --format json` before deploying.

**Security:** The skills directory must be admin-owned (mode 0750). Custom JSON files can grant high-impact tools like `bash` and `write` — treat them as code-equivalent configuration, not user data.

## Feature Flags

Boolean flags for incremental feature rollout. Defined in `src/proxy/feature_flags.py`.
Enable by setting the env var to `"1"`.

| Flag | Env Var | Default | Phase | Description |
|------|---------|---------|-------|-------------|
| `USE_DISPATCH_EXTRACTION` | `USE_DISPATCH_EXTRACTION` | `"1"` (ON) | Phase 3 | Dispatch extraction pipeline |
| `CACHE_METRICS_ENABLED` | `PROXY_CACHE_METRICS_ENABLED` | `"1"` (ON) | — | Prompt-cache observability (PostProcessStage) |

Historical flags removed from `feature_flags.py` (no effect if set):
`USE_PROVIDER_REGISTRY` (removed `bef35bd`), `USE_AGENT_ADAPTER`, `USE_AGENT_LIFECYCLE`.

Additional runtime toggles (not in `feature_flags.py` but follow the same pattern):

| Flag | Env Var | Default | Location | Description |
|------|---------|---------|----------|-------------|
| `VFS_ENABLED` | `PROXY_VFS_ENABLED` | `"1"` (ON) | `state.py` | Virtual Filesystem Layer |
| `MOTIVATOR_ENABLED` | `PROXY_MOTIVATOR_ENABLED` | `"1"` (ON) | `pipeline/motivator_stage.py` | Sub-agent motivator |
| `_METRICS_ENABLED` | `PROXY_METRICS_ENABLED` | `""` (OFF) | `metrics.py` | Prometheus metrics |
| `_LOCK_ORDERING_ENABLED` | `PROXY_DEBUG_LOCK_ORDERING` | `""` (OFF) | `lock_ordering.py` | Runtime lock-ordering enforcement |
| `ENABLE_TOKEN_CALCULATION` | N/A (API toggle) | `True` | `context.py` | Dynamic token calculation (toggled via `/v1/context/config/toggle`) |

## Rate Limiter Configuration

7 rate limiter instances defined in `state.py` (6 persisted to SQLite `rate_limiter_state` table, 1 in-process only):

| Name | Max Requests | Window | Persist | Purpose |
|------|-------------|--------|---------|---------|
| `command_rate_limiter` | 500 | 60s | yes | `^^command` rate limiting (session-keyed) |
| `escalation_rate_limiter` | 5 | 3600s | no | Sub-agent mode escalation rate limiting (session-keyed, in-process only) |
| `skill_rate_limiter` | 1000 | 60s | yes | Skill activation rate limiting (session-keyed) |
| `skill_ip_rate_limiter` | 3000 | 60s | yes | Per-IP skill activation |
| `task_rate_limiter` | 2000 | 60s | yes | Task operation rate limiting (session-keyed) |
| `chat_rate_limiter` | 1000 | 60s | yes | `/v1/chat/completions` rate limiting (IP-keyed) |
| `admin_rate_limiter` | 1000 | 60s | yes | Admin endpoint rate limiting (IP-keyed) |

⚠️ **Do NOT tighten these limits** without implementing per-account rate limiting first. The 50-100× values vs. an earlier 10-60/min baseline were raised in commit `872964c` to fix sub-agent self-DoS (per-session bucketing conflates all local sub-agent traffic into one bucket). Lowering limits reintroduces the hang.

## Tool Limits

Hard limits per request, defined in `src/proxy/tool_pipeline/tool_limits.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_PROXY_TICKET_CALLS` | 50 | Max `proxy_ticket` tool calls per request |
| `MAX_PROXY_WORK_CALLS` | 50 | Max `proxy_work` tool calls per request |
| `MAX_REPORT_BUG_CALLS` | 5 | Max `report_bug` tool calls per request |
| `MAX_FILE_READS` | 30 | Max unique file reads per request (each file readable once) |
| `MAX_SUBAGENTS` | 3 | Max sub-agent spawns per request |
| `MAX_RETRIES_PER_CALL` | 3 | Max retries per memory tool call |

File blocklist patterns (cannot be read by ticket-mode agents): `.env`, `.pem`, `credential`, `secret`, `token`, `.git/`, `.password`

## Tool Loop Limits

Memory tool interception loop, defined in `src/memory/tool_loop.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 5 | Max re-query loop iterations |
| `max_total_time` | 900.0s (15 min) | Total wall-clock budget for the tool loop |
| `max_result_size` | 16384 bytes | Max bytes per tool result before truncation |
| `max_calls_per_iteration` | 50 | Max memory tool calls processed per iteration (env-tunable `PROXY_MAX_CALLS_PER_ITERATION`, range [1, 200]) |
| `CONTEXT_TOKEN_LIMIT` | 26000 | Token limit for working messages in the loop |
| `USABLE_CONTEXT_RATIO` | 0.75 | Fraction of context limit usable before trimming |

## HTTP Client Configuration

Upstream call settings from `src/proxy/http_client.py` and `proxy_config.json`:

| Parameter | Default | Source | Description |
|-----------|---------|--------|-------------|
| Base timeout | 1800s (30 min) | `proxy_config.json` → `upstream.timeout` | Read timeout for upstream calls. Connect timeout is always 30s. |
| Timeout multiplier | 2.0 | `proxy_config.json` → `upstream.timeout_multiplier` | Scaling factor when body > 40 KB or messages > 20. |
| Max retry attempts | 3 | `proxy_config.json` → `upstream.retry_max_attempts` | Retries on transient errors (ConnectionError, ChunkedEncodingError, Timeout, 502/503/504). |
| Base retry delay | 2.0s | `proxy_config.json` → `upstream.retry_base_delay` | Initial exponential backoff delay. |
| Max retry delay | 30.0s | `proxy_config.json` → `upstream.retry_max_delay` | Maximum backoff delay cap. |
| 429 Retry-After | Up to 60s | Hardcoded | Honors `Retry-After` header, capped at 60s. Falls back to exponential backoff if absent. |
| Endpoint concurrency | Per-endpoint | `proxy_config.json` → `upstream.endpoint_concurrency` | Semaphore-based concurrency limiter per `scheme://host`. Omitted = unlimited. |
| SSE heartbeat interval | 5.0s | `proxy_config.json` → `streaming.heartbeat_interval` (env override `PROXY_STREAMING_HEARTBEAT_INTERVAL`) | Seconds between `: keepalive` SSE comments while the completion thread is still running (preflight, upstream, tool loops, post-processing). Zero or negative falls back to 5.0 with a WARN. |

## Config Files

### proxy_models.json (Model Registry)

Located at `src/proxy_models.json`. JSON object with two top-level keys:

- `"models"` — dict keyed by model alias. Each entry:
  - `backend` (string, required): Backend URL (e.g., `"https://api.stepfun.ai/step_plan/v1/chat/completions"`)
  - `tool_parser` (string, required): Tool parsing format — `"xml"`, `"json"`, `"openai"`, or `"dsml"`
  - `context_limit` (int, required): Max context tokens
  - `name` (string, optional): Human-readable display name
  - `api_key_env` (string, optional): Environment variable name containing the API key for this backend
  - `backend_model` (string, optional): Model name at the backend (if different from alias)
  - `ignore_subagents` (bool, optional): Whether to ignore sub-agent requests for this model
  - `memory_injection_enabled` (bool, optional): Whether to inject Neo4j memories for this model

- `"default"` — fallback model config with `backend`, `tool_parser`, and `context_limit`

### proxy_config.json (Runtime Overrides)

Located at project root. **NOT in repo** — copy from `proxy_config.example.json`. Gitignored.

Schema (all sections optional):

```json
{
  "security": {
    "config_auth_enabled": false,
    "config_api_key_env": null
  },
  "context": {
    "enable_token_calculation": true,
    "max_message_content_size_kb": 100,
    "max_total_content_size_kb": 500
  },
  "upstream": {
    "timeout": 1800,
    "timeout_multiplier": 2.0,
    "retry_max_attempts": 3,
    "retry_base_delay": 2.0,
    "retry_max_delay": 30.0,
    "endpoint_concurrency": {
      "https://ollama.com": 2
    }
  },
  "streaming": {
    "heartbeat_interval": 5.0
  }
}
```

## Startup

```bash
# Minimal startup
PYTHONPATH=src NEO4J_AUTH="neo4j:password" python3 src/qwen_proxy.py

# Full startup with SOTA memory extraction
PYTHONPATH=src NEO4J_AUTH="neo4j:password" \
  PROXY_SOTA_API_KEY="sk-..." \
  PROXY_SOTA_ENDPOINT="https://api.stepfun.ai/step_plan/v1" \
  PROXY_SOTA_MODEL="step-3.5-flash-2603" \
  PROXY_TEST_DB="data/memory.db" \
  python3 src/qwen_proxy.py

# With VFS disabled
PROXY_VFS_ENABLED=0 PYTHONPATH=src python3 src/qwen_proxy.py

# Sub-agent-only VFS
PROXY_VFS_SCOPE=subs PYTHONPATH=src python3 src/qwen_proxy.py

# With debug lock ordering enforcement
PROXY_DEBUG_LOCK_ORDERING=1 PYTHONPATH=src python3 src/qwen_proxy.py

# With Prometheus metrics
PROXY_METRICS_ENABLED=1 PYTHONPATH=src python3 src/qwen_proxy.py
```

Proxy listens on **port 6666** by default. Max request body size: **16 MB**.
