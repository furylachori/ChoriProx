# Environment Variables Reference

This document lists all environment variables used by the ChoriProx proxy system.

## Authentication

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_SESSION_SECRET` | (none) | HMAC secret for session tokens and canary signing. **Required** whenever any swarm-capable custom skill is configured (signed swarm canaries depend on it); otherwise strongly recommended for production. | Production (mandatory with swarm-capable skills) |
| `PROXY_API_KEY` | (none) | API key for proxy authentication via Bearer token or X-API-Key header | Production |
| `CONFIG_AUTH_PASSWORD` | (none) | Password for debug endpoint Basic authentication (admin:password) | For debug endpoints |

## Neo4j

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_NEO4J_BOLT_URL` | `bolt://localhost:7687` | Neo4j Bolt connection URL | No |
| `NEO4J_AUTH` | `neo4j:password` | Neo4j authentication in `user:password` format | No |

## Logging & Debug

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_DEBUG` | (not set) | Enable debug-level logging when set to any value | No |
| `PROXY_TRACE` | `0` | Enable detailed request/response tracing (set to "1" to enable) | No |
| `PARAM_MOD_LOG_INTERVAL` | `60` | Interval in seconds for parameter modification log messages | No |

## SOTA (State-of-the-Art) Memory Models

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_SOTA_API_KEY` | (none) | API key for SOTA model service (StepFun) | For memory features |
| `PROXY_SOTA_ENDPOINT` | `https://api.stepfun.ai/step_plan/v1` | SOTA service API endpoint | No |
| `PROXY_SOTA_MODEL` | `step-3.5-flash-2603` | Default SOTA model for general tasks | No |
| `PROXY_SOTA_MODEL_EXTRACTION` | `step-3.5-flash-2603` | Model for information extraction tasks | No |
| `PROXY_SOTA_MODEL_SUMMARIZATION` | `step-3.5-flash-2603` | Model for summarization tasks | No |
| `PROXY_SOTA_MODEL_COMPRESSION` | `step-3.5-flash-2603` | Model for context compression tasks | No |
| `PROXY_SOTA_MODEL_PLANNING` | `step-3.5-flash-2603` | Model for task planning | No |
| `PROXY_SOTA_TIMEOUT` | `180` | Timeout in seconds for SOTA API calls | No |
| `PROXY_SOTA_RETRY_AFTER_CAP` | `60` | Maximum seconds to sleep on a `Retry-After` response from the SOTA endpoint. Clamped to [1, 300]. Prevents worker starvation when upstream returns arbitrarily large Retry-After values. | No |
| `SOTA_MODEL_CTX` | `256000` | Context window size (tokens) for SOTA models | No |

## Feature Flags

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_USE_ADAPTERS` | `1` | Enable model adapter system (set to "0" to disable) | No |
| `USE_DISPATCH_EXTRACTION` | `1` | Enable dispatch extraction pipeline | No |
| `PROXY_CACHE_METRICS_ENABLED` | `1` | Enable prompt-cache observability | No |
| `PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS` | (unset) | Deployment-wide kill switch for swarm-capable custom JSON skills. Must be exactly `"1"` for any skill declaring a nested `subagents` block to load; unset/empty/other values cause such skills to be rejected at startup with a `swarm_*` code. Read once at startup — no per-request override. Also requires `PROXY_SESSION_SECRET` to be set; otherwise startup fails with `swarm_missing_secret`. Not authentication: this is operator-controlled deployment configuration (same trust boundary as the proxy process). See internal documentation §3. | No |

Note: `USE_AGENT_ADAPTER` and `USE_AGENT_LIFECYCLE` are listed in older revisions of this doc but were never implemented. `USE_PROVIDER_REGISTRY` was removed in `bef35bd`. Setting any of these has no effect.

## Session & Storage

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_MESSAGE_RETENTION_DAYS` | `7` | Number of days to retain messages in SQLite database | No |
| `CHAT_HISTORIES_MAX_ENTRIES` | `100` | Maximum entries in chat history cache per session | No |
| `AGENT_REGISTRY_MAX_AGENTS` | `100` | Maximum number of agents in the registry | No |
| `TASK_BATCH_MAX_SIZE` | `100` | Maximum batch size for task processing | No |

## Request Deduplication (CP4)

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_IDEMPOTENCY_SECRET` | (none) | HMAC secret for request fingerprinting. Falls back to `PROXY_SESSION_SECRET`, then a process-local random. | No (fallback chain) |
| `PROXY_DEDUP_MAX_COMPLETED` | `256` | Max COMPLETED entries in the dedup store. | No |
| `PROXY_DEDUP_MAX_BYTES` | `16777216` (16MB) | Max total stored bytes across COMPLETED entries. | No |
| `PROXY_DEDUP_MAX_FOLLOWERS` | `16` | Max followers per PENDING dedup key. | No |
| `PROXY_DEDUP_COMPLETED_TTL` | `120` | TTL (seconds) for COMPLETED dedup entries. | No |
| `PROXY_DEDUP_PENDING_MAX_AGE` | `600` | Max age (seconds) for PENDING dedup entries (crash guard). | No |
| `PROXY_DEDUP_NONSTREAM_FOLLOWER_TIMEOUT` | `60` | Non-stream follower wait timeout (seconds). On timeout, returns 504 + Retry-After. | No |

## Context & Performance

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_STREAMING_HEARTBEAT_INTERVAL` | `5.0` | Seconds between `: keepalive` SSE comments emitted by `_keepalive_stream` while the completion thread is still running (preflight, upstream, tool loops, post-processing). Must be a positive int/float; zero/negative falls back to `5.0` with a WARN. Overrides `proxy_config.json` → `streaming.heartbeat_interval`. | No |

## Custom Skills

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_SKILLS_DIR` | `<repo>/skills` | Directory containing JSON-defined custom skills. Absolute or relative to repo root. Set to empty string to disable custom-skill loading. Set to a non-existent path: startup fails (Wave 2 eager init). Unset + default missing: healthy, builtins only. Restart required after changes (no hot reload). Swarm-capable skills additionally require `PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS=1` and `PROXY_SESSION_SECRET` — see internal documentation §3. | No |

Semantics (membership in `os.environ`, not truthiness):

| Environment state | Outcome |
|---|---|
| Variable absent; default `<repo>/skills` exists | Load custom skills from the default directory |
| Variable absent; default `<repo>/skills` missing | Healthy; builtins only |
| `PROXY_SKILLS_DIR=""` (empty string) | Custom loading explicitly disabled; healthy; builtins only |
| `PROXY_SKILLS_DIR="/abs/path"` (exists + is dir) | Load custom skills from that path |
| `PROXY_SKILLS_DIR="/abs/path"` (missing or not a dir) | Startup fails (fatal `SkillRegistryConfigurationError`) |
| `PROXY_SKILLS_DIR="rel/path"` | Resolved relative to repo root; same fatal rules apply |

Hot reload is NOT supported. Restart the proxy to pick up changes.
See internal documentation §5 for the full contract.

## Testing

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `PROXY_TEST_DB` | (not set) | Set to `:memory:` to use in-memory SQLite for testing | No |

## External Services

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `ANTHROPIC_API_KEY` | (none) | Anthropic API key for Claude model integration | If using Claude models |
| `PROJECT_ROOT` | Current working directory | Project root directory for file path resolution | No |

## Notes

- **Production deployments** should always set `PROXY_SESSION_SECRET` and `PROXY_API_KEY` for security.
- **Memory features** require `PROXY_SOTA_API_KEY` and a valid Neo4j connection.
- **Debug endpoints** (`/v1/debug/*`) are disabled unless `CONFIG_AUTH_PASSWORD` is set.
- **Feature flags** default to production-ready values; only change for testing or specific use cases.
- **SOTA models** all default to `step-3.5-flash-2603` but can be customized per task type.
