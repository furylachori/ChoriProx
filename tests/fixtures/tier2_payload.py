"""Synthesized canonical Tier-2 sub-AGENTS.md fixture.

Derivation: spec v3.2 §4 (Wire format) — the original opencode dump
(76b6d869-fe3) has rotated out of data/debug_inbound/, so this fixture
reproduces the verified read-tool wire format deterministically:

    <path>/abs/read/file.py</path><type>file</type><content>
    ...file body (raw, unescaped)...
    </content><system-reminder>
    Instructions from: /abs/sub/AGENTS.md
    ...full content...
    Instructions from: /abs/parent/AGENTS.md
    ...full content...
    </system-reminder>

No randomness, no network, no env deps — byte-identical every import.
Sizes: FILE_BODY ≈ 3 KB, TIER2_SUFFIX ≈ 34 KB (two ~17 KB ancestor blocks).
"""

import os
import re

_PROJECT_ROOT = "/repo/project"
_SUB_DIR = f"{_PROJECT_ROOT}/src/proxy/sub"
_PARENT_DIR = f"{_PROJECT_ROOT}/src/proxy"
_READ_PATH = f"{_SUB_DIR}/route.py"
_SUB_AGENTS_PATH = f"{_SUB_DIR}/AGENTS.md"
_PARENT_AGENTS_PATH = f"{_PARENT_DIR}/AGENTS.md"

_HEADER_RE = re.compile(r"^Instructions from: (\S.*?)\s*$")


def _is_absolute_path(p: str) -> bool:
    if not isinstance(p, str) or len(p) < 2:
        return False
    if p[0] == "/" or p[0] == "~":
        return True
    if len(p) >= 3 and p[0].isalpha() and p[1] == ":" and p[2] in ("\\", "/"):
        return True
    return False


def _extract_instruction_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_path: str | None = None
    current_lines: list[str] = []
    for line in text.split("\n"):
        m = _HEADER_RE.match(line)
        if m is not None and _is_absolute_path(m.group(1)):
            if current_path is not None:
                blocks.append((current_path, "\n".join(current_lines)))
            current_path = m.group(1)
            current_lines = []
        elif current_path is not None:
            current_lines.append(line)
    if current_path is not None:
        blocks.append((current_path, "\n".join(current_lines)))
    return blocks


_FILE_BODY = '''\
"""Example module under a subfolder — realistic Python source."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Edge case: a literal close tag inside a string that must NOT split the parser.
# The split boundary is the full sequence of the content-close tag immediately
# followed by a system-reminder open tag. A bare content-close tag alone is
# insufficient (spec §8 edge case, §6.2 anchor strength).
CLOSE_TAG_LITERAL = "</content>"


@dataclass(frozen=True)
class RouteRule:
    """A single routing rule resolved by RouteResolutionStage."""

    provider: str
    model: str
    weight: float = 1.0
    fallback: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "weight": self.weight,
            "fallback": self.fallback,
            "metadata": dict(self.metadata),
        }


class RouteTable:
    """Ordered list of RouteRules; first match wins."""

    def __init__(self, rules: list[RouteRule] | None = None) -> None:
        self._rules: list[RouteRule] = list(rules or [])

    def add(self, rule: RouteRule) -> None:
        self._rules.append(rule)

    def resolve(self, session_id: str) -> RouteRule | None:
        if not self._rules:
            return None
        idx = hash(session_id) % len(self._rules)
        return self._rules[idx]

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)


def load_config(path: str) -> dict[str, Any]:
    """Load a JSON config file; return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("Config root is not a dict: %s", type(data))
            return {}
        return data
    except FileNotFoundError:
        logger.info("Config file not found: %s", path)
        return {}
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return {}


def resolve_model_route(
    model_name: str,
    registry: dict[str, Any],
    default: str = "step-3.5-flash-2603",
) -> str:
    """Resolve a model name via the model registry, falling back to *default*."""
    entry = registry.get(model_name)
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and "upstream_model" in entry:
        return str(entry["upstream_model"])
    return default


def _sanitize(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_", "."))


class ProviderClient:
    """Minimal upstream HTTP client wrapper (real one uses _HTTP_SESSION)."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # Real implementation delegates to proxy.state._HTTP_SESSION.request(...)
        logger.debug("POST %s headers=%s", url, list(headers))
        return {"status": "ok", "url": url}


if __name__ == "__main__":
    table = RouteTable()
    table.add(RouteRule(provider="stepfun", model="step-3.5-flash-2603"))
    table.add(RouteRule(provider="deepseek", model="deepseek-chat"))
    print(f"Loaded {len(table)} route rules")
'''


_SUB_AGENTS_MD = """\
# `src/proxy/sub/AGENTS.md` — Route Resolution Subfolder

> You are reading a file inside `src/proxy/sub/`. This sub-AGENTS.md
> supplements the parent `src/proxy/AGENTS.md` conventions below. Apply
> these rules only when the file under modification lives inside this
> subtree. The root `AGENTS.md` (Tier-1) is handled separately by
> `MotivatorStage`; this file is Tier-2 and is pinned via the read-tool
> capture pipeline.

## Module Purpose

This subfolder holds the route-resolution helpers and the provider-client
shim. It lives under `src/proxy/sub/` and is the second layer of the
proxy's dispatch chain (after `CleanupStage`, before `SecurityStage`).
All model routing logic MUST live here — never inline route decisions in
pipeline stage modules. The `RouteTable` is the single source of truth
for provider/model selection; `ProviderClient` is the single source of
truth for upstream HTTP calls from this subtree.

## Files in this module

| File | Purpose | Lines | Owner |
|---|---|---|---|
| `route.py` | `RouteTable` + `RouteRule` dataclass — ordered rule list, first-match semantics | ~180 | proxy-core |
| `provider_client.py` | Thin wrapper around `proxy.state._HTTP_SESSION` for upstream calls | ~95 | proxy-core |
| `config.py` | `load_config(path)` — JSON config loader with error fallback | ~45 | proxy-core |
| `sanitize.py` | `_sanitize(value)` — alphanumeric + `-_.` filter for user-supplied tokens | ~12 | proxy-core |
| `retry.py` | `with_retry(fn, max_attempts=3)` — exponential backoff for upstream calls | ~60 | proxy-core |
| `circuit.py` | `CircuitBreaker` state machine — open/half-open/closed transitions | ~110 | proxy-core |
| `metrics.py` | `record_latency(provider, ms)` — push to `state.request_counter` | ~30 | proxy-core |
| `types.py` | `ProviderEndpoint`, `RoutePredicate` type aliases | ~20 | proxy-core |
| `__init__.py` | Re-exports `RouteTable`, `RouteRule`, `ProviderClient` | ~8 | proxy-core |

## Cross-References

- `src/proxy/pipeline/AGENTS.md` — the 9-stage pipeline that consumes routes resolved here
- `src/proxy/AGENTS.md` — lock hierarchy and singleton rules (Section "Lock Hierarchy")
- `src/proxy/state.py` — `vfs_lock` (#9), `_HTTP_SESSION` singleton, rate limiters
- `src/clients/AGENTS.md` — client adapters that feed session IDs into route resolution
- `src/proxy/context_v2.py` — V2 chain-hash pipeline that caches upstream responses
- Parent `AGENTS.md` (below) — project-wide invariants that also apply here

## Navigation

```
/repo/project/
├── src/
│   ├── proxy/
│   │   ├── pipeline/           ← consumes routes from this subfolder
│   │   │   ├── cleanup.py
│   │   │   ├── dispatch.py
│   │   │   ├── route_resolution.py   ← calls RouteTable.resolve()
│   │   │   ├── security.py
│   │   │   ├── context_optimize.py
│   │   │   ├── motivator.py
│   │   │   ├── provider_compat.py
│   │   │   ├── upstream.py
│   │   │   └── post_process.py
│   │   ├── sub/                ← YOU ARE HERE
│   │   │   ├── __init__.py
│   │   │   ├── route.py
│   │   │   ├── provider_client.py
│   │   │   ├── config.py
│   │   │   ├── sanitize.py
│   │   │   ├── retry.py
│   │   │   ├── circuit.py
│   │   │   ├── metrics.py
│   │   │   └── types.py
│   │   ├── agent/
│   │   ├── canary.py
│   │   ├── state.py
│   │   └── context_v2.py
│   └── clients/
│       ├── opencode_adapter.py
│       ├── kilocode_adapter.py
│       └── model_registry.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│       └── tier2_payload.py    ← YOU ARE HERE
└── AGENTS.md                   ← Tier-1 root (NOT this file)
```

## Lock Hierarchy (relevant subset)

When this subfolder's code acquires locks, it MUST follow the global ordering
defined in `src/proxy/state.py:234-250`:

| # | Lock | Type | Notes |
|---|---|---|---|
| #1 | `context_config_lock` | RLock | guard route-config mutations |
| #4 | `tool_cache_lock` | Lock | guard provider-client response cache |
| #5 | `request_counter_lock` | Lock | metrics only — never block on I/O while holding |
| #9 | `vfs_lock` | Lock | only if writing config snapshots to VFS |

**Rule:** acquire in ascending order. Never hold #4 while waiting for #9.
Never hold #1 while waiting for #4 (deadlock risk — see commit `2753fbe`).

### Lock acquisition patterns

```python
# CORRECT — ascending order, short critical section
import proxy.state as state
with state.request_counter_lock:
    state.request_counter += 1

# CORRECT — nested, still ascending
with state.context_config_lock:
    config = dict(state.context_config)
    with state.tool_cache_lock:
        cache_snapshot = dict(state.tool_cache)

# WRONG — descending order (deadlock)
with state.vfs_lock:
    with state.context_config_lock:  # NEVER — #9 then #1
        ...
```

## Key Invariants

1. **First-match wins.** `RouteTable.resolve()` returns the first rule whose
   predicate matches. Reordering rules changes behavior — document any
   reordering in the commit message. The `RouteRule.weight` field is used
   only for logging/observability, NOT for selection.
2. **No bare `except:`**. Always catch `InfrastructureError`,
   `ConstraintViolation`, and `QueryTimeoutError` explicitly. Bare except
   swallows the error hierarchy the proxy relies on for fail-closed behavior.
   See `src/memory/query_executor.py` for the error hierarchy definition.
3. **Provider client is a thin wrapper.** It MUST NOT cache responses beyond
   a single request — caching is owned by `context_v2.py` chain state.
   The `_HTTP_SESSION` singleton in `proxy.state` handles connection pooling.
4. **Config is immutable after load.** `load_config()` returns a fresh dict
   each call. Callers that mutate the returned dict must `copy.deepcopy()`
   first. Never cache the returned dict across requests.
5. **Sanitize on ingress, not egress.** User-supplied tokens are sanitized
   by `_sanitize()` before any storage or routing decision. Never trust
   raw input past the dispatch boundary. This includes session IDs, model
   names, and provider identifiers.
6. **Circuit breaker is per-provider.** The `CircuitBreaker` in `circuit.py`
   tracks failure counts per `(provider, model)` pair. It MUST NOT share
   state across providers — a failing StepFun must not block DeepSeek calls.
7. **Retry is bounded.** `with_retry(fn, max_attempts=3)` uses exponential
   backoff with jitter. It MUST NOT retry on `ConstraintViolation` (duplicate
   key) — only on `InfrastructureError` and `QueryTimeoutError`.
8. **Metrics are fire-and-forget.** `record_latency()` increments counters
   under `request_counter_lock`. It MUST NOT perform I/O or acquire additional
   locks. Slow metrics collection must not block the request path.

## Landmines

| # | Landmine | Mitigation |
|---|---|---|
| 1 | `RouteTable.resolve()` uses `hash(session_id)` — non-deterministic across Python runs (PYTHONHASHSEED). For stable tests, seed with `PYTHONHASHSEED=0` or use the `deterministic=True` flag. | Pin seed in tests |
| 2 | `load_config()` returns `{}` on `JSONDecodeError` — silent failure. Callers must validate the returned dict is non-empty before use. | Assert non-empty |
| 3 | `ProviderClient.post()` logs the API key header name but NOT the value — safe for debug logs. Never log `self._api_key` directly. | Code review |
| 4 | `CLOSE_TAG_LITERAL = "</content>"` — a string that looks like an XML close tag. The Tier-2 parser MUST anchor on the full boundary sequence (content-close immediately followed by system-reminder-open), not the last content-close in the body. | Spec §6.2 |
| 5 | `CircuitBreaker` uses `time.monotonic()` for timeout tracking. Do NOT use `time.time()` — it is not monotonic and can cause premature state transitions under NTP adjustments. | Use monotonic clock |
| 6 | `with_retry()` adds jitter via `random.uniform(0, base)`. This is NOT cryptographically secure — do NOT use for security-sensitive operations. | Use `secrets` module for crypto |
| 7 | `_sanitize()` strips spaces — a path like `/my project/file.py` becomes `/myproject/file.py`. Never pass unsanitized paths to filesystem operations. | Sanitize early |
| 8 | `RouteTable` is not thread-safe. Concurrent `add()` calls from multiple requests can lose rules. Always hold `context_config_lock` when mutating the table. | Lock before mutate |

## Code Conventions

### Logging
```python
logger.info("route resolved", extra={"request_id": ctx.request_id})
logger.debug("provider=%s model=%s", rule.provider, rule.model)
```

### Error handling
```python
try:
    result = client.post("/v1/chat/completions", payload)
except InfrastructureError:
    raise  # let retry logic upstream handle it
except ConstraintViolation as exc:
    logger.error("constraint violation: %s", exc)
    return {"error": "constraint_violation"}
except QueryTimeoutError:
    logger.warning("upstream timeout after %s s", timeout)
    raise  # triggers circuit breaker
```

### State mutation
```python
import proxy.state as state
state.request_counter += 1  # CORRECT — module-level mutation

# WRONG — creates local variable
from proxy.state import request_counter
request_counter += 1
```

### Deep-copy discipline
```python
import copy

def update_config(new_entries):
    with state.context_config_lock:
        merged = copy.deepcopy(state.context_config)
        merged.update(new_entries)
        state.context_config.clear()
        state.context_config.update(merged)
```

### Re-import pattern
```python
def get_route_table():
    from proxy.sub.route import _table_instance  # re-import for current value
    return _table_instance
```

## Behavioral Contracts

- `RouteTable.resolve(session_id)` MUST return `None` for an empty table.
- `load_config(path)` MUST return `{}` (not raise) when the file is missing.
- `ProviderClient.post(path, payload)` MUST set the `Authorization` header
  to `Bearer {api_key}` and the `Content-Type` to `application/json`.
- `_sanitize(value)` MUST strip all characters except `[a-zA-Z0-9._-]`.
- `with_retry(fn, max_attempts=3)` MUST call `fn` at least once and at most
  `max_attempts` times. It MUST raise the last exception if all attempts fail.
- `CircuitBreaker.is_open(provider)` MUST return `True` after 3 consecutive
  failures within a 30-second window. It MUST transition to half-open after
  the cooldown period.
- `record_latency(provider, ms)` MUST increment `state.request_counter` and
  append to `state.param_mod_details` (under `request_counter_lock`).

## Test Patterns

```python
def test_route_table_resolve_empty():
    table = RouteTable()
    assert table.resolve("any-session") is None

def test_route_table_resolve_deterministic():
    table = RouteTable([RouteRule(provider="stepfun", model="step-3.5-flash-2603")])
    # PYTHONHASHSEED=0 for deterministic hash
    result = table.resolve("test-session")
    assert result.provider == "stepfun"

def test_load_config_missing():
    result = load_config("/nonexistent/path/config.json")
    assert result == {}

def test_sanitize_strips_special_chars():
    assert _sanitize("hello world!@#") == "helloworld"
    assert _sanitize("a-b_c.d") == "a-b_c.d"
```

## External Dependencies

- `proxy.state._HTTP_SESSION` — the shared `requests.Session()` for upstream calls
- `proxy.state.context_config_lock` — RLock guarding route-config mutations
- `proxy.state.tool_cache_lock` — Lock guarding provider-client response cache
- `memory.query_executor.InfrastructureError` — retryable infrastructure error
- `memory.query_executor.ConstraintViolation` — non-retryable duplicate-key error
- `memory.query_executor.QueryTimeoutError` — retryable timeout error

## Detailed Design — Route Resolution Algorithm

The `RouteTable.resolve(session_id)` algorithm:

1. If the table is empty, return `None` immediately (no fallback).
2. Compute `idx = hash(session_id) % len(self._rules)`.
3. Return `self._rules[idx]`.

This is a simple hash-based selection, NOT weighted random. The `weight`
field on `RouteRule` is purely for observability (exported to metrics).

### Why hash-based?

- Deterministic: the same session always routes to the same provider
  (enables cache hit in `context_v2.py`).
- O(1) selection — no iteration over rules.
- No state — no need to track "last used" or "round robin" position.

### Fallback behavior

When the selected provider's circuit breaker is open, `resolve()` does
NOT automatically fall back. The caller (`RouteResolutionStage`) handles
fallback by calling `resolve()` with a modified session ID:

```python
rule = table.resolve(session_id)
if rule and circuit_breaker.is_open(rule.provider):
    # Fallback: hash with a salt to get a different provider
    rule = table.resolve(session_id + ":fallback")
```

### Thread safety

`RouteTable` itself is NOT thread-safe. All mutations (add/remove/clear)
must be performed under `context_config_lock`. Reads (`resolve()`) are
lock-free because Python's GIL protects simple attribute access — but
concurrent mutation during iteration is undefined behavior.

## Detailed Design — Circuit Breaker State Machine

```
CLOSED  --[3 failures]-->  OPEN  --[cooldown]-->  HALF_OPEN  --[1 success]-->  CLOSED
                                                          |
                                                      [failure]
                                                          |
                                                          v
                                                        OPEN
```

- **CLOSED**: All requests pass through. Failures are counted.
- **OPEN**: All requests are rejected immediately (fail-fast). After
  `cooldown_seconds` (default 30), transition to HALF_OPEN.
- **HALF_ONE**: The next request is allowed through. If it succeeds,
  transition to CLOSED. If it fails, transition back to OPEN.

The breaker uses `time.monotonic()` for timeout tracking — NOT
`time.time()` (which is not monotonic and can cause premature transitions
under NTP adjustments).

## Detailed Design — Retry with Exponential Backoff

```python
def with_retry(fn, max_attempts=3, base_delay=0.5, max_delay=8.0):
    for attempt in range(max_attempts):
        try:
            return fn()
        except InfrastructureError:
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            time.sleep(delay + jitter)
        except QueryTimeoutError:
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay)
```

- Retries on `InfrastructureError` (transient) and `QueryTimeoutError`.
- Does NOT retry on `ConstraintViolation` (deterministic failure).
- Uses exponential backoff with full jitter (0-50% of delay).
- Caps delay at `max_delay` (default 8 seconds).

## Performance Characteristics

| Operation | Time Complexity | Lock Held |
|---|---|---|
| `RouteTable.resolve()` | O(1) | None |
| `RouteTable.add()` | O(1) amortized | `context_config_lock` |
| `load_config()` | O(file size) | None |
| `_sanitize(value)` | O(len(value)) | None |
| `ProviderClient.post()` | O(network latency) | None (uses `_HTTP_SESSION`) |
| `with_retry(fn)` | O(attempts * fn_time) | None |
| `CircuitBreaker.is_open()` | O(1) | None |
| `record_latency()` | O(1) | `request_counter_lock` |

## Monitoring and Observability

The `metrics.py` module exports the following counters (all under
`request_counter_lock`):

- `state.request_counter` — total requests processed
- `state.tool_cache_hits` — provider-client cache hits
- `state.tool_cache_misses` — provider-client cache misses
- `state.vfs_cache_hits` — VFS cache hits (if config uses VFS)
- `state.vfs_cache_misses` — VFS cache misses
- `state.vfs_write_conflicts` — VFS write conflict count
- `state.log_overflow_count` — log buffer overflow count

## Additional Notes

This subfolder is part of the proxy's Tier-2 instruction pinning system.
When opencode reads a file in this subtree, it appends both this
sub-AGENTS.md and the parent `src/proxy/AGENTS.md` to the tool result.
The proxy's capture pipeline extracts these blocks and pins them as a
stable system message so they survive V2 eviction.

For the full specification, see:
`docs/architecture/tier2-sub-agents-md-pinning-spec-v21.md`

## Size Notes

This block is ~17 KB of realistic AGENTS.md content — sections, bullets,
code snippets, landmine tables — matching the Tier-2 suffix size target
from spec §7 ("~-33 KB stripped per read"). It is intentionally verbose
to exercise the Tier-2 capture + strip pipeline at production scale.
"""

_PARENT_AGENTS_MD = """\
# `src/proxy/AGENTS.md` — Proxy Core (sub-file, NOT root)

> This file is `src/proxy/AGENTS.md` — a SUB-FILE, NOT the project root
> (F1/SC2 upheld). The root is `/repo/project/AGENTS.md` (Tier-1). This
> file is Tier-2 and is attached by opencode when any file under
> `src/proxy/` is read. It defines the conventions for the proxy shell,
> the 9-stage pipeline, the lock hierarchy, and the singleton-state rules.

## Module Purpose

The proxy shell owns:
- Flask app + routes (`app.py`) and the admin blueprint (`admin.py`)
- The single mutable-state module `proxy.state` (locks, counters, rate
  limiters, singletons)
- The 9-stage request pipeline (`pipeline/`)
- The canary HMAC subsystem (`canary.py`)
- Skill infrastructure (`skills/`) + concrete skill classes
- Tool subsystem (`tool_pipeline/`) + tool argument normalization
  (`tool_parsing/`)
- Agent + AgentRegistry (`agent/`)
- Session + workspace state (`session.py`, `chat_history.py`,
  `workspace_state.py`, `eviction_coordinator.py`)
- Event log + telemetry (`event_log.py`)
- VFS virtual filesystem layer (`vfs.py`)

## Files in this module

| File | Purpose | Lines | Owner |
|---|---|---|---|
| `app.py` | Flask routes — `_require_api_key`, `_enforce_fail_closed`, `parse_request`, admin blueprint | ~420 | proxy-core |
| `admin.py` | Admin blueprint — `/v1/models`, `/health`, debug endpoints | ~180 | proxy-core |
| `state.py` | Single source of truth for ALL shared mutable state | ~600 | proxy-core |
| `lock_ordering.py` | `make_lock(position, name, lock_class)` factory | ~45 | proxy-core |
| `types.py` | `RequestContext` dataclass, `BackendError`, `SubagentIngress` | ~80 | proxy-core |
| `session.py` | Session ID resolution + `cleanup_old_sessions` | ~95 | proxy-core |
| `chat_history.py` | `ChatHistoryCache` LRU dict + read-through to SQLite | ~130 | proxy-core |
| `workspace_state.py` | `WorkspaceStateStore` — per-session active workspace | ~110 | proxy-core |
| `eviction_coordinator.py` | `build_and_process_deferred_clears` — lock handoff | ~75 | proxy-core |
| `event_log.py` | SQLite `logging.Handler` + `BoundedQueueHandler` | ~150 | proxy-core |
| `mode.py` | `Mode` enum + `resolve_mode` + `apply_mode` | ~90 | proxy-core |
| `skill_dispatcher.py` | `SkillDispatcher` + `DispatchResult` | ~120 | proxy-core |
| `skill_lifecycle.py` | `register_lifecycle_callbacks(agent)` | ~40 | proxy-core |
| `skill_guard.py` | `check_sub_agent_proxy_access()` — CAT-3 trust gate | ~65 | proxy-core |
| `skill_resolver.py` | `resolve_mode` (fallback deleted in C3) | ~55 | proxy-core |
| `model_registry.py` | `_model_registry`, `resolve_model_route` | ~200 | proxy-core |
| `openai_compat.py` | `build_openai_response()` — synthetic response shape | ~35 | proxy-core |
| `streaming.py` | `make_sse_chunk`, `stream_response` | ~100 | proxy-core |
| `requery.py` | `requery_llm()` — re-query loop after memory tool execution | ~85 | proxy-core |
| `tool_loop_legacy.py` | `MessageBuffer` (legacy, do not confuse with new one) | ~70 | proxy-core |
| `vfs.py` | Virtual filesystem layer — caches, write-through, rollback | ~250 | proxy-core |
| `context.py` + `context_v2.py` | Context V2 chain-hash pipeline (V1 deleted) | ~1400 | proxy-core |
| `subagent.py` | Sub-agent ingress detection helpers | ~60 | proxy-core |
| `utils.py` | `_now_iso`, `safe_content`, `extract_text_content` | ~90 | proxy-core |
| `canary.py` | HMAC-signed canary parser + verifier | ~480 | proxy-core |
| `session_tokens.py` | HMAC-signed client identity | ~55 | proxy-core |
| `rate_limiter.py` | `RateLimiter(max_requests, window_seconds, persist=)` | ~100 | proxy-core |
| `client_instructions.py` | Tier-1 instruction extraction (`extract_client_instructions`) | ~115 | proxy-core |

## Pipeline Stage Order (fixed — DO NOT REORDER)

```
Cleanup → Dispatch → RouteResolution → Security → ContextOptimize → Motivator → ProviderCompat → Upstream → PostProcess
```

Each stage is a singleton defined in `src/proxy/pipeline/stages.py`.
`app.py` resets `DispatchStage._features` per request — no other stage
holds mutable instance state.

## Lock Hierarchy (full — authoritative)

| # | Lock | Where | Type |
|---|---|---|---|
| #0 | `AgentRegistry._lock` | `agent/agent_registry.py:46` | coarsest — global |
| #1 | `context_config_lock` | `state.py:100` | RLock |
| #2 | `chat_histories_lock` | `state.py` via `chat_history.py:13` | RLock |
| #3 | `chat_summaries_lock` | `state.py:102` | |
| #4 | `tool_cache_lock` | `state.py:101` | |
| #5 | `request_counter_lock` | `state.py:99` | |
| #6 | `active_workspace_lock` | `state.py:373` | RLock |
| #7 | `Agent._mode_lock` | `agent/agent.py:149` | per-instance |
| #8 | `AgentTelemetry._lock` | `pipeline/agent_telemetry.py` | per-instance |
| #9 | `vfs_lock` | `state.py:105` | |
| #10 | `_chain_state_lock` | `state.py:106` | |
| #11 | `canary_verify_cache_lock` | `state.py:107` | |
| #12 | `log_overflow_count_lock` | `state.py:175` | leaf |

**Rule:** acquire in ascending order; never hold a lower-numbered lock while
acquiring a higher one. Positions #7 and #8 are per-instance — not global.
Violations caused 8+ adversarial-review-round deadlocks.

### Lock ordering examples

```python
# CORRECT — ascending, short critical sections
with state.request_counter_lock:        # #5
    state.request_counter += 1

# CORRECT — nested ascending
with state.context_config_lock:         # #1
    config = dict(state.context_config)
    with state.tool_cache_lock:         # #4 (still ascending)
        cache = dict(state.tool_cache)

# WRONG — would deadlock
with state.vfs_lock:                    # #9
    with state.context_config_lock:     # #1 — LOWER number, HIGHER in stack
        ...  # DEADLOCK if another thread does #1 then #9
```

## Rate Limiters (`state.py:128-140`)

| Limiter | Limit | Window | Persist | Notes |
|---|---|---|---|---|
| `command_rate_limiter` | 500 | 60s | yes | |
| `enhancer_rate_limiter` | 500 | 60s | yes | ^^imp gate (DEAD path) |
| `escalation_rate_limiter` | 5 | 3600s | **no** | ticket→work sub-agent escalation |
| `skill_rate_limiter` | 1000 | 60s | yes | |
| `skill_ip_rate_limiter` | 3000 | 60s | yes | |
| `task_rate_limiter` | 2000 | 60s | yes | |
| `chat_rate_limiter` | 1000 | 60s | yes | `/v1/chat/completions` |
| `admin_rate_limiter` | 1000 | 60s | yes | `/v1/models`, `/health`, debug |

All auto-discovered + reset by `reset_all_rate_limiters()` (`state.py:143`).

## Singletons (single-instance rule)

| Singleton | Location | Notes |
|---|---|---|
| `_HTTP_SESSION` | `state.py:528` | One `requests.Session()` shared |
| `_chain_state` | `state.py:342` | OrderedDict keyed on `(session_id, model_name)` |
| `canary_verify_cache` | `state.py:562` | OrderedDict, max 4096, TTL 3600s |
| `Neo4j circuit breaker` | `state.py:518` | `CircuitBreaker(3, 30.0)` |
| `ToolRegistry` | `tool_pipeline/tool_registry.py:97` | Module-level singleton |
| `SkillRegistry` | `skills/skill_registry.py` | Double-checked locking singleton |
| `AgentRegistry` | `agent/agent_registry.py:29` | Singleton via `agent_registry.registry` |
| `workspace_store` | `state.py:384` | `WorkspaceStateStore` instance |

**Always acquire the singleton's lock through `proxy.state`/`proxy.X.lock`
— never `Lock()` locally.**

## Key Invariants

1. **Single mutable state owner.** `proxy.state` is the only module that
   defines shared locks/dicts/counters. Other modules import references.
2. **Lock ordering.** 13 positions; ascending order only.
3. **Agent mode lock is per-instance (#7).** Defined on the `Agent`
   instance in `agent.py:149`; NOT a global in `state.py`.
4. **`AppState` is immutable config.** All mutable state lives at module
   level, NOT in `AppState`.
5. **`_HTTP_SESSION` is the only HTTP session.** Per-request auth is via
   `headers=` param, not a fresh session.
6. **`parse_request` is pre-pipeline.** Called from `app.py` BEFORE
   `RequestPipeline.run()`; raises `StageAbort((dict, 401))` for invalid
   session token.
7. **VFS scope is env-driven.** `PROXY_VFS_SCOPE` = `"all"` (default) |
   `"subs"` | `"main"`. Invalid values log WARNING and fall back to `"all"`.

## Landmines

| # | Landmine | What NOT to do |
|---|---|---|
| 1 | Tool argument normalization case folding | Never add `.lower()` fallback that bypasses `valid_params` |
| 2 | Lock ordering violations | Never hold #0 while waiting for #7 |
| 3 | Canary mid-message parsing | Always route through `parse_canary` — never inline |
| 4 | Neo4j driver multiplicity | Use the shared driver from `QueryExecutor` — never create a new one |
| 5 | SQLite parameter limit | Chunk large IN clauses — SQLite caps at ~999 params |
| 6 | Shallow copy of message lists | Always `copy.deepcopy()`, never `list()`, in pipeline stages |
| 7 | TOCTOU on `active_mode` | Use `agent.snapshot_mode()`, never read `agent.active_mode` unlocked |
| 8 | Bare `except Exception: pass` | Always handle `InfrastructureError`, `ConstraintViolation`, `QueryTimeoutError` |
| 9 | `NEO4J_AUTH` separator | Use `:` not `/` — mismatch was a bug |
| 10 | Canary degradation | When `PROXY_SESSION_SECRET` is unset, canary signing returns unsigned — intentional fail-safe |

## Code Conventions

### Logging
```python
logger.info("stage complete", extra={"request_id": ctx.request_id})
```

### Re-import pattern for mutable state
```python
def get_chain_state():
    from proxy.context_v2 import _chain_state  # re-import for current value
    return _chain_state
```

### Deep-copy discipline
```python
import copy

def stage_process(messages):
    msgs = copy.deepcopy(messages)  # CORRECT — no aliasing
    # WRONG: msgs = list(messages) — shallow copy aliases dict items
    return msgs
```

### Sub-agent delegation rule
Main agents delegate non-trivial edits to sub-agents. Read-only
investigation is fine on the main agent. See `src/AGENTS.md` for the
full rule.

## Behavioral Contracts

- `CleanupStage` MUST run before any other stage — it normalizes the
  incoming request shape and rejects malformed payloads.
- `DispatchStage` MUST resolve the request to a pipeline entry point
  (main chat, admin, or sub-agent delegation).
- `RouteResolutionStage` MUST produce a valid `(provider, model)` pair
  before `UpstreamStage` is reached.
- `SecurityStage` MUST validate the session token; on failure, abort
  with HTTP 401.
- `ContextOptimizeStage` MUST assemble the full outbound message list
  (system + history + summaries + VFS context) before `MotivatorStage`
  injects the Tier-1 instruction pin.
- `MotivatorStage` MUST inject the `<CLIENT_PROJECT_INSTRUCTIONS>` block
  from the Tier-1 AGENTS.md root, NOT from any subfolder.
- `ProviderCompatStage` MUST normalize all tool schemas to the upstream
  provider's expected format before `UpstreamStage` serializes the payload.
- `UpstreamStage` MUST POST to the resolved backend and stream the SSE
  response back to the client.
- `PostProcessStage` MUST sync chat history and kick off background SOTA
  extraction after the response is fully streamed.

## Detailed Design — Context V2 Pipeline

The V2 context pipeline (`context_v2.py`) is the sole context assembly
entry point (V1 was deleted). It implements a chain-hash caching scheme:

1. **Input**: `full_messages` (the complete inbound message list from
   the client, including all historical tool results).
2. **Eviction**: `_evict_with_cap()` whole-drops the oldest messages
   beyond the cap (atomic — never partial).
3. **Tier-2 Capture**: On `raw_block` (pre-trim), scan `role="tool"`
    messages for the full boundary (content-close tag immediately followed
    by system-reminder-open tag). Extract
   `Instruction from:` blocks and record them in
   `state.instruction_files_by_workspace`.
4. **Trim**: `_trim_raw_block_to_budget()` whole-drops the largest body
   messages first (NOT tail-cut). The last-message content trim is the
   last resort.
5. **Strip**: Write back the stripped tool content (file body only, no
   Tier-2 suffix) by index into the outbound message list.
6. **Pin Emit**: Before EVERY return in `_select_context_v2`, emit the
   `<CLIENT_SUB_INSTRUCTIONS>` block from the map.
7. **Outbound**: The final message list sent to upstream.

### Chain-hash caching

`compute_chain(messages)` produces a deterministic hash of the message
list (excluding `role="system"` messages, which don't affect the cache
key). If the hash matches a previously seen chain, the upstream call
is skipped and the cached response is returned.

### Return paths (all must emit the pin)

- `:841` — empty input (no-op, empty map)
- `:857-862` — degenerate cases (×4, no-op)
- `:881` — sub-agent delegation (reaches `_build_subagent_context`)
- `:913` — passthrough (no trim needed)
- `:931` — compute_chain failure (emit pin anyway — map populated)
- `:1183` — warm-path CAS failure (emit pin anyway)
- `:1208` — cold-init CAS failure (emit pin anyway)
- `:1222` — main return (most common path)

## Detailed Design — Sub-agent Ingress

Sub-agent requests are detected by `subagent._detect_ingress()` which
looks for the canary marker at position 0 of the system message. The
canary is HMAC-signed and contains:
- `skill_name`: the skill that spawned the sub-agent
- `state_json`: serialized state (workspace, mode, etc.)
- `timestamp`: for freshness validation (5-minute window)

The sub-agent inherits the parent's workspace via
`resolve_inherited_workspace()`. The Tier-2 instruction map is keyed
by workspace, so sub-agents automatically see the parent's pinned
sub-instructions.

### Sub-agent overflow paths

When a sub-agent's context exceeds the token budget, the pipeline takes
one of four paths:
- `:314` — fits-safe (no trimming needed)
- `:335` — no-first-user (degenerate, return as-is)
- `:361` — trimmed-fits (trim to budget, still fits)
- `:369` — final fallback (force-trim to hard limit)

All four paths MUST emit the Tier-2 pin — the sub-agent overflow paths
are the landmine-exposed ones (big tool results).

## Detailed Design — VFS Virtual Filesystem

The VFS layer (`vfs.py`) provides a virtual filesystem backed by SQLite.
It supports:
- Read-through caching (LRU, max 1024 entries)
- Write-through to SQLite on flush
- Rollback snapshots (for atomic multi-file operations)
- Conflict detection (two sessions writing the same file)

All VFS operations are guarded by `vfs_lock` (#9). The VFS cache is
cleared by `_reset_all_shared_state()` in the test harness.

## Performance Characteristics

| Operation | Time Complexity | Lock Held |
|---|---|---|
| `parse_request()` | O(request size) | None |
| `RequestPipeline.run()` | O(stages × messages) | Stage-specific |
| `_select_context_v2()` | O(messages) | `_chain_state_lock` |
| `_evict_with_cap()` | O(messages) | None |
| `_trim_raw_block_to_budget()` | O(messages log n) | None |
| `compute_chain(messages)` | O(messages) | None |
| `AgentRegistry.get_or_load()` | O(1) amortized | `AgentRegistry._lock` |
| `workspace_store.get()` | O(1) | `active_workspace_lock` |

## Monitoring and Observables

The proxy exposes the following observability surfaces:

- **Log messages**: All operational logs include `request_id` via
  `extra={"request_id": ctx.request_id}`. Use structured logging
  (JSON format in production) for easy aggregation.
- **Rate limiter state**: `state.reset_all_rate_limiters()` clears all
  8 rate limiters. Per-limiter state is accessible via the
  `RateLimiter._buckets` dict (read-only, for debugging).
- **Chain cache hit rate**: The ratio of `compute_chain` cache hits
  to total calls. A high hit rate (>80%) indicates stable sessions.
  A drop after a feature change may indicate a cache-bust regression.

## Cross-References

- `/repo/project/AGENTS.md` — Tier-1 root (NOT this file)
- `src/AGENTS.md` — root project conventions and the top-10 landmines table
- `src/proxy/pipeline/AGENTS.md` — 9-stage pipeline lifecycle
- `src/proxy/tool_pipeline/AGENTS.md` — ToolRegistry + ToolGuard + 17 actions
- `src/proxy/tool_parsing/AGENTS.md` — schema-driven argument normalization
- `src/proxy/skills/AGENTS.md` — SkillRegistry + TrustToken + custom JSON loader
- `src/proxy/agent/AGENTS.md` — Agent dataclass + AgentRegistry
- `src/proxy/canary.py` — HMAC-signed canary parser + verifier
- `src/memory/AGENTS.md` — SQLite + Neo4j + SOTA
- `src/clients/AGENTS.md` — ClientAdapter + adapters + registries
- `tests/AGENTS.md` — test layers, builders, mocks, wipe gate, fixtures

## Additional Notes

This file is part of the proxy's Tier-2 instruction pinning system.
When opencode reads any file under `src/proxy/`, it appends this
AGENTS.md (and any deeper sub-AGENTS.md files) to the tool result.
The proxy's capture pipeline extracts these blocks and pins them as a
stable system message so they survive V2 eviction.

For the full specification, see:
`docs/architecture/tier2-sub-agents-md-pinning-spec-v21.md`

## Size Notes

This block is ~17 KB of realistic AGENTS.md content — sections, bullets,
code snippets, landmine tables, lock hierarchy, singleton rules. Combined
with the sub-AGENTS.md block above, the total Tier-2 suffix is ~34 KB —
matching spec §7's "~-33 KB stripped per read" measurement.
"""

TIER2_SUFFIX = (
    "<system-reminder>\n"
    f"Instructions from: {_SUB_AGENTS_PATH}\n"
    f"{_SUB_AGENTS_MD}\n"
    f"Instructions from: {_PARENT_AGENTS_PATH}\n"
    f"{_PARENT_AGENTS_MD}\n"
    "</system-reminder>"
)

FILE_BODY = f"<path>{_READ_PATH}</path><type>file</type><content>\n{_FILE_BODY}</content>"

RAW_TOOL_CONTENT = FILE_BODY + TIER2_SUFFIX

INSTRUCTION_BLOCKS = _extract_instruction_blocks(TIER2_SUFFIX)

RAW_TOOL_BODY_WITH_LITERAL_CLOSE = (
    "<path>/abs/edge/case.py</path><type>file</type><content>\n"
    '"""Edge case: file body with a literal </content> not followed by <system-reminder>."""\n'
    "\n"
    "# This string contains a literal close tag that must NOT be the Tier-2 split point.\n"
    'CLOSE_TAG = "</content>"\n'
    "\n"
    "def example():\n"
    "    payload = {\n"
    "        'tag': '</content>',\n"
    "        'description': 'must not trigger Tier-2 capture',\n"
    "    }\n"
    "    return payload\n"
    "\n"
    "# Another occurrence: data = \"</content>\" followed by more code.\n"
    'data = "</content>"\n'
    "result = len(data)\n"
    "</content>"
)

RAW_NON_READ_TOOL_RESULT = (
    "grep result:\n"
    "/abs/read/sub/file.py:12:from __future__ import annotations\n"
    "/abs/read/sub/file.py:14:import logging\n"
    "/abs/read/sub/file.py:15:import os\n"
    "/abs/read/sub/file.py:20:logger = logging.getLogger(__name__)\n"
    "/abs/read/sub/file.py:34:class RouteTable:\n"
    "/abs/read/sub/file.py:48:def resolve_model_route(\n"
    "/abs/read/sub/file.py:67:def _sanitize(value: str) -> str:\n"
    "/abs/read/sub/file.py:72:class ProviderClient:\n"
    "/abs/read/sub/file.py:85:    def post(self, path: str, payload: dict) -> dict:\n"
    "/abs/read/sub/file.py:93:if __name__ == \"__main__\":\n"
)


def test_fixture_shape():
    assert FILE_BODY + TIER2_SUFFIX == RAW_TOOL_CONTENT
    assert RAW_TOOL_CONTENT.count("</content><system-reminder>") == 1, (
        "canonical payload must carry the Tier-2 boundary exactly once; "
        f"found {RAW_TOOL_CONTENT.count('</content><system-reminder>')}"
    )
    assert FILE_BODY.endswith("</content>")
    assert TIER2_SUFFIX.startswith("<system-reminder>")
    assert len(INSTRUCTION_BLOCKS) == 2
    for abs_path, _content in INSTRUCTION_BLOCKS:
        assert os.path.isabs(abs_path), f"not absolute: {abs_path!r}"
    read_dir = os.path.dirname(_READ_PATH)
    first_instr_dir = os.path.dirname(INSTRUCTION_BLOCKS[0][0])
    assert read_dir.startswith(first_instr_dir + os.sep) or read_dir == first_instr_dir
    literal_close_idx = RAW_TOOL_BODY_WITH_LITERAL_CLOSE.find("</content>")
    after_close = RAW_TOOL_BODY_WITH_LITERAL_CLOSE[literal_close_idx + len("</content>"):]
    assert not after_close.startswith("<system-reminder>"), (
        "literal </content> must NOT be followed by <system-reminder>"
    )
    assert "<path>" not in RAW_NON_READ_TOOL_RESULT


if __name__ == "__main__":
    test_fixture_shape()
    print("OK: test_fixture_shape passed")
    print(f"FILE_BODY bytes: {len(FILE_BODY.encode('utf-8'))}")
    print(f"TIER2_SUFFIX bytes: {len(TIER2_SUFFIX.encode('utf-8'))}")
    print(f"RAW_TOOL_CONTENT bytes: {len(RAW_TOOL_CONTENT.encode('utf-8'))}")
    print(f"INSTRUCTION_BLOCKS: {len(INSTRUCTION_BLOCKS)} blocks")
    for p, c in INSTRUCTION_BLOCKS:
        print(f"  {p}: {len(c.encode('utf-8'))} bytes")
