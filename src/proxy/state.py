"""Single source of truth for all proxy shared state.

All locks and shared mutable dicts are defined ONCE here.
Every other proxy module imports REFERENCES from this module.
Never create new Lock() or dict() instances for shared state in other modules.

CRITICAL RULE: Integer counters (tool_cache_size, tool_cache_hits, tool_cache_misses)
MUST be mutated via:
    import proxy.state as state
    state.tool_cache_size += 1
NOT via:
    from proxy.state import tool_cache_size
    tool_cache_size += 1  # This creates a LOCAL variable, module-level unchanged!
"""

from collections import OrderedDict
import os
import sys
import time
import logging
import threading
import requests
from dataclasses import dataclass
from memory.circuit_breaker import CircuitBreaker
from proxy.rate_limiter import RateLimiter, RateLimitResult
from proxy.lock_ordering import make_lock
from proxy.agent.types import ModeState  # Phase 2A: re-export for proxy.state as third canonical home

__all__ = [
    "ModeState",
    "AppState",
    "request_counter",
    "request_id",
    "model_config_cache",
    "tool_cache",
    "tool_cache_size",
    "tool_cache_hits",
    "tool_cache_misses",
    "chat_summaries",
    "vfs_caches",
    "vfs_write_log",
    "VFS_WRITE_LOG_MAX",
    "vfs_rollback_snapshots",
    "VFS_ROLLBACK_TIMEOUT",
    "VFS_ROLLBACK_MAX",
    "request_counter_lock",
    "context_config_lock",
    "tool_cache_lock",
    "chat_summaries_lock",
    "param_mod_lock",
    "param_mod_log_lock",
    "vfs_lock",
    "canary_verify_cache_lock",
    "canary_verify_cache",
    "_CANARY_VERIFY_CACHE_MAX",
    "_CANARY_VERIFY_CACHE_TTL",
    "PARAM_MOD_LOG_INTERVAL",
    "param_mod_log_timestamps",
    "command_rate_limiter",
    "dedup_lock",
    "dedup_store",
    "enhancer_rate_limiter",
    "escalation_rate_limiter",
    "skill_rate_limiter",
    "skill_ip_rate_limiter",
    "task_rate_limiter",
    "chat_rate_limiter",
    "admin_rate_limiter",
    "log_overflow_count",
    "log_overflow_count_lock",
    "bump_log_overflow",
    "recovery_lock",
    "reset_all_rate_limiters",
    "load_all_rate_limiter_state",
    "MAX_CALLS_PER_ITERATION",
    "_HTTP_SESSION",
    "_chain_state",
    "_chain_state_lock",
    "CHAIN_STATE_MAX_ENTRIES",
    "invalidate_chain_state_for_session",
    "eviction_coordinator",
    "RateLimiter",
    "RateLimitResult",
    "CircuitBreaker",
]

# VFS type imports (TYPE_CHECKING to avoid circular dependency)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from proxy.vfs import VFSCache, VFSWriteRecord, VFSRollbackSnapshot

logger = logging.getLogger("proxy.state")

try:
    from config import CONFIG
except ImportError:
    CONFIG = {}

# =============================================================================
# Locks (created once, imported everywhere)
# =============================================================================
request_counter_lock = make_lock(5, "request_counter_lock", threading.Lock)
context_config_lock = make_lock(1, "context_config_lock", threading.RLock)
tool_cache_lock = make_lock(4, "tool_cache_lock", threading.Lock)
chat_summaries_lock = make_lock(3, "chat_summaries_lock", threading.Lock)
param_mod_lock = threading.Lock()
param_mod_log_lock = threading.RLock()  # rate-limits param-modification log messages
vfs_lock = make_lock(9, "vfs_lock", threading.Lock)
_chain_state_lock = make_lock(10, "_chain_state_lock", threading.Lock)
canary_verify_cache_lock = make_lock(11, "canary_verify_cache_lock", threading.Lock)

# Shared rate-limit state for parameter modification log messages.
# Hoisted here from app.py and provider_compat.py so both modules
# coordinate rate-limiting through one dict, one interval, and one lock.
PARAM_MOD_LOG_INTERVAL = int(os.environ.get("PARAM_MOD_LOG_INTERVAL", "60"))


class _ParamModLogDict(dict):
    """dict subclass that triggers cleanup of stale entries on write."""

    def __setitem__(self, key, value):
        # F2: Hold param_mod_log_lock across both the write and cleanup
        # to prevent RuntimeError from concurrent dict mutation during iteration.
        with param_mod_log_lock:
            super().__setitem__(key, value)
            _cleanup_param_log_timestamps()


param_mod_log_timestamps: dict[str, float] = _ParamModLogDict()

command_rate_limiter = RateLimiter(max_requests=500, window_seconds=60, name="command", persist=True)
enhancer_rate_limiter = RateLimiter(max_requests=500, window_seconds=60, name="enhancer", persist=True)
escalation_rate_limiter = RateLimiter(
    max_requests=5,
    window_seconds=3600,
    name="escalation",
    persist=False,
)
skill_rate_limiter = RateLimiter(max_requests=1000, window_seconds=60, name="skill", persist=True)
skill_ip_rate_limiter = RateLimiter(max_requests=3000, window_seconds=60, name="skill_ip", persist=True)
task_rate_limiter = RateLimiter(max_requests=2000, window_seconds=60, name="task", persist=True)
chat_rate_limiter = RateLimiter(max_requests=1000, window_seconds=60, name="chat", persist=True)
admin_rate_limiter = RateLimiter(max_requests=1000, window_seconds=60, name="admin", persist=True)


def reset_all_rate_limiters() -> None:
    """Reset every RateLimiter instance in proxy.state.

    Single source of truth for the "auto-discover RateLimiter attrs and
    call reset() on each" loop. Called from:

    - src/devtools/debug_session.py:_reset_all_shared_state
    - tests/integration/conftest.py:_reset_all_shared_state
    - tests/conftest.py:reset_rate_limiters autouse fixture

    Each limiter's reset() holds its own lock and zeros _entries,
    _call_count, _last_cleanup, and _last_save (the public API).

    Future-drift guard: asserts each limiter has a non-empty `_name`,
    which is the key DebugSession.rate_limit_status() uses to find it
    via the proxy.state attribute scan. A limiter without a name would
    be silently unreachable from the harness.
    """
    import sys as _sys
    self_module = _sys.modules[__name__]
    for _name in dir(self_module):
        attr = getattr(self_module, _name, None)
        if isinstance(attr, RateLimiter):
            if not attr._name:
                raise ValueError(
                    f"RateLimiter {_name!r} has empty _name — "
                    f"unreachable via rate_limit_status({_name!r}). Fix it."
                )
            attr.reset()


def load_all_rate_limiter_state() -> None:
    """Restore persisted RateLimiter state from SQLite for every
    persist=True instance in proxy.state.

    Mirrors the auto-discover pattern of ``reset_all_rate_limiters`` —
    iterates ``dir(self_module)`` and dispatches ``load_persisted_state``
    on every ``RateLimiter`` instance found. Called once at startup by
    ``proxy.app.initialize_proxy_at_startup`` AFTER ``memory.sync.init_db``
    has created the ``rate_limiter_state`` schema, so the SQLite call
    inside each limiter's hydration does NOT race schema creation.

    Per-limiter ``persist=False`` instances are no-ops inside
    ``load_persisted_state`` and return False without touching SQLite.

    Failures propagate — the caller (startup orchestration) decides how
    to handle a SQLite error during hydration. We do not silently swallow.

    The same future-drift guard applies: every RateLimiter must have a
    non-empty ``_name`` (the WHERE clause in the SELECT keys on it).
    """
    import sys as _sys
    self_module = _sys.modules[__name__]
    for _name in dir(self_module):
        attr = getattr(self_module, _name, None)
        if isinstance(attr, RateLimiter):
            if not attr._name:
                raise ValueError(
                    f"RateLimiter {_name!r} has empty _name — "
                    f"unreachable via rate_limit_status({_name!r}). Fix it."
                )
            attr.load_persisted_state()


log_overflow_count = 0
log_overflow_count_lock = make_lock(12, "log_overflow_count_lock", threading.Lock)

# Position #13 — SQLite corruption-recovery serialization.
# Acquired by memory.db_recovery.recover_corrupt_db() and
# memory.db_pool._create_connection() inline-recovery path. RLock because the
# recovery body may re-enter the gate via ensure_db_integrity() called from
# inside recover_corrupt_db(). Leaf lock — does not acquire any other proxy.state
# locks; safe to hold while calling into memory.db_recovery. See AGENTS.md
# "Lock ordering (positions #0–#13)" — this is the first position past #12.
recovery_lock = make_lock(13, "recovery_lock", threading.RLock)
# CRIT-3 fix: ``recovery_in_progress`` flag REMOVED. It was a process-wide
# "sticky" flag that, after one successful recovery, short-circuited every
# subsequent corruption event to ``ensure_db_integrity`` — permanently
# disabling auto-heal in long-running processes. Serialization is now
# provided solely by ``recovery_lock`` (RLock); the recovery body is
# idempotent on a healthy db (wasted work but not data loss).

# Position #14 — transparent request deduplication keyed completion map (CP4-1).
# Leaf lock — guards proxy.dedup.DedupStore's internal OrderedDict. The store
# also carries its own internal lock; this canonical position reserves #14 in
# the hierarchy so future wiring acquires it in ascending order. dedup.py does
# NOT import from state.py, so importing DedupStore here is cycle-free.
dedup_lock = make_lock(14, "dedup_lock", threading.Lock)

from proxy.dedup import DedupStore
dedup_store = DedupStore(lock=dedup_lock)


def bump_log_overflow() -> None:
    global log_overflow_count
    with log_overflow_count_lock:
        log_overflow_count += 1

# ── Canary verification cache (§5.4) ──────────────────────────────────────
# Session-scoped cache that remembers verified canaries so long-running
# sub-agents (>5 min) don't lose their ticket mode when the HMAC freshness
# window expires. Keyed by (session_id, sig_hash, state_hash, secret_hash).
# LRU eviction when over _CANARY_VERIFY_CACHE_MAX entries.
canary_verify_cache: "OrderedDict" = None  # populated at module init below

_CANARY_VERIFY_CACHE_MAX_RAW = os.environ.get("PROXY_CANARY_VERIFY_CACHE_MAX", "4096")
try:
    _CANARY_VERIFY_CACHE_MAX = int(_CANARY_VERIFY_CACHE_MAX_RAW)
    if _CANARY_VERIFY_CACHE_MAX < 1:
        logger = logging.getLogger("proxy.state")
        logger.warning(
            "PROXY_CANARY_VERIFY_CACHE_MAX=%s (< 1) — clamping to 1",
            _CANARY_VERIFY_CACHE_MAX_RAW,
        )
        _CANARY_VERIFY_CACHE_MAX = 1
except (ValueError, TypeError):
    logger = logging.getLogger("proxy.state")
    logger.warning(
        "PROXY_CANARY_VERIFY_CACHE_MAX=%r is not a valid integer — defaulting to 4096",
        _CANARY_VERIFY_CACHE_MAX_RAW,
    )
    _CANARY_VERIFY_CACHE_MAX = 4096

_CANARY_VERIFY_CACHE_TTL = 3600.0  # 1 hour — entries expire after this

# PROXY_BUG_REVIEW_COUNT — reviewers required per bug (default 2, valid 1..5)
try:
    _BUG_REVIEW_COUNT = int(os.environ.get("PROXY_BUG_REVIEW_COUNT", "2"))
    if not (1 <= _BUG_REVIEW_COUNT <= 5):
        raise ValueError(f"PROXY_BUG_REVIEW_COUNT must be 1..5, got {_BUG_REVIEW_COUNT}")
    PROXY_BUG_REVIEW_COUNT = _BUG_REVIEW_COUNT
except ValueError as e:
    logger.critical("Invalid PROXY_BUG_REVIEW_COUNT: %s — proxy will refuse to start", e)
    raise

# MAX_CALLS_PER_ITERATION — cap on memory tool calls per re-query iteration.
# Pin the default to 50. The only reader of the constant is
# tool_interception.py:28; memory.tool_loop and tool_pipeline.py hardcode 50 directly. Valid range [1, 200]; out-of-range values are clamped
# with a warning so a typo'd env var doesn't silently disable the loop.
MAX_CALLS_PER_ITERATION = int(os.environ.get("PROXY_MAX_CALLS_PER_ITERATION", "50"))
if not (1 <= MAX_CALLS_PER_ITERATION <= 200):
    logger.warning(
        "PROXY_MAX_CALLS_PER_ITERATION=%s out of range [1, 200], clamping",
        MAX_CALLS_PER_ITERATION,
    )
    MAX_CALLS_PER_ITERATION = max(1, min(200, MAX_CALLS_PER_ITERATION))

# =============================================================================
# Lock ordering (MUST be acquired in this order to prevent deadlocks)
# =============================================================================
# 0. AgentRegistry._lock          ← coarsest; must be acquired before all others (see agent_registry.py)
# 1. context_config_lock
# 2. chat_histories_lock          ← also used by ChatHistoryCache (re-entrant via RLock)
# 3. chat_summaries_lock
# 4. tool_cache_lock
# 5. request_counter_lock
# 6. active_workspace_lock      ← workspace state only; no cross-lock deps yet
# 7. agent._mode_lock             ← per-agent; mode resolution after workspace resolution
# 8. AgentTelemetry._lock         ← per-agent; motivator threshold tracking
# 9. vfs_lock                  ← VFS cache, write log, rollback snapshots
# 10. _chain_state_lock           ← V2 chain-hash state per (session_id, model_name)
# 11. canary_verify_cache_lock    ← canary verification cache (§5.4)
# 12. log_overflow_count_lock     ← leaf lock; never acquires other locks
# 13. recovery_lock               ← memory.db_recovery corruption-recovery serialization
#                                    (RLock — re-entry from ensure_db_integrity inside recover_corrupt_db)
# 14. dedup_lock                  ← leaf lock; DedupStore keyed completion map (CP4-1)
#
# Never acquire a lower-numbered lock while holding a higher-numbered one.
# If you need multiple locks, acquire them in ascending order.
# =============================================================================

# =============================================================================
# Shared mutable state (created once, imported everywhere)
# =============================================================================
request_counter = 0
request_id = ""
model_config_cache = OrderedDict()
tool_cache = OrderedDict()
tool_cache_size = 0
tool_cache_hits = 0
tool_cache_misses = 0
chat_summaries = {}

# =============================================================================
# VFS (Virtual Filesystem Layer) state
# =============================================================================
# Workspace-scoped VFS caches: workspace_id → VFSCache
vfs_caches = OrderedDict()  # workspace_id → VFSCache (OrderedDict for LRU eviction)

# Global write log (append-only, bounded)
vfs_write_log: list = []
VFS_WRITE_LOG_MAX = 10000

# Rollback snapshots for write-through recovery
vfs_rollback_snapshots: dict = {}  # tool_call_id → VFSRollbackSnapshot
VFS_ROLLBACK_TIMEOUT = 60   # seconds
VFS_ROLLBACK_MAX = 100      # max pending snapshots

# VFS configuration (env-driven)
VFS_TTL_SECONDS = int(os.environ.get("PROXY_VFS_TTL", "60"))
VFS_MAX_ENTRIES = int(os.environ.get("PROXY_VFS_MAX_ENTRIES", "500"))
VFS_MAX_ENTRY_BYTES = int(os.environ.get("PROXY_VFS_MAX_ENTRY_BYTES", str(1024 * 1024)))
VFS_ENABLED = os.environ.get("PROXY_VFS_ENABLED", "1") != "0"
VFS_MAX_WORKSPACES = int(os.environ.get("PROXY_VFS_MAX_WORKSPACES", "20"))
VFS_SCOPE = os.environ.get("PROXY_VFS_SCOPE", "all")  # "all" | "subs" | "main"
if VFS_SCOPE not in ("all", "subs", "main"):
    logger.warning("Invalid PROXY_VFS_SCOPE=%r, defaulting to 'all'", VFS_SCOPE)
    VFS_SCOPE = "all"

# =============================================================================
# Tier-2 sub-AGENTS.md pinning — instruction file map (guarded by vfs_lock #9)
# =============================================================================
instruction_files_by_workspace: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()
INSTRUCTION_FILES_MAX_PER_WORKSPACE = 64
INSTRUCTION_FILES_MAX_WORKSPACES = 128
_LAST_SCANNED_MAX = 1024
_last_scanned_by_tool_call: "OrderedDict[str, int]" = OrderedDict()


def record_instruction_file(workspace_id, path, content, session_id=None):
    """Refresh-on-resight. Caller MUST hold state.vfs_lock. Returns
    'new'|'updated'|'same'. First sight: append. Resight: update value iff
    content differs, but DO NOT move_to_end (position is first-seen-stable,
    spec D4). Inner cap 64 + outer cap 128 via popitem(last=False). Default-
    workspace keying: if workspace_id == DEFAULT_WORKSPACE, outer key is
    (workspace_id, session_id) tuple; else outer key is workspace_id.
    Each inner value MUST be constructed as OrderedDict() (NOT {} —
    popitem(last=False) raises TypeError on a plain dict)."""
    if workspace_id == DEFAULT_WORKSPACE:
        key = (workspace_id, session_id)
    else:
        key = workspace_id

    if key not in instruction_files_by_workspace:
        if len(instruction_files_by_workspace) >= INSTRUCTION_FILES_MAX_WORKSPACES:
            instruction_files_by_workspace.popitem(last=False)
        instruction_files_by_workspace[key] = OrderedDict()

    inner = instruction_files_by_workspace[key]

    if path in inner:
        if inner[path] == content:
            return "same"
        inner[path] = content
        return "updated"

    if len(inner) >= INSTRUCTION_FILES_MAX_PER_WORKSPACE:
        inner.popitem(last=False)
    # DEBUG/TEST-ONLY fault-injection hook (off by default). Mirrors
    # PROXY_DEBUG_LOCK_ORDERING: a self-DoS knob if env config is ever exposed
    # to untrusted operators. Never enable in production.
    if os.environ.get("PROXY_TIER2_TEST_RACE"):
        import time; time.sleep(0.02)
    inner[path] = content
    return "new"


def get_instruction_files(workspace_id, session_id=None) -> dict:
    """Return a SHALLOW COPY of {path: content} for the workspace. Acquires
    state.vfs_lock internally (pin-emit caller holds none). Same default-
    workspace keying. Missing workspace -> {}."""
    import proxy.state as state
    if workspace_id == DEFAULT_WORKSPACE:
        key = (workspace_id, session_id)
    else:
        key = workspace_id
    with state.vfs_lock:
        inner = state.instruction_files_by_workspace.get(key)
        if inner is None:
            return {}
        return dict(inner)

# v5: tunable context budget ratio (was hardcoded 0.80 everywhere).
# Controls: safe context tokens, recent message budget, memory confidence
# drop threshold, and tool loop usable context ratio.
CONTEXT_BUDGET_RATIO = float(os.environ.get("PROXY_CONTEXT_BUDGET_RATIO", "0.75"))
if not (0.1 <= CONTEXT_BUDGET_RATIO <= 0.95):
    logger.warning("PROXY_CONTEXT_BUDGET_RATIO=%s out of range [0.1, 0.95], clamping", CONTEXT_BUDGET_RATIO)
    CONTEXT_BUDGET_RATIO = max(0.1, min(0.95, CONTEXT_BUDGET_RATIO))

# VFS counters
vfs_cache_hits = 0
vfs_cache_misses = 0
vfs_write_conflicts = 0

# =============================================================================
# Chat history state (extracted to chat_history.py, re-exported for compatibility)
# =============================================================================
from proxy.chat_history import (
    chat_histories,
    chat_histories_lock,
    ChatHistoryCache,
    CHAT_HISTORIES_MAX_ENTRIES,
    _chat_histories_lru,
    _load_from_sqlite,
)
from proxy.eviction_coordinator import build_and_process_deferred_clears

# =============================================================================
# V2 chain-hash state (per session_id + model_name)
# =============================================================================
# OrderedDict keyed on (session_id, model_name) tuples. Value is a dict from
# new_chain_state_entry(). LRU semantics: any access (read OR write) bumps the
# entry to MRU; insertion past CHAIN_STATE_MAX_ENTRIES evicts the LRU entry.
#
# Canonical direct-access pattern (used by context_v2.py). Callers MUST hold
# _chain_state_lock around read+move_to_end (snapshot to locals), release it
# for I/O, then re-acquire for the CAS check + write + eviction:
#
#     with _chain_state_lock:
#         entry = _chain_state.get(key)
#         if entry is not None:
#             _chain_state.move_to_end(key)
#         # ... copy fields to locals ...
#     # lock released for I/O / cold-path computation
#     with _chain_state_lock:
#         current = _chain_state.get(key)
#         if current is None or current["version"] != snap_version:
#             ... CAS failure ...
#         current[...] = updated fields
#         current["version"] += 1
#         _chain_state.move_to_end(key)
#         while len(_chain_state) > CHAIN_STATE_MAX_ENTRIES:
#             _chain_state.popitem(last=False)
_chain_state: OrderedDict = OrderedDict()
try:
    CHAIN_STATE_MAX_ENTRIES = int(os.environ.get("PROXY_V2_CHAIN_STATE_MAX_ENTRIES", "100"))
    if CHAIN_STATE_MAX_ENTRIES < 1:
        logger.warning(
            "PROXY_V2_CHAIN_STATE_MAX_ENTRIES=%d < 1, clamping to 1",
            CHAIN_STATE_MAX_ENTRIES,
        )
        CHAIN_STATE_MAX_ENTRIES = 1
except (ValueError, TypeError):
    logger.warning(
        "Invalid PROXY_V2_CHAIN_STATE_MAX_ENTRIES=%r; falling back to default 100",
        os.environ.get("PROXY_V2_CHAIN_STATE_MAX_ENTRIES"),
    )
    CHAIN_STATE_MAX_ENTRIES = 100


def new_chain_state_entry() -> dict:
    """Return a fresh chain-state dict with the canonical key set and defaults."""
    return {
        "last_chain": None,
        "boundary_idx": 0,
        "boundary_chain": None,
        "summaries": [],
        "ancient": "",
        "version": 0,
    }


def invalidate_chain_state_for_session(session_id):
    """Drop all chain-state entries for a session (multi-model safe).

    Lock-order: position #10, acquired holding nothing.
    Next request for the session takes the cold path → fresh DB read.
    """
    with _chain_state_lock:
        for key in [k for k in _chain_state if k[0] == session_id]:
            del _chain_state[key]

# =============================================================================
# Workspace state (session-scoped workspace management)
# =============================================================================
active_workspace_lock = make_lock(6, "active_workspace_lock", threading.RLock)  # Position #6 — DO NOT MOVE or rename

from memory.workspace_persistence import (
    create_persist_callback,
    create_recovery_fn,
    cleanup_neo4j_sessions,
)

_persist_cb = create_persist_callback()  # None if Neo4j unavailable

from proxy.workspace_state import init_workspace_store, DEFAULT_WORKSPACE, WorkspaceStateStore
workspace_store = init_workspace_store(active_workspace_lock, persist_callback=_persist_cb)
workspace_store._recovery_fn = create_recovery_fn(workspace_store)  # post-construction wiring

# Backward-compatible aliases for tests
active_workspaces = workspace_store._active
recent_workspaces = workspace_store._recent
session_last_access = workspace_store._last_access

_last_cleanup_time: float = 0.0  # Track last cleanup for rate limiting
_cleanup_lock = threading.Lock()  # Guards _last_cleanup_time check-and-set


def make_workspace_id(workspace: str) -> str:
    """Return the canonical workspace_id (bare name in the v5 global model).

    The session_id is no longer embedded — workspaces are global, not per-session.
    Validates the name to prevent Cypher injection and ensure uniqueness.

    Backward compat: callers may still pass (session_id, workspace) — the
    second arg is ignored. This lets existing call sites pass without
    modification while we transition. New callers should pass only workspace.
    """
    # Backward compat: ignore session_id if passed positionally
    if not workspace or not workspace.strip():
        raise ValueError("workspace must not be empty")
    if len(workspace) > 64:
        raise ValueError("workspace name too long")
    if ':' in workspace or not workspace.replace('-', '').replace('_', '').isalnum():
        raise ValueError(
            f"workspace must contain only alphanumeric, '-', '_' characters: {workspace!r}"
        )
    return workspace


def cleanup_stale_workspaces(session_ttl: int = 28800) -> int:
    """Remove workspace state for sessions older than session_ttl seconds.
    MEM-001: Prevents unbounded growth of active_workspaces and recent_workspaces.
    Returns number of sessions cleaned up.
    """
    global _last_cleanup_time
    now = time.time()
    
    # Guard: only run cleanup once per 60 seconds (thread-safe)
    with _cleanup_lock:
        if now - _last_cleanup_time < 60:
            return 0
        _last_cleanup_time = now
    
    # Step 1: Get stale session IDs from workspace_store
    stale_session_ids = workspace_store.get_stale_sessions(session_ttl)

    if not stale_session_ids:
        return 0

    # Step 2: Acquire AgentRegistry._lock (#0) — coarsest — to check in-flight
    # status and retrieve agents. agent._mode_lock (#7) is fine inside #0.
    stale_with_modes = []
    try:
        from proxy.agent.agent_registry import registry as _agent_registry
        with _agent_registry._lock:
            for session_id in stale_session_ids:
                if _agent_registry._in_flight_requests.get(session_id, 0) > 0:
                    continue
                agent = _agent_registry._agents.get(session_id)
                mode_snap = None
                if agent is not None:
                    try:
                        mode_snap = agent.snapshot_mode()
                    except (ImportError, AttributeError):
                        pass
                stale_with_modes.append((session_id, mode_snap))
    except (ImportError, AttributeError):
        stale_with_modes = [(sid, None) for sid in stale_session_ids]

    if not stale_with_modes:
        return 0

    # Step 3: Pop workspace state for stale sessions
    # F1: Track re-activated sessions so step 4 can skip them too.
    reactivated_ids: set[str] = set()
    popped_ids: list[str] = []
    for session_id, _ in stale_with_modes:
        # F7: Re-check staleness - session may have been re-activated since step 2
        last = workspace_store.get_last_access(session_id)
        if last is not None and now - last < session_ttl:
            reactivated_ids.add(session_id)
            continue
        workspace_store.pop_session(session_id)
        popped_ids.append(session_id)
    
    # --- AgentRegistry._lock (#0) and active_workspace_lock (#6) released ---

    # Step 3b: Neo4j cleanup — OUTSIDE all locks
    # v5: cleanup_neo4j_sessions deletes :Session + edges, NEVER :Workspace
    if popped_ids:
        try:
            cleanup_neo4j_sessions(popped_ids)
        except Exception:
            pass  # Already logged internally; non-fatal
    
    # Step 4: Clear active modes outside all locks
    try:
        from proxy.agent.agent_registry import registry as _agent_registry
        for sid, mode_snap in stale_with_modes:
            # F1: Skip sessions that were re-activated between step 2 and step 3
            if sid in reactivated_ids:
                continue
            if mode_snap is not None:
                agent = _agent_registry.get_or_load(sid)
                if agent is not None:
                    try:
                        agent.clear_active_mode(expected_mode=mode_snap)
                    except Exception:
                        pass
    except ImportError:
        pass
    logger.info("Cleaned up %d stale workspaces", len(stale_with_modes))
    return len(stale_with_modes)

param_modifications = 0
param_drops = 0
param_clamps = 0
param_injects = 0
param_mod_details = {}

def _cleanup_param_log_timestamps() -> None:
    """Remove entries older than 1 hour from param_mod_log_timestamps."""
    cutoff = time.time() - 3600
    with param_mod_log_lock:
        stale = [k for k, v in param_mod_log_timestamps.items() if v < cutoff]
        for k in stale:
            dict.__delitem__(param_mod_log_timestamps, k)


neo4j_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout_seconds=30.0,
    name="neo4j",
)

# =============================================================================
# HTTP session (single instance shared across all requests)
# Auth headers are passed per-request via headers= param, NOT by mutating this.
# =============================================================================
_HTTP_SESSION = requests.Session()


@dataclass
class AppState:
    """Convenience wrapper. All shared mutable state lives in state.py module-level objects.

    AppState holds ONLY immutable configuration fields set once at startup.
    All mutable state (locks, dicts, counters) is accessed directly via
    state.py module-level objects:

        from proxy.state import chat_histories          # dict (mutable, shared)
        import proxy.state as state; state.tool_cache_size += 1  # int counter

    AppState does NOT re-create or wrap these objects. This prevents the
    split-state bug where AppState creates NEW dicts/locks that diverge
    from the module-level ones.
    """
    llama_url: str = "http://localhost:7777/v1/chat/completions"
    max_response_tokens: int = 1200
    cleanup_interval: int = 100
    session_ttl: int = 28800  # v5: bumped from 3600 (8h)
    max_tool_cache_size: int = 100
    max_model_cache_size: int = 100
    max_message_content_size: int = 100 * 1024
    max_total_content_size: int = 500 * 1024
    summary_max_tokens: int = 2000
    enable_token_calculation: bool = True
    test_mode: bool = False

# Self-reference for `from proxy.state import state` and `state.attr` access.
state = sys.modules[__name__]

# ── Canary verification cache (§5.4) — populated after class defs ─────────
canary_verify_cache = OrderedDict()

# Mirror of proxy.app:_PROXY_TRACE — read env var at module load.
# Hoisted here so provider_compat can import it without circular dep on proxy.app.
PROXY_TRACE = os.environ.get("PROXY_TRACE", "0") != "0"
