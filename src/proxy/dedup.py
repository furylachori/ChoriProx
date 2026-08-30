"""Transparent request deduplication — canonical JSON fingerprint + keyed store (CP4-1).

The fingerprint is derived from what the client already sends (validated body +
session_id + client_hint + model + workspace_name + is_subagent + mode). No
client change, no new header, no plugin. The HMAC secret fallback chain is:
PROXY_IDEMPOTENCY_SECRET → PROXY_SESSION_SECRET → process-local random.

The DedupStore is an OrderedDict keyed by fingerprint, protected by
the lock passed at construction (state.dedup_lock, position #14, in
production). LRU eviction via ``move_to_end`` + ``popitem(last=False)``.
TTL starts at COMPLETED. PENDING entries have a max-age cap (default 600s)
to prevent unbounded growth from crashed owners. FAILED entries are evicted
immediately by the next lookup_or_insert call.

CP4-2: the store is wired into app.py and the lookup result is ACTED ON
for streaming requests (in-flight dedup: same fingerprint + PENDING owner →
FOLLOWER attaches to the same CompletionRecord). Non-streaming requests
compute the fingerprint but do not act on it (CP4-4).

CP4-3: COMPLETED REPLAY. When lookup_or_insert finds a COMPLETED entry within
TTL whose artifact passes the safety gate (is_safe_for_replay: logical_status
"ok", no tool calls), it returns (entry, AdmissionKind.REPLAY) WITHOUT evicting.
The caller serves the cached artifact directly — no upstream, no pipeline, no
owner_setup. Unsafe COMPLETED entries (tool calls, error status, missing
artifact) are evicted and a new PENDING owner is created.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from proxy.completion_coordinator import AdmissionKind, CompletionArtifact, CompletionRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret resolution (read once at module load, consistent with session_tokens.py)
# ---------------------------------------------------------------------------

_IDEMPOTENCY_SECRET: bytes
_env_secret = os.environ.get("PROXY_IDEMPOTENCY_SECRET", "")
if _env_secret:
    _IDEMPOTENCY_SECRET = _env_secret.encode("utf-8")
else:
    _session_secret = os.environ.get("PROXY_SESSION_SECRET", "")
    if _session_secret:
        _IDEMPOTENCY_SECRET = _session_secret.encode("utf-8")
    else:
        # Process-local fallback: stable within this process, not across restarts.
        _IDEMPOTENCY_SECRET = os.urandom(32)


# ---------------------------------------------------------------------------
# Canonical JSON + fingerprint
# ---------------------------------------------------------------------------


def canonical_json(body: dict) -> bytes:
    """Deterministic JSON serialization for fingerprinting.

    Sorted keys, compact separators, UTF-8, no NaN/Infinity.
    Duplicate JSON keys are already collapsed by Flask's json.loads —
    no object_pairs_hook needed here (that's a json.loads parameter).
    """
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_fingerprint(ctx, mode_name: str = "") -> str:
    """HMAC-SHA256 fingerprint of the validated request, scoped by session.

    Includes: session_id, client_hint, model, workspace_name, is_subagent,
    mode (CP4-3: prevents mode collisions — same body in different modes
    produces different responses), and the full validated body. Truncated
    to 32 hex chars.

    The body is ctx.body (immutable after parse_request), NOT ctx.llama_body
    (which is mutated by ProviderCompatStage).
    """
    body = dict(getattr(ctx, "body", {}))
    body.pop("stream", None)  # stream is captured by is_stream, not body
    payload = {
        "session_id": getattr(ctx, "session_id", ""),
        "client_hint": getattr(ctx, "client_hint", ""),
        "model": getattr(ctx, "model_name", ""),
        "workspace_name": getattr(ctx, "workspace_name", ""),
        "is_subagent": getattr(ctx, "is_subagent", False),
        "mode": mode_name,
        "body": body,
    }
    raw = canonical_json(payload)
    digest = hmac.new(_IDEMPOTENCY_SECRET, raw, hashlib.sha256).hexdigest()
    return digest[:32]


# ---------------------------------------------------------------------------
# DedupEntry + DedupState
# ---------------------------------------------------------------------------


class DedupState(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class FollowerCapExceeded(Exception):
    """Raised when a PENDING key has reached max_followers_per_key."""


@dataclass
class DedupEntry:
    """A single keyed entry in the DedupStore."""

    fingerprint: str
    record: Optional[CompletionRecord] = None
    state: DedupState = DedupState.PENDING
    artifact: Optional[CompletionArtifact] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    follower_count: int = 0
    stored_bytes: int = 0
    is_stream: bool = False


def is_safe_for_replay(entry: DedupEntry) -> bool:
    """Pure, total, fail-closed safety gate for replay (CP4-3).

    Returns True only if the entry is COMPLETED with a well-formed,
    tool-call-free, ok-status artifact. Any uncertainty → False.
    """
    try:
        if entry.state is not DedupState.COMPLETED:
            return False
        art = entry.artifact
        if art is None:
            return False
        if getattr(art, "logical_status", None) != "ok":
            return False
        if getattr(art, "has_tool_calls", False):
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DedupStore
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        val = int(os.environ.get(name, str(default)))
        return max(val, minimum)
    except (ValueError, TypeError):
        return default


class DedupStore:
    """OrderedDict-backed keyed completion store with LRU + TTL eviction.

    Protected by the lock passed at construction (state.dedup_lock, position
    #14, in production). All public methods acquire the lock internally.

    Caps (env-tunable):
      - max_completed: max COMPLETED entries (default 256).
      - max_bytes: total stored bytes across COMPLETED entries (default 16MB).
      - max_followers_per_key: max followers per PENDING key (default 16).
      - completed_ttl_seconds: TTL for COMPLETED entries (default 120s).
      - pending_max_age_seconds: max age for PENDING entries (default 600s).
    """

    def __init__(self, lock: Optional[threading.Lock] = None):
        self._store: OrderedDict[str, DedupEntry] = OrderedDict()
        self._lock = lock if lock is not None else threading.Lock()
        self.max_completed = _env_int("PROXY_DEDUP_MAX_COMPLETED", 256)
        self.max_bytes = _env_int("PROXY_DEDUP_MAX_BYTES", 16 * 1024 * 1024)
        self.max_followers_per_key = _env_int("PROXY_DEDUP_MAX_FOLLOWERS", 16)
        self.completed_ttl_seconds = _env_int("PROXY_DEDUP_COMPLETED_TTL", 120)
        self.pending_max_age_seconds = _env_int("PROXY_DEDUP_PENDING_MAX_AGE", 600)
        # Observability counters
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.conflicts = 0
        self.releases = 0

    def lookup(self, fingerprint: str) -> Optional[DedupEntry]:
        """Look up an entry by fingerprint. Evicts expired entries on access.

        Returns the entry if found and still valid, None otherwise.
        COMPLETED entries past TTL are evicted. PENDING entries past max_age
        are evicted (crashed owner).
        """
        with self._lock:
            return self._lookup_locked(fingerprint)

    def _lookup_locked(self, fingerprint: str) -> Optional[DedupEntry]:
        """Internal lookup — caller MUST hold self._lock.

        Evicts expired entries on access. Returns the entry if found and
        still valid, None otherwise.
        """
        entry = self._store.get(fingerprint)
        if entry is None:
            self.misses += 1
            return None

        now = time.time()

        # Evict expired COMPLETED
        if entry.state == DedupState.COMPLETED and entry.completed_at is not None:
            if now - entry.completed_at > self.completed_ttl_seconds:
                self._evict(fingerprint)
                self.misses += 1
                return None

        # Evict stale PENDING (crashed owner)
        if entry.state == DedupState.PENDING:
            if now - entry.created_at > self.pending_max_age_seconds:
                self._evict(fingerprint)
                self.misses += 1
                return None

        # LRU bump
        self._store.move_to_end(fingerprint)
        self.hits += 1
        return entry

    def lookup_or_insert(self, fingerprint: str, record_factory, is_stream: bool):
        """Atomic check-and-insert. Returns (entry, AdmissionKind).

        - No entry / evicted → insert PENDING owner, (entry, OWNER).
        - PENDING found, follower_count < cap → attach follower, (entry, FOLLOWER).
        - PENDING found, follower_count >= cap → raise FollowerCapExceeded.
        - COMPLETED found + safe for replay → (entry, REPLAY). Entry NOT evicted.
        - COMPLETED found + NOT safe → evict + new PENDING owner, (entry, OWNER).
        - FAILED found → evict + new PENDING owner, (entry, OWNER).
        """
        from proxy.completion_coordinator import AdmissionKind

        with self._lock:
            entry = self._lookup_locked(fingerprint)
            if entry is None:
                entry = DedupEntry(
                    fingerprint=fingerprint,
                    record=record_factory(),
                    state=DedupState.PENDING,
                    is_stream=is_stream,
                )
                self._store[fingerprint] = entry
                self._store.move_to_end(fingerprint)
                self._evict_lru_completed()
                return entry, AdmissionKind.OWNER
            if entry.state is DedupState.PENDING:
                if entry.is_stream != is_stream:
                    self.conflicts += 1
                    return entry, AdmissionKind.CONFLICT
                if entry.follower_count >= self.max_followers_per_key:
                    raise FollowerCapExceeded(fingerprint)
                entry.follower_count += 1
                return entry, AdmissionKind.FOLLOWER
            if entry.state is DedupState.COMPLETED and is_safe_for_replay(entry):
                # REPLAY: return the existing entry WITHOUT evicting.
                self._store.move_to_end(fingerprint)
                return entry, AdmissionKind.REPLAY
            # COMPLETED (unsafe) or FAILED: evict and create new PENDING owner.
            self._evict(fingerprint)
            entry = DedupEntry(
                fingerprint=fingerprint,
                record=record_factory(),
                state=DedupState.PENDING,
                is_stream=is_stream,
            )
            self._store[fingerprint] = entry
            self._store.move_to_end(fingerprint)
            self._evict_lru_completed()
            return entry, AdmissionKind.OWNER

    def mark_completed(self, fingerprint: str, artifact: CompletionArtifact) -> None:
        """Mark an entry COMPLETED, store the artifact, start TTL."""
        with self._lock:
            entry = self._store.get(fingerprint)
            if entry is None:
                return
            entry.state = DedupState.COMPLETED
            entry.artifact = artifact
            entry.completed_at = time.time()
            if artifact is not None:
                if getattr(artifact, "body", None) is not None:
                    import json as _json
                    entry.stored_bytes = len(_json.dumps(artifact.body, separators=(",", ":")).encode("utf-8"))
                elif hasattr(artifact, "terminal_chunks"):
                    entry.stored_bytes = sum(
                        len(c.encode("utf-8")) for c in artifact.terminal_chunks
                    )
            self._store.move_to_end(fingerprint)
            self._evict_lru_completed()

    def mark_failed(self, fingerprint: str) -> None:
        """Mark an entry FAILED. Evicted by the next lookup_or_insert call."""
        with self._lock:
            entry = self._store.get(fingerprint)
            if entry is None:
                return
            entry.state = DedupState.FAILED
            entry.artifact = None
            entry.stored_bytes = 0

    def release_follower(self, fingerprint: str) -> bool:
        """Release a follower slot previously allocated by lookup_or_insert.

        Idempotent: safe to call multiple times. Only decrements when the
        entry is still PENDING (the cap is PENDING-only). Returns True iff
        a decrement occurred. Best-effort; safe to call from any context.
        """
        with self._lock:
            entry = self._store.get(fingerprint)
            if entry is None:
                return False
            if entry.state is not DedupState.PENDING:
                return False
            if entry.follower_count <= 0:
                return False
            entry.follower_count -= 1
            self.releases += 1
            return True

    def clear(self) -> None:
        """Clear all entries (test isolation)."""
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.conflicts = 0
            self.releases = 0

    def stats(self) -> dict:
        """Snapshot for /health endpoint."""
        with self._lock:
            pending = sum(1 for e in self._store.values() if e.state == DedupState.PENDING)
            completed = sum(1 for e in self._store.values() if e.state == DedupState.COMPLETED)
            total_bytes = sum(e.stored_bytes for e in self._store.values())
            return {
                "pending_entries": pending,
                "completed_entries": completed,
                "total_stored_bytes": total_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "conflicts": self.conflicts,
                "releases": self.releases,
            }

    # -- internal (caller holds _lock) --

    def _evict(self, fingerprint: str) -> None:
        entry = self._store.pop(fingerprint, None)
        if entry is not None:
            self.evictions += 1

    def _evict_lru_completed(self) -> None:
        """Evict oldest COMPLETED entries until under caps. Never evict PENDING."""
        # Count cap
        completed_count = sum(
            1 for e in self._store.values() if e.state == DedupState.COMPLETED
        )
        while completed_count > self.max_completed:
            # Find oldest COMPLETED (first in OrderedDict order)
            for fp, entry in self._store.items():
                if entry.state == DedupState.COMPLETED:
                    self._evict(fp)
                    completed_count -= 1
                    break
            else:
                break

        # Byte cap
        total_bytes = sum(e.stored_bytes for e in self._store.values())
        while total_bytes > self.max_bytes:
            for fp, entry in self._store.items():
                if entry.state == DedupState.COMPLETED:
                    total_bytes -= entry.stored_bytes
                    self._evict(fp)
                    break
            else:
                break
