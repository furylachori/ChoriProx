"""ToolCallTracker — per-request call tracking with provenance and guard limits.

Phase 5.5: Integration point for tool_guard.py and execute_ticket_tool.

Tracks:
  - proxy_ticket call count
  - seen short_hashes (for MODIFY provenance validation)
  - file reads (blocklist + read-once)
  - sub-agent spawns

All state is per-request, reset on new user message.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from proxy.tool_pipeline.tool_limits import (
    ToolLimits,
    MAX_PROXY_TICKET_CALLS,
    MAX_PROXY_WORK_CALLS,
    MAX_FILE_READS,
    MAX_SUBAGENTS,
    FILE_BLOCKLIST_PATTERNS,
)

logger = logging.getLogger("proxy.tool_pipeline")


# ---------------------------------------------------------------------------
# ToolCallTracker
# ---------------------------------------------------------------------------

class ToolCallTracker:
    """Per-request tool call tracking with provenance and guard limits.

    Singleton-per-session: one instance per session_id.
    All state resets on new_request().

    Integration:
      - tool_guard.py: uses tracker for rate limit checks
      - execute_ticket_tool: uses tracker for proxy_ticket count + hash provenance
      - read action: uses tracker for file blocklist + read-once enforcement
    """

    # Per-session tracker instances
    _instances: dict[str, "ToolCallTracker"] = {}
    _instances_lock = threading.RLock()

    @classmethod
    def get_instance(cls, session_id: str) -> "ToolCallTracker":
        """Get or create a tracker for the given session."""
        with cls._instances_lock:
            if session_id not in cls._instances:
                cls._instances[session_id] = cls(session_id)
            inst = cls._instances[session_id]
            inst._last_access = time.monotonic()
            return inst

    @classmethod
    def reset_instance(cls, session_id: str) -> None:
        """Reset tracker for a new request from the same session."""
        with cls._instances_lock:
            if session_id in cls._instances:
                cls._instances[session_id].reset()
                cls.evict_stale()

    @classmethod
    def evict_stale(cls, max_age_seconds: int = 3600) -> int:
        """Remove tracker instances not accessed in max_age_seconds. Returns count removed."""
        with cls._instances_lock:
            now = time.monotonic()
            stale = [sid for sid, inst in cls._instances.items()
                     if now - inst._last_access > max_age_seconds]
            for sid in stale:
                del cls._instances[sid]
            return len(stale)

    def __init__(self, session_id: str):
        self._session_id = session_id
        self._lock = threading.Lock()
        self._limits = ToolLimits()
        self._last_access = time.monotonic()

    def reset(self) -> None:
        """Reset all state for a new user request."""
        with self._lock:
            self._limits.reset()

    # -------------------------------------------------------------------------
    # proxy_ticket call counting
    # -------------------------------------------------------------------------

    def count_proxy_ticket(self, call_id: str) -> int:
        """Count a proxy_ticket call. Returns current call count."""
        return self._limits.count_proxy_ticket(call_id)

    def try_count_proxy_ticket(self, call_id: str, max_limit: Optional[int] = None) -> bool:
        """Atomically check limit AND count. Returns True if allowed, False if at limit."""
        return self._limits.try_count_proxy_ticket(call_id, max_limit)

    def can_proxy_ticket(self) -> bool:
        """Check if another proxy_ticket call is allowed."""
        return self._limits.can_proxy_ticket

    @property
    def proxy_ticket_count(self) -> int:
        return self._limits.proxy_ticket_count

    # -------------------------------------------------------------------------
    # proxy_work call counting
    # -------------------------------------------------------------------------

    def count_proxy_work(self, call_id: str) -> int:
        """Count a proxy_work call. Returns current call count."""
        return self._limits.count_proxy_work(call_id)

    def try_count_proxy_work(self, call_id: str, max_limit: Optional[int] = None) -> bool:
        """Atomically check limit AND count. Returns True if allowed, False if at limit."""
        return self._limits.try_count_proxy_work(call_id, max_limit)

    def can_proxy_work(self) -> bool:
        """Check if another proxy_work call is allowed."""
        return self._limits.can_proxy_work

    @property
    def proxy_work_count(self) -> int:
        return self._limits.proxy_work_count

    # -------------------------------------------------------------------------
    # report_bug call counting
    # -------------------------------------------------------------------------

    def try_count_report_bug(self, call_id: str) -> bool:
        """Atomically check limit AND count. Returns True if allowed, False if at limit."""
        return self._limits.try_count_report_bug(call_id)

    @property
    def report_bug_count(self) -> int:
        return self._limits.report_bug_count

    # -------------------------------------------------------------------------
    # Hash provenance (MODIFY validation)
    # -------------------------------------------------------------------------

    def add_seen_hashes(self, hashes: list[str]) -> None:
        """Record short_hashes returned by list_tasks for MODIFY provenance."""
        self._limits.add_seen_hashes(hashes)

    def register_created_hashes(self, hashes: list[str]) -> None:
        """Register hashes from create_tasks response for provenance tracking."""
        self._limits.add_seen_hashes(hashes)

    def has_seen_hash(self, short_hash: str) -> bool:
        """Check if a short_hash was returned by a prior list_tasks call.
        
        Used by edit/update actions to reject edits on unseen hashes.
        """
        return self._limits.has_seen_hash(short_hash)

    # -------------------------------------------------------------------------
    # File access
    # -------------------------------------------------------------------------

    def can_read_file(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Check if a file can be read (blocklist + read-once).
        
        Returns (True, None) if allowed.
        Returns (False, reason) if blocked.
        """
        return self._limits.can_read_file(file_path)

    def try_read_file(self, file_path: str, from_cache: bool = False) -> tuple[bool, Optional[str]]:
        """Atomically check and record a file read.

        M3-2: when `from_cache=True`, the read-once check is BYPASSED
        because a cache hit means the read already happened in a prior
        request and was already counted at that time. The blocklist
        still applies (a path is either blocked or not, regardless of
        cache). The read count is NOT incremented for cache hits —
        they don't count against the per-request read-once limit.
        """
        return self._limits.try_read_file(file_path, from_cache=from_cache)

    def record_file_read(self, file_path: str) -> None:
        """Record that a file was successfully read."""
        self._limits.record_file_read(file_path)

    # -------------------------------------------------------------------------
    # Sub-agent tracking
    # -------------------------------------------------------------------------

    def can_spawn_subagent(self) -> bool:
        """Check if a sub-agent can be spawned."""
        return self._limits.can_spawn_subagent

    def spawn_subagent(self, session_id: str, tool_call_id: str) -> bool:
        """Record a sub-agent spawn."""
        return self._limits.spawn_subagent(session_id, tool_call_id)

    def try_spawn_subagent(self, session_id: str, tool_call_id: str, max_limit: Optional[int] = None) -> bool:
        """Atomically check and record sub-agent spawn."""
        return self._limits.try_spawn_subagent(session_id, tool_call_id, max_limit)

    @property
    def subagent_count(self) -> int:
        return self._limits.subagent_count

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def status_summary(self) -> dict:
        return self._limits.status_summary()