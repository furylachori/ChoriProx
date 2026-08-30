"""Tool call hard limits for a single user request.

Phase 5.5: Per-request limits that reset on each new user message.
Kept separate from ToolCallTracker to allow independent use.

Limits:
   -   proxy_ticket max calls (default 50)
   - file reads (max 30, per file once)
   - sub-agents (max 3)

Also tracks:
   - file blocklist (.env*, *.pem, *credential*, *secret*, *token*)
   - hash provenance: short_hashes returned by list_tasks (for MODIFY validation)
   - SemanticRetryTracker: bounded retry enforcement (spec §15)
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PROXY_TICKET_CALLS = 50
MAX_PROXY_WORK_CALLS = 50
MAX_FILE_READS = 30
MAX_SUBAGENTS = 3
MAX_REPORT_BUG_CALLS = 5

# File patterns that cannot be read by ticket-mode agents
FILE_BLOCKLIST_PATTERNS = (
    ".env",
    ".pem",
    "credential",
    "secret",
    "token",
    ".git/",
    ".password",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FileAccessRecord:
    """Tracks one file read: path → read count (should be 1, but allow retries)."""
    path: str
    normalized_path: str
    read_count: int = 0



# ---------------------------------------------------------------------------
# ToolLimits
# ---------------------------------------------------------------------------

class ToolLimits:
    """Per-request hard limits for tool calls.
    
    All counters are per-instance and reset on new_request().
    Thread-safe for concurrent access within a single request.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._overrides: dict[str, int] = {}
        self.reset()

    def set_limit(self, field: str, n: int) -> None:
        """Override the module-level MAX_LIMIT for one field (debug API).

        `field` is one of: "proxy_ticket", "proxy_work", "report_bug",
        "file_reads", "subagents". An empty `self._overrides` dict preserves
        the original module-constant behavior, so this method is backwards
        compatible — no production caller is required to change.
        """
        with self._lock:
            self._overrides[field] = n

    def reset(self) -> None:
        """Reset all counters for a new user request."""
        with self._lock:
            self._proxy_ticket_count = 0
            self._proxy_ticket_counted = set()  # call ids already counted
            self._proxy_work_count = 0
            self._proxy_work_counted: set[str] = set()  # call ids already counted
            self._file_reads: dict[str, FileAccessRecord] = {}
            self._subagent_count = 0
            self._seen_hashes: set[str] = set()  # short_hashes from list_tasks
            self._report_bug_count = 0
            self._report_bug_counted: set[str] = set()  # call ids already counted

    # -------------------------------------------------------------------------
    # proxy_ticket counting
    # -------------------------------------------------------------------------

    def count_proxy_ticket(self, call_id: str) -> int:
        """Increment and return current proxy_ticket call count for this request.
        
        Returns the new count.
        """
        with self._lock:
            if call_id in self._proxy_ticket_counted:
                return self._proxy_ticket_count  # already counted this call
            self._proxy_ticket_counted.add(call_id)
            self._proxy_ticket_count += 1
            return self._proxy_ticket_count

    @property
    def proxy_ticket_count(self) -> int:
        with self._lock:
            return self._proxy_ticket_count

    def try_count_proxy_ticket(self, call_id: str, max_limit: Optional[int] = None) -> bool:
        """Atomically check limit AND count. Returns True if allowed, False if at limit."""
        with self._lock:
            limit = max_limit if max_limit is not None else self._overrides.get("proxy_ticket", MAX_PROXY_TICKET_CALLS)
            if call_id in self._proxy_ticket_counted:
                return True  # already counted
            if self._proxy_ticket_count >= limit:
                return False
            self._proxy_ticket_counted.add(call_id)
            self._proxy_ticket_count += 1
            return True

    @property
    def can_proxy_ticket(self) -> bool:
        with self._lock:
            return self._proxy_ticket_count < self._overrides.get("proxy_ticket", MAX_PROXY_TICKET_CALLS)

    # -------------------------------------------------------------------------
    # proxy_work counting
    # -------------------------------------------------------------------------

    def count_proxy_work(self, call_id: str) -> int:
        """Increment and return current proxy_work call count for this request.
        
        Returns the new count.
        """
        with self._lock:
            if call_id in self._proxy_work_counted:
                return self._proxy_work_count  # already counted this call
            self._proxy_work_counted.add(call_id)
            self._proxy_work_count += 1
            return self._proxy_work_count

    @property
    def proxy_work_count(self) -> int:
        with self._lock:
            return self._proxy_work_count

    def try_count_proxy_work(self, call_id: str, max_limit: Optional[int] = None) -> bool:
        """Atomically check limit AND count. Returns True if allowed, False if at limit."""
        with self._lock:
            limit = max_limit if max_limit is not None else self._overrides.get("proxy_work", MAX_PROXY_WORK_CALLS)
            if call_id in self._proxy_work_counted:
                return True  # already counted
            if self._proxy_work_count >= limit:
                return False
            self._proxy_work_counted.add(call_id)
            self._proxy_work_count += 1
            return True

    @property
    def can_proxy_work(self) -> bool:
        with self._lock:
            return self._proxy_work_count < self._overrides.get("proxy_work", MAX_PROXY_WORK_CALLS)

    # -------------------------------------------------------------------------
    # report_bug counting
    # -------------------------------------------------------------------------

    def try_count_report_bug(self, call_id: str) -> bool:
        """Atomically check limit AND count. Returns True if allowed, False if at limit."""
        with self._lock:
            limit = self._overrides.get("report_bug", MAX_REPORT_BUG_CALLS)
            if call_id in self._report_bug_counted:
                return True  # already counted
            if self._report_bug_count >= limit:
                return False
            self._report_bug_counted.add(call_id)
            self._report_bug_count += 1
            return True

    @property
    def report_bug_count(self) -> int:
        with self._lock:
            return self._report_bug_count

    # -------------------------------------------------------------------------
    # Hash provenance (MODIFY validation)
    # -------------------------------------------------------------------------

    def add_seen_hashes(self, hashes: list[str]) -> None:
        """Record short_hashes returned by a list_tasks call."""
        with self._lock:
            self._seen_hashes.update(h.lower() for h in hashes if h)

    def has_seen_hash(self, short_hash: str) -> bool:
        """Check if a short_hash was returned by a prior list_tasks call."""
        with self._lock:
            return short_hash.lower() in self._seen_hashes

    # -------------------------------------------------------------------------
    # File read tracking + blocklist
    # -------------------------------------------------------------------------

    def can_read_file(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Check if a file can be read (blocklist + read-once enforcement).
        
        Returns (True, None) if allowed.
        Returns (False, reason) if blocked.
        """
        with self._lock:
            normalized = os.path.normpath(os.path.realpath(file_path)).lower().replace("\\", "/")
            
            # Check blocklist
            for pattern in FILE_BLOCKLIST_PATTERNS:
                if pattern in normalized or normalized.endswith(pattern):
                    return False, f"File access blocked: '{file_path}' matches pattern '{pattern}'"
            
            # Check read-once
            record = self._file_reads.get(normalized)
            if record and record.read_count > 0:
                return False, f"File already read this request: '{file_path}'"
            
            return True, None

    def record_file_read(self, file_path: str) -> None:
        """Record that a file was successfully read."""
        with self._lock:
            normalized = os.path.normpath(os.path.realpath(file_path)).lower().replace("\\", "/")
            record = self._file_reads.get(normalized)
            if record:
                record.read_count += 1
            else:
                self._file_reads[normalized] = FileAccessRecord(
                    path=file_path,
                    normalized_path=normalized,
                    read_count=1,
                )
            if self._file_reads.get(normalized) and len(self._file_reads) >= MAX_FILE_READS:
                logger.warning(
                    "File read limit reached: %d unique files", len(self._file_reads)
                )

    def try_read_file(self, file_path: str, from_cache: bool = False) -> tuple[bool, Optional[str]]:
        """Atomically check blocklist + read-once AND record the read. Returns (True, None) if allowed, (False, reason) if blocked.

        M3-2: when `from_cache=True`, the read-once check is BYPASSED
        because a cache hit means the read already happened in a prior
        request and was already counted at that time. The blocklist
        still applies (a path is either blocked or not, regardless of
        cache). The read count is NOT incremented for cache hits —
        they don't count against the per-request read-once limit.
        """
        with self._lock:
            normalized = os.path.normpath(os.path.realpath(file_path)).lower().replace("\\", "/")

            # Check blocklist (always applies)
            for pattern in FILE_BLOCKLIST_PATTERNS:
                if pattern in normalized or normalized.endswith(pattern):
                    return False, f"File access blocked: '{file_path}' matches pattern '{pattern}'"

            # Cache hit: skip read-once check + record (already counted when first read).
            if from_cache:
                return True, None

            # Check read-once
            record = self._file_reads.get(normalized)
            if record and record.read_count > 0:
                return False, f"File already read this request: '{file_path}'"

            # Record the read
            if record:
                record.read_count += 1
            else:
                self._file_reads[normalized] = FileAccessRecord(
                    path=file_path,
                    normalized_path=normalized,
                    read_count=1,
                )
            return True, None

    # -------------------------------------------------------------------------
    # Sub-agent tracking
    # -------------------------------------------------------------------------

    @property
    def can_spawn_subagent(self) -> bool:
        with self._lock:
            return self._subagent_count < self._overrides.get("subagents", MAX_SUBAGENTS)

    def spawn_subagent(self, session_id: str, tool_call_id: str) -> bool:
        """Record a sub-agent spawn. Returns True if allowed, False if at limit."""
        with self._lock:
            if self._subagent_count >= self._overrides.get("subagents", MAX_SUBAGENTS):
                return False
            self._subagent_count += 1
            return True

    def try_spawn_subagent(self, session_id: str, tool_call_id: str, max_limit: Optional[int] = None) -> bool:
        """Atomically check limit AND record spawn. Returns True if allowed, False if at limit."""
        with self._lock:
            limit = max_limit if max_limit is not None else self._overrides.get("subagents", MAX_SUBAGENTS)
            if self._subagent_count >= limit:
                return False
            self._subagent_count += 1
            return True

    @property
    def subagent_count(self) -> int:
        with self._lock:
            return self._subagent_count

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def status_summary(self) -> dict:
        with self._lock:
            return {
                "proxy_ticket_calls": self._proxy_ticket_count,
                "proxy_ticket_max": self._overrides.get("proxy_ticket", MAX_PROXY_TICKET_CALLS),
                "proxy_work_calls": self._proxy_work_count,
                "proxy_work_max": self._overrides.get("proxy_work", MAX_PROXY_WORK_CALLS),
                "report_bug_calls": self._report_bug_count,
                "report_bug_max": self._overrides.get("report_bug", MAX_REPORT_BUG_CALLS),
                "file_reads": len(self._file_reads),
                "file_read_max": MAX_FILE_READS,
                "subagents_spawned": self._subagent_count,
                "subagent_max": self._overrides.get("subagents", MAX_SUBAGENTS),
                "seen_hashes_count": len(self._seen_hashes),
                "_overrides": dict(self._overrides),
            }


# ---------------------------------------------------------------------------
# SemanticRetryTracker (spec §15 — bounded semantic retry)
# ---------------------------------------------------------------------------

@dataclass
class _RetryKeyState:
    category: str
    code: Optional[str] = None
    field: Optional[str] = None
    total_attempts: int = 0
    consecutive_identical: int = 0
    last_args_signature: Optional[str] = None


class SemanticRetryTracker:
    """Per-invocation semantic retry enforcement. No lock needed —
    tool execution is sequential within a single run_memory_tool_loop call."""

    # Only "validation" and "infrastructure" budgets are consulted: check()
    # short-circuits "authorization" and "internal" to no-retry BEFORE any
    # budget lookup, so those categories never read this dict.
    BUDGETS = {"validation": 3, "infrastructure": 2}
    CONSECUTIVE_IDENTICAL_LIMIT = 2

    _PROXY_INJECTED_ARG_KEYS = ("session_id", "client_hint")

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], _RetryKeyState] = {}

    def check(
        self,
        tool_name: str,
        operation: str,
        tool_args: dict,
    ) -> Optional[tuple[str, str]]:
        key = (tool_name, operation)
        state = self._states.get(key)
        if state is None:
            return None
        if state.category in ("authorization", "internal"):
            return (f"{state.category}_no_retry", state.category)
        signature = self._args_signature(tool_args, state.field)
        if signature != state.last_args_signature:
            return None
        if state.category == "validation":
            if (
                state.consecutive_identical >= self.CONSECUTIVE_IDENTICAL_LIMIT
                or state.total_attempts >= self.BUDGETS["validation"]
            ):
                return ("validation_retry_exhausted", "validation")
        elif state.category == "infrastructure":
            if state.total_attempts >= self.BUDGETS["infrastructure"]:
                return ("infrastructure_retry_exhausted", "infrastructure")
        return None

    def record(
        self,
        tool_name: str,
        operation: str,
        result: Any,
        tool_args: dict,
    ) -> None:
        key = (tool_name, operation)
        status = result.get("status") if isinstance(result, dict) else None
        err_obj = result.get("error") if isinstance(result, dict) else None

        if status == "success":
            self._states.pop(key, None)
            return
        if status == "partial" and not isinstance(err_obj, dict):
            self._states.pop(key, None)
            return
        if status != "error" and not (
            status == "partial" and isinstance(err_obj, dict)
        ):
            return

        category = "internal"
        code: Optional[str] = None
        field: Optional[str] = None
        if isinstance(err_obj, dict):
            category = err_obj.get("category") or "internal"
            code = err_obj.get("code")
            field = err_obj.get("field")

        signature = self._args_signature(tool_args, field)
        prior = self._states.get(key)
        if prior is None:
            self._states[key] = _RetryKeyState(
                category=category,
                code=code,
                field=field,
                total_attempts=1,
                consecutive_identical=1,
                last_args_signature=signature,
            )
            return

        prior.total_attempts += 1
        same_triplet = (
            prior.category == category
            and prior.code == code
            and prior.field == field
        )
        if same_triplet and signature == prior.last_args_signature:
            prior.consecutive_identical += 1
        else:
            prior.consecutive_identical = 1
        prior.category = category
        prior.code = code
        prior.field = field
        prior.last_args_signature = signature

    def _args_signature(self, tool_args: Any, field: Optional[str]) -> str:
        if not isinstance(tool_args, dict):
            cleaned: dict = {}
        else:
            cleaned = {
                k: v
                for k, v in tool_args.items()
                if k not in self._PROXY_INJECTED_ARG_KEYS
            }
        if field is not None and field in cleaned:
            payload = {"field": field, "value": cleaned[field]}
        else:
            payload = cleaned
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _derive_operation(tool_name: str, tool_args: Any) -> str:
        if isinstance(tool_args, dict):
            action = tool_args.get("action")
            if isinstance(action, str) and action:
                return action
        return tool_name