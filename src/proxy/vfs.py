"""Virtual Filesystem Layer — core data structures.

Provides workspace-scoped LRU caching for file contents, enabling
cross-agent file sharing and write-through cache updates.

All VFSCache methods MUST be called while holding state.vfs_lock.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("proxy.vfs")


@dataclass(frozen=True)
class VFSEntry:
    """Immutable snapshot of a file at a point in time."""
    path: str                    # Normalized absolute path
    content: str                 # File content (capped at VFS_MAX_ENTRY_BYTES)
    content_hash: str            # SHA-256 hex digest of content
    size_bytes: int              # len(content)
    cached_at: float             # time.time() when cached
    cached_by: str               # session_id of the agent that populated this entry
    workspace_id: str            # "session_id:workspace" scope key
    verified: bool = True        # False for write-through (unconfirmed), True after result confirms
    line_count: int = 0          # Precomputed at construction


@dataclass
class VFSWriteRecord:
    """Record of a write/edit operation for conflict detection."""
    path: str                    # Normalized absolute path
    agent_id: str                # session_id of the writer
    workspace_id: str            # Workspace scope
    timestamp: float             # time.time() of the write
    operation: str               # "write" | "edit"
    old_hash: Optional[str]      # Hash before write (None for new files)
    new_hash: str                # Hash after write


@dataclass
class VFSRollbackSnapshot:
    """Snapshot of pre-write cache state for rollback on failure.
    
    Saved before write-through updates the cache. If the write fails
    on the client side, the snapshot is used to restore the old content.
    """
    path: str                    # Normalized path
    workspace_id: str            # Workspace scope
    old_entry: Optional[VFSEntry]  # None = file didn't exist before write
    new_hash: str                # Hash of the new (write-through) content
    tool_call_id: str            # OpenAI tool_call_id for matching with result
    created_at: float            # time.time() for timeout detection
    parent_snapshot_id: Optional[str] = None  # Tracks edit chains (R4C-003)


class VFSCache:
    """Workspace-scoped LRU cache for file contents.

    Thread-safe via state.vfs_lock (position #9).
    All public methods MUST be called while holding vfs_lock.
    """

    def __init__(self, max_entries: int = 500, max_entry_bytes: int = 1_048_576):
        self._entries: OrderedDict[str, VFSEntry] = OrderedDict()
        self._max_entries = max_entries
        self._max_entry_bytes = max_entry_bytes
        self._total_bytes = 0
        self._readers: dict[str, set[str]] = {}  # path → set of session_ids

    def get(self, path: str) -> Optional[VFSEntry]:
        """Return cached entry or None. Moves to end (MRU) on hit."""
        entry = self._entries.get(path)
        if entry is not None:
            self._entries.move_to_end(path)
        return entry

    def put(self, entry: VFSEntry) -> None:
        """Insert or update entry. Evicts LRU if at capacity."""
        if entry.size_bytes > self._max_entry_bytes:
            return  # Skip oversized files silently
        existing = self._entries.get(entry.path)
        if existing:
            self._total_bytes -= existing.size_bytes
        self._entries[entry.path] = entry
        self._entries.move_to_end(entry.path)
        self._total_bytes += entry.size_bytes
        self._evict_if_needed()

    def invalidate(self, path: str) -> Optional[VFSEntry]:
        """Remove entry by path. Returns removed entry or None."""
        entry = self._entries.pop(path, None)
        if entry:
            self._total_bytes -= entry.size_bytes
        self._readers.pop(path, None)
        return entry

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries whose path starts with prefix. Returns count."""
        to_remove = [p for p in self._entries if p.startswith(prefix)]
        for p in to_remove:
            self._total_bytes -= self._entries[p].size_bytes
            del self._entries[p]
            self._readers.pop(p, None)
        return len(to_remove)

    def invalidate_all(self) -> int:
        """Remove all entries. Returns count."""
        count = len(self._entries)
        self._entries.clear()
        self._readers.clear()
        self._total_bytes = 0
        return count

    def _evict_if_needed(self) -> None:
        """Evict LRU entries until under max_entries."""
        while len(self._entries) > self._max_entries:
            _path, entry = self._entries.popitem(last=False)
            self._total_bytes -= entry.size_bytes
            self._readers.pop(_path, None)

    def snapshot_paths(self) -> list[tuple[str, str, float]]:
        """Return [(path, hash, cached_at)] for context injection.
        Caller MUST hold vfs_lock.
        """
        return [
            (e.path, e.content_hash, e.cached_at)
            for e in self._entries.values()
        ]

    def snapshot_entries(self) -> list[VFSEntry]:
        """Return list of VFSEntry objects for context injection.
        
        Returns full entries instead of tuples, avoiding direct
        _entries access from outside the class.
        Caller MUST hold vfs_lock.
        """
        return list(self._entries.values())

    def record_reader(self, path: str, session_id: str) -> None:
        """Record that a session read a file. Caller MUST hold vfs_lock."""
        if path not in self._readers:
            self._readers[path] = set()
        self._readers[path].add(session_id)

    def get_readers(self, path: str) -> set[str]:
        """Return set of session_ids that read a file. Caller MUST hold vfs_lock."""
        return self._readers.get(path, set()).copy()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


# =============================================================================
# Helper functions
# =============================================================================

def _is_expired(entry: VFSEntry, ttl_seconds: int) -> bool:
    """Check if a cache entry has expired."""
    import time
    return (time.time() - entry.cached_at) > ttl_seconds


def _vfs_active_for_context(is_subagent: bool) -> bool:
    """Check if VFS is active for the given agent type.
    
    Returns True if VFS should process tool calls for this agent.
    """
    import proxy.state as state
    if not state.VFS_ENABLED:
        return False
    if state.VFS_SCOPE == "subs" and not is_subagent:
        return False
    if state.VFS_SCOPE == "main" and is_subagent:
        return False
    return True


def _get_or_create_workspace_cache(workspace_id: str) -> VFSCache:
    """Get or create workspace cache. Caller MUST hold vfs_lock.
    
    Enforces VFS_MAX_WORKSPACES with LRU eviction.
    """
    import proxy.state as state
    
    if state.VFS_MAX_WORKSPACES <= 0:
        # VFS workspace caching disabled — still create cache but no eviction
        cache = state.vfs_caches.get(workspace_id)
        if cache is not None:
            return cache
        cache = VFSCache(
            max_entries=state.VFS_MAX_ENTRIES,
            max_entry_bytes=state.VFS_MAX_ENTRY_BYTES,
        )
        state.vfs_caches[workspace_id] = cache
        return cache
    
    cache = state.vfs_caches.get(workspace_id)
    if cache is not None:
        state.vfs_caches.move_to_end(workspace_id)
        return cache
    
    # Evict LRU workspace if at capacity
    while len(state.vfs_caches) >= state.VFS_MAX_WORKSPACES:
        evicted_id, evicted_cache = state.vfs_caches.popitem(last=False)
        logger.info("VFS: Evicted workspace %s (%d entries)", evicted_id, evicted_cache.size)
    
    cache = VFSCache(
        max_entries=state.VFS_MAX_ENTRIES,
        max_entry_bytes=state.VFS_MAX_ENTRY_BYTES,
    )
    state.vfs_caches[workspace_id] = cache
    return cache


def _cache_file(file_path: str, content: str, workspace_id: str, session_id: str) -> None:
    """Cache a file from read tool results. Called from _populate_vfs_cache()."""
    import proxy.state as state
    import hashlib
    import time
    
    normalized = os.path.normpath(os.path.realpath(file_path))
    
    with state.vfs_lock:
        cache = _get_or_create_workspace_cache(workspace_id)
        
        # Fast-path: skip if already cached with same content and not expired
        existing = cache.get(normalized)
        if existing and not _is_expired(existing, state.VFS_TTL_SECONDS):
            if existing.content == content:
                return  # Already cached, skip
    
    # Slow path: compute hash outside lock
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    entry = VFSEntry(
        path=normalized,
        content=content,
        content_hash=content_hash,
        size_bytes=len(content),
        cached_at=time.time(),
        cached_by=session_id,
        workspace_id=workspace_id,
        verified=True,  # Read results are verified (came from disk)
        line_count=content.count('\n') + 1,
    )
    
    with state.vfs_lock:
        cache = _get_or_create_workspace_cache(workspace_id)
        cache.put(entry)


def _cache_file_write_through(path: str, content: str, workspace_id: str,
                               session_id: str, tool_call_id: str = None) -> None:
    """Write-through: update cache with new file content from write tool.
    
    Saves a rollback snapshot before updating the cache.
    """
    import proxy.state as state
    import hashlib
    import time
    
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    with state.vfs_lock:
        cache = _get_or_create_workspace_cache(workspace_id)
        existing = cache.get(path)
        old_hash = existing.content_hash if existing else None
        
        # Save rollback snapshot BEFORE updating
        if tool_call_id:
            if len(state.vfs_rollback_snapshots) >= state.VFS_ROLLBACK_MAX:
                _rollback_oldest_snapshot()
            state.vfs_rollback_snapshots[tool_call_id] = VFSRollbackSnapshot(
                path=path,
                workspace_id=workspace_id,
                old_entry=existing,
                new_hash=content_hash,
                tool_call_id=tool_call_id,
                created_at=time.time(),
            )
        
        entry = VFSEntry(
            path=path,
            content=content,
            content_hash=content_hash,
            size_bytes=len(content),
            cached_at=time.time(),
            cached_by=session_id,
            workspace_id=workspace_id,
            verified=False,  # Unconfirmed until write result arrives
            line_count=content.count('\n') + 1,
        )
        cache.put(entry)

    try:
        _log_write(path, session_id, workspace_id, "write", old_hash, content_hash)
    except Exception as exc:
        logger.warning(
            "VFS: _log_write failed for %s in workspace %s: %s",
            path, workspace_id, exc,
            extra={"request_id": "?"},
        )


def _cache_file_edit_through(path: str, old_str: str, new_str: str,
                              workspace_id: str, session_id: str,
                              replace_all: bool = False,
                              tool_call_id: str = None) -> None:
    """Write-through: apply edit to cached content if available.
    
    Three-phase approach to minimize lock hold time:
    Phase 1: Extract old content and compute new content under lock
    Phase 2: Compute hash OUTSIDE lock
    Phase 3: Re-acquire lock for cache.put() with staleness check
    """
    import proxy.state as state
    import hashlib
    import time
    
    # Phase 1: Extract old content under lock
    old_hash = None
    old_entry = None
    parent_id = None
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_id)
        if cache is None:
            return  # Not cached — next read will populate
        
        entry = cache.get(path)
        if entry is None:
            return  # Not cached — next read will populate
        
        old_content = entry.content
        old_hash = entry.content_hash
        old_entry = entry
        if old_str not in old_content:
            return  # old_str not found — edit will fail on client side
        
        # Find parent snapshot for edit chain tracking
        if tool_call_id:
            for snap in reversed(list(state.vfs_rollback_snapshots.values())):
                if snap.path == path and snap.workspace_id == workspace_id:
                    parent_id = snap.tool_call_id
                    break
        
        if replace_all:
            new_content = old_content.replace(old_str, new_str)
        else:
            new_content = old_content.replace(old_str, new_str, 1)
    
    # Phase 2: Compute hash OUTSIDE lock
    content_hash = hashlib.sha256(new_content.encode()).hexdigest()
    
    # Phase 3: Re-acquire lock for cache.put() with staleness check
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_id)
        if cache is None:
            return
        
        # Staleness check: verify entry hasn't changed since Phase 1
        current = cache.get(path)
        if current and current.content_hash != old_hash:
            logger.warning("VFS: Edit to %s skipped — another write superseded", path)
            return
        
        # Save rollback snapshot
        if tool_call_id:
            if len(state.vfs_rollback_snapshots) >= state.VFS_ROLLBACK_MAX:
                _rollback_oldest_snapshot()
            state.vfs_rollback_snapshots[tool_call_id] = VFSRollbackSnapshot(
                path=path,
                workspace_id=workspace_id,
                old_entry=old_entry,
                new_hash=content_hash,
                tool_call_id=tool_call_id,
                created_at=time.time(),
                parent_snapshot_id=parent_id,
            )
        
        new_entry = VFSEntry(
            path=path,
            content=new_content,
            content_hash=content_hash,
            size_bytes=len(new_content),
            cached_at=time.time(),
            cached_by=session_id,
            workspace_id=workspace_id,
            verified=False,
            line_count=new_content.count('\n') + 1,
        )
        cache.put(new_entry)

    try:
        _log_write(path, session_id, workspace_id, "edit", old_hash, content_hash)
    except Exception as exc:
        logger.warning(
            "VFS: _log_write failed for %s in workspace %s: %s",
            path, workspace_id, exc,
            extra={"request_id": "?"},
        )


def _log_write(file_path: str, agent_id: str, workspace_id: str,
               operation: str, old_hash: str, new_hash: str) -> None:
    """Append a write record to the global write log."""
    import proxy.state as state
    import time
    
    record = VFSWriteRecord(
        path=file_path,
        agent_id=agent_id,
        workspace_id=workspace_id,
        timestamp=time.time(),
        operation=operation,
        old_hash=old_hash,
        new_hash=new_hash,
    )
    with state.vfs_lock:
        state.vfs_write_log.append(record)
        if len(state.vfs_write_log) > state.VFS_WRITE_LOG_MAX:
            state.vfs_write_log[:] = state.vfs_write_log[-state.VFS_WRITE_LOG_MAX:]


def _rollback_snapshot(snapshot: VFSRollbackSnapshot) -> None:
    """Restore cache to pre-write state. Cascades to dependent edits."""
    import proxy.state as state
    
    with state.vfs_lock:
        cache = state.vfs_caches.get(snapshot.workspace_id)
        if cache is None:
            return
        
        current = cache.get(snapshot.path)
        if current and current.content_hash != snapshot.new_hash:
            # Another write superseded — find dependent snapshots
            dependents = [
                s for s in state.vfs_rollback_snapshots.values()
                if s.parent_snapshot_id == snapshot.tool_call_id
            ]
            if dependents:
                cache.invalidate(snapshot.path)
                for dep in dependents:
                    state.vfs_rollback_snapshots.pop(dep.tool_call_id, None)
            return
        
        if snapshot.old_entry is None:
            cache.invalidate(snapshot.path)
        else:
            cache.put(snapshot.old_entry)


def _rollback_stale_snapshots(workspace_id: str) -> None:
    """Rollback snapshots older than VFS_ROLLBACK_TIMEOUT.
    
    Pops stale snapshots under lock, processes rollbacks OUTSIDE lock
    to avoid deadlock (vfs_lock is not reentrant).
    """
    import proxy.state as state
    import time
    
    cutoff = time.time() - state.VFS_ROLLBACK_TIMEOUT
    stale_snapshots = []
    
    with state.vfs_lock:
        stale_ids = [
            tc_id for tc_id, snap in state.vfs_rollback_snapshots.items()
            if snap.workspace_id == workspace_id and snap.created_at < cutoff
        ]
        for tc_id in stale_ids:
            snap = state.vfs_rollback_snapshots.pop(tc_id, None)
            if snap:
                stale_snapshots.append(snap)
    
    # Process rollbacks OUTSIDE lock
    for snap in stale_snapshots:
        _rollback_snapshot(snap)


def _rollback_oldest_snapshot() -> None:
    """Discard the oldest rollback snapshot without rolling back.
    
    IMPORTANT: Caller MUST hold vfs_lock. This function does NOT acquire it.
    Called when VFS_ROLLBACK_MAX is reached.
    """
    import proxy.state as state
    
    if not state.vfs_rollback_snapshots:
        return
    oldest_id = min(
        state.vfs_rollback_snapshots,
        key=lambda tc_id: state.vfs_rollback_snapshots[tc_id].created_at
    )
    state.vfs_rollback_snapshots.pop(oldest_id, None)


def _looks_like_failure(content: str) -> bool:
    """Heuristic failure detection for write/edit tool results.
    
    Checks first 200 chars for failure patterns. Short-circuits on
    known success prefixes to reduce false positives.
    """
    if not content or not isinstance(content, str):
        return False
    
    header = content[:200].lower().lstrip()
    
    # Short-circuit on known success prefixes
    success_prefixes = [
        "file written", "successfully", "updated", "created",
        "wrote", "saved", "file created", "file updated",
        "applied", "done", "ok", "success",
    ]
    for prefix in success_prefixes:
        if header.startswith(prefix):
            return False
    
    import re
    failure_patterns = [
        r'\berror:', r'\berror\b', r'\bfailed:', r'\bfailed to\b',
        r'\benoent\b', r'\beacces\b', r'\benosp\b', r'\beisdir\b', r'\bebusy\b',
        r'\bpermission denied\b', r'\bno such file\b', r'\bnot found\b',
        r'\bno space left\b', r'\bread-only file system\b',
        r'\bold_string not found\b', r'\boldstring not found\b',
        r'\bno match found\b', r'\bmultiple matches\b',
        r'\bis_errored\b', r'\btool_error\b',
        r'\bexception\b', r'\btraceback\b', r'\berrno\b',
    ]
    return any(re.search(p, header) for p in failure_patterns)


def _build_vfs_context(session_id, ctx, budget_tokens, tokens_per_char):
    """Build VFS context message listing cached files for the workspace.
    
    Returns a system message dict or None if no cache data.
    Only active for main agents (sub-agents get inherited context instead).
    """
    import proxy.state as state

    if not _vfs_active_for_context(False):
        return None
    if getattr(ctx, "is_subagent", False):
        return None

    # v5: workspaces are global. Get the session's actual active workspace
    # (or 'default') — bare name, no session prefix.
    from proxy.workspace_state import resolve_inherited_workspace
    workspace_id = resolve_inherited_workspace(session_id, client_hint=getattr(ctx, 'client_hint', ''))

    now = time.time()
    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_id)
        if not cache or cache.size == 0:
            return None
        entries = [
            e for e in cache.snapshot_entries()
            if (now - e.cached_at) <= state.VFS_TTL_SECONDS
        ]

    if not entries:
        return None

    entries = sorted(entries, key=lambda e: e.path)

    if budget_tokens <= 0 or tokens_per_char <= 0:
        return None

    total_entries = len(entries)
    row_cap = min(total_entries, 30)

    base_lines = [
        f"Workspace: {workspace_id} | Cached files: {total_entries}",
        "",
        "Cached files (available for instant reads):",
    ]

    row_lines = [
        (
            f"  {entry.path} "
            f"(hash:{entry.content_hash[:8]}, {entry.size_bytes}B)"
        )
        for entry in entries[:row_cap]
    ]

    def render_content(retained_count, include_omission):
        lines = base_lines + row_lines[:retained_count]
        omitted_count = total_entries - retained_count
        if include_omission and omitted_count > 0:
            lines.append(f"  ... and {omitted_count} more files")
        return (
            "[VFS CONTEXT — Cached file state]\n"
            + "\n".join(lines)
            + "\n[END VFS CONTEXT]"
        )

    def fits(content):
        return int(len(content) * tokens_per_char) <= budget_tokens

    for retained_count in range(row_cap, 0, -1):
        omitted_count = total_entries - retained_count

        if omitted_count > 0:
            content = render_content(
                retained_count,
                include_omission=True,
            )
            if fits(content):
                return {
                    "role": "system",
                    "content": content,
                }

        content = render_content(
            retained_count,
            include_omission=False,
        )
        if fits(content):
            return {
                "role": "system",
                "content": content,
            }

    return None


def _build_vfs_inherited_context(ctx, budget_tokens, tokens_per_char):
    """Build a strictly budgeted inherited VFS block for sub-agents."""
    import proxy.state as state

    if not _vfs_active_for_context(True):
        return None
    if not getattr(ctx, "is_subagent", False):
        return None
    if budget_tokens <= 0 or tokens_per_char <= 0:
        return None

    session_id = getattr(ctx, "session_id", "")
    from proxy.workspace_state import resolve_inherited_workspace
    workspace_id = resolve_inherited_workspace(
        session_id,
        client_hint=getattr(ctx, "client_hint", ""),
    )

    with state.vfs_lock:
        cache = state.vfs_caches.get(workspace_id)
        if not cache or cache.size == 0:
            return None

        now = time.time()
        entries = []
        for entry in cache.snapshot_entries():
            if (now - entry.cached_at) > state.VFS_TTL_SECONDS:
                continue
            entries.append({
                "path": entry.path,
                "hash": entry.content_hash[:8],
                "size_bytes": entry.size_bytes,
                "line_count": entry.line_count,
            })

        write_cutoff = now - 300
        recent_writes = [
            record
            for record in state.vfs_write_log
            if (
                record.workspace_id == workspace_id
                and record.timestamp >= write_cutoff
            )
        ]

    if not entries:
        return None

    written_paths = {record.path for record in recent_writes}
    entries.sort(key=lambda entry: entry["path"])

    written_entries = [
        entry
        for entry in entries
        if entry["path"] in written_paths
    ]
    read_entries = [
        entry
        for entry in entries
        if entry["path"] not in written_paths
    ]

    row_candidates = [
        (
            "written",
            (
                f"  {entry['path']} "
                f"(hash:{entry['hash']}, {entry['line_count']} lines)"
            ),
        )
        for entry in written_entries[:10]
    ]
    row_candidates.extend(
        (
            "read",
            (
                f"  {entry['path']} "
                f"(hash:{entry['hash']}, {entry['line_count']} lines)"
            ),
        )
        for entry in read_entries[:15]
    )

    if not row_candidates:
        return None

    total_entries = len(entries)
    row_cap = len(row_candidates)
    parent_id = getattr(ctx, "parent_agent_id", "unknown")

    def render_content(retained_count, include_omission):
        selected = row_candidates[:retained_count]
        lines = [
            (
                f"Workspace: {workspace_id} | Direct parent: {parent_id} "
                f"| Cached files: {total_entries}"
            ),
            "",
        ]

        written_rows = [
            row
            for section, row in selected
            if section == "written"
        ]
        read_rows = [
            row
            for section, row in selected
            if section == "read"
        ]

        if written_rows:
            lines.append("Recently written (parent agent):")
            lines.extend(written_rows)
            lines.append("")

        if read_rows:
            lines.append("Recently read (parent agent):")
            lines.extend(read_rows)
            lines.append("")

        omitted_count = total_entries - retained_count
        if include_omission and omitted_count > 0:
            lines.append(f"  ... and {omitted_count} more files")
            lines.append("")

        lines.append(
            "These files are cached — reading them will be instant "
            "(no disk I/O)."
        )
        text = "\n".join(lines)
        return (
            "[VFS INHERITED CONTEXT — Parent agent's recent file activity]\n"
            + text
            + "\n[END VFS INHERITED CONTEXT]"
        )

    def fits(content):
        return (
            int(len(content) * tokens_per_char)
            <= budget_tokens
        )

    for retained_count in range(row_cap, 0, -1):
        omitted_count = total_entries - retained_count

        if omitted_count > 0:
            content = render_content(
                retained_count,
                include_omission=True,
            )
            if fits(content):
                return {
                    "role": "system",
                    "content": content,
                }

        content = render_content(
            retained_count,
            include_omission=False,
        )
        if fits(content):
            return {
                "role": "system",
                "content": content,
            }

    return None


def detect_write_conflicts(path: str, workspace_id: str, reader_agent_id: str,
                           since_seconds: float = 120.0) -> list:
    """Return write records by OTHER agents to this path within since_seconds.
    
    Used by _check_vfs_read_cache() to warn agents when they read a file
    that was recently modified by another agent in the same workspace.
    """
    import proxy.state as state
    
    cutoff = time.time() - since_seconds
    conflicts = []
    with state.vfs_lock:
        for record in state.vfs_write_log:
            if (record.path == path
                and record.workspace_id == workspace_id
                and record.agent_id != reader_agent_id
                and record.timestamp >= cutoff):
                conflicts.append(record)
    return conflicts


def _compute_diff_summary(old_content: str, new_content: str) -> str:
    """Produce a compact diff summary for conflict warnings.
    
    Returns a short string like "+15 lines" or "-3 lines" or "modified (same line count)".
    """
    old_lines = old_content.count('\n')
    new_lines = new_content.count('\n')
    delta = new_lines - old_lines
    if delta > 0:
        return f"+{delta} lines"
    elif delta < 0:
        return f"{delta} lines"
    return "modified (same line count)"
