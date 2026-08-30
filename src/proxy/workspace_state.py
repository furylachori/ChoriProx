"""Thread-safe store for per-session workspace state.

Extracted from state.py to reduce the god object. The lock passed to
__init__ MUST be the same object as state.py's active_workspace_lock
to preserve lock ordering position #6.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("proxy.workspace_state")

DEFAULT_WORKSPACE = "default"
MAX_RECENT_WORKSPACES = 10
CATALOG_CACHE_TTL_SECONDS = 300  # 5 minutes


class WorkspaceStateStore:
    """Thread-safe store for per-session workspace state.

    The lock MUST be the same object as state.py's active_workspace_lock
    (position #6 in lock hierarchy).
    """

    def __init__(self, lock: threading.RLock,
                 persist_callback=None, recovery_fn=None):
        self._lock = lock
        self._active: dict[str, str] = {}
        self._recent: dict[str, list[str]] = {}
        self._last_access: dict[str, float] = {}
        self._client_active: dict[str, str] = {}  # client_hint → workspace
        # v5: persist signature changed from (workspace_id, name, session_id)
        # to (workspace_name, session_id). workspace_id == workspace_name now.
        self._persist = persist_callback  # callable(workspace_name, session_id) | None
        self._recovery_fn = recovery_fn   # callable(session_id) | None (wired after construction)
        self._recovered_sessions: set[str] = set()  # sessions where recovery already succeeded
        # v5: global catalog cache. Populated lazily by recovery; invalidated
        # when a new workspace name is seen.
        self._catalog_cache: list[str] = []
        self._catalog_cache_ts: float = 0.0


    def activate(
        self,
        session_id: str,
        workspace: str,
        client_hint: str = ""
    ) -> tuple[str, list[str]]:
        """Set active + update recent + touch. Returns (old_active, updated_recent).

        Persistence callback is fired OUTSIDE the lock (fire-and-forget).
        The workspace is always activated in-memory before the persist call,
        so Neo4j failures never block workspace activation.

        Args:
            session_id: The session ID for this request.
            workspace: The workspace name to activate.
            client_hint: Optional client identifier (IP or client ID) for
                         cross-session persistence.
        """
        persist_payload = None
        with self._lock:
            old = self._active.get(session_id, DEFAULT_WORKSPACE)
            self._active[session_id] = workspace
            self._last_access[session_id] = time.time()

            # Update recent workspaces list
            recent = self._recent.setdefault(session_id, [])
            if workspace in recent:
                recent.remove(workspace)
            recent.insert(0, workspace)
            self._recent[session_id] = recent[:MAX_RECENT_WORKSPACES]

            # Persist workspace at client level if client_hint provided
            if client_hint and workspace != DEFAULT_WORKSPACE:
                self._client_active[client_hint] = workspace

            # Capture persist payload INSIDE lock, fire OUTSIDE
            if self._persist and workspace != DEFAULT_WORKSPACE:
                # v5: bare name, no session prefix. workspace_id == workspace.
                persist_payload = (workspace, session_id)
                # Invalidate catalog cache — a new workspace may have appeared
                if workspace not in self._catalog_cache:
                    self._catalog_cache = []
            recent_snapshot = list(self._recent.get(session_id, []))

        # --- lock released ---
        if persist_payload is not None:
            try:
                # v5: lazy rewire — if persist is None, try to acquire one
                # (handles Neo4j coming online after module load)
                if self._persist is None:
                    from memory.workspace_persistence import create_persist_callback
                    self._persist = create_persist_callback()
                if self._persist is not None:
                    self._persist(*persist_payload)
            except Exception:
                logger.warning(
                    "Workspace persist failed for %s:%s",
                    session_id[:8], workspace, exc_info=True
                )

        return old, recent_snapshot

    def deactivate(self, session_id: str, client_hint: str = "") -> tuple[str | None, list[str]]:
        """Clear active + pop recent + pop last_access. Returns (cleared_workspace, recent).

        Args:
            session_id: The session ID.
            client_hint: Optional client identifier to clear client-level workspace.

        NOTE: does NOT discard from _recovered_sessions — if the session was
        recovered from Neo4j, it stays recovered. Otherwise the next call to
        get_active() would trigger recovery again and restore the workspace
        from Neo4j (defeating the leave/skip intent).
        """
        with self._lock:
            ws = self._active.pop(session_id, None)
            recent = self._recent.pop(session_id, [])
            self._last_access.pop(session_id, None)

            # Clear client-level workspace if requested
            if client_hint and client_hint in self._client_active:
                del self._client_active[client_hint]

            return ws, recent

    def get_active(self, session_id: str, client_hint: str = "") -> str:
        """Get active workspace for session, falling back to client-level workspace.
        
        Lazy recovery: on first cold-session request, calls _recovery_fn (if set)
        to rebuild workspace state from Neo4j. Recovery is attempted once per
        session (tracked in _recovered_sessions). Subsequent calls skip
        recovery and return the in-memory state.
        
        Args:
            session_id: The session ID.
            client_hint: Optional client identifier for cross-session persistence.
        
        Returns:
            The active workspace name, or DEFAULT_WORKSPACE if none set.
        """
        with self._lock:
            # Check session-level first
            ws = self._active.get(session_id)
            if ws is not None:
                return ws
            
            # Fall back to client-level workspace
            if client_hint:
                ws = self._client_active.get(client_hint)
                if ws is not None:
                    # Propagate to session level for this session
                    self._active[session_id] = ws
                    self._last_access[session_id] = time.time()
                    # Update recent workspaces
                    recent = self._recent.setdefault(session_id, [])
                    if ws in recent:
                        recent.remove(ws)
                    recent.insert(0, ws)
                    self._recent[session_id] = recent[:MAX_RECENT_WORKSPACES]
                    return ws
            
            # Neither session nor client has a workspace — cold session
            ws = DEFAULT_WORKSPACE
        
        # --- lock released ---
        # Lazy recovery: attempt only for cold sessions where recovery hasn't
        # already succeeded. If recovery fails (Neo4j down, transient error),
        # _recovered_sessions is NOT updated — next get_active() retries.
        if (self._recovery_fn is not None
                and ws == DEFAULT_WORKSPACE
                and session_id not in self._recovered_sessions):
            try:
                self._recovery_fn(session_id)
            except Exception as e:
                logger.warning("workspace recovery failed session=%s: %s", session_id, e)
            # Re-read under lock; recovery wrote _active only on success
            with self._lock:
                ws = self._active.get(session_id, DEFAULT_WORKSPACE)
        
        return ws

    def try_recover(self, session_id: str, active: str, recent: list[str]) -> bool:
        """Atomically check-and-set recovery state. Acquires self._lock internally.

        Returns True if recovery was applied (session was cold).
        Returns False if session was already populated (concurrent activate() won).

        Also marks session in _recovered_sessions atomically with the write,
        preventing the pop_session() race. Guard against NULL/empty names
        (storing None in _active would break get_active()).
        """
        with self._lock:
            if session_id in self._active:
                return False
            if not active or not isinstance(active, str):
                return False
            self._active[session_id] = active
            self._recent[session_id] = recent[:MAX_RECENT_WORKSPACES]
            self._last_access[session_id] = time.time()
            self._recovered_sessions.add(session_id)
            return True

    def get_client_workspace(self, client_hint: str) -> str | None:
        """Get client-level workspace (does NOT propagate to session).
        
        Args:
            client_hint: The client identifier.
        
        Returns:
            The workspace name or None if not set.
        """
        with self._lock:
            return self._client_active.get(client_hint)

    def set_active(self, session_id: str, workspace: str, client_hint: str = "") -> str:
        """Set active workspace. Returns old workspace.

        Same pattern as activate() — persist fires OUTSIDE the lock.
        """
        persist_payload = None
        with self._lock:
            old = self._active.get(session_id, DEFAULT_WORKSPACE)
            self._active[session_id] = workspace
            self._last_access[session_id] = time.time()
            recent = self._recent.setdefault(session_id, [])
            if workspace in recent:
                recent.remove(workspace)
            recent.insert(0, workspace)
            self._recent[session_id] = recent[:MAX_RECENT_WORKSPACES]

            if client_hint and workspace != DEFAULT_WORKSPACE:
                self._client_active[client_hint] = workspace

            if self._persist and workspace != DEFAULT_WORKSPACE:
                # v5: bare name, no session prefix
                persist_payload = (workspace, session_id)
                if workspace not in self._catalog_cache:
                    self._catalog_cache = []

        if persist_payload is not None:
            try:
                # v5: lazy rewire
                if self._persist is None:
                    from memory.workspace_persistence import create_persist_callback
                    self._persist = create_persist_callback()
                if self._persist is not None:
                    self._persist(*persist_payload)
            except Exception:
                logger.warning(
                    "Workspace persist failed for %s:%s",
                    session_id[:8], workspace, exc_info=True
                )

        return old

    def get_global_catalog(self) -> list[str]:
        """Return the global workspace catalog (cached for 5 minutes).

        v5: workspaces are global. The catalog is loaded from Neo4j on cold
        start or after cache invalidation. Always includes 'default'.

        BUG #12: The Neo4j round-trip is performed OUTSIDE self._lock so
        concurrent callers (activate/get_active/deactivate across other
        sessions) are not blocked waiting on the fetch. The cache compare-
        and-set happens UNDER the lock, consistent with concurrent
        invalidations in activate()/set_active(). Lock ordering (#6) is
        preserved — no other lock is acquired while holding self._lock.
        """
        # Fast path: cache hit returns under a single short lock hold.
        with self._lock:
            now = time.time()
            if (self._catalog_cache
                    and now - self._catalog_cache_ts < CATALOG_CACHE_TTL_SECONDS):
                return list(self._catalog_cache)

        # Cache miss — fetch from Neo4j OUTSIDE the lock so the global
        # workspace lock is not held across the network round-trip.
        from memory.workspace_persistence import get_global_catalog as _fetch
        catalog = _fetch()

        # Compare-and-set under the lock. Re-check the cache first: another
        # thread may have populated or invalidated it while we fetched.
        with self._lock:
            now = time.time()
            if (self._catalog_cache
                    and now - self._catalog_cache_ts < CATALOG_CACHE_TTL_SECONDS):
                return list(self._catalog_cache)
            self._catalog_cache = catalog
            self._catalog_cache_ts = now
            return list(catalog)

    def set_catalog_cache(self, catalog: list[str]) -> None:
        """Set the catalog cache directly (used by recovery to avoid duplicate Neo4j reads)."""
        with self._lock:
            self._catalog_cache = list(catalog)
            self._catalog_cache_ts = time.time()

    def clear_active(self, session_id: str) -> str | None:
        with self._lock:
            return self._active.pop(session_id, None)

    def get_recent(self, session_id: str) -> list[str]:
        with self._lock:
            return list(self._recent.get(session_id, []))

    def touch(self, session_id: str) -> None:
        with self._lock:
            self._last_access[session_id] = time.time()

    def get_last_access(self, session_id: str) -> float | None:
        with self._lock:
            return self._last_access.get(session_id)

    def snapshot(self, session_id: str) -> tuple[str, list[str]]:
        """Return (active_workspace, recent_workspaces) atomically."""
        with self._lock:
            return (
                self._active.get(session_id, DEFAULT_WORKSPACE),
                list(self._recent.get(session_id, []))
            )

    def forget(self, session_id: str, workspace: str) -> bool:
        """Remove workspace from this session's MRU list. Does NOT touch active.

        Returns True if the workspace was removed, False if it wasn't in the list.
        """
        with self._lock:
            recent = self._recent.get(session_id)
            if not recent or workspace not in recent:
                return False
            recent.remove(workspace)
            return True

    def pop_session(self, session_id: str) -> tuple[str | None, list[str], float | None]:
        """Remove all state for session. Returns (workspace, recent, last_access)."""
        with self._lock:
            self._recovered_sessions.discard(session_id)
            ws = self._active.pop(session_id, None)
            recent = self._recent.pop(session_id, [])
            la = self._last_access.pop(session_id, None)
            return ws, recent, la

    def get_stale_sessions(self, ttl_seconds: float) -> list[str]:
        with self._lock:
            cutoff = time.time() - ttl_seconds
            return [sid for sid, ts in self._last_access.items() if ts < cutoff]
    
    def client_heartbeat(self, client_hint: str) -> None:
        """Update timestamp for all sessions belonging to this client.
        
        Prevents workspace state cleanup while client is active.
        """
        with self._lock:
            client_ws = self._client_active.get(client_hint)
            if not client_ws:
                return
            
            now = time.time()
            # Touch all sessions using this client's workspace
            for session_id, ws in self._active.items():
                if ws == client_ws:
                    self._last_access[session_id] = now

    @property
    def lock(self) -> threading.RLock:
        """Expose lock for callers needing custom compound operations."""
        return self._lock


workspace_store: WorkspaceStateStore = None  # Initialized by state.py


def init_workspace_store(lock: threading.RLock,
                        persist_callback=None,
                        recovery_fn=None) -> WorkspaceStateStore:
    global workspace_store
    workspace_store = WorkspaceStateStore(lock, persist_callback, recovery_fn)
    return workspace_store


def resolve_workspace(ctx) -> str:
    """N5: Resolve workspace for a request context.

    Replaces the defensive `workspace_store.get_active(getattr(ctx, 'session_id', ''), client_hint=getattr(ctx, 'client_hint', ''))`
    pattern at 19 call sites. Centralises the getattr fallback so call
    sites stop drifting on whether they handle missing attributes.

    For call sites where `client_hint` is a function parameter (not on
    ctx), continue using `workspace_store.get_active(sid, hint)` directly.
    """
    return workspace_store.get_active(
        getattr(ctx, 'session_id', ''),
        client_hint=getattr(ctx, 'client_hint', ''),
    )


def resolve_inherited_workspace(
    session_id: str,
    client_hint: str = "",
    *,
    log_failures: bool = True,
) -> str:
    """N3: Walk the parent_agent_id chain to find the root workspace.

    Returns the session's own active workspace unless it equals
    'default', in which case walks `agent_registry.get_or_load` parent
    chain to the root and returns the root's workspace.

    Args:
      session_id: the session to resolve
      client_hint: forwarded to workspace_store.get_active
      log_failures: if True (default), log warnings when the parent-chain
        walk raises. The pre-consolidation sites 3+4 silently swallowed
        errors via `except Exception: pass`, which hid failures that
        sites 1+2 logged. The drift is now ours per Bug Policy — the
        helper always logs unless the caller explicitly opts out.
    """
    ws = workspace_store.get_active(session_id, client_hint=client_hint)
    if ws != "default":
        return ws
    try:
        from proxy.agent.agent_registry import registry as _registry
        visited = set()
        current_sid = session_id
        while current_sid and current_sid not in visited:
            visited.add(current_sid)
            agent = _registry.get_or_load(current_sid)
            if not agent or not getattr(agent, 'parent_agent_id', None):
                break
            current_sid = agent.parent_agent_id
        # parent_agent_id is an agent_id, not a session_id. workspace_store
        # is keyed by session_id — convert via get_or_load(agent_id).session_id.
        root_agent = _registry.get_or_load(current_sid) if current_sid else None
        root_session_id = (
            getattr(root_agent, "session_id", None) if root_agent else None
        ) or current_sid
        root_ws = workspace_store.get_active(root_session_id)
    except Exception as e:
        if log_failures:
            logger.warning(
                "resolve_inherited_workspace: parent-chain walk failed for session_id=%s; "
                "falling back to session's own workspace. error=%s",
                str(session_id or "")[:8], e, exc_info=True,
            )
        return ws
    if root_ws and root_ws != "default":
        return root_ws
    return ws


def resolve_inherited_session(
    session_id: str,
    *,
    log_failures: bool = True,
) -> str:
    """N3: Walk the parent_agent_id chain to find the root session.

    Returns the root session_id — the session under which the root (main)
    agent captured Tier-2 instruction files. Needed because
    ``record_instruction_file`` keys default-workspace captures as
    ``(DEFAULT_WORKSPACE, session_id)``; a sub-agent looking up under its
    OWN session_id would miss the parent's capture.

    Args:
      session_id: the session to resolve
      log_failures: if True (default), log warnings when the parent-chain
        walk raises.
    """
    try:
        from proxy.agent.agent_registry import registry as _registry
        visited = set()
        current_sid = session_id
        while current_sid and current_sid not in visited:
            visited.add(current_sid)
            agent = _registry.get_or_load(current_sid)
            if not agent or not getattr(agent, 'parent_agent_id', None):
                break
            current_sid = agent.parent_agent_id
        root_agent = _registry.get_or_load(current_sid) if current_sid else None
        root_session_id = (
            getattr(root_agent, "session_id", None) if root_agent else None
        ) or current_sid
    except Exception as e:
        if log_failures:
            logger.warning(
                "resolve_inherited_session: parent-chain walk failed for session_id=%s; "
                "falling back to session's own id. error=%s",
                str(session_id or "")[:8], e, exc_info=True,
            )
        return session_id
    return root_session_id
