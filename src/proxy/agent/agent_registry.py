"""AgentRegistry - manages agent lifecycle, LRU caching, and tree tracking.

Phase 7: Base version with get_or_load() and LRU eviction.
Phase 8: Extended with get_tree(), get_all_roots(), agent_return().
"""
import contextlib
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from typing import Any, Dict, Optional, List

from proxy.agent.agent import Agent
from proxy.agent.agent_config import AgentConfig
from proxy.skill_lifecycle import register_lifecycle_callbacks
from proxy.lock_ordering import make_lock

logger = logging.getLogger("proxy.agent.registry")

# Configurable max agents (default 100)
# NOTE: Low AGENT_REGISTRY_MAX_AGENTS values (<100) combined with content-derived
# session IDs allow an attacker to force eviction of legitimate agents by varying
# the first message. Keep this value high enough for expected concurrent sessions.
AGENT_REGISTRY_MAX_AGENTS = int(os.environ.get("AGENT_REGISTRY_MAX_AGENTS", "100"))


class AgentRegistry:
    """Registry for managing Agent instances with LRU eviction.
    
    SQLite is the source of truth. This registry is an in-memory LRU cache
    backed by SQLite persistence.
    
    Lock ordering: This is the COARSEST lock (position 0 in the hierarchy).
    Always acquire AgentRegistry._lock BEFORE acquiring Agent._mode_lock or
    MessageBuffer._lock.
    See: state.py:66-78 for the full lock ordering documentation.
    """

    def __init__(self, max_agents: int = AGENT_REGISTRY_MAX_AGENTS):
        self.max_agents = max_agents
        self._agents: Dict[str, Agent] = {}
        self._agents_by_id: Dict[str, Agent] = {}
        self._lru: OrderedDict = OrderedDict()
        self._lock = make_lock(0, "AgentRegistry._lock", threading.Lock)
        
        # Phase 8: _children index for tree tracking (placeholder now)
        self._children: Dict[str, List[Agent]] = {}

        # In-flight request tracking: prevents LRU eviction / TTL expiry
        # of an agent while it is actively processing a tool loop.
        self._in_flight_requests: dict[str, int] = {}

        # H11: Rate limiting state for get_or_load enumeration protection
        # R2-HIGH-1: Use deque for O(1) popleft, prune periodically instead of every call
        self._gol_query_log: deque[tuple[float, str]] = deque()
        self._gol_call_count: int = 0

    # -------------------------------------------------------------------------
    # In-flight request tracking (R3-H8 / R3-H9)
    # -------------------------------------------------------------------------

    @contextlib.contextmanager
    def request_scope(self, session_id: str):
        """Context manager marking a request as in-flight for a session.

        Prevents LRU eviction and TTL cleanup during the request's lifetime.
        """
        with self._lock:
            self._in_flight_requests[session_id] = (
                self._in_flight_requests.get(session_id, 0) + 1
            )
        try:
            yield
        finally:
            with self._lock:
                count = self._in_flight_requests.get(session_id, 0) - 1
                if count <= 0:
                    self._in_flight_requests.pop(session_id, None)
                else:
                    self._in_flight_requests[session_id] = count

    # H11: Caller authorization + rate limiting for get_or_load().
    # Session IDs are content-derived hashes, making them enumerable.
    # Two defenses:
    #   1. caller_session_id validation — callers must prove ownership
    #   2. Rate limiting — blocks enumeration after 100 unique IDs per minute
    _GET_OR_LOAD_RATE_WINDOW = 60  # seconds
    _GET_OR_LOAD_RATE_LIMIT = 100  # unique session_ids per window

    def get_or_load(
        self,
        session_id: str,
        caller_session_id: Optional[str] = None,
    ) -> Optional[Agent]:
        """Load or retrieve an agent from the registry.

        Args:
            session_id: The session to load.
            caller_session_id: If provided, must match *session_id* or the
                request is denied.  Pass ``None`` only from trusted internal
                callers (e.g. tree traversal, agent_return).

        Returns:
            The Agent instance, or ``None`` if authorization fails or the
            caller is rate-limited.
        """
        # H11-1: Caller authorization check
        if caller_session_id is not None and caller_session_id != session_id:
            logger.warning(
                "get_or_load: caller %s attempted to access session %s — denied",
                caller_session_id[:12],
                session_id[:12],
            )
            return None

        # H11-2: Rate limiting — track unique session_ids queried per window
        # R2-HIGH-1: Prune periodically (every 10th call) using deque popleft for O(1)
        now = time.monotonic()
        with self._lock:
            self._gol_call_count += 1
            if self._gol_call_count % 10 == 0:
                cutoff = now - self._GET_OR_LOAD_RATE_WINDOW
                while self._gol_query_log and self._gol_query_log[0][0] < cutoff:
                    self._gol_query_log.popleft()
            # F3: Only rate-limit external callers; internal pipeline stages are trusted
            if caller_session_id is not None:
                if len(self._gol_query_log) >= self._GET_OR_LOAD_RATE_LIMIT:
                    logger.warning(
                        "get_or_load: rate limit exceeded (%d queries in window) — denying %s",
                        len(self._gol_query_log),
                        session_id[:12],
                    )
                    return None
                self._gol_query_log.append((now, session_id))

        # Fast path: check under lock first
        with self._lock:
            if session_id in self._agents:
                self._lru.move_to_end(session_id)
                return self._agents[session_id]
            # Also check by agent_id (for spawned agents)
            if session_id in self._agents_by_id:
                agent = self._agents_by_id[session_id]
                self._lru.move_to_end(agent.session_id)
                return agent
        
        # Slow path: load from DB (outside lock)
        agent = self._load_from_db(session_id)
        
        # F3: If session doesn't exist in authoritative store, return None.
        # This prevents privilege escalation via forged session IDs.
        if agent is None:
            return None
        
        evicted_agent = None
        
        # Re-acquire lock for registration
        with self._lock:
            # Double-check: another thread may have loaded while we waited
            if session_id in self._agents:
                self._lru.move_to_end(session_id)
                return self._agents[session_id]
            
            # Check capacity and evict if needed before registering
            if len(self._agents) >= self.max_agents:
                evicted_agent = self._evict_lru()
                # C-6: If eviction failed (all in-flight), reject the new agent
                if evicted_agent is None and len(self._agents) >= self.max_agents:
                    logger.warning(
                        "AgentRegistry: Overflow blocked — rejecting agent %s, "
                        "at capacity (%d/%d) with all existing agents in-flight.",
                        session_id[:12], len(self._agents), self.max_agents,
                    )
                    return None
            
            self._register(agent)
        
        # H-4: Clear active mode OUTSIDE self._lock to avoid deadlock.
        # clear_active_mode() may trigger on_deactivate() -> get_or_load()
        # which would re-acquire self._lock if called under it.
        # Race window: Between eviction (line 156) and clear_active_mode (below),
        # another thread could re-load the same session. This is benign because:
        # - The evicted agent is already removed from _agents and _agents_by_id
        # - A concurrent re-load creates a fresh agent instance
        # - The stale clear_active_mode only affects the evicted instance (harmless)
        if evicted_agent is not None:
            try:
                evicted_agent.clear_active_mode()
            except Exception:
                logger.exception(
                    "Failed to clear active mode during LRU eviction of agent %s",
                    evicted_agent.agent_id,
                )
        
        return agent

    def _load_from_db(self, session_id: str) -> Optional[Agent]:
        """Load an agent from SQLite.
        
        In Phase 7, this is a wrapper around existing chat_histories + session_state.
        In future phases, this will load directly from SQLite agent tables.
        
        Algorithm:
        1. Check if session exists in chat_histories (in-memory LRU cache)
        2. If not in cache, read through to SQLite (the authoritative store)
        3. If SQLite also has no data, return None — prevents privilege escalation
        4. Load messages and create Agent with default state
        
        F3 SECURITY: Returns None for unknown sessions. Callers (app.py, canary.py)
        rely on this to reject forged session IDs. Never create phantom agents for
        arbitrary IDs — that would bypass privilege escalation checks.
        
        F5 FIX: Read-through to SQLite when session is evicted from the in-memory
        LRU cache. Previously only checked chat_histories (cache), treating evicted
        sessions as non-existent despite them existing in SQLite.
        """
        from proxy.chat_history import chat_histories, chat_histories_lock, _load_from_sqlite
        from proxy.agent.agent_config import AgentConfig

        # F5: Check in-memory cache first, fall through to SQLite if not found.
        # chat_histories is an LRU cache — SQLite is the authoritative store.
        needs_sqlite_check = False

        with chat_histories_lock:
            if session_id in chat_histories:
                # F1 FIX: Use dict.__getitem__ to bypass ChatHistoryCache.__getitem__,
                # which would acquire AgentRegistry._lock (#0) via _build_deferred_clears
                # while chat_histories_lock (#2) is held — violating lock ordering.
                messages = list(dict.__getitem__(chat_histories, session_id))
            else:
                needs_sqlite_check = True

        # F5: SQLite read-through for sessions evicted from the LRU cache.
        # Done outside chat_histories_lock to respect lock ordering.
        if needs_sqlite_check:
            sqlite_msgs = _load_from_sqlite(session_id)
            # F2 FIX: Distinguish "session doesn't exist" (None) from "session
            # exists but has no messages yet" ([]). Only None indicates a forged
            # or invalid session ID — empty list is valid for new sessions whose
            # first message is a workspace command.
            if sqlite_msgs is None:
                return None
            # Re-populate cache with SQLite data.
            # chat_histories.__setitem__ handles LRU tracking and deferred clears
            # safely outside chat_histories_lock (lock ordering preserved).
            chat_histories[session_id] = sqlite_msgs
            messages = list(sqlite_msgs)

        # Create agent with default empty config
        agent = Agent(
            session_id=session_id,
            parent=None,
            config=AgentConfig(),
        )

        # Wire up IoC deactivation callbacks
        register_lifecycle_callbacks(agent)

        # Set up the buffer with loaded messages
        if messages:
            agent._buffer = Agent.__dataclass_fields__['_buffer'].default_factory()
            agent._buffer.extend(messages)

        return agent

    def _register(self, agent: Agent) -> None:
        """Add an agent to the registry and LRU.

        Assumes caller holds self._lock.

        Phase 8 will also update the _children index for tree tracking.
        """
        self._agents[agent.session_id] = agent
        self._agents_by_id[agent.agent_id] = agent
        self._lru[agent.session_id] = None
        self._lru.move_to_end(agent.session_id)

        logger.debug(
            f"AgentRegistry: Registered agent {agent.agent_id} "
            f"for session {agent.session_id[:8]}..."
        )

    def register(self, agent: Agent) -> None:
        """Public registration entry point.

        Acquires ``self._lock`` and (if needed) evicts the LRU agent before
        delegating to :meth:`_register`. Intended for callers (such as the
        pipeline) that need to register a freshly constructed Agent outside
        the normal ``get_or_load`` / ``spawn`` paths — e.g. a brand-new
        sub-agent whose session has no SQLite history yet.

        Eviction follows the same policy as :meth:`get_or_load`: if the
        registry is at capacity and every existing agent has an in-flight
        request, the new agent is rejected and ``None`` is returned
        (caller can decide whether to fall back). On success, the registered
        agent is returned.

        Lifecycle callbacks (Phase B IoC) are NOT wired up here — callers
        must invoke ``register_lifecycle_callbacks(agent)`` themselves if
        they need on_deactivate dispatch.
        """
        evicted_agent = None
        with self._lock:
            # Double-check: another thread may have registered while we waited
            if agent.session_id in self._agents:
                self._lru.move_to_end(agent.session_id)
                return
            if agent.agent_id in self._agents_by_id:
                existing = self._agents_by_id[agent.agent_id]
                self._lru.move_to_end(existing.session_id)
                return
            # Capacity check
            if len(self._agents) >= self.max_agents:
                evicted_agent = self._evict_lru()
                if evicted_agent is None and len(self._agents) >= self.max_agents:
                    logger.warning(
                        "AgentRegistry.register: Overflow blocked — rejecting "
                        "agent %s, at capacity (%d/%d) with all existing "
                        "agents in-flight.",
                        agent.session_id[:12], len(self._agents), self.max_agents,
                    )
                    return
            self._register(agent)

        # Clear evicted agent's active mode outside the lock (H-4 deadlock fix).
        if evicted_agent is not None:
            try:
                evicted_agent.clear_active_mode()
            except Exception:
                logger.exception(
                    "Failed to clear active mode during LRU eviction of agent %s",
                    evicted_agent.agent_id,
                )

    def _evict_lru(self) -> Optional[Agent]:
        """Evict the least recently used agent from the cache.

        Assumes caller holds self._lock.

        Returns the evicted Agent so the caller can clean up its resources
        OUTSIDE self._lock (avoids deadlock when clear_active_mode triggers
        on_deactivate -> get_or_load -> self._lock). Returns None if no
        agent could be evicted (empty LRU or all agents in-flight).

        Algorithm:
        1. Walk LRU from oldest to newest, skipping agents with in-flight requests
        2. Pop oldest eligible entry from LRU
        3. Remove from self._agents dict
        4. Log at INFO level
        5. Messages are already synced to SQLite at end-of-request, so no data loss

        R3-H8: Agents with in-flight requests are skipped to avoid evicting
        an agent mid-tool-loop.
        """
        if not self._lru:
            return None

        # Walk LRU from oldest to newest; skip in-flight agents (R3-H8)
        for _ in range(len(self._lru)):
            lru_key, _ = next(iter(self._lru.items()))
            # Resolve actual agent to get its session_id, since spawned agents
            # use agent_id as LRU key but _in_flight_requests is keyed by session_id (R4-H2 fix)
            agent = self._agents.get(lru_key)
            actual_session_id = agent.session_id if agent else lru_key
            if self._in_flight_requests.get(actual_session_id, 0) > 0:
                # Move to end (treat as recently used) and try next
                self._lru.move_to_end(lru_key)
                continue
            break
        else:
            # C-4: All agents are in-flight — caller will handle overflow
            logger.warning(
                "AgentRegistry: All %d agents are in-flight, cannot evict. "
                "Registry will temporarily exceed capacity (max=%d).",
                len(self._lru), self.max_agents,
            )
            return None

        # Pop oldest eligible (LRU)
        lru_key, _ = self._lru.popitem(last=False)

        # Remove from agents dict
        agent = self._agents.pop(lru_key, None)
        # Also remove from agents_by_id index
        if agent:
            self._agents_by_id.pop(agent.agent_id, None)
        if agent:
            # Clean up _children references for evicted agent
            self._children.pop(agent.agent_id, None)  # Remove as parent
            for parent_children in self._children.values():
                parent_children[:] = [
                    c for c in parent_children if c.agent_id != agent.agent_id
                ]

        if agent:
            logger.info(
                f"AgentRegistry: Evicted agent {agent.agent_id} "
                f"from LRU (capacity={self.max_agents})"
            )
            from proxy.metrics import record_agent_eviction
            record_agent_eviction()

        # H-4: Do NOT call clear_active_mode() under self._lock.
        # Return the agent; caller must invoke clear_active_mode() outside the lock.
        return agent

    def cleanup_stale_ticket_modes(self) -> None:
        """Clear expired ModeState entries (Phase B skill system).

        Iterates all agents, checks active_mode.ttl_seconds for expiration,
        and calls clear_active_mode() if TTL has passed.

        Thread safety: agents needing cleanup are snapshot under self._lock,
        then clear_active_mode() is called *outside* self._lock to avoid
        deadlock when on_deactivate callbacks re-enter the registry.

        Race fix: the mode *object* is snapshotted under lock. Before calling
        clear_active_mode(), we re-check that agent.active_mode is still the
        same object (identity check). If a new mode was set between snapshot
        and clear, the new mode is left untouched.
        """
        stale: list[tuple[Agent, object]] = []
        with self._lock:
            for agent in list(self._agents.values()):
                # R3-H9: Skip agents with in-flight requests — do not expire
                # active_mode mid-tool-loop.
                if self._in_flight_requests.get(agent.session_id, 0) > 0:
                    continue
                if agent.is_mode_expired():
                    stale.append((agent, agent.snapshot_mode()))
        for agent, snapshotted_mode in stale:
            try:
                agent.clear_active_mode(expected_mode=snapshotted_mode)
                logger.debug(
                    f"AgentRegistry: Cleared expired active_mode for agent {agent.agent_id}"
                )
            except Exception as e:
                logger.error(f"Failed to clear expired mode: {e}")

    # -------------------------------------------------------------------------
    # Phase 8: Tree and lifecycle methods
    # -------------------------------------------------------------------------

    def get_tree(self, agent_id: str) -> dict:
        """Return agent tree rooted at agent_id as nested dict.

        Algorithm:
        1. Load root agent: agent = self.get_or_load(agent_id)
        2. If not found, return {"error": "Agent not found"}
        3. Build tree recursively:
           tree = {
               "agent_id": agent.agent_id,
               "session_id": agent.session_id,
               "parent_id": agent.parent.agent_id if agent.parent else None,
               "children": []
           }
        4. For each child in self._children.get(agent_id, []):
           tree["children"].append(self.get_tree(child.agent_id))
        5. Return tree
        """
        agent = self.get_or_load(agent_id)
        if agent is None:
            return {"error": "Agent not found"}

        tree: Dict[str, Any] = {
            "agent_id": agent.agent_id,
            "session_id": agent.session_id,
            "parent_id": agent.parent.agent_id if agent.parent else None,
            "children": [],
        }

        with self._lock:
            children_snapshot = list(self._children.get(agent_id, []))
        for child in children_snapshot:
            tree["children"].append(self.get_tree(child.agent_id))

        return tree

    def get_all_roots(self) -> list:
        """Return all root agents (no parent)."""
        with self._lock:
            return [
                agent for agent in self._agents.values()
                if agent.parent is None
            ]

    def agent_return(self, child_agent_id: str, result: dict) -> None:
        """Record that a sub-agent has returned to its parent.

        Algorithm:
        1. Load child agent: child = self.get_or_load(child_agent_id)
        2. Load parent agent: parent = child.parent
        3. If parent is None (root agent), return — no parent to notify
        4. Append return message to parent's MessageBuffer:
           {"role": "tool", "content": result_summary,
            "tool_call_id": child_agent_id, "name": "sub_agent_return"}
        5. Set child status tracking if available
        6. Log the return
        """
        child = self.get_or_load(child_agent_id)
        if child is None:
            logger.warning(f"agent_return: child agent {child_agent_id} not found")
            return

        parent = child.parent
        if parent is None:
            # Root agent — no parent to notify
            logger.debug(f"Agent {child_agent_id} returned to root (no parent)")
            return

        # Build result summary
        result_summary = ""
        if isinstance(result, dict):
            content = result.get("content", "")
            if content:
                result_summary = str(content)[:500]
            elif result.get("tool_calls"):
                result_summary = f"[Tool calls: {len(result['tool_calls'])}]"
        elif result:
            result_summary = str(result)[:500]

        if not result_summary:
            result_summary = "Sub-agent completed"

        # Append return message to parent's buffer
        return_msg: dict = {
            "role": "tool",
            "content": result_summary,
            "tool_call_id": child_agent_id,
            "name": "sub_agent_return",
            "_sub_agent_return": True,
        }

        parent.append_message(return_msg)
        logger.info(
            f"Agent {child_agent_id} returned to parent {parent.agent_id}"
        )


# ---------------------------------------------------------------------------
# Global singleton registry
# ---------------------------------------------------------------------------

registry = AgentRegistry()
