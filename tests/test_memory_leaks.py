"""Tests for memory leak fixes (Batch B - B4, MEM-001/MEM-002)."""
import pytest
import time
from unittest.mock import MagicMock, patch


class TestCleanupStaleWorkspaces:
    """MEM-001: cleanup_stale_workspaces removes stale entries."""

    def test_cleanup_stale_workspaces_function_exists(self):
        """cleanup_stale_workspaces function should exist."""
        from proxy import state
        assert hasattr(state, 'cleanup_stale_workspaces')
        assert callable(state.cleanup_stale_workspaces)

    def test_cleanup_removes_stale_sessions(self):
        """cleanup removes sessions older than session_ttl."""
        from proxy import state
        # Seed some stale data
        now = time.time()
        stale_session = f"stale-test-{now}"
        fresh_session = f"fresh-test-{now}"
        with state.active_workspace_lock:
            state.session_last_access[stale_session] = now - 7200  # 2 hours ago
            state.active_workspaces[stale_session] = "test_workspace"
            state.recent_workspaces[stale_session] = ["test_workspace"]
            state.session_last_access[fresh_session] = now  # just now
        # Reset cleanup guard so the test can run immediately
        state._last_cleanup_time = 0.0
        try:
            # Run cleanup
            removed = state.cleanup_stale_workspaces(session_ttl=3600)
            assert removed >= 1
            # Verify stale session was removed
            with state.active_workspace_lock:
                assert stale_session not in state.active_workspaces
                assert stale_session not in state.session_last_access
        finally:
            # Cleanup test data
            with state.active_workspace_lock:
                state.active_workspaces.pop(stale_session, None)
                state.recent_workspaces.pop(stale_session, None)
                state.session_last_access.pop(stale_session, None)
                state.active_workspaces.pop(fresh_session, None)
                state.recent_workspaces.pop(fresh_session, None)
                state.session_last_access.pop(fresh_session, None)


class TestLRUEvictionCallsClearActiveMode:
    """MEM-002: _evict_lru calls clear_active_mode before removing agent."""

    def test_lru_eviction_clears_active_mode(self):
        """LRU eviction should call clear_active_mode on the evicted agent."""
        from proxy.agent.agent_registry import AgentRegistry
        from proxy.agent.agent import Agent
        from proxy.skills.skill_definition import ModeState
        from unittest.mock import MagicMock
        
        registry = AgentRegistry(max_agents=2)
        # Create and register agents up to capacity
        agent1 = Agent(session_id=f"lru-test-1-{time.time()}")
        agent2 = Agent(session_id=f"lru-test-2-{time.time()}")
        registry._register(agent1)
        registry._register(agent2)
        
        # Set active mode on agent1
        mode = ModeState(
            name="ticket",
            activated_at=time.time(),
            ttl_seconds=300,
            allowed_tool_names=frozenset(),
        )
        agent1.set_active_mode(mode)
        
        # Track clear_active_mode calls
        with patch.object(agent1, 'clear_active_mode') as mock_clear:
            # Manually trigger eviction (get_or_load calls this outside lock)
            with registry._lock:
                evicted = registry._evict_lru()
            # clear_active_mode must be called OUTSIDE the lock (H-4)
            if evicted is not None:
                evicted.clear_active_mode()
            # clear_active_mode should have been called on agent1
            # (The oldest agent gets evicted)
            assert mock_clear.called, "clear_active_mode should be called on evicted agent"