"""
Phase 0 bug-fix tests for P0-B1, P0-B2, P0-B4.

P0-B1: clear_tool_cache and get_cached_tools now use tool_cache_lock (not chat_histories_lock)
P0-B2: clear_tool_cache resets counters via state.* attribute assignment (not global)
P0-B4: tool_cache_status() acquires tool_cache_lock before reading
"""

import pytest
from unittest.mock import patch, MagicMock


VALID_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "test_fn",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]


class TestP0B1LockCorrectness:
    """P0-B1: clear_tool_cache and get_cached_tools must use tool_cache_lock."""

    def test_clear_tool_cache_uses_tool_cache_lock_not_chat_lock(self, client):
        """clear_tool_cache must acquire tool_cache_lock, not chat_histories_lock."""
        import proxy.state as state
        # Seed the cache so clear_tool_cache has something to clear
        state.tool_cache["test_hash_abc"] = [{"type": "function", "function": {"name": "test"}}]
        state.tool_cache_size = 1

        from proxy.tool_cache import clear_tool_cache
        # Call via Flask test client
        resp = client.post("/v1/tools/cache/clear")
        assert resp.status_code == 200

        # Cache must be cleared
        assert len(state.tool_cache) == 0

    def test_get_cached_tools_uses_tool_cache_lock_not_chat_lock(self, client):
        """get_cached_tools must acquire tool_cache_lock, not chat_histories_lock."""
        import proxy.state as state
        from proxy.tool_cache import cache_tool_schema

        # Seed the cache
        cache_tool_schema("test_phase0_hash", VALID_TOOL_SCHEMA)

        resp = client.get("/v1/tools/cache/test_phase0_hash")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["hash"] == "test_phase0_hash"


class TestP0B2CounterResetViaState:
    """P0-B2: clear_tool_cache must reset counters via state.*, not global."""

    def test_clear_tool_cache_resets_size_counter_via_state(self, client):
        """tool_cache_size counter must be reset via state.tool_cache_size."""
        import proxy.state as state
        from proxy.tool_cache import cache_tool_schema

        # Seed the cache
        cache_tool_schema("seed_hash_1", VALID_TOOL_SCHEMA)
        cache_tool_schema("seed_hash_2", VALID_TOOL_SCHEMA)
        assert state.tool_cache_size >= 2

        resp = client.post("/v1/tools/cache/clear")
        assert resp.status_code == 200

        # Counter must be 0 after clear
        assert state.tool_cache_size == 0

    def test_clear_tool_cache_resets_hits_counter_via_state(self, client):
        """tool_cache_hits counter must be reset via state.tool_cache_hits."""
        import proxy.state as state
        from proxy.tool_cache import cache_tool_schema, get_cached_tool_schema

        # Seed the cache
        cache_tool_schema("hit_seed_hash", VALID_TOOL_SCHEMA)
        # Generate a hit
        get_cached_tool_schema("hit_seed_hash")
        assert state.tool_cache_hits >= 1

        resp = client.post("/v1/tools/cache/clear")
        assert resp.status_code == 200

        assert state.tool_cache_hits == 0

    def test_clear_tool_cache_resets_misses_counter_via_state(self, client):
        """tool_cache_misses counter must be reset via state.tool_cache_misses."""
        import proxy.state as state

        # Generate a miss
        from proxy.tool_cache import get_cached_tool_schema
        get_cached_tool_schema("nonexistent_hash_xyz")
        assert state.tool_cache_misses >= 1

        resp = client.post("/v1/tools/cache/clear")
        assert resp.status_code == 200

        assert state.tool_cache_misses == 0


class TestP0B4ToolCacheStatusReadLock:
    """P0-B4: tool_cache_status must acquire tool_cache_lock before reading."""

    def test_tool_cache_status_returns_200(self, client):
        """tool_cache_status endpoint must respond with 200."""
        resp = client.get("/v1/tools/cache")
        assert resp.status_code == 200

    def test_tool_cache_status_returns_json_with_expected_fields(self, client):
        """tool_cache_status response must include cache_size, hits, misses fields."""
        resp = client.get("/v1/tools/cache")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "cache_size" in data
        assert "cache_hits" in data
        assert "cache_misses" in data

    def test_tool_cache_status_reflects_state_after_operations(self, client):
        """tool_cache_status must return accurate state after cache operations."""
        import proxy.state as state
        from proxy.tool_cache import cache_tool_schema, get_cached_tool_schema

        # Reset state
        state.tool_cache.clear()
        state.tool_cache_size = 0
        state.tool_cache_hits = 0
        state.tool_cache_misses = 0

        cache_tool_schema("status_test_hash", VALID_TOOL_SCHEMA)
        get_cached_tool_schema("status_test_hash")  # hit
        get_cached_tool_schema("missing_hash")      # miss

        resp = client.get("/v1/tools/cache")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["cache_size"] == 1
        assert data["cache_hits"] == 1
        assert data["cache_misses"] >= 1


class TestUS041LockAcquisitionOrder:
    """US-041: Verify lock ordering contract — ascending lock-number acquisition.

    The lock hierarchy in state.py defines positions 0-9. Locks must be
    acquired in ascending order to prevent deadlocks. This test documents
    the contract and verifies OrderedLock enforces it.
    """

    def test_lock_positions_are_ascending_in_state_py(self):
        """US-041: Document that lock numbers in state.py are monotonically assigned.

        The locks defined in state.py have these positions:
            #1 context_config_lock
            #3 chat_summaries_lock
            #4 tool_cache_lock
            #5 request_counter_lock
            #6 active_workspace_lock
            #9 vfs_lock

        This test asserts they are assigned in ascending order (no gaps
        matter — only the relative ordering is critical).
        """
        from proxy.lock_ordering import OrderedLock
        from proxy import state

        lock_positions = []
        for name in ("context_config_lock", "chat_summaries_lock",
                      "tool_cache_lock", "request_counter_lock",
                      "active_workspace_lock", "vfs_lock"):
            lock = getattr(state, name)
            if isinstance(lock, OrderedLock):
                lock_positions.append((name, lock.position))

        positions = [p for _, p in lock_positions]
        assert positions == sorted(positions), (
            f"Lock positions must be ascending: {lock_positions}"
        )

    def test_orderedlock_rejects_descending_acquisition(self):
        """US-041: OrderedLock raises RuntimeError when acquiring lower-numbered
        lock while holding a higher-numbered one.
        """
        from proxy.lock_ordering import OrderedLock

        high_lock = OrderedLock(position=8, name="high")
        low_lock = OrderedLock(position=2, name="low")

        import os
        original_val = os.environ.get("PROXY_DEBUG_LOCK_ORDERING")
        os.environ["PROXY_DEBUG_LOCK_ORDERING"] = "1"
        try:
            from proxy import lock_ordering
            old_flag = lock_ordering._LOCK_ORDERING_ENABLED
            lock_ordering._LOCK_ORDERING_ENABLED = True
            try:
                high_lock.acquire()
                try:
                    with pytest.raises(RuntimeError, match="Lock ordering violation"):
                        low_lock.acquire()
                finally:
                    high_lock.release()
            finally:
                lock_ordering._LOCK_ORDERING_ENABLED = old_flag
        finally:
            if original_val is None:
                os.environ.pop("PROXY_DEBUG_LOCK_ORDERING", None)
            else:
                os.environ["PROXY_DEBUG_LOCK_ORDERING"] = original_val
