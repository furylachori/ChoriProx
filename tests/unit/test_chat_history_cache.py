"""Integration tests for ChatHistoryCache LRU read-through behavior.
# US-042: ChatHistoryCache LRU Tracking
"""
import os
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_test_db():
    """Use in-memory SQLite so _load_from_sqlite works without a real DB file.

    Also cleans up module-level LRU tracking state to prevent cross-test
    contamination.
    """
    import proxy.state as state

    os.environ["PROXY_TEST_DB"] = ":memory:"
    from memory.db import init_db
    init_db()

    # Clean module-level shared state
    state.chat_histories.clear()
    state._chat_histories_lru.clear()

    yield

    # Teardown: clean up again and restore env
    state.chat_histories.clear()
    state._chat_histories_lru.clear()
    os.environ.pop("PROXY_TEST_DB", None)


# ---------------------------------------------------------------------------
# Read-through (__missing__ → _load_from_sqlite)
# ---------------------------------------------------------------------------

def test_read_through_loads_from_sqlite():
    """Cache miss triggers __missing__ → _load_from_sqlite → populates dict."""
    import proxy.state as state
    from memory.db import store_message, touch_session

    touch_session("sid-r1")
    store_message("sid-r1", 0, "user", "hello")

    with state.chat_histories_lock:
        result = state.chat_histories["sid-r1"]
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_read_through_cache_hit_no_sqlite():
    """Second access hits cache, no SQLite round-trip."""
    import proxy.state as state
    from memory.db import store_message, touch_session

    touch_session("sid-r2")
    store_message("sid-r2", 0, "user", "hello")

    with state.chat_histories_lock:
        _ = state.chat_histories["sid-r2"]  # populates cache via __missing__

    with patch("proxy.state._load_from_sqlite") as mock_load:
        with state.chat_histories_lock:
            _ = state.chat_histories["sid-r2"]
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# LRU eviction + transparent reload
# ---------------------------------------------------------------------------

def test_eviction_reloads_from_sqlite():
    """After LRU eviction, next access transparently reloads from SQLite.

    Uses state.chat_histories (the module-level singleton) because
    _evict_chat_histories_lru() is hardcoded to evict from it.
    """
    import proxy.state as state
    import proxy.chat_history as chat_history_module
    from memory.db import store_message

    old_max = chat_history_module.CHAT_HISTORIES_MAX_ENTRIES
    chat_history_module.CHAT_HISTORIES_MAX_ENTRIES = 2

    try:
        from memory.db import touch_session
        for i in range(3):
            store_message(f"sid-ev-{i}", 0, "user", f"msg-{i}")
            touch_session(f"sid-ev-{i}")

        with state.chat_histories_lock:
            for i in range(3):
                state.chat_histories[f"sid-ev-{i}"] = [
                    {"role": "user", "content": f"msg-{i}"}
                ]

        # Oldest entry should be evicted
        assert "sid-ev-0" not in state.chat_histories
        assert "sid-ev-1" in state.chat_histories
        assert "sid-ev-2" in state.chat_histories

        # Access evicted key → transparent SQLite reload via __missing__
        with state.chat_histories_lock:
            result = state.chat_histories["sid-ev-0"]
        assert len(result) == 1
        assert result[0]["content"] == "msg-0"
    finally:
        chat_history_module.CHAT_HISTORIES_MAX_ENTRIES = old_max


# ---------------------------------------------------------------------------
# pop() LRU cleanup
# ---------------------------------------------------------------------------

def test_pop_cleans_up_lru():
    """pop() removes key from both dict and LRU tracking."""
    import proxy.state as state

    with state.chat_histories_lock:
        state.chat_histories["sid-pop"] = [{"role": "user"}]
        assert "sid-pop" in state._chat_histories_lru

        state.chat_histories.pop("sid-pop")
        assert "sid-pop" not in state.chat_histories
        assert "sid-pop" not in state._chat_histories_lru


# ---------------------------------------------------------------------------
# get() with default
# ---------------------------------------------------------------------------

def test_get_with_default_avoids_keyerror():
    """get() with default returns default on miss, doesn't raise KeyError."""
    import proxy.state as state

    with state.chat_histories_lock:
        result = state.chat_histories.get("nonexistent", "fallback")
    assert result == "fallback"


# ---------------------------------------------------------------------------
# Bare __getitem__ KeyError on total miss
# ---------------------------------------------------------------------------

def test_getitem_raises_keyerror_on_total_miss():
    """Bare __getitem__ raises KeyError when key absent from cache AND DB."""
    import proxy.state as state
    import pytest

    with state.chat_histories_lock:
        with pytest.raises(KeyError):
            _ = state.chat_histories["definitely-does-not-exist"]


# ---------------------------------------------------------------------------
# __contains__ behavior
# ---------------------------------------------------------------------------

def test_contains_on_cached_key():
    """__contains__ returns True for cached keys without SQLite round-trip."""
    import proxy.state as state

    with state.chat_histories_lock:
        state.chat_histories["sid-contains"] = [{"role": "user"}]

    with patch("proxy.state._load_from_sqlite") as mock_load:
        with state.chat_histories_lock:
            result = "sid-contains" in state.chat_histories
        assert result is True
        mock_load.assert_not_called()


def test_contains_on_uncached_db_key():
    """__contains__ returns False for keys in DB but not in cache."""
    import proxy.state as state
    from memory.db import store_message

    store_message("sid-db-only", 0, "user", "hello")

    with state.chat_histories_lock:
        result = "sid-db-only" in state.chat_histories
    assert result is False
