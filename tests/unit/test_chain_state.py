"""Unit tests for V2 chain-hash shared state (state._chain_state LRU).

Exercises the canonical DIRECT-ACCESS pattern documented in state.py (used by
context_v2.py). There are no get/set wrapper accessors — callers hold
_chain_state_lock around read+move_to_end (snapshot to locals) and around the
CAS check + write + LRU eviction.
"""
import pytest

pytestmark = pytest.mark.unit


def _get(state, key):
    """Mirror context_v2.py read path: bump to MRU under lock, return entry."""
    with state._chain_state_lock:
        entry = state._chain_state.get(key)
        if entry is not None:
            state._chain_state.move_to_end(key)
        return entry


def _set(state, key, value):
    """Mirror context_v2.py write path: store, bump to MRU, evict LRU over cap."""
    with state._chain_state_lock:
        state._chain_state[key] = value
        state._chain_state.move_to_end(key)
        while len(state._chain_state) > state.CHAIN_STATE_MAX_ENTRIES:
            state._chain_state.popitem(last=False)


@pytest.fixture(autouse=True)
def _reset_chain_state():
    import proxy.state as state
    state._chain_state.clear()
    yield
    state._chain_state.clear()


def test_empty_by_default():
    import proxy.state as state
    assert len(state._chain_state) == 0
    assert _get(state, ("s1", "m1")) is None


def test_new_chain_state_entry_has_canonical_keys():
    import proxy.state as state
    entry = state.new_chain_state_entry()
    assert set(entry.keys()) == {
        "last_chain",
        "boundary_idx",
        "boundary_chain",
        "summaries",
        "ancient",
        "version",
    }


def test_new_chain_state_entry_ancient_is_string():
    """MEDIUM 3: ancient defaults to '' (string), not [] (list).

    All consumers expect a string; the prior [] default only avoided a crash
    because empty list is falsy and the `if ancient:` guard skipped rendering.
    """
    import proxy.state as state
    entry = state.new_chain_state_entry()
    assert entry["ancient"] == ""
    assert isinstance(entry["ancient"], str)


def test_set_and_get_roundtrip():
    import proxy.state as state
    entry = state.new_chain_state_entry()
    entry["last_chain"] = "abc123"
    _set(state, ("s1", "m1"), entry)
    got = _get(state, ("s1", "m1"))
    assert got is entry
    assert got["last_chain"] == "abc123"


def test_lru_eviction_at_cap():
    import proxy.state as state
    cap = state.CHAIN_STATE_MAX_ENTRIES
    for i in range(cap):
        _set(state, (f"s{i}", "m"), state.new_chain_state_entry())
    assert len(state._chain_state) == cap
    _set(state, ("sNEW", "m"), state.new_chain_state_entry())
    assert len(state._chain_state) == cap
    assert ("s0", "m") not in state._chain_state
    assert ("sNEW", "m") in state._chain_state


def test_read_bumps_to_mru():
    import proxy.state as state
    cap = state.CHAIN_STATE_MAX_ENTRIES
    for i in range(cap):
        _set(state, (f"s{i}", "m"), state.new_chain_state_entry())
    assert _get(state, ("s0", "m")) is not None
    _set(state, ("sNEW", "m"), state.new_chain_state_entry())
    assert ("s0", "m") in state._chain_state
    assert ("s1", "m") not in state._chain_state


def test_write_existing_bumps_to_mru():
    import proxy.state as state
    cap = state.CHAIN_STATE_MAX_ENTRIES
    for i in range(cap):
        _set(state, (f"s{i}", "m"), state.new_chain_state_entry())
    _set(state, ("s0", "m"), state.new_chain_state_entry())
    _set(state, ("sNEW", "m"), state.new_chain_state_entry())
    assert ("s0", "m") in state._chain_state
    assert ("s1", "m") not in state._chain_state


def test_concurrent_access_no_crash():
    import proxy.state as state
    from tests.concurrency_harness import run_concurrently

    def writer(n):
        for i in range(n):
            _set(state, (f"s{i % 10}", "m"), state.new_chain_state_entry())

    def reader(n):
        for i in range(n):
            _get(state, (f"s{i % 10}", "m"))

    run_concurrently(
        [
            lambda: writer(200),
            lambda: writer(200),
            lambda: reader(200),
            lambda: reader(200),
        ],
        timeout=15,
    )
    assert len(state._chain_state) <= state.CHAIN_STATE_MAX_ENTRIES
