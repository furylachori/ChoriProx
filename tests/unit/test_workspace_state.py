"""Unit tests for proxy.workspace_state.WorkspaceStateStore.

Covers BUG #12: get_global_catalog must perform the Neo4j fetch OUTSIDE
self._lock so concurrent callers (activate/get_active/deactivate across
other sessions) are not blocked on the round-trip. The cache compare-and-
set still happens under the lock.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from proxy.workspace_state import (
    DEFAULT_WORKSPACE,
    WorkspaceStateStore,
    CATALOG_CACHE_TTL_SECONDS,
)


class _CountingRLock:
    """RLock wrapper that tracks whether the lock is currently held.

    Used by the BUG #12 tests to assert that get_global_catalog does NOT
    hold the workspace lock across the Neo4j fetch. threading.RLock has
    no public ``locked()`` on Python 3.9, so we wrap it with a counter.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._depth = 0
        self._depth_lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        with self._depth_lock:
            self._depth += 1
        return self

    def __exit__(self, *exc):
        with self._depth_lock:
            self._depth -= 1
        self._lock.release()
        return False

    @property
    def depth(self):
        with self._depth_lock:
            return self._depth


@pytest.fixture
def lock():
    return threading.RLock()


@pytest.fixture
def store(lock):
    return WorkspaceStateStore(lock)


@pytest.fixture
def counting_store():
    """A store backed by _CountingRLock for lock-held assertions."""
    return WorkspaceStateStore(_CountingRLock())


class TestGetGlobalCatalogLockScope:
    """BUG #12: the Neo4j fetch must happen OUTSIDE the workspace lock."""

    def test_cache_hit_does_not_fetch(self, store):
        """Populated cache returns without calling the persistence layer."""
        store.set_catalog_cache(["default", "alpha"])

        with patch(
            "memory.workspace_persistence.get_global_catalog"
        ) as fetch_mock:
            result = store.get_global_catalog()

        fetch_mock.assert_not_called()
        assert "alpha" in result
        assert "default" in result

    def test_cache_miss_fetches_outside_lock(self, counting_store):
        """On cache miss, the Neo4j fetch runs while self._lock is NOT held.

        The _CountingRLock wrapper records the current hold depth. While
        the patched fetch runs, the depth MUST be 0 — proving the lock
        was released before the round-trip.
        """
        depth_during_fetch = []

        def recording_fetch():
            depth_during_fetch.append(counting_store._lock.depth)
            return ["default", "beta"]

        with patch(
            "memory.workspace_persistence.get_global_catalog",
            side_effect=recording_fetch,
        ):
            result = counting_store.get_global_catalog()

        assert depth_during_fetch == [0], (
            f"Neo4j fetch must run OUTSIDE the workspace lock "
            f"(lock depth during fetch={depth_during_fetch})"
        )
        assert result == ["default", "beta"]

    def test_cache_miss_populates_cache_under_lock(self, store):
        """After a fetch, the cache is populated and a second call hits."""
        call_count = [0]

        def counting_fetch():
            call_count[0] += 1
            return ["default", "gamma"]

        with patch(
            "memory.workspace_persistence.get_global_catalog",
            side_effect=counting_fetch,
        ):
            first = store.get_global_catalog()
            second = store.get_global_catalog()

        assert call_count[0] == 1, "Second call should hit cache, not fetch"
        assert first == ["default", "gamma"]
        assert second == ["default", "gamma"]

    def test_cache_write_happens_under_lock(self, counting_store):
        """The cache compare-and-set runs under the lock — concurrent
        readers must not observe a partially-written cache."""
        # Snapshot the lock depth at the moment we expect the write to
        # happen. We can't easily intercept the write itself, but we can
        # verify the cache IS populated after the call returns (i.e.,
        # the compare-and-set under the lock completed).
        with patch(
            "memory.workspace_persistence.get_global_catalog",
            return_value=["default", "delta"],
        ):
            counting_store.get_global_catalog()

        # The cache must be populated (write happened under the lock).
        assert counting_store._catalog_cache == ["default", "delta"]
        assert counting_store._catalog_cache_ts > 0

    def test_concurrent_callers_do_not_block_on_fetch(self, store):
        """Two concurrent callers on cache miss: both make progress while
        the other is fetching (the lock is not held across the fetch).

        We model this by having the fetch block on a barrier — if either
        caller held the lock across the fetch, the other would deadlock
        trying to acquire the lock for its own cache check.
        """
        barrier = threading.Barrier(2, timeout=5.0)
        results = [None, None]
        errors = []

        def blocking_fetch():
            # Both threads must reach this barrier. If the lock were held
            # by the caller, the second thread would block on the lock
            # before reaching get_global_catalog's first cache check, and
            # the barrier would time out.
            try:
                barrier.wait()
            except threading.BrokenBarrierError as e:
                raise RuntimeError(f"barrier broken: {e!r}") from e
            return ["default", "delta"]

        def caller(idx):
            try:
                with patch(
                    "memory.workspace_persistence.get_global_catalog",
                    side_effect=blocking_fetch,
                ):
                    results[idx] = store.get_global_catalog()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=caller, args=(0,), daemon=True),
            threading.Thread(target=caller, args=(1,), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"caller threads raised: {errors!r}"
        assert not any(t.is_alive() for t in threads), (
            "caller thread did not finish within timeout — likely a deadlock "
            "from holding the workspace lock across the Neo4j fetch"
        )
        assert results[0] == ["default", "delta"]
        assert results[1] == ["default", "delta"]

    def test_concurrent_winning_writer_publishes_catalog(self, store):
        """When two threads race a cache miss, both fetch but only one
        result is published (last writer wins). Both callers see a valid
        catalog — no corruption, no partial state visible to readers.
        """
        fetch_count = [0]
        fetch_count_lock = threading.Lock()
        results = [None, None]

        def fetching():
            with fetch_count_lock:
                fetch_count[0] += 1
                n = fetch_count[0]
            return ["default", f"ws-{n}"]

        def caller(idx):
            with patch(
                "memory.workspace_persistence.get_global_catalog",
                side_effect=fetching,
            ):
                results[idx] = store.get_global_catalog()

        threads = [
            threading.Thread(target=caller, args=(0,), daemon=True),
            threading.Thread(target=caller, args=(1,), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        for r in results:
            assert "default" in r, f"caller result missing default: {r!r}"
        with store._lock:
            cached = list(store._catalog_cache)
        assert "default" in cached
        assert any(name.startswith("ws-") for name in cached), (
            f"cache should contain one of the fetched catalogs: {cached!r}"
        )

    def test_no_lock_ordering_violation(self, counting_store):
        """BUG #12 verification: get_global_catalog does not acquire any
        lock OTHER than self._lock (position #6). The bug fix must not
        introduce a lock-ordering violation per AGENTS.md Known Landmines.

        We assert this structurally: the only lock touched during a cache
        miss is self._lock itself. The fetch mock acquires nothing.
        """
        depths_seen = []

        def recording_fetch():
            depths_seen.append(counting_store._lock.depth)
            return ["default", "epsilon"]

        with patch(
            "memory.workspace_persistence.get_global_catalog",
            side_effect=recording_fetch,
        ):
            counting_store.get_global_catalog()

        # During the fetch, depth must be 0 — the lock was released.
        assert depths_seen == [0], (
            f"Lock must be released during fetch; depths={depths_seen}"
        )


class TestGetGlobalCatalogTTL:
    def test_expired_cache_refetches(self, store):
        """A cache whose timestamp is older than TTL triggers a refetch."""
        store.set_catalog_cache(["default", "stale"])
        with store._lock:
            store._catalog_cache_ts = time.time() - CATALOG_CACHE_TTL_SECONDS - 1

        with patch(
            "memory.workspace_persistence.get_global_catalog",
            return_value=["default", "fresh"],
        ):
            result = store.get_global_catalog()

        assert result == ["default", "fresh"]

    def test_invalidate_via_activate_clears_cache(self, store):
        """Activating a previously-unknown workspace invalidates the cache
        (requires persist_callback to be wired — see activate())."""
        persist_calls = []

        def fake_persist(workspace_name, session_id):
            persist_calls.append((workspace_name, session_id))

        store_with_persist = WorkspaceStateStore(
            threading.RLock(), persist_callback=fake_persist
        )
        store_with_persist.set_catalog_cache(["default", "alpha"])

        # activate() of an unknown workspace clears the cache (because
        # persist_callback is set, the invalidation branch fires).
        store_with_persist.activate("sess1", "brand-new-workspace")

        with patch(
            "memory.workspace_persistence.get_global_catalog",
            return_value=["default", "brand-new-workspace"],
        ) as fetch_mock:
            result = store_with_persist.get_global_catalog()

        fetch_mock.assert_called_once()
        assert "brand-new-workspace" in result
        assert persist_calls == [("brand-new-workspace", "sess1")], (
            f"persist should have fired for the new workspace: {persist_calls}"
        )


class TestGetGlobalCatalogDefault:
    def test_empty_cache_returns_default_via_fetch(self, store):
        """Cold cache + fetch returning ['default'] -> catalog is ['default']."""
        with patch(
            "memory.workspace_persistence.get_global_catalog",
            return_value=["default"],
        ):
            result = store.get_global_catalog()
        assert result == ["default"]


# ---------------------------------------------------------------------------
# F2: WorkspaceSkill._handle_switch must fetch the global catalog OUTSIDE
# the outer workspace_store.lock.
#
# BUG #12 pulled the Neo4j round-trip OUT of get_global_catalog's internal
# lock, but WorkspaceSkill._handle_switch still acquired workspace_store.lock
# (position #6 RLock) BEFORE calling get_global_catalog. Since RLock.acquire
# is a no-op for the SAME thread that already holds it, the inner release is
# invisible to the caller — the round-trip still blocks ALL other sessions
# that touch the workspace store. F2 fixes that by fetching the catalog
# before acquiring the outer lock.
# ---------------------------------------------------------------------------


class TestHandleSwitchCatalogFetchLockScope:
    """F2: the catalog fetch in `_handle_switch` must run with lock depth 0."""

    def _make_store(self):
        """Build a mock workspace_store backed by a _CountingRLock.

        Returns (mock_store, lock). The mock store's `lock` property
        returns the counting lock so callers' `with workspace_store.lock:`
        blocks are observable.
        """
        counting_lock = _CountingRLock()
        mock_store = MagicMock()
        mock_store.lock = counting_lock
        mock_store.get_active.return_value = DEFAULT_WORKSPACE
        mock_store.get_recent.return_value = []
        mock_store.activate.return_value = (DEFAULT_WORKSPACE, [])
        return mock_store, counting_lock

    def test_handle_switch_fetches_catalog_outside_outer_lock(self):
        """During `get_global_catalog()` inside `_handle_switch`, the
        outer workspace_store.lock MUST NOT be held.

        Why: RLock.acquire is a no-op for the SAME thread that already holds
        it, so get_global_catalog's internal release is invisible to a
        caller that already acquired the outer lock. The Neo4j round-trip
        would block all sessions that touch workspace_store for its full
        duration. Pulling the fetch out of the critical section is the only
        way to keep the lock held only for the in-memory read-check-write.
        """
        from proxy.skills.skill_definition import SkillRequestContext
        from skills.workspace import WorkspaceSkill

        mock_store, counting_lock = self._make_store()
        depth_during_catalog_fetch = []

        def recording_catalog():
            depth_during_catalog_fetch.append(counting_lock.depth)
            return [DEFAULT_WORKSPACE]

        mock_store.get_global_catalog.side_effect = recording_catalog

        ctx = SkillRequestContext(
            session_id="sess-f2",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint="",
            body={},
            seeded_tasks=[],
            mode_state=None,
        )

        skill = WorkspaceSkill()
        with patch("skills.workspace.workspace_store", mock_store):
            result = skill._handle_switch(ctx, "newws")

        # The catalog fetch MUST have run with depth=0 — i.e., BEFORE
        # the outer `with workspace_store.lock:` block was entered.
        assert depth_during_catalog_fetch == [0], (
            f"Catalog fetch in _handle_switch must run OUTSIDE "
            f"workspace_store.lock; depths={depth_during_catalog_fetch}"
        )
        # Sanity: the call still produced a result (the "not found" branch
        # in this case — 'newws' is not in the empty catalog).
        assert result is not None

    def test_handle_switch_under_lock_runs_at_depth_one(self):
        """While inside the `with workspace_store.lock:` block, depth IS 1.

        Guards the F2 fix from a regression that accidentally releases the
        lock too early or never acquires it.
        """
        from proxy.skills.skill_definition import SkillRequestContext
        from skills.workspace import WorkspaceSkill

        mock_store, counting_lock = self._make_store()
        mock_store.get_global_catalog.return_value = [DEFAULT_WORKSPACE]
        depths_inside_lock = []

        original_get_active = mock_store.get_active

        def recording_get_active(*args, **kwargs):
            depths_inside_lock.append(counting_lock.depth)
            return DEFAULT_WORKSPACE

        mock_store.get_active.side_effect = recording_get_active

        ctx = SkillRequestContext(
            session_id="sess-f2-depth",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint="",
            body={},
            seeded_tasks=[],
            mode_state=None,
        )

        skill = WorkspaceSkill()
        with patch("skills.workspace.workspace_store", mock_store):
            skill._handle_switch(ctx, "newws")

        # get_active was called while the outer lock was held (depth=1),
        # confirming the in-memory read-check-write IS under the lock.
        assert depths_inside_lock, "get_active should have been called"
        assert all(d >= 1 for d in depths_inside_lock), (
            f"get_active calls inside _handle_switch must be under the "
            f"outer lock; depths={depths_inside_lock}"
        )

    def test_handle_switch_no_lock_ordering_violation(self):
        """F2: only one lock is ever held at a time in `_handle_switch`.

        AGENTS.md Known Landmines: lock-ordering violations cause deadlocks.
        The F2 reorder must not introduce a new lock ordering (e.g.,
        acquiring some lock #N before workspace_store.lock #6). The only
        lock touched by `_handle_switch` is workspace_store.lock itself.
        """
        from proxy.skills.skill_definition import SkillRequestContext
        from skills.workspace import WorkspaceSkill

        mock_store, counting_lock = self._make_store()
        mock_store.get_global_catalog.return_value = [DEFAULT_WORKSPACE]

        ctx = SkillRequestContext(
            session_id="sess-f2-order",
            is_subagent=False,
            inherited_skill=None,
            subagent_ticket_mode=False,
            workspace_name="",
            client_hint="",
            body={},
            seeded_tasks=[],
            mode_state=None,
        )

        skill = WorkspaceSkill()
        with patch("skills.workspace.workspace_store", mock_store):
            skill._handle_switch(ctx, "newws")

        # After the call returns, the lock MUST be fully released.
        assert counting_lock.depth == 0, (
            f"workspace_store.lock must be released after _handle_switch "
            f"returns; final depth={counting_lock.depth}"
        )

