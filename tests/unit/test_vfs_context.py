"""Unit tests for VFS context injection functions."""

import logging
import time
import hashlib
import pytest
from unittest.mock import patch
import proxy.state as state
from proxy.vfs import (
    VFSEntry, VFSCache, VFSWriteRecord,
    _build_vfs_context, _build_vfs_inherited_context,
    _cache_file, _cache_file_write_through,
    _get_or_create_workspace_cache,
)


class MockCtx:
    """Mock request context for testing."""
    def __init__(self, session_id="test_session", is_subagent=False, parent_agent_id=None):
        self.session_id = session_id
        self.is_subagent = is_subagent
        self.parent_agent_id = parent_agent_id


@pytest.fixture(autouse=True)
def clean_vfs_state():
    with state.vfs_lock:
        state.vfs_caches.clear()
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()
    yield
    with state.vfs_lock:
        state.vfs_caches.clear()
        state.vfs_write_log.clear()
        state.vfs_rollback_snapshots.clear()


@pytest.fixture
def vfs_enabled(monkeypatch):
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)


def _populate_cache(workspace_id, files):
    """Helper to populate cache with test files."""
    with state.vfs_lock:
        cache = _get_or_create_workspace_cache(workspace_id)
    for path, content in files.items():
        _cache_file(path, content, workspace_id, "test_session")


class TestBuildVfsContext:
    """US-058: VFS Inherited Context for Sub-Agents — _build_vfs_context coverage.

    Tests the main-agent path that formats cached entries into a system
    message for the current request.
    """
    def test_returns_none_when_disabled(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", False)
        ctx = MockCtx()
        result = _build_vfs_context("test", ctx, 1000, 0.25)
        assert result is None

    @pytest.mark.parametrize(
        ("scope", "enabled", "expects_context"),
        [
            pytest.param("all", True, True, id="all-enabled"),
            pytest.param("main", True, True, id="main-enabled"),
            pytest.param("subs", True, False, id="subs-suppresses-main"),
            pytest.param("all", False, False, id="disabled"),
        ],
    )
    def test_prompt_emission_scope_matrix(
        self,
        vfs_enabled,
        monkeypatch,
        scope,
        enabled,
        expects_context,
    ):
        """US-057: Main prompt emission honors scope at render time."""
        _populate_cache("default", {"src/app.py": "content"})
        monkeypatch.setattr(state, "VFS_SCOPE", scope)
        monkeypatch.setattr(state, "VFS_ENABLED", enabled)
        ctx = MockCtx(session_id="test_session", is_subagent=False)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ) as resolve_workspace:
            result = _build_vfs_context(
                "test_session",
                ctx,
                1000,
                0.25,
            )

        assert (result is not None) is expects_context

        if expects_context:
            assert result["role"] == "system"
            assert "[VFS CONTEXT" in result["content"]
            assert "src/app.py" in result["content"]
            resolve_workspace.assert_called_once_with(
                "test_session",
                client_hint="",
            )
        else:
            resolve_workspace.assert_not_called()

    def test_scope_change_suppresses_already_populated_main_cache(
        self,
        vfs_enabled,
        monkeypatch,
    ):
        """US-057: scope=subs suppresses warm main cache at emission."""
        _populate_cache("default", {"src/app.py": "content"})
        ctx = MockCtx(session_id="test_session", is_subagent=False)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ) as resolve_workspace:
            before = _build_vfs_context(
                "test_session",
                ctx,
                1000,
                0.25,
            )

            monkeypatch.setattr(state, "VFS_SCOPE", "subs")

            after = _build_vfs_context(
                "test_session",
                ctx,
                1000,
                0.25,
            )

        assert before is not None
        assert "[VFS CONTEXT" in before["content"]
        assert after is None

        resolve_workspace.assert_called_once_with(
            "test_session",
            client_hint="",
        )

        with state.vfs_lock:
            assert state.vfs_caches["default"].size == 1

    def test_returns_none_for_subagent_with_populated_cache(
        self,
        vfs_enabled,
    ):
        """Main builder rejects sub-agent ctx independently of scope."""
        _populate_cache("default", {"src/app.py": "content"})
        ctx = MockCtx(session_id="test_session", is_subagent=True)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ) as resolve_workspace:
            result = _build_vfs_context(
                "test_session",
                ctx,
                5000,
                0.25,
            )

        assert result is None
        resolve_workspace.assert_not_called()

    def test_returns_none_when_empty(self, vfs_enabled):
        ctx = MockCtx()
        result = _build_vfs_context("test", ctx, 1000, 0.25)
        assert result is None

    def test_returns_system_message(self, vfs_enabled):
        ws = "default"
        _populate_cache(ws, {"src/app.py": "content"})
        ctx = MockCtx(session_id="test_session")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_context("test_session", ctx, 1000, 0.25)
        assert result is not None
        assert result["role"] == "system"
        assert "[VFS CONTEXT" in result["content"]
        assert "[END VFS CONTEXT]" in result["content"]
        assert "src/app.py" in result["content"]

    def test_lists_multiple_files(self, vfs_enabled):
        ws = "default"
        _populate_cache(ws, {
            "src/app.py": "app",
            "src/state.py": "state",
            "tests/test_app.py": "test",
        })
        ctx = MockCtx(session_id="test_session")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_context("test_session", ctx, 5000, 0.25)
        assert "src/app.py" in result["content"]
        assert "src/state.py" in result["content"]
        assert "tests/test_app.py" in result["content"]

    def test_shows_hash_prefix(self, vfs_enabled):
        ws = "default"
        _populate_cache(ws, {"src/app.py": "content"})
        ctx = MockCtx(session_id="test_session")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_context("test_session", ctx, 1000, 0.25)
        # Should contain 8-char hash prefix
        assert "hash:" in result["content"]

    # ------------------------------------------------------------------
    # Wave-3 strict budget tests (Caches-VFS spec / US-127)
    # ------------------------------------------------------------------

    def test_full_rendered_block_fits_with_all_sorted_rows(self, vfs_enabled):
        """Wave 3: all sorted rows render when budget accommodates."""
        _populate_cache("default", {
            "src/zeta.py": "z",
            "src/alpha.py": "a",
            "src/middle.py": "m",
        })
        ctx = MockCtx(session_id="test_session", is_subagent=False)
        budget = 10_000
        tpc = 0.25

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context("test_session", ctx, budget, tpc)

        assert result is not None
        assert result["role"] == "system"
        content = result["content"]
        assert content.count("[VFS CONTEXT") == 1
        assert content.count("[END VFS CONTEXT]") == 1
        assert "src/alpha.py" in content
        assert "src/middle.py" in content
        assert "src/zeta.py" in content
        assert "alpha.py" in content
        assert "middle.py" in content
        assert "zeta.py" in content
        pos_alpha = content.find("alpha.py")
        pos_middle = content.find("middle.py")
        pos_zeta = content.find("zeta.py")
        assert pos_alpha < pos_middle < pos_zeta, (
            f"Paths must be in alphabetical order; "
            f"got alpha@{pos_alpha}, middle@{pos_middle}, zeta@{pos_zeta}"
        )
        assert "more files" not in content
        assert int(len(content) * tpc) <= budget

    def test_smallest_valid_block_fits_at_exact_integer_cost(self, vfs_enabled):
        """Wave 3: exact-integer-cost budget still admits the smallest block."""
        _populate_cache("default", {"src/app.py": "x"})
        ctx = MockCtx(session_id="test_session", is_subagent=False)
        tpc = 1

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context("test_session", ctx, 10_000, tpc)

        assert result is not None
        assert result["role"] == "system"
        content = result["content"]
        row_lines = [
            line for line in content.split("\n")
            if line.startswith("  ") and "src/app.py" in line
        ]
        assert len(row_lines) == 1
        actual_row = row_lines[0]
        expected_inner = "\n".join([
            "Workspace: default | Cached files: 1",
            "",
            "Cached files (available for instant reads):",
            actual_row,
        ])
        expected_content = (
            "[VFS CONTEXT — Cached file state]\n"
            + expected_inner
            + "\n[END VFS CONTEXT]"
        )
        exact_budget = int(len(expected_content) * tpc)
        assert exact_budget == len(expected_content)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context(
                "test_session", ctx, exact_budget, tpc,
            )

        assert result is not None
        assert result["content"] == expected_content
        assert int(len(result["content"]) * tpc) == exact_budget
        assert "more files" not in result["content"]

    def test_returns_none_when_full_rendered_block_exceeds_budget(self, vfs_enabled):
        """Wave 3: builder returns None when full wrapped estimate exceeds budget.

        Under the OLD inner-only estimate, this same budget would have admitted
        the wrapped block whose total length exceeds the budget. The strict
        complete-rendered estimate rejects it.
        """
        _populate_cache("default", {"src/app.py": "x"})
        ctx = MockCtx(session_id="test_session", is_subagent=False)
        tpc = 1

        with state.vfs_lock:
            cache = state.vfs_caches["default"]
            entry = next(iter(cache.snapshot_entries()))
        normalized_path = entry.path
        hash_prefix = entry.content_hash[:8]

        inner_text = "\n".join([
            "Workspace: default | Cached files: 1",
            "",
            "Cached files (available for instant reads):",
            f"  {normalized_path} (hash:{hash_prefix}, {entry.size_bytes}B)",
        ])
        full_content = (
            "[VFS CONTEXT — Cached file state]\n"
            + inner_text
            + "\n[END VFS CONTEXT]"
        )
        inner_cost = int(len(inner_text) * tpc)
        full_cost = int(len(full_content) * tpc)
        budget = full_cost - 1

        assert inner_cost <= budget < full_cost, (
            f"Fixture must satisfy inner_cost={inner_cost} <= "
            f"budget={budget} < full_cost={full_cost}"
        )

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context("test_session", ctx, budget, tpc)

        assert result is None

    def test_selects_largest_sorted_prefix_whose_full_content_fits(self, vfs_enabled):
        """Wave 3: picks the largest prefix whose complete content fits."""
        files = {
            "src/zeta.py": "z" * 50,
            "src/alpha.py": "a" * 50,
            "src/middle.py": "m" * 50,
            "src/beta.py": "b" * 50,
        }
        _populate_cache("default", files)
        ctx = MockCtx(session_id="test_session", is_subagent=False)
        tpc = 1

        with state.vfs_lock:
            cache = state.vfs_caches["default"]
            entries = sorted(cache.snapshot_entries(), key=lambda e: e.path)
        paths_and_hashes = [
            (e.path, e.content_hash[:8], e.size_bytes) for e in entries
        ]

        sorted_paths = [p for p, _, _ in paths_and_hashes]
        assert sorted_paths[0].endswith("alpha.py")
        assert sorted_paths[1].endswith("beta.py")
        assert sorted_paths[2].endswith("middle.py")
        assert sorted_paths[3].endswith("zeta.py")

        def _content(retained, with_omission):
            lines = [
                "Workspace: default | Cached files: 4",
                "",
                "Cached files (available for instant reads):",
            ]
            for path, hash_p, size in paths_and_hashes[:retained]:
                lines.append(f"  {path} (hash:{hash_p}, {size}B)")
            if with_omission:
                lines.append(f"  ... and {4 - retained} more files")
            return (
                "[VFS CONTEXT — Cached file state]\n"
                + "\n".join(lines)
                + "\n[END VFS CONTEXT]"
            )

        cost_2_with = int(len(_content(2, True)) * tpc)
        cost_3_without = int(len(_content(3, False)) * tpc)
        budget = cost_2_with
        assert cost_3_without > budget, (
            f"Fixture must satisfy cost(3 rows without marker)={cost_3_without} "
            f"> budget={budget}"
        )

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context("test_session", ctx, budget, tpc)

        assert result is not None
        content = result["content"]
        assert "alpha.py" in content
        assert "beta.py" in content
        assert "middle.py" not in content
        assert "zeta.py" not in content
        assert "more files" in content
        omission_line = [
            line for line in content.split("\n")
            if "more files" in line
        ][0]
        assert "2 more files" in omission_line
        assert int(len(content) * tpc) <= budget

    def test_keeps_larger_row_prefix_when_omission_marker_would_overflow(
        self, vfs_enabled,
    ):
        """Wave 3: real rows have priority over the omission marker."""
        files = {
            "src/zeta.py": "z" * 50,
            "src/alpha.py": "a" * 50,
            "src/middle.py": "m" * 50,
        }
        _populate_cache("default", files)
        ctx = MockCtx(session_id="test_session", is_subagent=False)
        tpc = 1

        with state.vfs_lock:
            cache = state.vfs_caches["default"]
            entries = sorted(cache.snapshot_entries(), key=lambda e: e.path)
        paths_and_hashes = [
            (e.path, e.content_hash[:8], e.size_bytes) for e in entries
        ]

        def _content(retained, with_omission):
            lines = [
                "Workspace: default | Cached files: 3",
                "",
                "Cached files (available for instant reads):",
            ]
            for path, hash_p, size in paths_and_hashes[:retained]:
                lines.append(f"  {path} (hash:{hash_p}, {size}B)")
            if with_omission:
                lines.append(f"  ... and {3 - retained} more files")
            return (
                "[VFS CONTEXT — Cached file state]\n"
                + "\n".join(lines)
                + "\n[END VFS CONTEXT]"
            )

        k = None
        for candidate in range(1, 3):
            cost_cand_without = int(len(_content(candidate, False)) * tpc)
            cost_next_without = int(len(_content(candidate + 1, False)) * tpc)
            cost_cand_with = int(len(_content(candidate, True)) * tpc)
            if cost_cand_without < cost_next_without and cost_cand_with > cost_cand_without:
                k = candidate
                break
        assert k is not None, (
            "Fixture must have a k where k-without fits but k-with doesn't and "
            "k+1-without doesn't"
        )
        budget = int(len(_content(k, False)) * tpc)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context("test_session", ctx, budget, tpc)

        assert result is not None
        content = result["content"]
        for i in range(k):
            assert paths_and_hashes[i][0] in content
        for i in range(k, 3):
            assert paths_and_hashes[i][0] not in content
        assert "more files" not in content
        assert "Cached files: 3" in content
        assert int(len(content) * tpc) <= budget

    def test_large_budget_caps_rows_at_thirty_and_reports_all_omissions(
        self, vfs_enabled,
    ):
        """Wave 3: at most 30 rows are emitted; remaining count is accurate."""
        files = {f"src/file_{i:02d}.py": str(i) for i in range(35)}
        _populate_cache("default", files)
        ctx = MockCtx(session_id="test_session", is_subagent=False)
        tpc = 0.25

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context("test_session", ctx, 50_000, tpc)

        assert result is not None
        content = result["content"]
        file_rows = [
            line for line in content.split("\n")
            if line.startswith("  ") and "hash:" in line and ".py" in line
        ]
        assert len(file_rows) == 30, (
            f"Expected exactly 30 file rows, got {len(file_rows)}"
        )
        for i in range(30):
            assert f"file_{i:02d}.py" in content
        for i in range(30, 35):
            assert f"file_{i:02d}.py" not in content
        assert "5 more files" in content
        assert "Cached files: 35" in content
        assert int(len(content) * tpc) <= 50_000

    @pytest.mark.parametrize(
        ("budget_tokens", "tokens_per_char"),
        [
            pytest.param(0, 0.25, id="zero-budget"),
            pytest.param(-1, 0.25, id="negative-budget"),
            pytest.param(1000, 0, id="zero-token-ratio"),
            pytest.param(1000, -0.25, id="negative-token-ratio"),
        ],
    )
    def test_nonpositive_budget_inputs_omit_main_vfs(
        self, vfs_enabled, budget_tokens, tokens_per_char,
    ):
        """Wave 3: nonpositive budget inputs return None without rendering."""
        _populate_cache("default", {"src/app.py": "x"})
        ctx = MockCtx(session_id="test_session", is_subagent=False)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ):
            result = _build_vfs_context(
                "test_session", ctx, budget_tokens, tokens_per_char,
            )

        assert result is None


class TestVfsContentDeterminism:
    """VFS content must be byte-identical across consecutive turns where
    no file changes. Volatile fields (age, verified) invalidate the upstream
    provider's prefix cache and tank the cache hit rate.

    Regression: external round-4 review found VFS embedded
    `_format_age(cached_at)` and `verified` in each line, producing
    "45s ago" / " [unconfirmed]" suffixes that flipped every turn. Cache
    hit rate dropped 94% → 2%.
    """

    def test_vfs_content_stable_across_consecutive_calls(self, vfs_enabled):
        """Two consecutive _build_vfs_context calls (no cache mutation
        between them) must produce byte-identical content."""
        ws = "default"
        _populate_cache(ws, {
            "src/app.py": "content",
            "src/state.py": "state",
        })
        ctx = MockCtx(session_id="test_session")

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            r1 = _build_vfs_context("test_session", ctx, 1000, 0.25)
            r2 = _build_vfs_context("test_session", ctx, 1000, 0.25)

        assert r1 is not None
        assert r2 is not None
        assert r1["content"] == r2["content"], (
            "VFS content must be byte-identical across turns; volatile "
            "fields invalidate the upstream prefix cache"
        )

    def test_vfs_content_stable_under_time_advance(self, vfs_enabled):
        """Even when time.time() advances between calls (simulating a later
        turn), VFS content must not change."""
        ws = "default"
        _populate_cache(ws, {"src/app.py": "content"})
        ctx = MockCtx(session_id="test_session")

        with patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch("time.time", return_value=1000.0):
            mock_ws.get_active.return_value = "default"
            r1 = _build_vfs_context("test_session", ctx, 1000, 0.25)
        with patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch("time.time", return_value=1010.0):
            mock_ws.get_active.return_value = "default"
            r2 = _build_vfs_context("test_session", ctx, 1000, 0.25)

        assert r1["content"] == r2["content"], (
            "VFS content must not depend on time.time()"
        )

    def test_vfs_line_excludes_volatile_fields(self, vfs_enabled):
        """VFS line must not contain age suffix ('ago') or verified flag
        ('unconfirmed'). Both change between turns and break prefix cache."""
        ws = "default"
        _cache_file_write_through("src/app.py", "new", ws, "s1", "call_1")
        ctx = MockCtx(session_id="test_session")

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_context("test_session", ctx, 1000, 0.25)

        assert "ago" not in result["content"], (
            "VFS line must not include age ('ago') — changes every turn"
        )
        assert "unconfirmed" not in result["content"], (
            "VFS line must not include 'unconfirmed' flag — flips on write confirmation"
        )

    def test_vfs_line_uses_only_stable_fields(self, vfs_enabled):
        """VFS line content depends only on {path, content_hash, size_bytes}.
        Two entries with identical {path, hash, size} must produce identical
        lines even if cached_at differs (within TTL)."""
        import hashlib
        ws = "default"
        hash_a = hashlib.sha256(b"content-a").hexdigest()
        now = time.time()
        e1 = VFSEntry(
            path="src/app.py", content="content-a",
            content_hash=hash_a, size_bytes=9,
            cached_at=now - 5, cached_by="s1", workspace_id=ws,
            verified=True,
        )
        e2 = VFSEntry(
            path="src/app.py", content="content-a",
            content_hash=hash_a, size_bytes=9,
            cached_at=now - 2, cached_by="s2", workspace_id=ws,
            verified=False,
        )
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache(ws)
            cache.put(e1)
            cache.put(e2)

        ctx = MockCtx(session_id="test_session")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            r1 = _build_vfs_context("test_session", ctx, 1000, 0.25)
            r2 = _build_vfs_context("test_session", ctx, 1000, 0.25)

        assert r1 is not None and r2 is not None
        assert r1["content"] == r2["content"]

    def test_vfs_entry_ordering_stable_after_cache_hits(self, vfs_enabled):
        """Bug G: VFS content order must be stable across turns even when
        cache.get() mutates LRU. The model reads a file mid-turn → move_to_end
        must NOT change the VFS system message content. Without the sort fix,
        the file list reorders every turn and invalidates the upstream prefix
        cache (~94% → ~2% hit rate)."""
        ws = "default"
        _populate_cache(ws, {
            "a.py": "a-content",
            "b.py": "b-content",
            "c.py": "c-content",
        })
        ctx = MockCtx(session_id="test_session")

        with state.vfs_lock:
            cache = state.vfs_caches.get(ws)

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            r1 = _build_vfs_context("test_session", ctx, 1000, 0.25)
            with state.vfs_lock:
                cache.get("b.py")
            r2 = _build_vfs_context("test_session", ctx, 1000, 0.25)
            with state.vfs_lock:
                cache.get("a.py")
                cache.get("c.py")
                cache.get("b.py")
                cache.get("a.py")
            r3 = _build_vfs_context("test_session", ctx, 1000, 0.25)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None
        assert r1["content"] == r2["content"], (
            "VFS content must be byte-identical after LRU mutation"
        )
        assert r2["content"] == r3["content"], (
            "VFS content must be byte-identical after repeated LRU mutations"
        )

    def test_lru_eviction_unaffected_by_sort_fix(self, vfs_enabled):
        """Bug G fix uses sorted(snapshot_entries(), key=path). LRU eviction
        uses _entries.popitem(last=False) directly — must still evict
        oldest-inserted entry when over capacity."""
        ws = "default"
        with state.vfs_lock:
            cache = _get_or_create_workspace_cache(ws)
        cache._max_entries = 3

        with state.vfs_lock:
            for name in ("first.py", "second.py", "third.py", "fourth.py"):
                cache.put(VFSEntry(
                    path=name, content="x", content_hash=f"h-{name}",
                    size_bytes=1, cached_at=time.time(),
                    cached_by="s1", workspace_id=ws,
                ))

        assert cache.size == 3, f"Expected 3 entries after eviction, got {cache.size}"
        paths = [e.path for e in cache._entries.values()]
        assert "first.py" not in paths, "Oldest entry (first.py) should have been evicted"
        assert "second.py" in paths
        assert "third.py" in paths
        assert "fourth.py" in paths

        with state.vfs_lock:
            snap = cache.snapshot_entries()
        snap_paths = [e.path for e in snap]
        assert snap_paths == ["second.py", "third.py", "fourth.py"], (
            f"snapshot_entries returns LRU insertion order; got {snap_paths}"
        )


class TestBuildVfsInheritedContext:
    """US-058: VFS Inherited Context for Sub-Agents — _build_vfs_inherited_context coverage.

    Tests the sub-agent path that inherits cached entries from the parent
    agent's workspace.
    """
    @pytest.mark.parametrize(
        ("scope", "enabled", "expects_context"),
        [
            pytest.param("all", True, True, id="all-enabled"),
            pytest.param("subs", True, True, id="subs-enabled"),
            pytest.param("main", True, False, id="main-suppresses-inherited"),
            pytest.param("all", False, False, id="disabled"),
        ],
    )
    def test_prompt_emission_scope_matrix(
        self,
        vfs_enabled,
        monkeypatch,
        scope,
        enabled,
        expects_context,
    ):
        """US-057: Inherited prompt emission honors scope at render time."""
        _populate_cache("default", {"src/app.py": "content"})
        monkeypatch.setattr(state, "VFS_SCOPE", scope)
        monkeypatch.setattr(state, "VFS_ENABLED", enabled)
        ctx = MockCtx(
            session_id="child_session",
            is_subagent=True,
            parent_agent_id="parent_session",
        )

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ) as resolve_workspace:
            result = _build_vfs_inherited_context(
                ctx,
                5000,
                0.25,
            )

        assert (result is not None) is expects_context

        if expects_context:
            assert result["role"] == "system"
            assert "[VFS INHERITED CONTEXT" in result["content"]
            assert "src/app.py" in result["content"]
            resolve_workspace.assert_called_once_with(
                "child_session",
                client_hint="",
            )
        else:
            resolve_workspace.assert_not_called()

    def test_scope_change_suppresses_already_populated_inherited_cache(
        self,
        vfs_enabled,
        monkeypatch,
    ):
        """US-057: scope=main suppresses warm inherited cache at emission."""
        _populate_cache("default", {"src/app.py": "content"})
        ctx = MockCtx(
            session_id="child_session",
            is_subagent=True,
            parent_agent_id="parent_session",
        )

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ) as resolve_workspace:
            before = _build_vfs_inherited_context(
                ctx,
                5000,
                0.25,
            )

            monkeypatch.setattr(state, "VFS_SCOPE", "main")

            after = _build_vfs_inherited_context(
                ctx,
                5000,
                0.25,
            )

        assert before is not None
        assert "[VFS INHERITED CONTEXT" in before["content"]
        assert after is None

        resolve_workspace.assert_called_once_with(
            "child_session",
            client_hint="",
        )

        with state.vfs_lock:
            assert state.vfs_caches["default"].size == 1

    def test_returns_none_for_main_agent_with_populated_cache(
        self,
        vfs_enabled,
    ):
        """Inherited builder rejects main ctx independently of scope."""
        _populate_cache("default", {"src/app.py": "content"})
        ctx = MockCtx(session_id="test_session", is_subagent=False)

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="default",
        ) as resolve_workspace:
            result = _build_vfs_inherited_context(
                ctx,
                5000,
                0.25,
            )

        assert result is None
        resolve_workspace.assert_not_called()

    def test_returns_none_when_disabled(self, vfs_enabled, monkeypatch):
        monkeypatch.setattr(state, "VFS_ENABLED", False)
        ctx = MockCtx(is_subagent=True)
        result = _build_vfs_inherited_context(ctx, 1000, 0.25)
        assert result is None

    def test_returns_none_when_empty(self, vfs_enabled):
        ctx = MockCtx(is_subagent=True, parent_agent_id="parent")
        result = _build_vfs_inherited_context(ctx, 1000, 0.25)
        assert result is None

    def test_returns_system_message(self, vfs_enabled):
        ws = "default"
        _populate_cache(ws, {"src/app.py": "content"})
        ctx = MockCtx(session_id="child_session", is_subagent=True, parent_agent_id="parent_session")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_inherited_context(ctx, 5000, 0.25)
        assert result is not None
        assert result["role"] == "system"
        assert "[VFS INHERITED CONTEXT" in result["content"]

    def test_filters_expired_entries(self, vfs_enabled):
        ws = "default"
        _populate_cache(ws, {"src/app.py": "content"})
        
        # Manually expire all entries (paths are normalized to absolute)
        with state.vfs_lock:
            cache = state.vfs_caches.get(ws)
            if cache:
                for entry in list(cache.snapshot_entries()):
                    expired = VFSEntry(
                        path=entry.path, content=entry.content,
                        content_hash=entry.content_hash, size_bytes=entry.size_bytes,
                        cached_at=time.time() - 120,
                        cached_by=entry.cached_by, workspace_id=entry.workspace_id,
                    )
                    cache.put(expired)
        
        ctx = MockCtx(session_id="test_session", is_subagent=True, parent_agent_id="parent")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_inherited_context(ctx, 5000, 0.25)
        # All entries expired — should return None or empty
        assert result is None

    def test_shows_written_files_first(self, vfs_enabled):
        ws = "default"
        _populate_cache(ws, {"src/read_only.py": "read content"})
        _cache_file_write_through("src/written.py", "written content", ws, "s1", "w1")
        
        ctx = MockCtx(session_id="test_session", is_subagent=True, parent_agent_id="parent")
        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_inherited_context(ctx, 5000, 0.25)
        assert result is not None
        content = result["content"]
        written_pos = content.find("src/written.py")
        read_pos = content.find("src/read_only.py")
        assert written_pos >= 0
        assert read_pos >= 0
        assert written_pos < read_pos

    def test_inherited_content_stable_across_consecutive_calls(self, vfs_enabled):
        """Bug F + G: Two consecutive _build_vfs_inherited_context calls
        (no cache mutation between them) must produce byte-identical content."""
        ws = "default"
        _populate_cache(ws, {
            "src/app.py": "content",
            "src/state.py": "state",
        })
        ctx = MockCtx(
            session_id="test_session",
            is_subagent=True,
            parent_agent_id="parent_session",
        )

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            r1 = _build_vfs_inherited_context(ctx, 5000, 0.25)
            r2 = _build_vfs_inherited_context(ctx, 5000, 0.25)

        if r1 is None:
            pytest.skip("workspace resolution didn't match 'default'")
        assert r1["content"] == r2["content"], (
            "Inherited VFS content must be byte-identical across turns"
        )

    def test_inherited_content_stable_under_lru_mutation(self, vfs_enabled):
        """Bug G: Even when cache.get() mutates LRU mid-turn, inherited VFS
        content must be byte-identical (sorted by path, not LRU)."""
        ws = "default"
        _populate_cache(ws, {
            "a.py": "a-content",
            "b.py": "b-content",
            "c.py": "c-content",
        })
        ctx = MockCtx(
            session_id="test_session",
            is_subagent=True,
            parent_agent_id="parent_session",
        )

        with state.vfs_lock:
            cache = state.vfs_caches.get(ws)

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            r1 = _build_vfs_inherited_context(ctx, 5000, 0.25)
            with state.vfs_lock:
                cache.get("b.py")
                cache.get("a.py")
            r2 = _build_vfs_inherited_context(ctx, 5000, 0.25)

        if r1 is None:
            pytest.skip("workspace resolution didn't match 'default'")
        assert r1["content"] == r2["content"], (
            "Inherited VFS content must be byte-identical after LRU mutation"
        )

    def test_inherited_line_excludes_volatile_fields(self, vfs_enabled):
        """Bug F: Inherited VFS line must not contain age suffix ('ago')
        or verified flag ('unconfirmed'). Both change between turns."""
        ws = "default"
        _cache_file_write_through("src/written.py", "new content", ws, "s1", "w1")
        _populate_cache(ws, {"src/read.py": "read content"})

        ctx = MockCtx(
            session_id="test_session",
            is_subagent=True,
            parent_agent_id="parent_session",
        )

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_inherited_context(ctx, 5000, 0.25)

        if result is None:
            pytest.skip("workspace resolution didn't match 'default'")
        assert "ago" not in result["content"], (
            "Inherited VFS line must not include 'ago' — changes every turn"
        )
        assert "unconfirmed" not in result["content"], (
            "Inherited VFS line must not include 'unconfirmed' — flips on confirmation"
        )

    def test_inherited_sorted_by_path(self, vfs_enabled):
        """Bug G: Inherited VFS lists files in alphabetical path order,
        not LRU or age order. Populate in non-alphabetical insertion order."""
        ws = "default"
        _populate_cache(ws, {
            "z_last.py": "z",
            "a_first.py": "a",
            "m_middle.py": "m",
        })
        ctx = MockCtx(
            session_id="test_session",
            is_subagent=True,
            parent_agent_id="parent_session",
        )

        with patch("proxy.workspace_state.workspace_store") as mock_ws:
            mock_ws.get_active.return_value = "default"
            result = _build_vfs_inherited_context(ctx, 5000, 0.25)

        if result is None:
            pytest.skip("workspace resolution didn't match 'default'")
        pos_a = result["content"].find("a_first.py")
        pos_m = result["content"].find("m_middle.py")
        pos_z = result["content"].find("z_last.py")
        assert pos_a >= 0 and pos_m >= 0 and pos_z >= 0
        assert pos_a < pos_m < pos_z, (
            f"Files must be in alphabetical path order; "
            f"got a={pos_a}, m={pos_m}, z={pos_z}"
        )

    def test_slice_4e_full_rendered_content_exact_fit_boundary(
        self,
        vfs_enabled,
    ):
        """The inherited wrapper counts against its five-percent budget."""
        _populate_cache("ws", {"/p.py": "x"})
        ctx = MockCtx(
            session_id="child",
            is_subagent=True,
            parent_agent_id="parent",
        )

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="ws",
        ):
            exact = _build_vfs_inherited_context(
                ctx,
                budget_tokens=279,
                tokens_per_char=1.0,
            )
            too_small = _build_vfs_inherited_context(
                ctx,
                budget_tokens=278,
                tokens_per_char=1.0,
            )

        assert exact is not None
        assert exact["role"] == "system"
        assert len(exact["content"]) == 279
        assert exact["content"] == (
            "[VFS INHERITED CONTEXT — Parent agent's recent file activity]\n"
            "Workspace: ws | Direct parent: parent | Cached files: 1\n"
            "\n"
            "Recently read (parent agent):\n"
            "  /p.py (hash:2d711642, 1 lines)\n"
            "\n"
            "These files are cached — reading them will be instant "
            "(no disk I/O).\n"
            "[END VFS INHERITED CONTEXT]"
        )
        assert too_small is None

    @pytest.mark.parametrize(
        ("budget_tokens", "tokens_per_char"),
        [
            pytest.param(0, 0.25, id="zero-budget"),
            pytest.param(-1, 0.25, id="negative-budget"),
            pytest.param(1000, 0, id="zero-token-ratio"),
            pytest.param(1000, -0.25, id="negative-token-ratio"),
        ],
    )
    def test_slice_4e_nonpositive_budget_omits_inherited_vfs(
        self,
        vfs_enabled,
        budget_tokens,
        tokens_per_char,
    ):
        _populate_cache("ws", {"/p.py": "x"})
        ctx = MockCtx(
            session_id="child",
            is_subagent=True,
            parent_agent_id="parent",
        )

        with patch(
            "proxy.workspace_state.resolve_inherited_workspace",
            return_value="ws",
        ):
            result = _build_vfs_inherited_context(
                ctx,
                budget_tokens,
                tokens_per_char,
            )

        assert result is None


# ---------------------------------------------------------------------------
# BUG #15: parent-chain walk failure must be logged (no silent except/pass)
# ---------------------------------------------------------------------------


class MockCtxForFailure:
    """Mock ctx with a parent_agent_id to trigger the parent-chain walk."""
    def __init__(self, session_id="test_session", is_subagent=False, parent_agent_id=None):
        self.session_id = session_id
        self.is_subagent = is_subagent
        self.parent_agent_id = parent_agent_id


class TestParentChainFailureLogging:
    """BUG #15: a registry failure during the parent-chain workspace walk
    must emit a WARNING (with session_id) rather than be swallowed by a
    bare `except Exception: pass`. The fallback behavior (use the
    session's own workspace) is unchanged.
    """

    def test_build_vfs_context_logs_warning_on_registry_failure(
        self, vfs_enabled, caplog
    ):
        """A registry exception in _build_vfs_context's parent walk is logged."""
        ctx = MockCtxForFailure(session_id="sess-pcf-1")

        def boom_get_or_load(_session_id):
            raise RuntimeError("simulated registry failure")

        with patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch(
                 "proxy.agent.agent_registry.registry.get_or_load",
                 side_effect=boom_get_or_load,
             ):
            mock_ws.get_active.return_value = "default"
            with caplog.at_level(logging.WARNING, logger="proxy.vfs"):
                result = _build_vfs_context("sess-pcf-1", ctx, 1000, 0.25)

        # Fallback behavior preserved: returns None when default cache empty.
        assert result is None
        # The exception must be logged with the session_id.
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "parent-chain" in r.getMessage().lower()
        ]
        assert warnings, (
            "Expected a WARNING log about the parent-chain walk failure"
        )
        msg = warnings[0].getMessage()
        assert "sess-pcf" in msg, (
            f"Log message must include session_id; got: {msg!r}"
        )
        assert warnings[0].exc_info is not None, (
            "Log must carry exc_info so the traceback is preserved"
        )

    def test_build_vfs_inherited_context_logs_warning_on_registry_failure(
        self, vfs_enabled, caplog
    ):
        """A registry exception in _build_vfs_inherited_context's parent
        walk is logged."""
        ctx = MockCtxForFailure(
            session_id="sess-pcf-2",
            is_subagent=True,
            parent_agent_id="parent-sess",
        )

        def boom_get_or_load(_session_id):
            raise RuntimeError("simulated registry failure")

        with patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch(
                 "proxy.agent.agent_registry.registry.get_or_load",
                 side_effect=boom_get_or_load,
             ):
            mock_ws.get_active.return_value = "default"
            with caplog.at_level(logging.WARNING, logger="proxy.vfs"):
                result = _build_vfs_inherited_context(ctx, 1000, 0.25)

        # Fallback behavior preserved: no entries cached for "default" -> None.
        assert result is None
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "parent-chain" in r.getMessage().lower()
        ]
        assert warnings, (
            "Expected a WARNING log about the parent-chain walk failure"
        )
        msg = warnings[0].getMessage()
        assert "sess-pcf" in msg, (
            f"Log message must include session_id; got: {msg!r}"
        )

    def test_build_vfs_context_fallback_returns_default_workspace(
        self, vfs_enabled
    ):
        """When the parent walk fails, _build_vfs_context still serves the
        session's own workspace cache (fallback behavior unchanged)."""
        # Populate the session's own workspace cache (default).
        _populate_cache("default", {"src/app.py": "content"})
        ctx = MockCtxForFailure(session_id="sess-pcf-3")

        def boom_get_or_load(_session_id):
            raise RuntimeError("simulated registry failure")

        with patch("proxy.workspace_state.workspace_store") as mock_ws, \
             patch(
                 "proxy.agent.agent_registry.registry.get_or_load",
                 side_effect=boom_get_or_load,
             ):
            mock_ws.get_active.return_value = "default"
            # Should NOT raise — falls back to session's own workspace.
            result = _build_vfs_context("sess-pcf-3", ctx, 5000, 0.25)

        # Fallback served the default workspace's cache.
        assert result is not None
        assert "src/app.py" in result["content"]
