"""Unit tests for VFS Phase 3: Write Tracking & Conflict Detection."""

import os
import time
import hashlib
import pytest
import proxy.state as state
from proxy.vfs import (
    VFSEntry, VFSWriteRecord, VFSCache,
    _cache_file, _cache_file_write_through, _cache_file_edit_through,
    _log_write, detect_write_conflicts, _compute_diff_summary,
    _get_or_create_workspace_cache,
)

NORM_APP_PY = os.path.normpath(os.path.realpath("src/app.py"))


@pytest.fixture(autouse=True)
def clean_vfs_state():
    """Reset VFS state before each test."""
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
    """Ensure VFS is enabled with default settings."""
    monkeypatch.setattr(state, "VFS_ENABLED", True)
    monkeypatch.setattr(state, "VFS_SCOPE", "all")
    monkeypatch.setattr(state, "VFS_TTL_SECONDS", 60)
    monkeypatch.setattr(state, "VFS_MAX_ENTRIES", 500)
    monkeypatch.setattr(state, "VFS_MAX_ENTRY_BYTES", 1_048_576)
    monkeypatch.setattr(state, "VFS_MAX_WORKSPACES", 20)
    monkeypatch.setattr(state, "VFS_ROLLBACK_MAX", 100)
    monkeypatch.setattr(state, "VFS_ROLLBACK_TIMEOUT", 60)
    monkeypatch.setattr(state, "VFS_WRITE_LOG_MAX", 10000)


class TestDetectWriteConflicts:
    """Tests for detect_write_conflicts()."""

    def test_no_conflicts_empty_log(self, vfs_enabled):
        """Empty write log returns no conflicts."""
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert conflicts == []

    def test_no_conflicts_own_writes(self, vfs_enabled):
        """Agent's own writes are not conflicts."""
        _log_write("src/app.py", "agent_a", "ws1", "write", None, "hash1")
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert conflicts == []

    def test_conflict_other_agent_write(self, vfs_enabled):
        """Write by another agent is a conflict."""
        _log_write("src/app.py", "agent_b", "ws1", "write", None, "hash1")
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert len(conflicts) == 1
        assert conflicts[0].agent_id == "agent_b"
        assert conflicts[0].operation == "write"

    def test_conflict_other_agent_edit(self, vfs_enabled):
        """Edit by another agent is a conflict."""
        _log_write("src/app.py", "agent_b", "ws1", "edit", "old_hash", "new_hash")
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert len(conflicts) == 1
        assert conflicts[0].operation == "edit"

    def test_no_conflict_different_path(self, vfs_enabled):
        """Write to different path is not a conflict."""
        _log_write("src/other.py", "agent_b", "ws1", "write", None, "hash1")
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert conflicts == []

    def test_no_conflict_different_workspace(self, vfs_enabled):
        """Write in different workspace is not a conflict."""
        _log_write("src/app.py", "agent_b", "ws2", "write", None, "hash1")
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert conflicts == []

    def test_no_conflict_old_write(self, vfs_enabled):
        """Write older than since_seconds is not a conflict."""
        # Manually create an old write record
        old_record = VFSWriteRecord(
            path="src/app.py",
            agent_id="agent_b",
            workspace_id="ws1",
            timestamp=time.time() - 300,  # 5 minutes ago
            operation="write",
            old_hash=None,
            new_hash="hash1",
        )
        with state.vfs_lock:
            state.vfs_write_log.append(old_record)
        
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a", since_seconds=120)
        assert conflicts == []

    def test_multiple_conflicts(self, vfs_enabled):
        """Multiple writes by different agents are all returned."""
        _log_write("src/app.py", "agent_b", "ws1", "write", None, "hash1")
        _log_write("src/app.py", "agent_c", "ws1", "edit", "hash1", "hash2")
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a")
        assert len(conflicts) == 2

    def test_custom_since_seconds(self, vfs_enabled):
        """Custom since_seconds parameter works."""
        _log_write("src/app.py", "agent_b", "ws1", "write", None, "hash1")
        # With very short window, should still find it (just written)
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a", since_seconds=5)
        assert len(conflicts) == 1
        # With zero window, should not find it
        conflicts = detect_write_conflicts("src/app.py", "ws1", "agent_a", since_seconds=0)
        assert len(conflicts) == 0


class TestComputeDiffSummary:
    """Tests for _compute_diff_summary()."""

    def test_added_lines(self):
        result = _compute_diff_summary("line1\nline2\n", "line1\nline2\nline3\nline4\n")
        assert "+2 lines" in result

    def test_removed_lines(self):
        result = _compute_diff_summary("line1\nline2\nline3\n", "line1\n")
        assert "-2 lines" in result

    def test_same_line_count(self):
        result = _compute_diff_summary("line1\nline2\n", "new1\nnew2\n")
        assert "same line count" in result

    def test_empty_to_content(self):
        result = _compute_diff_summary("", "line1\nline2\n")
        assert "+2 lines" in result

    def test_content_to_empty(self):
        result = _compute_diff_summary("line1\nline2\n", "")
        assert "-2 lines" in result

    def test_both_empty(self):
        result = _compute_diff_summary("", "")
        assert "same line count" in result


class TestWriteLogIntegration:
    """Tests for write log integration with conflict detection."""

    def test_write_through_populates_log(self, vfs_enabled):
        """_cache_file_write_through populates write log."""
        _cache_file_write_through(NORM_APP_PY, "new content", "ws1", "agent_a", "call_1")
        assert len(state.vfs_write_log) == 1
        assert state.vfs_write_log[0].agent_id == "agent_a"
        assert state.vfs_write_log[0].operation == "write"

    def test_edit_through_populates_log(self, vfs_enabled):
        """_cache_file_edit_through populates write log."""
        _cache_file("src/app.py", "hello world", "ws1", "agent_a")
        _cache_file_edit_through(NORM_APP_PY, "world", "universe", "ws1", "agent_a", tool_call_id="e1")
        assert len(state.vfs_write_log) == 1
        assert state.vfs_write_log[0].operation == "edit"

    def test_conflict_after_cross_agent_write(self, vfs_enabled):
        """Full flow: agent A reads, agent B writes, agent A detects conflict."""
        ws = "ws1"
        # Agent A reads file
        _cache_file("src/app.py", "original content", ws, "agent_a")
        
        # Agent B writes file (write-through)
        _cache_file_write_through(NORM_APP_PY, "modified content", ws, "agent_b", "call_b1")
        
        # Agent A checks for conflicts
        conflicts = detect_write_conflicts(NORM_APP_PY, ws, "agent_a")
        assert len(conflicts) == 1
        assert conflicts[0].agent_id == "agent_b"

    def test_no_conflict_after_own_write(self, vfs_enabled):
        """Agent's own write is not a conflict."""
        ws = "ws1"
        _cache_file("src/app.py", "original", ws, "agent_a")
        _cache_file_write_through(NORM_APP_PY, "modified", ws, "agent_a", "call_a1")
        
        conflicts = detect_write_conflicts(NORM_APP_PY, ws, "agent_a")
        assert conflicts == []


class TestWriteLogTruncation:
    """Tests for write log truncation when exceeding VFS_WRITE_LOG_MAX."""

    def test_truncation_keeps_last_n(self, vfs_enabled, monkeypatch):
        """Write log is truncated to VFS_WRITE_LOG_MAX entries."""
        monkeypatch.setattr(state, "VFS_WRITE_LOG_MAX", 5)
        
        for i in range(10):
            _log_write(f"file{i}.py", "agent_a", "ws1", "write", None, f"hash{i}")
        
        assert len(state.vfs_write_log) == 5
        # Should keep the last 5 (file5 through file9)
        paths = [r.path for r in state.vfs_write_log]
        assert "file0.py" not in paths
        assert "file4.py" not in paths
        assert "file5.py" in paths
        assert "file9.py" in paths

    def test_no_truncation_under_limit(self, vfs_enabled, monkeypatch):
        """Write log is not truncated when under limit."""
        monkeypatch.setattr(state, "VFS_WRITE_LOG_MAX", 100)
        
        for i in range(10):
            _log_write(f"file{i}.py", "agent_a", "ws1", "write", None, f"hash{i}")
        
        assert len(state.vfs_write_log) == 10


class TestConflictWarningIntegration:
    """Tests for _check_vfs_read_cache conflict warning text."""

    def test_no_warning_without_conflicts(self, vfs_enabled):
        """Cached content returned without warning when no conflicts."""
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline
        
        ws = "ws1"
        _cache_file("src/app.py", "original content", ws, "agent_a")
        
        tp = ToolPipeline.__new__(ToolPipeline)
        tp._cached_reads = {}
        
        # Mock tool call
        class MockToolCall:
            name = "read"
            arguments = {"file_path": "src/app.py"}
            id = "call_1"
        
        # Need to mock _resolve_workspace_id
        tp._resolve_workspace_id = lambda sid, ch: ws
        
        result = tp._check_vfs_read_cache(MockToolCall(), "agent_a")
        
        assert result is not None
        assert "original content" in result
        assert "VFS CONFLICT WARNING" not in result

    def test_warning_with_conflicts(self, vfs_enabled):
        """Conflict warning appended when other agent wrote recently."""
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline
        
        ws = "ws1"
        _cache_file("src/app.py", "original content", ws, "agent_a")
        
        # Another agent writes to the same file
        _log_write(NORM_APP_PY, "agent_b", ws, "write", "old", "new")
        
        tp = ToolPipeline.__new__(ToolPipeline)
        tp._cached_reads = {}
        
        class MockToolCall:
            name = "read"
            arguments = {"file_path": "src/app.py"}
            id = "call_1"
        
        tp._resolve_workspace_id = lambda sid, ch: ws
        
        result = tp._check_vfs_read_cache(MockToolCall(), "agent_a")
        
        assert result is not None
        assert "original content" in result
        assert "VFS CONFLICT WARNING" in result
        assert "agent_b" in result

    def test_warning_capped_at_5_conflicts(self, vfs_enabled):
        """Conflict warning shows at most 5 conflict lines."""
        from proxy.tool_pipeline.tool_pipeline import ToolPipeline
        
        ws = "ws1"
        _cache_file("src/app.py", "original content", ws, "agent_a")
        
        # 10 different agents write to the same file
        for i in range(10):
            _log_write(NORM_APP_PY, f"agent_{i}", ws, "write", "old", f"hash{i}")
        
        tp = ToolPipeline.__new__(ToolPipeline)
        tp._cached_reads = {}
        
        class MockToolCall:
            name = "read"
            arguments = {"file_path": "src/app.py"}
            id = "call_1"
        
        tp._resolve_workspace_id = lambda sid, ch: ws
        
        result = tp._check_vfs_read_cache(MockToolCall(), "agent_a")
        
        assert result is not None
        assert "VFS CONFLICT WARNING" in result
        # Count conflict lines (lines starting with "  - ")
        conflict_lines = [l for l in result.split("\n") if l.strip().startswith("- ")]
        assert len(conflict_lines) <= 5
        # Should have "and N more" line
        assert "more" in result
