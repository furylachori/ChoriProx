"""Unit tests for VFS data structures and VFSCache operations."""

import time
import pytest
from proxy.vfs import VFSEntry, VFSWriteRecord, VFSRollbackSnapshot, VFSCache


def _make_entry(path="src/app.py", content="hello world", session="s1",
                workspace="ws1", verified=True, cached_at=None):
    """Helper to create VFSEntry with sensible defaults."""
    import hashlib
    return VFSEntry(
        path=path,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        size_bytes=len(content),
        cached_at=cached_at or time.time(),
        cached_by=session,
        workspace_id=workspace,
        verified=verified,
    )


class TestVFSEntry:
    """Tests for VFSEntry dataclass."""

    def test_frozen(self):
        """VFSEntry is immutable."""
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.content = "changed"

    def test_fields(self):
        """All fields are accessible."""
        entry = _make_entry(path="foo.py", content="bar", session="s1")
        assert entry.path == "foo.py"
        assert entry.content == "bar"
        assert entry.cached_by == "s1"
        assert entry.verified is True

    def test_default_verified(self):
        """verified defaults to True."""
        entry = _make_entry()
        assert entry.verified is True

    def test_unverified(self):
        """Can create unverified entry."""
        entry = _make_entry(verified=False)
        assert entry.verified is False


class TestVFSWriteRecord:
    """Tests for VFSWriteRecord dataclass."""

    def test_fields(self):
        record = VFSWriteRecord(
            path="src/app.py",
            agent_id="agent1",
            workspace_id="ws1",
            timestamp=time.time(),
            operation="write",
            old_hash="abc123",
            new_hash="def456",
        )
        assert record.path == "src/app.py"
        assert record.operation == "write"
        assert record.old_hash == "abc123"

    def test_new_file_no_old_hash(self):
        record = VFSWriteRecord(
            path="new.py", agent_id="a1", workspace_id="ws1",
            timestamp=time.time(), operation="write",
            old_hash=None, new_hash="abc",
        )
        assert record.old_hash is None


class TestVFSRollbackSnapshot:
    """Tests for VFSRollbackSnapshot dataclass."""

    def test_with_old_entry(self):
        old = _make_entry(content="old content")
        snap = VFSRollbackSnapshot(
            path="src/app.py",
            workspace_id="ws1",
            old_entry=old,
            new_hash="new_hash",
            tool_call_id="call_123",
            created_at=time.time(),
        )
        assert snap.old_entry is old
        assert snap.parent_snapshot_id is None

    def test_new_file_no_old_entry(self):
        snap = VFSRollbackSnapshot(
            path="new.py", workspace_id="ws1",
            old_entry=None, new_hash="abc",
            tool_call_id="call_456", created_at=time.time(),
        )
        assert snap.old_entry is None

    def test_parent_snapshot_id(self):
        snap = VFSRollbackSnapshot(
            path="src/app.py", workspace_id="ws1",
            old_entry=None, new_hash="abc",
            tool_call_id="call_789", created_at=time.time(),
            parent_snapshot_id="call_123",
        )
        assert snap.parent_snapshot_id == "call_123"


class TestVFSCache:
    """Tests for VFSCache LRU cache."""

    def test_put_and_get(self):
        cache = VFSCache(max_entries=10)
        entry = _make_entry(path="a.py", content="aaa")
        cache.put(entry)
        result = cache.get("a.py")
        assert result is entry
        assert cache.size == 1

    def test_get_miss(self):
        cache = VFSCache(max_entries=10)
        assert cache.get("nonexistent.py") is None

    def test_put_update(self):
        cache = VFSCache(max_entries=10)
        e1 = _make_entry(path="a.py", content="v1")
        e2 = _make_entry(path="a.py", content="v2")
        cache.put(e1)
        cache.put(e2)
        assert cache.size == 1
        assert cache.get("a.py").content == "v2"
        assert cache.total_bytes == len("v2")

    def test_lru_eviction(self):
        cache = VFSCache(max_entries=3)
        cache.put(_make_entry(path="a.py", content="a"))
        cache.put(_make_entry(path="b.py", content="b"))
        cache.put(_make_entry(path="c.py", content="c"))
        # Cache is full, adding d.py should evict a.py (LRU)
        cache.put(_make_entry(path="d.py", content="d"))
        assert cache.get("a.py") is None  # Evicted
        assert cache.get("b.py") is not None
        assert cache.get("c.py") is not None
        assert cache.get("d.py") is not None
        assert cache.size == 3

    def test_get_promotes_to_mru(self):
        cache = VFSCache(max_entries=3)
        cache.put(_make_entry(path="a.py", content="a"))
        cache.put(_make_entry(path="b.py", content="b"))
        cache.put(_make_entry(path="c.py", content="c"))
        # Access a.py to promote it to MRU
        cache.get("a.py")
        # Now b.py is LRU, adding d.py should evict b.py
        cache.put(_make_entry(path="d.py", content="d"))
        assert cache.get("a.py") is not None  # Still here (was promoted)
        assert cache.get("b.py") is None       # Evicted (was LRU)
        assert cache.get("c.py") is not None
        assert cache.get("d.py") is not None

    def test_invalidate(self):
        cache = VFSCache(max_entries=10)
        entry = _make_entry(path="a.py", content="aaa")
        cache.put(entry)
        removed = cache.invalidate("a.py")
        assert removed is entry
        assert cache.get("a.py") is None
        assert cache.size == 0
        assert cache.total_bytes == 0

    def test_invalidate_miss(self):
        cache = VFSCache(max_entries=10)
        assert cache.invalidate("nonexistent.py") is None

    def test_invalidate_prefix(self):
        cache = VFSCache(max_entries=10)
        cache.put(_make_entry(path="src/a.py", content="a"))
        cache.put(_make_entry(path="src/b.py", content="b"))
        cache.put(_make_entry(path="tests/c.py", content="c"))
        count = cache.invalidate_prefix("src/")
        assert count == 2
        assert cache.get("src/a.py") is None
        assert cache.get("src/b.py") is None
        assert cache.get("tests/c.py") is not None
        assert cache.size == 1

    def test_invalidate_all(self):
        cache = VFSCache(max_entries=10)
        cache.put(_make_entry(path="a.py", content="a"))
        cache.put(_make_entry(path="b.py", content="b"))
        count = cache.invalidate_all()
        assert count == 2
        assert cache.size == 0
        assert cache.total_bytes == 0

    def test_oversized_entry_skipped(self):
        cache = VFSCache(max_entries=10, max_entry_bytes=5)
        entry = _make_entry(path="big.py", content="this is way more than 5 bytes")
        cache.put(entry)
        assert cache.get("big.py") is None
        assert cache.size == 0

    def test_total_bytes_tracking(self):
        cache = VFSCache(max_entries=10)
        cache.put(_make_entry(path="a.py", content="hello"))  # 5 bytes
        cache.put(_make_entry(path="b.py", content="world!"))  # 6 bytes
        assert cache.total_bytes == 11
        cache.invalidate("a.py")
        assert cache.total_bytes == 6

    def test_snapshot_paths(self):
        cache = VFSCache(max_entries=10)
        e1 = _make_entry(path="a.py", content="aaa")
        e2 = _make_entry(path="b.py", content="bbb")
        cache.put(e1)
        cache.put(e2)
        paths = cache.snapshot_paths()
        assert len(paths) == 2
        assert paths[0][0] == "a.py"
        assert paths[0][1] == e1.content_hash
        assert paths[1][0] == "b.py"

    def test_snapshot_entries(self):
        cache = VFSCache(max_entries=10)
        e1 = _make_entry(path="a.py", content="aaa")
        cache.put(e1)
        entries = cache.snapshot_entries()
        assert len(entries) == 1
        assert entries[0] is e1

    def test_record_reader(self):
        cache = VFSCache(max_entries=10)
        cache.record_reader("a.py", "session1")
        cache.record_reader("a.py", "session2")
        cache.record_reader("a.py", "session1")  # Duplicate
        readers = cache.get_readers("a.py")
        assert readers == {"session1", "session2"}

    def test_get_readers_empty(self):
        cache = VFSCache(max_entries=10)
        readers = cache.get_readers("nonexistent.py")
        assert readers == set()

    def test_get_readers_returns_copy(self):
        """get_readers returns a copy, not the internal set."""
        cache = VFSCache(max_entries=10)
        cache.record_reader("a.py", "s1")
        readers = cache.get_readers("a.py")
        readers.add("s2")  # Modifying the copy
        assert cache.get_readers("a.py") == {"s1"}  # Internal unchanged

    def test_eviction_updates_total_bytes(self):
        cache = VFSCache(max_entries=2)
        cache.put(_make_entry(path="a.py", content="aaaa"))  # 4 bytes
        cache.put(_make_entry(path="b.py", content="bbbb"))  # 4 bytes
        assert cache.total_bytes == 8
        cache.put(_make_entry(path="c.py", content="cccc"))  # Evicts a.py
        assert cache.total_bytes == 8  # Still 8 (b + c)
        assert cache.get("a.py") is None

    def test_empty_cache_properties(self):
        cache = VFSCache()
        assert cache.size == 0
        assert cache.total_bytes == 0
        assert cache.snapshot_paths() == []
        assert cache.snapshot_entries() == []
