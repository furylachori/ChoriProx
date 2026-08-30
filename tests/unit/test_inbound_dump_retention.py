"""Tests for inbound dump retention cap (M8).

``dump_inbound_request_body`` writes to ``data/debug_inbound/`` with no
rotation by default. With ``PROXY_DUMP_REQUESTS=1`` this is an unbounded
disk-fill risk. The retention cap (PROXY_DUMP_MAX_FILES, default 250)
deletes the oldest files (by mtime) before writing a new dump so the
directory never exceeds the cap. Cleanup failure must never break a dump.
"""
import json
import os
import time

import pytest

pytestmark = pytest.mark.unit

from proxy.http_client import dump_inbound_request_body


def _seed_dump_files(dump_dir, count):
    """Create `count` .json dump files with increasing mtimes."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    base = time.time() - count
    for i in range(count):
        p = dump_dir / f"req-{i:04d}-20260101T000000000000.json"
        p.write_text(json.dumps({"i": i}))
        mtime = base + i
        os.utime(p, (mtime, mtime))


def _count_json(dump_dir):
    return len([n for n in os.listdir(dump_dir) if n.endswith(".json")])


class TestDumpRetention:
    def test_cleanup_deletes_oldest_when_over_cap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_INBOUND_DIR", str(tmp_path))
        monkeypatch.setenv("PROXY_DUMP_MAX_FILES", "100")
        _seed_dump_files(tmp_path, 110)

        dump_inbound_request_body("sess", "newreq", {"hello": "world"})

        # cap is 100; after writing one new file the pre-write cleanup removed
        # enough old files so the total is <= 100.
        assert _count_json(tmp_path) <= 100

    def test_oldest_files_are_deleted_first(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_INBOUND_DIR", str(tmp_path))
        monkeypatch.setenv("PROXY_DUMP_MAX_FILES", "5")
        _seed_dump_files(tmp_path, 10)

        dump_inbound_request_body("sess", "newreq", {"hello": "world"})

        names = sorted(n for n in os.listdir(tmp_path) if n.endswith(".json"))
        # The oldest seeded files (req-0000 ... req-0004) must be gone;
        # the newest seeded files + the new dump remain.
        assert all("req-0000" not in n for n in names)
        assert any("newreq" in n for n in names)
        assert _count_json(tmp_path) <= 5

    def test_no_cleanup_when_under_cap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_INBOUND_DIR", str(tmp_path))
        monkeypatch.setenv("PROXY_DUMP_MAX_FILES", "100")
        _seed_dump_files(tmp_path, 3)

        dump_inbound_request_body("sess", "newreq", {"hello": "world"})

        assert _count_json(tmp_path) == 4

    def test_custom_cap_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_INBOUND_DIR", str(tmp_path))
        monkeypatch.setenv("PROXY_DUMP_MAX_FILES", "3")
        _seed_dump_files(tmp_path, 8)

        dump_inbound_request_body("sess", "newreq", {"hello": "world"})

        assert _count_json(tmp_path) <= 3

    def test_invalid_cap_env_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_INBOUND_DIR", str(tmp_path))
        monkeypatch.setenv("PROXY_DUMP_MAX_FILES", "not-a-number")
        _seed_dump_files(tmp_path, 255)

        dump_inbound_request_body("sess", "newreq", {"hello": "world"})

        assert _count_json(tmp_path) <= 250

    def test_cleanup_failure_does_not_break_dump(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROXY_DUMP_REQUESTS", "1")
        monkeypatch.setenv("PROXY_DUMP_INBOUND_DIR", str(tmp_path))
        monkeypatch.setenv("PROXY_DUMP_MAX_FILES", "5")
        _seed_dump_files(tmp_path, 10)

        real_remove = os.remove

        def flaky_remove(path, *a, **kw):
            # First removal attempt raises; the helper must swallow it and
            # continue (the dump itself must still succeed).
            raise OSError("simulated removal failure")

        import proxy.http_client as hc
        monkeypatch.setattr(os, "remove", flaky_remove)
        # The dump must not raise even though cleanup fails.
        dump_inbound_request_body("sess", "newreq", {"hello": "world"})
        monkeypatch.setattr(os, "remove", real_remove)
        # The new dump file should exist regardless of cleanup failure.
        names = [n for n in os.listdir(tmp_path) if n.endswith(".json")]
        assert any("newreq" in n for n in names)