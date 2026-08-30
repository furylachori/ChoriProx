"""Tests for proxy.skills.path_utils (read_contained_regular_file + get_skills_dir).

Per ``docs/specs/custom-skills-remediation.md`` §5.4 (Safe file reading):
descriptor-based safe reads, symlink/TOCTOU/escape/traversal/oversize.

These tests use REAL filesystem objects (tmp_path). They do NOT mock the
read function — every test exercises the production code path end-to-end.

**Tautology note:** the test names include words like "absolute" and
"traversal"; pytest's auto-generated ``tmp_path`` embeds the test name.
Assertions therefore check for a substring that ONLY appears in the
PRODUCTION error message — never on a path string that happens to contain
the test name. For "absolute paths" we assert ``"must be relative"`` (the
production message); for "traversal" we assert ``"must not contain"``.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from proxy.skills.path_utils import (
    SkillPathError,
    _REPO_ROOT,
    get_skills_dir,
    read_contained_regular_file,
)


@pytest.fixture(autouse=True)
def _isolate_proxy_skills_dir(monkeypatch):
    """Snapshot/restore PROXY_SKILLS_DIR around every test in this module."""
    original = os.environ.get("PROXY_SKILLS_DIR")
    yield
    if original is None:
        monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
    else:
        monkeypatch.setenv("PROXY_SKILLS_DIR", original)


class TestGetSkillsDir:
    def test_returns_none_when_env_empty_string(self, monkeypatch):
        monkeypatch.setenv("PROXY_SKILLS_DIR", "")
        assert get_skills_dir() is None

    def test_returns_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
        assert get_skills_dir() == _REPO_ROOT / "skills"

    def test_resolves_absolute(self, monkeypatch, tmp_path):
        unique = tmp_path / "abs_target_dir"
        monkeypatch.setenv("PROXY_SKILLS_DIR", str(unique))
        assert get_skills_dir() == unique

    def test_resolves_relative_to_repo_root(self, monkeypatch):
        rel = "skills_relative_to_repo_unique"
        monkeypatch.setenv("PROXY_SKILLS_DIR", rel)
        assert get_skills_dir() == _REPO_ROOT / rel

    def test_explicit_disable_takes_precedence(self, monkeypatch):
        monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
        assert get_skills_dir() == _REPO_ROOT / "skills"
        monkeypatch.setenv("PROXY_SKILLS_DIR", "")
        assert get_skills_dir() is None
        monkeypatch.delenv("PROXY_SKILLS_DIR", raising=False)
        assert get_skills_dir() == _REPO_ROOT / "skills"


class TestReadContainedRegularFileHappyPath:
    def test_happy_path(self, tmp_path):
        f = tmp_path / "ok.md"
        f.write_text("hello world", encoding="utf-8")
        assert (
            read_contained_regular_file(tmp_path, "ok.md", max_bytes=1024)
            == "hello world"
        )

    def test_happy_path_nested(self, tmp_path):
        sub = tmp_path / "prompts"
        sub.mkdir()
        (sub / "adversary.md").write_text("nested", encoding="utf-8")
        assert (
            read_contained_regular_file(
                tmp_path, "prompts/adversary.md", max_bytes=1024
            )
            == "nested"
        )

    def test_returns_empty_string_for_zero_byte_file(self, tmp_path):
        (tmp_path / "empty.md").write_text("", encoding="utf-8")
        assert (
            read_contained_regular_file(
                tmp_path, "empty.md", max_bytes=1024
            )
            == ""
        )


class TestReadContainedRegularFileRejections:
    def test_rejects_absolute_path(self, tmp_path):
        """Absolute path is rejected with the 'must be relative' message."""
        absolute = str(tmp_path / "x.md")
        Path(absolute).write_text("x")
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(tmp_path, absolute, max_bytes=1024)
        msg = str(cm.value).lower()
        assert "must be relative" in msg, (
            f"expected 'must be relative' in {msg!r}"
        )

    def test_rejects_empty_path(self, tmp_path):
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(tmp_path, "", max_bytes=1024)
        assert "non-empty" in str(cm.value).lower()

    def test_rejects_traversal(self, tmp_path):
        outside = tempfile.mkdtemp()
        try:
            evil_name = Path(outside).name
            evil = Path(outside) / "evil.md"
            evil.write_text("evil")
            with pytest.raises(SkillPathError) as cm:
                read_contained_regular_file(
                    tmp_path,
                    "../" + evil_name + "/evil.md",
                    max_bytes=1024,
                )
            msg = str(cm.value).lower()
            assert "must not contain" in msg, (
                f"expected 'must not contain' in {msg!r}"
            )
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_rejects_traversal_bare(self, tmp_path):
        with pytest.raises(SkillPathError):
            read_contained_regular_file(tmp_path, "..", max_bytes=1024)

    def test_rejects_symlink_at_leaf(self, tmp_path):
        outside = tempfile.mkdtemp()
        try:
            outside_file = Path(outside) / "target.md"
            outside_file.write_text("target")
            link_path = tmp_path / "sneaky.md"
            os.symlink(str(outside_file), str(link_path))
            with pytest.raises(SkillPathError):
                read_contained_regular_file(
                    tmp_path, "sneaky.md", max_bytes=1024
                )
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_rejects_symlink_in_intermediate(self, tmp_path):
        outside = tempfile.mkdtemp()
        try:
            outside_dir = Path(outside) / "real"
            outside_dir.mkdir()
            (outside_dir / "good.md").write_text("hi")
            link = tmp_path / "prompts"
            os.symlink(str(outside_dir), str(link))
            with pytest.raises(SkillPathError) as cm:
                read_contained_regular_file(
                    tmp_path, "prompts/good.md", max_bytes=1024
                )
            assert "symlink" in str(cm.value).lower()
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_rejects_directory(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(
                tmp_path, "subdir", max_bytes=1024
            )
        assert "not a regular file" in str(cm.value).lower()

    def test_rejects_oversize(self, tmp_path):
        f = tmp_path / "big.md"
        f.write_text("a" * 100, encoding="utf-8")
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(tmp_path, "big.md", max_bytes=10)
        assert "exceeds" in str(cm.value).lower()

    def test_rejects_invalid_utf8(self, tmp_path):
        f = tmp_path / "bad.md"
        f.write_bytes(b"\xff\xfe\xff\x00abc")
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(tmp_path, "bad.md", max_bytes=1024)
        assert "utf-8" in str(cm.value).lower()

    def test_strict_utf8_no_bom(self, tmp_path):
        f = tmp_path / "bom.md"
        f.write_bytes(b"\xef\xbb\xbfhello")
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(tmp_path, "bom.md", max_bytes=1024)
        assert "bom" in str(cm.value).lower()

    def test_rejects_nonexistent_leaf(self, tmp_path):
        with pytest.raises(SkillPathError):
            read_contained_regular_file(
                tmp_path, "missing.md", max_bytes=1024
            )

    def test_rejects_missing_intermediate(self, tmp_path):
        (tmp_path / "ok.md").write_text("ok")
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(
                tmp_path, "no_such_dir/ok.md", max_bytes=1024
            )
        assert "not found" in str(cm.value).lower()

    def test_rejects_max_bytes_zero_when_file_has_content(self, tmp_path):
        (tmp_path / "x.md").write_text("anything", encoding="utf-8")
        with pytest.raises(SkillPathError) as cm:
            read_contained_regular_file(tmp_path, "x.md", max_bytes=0)
        assert "exceeds" in str(cm.value).lower()
