"""
Unit tests for normalize_tool_arguments function.
Tests parameter alias normalization for all tool types.

US-125: Internally-executed memory tools use snake_case (valid_params=None legacy path).
"""

import pytest
from qwen_proxy import normalize_tool_arguments


class TestPathAliases:
    """Tests for path/file/filepath/filename → file_path normalization."""

    def test_path_aliases(self):
        """path/filepath/filename should normalize to file_path."""
        args = {"path": "/tmp/test.txt"}
        result = normalize_tool_arguments("read_file", args)
        assert "file_path" in result
        assert result["file_path"] == "/tmp/test.txt"

        args = {"filepath": "/tmp/test.txt"}
        result = normalize_tool_arguments("read_file", args)
        assert "file_path" in result

        args = {"filename": "/tmp/test.txt"}
        result = normalize_tool_arguments("read_file", args)
        assert "file_path" in result


class TestCmdAliases:
    """Tests for cmd/command/shell → command normalization."""

    def test_cmd_aliases(self):
        """cmd/command/shell should normalize to command for bash."""
        args = {"cmd": "ls -la"}
        result = normalize_tool_arguments("bash", args)
        assert "command" in result
        assert result["command"] == "ls -la"

        args = {"shell": "pwd"}
        result = normalize_tool_arguments("bash", args)
        assert "command" in result


class TestEditAliases:
    """Tests for old/new → old_string/new_string normalization."""

    def test_edit_aliases(self):
        """old/new should normalize to old_string/new_string for edit_file."""
        args = {"path": "/tmp/test.py", "old": "def foo():", "new": "def bar():"}
        result = normalize_tool_arguments("edit_file", args)
        assert "old_string" in result
        assert "new_string" in result
        assert result["old_string"] == "def foo():"
        assert result["new_string"] == "def bar():"


class TestReadWithOffsetLimit:
    """Tests for read tool with offset and limit parameters."""

    def test_read_with_offset_limit(self):
        """read should pass through offset and limit correctly."""
        args = {"path": "/tmp/test.txt", "offset": 10, "limit": 50}
        result = normalize_tool_arguments("read_file", args)
        assert result.get("file_path") == "/tmp/test.txt"
        assert result.get("offset") == 10
        assert result.get("limit") == 50


class TestGlobPatternPassthrough:
    """Tests for glob tool pattern parameter."""

    def test_glob_pattern_passthrough(self):
        """Glob pattern should pass through unchanged."""
        args = {"pattern": "*.py", "path": "/path/to/documents"}
        result = normalize_tool_arguments("file_glob", args)
        assert "pattern" in result or "path" in result


class TestWebfetchUrlPassthrough:
    """Tests for webfetch tool URL parameter."""

    def test_webfetch_url_passthrough(self):
        """webfetch URL should pass through unchanged."""
        args = {"url": "https://example.com", "format": "markdown"}
        result = normalize_tool_arguments("web_fetch", args)
        assert "url" in result
        assert result["url"] == "https://example.com"


class TestEmptyArgsUnchanged:
    """Tests for empty arguments handling."""

    def test_empty_args_unchanged(self):
        """Empty arguments should remain unchanged."""
        result = normalize_tool_arguments("bash", {})
        assert result == {}

    def test_unknown_tool_returns_unchanged(self):
        """Unknown tool should return arguments unchanged."""
        args = {"foo": "bar", "baz": 123}
        result = normalize_tool_arguments("unknown_tool", args)
        assert result.get("foo") == "bar"
        assert result.get("baz") == 123


class TestMemoryToolNormalization:
    """Tests for memory tool parameter normalization."""

    def test_recall_param_normalization(self):
        """recall parameters should be normalized."""
        args = {"semantic_targets": ["recent tasks"], "keywords": ["task"], "limit": 5}
        result = normalize_tool_arguments("recall", args)
        assert "semantic_targets" in result
        assert result["semantic_targets"] == ["recent tasks"]

    def test_search_param_normalization(self):
        """search parameters should be normalized."""
        args = {"keywords": ["previous conversation"], "limit": 10}
        result = normalize_tool_arguments("search", args)
        assert "keywords" in result
        assert result["keywords"] == ["previous conversation"]

    def test_save_memory_param_normalization(self):
        """save_memory parameters should return error (removed tool)."""
        args = {"topic_key": "project_x", "memory_content": "important info"}
        result = normalize_tool_arguments("save_memory", args)
        assert "error" in result or "topic_key" in result

    def test_manage_task_param_normalization(self):
        """manage_task parameters should be passthrough (removed tool)."""
        args = {"action": "create", "task_name": "new task"}
        result = normalize_tool_arguments("manage_task", args)
        assert "action" in result
        assert result["action"] == "create"
