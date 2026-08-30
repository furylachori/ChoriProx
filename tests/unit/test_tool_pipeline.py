"""Unit tests for tool_pipeline module helpers."""

from proxy.tool_parsing import extract_file_path


class TestExtractFilePath:
    """extract_file_path pulls a path across snake_case, camelCase, and bare keys."""

    def test_snake_case_file_path(self):
        assert extract_file_path({"file_path": "/tmp/a.txt"}) == "/tmp/a.txt"

    def test_camel_case_file_path(self):
        assert extract_file_path({"filePath": "/tmp/a.txt"}) == "/tmp/a.txt"

    def test_bare_path_key(self):
        assert extract_file_path({"path": "/tmp/a.txt"}) == "/tmp/a.txt"

    def test_empty_dict_returns_empty(self):
        assert extract_file_path({}) == ""

    def test_non_dict_returns_empty(self):
        assert extract_file_path(None) == ""
        assert extract_file_path("/tmp/a.txt") == ""
        assert extract_file_path(["file_path", "/tmp/a.txt"]) == ""

    def test_snake_takes_precedence_over_path(self):
        assert extract_file_path({"file_path": "/a", "path": "/b"}) == "/a"

    def test_camel_takes_precedence_over_path(self):
        assert extract_file_path({"filePath": "/a", "path": "/b"}) == "/a"
