import pytest
import json


def _make_tool_call(name, arguments, idx=0):
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments)
            if isinstance(arguments, dict)
            else arguments,
        },
    }


class TestParameterCorrectnessIntegration:
    def test_read_params_valid_after_normalization(self):
        from qwen_proxy import normalize_tool_arguments

        result = normalize_tool_arguments("read_file", {"file": "/tmp/test.txt"})
        assert "file_path" in result
        assert result["file_path"] == "/tmp/test.txt"

    def test_write_params_valid_after_normalization(self):
        from qwen_proxy import normalize_tool_arguments

        result = normalize_tool_arguments(
            "write_file", {"file": "/tmp/output.txt", "data": "hello"}
        )
        assert "file_path" in result
        assert result["file_path"] == "/tmp/output.txt"
        assert "content" in result
        assert result["content"] == "hello"

    def test_edit_params_valid_after_normalization(self):
        from qwen_proxy import normalize_tool_arguments

        result = normalize_tool_arguments(
            "edit_file",
            {
                "file": "/tmp/test.txt",
                "old": "hello",
                "new": "world",
            },
        )
        assert "file_path" in result
        assert result["file_path"] == "/tmp/test.txt"
        assert "old_string" in result
        assert result["old_string"] == "hello"
        assert "new_string" in result
        assert result["new_string"] == "world"

    def test_read_params_schema_canonicalization_via_valid_params(self):
        from proxy.tool_parsing.normalize import (
            build_valid_params_by_tool,
            normalize_tool_arguments,
        )

        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                        },
                    },
                },
            }
        ]
        vpbt = build_valid_params_by_tool(tools_schema)
        result = normalize_tool_arguments(
            "read", {"file": "/tmp/test.txt"}, valid_params=vpbt.get("read")
        )
        assert "file_path" in result
        assert result["file_path"] == "/tmp/test.txt"
