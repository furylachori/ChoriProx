import pytest
import json


class TestArgumentNormalizationIntegration:
    def test_all_alias_mappings_roundtrip(self):
        from qwen_proxy import normalize_tool_arguments

        alias_payloads = [
            {"path": "/tmp/test.txt"},
            {"file": "/tmp/test.txt"},
            {"filename": "/tmp/test.txt"},
            {"filepath": "/tmp/test.txt"},
        ]

        for alias_args in alias_payloads:
            result = normalize_tool_arguments("read_file", alias_args)
            assert "file_path" in result, f"Alias mapping failed for {alias_args}"
            assert (
                result["file_path"] == alias_args.get("path")
                or alias_args.get("file")
                or alias_args.get("filename")
                or alias_args.get("filepath")
            )

    def test_opencode_to_canonical_mapping(self):
        from qwen_proxy import normalize_tool_arguments

        result = normalize_tool_arguments("read_file", {"file": "/tmp/data.json"})
        assert "file_path" in result
        assert result["file_path"] == "/tmp/data.json"
