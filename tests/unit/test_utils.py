import pytest
from memory.utils import strip_args

pytestmark = pytest.mark.unit


class TestStripArgs:
    def test_valid_tool_summary(self):
        assert strip_args("[tools: bash: rm -rf, write_file: foo.py: content]") == "[tools: bash, write_file]"

    def test_single_tool(self):
        assert strip_args("[tools: bash: some args]") == "[tools: bash]"

    def test_empty_tools(self):
        assert strip_args("[tools:]") == "[tools:]"

    def test_not_a_tool_summary(self):
        assert strip_args("random text") == "random text"

    def test_none_input(self):
        assert strip_args(None) is None

    def test_empty_string(self):
        assert strip_args("") == ""

    def test_whitespace_handling(self):
        assert strip_args("[tools: bash : arg1 , write_file : arg2]") == "[tools: bash, write_file]"

    def test_exact_format(self):
        assert strip_args("[tools: tool1: arg1]") == "[tools: tool1]"
