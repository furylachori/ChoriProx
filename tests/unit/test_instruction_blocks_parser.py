"""Unit tests for extract_instruction_blocks + _split_tool_content_for_tier2.

Asserts exact equality against the Wave 0 canonical fixture in
tests/fixtures/tier2_payload.py. Revert-test performed (see report below).
"""
from proxy.client_instructions import (
    _split_tool_content_for_tier2,
    extract_instruction_blocks,
)
from tests.fixtures.tier2_payload import (
    FILE_BODY,
    INSTRUCTION_BLOCKS,
    RAW_TOOL_BODY_WITH_LITERAL_CLOSE,
    RAW_TOOL_CONTENT,
    TIER2_SUFFIX,
)


class TestExtractInstructionBlocks:
    def test_exact_against_fixture(self):
        assert extract_instruction_blocks(TIER2_SUFFIX) == INSTRUCTION_BLOCKS

    def test_empty_string(self):
        assert extract_instruction_blocks("") == []

    def test_none_input(self):
        assert extract_instruction_blocks(None) == []

    def test_three_blocks_strict_boundary(self):
        text = (
            "Instructions from: /abs/a/AGENTS.md\n"
            "body-a-line-1\n"
            "body-a-line-2\n"
            "Instructions from: /abs/b/AGENTS.md\n"
            "body-b-line-1\n"
            "Instructions from: /abs/c/AGENTS.md\n"
            "body-c-line-1\n"
        )
        result = extract_instruction_blocks(text)
        assert len(result) == 3
        assert result[0] == ("/abs/a/AGENTS.md", "body-a-line-1\nbody-a-line-2")
        assert result[1] == ("/abs/b/AGENTS.md", "body-b-line-1")
        assert result[2] == ("/abs/c/AGENTS.md", "body-c-line-1\n")

    def test_non_header_line_between_blocks(self):
        text = (
            "Instructions from: /abs/a/AGENTS.md\n"
            "body-a\n"
            "garbage line that is not a header\n"
            "Instructions from: /abs/b/AGENTS.md\n"
            "body-b\n"
        )
        result = extract_instruction_blocks(text)
        assert len(result) == 2
        assert result[0] == ("/abs/a/AGENTS.md", "body-a\ngarbage line that is not a header")
        assert result[1] == ("/abs/b/AGENTS.md", "body-b\n")


class TestSplitToolContentForTier2:
    def test_exact_against_fixture(self):
        file_content, suffix = _split_tool_content_for_tier2(RAW_TOOL_CONTENT)
        assert file_content == FILE_BODY
        assert suffix == TIER2_SUFFIX

    def test_file_content_endswith_close_tag(self):
        file_content, suffix = _split_tool_content_for_tier2(RAW_TOOL_CONTENT)
        assert file_content.endswith("</content>")

    def test_no_boundary_returns_none(self):
        content = "<path>/abs/x.py</path><type>file</type><content>\nplain body</content>"
        file_content, suffix = _split_tool_content_for_tier2(content)
        assert file_content == content
        assert suffix is None

    def test_literal_close_not_followed_by_system_reminder(self):
        file_content, suffix = _split_tool_content_for_tier2(RAW_TOOL_BODY_WITH_LITERAL_CLOSE)
        assert file_content == RAW_TOOL_BODY_WITH_LITERAL_CLOSE
        assert suffix is None
