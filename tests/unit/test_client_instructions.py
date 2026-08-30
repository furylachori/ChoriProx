"""Unit tests for proxy.client_instructions (spec: client-instructions-preservation).

Pure tests for ``extract_client_instructions`` + ``format_instruction_section``.
No I/O, no state, no mocks — the functions under test are pure.
"""
import pytest

from proxy.client_instructions import (
    extract_client_instructions,
    format_instruction_section,
    _is_absolute_path,
)


# ---------------------------------------------------------------------------
# No-op / empty inputs
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty_pair():
    assert extract_client_instructions("") == ("", "")


def test_none_returns_empty_pair():
    assert extract_client_instructions(None) == ("", "")  # type: ignore[arg-type]


def test_no_blocks_returns_content_stripped():
    """When there are no instruction blocks, remaining = stripped content."""
    content = "  env facts\nskills text\n  "
    remaining, instr = extract_client_instructions(content)
    assert remaining == "env facts\nskills text"
    assert instr == ""


# ---------------------------------------------------------------------------
# Single block extraction
# ---------------------------------------------------------------------------

def test_single_agents_md_block_captured():
    content = (
        "Instructions from: /home/user/proj/AGENTS.md\n"
        "# Project Rules\n"
        "Use 4-space indentation.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: /home/user/proj/AGENTS.md" in instr
    assert "# Project Rules" in instr
    assert "Use 4-space indentation." in instr


def test_single_block_with_leading_env_stripped_in_remaining():
    content = (
        "ENV: production\n"
        "\n"
        "Instructions from: /home/user/proj/AGENTS.md\n"
        "Rules here.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == "ENV: production"
    assert "Instructions from: /home/user/proj/AGENTS.md" in instr
    assert "Rules here." in instr


# ---------------------------------------------------------------------------
# Multiple blocks
# ---------------------------------------------------------------------------

def test_two_blocks_both_captured_order_kept():
    content = (
        "Instructions from: /home/user/proj/AGENTS.md\n"
        "AGENTS rules.\n"
        "Instructions from: /home/user/proj/CLAUDE.md\n"
        "CLAUDE rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: /home/user/proj/AGENTS.md" in instr
    assert "Instructions from: /home/user/proj/CLAUDE.md" in instr
    assert "AGENTS rules." in instr
    assert "CLAUDE rules." in instr
    # Order kept
    assert instr.index("AGENTS rules.") < instr.index("CLAUDE rules.")
    # Blocks separated by blank line
    assert "AGENTS rules.\n\nInstructions from:" in instr


# ---------------------------------------------------------------------------
# In-content "Instructions from:" lines
# ---------------------------------------------------------------------------

def test_in_content_non_absolute_does_not_terminate_block():
    """An in-content `Instructions from: see docs` (non-absolute path) is
    treated as block content, not a header (R1 H6)."""
    content = (
        "Instructions from: /home/user/proj/AGENTS.md\n"
        "Some rules here.\n"
        "Instructions from: see docs\n"
        "More rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "see docs" in instr
    assert "More rules." in instr
    # Only ONE absolute-path header was matched (the non-absolute line is content)
    assert instr.count("Instructions from: /home/user/proj/") == 1


def test_in_content_absolute_terminates_block():
    """An in-content `Instructions from: /real/path/AGENTS.md` (absolute path
    at column 0) DOES terminate the block — documented v1 over-capture
    residual (R1 H6 narrow)."""
    content = (
        "Instructions from: /home/user/proj/AGENTS.md\n"
        "Some rules here.\n"
        "Instructions from: /real/path/AGENTS.md\n"
        "Other rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    # Two headers matched
    assert instr.count("Instructions from:") == 2
    assert "Some rules here." in instr
    assert "Other rules." in instr
    # First block closed, content before second header
    assert "Some rules here.\n\nInstructions from: /real/path" in instr


# ---------------------------------------------------------------------------
# CRLF normalization
# ---------------------------------------------------------------------------

def test_crlf_input_normalized():
    content = (
        "ENV facts.\r\n"
        "Instructions from: /home/user/proj/AGENTS.md\r\n"
        "CRLF rules.\r\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == "ENV facts."
    assert "Instructions from: /home/user/proj/AGENTS.md" in instr
    assert "CRLF rules." in instr
    # No CR in output
    assert "\r" not in instr
    assert "\r" not in remaining


# ---------------------------------------------------------------------------
# Windows / tilde / user-config paths (CRIT-N1: absolute-path guard, not
# filename allowlist)
# ---------------------------------------------------------------------------

def test_windows_drive_path_captured():
    content = (
        "Instructions from: C:\\Users\\me\\repo\\AGENTS.md\n"
        "Windows rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: C:\\Users\\me\\repo\\AGENTS.md" in instr
    assert "Windows rules." in instr


def test_windows_forward_slash_drive_path_captured():
    content = (
        "Instructions from: D:/code/proj/AGENTS.md\n"
        "Forward-slash drive rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: D:/code/proj/AGENTS.md" in instr


def test_tilde_path_captured():
    content = (
        "Instructions from: ~/proj/AGENTS.md\n"
        "Tilde rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: ~/proj/AGENTS.md" in instr
    assert "Tilde rules." in instr


def test_user_config_contributing_md_captured():
    """CRIT-N1: user-configured instruction files (not AGENTS.md/CLAUDE.md)
    MUST be preserved — the guard is absolute-path, NOT a filename allowlist."""
    content = (
        "Instructions from: /abs/path/CONTRIBUTING.md\n"
        "Contributing rules.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: /abs/path/CONTRIBUTING.md" in instr
    assert "Contributing rules." in instr


def test_relative_path_not_matched():
    """A relative path does not match — stays in remaining content."""
    content = (
        "Instructions from: relative/AGENTS.md\n"
        "Some text.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert instr == ""
    assert "Instructions from: relative/AGENTS.md" in remaining
    assert "Some text." in remaining


# ---------------------------------------------------------------------------
# Empty-body block
# ---------------------------------------------------------------------------

def test_empty_body_block_header_only():
    """An AGENTS.md with zero body emits a header-only block.
    The block string is header + newline (content_lines empty)."""
    content = "Instructions from: /home/user/proj/AGENTS.md\n"
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    # The header is present; instruction_text is non-empty (header remains)
    assert "Instructions from: /home/user/proj/AGENTS.md" in instr
    assert instr.strip() != ""


# ---------------------------------------------------------------------------
# Over-capture (v1 documented limitation)
# ---------------------------------------------------------------------------

def test_over_capture_trailing_content_rides_along():
    """Trailing non-block content after the last block rides along in the
    last block's content (documented v1 behavior)."""
    content = (
        "Instructions from: /home/user/proj/AGENTS.md\n"
        "Project rules.\n"
        "MCP: some mcp text that rides along.\n"
    )
    remaining, instr = extract_client_instructions(content)
    assert remaining == ""
    assert "Instructions from: /home/user/proj/AGENTS.md" in instr
    assert "Project rules." in instr
    # Trailing content captured in the block (over-capture)
    assert "MCP: some mcp text that rides along." in instr


# ---------------------------------------------------------------------------
# _is_absolute_path unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    ("/abs/path", True),
    ("/a", True),
    ("~/home/path", True),
    ("~/.config/file", True),
    ("C:\\Users\\me", True),
    ("D:/code/repo", True),
    ("/", False),          # bare slash
    ("~", False),          # bare tilde
    ("relative/path", False),
    ("AGENTS.md", False),
    ("see docs", False),
    ("", False),
    ("A:", False),         # drive letter but no separator
    ("AB", False),         # not a drive, not absolute
])
def test_is_absolute_path(path, expected):
    assert _is_absolute_path(path) is expected


# ---------------------------------------------------------------------------
# format_instruction_section
# ---------------------------------------------------------------------------

def test_format_instruction_section_wraps_text():
    escaped = "some escaped rules"
    result = format_instruction_section(escaped)
    assert '<CLIENT_PROJECT_INSTRUCTIONS priority="project-convention">' in result
    assert "</CLIENT_PROJECT_INSTRUCTIONS>" in result
    assert "STRICTLY SUBORDINATE" in result
    assert escaped in result


def test_format_instruction_section_does_not_re_escape():
    """The function takes PRE-ESCAPED text and must NOT re-escape it."""
    pre_escaped = "text with &lt;tag&gt; and &amp;"
    result = format_instruction_section(pre_escaped)
    assert "&lt;tag&gt;" in result
    assert "&amp;amp;" not in result  # no double-escape of ampersand


def test_format_instruction_section_structure_intact():
    """The wrapper is a well-formed tag with exactly one open and one close."""
    result = format_instruction_section("content")
    assert result.count("<CLIENT_PROJECT_INSTRUCTIONS") == 1
    assert result.count("</CLIENT_PROJECT_INSTRUCTIONS>") == 1
    assert result.startswith('<CLIENT_PROJECT_INSTRUCTIONS')
    assert result.endswith("</CLIENT_PROJECT_INSTRUCTIONS>")
