"""Qwen XML/JSON/YAML tool call parsing and normalization.

Backward-compatible re-exports from split submodules.
"""

from proxy.tool_parsing.prompts import (
    detect_tool_mode,
    OUR_SYSTEM_PROMPT,
    OUR_SUB_PROMPT,
    SKILL_SUBAGENT_PROMPT,
)
from proxy.tool_parsing.ticket_prompts import (
    TICKET_MAIN_PROMPT,
    TICKET_SUB_PROMPT,
    TICKET_SHARED_REFERENCE,
)
from proxy.tool_parsing.parsers import (
    parse_yaml_tool_blocks,
    parse_all_tools_from_content,
    parse_tool_from_content,
)
from proxy.tool_parsing.normalize import (
    normalize_tool_arguments,
    provide_defaults_for_optional_params,
    make_tool_calls,
    build_valid_params_by_tool,
    extract_file_path,
    TOOL_NAME_ALIASES,
    canonicalize_tool_name,
)

__all__ = [
    "detect_tool_mode",
    "OUR_SYSTEM_PROMPT",
    "OUR_SUB_PROMPT",
    "SKILL_SUBAGENT_PROMPT",
    "TICKET_MAIN_PROMPT",
    "TICKET_SUB_PROMPT",
    "TICKET_SHARED_REFERENCE",
    "parse_yaml_tool_blocks",
    "parse_all_tools_from_content",
    "parse_tool_from_content",
    "normalize_tool_arguments",
    "provide_defaults_for_optional_params",
    "make_tool_calls",
    "build_valid_params_by_tool",
    "extract_file_path",
    "TOOL_NAME_ALIASES",
    "canonicalize_tool_name",
]
