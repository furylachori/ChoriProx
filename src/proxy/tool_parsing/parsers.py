"""YAML, JSON, XML, and DSML tool call parsers."""

import json
import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def parse_yaml_tool_blocks(content):
    """Parse YAML-style tool calls from Qwen content.

    Format:
    tool_name:
      key1: value1
      key2: |
        multi-line
        content here

    Returns a list of {"name": ..., "arguments": ...} dicts.
    """
    tools = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        tool_match = re.match(r"^([a-zA-Z_]\w*):\s*$", line)
        if tool_match:
            tool_name = tool_match.group(1)
            known_tools = {
                # Old LLM names (auto-healed by canonicalize_tool_name)
                "write_file", "read_file", "edit_file", "file_glob",
                "run_shell_command", "web_fetch",
                # Canonical names
                "bash", "shell", "grep",
                "read", "write", "edit", "glob", "webfetch",
                # Memory / task tools
                "todowrite", "todo_write",
                "MiniMax_web_search",
                "task", "ask", "question",
            }
            if tool_name in known_tools:
                args = {}
                i += 1
                while i < len(lines):
                    arg_line = lines[i]
                    if not arg_line.startswith("  ") and arg_line.strip() != "":
                        break
                    if arg_line.strip() == "":
                        next_i = i + 1
                        while next_i < len(lines) and lines[next_i].strip() == "":
                            next_i += 1
                        if next_i < len(lines) and re.match(
                            r"^[a-zA-Z_]\w*:\s*$", lines[next_i]
                        ):
                            break
                        i += 1
                        continue

                    kv_match = re.match(r"^\s+(\w+):\s*(.*)", arg_line)
                    if kv_match:
                        key = kv_match.group(1)
                        value = kv_match.group(2).strip()

                        if value == "|":
                            value_lines = []
                            i += 1
                            while i < len(lines) and lines[i].startswith("    "):
                                value_lines.append(lines[i][4:])
                                i += 1
                            value = "\n".join(value_lines)
                        else:
                            i += 1

                        args[key] = value
                    else:
                        i += 1

                if args:
                    tools.append({"name": tool_name, "arguments": args})
                continue
        i += 1

    return tools


def _extract_balanced_json(text: str, max_depth: int = 20) -> list[str]:
    """Extract balanced JSON objects from text using a depth-limited scanner.

    Tracks string literals and escape sequences so that braces inside JSON
    string values (e.g. {"key": "val}ue"}) do not affect depth counting.
    """
    results = []
    for m in re.finditer(r'\{', text):
        depth, i = 0, m.start()
        start = i
        in_string = False
        escape_next = False
        while i < len(text) and depth < max_depth:
            char = text[i]
            if escape_next:
                escape_next = False
            elif char == '\\':
                escape_next = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        results.append(text[start:i + 1])
                        break
            i += 1
    return results


def parse_all_tools_from_content(content):
    """Parse ALL tool calls from Qwen's responses.
    Returns a list of {"name": ..., "arguments": ...} dicts.
    """
    if not content:
        return []

    content = content.strip()
    tools = []

    yaml_tools = parse_yaml_tool_blocks(content)
    tools.extend(yaml_tools)

    for dsml_match in re.finditer(
        r'<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>', content, re.DOTALL
    ):
        dsml_block = dsml_match.group(1)
        for invoke_match in re.finditer(
            r'<｜DSML｜invoke\s+name="([^"]+)">(.*?)</｜DSML｜invoke>', dsml_block, re.DOTALL
        ):
            tool_name = invoke_match.group(1)
            params_block = invoke_match.group(2)
            args = {}
            for param_match in re.finditer(
                r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="true">(.+?)</｜DSML｜parameter>', params_block, re.DOTALL
            ):
                args[param_match.group(1)] = param_match.group(2).strip()
            for param_match in re.finditer(
                r'<｜DSML｜parameter\s+name="([^"]+)">(.*?)</｜DSML｜parameter>', params_block, re.DOTALL
            ):
                raw = param_match.group(0)
                if 'string="true"' not in raw:
                    val = param_match.group(2).strip()
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    args[param_match.group(1)] = val
            if tool_name and args:
                tools.append({"name": tool_name, "arguments": args})

    try:
        root = ET.fromstring(content)
        if root.tag == "tools" and root.text:
            tool_data = json.loads(root.text.strip())
            if "name" in tool_data and "arguments" in tool_data:
                name = tool_data.get("name", "")
                if not name or not name.strip():
                    pass
                else:
                    tools.append(tool_data)
    except (ET.ParseError, json.JSONDecodeError):
        pass

    for tools_match in re.finditer(
        r"<tools>\s*(\{.*?\})\s*</tools>", content, re.DOTALL
    ):
        try:
            tool_data = json.loads(tools_match.group(1))
            if "name" in tool_data and "arguments" in tool_data:
                name = tool_data.get("name", "")
                if not name or not name.strip():
                    continue
                tools.append(tool_data)
        except json.JSONDecodeError:
            pass
    for func_match in re.finditer(
        r"<function=(\w+)>(.*?)</function>", content, re.DOTALL
    ):
        func_name = func_match.group(1)
        params_block = func_match.group(2)
        args = {}
        for param_match in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_block, re.DOTALL
        ):
            args[param_match.group(1)] = param_match.group(2).strip()
        if func_name and args:
            tools.append({"name": func_name, "arguments": args})

    for json_match in re.finditer(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL):
        try:
            tool_data = json.loads(json_match.group(1))
            if "name" in tool_data and "arguments" in tool_data:
                name = tool_data.get("name", "")
                if not name or not name.strip():
                    continue
                tools.append(tool_data)
        except json.JSONDecodeError:
            pass

    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    depth_limited_patterns = _extract_balanced_json(content)
    for json_str in depth_limited_patterns:
        try:
            tool_data = json.loads(json_str)
            if "name" in tool_data and "arguments" in tool_data:
                name = tool_data.get("name", "")
                if not name or not name.strip():
                    continue
                tools.append(tool_data)
        except json.JSONDecodeError:
            continue

    seen = set()
    unique_tools = []
    for tool in tools:
        try:
            tool_key = (tool["name"], json.dumps(tool["arguments"], sort_keys=True))
            if tool_key not in seen:
                seen.add(tool_key)
                unique_tools.append(tool)
        except (KeyError, TypeError):
            unique_tools.append(tool)

    if len(unique_tools) != len(tools):
        logger.info(f"Deduplicated: {len(tools)} → {len(unique_tools)} tools")

    return unique_tools


def parse_tool_from_content(content):
    """Parse tool calls from Qwen's XML/JSON/YAML responses embedded in content.

    Returns the FIRST tool call found, or None.
    For multiple tool calls, use parse_all_tools_from_content.
    """
    tools = parse_all_tools_from_content(content)
    return tools[0] if tools else None
