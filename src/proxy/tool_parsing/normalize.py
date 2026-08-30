"""Tool argument normalization and OpenAI tool_calls construction.

⚠ INVARIANT — read src/proxy/tool_parsing/AGENTS.md "Tool-call argument normalization" before editing.
The client's incoming `tools[]` is the SINGLE source of truth for tool names and
argument param names. `normalize_tool_arguments()` reconciles LLM-degraded names
back to that schema via `valid_params`. NEVER add a case-folding / `.lower()`
fallback that bypasses `valid_params` — that silently rewrites opencode's
camelCase (`filePath`) to snake_case (`file_path`) and the client rejects the
call with SchemaError. This has been broken 5+ times the same way.
`valid_params=None` is legacy snake_case for internal memory tools only.
"""

import json
import logging
import re
import uuid
from typing import Optional, Set

logger = logging.getLogger("proxy.tool_parsing")

# Static tool name aliases for LLM hallucinations — names the model
# invents that have no relationship to what the client sent.
# Client-naming differences (read_file → read, etc.) are handled
# dynamically via canonicalize_tool_name() using the incoming tool list.
TOOL_NAME_ALIASES: dict[str, str] = {
    # LLM hallucinations (names the model invents with no relation to client tools)
    "todo_write": "todowrite",
    "execute_code": "bash",
    "search_files": "grep",
    # Client-naming edge cases the dynamic resolver can't handle
    "run_shell_command": "bash",
}

# Suffixes and prefixes stripped by the dynamic resolver when matching
# client tool names. E.g., client sent 'read', LLM said 'read_file' →
# strip '_file' → 'read' matches client name.
_DYNAMIC_SUFFIXES = ("_file", "_search", "_fetch", "_find", "_write", "_read")
_DYNAMIC_PREFIXES = ("file_", "run_shell_", "web_", "shell_")

_FUZZY_SEP = re.compile(r'[-_]')

# Proxy-internal control keys that are NEVER injected into the client tool
# schema. Their paraphrase aliases (mode → subagent_mode, type → subagent_type,
# etc.) MUST apply unconditionally regardless of valid_params so the proxy can
# read them. See H1 / src/proxy/tool_parsing/AGENTS.md "Tool-call argument normalization".
_PROXY_INTERNAL_CONTROL_KEYS = frozenset({"subagent_mode", "subagent_type"})


def _fuzzy(s):
    """Case- and separator-insensitive key: file_path == filePath == filepath."""
    if not isinstance(s, str):
        return s
    return _FUZZY_SEP.sub('', s.lower())


def build_valid_params_by_tool(tools_schema):
    """Map {tool_name: frozenset(real_param_names)} from an OpenAI tools[] array.

    Returns {} for missing/malformed input. Tolerates tools whose `parameters`
    is absent or has no `properties`.
    """
    result = {}
    if not isinstance(tools_schema, list):
        return result
    for t in tools_schema:
        try:
            fn = t.get("function", t) if isinstance(t, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            params = fn.get("parameters")
            props = params.get("properties") if isinstance(params, dict) else None
            if isinstance(name, str) and name and isinstance(props, dict) and props:
                result[name] = frozenset(k for k in props.keys() if isinstance(k, str))
        except (AttributeError, TypeError):
            continue
    return result


def extract_file_path(args):
    """Return a filesystem path from tool-call args, casing-agnostic.

    Tries snake_case `file_path`, camelCase `filePath` (opencode), and the bare
    `path` key (kilocode internal). Returns "" if none present or args is not a dict.
    """
    if not isinstance(args, dict):
        return ""
    return args.get("file_path") or args.get("filePath") or args.get("path") or ""


def _try_dynamic_match(name: str, valid_names: Set[str]) -> Optional[str]:
    """Try to match name against valid_names via suffix/prefix stripping."""
    for suffix in _DYNAMIC_SUFFIXES:
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            if base in valid_names:
                return base
    for prefix in _DYNAMIC_PREFIXES:
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            if remainder in valid_names:
                return remainder
    # Try: web_fetch → webfetch (remove underscore), glob_search → glob
    if "_" in name:
        collapsed = name.replace("_", "")
        if collapsed in valid_names:
            return collapsed
    return None


def canonicalize_tool_name(name: str, valid_names: Optional[Set[str]] = None) -> str:
    """Return the canonical tool name for the given client request.

    Resolution order:
    1. If name is already in valid_names (what the client sent), return as-is.
    2. If name is in the static TOOL_NAME_ALIASES, return the mapped name.
    3. Try stripping common suffixes (_file, _search, ...) and see if the
       resulting base name is in valid_names (dynamic client-name resolution).
    4. Fall back: return name unchanged.

    When valid_names is None (e.g., in tests or early bootstrap), only
    the static alias map is used.
    """
    if not isinstance(name, str):
        return name

    # Already valid for this client — no correction needed.
    if valid_names and name in valid_names:
        return name

    # Static alias (LLM hallucinations).
    canonical = TOOL_NAME_ALIASES.get(name)
    if canonical is not None and canonical != name:
        logger.info("Tool name auto-heal (static): %r → %r", name, canonical)
        return canonical

    # Dynamic: try stripping known client-specific suffixes/prefixes.
    # E.g., client sent 'read', LLM said 'read_file' → resolve to 'read'.
    if valid_names:
        match = _try_dynamic_match(name, valid_names)
        if match is not None:
            logger.info("Tool name auto-heal (dynamic): %r → %r", name, match)
            return match

    return name


def normalize_tool_arguments(tool_name, arguments, valid_params=None):
    """Normalize tool arguments to match expected parameter names.

    When valid_params (the set of real parameter names from the client tool
    schema) is provided, argument keys are reconciled against it: exact matches
    are preserved, and casing/separator drift (file_path vs filePath) plus
    synonym aliases are resolved to the real schema param name. When
    valid_params is None, the legacy snake_case synonym behavior is used
    (for internally-executed memory tools that have no client schema).
    """
    if not isinstance(arguments, dict):
        return arguments

    # SEE src/proxy/tool_parsing/AGENTS.md "Tool-call argument normalization": targets are INTERMEDIATE
    # snake_case tokens, fuzzy-resolved against valid_params at runtime. Do NOT
    # change a target's casing and do NOT add .lower() lookups outside _resolve().
    param_mappings = {
        "read": {
            "path": "file_path",
            "file": "file_path",
            "filename": "file_path",
            "location": "file_path",
            "filepath": "file_path",
            "file_name": "file_path",
        },
        "write": {
            "path": "file_path",
            "file": "file_path",
            "filename": "file_path",
            "location": "file_path",
            "filepath": "file_path",
            "file_name": "file_path",
            "data": "content",
            "text": "content",
            "body": "content",
        },
        "edit": {
            "path": "file_path",
            "file": "file_path",
            "filename": "file_path",
            "location": "file_path",
            "filepath": "file_path",
            "file_name": "file_path",
            "new_content": "content",
            "text": "content",
            "data": "content",
            "old": "old_string",
            "find": "old_string",
            "search": "old_string",
            "new": "new_string",
            "replace": "new_string",
            "replacement": "new_string",
        },
        "bash": {
            "cmd": "command",
            "shell": "command",
            "script": "command",
            "execute": "command",
            "run": "command",
            "code": "command",
            "desc": "description",
            "explanation": "description",
            "purpose": "description",
            "dir": "workdir",
            "directory": "workdir",
            "cwd": "workdir",
        },
        "webfetch": {
            "url": "url",
            "link": "url",
            "address": "url",
            "uri": "url",
            "format": "format",
            "output": "format",
            "type": "format",
            "time": "timeout",
            "delay": "timeout",
            "wait": "timeout",
        },
        "grep": {
            "pattern": "pattern",
            "search": "pattern",
            "query": "pattern",
            "regex": "pattern",
            "match": "pattern",
            "text": "pattern",
            "files": "include",
            "filter": "include",
            "file_pattern": "include",
            "dir": "path",
            "directory": "path",
            "location": "path",
        },
        "task": {
            "instruction": "prompt",
            "query": "prompt",
            "request": "prompt",
            "task": "prompt",
            "agent_type": "subagent_type",
            "agent": "subagent_type",
            "type": "subagent_type",
            "subagent_mode": "subagent_mode",
            "subagentmode": "subagent_mode",
            "agent_mode": "subagent_mode",
            "mode": "subagent_mode",
        },
    }

    # Canonicalize the tool name before lookup so that LLM-emitted variants
    # like "read_file", "write_file", "edit_file" resolve to their canonical
    # counterparts ("read", "write", "edit") and pick up the right mappings.
    tool_name = canonicalize_tool_name(tool_name, set(param_mappings.keys()))

    mappings = param_mappings.get(tool_name, {})

    fuzzy_real = {}
    if valid_params:
        for p in valid_params:
            if isinstance(p, str) and p:
                f = _fuzzy(p)
                if f in fuzzy_real and fuzzy_real[f] != p:
                    logger.warning("normalize_tool_arguments: fuzzy param collision %r vs %r for tool %r", p, fuzzy_real[f], tool_name)
                fuzzy_real[f] = p

    # SEE src/proxy/tool_parsing/AGENTS.md: resolver order is FIXED (exact → fuzzy → alias → passthrough).
    # valid_params is checked BEFORE any .lower() — never bypass it.
    #
    # H1: proxy-internal control keys (subagent_mode, subagent_type) are NOT in
    # the client tool schema — they are proxy-internal control keys injected by
    # the proxy. Their paraphrase aliases MUST apply regardless of valid_params
    # so canary.py:_resolve_task_canary (which reads args.get("subagent_mode"))
    # sees the normalized value. This special-case runs AFTER the exact-schema-
    # match check so it only fires when the schema lacks the key (the key is a
    # paraphrase that needs healing to the proxy-internal name). If the schema
    # legitimately contains `type`/`mode`, that exact match wins — the control-
    # key alias is a synonym-alias-class op and MUST run after exact match.
    def _resolve(key):
        if not isinstance(key, str):
            return key
        mapped = mappings.get(key)
        if mapped is None:
            lowered = key.lower().replace("-", "_")
            if lowered != key:
                mapped = mappings.get(lowered)
        if valid_params is not None and key in valid_params:
            return key
        if mapped is not None and mapped in _PROXY_INTERNAL_CONTROL_KEYS:
            return mapped
        if valid_params is not None:
            real = fuzzy_real.get(_fuzzy(key))
            if real is not None:
                return real
        if mapped is not None and mapped != key:
            if valid_params is not None:
                if mapped in valid_params:
                    return mapped
                real = fuzzy_real.get(_fuzzy(mapped))
                if real is not None:
                    return real
                return key
            return mapped
        return key

    return {_resolve(k): v for k, v in arguments.items()}


def provide_defaults_for_optional_params(tool_name, args, valid_params=None):
    """Provide default values for optional parameters that would otherwise cause validation errors."""
    if not isinstance(args, dict):
        return args

    defaults = {
        "bash": {"description": "shell command"},
        "task": {"subagent_type": "general"},
        "grep": {"path": "."},
    }

    if tool_name in defaults:
        fuzzy_real = {_fuzzy(p): p for p in (valid_params or []) if isinstance(p, str)} if valid_params else {}
        for param, default_value in defaults[tool_name].items():
            real = param
            if valid_params and param not in valid_params:
                real = fuzzy_real.get(_fuzzy(param), param)
            if real not in args:
                args[real] = default_value

    return args


def make_tool_calls(tool_calls_list, existing_ids=None, valid_params_by_tool=None):
    """Convert parsed tool call dicts into an OpenAI tool_calls array.

    ALWAYS pass `valid_params_by_tool` (built from the client's tools[] via
    build_valid_params_by_tool) for any array returned to the client — otherwise
    arg names are not reconciled to the client schema and opencode SchemaErrors.
    See src/proxy/tool_parsing/AGENTS.md "Tool-call argument normalization".
    """
    if not tool_calls_list:
        return None

    result = []
    schema_names = set((valid_params_by_tool or {}).keys())
    for i, tool_call in enumerate(tool_calls_list):
        name = canonicalize_tool_name(tool_call["name"], schema_names)
        valid_params = (valid_params_by_tool or {}).get(name)
        args = tool_call["arguments"]

        if isinstance(args, dict):
            args = normalize_tool_arguments(name, args, valid_params=valid_params)
            args = provide_defaults_for_optional_params(name, args, valid_params=valid_params)
            args = json.dumps(args, sort_keys=True)

        tc_id = existing_ids[i] if existing_ids and i < len(existing_ids) else f"call_{uuid.uuid4().hex[:24]}"

        result.append(
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        )
    return result
