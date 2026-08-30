"""Tool schema caching and optimization.

Model configs (proxy_models.json) are validated at load time by
clients.model_registry.ModelRegistry._load — backend type, URL scheme,
name pattern, and ModelConfig ctor are all checked; invalid entries are
skipped with a WARNING. We do not re-validate model fields here. (M-B2
stale audit ref verified: validation IS in place, contrary to the audit
claim that 20 new entries have no validation.) The FATAL guard is in
model_registry._load itself (lines that raise on parse/construct errors).
"""
import hashlib
import json
import logging
from typing import Optional

from collections import OrderedDict

from proxy.state import (
    tool_cache,
    tool_cache_lock,
)
import proxy.state as state
from proxy.types import ToolingPayload
logger = logging.getLogger(__name__)

MAX_TOOL_CACHE_SIZE = 100

# Passthrough tool names — derived from ToolRegistry
try:
    from proxy.tool_pipeline.tool_registry import registry as _tool_reg
    _resolved_passthrough = _tool_reg.get_passthrough_tools()
    if not isinstance(_resolved_passthrough, (set, frozenset, list, tuple)):
        logger.warning(
            "tool_cache: get_passthrough_tools() returned %s (type %s); using hardcoded fallback",
            _resolved_passthrough, type(_resolved_passthrough).__name__,
        )
        PASSTHROUGH_TOOL_NAMES = {
            "todowrite",
            "ask",
            "question",
            "new_task",
            "TaskCreateTool",
            "TaskUpdateTool",
        }
    else:
        PASSTHROUGH_TOOL_NAMES = set(_resolved_passthrough)
except (ImportError, AttributeError) as e:
    logger.warning(
        "tool_cache: _tool_reg.get_passthrough_tools() unavailable (%s: %s), using hardcoded fallback",
        type(e).__name__, e,
    )
    PASSTHROUGH_TOOL_NAMES = {
        "todowrite",
        "ask",
        "question",
        "new_task",
        "TaskCreateTool",
        "TaskUpdateTool",
    }


def _get_injectable_memory_tools(skill=None, role="main"):
    """Return proxy-owned tool schemas available for the given skill+role.

    Resolves from capability_table.injectable_proxy_tools(role), then
    looks up each tool name in TICKET_TOOL_SCHEMAS (for ticket tools)
    or PROXY_WORK_TOOL_DEFINITION (for work tools). Unknown names are
    logged and skipped.

    If the skill has no capability table, inject nothing and log a warning.
    """
    from proxy.skills.skill_definition import CapabilityTable
    from memory.tools import PROXY_WORK_TOOL_DEFINITION
    from memory.tools.schemas import TICKET_TOOL_SCHEMAS

    if skill is None:
        return []

    table = getattr(skill, 'capability_table', None)
    if not isinstance(table, CapabilityTable):
        skill_name = getattr(skill, 'name', 'unknown')
        logger.warning(
            "Skill '%s' has no capability table; injecting no proxy tools.",
            skill_name,
        )
        return []

    injectable = table.injectable_proxy_tools(role)
    schemas = []
    for name in injectable:
        if name.startswith("proxy_ticket_"):
            schema = TICKET_TOOL_SCHEMAS.get(name)
            if schema:
                schemas.append(schema)
            else:
                logger.warning(
                    "Injectable tool '%s' not found in TICKET_TOOL_SCHEMAS; skipping.",
                    name,
                )
        elif name == "proxy_ticket":
            logger.warning(
                "Injectable tool 'proxy_ticket' (monolithic) requested; "
                "skipping (decomposed in 8b)."
            )
        elif name == "proxy_work":
            schemas.append(PROXY_WORK_TOOL_DEFINITION)
        else:
            logger.warning(
                "Unknown proxy tool '%s' from injectable_proxy_tools; skipping.",
                name,
            )
    return schemas


def prepare_tooling_payload(body: dict, model_cfg, ticket_mode: bool = False, is_subagent: bool = False, skill_name: Optional[str] = None, session_id: str = "", agent_id: str = "") -> ToolingPayload:
    """Filter, cache, and augment tool definitions."""
    logger.debug(
        "[DIAG-TOOLS-PREPARE] entry: ticket_mode=%s is_subagent=%s skill_name=%r session_id=%r agent_id=%r",
        ticket_mode, is_subagent, skill_name, session_id, agent_id,
    )
    from proxy.model_registry import _tool_parser_registry, USE_ADAPTERS, ACTIVE_CLIENT
    from proxy.tool_pipeline.tool_registry import registry as _tool_reg

    # --- Map ticket_mode to skill_name internally ---
    if ticket_mode and not skill_name:
        skill_name = "ticket"
        logger.debug("[DIAG-TOOLS-PREPARE] ticket_mode=True, setting skill_name='ticket'")

    # --- Skill Mode: use SkillRegistry to filter tools ---
    # Note: ticket_mode=True is now equivalent to skill_name="ticket"
    if skill_name:
        logger.debug("[DIAG-TOOLS-PREPARE] Entering skill mode path, skill_name=%r", skill_name)
        from proxy.skills.skill_registry import SkillRegistry, SkillContext

        registry = SkillRegistry()
        skill = registry.get(skill_name)

        if skill:
            # Build minimal SkillContext for tool filtering
            skill_ctx = SkillContext(
                session_id=session_id,
                agent_id=agent_id,
                messages=[],
                skill_state={},
                is_subagent=is_subagent,
            )
            tool_filter = skill.get_tool_filter(skill_ctx)

            if tool_filter is None:
                # Three-state policy (C6 fix): None = unrestricted. Do NOT
                # filter — fall through to the default adapter path below so
                # the standard OPENCODE_TOOL_NAMES filter applies to every
                # incoming tool. No cache write (no filter outcome to hash)
                # and no skill-mode log line (the filter was a no-op).
                logger.debug(
                    "[DIAG-TOOLS-PREPARE] tool_filter=None (unrestricted) for skill=%r; falling through to default path",
                    skill_name,
                )
            elif not tool_filter:
                # Three-state policy (C6 fix): empty frozenset = deny all.
                # Override body['tools'] to [] and short-circuit. No cache
                # write (no tools to hash). Emit the [SKILL_MODE-TOOLS]
                # log line with 0 tools so observability is identical to
                # the non-empty filter path.
                augmented_tools = []
                logger.info(
                    "[SKILL_MODE-TOOLS] Filtered to 0 tools using skill '%s' (deny-all): []",
                    skill_name,
                )
                logger.debug(
                    "[DIAG-TOOLS-PREPARE] return: count=0 names=[] cached=False path=skill_deny_all",
                )
                return ToolingPayload(
                    tools=augmented_tools,
                    tool_hash=None,
                    cached=False,
                    augmented_tools=augmented_tools,
                )
            else:
                _tools = body.get("tools", [])
                filtered_tools = [
                    t for t in _tools
                    if t.get("function", {}).get("name", t.get("name", "")) in tool_filter
                ]
                augmented_tools = list(filtered_tools)

                # In skill mode, append ONLY the search schema from MEMORY_TOOL_DEFINITIONS
                try:
                    from memory.tools import MEMORY_TOOL_DEFINITIONS
                    for canonical_tool in MEMORY_TOOL_DEFINITIONS:
                        name = canonical_tool.get("function", {}).get("name", "")
                        if name == "search":
                            augmented_tools.append(canonical_tool)
                except ImportError:
                    pass

                # Inject proxy-owned tools available in this skill (dedup against filtered + search)
                _existing_names = {
                    t.get("function", {}).get("name", "") for t in augmented_tools
                }
                _role = "subagent" if is_subagent else "main"
                _injected_schemas = _get_injectable_memory_tools(skill, _role)
                _injected_names = [
                    s.get("function", {}).get("name", "?") for s in _injected_schemas
                ]
                for schema in _injected_schemas:
                    name = schema.get("function", {}).get("name", "")
                    if name and name not in _existing_names:
                        augmented_tools.append(schema)
                        _existing_names.add(name)
                logger.debug(
                    "[DIAG-TOOLS-PREPARE] _get_injectable_memory_tools(skill_name=%r) returned %d schemas: %s",
                    skill_name, len(_injected_names), _injected_names,
                )

                # M-4: Hash the FILTERED+augmented tools, not the raw incoming
                # _tools list. Two requests with the same raw tool list but
                # different filter outcomes (skill vs default) would otherwise
                # produce the same cache hash → cross-filter collision.
                _tool_hash = _generate_tool_hash_with_mode(augmented_tools, ticket_mode=True, skill_name=skill_name)
                tool_hash = None
                if _tool_hash:
                    cached_tools = get_cached_tool_schema(_tool_hash)
                    if cached_tools is not None:
                        _expected_names = {t.get("function", {}).get("name", "") for t in augmented_tools}
                        _cached_name_set = {t.get("function", {}).get("name", "") for t in cached_tools}
                        if _cached_name_set == _expected_names and len(cached_tools) == len(augmented_tools):
                            logger.debug("[DIAG-TOOLS-CACHE] CACHE HIT hash=%s → %d tools: %s", _tool_hash, len(cached_tools), sorted(_cached_name_set))
                            logger.debug(
                                "[DIAG-TOOLS-PREPARE] return: count=%d names=%s cached=%s path=skill_cache_hit",
                                len(cached_tools),
                                sorted(_cached_name_set),
                                True,
                            )
                            _cached_tools = list(cached_tools)
                            return ToolingPayload(tools=_cached_tools, tool_hash=_tool_hash, cached=True, augmented_tools=_cached_tools)
                        else:
                            logger.debug(
                                "Cache miss (skill-mode validity) — tool name set mismatch for %s (mode=%s)",
                                skill_name, "ticket" if ticket_mode else "work",
                                extra={"request_id": None},
                            )
                    cache_tool_schema(_tool_hash, augmented_tools)
                    tool_hash = _tool_hash

                logger.info(
                    "[SKILL_MODE-TOOLS] Filtered to %d tools using skill '%s': %s",
                    len(augmented_tools),
                    skill_name,
                    [t.get("function", {}).get("name", t.get("name", "?")) for t in augmented_tools],
                )

                logger.debug(
                    "[DIAG-TOOLS-PREPARE] return: count=%d names=%s cached=%s path=skill_filtered",
                    len(augmented_tools),
                    [t.get("function", {}).get("name", t.get("name", "?")) for t in augmented_tools],
                    False,
                )
                return ToolingPayload(
                    tools=augmented_tools,
                    tool_hash=tool_hash,
                    cached=False,
                    augmented_tools=augmented_tools,
                )
    else:
        logger.debug("[DIAG-TOOLS-PREPARE] skill_name is None, entering DEFAULT adapter path")

    if "tools" in body:
        incoming_tools = body["tools"]
        incoming_tool_names = [
            t.get("function", {}).get("name", t.get("name", "?"))
            for t in incoming_tools
        ]

        _active_client = ACTIVE_CLIENT
        _default_tool_names = _tool_reg.get_default_tools()
        if _active_client and USE_ADAPTERS:
            OPENCODE_TOOL_NAMES = _active_client.get_tool_names()
        else:
            OPENCODE_TOOL_NAMES = _default_tool_names

        logger.debug(
            "[DIAG-TOOLS-FILTER] Using %s tool names from %s: %s",
            len(OPENCODE_TOOL_NAMES),
            "adapter" if (_active_client and USE_ADAPTERS) else "ToolRegistry",
            sorted(OPENCODE_TOOL_NAMES),
        )

        filtered_tools = [
            t
            for t in incoming_tools
            if t.get("function", {}).get("name", t.get("name", ""))
            in (OPENCODE_TOOL_NAMES | PASSTHROUGH_TOOL_NAMES)
        ]
        filtered_tool_names = [
            t.get("function", {}).get("name", t.get("name", ""))
            for t in filtered_tools
        ]

        logger.info(
            f"Incoming: {len(incoming_tools)} tools → Filtering to {len(filtered_tools)}"
        )
        logger.info(f"Keeping: {filtered_tool_names}")

        dropped_tool_names = set(incoming_tool_names) - set(filtered_tool_names)
        if dropped_tool_names:
            logger.warning(
                "Dropped %d tools not in allowlist: %s",
                len(dropped_tool_names),
                sorted(dropped_tool_names),
            )

        passthrough_found = [
            t.get("function", {}).get("name", t.get("name", ""))
            for t in incoming_tools
            if t.get("function", {}).get("name", t.get("name", ""))
            in PASSTHROUGH_TOOL_NAMES
        ]
        if passthrough_found:
            logger.info(
                "Passthrough client tools (will be deduplicated via OPENCODE_TOOL_NAMES "
                "filter in adapter): %s",
                passthrough_found,
            )
        augmented_tools = list(filtered_tools)

        # M-4: Hash the FILTERED+augmented tools, not the raw incoming list.
        # Two requests with the same raw tool list but different filter
        # outcomes (skill vs default) would otherwise produce the same
        # cache hash → cross-filter collision. Compute the hash here, after
        # augmented_tools exists, then look up the cache.
        tool_hash = _generate_tool_hash_with_mode(augmented_tools, ticket_mode=False)
        cached_tools = get_cached_tool_schema(tool_hash) if tool_hash else None

        if cached_tools is not None:
            _cached_names = [
                t.get("function", {}).get("name", t.get("name", "?"))
                for t in cached_tools
            ]
            logger.debug(
                "[DIAG-TOOLS-CACHE] CACHE HIT hash=%s → %d tools: %s",
                tool_hash, len(cached_tools), _cached_names,
            )
            logger.debug(
                "[DIAG-TOOLS-PREPARE] return: count=%d names=%s cached=%s path=default_cache_hit",
                len(cached_tools),
                _cached_names,
                True,
            )
            # Shallow-copy: get_cached_tool_schema returns a reference into the
            # shared cache; downstream reassignment/mutation of body["tools"]
            # must not corrupt the cached entry. Mirrors the non-cached path
            # (augmented_tools = list(filtered_tools)).
            _cached_tools = list(cached_tools)
            return ToolingPayload(
                tools=_cached_tools,
                tool_hash=tool_hash,
                cached=True,
                augmented_tools=_cached_tools,
            )

        if tool_hash:
            cache_tool_schema(tool_hash, augmented_tools)
            logger.info(
                f"Sending {len(augmented_tools)} tools to LLM (cache hash: {tool_hash})"
            )

        logger.debug(
            "[DIAG-TOOLS-PREPARE] return: count=%d names=%s cached=%s path=default_filtered",
            len(augmented_tools),
            [t.get("function", {}).get("name", t.get("name", "?")) for t in augmented_tools],
            False,
        )
        return ToolingPayload(
            tools=augmented_tools,
            tool_hash=tool_hash,
            cached=False,
            augmented_tools=augmented_tools,
        )

    logger.debug(
        "[DIAG-TOOLS-PREPARE] return: count=0 names=[] cached=False path=no_tools"
    )
    return ToolingPayload(
        tools=[],
        tool_hash=None,
        cached=False,
        augmented_tools=[],
    )


def _generate_tool_hash_with_mode(tools: list, *, ticket_mode: bool = False, skill_name: Optional[str] = None) -> Optional[str]:
    """Generate a cache hash that includes ticket_mode/skill_name to prevent cross-mode collisions."""
    import hashlib
    import json
    if not tools:
        return None

    canonical_tools = []
    for tool in tools:
        canonical_tools.append({
            "type": tool.get("type"),
            "function": {
                "name": tool.get("function", {}).get("name"),
                "description": tool.get("function", {}).get("description"),
                "parameters": tool.get("function", {}).get("parameters", {}),
            },
        })
    canonical_tools.sort(key=lambda x: x["function"]["name"])

    ticket_mode_bit = f":{skill_name}" if skill_name else (":ticket" if ticket_mode else "")
    tool_json = json.dumps(canonical_tools, sort_keys=True, separators=(",", ":")) + ticket_mode_bit
    return hashlib.sha256(tool_json.encode()).hexdigest()[:16]

def cache_tool_schema(tool_hash, tools):
    """Cache tool schema for future use."""
    if tool_hash and tools:
        if not validate_tool_schema(tools):
            logger.info(f"Skipping invalid tool schema for hash {tool_hash}")
            return

        with tool_cache_lock:
            if tool_hash not in tool_cache:
                tool_cache[tool_hash] = tools
                state.tool_cache_size += 1
            else:
                tool_cache[tool_hash] = tools

            if state.tool_cache_size > MAX_TOOL_CACHE_SIZE:
                oldest_key = next(iter(tool_cache))
                del tool_cache[oldest_key]
                state.tool_cache_size -= 1


def get_cached_tool_schema(tool_hash):
    """Get cached tool schema by hash."""
    with tool_cache_lock:
        if tool_hash in tool_cache:
            tool_cache.move_to_end(tool_hash)
            state.tool_cache_hits += 1
            return tool_cache[tool_hash]
        else:
            state.tool_cache_misses += 1
            return None

def validate_tool_schema(tools):
    """Validate that tool schema is properly formatted."""
    if not isinstance(tools, list):
        return False

    for tool in tools:
        if not isinstance(tool, dict):
            return False
        if "type" not in tool or tool["type"] != "function":
            return False
        if "function" not in tool:
            return False

        function = tool["function"]
        if not isinstance(function, dict):
            return False
        if "name" not in function or not function["name"]:
            return False
        if "description" not in function:
            return False
        if "parameters" not in function:
            return False

        params = function["parameters"]
        if not isinstance(params, dict):
            return False
        if "type" not in params or params["type"] != "object":
            return False
        if "properties" not in params or not isinstance(params["properties"], dict):
            return False
        if "required" in params:
            if not isinstance(params["required"], list):
                return False
            for field in params["required"]:
                if field not in params["properties"]:
                    return False

    return True


def restore_tool_payload(body, tool_hash, original_tools=None):
    """No-op: LLM server requires full tool arrays, not cached references."""
    if tool_hash:
        logger.info(f"Cache tracking active for hash {tool_hash}")
    return body


def clear_tool_cache():
    """Clear the tool cache and reset counters.

    IMPORTANT: Use state.tool_cache.clear() (mutate in-place), NOT
    tool_cache = {} (rebinds local variable, module-level dict unchanged).
    Integer counters mutated via 'import proxy.state as state' (see N2 rule).
    """
    with tool_cache_lock:
        tool_cache.clear()
        state.tool_cache_size = 0
        state.tool_cache_hits = 0
        state.tool_cache_misses = 0
