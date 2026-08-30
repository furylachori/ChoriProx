"""^^Command system — detection, dispatch, and built-in commands."""

import threading
import time
from dataclasses import dataclass
from typing import Callable

from proxy.openai_compat import build_openai_response

__all__ = [
    "COMMAND_PREFIX",
    "COMMAND_REGISTRY",
    "COMMAND_HELP",
    "CommandContext",
    "register",
    "is_command",
    "parse_command",
    "make_command_response",
    "dispatch_command",
]

COMMAND_PREFIX = "^^"

COMMAND_REGISTRY: dict[str, Callable] = {}
COMMAND_HELP: dict[str, str] = {}
_registry_lock = threading.RLock()


@dataclass
class CommandContext:
    raw_text: str
    session_id: str
    # Bug #7: threaded from RequestContext.client_hint at dispatch.py build
    # site so ^^ws/<name> can populate workspace_store._client_active via
    # the command path (not just session-level activation).
    client_hint: str = ""


def register(name: str, help_text: str = ""):
    """Decorator to register a command handler."""
    def decorator(fn):
        with _registry_lock:
            if name in COMMAND_REGISTRY:
                import warnings
                warnings.warn(f"Command '{name}' is already registered. Previous handler will be overwritten.", UserWarning, stacklevel=2)
            COMMAND_REGISTRY[name] = fn
            COMMAND_HELP[name] = help_text
        return fn
    return decorator


def is_command(text: str) -> bool:
    """Return True if text starts with COMMAND_PREFIX and has more characters after it."""
    return text.startswith(COMMAND_PREFIX) and len(text) > len(COMMAND_PREFIX)


def parse_command(text: str) -> tuple[str, list[str]]:
    """Parse a command string into (command_name, args_list).

    The command name is lowercased for case-insensitive matching.
    """
    parts = text.strip().split()
    return parts[0].lower(), parts[1:]


def make_command_response(content: str, model: str = "skill") -> dict:
    """Build a synthetic OpenAI chat completion response with HTML-escaped content."""
    ts = int(time.time())
    return build_openai_response(content, model=model, response_id=f"chatcmpl-cmd-{ts}", created=ts)


def dispatch_command(ctx: CommandContext) -> dict:
    """Look up and execute the command handler, or return an unknown-command response."""
    name, args = parse_command(ctx.raw_text)
    with _registry_lock:
        handler = COMMAND_REGISTRY.get(name)
    if handler is None:
        return make_command_response(
            f"Unknown command: {name}. Type ^^help for available commands.",
            model="^^command",
        )
    return handler(ctx, args)


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------

@register("^^help", "List commands and skills")
def cmd_help(ctx, args):
    """List all registered commands and skills per spec §8.1.

    Output format:
        Available Commands:

        Commands:
        - ^^help — List commands and skills
        - ^^status — Show proxy and custom-skills status
        - ^^workspace — Manage workspace context
          aliases: ^^ws

        Skills:
        - ^^ticket — Create and manage tasks [builtin]
        - ^^work — Execute a selected task [builtin]
        - ^^build — Build the project [custom]
          aliases: ^^t

    Custom skills are loaded from PROXY_SKILLS_DIR (default <repo>/skills).
    """
    lines = ["Available Commands:\n"]
    lines.append("Commands:")
    for name, text in sorted(COMMAND_HELP.items()):
        lines.append(f"- {name} — {text}")
    try:
        from proxy.skills.skill_registry import get_default_registry
        registry = get_default_registry()
        skills = registry.list_skills()
    except Exception:
        skills = []
    if skills:
        lines.append("")
        lines.append("Skills:")
        for skill in sorted(skills, key=lambda s: s.trigger):
            tag = "[builtin]" if getattr(skill, "is_builtin", False) else "[custom]"
            lines.append(f"- {skill.trigger} — {skill.description} {tag}")
            aliases = getattr(skill, "aliases", ()) or ()
            if aliases:
                lines.append(f"  aliases: {', '.join(aliases)}")
    return make_command_response("\n".join(lines), model="^^command")


@register("^^status", "Show proxy health and stats")
def cmd_status(ctx, args):
    """Show proxy health and stats."""
    from proxy.state import request_counter, tool_cache_size, tool_cache_hits, \
        tool_cache_misses, chat_histories, log_overflow_count
    from proxy.model_registry import _model_registry
    model_count = len(_model_registry.list_models()) if _model_registry else 0
    total = tool_cache_hits + tool_cache_misses
    hit_pct = f"{tool_cache_hits / total * 100:.0f}%" if total > 0 else "n/a"
    lines = [
        "Proxy Status:\n",
        f"- Models loaded: {model_count}",
        f"- Total requests: {request_counter}",
        f"- Sessions in memory: {len(chat_histories)}",
        f"- Tool schemas cached: {tool_cache_size} "
        f"({tool_cache_hits} reuses, {tool_cache_misses} first-loads, {hit_pct} hit rate)",
        f"- Log lines dropped (queue overflow): {log_overflow_count}",
    ]
    # Custom Skills block (Wave 2, spec §8.2) — concise; detailed per-file
    # issues belong in /health and startup logs, not in ^^status.
    try:
        from proxy.skills.skill_registry import get_default_registry
        cs = get_default_registry().custom_skills_status()
        swarm_enabled = bool(getattr(cs, "swarm_enabled", False))
        swarm_capable_skills = tuple(getattr(cs, "swarm_capable_skills", ()) or ())
        lines.append("")
        lines.append("Custom Skills:")
        lines.append(f"- Status: {cs.state}")
        lines.append(f"- Loading: {'enabled' if cs.enabled else 'disabled'}")
        if cs.directory:
            lines.append(f"- Directory: {cs.directory}")
        lines.append(f"- Discovered: {cs.discovered}")
        if cs.loaded_names:
            lines.append(f"- Loaded: {len(cs.loaded_names)} ({', '.join(cs.loaded_names)})")
        else:
            lines.append("- Loaded: 0")
        lines.append("- Restart: required for changes to take effect" if cs.restart_required else "- Restart: not required")
        # Swarm operator-visibility (spec AC 16.15).
        lines.append(f"- Swarm-enabled: {'yes' if swarm_enabled else 'no'}")
        if swarm_capable_skills:
            names = ", ".join(swarm_capable_skills)
            lines.append(
                f"- Swarm-capable skills: {len(swarm_capable_skills)} skill(s): {names}"
            )
        else:
            lines.append("- Swarm-capable skills: 0")
    except Exception:
        lines.append("")
        lines.append("Custom Skills: unavailable — registry not initialized")
    return make_command_response("\n".join(lines), model="^^command")


@register("^^context", "Set the context budget the proxy enforces (in thousands of tokens)")
def cmd_context(ctx, args):
    """Set or report the per-session reduce-only context budget (in thousands of tokens).

    Args:
        ctx: CommandContext (only ctx.session_id is usable; dispatch runs
            before route resolution so model_cfg is not known here).
        args: list of string arguments.

    Returns:
        A synthetic OpenAI response dict (no upstream round-trip).
    """
    if len(args) > 1:
        return make_command_response(
            "Usage: ^^context <N>  (200..1000, in thousands of tokens). "
            "Example: ^^context 500 sets a 500,000-token budget.",
            model="^^command",
        )

    if not args:
        # Report current budget or model default.
        from proxy.agent.agent_registry import registry as agent_registry
        agent = agent_registry.get_or_load(ctx.session_id)
        override = agent.get_context_limit_override() if agent else None
        if override is None:
            return make_command_response(
                "Context budget: model default (no override set). "
                "Use ^^context <N> to set one (200..1000, in thousands of tokens).",
                model="^^command",
            )
        return make_command_response(
            f"Context budget: {override:,} tokens (capped at the model's context window if smaller).",
            model="^^command",
        )

    # One arg: parse and validate.
    raw = args[0]
    try:
        n = int(raw)
    except (ValueError, TypeError):
        return make_command_response(
            f"Invalid value '{raw}': ^^context requires an integer between 200 and 1000.",
            model="^^command",
        )
    if n < 200 or n > 1000:
        return make_command_response(
            f"Value {n} out of range: ^^context requires 200 <= N <= 1000 "
            f"(got {n}, would be {n*1000:,} tokens).",
            model="^^command",
        )

    tokens = n * 1000
    # Lazy imports to avoid circular dependency with proxy.pipeline.dispatch.
    from proxy.agent.agent_registry import registry as agent_registry
    agent = agent_registry.get_or_load(ctx.session_id)
    if agent is None:
        from memory.db import touch_session
        touch_session(ctx.session_id)
        agent = agent_registry.get_or_load(ctx.session_id)
    if agent is None:
        return make_command_response(
            "Error: could not load or create an agent for this session.",
            model="^^command",
        )
    agent.set_context_limit_override(tokens)
    return make_command_response(
        f"Context budget set to {tokens:,} tokens (capped at the model's context window if smaller).",
        model="^^command",
    )


# ---------------------------------------------------------------------------
# Workspace commands — registered via import (R2-Issue 3 restoration)
# ---------------------------------------------------------------------------
# Importing the module runs the @register decorators inside it, adding
# `^^workspace` and `^^ws` (alias) to COMMAND_REGISTRY / COMMAND_HELP.
# Kept at the bottom of __init__.py so the register() helper, CommandContext,
# and make_command_response are defined before the import side effects run.
from commands import workspace_commands as _workspace_commands  # noqa: E402,F401