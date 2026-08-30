"""Dispatch extraction — Phase 2B skill activation pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from memory.task_graph import AmbiguousHashError
from proxy.skills.skill_definition import SkillValidationError
from proxy.state import command_rate_limiter, enhancer_rate_limiter, skill_rate_limiter, skill_ip_rate_limiter
from proxy.rate_limit_response import rate_limited_error_response
from proxy.utils import extract_text_content

logger = logging.getLogger("proxy.pipeline.dispatch")


# ---------------------------------------------------------------------------
# DispatchResult — the bridge between dispatch.py and app.py
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Result of a dispatch handler.

    Attributes:
        rewritten_text: The (possibly rewritten) user message text to send upstream.
                        None means the request was fully handled (no upstream call).
        response:        Plain response dict (or Flask JSON tuple). Set when handled=True.
                        Streaming skill responses are returned as plain dicts and
                        serialized by the canonical _keepalive_stream path in
                        proxy.app — dispatch does NOT own SSE framing.
        status_code:     HTTP status code for the response.
        handled:         True if the handler consumed the request fully (no upstream).
        skill:           The matched SkillDefinition (Phase 2B, set by new dispatcher path).
    """
    rewritten_text: Optional[str]
    response: Any = None
    status_code: int = 200
    handled: bool = False
    skill: Any = None


# ---------------------------------------------------------------------------
# Handler 1 — enhancer
# ---------------------------------------------------------------------------
#
# ^^imp is a planned/future feature. The infrastructure (enhancer_rate_limiter
# in state.py, dispatch hook in try_enhancer, enhancer package in src/enhancer/)
# exists but the actual prompt enhancement logic is not implemented. The
# feature is gated by `enhancer_enabled` which references a ProxyFeatures
# field that doesn't exist on the dataclass (see
# `src/clients/orchestrator_config.py:ProxyFeatures` — no `enhancer_enabled`
# attribute). Calls fall through to the command dispatcher as 'Unknown
# command' until the feature is built.
#
# The only thing that exists for this path is the rate limiter (Phase 3 P3.1,
# commit 2d815dd — added `enhancer_rate_limiter` to state.py). The feature
# itself remains unimplemented. The gate `getattr(features, "enhancer_enabled",
# False)` always returns False, so this entire code path is currently
# dead code at runtime.
#
# If you intend to wire up ^^imp: add `enhancer_enabled: bool = False` to
# the `ProxyFeatures` dataclass, set it to True in the relevant client
# config, and implement the prompt-enhancement logic in `dispatch_imp`.


def try_enhancer(ctx: Any, features: Any) -> Optional[DispatchResult]:
    """Handle IMP trigger via enhancer dispatch.

    **FUTURE FEATURE — NOT IMPLEMENTED.** ^^imp is a planned prompt-enhancer
    feature whose infrastructure (rate limiter, dispatch hook) is in place
    but whose core logic has not been written. The function exits early on
    the gate at line ~71+ (see the section docstring above) because
    `ProxyFeatures.enhancer_enabled` does not exist on the dataclass, so
    `getattr(features, "enhancer_enabled", False)` always returns False.

    Returns DispatchResult if triggered, None otherwise. Currently always
    returns None.
    """
    if not getattr(features, "enhancer_enabled", False):
        return None

    text = extract_text_content(ctx.messages[-1].get("content", "")) if ctx.messages else ""
    enh_result = enhancer_rate_limiter.check_only(ctx.session_id)
    if not enh_result.allowed:
        logger.warning(
            "[ENHANCER_RATE_LIMIT] Session %s exceeded enhancer rate limit",
            ctx.session_id[:8],
        )
        body, status_code = rate_limited_error_response("Rate limit exceeded for ^^imp", int(enh_result.retry_after))
        return DispatchResult(
            rewritten_text=None,
            response=body,
            status_code=status_code,
            handled=True,
        )
    try:
        from enhancer import is_imp_trigger, dispatch_imp
        if is_imp_trigger(text.strip()):
            # M6 #2: commit the rate-limit slot only AFTER the
            # trigger matched (i.e. we're going to dispatch). If the
            # client floods without ^^imp, no slot is consumed.
            enhancer_rate_limiter.check(ctx.session_id)
            resp = dispatch_imp(ctx.session_id, text)
            return DispatchResult(
                rewritten_text=None,
                response=resp,
                status_code=200,
                handled=True,
            )
    except Exception as e:
        exc_name = type(e).__name__
        if any(
            kw in exc_name.lower()
            for kw in ["timeout", "apierror", "parseerror", "circuitbreaker", "value"]
        ):
            logger.warning("^^imp SOTA failure: %s", e)
            from enhancer.core import format_error
            return DispatchResult(
                rewritten_text=None,
                response=format_error("Enhancement unavailable, sending original"),
                status_code=200,
                handled=True,
            )
        raise  # All other exceptions bubble up

    return None


# ---------------------------------------------------------------------------
# Handler 3 — command dispatch
# ---------------------------------------------------------------------------

def try_command(ctx: Any, features: Any) -> Optional[DispatchResult]:
    """Handle command dispatch — rate-limit + dispatch_command.

    Returns DispatchResult if triggered, None otherwise.
    """
    if not getattr(features, "intercepts_commands", False):
        return None

    text = extract_text_content(ctx.messages[-1].get("content", "")) if ctx.messages else ""
    try:
        from commands import is_command
    except ImportError:
        logger.exception("try_command: Failed to import 'commands' module — ^^ commands are BROKEN")
        return None

    if not is_command(text.strip()):
        return None

    # --- Rate limit check ---
    cmd_result = command_rate_limiter.check(ctx.session_id)
    if not cmd_result.allowed:
        logger.warning(
            "[CMD_RATE_LIMIT] Session %s exceeded command rate limit",
            ctx.session_id[:8],
        )
        body, status_code = rate_limited_error_response("Rate limit exceeded for commands", int(cmd_result.retry_after))
        return DispatchResult(
            rewritten_text=None,
            response=body,
            status_code=status_code,
            handled=True,
        )

    try:
        from commands import dispatch_command
        from commands import CommandContext

        # Bug #7: thread client_hint from RequestContext so ^^ws <name>
        # reaches workspace_store.activate and populates _client_active.
        # Type-validate (mirrors SkillContext.from_request_context) so a
        # non-string attr (e.g. an unset MagicMock in tests) coerces to "".
        _raw_client_hint = getattr(ctx, "client_hint", "")
        cmd_client_hint = _raw_client_hint if isinstance(_raw_client_hint, str) else ""

        cmd_ctx = CommandContext(
            raw_text=text.strip(),
            session_id=ctx.session_id,
            client_hint=cmd_client_hint,
        )
        resp = dispatch_command(cmd_ctx)

        return DispatchResult(
            rewritten_text=None,
            response=resp,
            status_code=200,
            handled=True,
        )
    except Exception as e:
        logger.exception("Command dispatch failed: %s", e)
        return DispatchResult(
            rewritten_text=None,
            response={
                "error": {
                    "message": "Command execution failed",
                    "type": "command_error",
                }
            },
            status_code=500,
            handled=True,
        )


# ---------------------------------------------------------------------------
# Skill dispatch helpers (Phase B)
# ---------------------------------------------------------------------------

def _get_agent(ctx: Any) -> Any:
    """Load the agent for the current session.

    If the agent doesn't exist (new session), create the session in SQLite
    and retry. This allows skill commands to work on first request.
    """
    from proxy.agent.agent_registry import registry as agent_registry
    agent = agent_registry.get_or_load(ctx.session_id)
    if agent is None:
        # Session may not exist yet (first request). Create it and retry.
        from memory.db import touch_session as db_touch_session
        db_touch_session(ctx.session_id)
        agent = agent_registry.get_or_load(ctx.session_id)
    return agent


# ---------------------------------------------------------------------------
# Handler 2 (Phase B) — skill activation
# ---------------------------------------------------------------------------

def try_skill_activation(ctx, features) -> Optional[DispatchResult]:
    """Phase 2B: skill activation handler.

    Detects skill triggers (^^ticket, ^^workspace, etc.) in the last user
    message, activates the skill, and returns a DispatchResult with the
    rewritten text.
    """
    # Sub-agents must not trigger skill activation — that's the parent agent's role.
    if getattr(ctx, "is_subagent", False):
        return None

    from proxy.skill_dispatcher import SkillDispatcher as NewSkillDispatcher
    from proxy.skills.skill_registry import SkillRegistry

    text = extract_text_content(ctx.messages[-1].get("content", "")) if ctx.messages else ""
    registry = SkillRegistry()
    dispatcher = NewSkillDispatcher(registry)

    # Get agent (needed for dispatch).
    agent = _get_agent(ctx)

    # C4 §3.4: rate-limit BEFORE dispatch installs the mode (otherwise a 429'd
    # request leaves a zombie mode on the agent). detect_command has no side
    # effects — it just returns the matched SkillDefinition or None.
    detected_skill = dispatcher.detect_command(text)
    if detected_skill is not None:
        session_result = skill_rate_limiter.check(ctx.session_id)
        if not session_result.allowed:
            logger.warning(
                "[SKILL_RATE_LIMIT] Session %s exceeded skill rate limit",
                ctx.session_id[:8],
            )
            body, status_code = rate_limited_error_response(
                "Rate limit exceeded for skill activation",
                int(session_result.retry_after),
            )
            return DispatchResult(
                rewritten_text=None,
                response=body,
                status_code=status_code,
                handled=True,
            )
        remote_addr = getattr(ctx, 'remote_addr', '')
        if remote_addr:
            ip_result = skill_ip_rate_limiter.check(remote_addr)
            if not ip_result.allowed:
                logger.warning(
                    "[SKILL_RATE_LIMIT_IP] IP %s exceeded skill IP rate limit",
                    remote_addr[:16],
                )
                body, status_code = rate_limited_error_response(
                    "Rate limit exceeded for skill activation",
                    int(ip_result.retry_after),
                )
                return DispatchResult(
                    rewritten_text=None,
                    response=body,
                    status_code=status_code,
                    handled=True,
                )

    # Dispatch — always assign result so exception handlers below can reference it.
    result = None
    try:
        result = dispatcher.dispatch(text, ctx, agent)
    except AmbiguousHashError:
        return DispatchResult(
            rewritten_text=None,
            response={
                "error": {
                    "message": "Ambiguous hash — be more specific",
                    "type": "ambiguous_hash",
                }
            },
            status_code=409,
            handled=True,
        )
    except (ValueError, PermissionError) as _skill_err:
        # F1: a SkillValidationError carries an explicit http_status (e.g.
        # 404 for an unknown task hash from WorkSkill.on_activate). Honor
        # it BEFORE the generic 500 path so expected, user-correctable
        # errors return the right 4xx instead of a server error. Genuine,
        # unexpected ValueErrors still fall through to 500 below.
        if isinstance(_skill_err, SkillValidationError):
            logger.info(
                "Skill validation error for session=%s: %s",
                getattr(ctx, 'session_id', '?')[:8], _skill_err,
            )
            from proxy.skills.skill_registry import error_response
            _err_resp = error_response(str(_skill_err), _skill_err.http_status)
            # F3: strip the internal _status_code so it does not leak to the
            # client JSON. The status is carried by DispatchResult.status_code.
            _err_resp.pop("_status_code", None)
            return DispatchResult(
                rewritten_text=None,
                response=_err_resp,
                status_code=_skill_err.http_status,
                handled=True,
            )
        skill_for_log = getattr(result, 'skill', None) if result else None
        logger.exception("Skill '%s' parse_subcommand failed", skill_for_log.name if skill_for_log else '?')
        return DispatchResult(
            rewritten_text=None,
            response={
                "error": {
                    "message": "Skill activation failed",
                    "type": "skill_error",
                }
            },
            status_code=500,
            handled=True,
        )
    except Exception as _dispatch_err:
        logger.exception("Skill dispatch via SkillDispatcher failed: %s", _dispatch_err)
        return DispatchResult(
            rewritten_text=None,
            response={
                "error": {
                    "message": "Command handling failed",
                    "type": "dispatch_error",
                }
            },
            status_code=500,
            handled=True,
        )

    skill = getattr(result, 'skill', None)

    if skill is not None:
        logger.info(
            "[SKILL_MATCH] skill=%r session=%s last_msg_len=%d",
            skill.name, getattr(ctx, 'session_id', '?')[:8], len(text),
        )

    # If no skill matched, fall through immediately (no rate limit, no further processing).
    if skill is None:
        return None

    # Log [SKILL_HANDLED] when handle_direct handled the request.
    if getattr(result, 'handled', False):
        action = getattr(result, 'action', None)
        logger.info(
            "[SKILL_HANDLED] skill=%r action=%r session=%s",
            skill.name if skill else '?', action, getattr(ctx, 'session_id', '?')[:8],
        )

    rewritten_text = None
    if getattr(result, 'has_rewrite', False):
        # Agent is required for activation — return 500 if missing.
        if agent is None:
            logger.warning("Cannot activate skill '%s': agent not found for session", skill.name)
            return DispatchResult(
                rewritten_text=None,
                response={
                    "error": {
                        "message": "Skill activation failed: agent not found",
                        "type": "skill_error",
                    }
                },
                status_code=500,
                handled=True,
            )
        from proxy.skills.skill_definition import SkillRequestContext
        skill_ctx = SkillRequestContext.from_request_context(ctx)
        skill_ctx.skill_state = {"action": getattr(result, 'action', ''), "arg": getattr(result, 'arg', '') or ""}
        skill_ctx.raw_user_text = text
        # Spec §7.3: if `get_activation_rewrite` raises AFTER activation
        # succeeded, the mode is installed on the agent but no rewrite
        # is available. Roll back via compare-and-clear using the snapshot
        # SkillDispatcher.dispatch() captured before activation, then
        # return 500 so DispatchStage short-circuits the request.
        expected_mode = getattr(result, 'expected_mode', None)
        try:
            rewritten_text = skill.get_activation_rewrite(skill_ctx)
        except (ValueError, PermissionError):
            # Re-raise security exceptions after rolling back the mode
            # so the caller can still surface them upstream.
            if expected_mode is not None:
                try:
                    agent.clear_active_mode(expected_mode=expected_mode)
                except Exception as _rb_err:
                    logger.warning(
                        "dispatch: rollback clear_active_mode after security exception failed: %s",
                        _rb_err,
                    )
            raise
        except Exception as _rewrite_err:
            logger.warning(
                "skill.get_activation_rewrite failed for %r: %s — rolling back activation",
                skill.name if skill else '?', _rewrite_err,
            )
            if expected_mode is not None:
                try:
                    agent.clear_active_mode(expected_mode=expected_mode)
                except Exception as _rb_err:
                    logger.warning(
                        "dispatch: rollback clear_active_mode failed: %s", _rb_err,
                    )
            return DispatchResult(
                rewritten_text=None,
                response={
                    "error": {
                        "message": "Skill activation rewrite failed",
                        "type": "skill_runtime_error",
                    }
                },
                status_code=500,
                handled=True,
            )
        if not rewritten_text:
            rewritten_text = getattr(ctx, 'messages', [{}])[-1].get("content", "") if ctx.messages else text
    elif skill is not None and not getattr(result, 'handled', False):
        # Skill matched but activation failed — return 500.
        # (Only fires when activation was needed but has_rewrite=False.)
        logger.warning("Cannot activate skill '%s': activation failed", skill.name)
        return DispatchResult(
            rewritten_text=None,
            response={
                "error": {
                    "message": f"Skill activation failed: could not activate '{skill.name}'",
                    "type": "skill_error",
                }
            },
            status_code=500,
            handled=True,
        )

    ctx._skill_activated = getattr(result, 'has_rewrite', False)
    if getattr(result, 'has_rewrite', False):
        logger.info(
            "[SKILL_ACTIVATED] skill=%r session=%s",
            skill.name if skill else '?',
            getattr(ctx, 'session_id', '?')[:8],
        )

    skill_response = getattr(result, 'response', None)
    if isinstance(skill_response, dict):
        response = skill_response
    elif skill_response is None:
        response = {
            "error": {
                "message": "Skill dispatch returned no response",
                "type": "skill_error",
            }
        }
    else:
        response = skill_response

    # Determine status code:
    # - error_response() embeds `_status_code` on the response dict (e.g.
    #   404 for an unknown task hash, 400 for a missing arg). Honor it so
    #   the skill error reaches the client with the intended HTTP status.
    # - Otherwise: 500 for error-only responses, 200 for handled success,
    #   None for unhandled (caller falls through).
    if isinstance(response, dict):
        embedded_status = response.get("_status_code")
        if isinstance(embedded_status, int):
            status_code = embedded_status
        elif "error" in response and "choices" not in response:
            status_code = 500
        else:
            status_code = 200 if getattr(result, 'handled', False) else None
        # F3: strip the internal _status_code so it does not leak to the
        # client JSON body. The status is carried by DispatchResult.status_code
        # (and threaded to StageAbort by DispatchStage.execute).
        response.pop("_status_code", None)
    else:
        status_code = 200 if getattr(result, 'handled', False) else None

    return DispatchResult(
        rewritten_text=rewritten_text,
        response=response,
        status_code=status_code,
        handled=getattr(result, 'handled', False),
        skill=skill,
    )



# ---------------------------------------------------------------------------
# Public API — run_dispatch
# ---------------------------------------------------------------------------

# Handlers in priority order (lower int = higher priority)
# Priority 3 is reserved for future use (e.g., skill-specific middleware)
_DISPATCH_HANDLERS: dict[int, Any] = {
    1: try_enhancer,
    2: try_skill_activation,
    # 3 reserved for future skill-specific middleware
    4: try_command,
}


def run_dispatch(ctx: Any, features: Any) -> Optional[DispatchResult]:
    """Run the priority dispatch loop.

    Iterates handlers in priority order (enhancer → task_plan → commands)
    and returns the first non-None DispatchResult. Returns None if no handler
    matched the current request.
    """
    if not (features and ctx.messages):
        return None

    last_msg = ctx.messages[-1]
    if not (isinstance(last_msg, dict) and last_msg.get("role", "").lower() == "user"):
        return None

    for _priority, handler in sorted(_DISPATCH_HANDLERS.items()):
        result = handler(ctx, features)
        if result is not None:
            logger.info(
                "[DISPATCH] Handler %s returned DispatchResult(handled=%s, has_rewrite=%s) — session=%s",
                handler.__name__, result.handled,
                result.rewritten_text is not None,
                getattr(ctx, 'session_id', '?')[:8],
            )
            return result

    return None
