"""ContextOptimizeStage — context budget, mode resolution, cascade fallback."""

import copy
import logging
from typing import Any, Optional

from proxy.pipeline.exceptions import StageAbort

logger = logging.getLogger("proxy.pipeline.stages")


_PROPAGATION_NOT_SET = object()


class ContextOptimizeStage:
    """System prompt injection, context window management.

    Uses:
    - enforce_context_limits() from context.py
    - apply_mode() from proxy/mode.py
    """

    def execute(self, ctx: Any) -> Optional[dict]:
        from proxy.agent.agent_registry import registry as agent_registry
        from proxy.context import ContextLimitError

        # Spec §11.3 — request-local immutable policy reference. Reset at
        # the START of every execute() so:
        #   1. reused test contexts and stale carrier contexts cannot
        #      leak a previous request's policy.
        #   2. fallback / cascade paths leave the field at None.
        ctx.swarm_policy = None
        ctx.context_limit_override = None

        logger.debug(
            "[DIAG-CTX-OPT] entry: ctx.is_subagent=%s ctx.inherited_skill=%r ctx.subagent_ticket_mode=%s",
            getattr(ctx, 'is_subagent', None),
            getattr(ctx, 'inherited_skill', None),
            getattr(ctx, 'subagent_ticket_mode', None),
        )

        model_cfg = getattr(ctx, "model_cfg", None)

        try:
            from proxy.context import enforce_context_limits
            enforce_context_limits(ctx.body, model_cfg)
        except ValueError as e:
            logger.error("ContextOptimizeStage: Context limit exceeded: %s", e,
                         extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            raise StageAbort(
                ({"error": {"message": str(e), "type": "context_limit_exceeded"}}, 413)
            )
        except Exception as e:
            logger.warning("ContextOptimizeStage: enforce_context_limits failed (non-fatal): %s", e,
                           extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})

        agent = agent_registry.get_or_load(ctx.session_id)

        # DIAG: log which path we'll take. agent=None means get_or_load()
        # returned no Agent (e.g. a brand-new sub-agent session that has
        # never been registered and has no SQLite history). In that case
        # we skip resolve_mode() entirely and fall into the DEFAULT cascade
        # fallback below — losing the SKILL_SUBAGENT mode that the sub-agent
        # should have inherited.
        logger.debug(
            "[DIAG-CTX-OPT-AGENT] agent is None? %s | agent_id=%s | session_id=%s | is_subagent=%s | inherited_skill=%r | subagent_ticket_mode=%s",
            agent is None,
            getattr(agent, "agent_id", None) if agent is not None else None,
            ctx.session_id,
            getattr(ctx, 'is_subagent', None),
            getattr(ctx, 'inherited_skill', None),
            getattr(ctx, 'subagent_ticket_mode', None),
        )

        # Sub-agent auto-registration fallback: a sub-agent request can
        # carry a brand-new session_id that has no Agent in the registry
        # AND no chat history in SQLite (the LRU cache and the DB both
        # come up empty). Without an Agent, resolve_mode() never runs and
        # we drop into the DEFAULT cascade — losing the SKILL_SUBAGENT
        # mode and the proxy_ticket tool. Auto-register a fresh Agent for
        # the sub-agent's session_id so the primary resolve_mode/apply_mode
        # path can run. Trust assumption: ctx.is_subagent is set by the
        # ingress parser from request metadata that the parent agent
        # controls, so this is not an unauthenticated registration path.
        if agent is None and getattr(ctx, "is_subagent", False):
            from proxy.agent.agent import Agent
            from proxy.skill_lifecycle import register_lifecycle_callbacks
            try:
                new_agent = Agent(
                    session_id=ctx.session_id,
                    agent_id=ctx.session_id[:12],
                    is_subagent=True,
                )
            except Exception as agent_ctor_err:
                logger.error(
                    "ContextOptimizeStage: failed to construct Agent for "
                    "sub-agent session %s: %s",
                    ctx.session_id[:12], agent_ctor_err,
                    extra={"request_id": getattr(ctx, 'request_id', '?'),
                           "session_id": ctx.session_id},
                )
            else:
                # Wire IoC deactivation callbacks (consistent with
                # _load_from_db and spawn paths).
                try:
                    register_lifecycle_callbacks(new_agent)
                except Exception:
                    logger.exception(
                        "ContextOptimizeStage: register_lifecycle_callbacks "
                        "failed for sub-agent session %s — continuing",
                        ctx.session_id[:12],
                    )
                # Register under the registry lock; on success, retry
                # get_or_load so the canonical lookup path returns it.
                try:
                    agent_registry.register(new_agent)
                except Exception as reg_err:
                    logger.error(
                        "ContextOptimizeStage: agent_registry.register "
                        "failed for sub-agent session %s: %s",
                        ctx.session_id[:12], reg_err,
                        extra={"request_id": getattr(ctx, 'request_id', '?'),
                               "session_id": ctx.session_id},
                    )
                else:
                    # Re-resolve so downstream code (and the DIAG log
                    # above) see the freshly registered agent.
                    agent = agent_registry.get_or_load(ctx.session_id) or new_agent
                    logger.info(
                        "ContextOptimizeStage: auto-registered Agent "
                        "agent_id=%s for sub-agent session_id=%s "
                        "(inherited_skill=%r subagent_ticket_mode=%s)",
                        getattr(agent, "agent_id", "?")[:12],
                        ctx.session_id[:12],
                        getattr(ctx, 'inherited_skill', None),
                        getattr(ctx, 'subagent_ticket_mode', None),
                    )

        from proxy.mode import (
            resolve_mode,
            apply_mode,
            Mode,
            ActiveSkillUnavailable,
            SkillRuntimeError,
        )

        # Deepcopy snapshot: apply_mode mutates body in-place. The snapshot
        # provides a clean body for the fallback paths and original messages
        # for MotivatorStage's turn-aware reminders.
        try:
            body_original = copy.deepcopy(ctx.body)
        except Exception as e:
            logger.error("ContextOptimizeStage: deepcopy of body failed: %s", e,
                         extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            raise StageAbort(
                ({"error": {"message": "Internal error preparing context", "type": "internal_error"}}, 500)
            )

        # --- Primary path: agent-aware mode resolution ---
        if agent is not None:
            # Sub-agents do not inherit the parent's context_limit_override;
            # the top-of-execute reset already covers sub-agents / reused ctx,
            # so this guard sets the override ONLY for non-subagent requests.
            if not getattr(ctx, "is_subagent", False):
                ctx.context_limit_override = agent.get_context_limit_override()

            # Capture the active mode BEFORE apply_mode runs so we can
            # compare-and-clear if apply_mode raises ActiveSkillUnavailable
            # or SkillRuntimeError (spec §7.2).
            expected_mode_snapshot = agent.snapshot_mode()
            try:
                mode = resolve_mode(ctx, agent)
                ctx.messages_original = copy.deepcopy(body_original.get("messages", []))
                logger.debug(
                    "[DIAG-CTX-OPT-PATH] primary path: about to apply_mode(mode=%s, agent_id=%s)",
                    mode, getattr(agent, "agent_id", None),
                )
                ctx.body = apply_mode(ctx, agent, ctx.body, mode)
                ctx.messages = copy.deepcopy(ctx.body.get("messages", []))

                # Capture the pre-upstream mode snapshot for CAS
                # compare-and-clear in PostProcessStage.finalize_skill_mode.
                # R2 #1 (TOCTOU at context_optimize.py:178): propagate the
                # exact ModeState identity that apply_mode read at
                # mode.py:170 — do NOT re-snapshot here. Two atomic reads
                # of `active_mode` open a window where a concurrent
                # replacement can be CAS-cleared (the F10 clobber).
                propagated = getattr(
                    ctx,
                    "_mode_snapshot_propagated_from_apply_mode",
                    _PROPAGATION_NOT_SET,
                )
                if propagated is _PROPAGATION_NOT_SET:
                    propagated = agent.snapshot_mode()
                ctx.mode_snapshot_pre_upstream = propagated

                # Spec §11.3 — install the immutable SubagentPolicy
                # reference AFTER apply_mode succeeds. Conditions:
                #   1. NOT a sub-agent (universal no-nesting rule §14; any
                #      sub-agent MUST keep swarm_policy=None even if its
                #      inherited skill has subagents configured).
                #   2. Active skill resolves in the registry.
                #   3. The active skill has a non-None subagents attribute.
                # The assignment is the exact frozen object published on
                # SkillDefinition — no copy, no dict conversion.
                _install_swarm_policy(ctx, agent)

                last_user_msg = ""
                for msg in reversed(ctx.body.get("messages", [])):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")
                        break
                logger.debug("[ContextOptimizeStage] after select_context: LAST user message (truncated 200) = %r", last_user_msg[:200] if isinstance(last_user_msg, str) else "")

                return None
            except StageAbort:
                raise
            except ActiveSkillUnavailable as exc:
                # Spec §7.2: active mode references a skill not in the
                # registered snapshot. HTTP 409 skill_state_conflict; the
                # underlying exception text MUST NOT be echoed to the
                # client. Mode finalization is owned by RequestPipeline
                # via finalize_skill_mode_on_exception (reads
                # ctx._mode_snapshot_propagated_from_apply_mode, set
                # inside apply_mode before the fail-closed check).
                logger.error(
                    "ContextOptimizeStage: active skill %r not in registry — returning 409",
                    exc.skill_name,
                    extra={"request_id": getattr(ctx, 'request_id', '?'),
                           "session_id": ctx.session_id},
                )
                raise StageAbort(({
                    "error": {
                        "message": "Active skill is not registered",
                        "type": "skill_state_conflict",
                    }
                }, 409))
            except SkillRuntimeError as exc:
                # Spec §7.2: prompt/tool-policy callback raised. HTTP 500
                # skill_runtime_error; the underlying exception text MUST
                # NOT be echoed to the client. Mode finalization is owned
                # by RequestPipeline via finalize_skill_mode_on_exception
                # (reads ctx._mode_snapshot_propagated_from_apply_mode,
                # set inside apply_mode before any skill callback runs).
                logger.error(
                    "ContextOptimizeStage: skill %r callback raised %s — returning 500",
                    exc.skill_name, exc.source,
                    extra={"request_id": getattr(ctx, 'request_id', '?'),
                           "session_id": ctx.session_id},
                )
                raise StageAbort(({
                    "error": {
                        "message": "Skill runtime error",
                        "type": "skill_runtime_error",
                    }
                }, 500))
            except Exception as e:
                logger.warning(
                    "ContextOptimizeStage: resolve_mode/apply_mode failed: %s", e,
                    extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id},
                )
                _raise_413_if_context_limit(e, ctx)
                # Cascade is RESTRICTED to genuinely mode-free requests
                # (spec §7.2). If an active mode is still installed, or
                # this is a skill-inheriting sub-agent, the cascade to
                # DEFAULT would silently drop the skill's prompt and tool
                # policy — re-raise instead.
                #
                # We use `agent.active_mode` directly here (racy under
                # concurrent updates, but this is a one-shot error-path
                # decision). `agent.snapshot_mode()` is the canonical
                # thread-safe read but it can return a default MagicMock
                # in tests where `active_mode` was explicitly set to None.
                # Trusting the explicit `active_mode` attribute gives the
                # correct semantic: a real ModeState carries a `.name`,
                # so we check that to distinguish from a default MagicMock.
                raw_active_mode = getattr(agent, "active_mode", None) if agent is not None else None
                has_active_mode = (
                    raw_active_mode is not None
                    and getattr(raw_active_mode, "name", None) is not None
                )
                # For inherited_skill / subagent_ticket_mode, use strict
                # type checks so auto-created MagicMock attributes (in
                # tests that don't set them) don't accidentally look truthy.
                inherited_skill_value = getattr(ctx, "inherited_skill", None)
                has_inherited_skill = (
                    isinstance(inherited_skill_value, str) and inherited_skill_value
                )
                subagent_ticket_mode = getattr(ctx, "subagent_ticket_mode", False)
                is_subagent_ticket = subagent_ticket_mode is True
                if (
                    has_active_mode
                    or has_inherited_skill
                    or is_subagent_ticket
                ):
                    logger.warning(
                        "ContextOptimizeStage: non-skill error %s with active "
                        "skill set — re-raising without cascade",
                        type(e).__name__,
                        extra={"request_id": getattr(ctx, 'request_id', '?'),
                               "session_id": ctx.session_id},
                    )
                    raise
                # Fall through to DEFAULT cascade fallback
                logger.debug(
                    "[DIAG-CTX-OPT-PATH] primary path raised, falling through to cascade fallback",
                )

        # --- Cascade fallback: DEFAULT mode → minimal → empty ---
        logger.debug(
            "[DIAG-CTX-OPT-PATH] cascade fallback path: agent_was_none=%s ctx.is_subagent=%s ctx.inherited_skill=%r ctx.subagent_ticket_mode=%s",
            agent is None,
            getattr(ctx, 'is_subagent', None),
            getattr(ctx, 'inherited_skill', None),
            getattr(ctx, 'subagent_ticket_mode', None),
        )
        ctx._mode = Mode.DEFAULT
        ctx.messages_original = copy.deepcopy(body_original.get("messages", []))
        _cascade_apply_default(ctx, body_original, agent_was_none=(agent is None), agent=agent)
        ctx.messages = copy.deepcopy(ctx.body.get("messages", []))
        return None


def _raise_413_if_context_limit(e: Exception, ctx: Any) -> None:
    """If `e` is ContextLimitError, raise StageAbort(413) immediately."""
    from proxy.context import ContextLimitError
    if isinstance(e, ContextLimitError):
        logger.error(
            "ContextOptimizeStage: Context limit exceeded: %s", e,
            extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
        raise StageAbort(
            ({"error": {"message": str(e), "type": "context_limit_exceeded"}}, 413)
        )


def _install_swarm_policy(ctx: Any, agent: Any) -> None:
    """Install the immutable SubagentPolicy on ctx after a successful apply_mode.

    Spec §11.3 — ContextOptimizeStage writes; the PostProcess swarm boundary
    reads. Conditions (any failure leaves ``ctx.swarm_policy = None``):

      1. NOT a sub-agent (universal no-nesting rule §14).
      2. Active skill resolves in the registry (ActiveSkillUnavailable is
         handled earlier by apply_mode; if we reach here, the skill is
         registered).
      3. The active skill has a non-None ``subagents`` attribute (i.e. a
         swarm-capable custom skill).

    The reference is the exact frozen object published on
    ``SkillDefinition.subagents`` — no copy, no deep-copy, no dict
    conversion. Downstream code uses ``is`` identity checks.

    Side effects: zero. Does not read any mutable state on the Agent;
    uses ``snapshot_mode()`` under lock #7.
    """
    # CONDITION 1: universal no-nesting — every sub-agent MUST keep
    # swarm_policy=None even if the inherited skill has subagents set.
    if getattr(ctx, "is_subagent", False):
        ctx.swarm_policy = None
        return

    # CONDITION 2: resolve the active skill name. agent may be None on
    # the cascade fallback path (not reached here; the early-return above
    # guards against it). snapshot_mode() is the canonical atomic read.
    skill = None
    if agent is not None:
        ms = agent.snapshot_mode()
        skill_name = getattr(ms, "name", None) if ms is not None else None
        if skill_name:
            # Canonical singleton accessor per src/proxy/skills/AGENTS.md.
            # `get_default_registry()` returns the unique process-wide
            # registry which is populated by `populate_builtins()` at
            # startup and extended by the custom-skill loader. The
            # SkillRegistry() __new__ shim also returns this singleton,
            # but the canonical helper name documents intent and
            # avoids the misleading "constructor call" pattern.
            from proxy.skills.skill_registry import get_default_registry
            registry = get_default_registry()
            skill = registry.get(skill_name)

    # CONDITION 3: swarm-capable custom skill. The loader rejected any
    # skill with a `subagents` key that failed structural validation, so
    # reaching here with skill.subagents non-None implies a published,
    # validated, frozen SubagentPolicy.
    policy = getattr(skill, "subagents", None) if skill is not None else None
    if policy is not None:
        ctx.swarm_policy = policy
    else:
        ctx.swarm_policy = None


def _cascade_apply_default(ctx: Any, body_original: dict, agent_was_none: bool = False, agent: Any = None) -> None:
    """Cascade fallback: DEFAULT mode → minimal (system+last) → empty."""
    from proxy.mode import apply_mode, Mode
    from proxy.context import ContextLimitError

    # DIAG: log invocation of cascade fallback. If agent_was_none=True and
    # ctx.is_subagent=True, this is the BUG PATH: resolve_mode() was skipped
    # because get_or_load() returned None for a new sub-agent session, so
    # we end up applying Mode.DEFAULT instead of Mode.SKILL_SUBAGENT.
    logger.debug(
        "[DIAG-CASCADE-INVOKED] _cascade_apply_default called: agent_was_none=%s ctx.is_subagent=%s ctx.inherited_skill=%r ctx.subagent_ticket_mode=%s session_id=%s",
        agent_was_none,
        getattr(ctx, 'is_subagent', None),
        getattr(ctx, 'inherited_skill', None),
        getattr(ctx, 'subagent_ticket_mode', None),
        getattr(ctx, 'session_id', '?'),
    )

    try:
        logger.debug(
            "[DIAG-CASCADE-APPLY] about to apply_mode(ctx, agent=%s, body_original, Mode.DEFAULT) — mode=DEFAULT",
            getattr(agent, "agent_id", None),
        )
        ctx.body = copy.deepcopy(apply_mode(ctx, agent, body_original, Mode.DEFAULT))
    except Exception as fallback_err:
        if isinstance(fallback_err, ContextLimitError):
            logger.error("ContextOptimizeStage: Context limit exceeded in cascade fallback: %s", fallback_err,
                         extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            raise StageAbort(
                ({"error": {"message": str(fallback_err), "type": "context_limit_exceeded"}}, 413)
            )
        # agent-None path: ValueError from apply_mode → 413 (P1-1)
        if agent_was_none and isinstance(fallback_err, ValueError):
            logger.error("ContextOptimizeStage: Context limit exceeded in cascade fallback (agent-None): %s", fallback_err,
                         extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            raise StageAbort(
                ({"error": {"message": str(fallback_err), "type": "context_limit_exceeded"}}, 413)
            )
        logger.error("ContextOptimizeStage: fallback also failed: %s", fallback_err,
                     extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
        try:
            from proxy.tool_parsing import OUR_SYSTEM_PROMPT as _FALLBACK_PROMPT
            # Walk backward to find the last user-role message (M-D2 fix).
            # Without this, a trailing tool result becomes the "last message"
            # and the model loses the user's actual instruction.
            last_user_msg = None
            for m in reversed(body_original.get("messages", []) or []):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user_msg = m
                    break
            sys_prompt_msg = {"role": "system", "content": _FALLBACK_PROMPT}
            minimal = [m for m in [sys_prompt_msg, last_user_msg] if m]
            ctx.body = {"messages": minimal}
            ctx.messages = minimal
            ctx.messages_original = copy.deepcopy(minimal)
        except Exception:
            logger.error("ContextOptimizeStage: minimal fallback failed — context is empty",
                         extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            ctx.body = {"messages": []}
            ctx.messages = []
            ctx.messages_original = []
