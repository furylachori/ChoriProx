"""Mode resolution and application for proxy request processing.

Centralizes mode logic (ticket, sub-agent, review, adversary) into:
- `resolve_mode` — picks the operating mode for the request.
- `apply_mode` — pre-upstream prompt + tool filtering.
- `classify_turn_response` / `TurnDisposition` — single-source classifier
  for terminal disposition of one LLM turn (spec §7.4).
- `finalize_skill_mode` — compare-and-clear the active mode based on
  the disposition (called from PostProcessStage.execute()'s `finally`
  block per spec §7.5).

See docs/specs/custom-skills-remediation.md §7.4 / §7.5 for the
post-process split.
"""

import enum
import logging

from proxy.pipeline.exceptions import StageAbort

logger = logging.getLogger(__name__)


SWARM_RUNTIME_STATUS = {
    "swarm_max_calls_exceeded": 400,
    "swarm_subagent_cannot_spawn": 403,
    "swarm_missing_secret": 500,
    "swarm_signing_failed": 500,
    "swarm_policy_unavailable": 500,
    "swarm_malformed_task_call": 502,
    "swarm_state_too_large": 500,
}


def swarm_error(*, code: str, message: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": "custom_skill_swarm_error",
            "param": "tool_calls",
            "code": code,
        }
    }


def _raise_swarm_abort(*, code: str, status: int, message: str) -> None:
    raise StageAbort((swarm_error(code=code, message=message), status))


class TurnDisposition(enum.Enum):
    """Disposition of a single LLM turn response (spec §7.4).

    CONTINUE: the turn is part of a multi-step orchestration; the skill
              mode (if any) is preserved for the next call.
    COMPLETE: the turn is terminal with a successful finish_reason; the
              skill mode is cleared.
    FAILED:   the turn is terminal but terminated abnormally (content
              filter, malformed response, unknown reason); the skill
              mode is cleared and any canary/processing is canceled.
    """

    CONTINUE = enum.auto()
    COMPLETE = enum.auto()
    FAILED = enum.auto()


# Finish reason constant (STY-010)
FINISH_REASON_STOP = "stop"


class ActiveSkillUnavailable(RuntimeError):
    """The active mode references a skill not present in the registered snapshot.

    Per spec §7.2, this is a fail-closed condition: the request is rejected
    with HTTP 409 skill_state_conflict and the active mode is cleared (compare-
    and-clear against the request's expected mode snapshot).

    Carries the missing skill name for diagnostic logging.
    """

    def __init__(self, skill_name: str, message: str = ""):
        self.skill_name = skill_name
        super().__init__(
            message or f"Active skill {skill_name!r} is not registered"
        )


class SkillRuntimeError(RuntimeError):
    """A skill callback (prompt, tool policy, or activation rewrite) raised.

    Per spec §7.2, this is a fail-closed condition: the request is rejected
    with HTTP 500 skill_runtime_error and the active mode is cleared. The
    underlying exception text is NOT echoed to the client.
    """

    def __init__(self, skill_name: str, source: str, original: BaseException):
        self.skill_name = skill_name
        self.source = source  # "prompt", "tool_policy", "rewrite", etc.
        self.original = original
        super().__init__(
            f"Skill {skill_name!r} {source} callback raised: "
            f"{type(original).__name__}"
        )


class Mode(enum.Enum):
    """Operating mode for a request.

    Each mode determines the system prompt, tool set, and post-processing.
    """

    DEFAULT = enum.auto()  # Standard coding assistant
    SKILL = enum.auto()  # Main session, skill active
    SUBAGENT = enum.auto()  # Sub-agent, no skill inheritance
    SKILL_SUBAGENT = enum.auto()  # Sub-agent, inherited skill mode


_MODE_MAP = {str(m.value): m for m in Mode}


def resolve_mode(ctx, agent) -> Mode:
    """Resolve the operating mode for this request.

    Called from ContextOptimizeStage. Sets ctx._mode as a side effect.
    """
    from proxy.skill_resolver import resolve_mode as _new_resolve_mode
    from proxy.skills.skill_registry import SkillRegistry
    registry = SkillRegistry()
    mode_str = _new_resolve_mode(ctx, agent, registry)
    # Map the string return to Mode enum
    resolved = _MODE_MAP.get(mode_str, Mode.DEFAULT)
    ctx._mode = resolved
    return resolved


def apply_mode(ctx, agent, body, mode) -> dict:
    """Apply mode-specific prompt, tool filtering, and context injection.

    Called from ContextOptimizeStage. Returns modified body dict.

    Per spec §7.2 (fail-closed):
    - If `agent.snapshot_mode()` returns a ModeState whose name is not in
      the registry, raise `ActiveSkillUnavailable`. The caller
      (ContextOptimizeStage) compare-and-clears the snapshot before
      re-raising as HTTP 409 skill_state_conflict.
    - If any skill callback (assemble_system_prompt, select_tools, etc.)
      raises, convert the exception to `SkillRuntimeError`. The caller
      compare-and-clears before re-raising as HTTP 500 skill_runtime_error.
    - `ContextLimitError` propagates unchanged so the caller can emit
      HTTP 413 context_limit_exceeded.

    For mode-free requests (agent is None or mode is DEFAULT/SUBAGENT
    without an active skill snapshot), no skill callbacks run; apply_mode
    proceeds normally.

    NOTE: 413-wiring (context-limit → HTTP 413) is delegated to the
    module-level helper `_raise_413_if_context_limit` in
    `proxy.pipeline.stage_modules.context_optimize`. (M-A2 — ref corrected
    post-Tier-4 split.)
    """
    from proxy.skill_prompts import assemble_system_prompt
    from proxy.skill_tools import select_tools
    from proxy.skills.skill_registry import SkillRegistry
    from proxy.tool_cache import prepare_tooling_payload
    from proxy.tool_pipeline.tool_registry import ToolMeta
    from proxy.context import select_context, ContextLimitError

    registry = SkillRegistry()
    # agent may be None on cascade fallback paths; tolerate here.
    ms = agent.snapshot_mode() if agent is not None else None
    skill_name = getattr(ms, 'name', None) if ms else "<none>"

    # R2 #1 (TOCTOU at context_optimize.py:178): propagate the exact
    # ModeState identity that apply_mode used for the fail-closed
    # check + tool/prompt work, so the caller can store it on
    # `ctx.mode_snapshot_pre_upstream` without a second atomic read.
    # The value is the identity (object `is` check) that PostProcess's
    # CAS compares against; a second snapshot would open a window where
    # a concurrent replacement could be CAS-cleared (the F10 clobber).
    # None here means either no agent or no active mode — the caller's
    # None guard must NOT trigger an unconditional clear.
    #
    # Assignment is UNCONDITIONAL. `RequestContext` is a non-frozen
    # dataclass (types.py:108), so the instance attribute is created on
    # first set — a guard like `if hasattr(ctx, "...")` would silently
    # no-op on a fresh ctx (no MagicMock, no pre-declared field) and
    # re-open the TOCTOU window the comment below claims to close.
    ctx._mode_snapshot_propagated_from_apply_mode = ms

    # FAIL-CLOSED (spec §7.2): verify the active skill exists in the
    # registered snapshot BEFORE any skill callback runs. If not, raise
    # ActiveSkillUnavailable. The caller (ContextOptimizeStage) is
    # responsible for compare-and-clearing the expected_mode_snapshot
    # before re-raising as HTTP 409 skill_state_conflict.
    if ms is not None and mode in (Mode.SKILL, Mode.SKILL_SUBAGENT):
        active_skill = registry.get(ms.name)
        if active_skill is None:
            logger.error(
                "apply_mode: active skill %r not in registry (fail-closed)",
                ms.name,
            )
            raise ActiveSkillUnavailable(ms.name)

    try:
        # Inject tools via prepare_tooling_payload
        model_cfg = getattr(ctx, 'model_cfg', None)
        is_ticket_mode = mode in (Mode.SKILL, Mode.SKILL_SUBAGENT)
        is_subagent = mode in (Mode.SUBAGENT, Mode.SKILL_SUBAGENT)
        tooling = prepare_tooling_payload(
            body, model_cfg,
            ticket_mode=is_ticket_mode,
            is_subagent=is_subagent,
            skill_name=skill_name,
            session_id=getattr(ctx, 'session_id', ''),
            agent_id=getattr(agent, 'agent_id', '') if agent else '',
        )
        if tooling.augmented_tools:
            body["tools"] = tooling.augmented_tools
        # Apply skill tool filter via select_tools
        # Build name → original-dict map before filtering so the full tool
        # schema (description, parameters) is preserved for each filtered tool.
        tool_by_name = {
            t.get("function", {}).get("name", ""): t
            for t in (body.get("tools", []) or [])
            if isinstance(t, dict)
        }
        tools_meta = select_tools(agent, ctx, body, registry)
        # Convert ToolMeta objects back to dict format expected by the pipeline,
        # looking up the original schema by name. Non-ToolMeta objects (e.g.
        # SkillDefinition) pass through unchanged.
        final_tools = []
        for t in tools_meta:
            if isinstance(t, ToolMeta):
                schema = tool_by_name.get(t.name)
                if schema is not None:
                    final_tools.append(schema)
                else:
                    logger.warning(
                        "Skipping tool %r: no matching client schema in body[\"tools\"]",
                        t.name,
                    )
            else:
                final_tools.append(t)
        body["tools"] = final_tools
        logger.info(
            "[FINAL-TOOLS] count=%d names=%s mode=%s skill=%s",
            len(body.get("tools", [])),
            [t.get("function", {}).get("name", t.get("name", "?")) for t in body.get("tools", [])],
            mode.name,
            skill_name,
        )
        # Capture tools snapshot (after all filtering)
        ctx._tools_snapshot = body.get("tools", [])
        # Build and merge system prompt
        prompt = assemble_system_prompt(agent=agent, ctx=ctx, body=body, registry=registry)
        # Merge the prompt into body["messages"][0] with role awareness
        if "messages" in body and len(body["messages"]) > 0:
            if isinstance(body["messages"][0], dict) and body["messages"][0].get("role") == "system":
                body["messages"][0]["content"] = prompt
            else:
                body["messages"].insert(0, {"role": "system", "content": prompt})

        # Prune the context window via the V2 three-tier pipeline.
        # This was stripped in c9b6755 and must be restored — without it,
        # the full unpruned message history goes upstream (4133 msgs /
        # 9.2MB instead of ~711 msgs / 1.3MB), breaking the budget pipeline.
        # select_context runs AFTER the system prompt is merged so the
        # prompt counts against the token budget; _select_context_v2
        # preserves the leading system message and only prunes
        # user/assistant/tool messages.
        logger.debug(
            "[DIAG-APPLY-MODE] before select_context: message count=%d",
            len(body.get("messages", [])),
        )
        model_name_for_context = getattr(ctx, "model_name", None)
        optimized = select_context(
            getattr(ctx, 'session_id', ''),
            body["messages"],
            model_name_for_context,
            model_cfg=model_cfg,
            is_subagent=getattr(ctx, 'is_subagent', False),
            ctx=ctx,
            tools=body.get("tools"),
        )
        if optimized is not None:
            body["messages"] = optimized
        logger.debug(
            "[DIAG-APPLY-MODE] after select_context: message count=%d",
            len(body.get("messages", [])),
        )
        return body
    except ActiveSkillUnavailable:
        # Already wrapped; pass through to caller.
        raise
    except ContextLimitError:
        # ContextLimitError must propagate to ContextOptimizeStage for 413
        # handling. Do NOT wrap as SkillRuntimeError.
        raise
    except Exception as e:
        # Skill callback failure (assemble_system_prompt, select_tools,
        # prepare_tooling_payload, etc.). Wrap so the caller can
        # compare-and-clear the snapshot and return HTTP 500
        # skill_runtime_error. The original exception is preserved on
        # SkillRuntimeError.original for diagnostic logging; the response
        # envelope MUST NOT echo it to the client.
        logger.error(
            "apply_mode: skill %r callback raised %s: %s",
            skill_name, type(e).__name__, e,
        )
        raise SkillRuntimeError(skill_name, source="apply_mode", original=e)


def classify_turn_response(data) -> TurnDisposition:
    """Classify a single LLM turn response per spec §7.4.

    Rules (in order):
    - data is not a dict OR has no "choices" key OR choices is empty ->
      FAILED (malformed).
    - choices[0] is not a dict -> FAILED (malformed).
    - finish_reason == "content_filter" -> FAILED (spec §7.4). Checked
      BEFORE tool_calls: a provider-redacted completion stays fail-closed
      even if it carries tool calls.
    - message.tool_calls is a non-empty list -> CONTINUE (the fix:
      client-executable tool calls win over a contradictory
      "stop"/None/unknown finish_reason; providers such as MiniMax emit
      finish_reason="stop" alongside tool_calls). The classifier must be
      correct on its own, not depend on the tool pipeline normalizing
      finish_reason to "tool_calls" upstream. A non-list "tool_calls"
      (string/dict) or an empty list is NOT a continuation.
    - finish_reason == "stop" -> COMPLETE.
    - finish_reason == "tool_calls" -> CONTINUE.
    - finish_reason == "length" -> CONTINUE (truncated but resumable).
    - finish_reason is None -> COMPLETE (default; no termination signal
      received but no continuation signal either).
    - any other non-null finish_reason -> FAILED (unknown reason).

    The is_final_response check in post_process.py:80 (memory sync) is
    INDEPENDENT and unchanged — it gates SOTA background extraction on
    finish_reason in ("stop", "length", "content_filter"), not on the
    TurnDisposition.
    """
    if not isinstance(data, dict):
        return TurnDisposition.FAILED
    choices = data.get("choices") or []
    if not choices:
        return TurnDisposition.FAILED
    first = choices[0]
    if not isinstance(first, dict):
        return TurnDisposition.FAILED
    finish_reason = first.get("finish_reason")
    message = first.get("message")
    if not isinstance(message, dict):
        message = {}
    raw_tool_calls = message.get("tool_calls")
    has_tool_calls = isinstance(raw_tool_calls, list) and len(raw_tool_calls) > 0

    # content_filter is fail-closed even when tool calls are present: a
    # provider-redacted completion must not drive a continuation.
    if finish_reason == "content_filter":
        return TurnDisposition.FAILED
    # Tool-calls-first: a response carrying client-executable tool calls is a
    # continuation regardless of a contradictory finish_reason (providers such as
    # MiniMax emit finish_reason="stop" alongside tool_calls). The classifier must
    # be correct on its own, not depend on the tool pipeline normalizing
    # finish_reason to "tool_calls" upstream.
    if has_tool_calls:
        return TurnDisposition.CONTINUE
    if finish_reason == "stop":
        return TurnDisposition.COMPLETE
    if finish_reason in ("tool_calls", "length"):
        return TurnDisposition.CONTINUE
    if finish_reason is None:
        return TurnDisposition.COMPLETE
    # Unknown non-null reason
    return TurnDisposition.FAILED


def finalize_skill_mode(ctx, agent, expected_mode) -> bool:
    """Finalize the agent's active mode after a turn completes (spec §7.5).

    Called from PostProcessStage.execute()'s `finally` block. Returns True
    if the mode was cleared, False if it was preserved (concurrent
    replacement detected via CAS).

    On CONTINUE: keep the mode alive for the next turn (no clear).
    On COMPLETE / FAILED: clear_active_mode(expected_mode=expected_mode).

    The expected_mode is the snapshot captured BEFORE this turn's
    upstream call. ContextOptimizeStage captures it via
    agent.snapshot_mode() IMMEDIATELY AFTER apply_mode succeeds (so the
    mode is resolved) but BEFORE upstream runs, and stores it on
    ctx.mode_snapshot_pre_upstream. PostProcessStage reads that field
    instead of re-snapshotting at the top of finalization. This allows
    compare-and-clear to detect concurrent replacement (e.g., Request B
    installing a newer mode while Request A's completion thread was
    finishing upstream) — the pre-upstream snapshot mismatches current,
    so the CAS no-ops and B's mode survives.

    If `ctx.response_data is None`, no real turn completed (preflight,
    deferred upstream, etc.). In that case there is nothing to classify,
    so the mode MUST be preserved — passing `expected_mode=None` into
    `clear_active_mode` would otherwise bypass the CAS and unconditional-
    clear the agent's mode mid-streaming, which is what causes a streaming
    preflight to wipe the ticket mode before the daemon thread's
    PostProcessStage ever runs (Wave 4 fix).

    IMPORTANT: this function does NOT snapshot the current mode itself.
    The CAS in clear_active_mode(expected_mode=...) is the atomic op.
    The caller is responsible for capturing the expected_mode BEFORE the
    upstream call.
    """
    data = getattr(ctx, "response_data", None)
    if data is None:
        # No turn completed (preflight / streaming deferred upstream / etc.).
        # Don't clear the mode.
        return False
    # R2 #1 (None-ownership): if no pre-upstream snapshot was captured
    # (agent is None, no active mode at apply_mode time, or the
    # propagation chain didn't reach us), DO NOT pass `expected_mode=None`
    # to clear_active_mode — that bypasses the CAS identity check
    # (`agent.py:225-226`: `if expected_mode is not None and self.active_mode
    # is not expected_mode`) and would unconditionally clear any current
    # mode a concurrent request might have installed. Preserve the mode
    # instead.
    if expected_mode is None:
        return False
    disposition = classify_turn_response(data)
    if disposition == TurnDisposition.CONTINUE:
        return False  # mode preserved
    return agent.clear_active_mode(expected_mode=expected_mode)


def finalize_skill_mode_on_exception(ctx, agent) -> bool:
    """Finalize the agent's active mode on a CRITICAL-stage exception.

    R2 #2 (exception-path never finalizes): when RouteResolution,
    Security, ContextOptimize, or Upstream raises, PostProcessStage's
    `finally` block never runs, so the agent's mode is never
    CAS-finalized. Without this call the agent leaks the install into
    the next turn, which re-snapshots from a possibly-stale state.

    Rules:
    - Idempotent: callable once; the CAS guards against double-clear.
    - No `response_data` requirement: there is no completed turn to
      classify. We only act on the CAS.
    - `expected_mode` is read with a documented precedence:
        1. `ctx.mode_snapshot_pre_upstream` — the public field, set by
           ContextOptimizeStage AFTER `apply_mode` returns and BEFORE
           upstream runs. This is the canonical reader.
        2. `ctx._mode_snapshot_propagated_from_apply_mode` — fallback
           for the case where `apply_mode` raised before
           context_optimize.py:178 had a chance to populate the public
           field (e.g., a generic exception in the primary path caught
           at context_optimize.py:241 and re-raised at :287 because
           `has_active_mode=True`). The propagation in Edit A1 ran
           unconditionally, so this fallback is populated whenever
           apply_mode entered.
    - If both are None, no-op (preserves the agent's current mode).
    - `agent` may be None: no-op.

    Returns True if the mode was cleared, False otherwise (preserved
    or no-op).

    Disjoint ownership note: this function is reached only via
    `RequestPipeline._finalize_mode_on_critical_failure` (Edit A5),
    which fires when a CRITICAL stage raises. PostProcessStage.execute()
    (which calls `_finalize_skill_lifecycle_with_flag`) cannot run in
    that case because the pipeline re-raises before PostProcess. The
    CAS-idempotency is a defense-in-depth backstop, not the primary
    safety; do not couple these paths in a future refactor.
    """
    if agent is None:
        return False
    expected_mode = getattr(ctx, "mode_snapshot_pre_upstream", None)
    if expected_mode is None:
        # Fallback to the propagation (see docstring precedence #2).
        expected_mode = getattr(
            ctx, "_mode_snapshot_propagated_from_apply_mode", None
        )
    if expected_mode is None:
        return False
    return agent.clear_active_mode(expected_mode=expected_mode)


def apply_skill_response_transform(ctx, agent, response_data) -> dict:
    """Apply the active skill's post_process() callback to the response.

    Spec §7.5: this is the FIRST of the three post-process ops invoked from
    PostProcessStage.execute(). Pure data transform; does not touch agent
    state. Wraps skill.post_process() with error handling so a buggy skill
    callback cannot poison the rest of post-processing.

    Behavior:
    - If the agent has no active skill mode, return response_data unchanged.
    - If the active skill name is not in the registry, return response_data
      unchanged (with a WARNING log). Per spec §7.2 fail-closed semantics,
      this is a defensive fallback — ContextOptimizeStage would have raised
      ActiveSkillUnavailable earlier, but if we land here without an active
      skill (e.g., mode was already cleared), we must not crash.
    - On exception from skill.post_process(), log a WARNING and return
      response_data unchanged. The skill's transform is a soft hook, not a
      correctness gate.
    """
    from proxy.skills.skill_registry import SkillContext, SkillRegistry

    if not isinstance(response_data, dict):
        return response_data

    ms = agent.snapshot_mode() if agent is not None else None
    if ms is None:
        return response_data

    registry = SkillRegistry()
    skill = registry.get(ms.name)
    if skill is None:
        logger.warning(
            "apply_skill_response_transform: active skill %r not in registry",
            ms.name,
        )
        return response_data

    skill_ctx = SkillContext(
        session_id=getattr(ctx, "session_id", ""),
        agent_id=getattr(agent, "agent_id", ""),
        messages=list(getattr(ctx, "messages", []) or []),
        is_subagent=bool(getattr(ctx, "is_subagent", False)),
        skill_state=dict(getattr(ms, "skill_state", {}) or {}),
    )
    try:
        return skill.post_process(skill_ctx, response_data)
    except Exception as e:
        logger.warning(
            "apply_skill_response_transform: skill %r post_process raised: %s",
            ms.name, e,
        )
        return response_data


def inject_task_canaries(ctx, agent, response_data, mode) -> dict:
    """Insert the appropriate skill canary into any `task` tool calls.

    Spec §7.5 + custom-skill research swarms §11/§13/§14. THREE
    mutually exclusive paths:

    1. **Universal no-nesting (§14):** if `ctx.is_subagent` AND the final
       response contains any native `task` tool call, reject the entire
       response with ``StageAbort((swarm_subagent_cannot_spawn, 403))``.

    2. **MAIN swarm parent (§11/§13):** when `ctx.swarm_policy` is not None,
       build ONE strict SKILL canary via `build_swarm_skill_canary`,
       then delegate the atomic detached response transformation to
       `proxy.canary.transform_swarm_task_calls`.

    3. **Legacy MAIN (ctx.swarm_policy is None):** preserve the existing
       path byte-for-byte.
    """
    from proxy.canary import (
        SwarmCanaryError,
        _inject_canary_into_task_calls,
        build_swarm_skill_canary,
        sign_canary,
        transform_swarm_task_calls,
    )
    from proxy.skills.skill_definition import SubagentPolicy
    from proxy.skills.skill_registry import SkillContext, SkillRegistry
    from proxy.workspace_state import DEFAULT_WORKSPACE, resolve_workspace

    if not isinstance(response_data, dict):
        return response_data

    task_calls = _collect_native_task_calls(response_data)

    # ===== Path A: universal no-nesting (§14) =====
    if getattr(ctx, "is_subagent", False) and task_calls:
        _raise_swarm_abort(
            code="swarm_subagent_cannot_spawn",
            status=SWARM_RUNTIME_STATUS["swarm_subagent_cannot_spawn"],
            message="Sub-agents cannot spawn sub-agents",
        )

    # ===== Path B: MAIN swarm parent =====
    swarm_policy = getattr(ctx, "swarm_policy", None)
    if swarm_policy is not None and not isinstance(swarm_policy, SubagentPolicy):
        swarm_policy = None

    if swarm_policy is not None:
        ms = agent.snapshot_mode() if agent is not None else None
        skill_name = getattr(ms, "name", None) if ms is not None else None
        if not skill_name:
            _raise_swarm_abort(
                code="swarm_policy_unavailable",
                status=SWARM_RUNTIME_STATUS["swarm_policy_unavailable"],
                message="Swarm parent policy requires an active mode",
            )

        try:
            active_ws = resolve_workspace(ctx)
        except Exception:
            active_ws = None
        if active_ws and active_ws != DEFAULT_WORKSPACE:
            state = {"workspace_name": active_ws}
        else:
            state = {}

        try:
            canary = build_swarm_skill_canary(
                skill_name,
                state,
                session_id=getattr(ctx, "session_id", "") or "",
            )
        except SwarmCanaryError as exc:
            code = exc.code
            if code in ("swarm_missing_secret", "swarm_signing_failed", "swarm_state_too_large"):
                mapped = code
            else:
                mapped = "swarm_signing_failed"
            _raise_swarm_abort(
                code=mapped,
                status=SWARM_RUNTIME_STATUS[mapped],
                message="Strict canary build failed",
            )
        except Exception:
            _raise_swarm_abort(
                code="swarm_signing_failed",
                status=SWARM_RUNTIME_STATUS["swarm_signing_failed"],
                message="Strict canary build failed",
            )

        from proxy.canary import _SwarmTransformError
        try:
            result = transform_swarm_task_calls(
                response_data,
                canary=canary,
                max_task_calls=swarm_policy.max_task_calls_per_request,
            )
        except _SwarmTransformError as exc:
            _raise_swarm_abort(
                code=exc.code,
                status=SWARM_RUNTIME_STATUS[exc.code],
                message=exc.message,
            )

        total_bytes = sum(_get_prompt_bytes(tc) for tc in task_calls)
        logger.debug(
            "swarm_inject request_id=%s parent_skill=%s task_count=%d "
            "prompt_byte_len=%d code=ok",
            getattr(ctx, "request_id", "?"),
            skill_name,
            len(task_calls),
            total_bytes,
        )
        return result

    # ===== Path C: legacy MAIN =====
    if getattr(ctx, "is_subagent", False):
        return response_data

    ms = agent.snapshot_mode() if agent is not None else None
    canary = None
    if ms is not None:
        registry = SkillRegistry()
        skill = registry.get(ms.name)
        if skill is not None:
            skill_ctx = SkillContext(
                session_id=getattr(ctx, "session_id", ""),
                agent_id=getattr(agent, "agent_id", ""),
                messages=list(getattr(ctx, "messages", []) or []),
                skill_state=dict(getattr(ms, "skill_state", {}) or {}),
                client_hint=getattr(ctx, "client_hint", "") or "",
            )
            try:
                canary = skill.get_canary(skill_ctx)
            except Exception as e:
                logger.warning(
                    "inject_task_canaries: skill %r get_canary raised: %s",
                    ms.name, e,
                )
                canary = None

    if canary is None:
        active_ws = resolve_workspace(ctx)
        if active_ws and active_ws != DEFAULT_WORKSPACE:
            sig = sign_canary(
                f"workspace:{active_ws}",
                session_id=getattr(ctx, "session_id", "") or "",
            )
            canary = f"[[__SUBAGENT__:WORKSPACE:{active_ws}:{sig}]]\n"
        else:
            canary = "[[__SUBAGENT__]]\n"

    return _inject_canary_into_task_calls(
        response_data,
        canary,
        request_ctx=ctx,
        default_mode_name=(ms.name if ms is not None else None),
    )


def _collect_native_task_calls(response_data: dict) -> list:
    """Return the list of native ``function.name == "task"`` tool calls."""
    try:
        choices = response_data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return []
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            return []
        tc_list = message.get("tool_calls") or []
        if not isinstance(tc_list, list):
            return []
    except (IndexError, KeyError, TypeError, AttributeError):
        return []
    out = []
    for tc in tc_list:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if isinstance(func, dict) and func.get("name") == "task":
            out.append(tc)
    return out


def _get_prompt_bytes(tc: dict) -> int:
    """Best-effort byte length of a task call's prompt argument."""
    try:
        import json as _json
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            return 0
        args_raw = func.get("arguments")
        if not isinstance(args_raw, str):
            return 0
        args = _json.loads(args_raw)
        if not isinstance(args, dict):
            return 0
        prompt = args.get("prompt", "")
        if isinstance(prompt, str):
            return len(prompt.encode("utf-8"))
    except Exception:
        return 0
    return 0
