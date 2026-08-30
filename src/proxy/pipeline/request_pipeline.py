"""RequestPipeline - orchestrates request processing through stages.

Request parsing runs before the pipeline (in app.py:parse_request).
DEFAULT_STAGES contains the 9 stages that run inside the pipeline:

1. CleanupStage - session cleanup, request counter
2. DispatchStage - command/enhancer/task dispatch
3. RouteResolutionStage - resolve upstream provider route
4. SecurityStage - malicious content detection
5. ContextOptimizeStage - system prompt, context limits
6. MotivatorStage - sub-agent context-usage reminders
7. ProviderCompatStage - provider-specific transformations
8. UpstreamStage - HTTP call to LLM backend
9. PostProcessStage - tool interception, requery loop, sync

Error handling:
- StageAbort: Normal early exit (dispatch handled the request)
- CRITICAL stages: RouteResolutionStage, SecurityStage, UpstreamStage - fail = abort
- Non-critical: continue in degraded mode
"""
import logging
from typing import Any, List, Optional

from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.protocols import PipelineStage
from proxy.pipeline.stages import (
    DEFAULT_STAGES,
    CleanupStage,
    DispatchStage,
    RouteResolutionStage,
    SecurityStage,
    ContextOptimizeStage,
    ProviderCompatStage,
    UpstreamStage,
    PostProcessStage,
)

logger = logging.getLogger("proxy.pipeline")


class RequestPipeline:
    """Orchestrates request processing through stages.

    Stateless - configured by Agent, executes 9 stages in order.

    Usage:
        pipeline = RequestPipeline()
        response = pipeline.run(ctx)
    """

    CRITICAL_STAGES = {
        "RouteResolutionStage",
        "SecurityStage",
        "ContextOptimizeStage",
        "UpstreamStage",
    }

    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        """Create a pipeline with the given stages.

        Args:
            stages: List of stages to execute in order.
                   Uses DEFAULT_STAGES if not provided.
        """
        self.stages = stages if stages is not None else list(DEFAULT_STAGES)

    def run(self, ctx: Any) -> dict:
        """Run the pipeline on the given context.

        Args:
            ctx: RequestContext with session, body, messages, etc.

        Returns:
            Response dict (or raises exception for critical failures).

        Error handling:
        - StageAbort: Early exit with response (e.g., dispatch produced a response)
        - Exception in CRITICAL stage: Re-raise
        - Exception in non-critical stage: Log and continue
        """
        # R3-H8 / R3-H9: Wrap entire pipeline in request_scope so the agent
        # is marked in-flight for the duration of request processing (including
        # any tool loops). This prevents LRU eviction and TTL expiry mid-request.
        from proxy.agent.agent_registry import registry as agent_registry
        session_id = getattr(ctx, "session_id", None)
        if session_id:
            with agent_registry.request_scope(session_id):
                return self._run_stages(ctx)
        else:
            return self._run_stages(ctx)

    def _run_stages(self, ctx: Any) -> dict:
        for stage in self.stages:
            stage_name = stage.__class__.__name__

            try:
                logger.debug(f"RequestPipeline: Executing {stage_name}")

                result = stage.execute(ctx)

                if result is not None:
                    logger.debug(
                        f"RequestPipeline: {stage_name} returned response, "
                        f"short-circuiting pipeline"
                    )
                    return result

            except StageAbort as abort:
                logger.warning(
                    f"RequestPipeline: {stage_name} raised StageAbort — "
                    f"pipeline SHORT-CIRCUIT, remaining stages SKIPPED. "
                    f"session={getattr(ctx, 'session_id', '?')[:8]} "
                    f"msg_count={len(getattr(ctx, 'messages', []) or [])}"
                )
                # R2 #2 (exception-path never finalizes mode, StageAbort
                # half): CRITICAL stages can raise StageAbort directly
                # (e.g. UpstreamStage wraps a non-AuthConfigError as
                # StageAbort((upstream_protocol_error, 502)) at
                # upstream.py:76). Without this finalize, the agent's
                # mode installed by ContextOptimize leaks into the next
                # turn. Idempotent (CAS-guarded) and a no-op when no
                # snapshot was captured, so safe on every CRITICAL
                # StageAbort path.
                if stage_name in self.CRITICAL_STAGES:
                    self._finalize_mode_on_critical_failure(ctx)
                raise

            except Exception as e:
                stage_name = stage.__class__.__name__

                logger.error(
                    f"RequestPipeline: Stage {stage_name} failed: {e}",
                    exc_info=True
                )

                if stage_name in self.CRITICAL_STAGES:
                    logger.error(
                        f"RequestPipeline: CRITICAL stage {stage_name} failed, "
                        f"aborting request"
                    )
                    # R2 #2 (exception-path never finalizes mode, generic
                    # Exception half): the matching StageAbort branch is
                    # above. finalize_skill_mode_on_exception is
                    # idempotent (CAS-guarded) and a no-op when no
                    # snapshot was captured, so it is safe on every
                    # CRITICAL-stage failure path.
                    self._finalize_mode_on_critical_failure(ctx)
                    raise

                logger.warning(
                    f"RequestPipeline: Non-critical stage {stage_name} failed, "
                    f"continuing with degraded context"
                )
                continue

        return getattr(ctx, "response_data", {})

    def _finalize_mode_on_critical_failure(self, ctx: Any) -> None:
        """Finalize the agent's mode when a CRITICAL stage raises or aborts.

        R2 #2. Delegates to `proxy.mode.finalize_skill_mode_on_exception`.
        Wrapped in try/except so a finalize failure cannot mask the
        original exception being re-raised by `_run_stages`.

        Lock-ordering note (verified by R1-A plan reviewer): this method
        acquires `#0 AgentRegistry._lock` briefly via get_or_load,
        releases it, then `#7 Agent._mode_lock` per-instance via
        clear_active_mode. Ascending order is correct. The enclosing
        `request_scope` (`run()` line 87) holds `#0` only across the
        `with` body, so by the time `_finalize_mode_on_critical_failure`
        runs from inside `_run_stages`, the scope is still active and
        `#0` is held by the thread — but `get_or_load` re-acquires it
        under the registry's own context manager (agent_registry.py:64-82
        releases on entry, re-acquires on exit). No deadlock.
        """
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            from proxy.mode import finalize_skill_mode_on_exception

            session_id = getattr(ctx, "session_id", None)
            if not session_id:
                logger.info(
                    "RequestPipeline._finalize_mode_on_critical_failure: "
                    "no session_id on ctx, skipping finalize"
                )
                return
            agent = agent_registry.get_or_load(session_id)
            if agent is None:
                logger.info(
                    "RequestPipeline._finalize_mode_on_critical_failure: "
                    "no agent in registry for session_id=%s, skipping finalize",
                    session_id[:12],
                )
                return
            finalize_skill_mode_on_exception(ctx, agent)
        except Exception as finalize_err:
            # NEVER mask the original CRITICAL-stage exception. Log and
            # swallow; the caller (request_pipeline._run_stages) re-raises
            # the original exception.
            logger.warning(
                "RequestPipeline._finalize_mode_on_critical_failure: "
                "finalize_skill_mode_on_exception failed: %s",
                finalize_err,
                exc_info=True,
            )


def create_pipeline() -> RequestPipeline:
    """Create a default pipeline with all 9 stages.

    Convenience factory function.
    """
    return RequestPipeline()


def is_critical_stage(stage: PipelineStage) -> bool:
    """Check if a stage is critical (failure = request abort)."""
    return stage.__class__.__name__ in RequestPipeline.CRITICAL_STAGES
