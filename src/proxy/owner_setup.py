"""Owner-only request setup (CP2B).

These mutations touch shared / per-session state and MUST run exactly once, on
the owner execution path, immediately before pipeline.run(ctx). They are split
out of parse_request (pure ingress) and the pre-admission route body so they
never fire for rejected (503) or, in Phase 3, coalesced follower requests.
"""
import logging

logger = logging.getLogger("proxy.owner_setup")


def activate_inherited_workspace(ctx) -> None:
    """Owner-only: activate a sub-agent's inherited workspace in the shared
    workspace_store. Moved verbatim out of parse_request (CP2B)."""
    if not (getattr(ctx, "is_subagent", False)
            and getattr(ctx, "workspace_name", None)
            and ctx.workspace_name != "default"):
        return
    _ws = ctx.workspace_name
    try:
        from proxy.state import make_workspace_id
        _ws = make_workspace_id(_ws)
        from proxy.workspace_state import workspace_store
        workspace_store.set_active(ctx.session_id, _ws)
        logger.debug("owner_setup: activated inherited workspace=%r for sub-agent session=%s",
                     _ws, ctx.session_id[:8],
                     extra={"request_id": getattr(ctx, "request_id", "?")})
    except ValueError:
        logger.warning("owner_setup: rejecting invalid inherited workspace_name=%r for session=%s",
                       _ws, ctx.session_id[:8],
                       extra={"request_id": getattr(ctx, "request_id", "?")})


def owner_setup(ctx, pipeline, features) -> None:
    """Run the owner-only mutations in order, immediately before pipeline.run(ctx)."""
    ctx._features = features
    from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker
    ToolCallTracker.reset_instance(ctx.session_id)
    activate_inherited_workspace(ctx)
    # DispatchStage._features is a documented per-request reset (pipeline/AGENTS.md
    # landmine #9). Safe to set on the shared singleton from a worker thread ONLY
    # because `features` (_FEATURES) is a process constant identical for every
    # request. If _features ever becomes per-request, this becomes a cross-request race.
    from proxy.pipeline.stage_modules.dispatch_stage import DispatchStage
    for stage in pipeline.stages:
        if isinstance(stage, DispatchStage):
            stage._features = features
            break
