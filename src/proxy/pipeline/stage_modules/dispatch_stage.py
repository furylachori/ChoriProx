"""DispatchStage — command dispatch, enhancer, task planning."""

import logging
from typing import Any, Optional

from proxy.pipeline.exceptions import StageAbort

logger = logging.getLogger("proxy.pipeline.stages")


class DispatchStage:
    """Command dispatch, enhancer, task planning.

    Uses: src/proxy/pipeline/dispatch.py:run_dispatch()

    If dispatch handles the request fully, raises StageAbort with response.
    If dispatch rewrites the text (^^ticket case), modifies ctx.messages in-place.
    """

    def __init__(self, features: Any = None):
        self._features = features

    def execute(self, ctx: Any) -> Optional[Any]:
        features = self._features or getattr(ctx, "_features", None)
        if not features:
            try:
                from clients.orchestrator_config import get_features_for_client

                client_hint = getattr(ctx, "client_hint", "unknown")
                features = get_features_for_client(client_hint)
            except ImportError:
                logger.exception("DispatchStage: Failed to import orchestrator_config — dispatch pipeline DISABLED, all commands/skills bypassed")
                return None

        if not features or not ctx.messages:
            return None

        try:
            from proxy.pipeline.dispatch import run_dispatch
        except ImportError:
            logger.exception("DispatchStage: Failed to import proxy.pipeline.dispatch — dispatch pipeline DISABLED")
            return None

        result = run_dispatch(ctx, features)
        if result is None:
            return None

        if result.handled:
            # Bug #5: thread the skill-error status code (e.g. 404 for an
            # unknown-hash, 400 for bad arg) through to the Flask response
            # layer. `_response_from_stage_abort` (app.py) unpacks the
            # `(body, status)` tuple shape. Dispatch now always returns
            # plain dicts (no Flask streaming Response objects), so we only
            # wrap dict bodies whose status_code is a non-default 4xx/5xx;
            # 200 is the implicit default for plain dict passthrough.
            if (
                isinstance(result.response, dict)
                and isinstance(result.status_code, int)
                and result.status_code != 200
            ):
                raise StageAbort((result.response, result.status_code))
            raise StageAbort(result.response)

        if result.rewritten_text is not None:
            if ctx.messages:
                last_msg = ctx.messages[-1]
                if isinstance(last_msg, dict) and last_msg.get("role") == "user":
                    # Spec §7.3: any failure during rewrite application
                    # must roll back the mode that SkillDispatcher.activate
                    # already installed. The expected_mode snapshot,
                    # captured before activation in
                    # SkillDispatcher.dispatch(), is the rollback anchor.
                    try:
                        last_msg["content"] = result.rewritten_text
                        if "messages" in ctx.body and ctx.body["messages"]:
                            body_last = ctx.body["messages"][-1]
                            if isinstance(body_last, dict):
                                body_last["content"] = result.rewritten_text
                    except Exception as rewrite_err:
                        logger.error(
                            "DispatchStage: rewrite failed for skill %r: %s - rolling back mode",
                            result.skill.name if result.skill else "?",
                            rewrite_err,
                            extra={
                                "request_id": getattr(ctx, "request_id", "?"),
                                "session_id": getattr(ctx, "session_id", "?"),
                            },
                        )
                        # Compare-and-clear the expected mode snapshot.
                        agent = getattr(ctx, "_active_agent", None)
                        if agent is None:
                            from proxy.agent.agent_registry import registry as agent_registry
                            agent = agent_registry.get_or_load(ctx.session_id)
                        if agent is not None and getattr(result, "expected_mode", None) is not None:
                            try:
                                agent.clear_active_mode(expected_mode=result.expected_mode)
                            except Exception as rb_err:
                                logger.warning(
                                    "DispatchStage: rollback clear_active_mode failed: %s",
                                    rb_err,
                                )
                        raise StageAbort(({
                            "error": {
                                "message": "Skill activation rewrite failed",
                                "type": "skill_runtime_error",
                            }
                        }, 500))

        return None
