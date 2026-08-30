"""RouteResolutionStage — resolve upstream provider route."""

import logging
from typing import Any, Optional

from proxy.pipeline.exceptions import StageAbort

logger = logging.getLogger("proxy.pipeline.stages")


class RouteResolutionStage:
    """Resolve upstream provider route.

    Resolves which upstream provider to use based on the model name,
    setting ctx.model_route and ctx.model_cfg for downstream stages.
    """

    def execute(self, ctx: Any) -> Optional[dict]:
        route = getattr(ctx, "model_route", None)
        if route is None:
            try:
                from proxy.model_registry import resolve_model_route, _model_registry
                route = resolve_model_route(ctx.model_name, _model_registry)
                ctx.model_route = route
                ctx.model_cfg = route.model_cfg
            except Exception as e:
                logger.error(
                    "RouteResolutionStage: route resolution failed: %s", e,
                    extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id},
                )
                raise StageAbort(
                    ({"error": {"message": f"Route resolution failed: {e}", "type": "route_error"}}, 502)
                )

        return None
