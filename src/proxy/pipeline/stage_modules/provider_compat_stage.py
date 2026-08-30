"""ProviderCompatStage — provider-specific request transformations."""

import logging
from typing import Any, Optional

logger = logging.getLogger("proxy.pipeline.stages")


class ProviderCompatStage:
    """Provider-specific request transformations.

    Uses: pipeline/provider_compat.py (Phase 3 extracted)
    """

    def execute(self, ctx: Any) -> Optional[dict]:
        route = getattr(ctx, "model_route", None)
        model_cfg = route.model_cfg if route else getattr(ctx, "model_cfg", None)

        llama_body = {**ctx.body, "stream": bool(getattr(ctx, "wants_stream", False))}
        llama_body.pop("stream_options", None)

        if "tools" in llama_body:
            from proxy_recovery_system import apply_proxy_recovery_system

            try:
                llama_body = apply_proxy_recovery_system(llama_body, ctx.session_id)
            except Exception as e:
                logger.warning(f"ProviderCompatStage: apply_proxy_recovery_system failed: {e}")

        from proxy.pipeline.provider_compat import apply_provider_compat

        llama_body = apply_provider_compat(llama_body, model_cfg)

        ctx.llama_body = llama_body

        return None
