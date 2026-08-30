"""UpstreamStage + prepare_upstream_request — HTTP call to LLM backend, request prep."""

import logging
import os
from typing import Any, Optional

from proxy.auth import AuthConfigError
from proxy.pipeline.exceptions import StageAbort
from proxy.types import UpstreamRequest

logger = logging.getLogger("proxy.pipeline.stages")


def prepare_upstream_request(body: dict, model_cfg, model_route) -> UpstreamRequest:
    """Rewrite model alias, build auth headers, clamp temperature."""
    from proxy.model_registry import TEST_MODE

    original_model = body.get("model", "")

    if model_cfg and model_cfg.backend_model:
        body["model"] = model_cfg.backend_model

    _auth_headers = {}

    if TEST_MODE:
        logger.debug("Returning mock response for context management testing")
    else:
        if model_cfg and "minimax" in (model_cfg.backend_model or "").lower():
            temp = body.get("temperature")
            if temp is not None:
                body["temperature"] = max(0.01, min(1.0, temp))

        if model_cfg and model_cfg.api_key_env:
            _api_key = os.environ.get(model_cfg.api_key_env, "").strip()
            if _api_key:
                _auth_headers["Authorization"] = f"Bearer {_api_key}"
            else:
                logger.warning(
                    f"API key env var '{model_cfg.api_key_env}' is not set or empty for model '{model_cfg.name}'"
                )
                raise AuthConfigError(
                    model_cfg.name or "unknown", model_cfg.api_key_env
                )

    return UpstreamRequest(
        model_route.upstream_url, body, _auth_headers, original_model
    )


class UpstreamStage:
    """HTTP call to LLM backend.

    Uses prepare_upstream_request() and call_upstream() from this module.
    """

    def execute(self, ctx: Any) -> Optional[Any]:

        route = getattr(ctx, "model_route", None)
        if route is None:
            raise RuntimeError(
                "UpstreamStage: ctx.model_route is not set. "
                "RouteResolutionStage may have failed silently."
            )

        model_cfg = route.model_cfg

        try:
            llama_body = getattr(ctx, "llama_body", ctx.body)
            upstream = prepare_upstream_request(llama_body, model_cfg, route)
            ctx.upstream = upstream
        except AuthConfigError:
            raise
        except Exception as e:
            logger.error("UpstreamStage: prepare_upstream_request failed: %s", e,
                         extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id})
            raise StageAbort(
                ({"error": {"message": f"Upstream request preparation failed: {e}", "type": "upstream_error"}}, 502)
            )

        try:
            from proxy.model_registry import TEST_MODE

            if TEST_MODE:
                import time
                from proxy.openai_compat import build_openai_response

                data = build_openai_response("Mock response for testing. Context was optimized.", model=upstream.original_model, response_id=f"chatcmpl-{int(time.time())}", usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
                ctx.response_data = data
                return None
        except ImportError:
            logger.exception("UpstreamStage: Failed to import TEST_MODE — mock responses DISABLED")
            pass

        from proxy.http_client import call_upstream
        logger.info(
            "[%s] upstream call: url=%s",
            getattr(ctx, 'request_id', '?'),
            upstream.url,
            extra={"request_id": getattr(ctx, 'request_id', '?'), "session_id": ctx.session_id},
        )
        data = call_upstream(upstream.url, upstream.body, upstream.headers, bool(getattr(ctx, "wants_stream", False)))
        ctx.response_data = data

        return None
