"""Pipeline stages façade — re-exports all 9 stages + DEFAULT_STAGES + parse/prepare helpers.

Stage implementations live in stage_modules/<name>.py (+ motivator_stage.py /
post_process.py). This module is the public import surface and the DEFAULT_STAGES
composition point.
"""
from proxy.pipeline.stage_modules.cleanup import CleanupStage
from proxy.pipeline.stage_modules.dispatch_stage import DispatchStage
from proxy.pipeline.stage_modules.route_resolution import RouteResolutionStage
from proxy.pipeline.stage_modules.security import SecurityStage
from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage
from proxy.pipeline.stage_modules.provider_compat_stage import ProviderCompatStage
from proxy.pipeline.stage_modules.request_parse import parse_request
from proxy.pipeline.stage_modules.upstream import UpstreamStage, prepare_upstream_request
from proxy.pipeline.motivator_stage import MotivatorStage
from proxy.pipeline.post_process import PostProcessStage

DEFAULT_STAGES = [
    CleanupStage(),
    DispatchStage(),
    RouteResolutionStage(),
    SecurityStage(),
    ContextOptimizeStage(),
    MotivatorStage(),
    ProviderCompatStage(),
    UpstreamStage(),
    PostProcessStage(),
]
