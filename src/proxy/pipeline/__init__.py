"""proxy.pipeline — request processing pipeline.

Phase 3: Provider registry + dispatch extraction
Phase 7: Full RequestPipeline with 9 stages
"""

from proxy.pipeline.dispatch import (
    run_dispatch,
    DispatchResult,
    try_enhancer,
    try_skill_activation,
    try_command,
)
from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.protocols import PipelineStage
from proxy.pipeline.tool_interception import handle_tool_interception
from proxy.pipeline.stages import (
    CleanupStage,
    DispatchStage,
    RouteResolutionStage,
    SecurityStage,
    ContextOptimizeStage,
    MotivatorStage,
    ProviderCompatStage,
    UpstreamStage,
    PostProcessStage,
    DEFAULT_STAGES,
    parse_request,
    prepare_upstream_request,
)
from proxy.pipeline.provider_compat import (
    apply_provider_compat,
    ProviderAdapter,
    DeepSeekAdapter,
    MiMoAdapter,
    KimiAdapter,
    MiniMaxAdapter,
)
from proxy.pipeline.request_pipeline import (
    RequestPipeline,
    create_pipeline,
    is_critical_stage,
)
from proxy.pipeline.conditional_stage import ConditionalStage
from proxy.pipeline.prompt_injector import PromptInjector
from proxy.pipeline.threshold_engine import ThresholdEngine
from proxy.pipeline.agent_telemetry import AgentTelemetry

__all__ = [
    "run_dispatch",
    "DispatchResult",
    "try_enhancer",
    "try_skill_activation",
    "try_command",
    "PipelineStage",
    "StageAbort",
    "CleanupStage",
    "DispatchStage",
    "RouteResolutionStage",
    "SecurityStage",
    "ContextOptimizeStage",
    "MotivatorStage",
    "ProviderCompatStage",
    "UpstreamStage",
    "PostProcessStage",
    "DEFAULT_STAGES",
    "parse_request",
    "prepare_upstream_request",
    "handle_tool_interception",
    "apply_provider_compat",
    "ProviderAdapter",
    "DeepSeekAdapter",
    "MiMoAdapter",
    "KimiAdapter",
    "MiniMaxAdapter",
    "RequestPipeline",
    "create_pipeline",
    "is_critical_stage",
    "ConditionalStage",
    "PromptInjector",
    "ThresholdEngine",
    "AgentTelemetry",
]
