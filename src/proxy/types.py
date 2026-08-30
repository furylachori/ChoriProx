"""Data classes for function signatures and shared exception types."""
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from proxy.skills.skill_definition import SubagentPolicy


class BackendError(Exception):
    """Raised when upstream backend returns an error response.

    Defined here (not in app.py) to avoid circular imports — http_client.py
    needs to raise it, and app.py needs to catch it. Both import from types.py.
    """
    def __init__(self, status_code, error_body):
        self.status_code = status_code
        self.error_body = error_body
        super().__init__(f"Backend returned {status_code}")


class InvalidRequestError(ValueError):
    """Raised when the incoming request body fails structural validation.

    Per spec §7.1, carries the failing field path and a stable error code
    so the chat handler can render the structured 400 envelope without
    re-inspecting the exception.

    Attributes:
        param: JSON-pointer-style path to the failing field (e.g.
               "messages[2].role", "stream").
        code:  Stable string identifier (e.g. "invalid_messages",
               "invalid_role", "invalid_content").
        message: Human-readable description for the operator.
    """
    def __init__(self, param: str, code: str, message: str):
        self.param = param
        self.code = code
        self.message = message
        super().__init__(f"{message} (param={param!r}, code={code!r})")


@dataclass
class SubagentIngress:
    """Result of sub-agent ingress detection.

    Fields:
        is_subagent:           True if request originated from a sub-agent.
        subagent_ticket_mode:  True if parent had ticket mode (or canary says "ticket").
        inherited_skill:       Skill name inherited from parent canary (None otherwise).
        workspace_name:        Workspace name from canary (e.g., "feature-x" from
                               [[__SUBAGENT__:WORKSPACE:feature-x]]). Empty string if
                               not a workspace canary.
    """
    is_subagent: bool = False
    subagent_ticket_mode: bool = False
    inherited_skill: Optional[str] = None
    workspace_name: str = ""
    task_hash: Optional[str] = None


@dataclass
class RequestContext:
    """Carries request state through the 7-stage pipeline.

    Field ownership (writer → readers):
        body              — IngressGateway (write), all stages (read), DispatchStage (rewrite)
        model_name        — IngressGateway (write), ProviderCompatStage (read)
        wants_stream      — IngressGateway (write), UpstreamStage (read)
        session_id        — IngressGateway (write), all stages (read)
        client_hint       — IngressGateway (write), DispatchStage (read)
        messages          — IngressGateway (write), ContextOptimizeStage (rewrite), UpstreamStage (read)
        remote_addr       — IngressGateway (write), SecurityStage (read)
        is_subagent       — CanaryDetector (write), ToolGuard (read), ModePipeline (read)
        messages_original — ContextOptimizeStage (write, backup before rewrite)
        subagent_ticket_mode — CanaryDetector (write), DispatchStage (read)
        upstream_url      — RouteResolutionStage (write), UpstreamStage (read)
        tool_parser       — DispatchStage (write), UpstreamStage (read)
        auth_headers      — ProviderCompatStage (write), UpstreamStage (read)
        original_model    — ProviderCompatStage (write), PostProcessStage (read)
        model_cfg         — RouteResolutionStage (write), ProviderCompatStage (read)
        execute_fn        — DispatchStage (write), UpstreamStage (read)
        parse_tools_fn    — DispatchStage (write), PostProcessStage (read)
        max_iterations    — DispatchStage (write), UpstreamStage (read)
        max_total_time    — DispatchStage (write), UpstreamStage (read)
        max_result_size   — DispatchStage (write), PostProcessStage (read)
        tooling_payload   — DispatchStage (write), UpstreamStage (read)
        response_data     — UpstreamStage (write), PostProcessStage (read/write)
        agent             — DispatchStage (write), PostProcessStage (read)
        dispatch_response — DispatchStage (write), PostProcessStage (read)
        seeded_tasks      — DispatchStage (write), PostProcessStage (read)
        _skill_activated  — DispatchStage (write), PostProcessStage (read)
        inherited_skill   — CanaryDetector (write), DispatchStage (read)
        workspace_name    — CanaryDetector (write), ToolEntryPoints (read)
        _mode             — DispatchStage (write), PostProcessStage (read)
        swarm_policy      — ContextOptimizeStage (write after mode application),
                           PostProcess swarm boundary (read)

    V2 chain-hash boundary fields:
        v2_boundary_idx: Optional[int] = None
            Set by _select_context_v2() after V2 chain-hash pipeline runs.
            Read by PostProcessStage to persist raw_start_message_index.
            None for sub-agents or cold-path fallback (CAS failure).
        v2_boundary_chain: Optional[str] = None
            Set alongside v2_boundary_idx. Transient — used for validation,
            not persisted to DB.
    """
    body: dict
    model_name: str
    wants_stream: bool
    session_id: str
    client_hint: str
    messages: list
    remote_addr: str = ""
    is_subagent: bool = False
    unattended: bool = False  # True if no human partner is available
    messages_original: Optional[list] = None
    subagent_ticket_mode: bool = False
    upstream_url: str = ""
    tool_parser: Any = None
    auth_headers: dict = field(default_factory=dict)
    original_model: str = ""
    model_cfg: Any = None
    execute_fn: Any = None
    parse_tools_fn: Any = None
    max_iterations: int = 5
    max_total_time: float = 90.0
    max_result_size: int = 16384
    tooling_payload: Any = None
    response_data: dict = field(default_factory=dict)
    agent: Any = None
    dispatch_response: Any = None
    seeded_tasks: list = field(default_factory=list)
    _skill_activated: bool = False
    inherited_skill: Optional[str] = None
    workspace_name: str = ""
    inherited_task_hash: Optional[str] = None
    _mode: Optional[Any] = None
    v2_boundary_idx: Optional[int] = None
    v2_boundary_chain: Optional[str] = None
    # Immutable SubagentPolicy reference for swarm-capable custom skill
    # requests. Set ONLY by ContextOptimizeStage AFTER apply_mode succeeds
    # and the active skill's subagents policy is validated. None for:
    #   - non-swarm MAIN requests,
    #   - ordinary MAIN requests (active skill has subagents=None),
    #   - all sub-agent requests (universal no-nesting rule),
    #   - failed mode-application paths,
    #   - reused RequestContext instances (reset at start of every execute()).
    # The reference is the immutable frozen object published by
    # SkillDefinition.subagents — do not copy, deep-copy, or convert to
    # a dict; downstream code uses `is` identity checks.
    swarm_policy: Optional["SubagentPolicy"] = None
    # Reduce-only per-session context budget override (raw tokens), set by the
    # `^^context` command on the Agent. Set by ContextOptimizeStage: None for
    # sub-agents (no inheritance) / reused ctx / agent-None paths; the agent's
    # get_context_limit_override() value otherwise. Read in context_v2 to
    # thread into get_context_config.
    context_limit_override: Optional[int] = None
    # Pre-upstream mode snapshot for CAS compare-and-clear in
    # PostProcessStage.finalize_skill_mode. Captured by ContextOptimizeStage
    # via agent.snapshot_mode() IMMEDIATELY AFTER apply_mode succeeds — i.e.
    # AFTER mode resolution but BEFORE the upstream call. This is the
    # `expected_mode` that clear_active_mode(expected_mode=...) compares
    # against, so a concurrent replacement (e.g. Request B installing a newer
    # mode while Request A's completion thread is finishing upstream) is
    # detected and B's mode is preserved. None when mode resolution did not
    # run (cascade fallback, preflight short-circuit, agent is None).
    mode_snapshot_pre_upstream: Optional[Any] = None
    """Pre-upstream mode snapshot for CAS compare-and-clear in
    `proxy.mode.finalize_skill_mode` AND
    `proxy.mode.finalize_skill_mode_on_exception`.

    R2 #1 (TOCTOU at context_optimize.py:178): the value is the EXACT
    ModeState identity propagated from `proxy.mode.apply_mode`'s entry-
    function `agent.snapshot_mode()` read at `mode.py:170` — NOT a
    second `snapshot_mode()` call after `apply_mode` returns. Two atomic
    reads of `active_mode` open a window where a concurrent replacement
    can be CAS-cleared (the F10 clobber).

    R2 #1 (None-ownership): `None` here means either (a) agent is None,
    (b) no active mode at apply_mode time, or (c) the propagation chain
    did not run (legacy call site or apply_mode raised early).
    `finalize_skill_mode` and `finalize_skill_mode_on_exception` both
    return False on `None` — they never pass `None` to
    `clear_active_mode` (which would bypass the CAS and unconditional-
    clear any current mode).

    Captured by `proxy.pipeline.stage_modules.context_optimize`
    ContextOptimizeStage.execute() after `apply_mode` succeeds but
    before upstream runs. Read by PostProcessStage (`finally` block) and
    by `RequestPipeline._finalize_mode_on_critical_failure`.
    """


@dataclass
class ModelRoute:
    """Resolved model configuration and routing."""
    model_cfg: Any
    upstream_url: str
    tool_parser: Any
    is_remote: bool


@dataclass
class UpstreamRequest:
    """Prepared upstream HTTP request."""
    url: str
    body: dict
    headers: dict
    original_model: str


@dataclass
class ToolingPayload:
    """Processed tool definitions."""
    tools: list
    tool_hash: str
    cached: bool
    augmented_tools: list


@dataclass
class RequeryResult:
    """Result from requery_llm(). Field names match the dict keys
    returned by the current requery_llm() implementation in memory/tools.py."""
    requery_failed: bool = False
    requery_data: Optional[dict] = None
    working_messages: list = field(default_factory=list)
    _raw_tool_id_map: dict = field(default_factory=dict)
    new_tool_calls: list = field(default_factory=list)
    requery_content: Optional[str] = None
    should_continue: bool = False
    upstream_finish_reason: Optional[str] = None


@dataclass
class TaskResult:
    """Standard result object for proxy task operations."""
    action: str
    session_id: str
    success: bool = True
    error: Optional[str] = None
    message: Optional[str] = None
    data: Optional[dict] = None


@dataclass
class MessageProvenance:
    """Metadata tracking the origin and processing stage of a message.

    Attached to messages as _provenance key for debugging and audit.
    NOT synced to SQLite (ephemeral metadata only).

    Fields:
        agent_id:       The agent that processed this message
        message_type:   "user" | "assistant" | "tool" | "system"
        stage:          "ingress" | "pipeline" | "requery" | "sync"
        provider:       Provider name if known (deepseek, kimi, etc.)
        timestamp:      When this message was processed
    """
    agent_id: str
    message_type: str
    stage: str
    provider: Optional[str] = None
    timestamp: float = field(default_factory=time.time)