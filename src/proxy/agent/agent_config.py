"""Agent configuration — per-agent settings.

Part of Phase 7: Full Agent Ownership.
"""
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class AgentConfig:
    """Configuration for an Agent instance.
    
    Controls the agent's behavior, tools, and context limits.
    
    Fields:
        provider_adapter: Key for PROVIDER_REGISTRY (e.g., "deepseek", "mimo", "kimi", "minimax")
        tool_registry: Optional ToolRegistry override (for ticket mode)
        system_prompt: The system prompt to inject for this agent
        context_budget: Token budget for context window (default: 32000)
        max_tool_iterations: Maximum tool loop iterations (default: 5)
        tool_loop_timeout: Tool loop timeout in seconds (default: 90.0)
        ignore_subagents: Whether to skip sub-agent injection/handling
        motivator_thresholds: Percentage thresholds for motivator reminders (default: (50, 75, 90))
    """
    provider_adapter: Optional[str] = None
    tool_registry: Optional[Any] = None
    system_prompt: str = ""
    context_budget: int = 32000
    max_tool_iterations: int = 5
    tool_loop_timeout: float = 90.0
    ignore_subagents: bool = False
    motivator_thresholds: tuple = (50, 75, 90)
    # Default is large enough to cover mini-m3 (1M tokens ≈ 4M chars).
    # The actual window is resolved per-request from ctx.model_cfg.context_limit;
    # this default is only used when no model_cfg is available.
    motivator_context_window_chars: int = 4_000_000  # 1M tokens × 4 chars/token
