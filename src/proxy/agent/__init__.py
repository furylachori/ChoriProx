"""Agent package — Agent abstraction and lifecycle management.

Phase 5: Thin Agent adapter over existing state.
Phase 7: Full Agent ownership with MessageBuffer, AgentConfig, AgentRegistry.
"""

from proxy.agent.agent import Agent
from proxy.agent.agent_config import AgentConfig
from proxy.agent.message_buffer import MessageBuffer
from proxy.agent.agent_registry import AgentRegistry, registry

__all__ = [
    "Agent",
    "AgentConfig",
    "MessageBuffer",
    "AgentRegistry",
    "registry",
]
