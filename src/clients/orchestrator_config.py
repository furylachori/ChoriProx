from dataclasses import dataclass
from proxy.tool_pipeline.tool_registry import registry as _tool_reg

_memory_tool_names_default = frozenset(_tool_reg.get_memory_tool_names())


@dataclass(frozen=True)
class HookCapabilities:
    provider: bool = False
    auth: bool = False
    tool_definition: bool = False
    tool_execute_before: bool = False
    tool_execute_after: bool = False
    chat_params: bool = False
    chat_headers: bool = False
    chat_messages_transform: bool = False
    system_prompt_transform: bool = False
    permission_ask: bool = False
    shell_env: bool = False
    session_compacting: bool = False
    event: bool = False
    config: bool = False
    command_execute: bool = False
    text_complete: bool = False


@dataclass(frozen=True)
class ProxyFeatures:
    version: int = 1
    intercepts_memory: bool = True
    runs_tool_loop: bool = True
    injects_context: bool = True
    replaces_system_prompt: bool = True
    memory_tool_names: frozenset[str] = _memory_tool_names_default
    max_tool_iterations: int = 15
    ticket_mode_enabled: bool = True
    tool_loop_timeout: int = 90
    intercepts_commands: bool = True
    task_commands_enabled: bool = True


def get_features_for_client(client_hint: str = "unknown") -> ProxyFeatures:
    """Return ProxyFeatures for a given client hint. Currently returns defaults."""
    return ProxyFeatures()
