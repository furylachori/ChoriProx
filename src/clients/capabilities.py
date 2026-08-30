from clients.interface import CapabilitySet, ToolFormat, StreamingFormat, MessageFormat
from clients.orchestrator_config import HookCapabilities

KILOCODE_CAPABILITIES = CapabilitySet(
    tool_format=ToolFormat.OPENAI,
    streaming_format=StreamingFormat.SSE_OPENAI,
    message_format=MessageFormat.OPENAI,
    supports_parallel_tool_calls=True,
    supports_streaming=True,
    has_plugin_system=True,
    hooks=HookCapabilities(
        provider=True,
        auth=True,
        tool_definition=True,
        tool_execute_before=True,
        tool_execute_after=True,
        chat_params=True,
        chat_headers=True,
        chat_messages_transform=True,
        system_prompt_transform=True,
        permission_ask=True,
        shell_env=True,
        session_compacting=True,
        event=True,
        config=True,
        command_execute=True,
        text_complete=True,
    ),
    config_format="json",
    agent_config_dir=".kilo",
    command_config_dir=".kilo",
    auth_mechanism="bearer",
)

OPENCODE_CAPABILITIES = CapabilitySet(
    tool_format=ToolFormat.OPENAI,
    streaming_format=StreamingFormat.SSE_OPENAI,
    message_format=MessageFormat.OPENAI,
    supports_parallel_tool_calls=True,
    supports_streaming=True,
    hooks=HookCapabilities(),
    config_format="markdown_frontmatter",
    agent_config_dir=".opencode",
    command_config_dir=".opencode",
    auth_mechanism="bearer",
)
