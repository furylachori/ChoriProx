# 1. Imports
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from clients.orchestrator_config import HookCapabilities
from proxy.tool_pipeline.tool_registry import registry


# 2. Enums
class ToolFormat(Enum):
    OPENAI = auto()  # {"type":"function","function":{name,parameters}}
    ANTHROPIC = auto()  # {"name","input_schema"}
    CUSTOM = auto()


class StreamingFormat(Enum):
    SSE_OPENAI = auto()
    SSE_ANTHROPIC = auto()
    JSON = auto()
    CUSTOM = auto()


class MessageFormat(Enum):
    OPENAI = auto()  # {role, content}
    ANTHROPIC = auto()  # {role, content: [{type, text}]}
    CUSTOM = auto()


# 3. Exception hierarchy
class ClientsError(Exception):
    """Base exception for the clients layer."""


class RegistryError(ClientsError):
    """Raised for registry lookup failures."""


class ConfigError(ClientsError):
    """Raised for configuration loading failures."""


class DetectionError(ClientsError):
    """Raised when no client can be detected."""


# 4. CapabilitySet
@dataclass(frozen=True)
class CapabilitySet:
    tool_format: ToolFormat = ToolFormat.OPENAI
    streaming_format: StreamingFormat = StreamingFormat.SSE_OPENAI
    message_format: MessageFormat = MessageFormat.OPENAI
    supports_parallel_tool_calls: bool = True
    supports_streaming: bool = True
    supports_streaming_events: bool = False
    max_tools_per_request: Optional[int] = None
    supports_vision: bool = False
    has_plugin_system: bool = False
    hooks: "HookCapabilities" = field(default_factory=lambda: HookCapabilities())
    config_format: str = "json"
    agent_config_dir: str = ".opencode"
    command_config_dir: str = ".opencode"
    auth_mechanism: str = "bearer"


# 5. ClientConfig (for config normalizer)
@dataclass
class ClientConfig:
    name: str = "unknown"
    provider_id: str = "openai"
    base_url: str = ""
    model: str = ""
    api_key: Optional[str] = field(default=None, repr=False)
    temperature: float = 0.1
    extra: dict = field(default_factory=dict)


# 6. ConfigLoader protocol (for testability)
class ConfigLoader:
    """Filesystem abstraction for configuration loading.
    Default implementation uses os/open. Tests can inject mocks."""

    def list_dir(self, path: str) -> list[str]:
        import os

        return os.listdir(path)

    def read_file(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def is_dir(self, path: str) -> bool:
        import os

        return os.path.isdir(path)

    def is_file(self, path: str) -> bool:
        import os

        return os.path.isfile(path)

    def path_join(self, *parts: str) -> str:
        import os

        return os.path.join(*parts)


# 7. ClientAdapter ABC (4 abstract methods, rest concrete with defaults)
class ClientAdapter(ABC):
    name: str = "unknown"

    @abstractmethod
    def get_capabilities(self) -> CapabilitySet:
        """Return the capability set for this client."""
        ...

    @abstractmethod
    def get_tool_names(self) -> set[str]:
        """Return set of tool names this client supports."""
        ...

    @abstractmethod
    def normalize_tool_args(self, tool_name: str, arguments: dict) -> dict:
        """Normalize client-specific tool arg names to canonical form."""
        ...

    def denormalize_tool_args(self, tool_name: str, arguments: dict) -> dict:
        """Convert canonical arg names back to client-specific form."""
        return dict(arguments)

    def format_response(self, data: dict, wants_stream: bool) -> Any:
        """Format response for this client."""
        return data

    def format_sse_chunk(self, chunk: dict) -> str:
        """Format a single SSE chunk line."""
        return f"data: {json.dumps(chunk)}\n\n"

    def load_config(
        self, project_root: str, loader: Optional["ConfigLoader"] = None
    ) -> ClientConfig:
        """Load client configuration from project root."""
        return ClientConfig(name=self.name)

    def get_system_prompt(self, context: dict) -> str:
        """Return system prompt for this client."""
        return "You are an AI assistant with access to tools."

    def format_tool_result(self, tool_name: str, result: Any) -> dict:
        """Format tool execution result."""
        return {"role": "tool", "content": str(result)}

    def format_error(self, error: Exception, role: str = "tool") -> dict:
        """Format error for this client."""
        return {"role": role, "content": f"Error: {error}"}

    def generate_tool_call_id(self, tool_name: str) -> str:
        """Generate unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"

    def get_plugin_adapter(self) -> Optional[Any]:
        """Return plugin adapter if supported."""
        return None

    def validate_tool_result(self, tool_name: str, result: Any) -> bool:
        """Validate tool result structure."""
        return True

    def get_auth_headers(self) -> dict:
        """Return auth headers for this client."""
        return {"Authorization": "Bearer not-needed"}


# 8. ToolCall dataclass (canonical format)
@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


# 9. FormattedRequest dataclass
@dataclass
class FormattedRequest:
    messages: list[dict]
    system: str = ""
    extra_headers: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.messages, list):
            raise TypeError(f"messages must be a list, got {type(self.messages)}")
