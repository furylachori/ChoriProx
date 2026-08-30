# Block: N-Client Abstraction Layer

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 1. Current State

Phase 1 (N-client abstraction layer) complete -- 10 files in src/clients/.
Phase 2 (wiring into qwen_proxy.py) complete -- 5 integration points active.
Phase 3 (tool loop extraction) complete -- src/memory/tool_loop.py extracted.
676 tests pass across unit, integration, and E2E suites.

---

## 2. N-Client Abstraction Layer

### 2.1 Interface & ABCs

**File:** src/clients/interface.py (307 lines)

**Enums:**
```python
class ToolFormat(Enum):      # OPENAI, ANTHROPIC, CUSTOM
class StreamingFormat(Enum):  # SSE_OPENAI, SSE_ANTHROPIC, JSON, CUSTOM
class MessageFormat(Enum):    # OPENAI, ANTHROPIC, CUSTOM
```

**Exception hierarchy:**
```
ClientsError (base)
+-- RegistryError
+-- ConfigError
+-- DetectionError
```

**ClientAdapter ABC** -- 4 abstract methods, rest concrete with defaults:
```python
class ClientAdapter(ABC):
    name: str = "unknown"

    @abstractmethod
    def get_capabilities(self) -> CapabilitySet: ...
    @abstractmethod
    def get_tool_names(self) -> set[str]: ...
    @abstractmethod
    def normalize_tool_args(self, tool_name: str, arguments: dict) -> dict: ...

    # Concrete with defaults:
    def denormalize_tool_args(self, tool_name: str, arguments: dict) -> dict
    def format_response(self, data: dict, wants_stream: bool) -> Any
    def format_sse_chunk(self, chunk: dict) -> str
    def load_config(self, project_root: str, loader: Optional[ConfigLoader] = None) -> ClientConfig
    def get_system_prompt(self, context: dict) -> str
    def interpret_command(self, user_input: str) -> Optional[dict]
    def format_tool_result(self, tool_name: str, result: Any) -> dict
    def format_error(self, error: Exception) -> dict
    def generate_tool_call_id(self, tool_name: str) -> str
    def get_plugin_adapter(self) -> Optional[Any]
    def validate_tool_result(self, tool_name: str, result: Any) -> bool
    def get_auth_headers(self) -> dict
```

**BackendAdapter ABC** -- 4 abstract methods:
```python
class BackendAdapter(ABC):
    @abstractmethod
    def format_tool_definitions(self, tools: list[dict]) -> list[dict]: ...
    @abstractmethod
    def format_messages(self, messages: list[dict]) -> FormattedRequest: ...
    @abstractmethod
    def format_streaming_response(self, raw_chunk: bytes) -> dict: ...
    @abstractmethod
    def get_request_headers(self) -> dict: ...
```

**Backend implementations:**
- OpenAIBackendAdapter (interface.py:217) -- passthrough for llama-server
- AnthropicBackendAdapter (interface.py:245) -- extracts system messages,
  formats content as [{"type":"text","text":...}]

**ConfigLoader** (interface.py:81) -- filesystem abstraction for testability:
list_dir, read_file, is_dir, is_file, path_join.

### 2.2 Dataclasses

**File:** src/clients/interface.py:50-77

```python
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
    hooks: HookCapabilities = field(default_factory=lambda: HookCapabilities())
    config_format: str = "json"
    agent_config_dir: str = ".opencode"
    command_config_dir: str = ".opencode"
    auth_mechanism: str = "bearer"

@dataclass
class ClientConfig:
    name: str = "unknown"
    provider_id: str = "openai"
    base_url: str = ""
    model: str = ""
    api_key: Optional[str] = field(default=None, repr=False)
    temperature: float = 0.1
    extra: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict

@dataclass
class FormattedRequest:
    messages: list[dict]
    system: str = ""
    extra_headers: dict = field(default_factory=dict)
```

### 2.3 Orchestrator Config

**File:** src/clients/orchestrator_config.py (42 lines)

```python
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
    memory_tool_names: frozenset[str] = frozenset({
        "recall", "search",
    })
    max_tool_iterations: int = 3
    tool_loop_timeout: int = 90
```

### 2.4 Registry

**File:** src/clients/registry.py (83 lines)

```python
class ClientRegistry:
    _instance: Optional["ClientRegistry"] = None
    _lock = threading.RLock()

    @classmethod
    def instance(cls) -> "ClientRegistry"        # Double-checked locking singleton
    @classmethod
    def reset(cls) -> None                       # Reset singleton (testing)
    @classmethod
    def _for_testing(cls) -> "ClientRegistry"    # Isolated instance for tests

    def register(self, adapter: ClientAdapter) -> None
    def get(self, name: str) -> ClientAdapter          # Raises RegistryError
    def active(self) -> ClientAdapter                  # Raises RegistryError if no active
    def set_active(self, name: str) -> None
    def detect(self, project_root: str) -> ClientAdapter  # Raises DetectionError
    def list_clients(self) -> list[str]
```

detect() checks os.path.isdir(project_root/caps.agent_config_dir) for each
registered adapter. Uses os.path.realpath() for path traversal safety (registry.py:66).

### 2.5 Model Registry

**File:** src/clients/model_registry.py (159 lines)

```python
@dataclass(frozen=True)
class ModelConfig:
    backend: str
    tool_parser: str
    system_prompt: str = "default"
    context_limit: int = 32000
    supports_vision: bool = False
    supports_parallel_tools: bool = True
```

**Tool parsers** (ABC ToolParser with .name property and .parse(content) -> list[dict]):
- OpenAIToolParser (model_registry.py:43) -- returns [] (passthrough)
- QwenXMLToolParser (`clients/model_registry.py:51`) -- delegates to `proxy.tool_parsing.parsers.parse_all_tools_from_content` (canonical, post-`qwen_xml.py`-deletion, M7 #2 / commit `45ecfec`). Note: the module file `clients/parsers/qwen_xml.py` is deleted; the class `QwenXMLToolParser` itself is live and delegates to the canonical 7-extractor parser.
- ToolParserRegistry (model_registry.py:61) -- maps parser name to factory.
Default fallback: "xml" (changed from "openai" in Phase 3).

**ModelRegistry** (model_registry.py:100) -- loads from proxy_models.json.
Supports prefix matching ("qwen3-*").
Default: localhost:7777, tool_parser="xml", context_limit=32000.

### 2.6 Capabilities

**File:** src/clients/capabilities.py (59 lines)

Three pre-built CapabilitySet instances:
- KILOCODE_CAPABILITIES -- all 16 hooks enabled, .kilo config dir, has_plugin_system=True
- OPENCODE_CAPABILITIES -- no hooks, .opencode config dir, config_format="markdown_frontmatter"
- ANTHROPIC_CAPABILITIES -- Anthropic message/tool format, .claude config dir,
  auth_mechanism="api_key"

### 2.7 Adapters

**KiloCodeAdapter** (src/clients/kilocode.py, 101 lines):
- Tool names: bash, read, glob, edit, write, webfetch, 6 memory tools, task
- normalize_tool_args: maps filePath->path, oldString->old_string, newString->new_string
- denormalize_tool_args: inverse of above
- load_config: reads .kilo/agent/*.json

**OpenCodeAdapter** (src/clients/opencode.py, 74 lines):
- Same tool names as KiloCode
- normalize_tool_args: passthrough (no mapping needed)
- load_config: reads .opencode/*.md frontmatter via yaml.safe_load

### 2.8 Parsers

**File:** src/proxy/tool_parsing/parsers.py:121+ (clients/parsers/qwen_xml.py deleted in M7 #2; the canonical 7-extractor parser is the single source of truth)

parse_all_tools_from_content(content: str) -> list[dict] -- 5 extraction strategies:
1. XML wrapper with JSON body (xml.etree.ElementTree)
2. Regex: <tools>JSON</tools>
3. Regex: <function=name>...<parameter=key>value
---

### 2.9 Corrected extraction count (2026-07-01)

**File:** src/proxy/tool_parsing/parsers.py:121+

The previous header said "7-extractor parser" but listed only 5 strategies. The canonical parser has **7 extraction strategies**:

1. XML wrapper with JSON body (xml.etree.ElementTree)
2. Regex: <tools>JSON</tools>
3. Regex: <function=name>...<parameter=key>value</parameter>...</function>
4. Function-call tags with attribute parsing
5. JSON code blocks (```json {...} ```)
6. YAML extraction (YAML document markers)
7. DSML extraction (Data Structure Markup Language)

Plus dedup by (name, json.dumps(args, sort_keys=True)).
