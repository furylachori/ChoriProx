# `src/clients/AGENTS.md` — Client Adapters

> The `ClientAdapter` ABC + OpenCode/KiloCode adapters + ClientRegistry +
> ModelRegistry + ToolParserRegistry. Note: **opencode/kilocode
> `normalize_tool_args` are VESTIGIAL/DEAD** — the schema-driven path in
> `proxy/tool_parsing/normalize.py` supersedes them. Never wire per-client
> casing tables.

## Module Purpose

Owns:
- ClientAdapter ABC + dataclasses (`interface.py`)
- Capability constants (`capabilities.py`)
- Adapter configurations (`orchestrator_config.py`)
- OpenCode + KiloCode adapters (`opencode.py`, `kilocode.py`)
- ClientRegistry singleton + project-root detection (`registry.py`)
- ModelRegistry + ToolParserRegistry (`model_registry.py`)
- SOTA layering firewall (`sota_client.py`) — the ONLY legal path for
  non-`memory` code to reach SOTA

## Files in this module

| File | LOC | Purpose |
|---|---|---|
| `__init__.py` | 52 | Public façade — re-exports `ClientAdapter`, `CapabilitySet`, `ClientConfig`, exceptions, `ProxyFeatures`, `HookCapabilities`, `register_client` / `get_client` / `detect_client` / `active_client` / `list_clients` |
| `interface.py` | 193 | `ClientAdapter` ABC (3 abstract + 8 concrete-with-defaults methods) + `CapabilitySet` (frozen) + `ClientConfig` + `ToolFormat`/`StreamingFormat`/`MessageFormat` enums + exception hierarchy + `ConfigLoader` protocol + `ToolCall` + `FormattedRequest` |
| `opencode.py` | 51 | `OpenCodeAdapter` — YAML-frontmatter agent configs from `.opencode/`, **identity passthrough** for `normalize_tool_args` |
| `kilocode.py` | 77 | `KiloCodeAdapter` — JSON agent configs from `.kilo/agent/`, **DEAD code** `normalize_tool_args` that rewrites `filePath↔path`, `oldString↔old_string`, `newString↔new_string` |
| `registry.py` | 83 | `ClientRegistry` singleton (double-checked locking) + `register` / `get` / `active` / `set_active` / `detect` (filesystem-rooted) + `_for_testing` |
| `model_registry.py` | 227 | `ModelConfig` (frozen) + `ToolParser` ABC + `OpenAIToolParser` (STUB) + `QwenXMLToolParser` (delegates to canonical parser) + `ModelRegistry` + `ToolParserRegistry` |
| `orchestrator_config.py` | 44 | `HookCapabilities` + `ProxyFeatures` (frozen dataclasses) + `get_features_for_client(client_hint)` (currently returns `ProxyFeatures()` regardless of input) |
| `capabilities.py` | 46 | `KILOCODE_CAPABILITIES` + `OPENCODE_CAPABILITIES` (frozen module-level instances) |
| `sota_client.py` | 37 | `call_sota_extraction` — the ONLY legal path from `enhancer/` to `memory/*` SOTA code |
| `parsers/__init__.py` | 0 | Empty package marker. **`qwen_xml.py` was DELETED in commit `45ecfec` (M7 #2)** — superseded by `proxy/tool_parsing/parsers.parse_all_tools_from_content` |

## ClientAdapter ABC (interface.py:112)

```python
class ClientAdapter(ABC):
    name: str = "unknown"                                          # class attribute

    # Abstract (3 methods) — every adapter MUST implement
    def get_capabilities(self) -> CapabilitySet: ...                # line 116
    def get_tool_names(self) -> set[str]: ...                      # line 121
    def normalize_tool_args(self, tool_name, arguments) -> dict:   # line 126

    # Concrete with defaults (8 methods) — adapter may override
    def denormalize_tool_args(self, tool_name, arguments) -> dict: ...  # line 130
    def format_response(self, response) -> dict: ...                    # line 134
    def format_sse_chunk(self, chunk) -> str: ...                      # line 138
    def load_config(self, project_root, loader=None) -> ClientConfig:   # line 142
    def get_system_prompt(self, context) -> str: ...                   # line 148
    def format_tool_result(self, result) -> dict: ...                  # line 152
    def format_error(self, error) -> dict: ...                         # line 156
    def generate_tool_call_id(self) -> str: ...                        # line 160
    def get_plugin_adapter(self): ...                                  # line 164
    def validate_tool_result(self, result) -> bool: ...                # line 168
    def get_auth_headers(self) -> dict: ...                            # line 172
```

### `CapabilitySet` (frozen dataclass, line 52)

```python
@dataclass(frozen=True)
class CapabilitySet:
    tool_format: ToolFormat
    streaming_format: StreamingFormat
    message_format: MessageFormat
    supports_parallel_tool_calls: bool
    supports_streaming: bool
    supports_streaming_events: bool
    max_tools_per_request: int
    supports_vision: bool
    has_plugin_system: bool
    hooks: HookCapabilities
    config_format: str
    agent_config_dir: str
    command_config_dir: str
    auth_mechanism: str
```

`frozen=True` — mutation raises `FrozenInstanceError`.

### `ClientConfig` (mutable dataclass, line 71)

`api_key` has `repr=False` — prevents accidental logging via `repr(cfg)`.

### `ConfigLoader` protocol (line 82)

Filesystem-isolating protocol: `list_dir`, `read_file`, `is_dir`, `is_file`,
`path_join`. The default `ClientAdapter.load_config` returns
`ClientConfig(name=self.name)` and never reads any filesystem.

## OpenCodeAdapter (opencode.py)

```python
class OpenCodeAdapter(ClientAdapter):
    name = "opencode"
    # get_capabilities() → OPENCODE_CAPABILITIES (no hooks, markdown_frontmatter, .opencode)
    # get_tool_names() → registry.get_default_tools()  # lazy import
    # normalize_tool_args(tool_name, arguments) → dict(arguments)  # identity passthrough
    # load_config(project_root, loader=None) → reads .opencode/*.md frontmatter
    # get_system_prompt(context) → "You are an AI coding assistant with access to tools.\n..."
```

### KEY INVARIANT (line 21-22)

`normalize_tool_args` is **identity passthrough** with a comment explicitly
citing the schema-driven normalizer:

```python
def normalize_tool_args(self, tool_name, arguments):
    """Identity passthrough. Schema-driven normalization in
    proxy.tool_parsing.normalize supersedes this."""
    return dict(arguments)
```

**This is a pinned test invariant** at `tests/integration/clients/test_adapter_wiring.py:90-94`.

### Dead imports

- Line 1: `import json` is unused (dead code).
- `yaml` is lazy-imported inside `load_config` (line 27).

## KiloCodeAdapter (kilocode.py)

```python
class KiloCodeAdapter(ClientAdapter):
    name = "kilocode"
    # get_capabilities() → KILOCODE_CAPABILITIES (all hooks enabled, JSON config, .kilo)
    # get_tool_names() → registry.get_default_tools()  # lazy import
    # normalize_tool_args(tool_name, arguments) → REWRITES filePath↔path, oldString↔old_string, newString↔new_string
    # denormalize_tool_args(tool_name, arguments) → INVERSE rewrite
    # load_config(project_root, loader=None) → reads .kilo/agent/*.json
    # get_system_prompt(context) → "You are an AI assistant with access to tools.\n..." (no "coding")
```

### `normalize_tool_args` / `denormalize_tool_args` — DEAD IN PRODUCTION

**CRITICAL:** `grep -r "normalize_tool_args\|denormalize_tool_args" src/` returns
matches ONLY in `src/clients/{interface.py, opencode.py, kilocode.py}`. **No
production caller in `src/` outside the clients/ package itself.** The
methods are wired with non-trivial logic (camelCase↔snake_case rewrites for
`read`/`write`/`edit`), but production code never calls them.

The schema-driven path in `proxy/tool_parsing/normalize.py` is the actual
production normalizer — it auto-adapts to any client via
`valid_params=...` from the incoming `tools[]` schema.

### Dead imports

- Line 3: `import os` is unused (dead code).

## ClientRegistry (registry.py)

```python
class ClientRegistry:
    _instance: Optional["ClientRegistry"] = None
    _lock: threading.RLock
    _clients: Dict[str, ClientAdapter]
    _active: Optional[str]

    # Public API
    @classmethod
    def instance() -> "ClientRegistry": ...         # line 15 — double-checked locking singleton
    @classmethod
    def reset() -> None: ...                       # line 23 — clears singleton (test-only)
    @classmethod
    def _for_testing() -> "ClientRegistry": ...    # line 28 — bypasses singleton, no lock

    def register(self, adapter) -> None: ...       # line 35 — stores in _clients
    def get(self, name) -> ClientAdapter: ...      # line 39 — raises RegistryError if absent
    def active(self) -> ClientAdapter: ...         # line 48 — raises RegistryError if _active is None
    def set_active(self, name) -> None: ...       # line 54
    def detect(self, project_root: str) -> ClientAdapter: ...  # line 60
    def list_clients(self) -> list[str]: ...       # line 81
```

### `detect(project_root)` (line 60) — first-match-wins

1. `realpath(project_root)` — sanitizes symlinks
2. Validates non-empty + is-dir (raises `DetectionError` if invalid)
3. Iterates `_clients` in registration order
4. For each adapter checks `real_root/<caps.agent_config_dir>` exists
5. First match wins — sets `_active = name` and returns adapter
6. Raises `DetectionError` if no match

**Registration order matters.** Register opencode first → `.kilo`+`.opencode`
both exist → returns opencode. Reverse registration order → returns kilocode.

Pinned by `tests/integration/clients/test_registry_detection.py::test_detect_prefers_first_match_order`.

### `_for_testing()` (line 28)

Bypasses `_lock`. Tests rely on this for parallel isolation. Documented
as test-only.

## ModelRegistry + ToolParserRegistry (model_registry.py)

### `ModelConfig` (frozen dataclass, line 19)

```python
@dataclass(frozen=True)
class ModelConfig:
    backend: str
    tool_parser: str
    name: str = ""
    context_limit: int = 32000
    supports_vision: bool = False
    supports_parallel_tools: bool = True
    api_key_env: str = ""
    backend_model: str = ""
    ignore_subagents: bool = False
    memory_enabled: bool = True
    memory_injection_enabled: bool = False
    raw_exchange_count: int = 0
    budget_ratio: Optional[float] = None
```

### `ToolParser` ABC + implementations (line 35-58)

```python
class ToolParser(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def parse(self, content: str) -> list[dict]: ...

class OpenAIToolParser(ToolParser):
    """STUB. Returns []."""
    name = "openai"
    def parse(self, content): return []

class QwenXMLToolParser(ToolParser):
    """Delegates to canonical 7-extractor parser."""
    name = "xml"
    def parse(self, content):
        from proxy.tool_parsing.parsers import parse_all_tools_from_content
        return parse_all_tools_from_content(content)  # lazy import
```

**WARNING:** `OpenAIToolParser.parse` returns `[]` unconditionally. If ever
called on a real LLM response, would silently lose all tool calls. The
production code path in `tool_interception.py:84-85` calls
`parse_all_tools_from_content` directly, **bypassing** `ToolParserRegistry`.

### `ModelRegistry` (line 94)

Reads `proxy_models.json` on init. `get(model_name)` has 5 fallback layers:
1. Explicit name match
2. Wildcard prefix match (`prefix*` with longest-prefix wins)
3. Case-insensitive match
4. Case-insensitive with hyphen/underscore normalization
5. Match against `cfg.backend_model`
6. **Else default** — never raises

`auto_detect(request_body)` calls `get(request_body.get("model", ""))`.

### `ALLOWED_SCHEMES = {"http", "https"}` (line 13)

URL scheme validation rejects bare schemes. `MAX_NAME_LENGTH = 128`,
`NAME_PATTERN = r"^[\w\s\-./]+$"`.

### `ToolParserRegistry` (line 61)

Pre-instantiates all parsers at construction (lines 68-72). `get(parser_name, strict=False)`
returns the parser; unknown names log WARNING and fall back to `"xml"`.

## `orchestrator_config.py` — ProxyFeatures

```python
@dataclass(frozen=True)
class ProxyFeatures:
    version: int = 1
    intercepts_memory: bool = True
    runs_tool_loop: bool = True
    injects_context: bool = True
    replaces_system_prompt: bool = True
    memory_tool_names: frozenset = _memory_tool_names_default  # captured at import time
    max_tool_iterations: int = 15
    ticket_mode_enabled: bool = True
    tool_loop_timeout: int = 90
    intercepts_commands: bool = True
    task_commands_enabled: bool = True
```

### `get_features_for_client(client_hint: str = "unknown") -> ProxyFeatures` (line 42)

**Currently always returns `ProxyFeatures()`** — the `client_hint` parameter
is unused. Stub for future per-client feature negotiation.

### `_memory_tool_names_default = frozenset(_tool_reg.get_memory_tool_names())` (line 4)

Captured at import time. Mutating `ToolRegistry` AFTER
`clients.orchestrator_config` is imported does NOT update
`ProxyFeatures().memory_tool_names`.

## `sota_client.py` — the layering firewall

```python
def call_sota_extraction(prompt, system_prompt="", *, temperature=None,
                         max_tokens=None, session_id="enhancer") -> str:
    """Route through clients/ for non-memory callers.
    
    The ONLY legal path from enhancer/ (or any non-memory code) to SOTA.
    Returns str(result.parsed).strip().
    """
```

`SOTAChannel(mode=SOTAMode.EXTRACTION, ...)` is constructed per call (no
caching); the underlying OpenAI HTTP client IS a singleton via
`_sota_http.get_sota_client()`.

The docstring states the layering rule verbatim:
> "clients/ is the only package allowed to reach into memory/ for SOTA transport details."

## Key Invariants

1. **`ClientRegistry` is a singleton** with double-checked locking.
2. **First-match-wins-by-registration-order** for `detect()`. Order matters when both `.kilo` and `.opencode` exist.
3. **`CapabilitySet` is frozen.** `OPENCODE_CAPABILITIES.has_plugin_system = True` raises `FrozenInstanceError`.
4. **`OpenCodeAdapter.normalize_tool_args` is identity passthrough.** Pinned test invariant.
5. **`KiloCodeAdapter.normalize_tool_args` is DEAD in production.** Wired with casing-table logic but never called outside the clients/ package.
6. **The schema-driven path in `proxy/tool_parsing/normalize.py` is the actual production normalizer.** Auto-adapts to any client.
7. **`ModelRegistry.get` never raises.** Falls through to default.
8. **`OpenAIToolParser` is a STUB.** The production tool parsing happens via `parse_all_tools_from_content` directly.
9. **`QwenXMLToolParser` delegates to canonical parser.** The `clients/parsers/qwen_xml.py` was deleted in commit `45ecfec` (M7 #2).
10. **`ProxyFeatures` is frozen + `_memory_tool_names_default` is captured at import time.** Hot-reload won't pick up ToolRegistry changes.
11. **`get_features_for_client` ignores its parameter.** Returns `ProxyFeatures()` regardless.

## Landmines

- **Wiring opencode-specific or kilocode-specific casing logic into `normalize_tool_args`** — re-introduces the casing-table antipattern that `proxy/tool_parsing/normalize.py` was designed to eliminate. Per AGENTS.md "Tool-call argument normalization" landmine, these methods exist ONLY as no-op stubs.
- **`tests/integration/clients/test_canary_preservation.py` pins that the canary prefix survives both adapters' transformations.** If you wire `normalize_tool_args` into production, that test catches it.
- **`DetectionError` is raised in `registry.py:64,68,76`** (detect validation + no-match). `RegistryError` is raised at `registry.py:42,51,57` (duplicate register, no active, not found). `ConfigError` is exported but unused.
- **`get_tool_names()` lazy import of `proxy.tool_pipeline.tool_registry.registry`** — `opencode.py:17`, `kilocode.py:17`. This is INTENTIONAL to avoid the load-order trap (don't move to module-level).
- **`CapabilitySet.hooks` field is unused** — grep `src/` confirms no code reads `capabilities.hooks` for anything. The data is informative but vestigial.
- **`proxy.tool_pipeline.tool_registry.registry` is imported at module level in `interface.py:10`** but never referenced — vestigial import, top-level side effect.
- **`OpenAIToolParser` returning `[]`** — if `tool_parser == "openai"` in `proxy_models.json` and the parser is actually invoked, the proxy gets an empty tool list from this parser. Most explicit entries still set `tool_parser: "openai"` (default `ModelConfig` is `"xml"`).
- **`budget_ratio=None` default** — `proxy/model_registry.py:179-180` falls back to `proxy.state.CONTEXT_BUDGET_RATIO` (= 0.75).
- **`MAX_NAME_LENGTH = 128`** — generous; real LLM model names are < 64. A model name with special chars (e.g. `:` for namespaced OpenAI `gpt-4:custom`) is rejected by `NAME_PATTERN`.
- **`name.startswith("__")` for client names** — `_for_testing()` bypasses `_lock`. Tests rely on this for parallel isolation.

## Code Conventions

```python
# CORRECT — get a client
from clients import get_client, detect_client
adapter = get_client("opencode")
adapter = detect_client(project_root)

# CORRECT — use the canonical normalizer (NOT the adapter's normalize_tool_args)
from proxy.tool_parsing import (
    build_valid_params_by_tool, normalize_tool_arguments, make_tool_calls, extract_file_path,
)
vpbt = build_valid_params_by_tool(body.get("tools", []))
args = normalize_tool_arguments(name, args, valid_params=vpbt.get(name))

# WRONG — adapter-specific casing
adapter.normalize_tool_args("read", {"filePath": "..."})  # dead in production

# CORRECT — fresh instance for tests (parallel-safe)
from clients.registry import ClientRegistry
fresh = ClientRegistry._for_testing()

# CORRECT — SOTA layering firewall
from clients.sota_client import call_sota_extraction
result = call_sota_extraction(prompt, system_prompt=sp, temperature=0.2, max_tokens=512, session_id="my_session")

# WRONG — reaching into memory/ for SOTA from non-memory code
from memory.sota_client import SOTAClient  # not allowed for enhancer/, skills/, etc.
```

## External Dependencies

**Imports from:**
- `proxy.tool_pipeline.tool_registry.registry` — `get_default_tools()` (lazy, inside `get_tool_names`).
- `proxy.tool_parsing.parsers.parse_all_tools_from_content` — QwenXMLToolParser delegation (lazy).
- `memory.sota_channel.SOTAChannel` + `memory._sota_http.SOTAMode` — `sota_client.py` (lazy).

**Imported by:**
- `src/proxy/model_registry.py` — registers both adapters at module load.
- `src/enhancer/core.py:90` — calls `clients.sota_client.call_sota_extraction` (the layering firewall).
- `src/devtools/debug_session.py:1455` — lazy.
- Tests under `tests/unit/clients/`, `tests/integration/clients/`, `tests/integration/test_*`.

## Cross-References

- `src/proxy/tool_parsing/AGENTS.md` — THE LANDMINE. The schema-driven normalizer supersedes the dead `normalize_tool_args` methods.
- `src/proxy/AGENTS.md` — `proxy.model_registry` is the canonical entry for model routing.
- `src/enhancer/AGENTS.md` — only valid caller of `clients.sota_client`.
- `tests/AGENTS.md` (planned) — `tests/integration/clients/test_adapter_wiring.py` pins the identity-passthrough invariant.