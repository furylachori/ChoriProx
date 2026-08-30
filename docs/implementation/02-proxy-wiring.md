# Block: Proxy Wiring

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 3. Proxy Wiring Points

5 integration points in qwen_proxy.py where adapters plug in.

### 3.1 Tool Filtering (qwen_proxy.py:1204-1244)

Uses adapter's tool set if available, else fallback to `ToolRegistry.get_default_tools()` (the legacy `DEFAULT_TOOL_NAMES` literal was deleted in Tier 3 Phase 3, commit `93f5396`).
Per-request header X-Client-Name overrides startup default.

```python
# qwen_proxy.py:1207-1218
_client_name = request.headers.get("X-Client-Name")
if _client_name and USE_ADAPTERS:
    _active_client = get_client(_client_name)  # per-request override
else:
    _active_client = ACTIVE_CLIENT              # module-level cached fallback
if _active_client and USE_ADAPTERS:
    OPENCODE_TOOL_NAMES = _active_client.get_tool_names()
else:
    OPENCODE_TOOL_NAMES = _registry.get_default_tools()
```

Then filters incoming tools to OPENCODE_TOOL_NAMES and appends MEMORY_TOOL_DEFINITIONS.

### 3.2 System Prompt (qwen_proxy.py:1301-1324)

Replaces entire system prompt with adapter's version:

```python
# qwen_proxy.py:1311-1315
_active_client = ACTIVE_CLIENT
if _active_client and USE_ADAPTERS:
    _system_prompt = _active_client.get_system_prompt({})
else:
    _system_prompt = OUR_SYSTEM_PROMPT
```

### 3.3 Client Detection (qwen_proxy.py:66-99)

Startup detection via config directory scan:

```python
# qwen_proxy.py:84-87
if USE_ADAPTERS:
    ACTIVE_CLIENT = detect_client(_project_root)
    print(f"[ADAPTER] Active client detected: {ACTIVE_CLIENT.name}")
```

Per-request override via X-Client-Name header (see 3.1).

### 3.4 Kill Switch (qwen_proxy.py:68-69)

Environment variable PROXY_USE_ADAPTERS=0 disables all adapter usage:

```python
USE_ADAPTERS = os.environ.get("PROXY_USE_ADAPTERS", "1") != "0"
```

When disabled, proxy uses the default toolset from `ToolRegistry.get_default_tools()` and `OUR_SYSTEM_PROMPT`.

### 3.5 Re-query Loop (qwen_proxy.py:1404-1448)

Memory tool interception loop extracted to memory.tool_loop:

```python
# qwen_proxy.py:1419-1430
result = run_memory_tool_loop(
    tool_calls_list=tool_calls_list,
    working_messages=working_messages,
    tools=llama_body.get("tools", []),
    model=llama_body.get("model", body.get("model", "default")),
    session_id=session_id,
    llama_url=LLAMA_URL,
    body=body,
    llama_body=llama_body,
    chat_histories=chat_histories,
    chat_histories_lock=chat_histories_lock,
)
```

Result is applied to the response message (tool_calls or content + finish_reason).
