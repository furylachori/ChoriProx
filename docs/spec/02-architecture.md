# Component: Architecture

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/02-architecture.md)

## Directory Structure

```
qwen-adapter/
  src/
    qwen_proxy.py             # Proxy server (port 8000)
    qwen_tools.py             # Standalone client (port 7777)
    keyword_analyzer.py       # Tool extraction from LLM responses
    proxy_recovery_system.py  # Malformed call detection and auto-fix
    session_retry_manager.py  # Session retry tracking, circuit breaker
    smart_proxy_validation.py # Validation middleware
    tool_optimizer.py         # LLM-optimized tool descriptions
    memory/
      sync.py                 # Sync pipeline
      sota_client.py          # SOTA API client
      db.py                   # SQLite database layer
      graph.py                # Neo4j graph storage
      tools.py                # Memory tool definitions
  tests/                      # Unit, integration, E2E tests
  data/                       # SQLite databases (memory.db)
  config.json                 # Configuration
```

## Components

| Component | Purpose |
|-----------|---------|
| `qwen_proxy.py` | Proxy server (port 8000). Translates Qwen XML/JSON to OpenAI `tool_calls`. Forces `stream=false` to llama-server, converts back to SSE. |
| `qwen_tools.py` | Standalone CLI client. Talks directly to port 7777, parses tool calls, executes locally. |
| `proxy_recovery_system.py` | Malformed call detection, auto-fix, progressive escalation. |
| `session_retry_manager.py` | Session-based retry tracking with circuit breaker and stuck-pattern detection. |
| `smart_proxy_validation.py` | Tool call validation and error recovery middleware. |
| `tool_optimizer.py` | LLM-optimized tool descriptions (80-90% size reduction). |
| `keyword_analyzer.py` | Extracts tool calls from LLM text responses. |
| `memory/` | Memory system — sync pipeline, SOTA client, SQLite + Neo4j storage, tool definitions. |

## Request Flow (Integrated Mode)

1. KiloCode sends request to proxy (port 8000) with `stream: true`
2. Proxy forces `stream: false`, forwards to llama-server (port 7777)
3. Qwen responds with XML/JSON tool format
4. Proxy parses content for tool invocations (4 supported formats)
5. Proxy converts to OpenAI `tool_calls` format
6. Proxy streams SSE response back to KiloCode

## Tool Schema (6 core + 6 memory = 12 total)

Core tools: `read_file`, `write_file`, `edit_file`, `glob`, `bash`, `webfetch`.
Memory tools: `recall`, `search` (read-only); write tools removed (sync pipeline handles memory writes, task via client passthrough).

## System Prompt

Minimal 2-line prompt (shorter prompts reduce death loop risk):

```
You are an AI assistant with access to tools.
Use tools when needed. Keep responses concise.
```
