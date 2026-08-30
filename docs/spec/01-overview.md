# Component: Overview

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/01-overview.md)

qwen-proxy enables KiloCode to communicate with llama-server running Qwen3-Coder locally. Since Qwen3-Coder does not emit OpenAI-compatible `tool_calls`, the proxy translates Qwen's XML/JSON tool format into the standard OpenAI `tool_calls` format and resolves the streaming protocol mismatch (KiloCode sends `stream: true`, but the proxy needs the full response before it can parse tool calls).

The system operates in two modes: **standalone** (`qwen_tools.py` talks directly to llama-server on port 7777) and **integrated** (KiloCode connects to `qwen_proxy.py` on port 8000, which forwards to llama-server on port 7777). The proxy intercepts responses, parses four recognized tool invocation formats (XML+JSON, custom function/parameter XML, markdown code block JSON, bare JSON), converts them to OpenAI `tool_calls`, and streams the result back as SSE.

The proxy also provides a persistent memory system backed by SQLite and Neo4j, a context budget architecture that enforces strict token limits to eliminate LLM timeouts, and a multi-layer recovery system that detects and auto-fixes malformed tool calls.
