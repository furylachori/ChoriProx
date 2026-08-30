# Qwen3-Coder Local Tool Calling via TurboQuant

Run Qwen3-Coder locally on Apple Silicon with tool execution capabilities,
both standalone (CLI) and integrated into Kilocode/OpenCode.

---

## Architecture

```
                   Standalone CLI                    Kilocode Integration
                   ──────────────                    ────────────────────

                   qwen_tools.py                     Kilocode CLI / VS Code
                   (parses tool calls,                      |
                    executes locally,               /model qwencode
                    loops until done)                       |
                        |                                  v
                        |                          qwen_proxy.py :8000
                        |                          (intercepts stream,
                        |                           forces stream=false,
                        |                           parses Qwen XML/JSON,
                        |                           converts to OpenAI
                        |                           tool_calls format,
                        |                           streams back as SSE)
                        |                                  |
                        v                                  v
                   llama-server :7777  <--------------------
                   (Qwen3-Coder GGUF + TurboQuant KV cache)
```

**The core problem solved by the proxy**: Qwen3-Coder does not emit OpenAI-
compatible `tool_calls` in its responses. Instead it wraps tool invocations
in XML (`<tools>...</tools>`) or a custom `<function=name>` syntax. Kilocode
expects standard OpenAI `tool_calls` objects. The proxy sits in the middle,
translates the format, and handles the streaming protocol mismatch (kilocode
sends `stream: true`, but we need the full response to parse tool calls).

---

## Prerequisites

| Requirement | Details |
|---|---|
| Hardware | Apple M4 Pro (24 GB unified RAM) or similar |
| macOS | With Xcode Command Line Tools |
| Python 3.8+ | System Python or pyenv |
| pip packages | `requests`, `flask`, `openai`, `neo4j` |
| CMake 3.16+ | For building llama.cpp |
| Neo4j | Graph database for memory system (`brew install neo4j`) |
| Kilocode CLI | `npm install -g @kilocode/cli` (for integration only) |

Install Python dependencies:

```bash
pip3 install requests flask openai neo4j
```

Install Neo4j (macOS):

```bash
brew install neo4j
```

Set Neo4j password:

```bash
neo4j-admin dbms set-initial-password your-neo4j-password
```

---

## File Reference

### Core Files (you maintain these)

| File | Purpose |
|---|---|
| `qwen_proxy.py` | **Proxy server**. Sits between kilocode and llama-server. Converts Qwen tool format to OpenAI format. Handles streaming. Port 8000. |
| `qwen_tools.py` | **Standalone CLI client**. Talks directly to llama-server, parses tool calls, executes them locally, loops until final answer. Port 7777. |
| `SPEC.md` | System architecture specification (merged from original docs). |
| `IMPLEMENTATION_SPEC.md` | Implementation details and readiness spec (merged from original docs). |
| `MEM_TOOLS_PROPOSAL_SPEC.md` | Upcoming memory tool changes proposal. |

### Kilocode Configuration (global -- applies to all projects)

| File | Purpose |
|---|---|
| `~/.config/kilo/kilo.jsonc` | Main kilocode config. Defines the `qwencode-proxy` provider and `qwencode` model pointing at port 8000. |
| `~/.config/kilo/agent/qwencode.json` | Agent definition for the qwencode model. |
| `~/.config/kilo/command/qwencode.md` | `/qwencode` command documentation. |

### Kilocode Configuration (project-local -- this repo only)

| File | Purpose |
|---|---|
| `.kilo/agent/qwencode.json` | Same as global agent config (project override). |
| `.kilo/command/qwencode.md` | Same as global command config (project override). |

### Memory System Files

| File | Purpose |
|---|---|
| `src/memory/` | Memory system module (sync pipeline, SOTA client, database). |
| `src/memory/sync.py` | Sync pipeline - extracts memories from conversations. |
| `src/memory/sota_client.py` | SOTA API client for memory extraction. |
| `src/memory/db.py` | SQLite database layer for message storage and extraction batches. |
| `src/memory/graph.py` | Neo4j graph storage for memory relationships. |
| `src/memory/tools.py` | Memory tool definitions for Kilocode. |
| `data/memory.db` | Production SQLite database. |
| `data/memory_test.db` | Test SQLite database (created by test fixtures). |

### Test / Utility Files (safe to delete)

| File | Purpose |
|---|---|
| `test_proxy.py` | Test script for proxy non-streaming mode. |
| `test_tool_calls.py` | Test script for OpenAI tool_calls format. |
| `run_proxy_test.sh` | Shell script to start proxy and run tests. |
| `start_qwencode.sh` | Convenience script to start the proxy. |
| `qwen_tools.py.backup` | Backup of earlier qwen_tools.py. |
| `test.txt`, `test_output.txt`, `test_proxy_output.txt`, `test_tool_calls.txt` | Test artifacts. |
| `KILOCODE_INTEGRATION.md` | Archived — merged into `SPEC.md`. |
| `kilocode/` | Empty directory from earlier experiments. Safe to delete. |

### Not Documented Here

| Directory | Notes |
|---|---|
| `llama-cpp-turboquant/` | The TurboQuant fork of llama.cpp. Separate project -- not covered here. |

---

## Startup Procedure

### Step 1: Start llama-server (port 7777)

```bash
./llama-cpp-turboquant/build/bin/llama-server \
  -m /path/to/local/file \
  --jinja \
  --port 7777 \
  --cache-type-k turbo4 \
  --cache-type-v turbo4 \
  -c 32768 \
  --no-warmup
```

Verify it's running:

```bash
curl -s http://localhost:7777/v1/models | python3 -m json.tool
```

### Step 2a: Standalone CLI usage (no proxy needed)

```bash
python3 qwen_tools.py "Read the README.md and tell me what it's about"
python3 qwen_tools.py "Create a file called hello.txt with 'hello world' inside"
python3 qwen_tools.py "List all Python files in this directory"
```

This talks directly to port 7777, parses Qwen's tool format internally, and
executes tools in the same Python process.

### Step 2b: Kilocode integration (requires proxy)

Start Neo4j (required for memory system):

```bash
# macOS with Homebrew
neo4j start

# Verify it's running
neo4j status
```

Start the proxy:

```bash
PYTHONPATH=src \
NEO4J_AUTH="neo4j:your-neo4j-password" \
PROXY_SOTA_API_KEY="sk-..." \
PROXY_SOTA_ENDPOINT="https://api.minimax.io/v1" \
PROXY_SOTA_MODEL="MiniMax-M2.7" \
python3 src/qwen_proxy.py
```

You should see:

```
============================================================
Qwen Tool Proxy for Kilocode
============================================================
  Listening on    : http://0.0.0.0:8000
  Forwarding to   : http://localhost:7777/v1/chat/completions
  Streaming       : kilocode streams -> proxy buffers -> streams back
============================================================
```

Then in kilocode:

```
/model qwencode
```

Select **"Qwen3-Coder with Tools"** under the **"Qwen3-Coder with Tool Proxy"** provider.

**Important**: Do NOT use "Qwen3-Coder TurboQuant" for tool tasks in kilocode.
That model connects directly to port 7777 and bypasses the proxy, so tool
calls will be printed as raw XML instead of being executed.

---

## Configuration Details

### `~/.config/kilo/kilo.jsonc`

Two providers are registered:

```jsonc
{
  "provider": {
    // Direct connection -- no tool conversion, for chat only
    "llama-cpp-turbo": {
      "name": "Llama.cpp TurboQuant",
      "api": "http://localhost:7777/v1",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "apiKey": "not-needed",
        "baseURL": "http://localhost:7777/v1"
      },
      "models": {
        "qwen3-coder-turbo": {
          "id": "qwen3-coder-turbo",
          "name": "Qwen3-Coder TurboQuant",
          "temperature": true,
          "tool_call": true,
          "cost": { "input": 0, "output": 0 },
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    },

    // Via proxy -- tool calls converted to OpenAI format
    "qwencode-proxy": {
      "name": "Qwen3-Coder with Tool Proxy",
      "api": "http://localhost:8000/v1",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "apiKey": "not-needed",
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwencode": {
          "id": "qwencode",
          "name": "Qwen3-Coder with Tools",
          "temperature": true,
          "tool_call": true,
          "cost": { "input": 0, "output": 0 },
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

### `~/.config/kilo/agent/qwencode.json`

```json
{
  "name": "qwencode",
  "description": "Qwen3-Coder with local tool execution via proxy",
  "provider": "openai",
  "config": {
    "apiKey": "not-needed",
    "baseURL": "http://localhost:8000/v1"
  },
  "model": "qwen3-coder",
  "capabilities": ["tools"],
  "temperature": 0.1
}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_AUTH` | Yes | — | Neo4j auth credentials (`neo4j:password`) |
| `PROXY_SOTA_API_KEY` | Yes | — | SOTA API key for memory extraction |
| `PROXY_SOTA_ENDPOINT` | Yes | `https://api.minimax.io/v1` | SOTA API endpoint |
| `PROXY_SOTA_MODEL` | Yes | `MiniMax-M2.7` | SOTA model name |
| `PROXY_TEST_DB` | No | `data/memory.db` | SQLite database path (set to use test database) |

### Proxy Dependencies

- **Neo4j**: Graph database for memory storage. Install via Homebrew (`brew install neo4j`) or Docker.
- **SOTA API**: External API for memory extraction from conversations.
- **llama-server**: Local LLM server on port 7777.

---

## How the Proxy Works (`qwen_proxy.py`)

### Request Flow

1. Kilocode sends a request to `http://localhost:8000/v1/chat/completions`
   with `stream: true` and kilocode's own tool definitions.
2. The proxy **forces `stream: false`** and forwards to llama-server at
   `http://localhost:7777/v1/chat/completions`.
3. The proxy receives a complete JSON response from llama-server.
4. It parses the `content` field looking for tool invocations in these
   formats (in priority order):

   **Format 1** -- XML wrapper with JSON body:
   ```
   <tools>
   {"name": "write_file", "arguments": {"path": "foo.txt", "content": "bar"}}
   </tools>
   ```

   **Format 2** -- Custom function/parameter XML:
   ```
   <function=write_file>
   <parameter=path>/path/to/local/file
   <parameter=content>bar</parameter>
   </function>
   ```

   **Format 3** -- JSON in a markdown code block:
   ````
   ```json
   {"name": "write_file", "arguments": {"path": "foo.txt", "content": "bar"}}
   ```
   ````

   **Format 4** -- Bare JSON object in content:
   ```
   {"name": "write_file", "arguments": {"path": "foo.txt", "content": "bar"}}
   ```

5. If a tool call is found, the proxy converts it to OpenAI `tool_calls` format:
   ```json
   {
     "tool_calls": [{
       "id": "call_1234567890",
       "type": "function",
       "function": {
         "name": "write_file",
         "arguments": "{\"path\": \"foo.txt\", \"content\": \"bar\"}"
       }
     }],
     "content": null
   }
   ```

6. The proxy **streams the response back as SSE** to kilocode, which then
   executes the tool using its own built-in tool runner.

### Proxy Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Main proxy endpoint. Handles streaming and tool conversion. |
| `/v1/models` | GET | Proxies the models list from llama-server. |
| `/health` | GET | Returns `{"status": "ok"}`. |

---

## Standalone Client (`qwen_tools.py`)

### Tool Definitions

The standalone client defines its own tools with custom names:

| Tool Name | Description | Parameters |
|---|---|---|
| `read_file` | Read file contents | `path`, `offset` (default 1), `limit` (default 2000) |
| `write_file` | Write/overwrite a file | `path`, `content` |
| `edit_file` | Replace first occurrence of text | `path`, `old_string`, `new_string` |
| `file_glob` | Find files by glob pattern | `pattern`, `root_dir` (default `.`) |
| `run_shell_command` | Execute a shell command | `command` |
| `web_fetch` | Fetch a URL | `url`, `timeout` (default 10) |

### Conversation Loop

```
User prompt
    |
    v
Send to llama-server with tools + stream=false
    |
    v
Parse response content for tool call
    |
    +--> Tool found? Execute it, append result, loop back
    |
    +--> No tool? Return content as final answer
    |
    (max 10 iterations to prevent infinite loops)
```

---

## Troubleshooting

### Proxy returns 500 errors

**Most likely cause**: Kilocode sends `stream: true` but an older version of
the proxy tried to `response.json()` on a streaming SSE response. Make sure
you are running the latest `qwen_proxy.py` which forces `stream: false` to
llama-server.

Check the proxy terminal output for detailed error messages and tracebacks.

### Tools display as text instead of executing

You are using the wrong model. In kilocode, use `/model qwencode` to select
the proxy-backed model. The direct model (`Qwen3-Coder TurboQuant`) bypasses
the proxy and returns raw Qwen XML that kilocode cannot execute.

### llama-server not responding

```bash
# Check if it's running
ps aux | grep llama-server

# Test it directly
curl -s http://localhost:7777/v1/models
```

### Proxy not running

```bash
# Check health
curl -s http://localhost:8000/health

# If port in use
lsof -ti:8000 | xargs kill -9
python3 qwen_proxy.py
```

### Slow first response

Normal. The first request takes 30-60 seconds for model loading. Subsequent
requests are 2-10 seconds depending on prompt length.

---

## Performance Notes

| Metric | Value |
|---|---|
| First request latency | 30-60s (model loading) |
| Subsequent requests | 2-10s |
| Context window | 32,768 tokens (TurboQuant) |
| RAM usage | ~15-20 GB on M4 Pro 24 GB |
| Quantization | turbo4 KV cache (3.8-6.4x reduction) |

To reduce memory, lower the context window:

```bash
# 16K context instead of 32K
llama-server ... -c 16384
```

---

## Known Limitations / TODO

- **WIP**: Kilocode integration via proxy is still being debugged. The proxy
  correctly converts formats but end-to-end tool execution in kilocode has not
  been fully validated yet.
- The proxy does not support multiple concurrent tool calls in a single
  response (Qwen rarely emits these, but if it does, only the first is caught).
- No graph memory integration yet (placeholder in `qwen_tools.py`).
- The standalone client truncates long tool results to 200 chars before
  sending back to the model, which may lose context on large file reads.
