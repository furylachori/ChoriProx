# Adding a New Model to ChoriProx

Complete guide for registering a new model in the proxy system.

---

## Quick-Reference Checklist

```
[ ] 1. Add model entry to src/proxy_models.json
        - File is valid JSON (run `python3 -c "import json; json.load(open('src/proxy_models.json'))"`)
        - Key passes MODEL_NAME_PATTERN (no colons, max 64 chars)
        - backend is a valid http/https URL
        - tool_parser is one of: "openai", "xml", "anthropic", "json"
        - context_limit is set
        - api_key_env is set (if remote backend requires auth)
        - backend_model is set (if upstream model name differs from key)

[ ] 2. Export API key in start.sh
        - Add export SOME_API_KEY="..." if using a new api_key_env

[ ] 3. Add model to your client config:
        OpenCode → ~/.config/opencode/opencode.jsonc
        Kilo    → ~/.config/kilo/kilo.jsonc
        ⚠️ If you also have a direct provider block for the same upstream
           (e.g. `"agnes"`), you MUST also add the model to the
           `"chori-prox".models` block — otherwise opencode won't list it
           under the chori-prox picker and requests bypass the proxy.

[ ] 4. Restart the proxy
        - ./start.sh or kill + relaunch
        - Registry loads at module import time — no hot-reload

[ ] 5. Verify
        - Check proxy logs for "ModelRegistry" warnings
        - Send a test request: curl http://localhost:6666/v1/chat/completions
```

---

## Step 1: Register in `src/proxy_models.json`

This is the **source of truth** for the proxy. The `ModelRegistry` class
(`src/clients/model_registry.py`) loads this file at startup and creates
a `ModelConfig` dataclass for each entry.

### Entry structure

```jsonc
{
  "models": {
    "<model-key>": {
      "backend": "https://api.example.com/v1",   // REQUIRED — http or https URL
      "tool_parser": "openai",                    // REQUIRED — see valid values below
      "context_limit": 131072,                    // REQUIRED — max context window in tokens
      "name": "Human-Readable Display Name",      // optional — shown in logs
      "api_key_env": "EXAMPLE_API_KEY",           // optional — env var holding the API key
      "backend_model": "actual-model-name:v1",    // optional — upstream model name (CAN contain colons)
      "memory_injection_enabled": false,          // optional — defaults to false
      "ignore_subagents": true                    // optional — defaults to false
    }
  },
  "default": {
    "backend": "http://localhost:7777",
    "tool_parser": "xml",
    "context_limit": 32000
  }
}
```

### Model key rules

The key is validated against this regex at request time (`proxy/model_registry.py`):

```python
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_/][a-zA-Z0-9_\-./]{0,63}$")
```

**Allowed:** letters, digits, underscores (`_`), hyphens (`-`), dots (`.`), forward slashes (`/`).
**Max length:** 64 characters.
**Forbidden:** colons (`:`), spaces, `@`, or any other special characters.

> **Important:** The key you choose is what clients send in the `"model"` field of
> API requests. It is NOT the same as the upstream model name — use `backend_model`
> for that.

### Required fields

| Field | Type | Description |
|---|---|---|
| `backend` | string | Upstream API base URL. Must start with `http://` or `https://`. The proxy appends `/v1/chat/completions` automatically (or uses the URL as-is if it already ends with that path). |
| `tool_parser` | string | How to parse tool calls from the model's response. See valid values below. |
| `context_limit` | integer | Maximum context window in tokens. Used for token estimation and context management. |

### Optional fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | `""` | Human-readable name for logs. Max 128 chars, pattern `^[\w\s\-./]+$`. |
| `api_key_env` | string | `""` | Name of environment variable holding the API key. If set and the env var is missing, the proxy raises `AuthConfigError`. |
| `backend_model` | string | `""` | The actual model name to send upstream. **Can contain colons** (e.g. `"deepseek-v4-pro:cloud"`). If unset, the model key is sent as-is. |
| `memory_injection_enabled` | bool | `false` | Whether to inject memory context into prompts. |
| `ignore_subagents` | bool | `false` | Whether to skip subagent routing for this model. |

### Valid `tool_parser` values

| Value | Parser class | Use case |
|---|---|---|
| `"openai"` | `OpenAIToolParser` | Models that return tool calls in OpenAI's native JSON format |
| `"xml"` | `QwenXMLToolParser` | Models that return tool calls in XML blocks (Qwen, DeepSeek) |
| `"anthropic"` | `AnthropicToolParser` | Models returning tool calls as Anthropic-format JSON |
| `"json"` | `JSONToolParser` | Models returning tool calls as raw JSON objects |

If an unknown parser name is provided, it falls back to `"xml"` with a warning.

### Real examples from the codebase

**Local backend (no auth):**
```json
"qwen3-coder": {
  "backend": "http://localhost:7777",
  "tool_parser": "xml",
  "context_limit": 32000,
  "name": "ChoriProx Qwen3 Coder",
  "memory_injection_enabled": false
}
```

**Remote backend with API key and backend_model:**
```json
"deepseek-v4-flash": {
  "backend": "https://ollama.com/v1",
  "tool_parser": "xml",
  "context_limit": 1048576,
  "name": "Ollama Cloud DeepSeek V4 Flash",
  "api_key_env": "OLLAMA_API_KEY",
  "backend_model": "deepseek-v4-flash:cloud",
  "memory_injection_enabled": false
}
```

Note: The key `deepseek-v4-flash` has no colons, but `backend_model` is `"deepseek-v4-flash:cloud"` — this is how the upstream API receives it.

**Another example — OpenCode backend:**
```json
"DeepSeek4_Pro": {
  "backend": "https://opencode.ai/zen/go/v1",
  "tool_parser": "xml",
  "context_limit": 1048576,
  "name": "OpenCode DeepSeek V4 Pro",
  "api_key_env": "OPENCODE_API_KEY",
  "backend_model": "deepseek-v4-pro",
  "memory_injection_enabled": false
}
```

**With `ignore_subagents`:**
```json
"qwen3-coder-next": {
  "backend": "https://ollama.com/v1",
  "tool_parser": "openai",
  "context_limit": 131072,
  "name": "Ollama Cloud Qwen3 Coder Next",
  "api_key_env": "OLLAMA_API_KEY",
  "backend_model": "qwen3-coder-next",
  "ignore_subagents": true,
  "memory_injection_enabled": false
}
```

---

## Step 2: Export the API key in `start.sh`

If your model uses `api_key_env`, add an export to `start.sh`:

```bash
export EXAMPLE_API_KEY="${EXAMPLE_API_KEY:-your-default-key-here}"
```

The proxy reads the key at request time via `os.environ[model_cfg.api_key_env]`.
If the env var is missing or empty, requests to that model will fail with
`AuthConfigError` (HTTP 401).

---

## Step 3: Add the model to your client config

The proxy serves models via `/v1/models`, but each client needs its own
config entry to display the model in its UI picker. You MUST add the model
to the correct config file — skipping this step is the #1 cause of "the model
doesn't show up."

### OpenCode: `~/.config/opencode/opencode.jsonc`

OpenCode uses a global config at `~/.config/opencode/opencode.jsonc` (JSONC —
trailing commas and comments allowed). This is **not** the same as
`~/.opencode/opencode.json` (project-override, not the model catalog).

Add your model under the `chori-prox` provider:

```jsonc
{
  "provider": {
    "chori-prox": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ChoriProx (localhost:6666)",
      "options": {
        "baseURL": "http://localhost:6666/v1"
      },
      "models": {
        // ... existing models ...
        "your-model-key": {
          "name": "Human-Readable Name",
          "limit": {
            "context": 131072,
            "output": 131072
          }
        }
      }
    }
  }
}
```

> **Critical:** The key must match `proxy_models.json` exactly. OpenCode
> auto-discovers models from `/v1/models` and may hide or deduplicate entries
> with similar prefixes. If your model doesn't appear, check:
> 1. The key in opencode.jsonc matches proxy_models.json exactly
> 2. The key doesn't share a prefix with another model already in the config
>    (e.g., `deepseek-v4-flash` and `deepseek-v4-flash-direct` will collide)
> 3. You restarted OpenCode after editing the config
> 
> **Note — chori-prox provider block:** if you also defined a separate
> `<provider-name>` block for the same upstream (e.g. the `"agnes"`
> provider block above), opencode will route that provider directly and
> bypass the proxy. To make the model accessible **through chori-prox**
> (the proxy on `localhost:6666`), you MUST add it to the
> `"chori-prox".models` block too — otherwise opencode won't list it
> under the chori-prox picker and you won't get proxy-side
> session/cost/skills features. Do not stop at
> `src/proxy_models.json`; both edits are required.
> 
> See Common Mistake **3c** below for the full explanation.

### Kilo: `~/.config/kilo/kilo.jsonc`

Kilo needs its own entry so it can list the model in its UI and know the
context window. Add the model under the appropriate provider's `"models"` object.

### For the `choriprox` provider (routes through the proxy):

```jsonc
{
  "provider": {
    "choriprox": {
      "name": "ChoriProx Proxy",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "apiKey": "none",
        "baseURL": "http://localhost:6666/v1"
      },
      "models": {
        // ... existing models ...
        "your-model-key": {
          "name": "Human-Readable Name",
          "options": { "contextWindow": 131072 }
        }
      }
    }
  }
}
```

### For direct provider access (bypassing the proxy):

If you're adding a model to a direct provider like `ollama`, use the
**backend model name** as the key (since there's no proxy translation layer):

```jsonc
{
  "provider": {
    "ollama": {
      "models": {
        "your-backend-model-name": {
          "name": "Display Name",
          "options": { "contextWindow": 131072 }
        }
      }
    }
  }
}
```

### Key matching rules

- **OpenCode `chori-prox` provider / Kilo `choriprox` provider:** The key must match `proxy_models.json` exactly (e.g. `"deepseek-v4-pro-cloud"`).
- **Direct providers:** The key is the upstream model name (e.g. `"deepseek-v4-pro:cloud"`).
- `contextWindow` / `context` limit should match the `context_limit` in `proxy_models.json`.

---

## Step 4: Restart the proxy

The `ModelRegistry` is instantiated at module import time in `proxy/model_registry.py`:

```python
def _initialize_adapters():
    # ...
    _model_registry = ModelRegistry(
        config_path="proxy_models.json",
        base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    # ...

_initialize_adapters()  # runs on import
```

**There is no hot-reload.** After editing `proxy_models.json`, you must restart:

```bash
./start.sh
# or
pkill -f qwen_proxy.py && ./start.sh
```

---

## Step 5: Verify

### Check logs for warnings

On startup, the proxy logs any issues with model configs:

```
WARNING - Invalid scheme for model 'my-model': ftp
WARNING - Model 'my-model' references env var 'MY_KEY' which is not set
WARNING - Remote model 'my-model' has no backend_model set — model rewriting disabled
```

### Send a test request

```bash
curl -s http://localhost:6666/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-key",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

---

## Common Mistakes

### 1. Using colons in the model key

```
WRONG:  "deepseek-v4-pro:cloud": { ... }   # FAILS MODEL_NAME_PATTERN
RIGHT:  "deepseek-v4-pro-cloud": { ... }   # with backend_model: "deepseek-v4-pro:cloud"
```

The key is what clients send as the `"model"` field in API requests. It must pass
validation. The colon-containing upstream name goes in `backend_model`.

### 2. Forgetting to restart the proxy

The registry loads once at import. Edits to `proxy_models.json` have no effect
until the process restarts.

### 3. Mismatched keys between `proxy_models.json` and client config

If Kilo sends `"model": "deepseek-v4-pro-cloud"` but the proxy has the key as
`"deepseek_v4_pro_cloud"`, the request will fall through to the default config
or fail. Keys must match exactly.

### 3b. Prefix collision in OpenCode

OpenCode auto-discovers models from `/v1/models` and may deduplicate entries
with shared prefixes. For example, `deepseek-v4-flash` (ollama) and
`deepseek-v4-flash-direct` (direct) share the prefix `deepseek-v4-flash-` —
OpenCode will only show one. To avoid this, use distinct keys that don't share
a prefix (e.g., `deepseek-flash-direct` instead of `deepseek-v4-flash-direct`).
This applies regardless of which provider section the model is listed under.

### 3c. Missing `chori-prox` provider entry when a direct provider block also exists

If you added the model to opencode's direct provider block (e.g. the `"agnes"`
block in `~/.config/opencode/opencode.jsonc`) but forgot the `"chori-prox"`
entry, the model will appear in the opencode picker under the direct provider
but NOT under the chori-prox picker. This means the request goes straight to
the upstream API — **bypassing the proxy entirely** (no session tracking, no
cost accounting, no skills). The fix is simple: add the same model key to the
`"chori-prox".models` block in your opencode config. Both blocks are needed
when you want the model accessible through both the direct provider AND the
proxy.

### 4. Missing or wrong `api_key_env`

If you set `"api_key_env": "MY_KEY"` but don't export `MY_KEY`, every request
to that model will fail with `AuthConfigError` (401). Make sure:
1. The env var name in `proxy_models.json` matches what you export in `start.sh`
2. The env var is actually set when the proxy starts

### 5. Wrong `tool_parser` value

If the model returns tool calls in XML but you set `"tool_parser": "openai"`,
tool calls will silently fail to parse. Match the parser to the model's actual
output format:

- Qwen, DeepSeek → `"xml"`
- OpenAI-compatible models → `"openai"`
- Anthropic-style → `"anthropic"`

### 6. Not setting `backend_model` for remote backends

If `backend_model` is empty, the proxy sends the model key as the upstream
model name. This works for local backends but usually fails for remote APIs
that expect specific model identifiers. The proxy logs a warning:

```
WARNING - Remote model 'my-model' has no backend_model set — model rewriting disabled
```

### 7. Invalid JSON in `proxy_models.json` (proxy refuses to start)

`ModelRegistry` is **fail-fast on JSON errors** as of the loader hardening
fix. A missing brace, trailing comma, unclosed string, or any other JSON
syntax error causes the proxy to abort startup with:

```
ERROR - FATAL: proxy_models.json is malformed: Expecting ',' delimiter: line N column M (char K). Refusing to start.
```

**This is intentional.** The old behavior silently fell back to the `"default"`
config (which points at `http://localhost:7777` with no backend running),
so every request returned `Internal proxy error` with no obvious cause in
the proxy logs — only a `WARNING`. The proxy now refuses to boot instead.

If you see this error:
1. Run `python3 -c "import json; json.load(open('src/proxy_models.json'))"` to get the exact line/column
2. Fix the syntax (most often a missing `}` or stray comma after a model entry)
3. Restart the proxy

The regression test `tests/unit/clients/test_proxy_models_json.py` loads
the on-disk file and asserts it parses with all required models present, so
this class of breakage is caught in CI before it reaches the proxy.

---

## How Resolution Works

When a request arrives with `"model": "your-key"`:

1. **Exact match** — looks up `"your-key"` in the registry
2. **Wildcard match** — checks keys ending with `*` (e.g. `"qwen-*"`)
3. **Case-insensitive** — tries lowercase match
4. **Hyphen/underscore normalization** — `deepseek-v4` matches `deepseek_v4`
5. **backend_model match** — checks if the requested name matches any model's `backend_model` field
6. **Default fallback** — returns the `"default"` config if nothing matches

This means clients can sometimes send the `backend_model` name directly and
still get routed correctly, but you should not rely on this — always use the
registered key.

---

## File Summary

| File | Purpose | When to edit |
|---|---|---|
| `src/proxy_models.json` | Proxy model registry (source of truth) | Always — must be valid JSON or proxy refuses to start |
| `start.sh` | API key exports | When adding a new `api_key_env` |
| `~/.config/kilo/kilo.jsonc` | Kilo editor config | Always (for Kilo to see the model) |
| `~/.config/opencode/opencode.jsonc` | OpenCode global model catalog | Always (for OpenCode to see the model) |
| `src/proxy/model_registry.py` | Context config overrides (rarely needed) | Only for special context tuning |
| `tests/unit/clients/test_proxy_models_json.py` | Regression test that loads the on-disk `proxy_models.json` and asserts all required models are present | Edit when adding required models to the assertion list |
