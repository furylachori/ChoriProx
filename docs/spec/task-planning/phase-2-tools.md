# Task Planning v2 — Phase 2: Read-Only Tool Suite

> **Depends on:** [Phase 1 — Context Assembly](./phase-1-context.md)
> **Output:** `src/planning/task_tools.py`
> **Next:** [Phase 3 — Planning Tool Loop](./phase-3-loop.md)

---

## 5. Phase 2: Read-Only Tool Suite (`task_tools.py`)

All planner tools are **read-only**. No writes, no shell execution, no edits. The write-capable tools (`write`, `edit`, `bash`) are excluded from `PLANNING_TOOLS`.

### 5.1 `PLANNING_TOOLS` List

```python
# src/planning/task_tools.py

PLANNING_TOOLS = [
    READ_FILE_TOOL,
    FILE_GLOB_TOOL,
    CODE_SEARCH_TOOL,
    WEB_FETCH_TOOL,
    RECALL_TOOL,
    SEARCH_TOOL,
    ADD_TASKS_TOOL,
]
```

`recall` and `search` are included even though their handlers in `src/memory/tools.py` are currently stubbed/disabled. When their Neo4j/read paths are activated, the planner gets them for free.

### 5.2 Tool Schemas (OpenAI Function-Calling Format)

#### `read_file`

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read the contents of a file within the working directory. Max 500 lines / 100KB. Returns truncated content if limits exceeded.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Relative or absolute file path. Must resolve inside the working directory."
        },
        "offset": {
          "type": "integer",
          "default": 1,
          "description": "1-based starting line."
        },
        "limit": {
          "type": "integer",
          "default": 200,
          "description": "Maximum lines to read (max 500)."
        }
      },
      "required": ["path"]
    }
  }
}
```

#### `file_glob`

```json
{
  "type": "function",
  "function": {
    "name": "file_glob",
    "description": "Find files matching a glob pattern within the working directory. Max 50 results.",
    "parameters": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "description": "Glob pattern, e.g. 'src/**/*.py' or '*.md'."
        }
      },
      "required": ["pattern"]
    }
  }
}
```

#### `code_search`

```json
{
  "type": "function",
  "function": {
    "name": "code_search",
    "description": "Search file contents using a Python regex pattern across the working directory. Max 50 matches. No subprocess / grep dependency.",
    "parameters": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "description": "Python regex pattern (re syntax)."
        },
        "include": {
          "type": "string",
          "default": "*",
          "description": "Glob filter for files to search, e.g. '*.py'."
        },
        "limit": {
          "type": "integer",
          "default": 50,
          "description": "Maximum match count (hard cap 50)."
        }
      },
      "required": ["pattern"]
    }
  }
}
```

#### `web_fetch`

```json
{
  "type": "function",
  "function": {
    "name": "web_fetch",
    "description": "Fetch content from a public URL. Max 8KB returned. Timeout 10s.",
    "parameters": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string",
          "description": "Fully-qualified URL (http or https)."
        },
        "timeout": {
          "type": "integer",
          "default": 10,
          "description": "Request timeout in seconds (max 30)."
        }
      },
      "required": ["url"]
    }
  }
}
```

#### `recall`

Reuses the existing schema from `src/memory/tools.py` (`MEMORY_TOOL_DEFINITIONS[0]`).

```json
{
  "type": "function",
  "function": {
    "name": "recall",
    "description": "Search memory across all stores. Provide queries (concepts to find) AND keywords (exact terms to match). Results hitting both paths are highest confidence. Returns unified feed sorted by recency.",
    "parameters": {
      "type": "object",
      "properties": {
        "queries": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Free-text search terms matched against memory content (substring CONTAINS matching)."
        },
        "keywords": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Exact terms to match against topic keys and entity names."
        },
        "session_id": {
          "type": "string",
          "description": "Current session ID (automatically provided by the proxy)."
        },
        "limit": {
          "type": "integer",
          "description": "Max results (default 10)."
        }
      },
      "required": ["queries", "keywords"]
    }
  }
}
```

#### `search`

Reuses the existing schema from `src/memory/tools.py` (`MEMORY_TOOL_DEFINITIONS[1]`).

```json
{
  "type": "function",
  "function": {
    "name": "search",
    "description": "Search chat history by keywords. Matches against raw message content, returns summaries for matched messages sorted by recency.",
    "parameters": {
      "type": "object",
      "properties": {
        "keywords": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Terms to match against raw message content."
        },
        "match_all": {
          "type": "boolean",
          "description": "If true, only return messages matching ALL keywords."
        },
        "session_id": {
          "type": "string",
          "description": "Current session ID (automatically provided by the proxy)."
        },
        "time_start": {
          "type": "string",
          "description": "ISO 8601 start time filter."
        },
        "time_end": {
          "type": "string",
          "description": "ISO 8601 end time filter."
        },
        "limit": {
          "type": "integer",
          "description": "Max results (default 10, max 50)."
        }
      },
      "required": ["keywords"]
    }
  }
}
```

#### `add_tasks`

```json
{
  "type": "function",
  "function": {
    "name": "add_tasks",
    "description": "Propose tasks to be created. Each task must have a name and priority. Use tags (A, B, C...) to group parallel tasks. Call this only when you have enough information to produce a concrete plan.",
    "parameters": {
      "type": "object",
      "properties": {
        "tasks": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string",
                "description": "Short, descriptive task name (max ~60 chars)."
              },
              "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"]
              },
              "summary": {
                "type": "string",
                "description": "1-2 sentence description."
              },
              "tag": {
                "type": "string",
                "description": "Alphabetic tag (A, B, C...) for parallelism groups."
              }
            },
            "required": ["name", "priority"]
          }
        }
      },
      "required": ["tasks"]
    }
  }
}
```

### 5.3 Security Constraints

All constraints are enforced inside `execute_readonly_tool()`.

| Constraint | Implementation |
|------------|----------------|
| **Working-directory containment** | Resolve candidate path with `Path.resolve()`. Verify `resolved_path.is_relative_to(working_dir_path)`. Also reject if `..` appears in path parts. Reject with `"error": "Path outside working directory"`. |
| **Directory denylist** | `.git/`, `.ssh/`, `.aws/`, `.npm/`, `.docker/`, `node_modules/` — checked against full resolved path. |
| **Read line / size cap** | `read_file` rejects files > 100KB before reading. Returns at most 500 lines. |
| **Glob result cap** | `file_glob` returns at most 50 paths via a generator that stops early. Excess is discarded silently (logged at INFO). |
| **Search match cap** | `code_search` returns at most 50 matches. Excess is discarded. Skips binary extensions. |
| **Secret pattern block** | Any `read_file` path matching `.env`, `.env.*`, `*.key`, `*secret*`, `.pem`, `.crt`, `.cer`, `.p12`, `.pfx`, `.jks`, `id_rsa`, `id_ed25519`, `id_ecdsa`, `credentials*.json`, `token*`, `private*` (case-insensitive), or residing in `.git/`, `.ssh/`, `.aws/`, `.npm/`, `.docker/`, `node_modules/` is rejected with `"error": "Sensitive file access blocked"`. |
| **URL scheme allowlist** | `web_fetch` only allows `http://` and `https://`. No `file://`, `ftp://`, etc. |
| **Web fetch size cap** | 8KB per fetch. Content-Length checked before reading body. |
| **Tool result size cap** | 8,000 chars per result. |
| **No subprocess** | `code_search` uses `pathlib.Path.rglob()` + `re.search()` in pure Python. |

### 5.4 `execute_readonly_tool()` Implementation Sketch

```python
# src/planning/task_tools.py

import json
import re
import logging
from pathlib import Path
from urllib import request, error

logger = logging.getLogger("planning.task_tools")

SENSITIVE_PATTERNS = [
    re.compile(r"\.env($|\..*)", re.I),
    re.compile(r"\.key$", re.I),
    re.compile(r"secret", re.I),
    re.compile(r"\.pem$", re.I),
    re.compile(r"\.crt$", re.I),
    re.compile(r"\.cer$", re.I),
    re.compile(r"\.p12$", re.I),
    re.compile(r"\.pfx$", re.I),
    re.compile(r"\.jks$", re.I),
    re.compile(r"id_rsa", re.I),
    re.compile(r"id_ed25519", re.I),
    re.compile(r"id_ecdsa", re.I),
    re.compile(r"credentials.*\.json", re.I),
    re.compile(r"token.*", re.I),
    re.compile(r"private.*", re.I),
]
MAX_READ_LINES = 500
MAX_READ_BYTES = 100_000
MAX_GLOB_RESULTS = 50
MAX_SEARCH_MATCHES = 50
WEB_FETCH_MAX_BYTES = 8_000
WEB_FETCH_TIMEOUT = 10


DIR_DENYLIST = {".git/", ".ssh/", ".aws/", ".npm/", ".docker/", "node_modules/"}


def _is_sensitive(path: str) -> bool:
    resolved_str = str(Path(path).resolve())
    for denied_dir in DIR_DENYLIST:
        if denied_dir in resolved_str:
            return True
    name = Path(path).name
    return any(p.search(name) for p in SENSITIVE_PATTERNS)


def _resolve_in_workdir(path: str, working_dir: str) -> Path:
    if ".." in Path(path).parts:
        raise PermissionError(f"Path traversal denied: {path}")
    wd = Path(working_dir).resolve()
    candidate = (wd / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not candidate.is_relative_to(wd):
        raise PermissionError(f"Path {candidate} is outside working directory {wd}")
    return candidate


def execute_readonly_tool(name: str, arguments: dict, working_dir: str) -> dict:
    """Execute a read-only planning tool and return a JSON-safe result dict."""
    try:
        if name == "read_file":
            return _tool_read_file(arguments, working_dir)
        elif name == "file_glob":
            return _tool_file_glob(arguments, working_dir)
        elif name == "code_search":
            return _tool_code_search(arguments, working_dir)
        elif name == "web_fetch":
            return _tool_web_fetch(arguments)
        elif name == "recall":
            # NOTE: `recall` and `search` are currently stubbed in
            # `src/memory/tools.py`. When enabled, no changes are needed here.
            from memory.tools import execute_memory_tool
            return execute_memory_tool(name, arguments)
        elif name == "search":
            from memory.tools import execute_memory_tool
            return execute_memory_tool(name, arguments)
        else:
            return {"error": f"Forbidden tool in planning context: {name}"}
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.warning("Tool %s failed: %s", name, e, exc_info=True)
        return {"error": f"Tool execution failed: {type(e).__name__}: {str(e)[:200]}"}


def _tool_read_file(arguments: dict, working_dir: str) -> dict:
    path = arguments.get("path", "")
    if _is_sensitive(path):
        return {"error": "Sensitive file access blocked"}
    resolved = _resolve_in_workdir(path, working_dir)
    if not resolved.is_file():
        return {"error": f"File not found: {path}"}
    try:
        if resolved.stat().st_size > MAX_READ_BYTES:
            return {"error": f"File too large: {resolved.stat().st_size} bytes (max {MAX_READ_BYTES})"}
        content = resolved.read_text(errors="ignore")
    except Exception as e:
        return {"error": f"Read failed: {e}"}
    lines = content.splitlines()
    offset = max(0, arguments.get("offset", 1) - 1)
    limit = min(arguments.get("limit", 200), MAX_READ_LINES)
    selected = lines[offset:offset + limit]
    result = "\n".join(selected)
    return {
        "path": str(resolved.relative_to(Path(working_dir).resolve())),
        "offset": offset + 1,
        "lines_returned": len(selected),
        "total_lines": len(lines),
        "content": result,
    }


def _tool_file_glob(arguments: dict, working_dir: str) -> dict:
    pattern = arguments.get("pattern", "")
    wd = Path(working_dir).resolve()
    results = [str(p.relative_to(wd)) for p in _limited_rglob(wd, pattern, MAX_GLOB_RESULTS)]
    return {"pattern": pattern, "matches": results, "truncated": False}


def _limited_rglob(wd: Path, pattern: str, limit: int):
    """Generator that yields file paths matching pattern, stopping after limit."""
    count = 0
    for p in wd.rglob(pattern):
        if p.is_file():
            yield p
            count += 1
            if count >= limit:
                break


_BINARY_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".zip", ".tar", ".gz",
    ".so", ".dll", ".exe", ".bin", ".dat", ".db", ".sqlite",
}


def _is_binary_ext(filepath: Path) -> bool:
    return filepath.suffix.lower() in _BINARY_EXTENSIONS


def _tool_code_search(arguments: dict, working_dir: str) -> dict:
    pattern = arguments.get("pattern", "")
    include = arguments.get("include", "*")
    limit = min(arguments.get("limit", 50), MAX_SEARCH_MATCHES)
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}
    wd = Path(working_dir).resolve()
    matches = []
    for filepath in wd.rglob(include):
        if not filepath.is_file():
            continue
        if _is_sensitive(str(filepath)):
            continue
        if _is_binary_ext(filepath):
            continue
        try:
            text = filepath.read_text(errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append({
                        "file": str(filepath.relative_to(wd)),
                        "line": lineno,
                        "content": line[:200],
                    })
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        except Exception:
            continue
    return {"pattern": pattern, "matches": matches, "truncated": len(matches) >= limit}


def _tool_web_fetch(arguments: dict) -> dict:
    import ipaddress
    from urllib.parse import urlparse
    url = arguments.get("url", "")
    timeout = min(arguments.get("timeout", WEB_FETCH_TIMEOUT), 30)
    if not url.startswith(("http://", "https://")):
        return {"error": "Only http/https URLs are allowed"}

    def _is_internal(parsed) -> bool:
        hostname = parsed.hostname or ""
        try:
            addr = ipaddress.ip_address(hostname)
            return (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_multicast
                or addr.is_reserved
            )
        except ValueError:
            # hostname is not an IP — allow public DNS names
            return False

    parsed = urlparse(url)
    if _is_internal(parsed):
        return {"error": "URL resolves to an internal address"}

    try:
        req = request.Request(url, headers={"User-Agent": "ChoriProx-Planner/1.0"})
        with request.urlopen(req, timeout=timeout) as resp:
            # Validate redirect target if Location header present
            final_url = resp.geturl()
            final_parsed = urlparse(final_url)
            if _is_internal(final_parsed):
                return {"error": "Redirect resolved to an internal address"}

            cl = resp.headers.get("Content-Length")
            if cl and int(cl) > WEB_FETCH_MAX_BYTES:
                return {"error": f"Content-Length {cl} exceeds max {WEB_FETCH_MAX_BYTES}"}

            content = resp.read().decode("utf-8", errors="ignore")
            if len(content) > WEB_FETCH_MAX_BYTES:
                content = content[:WEB_FETCH_MAX_BYTES] + "\n...[truncated]"
            return {"url": url, "content": content}
    except error.URLError as e:
        return {"error": f"Fetch failed: {e.reason}"}
    except Exception as e:
        return {"error": f"Fetch failed: {e}"}
```

---

## 14. Appendix B: Tool Schema Reference

All schemas in one place for copy-paste into implementation.

```python
# src/planning/task_tools.py

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file within the working directory. Max 500 lines / 100KB. Returns truncated content if limits exceeded.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute file path. Must resolve inside the working directory."},
                "offset": {"type": "integer", "default": 1, "description": "1-based starting line."},
                "limit": {"type": "integer", "default": 200, "description": "Maximum lines to read (max 500)."},
            },
            "required": ["path"],
        },
    },
}

FILE_GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "file_glob",
        "description": "Find files matching a glob pattern within the working directory. Max 50 results.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. 'src/**/*.py' or '*.md'."},
            },
            "required": ["pattern"],
        },
    },
}

CODE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "code_search",
        "description": "Search file contents using a Python regex pattern across the working directory. Max 50 matches. No subprocess / grep dependency.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex pattern (re syntax)."},
                "include": {"type": "string", "default": "*", "description": "Glob filter for files to search, e.g. '*.py'."},
                "limit": {"type": "integer", "default": 50, "description": "Maximum match count (hard cap 50)."},
            },
            "required": ["pattern"],
        },
    },
}

WEB_FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch content from a public URL. Max 8KB returned. Timeout 10s.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Fully-qualified URL (http or https)."},
                "timeout": {"type": "integer", "default": 10, "description": "Request timeout in seconds (max 30)."},
            },
            "required": ["url"],
        },
    },
}

# Re-exported from memory.tools
from memory.tools import MEMORY_TOOL_DEFINITIONS
RECALL_TOOL = MEMORY_TOOL_DEFINITIONS[0]
SEARCH_TOOL = MEMORY_TOOL_DEFINITIONS[1]

ADD_TASKS_TOOL = {
    "type": "function",
    "function": {
        "name": "add_tasks",
        "description": "Propose tasks to be created. Each task must have a name and priority. Use tags (A, B, C...) to group parallel tasks. Call this only when you have enough information to produce a concrete plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Short, descriptive task name (max ~60 chars)."},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "summary": {"type": "string", "description": "1-2 sentence description."},
                            "tag": {"type": "string", "description": "Alphabetic tag (A, B, C...) for parallelism groups."},
                        },
                        "required": ["name", "priority"],
                    },
                },
            },
            "required": ["tasks"],
        },
    },
}

PLANNING_TOOLS = [
    READ_FILE_TOOL,
    FILE_GLOB_TOOL,
    CODE_SEARCH_TOOL,
    WEB_FETCH_TOOL,
    RECALL_TOOL,
    SEARCH_TOOL,
    ADD_TASKS_TOOL,
]
```

---

*End of Specification*
