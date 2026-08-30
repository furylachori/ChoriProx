#!/usr/bin/env python3
"""Template D1 — memory-tool re-query (2 canned entries).

The proxy re-queries the backend ONLY for memory tools (`recall`, `search`,
`proxy_ticket`, `proxy_work`, …) — `tool_loop.py:185-196` short-circuits
otherwise. So a memory-tool round-trip consumes TWO canned entries: the
`tool_calls` response, then the post-requery `stop`.

External tools (`read`/`edit`/`bash`/`task`) consume exactly one — the proxy
forwards them to the client.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/d1_memory_tool_loop.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

# Two entries: memory tool_call, then the post-requery stop.
canned = [
    {"choices":[{"message":{"role":"assistant","content":None,
        "tool_calls":[{"id":"call_1","type":"function",
            "function":{"name":"recall","arguments":"{\"query\":\"tasks\"}"}}]},
        "finish_reason":"tool_calls"}]},
    {"choices":[{"message":{"role":"assistant","content":"found 2 tasks"},"finish_reason":"stop"}]},
]

with debug_session(canned_responses=canned) as c:
    r = c.chat(messages=[{"role":"user","content":"what do you recall?"}])
    body = r.json() or {}
    final = body.get("choices",[{}])[0].get("message",{}).get("content","")
    _emit(1, "re-query happened (2nd entry consumed)", "found 2 tasks" in final,
          f"final={final!r}")

_finish()