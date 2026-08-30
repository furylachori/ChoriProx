#!/usr/bin/env python3
"""Template D2 — tool-name/argument normalization healing (AGENTS.md #1 landmine).

The mock emits a degraded tool name/args; the proxy must heal them against
the client's `tools[]` schema. The observable is the response body's
`tool_calls`, not the loop.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/d2_normalization_healing.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

# Mock emits the WRONG name (read_file) and WRONG arg casing (file_path).
# Client schema declares: read / filePath. Proxy must heal both.
canned = [{
    "choices":[{"message":{"role":"assistant","content":None,
        "tool_calls":[{"id":"call_1","type":"function",
            "function":{"name":"read_file","arguments":"{\"file_path\":\"/x\"}"}}]},
        "finish_reason":"tool_calls"}]
}]

tools = [{"type":"function","function":{"name":"read",
    "parameters":{"type":"object","properties":{"filePath":{"type":"string"}}}}}]  # canonical

with debug_session(canned_responses=canned) as c:
    r = c.chat(messages=[{"role":"user","content":"read /x"}], tools=tools)
    body = r.json() or {}
    tc = (body.get("choices",[{}])[0].get("message",{}).get("tool_calls") or [{}])[0]
    healed_name = tc.get("function",{}).get("name")
    healed_args = tc.get("function",{}).get("arguments","{}")
    _emit(1, "name healed read_file->read", healed_name == "read", f"got {healed_name!r}")
    _emit(2, "arg healed file_path->filePath", "filePath" in healed_args, f"args={healed_args!r}")

_finish()