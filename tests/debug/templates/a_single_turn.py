#!/usr/bin/env python3
"""Template A — single-turn assertion.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/a_single_turn.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

canned = [{"choices":[{"message":{"role":"assistant","content":"hello back"},"finish_reason":"stop"}]}]

with debug_session(canned_responses=canned) as c:
    r = c.chat(messages=[{"role":"user","content":"hi"}])
    _emit(1, "status==200", r.status_code == 200, f"got {r.status_code}")
    body = r.json() or {}
    content = body.get("choices",[{}])[0].get("message",{}).get("content","") or ""
    ok = "hello back" in content
    _emit(2, "content matches", ok, f"content={content!r}")

_finish()