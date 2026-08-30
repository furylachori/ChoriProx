#!/usr/bin/env python3
"""Template C — sub-agent canary (inherit / strip).

Use `c.sign_canary()` (returns the wrapped `[[__SUBAGENT__:...]]` marker, NOT
`proxy.canary.sign_canary` which returns a bare `sig:ts`). `send_as_subagent`
embeds the canary at position-0 internally.

Do NOT use `capture_canary()` to verify outbound — canaries live in
`tool_calls[].function.arguments.prompt`, not `messages[].content`.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/c_subagent_canary.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

canned = [{"choices":[{"message":{"role":"assistant","content":"sub-agent done"},"finish_reason":"tool_calls"}]}]

import time

with debug_session(canned_responses=canned) as c:
    c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "c-test"
    canary = c.sign_canary("ticket", "")                 # wrapped marker, ready to embed
    _emit(1, "canary generated", isinstance(canary,str) and canary.startswith("[[__SUBAGENT__"),
          f"canary={canary!r}")

    # Pre-compute the sub-agent's session_id (proxy derives it from
    # canary + subagent_prompt + client_hint). Pre-seed ticket mode under that
    # hash so the proxy's get_or_load() lookup hits.
    sub_prompt = "list pending tasks"                  # ADJUST: the sub-agent's task
    sub_sid = c.session_id_for(
        [{"role": "user", "content": f"{canary}\n{sub_prompt}"}],
        client_hint="c-test",
    )
    c.activate_ticket(sub_sid)
    pre_seed_time = time.time()                         # capture for H-1 inheritance check

    r2, returned_sid = c.send_as_subagent(
        canary=canary,
        subagent_prompt=sub_prompt,
        subagent_messages=[],
        client_hint="c-test",
    )
    _emit(2, "sub-agent 200", r2.status_code == 200, f"status={r2.status_code}")

    sub_mode = c.inspect_agent(returned_sid)
    # H-1: assert activated_at > pre_seed_time to prove the proxy's _inherit_skill
    # actually re-activated the mode (not just that the pre-seed survived).
    inherited = (
        sub_mode is not None
        and sub_mode["name"] == "ticket"
        and sub_mode["activated_at"] > pre_seed_time
    )
    _emit(3, "sub-agent inherited ticket (activated_at > pre-seed)",
          inherited,
          f"mode={sub_mode} pre_seed_time={pre_seed_time}")

_finish()