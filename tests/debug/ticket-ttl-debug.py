#!/usr/bin/env python3
"""Worked example — exercise the sub-agent workflow with the devtools harness.

Demonstrates the sub-agent workflow (parent + sub-agent 200s), NOT TTL
expiry — it never calls time_travel(). For the canonical TTL pattern see
tests/debug/templates/e_ttl_expiry.py (Template E).

This is a scratch script, NOT a pytest test. Run with:

    PYTHONPATH=src PROXY_SESSION_SECRET=... python tests/debug/ticket-ttl-debug.py

Demonstrates the full debug workflow:
  1. Pin session_id via HTTP_X_SESSION_CLIENT header
  2. Pre-seed agent ticket mode + workspace using `session_id_for()`
     (which computes the proxy's content-derived hash)
  3. Send a parent message via c.chat()
  4. Capture the canary / upstream body the proxy sent
  5. Spawn a sub-agent turn via c.send_as_subagent()
  6. Verify both requests completed with status 200

NOTES on the proxy model:
  - session_id is content-derived: sha256(first 10 user msgs + ":" + client_hint)[:16]
  - To pre-seed agent state the proxy can see, compute the hash via
    `c.session_id_for(messages, client_hint=...)` and register under that key.
  - Ticket mode is per-turn: the proxy clears it after finish_reason=stop.
  - Canary injection only happens in `apply_post_process()` for `task` tool calls,
    so a plain chat won't show the [[__SUBAGENT__:SKILL:...]] canary in the body.
"""
from __future__ import annotations

import os
import sys

from devtools.debug_session import debug_session


def main() -> int:
    canned = [{"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}]

    with debug_session(canned_responses=canned) as c:
        client_hint = "ttl-debug"
        c.client.environ_base["HTTP_X_SESSION_CLIENT"] = client_hint

        # 1. Compute the proxy's content-derived session_id for the parent
        parent_msgs = [{"role": "user", "content": "^^ticket plan a fix"}]
        parent_sid = c.session_id_for(parent_msgs, client_hint=client_hint)
        print(f"[1] parent session_id (proxy hash): {parent_sid}")

        # 2. Pre-seed ticket mode + workspace under the hash
        c.activate_ticket(parent_sid, workspace="choriprox")
        mode = c.inspect_agent(parent_sid)
        print(f"[2] agent mode pre-chat: {mode['name'] if mode else None}")

        # 3. Send the parent message
        r1 = c.chat(messages=parent_msgs, client_hint=client_hint)
        print(f"[3] parent response status: {r1.status_code}")
        if r1.status_code != 200:
            print(f"FAIL: parent response was {r1.content[:200]}")
            return 1

        # 4. Inspect what the proxy sent upstream
        if not c.captured_requests:
            print("FAIL: no outbound requests captured")
            return 1
        upstream = c.captured_requests[-1]["body"]
        print(f"[4] upstream message count: {len(upstream['messages'])}")
        last_msg = upstream["messages"][-1]
        preview = last_msg.get("content", "")[:80]
        print(f"[4] upstream last message preview: {preview!r}")

        # 5. dump_captured() shows the full record
        captured = c.dump_captured()
        print(f"[5] dump_captured: {len(captured['inbound'])} inbound, "
              f"{len(captured['outbound'])} outbound")

        # 6. Send a second turn as a sub-agent (using send_as_subagent)
        # Compute session_id the same way send_as_subagent does (canary + prompt as first message)
        canary = "[[__SUBAGENT__:SKILL:ticket:demo]]"
        combined_first = f"{canary}\nlist pending tasks"
        sub_msgs_with_canary = [{"role": "user", "content": combined_first}]
        sub_sid = c.session_id_for(sub_msgs_with_canary, client_hint=client_hint)
        r2, returned_sid = c.send_as_subagent(
            canary=canary,
            subagent_prompt="list pending tasks",
            subagent_messages=[],
            client_hint=client_hint,
        )
        print(f"[6] sub-agent response status: {r2.status_code}")
        print(f"[6] sub-agent session_id: {returned_sid}")
        if returned_sid != sub_sid:
            print(f"FAIL: returned session_id {returned_sid} != computed {sub_sid}")
            return 1
        if r2.status_code != 200:
            print(f"FAIL: sub-agent response was {r2.content[:200]}")
            return 1

        print("PASS: workflow completes end-to-end (parent + sub-agent both 200)")
        return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
