#!/usr/bin/env python3
"""Bug-class reproducer: canary NOT at position 0 of the user message is
silently rejected by the canary parser (per `src/proxy/canary.py:443` —
`loc.start == 0` is required).

No direct template equivalent. Template C (tests/debug/templates/c_subagent_canary.py)
uses `send_as_subagent()` to test the sub-agent ingress + mode inheritance path.
This repro tests the **parser position-0 invariant** directly via `c.chat()` with
a hand-rolled canary string — a different code path.

Symptom: a sub-agent embedded the canary in its prompt but the upstream LLM
runs in DEFAULT mode (no proxy_ticket tool, plain system prompt). The proxy
wraps the offending user text in a `<TASK_PAYLOAD>` "untrusted data" guard
instead of activating the skill mode.

Signal checked: the OUTBOUND body the proxy sent upstream. Position-0 canary
yields a ticket-mode system prompt (mentions "PLANNER" + proxy_ticket). The
non-position-0 canary yields the `<TASK_PAYLOAD>` wrapper around the canary
text — proof the canary was NOT consumed.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET=demo python3 tests/debug/canary-position-repro.py
"""
import json
import os
import sys

from devtools.debug_session import debug_session


def main() -> int:
    from proxy.canary import sign_canary

    sig_ts = sign_canary("ticket")
    canary = f"[[__SUBAGENT__:SKILL:ticket:{sig_ts}]]"

    with debug_session() as c:
        msg_ok  = f"{canary}\nlist pending"
        msg_bad = f"preamble text\n{canary}\nlist pending"
        c.chat(messages=[{"role": "user", "content": msg_ok}],  client_hint="x")
        c.chat(messages=[{"role": "user", "content": msg_bad}], client_hint="x")

        out_ok  = c.captured_requests[0]["body"]
        out_bad = c.captured_requests[1]["body"]
        sys_ok  = out_ok["messages"][0]["content"]
        sys_bad = out_bad["messages"][0]["content"]
        tools_ok_names = {t["function"]["name"]
                          for t in out_ok.get("tools", [])}
        tools_bad_names = {t["function"]["name"]
                           for t in out_bad.get("tools", [])}
        report = {
            "canary@pos0 system has 'PLANNER' marker":     "PLANNER" in sys_ok,
            "canary@pos0 tools include proxy_ticket":      "proxy_ticket" in tools_ok_names,
            "canary@pos>0 system has 'TASK_PAYLOAD' wrap": "TASK_PAYLOAD" in sys_bad,
            "canary@pos>0 tools include proxy_ticket":     "proxy_ticket" in tools_bad_names,
        }
        for k, v in report.items():
            print(f"  {k}: {v}")
        bug = (report["canary@pos0 system has 'PLANNER' marker"]
               and report["canary@pos0 tools include proxy_ticket"]
               and report["canary@pos>0 system has 'TASK_PAYLOAD' wrap"]
               and not report["canary@pos>0 tools include proxy_ticket"])
        print(f"\nBUG_REPRODUCED={'YES' if bug else 'NO'}")
        return 0 if bug else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
