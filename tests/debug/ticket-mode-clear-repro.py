#!/usr/bin/env python3
"""Bug-class reproducer: per-turn ticket mode clears after finish_reason=stop.

Canonical pattern: see tests/debug/templates/b_mode_lifecycle.py (Template B).

User reports: "I activated ^^ticket, sent my prompt, then sent a follow-up and
the LLM was suddenly running in default mode — ticket tools are gone."

This is EXPECTED behavior (ticket is per-turn by design — see
`src/skills/ticket_skill.py:867 should_deactivate`). The repro documents the
lifecycle so an engineer triaging the next ticket-mode complaint can hand the
reporter this 8-line script and say "this is what the proxy does."

Run with:
    PYTHONPATH=src python3 tests/debug/ticket-mode-clear-repro.py
"""
import os
import sys

from devtools.debug_session import debug_session


def main() -> int:
    canned = [
        {"choices": [{"message": {"role": "assistant", "content": "ack 1"},
                      "finish_reason": "stop"}]},
        {"choices": [{"message": {"role": "assistant", "content": "ack 2"},
                      "finish_reason": "stop"}]},
    ]
    with debug_session(canned_responses=canned) as c:
        hint = "per-turn-demo"
        msgs = [{"role": "user", "content": "^^ticket plan foo"}]
        sid = c.session_id_for(msgs, client_hint=hint)
        c.activate_ticket(sid, workspace="ws")
        # Pre-chat: mode should be active
        pre = c.inspect_agent(sid)
        r1 = c.chat(messages=msgs, client_hint=hint)
        post = c.inspect_agent(sid)
        # Follow-up turn: no re-activation, mode should still be absent
        r2 = c.chat(messages=[{"role": "user", "content": "follow-up"}],
                    client_hint=hint)
        after2 = c.inspect_agent(sid)
        report = (
            f"pre:  {pre['name'] if pre else None}\n"
            f"post turn 1 (finish=stop): {post}\n"
            f"post turn 2:              {after2}\n"
            f"r1.status={r1.status_code} r2.status={r2.status_code}\n"
        )
        # Bug indicator: if post is NOT None, ticket mode did NOT clear on stop
        bug_present = post is not None or after2 is not None
        report += f"\nBUG_REPRODUCED={'YES' if bug_present else 'NO (expected)'}"
        print(report)
        return 0 if not bug_present else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
