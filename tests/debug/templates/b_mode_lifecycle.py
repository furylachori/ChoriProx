#!/usr/bin/env python3
"""Template B — mode lifecycle (set on activate, clear on stop).

Ticket/work/custom skills are PER-TURN — the proxy clears the mode after
`finish_reason:"stop"` post-processing. So `inspect_agent` returns the mode
*before* the chat and `None` *after* (for a stop response). Compare pre vs
post to detect a real leak.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/b_mode_lifecycle.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

canned = [{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}]

with debug_session(canned_responses=canned) as c:
    # H-2: message does NOT contain ^^ticket — pre-seed is the ONLY way the mode
    # gets set. This isolates the per-turn stop-clears-mode contract from the
    # dispatcher's trigger-based re-activation path.
    msgs = [{"role":"user","content":"plan foo"}]
    sid = c.session_id_for(msgs, client_hint="b-test")   # ADJUST: your messages + hint
    c.activate_ticket(sid, workspace="choriprox")        # ADJUST: workspace or omit

    before = c.inspect_agent(sid)
    _emit(1, "mode set before chat", before is not None and before["name"]=="ticket",
          f"before={before}")

    r = c.chat(messages=msgs, client_hint="b-test")
    _emit(2, "chat 200", r.status_code == 200, f"status={r.status_code} body={r.data[:200]!r}")

    after = c.inspect_agent(sid)
    # Per-turn contract: stop response CLEARS the mode. after is None = correct.
    _emit(3, "mode cleared after stop", after is None, f"after={after}")

_finish()