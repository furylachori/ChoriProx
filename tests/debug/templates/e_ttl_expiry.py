#!/usr/bin/env python3
"""Template E — TTL expiry (two-session design).

TTL cleanup has two structural constraints that defeat a naive test:
1. `cleanup_stale_ticket_modes()` (`agent_registry.py:429`) skips in-flight
   agents. The entire pipeline runs inside `request_scope(session_id)`
   (`request_pipeline.py:87`), so during CleanupStage the triggering request's
   own session is always in-flight → a session can never clean up its own
   expired mode via its own request.
2. `finish_reason:"stop"` clears the mode via `TicketSkill.should_deactivate()`
   regardless of TTL — so a single-session test with a stop response passes
   for the wrong reason.

Fix: two sessions. Session A activates + time_travel(400). Session B
(different content → different session_id) sends a plain chat — its
CleanupStage cleans up A's expired mode (A is NOT in-flight during B's
request).

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/e_ttl_expiry.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

# Two canned entries: one for the triggering request (B), one spare.
canned = [
    {"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]},
    {"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]},
]

with debug_session(canned_responses=canned) as c:
    # Session A: activate ticket mode
    msgs_a = [{"role":"user","content":"session A content"}]
    sid_a = c.session_id_for(msgs_a, client_hint="e-a")
    c.activate_ticket(sid_a)
    _emit(1, "session A mode set", c.inspect_agent(sid_a) is not None,
          f"before={c.inspect_agent(sid_a)}")

    with c.time_travel(400):                            # > 300s TTL
        # Session B: different content + hint → different session_id.
        # B's CleanupStage runs cleanup_stale_ticket_modes(); A is NOT in-flight
        # during B's request, so A's expired mode gets cleared.
        msgs_b = [{"role":"user","content":"session B trigger"}]
        r = c.chat(messages=msgs_b, client_hint="e-b")
        _emit(2, "triggering request 200", r.status_code == 200,
              f"status={r.status_code} body={r.data[:200]!r}")

    _emit(3, "session A mode cleared by TTL (not by stop-deactivation)",
          c.inspect_agent(sid_a) is None,
          f"after={c.inspect_agent(sid_a)}")

_finish()