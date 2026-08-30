#!/usr/bin/env python3
"""Template F — provider-compat body mutation (Kimi temperature clamp).

Primary observable: Kimi temperature clamping (`provider_compat.py:244-248`)
— works on ANY request body with `temperature` set, no assistant message
needed. Send temperature=1.5 via extra_body; KimiAdapter clamps to [0,1] →
outbound 1.0.

For DeepSeek/MiMo reasoning_content injection (requires HF2 to wire those
models), the conversation MUST include a prior assistant message — the
adapter only mutates role=="assistant" messages.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/f_provider_compat.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

canned = [{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}]

# Kimi is already wired in _make_proxy_models_config (no HF2 needed for this path).
with debug_session(canned_responses=canned, models=["kimi"]) as c:
    r = c.chat(
        messages=[{"role":"user","content":"hi"}],
        model="kimi",
        extra_body={"temperature": 1.5},        # HF6 — injected into the request body
    )
    _emit(1, "chat 200", r.status_code == 200, f"status={r.status_code}")
    outbound = c.captured_requests[-1]["body"]
    out_temp = outbound.get("temperature")
    # KimiAdapter clamps to max(0.0, min(1.0, 1.5)) == 1.0
    _emit(2, "temperature clamped 1.5->1.0", out_temp == 1.0, f"got temp={out_temp}")

_finish()