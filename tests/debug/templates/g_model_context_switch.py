#!/usr/bin/env python3
"""Template G — model context-size transition (1M -> 200K).

Pins the behavior when a conversation that fits one model's window is
replayed against a smaller-window model. Mounts BOTH glm-5.2 (1M ctx) and
mini (200K ctx) and replays the same payload against each.

Two scenarios:

  Scenario A — multi-message history (~250K tokens, fits 1M, exceeds 200K):
    * glm-5.2 (1M):   SUCCEEDS (200); content retained within the 1M window.
    * mini   (200K):  SUCCEEDS (200) via aggressive trim — does NOT 413.
      Per enforce_context_limits (src/proxy/context.py:60-72): when the
      estimate exceeds the raw limit AND there is more than one
      non-system message, it logs a warning and defers to select_context()
      rather than raising. The trimmed outbound is observably smaller than
      the input.

  Scenario B — single oversized message (~250K tokens):
    * glm-5.2 (1M):   SUCCEEDS (200).
    * mini   (200K):  SUCCEEDS (200) — the message is TRIMMED, NOT 413.
      PINNED ACTUAL BEHAVIOR (differs from the naive expectation of a 413):
      enforce_context_limits only raises when estimated_tokens > raw_limit
      AND len(non_system) <= 1. But calculate_tokens_from_request_body
      CAPS each message at max_message_size (= 10% of the safe budget /
      tokens_per_char), so a single message's estimate tops out around
      ~15K tokens for mini — far below the 200K raw limit. The 413
      single-message path is therefore only reachable when
      enforce_context_limits is mocked to raise (see
      tests/integration/test_pipeline_errors.py::test_context_limit_error_returns_413).
      In the real un-mocked flow the oversized single message is trimmed
      by select_context() and returns 200.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/g_model_context_switch.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

# --- sizing -----------------------------------------------------------------
# Heuristic mirrors the proxy's tokens_per_char=0.25 (chars/4 ~= tokens).
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 250_000                       # fits 1M, exceeds 200K
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # 1_000_000

FILLER_UNIT = "lorem ipsum "                  # 12 chars


def _outbound_chars(c):
    """Sum of content lengths in the most recent outbound (captured) body."""
    body = c.captured_requests[-1]["body"]
    return sum(len(m.get("content", "") or "") for m in body.get("messages", []))


# --- Scenario A: multi-message history --------------------------------------
N_TURNS = 50
_chars_per_msg = TARGET_CHARS // N_TURNS      # 20_000
_filler_reps = max(1, _chars_per_msg // len(FILLER_UNIT))


def _multi_messages():
    msgs = []
    for i in range(N_TURNS):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": (FILLER_UNIT * _filler_reps) + f" [{i}]"})
    return msgs


# --- canned responses (mock backend pops one per POST) ----------------------
_OK = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
canned = [_OK, _OK, _OK, _OK, _OK, _OK]

with debug_session(canned_responses=canned, models=["glm-5.2", "mini"]) as c:
    multi = _multi_messages()
    input_chars_multi = sum(len(m["content"]) for m in multi)

    # ===================== Scenario A: multi-message =======================
    r_glm_a = c.chat(messages=multi, model="glm-5.2")
    _emit(
        1, "A: glm-5.2 (1M) status==200",
        r_glm_a.status_code == 200,
        f"got {r_glm_a.status_code} (1M ctx absorbs ~{TARGET_TOKENS} tokens)",
    )

    r_mini_a = c.chat(messages=multi, model="mini")
    _emit(
        2, "A: mini (200K) status==200 (trims, not 413)",
        r_mini_a.status_code == 200,
        f"got {r_mini_a.status_code} (200K ctx, multi-msg overflow -> trim via select_context)",
    )
    _emit(
        3, "A: mini did NOT 413",
        r_mini_a.status_code != 413,
        f"status={r_mini_a.status_code} (enforce_context_limits defers multi-msg overflow)",
    )
    out_chars_mini_a = _outbound_chars(c)
    _emit(
        4, "A: mini outbound trimmed (< input)",
        out_chars_mini_a < input_chars_multi,
        f"out={out_chars_mini_a} input={input_chars_multi}",
    )

    # ===================== Scenario B: single message ======================
    huge = FILLER_UNIT * (TARGET_CHARS // len(FILLER_UNIT))
    single = [{"role": "user", "content": huge}]
    input_chars_single = len(huge)

    r_glm_b = c.chat(messages=single, model="glm-5.2")
    _emit(
        5, "B: glm-5.2 (1M) status==200",
        r_glm_b.status_code == 200,
        f"got {r_glm_b.status_code} (1M ctx, single ~{TARGET_TOKENS}-tok msg fits)",
    )

    r_mini_b = c.chat(messages=single, model="mini")
    # PINNED actual behavior: a single oversized message does NOT 413 in the
    # real flow. See module docstring for the estimation-cap rationale.
    _emit(
        6, "B: mini (200K) status==200 (NOT 413)",
        r_mini_b.status_code == 200,
        f"got {r_mini_b.status_code} (200K ctx, single oversized msg completes)",
    )
    _emit(
        7, "B: mini did NOT 413",
        r_mini_b.status_code != 413,
        f"status={r_mini_b.status_code}; 413 only via mocked enforce_context_limits",
    )
    # The request completed end-to-end (well-formed completion, not an error
    # body): a real assistant reply is present. (We do NOT assert on outbound
    # size here: for a single user message the MiniMax provider-compat adapter
    # restructures the payload — merging the system prompt into a user message
    # and inserting a barrier — so outbound size is not a reliable trim signal
    # the way it is for the multi-message case in Scenario A.)
    body_b = r_mini_b.json() or {}
    msg_b = (body_b.get("choices", [{}])[0].get("message", {}) or {})
    _emit(
        8, "B: mini returned a well-formed completion",
        bool(msg_b.get("content")),
        f"finish_reason={body_b.get('choices',[{}])[0].get('finish_reason')!r}",
    )

_finish()
