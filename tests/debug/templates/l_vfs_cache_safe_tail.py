#!/usr/bin/env python3
"""Template L — main-agent VFS cache-safe tail placement (US-127).

Pins the canonical positioning of the main-agent VFS context block
end-to-end through debug_session + real VFS builder + real V2 context
pipeline + mock backend.

Authority: docs/specs/main-agent-vfs-cache-safe-tail.md §1 canonical
ordering and §13 budget contract; USER-STORIES.md US-127.

Canonical ordering (spec §1, §16):
    primary system
    Tier-2 pin, if present
    summaries
    ancient history
    completed retained conversation
    VFS context
    entire live user-led exchange

Scenarios:

  Scenario A — VFS placed immediately before live exchange:
    * cache one file in the active workspace
    * send [system, completed user, completed assistant, live user]
    * the captured outbound body has exactly one VFS block
    * the VFS block sits between completed history and the live user
    * no VFS content bleeds into the live user message
    * the live user message is preserved verbatim

  Scenario B — VFS dropped (not live exchange) when budget tight:
    * cache one file
    * send an oversized live user message that overflows safe budget
    * the proxy returns 200 (trim, not 413) per g_model_context_switch.py
    * the outbound body's live user content is trimmed with
      "... [truncated]" marker (§13.5 safety net)
    * VFS is dropped (§13.1 #7 / AC-15: drop VFS, do not split live-
      exchange TO MAKE ROOM FOR VFS)
    * system prompt survives; request reaches upstream backend

  Scenario C — byte-identical VFS across multiple requests:
    * cache one file
    * send the same body twice
    * the captured VFS block in both requests is byte-identical
      (spec §1 "stable VFS state MUST render byte-identically regardless
      of cache age within TTL or LRU access order")

  Scenario D — workspace switch replaces only the VFS block:
    * cache file_A in workspace_A, file_B in workspace_B
    * send the same body against workspace_A, then workspace_B
    * the VFS block content differs (file_A path/hash vs file_B)
    * the prefix before VFS (system + completed history) is unchanged

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/l_vfs_cache_safe_tail.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session
import proxy.state as state
from proxy.vfs import _cache_file


MAIN_VFS_MARKER = "[VFS CONTEXT — Cached file state]"


def _captured_outbound(c):
    """Return the messages list from the most recent captured upstream body."""
    body = c.captured_requests[-1]["body"]
    return body.get("messages", [])


def _vfs_indices(msgs):
    return [
        i for i, m in enumerate(msgs)
        if MAIN_VFS_MARKER in str(m.get("content") or "")
    ]


def _role_index(msgs, role, content_contains=None):
    """Find the index of the first message with role and optional content match."""
    for i, m in enumerate(msgs):
        if m.get("role") != role:
            continue
        if content_contains is None:
            return i
        if content_contains in str(m.get("content") or ""):
            return i
    return -1


def _activate(client_hint, workspace):
    """Activate workspace at the CLIENT level (the proxy looks up by client_hint).

    `c.activate_workspace` only sets session-level state — but the proxy
    derives session_id from the request body's user messages, so we can't
    know it ahead of time. Setting client-level workspace mapping via
    `client_hint` is the robust path (workspace_state.py:152-164).
    """
    state.workspace_store.set_active(
        "__l_vfs_preseed__", workspace, client_hint=client_hint,
    )


# --- Canned responses (one per chat call; we make 7 total) ------------------
_OK = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
canned = [_OK] * 8


with debug_session(
    canned_responses=canned,
    secret="l-vfs-cache-safe-tail-secret",
    vfs_enabled=True,
    models=["qwen3-coder"],   # 32K ctx — small enough to force trim on Scenario B
) as c:
    # =========================================================================
    # Scenario A — VFS placed immediately before live exchange
    # =========================================================================
    workspace_a = "l-vfs-ws-A"
    client_hint_a = "l-vfs-client-A"
    cache_path_a = "/l_vfs/scenario_a.py"
    cache_content_a = "def hello():\n    return 'vfs scenario A'\n"

    _activate(client_hint_a, workspace_a)
    _cache_file(cache_path_a, cache_content_a, workspace_a, "__l_vfs_preseed__")

    msgs_a = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Please review the auth module."},
        {"role": "assistant", "content": "I'll start by reading the file."},
        {"role": "user", "content": "Go ahead and read scenario_a.py"},
    ]
    r_a = c.chat(messages=msgs_a, model="qwen3-coder", client_hint=client_hint_a)
    _emit(
        1, "A: status==200 (VFS placement succeeds)",
        r_a.status_code == 200,
        f"got {r_a.status_code}: {(r_a.content or '')[:200]}",
    )

    out_a = _captured_outbound(c)
    vfs_idxs_a = _vfs_indices(out_a)
    _emit(
        2, "A: exactly one VFS block present",
        len(vfs_idxs_a) == 1,
        f"got {len(vfs_idxs_a)} VFS blocks at indices {vfs_idxs_a}; "
        f"outbound roles={[m.get('role') for m in out_a]}",
    )

    if vfs_idxs_a:
        vfs_idx_a = vfs_idxs_a[0]
        live_idx_a = _role_index(out_a, "user", "Go ahead and read scenario_a.py")
        _emit(
            3, "A: VFS precedes live user",
            0 <= vfs_idx_a < live_idx_a,
            f"vfs_idx={vfs_idx_a} live_idx={live_idx_a}",
        )
        # VFS content contains the cached path.
        vfs_msg_a = out_a[vfs_idx_a]
        _emit(
            4, "A: VFS contains cached path",
            cache_path_a in str(vfs_msg_a.get("content") or ""),
            f"vfs_msg tail={str(vfs_msg_a.get('content') or '')[-120:]!r}",
        )
        # VFS does NOT bleed live-user content.
        _emit(
            5, "A: VFS distinct from live user (no bleed)",
            "Go ahead and read scenario_a.py" not in str(vfs_msg_a.get("content") or ""),
            "VFS message must not contain the live user prompt",
        )
        # Completed history precedes VFS.
        completed_asst_idx_a = _role_index(out_a, "assistant", "I'll start by reading")
        _emit(
            6, "A: completed history precedes VFS",
            0 <= completed_asst_idx_a < vfs_idx_a,
            f"completed_asst_idx={completed_asst_idx_a} vfs_idx={vfs_idx_a}",
        )
        # System prompt is first.
        _emit(
            7, "A: primary system prompt at index 0",
            out_a[0].get("role") == "system",
            f"out_a[0]={out_a[0]!r}",
        )
        # Live user content is preserved verbatim.
        _emit(
            8, "A: live user content preserved verbatim",
            out_a[live_idx_a].get("content") == "Go ahead and read scenario_a.py",
            f"got {out_a[live_idx_a].get('content')!r}",
        )

    # =========================================================================
    # Scenario B — VFS dropped, live exchange trimmed (not 413)
    # =========================================================================
    workspace_b = "l-vfs-ws-B"
    client_hint_b = "l-vfs-client-B"
    cache_path_b = "/l_vfs/scenario_b.py"
    cache_content_b = "x = 'vfs scenario B'\n"

    _activate(client_hint_b, workspace_b)
    _cache_file(cache_path_b, cache_content_b, workspace_b, "__l_vfs_preseed__")

    # qwen3-coder: 32K context, safe ~30K tokens = ~120K chars at tpc=0.25.
    # 200K chars of live user = 50K tokens > 30K safe → trim-eligible overflow.
    huge_live = "ANALYZE_THIS_LARGE_PROMPT_BODY_" + ("Z" * 200_000)
    msgs_b = [
        {"role": "system", "content": "You are a code reviewer."},
        {"role": "user", "content": huge_live},
    ]
    r_b = c.chat(messages=msgs_b, model="qwen3-coder", client_hint=client_hint_b)
    _emit(
        9, "B: status==200 (trim, not 413) — g_model_context_switch.py contract",
        r_b.status_code == 200,
        f"got {r_b.status_code}: {(r_b.content or '')[:200]}",
    )
    _emit(
        10, "B: did NOT 413",
        r_b.status_code != 413,
        f"status={r_b.status_code}",
    )

    out_b = _captured_outbound(c)
    if r_b.status_code == 200:
        # The live user content must be trimmed with the marker (§13.5 safety net).
        user_msgs_b = [m for m in out_b if m.get("role") == "user"]
        _emit(
            11, "B: live user message survived (in some form)",
            len(user_msgs_b) >= 1,
            f"got {len(user_msgs_b)} user messages in outbound",
        )
        if user_msgs_b:
            last_user_b = user_msgs_b[-1]
            trimmed_content_b = str(last_user_b.get("content") or "")
            _emit(
                12, "B: live user trimmed with marker (§13.5)",
                "... [truncated]" in trimmed_content_b,
                f"tail={trimmed_content_b[-80:]!r}",
            )
            _emit(
                13, "B: trimmed content shorter than original",
                len(trimmed_content_b) < len(huge_live),
                f"trimmed={len(trimmed_content_b)} original={len(huge_live)}",
            )
        # VFS should be dropped (priority: drop VFS, not live-exchange, per §13.1 #7).
        vfs_idxs_b = _vfs_indices(out_b)
        _emit(
            14, "B: VFS dropped when live exchange needs the budget",
            len(vfs_idxs_b) == 0,
            f"got {len(vfs_idxs_b)} VFS blocks (expected 0 — VFS is discretionary)",
        )
        # System prompt survives.
        _emit(
            15, "B: system prompt survived",
            out_b and out_b[0].get("role") == "system",
            f"out_b[0]={out_b[0] if out_b else None!r}",
        )

    # =========================================================================
    # Scenario C — byte-identical VFS across multiple requests
    # =========================================================================
    workspace_c = "l-vfs-ws-C"
    client_hint_c = "l-vfs-client-C"
    cache_path_c = "/l_vfs/scenario_c.py"
    cache_content_c = "# scenario C content\nCACHE_STABLE = True\n"

    _activate(client_hint_c, workspace_c)
    _cache_file(cache_path_c, cache_content_c, workspace_c, "__l_vfs_preseed__")

    msgs_c = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Please check scenario_c.py"},
    ]
    r_c1 = c.chat(messages=msgs_c, model="qwen3-coder", client_hint=client_hint_c)
    out_c1 = _captured_outbound(c)
    vfs_c1_idxs = _vfs_indices(out_c1)

    r_c2 = c.chat(messages=msgs_c, model="qwen3-coder", client_hint=client_hint_c)
    out_c2 = _captured_outbound(c)
    vfs_c2_idxs = _vfs_indices(out_c2)

    _emit(
        16, "C: both requests had exactly one VFS block",
        len(vfs_c1_idxs) == 1 and len(vfs_c2_idxs) == 1,
        f"req1={len(vfs_c1_idxs)} req2={len(vfs_c2_idxs)}",
    )

    if vfs_c1_idxs and vfs_c2_idxs:
        vfs_c1_content = str(out_c1[vfs_c1_idxs[0]].get("content") or "")
        vfs_c2_content = str(out_c2[vfs_c2_idxs[0]].get("content") or "")
        _emit(
            17, "C: byte-identical VFS content across requests",
            vfs_c1_content == vfs_c2_content,
            f"len1={len(vfs_c1_content)} len2={len(vfs_c2_content)} "
            f"equal={vfs_c1_content == vfs_c2_content}",
        )

    # =========================================================================
    # Scenario D — workspace switch replaces only the VFS block
    # =========================================================================
    workspace_d1 = "l-vfs-ws-D1"
    workspace_d2 = "l-vfs-ws-D2"
    client_hint_d1 = "l-vfs-client-D1"
    client_hint_d2 = "l-vfs-client-D2"
    cache_path_d1 = "/l_vfs/scenario_d1.py"
    cache_path_d2 = "/l_vfs/scenario_d2.py"
    cache_content_d1 = "D1 = 'workspace one'\n"
    cache_content_d2 = "D2 = 'workspace two'\n"

    _activate(client_hint_d1, workspace_d1)
    _cache_file(cache_path_d1, cache_content_d1, workspace_d1, "__l_vfs_preseed__")
    _activate(client_hint_d2, workspace_d2)
    _cache_file(cache_path_d2, cache_content_d2, workspace_d2, "__l_vfs_preseed__")

    msgs_d = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Check the file in my current workspace."},
    ]
    r_d1 = c.chat(messages=msgs_d, model="qwen3-coder", client_hint=client_hint_d1)
    out_d1 = _captured_outbound(c)
    vfs_d1_idxs = _vfs_indices(out_d1)

    r_d2 = c.chat(messages=msgs_d, model="qwen3-coder", client_hint=client_hint_d2)
    out_d2 = _captured_outbound(c)
    vfs_d2_idxs = _vfs_indices(out_d2)

    _emit(
        18, "D: both workspace requests had exactly one VFS block",
        len(vfs_d1_idxs) == 1 and len(vfs_d2_idxs) == 1,
        f"ws1={len(vfs_d1_idxs)} ws2={len(vfs_d2_idxs)}",
    )

    if vfs_d1_idxs and vfs_d2_idxs:
        vfs_d1_content = str(out_d1[vfs_d1_idxs[0]].get("content") or "")
        vfs_d2_content = str(out_d2[vfs_d2_idxs[0]].get("content") or "")
        _emit(
            19, "D: VFS content differs across workspaces",
            vfs_d1_content != vfs_d2_content,
            f"d1_has_path1={cache_path_d1 in vfs_d1_content} "
            f"d2_has_path2={cache_path_d2 in vfs_d2_content}",
        )
        _emit(
            20, "D: ws1 VFS references ws1 cache path",
            cache_path_d1 in vfs_d1_content,
            f"vfs_d1 tail={vfs_d1_content[-80:]!r}",
        )
        _emit(
            21, "D: ws2 VFS references ws2 cache path",
            cache_path_d2 in vfs_d2_content,
            f"vfs_d2 tail={vfs_d2_content[-80:]!r}",
        )
        # Prefix before VFS (system prompt) is unchanged.
        prefix_d1 = out_d1[:vfs_d1_idxs[0]]
        prefix_d2 = out_d2[:vfs_d2_idxs[0]]
        _emit(
            22, "D: prefix before VFS unchanged across workspace switch",
            prefix_d1 == prefix_d2,
            f"prefix1_roles={[m.get('role') for m in prefix_d1]} "
            f"prefix2_roles={[m.get('role') for m in prefix_d2]}",
        )

_finish()
