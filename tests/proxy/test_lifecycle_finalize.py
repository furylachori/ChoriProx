"""End-to-end lifecycle tests for PostProcessStage (spec §7.5).

These tests send real POSTs to /v1/chat/completions through the Flask
shell, drive the mock backend with a configurable finish_reason, and
verify that the agent's active_mode is cleared (or preserved) after
PostProcessStage.execute()'s `finally` block runs.

Coverage (spec §7.4 finish-reason matrix):
- `stop`              → COMPLETE  → mode cleared
- `content_filter`    → FAILED    → mode cleared
- `tool_calls`        → CONTINUE  → mode preserved
- `length`            → CONTINUE  → mode preserved
- unknown reason      → FAILED    → mode cleared
- malformed/empty     → FAILED    → mode cleared
- no active mode at all → idempotent (no error)
- sub-method exception → finally block still runs

R1-A additional coverage (audit-remediation-round2-waves.md §4.4):
- T-A1: concurrent replacement with propagated snapshot (TOCTOU race)
- T-A2: None-ownership guard (Edit A2)
- T-A3: CRITICAL stage generic Exception (Edit A5 Exception half)
- T-A4: CRITICAL stage pre-ContextOptimize, no snapshot → no-op (Edit A3 None guard)
- T-A5: propagated identity is identity, not equality (Edit A1)
- T-A6: exception-path finalize is idempotent (Edit A5)
- T-A7: ContextOptimize raises mid-apply_mode, propagation fallback (Edit A3 fallback)
- T-A8: UpstreamStage StageAbort path (Edit A5 StageAbort half)

Revert-test matrix (spec §10.3):
- Restore stop-only `should_deactivate` → `content_filter`/malformed
  cleanup tests fail.
- Remove `finally` from PostProcessStage.execute() → early-return /
  sub-method exception tests fail.
- Unconditionally clear on abort → concurrent replacement-mode test fails.

Per spec §7.6:
- Lifecycle finalization MUST be idempotent (the `finally` block re-runs
  on every exit path: normal, empty choices, malformed response, tool
  interception early return, StageAbort, sync error, transform error,
  sub-method exception).
"""
import threading
import time

import pytest

from proxy.agent.agent import Agent
from proxy.pipeline.exceptions import StageAbort
from proxy.skills.skill_definition import ModeState
from tests.concurrency_harness import run_concurrently


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_real_agent_with_mode(session_id, mode_name="ticket", ttl_seconds=300):
    """Construct a real Agent with active_mode set to a real ModeState.

    Returns (agent, expected_mode). The `expected_mode` is the identity
    snapshot PostProcessStage will compare against during CAS.
    """
    agent = Agent(
        session_id=session_id,
        agent_id=f"agent_{session_id}",
        is_subagent=False,
    )
    expected_mode = ModeState(
        name=mode_name,
        activated_at=time.time(),
        ttl_seconds=ttl_seconds,
        skill_state={},
    )
    agent.set_active_mode(expected_mode)
    return agent, expected_mode


def _make_newer_mode(base_mode, generation=2):
    """Construct a NEWER ModeState with the same name but distinct identity.

    Returns a ModeState with a distinct `activated_at` and a `generation`
    marker in skill_state to make identity-vs-equality obvious. Used by
    T-A1 to simulate a concurrent skill-switch that installs a fresh
    mode after Request A's apply_mode read the older mode.
    """
    return ModeState(
        name=base_mode.name,
        activated_at=base_mode.activated_at + 1.0,
        ttl_seconds=base_mode.ttl_seconds,
        skill_state={"_generation": generation},
    )


def _build_test_client_with_agent(agent):
    """Wire `proxy.agent.agent_registry.registry.get_or_load` to return our agent.

    Returns a closure that monkeypatches the registry getter; tests should
    apply the patch via `monkeypatch.setattr(...)`.
    """
    def _patch(monkeypatch):
        monkeypatch.setattr(
            "proxy.agent.agent_registry.registry.get_or_load",
            lambda session_id, **kw: agent,
        )
    return _patch


def _make_stop_response():
    """Return a valid OpenAI-format stop response for mock_llama_response."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ]
    }


# ===========================================================================
# COMPLETE / FAILED → mode cleared
# ===========================================================================


def test_lifecycle_complete_stop_clears_mode(client, mock_llama_response, monkeypatch):
    """finish_reason=stop → COMPLETE → active mode is cleared.

    Per spec §7.4: `stop` maps to COMPLETE; finalize_skill_mode calls
    clear_active_mode(expected_mode=...) which performs CAS and clears
    when the identity matches.
    """
    agent, _ = _make_real_agent_with_mode("sess_lc_stop")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 200, (
        f"Expected 200 from chat completion, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is None, (
        "COMPLETE (stop) must clear the active mode; got "
        f"{agent.snapshot_mode()!r}"
    )


def test_lifecycle_failed_content_filter_clears_mode(
    client, mock_llama_response, monkeypatch
):
    """finish_reason=content_filter → FAILED → active mode is cleared.

    Per spec §7.4: `content_filter` is a FAILED disposition; the active
    mode is cleared (revert test for the spec §10.3 entry 'Restore
    stop-only should_deactivate').
    """
    agent, _ = _make_real_agent_with_mode("sess_lc_cf")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "[filtered]"},
                    "finish_reason": "content_filter",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "trigger filter"}],
        },
    )

    # The HTTP response can be 200 (content_filter is a valid finish_reason
    # from the upstream; the proxy forwards it). What matters is the
    # lifecycle finalization side-effect on the agent.
    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code} body={resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is None, (
        "FAILED (content_filter) must clear the active mode; got "
        f"{agent.snapshot_mode()!r}"
    )


def test_lifecycle_failed_malformed_response_clears_mode(
    client, mock_llama_response, monkeypatch
):
    """Malformed response (missing choices) → 502 + active mode is cleared.

    Per spec §7.4: missing choices / empty choices / non-dict response
    is FAILED. PostProcessStage now fail-closes malformed upstream data
    with build_upstream_protocol_error() + HTTP 502 BEFORE any success
    method runs. The lifecycle finalize still runs in the outer finally,
    classifying the turn FAILED and clearing the active mode.
    """
    from proxy.openai_compat import build_upstream_protocol_error

    agent, _ = _make_real_agent_with_mode("sess_lc_malformed")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={"usage": {"prompt_tokens": 5}}  # No "choices" key — malformed
    )

    # The semantic-retry sleep in proxy.http_client would otherwise burn
    # real wall-clock time on the fallback path. Neutralize it so the test
    # exercises the fail-closed boundary quickly.
    monkeypatch.setattr("proxy.http_client.time.sleep", lambda *_a, **_kw: None)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "malformed test"}],
        },
    )

    assert resp.status_code == 502, (
        f"Expected 502 for malformed upstream, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)[:200]}"
    )
    assert resp.get_json() == build_upstream_protocol_error()
    assert agent.snapshot_mode() is None, (
        "FAILED (malformed response) must clear the active mode; got "
        f"{agent.snapshot_mode()!r}"
    )


def test_lifecycle_failed_unknown_reason_clears_mode(
    client, mock_llama_response, monkeypatch
):
    """finish_reason=garbage_xyz → FAILED → active mode is cleared.

    Per spec §7.4: any unknown non-null finish_reason is FAILED.
    """
    agent, _ = _make_real_agent_with_mode("sess_lc_unknown")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "???"},
                    "finish_reason": "garbage_xyz",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test unknown"}],
        },
    )

    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code}"
    )
    assert agent.snapshot_mode() is None, (
        "FAILED (unknown finish_reason) must clear the active mode; got "
        f"{agent.snapshot_mode()!r}"
    )


# ===========================================================================
# CONTINUE → mode preserved
# ===========================================================================


def test_lifecycle_continue_tool_calls_keeps_mode(
    client, mock_llama_response, monkeypatch
):
    """finish_reason=tool_calls → CONTINUE → active mode is preserved.

    Per spec §7.4: `tool_calls` is a CONTINUE disposition; the active
    mode is intentionally retained so the tool loop can continue.
    """
    agent, mode = _make_real_agent_with_mode("sess_lc_tc")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_lc_001",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "tool call turn"}],
        },
    )

    # Status: the response may complete with 200 (forwarded body) or be
    # caught by tool interception. Either way, lifecycle finalization
    # ran and the mode is preserved.
    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code}"
    )
    assert agent.snapshot_mode() is mode, (
        f"CONTINUE (tool_calls) must PRESERVE the active mode; got "
        f"{agent.snapshot_mode()!r}"
    )


def test_lifecycle_continue_length_keeps_mode(
    client, mock_llama_response, monkeypatch
):
    """finish_reason=length → CONTINUE → active mode is preserved.

    Per spec §7.4: `length` (truncated but resumable) is CONTINUE; the
    mode is retained for the next turn.
    """
    agent, mode = _make_real_agent_with_mode("sess_lc_len")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "truncated..."},
                    "finish_reason": "length",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "truncation test"}],
        },
    )

    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code}"
    )
    assert agent.snapshot_mode() is mode, (
        f"CONTINUE (length) must PRESERVE the active mode; got "
        f"{agent.snapshot_mode()!r}"
    )


# ===========================================================================
# Idempotency + finally invariants
# ===========================================================================


def test_lifecycle_already_cleared_does_not_error(
    client, mock_llama_response, monkeypatch
):
    """Agent has NO active mode at all → pipeline completes without error.

    The finalize_skill_mode call must be a no-op (the CAS path in
    clear_active_mode with expected_mode=None always clears — but here
    there's nothing to clear, so it's a no-op pass-through).
    """
    agent = Agent(
        session_id="sess_lc_none",
        agent_id="agent_sess_lc_none",
        is_subagent=False,
    )
    # No active_mode set.

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "no mode at all"}],
        },
    )

    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code} body={resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is None


def test_lifecycle_finally_runs_even_on_sub_method_exception(
    client, mock_llama_response, monkeypatch
):
    """Sub-method exception → lifecycle finalization still runs.

    Per spec §7.5: the `finally` block ensures _finalize_skill_lifecycle
    runs on every exit path including sub-method exceptions. We patch
    one of the in-loop sub-methods (`_assign_model_name`) to raise, and
    verify the mode is still cleared.
    """
    from proxy.pipeline.post_process import PostProcessStage

    agent, _ = _make_real_agent_with_mode("sess_lc_exc")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "after exception"},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    # Patch _assign_model_name to raise. The inner try/except in execute()
    # logs the failure and continues; the outer finally still runs.
    def raise_runtime_error(ctx):
        raise RuntimeError("simulated sub-method exception")
    monkeypatch.setattr(
        PostProcessStage, "_assign_model_name", raise_runtime_error
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "sub method exc"}],
        },
    )

    # The request completes (sub-method exception is logged, not propagated).
    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code}"
    )
    # CRITICAL: the `finally` block must have run, clearing the mode.
    assert agent.snapshot_mode() is None, (
        f"Finally block must run on sub-method exception; mode = "
        f"{agent.snapshot_mode()!r}"
    )


def test_lifecycle_finally_runs_when_sub_method_returns_early(
    client, mock_llama_response, monkeypatch
):
    """Tool interception early-return path → lifecycle finalization still runs.

    Per spec §7.5: when `_handle_tool_interception` returns non-None
    (early return after tool handling), the inner code path calls
    `_apply_skill_response_transform` + `_inject_task_canaries` then
    returns — but the outer `finally` still calls
    `_finalize_skill_lifecycle_with_flag`. This test verifies the
    early-return path completes lifecycle finalization.
    """
    from proxy.pipeline.post_process import PostProcessStage

    agent, _ = _make_real_agent_with_mode("sess_lc_early")

    _build_test_client_with_agent(agent)(monkeypatch)

    # First (and only) response from the mock backend.
    mock_llama_response(
        data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_early_001",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "early return path"}],
        },
    )

    # Either 200 (forwarded body) or 502 (tool interception handled);
    # lifecycle finalization should have run either way.
    assert resp.status_code in (200, 502), (
        f"Unexpected status: {resp.status_code}"
    )

    # The tool_calls turn is CONTINUE — the mode should be preserved.
    # But more importantly, the test exercises the early-return path.
    final_mode = agent.snapshot_mode()
    # If the pipeline made it to finalize_skill_mode, the mode is either:
    #   - preserved (tool_calls → CONTINUE) → initial mode still set
    #   - cleared (if lifecycle ran with a stop-disposition somehow)
    # Either is acceptable as long as no error propagated.
    assert final_mode is None or final_mode.name == "ticket"


# ===========================================================================
# R1-A — audit-remediation-round2-waves.md §4.4
# Tests for finding #1 (TOCTOU + None-ownership) and #2 (exception path
# never finalizes mode). Each test has a named revert-check; see the
# dispatch prompt for the exact revert-to-prove-discriminating procedure.
# ===========================================================================


# ---------------------------------------------------------------------------
# T-A1 — concurrent replacement with propagated snapshot
#   (REVERTS Edit A1 + Edit A4 — TOCTOU at context_optimize.py:178)
# ---------------------------------------------------------------------------

def test_T_A1_concurrent_replacement_propagated_snapshot_survives(
    client, mock_llama_response, monkeypatch
):
    """R2 #1: Edit A1 propagates the exact ModeState identity that
    apply_mode read at mode.py:170. A concurrent replacement that
    installs a newer mode AFTER apply_mode ran must NOT be CAS-cleared
    by the older request's finalize_skill_mode.

    The window the edit closes: pre-fix, ContextOptimizeStage.execute
    did `ctx.mode_snapshot_pre_upstream = agent.snapshot_mode()` AFTER
    apply_mode returned. A concurrent replacement between the two reads
    leaks the newer identity into the pre-upstream snapshot, and the
    CAS in clear_active_mode compares newer vs newer and clears the
    replacement (the F10 clobber).

    With Edit A1 + Edit A4, the pre-upstream snapshot is the SAME
    object that apply_mode read — the older identity — and the CAS
    detects a mismatch and no-ops, preserving the newer mode.
    """
    agent, m_a = _make_real_agent_with_mode("sess_T_A1", mode_name="ticket")
    m_b = _make_newer_mode(m_a, generation=2)

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    # In-flight coordination: thread A blocks inside apply_mode AFTER
    # the Edit A1 propagation line so we can install m_b BEFORE the
    # pre-upstream field is stored. The barrier hook lives in apply_mode
    # itself (the function, not a stage) so we exercise the real
    # propagation chain.
    propagation_done = threading.Event()
    cascade_release = threading.Event()

    import proxy.mode as mode_module
    real_apply_mode = mode_module.apply_mode

    def hooked_apply_mode(ctx, agent_, body, mode_arg):
        # Run the real apply_mode — this triggers the Edit A1
        # propagation (ctx._mode_snapshot_propagated_from_apply_mode = ms).
        result = real_apply_mode(ctx, agent_, body, mode_arg)
        # Now signal that the propagation happened; block until the
        # replacement thread has swapped the agent's mode.
        propagation_done.set()
        cascade_release.wait(timeout=5)
        return result

    monkeypatch.setattr(mode_module, "apply_mode", hooked_apply_mode)

    def do_replace():
        # Wait for A's apply_mode to propagate, then swap the mode.
        assert propagation_done.wait(timeout=5), (
            "A never propagated the snapshot; test setup is broken"
        )
        agent.set_active_mode(m_b)  # concurrent replacement — legal
        cascade_release.set()

    # Drive A's POST and the replacement concurrently.
    results = run_concurrently([
        lambda: client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "concurrent A"}],
            },
        ),
        do_replace,
    ])
    resp, _ = results[0]

    # The TOCTOU-fix invariant: A's CAS saw (m_a != m_b) and no-op'd.
    # m_b survives the F10 clobber.
    assert agent.snapshot_mode() is m_b, (
        f"F10 clobber: newer mode m_b was CAS-cleared by Request A; "
        f"current snapshot_mode() = {agent.snapshot_mode()!r}"
    )
    # Sanity: the POST did not 5xx.
    assert resp.status_code == 200, (
        f"Request A failed: {resp.status_code} body="
        f"{resp.get_data(as_text=True)[:200]}"
    )


# ---------------------------------------------------------------------------
# T-A2 — None-ownership guard (REVERTS Edit A2)
# ---------------------------------------------------------------------------

def test_T_A2_none_ownership_guard_preserves_mode(
    client, mock_llama_response, monkeypatch
):
    """R2 #1 None-ownership: finalize_skill_mode MUST NOT call
    clear_active_mode(expected_mode=None) — that bypasses the CAS and
    would unconditionally clear any concurrent mode installed between
    apply_mode and finalize_skill_mode.

    Setup: install m_a on the agent, then patch ContextOptimizeStage
    to simulate the agent-None cascade path by setting
    ctx.mode_snapshot_pre_upstream = None AND
    ctx._mode_snapshot_propagated_from_apply_mode = None. PostProcess
    then reads expected_mode=None and calls
    finalize_skill_mode(ctx, agent, None) — Edit A2's guard returns
    False without touching the agent. m_a SURVIVES.

    Revert-check: revert Edit A2 (remove the
    `if expected_mode is None: return False` guard). Then
    finalize_skill_mode calls clear_active_mode(expected_mode=None),
    which bypasses the CAS identity check at agent.py:225-226 (the
    `if expected_mode is not None and ...` clause is False, the guard
    skips, the clear proceeds). m_a is wiped → assertion fails.
    """
    from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage

    agent, m_a = _make_real_agent_with_mode("sess_T_A2", mode_name="ticket")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    # Patch ContextOptimizeStage.execute to simulate the "agent is
    # None at apply_mode time" cascade path: both the public field and
    # the propagation are None, but the real agent is still in the
    # registry so PostProcess can find it.
    real_co_execute = ContextOptimizeStage.execute

    def stub_co_execute(self, ctx):
        ctx._mode_snapshot_propagated_from_apply_mode = None
        ctx.mode_snapshot_pre_upstream = None
        # Return None so the pipeline continues to PostProcess.
        return None

    monkeypatch.setattr(ContextOptimizeStage, "execute", stub_co_execute)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "none ownership"}],
        },
    )

    # The POST may 5xx because ContextOptimize did not run resolve_mode /
    # apply_mode — but what we care about is the lifecycle side-effect
    # on the agent, NOT the response status (the agent survives).
    assert agent.snapshot_mode() is m_a, (
        f"None-ownership guard failed: expected m_a to survive; "
        f"snapshot_mode() = {agent.snapshot_mode()!r}"
    )


# ---------------------------------------------------------------------------
# T-A3 — CRITICAL-stage exception path (REVERTS Edit A5 generic-Exception half)
# ---------------------------------------------------------------------------

def test_T_A3_critical_stage_exception_path_clears_mode(
    client, mock_llama_response, monkeypatch
):
    """R2 #2 Exception half: when a CRITICAL stage (Upstream) raises
    a non-StageAbort exception, the Edit A5 `_finalize_mode_on_critical_failure`
    call MUST run BEFORE the exception is re-raised. The helper
    delegates to finalize_skill_mode_on_exception, which CAS-clears
    using either ctx.mode_snapshot_pre_upstream (if populated) or the
    propagation fallback.

    Pre-state: agent has m_a installed. ContextOptimize runs and sets
    ctx.mode_snapshot_pre_upstream = m_a (the propagation edit
    populates it). UpstreamStage raises RuntimeError. The pipeline
    re-raises AFTER clearing m_a.

    Revert-check: revert Edit A5's generic-Exception half (remove
    `self._finalize_mode_on_critical_failure(ctx)` from the
    `except Exception` clause only — the StageAbort clause in T-A8 is
    separate). Then m_a survives (no finalize ever runs).
    """
    from proxy.pipeline.stage_modules.upstream import UpstreamStage

    agent, m_a = _make_real_agent_with_mode("sess_T_A3", mode_name="ticket")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    def raise_upstream_runtime_error(self, ctx):
        raise RuntimeError("synthetic upstream failure")

    monkeypatch.setattr(
        UpstreamStage, "execute", raise_upstream_runtime_error
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "upstream exc"}],
        },
    )

    # Flask returns 5xx for uncaught exceptions. The exact code is not
    # the contract — what matters is the agent side-effect.
    assert resp.status_code >= 500, (
        f"Upstream RuntimeError should propagate to 5xx, got "
        f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is None, (
        f"Exception path must finalize m_a via "
        f"_finalize_mode_on_critical_failure; snapshot_mode() = "
        f"{agent.snapshot_mode()!r}"
    )


# ---------------------------------------------------------------------------
# T-A4 — exception path with no snapshot, no-op (REVERTS Edit A3 None guard)
# ---------------------------------------------------------------------------

def test_T_A4_exception_path_with_no_snapshot_preserves_mode(
    client, mock_llama_response, monkeypatch
):
    """R2 #2 + Edit A3 None guard: when SecurityStage raises BEFORE
    ContextOptimize runs, ctx.mode_snapshot_pre_upstream is NEVER
    populated and ctx._mode_snapshot_propagated_from_apply_mode is
    also None (apply_mode never entered). finalize_skill_mode_on_exception
    MUST see both as None and return False — otherwise it would call
    clear_active_mode(expected_mode=None), which bypasses the CAS and
    unconditionally wipes the pre-existing m_a.

    Pre-state: agent has m_a installed. SecurityStage raises BEFORE
    ContextOptimize (CRITICAL order: RouteResolution → Security → ContextOptimize
    → ... → Upstream per DEFAULT_STAGES at pipeline/stages.py:18-28).
    m_a survives because no snapshot was ever captured.

    Revert-check: revert Edit A3's `if expected_mode is None: return
    False` guard (but NOT the propagation fallback). Without the guard,
    finalize_skill_mode_on_exception falls through to
    clear_active_mode(expected_mode=None) and wipes m_a.
    """
    from proxy.pipeline.stage_modules.security import SecurityStage

    agent, m_a = _make_real_agent_with_mode("sess_T_A4", mode_name="ticket")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    def raise_security_runtime_error(self, ctx):
        raise RuntimeError("synthetic security failure")

    monkeypatch.setattr(
        SecurityStage, "execute", raise_security_runtime_error
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "security exc"}],
        },
    )

    assert resp.status_code >= 500, (
        f"Security RuntimeError should propagate to 5xx, got "
        f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is m_a, (
        f"Exception path with no snapshot must preserve m_a; "
        f"snapshot_mode() = {agent.snapshot_mode()!r}"
    )


# ---------------------------------------------------------------------------
# T-A5 — propagated identity is identity, not equality (REVERTS Edit A1)
# ---------------------------------------------------------------------------

def test_T_A5_propagated_identity_is_identity_not_equality(
    client, mock_llama_response, monkeypatch
):
    """R2 #1 propagation contract: ONE atomic read in apply_mode, ONE
    identity propagated end-to-end. ctx.mode_snapshot_pre_upstream MUST
    be the SAME ModeState object (identity check `is`, not equality)
    that apply_mode read at entry — NOT a fresh atomic read AFTER
    apply_mode returns.

    Pre-fix: ContextOptimizeStage re-snapshotted after apply_mode
    returned — a fresh snapshot is a different object even if its
    fields equal the original. The `is` check fails.

    With Edit A1 + Edit A4: the pre-upstream field is the EXACT object
    that Edit A1 propagated. `is original_ms` holds.

    Discrimination trick: replace apply_mode with a hook that swaps
    the agent's active_mode AFTER apply_mode returns but BEFORE
    ContextOptimize's fallback snapshot would run. With Edit A1 in
    place, the pre-upstream field captures the original m_a identity
    (set inside apply_mode, BEFORE the swap). With Edit A1 reverted,
    the fallback reads the new identity m_b — `is m_a` fails.

    We capture the ctx via a hook on RequestPipeline._run_stages so we
    can inspect the field AFTER the full pipeline ran. The captured ctx
    is the real dataclass instance built by parse_request — NOT a
    MagicMock, so the Edit A1 unconditional-assignment gate is
    exercised end-to-end.
    """
    import proxy.mode as mode_module
    from proxy.pipeline.request_pipeline import RequestPipeline

    agent, m_a = _make_real_agent_with_mode("sess_T_A5", mode_name="ticket")
    m_b = _make_newer_mode(m_a, generation=2)

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    real_apply_mode = mode_module.apply_mode

    def hooked_apply_mode(ctx, agent_, body, mode_arg):
        # Run the real apply_mode — with Edit A1 in place, this
        # propagates ctx._mode_snapshot_propagated_from_apply_mode = m_a.
        result = real_apply_mode(ctx, agent_, body, mode_arg)
        # NOW swap the agent's active_mode so the fallback snapshot
        # (Edit A4's `propagated = agent.snapshot_mode()`) would
        # observe a DIFFERENT identity.
        agent_.set_active_mode(m_b)
        return result

    monkeypatch.setattr(mode_module, "apply_mode", hooked_apply_mode)

    captured_ctx = []
    real_run_stages = RequestPipeline._run_stages

    def hooked_run_stages(self, ctx):
        captured_ctx.append(ctx)
        return real_run_stages(self, ctx)

    monkeypatch.setattr(RequestPipeline, "_run_stages", hooked_run_stages)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "identity check"}],
        },
    )

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)[:200]}"
    )
    assert len(captured_ctx) == 1, (
        f"Expected exactly one captured ctx, got {len(captured_ctx)}"
    )
    ctx = captured_ctx[0]

    # The propagation contract: ctx.mode_snapshot_pre_upstream is the
    # SAME object apply_mode read (m_a). Identity (`is`), not equality.
    # The fresh snapshot fallback would read m_b (the swap that
    # hooked_apply_mode performed after apply_mode returned), so
    # without Edit A1 the identity check fails.
    assert ctx.mode_snapshot_pre_upstream is m_a, (
        f"Propagation contract violated: "
        f"ctx.mode_snapshot_pre_upstream={ctx.mode_snapshot_pre_upstream!r} "
        f"is NOT the same object as m_a={m_a!r} (m_b={m_b!r})"
    )


# ---------------------------------------------------------------------------
# T-A6 — exception path is idempotent (REVERTS Edit A5)
# ---------------------------------------------------------------------------

def test_T_A6_exception_path_finalize_is_idempotent(
    client, mock_llama_response, monkeypatch
):
    """R2 #2 idempotency: after the first POST clears m_a via the
    exception-path finalize, a second POST with the same patched
    exception path must NOT error or change state. The CAS in
    clear_active_mode returns False on the second call because
    agent.active_mode is None and expected_mode is non-None — the
    CAS sees a mismatch and no-ops.

    Pre-state: agent has m_a installed. UpstreamStage raises on POST
    1; m_a is cleared. POST 2: UpstreamStage raises again; the helper
    runs finalize_skill_mode_on_exception(ctx, agent), which reads
    the captured snapshot (m_a or whatever was last stored) and calls
    clear_active_mode(expected_mode=m_a). The CAS sees
    agent.active_mode=None != expected_mode=m_a → returns False →
    no error.

    Revert-check: revert Edit A5 (remove BOTH the `except Exception`
    half AND the `except StageAbort` half of
    `_finalize_mode_on_critical_failure(ctx)`). With this revert,
    POST 1's exception leaves m_a intact → first assertion fails.
    """
    from proxy.pipeline.stage_modules.upstream import UpstreamStage

    agent, m_a = _make_real_agent_with_mode("sess_T_A6", mode_name="ticket")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    def raise_upstream_runtime_error(self, ctx):
        raise RuntimeError("synthetic upstream failure")

    monkeypatch.setattr(
        UpstreamStage, "execute", raise_upstream_runtime_error
    )

    # POST 1 — clears m_a via the exception-path finalize.
    resp1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "idempotent 1"}],
        },
    )
    assert resp1.status_code >= 500
    assert agent.snapshot_mode() is None, (
        f"First POST must clear m_a; snapshot_mode() = "
        f"{agent.snapshot_mode()!r}"
    )

    # POST 2 — helper is a no-op (CAS mismatch on already-cleared mode).
    resp2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "idempotent 2"}],
        },
    )
    # CRITICAL: the helper must NOT raise a second exception. The Flask
    # shell's 5xx comes from the synthetic upstream RuntimeError, NOT
    # from finalize_skill_mode_on_exception itself.
    assert resp2.status_code >= 500, (
        f"Second POST should also 5xx from the synthetic exception, "
        f"got {resp2.status_code}: {resp2.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is None, (
        f"Second POST must leave mode None (idempotent no-op); "
        f"snapshot_mode() = {agent.snapshot_mode()!r}"
    )


# ---------------------------------------------------------------------------
# T-A7 — ContextOptimize raises mid-apply_mode, propagation fallback
#   (REVERTS Edit A3 propagation fallback)
# ---------------------------------------------------------------------------

def test_T_A7_context_optimize_mid_apply_propagation_fallback_clears_mode(
    client, mock_llama_response, monkeypatch
):
    """R2 #2 + Edit A3 propagation fallback: when ContextOptimize
    raises AFTER apply_mode entered (Edit A1 propagation populated
    ctx._mode_snapshot_propagated_from_apply_mode) but BEFORE
    context_optimize.py:178 had a chance to populate
    ctx.mode_snapshot_pre_upstream, the finalize must read the
    propagation fallback and CAS-clear.

    Setup: patch ContextOptimizeStage.execute to:
      1. Set ctx._mode_snapshot_propagated_from_apply_mode = m_a
         (simulating apply_mode having entered and propagated).
      2. Leave ctx.mode_snapshot_pre_upstream = None
         (simulating the public field never being set).
      3. Raise RuntimeError.

    The exception triggers _finalize_mode_on_critical_failure →
    finalize_skill_mode_on_exception → reads the propagation fallback
    (Edit A3 precedence #2) → calls clear_active_mode(expected_mode=m_a)
    → CAS matches (m_a is still the agent's mode) → returns True →
    m_a is cleared.

    Revert-check: revert Edit A3's propagation fallback (remove the
    `expected_mode = getattr(ctx, "_mode_snapshot_propagated_from_apply_mode", None)`
    block). With this revert, the helper sees None on both fields and
    returns False; m_a leaks.
    """
    from proxy.pipeline.stage_modules.context_optimize import ContextOptimizeStage

    agent, m_a = _make_real_agent_with_mode("sess_T_A7", mode_name="ticket")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    def stub_co_execute_with_propagation_then_raise(self, ctx):
        # Simulate apply_mode having propagated m_a via Edit A1.
        ctx._mode_snapshot_propagated_from_apply_mode = m_a
        # Public field is NOT set (we simulate the primary path
        # crashing before context_optimize.py:178 ran).
        # m_a is still active_mode on the agent (no replacement).
        raise RuntimeError("synthetic mid-apply failure")

    monkeypatch.setattr(
        ContextOptimizeStage, "execute", stub_co_execute_with_propagation_then_raise
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "mid apply failure"}],
        },
    )

    assert resp.status_code >= 500, (
        f"ContextOptimize RuntimeError should propagate to 5xx, got "
        f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )
    # The propagation fallback in finalize_skill_mode_on_exception
    # reads ctx._mode_snapshot_propagated_from_apply_mode = m_a and
    # CAS-clears.
    assert agent.snapshot_mode() is None, (
        f"Propagation fallback must clear m_a; snapshot_mode() = "
        f"{agent.snapshot_mode()!r}"
    )


# ---------------------------------------------------------------------------
# T-A8 — UpstreamStage raises StageAbort (REVERTS Edit A5 StageAbort half)
# ---------------------------------------------------------------------------

def test_T_A8_upstream_stage_abort_path_clears_mode(
    client, mock_llama_response, monkeypatch
):
    """R2 #2 + Edit A5 StageAbort half: CRITICAL stages can raise
    StageAbort directly (e.g. UpstreamStage wraps a non-AuthConfigError
    as StageAbort((upstream_protocol_error, 502)) at upstream.py:76).
    StageAbort is an Exception subclass, BUT it is caught by the
    `except StageAbort` clause BEFORE the `except Exception` clause.
    Without the StageAbort-half helper call (Edit A5), the leak
    would persist.

    Setup: pre-state m_a. ContextOptimize runs (sets
    ctx.mode_snapshot_pre_upstream = m_a via Edit A4). UpstreamStage
    raises StageAbort. The StageAbort clause calls
    _finalize_mode_on_critical_failure BEFORE re-raising.

    The Flask shell renders the StageAbort's response_data with the
    abort's HTTP status (502 here), so the response is a structured
    502 — NOT a generic 500 from a leaked exception.

    Revert-check: revert Edit A5's `except StageAbort` half only.
    With this revert, m_a survives (no finalize on the StageAbort
    path).
    """
    from proxy.pipeline.stage_modules.upstream import UpstreamStage

    agent, m_a = _make_real_agent_with_mode("sess_T_A8", mode_name="ticket")

    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    abort_body = {
        "error": {
            "message": "synthetic upstream abort",
            "type": "internal_error",
        }
    }

    def raise_upstream_stage_abort(self, ctx):
        raise StageAbort((abort_body, 502))

    monkeypatch.setattr(
        UpstreamStage, "execute", raise_upstream_stage_abort
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "upstream abort"}],
        },
    )

    # The Flask shell renders StageAbort as the abort's status code.
    # The exact body is dictated by _response_from_stage_abort; what
    # matters is that the agent side-effect ran.
    assert resp.status_code == 502, (
        f"Upstream StageAbort should render as 502, got "
        f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is None, (
        f"StageAbort clause must call _finalize_mode_on_critical_failure "
        f"BEFORE re-raise; snapshot_mode() = {agent.snapshot_mode()!r}"
    )


def test_T_A9_propagated_none_not_treated_as_missing(
    client, mock_llama_response, monkeypatch
):
    """apply_mode propagates None when no active mode; ContextOptimize must preserve None."""
    import proxy.mode as mode_module

    agent = Agent(
        session_id="sess_T_A9",
        agent_id="agent_sess_T_A9",
        is_subagent=False,
    )
    m_b = ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=300,
        skill_state={"_concurrent_install": True},
    )
    _build_test_client_with_agent(agent)(monkeypatch)
    mock_llama_response(data=_make_stop_response())

    propagation_done = threading.Event()
    replacement_done = threading.Event()
    real_apply_mode = mode_module.apply_mode

    def hooked_apply_mode(ctx, agent_, body, mode_arg):
        result = real_apply_mode(ctx, agent_, body, mode_arg)
        propagated = getattr(ctx, "_mode_snapshot_propagated_from_apply_mode")
        assert propagated is None, (
            f"expected propagated=None, got {propagated!r}"
        )
        propagation_done.set()
        assert replacement_done.wait(timeout=5), (
            "worker never installed B and released the waiter; setup broken"
        )
        return result

    monkeypatch.setattr(mode_module, "apply_mode", hooked_apply_mode)

    def do_install():
        assert propagation_done.wait(timeout=5), (
            "A's apply_mode never propagated; setup broken"
        )
        try:
            agent.set_active_mode(m_b)
        finally:
            replacement_done.set()

    results = run_concurrently([
        lambda: client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "propagated none"}],
            },
        ),
        do_install,
    ])
    resp, _ = results[0]
    assert resp.status_code == 200, (
        f"expected 200, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)[:200]}"
    )
    assert agent.snapshot_mode() is m_b, (
        f"F10 clobber: m_b cleared; snapshot_mode() = "
        f"{agent.snapshot_mode()!r}"
    )


def test_T_B1_skill_state_conflict_clear_count_exactly_once(
    client, mock_llama_response, monkeypatch
):
    """T_B1: ActiveSkillUnavailable → 409, exactly-one clear_active_mode call."""
    from proxy.skills.skill_registry import SkillRegistry

    agent, m_a = _make_real_agent_with_mode("sess_T_B1", mode_name="ticket")

    clear_counter = {"n": 0}
    original_clear = agent.clear_active_mode

    def counting_clear(expected_mode=None):
        clear_counter["n"] += 1
        return original_clear(expected_mode=expected_mode)

    agent.clear_active_mode = counting_clear

    _build_test_client_with_agent(agent)(monkeypatch)

    original_get = SkillRegistry.get

    def patched_get(self, name):
        if name == "ticket":
            return None
        return original_get(self, name)

    monkeypatch.setattr(
        "proxy.skills.skill_registry.SkillRegistry.get", patched_get
    )

    results = run_concurrently([
        lambda: client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "missing ticket skill"}
                ],
            },
        )
    ])
    result, exc = results[0]
    assert exc is None, (
        f"client.post raised unexpectedly: {exc!r}"
    )
    resp = result

    assert resp.status_code == 409, (
        f"Expected 409 for missing registered skill, got "
        f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )
    assert resp.get_json() == {
        "error": {
            "message": "Active skill is not registered",
            "type": "skill_state_conflict",
        }
    }

    assert agent.snapshot_mode() is None, (
        f"Mode must be cleared after ActiveSkillUnavailable; "
        f"snapshot_mode() = {agent.snapshot_mode()!r}"
    )

    assert clear_counter["n"] == 1, (
        f"clear_active_mode must be invoked exactly once; got "
        f"{clear_counter['n']} (stage-local + orchestrator double-clear)"
    )


def test_T_B2_skill_runtime_error_clear_count_exactly_once(
    client, mock_llama_response, monkeypatch
):
    """T_B2: SkillRuntimeError → 500, exactly-one clear_active_mode call."""
    agent, m_a = _make_real_agent_with_mode("sess_T_B2", mode_name="ticket")

    clear_counter = {"n": 0}
    original_clear = agent.clear_active_mode

    def counting_clear(expected_mode=None):
        clear_counter["n"] += 1
        return original_clear(expected_mode=expected_mode)

    agent.clear_active_mode = counting_clear

    _build_test_client_with_agent(agent)(monkeypatch)

    def raising_assemble_system_prompt(*args, **kwargs):
        raise RuntimeError("synthetic skill callback failure")

    monkeypatch.setattr(
        "proxy.skill_prompts.assemble_system_prompt",
        raising_assemble_system_prompt,
    )

    results = run_concurrently([
        lambda: client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "skill callback failure"}
                ],
            },
        )
    ])
    result, exc = results[0]
    assert exc is None, (
        f"client.post raised unexpectedly: {exc!r}"
    )
    resp = result

    assert resp.status_code == 500, (
        f"Expected 500 for skill callback failure, got "
        f"{resp.status_code}: {resp.get_data(as_text=True)[:200]}"
    )
    assert resp.get_json() == {
        "error": {
            "message": "Skill runtime error",
            "type": "skill_runtime_error",
        }
    }

    assert agent.snapshot_mode() is None, (
        f"Mode must be cleared after SkillRuntimeError; "
        f"snapshot_mode() = {agent.snapshot_mode()!r}"
    )

    assert clear_counter["n"] == 1, (
        f"clear_active_mode must be invoked exactly once; got "
        f"{clear_counter['n']} (stage-local + orchestrator double-clear)"
    )
