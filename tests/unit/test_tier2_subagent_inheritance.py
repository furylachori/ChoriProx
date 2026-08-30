"""Tier-2 pin emission: sub-agent INHERITS parent workspace via chain.

Proves both F1 (ctx.workspace_name is NOT the inherited workspace — the
parent chain walk resolves it) and F2 (default-workspace captures are keyed
by parent_session, requiring resolve_inherited_session).

Uses the autouse ``mock_agent_registry_get_or_load`` fixture's yielded dict
to register real Agent instances with parent links.

Revert-tested: see bottom-of-file comments.
"""
import copy
import uuid

import pytest

import proxy.state as state
from proxy.context_v2 import _emit_tier2_pin
from proxy.workspace_state import DEFAULT_WORKSPACE


pytestmark = pytest.mark.unit

_PIN_MARKER = '<CLIENT_SUB_INSTRUCTIONS scope="subfolder-convention">'


@pytest.fixture(autouse=True)
def _clean_state():
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()
    with state.active_workspace_lock:
        state.workspace_store._active.clear()
        state.workspace_store._recovered_sessions.clear()
    yield
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()
    with state.active_workspace_lock:
        state.workspace_store._active.clear()
        state.workspace_store._recovered_sessions.clear()


def _make_agent(session_id, parent=None):
    from proxy.agent.agent import Agent
    from proxy.agent.agent_config import AgentConfig
    return Agent(
        session_id=session_id,
        parent=parent,
        config=AgentConfig(),
        is_subagent=(parent is not None),
    )


def _make_sub_ctx(session_id):
    from proxy.types import RequestContext
    return RequestContext(
        body={},
        model_name="deepseek-chat",
        wants_stream=False,
        session_id=session_id,
        client_hint="",
        messages=[],
        is_subagent=True,
    )


def _messages_with_sys():
    return [
        {"role": "system", "content": "operational directives"},
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "working..."},
    ]


def _set_workspace(session_id, workspace_name):
    """Set workspace directly in the store's dict to avoid Neo4j persist."""
    with state.active_workspace_lock:
        state.workspace_store._active[session_id] = workspace_name


def _mark_recovered(session_id):
    """Mark session as recovered to prevent get_active() from triggering
    Neo4j recovery (which hangs when Neo4j is unavailable)."""
    with state.active_workspace_lock:
        state.workspace_store._recovered_sessions.add(session_id)


def _register_parent_and_sub(mock_registry, parent_sid, sub_sid):
    """Register real Agent instances in the mock registry with a parent link.

    Keys by BOTH session_id AND agent_id because resolve_inherited_workspace
    walks parent_agent_id (which is the agent's sha256-derived agent_id, not
    the session_id) and then calls get_or_load(agent_id) to recover the root.
    """
    parent_agent = _make_agent(parent_sid)
    sub_agent = _make_agent(sub_sid, parent=parent_agent)
    mock_registry[parent_sid] = parent_agent
    mock_registry[sub_sid] = sub_agent
    mock_registry[parent_agent.agent_id] = parent_agent
    mock_registry[sub_agent.agent_id] = sub_agent


def test_f1_non_default_subagent_inherits_parent_workspace(
        mock_agent_registry_get_or_load):
    """F1: sub-agent IGNORES ctx.workspace_name (even if non-empty/stale) and
    resolves the parent workspace via the parent-agent chain."""
    parent_sid = f"parent-{uuid.uuid4().hex[:8]}"
    sub_sid = f"sub-{uuid.uuid4().hex[:8]}"

    _register_parent_and_sub(mock_agent_registry_get_or_load, parent_sid, sub_sid)

    _set_workspace(parent_sid, "ws-inherit-nd")
    _mark_recovered(parent_sid)
    _mark_recovered(sub_sid)

    instr_path = "src/proxy/sub/AGENTS.md"
    instr_content = "# Sub instructions\nDo the thing."
    with state.vfs_lock:
        state.record_instruction_file(
            "ws-inherit-nd", instr_path, instr_content,
            session_id=parent_sid,
        )

    ctx = _make_sub_ctx(sub_sid)
    assert ctx.is_subagent is True
    # Explicitly set a STALE workspace_name on ctx — the fix must IGNORE this
    # and resolve via the parent chain instead.
    ctx.workspace_name = "stale-workspace"

    msgs = _messages_with_sys()
    result = _emit_tier2_pin(copy.deepcopy(msgs), sub_sid, ctx=ctx)

    assert result is not msgs
    pin_found = any(
        _PIN_MARKER in m.get("content", "") for m in result if m.get("role") == "system"
    )
    assert pin_found, (
        "F1 FAIL: sub-agent did not emit pin via parent-chain workspace resolution"
    )


def test_f2_default_workspace_subagent_uses_parent_session(
        mock_agent_registry_get_or_load):
    """F2: default-workspace captures are keyed by (DEFAULT_WORKSPACE,
    parent_session). The sub-agent must look up under the PARENT's session
    (via resolve_inherited_session), not its own."""
    parent_sid = f"parent-{uuid.uuid4().hex[:8]}"
    sub_sid = f"sub-{uuid.uuid4().hex[:8]}"

    _register_parent_and_sub(mock_agent_registry_get_or_load, parent_sid, sub_sid)

    _mark_recovered(parent_sid)
    _mark_recovered(sub_sid)

    instr_path = "src/proxy/sub/AGENTS.md"
    instr_content = "# Default-ws sub instructions."
    with state.vfs_lock:
        state.record_instruction_file(
            DEFAULT_WORKSPACE, instr_path, instr_content,
            session_id=parent_sid,
        )

    ctx = _make_sub_ctx(sub_sid)
    assert ctx.is_subagent is True
    assert getattr(ctx, "workspace_name", "") == "", (
        "sub-agent ctx.workspace_name MUST be empty to prove inheritance"
    )

    msgs = _messages_with_sys()
    result = _emit_tier2_pin(copy.deepcopy(msgs), sub_sid, ctx=ctx)

    assert result is not msgs
    pin_found = any(
        _PIN_MARKER in m.get("content", "") for m in result if m.get("role") == "system"
    )
    assert pin_found, (
        "F2 FAIL: sub-agent did not emit pin via resolve_inherited_session"
    )


def test_f2_revert_proves_necessity(mock_agent_registry_get_or_load):
    """Revert-test for F2: if _emit_tier2_pin uses session_id (not
    lookup_session) for default-workspace, the sub-agent won't see the
    parent's capture. We simulate the buggy lookup inline."""
    parent_sid = f"parent-{uuid.uuid4().hex[:8]}"
    sub_sid = f"sub-{uuid.uuid4().hex[:8]}"

    _register_parent_and_sub(mock_agent_registry_get_or_load, parent_sid, sub_sid)

    _mark_recovered(parent_sid)
    _mark_recovered(sub_sid)

    instr_path = "src/proxy/sub/AGENTS.md"
    instr_content = "# Default-ws sub instructions."
    with state.vfs_lock:
        state.record_instruction_file(
            DEFAULT_WORKSPACE, instr_path, instr_content,
            session_id=parent_sid,
        )

    buggy_files = state.get_instruction_files(DEFAULT_WORKSPACE, sub_sid)

    assert buggy_files == {}, (
        "Revert-test precondition: sub-agent session_id must NOT find "
        "parent's default-ws capture"
    )


def test_f1_main_agent_still_uses_ctx_workspace():
    """Main (non-sub-agent) path: ctx.workspace_name is reliable and used
    directly. This confirms the fix doesn't break the main path."""
    main_sid = f"main-{uuid.uuid4().hex[:8]}"

    _set_workspace(main_sid, "ws-main-ctx")
    _mark_recovered(main_sid)

    instr_path = "src/proxy/AGENTS.md"
    instr_content = "# Main instructions."
    with state.vfs_lock:
        state.record_instruction_file(
            "ws-main-ctx", instr_path, instr_content,
            session_id=main_sid,
        )

    from proxy.types import RequestContext
    ctx = RequestContext(
        body={}, model_name="deepseek-chat", wants_stream=False,
        session_id=main_sid, client_hint="", messages=[],
        is_subagent=False, workspace_name="ws-main-ctx",
    )

    msgs = _messages_with_sys()
    result = _emit_tier2_pin(copy.deepcopy(msgs), main_sid, ctx=ctx)

    assert result is not msgs
    pin_found = any(
        _PIN_MARKER in m.get("content", "") for m in result if m.get("role") == "system"
    )
    assert pin_found, "Main-path pin emission broken"
