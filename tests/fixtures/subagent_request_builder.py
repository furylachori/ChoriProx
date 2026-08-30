"""Sub-agent request builder for Tier-2 pin emission tests.

Builds (messages, ctx) pairs that drive ``_select_context_v2`` into the
sub-agent branch (``is_subagent=True``) and reach each of the four
``_build_subagent_context`` return paths:

  :314  fits           — small messages, all fit within safe budget
  :335  no-first-user  — overflow, no user message in non-system messages
  :361  trimmed-fits   — overflow, has first_user, trimmed last unit fits
  :369  fallback        — overflow, first_user itself exhausts budget

Usage::

    from tests.fixtures.subagent_request_builder import make_subagent_request
    msgs, ctx = make_subagent_request(parent_workspace="ws", overflow=False)
"""
import copy
import logging
import uuid

import proxy.state as state
from proxy.types import RequestContext

logger = logging.getLogger(__name__)

_TOKENS_PER_CHAR = 0.25
_TOOL_SCHEMA_TOKENS = 2000
_MODEL_SAFE_TOKENS = 24000
_NON_SYSTEM_BUDGET = _MODEL_SAFE_TOKENS - _TOOL_SCHEMA_TOKENS  # 22000 tokens
_NON_SYSTEM_BUDGET_CHARS = int(_NON_SYSTEM_BUDGET / _TOKENS_PER_CHAR)  # 88000 chars
_TRUNC_MARKER = "... [truncated]"


def _unique_session_id(prefix="sub"):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _make_ctx(session_id, is_subagent=True, parent_agent_id=None, workspace_name=""):
    ctx = RequestContext(
        body={},
        model_name="deepseek-chat",
        wants_stream=False,
        session_id=session_id,
        client_hint="",
        messages=[],
        is_subagent=is_subagent,
    )
    if parent_agent_id is not None:
        ctx.inherited_skill = None
    return ctx


def _register_parent_agent(session_id, workspace_name):
    from proxy.agent.agent import Agent
    from proxy.agent.agent_config import AgentConfig
    from proxy.agent.agent_registry import registry

    agent = Agent(
        session_id=session_id,
        parent=None,
        config=AgentConfig(),
    )
    with registry._lock:
        registry._register(agent)
    if workspace_name and workspace_name != "default":
        state.workspace_store.activate(session_id, workspace_name)


def _register_subagent_with_parent(sub_session_id, parent_session_id):
    from proxy.agent.agent import Agent
    from proxy.agent.agent_config import AgentConfig
    from proxy.agent.agent_registry import registry

    parent_agent = None
    with registry._lock:
        parent_agent = registry._agents.get(parent_session_id)
    agent = Agent(
        session_id=sub_session_id,
        parent=parent_agent,
        config=AgentConfig(),
        is_subagent=True,
    )
    with registry._lock:
        registry._register(agent)


def make_subagent_request(
    parent_workspace="",
    messages=None,
    overflow=False,
    session_id=None,
    parent_session_id=None,
    model_cfg=None,
    target_return=None,
):
    """Build a (messages, ctx) pair that drives ``_select_context_v2`` into
    the sub-agent branch and reaches one of the 4 ``_build_subagent_context``
    returns.

    WARNING — S1 PATH-COVERAGE ONLY. When ``parent_workspace`` is set, this
    builder activates the workspace directly on the sub-agent session
    (``workspace_store.activate``), which MASKS the inheritance path. Use this
    builder only for sub-agent return-path coverage (:314/:335/:361/:369). For
    any assertion about workspace INHERITANCE (F1/F2), use
    ``make_inherited_subagent_request`` instead — it leaves ctx.workspace_name
    empty and proves the parent-chain walk.

    Parameters
    ----------
    parent_workspace : str
        Workspace name the sub-agent should resolve to. If non-empty, the
        parent agent is registered with this workspace and the sub-agent
        session is set to inherit it via ``workspace_store``.
    messages : list | None
        Custom message list. When None, messages are synthesized based on
        ``overflow`` and ``target_return``.
    overflow : bool
        If True (and messages is None), synthesize messages that overflow
        the safe budget to reach :335/:361/:369. If False, synthesize
        small messages that fit (:314).
    session_id : str | None
        Sub-agent session ID. Auto-generated if None.
    parent_session_id : str | None
        Parent session ID. Auto-generated if None.
    model_cfg : object | None
        Unused (kept for API compat). Model config is derived from
        model_name inside ``_select_context_v2``.
    target_return : str | None
        Force a specific return path: ``"fits"``, ``"no-first-user"``,
        ``"trimmed-fits"``, or ``"fallback"``. Overrides ``overflow``.

    Returns
    -------
    (messages, ctx) : tuple[list, RequestContext]
        ``ctx.is_subagent`` is True; ``ctx.session_id`` is the sub-agent
        session; workspace inheritance is wired so
        ``resolve_inherited_workspace(session_id)`` returns
        ``parent_workspace``.

    Cleanup
    -------
    This builder registers agents in ``AgentRegistry`` and sets workspace
    state. Callers should reset shared state between tests::

        import proxy.state as state
        with state.vfs_lock:
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()
        from proxy.agent.agent_registry import registry as _reg
        with _reg._lock:
            _reg._agents.clear()
            _reg._agents_by_id.clear()
            _reg._lru.clear()
    """
    sub_sid = session_id or _unique_session_id("sub")
    parent_sid = parent_session_id or _unique_session_id("parent")

    if parent_workspace:
        _register_parent_agent(parent_sid, parent_workspace)
        _register_subagent_with_parent(sub_sid, parent_sid)
        state.workspace_store.activate(sub_sid, parent_workspace)

    ctx = _make_ctx(sub_sid)

    if messages is not None:
        return copy.deepcopy(messages), ctx

    if target_return is None:
        target_return = "fits" if not overflow else "trimmed-fits"

    if target_return == "fits":
        msgs = [
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": "working on it..."},
        ]
    elif target_return == "no-first-user":
        big = "A" * (_NON_SYSTEM_BUDGET_CHARS + 1000)
        msgs = [
            {"role": "assistant", "content": big},
        ]
    elif target_return == "trimmed-fits":
        user_msg = {"role": "user", "content": "please complete this assignment"}
        big = "B" * (_NON_SYSTEM_BUDGET_CHARS + 1000)
        assistant_msg = {"role": "assistant", "content": big}
        msgs = [user_msg, assistant_msg]
    elif target_return == "fallback":
        huge = "C" * (_NON_SYSTEM_BUDGET_CHARS + 1000)
        tail = {"role": "assistant", "content": "final response"}
        msgs = [
            {"role": "user", "content": huge},
            tail,
        ]
    else:
        raise ValueError(f"Unknown target_return={target_return!r}")

    return msgs, ctx


def make_inherited_subagent_request(
    parent_workspace="",
    messages=None,
    session_id=None,
    parent_session_id=None,
    target_return=None,
):
    """Build a (messages, ctx) pair where the sub-agent INHERITS the parent
    workspace via the parent-agent chain — NOT via ctx.workspace_name.

    The parent agent is registered with ``parent_workspace``; the sub-agent
    ctx has ``is_subagent=True`` and ``workspace_name=""`` (or "default").
    ``resolve_inherited_workspace`` must walk the parent chain to find the
    workspace. This proves the pin emits via inheritance, not via a manually
    set ctx.workspace_name (which was the old masking pattern).

    For ``parent_workspace="default"``, the parent's session_id is what
    ``record_instruction_file`` uses as the tuple key; the sub-agent must
    resolve the parent session via ``resolve_inherited_session``.
    """
    sub_sid = session_id or _unique_session_id("sub")
    parent_sid = parent_session_id or _unique_session_id("parent")

    _register_parent_agent(parent_sid, parent_workspace)
    _register_subagent_with_parent(sub_sid, parent_sid)

    ctx = _make_ctx(sub_sid)
    ctx.is_subagent = True

    if messages is not None:
        return copy.deepcopy(messages), ctx

    if target_return is None:
        target_return = "fits"

    if target_return == "fits":
        msgs = [
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": "working on it..."},
        ]
    elif target_return == "no-first-user":
        big = "A" * (_NON_SYSTEM_BUDGET_CHARS + 1000)
        msgs = [
            {"role": "assistant", "content": big},
        ]
    elif target_return == "trimmed-fits":
        user_msg = {"role": "user", "content": "please complete this assignment"}
        big = "B" * (_NON_SYSTEM_BUDGET_CHARS + 1000)
        assistant_msg = {"role": "assistant", "content": big}
        msgs = [user_msg, assistant_msg]
    elif target_return == "fallback":
        huge = "C" * (_NON_SYSTEM_BUDGET_CHARS + 1000)
        tail = {"role": "assistant", "content": "final response"}
        msgs = [
            {"role": "user", "content": huge},
            tail,
        ]
    else:
        raise ValueError(f"Unknown target_return={target_return!r}")

    return msgs, ctx


def _detect_return_path(messages, result):
    """Heuristic to infer which ``_build_subagent_context`` return path
    produced ``result``. Returns one of: ``"fits"``, ``"no-first-user"``,
    ``"trimmed-fits"``, ``"fallback"``, ``"unknown"``.

    Evidence used:
    - fallback: only the first-user message survives. The content may
      be unchanged (B1/B3 dropped), or shortened with the slice-4D
      restoration trim-safety-net suffix ``... [truncated]`` (when the
      §13.5 finalizer content-trimmed an otherwise-raiseable user
      message). Detected by: exactly 1 non-system result message, role
      user, content starts with ``orig_content[:50]`` OR ends with the
      truncation marker.
    - no-first-user: no user-role message in result but content present.
    - trimmed-fits: multiple non-system messages, at least one has the
      ``... [truncated]`` suffix indicating _trim_unit_to_budget ran.
    - fits: all original non-system messages preserved verbatim (identity
      or deep-equal content, no truncation suffix).
    """
    original_non_system = [m for m in messages if m.get("role") != "system"]
    result_non_system = [m for m in result if m.get("role") != "system"]

    if not result_non_system:
        return "unknown"

    has_user = any(m.get("role") == "user" for m in result_non_system)
    orig_user = next((m for m in original_non_system if m.get("role") == "user"), None)

    if len(result_non_system) == 1 and has_user and orig_user is not None:
        only_content = result_non_system[0].get("content", "")
        orig_content = orig_user.get("content", "")
        if only_content == orig_content:
            return "fallback"
        if (
            len(orig_content) > 100
            and only_content[:50] == orig_content[:50]
        ):
            return "fallback"
        if (
            len(only_content) > len(_TRUNC_MARKER)
            and only_content.endswith(_TRUNC_MARKER)
        ):
            prefix = only_content[: -len(_TRUNC_MARKER)]
            if orig_content.startswith(prefix):
                return "fallback"

    if not has_user:
        return "no-first-user"

    any_truncated = any(
        _TRUNC_MARKER in str(m.get("content", ""))
        for m in result_non_system
    )
    if any_truncated:
        return "trimmed-fits"

    if len(result_non_system) == len(original_non_system):
        all_match = all(
            r.get("content") == o.get("content")
            for r, o in zip(result_non_system, original_non_system)
        )
        if all_match:
            return "fits"

    return "unknown"


if __name__ == "__main__":
    from proxy.context_v2 import _select_context_v2

    print("=== Testing make_subagent_request builder ===\n")

    test_cases = [
        ("fits", {}),
        ("no-first-user", {"target_return": "no-first-user"}),
        ("trimmed-fits", {"target_return": "trimmed-fits"}),
        ("fallback", {"target_return": "fallback"}),
    ]

    for name, kwargs in test_cases:
        print(f"--- target_return={name} ---")
        msgs, ctx = make_subagent_request(**kwargs)
        result = _select_context_v2(
            session_id=ctx.session_id,
            full_messages=copy.deepcopy(msgs),
            model_name="deepseek-chat",
            is_subagent=True,
            ctx=ctx,
        )
        detected = _detect_return_path(msgs, result)
        print(f"  messages: {len(msgs)} msgs, "
              f"result: {len(result)} msgs, detected path: {detected}")
        for m in result:
            c = str(m.get("content", ""))[:50]
            print(f"    {m['role']}: {c}")
        match = "OK" if detected == name else f"MISMATCH (expected {name})"
        print(f"  -> {match}\n")

        # Clean up between runs
        from proxy.agent.agent_registry import registry as _reg
        with _reg._lock:
            _reg._agents.clear()
            _reg._agents_by_id.clear()
            _reg._lru.clear()
        with state.vfs_lock:
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()


def test_make_subagent_request_reaches_all_paths():
    """Smoke test: each target_return reaches the expected sub-agent path."""
    from proxy.context_v2 import _select_context_v2

    for target in ("fits", "no-first-user", "trimmed-fits", "fallback"):
        msgs, ctx = make_subagent_request(target_return=target)
        result = _select_context_v2(
            session_id=ctx.session_id,
            full_messages=copy.deepcopy(msgs),
            model_name="deepseek-chat",
            is_subagent=True,
            ctx=ctx,
        )
        detected = _detect_return_path(msgs, result)
        assert detected == target, (
            f"target_return={target!r} produced path={detected!r}"
        )
        assert ctx.is_subagent is True

        from proxy.agent.agent_registry import registry as _reg
        with _reg._lock:
            _reg._agents.clear()
            _reg._agents_by_id.clear()
            _reg._lru.clear()
        with state.vfs_lock:
            state.instruction_files_by_workspace.clear()
            state._last_scanned_by_tool_call.clear()
