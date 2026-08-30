

def test_resolve_workspace_uses_session_id_and_client_hint():
    """N5: resolve_workspace(ctx) must forward both fields to workspace_store."""
    from unittest.mock import patch, MagicMock
    from proxy.workspace_state import resolve_workspace

    ctx = MagicMock()
    ctx.session_id = "sess-1"
    ctx.client_hint = "kilocode"

    with patch("proxy.workspace_state.workspace_store") as mock_store:
        mock_store.get_active.return_value = "ws-foo"
        result = resolve_workspace(ctx)

    mock_store.get_active.assert_called_once_with("sess-1", client_hint="kilocode")
    assert result == "ws-foo"


def test_resolve_workspace_handles_missing_attributes():
    """N5: resolve_workspace must not raise if ctx has no session_id or client_hint."""
    from unittest.mock import patch
    from proxy.workspace_state import resolve_workspace

    class _Bare:
        pass

    ctx = _Bare()
    with patch("proxy.workspace_state.workspace_store") as mock_store:
        mock_store.get_active.return_value = "default"
        result = resolve_workspace(ctx)

    mock_store.get_active.assert_called_once_with("", client_hint="")
    assert result == "default"


def test_resolve_inherited_workspace_returns_session_workspace_if_not_default():
    """N3: short-circuit if session's own workspace is non-default."""
    from unittest.mock import patch
    from proxy.workspace_state import resolve_inherited_workspace

    with patch("proxy.workspace_state.workspace_store") as mock_store:
        mock_store.get_active.return_value = "ws-explicit"
        result = resolve_inherited_workspace("sess-1")

    assert result == "ws-explicit"
    # No parent walk should happen
    assert mock_store.get_active.call_count == 1


def test_resolve_inherited_workspace_walks_parent_chain():
    """N3: walks parent_agent_id chain when session workspace is 'default'.

    Regression test: parent_agent_id is an agent_id, NOT a session_id.
    The walker must convert it back to the root agent's session_id
    before querying workspace_store. Mock store is session_id-keyed.
    """
    from unittest.mock import patch, MagicMock
    from proxy.workspace_state import resolve_inherited_workspace

    with patch("proxy.workspace_state.workspace_store") as mock_store, \
         patch("proxy.agent.agent_registry.registry") as mock_registry:
        # Session's own workspace is default
        # workspace_store is keyed by session_id — return "ws-root" ONLY
        # for the root session_id. The parent_agent_id "parent-1" must
        # NOT return "ws-root" (it is an agent_id, not a session_id).
        root_session_id = "root-session"
        session_workspaces = {
            "sess-1": "default",
            root_session_id: "ws-root",
        }
        mock_store.get_active.side_effect = (
            lambda sid, **kw: session_workspaces.get(sid, "default")
        )
        # Parent chain: sess-1 -> parent-1 (agent_id of root, no further parent)
        agent = MagicMock()
        agent.parent_agent_id = "parent-1"
        root_agent = MagicMock()
        root_agent.parent_agent_id = None
        root_agent.session_id = root_session_id
        mock_registry.get_or_load.side_effect = lambda sid, **kw: {
            "sess-1": agent, "parent-1": root_agent,
        }.get(sid)

        result = resolve_inherited_workspace("sess-1")

    assert result == "ws-root"


def test_resolve_inherited_workspace_cycle_guard():
    """N3: cycle in parent chain must not loop forever."""
    from unittest.mock import patch, MagicMock
    from proxy.workspace_state import resolve_inherited_workspace

    with patch("proxy.workspace_state.workspace_store") as mock_store, \
         patch("proxy.agent.agent_registry.registry") as mock_registry:
        mock_store.get_active.return_value = "default"
        # Cycle: a -> b -> a
        a = MagicMock(parent_agent_id="b")
        b = MagicMock(parent_agent_id="a")
        mock_registry.get_or_load.side_effect = lambda sid: {"a": a, "b": b}.get(sid)

        result = resolve_inherited_workspace("a")

    # Should not hang; falls back to session's own workspace
    assert result == "default"


def test_resolve_inherited_workspace_logs_on_failure():
    """N3: helper must LOG on parent-walk failure, not silently swallow (Bug Policy)."""
    from unittest.mock import patch, MagicMock
    import logging
    from proxy.workspace_state import resolve_inherited_workspace

    with patch("proxy.workspace_state.workspace_store") as mock_store, \
         patch("proxy.agent.agent_registry.registry") as mock_registry:
        mock_store.get_active.return_value = "default"
        mock_registry.get_or_load.side_effect = RuntimeError("boom")

        with patch("proxy.workspace_state.logger") as mock_logger:
            result = resolve_inherited_workspace("sess-1")

    # Helper logged the error
    assert mock_logger.warning.called, "helper must log on parent-walk failure"
    # Falls back to session's own workspace
    assert result == "default"


def test_resolve_inherited_workspace_log_failures_false_silences_logs():
    """N3: caller can opt out of logging with log_failures=False."""
    from unittest.mock import patch, MagicMock
    from proxy.workspace_state import resolve_inherited_workspace

    with patch("proxy.workspace_state.workspace_store") as mock_store, \
         patch("proxy.agent.agent_registry.registry") as mock_registry:
        mock_store.get_active.return_value = "default"
        mock_registry.get_or_load.side_effect = RuntimeError("boom")

        with patch("proxy.workspace_state.logger") as mock_logger:
            result = resolve_inherited_workspace("sess-1", log_failures=False)

    assert not mock_logger.warning.called
    assert result == "default"
