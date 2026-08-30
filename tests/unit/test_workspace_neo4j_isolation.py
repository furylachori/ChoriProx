"""Guard test: workspace_store.activate must NEVER reach real Neo4j under the
test regime.

Pins the _isolate_workspace_neo4j autouse fixture in tests/conftest.py. This
is the 2753fbe-class guard (AGENTS.md Universal Landmine #1) for the
workspace-persistence vector: binding workspace_store._persist to the real
_persist_workspace callback at import (state.py:444) caused test :Workspace
nodes to MERGE into production Neo4j on every test run (230 nodes polluted).

Non-tautological contract (revert-test verified): if the fixture is removed,
_persist is the real callback, activate() calls _persist_workspace →
_get_executor → execute_write(MERGE :Workspace ...), the mock IS called, and
assert_not_called() FAILS.
"""
from unittest.mock import MagicMock, patch

import proxy.state as state


def test_workspace_store_persist_is_neutralized():
    """The autouse fixture must have set workspace_store._persist to None."""
    assert state.workspace_store._persist is None, (
        "workspace_store._persist must be None under the test regime; the "
        "_isolate_workspace_neo4j autouse fixture did not fire."
    )


def test_workspace_activate_does_not_reach_neo4j():
    """activate() must mutate only in-memory state — never the Neo4j write path.

    Patches the real executor factory so that IF the persist callback were
    live, the workspace MERGE would land on the mock. With the fixture active,
    _persist is None so _persist_workspace is never invoked and execute_write
    is never called.
    """
    mock_executor = MagicMock()
    session_id = "guard-session"
    workspace = "ws-should-not-persist"

    try:
        with patch(
            "memory.workspace_persistence._get_executor", return_value=mock_executor
        ):
            state.workspace_store.activate(session_id, workspace)

        mock_executor.execute_write.assert_not_called()
        assert state.workspace_store._active.get(session_id) == workspace, (
            "activate() must still mutate in-memory state (proves it ran); "
            "isolation must not turn it into a no-op."
        )
    finally:
        state.workspace_store.pop_session(session_id)
