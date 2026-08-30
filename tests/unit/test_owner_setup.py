"""CP2B: owner_setup performs the owner-only mutations exactly once.

These pins close the migration-coverage gap flagged in impl review: three
side effects moved out of the Flask route (app.py) into
``proxy.owner_setup.owner_setup``:

  1. ``ctx._features = features``
  2. ``ToolCallTracker.reset_instance(ctx.session_id)``
  3. ``DispatchStage._features = features`` on the pipeline's DispatchStage

Without these pins, a future refactor could silently drop ``reset_instance``
(tool-call-limit leakage across turns) or the ``_features`` assignments
(DispatchStage falls back to ``get_features_for_client``, masking the drop)
and no test would fail. Each assertion fails if the corresponding line in
``owner_setup`` is removed (revert-test discipline).
"""
from types import SimpleNamespace
from unittest.mock import patch

from proxy.owner_setup import owner_setup
from proxy.pipeline.stage_modules.dispatch_stage import DispatchStage


class _FakePipeline:
    """Minimal pipeline exposing a ``.stages`` list containing a DispatchStage,
    mirroring RequestPipeline's shape without the import weight."""

    def __init__(self):
        self.stages = [object(), DispatchStage(), object()]


def _make_ctx(session_id="sess-owner-1"):
    # Non-subagent ctx so activate_inherited_workspace is a no-op and does not
    # touch the shared workspace_store — this test isolates the OTHER three
    # side effects.
    return SimpleNamespace(
        session_id=session_id,
        is_subagent=False,
        workspace_name="",
        request_id="req-owner-1",
    )


def test_owner_setup_sets_ctx_features():
    ctx = _make_ctx()
    pipeline = _FakePipeline()
    sentinel = object()
    with patch("proxy.tool_pipeline.tool_call_tracker.ToolCallTracker.reset_instance"):
        owner_setup(ctx, pipeline, sentinel)
    assert ctx._features is sentinel


def test_owner_setup_sets_dispatch_stage_features():
    ctx = _make_ctx()
    pipeline = _FakePipeline()
    sentinel = object()
    with patch("proxy.tool_pipeline.tool_call_tracker.ToolCallTracker.reset_instance"):
        owner_setup(ctx, pipeline, sentinel)
    dispatch = next(s for s in pipeline.stages if isinstance(s, DispatchStage))
    assert dispatch._features is sentinel


def test_owner_setup_resets_tool_call_tracker_once():
    ctx = _make_ctx(session_id="sess-owner-reset")
    pipeline = _FakePipeline()
    with patch(
        "proxy.tool_pipeline.tool_call_tracker.ToolCallTracker.reset_instance"
    ) as reset_mock:
        owner_setup(ctx, pipeline, object())
    reset_mock.assert_called_once_with("sess-owner-reset")
