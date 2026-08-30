"""Bug #5: skill error_response() must reach the client with the right HTTP status.

Previously `dispatch.py:try_skill_activation` read `getattr(result, 'status_code', None)`
(an attribute on the inner dispatcher DispatchResult — which has no such field),
so the `_status_code` key that `error_response()` embeds in the response dict had
zero readers. As a result `^^ticket <unknown-hash>` returned "Error: No task found"
with HTTP 200 instead of 404.

These tests pin the three links in the chain:

1. ``try_skill_activation`` returns a ``DispatchResult`` whose ``status_code``
   mirrors the embedded ``_status_code`` (e.g. 404).
2. ``DispatchStage.execute`` wraps that into ``StageAbort((body, status))`` so
   the Flask layer's ``_response_from_stage_abort`` returns the 4xx response.
3. ``error_response`` (the producer) still embeds the status key.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from proxy.pipeline.dispatch import DispatchResult, try_skill_activation
from proxy.pipeline.exceptions import StageAbort
from proxy.pipeline.stages import DispatchStage
from proxy.skills.skill_definition import BaseSkill
from proxy.skills.skill_registry import (
    SkillActivation,
    SkillMeta,
    SkillRegistry,
    error_response,
    reset_default_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx(text: str, session_id: str = "test-sess", wants_stream: bool = False):
    """Minimal RequestContext-like mock for ``try_skill_activation``."""
    ctx = MagicMock()
    ctx.session_id = session_id
    ctx.wants_stream = wants_stream
    ctx.messages = [{"role": "user", "content": text}]
    ctx.is_subagent = False
    ctx.parent_agent_id = None
    ctx.inherited_skill = None
    ctx.working_directory = "/tmp"
    ctx.seeded_tasks = []
    ctx.active_workspace = None
    # Bug #7 adjoint: client_hint must be a real string for type-validation.
    ctx.client_hint = ""
    return ctx


def _make_features():
    features = MagicMock()
    return features


@pytest.fixture(autouse=True)
def flask_app_context():
    from qwen_proxy import app
    with app.app_context():
        yield


@pytest.fixture
def clean_registry():
    """Reset the singleton registry and skill rate limiter between tests."""
    from proxy.state import skill_rate_limiter
    reset_default_registry()
    skill_rate_limiter.reset()
    registry = SkillRegistry()
    registry.clear_all()
    yield registry, _make_features()
    skill_rate_limiter.reset()
    reset_default_registry()


# ---------------------------------------------------------------------------
# error_response (producer side)
# ---------------------------------------------------------------------------


class TestErrorResponseEmbedsStatusCode:
    def test_default_status_is_400(self):
        resp = error_response("bad input")
        assert resp["_status_code"] == 400
        assert resp["choices"][0]["message"]["content"] == "Error: bad input"

    def test_explicit_404_status(self):
        resp = error_response("No task found", 404)
        assert resp["_status_code"] == 404


# ---------------------------------------------------------------------------
# try_skill_activation (consumer side — dispatch.py)
# ---------------------------------------------------------------------------


class TestSkillActivationRespectsEmbeddedStatus:
    """``handle_direct`` returns ``error_response(...)`` → dispatch sets the right status."""

    def _register_error_skill(self, registry, *, status_code: int, message: str):
        # Use a non-builtin skill name ("custom") so the trust gate does not
        # reject the registration from this test module (only "ticket" /
        # "work" are TRUSTED_BUILTIN_SKILLS, gated to specific caller modules).
        class ErrorSkill(BaseSkill):
            name = "custom"
            trigger = "^^custom"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("show", "deadbeef")

            def handle_direct(self, ctx, action, arg):
                return error_response(message, status_code)

            def get_meta(self):
                return SkillMeta(name="custom", trigger="^^custom")

        registry.register(ErrorSkill())

    def test_unknown_hash_returns_404(self, clean_registry):
        """^^custom <unknown-hash> → status_code 404, not 200."""
        registry, features = clean_registry
        self._register_error_skill(registry, status_code=404, message="No task found")
        ctx = _make_ctx("^^custom deadbeef")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.status_code == 404, (
            f"Expected 404 from embedded _status_code; got {result.status_code}"
        )
        # F3: error_response() embeds _status_code so dispatch.py can read the
        # intended HTTP status. After reading it, dispatch.py MUST pop it so
        # the internal field does not leak to the client JSON body.
        body = result.response
        assert "_status_code" not in body, (
            f"_status_code must be stripped from the client-visible body; "
            f"found: {body.get('_status_code')!r}"
        )
        assert "No task found" in body["choices"][0]["message"]["content"]

    def test_bad_arg_returns_400(self, clean_registry):
        """error_response default (400) is honored too."""
        registry, features = clean_registry
        self._register_error_skill(registry, status_code=400, message="bad arg")
        ctx = _make_ctx("^^custom deadbeef")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.status_code == 400
        # F3: same stripping guarantee for the 400 branch.
        assert "_status_code" not in result.response




    def test_success_path_still_returns_200(self, clean_registry):
        """A handle_direct that returns a plain choices dict (no _status_code)
        MUST still resolve to status_code 200 — do not break the success path."""
        registry, features = clean_registry

        class OkSkill(BaseSkill):
            name = "custom"
            trigger = "^^custom"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("status", "")

            def handle_direct(self, ctx, action, arg):
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ],
                    "model": "skill",
                }

            def get_meta(self):
                return SkillMeta(name="custom", trigger="^^custom")

        registry.register(OkSkill())
        ctx = _make_ctx("^^custom status")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        assert result.status_code == 200
        assert "_status_code" not in result.response


# ---------------------------------------------------------------------------
# SkillValidationError → 4xx (not 500)
# ---------------------------------------------------------------------------


class TestSkillValidationErrorRoutesTo4xx:
    """F1: a `SkillValidationError` raised inside `skill.on_activate` (e.g.
    WorkSkill rejecting an unknown task hash) MUST be surfaced as the
    exception's declared HTTP status (404 by default), not the generic 500
    that an unrelated `ValueError` would receive.

    The exception subclasses `ValueError`, so dispatch.py must check for it
    FIRST in the `except (ValueError, PermissionError)` branch and honor its
    `http_status` attribute. The generic 500 path stays in place for genuine,
    unexpected ValueErrors.
    """

    def test_work_bad_hash_returns_404_not_500(self, clean_registry):
        """End-to-end: `^^work <unknown-hash>` → status_code 404 (not 500)."""
        from proxy.skills.skill_definition import SkillValidationError

        registry, features = clean_registry

        class _WorkLikeSkill(BaseSkill):
            """Stand-in for WorkSkill whose on_activate raises SkillValidationError.

            We don't use the real WorkSkill here because the test would then
            need a live task graph and workspace store. The dispatch path
            we care about (SkillDispatcher.dispatch → activate → on_activate
            raises) is identical regardless of which skill does the raising.

            The skill name MUST NOT be "work" — that's a TRUSTED_BUILTIN_SKILL
            and the trust gate rejects re-registration from this test module.
            Use a benign name + trigger that mirrors the `^^work` flow.
            """
            name = "worklike"
            trigger = "^^worklike"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("task", "deadbeef")

            def handle_direct(self, ctx, action, arg):
                # Fall through to activation by returning None.
                return None

            def get_meta(self):
                return SkillMeta(name="worklike", trigger="^^worklike")

            def on_activate(self, ctx, mode_state=None):
                raise SkillValidationError(
                    "No task found with hash 'deadbeef'. "
                    "Use ^^ticket list to find a valid task hash.",
                    http_status=404,
                )

        registry.register(_WorkLikeSkill())
        ctx = _make_ctx("^^worklike deadbeef")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result is not None, (
            "^^worklike <bad-hash> must produce a DispatchResult (not None)"
        )
        assert result.handled is True
        assert result.status_code == 404, (
            f"SkillValidationError(http_status=404) must yield 404, not "
            f"{result.status_code}. The dispatch layer is dropping the "
            f"declared http_status and falling through to the generic 500 path."
        )
        # F3: the body's _status_code must be stripped before it reaches the client.
        body = result.response
        assert "_status_code" not in body, (
            f"_status_code must be stripped from the client-visible body; "
            f"found: {body.get('_status_code')!r}"
        )
        assert "No task found with hash 'deadbeef'" in body["choices"][0]["message"]["content"]




    def test_generic_value_error_still_returns_500(self, clean_registry):
        """A bare ValueError (NOT a SkillValidationError) MUST still go to 500.

        Regression guard: F1's isinstance check must not accidentally route
        all ValueErrors to 4xx — only SkillValidationError is special-cased.
        """
        registry, features = clean_registry

        class _RaiserSkill(BaseSkill):
            name = "custom"
            trigger = "^^raiser"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("task", "x")

            def handle_direct(self, ctx, action, arg):
                return None

            def get_meta(self):
                return SkillMeta(name="custom", trigger="^^raiser")

            def on_activate(self, ctx, mode_state=None):
                # Plain ValueError — NOT a SkillValidationError.
                raise ValueError("unexpected internal failure")

        registry.register(_RaiserSkill())
        ctx = _make_ctx("^^raiser x")

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result is not None
        assert result.handled is True
        assert result.status_code == 500, (
            f"Plain ValueError must remain on the generic 500 path; "
            f"got {result.status_code}. F1's isinstance check is too broad."
        )


# ---------------------------------------------------------------------------
# DispatchStage (consumer side — stage_modules/dispatch_stage.py)
# ---------------------------------------------------------------------------


class TestDispatchStageThreadsStatusIntoStageAbort:
    """Bug #5: StageAbort must carry (body, status) for 4xx, body for 200."""

    def test_404_status_wraps_into_tuple(self):
        """4xx dict response → StageAbort carries (response, status_code) tuple."""
        resp = error_response("No task found", 404)
        result = DispatchResult(
            rewritten_text=None,
            response=resp,
            status_code=404,
            handled=True,
        )
        ctx = MagicMock()
        stage = DispatchStage(features=MagicMock())

        with patch("proxy.pipeline.dispatch.run_dispatch", return_value=result):
            with pytest.raises(StageAbort) as exc_info:
                stage.execute(ctx)

        payload = exc_info.value.response_data
        assert isinstance(payload, tuple), (
            f"4xx response must be wrapped as (body, status); got {type(payload)}"
        )
        body, status = payload
        assert status == 404
        assert body is resp

    def test_200_status_passes_dict_through(self):
        """200 dict response → StageAbort carries the dict as-is (no wrapping).

        Regression guard: the existing contract (test_pipeline_stages.py::
        TestDispatchStage::test_dispatch_stage_raises_stageabort_on_handled)
        asserts ``response_data is resp``. We must not regress that for 200.
        """
        resp = {"object": "chat.completion", "result": "done"}
        result = DispatchResult(
            rewritten_text=None,
            response=resp,
            status_code=200,
            handled=True,
        )
        ctx = MagicMock()
        stage = DispatchStage(features=MagicMock())

        with patch("proxy.pipeline.dispatch.run_dispatch", return_value=result):
            with pytest.raises(StageAbort) as exc_info:
                stage.execute(ctx)

        # 200 → dict passthrough (not wrapped).
        assert exc_info.value.response_data is resp


# ---------------------------------------------------------------------------
# Streaming path: error_response + ctx.wants_stream
# ---------------------------------------------------------------------------


class TestStreamingPathHonorsEmbeddedStatus:
    """Bug #5 / sub-agent #5: the streaming code path
    (``dispatch.py`` lines 482-523) reads ``_status_code`` from a skill's
    ``error_response(...)`` dict and stamps it on the SSE ``Response``
    object. Previously only the non-streaming branch was tested.

    This test pins the streaming end-to-end: ``error_response("no task", 404)``
    in, ``ctx.wants_stream=True``, ``try_skill_activation`` out, the returned
    ``DispatchResult.response`` is a Flask ``Response`` whose ``.status_code``
    is 404 (not the default 200).
    """

    def _register_error_skill(self, registry, *, status_code: int, message: str):
        class ErrorSkill(BaseSkill):
            name = "custom"
            trigger = "^^custom"
            activation = SkillActivation.COMMAND
            ttl_seconds = 300

            def parse_subcommand(self, text):
                return ("show", "deadbeef")

            def handle_direct(self, ctx, action, arg):
                return error_response(message, status_code)

            def get_meta(self):
                return SkillMeta(name="custom", trigger="^^custom")

        registry.register(ErrorSkill())

    def test_streaming_response_carries_404_status(self, clean_registry):
        """^^custom <unknown-hash> with wants_stream=True → DispatchResult
        status_code 404 (preserved on the plain dict back to the route;
        SSE framing is owned by app:_keepalive_stream which honors
        DispatchResult.status_code)."""
        registry, features = clean_registry
        self._register_error_skill(registry, status_code=404, message="no task")
        ctx = _make_ctx("^^custom deadbeef", wants_stream=True)

        with patch("proxy.skills.skill_registry.SkillRegistry", return_value=registry):
            result = try_skill_activation(ctx, features)

        assert result.handled is True
        # The new contract: dispatch hands a plain dict back to the route
        # with status_code preserved. SSE framing (and the eventual Flask
        # Response) is built by proxy.app:_keepalive_stream, not by
        # dispatch.
        assert isinstance(result.response, dict), (
            f"Streaming path must hand a plain dict to the route; "
            f"got {type(result.response).__name__}"
        )
        assert result.status_code == 404, (
            f"DispatchResult.status_code must carry the embedded _status_code (404); "
            f"got {result.status_code}. The dispatch path drops "
            f"error_response's status."
        )
