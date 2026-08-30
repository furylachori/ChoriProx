"""Integration tests for C7 — observability, sync honesty, downgrade coverage.

D7: per-request v2 INFO telemetry + VFS build failure WARNING
D4: sync submission honesty (executors + post_process)
D6: cold-path coverage gap detection (monitoring + catch-up only, NOT current-turn fix)
"""
import copy
import logging
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.integration


SESSION = "test-c7-observability"


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "memory.db")
    monkeypatch.setenv("PROXY_TEST_DB", db_path)
    from memory.db import init_db
    init_db(db_path)
    yield db_path


def _seed_conv_row(session_id):
    from memory.db_summaries import update_ancient_history
    update_ancient_history(session_id, "seed-ancient-for-bypass")


def _make_cfg(safe_tokens=100_000):
    return SimpleNamespace(backend_model="m")


def _patch_context_config(safe_tokens=100_000):
    cfg = {
        "max_context_tokens": int(safe_tokens / 0.75),
        "safe_context_tokens": safe_tokens,
        "optimal_context_tokens": int(safe_tokens * 0.47),
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": max(5, int(safe_tokens / 0.75) // 4000),
    }
    return patch("proxy.context.get_context_config", return_value=cfg)


def _patch_vfs():
    return (
        patch("proxy.vfs._build_vfs_context", return_value=None),
        patch("proxy.vfs._build_vfs_inherited_context", return_value=None),
    )


def _make_request(n_exchanges, chars_per_msg=100):
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"u{i} " + "U" * chars_per_msg, "message_index": i * 2})
        msgs.append({"role": "assistant", "content": f"a{i} " + "A" * chars_per_msg, "message_index": i * 2 + 1})
    return msgs


# ============================================================================
# D7 — INFO telemetry (one per assembled request)
# ============================================================================


class TestD7InfoTelemetry:
    def test_d7_info_telemetry_per_request(self, caplog):
        from proxy.context_v2 import _select_context_v2

        _seed_conv_row(SESSION)

        with caplog.at_level(logging.INFO, logger="proxy.context_v2"):
            req = _make_request(5, chars_per_msg=100)
            with ExitStack() as ctx_stack:
                ctx_stack.enter_context(_patch_context_config(100_000))
                for p in _patch_vfs():
                    ctx_stack.enter_context(p)
                _select_context_v2(
                    SESSION, req, model_name="m",
                    model_cfg=_make_cfg(100_000),
                )

        info_lines = [r for r in caplog.records if r.levelno == logging.INFO]
        v2_lines = [r for r in info_lines if "v2:" in r.getMessage()]
        assert len(v2_lines) == 1, (
            f"Expected exactly 1 INFO 'v2:' line, got {len(v2_lines)}: "
            f"{[r.getMessage() for r in v2_lines]}"
        )

        msg = v2_lines[0].getMessage()
        for field in (
            "raw_tok=",
            "summaries=",
            "ancient_chars=",
            "vfs_entries=",
            "boundary=",
            "cold=",
        ):
            assert field in msg, f"Missing field {field!r} in INFO line: {msg!r}"


# ============================================================================
# D7 — VFS build failure WARNING
# ============================================================================


class TestD7VfsWarning:
    def test_d7_vfs_warning_on_failure(self, caplog):
        from proxy.context_v2 import _select_context_v2
        from proxy.types import RequestContext

        _seed_conv_row(SESSION)

        req = _make_request(5, chars_per_msg=100)
        req.append({"role": "user", "content": "current D7 user request"})

        request_ctx = RequestContext(
            body={"messages": req},
            model_name="m",
            wants_stream=False,
            session_id=SESSION,
            client_hint="",
            messages=req,
            remote_addr="127.0.0.1",
        )
        request_ctx.request_id = "test-d7-vfs-warning"

        req_before = copy.deepcopy(req)

        with caplog.at_level(logging.WARNING, logger="proxy.context_v2"):
            with ExitStack() as ctx_stack:
                ctx_stack.enter_context(_patch_context_config(100_000))
                mock_builder = ctx_stack.enter_context(
                    patch(
                        "proxy.vfs._build_vfs_context",
                        side_effect=RuntimeError("vfs broke"),
                    )
                )
                result = _select_context_v2(
                    SESSION,
                    req,
                    model_name="m",
                    model_cfg=_make_cfg(100_000),
                    ctx=request_ctx,
                )

        mock_builder.assert_called_once_with(
            SESSION,
            request_ctx,
            5_000,
            0.25,
        )

        vfs_warnings = [
            r for r in caplog.records
            if r.name == "proxy.context_v2"
            and r.levelno == logging.WARNING
            and r.getMessage().startswith("V2 vfs build failed")
        ]
        assert len(vfs_warnings) == 1, (
            f"Expected exactly one WARNING for V2 normal-path VFS build failure, got: "
            f"{[(r.levelname, r.name, r.getMessage()) for r in caplog.records]}"
        )
        assert "vfs broke" in vfs_warnings[0].getMessage(), (
            f"Expected injected exception text in warning; got: {vfs_warnings[0].getMessage()!r}"
        )
        assert vfs_warnings[0].request_id == "test-d7-vfs-warning", (
            f"Expected warning request_id to match; got: {vfs_warnings[0].request_id!r}"
        )

        for m in result:
            content = m.get("content", "")
            if isinstance(content, str):
                assert "[VFS CONTEXT" not in content, (
                    f"VFS marker must not appear in result; got: {m!r}"
                )
                assert "[VFS INHERITED CONTEXT" not in content, (
                    f"Inherited VFS marker must not appear in result; got: {m!r}"
                )
        assert any(
            m.get("role") == "user"
            and (m.get("content") or "") == "current D7 user request"
            for m in result
        ), "Current user must remain in result"
        assert req == req_before, "Input request must remain unchanged"


# ============================================================================
# D4 — sync submission honesty
# ============================================================================


def _build_minimal_postprocess_ctx(session_id, request_id="test-c7-1"):
    return SimpleNamespace(
        request_id=request_id,
        is_subagent=False,
        session_id=session_id,
        response_data={
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}],
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
        },
        model_route=SimpleNamespace(
            model_cfg=SimpleNamespace(
                ignore_subagents=False,
                memory_enabled=True,
                backend_model="m",
            ),
            is_remote=False,
            tool_parser=None,
        ),
        upstream=None,
        body={"messages": []},
        messages_original=None,
        _features=None,
        llama_body=None,
        client_hint="",
        agent=None,
        model_name="m",
    )


class TestD4SyncSubmitted:
    def test_d4_post_process_logs_submitted_when_future_returned(self, caplog):
        from proxy.pipeline.post_process import PostProcessStage

        fake_future = MagicMock()
        ctx = _build_minimal_postprocess_ctx(SESSION)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(
                patch("memory.sync.submit_sync", return_value=fake_future)
            )
            with caplog.at_level(logging.INFO, logger="proxy.pipeline.stages"):
                PostProcessStage().execute(ctx)

        submit_info = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and r.getMessage() == f"Submitted sync for session {SESSION}"
        ]
        assert len(submit_info) == 1, (
            f"Expected exactly 1 'Submitted sync' INFO from post_process, got "
            f"{[r.getMessage() for r in submit_info]}"
        )

    def test_d4_post_process_logs_deferred_when_future_none(self, caplog):
        from proxy.pipeline.post_process import PostProcessStage

        ctx = _build_minimal_postprocess_ctx(SESSION, request_id="test-c7-2")

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(
                patch("memory.sync.submit_sync", return_value=None)
            )
            with caplog.at_level(logging.INFO, logger="proxy.pipeline.stages"):
                PostProcessStage().execute(ctx)

        deferred_info = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and r.getMessage() == f"Sync deferred (debounced) for session {SESSION}"
        ]
        assert len(deferred_info) == 1, (
            f"Expected exactly 1 'Sync deferred' INFO from post_process, got "
            f"{[r.getMessage() for r in deferred_info]}"
        )


# ============================================================================
# D4 — sync debounced path (executors: per-session lock already held)
# ============================================================================


class TestD4SyncDebounced:
    def test_d4_executors_logs_debounce_at_info_level(self, caplog):
        from memory import executors as exec_mod

        from memory.executors import _session_locks, _session_locks_lock
        with _session_locks_lock:
            _session_locks[SESSION] = object()

        try:
            with caplog.at_level(logging.INFO, logger="memory.executors"):
                result = exec_mod.submit_sync(lambda: None, SESSION)

            assert result is None, "submit_sync should return None when debounced"
            debounce_logs = [
                r for r in caplog.records
                if r.levelno == logging.INFO and "debounced" in r.getMessage()
            ]
            assert len(debounce_logs) >= 1, (
                f"Expected INFO debounce log, got: "
                f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
            )
            assert "Sync debounced" in debounce_logs[0].getMessage(), (
                f"Expected 'Sync debounced' in log message, got: "
                f"{debounce_logs[0].getMessage()!r}"
            )
        finally:
            with _session_locks_lock:
                _session_locks.pop(SESSION, None)


# ============================================================================
# D6 — cold-path coverage gap detection
# ============================================================================


def _seed_messages_with_exchange_index(session_id, n_exchanges, chars_per_msg=500):
    from memory.db import store_messages_batch
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"u{i} " + "U" * chars_per_msg})
        msgs.append({"role": "assistant", "content": f"a{i} " + "A" * chars_per_msg})
    store_messages_batch(session_id, msgs, start_index=0)
    return msgs


class TestD6CoverageGap:
    def test_d6_coverage_gap_warning_and_catch_up(self, caplog):
        from memory.db_summaries import set_last_summarized_exchange_index
        from proxy.context_v2 import _select_context_v2

        SESSION_GAP = SESSION + "-gap"
        _seed_messages_with_exchange_index(SESSION_GAP, 20, chars_per_msg=500)
        set_last_summarized_exchange_index(SESSION_GAP, exchange_index=2)

        req = _make_request(20, chars_per_msg=500)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(1_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            mock_submit = ctx_stack.enter_context(
                patch("memory.sync.submit_sync")
            )
            with caplog.at_level(logging.WARNING, logger="proxy.context_v2"):
                _select_context_v2(
                    SESSION_GAP, req, model_name="m",
                    model_cfg=_make_cfg(1_000),
                )

        coverage_logs = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "coverage gap" in r.getMessage()
        ]
        assert len(coverage_logs) >= 1, (
            f"Expected D6 coverage-gap WARNING, got: "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        assert mock_submit.called, (
            "Coverage gap detected but catch-up submit_sync was not called"
        )

    def test_d6_no_gap_no_warning(self, caplog):
        from memory.db_summaries import set_last_summarized_exchange_index
        from proxy.context_v2 import _select_context_v2

        SESSION_NOGAP = SESSION + "-no-gap"
        _seed_messages_with_exchange_index(SESSION_NOGAP, 20, chars_per_msg=500)
        set_last_summarized_exchange_index(SESSION_NOGAP, exchange_index=100)

        req = _make_request(20, chars_per_msg=500)

        with ExitStack() as ctx_stack:
            ctx_stack.enter_context(_patch_context_config(1_000))
            for p in _patch_vfs():
                ctx_stack.enter_context(p)
            mock_submit = ctx_stack.enter_context(
                patch("memory.sync.submit_sync")
            )
            with caplog.at_level(logging.WARNING, logger="proxy.context_v2"):
                _select_context_v2(
                    SESSION_NOGAP, req, model_name="m",
                    model_cfg=_make_cfg(1_000),
                )

        coverage_logs = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "coverage gap" in r.getMessage()
        ]
        assert len(coverage_logs) == 0, (
            f"No gap expected but coverage-gap WARNING logged: "
            f"{[r.getMessage() for r in coverage_logs]}"
        )
        assert not mock_submit.called, (
            "No gap expected but catch-up submit_sync was called"
        )

