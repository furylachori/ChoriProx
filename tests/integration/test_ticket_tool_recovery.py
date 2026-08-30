"""Slice 3 closure phase — end-to-end integration test.

Exercises the real full path: PostProcessStage.execute → handle_tool_interception
→ ToolPipeline.run → run_memory_tool_loop → execute_memory_tool →
execute_proxy_ticket → real action → requery_llm → closure →
ToolPipeline._format. Only upstream HTTP and the TaskGraph singleton are
mocked. The proxy_ticket action reaches the real dispatcher and the
recorded TaskGraph.

Five scenarios per the Slice 3 mandate:
  1) observed flow: list_tasks success → 2 invalid create_tasks → corrected
     create_tasks success → requery failure → closure LLM success
  2) closure LLM success (simple)
  3) closure emits native tool_call → fallback
  4) closure upstream error/timeout → fallback
  5) happy-path clean no-tool final → no closure call
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration
real_agent_registry = pytest.mark.real_agent_registry


class _RecordingTaskGraph:
    def __init__(self):
        self.created_tasks: list[dict] = []
        self.create_calls: list[dict] = []
        self.list_calls: int = 0

    def create_tasks_batch(self, batch, workspace_id=None):
        self.create_calls.append({"batch": list(batch), "workspace_id": workspace_id})
        results = []
        for entry in batch:
            name = entry.get("name", "")
            short_hash = f"hash_{hashlib.md5(name.encode()).hexdigest()[:7]}"
            self.created_tasks.append({
                "short_hash": short_hash,
                "name": name,
                "priority": entry.get("priority", "medium"),
                "workspace_id": workspace_id,
            })
            results.append({
                "short_hash": short_hash,
                "name": name,
                "status": "success",
            })
        return results

    def get_actionable_tasks(self, workspace_id=None, status=None,
                            status_filter=None, limit=50):
        self.list_calls += 1
        return [
            {
                "short_hash": t["short_hash"],
                "name": t["name"],
                "status": "open",
                "priority": t["priority"],
                "workspace_id": t.get("workspace_id"),
            }
            for t in self.created_tasks
        ]

    def get_task_enriched(self, short_hash, workspace_id=None):
        for t in self.created_tasks:
            if t["short_hash"] == short_hash:
                return {"found": True, **t}
        return {"found": False}

    def update_task_status(self, short_hash, new_status, workspace_id=None,
                           caller_agent_id=None):
        for t in self.created_tasks:
            if t["short_hash"] == short_hash:
                return {"ok": True, "status": new_status, "changed": True}
        return {"ok": False, "status": "not_found"}

    def resolve_blocked_by_hashes(self, *args, **kwargs):
        return []

    def resolve_parent_hash(self, *args, **kwargs):
        return {"ok": True}

    def append_review_note(self, *args, **kwargs):
        return {"ok": True}


class _MockUpstream(BaseHTTPRequestHandler):
    responses: list = []
    captured: list = []
    request_count: int = 0
    cursor: int = 0
    _repeat_payload: dict | None = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {"__raw__": raw[:200].decode("utf-8", "replace")}
        _MockUpstream.captured.append(body)
        _MockUpstream.request_count += 1

        if _MockUpstream._repeat_payload is not None:
            payload = _MockUpstream._repeat_payload
            _MockUpstream._repeat_payload = None
        else:
            idx = _MockUpstream.cursor
            if idx < len(_MockUpstream.responses):
                payload = _MockUpstream.responses[idx]
                _MockUpstream.cursor += 1
                if isinstance(payload, dict) and payload.get("__repeat"):
                    _MockUpstream._repeat_payload = payload
            else:
                payload = {
                    "choices": [{
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }]
                }

        encoded = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception:
            pass

    def log_message(self, format, *args):
        pass


def _failure_response():
    return {"__repeat": True, "choices": []}


@pytest.fixture
def mock_upstream():
    _MockUpstream.responses = []
    _MockUpstream.captured = []
    _MockUpstream.request_count = 0
    _MockUpstream.cursor = 0
    _MockUpstream._repeat_payload = None
    server = HTTPServer(("127.0.0.1", 0), _MockUpstream)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1/chat/completions"
    server.shutdown()


@pytest.fixture
def recording_tg():
    recording = _RecordingTaskGraph()
    import proxy.context as ctx_mod
    original = ctx_mod._task_graph_singleton
    ctx_mod._task_graph_singleton = recording
    try:
        yield recording
    finally:
        ctx_mod._task_graph_singleton = original


def _text_response(content, finish_reason="stop"):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }]
    }


def _tool_call_response(name, arguments, call_id="call_1"):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }]
    }


def _build_real_request_context(
    session_id: str,
    initial_response_data: dict,
    upstream_url: str,
    recording_tg,
):
    """Build a minimally valid RequestContext that drives the real
    PostProcessStage through handle_tool_interception with a real
    permitted ticket sub-agent.
    """
    from proxy.agent.agent import Agent
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.skills.skill_definition import ModeState
    from proxy.skill_lifecycle import register_lifecycle_callbacks
    from proxy.types import (
        ModelRoute, RequestContext, UpstreamRequest,
    )
    from clients.orchestrator_config import ProxyFeatures

    agent = Agent(session_id=session_id, agent_id=f"agent-{session_id}")
    agent.is_subagent = True
    agent._ticket_mode_ts = time.time()
    mode_state = ModeState(
        name="ticket",
        activated_at=time.time(),
        ttl_seconds=300,
        allowed_tool_names=frozenset([
            "read", "glob", "grep", "search",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
            "proxy_work",
        ]),
        sub_agent_tool_filter=frozenset([
            "read", "glob", "grep", "search",
            "proxy_ticket_list_tasks", "proxy_ticket_get_task",
            "proxy_ticket_create_tasks", "proxy_ticket_update_task",
            "proxy_ticket_edit", "proxy_ticket_set_parent",
            "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
            "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
            "webfetch",
        ]),
        skill_state={"subcommand": "plan", "task_hash": None},
    )
    agent.set_active_mode(mode_state)
    register_lifecycle_callbacks(agent)
    with agent_registry._lock:
        agent_registry._agents.pop(session_id, None)
        agent_registry._agents_by_id.pop(agent.agent_id, None)
    agent_registry.register(agent)
    retrieved = agent_registry.get_or_load(session_id)
    if retrieved is None or retrieved.snapshot_mode() is None:
        raise RuntimeError(
            f"could not register permitted ticket agent for {session_id}"
        )

    body = {
        "model": "kimi",
        "messages": [{"role": "user", "content": "do work"}],
        "tools": [],
    }
    llama_body = {
        "model": "kimi",
        "messages": [{"role": "user", "content": "do work"}],
        "tools": [],
    }

    ctx = RequestContext(
        body=body,
        model_name="kimi",
        wants_stream=False,
        session_id=session_id,
        client_hint="closure-test",
        messages=[{"role": "user", "content": "do work"}],
        upstream_url=upstream_url,
        tool_parser=None,
        auth_headers={},
        original_model="kimi",
        model_cfg=type("Cfg", (), {
            "memory_enabled": True,
            "tool_parser": None,
            "backend_model": "kimi",
        })(),
        max_iterations=5,
        max_total_time=900.0,
        max_result_size=16384,
        response_data=initial_response_data,
        is_subagent=True,
        subagent_ticket_mode=True,
        unattended=True,
        agent=retrieved,
    )
    ctx.model_route = ModelRoute(
        model_cfg=ctx.model_cfg,
        upstream_url=upstream_url,
        tool_parser=None,
        is_remote=True,
    )
    ctx.upstream = UpstreamRequest(
        url=upstream_url,
        body=llama_body,
        headers={},
        original_model="kimi",
    )
    ctx.llama_body = llama_body
    ctx._features = ProxyFeatures(max_tool_iterations=5, tool_loop_timeout=900)
    return ctx


def _build_bare_main_request_context(
    session_id: str,
    initial_response_data: dict,
    upstream_url: str,
):
    """Build a RequestContext for a bare MAIN agent (no active skill mode).

    Used only by the Slice 10 S2 regression test. The ticket capability
    table — part of the proxy_ticket decomposition — rejects the deleted
    monolithic ``proxy_ticket`` name for every ticket-mode role (main and
    sub-agent) before the call can reach the dispatcher. To exercise the
    dispatch-layer ``unknown_memory_tool`` branch through the real full
    path (and make the S2 revert-test meaningful), the call must come from
    an agent that is NOT gated by the ticket capability table — a bare main
    agent with no active mode. The guard chain then admits ``proxy_ticket``
    (mode allowlist unrestricted, no work-mode/ticket-orchestrator/sub-agent
    gate applies) and the real dispatcher rejects it as unknown_memory_tool.
    """
    from proxy.agent.agent import Agent
    from proxy.agent.agent_registry import registry as agent_registry
    from proxy.types import (
        ModelRoute, RequestContext, UpstreamRequest,
    )
    from clients.orchestrator_config import ProxyFeatures

    agent = Agent(session_id=session_id, agent_id=f"agent-{session_id}")
    # Deliberately NO active mode and NOT a sub-agent.
    with agent_registry._lock:
        agent_registry._agents.pop(session_id, None)
        agent_registry._agents_by_id.pop(agent.agent_id, None)
    agent_registry.register(agent)
    retrieved = agent_registry.get_or_load(session_id)
    if retrieved is None:
        raise RuntimeError(
            f"could not register bare main agent for {session_id}"
        )

    body = {
        "model": "kimi",
        "messages": [{"role": "user", "content": "do work"}],
        "tools": [],
    }
    llama_body = {
        "model": "kimi",
        "messages": [{"role": "user", "content": "do work"}],
        "tools": [],
    }

    ctx = RequestContext(
        body=body,
        model_name="kimi",
        wants_stream=False,
        session_id=session_id,
        client_hint="slice10-s2",
        messages=[{"role": "user", "content": "do work"}],
        upstream_url=upstream_url,
        tool_parser=None,
        auth_headers={},
        original_model="kimi",
        model_cfg=type("Cfg", (), {
            "memory_enabled": True,
            "tool_parser": None,
            "backend_model": "kimi",
        })(),
        max_iterations=5,
        max_total_time=900.0,
        max_result_size=16384,
        response_data=initial_response_data,
        is_subagent=False,
        subagent_ticket_mode=False,
        unattended=True,
        agent=retrieved,
    )
    ctx.model_route = ModelRoute(
        model_cfg=ctx.model_cfg,
        upstream_url=upstream_url,
        tool_parser=None,
        is_remote=True,
    )
    ctx.upstream = UpstreamRequest(
        url=upstream_url,
        body=llama_body,
        headers={},
        original_model="kimi",
    )
    ctx.llama_body = llama_body
    ctx._features = ProxyFeatures(max_tool_iterations=5, tool_loop_timeout=900)
    return ctx


@real_agent_registry
@pytest.mark.usefixtures("mock_upstream", "recording_tg")
class TestSlice3ClosureE2E:
    """Real full-path Slice 3 closure end-to-end scenarios via
    PostProcessStage.execute(ctx)."""

    def _drive_post_process(self, ctx):
        """Drive the real full path through PostProcessStage.execute."""
        from proxy.pipeline.post_process import PostProcessStage
        return PostProcessStage().execute(ctx)

    def test_observed_full_path_closure_success(
        self, mock_upstream, recording_tg
    ):
        """Exact observed: list_tasks → invalid create → invalid create
        → corrected create → requery fail (with retry) → closure LLM
        success. The fake TaskGraph emits a deterministic hash/name/
        priority; the closure report must carry those exact details.
        """
        session_id = "closure-e2e-observed-flow"
        _MockUpstream.responses = [
            _tool_call_response(
                "proxy_ticket_create_tasks", {"tasks": []},
                call_id="tc_bad_1",
            ),
            _tool_call_response(
                "proxy_ticket_create_tasks", {"tasks": []},
                call_id="tc_bad_2",
            ),
            _tool_call_response(
                "proxy_ticket_create_tasks",
                {"tasks": [{"name": "Fix the bug", "priority": "high"}]},
                call_id="tc_good",
            ),
            _failure_response(),
            _text_response(
                "PARTIAL Confirmed operations:\n"
                "- created hash_4adc494 priority=high name=Fix the bug\n"
                "Errors:\n"
                "- proxy_ticket_create_tasks create_tasks failed (attempt 1) "
                "code=invalid_input "
                "create_tasks requires a non-empty tasks array\n"
            ),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_list_tasks",
                                    "arguments": json.dumps({}),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(session_id, initial, mock_upstream, recording_tg)

        result_data = self._drive_post_process(ctx)

        message = result_data["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"]
        content = message.get("content", "")
        assert content.startswith("PARTIAL"), (
            f"closure content must begin with PARTIAL at position 0, "
            f"got {content!r}"
        )
        assert "Fix the bug" in content
        assert "priority=high" in content
        assert "code=invalid_input" in content
        assert "create_tasks requires a non-empty tasks array" in content
        assert result_data["choices"][0]["finish_reason"] == "stop"

        assert len(recording_tg.create_calls) == 1
        created_names = [c["name"] for c in recording_tg.created_tasks]
        assert created_names == ["Fix the bug"]
        expected_hash = recording_tg.created_tasks[0]["short_hash"]
        assert expected_hash in content

        # Account for: 4 requery calls + 1 retry (failure_marker) + 1
        # closure call = 6 upstream HTTP calls. The mock's request_count
        # counts each POST.
        assert _MockUpstream.request_count == 6, (
            f"expected 6 upstream POSTs (4 requery + 1 retry + 1 closure), "
            f"got {_MockUpstream.request_count}"
        )

        closure_body = _MockUpstream.captured[-1]
        for forbidden in ("tools", "tool_choice", "parallel_tool_calls",
                          "functions", "function_call"):
            assert forbidden not in closure_body, (
                f"closure body must not contain {forbidden!r}; got keys: "
                f"{list(closure_body.keys())}"
            )
        assert expected_hash in closure_body["messages"][-1]["content"]
        assert "code=invalid_input" in closure_body["messages"][-1]["content"]

    def test_closure_lmm_success_simple(self, mock_upstream, recording_tg):
        session_id = "closure-e2e-simple"
        _MockUpstream.responses = [
            _failure_response(),
            _text_response(
                "COMPLETE Confirmed operations:\n"
                "- created hash_ce499de priority=high name=T1\n"
            ),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_create_tasks",
                                    "arguments": json.dumps({
                                        "tasks": [{"name": "T1",
                                                   "priority": "high"}],
                                    }),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(session_id, initial, mock_upstream, recording_tg)
        result_data = self._drive_post_process(ctx)

        message = result_data["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"]
        content = message.get("content", "")
        assert content.startswith("COMPLETE")
        assert "T1" in content
        expected_hash = recording_tg.created_tasks[0]["short_hash"]
        assert expected_hash in content
        assert result_data["choices"][0]["finish_reason"] == "stop"
        assert _MockUpstream.request_count == 3
        assert len(recording_tg.create_calls) == 1

    def test_closure_emits_native_tool_call_fallback(
        self, mock_upstream, recording_tg
    ):
        session_id = "closure-e2e-native-leak"
        _MockUpstream.responses = [
            _failure_response(),
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "PARTIAL something",
                        "tool_calls": [{
                            "id": "tc_closure_leak",
                            "type": "function",
                            "function": {
                                "name": "bash", "arguments": "{}",
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_create_tasks",
                                    "arguments": json.dumps({
                                        "tasks": [{"name": "T1"}],
                                    }),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(session_id, initial, mock_upstream, recording_tg)
        result_data = self._drive_post_process(ctx)

        message = result_data["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"]
        content = message.get("content", "")
        assert content.startswith("COMPLETE")
        assert "T1" in content
        assert "hash_ce499de" in content
        assert result_data["choices"][0]["finish_reason"] == "stop"

    def test_closure_upstream_error_fallback(
        self, mock_upstream, recording_tg
    ):
        session_id = "closure-e2e-error"

        class _ErroringHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(content_length)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"upstream error"}')

            def log_message(self, format, *args):
                pass

        err_server = HTTPServer(("127.0.0.1", 0), _ErroringHandler)
        err_port = err_server.server_address[1]
        threading.Thread(
            target=err_server.serve_forever, daemon=True
        ).start()

        _MockUpstream.responses = [
            _failure_response(),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_create_tasks",
                                    "arguments": json.dumps({
                                        "tasks": [{"name": "T1",
                                                   "priority": "high"}],
                                    }),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        try:
            ctx = _build_real_request_context(
                session_id, initial,
                f"http://127.0.0.1:{err_port}/v1/chat/completions",
                recording_tg,
            )
            result_data = self._drive_post_process(ctx)
        finally:
            err_server.shutdown()

        message = result_data["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"]
        content = message.get("content", "")
        assert content.startswith("COMPLETE")
        assert "T1" in content
        assert "hash_ce499de" in content
        assert result_data["choices"][0]["finish_reason"] == "stop"

    def test_happy_path_clean_no_closure_call(
        self, mock_upstream, recording_tg
    ):
        session_id = "closure-e2e-happy"
        _MockUpstream.responses = [
            _text_response("The work is complete.", finish_reason="stop"),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "recall",
                                    "arguments": json.dumps({
                                        "semantic_targets": ["x"],
                                        "keywords": ["x"],
                                    }),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(session_id, initial, mock_upstream, recording_tg)
        result_data = self._drive_post_process(ctx)

        message = result_data["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"]
        assert message.get("content") == "The work is complete."
        assert result_data["choices"][0]["finish_reason"] == "stop"

        assert _MockUpstream.request_count == 1

        # No dangling pairs in chat history.
        from proxy.chat_history import chat_histories
        history = chat_histories.get(session_id, [])
        valid_ids = set()
        present_ids = set()
        for m in history:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("id"):
                        valid_ids.add(tc["id"])
            elif m.get("role") == "tool" and m.get("tool_call_id"):
                present_ids.add(m["tool_call_id"])
        assert not (valid_ids - present_ids), (
            f"history contains dangling assistant tool_call(s) with no "
            f"paired tool result: {sorted(valid_ids - present_ids)}"
        )


@real_agent_registry
@pytest.mark.usefixtures("mock_upstream", "recording_tg")
class TestSlice10MiniMaxRegression:
    """Slice 10 — full-path MiniMax regression tests.

    Reproduces the production failure class (a ticket sub-agent dumping a
    raw tool result / raw tool call as its final response) through the REAL
    production path:

        PostProcessStage.execute → handle_tool_interception
        → ToolPipeline.run → run_memory_tool_loop → execute_memory_tool
        → execute_ticket_tool → dispatch_action → real action
        → requery_llm → closure → ToolPipeline._format

    Only upstream HTTP (mock backend), the TaskGraph singleton (recording
    fake) and the closure LLM response (scripted upstream payload) are
    mocked. classify_turn_response, finalize_skill_mode, ToolPipeline.run,
    run_memory_tool_loop, execute_ticket_tool, dispatch_action and every
    action class run for real.

    Scenarios:
      S1 — malformed JSON arguments reach the genuine malformed_arguments
           branch and surface as a FAILED closure (no raw dump).
      S2 — the deleted monolithic ``proxy_ticket`` name reaches the real
           dispatcher and is rejected as ``unknown_memory_tool``.
      S3 — ``proxy_ticket_update_task`` missing ``new_status`` yields a
           canonical ``invalid_status`` error naming the offending field.
      S4 — three identical invalid create_tasks calls are bounded by the
           SemanticRetryTracker (2 executions + 1 not-executed rejection)
           and close as FAILED with ``validation_retry_exhausted``.
      S5 — a successful mutation followed by an upstream requery failure
           closes cleanly with no ``tool_calls`` array and no raw envelope
           / internal tool-name leak in the client-facing content.
    """

    def _drive_post_process(self, ctx):
        """Drive the real full path through PostProcessStage.execute."""
        from proxy.pipeline.post_process import PostProcessStage
        return PostProcessStage().execute(ctx)

    @staticmethod
    def _closure_body():
        """The last upstream POST body — the closure finalization request."""
        return _MockUpstream.captured[-1]

    @staticmethod
    def _assert_clean_final_message(result_data):
        """Assert the client-facing message has no tool_calls and a
        COMPLETE/PARTIAL/FAILED closure content with finish_reason=stop."""
        message = result_data["choices"][0]["message"]
        assert "tool_calls" not in message or not message["tool_calls"], (
            f"final client message must not carry tool_calls; got: "
            f"{message.get('tool_calls')!r}"
        )
        content = message.get("content", "") or ""
        assert content.startswith(("COMPLETE", "PARTIAL", "FAILED")), (
            f"final client content must begin with COMPLETE/PARTIAL/FAILED, "
            f"got {content!r}"
        )
        assert result_data["choices"][0]["finish_reason"] == "stop"
        return content

    def test_s1_malformed_json_arguments_reach_malformed_branch(
        self, mock_upstream, recording_tg
    ):
        """A tool call whose arguments are a malformed JSON string (not a
        JSON object) must hit the genuine ``malformed_arguments`` branch in
        run_memory_tool_loop — NOT be silently coerced to ``{}`` — and the
        proxy must still produce a proper FAILED closure rather than dumping
        the raw tool result.

        The malformed call is delivered by the first re-query (not the
        initial response): a malformed argument on the initial native tool
        call is rejected upstream of the loop, so the loop only ever sees a
        malformed argument string via a re-query-produced tool call.
        """
        session_id = "slice10-s1-malformed-json"
        _MockUpstream.responses = [
            # requery #1 (after the valid list_tasks) → a tool call whose
            # arguments are a JSON *string scalar* wrapping malformed JSON
            # object text. requery parsing yields a Python str argument that
            # fails the tool loop's json.loads object expectation.
            _tool_call_response(
                "proxy_ticket_create_tasks", "{tasks: [BROKEN",
                call_id="tc_malformed",
            ),
            # requery #2 (after the malformed call) → empty choices (with the
            # mock's __repeat retry) → requery_failed → forces closure.
            _failure_response(),
            # closure finalization response.
            _text_response(
                "FAILED The proxy_ticket_create_tasks call carried malformed "
                "JSON arguments and could not be executed. No task was created."
            ),
        ]

        # Initial call: a valid list_tasks so the loop starts cleanly and the
        # malformed call arrives via the re-query turn.
        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_list_tasks",
                                    "arguments": json.dumps({}),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(
            session_id, initial, mock_upstream, recording_tg
        )

        result_data = self._drive_post_process(ctx)

        self._assert_clean_final_message(result_data)
        # Nothing was created — the malformed call never reached the action.
        assert recording_tg.create_calls == []
        assert recording_tg.created_tasks == []

        # REVERT-SENSITIVE: the malformed_arguments error must have reached
        # the execution ledger sent to the closure LLM. If the malformed
        # branch is reverted to fall through with tool_args={}, the action
        # runs create_tasks({}) → invalid_input, and "malformed_arguments"
        # never appears in the closure request.
        closure_serialized = json.dumps(self._closure_body(), default=str)
        assert "malformed_arguments" in closure_serialized, (
            "closure ledger must carry the malformed_arguments error code"
        )
        # The closure request must be a tools-disabled finalization request.
        for forbidden in ("tools", "tool_choice", "parallel_tool_calls",
                          "functions", "function_call"):
            assert forbidden not in self._closure_body(), (
                f"closure body must not contain {forbidden!r}; got keys: "
                f"{list(self._closure_body().keys())}"
            )

    def test_s2_monolithic_proxy_ticket_is_unknown_memory_tool(
        self, mock_upstream, recording_tg
    ):
        """If a model emits the deleted monolithic ``proxy_ticket`` name and
        the call reaches the dispatcher, it must be rejected as
        ``unknown_memory_tool`` (then close cleanly) — never crash, never
        leak a raw result.

        A bare main agent (no active skill mode) is used so the call
        traverses the real guard chain and reaches the dispatcher: the
        ticket capability table — part of the decomposition — rejects the
        monolithic name for every ticket-mode role before dispatch, so a
        ticket-mode agent can never reach the dispatch-layer unknown branch.
        The dispatcher has no entry for ``proxy_ticket`` (it was deleted),
        so it returns unknown_memory_tool.
        """
        session_id = "slice10-s2-unknown-tool"
        _MockUpstream.responses = [
            _failure_response(),
            _text_response(
                "FAILED The requested proxy_ticket tool is not available; "
                "only action-specific proxy_ticket_* tools are supported. "
                "No operation was performed."
            ),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket",
                                    "arguments": json.dumps(
                                        {"action": "list_tasks"}
                                    ),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_bare_main_request_context(
            session_id, initial, mock_upstream
        )

        result_data = self._drive_post_process(ctx)

        self._assert_clean_final_message(result_data)
        assert recording_tg.create_calls == []
        assert recording_tg.list_calls == 0

        # REVERT-SENSITIVE: the dispatcher's unknown-tool branch must have
        # produced unknown_memory_tool. If ``proxy_ticket`` is re-added to
        # the dispatch map, the call routes to a real action and the ledger
        # no longer contains unknown_memory_tool.
        closure_serialized = json.dumps(self._closure_body(), default=str)
        assert "unknown_memory_tool" in closure_serialized, (
            "closure ledger must carry the unknown_memory_tool error code"
        )

    def test_s3_update_task_missing_field_names_the_field(
        self, mock_upstream, recording_tg
    ):
        """``proxy_ticket_update_task`` with a known hash but no
        ``new_status`` must return a canonical ``invalid_status`` error that
        names the offending field, and still close cleanly.
        """
        from proxy.tool_pipeline.tool_call_tracker import ToolCallTracker

        session_id = "slice10-s3-missing-field"
        known_hash = "hash_abc1234"
        # Seed hash provenance as a prior list_tasks within this request
        # would have, so the guard admits the modify call and it reaches the
        # real UpdateTaskAction field validation.
        ToolCallTracker.get_instance(session_id).add_seen_hashes([known_hash])

        _MockUpstream.responses = [
            _failure_response(),
            _text_response(
                "FAILED The update_task call was rejected because it was "
                "missing the required new_status field."
            ),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_update_task",
                                    "arguments": json.dumps(
                                        {"short_hash": known_hash}
                                    ),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(
            session_id, initial, mock_upstream, recording_tg
        )

        result_data = self._drive_post_process(ctx)

        self._assert_clean_final_message(result_data)
        # The invalid call must not have mutated the TaskGraph.
        assert recording_tg.created_tasks == []

        # The canonical invalid_status error (naming new_status) must have
        # reached the closure ledger.
        closure_serialized = json.dumps(self._closure_body(), default=str)
        assert "invalid_status" in closure_serialized, (
            "closure ledger must carry the invalid_status error code"
        )
        assert "new_status" in closure_serialized, (
            "closure ledger must name the offending new_status field"
        )

    def test_s4_identical_invalid_calls_bounded_by_retry_tracker(
        self, mock_upstream, recording_tg
    ):
        """Three identical invalid create_tasks calls must be bounded by the
        SemanticRetryTracker: the first two execute (and fail validation),
        the third identical call is rejected without execution
        (validation_retry_exhausted), and the loop closes as FAILED.
        """
        session_id = "slice10-s4-retry-exhaustion"
        _MockUpstream.responses = [
            # requery #1 → identical invalid call (2nd execution)
            _tool_call_response(
                "proxy_ticket_create_tasks", {"tasks": []}, call_id="tc_rep_1",
            ),
            # requery #2 → identical invalid call (3rd, rejected by tracker)
            _tool_call_response(
                "proxy_ticket_create_tasks", {"tasks": []}, call_id="tc_rep_2",
            ),
            # closure finalization response.
            _text_response(
                "FAILED The create_tasks call repeated the same invalid "
                "arguments and exhausted its retry budget. No task was created."
            ),
        ]

        # Initial call: identical invalid create_tasks (empty tasks array).
        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_create_tasks",
                                    "arguments": json.dumps({"tasks": []}),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(
            session_id, initial, mock_upstream, recording_tg
        )

        result_data = self._drive_post_process(ctx)

        content = self._assert_clean_final_message(result_data)
        # No task was ever created — every attempt failed validation.
        assert recording_tg.create_calls == []
        assert recording_tg.created_tasks == []

        # REVERT-SENSITIVE: the third identical call must have been rejected
        # by the SemanticRetryTracker (validation_retry_exhausted) and that
        # not-executed record must appear in the closure ledger. If the
        # identical-call detection is disabled, no rejection record exists
        # and "validation_retry_exhausted" never appears.
        closure_serialized = json.dumps(self._closure_body(), default=str)
        assert "validation_retry_exhausted" in closure_serialized, (
            "closure ledger must record the validation_retry_exhausted "
            "rejection of the third identical call"
        )
        # The closure request must be a tools-disabled finalization request.
        for forbidden in ("tools", "tool_choice", "parallel_tool_calls",
                          "functions", "function_call"):
            assert forbidden not in self._closure_body(), (
                f"closure body must not contain {forbidden!r}; got keys: "
                f"{list(self._closure_body().keys())}"
            )

    def test_s5_no_tool_call_or_raw_result_leak_to_client(
        self, mock_upstream, recording_tg
    ):
        """A successful mutation followed by an upstream requery failure must
        close cleanly: the client-facing message carries no ``tool_calls``
        array and no raw envelope / internal tool-name leak — only the
        closure prose.
        """
        session_id = "slice10-s5-no-leak"
        _MockUpstream.responses = [
            # requery after the successful create → empty choices (with the
            # mock's __repeat retry) → requery_failed → forces closure.
            _failure_response(),
            # closure finalization response.
            _text_response(
                "COMPLETE Created 1 task: S5 regression task (priority=high)."
            ),
        ]

        initial = {
            "choices": [{
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "tc_init", "type": "function",
                                "function": {
                                    "name": "proxy_ticket_create_tasks",
                                    "arguments": json.dumps({
                                        "tasks": [{
                                            "name": "S5 regression task",
                                            "priority": "high",
                                        }],
                                    }),
                                },
                            }]},
                "finish_reason": "tool_calls",
            }],
            "model": "kimi",
        }
        ctx = _build_real_request_context(
            session_id, initial, mock_upstream, recording_tg
        )

        result_data = self._drive_post_process(ctx)

        message = result_data["choices"][0]["message"]
        content = self._assert_clean_final_message(result_data)

        # The real action ran exactly once and created the task.
        assert len(recording_tg.create_calls) == 1
        assert [t["name"] for t in recording_tg.created_tasks] == [
            "S5 regression task"
        ]
        # The created task's hash must have reached the closure ledger sent
        # to the closure LLM (proves the real success record survived to the
        # closure phase, not just the scripted closure text).
        expected_hash = recording_tg.created_tasks[0]["short_hash"]
        assert expected_hash in json.dumps(self._closure_body(), default=str)

        # No raw action envelope may cross the boundary as client content.
        assert not content.strip().startswith("{"), (
            f"client content must not be a raw JSON dump, got {content!r}"
        )
        for leak_marker in ('"ok":', '"operation":', '"session_id":',
                            '"status": "success"', '"tool_calls"'):
            assert leak_marker not in content, (
                f"client content leaks raw envelope marker {leak_marker!r}: "
                f"{content!r}"
            )
        # No internal memory tool name leaked as a raw tool_call block.
        assert "proxy_ticket_create_tasks" not in json.dumps(
            message.get("tool_calls") or []
        )