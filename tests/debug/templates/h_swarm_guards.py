#!/usr/bin/env python3
"""Template H — custom research swarm guards and failures.

Exercises every contract guard of the custom-skill research-swarm runtime
through ``debug_session(..., skills_dir=<tmp>, enable_custom_skill_subagents=True)``:

  A. Exact budget accepted (max=1, 1 task call → HTTP 200)
  B. Over budget rejected atomically (max=1, 2 task calls → HTTP 400,
     code ``swarm_max_calls_exceeded``, no choices, no canary in body)
  C. Universal no-nesting (sub-agent attempted nested task → HTTP 403,
     code ``swarm_subagent_cannot_spawn``, no choices)
  D. No-task pass-through (content-only response → HTTP 200, content
     unchanged, no canary, no signing error)
   E. Operator visibility (``/health`` reports swarm_enabled + the loaded
      skill; ``^^status`` reports "Swarm-enabled: yes" and the skill name).

Each scenario uses a FRESH ``tempfile.TemporaryDirectory`` and a FRESH
``debug_session`` — no ``c.reset()`` between different skill dirs.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/h_swarm_guards.py
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from _helpers import _emit, _finish
from devtools.debug_session import debug_session

SECRET = "debug-swarm-secret-at-least-32-bytes-long-0123456789"


def _read_tool():
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
            },
        },
    }


def _grep_tool():
    return {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
            },
        },
    }


def _task_tool():
    return {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a sub-agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["prompt"],
            },
        },
    }


def _build_skill(*, max_task_calls):
    return {
        "name": "research",
        "trigger": "^^research",
        "description": "Debug-session research swarm scenario.",
        "prompt": "RESEARCH_SWARM_DEBUG_PROMPT",
        "tool_whitelist": ["read", "grep", "task"],
        "subagents": {
            "tool_whitelist": ["read", "grep"],
            "max_task_calls_per_request": max_task_calls,
        },
    }


def _write_skill(d, max_task_calls=1):
    with open(os.path.join(d, "research.json"), "w") as f:
        json.dump(_build_skill(max_task_calls=max_task_calls), f)


def main():
    secret = SECRET
    parent_tools = [_read_tool(), _grep_tool(), _task_tool()]

    # ── Scenario A: exact budget accepted (max=1, 1 task) ────────────────
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, max_task_calls=1)
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "BUDGET_PROMPT_ONE",
                                             "description": "research"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ]
        with debug_session(
            canned_responses=canned,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "h-scenario-a"
            parent_sid = c.session_id_for(
                [{"role": "user", "content": "^^research do something"}],
                client_hint="h-scenario-a",
            )
            c.ensure_agent(parent_sid)
            resp = c.chat(
                messages=[{"role": "user", "content": "^^research do something"}],
                tools=parent_tools,
                client_hint="h-scenario-a",
            )
            _emit(1, "A: HTTP 200", resp.status_code == 200,
                  f"status={resp.status_code}")
            prompts = c.extract_task_prompts(resp)
            _emit(2, "A: one signed task prompt", len(prompts) == 1,
                  f"count={len(prompts)}")

    # ── Scenario B: over budget rejected atomically (max=1, 2 tasks) ─────
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, max_task_calls=1)
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "BUDGET_PROMPT_ONE",
                                             "description": "research"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                },
                                {
                                    "id": "call_task_2",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "BUDGET_PROMPT_TWO",
                                             "description": "research"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ]
        with debug_session(
            canned_responses=canned,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "h-scenario-b"
            parent_sid = c.session_id_for(
                [{"role": "user", "content": "^^research do something"}],
                client_hint="h-scenario-b",
            )
            c.ensure_agent(parent_sid)
            resp = c.chat(
                messages=[{"role": "user", "content": "^^research do something"}],
                tools=parent_tools,
                client_hint="h-scenario-b",
            )
            body = resp.json() or {}
            err = (body or {}).get("error", {})
            _emit(3, "B: HTTP 400", resp.status_code == 400,
                  f"status={resp.status_code}")
            _emit(4, "B: code swarm_max_calls_exceeded",
                  err.get("code") == "swarm_max_calls_exceeded",
                  json.dumps(err))
            _emit(5, "B: no choices or tool_calls (top-level)",
                  "choices" not in body,
                  f"body={json.dumps(body)[:200]}")
            _emit(6, "B: no canary leaked in body",
                  "[[__SUBAGENT__" not in json.dumps(body),
                  "leak_check=clean")

    # ── Scenario C: universal no-nested-task (sub cannot spawn sub) ───────
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, max_task_calls=1)
        canned_parent = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "nested prompt",
                                             "description": "research"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ]
        canned_child = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "attempted nested task",
                                             "description": "research"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ]
        with debug_session(
            canned_responses=canned_parent + canned_child,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "h-scenario-c"
            parent_sid = c.session_id_for(
                [{"role": "user", "content": "^^research spawn child"}],
                client_hint="h-scenario-c",
            )
            c.ensure_agent(parent_sid)
            parent_resp = c.chat(
                messages=[{"role": "user", "content": "^^research spawn child"}],
                tools=parent_tools,
                client_hint="h-scenario-c",
            )
            canary = c.extract_swarm_canary(parent_resp)
            _emit(7, "C: canary obtained", canary is not None,
                  f"canary={'yes' if canary else 'none'}")
            if canary is None:
                _emit(8, "C: HTTP 403 (skipped)", False, "no canary")
                _emit(9, "C: code swarm_subagent_cannot_spawn (skipped)", False,
                      "no canary")
                _emit(10, "C: no choices (skipped)", False, "no canary")
            else:
                child_resp, _sid = c.send_as_subagent(
                    canary=canary,
                    subagent_prompt="nested prompt",
                    subagent_messages=[],
                    client_hint="h-scenario-c-child",
                )
                body = child_resp.json() or {}
                err = (body or {}).get("error", {})
                _emit(8, "C: HTTP 403", child_resp.status_code == 403,
                      f"status={child_resp.status_code}")
            _emit(9, "C: code swarm_subagent_cannot_spawn",
                  err.get("code") == "swarm_subagent_cannot_spawn",
                  json.dumps(err))
            _emit(10, "C: no choices",
                  "choices" not in body,
                  f"body={json.dumps(body)[:200]}")

    # ── Scenario D: no-task pass-through (content-only) ──────────────────
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, max_task_calls=1)
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "plain response",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ]
        with debug_session(
            canned_responses=canned,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "h-scenario-d"
            parent_sid = c.session_id_for(
                [{"role": "user", "content": "^^research just text"}],
                client_hint="h-scenario-d",
            )
            c.ensure_agent(parent_sid)
            resp = c.chat(
                messages=[{"role": "user", "content": "^^research just text"}],
                tools=parent_tools,
                client_hint="h-scenario-d",
            )
            body = resp.json() or {}
            _emit(11, "D: HTTP 200", resp.status_code == 200,
                  f"status={resp.status_code}")
            _emit(12, "D: content unchanged",
                  body.get("choices", [{}])[0].get("message", {}).get("content") == "plain response",
                  json.dumps(body.get("choices", [{}])[0].get("message", {})))
            _emit(13, "D: no canary",
                  c.extract_swarm_canary(resp) is None,
                  "canary=none")
            _emit(14, "D: no signer error",
                  "swarm_signing_failed" not in json.dumps(body),
                  "signer=clean")

    # ── Scenario E: operator visibility (/health) ─────────────────────────
    # Verifies the operator-visible "/health" endpoint reports
    # swarm_enabled + the loaded swarm-capable custom skill. No parent
    # turn needed — custom_skills_status() reads the *registry* state,
    # not a per-session active mode, so it survives without a chat.
    with tempfile.TemporaryDirectory() as d:
        _write_skill(d, max_task_calls=1)
        canned = [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_task_1",
                                    "type": "function",
                                    "function": {
                                        "name": "task",
                                        "arguments": json.dumps(
                                            {"prompt": "op prompt",
                                             "description": "research"},
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "status reply",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
        with debug_session(
            canned_responses=canned,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "h-scenario-e"
            # Parent chat installs the skill+mode so we can assert via the
            # per-agent mode helper too. (The registry-level /health will
            # show True even without this turn.)
            parent_sid = c.session_id_for(
                [{"role": "user", "content": "^^research op prompt"}],
                client_hint="h-scenario-e",
            )
            c.ensure_agent(parent_sid)
            parent_resp = c.chat(
                messages=[{"role": "user", "content": "^^research op prompt"}],
                tools=parent_tools,
                client_hint="h-scenario-e",
            )
            health_resp = c.client.get(
                "/health",
                headers={"Authorization": "Bearer test"},
            )
            _emit(15, "E: parent HTTP 200",
                  parent_resp.status_code == 200,
                  f"parent_status={parent_resp.status_code}")
            _emit(16, "E: health HTTP 200",
                  health_resp.status_code == 200,
                  f"status={health_resp.status_code}")
            health_body = health_resp.get_json(silent=True) or {}
            cs = health_body.get("custom_skills", {})
            _emit(17, "E: health.swarm_enabled True",
                  cs.get("swarm_enabled") is True,
                  f"swarm_enabled={cs.get('swarm_enabled')}")
            _emit(18, "E: health.swarm_capable has research",
                  "research" in cs.get("swarm_capable_skills", []),
                  f"skills={cs.get('swarm_capable_skills')}")
            # Per-agent active-mode verification (research mode is live
            # because parent's finish_reason is tool_calls → CONTINUE).
            parent_mode = c.inspect_agent(parent_sid)
            _emit(19, "E: parent still has active research mode",
                  parent_mode is not None and parent_mode.get("name") == "research",
                  f"mode={'research' if parent_mode is not None and parent_mode.get('name') == 'research' else parent_mode}")
            # ^^status verification: the in-session command dispatcher reports
            # Swarm-enabled: yes and the loaded swarm-capable skill.
            status_resp = c.chat(
                messages=[{"role": "user", "content": "^^status"}],
                client_hint="h-scenario-e-status",
            )
            _status_body = status_resp.json() or {}
            _status_content = _status_body.get("choices", [{}])[0].get("message", {}).get("content", "")
            _emit(20, "E: status HTTP 200", status_resp.status_code == 200, f"status={status_resp.status_code}")
            _emit(21, "E: status shows Swarm-enabled: yes", "Swarm-enabled: yes" in _status_content, "has_swarm_line=yes")
            _emit(22, "E: status shows research name", "research" in _status_content, "has_name=yes")

    _finish()


if __name__ == "__main__":
    main()
