#!/usr/bin/env python3
"""Template G — custom research swarm parent/child happy path.

Exercises the full custom-skill research-swarm contract end-to-end through
``debug_session`` with a real custom JSON skill loaded on the fly:

1. **Parent** — ``^^research`` fires the swarm-capable custom skill; the
   upstream emits a native ``task`` call; the proxy post-process injects a
   signed SKILL canary for ``research`` into the prompt.
2. **Child** — ``send_as_subagent`` embeds the extracted canary marker at
   position-0; the proxy strips the canary, activates research mode for the
   child session, and forwards only the child tool whitelist (read + grep —
   NOT task / question / proxy_ticket / proxy_work).

Research skill JSON (written to a fres
h ``tempfile.TemporaryDirectory``) —
``skills_dir=`` + ``enable_custom_skill_subagents=True`` wires the runtime
registry via the new ``debug_session`` args; custom skills activate via the
real ``^^<name>`` trigger (no activate_ticket / activate_work needed).

Detached transform atomicity is pinned in
tests/proxy/test_swarm_post_process.py. This external HTTP scenario cannot
observe object identity inside PostProcess.

Run with:
    PYTHONPATH=src PROXY_SESSION_SECRET="" python3 tests/debug/templates/g_swarm_parent_child.py
"""
import sys
import os
import json
import copy
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


def _question_tool():
    return {
        "type": "function",
        "function": {
            "name": "question",
            "description": "Ask the operator a question",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _proxy_ticket_tool():
    return {
        "type": "function",
        "function": {
            "name": "proxy_ticket",
            "description": "Create/list tasks (swarm-PERMANENTLY-DENIED for custom)",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _proxy_work_tool():
    return {
        "type": "function",
        "function": {
            "name": "proxy_work",
            "description": "Execute a task (swarm-PERMANENTLY-DENIED for custom)",
            "parameters": {"type": "object", "properties": {}},
        },
    }


PARENT_TOOLS = [
    _read_tool(),
    _grep_tool(),
    _task_tool(),
    _question_tool(),
    _proxy_ticket_tool(),
    _proxy_work_tool(),
]


def main():
    secret = SECRET
    with tempfile.TemporaryDirectory() as d:
        research_json = {
            "name": "research",
            "trigger": "^^research",
            "description": "Debug-session research swarm scenario.",
            "prompt": "RESEARCH_SWARM_DEBUG_PROMPT",
            "tool_whitelist": ["read", "grep", "task"],
            "subagents": {
                "tool_whitelist": ["read", "grep"],
                "max_task_calls_per_request": 2,
            },
        }
        with open(os.path.join(d, "research.json"), "w") as f:
            json.dump(research_json, f)

        # Parent emits one native task call; child is content-only.
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
                                            {
                                                "prompt": "inspect authentication handling",
                                                "description": "research",
                                            },
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
                            "content": "sub-agent done",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ]

        # Snapshot the original canned[0] BEFORE passing it to the proxy so
        # we can assert post-session that the mock canned fixture is isolated
        # across HTTP serialization (the HTTP layer decodes a fresh object).
        source_snapshot = copy.deepcopy(canned[0])

        with debug_session(
            canned_responses=canned,
            secret=secret,
            skills_dir=d,
            enable_custom_skill_subagents=True,
        ) as c:
            # ── Parent turn ───────────────────────────────────────────────
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "swarm-parent"
            parent_messages = [
                {"role": "user", "content": "^^research inspect authentication handling"}
            ]
            parent_sid = c.session_id_for(parent_messages, client_hint="swarm-parent")
            # Pre-seed agent so first-request skill activation finds the agent.
            c.ensure_agent(parent_sid)
            parent_resp = c.chat(
                messages=parent_messages,
                tools=PARENT_TOOLS,
                client_hint="swarm-parent",
            )
            _emit(1, "parent HTTP 200", parent_resp.status_code == 200,
                  f"status={parent_resp.status_code}")

            parent_prompts = c.extract_task_prompts(parent_resp)
            _emit(2, "exactly one native task call",
                  len(parent_prompts) == 1,
                  f"count={len(parent_prompts)}")

            canary = c.extract_swarm_canary(parent_resp)
            _emit(3, "prompt begins with a canary marker",
                  canary is not None and canary.startswith("[[__SUBAGENT__:SKILL:"),
                  f"canary={'yes' if canary else 'none'}")

            if canary is not None:
                from proxy.canary import parse_canary
                parsed = parse_canary(canary)
                _emit(4, "parse_canary kind=skill name=research verified=True",
                      parsed is not None
                      and parsed.kind == "skill"
                      and parsed.skill_name == "research"
                      and parsed.verified,
                      f"parsed={parsed!r}")
            else:
                _emit(4, "parse_canary kind=skill name=research verified=False",
                      False, "no canary")

            if canary is not None and len(parent_prompts) == 1:
                full_prompt = parent_prompts[0]
                remainder = full_prompt[len(canary):]
                _emit(5, "prompt remainder equals original child prompt exactly once",
                      remainder == "inspect authentication handling",
                      f"remainder={remainder!r}")
            else:
                _emit(5, "prompt remainder check (skipped — missing canary/task)",
                      False, "precondition failed")

            # Parent source response (mock) must be byte-for-byte what we
            # queued — canary only appears in the proxy response, never in
            # the original canned dict.
            _emit(6, "mock canned fixture remains unchanged across HTTP serialization",
                  canned[0] == source_snapshot,
                  f"fixture_unchanged={canned[0] == source_snapshot}")

            # ── Child turn ────────────────────────────────────────────────
            if canary is None:
                _emit(7, "child HTTP 200 (skipped)", False, "no canary")
                _emit(8, "child inherited research (skipped)", False, "no canary")
                _emit(9, "child system prompt (skipped)", False, "no canary")
                _emit(10, "child tools read+grep (skipped)", False, "no canary")
                _emit(11, "child tools no task/question/proxy_* (skipped)", False,
                      "no canary")
                _emit(12, "canary stripped from user prompt (skipped)", False,
                      "no canary")
                _emit(13, "original child instruction once (skipped)", False,
                      "no canary")
                _finish()
                return

            # Derive the child's session_id the same way the proxy does so a
            # possible mode pre-seed (not required) lines up.
            child_messages = [
                {"role": "user", "content": f"{canary}\ninspect authentication handling"}
            ]
            child_sid = c.session_id_for(child_messages, client_hint="swarm-child")

            # The child request carries the parent's full tool set; the
            # proxy's `select_tools` filters it to EXACTLY the subagent
            # whitelist (read + grep). Passing tools here is what lets us
            # assert the filter worked on the outbound request.
            child_resp, returned_sid = c.send_as_subagent(
                canary=canary,
                subagent_prompt="inspect authentication handling",
                subagent_messages=[],
                client_hint="swarm-child",
                tools=PARENT_TOOLS,
            )
            _emit(7, "child HTTP 200", child_resp.status_code == 200,
                  f"status={child_resp.status_code}")

            # 8: inherited mode — may be None after terminal stop. Treat as
            # diagnostic-only detail, not a failure.
            child_mode = c.inspect_agent(returned_sid)
            if child_mode is not None and child_mode.get("name") == "research":
                _emit(8, "child inherited research mode (post-terminal)",
                      True, f"mode={child_mode['name']}")
            else:
                _emit(8, "child inherited research mode (post-terminal) "
                      "[diagnostic — mode cleared by stop]",
                      True,
                      f"mode={'cleared' if child_mode is None else child_mode}")

            # 9–13: OUTBOUND assertions (authoritative — these survive a
            # stop-disposition clear because they target the request the
            # proxy sent to upstream).
            if len(c.captured_requests) >= 2:
                outbound_body = c.captured_requests[1]["body"]
                sys_messages = [m for m in outbound_body.get("messages", [])
                                 if m.get("role") == "system"]
                sys_text = " ".join(
                    m.get("content", "") for m in sys_messages
                    if isinstance(m.get("content", ""), str)
                )
                user_messages = [m for m in outbound_body.get("messages", [])
                                 if m.get("role") == "user"]
                last_user = user_messages[-1] if user_messages else {}
                user_text = last_user.get("content", "")

                _emit(9, "outbound contains RESEARCH_SWARM_DEBUG_PROMPT",
                      "RESEARCH_SWARM_DEBUG_PROMPT" in sys_text,
                      f"sys_count={len(sys_messages)}")

                raw_tools = outbound_body.get("tools", []) or []
                outbound_names = sorted(
                    t.get("function", {}).get("name", "")
                    for t in raw_tools
                    if isinstance(t, dict) and isinstance(t.get("function"), dict)
                )
                _emit(10, "outbound child tools EXACTLY read + grep",
                      outbound_names == ["grep", "read"],
                      f"tools={outbound_names}")
                _emit(11, "outbound child tools exclude task/question/proxyTicket/proxyWork",
                      not any(n in outbound_names
                              for n in ("task", "question", "proxy_ticket", "proxy_work")),
                      f"tools={outbound_names}")

                _emit(12, "canary stripped from outbound user prompt",
                      isinstance(user_text, str)
                      and "[[__SUBAGENT__" not in user_text,
                      f"user_preview={user_text[:80]!r}")
                _emit(13, "original child instruction appears exactly once in outbound",
                      isinstance(user_text, str)
                      and user_text.count("inspect authentication handling") == 1,
                      f"count={user_text.count('inspect authentication handling') if isinstance(user_text, str) else 'n/a'}")
            else:
                _emit(9, "skipped — outbound not found", False, "captured<2")
                _emit(10, "skipped — outbound not found", False, "captured<2")
                _emit(11, "skipped — outbound not found", False, "captured<2")
                _emit(12, "skipped — outbound not found", False, "captured<2")
                _emit(13, "skipped — outbound not found", False, "captured<2")

    _finish()


if __name__ == "__main__":
    main()
