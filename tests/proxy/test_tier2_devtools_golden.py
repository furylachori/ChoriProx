"""§9.8 — DevTools golden: subfolder read → pin in assembled context, for
BOTH main and sub-agent (incl. overflow).

E2E proof that the full pipeline (capture → pin → strip) wires end-to-end:
  1. Send a tool result whose content carries the Tier-2 suffix
     (</content><system-reminder>Instructions from: ...).
  2. The proxy's _capture_tier2_instructions records the blocks into
     instruction_files_by_workspace.
  3. _emit_tier2_pin inserts a <CLIENT_SUB_INSTRUCTIONS> system message
     into the outbound message list sent to upstream.
  4. We assert the captured outbound request body contains the pin.

Uses debug_session (the canonical pytest-free harness) since proxy_with_mock
lives in tests/integration/conftest.py. The debug_session context manager
provides the same lifecycle: mock backend + Flask test client + state reset.
"""
import json

import pytest

import proxy.state as state
from tests.fixtures.tier2_payload import RAW_TOOL_CONTENT


_PIN_MARKER = '<CLIENT_SUB_INSTRUCTIONS scope="subfolder-convention">'


@pytest.fixture(autouse=True)
def _clean_tier2_state():
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()
    yield
    with state.vfs_lock:
        state.instruction_files_by_workspace.clear()
        state._last_scanned_by_tool_call.clear()


def _build_tool_result_message(call_id="call_tier2_read") -> dict:
    """Build a role=tool message carrying the canonical Tier-2 suffix."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": RAW_TOOL_CONTENT,
    }


def _build_assistant_tool_call(call_id="call_tier2_read") -> dict:
    """Build an assistant message with a read tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps(
                        {"filePath": "/repo/project/src/proxy/sub/route.py"}
                    ),
                },
            }
        ],
    }


def _assert_pin_in_messages(messages: list, label: str) -> None:
    """Assert the Tier-2 pin is present in the outbound message list."""
    assert messages, f"{label}: no outbound messages captured"

    pin_msgs = [
        m for m in messages
        if m.get("role") == "system" and _PIN_MARKER in str(m.get("content", ""))
    ]
    assert len(pin_msgs) >= 1, (
        f"{label}: expected at least one {_PIN_MARKER!r} system message, "
        f"got {len(pin_msgs)}. Messages: "
        f"{[(m.get('role'), str(m.get('content', ''))[:80]) for m in messages]}"
    )

    pin_content = pin_msgs[0]["content"]
    assert "Instructions from:" in pin_content, (
        f"{label}: pin content missing 'Instructions from:' header"
    )
    assert "sub/AGENTS.md" in pin_content, (
        f"{label}: pin content missing sub-AGENTS.md path reference"
    )


@pytest.mark.integration
class TestTier2DevToolsGolden:
    """E2E: Tier-2 subfolder read → pin in outbound context."""

    def test_main_agent_pin_in_outbound(self):
        """Main agent: a read tool result with Tier-2 suffix → pin in outbound."""
        from devtools.debug_session import debug_session

        canned = [
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ]

        with debug_session(canned_responses=canned) as c:
            call_id = "call_main_1"
            messages = [
                {"role": "system", "content": "operational directives"},
                {"role": "user", "content": "read src/proxy/sub/route.py"},
                _build_assistant_tool_call(call_id),
                _build_tool_result_message(call_id),
                {"role": "user", "content": "now summarize"},
            ]

            resp = c.chat(messages=messages, model="kimi")
            assert resp.status_code == 200, f"unexpected status: {resp.status_code}"

            dump = c.dump_captured()
            outbound = dump.get("outbound", [])
            assert outbound, "no outbound request captured by mock backend"

            last_body = outbound[-1].get("body", {})
            outbound_msgs = last_body.get("messages", [])

            _assert_pin_in_messages(outbound_msgs, "main-agent")

    def test_main_agent_tool_result_stripped(self):
        """The Tier-2 suffix is stripped from tool results in the outbound."""
        from devtools.debug_session import debug_session

        canned = [
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ]

        with debug_session(canned_responses=canned) as c:
            call_id = "call_strip_1"
            messages = [
                {"role": "system", "content": "operational directives"},
                {"role": "user", "content": "read the file"},
                _build_assistant_tool_call(call_id),
                _build_tool_result_message(call_id),
            ]

            resp = c.chat(messages=messages, model="kimi")
            assert resp.status_code == 200

            dump = c.dump_captured()
            outbound = dump.get("outbound", [])
            assert outbound, "no outbound request captured"

            last_body = outbound[-1].get("body", {})
            outbound_msgs = last_body.get("messages", [])

            for m in outbound_msgs:
                if m.get("role") == "tool":
                    content = m.get("content", "")
                    assert "<system-reminder>" not in content, (
                        "Tier-2 suffix was NOT stripped from outbound tool result"
                    )
                    assert "Instructions from:" not in content, (
                        "Tier-2 instruction blocks leaked into outbound tool result"
                    )

    def test_subagent_pin_in_outbound(self, mock_agent_registry_get_or_load):
        """Sub-agent: Tier-2 captured on parent turn → pin in sub-agent outbound.

        Uses a non-default workspace so the sub-agent resolves to the same
        workspace as the parent. The Tier-2 data is keyed by workspace, so
        the sub-agent finds it via resolve_inherited_workspace.
        """
        from devtools.debug_session import debug_session
        from proxy.model_registry import resolve_session_id

        canned = [
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "sub done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]

        with debug_session(canned_responses=canned) as c:
            from proxy.agent.agent import Agent

            # Compute the content-derived session_id for the parent
            parent_call_id = "call_parent_cap"
            parent_messages = [
                {"role": "system", "content": "operational directives"},
                {"role": "user", "content": "read src/proxy/sub/route.py"},
                _build_assistant_tool_call(parent_call_id),
                _build_tool_result_message(parent_call_id),
            ]
            parent_sid = resolve_session_id(
                {"messages": parent_messages}, "parent-golden"
            )

            # Register parent agent with a non-default workspace
            parent_agent = Agent(session_id=parent_sid, agent_id=parent_sid)
            state.workspace_store.activate(parent_sid, "ws-tier2-test")
            mock_agent_registry_get_or_load[parent_sid] = parent_agent

            # ── Phase 1: parent turn captures Tier-2 ──
            resp1 = c.chat(
                messages=parent_messages, model="kimi",
                client_hint="parent-golden",
            )
            assert resp1.status_code == 200

            # Verify the map was populated under the non-default workspace
            with state.vfs_lock:
                assert len(state.instruction_files_by_workspace) > 0, (
                    "Tier-2 capture did not populate instruction_files_by_workspace"
                )
                assert "ws-tier2-test" in state.instruction_files_by_workspace, (
                    f"expected 'ws-tier2-test' key, got: "
                    f"{list(state.instruction_files_by_workspace.keys())}"
                )

            # ── Phase 2: sub-agent turn inherits workspace → pin emits ──
            # Compute the sub-agent's content-derived session_id from the
            # FULL content (including the canary). The pipeline computes
            # the session_id in parse_request BEFORE detect_subagent_ingress
            # strips the canary.
            canary = c.sign_canary("ticket")
            sub_prompt = "now summarize the route logic"
            sub_user_content = f"{canary}\n{sub_prompt}"
            sub_sid = resolve_session_id(
                {"messages": [{"role": "user", "content": sub_user_content}]},
                "sub-golden",
            )
            sub_agent = Agent(
                session_id=sub_sid, agent_id=sub_sid,
                parent=parent_agent, is_subagent=True,
            )
            mock_agent_registry_get_or_load[sub_sid] = sub_agent
            # NOTE: Do NOT activate the workspace on the sub-agent session.
            # The sub-agent must INHERIT the workspace via the parent chain
            # (resolve_inherited_workspace walks agent.parent_agent_id).
            # Manually activating the same workspace on the sub-agent was
            # the old masking pattern that hid F1.

            resp2 = c.client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": sub_user_content}],
                    "model": "kimi",
                    "stream": False,
                },
                headers={"X-Session-Client": "sub-golden"},
            )
            assert resp2.status_code == 200, f"unexpected status: {resp2.status_code}"

            dump = c.dump_captured()
            outbound = dump.get("outbound", [])
            assert len(outbound) >= 2, "expected at least 2 outbound requests"

            last_body = outbound[-1].get("body", {})
            outbound_msgs = last_body.get("messages", [])

            _assert_pin_in_messages(outbound_msgs, "sub-agent")

    def test_subagent_overflow_pin_in_outbound(self, mock_agent_registry_get_or_load):
        """Sub-agent with overflow (large messages): pin still emits.

        Registers a parent agent with a non-default workspace keyed by the
        content-derived session_id, then drives the sub-agent with messages
        that overflow the safe budget (_build_subagent_context overflow
        paths :335/:361/:369). The pin MUST still be present.
        """
        from devtools.debug_session import debug_session
        from proxy.model_registry import resolve_session_id

        canned = [
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "sub done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]

        with debug_session(canned_responses=canned) as c:
            from proxy.agent.agent import Agent

            # Compute the content-derived session_id for the parent
            parent_call_id = "call_parent_overflow"
            parent_messages = [
                {"role": "system", "content": "operational directives"},
                {"role": "user", "content": "read src/proxy/sub/route.py"},
                _build_assistant_tool_call(parent_call_id),
                _build_tool_result_message(parent_call_id),
            ]
            parent_sid = resolve_session_id(
                {"messages": parent_messages}, "parent-overflow"
            )

            # Register parent agent with a non-default workspace
            parent_agent = Agent(session_id=parent_sid, agent_id=parent_sid)
            state.workspace_store.activate(parent_sid, "ws-overflow-test")
            mock_agent_registry_get_or_load[parent_sid] = parent_agent

            # ── Phase 1: capture Tier-2 ──
            resp1 = c.chat(
                messages=parent_messages, model="kimi",
                client_hint="parent-overflow",
            )
            assert resp1.status_code == 200

            with state.vfs_lock:
                assert len(state.instruction_files_by_workspace) > 0
                assert "ws-overflow-test" in state.instruction_files_by_workspace

            # ── Phase 2: sub-agent with overflow messages ──
            # Compute the sub-agent's content-derived session_id and
            # pre-register it with a parent link BEFORE sending.
            canary = c.sign_canary("ticket")
            _NON_SYSTEM_BUDGET_CHARS = 88000
            big_content = "B" * (_NON_SYSTEM_BUDGET_CHARS + 5000)
            sub_user_content = f"{canary}\nprocess this large assignment"
            # Compute session_id from ALL user messages (up to 10),
            # matching what resolve_session_id does in the pipeline.
            sub_sid = resolve_session_id(
                {"messages": [
                    {"role": "user", "content": sub_user_content},
                    {"role": "user", "content": "please complete this assignment"},
                ]},
                "sub-overflow",
            )
            sub_agent = Agent(
                session_id=sub_sid, agent_id=sub_sid,
                parent=parent_agent, is_subagent=True,
            )
            mock_agent_registry_get_or_load[sub_sid] = sub_agent
            # NOTE: Do NOT activate the workspace on the sub-agent session.
            # The sub-agent must INHERIT via the parent chain (F1 fix).

            resp2 = c.client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": sub_user_content},
                        {"role": "user", "content": "please complete this assignment"},
                        {"role": "assistant", "content": big_content},
                    ],
                    "model": "kimi",
                    "stream": False,
                },
                headers={"X-Session-Client": "sub-overflow"},
            )
            assert resp2.status_code == 200, f"unexpected status: {resp2.status_code}"

            dump = c.dump_captured()
            outbound = dump.get("outbound", [])
            assert len(outbound) >= 2

            last_body = outbound[-1].get("body", {})
            outbound_msgs = last_body.get("messages", [])

            _assert_pin_in_messages(outbound_msgs, "sub-agent-overflow")
