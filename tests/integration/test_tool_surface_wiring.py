"""Integration tests: ticket-mode tool-surface wiring end-to-end.

Ticket MAIN receives read-only proxy_ticket and no proxy_work. Guard
rejections are returned to the model as role="tool" results.

These tests use ``devtools.debug_session.debug_session()`` directly (the
canonical harness that ``proxy_with_mock`` wraps) so they have access to the
DebugSession helpers (``session_id_for``, ``activate_ticket``,
``captured_requests``). They do NOT mock _select_context_v2 / apply_mode /
prepare_tooling_payload / select_tools — the point is to test the wiring.
"""
from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


def _text_response(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _tool_names_in_request(captured: list[dict], *, request_index: int) -> set[str]:
    """Extract tool function names from an explicitly selected request body."""
    assert captured, "no outbound requests captured"
    body = captured[request_index]["body"]
    names = set()
    for tool in body.get("tools", []):
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if name:
            names.add(name)
    return names


def _initial_and_requery_bodies(captured: list[dict]) -> tuple[dict, dict]:
    assert len(captured) == 2, (
        f"expected initial request plus one re-query, got {len(captured)}"
    )
    initial = captured[0]["body"]
    requery = captured[1]["body"]

    assert not any(
        message.get("role") == "tool"
        for message in initial.get("messages", [])
    )
    assert any(
        message.get("role") == "assistant"
        and message.get("tool_calls")
        for message in requery.get("messages", [])
    )
    return initial, requery


def _decoded_tool_result(requery_body: dict, *, tool_call_id: str) -> dict:
    matches = [
        message
        for message in requery_body.get("messages", [])
        if (
            message.get("role") == "tool"
            and message.get("tool_call_id") == tool_call_id
        )
    ]
    assert len(matches) == 1, (
        f"expected one role=tool result for {tool_call_id!r}, got {matches!r}"
    )

    raw_content = matches[0].get("content")
    assert isinstance(raw_content, str)
    payload = json.loads(raw_content)
    assert isinstance(payload, dict)
    return payload


class TestToolSurfaceWiring:
    """End-to-end tool-surface wiring for ticket mode."""

    def test_proxy_ticket_in_upstream_tools_in_ticket_mode(self):
        """Decomposed proxy_ticket_* read tools ARE in upstream tools[] when ticket mode is active.

        Pins the Slice-8b tool-surface wiring: TicketSkill.get_tool_filter
        exposes the read-only decomposed tools (``proxy_ticket_list_tasks``
        and ``proxy_ticket_get_task``) to MAIN. Revert-test: remove either
        tool from the ticket MAIN filter -> this fails.
        """
        from devtools.debug_session import debug_session

        with debug_session(canned_responses=[_text_response("ok")]) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "ws-sess-1"
            messages = [{"role": "user", "content": "^^ticket plan a login feature"}]
            sid = c.session_id_for(messages, client_hint="ws-sess-1")
            c.activate_ticket(sid, workspace="wstest")

            c.chat(messages=messages, client_hint="ws-sess-1")

            outbound = c.captured_requests
            names = _tool_names_in_request(outbound, request_index=0)
            assert "proxy_ticket_list_tasks" in names, (
                f"proxy_ticket_list_tasks must be in upstream tools[] in ticket mode; "
                f"got: {sorted(names)}"
            )
            assert "proxy_ticket_get_task" in names, (
                f"proxy_ticket_get_task must be in upstream tools[] in ticket mode; "
                f"got: {sorted(names)}"
            )

    def test_default_mode_does_not_get_proxy_ticket(self):
        """Default mode (no skill) does NOT include proxy_ticket or decomposed tools in tools[].

        A plain non-ticket request in a fresh session must not surface
        proxy_ticket (monolithic) or any of the 10 decomposed
        ``proxy_ticket_*`` tools to the LLM. They are PROXY_ADMIN_TOOLS
        and require skill context.
        """
        from proxy.tool_pipeline.tool_registry import _PROXY_ADMIN_TOOLS
        from devtools.debug_session import debug_session

        with debug_session(canned_responses=[_text_response("ok")]) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "plain-sess-1"
            messages = [{"role": "user", "content": "hello"}]

            c.chat(messages=messages, client_hint="plain-sess-1")

            outbound = c.captured_requests
            names = _tool_names_in_request(outbound, request_index=0)
            leaked = _PROXY_ADMIN_TOOLS & names
            assert not leaked, (
                f"Proxy admin tools must NOT be in tools[] in default mode; "
                f"got leaked: {sorted(leaked)}"
            )

    def test_main_agent_read_only_enforced_at_action_layer(self):
        """Ticket MAIN write denial and read execution use the real tool path.

        Slice 8b: ``proxy_ticket_create_tasks`` is the decomposed write
        tool. It is NOT in the ticket MAIN ``allowed_tool_names`` filter
        (TicketSkill.get_tool_filter returns only
        ``proxy_ticket_list_tasks`` + ``proxy_ticket_get_task`` for the
        MAIN role), so the upstream tool list sent to the LLM does not
        include it. If the LLM hallucinates a call anyway, the guard
        rejects it with the standard
        ``"Tool 'X' is not allowed. Active skill 'Y' restricts tools to: [...]"``
        message and the denial reaches the second captured request as a
        ``role=tool`` result. ``proxy_ticket_list_tasks`` passes the guard
        and executes the real ``ListTasksAction`` with only its Neo4j-
        facing dependencies patched.
        """
        from devtools.debug_session import debug_session

        canned_write = [
            _tool_call_response(
                "proxy_ticket_create_tasks",
                {
                    "tasks": [{"name": "X", "priority": "high"}],
                },
                call_id="call_write",
            ),
            _text_response("done"),
        ]
        with debug_session(canned_responses=canned_write) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "ro-sess-write"
            messages = [{"role": "user", "content": "^^ticket plan a login feature"}]
            sid = c.session_id_for(messages, client_hint="ro-sess-write")
            c.activate_ticket(sid, workspace="wstest")

            c.chat(messages=messages, client_hint="ro-sess-write")

            _initial, requery = _initial_and_requery_bodies(c.captured_requests)
            write_payload = _decoded_tool_result(
                requery, tool_call_id="call_write"
            )
            # Slice 8b: the standard guard denial format. The tool name
            # itself carries the action, so the message names
            # ``proxy_ticket_create_tasks`` (not the legacy
            # ``proxy_ticket(action=create_tasks)`` form).
            assert "error" in write_payload, (
                f"proxy_ticket_create_tasks from ticket MAIN must be denied; got: {write_payload}"
            )
            error_msg = write_payload["error"]
            assert "proxy_ticket_create_tasks" in error_msg, (
                f"denial must reference the offending tool name; got: {error_msg!r}"
            )
            assert "not allowed" in error_msg.lower(), (
                f"denial must say the tool is not allowed; got: {error_msg!r}"
            )
            assert "ticket" in error_msg.lower(), (
                f"denial must name the active skill; got: {error_msg!r}"
            )

        canned_read = [
            _tool_call_response(
                "proxy_ticket_list_tasks",
                {},
                call_id="call_read",
            ),
            _text_response("done"),
        ]
        mock_tg = MagicMock()
        mock_tg.get_actionable_tasks.return_value = [
            {
                "short_hash": "abc12345",
                "name": "Existing task",
                "priority": "medium",
                "status": "pending",
            }
        ]
        mock_tg.get_store_status.return_value = {"total": 1}
        mock_tg._executor = MagicMock()

        proxy_ticket_module = importlib.import_module("memory.tools.proxy_ticket")
        with patch(
            "proxy.context._get_task_graph_singleton", return_value=mock_tg
        ), patch.object(
            proxy_ticket_module,
            "_get_task_graph_singleton",
            return_value=mock_tg,
        ), patch(
            "proxy.tool_pipeline.actions.list_tasks.get_recent_review_notes",
            return_value={},
        ), debug_session(canned_responses=canned_read) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "ro-sess-read"
            messages = [{"role": "user", "content": "^^ticket plan a login feature"}]
            sid = c.session_id_for(messages, client_hint="ro-sess-read")
            c.activate_ticket(sid, workspace="wstest")

            c.chat(messages=messages, client_hint="ro-sess-read")

            _initial, requery = _initial_and_requery_bodies(c.captured_requests)
            read_payload = _decoded_tool_result(
                requery, tool_call_id="call_read"
            )
            # Slice 8b + slice-1 envelope: the action now travels in the
            # tool name (``proxy_ticket_list_tasks``) and the canonical
            # envelope carries ``operation`` + nested ``data`` (spec §11).
            assert read_payload["operation"] == "list_tasks"
            assert read_payload["status"] == "success"
            assert read_payload["ok"] is True
            data = read_payload["data"]
            assert data["total"] == 1
            assert data["tasks"][0]["short_hash"] == "abc12345"
            assert data["tasks"][0]["name"] == "Existing task"
            assert data["store_status"] == {"total": 1}
            assert "error" not in read_payload or read_payload["error"] is None
            mock_tg.get_actionable_tasks.assert_called_once()

    def test_workspace_isolation_no_real_neo4j(self):
        """workspace_store.activate does NOT reach real Neo4j during ticket flow.

        Asserts ``workspace_store._persist is None`` after the run (the
        ``_isolate_workspace_neo4j`` autouse fixture sets it to None) AND that
        ``workspace_store._active[session_id] == "wstest"`` (proves activate
        actually ran in-memory). Does NOT mock ``workspace_store.activate``
        itself (that would be tautological for this invariant).
        """
        import proxy.state as state
        from devtools.debug_session import debug_session

        with debug_session(canned_responses=[_text_response("ok")]) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "ws-sess-2"
            messages = [{"role": "user", "content": "^^ticket plan a login feature"}]
            sid = c.session_id_for(messages, client_hint="ws-sess-2")
            c.activate_ticket(sid, workspace="wstest")

            c.chat(messages=messages, client_hint="ws-sess-2")

            assert state.workspace_store._persist is None, (
                "_isolate_workspace_neo4j must keep _persist=None so activate "
                "does not reach real Neo4j"
            )
            assert state.workspace_store._active.get(sid) == "wstest", (
                f"activate must have set _active[sid]='wstest'; got: "
                f"{state.workspace_store._active.get(sid)!r}"
            )

    def test_ticket_main_mode_allowlist_rejects_hallucinated_proxy_work(self):
        """Ticket MAIN rejects a hallucinated proxy_work call at the first guard.

        Ticket MAIN's initial tool surface excludes ``proxy_work``. If the
        provider hallucinates it anyway, ``_check_mode_allowlist()`` rejects
        the call and the denial reaches the re-query as a role=tool result.
        This normal path does not reach ``_check_ticket_mode_orchestrator()``.
        """
        from devtools.debug_session import debug_session

        canned = [
            _tool_call_response(
                "proxy_work",
                {"action": "start_work", "short_hash": "abc12345"},
            ),
            _text_response("done"),
        ]
        with debug_session(canned_responses=canned) as c:
            c.client.environ_base["HTTP_X_SESSION_CLIENT"] = "ws-sess-3"
            messages = [{"role": "user", "content": "^^ticket plan a login feature"}]
            sid = c.session_id_for(messages, client_hint="ws-sess-3")
            c.activate_ticket(sid, workspace="wstest")

            c.chat(messages=messages, client_hint="ws-sess-3")

            initial, requery = _initial_and_requery_bodies(c.captured_requests)

            initial_names = {
                tool.get("function", {}).get("name")
                for tool in initial.get("tools", [])
                if isinstance(tool, dict)
            }
            assert "proxy_work" not in initial_names
            # Slice 8b: ticket MAIN sees the decomposed read tools, not
            # the monolithic ``proxy_ticket``.
            assert "proxy_ticket" not in initial_names
            assert "proxy_ticket_list_tasks" in initial_names
            assert "proxy_ticket_get_task" in initial_names

            work_payload = _decoded_tool_result(requery, tool_call_id="call_1")
            assert set(work_payload) == {"error"}
            error = work_payload["error"]
            assert "proxy_work" in error
            assert "not allowed" in error.lower()
            assert "ticket" in error.lower()
            assert "restricts tools to" in error.lower()
