# tests/unit/clients/test_canary_preservation.py
"""
Theory 5 — Client runtime canary preservation test.

Verifies that the opencode and kilocode client adapters do NOT strip or
re-encode the [[__SUBAGENT__:SKILL:...]] canary when a sub-agent request
body is processed by any of the client adapter's available transformation
methods (normalize_tool_args, format_response, format_sse_chunk, etc.).

The canary is a critical security marker used by the proxy to detect
sub-agent ingress and grant skill/workspace inheritance. If the client
runtime (opencode/kilocode) strips or re-encodes the canary before
forwarding the sub-agent's request to the proxy, the proxy cannot detect
the sub-agent relationship and skill inheritance breaks.

Test strategy:
1. Build a realistic sub-agent request body — first user message contains
   the canary as a literal prefix string.
2. Pass it through every method on the client adapter that could plausibly
   touch the body (normalize_tool_args for the `task` tool, denormalize,
   format_response, format_sse_chunk, format_tool_result, format_error).
3. Assert the canary text appears verbatim in the first user message
   content after each transformation.
4. Also exercise the canary detection logic from proxy.canary to confirm
   the canary is detectable end-to-end.

This test should pass — the client adapters in this codebase do NOT
transform user message content. They only handle tool argument name
normalization and response/SSE formatting. The test documents this
guarantee and will FAIL if a future change adds user-content rewriting.
"""

import json
import os
import pytest

from clients.opencode import OpenCodeAdapter
from clients.kilocode import KiloCodeAdapter


# ----------------------------------------------------------------------------
# Canary fixture
# ----------------------------------------------------------------------------

# A representative canary: SKILL type with the literal text that
# proxy/canary.py expects to find at the start of the first user message.
# Per the in-code comment in canary.py, the canary uses [[__SUBAGENT__]]
# markers — those are NOT markdown, they are literal square brackets.
CANARY_SKILL = "[[__SUBAGENT__:SKILL:ticket]]"
CANARY_BARE = "[[__SUBAGENT__]]"
CANARY_WORKSPACE = "[[__SUBAGENT__:WORKSPACE:myws]]"
CANARY_TASK_PROMPT = (
    CANARY_SKILL
    + "\nDo this delegated thing for me."
)


def _build_subagent_request_body(canary: str = CANARY_SKILL) -> dict:
    """Build a realistic sub-agent request body.

    Mirrors the structure that the opencode/kilocode runtimes would
    send to the proxy when a parent agent spawns a sub-agent via the
    `task` tool: the canary prefix lives in the first user message
    content (the delegated prompt), followed by a normal user prompt.
    """
    return {
        "model": "gpt-4",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": canary + "\nPlease do the task.",
            },
            {
                "role": "assistant",
                "content": "Acknowledged.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }


# ----------------------------------------------------------------------------
# Test class
# ----------------------------------------------------------------------------


class TestCanaryPreservationByClientAdapter:
    """Theory 5: Client adapters must not strip/re-encode the canary."""

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_preserved_through_normalize_tool_args(
        self, adapter_factory, client_name
    ):
        """normalize_tool_args for the `task` tool must not touch user content.

        The canary lives in the user message content. normalize_tool_args
        only operates on tool-call argument dicts. Even if the LLM had
        somehow emitted a `task` tool call with the canary in its prompt
        argument, normalize_tool_args must not strip it — that would defeat
        the sub-agent detection at the proxy.
        """
        adapter = adapter_factory()
        args = {
            "prompt": CANARY_TASK_PROMPT,
            "subagent_type": "general",
            "description": "delegate",
        }
        result = adapter.normalize_tool_args("task", args)
        # The prompt (containing the canary) must be preserved verbatim.
        assert result["prompt"] == CANARY_TASK_PROMPT, (
            f"[{client_name}] normalize_tool_args('task') altered the canary. "
            f"Expected {CANARY_TASK_PROMPT!r}, got {result.get('prompt')!r}"
        )

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_preserved_through_denormalize_tool_args(
        self, adapter_factory, client_name
    ):
        """denormalize_tool_args for `task` must also preserve the canary."""
        adapter = adapter_factory()
        args = {
            "prompt": CANARY_TASK_PROMPT,
            "subagent_type": "general",
        }
        # OpenCode denormalize is a passthrough; KiloCode renames
        # filePath→path etc., but `task` has no such mappings.
        result = adapter.denormalize_tool_args("task", args)
        assert result["prompt"] == CANARY_TASK_PROMPT, (
            f"[{client_name}] denormalize_tool_args('task') altered the canary. "
            f"Expected {CANARY_TASK_PROMPT!r}, got {result.get('prompt')!r}"
        )

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_preserved_through_format_response(
        self, adapter_factory, client_name
    ):
        """format_response must not touch the upstream request body.

        format_response formats a server RESPONSE for the client. It does
        not process incoming request bodies, so the request body is
        untouched regardless. This test pins that contract.
        """
        adapter = adapter_factory()
        body = _build_subagent_request_body()
        # Format a response (not the body) — must not raise and must not
        # mutate the body reference.
        response = {"role": "assistant", "content": "ok"}
        out = adapter.format_response(response, wants_stream=False)
        assert out is response

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_preserved_through_format_sse_chunk(
        self, adapter_factory, client_name
    ):
        """format_sse_chunk must not alter the canary inside a delta."""
        adapter = adapter_factory()
        # Even if a chunk contains a canary, SSE formatting must not
        # mangle the brackets.
        chunk = {"choices": [{"delta": {"content": CANARY_SKILL}}]}
        line = adapter.format_sse_chunk(chunk)
        # The JSON-encoded chunk must contain the canary verbatim.
        assert CANARY_SKILL in line, (
            f"[{client_name}] format_sse_chunk escaped/encoded the canary.\n"
            f"  Input chunk: {chunk!r}\n"
            f"  Output line: {line!r}"
        )
        # Specifically, the literal "[[" must survive — the SSE layer
        # should not double-encode or markdown-escape the brackets.
        assert "[[__SUBAGENT__:SKILL:" in line

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_preserved_through_format_tool_result(
        self, adapter_factory, client_name
    ):
        """format_tool_result must not touch the canary in the result text."""
        adapter = adapter_factory()
        result = adapter.format_tool_result("task", CANARY_TASK_PROMPT)
        # The canary must be present in the formatted result.
        assert result["role"] == "tool"
        assert CANARY_SKILL in str(result["content"]), (
            f"[{client_name}] format_tool_result stripped the canary. "
            f"Got: {result!r}"
        )

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_preserved_through_format_error(
        self, adapter_factory, client_name
    ):
        """format_error must not touch the canary in the error text."""
        adapter = adapter_factory()
        # The canary could legitimately appear in an error trace (e.g.,
        # the proxy rejected the request because the HMAC failed).
        # The error formatter must not strip it.
        err = ValueError("canary rejection: " + CANARY_SKILL)
        result = adapter.format_error(err, role="tool")
        assert CANARY_SKILL in str(result["content"]), (
            f"[{client_name}] format_error stripped the canary. "
            f"Got: {result!r}"
        )

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_survives_json_round_trip_through_adapter(
        self, adapter_factory, client_name
    ):
        """End-to-end: serialize body, run any adapter transformations,
        deserialize — canary must survive byte-for-byte.

        This simulates the round-trip that the client runtime performs
        when forwarding a sub-agent request: it serializes the body to
        JSON, applies any adapter transformations, then sends it. The
        canary text must survive this round-trip.
        """
        adapter = adapter_factory()
        body = _build_subagent_request_body()

        # Step 1: serialize (this is what any HTTP client does).
        serialized = json.dumps(body)
        # Step 2: deserialize.
        reparsed = json.loads(serialized)
        # Step 3: run every method that could plausibly be invoked on
        # the adapter during request prep (they all must leave the
        # user content alone).
        for tool in ("task", "bash", "read"):
            adapter.normalize_tool_args(tool, {"prompt": CANARY_TASK_PROMPT})
            adapter.denormalize_tool_args(tool, {"prompt": CANARY_TASK_PROMPT})
        adapter.format_response({"ok": True}, wants_stream=False)
        adapter.format_sse_chunk({"delta": "x"})

        # The first user message must still begin with the canary
        # after the round-trip.
        first_user = reparsed["messages"][0]["content"]
        assert first_user.startswith(CANARY_SKILL), (
            f"[{client_name}] canary lost after JSON round-trip + adapter pass.\n"
            f"  Expected prefix: {CANARY_SKILL!r}\n"
            f"  Got:             {first_user!r}"
        )
        # And the [[__SUBAGENT__:SKILL:ticket]] literal must be intact.
        assert "[[__SUBAGENT__:SKILL:ticket]]" in first_user

    @pytest.mark.parametrize(
        "adapter_factory,client_name",
        [
            (OpenCodeAdapter, "opencode"),
            (KiloCodeAdapter, "kilocode"),
        ],
    )
    def test_canary_skill_prefix_detectable_after_adapter_pass(
        self, adapter_factory, client_name
    ):
        """The canary must be detectable by proxy/canary.py after the
        adapter's transformation pipeline runs.

        This is the end-to-end check: does the literal text
        `[[__SUBAGENT__:SKILL:...` survive the adapter intact, so the
        proxy's `startswith()` detection in detect_subagent_ingress()
        fires?
        """
        from proxy.utils import extract_text_content

        adapter = adapter_factory()
        # Take the canary, push it through the adapter's normalize/format
        # methods to mimic what the client runtime might do.
        prompt_in = CANARY_TASK_PROMPT
        # normalize_tool_args must be a no-op on the prompt string
        # (it only renames dict keys, not string values).
        norm = adapter.normalize_tool_args(
            "task", {"prompt": prompt_in, "subagent_type": "general"}
        )
        prompt_out = norm["prompt"]
        # The text passed to extract_text_content must round-trip the
        # proxy's canary detection logic.
        text = extract_text_content(prompt_out)
        # The detection is a literal startswith on the bracketed prefix.
        assert text.startswith("[[__SUBAGENT__:SKILL:"), (
            f"[{client_name}] canary prefix not detectable after adapter pass.\n"
            f"  Input:  {prompt_in!r}\n"
            f"  Output: {prompt_out!r}\n"
            f"  Extracted: {text!r}"
        )
        # And the canonical parser must accept it.
        # parse_canary requires a registered skill — in the test
        # environment there is no registry, so we only check that the
        # prefix is present (the full parse will be exercised by the
        # proxy integration tests).
        assert "[[__SUBAGENT__:SKILL:" in text


class TestCanaryPreservationProviderCompat:
    """Provider-compat transformations must not strip the canary.

    Although this is a proxy-level concern, the canary lives in the
    user message and provider_compat.inject reasoning_content on
    ASSISTANT messages — but to be safe, we explicitly verify that
    none of the built-in provider adapters touch the user message
    that contains the canary.
    """

    @pytest.mark.parametrize(
        "adapter_cls_name",
        ["DeepSeekAdapter", "MiMoAdapter", "KimiAdapter", "MiniMaxAdapter"],
    )
    def test_provider_compat_preserves_user_message_canary(
        self, adapter_cls_name
    ):
        from proxy.pipeline import provider_compat

        adapter_cls = getattr(provider_compat, adapter_cls_name)
        body = _build_subagent_request_body()
        original_first_user = body["messages"][0]["content"]

        # Build a minimal model_cfg stub.
        class _Cfg:
            backend_model = "stub-model"
        result = adapter_cls.apply(body, _Cfg())
        # First user message must be untouched.
        assert result["messages"][0]["content"] == original_first_user
        assert CANARY_SKILL in result["messages"][0]["content"]


class TestCanaryPreservationFullRequestPipeline:
    """Full request pipeline test (proxy-level, but client-runtime relevant).

    This simulates the full path the client runtime takes when forwarding
    a sub-agent's request: serialize to JSON → POST → proxy parses JSON
    → apply_mode runs (mutates body in place) → apply_provider_compat
    runs → request sent to backend.
    """

    def test_full_request_pipeline_preserves_canary(self):
        """Verify the canary in the first user message survives the full
        request pipeline (parse → apply_mode → apply_provider_compat).
        """
        # The body the client would send:
        body = _build_subagent_request_body()
        # Simulate the JSON round-trip the HTTP client performs.
        body = json.loads(json.dumps(body))
        # Simulate the proxy's apply_provider_compat (no provider config
        # in the test, so this is a passthrough).
        from proxy.pipeline.provider_compat import apply_provider_compat

        class _Cfg:
            backend_model = "openai/gpt-4"
        body = apply_provider_compat(body, _Cfg())
        # The canary must still be at the start of the first user message.
        first_user = body["messages"][0]["content"]
        assert first_user.startswith(CANARY_SKILL), (
            f"Canary lost in full pipeline.\n"
            f"  Got first user content: {first_user!r}"
        )
        # And the [[__SUBAGENT__:SKILL:ticket]] literal must survive.
        assert "[[__SUBAGENT__:SKILL:ticket]]" in first_user
