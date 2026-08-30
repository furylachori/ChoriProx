"""Unit tests for proxy.pipeline.provider_compat.

Covers the 4 built-in adapters (DeepSeek, MiMo, Kimi, MiniMax) and the
`apply_provider_compat` dispatch function. These adapters were previously
zero-tested.
"""
import copy
from unittest.mock import MagicMock

from proxy.pipeline.provider_compat import (
    apply_provider_compat,
    DeepSeekAdapter,
    KimiAdapter,
    MiMoAdapter,
    MiniMaxAdapter,
)


def _make_model_cfg(backend_model: str) -> MagicMock:
    cfg = MagicMock()
    cfg.backend_model = backend_model
    return cfg


class TestApplyProviderCompat:
    """Tests for the apply_provider_compat dispatcher."""

    def test_apply_provider_compat_kimi_clamps_temperature(self):
        """US-032: Kimi Temperature Clamping — apply_provider_compat entry point."""
        body = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x"}}],
                },
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [{"type": "function", "function": {"name": "x"}}],
            "temperature": 1.5,
        }
        cfg = _make_model_cfg("kimi-k2-0905")

        result = apply_provider_compat(body, cfg)

        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}
        assert result["tools"] == body["tools"]
        assert result["temperature"] == 1.0

    def test_apply_provider_compat_deepseek_injects_reasoning_content(self):
        """US-031: DeepSeek reasoning_content Injection — apply_provider_compat entry point."""
        body = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
            "temperature": 2.5,
        }
        cfg = _make_model_cfg("deepseek-v4-pro")

        result = apply_provider_compat(body, cfg)

        assert result["temperature"] == 2.5
        assistant = result["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["reasoning_content"] == " "

    def test_apply_provider_compat_minimax_converts_system_to_user(self):
        """US-033: MiniMax Role Conversion — apply_provider_compat entry point.

        system→user conversion preserves the converted message as a distinct
        user turn so ContextOptimizeStage's TASK_PAYLOAD wrapping doesn't lose
        the system message into the auto-injected prompt. A synthetic
        assistant barrier is inserted between the converted-system user and
        the following real user to break the adjacency (MiniMax rejects
        back-to-back user turns).
        """
        body = {
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 0.5,
        }
        cfg = _make_model_cfg("minimax-2.5")

        result = apply_provider_compat(body, cfg)

        msgs = result["messages"]
        assert len(msgs) == 3, (
            "system→user + barrier + real user (no merge to preserve system content)"
        )
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "you are a helpful assistant"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == "hi"
        assert result["temperature"] == 0.5


class TestKimiAdapterDetails:
    """KimiAdapter-specific edge cases."""

    def test_kimi_clamps_below_zero(self):
        body = {"messages": [], "temperature": -0.5}
        out = KimiAdapter.apply(body, _make_model_cfg("kimi-k2"))
        assert out["temperature"] == 0.0

    def test_kimi_leaves_temperature_in_range(self):
        body = {"messages": [], "temperature": 0.7}
        out = KimiAdapter.apply(body, _make_model_cfg("kimi-k2"))
        assert out["temperature"] == 0.7

    def test_kimi_skips_messages_without_tool_calls(self):
        body = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "no tools here"},
            ],
        }
        out = KimiAdapter.apply(body, _make_model_cfg("kimi-k2"))
        assert "reasoning_content" not in out["messages"][1]


class TestMiniMaxAdapterDetails:
    """MiniMaxAdapter-specific edge cases."""

    def test_minimax_clamps_temperature_above_one(self):
        body = {"messages": [], "temperature": 5.0}
        out = MiniMaxAdapter.apply(body, _make_model_cfg("minimax-2.5"))
        assert out["temperature"] == 1.0

    def test_minimax_clamps_temperature_at_or_below_zero(self):
        body = {"messages": [], "temperature": 0.0}
        out = MiniMaxAdapter.apply(body, _make_model_cfg("minimax-2.5"))
        assert out["temperature"] == 0.01

    def test_minimax_system_vfs_converts_to_distinct_user_with_exact_barrier(self):
        """US: Wave 7 — system-origin VFS converts to user but stays distinct
        from the current user via an exact single-space assistant barrier.

        Real main-agent VFS enters MiniMaxAdapter.apply with role=system.
        MiniMax converts it to role=user while tracking the conversion so
        the ordinary-user merger skips it. The current user remains
        a separate user message, and MiniMax inserts an exact
        `{"role": "assistant", "content": " "}` barrier between them.
        """
        vfs_content = (
            "[VFS CONTEXT — Cached file state]\n"
            "Workspace: wave7-alpha | Cached files: 1\n"
            "\n"
            "Cached files (available for instant reads):\n"
            "  /wave7/provider.py (hash:12345678, 42B)\n"
            "[END VFS CONTEXT]"
        )
        body = {
            "messages": [
                {"role": "system", "content": vfs_content},
                {"role": "user", "content": "wave7 current request"},
            ],
            "temperature": 0.5,
        }
        body_before = copy.deepcopy(body)

        out = apply_provider_compat(body, _make_model_cfg("minimax-2.5"))

        # Exact role sequence.
        assert [m.get("role") for m in out["messages"]] == [
            "user",
            "assistant",
            "user",
        ], f"Expected [user(converted), assistant, user(current)], got {[m.get('role') for m in out['messages']]}"

        # Exact content equality.
        assert out["messages"][0] == {
            "role": "user",
            "content": vfs_content,
        }, f"VFS content must be preserved byte-for-byte; got {out['messages'][0]!r}"
        assert out["messages"][1] == {
            "role": "assistant",
            "content": " ",
        }, f"Barrier must be exactly one ASCII space; got {out['messages'][1]!r}"
        assert out["messages"][2] == {
            "role": "user",
            "content": "wave7 current request",
        }, f"Current user must be preserved; got {out['messages'][2]!r}"

        # No system role reaches MiniMax.
        assert not any(m.get("role") == "system" for m in out["messages"]), (
            f"No system role should remain in MiniMax output; got "
            f"{[m.get('role') for m in out['messages']]}"
        )

        # VFS occurs exactly once.
        vfs_indices = [
            i for i, m in enumerate(out["messages"])
            if "[VFS CONTEXT" in (m.get("content") or "")
        ]
        assert len(vfs_indices) == 1, (
            f"Exactly one VFS block expected; got {len(vfs_indices)}"
        )

        # VFS and current-user content were not concatenated.
        assert "wave7 current request" not in (out["messages"][0].get("content") or ""), (
            "Current-user content must not appear in VFS message"
        )
        assert "[VFS CONTEXT" not in (out["messages"][2].get("content") or ""), (
            "VFS content must not appear in current-user message"
        )

        # Input immutability: apply_provider_compat deepcopies the body, so
        # the caller's body is untouched.
        assert body == body_before, (
            f"Input body must be unchanged (apply_provider_compat deepcopies); "
            f"diff: {[(i, e, r) for i, (e, r) in enumerate(zip(body_before['messages'], body['messages'])) if e != r]}"
        )

    def test_minimax_merges_ordinary_consecutive_user_messages(self):
        """US: Wave 7 — generic ordinary consecutive user messages merge with
        `\\n\\n` separator and produce no synthetic barrier."""
        body = {
            "messages": [
                {"role": "user", "content": "first user fragment"},
                {"role": "user", "content": "second user fragment"},
            ],
        }
        body_before = copy.deepcopy(body)

        out = apply_provider_compat(body, _make_model_cfg("minimax-2.5"))

        assert out["messages"] == [
            {
                "role": "user",
                "content": (
                    "first user fragment\n\n"
                    "second user fragment"
                ),
            },
        ], f"Expected single merged user message; got {out['messages']!r}"

        # No synthetic barrier between ordinary user fragments.
        assert not any(
            m.get("role") == "assistant" and m.get("content") == " "
            for m in out["messages"]
        ), "No synthetic barrier between ordinary user fragments"

        # Input immutability.
        assert body == body_before, "Input body must be unchanged"

    def test_minimax_system_vfs_followed_by_assistant_needs_no_barrier(self):
        """US: Wave 7 — system→user conversion followed by assistant does not
        produce an extra barrier because no user/user adjacency results."""
        vfs_content = (
            "[VFS CONTEXT — Cached file state]\n"
            "Workspace: wave7-alpha | Cached files: 1\n"
            "[END VFS CONTEXT]"
        )
        body = {
            "messages": [
                {"role": "system", "content": vfs_content},
                {"role": "assistant", "content": "completed"},
            ],
        }
        out = MiniMaxAdapter.apply(body, _make_model_cfg("minimax-2.5"))

        assert [m.get("role") for m in out["messages"]] == [
            "user",
            "assistant",
        ], f"Expected [user(converted), assistant]; got {[m.get('role') for m in out['messages']]}"
        assert out["messages"][0] == {
            "role": "user",
            "content": vfs_content,
        }
        assert out["messages"][1] == {
            "role": "assistant",
            "content": "completed",
        }




    def test_minimax_merge_preserves_non_user_messages(self):
        """Non-adjacent user messages (separated by assistant) are untouched."""
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
        }
        out = MiniMaxAdapter.apply(body, _make_model_cfg("minimax-2.5"))
        assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user"]
        assert out["messages"][0]["content"] == "first"
        assert out["messages"][2]["content"] == "second"

    def test_minimax_merge_coerces_non_str_content(self):
        """Non-string content is coerced via ``str(content or "")``.

        None is falsy → contributes the empty string; a list is truthy → its
        repr is used. They are joined with the double-newline separator.
        """
        body = {
            "messages": [
                {"role": "user", "content": None},
                {"role": "user", "content": ["block", "of", "text"]},
            ],
        }
        out = MiniMaxAdapter.apply(body, _make_model_cfg("minimax-2.5"))
        assert [m["role"] for m in out["messages"]] == ["user"]
        content = out["messages"][0]["content"]
        # None → "" so the merge starts with the separator; list → its repr.
        assert content.startswith("\n\n")
        assert "['block', 'of', 'text']" in content


class TestDeepSeekAdapterDetails:
    """DeepSeekAdapter-specific edge cases."""

    def test_deepseek_does_not_clamp_temperature(self):
        body = {"messages": [], "temperature": 2.5}
        out = DeepSeekAdapter.apply(body, _make_model_cfg("deepseek-v4-pro"))
        assert out["temperature"] == 2.5

    def test_deepseek_skips_existing_reasoning_content(self):
        body = {
            "messages": [
                {"role": "assistant", "content": "ok", "reasoning_content": "thoughts"},
            ],
        }
        out = DeepSeekAdapter.apply(body, _make_model_cfg("deepseek-v4-pro"))
        assert out["messages"][0]["reasoning_content"] == "thoughts"


class TestMiMoAdapterDetails:
    """MiMoAdapter-specific edge cases."""

    def test_mimo_injects_reasoning_content(self):
        body = {
            "messages": [
                {"role": "assistant", "content": "ok"},
            ],
        }
        out = MiMoAdapter.apply(body, _make_model_cfg("mimo-v2"))
        assert out["messages"][0]["reasoning_content"] == " "


class TestApplyProviderCompatDispatch:
    """apply_provider_compat dispatch behavior."""

    def test_returns_body_unchanged_when_model_cfg_is_none(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 2.0}
        out = apply_provider_compat(body, None)
        assert out is body

    def test_unknown_backend_passes_through(self):
        body = {"messages": [{"role": "assistant", "content": "ok"}], "temperature": 1.5}
        cfg = _make_model_cfg("some-unknown-model")
        out = apply_provider_compat(body, cfg)
        assert "reasoning_content" not in out["messages"][0]
        assert out["temperature"] == 1.5

    def test_does_not_mutate_caller_messages(self):
        original_messages = [{"role": "assistant", "content": "ok"}]
        body = {"messages": original_messages}
        cfg = _make_model_cfg("deepseek-v4-pro")
        apply_provider_compat(body, cfg)
        assert "reasoning_content" not in original_messages[0]

    def test_backend_match_is_case_insensitive(self):
        body = {"messages": [{"role": "assistant", "content": "ok"}]}
        cfg = _make_model_cfg("Kimi-K2")
        out = apply_provider_compat(body, cfg)
        assert out["messages"][0].get("role") == "assistant"
