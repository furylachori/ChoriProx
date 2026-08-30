"""Tests for proxy.state.make_workspace_id (v5 global catalog).

The v5 spec changes make_workspace_id from a 2-arg (session_id, workspace)
helper returning a compound `session_id:workspace` id to a 1-arg
`make_workspace_id(workspace)` helper returning the bare workspace name.

These tests pin down the current behavior of the rewritten function:
- bare name passthrough
- rejection of empty, colon-containing, and otherwise non-alphanumeric names
- character-class allowlist (alphanumeric + '-' + '_')
- positional arity: only one positional arg accepted (legacy 2-arg call raises TypeError)
- length handling: 64-char max enforced (mirrors canary._MAX_WORKSPACE_NAME_LENGTH)
"""
import pytest

from proxy.state import make_workspace_id


class TestBareName:
    def test_returns_bare_name(self):
        """The canonical case — bare name passes through unchanged."""
        assert make_workspace_id("choriprox") == "choriprox"

    def test_returns_short_name(self):
        assert make_workspace_id("a") == "a"

    def test_returns_long_name(self):
        assert make_workspace_id("alpha-numeric_workspace-3") == "alpha-numeric_workspace-3"

    def test_no_colon_in_output(self):
        """v5 invariant: the returned value never contains ':'."""
        out = make_workspace_id("choriprox")
        assert ":" not in out


class TestRejectsEmpty:
    def test_rejects_empty_workspace(self):
        with pytest.raises(ValueError):
            make_workspace_id("")

    def test_rejects_whitespace_only(self):
        """All-whitespace is rejected via the .strip() check in the source."""
        with pytest.raises(ValueError):
            make_workspace_id("   ")


class TestRejectsColon:
    def test_rejects_workspace_with_colon(self):
        """v5 invariant: colons are reserved as the legacy compound separator."""
        with pytest.raises(ValueError):
            make_workspace_id("choriprox:prod")

    def test_rejects_legacy_compound_form(self):
        """Mimic the v4 `session_id:workspace` shape — must be rejected."""
        with pytest.raises(ValueError):
            make_workspace_id("8d959d6d6f71ba93:choriprox")


class TestRejectsSpecialCharacters:
    def test_rejects_workspace_with_special_chars(self):
        """Non-alphanumeric characters outside the allowlist are rejected."""
        with pytest.raises(ValueError):
            make_workspace_id("choriprox@home")

    def test_rejects_workspace_with_space(self):
        with pytest.raises(ValueError):
            make_workspace_id("chori prox")

    def test_rejects_workspace_with_dot(self):
        with pytest.raises(ValueError):
            make_workspace_id("chori.prox")

    def test_rejects_workspace_with_slash(self):
        with pytest.raises(ValueError):
            make_workspace_id("chori/prox")

    def test_rejects_workspace_with_quote(self):
        """Defends against Cypher injection (single quote)."""
        with pytest.raises(ValueError):
            make_workspace_id("chori'prox")

    def test_rejects_workspace_with_semicolon(self):
        with pytest.raises(ValueError):
            make_workspace_id("chori;prox")


class TestAcceptsAlphanumericWithHyphensAndUnderscores:
    def test_accepts_alphanumeric_with_hyphens_and_underscores(self):
        assert make_workspace_id("my-ws_1") == "my-ws_1"

    def test_accepts_leading_underscore(self):
        assert make_workspace_id("_ws") == "_ws"

    def test_accepts_leading_digit(self):
        """Pure-numeric names are also allowed by .isalnum()."""
        assert make_workspace_id("123") == "123"


class TestLengthHandling:
    """Length-handling tests for the bare-name function.

    The v5 workspace rework specifies a 64-character maximum length for
    workspace names (matching the canary regex's
    `_MAX_WORKSPACE_NAME_LENGTH = 64`). `make_workspace_id` now
    enforces this limit.
    """

    def test_exactly_64_chars_accepted(self):
        """A 64-character bare name is accepted."""
        name = "a" * 64
        assert make_workspace_id(name) == name

    def test_max_length_64(self):
        """A 65-character name is rejected (length IS enforced)."""
        name = "a" * 65
        with pytest.raises(ValueError, match="too long"):
            make_workspace_id(name)


class TestNoSessionIdInOutput:
    """v5 signature: single arg. The legacy 2-arg call is rejected at the
    Python call site (TypeError), so output is always bare by construction.
    """

    def test_legacy_two_arg_call_raises_typeerror(self):
        """v5 deliberately drops session_id. Passing it positionally raises
        TypeError — clean break from the v4 signature.
        """
        with pytest.raises(TypeError):
            make_workspace_id("sess1", "choriprox")

    def test_legacy_two_arg_call_with_kwargs_raises_too(self):
        """A keyword-style 'session_id' is not a recognized parameter."""
        with pytest.raises(TypeError):
            make_workspace_id(session_id="sess1", workspace="choriprox")
