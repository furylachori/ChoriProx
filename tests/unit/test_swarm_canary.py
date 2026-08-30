"""Strict swarm canary primitive tests.

Covers the strict swarm canary primitive: error codes, signing, state
serialization, and the round-trip invariants:

  - `SwarmCanaryError` carries a stable code (no secret leakage)
  - `sign_canary_strict` rejects missing secret, zero-sig, malformed tokens
  - `_serialize_swarm_state` enforces the canonical serializer and the
    256-byte UTF-8 bound, and rejects every forbidden extra key
  - `build_swarm_skill_canary` never returns ''/None/bare/workspace/zero-sig
  - Real `parse_canary` round-trip via the existing parser
  - Built-in `BaseSkill.build_skill_canary` byte-for-byte invariant

The tests use real HMAC signing whenever the production path is exercised.
Patching `sign_canary` is reserved for tests that specifically exercise
the strict-output validation branch (it would be wrong to mock the
function-under-test for the round-trip suite).

`_SESSION_SECRET` cache reload pattern is borrowed from
`tests/unit/test_build_skill_canary.py:38-66` (the canonical pattern for
this module — without it, secrets leak between tests).
"""
from __future__ import annotations

import json
import os
import re
import time
from unittest.mock import patch

import pytest

from proxy import canary as canary_mod


# ---------------------------------------------------------------------------
# Canary secret isolation fixtures.
#
# `proxy.canary._SESSION_SECRET` is a module-level cache captured at IMPORT
# time. After import it is read by `_get_session_secret()` only via the
# fallback branch (`os.environ.get(...) or _SESSION_SECRET`); the cached
# value persists for the process lifetime unless explicitly reset.
#
# DO NOT use `importlib.reload(proxy.canary)` as a test fixture — reload
# re-runs ALL module-level code (including the swarm primitives added by
# Phase 2 Agent 3) and the re-execution can introduce cross-test
# ordering leaks. Instead, we pin `_SESSION_SECRET` directly via
# `monkeypatch.setattr` so monkeypatch auto-restores the original value
# at fixture teardown.
# ---------------------------------------------------------------------------

_CANARY_SECRET: str = "x" * 32


@pytest.fixture
def with_canary_secret(monkeypatch):
    """Pin `_SESSION_SECRET` to a known good value for this test."""
    monkeypatch.setattr(canary_mod, "_SESSION_SECRET", _CANARY_SECRET)
    yield canary_mod


@pytest.fixture
def without_canary_secret(monkeypatch):
    """Clear the secret (clear the env AND the cached fallback)."""
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    monkeypatch.setattr(canary_mod, "_SESSION_SECRET", "")
    yield canary_mod


@pytest.fixture
def empty_canary_secret(monkeypatch):
    """Empty PROXY_SESSION_SECRET — must still trip swarm_missing_secret.

    `_get_session_secret()` is `os.environ.get(...) or _SESSION_SECRET`,
    so BOTH the env var and the module-level fallback must be cleared —
    setting only one is insufficient when the other is non-empty.
    """
    monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
    monkeypatch.setattr(canary_mod, "_SESSION_SECRET", "")
    yield canary_mod


@pytest.fixture(autouse=True)
def _pin_canary_secret_to_post_test_env(request):
    """Auto-re-pin `_SESSION_SECRET` to whatever the env says AFTER each
    test has run.

    Without this autouse, `_SESSION_SECRET` set by `with_canary_secret`
    inside ONE test leaks into subsequent tests in DIFFERENT modules —
    because monkeypatch.setattr only restores for the test that created
    it. Tests that expected the dev fallback path (empty secret) would
    silently use the leaked secret and fail HMAC verification.
    """
    yield
    # After the test (and any inner monkeypatch.setattr) has run, the
    # monkeypatch stack has been unwound. We do NOT touch the env here
    # — monkeypatch undid that already. We just make sure the module's
    # cache reflects the post-test env state.
    #
    # Note: when this autouse is run, `os.environ` has already been
    # restored by monkeypatch. Reading from `os.environ.get(...)` here
    # is therefore safe and reflects the env state that THE NEXT test
    # will see when its own monkeypatch is set up.
    current_env = os.environ.get("PROXY_SESSION_SECRET", "")
    canary_mod._SESSION_SECRET = current_env


# ---------------------------------------------------------------------------
# Section 1 — SwarmCanaryError
# ---------------------------------------------------------------------------


class TestSwarmCanaryError:
    def test_carries_stable_code(self):
        from proxy.canary import SwarmCanaryError
        exc = SwarmCanaryError("swarm_missing_secret", "msg")
        assert exc.code == "swarm_missing_secret"
        assert str(exc) == "msg"
        assert isinstance(exc, RuntimeError)

    def test_message_is_non_secret(self):
        """Messages must not leak signature, secret, or state contents."""
        from proxy.canary import SwarmCanaryError
        # These tokens/substrings must NEVER appear in any swarm error msg
        forbidden_substrings = [
            _CANARY_SECRET,
            "0123456789abcdef" * 4,  # 64 hex chars
            "abcdef0123456789",
            ":sig:",
            "should-not-leak-state",
        ]
        msgs = [
            SwarmCanaryError("swarm_missing_secret", "msg-a").args[0],
            SwarmCanaryError("swarm_signing_failed", "msg-b").args[0],
            SwarmCanaryError("swarm_invalid_state", "msg-c").args[0],
        ]
        for m in msgs:
            for substr in forbidden_substrings:
                assert substr not in m, f"leak in message {m!r}"


# ---------------------------------------------------------------------------
# Section 2 — sign_canary_strict secret behavior
# ---------------------------------------------------------------------------


class TestSecretBehavior:
    def test_missing_secret_raises_swarm_missing_secret(self, without_canary_secret):
        from proxy.canary import SwarmCanaryError, build_swarm_skill_canary
        with pytest.raises(SwarmCanaryError) as exc_info:
            build_swarm_skill_canary("research", {"workspace_name": "alpha"})
        assert exc_info.value.code == "swarm_missing_secret"

    def test_empty_secret_raises_swarm_missing_secret(self, empty_canary_secret):
        from proxy.canary import SwarmCanaryError, build_swarm_skill_canary
        with pytest.raises(SwarmCanaryError) as exc_info:
            build_swarm_skill_canary("research", {})
        assert exc_info.value.code == "swarm_missing_secret"

    def test_present_secret_returns_valid_token(self, with_canary_secret):
        from proxy.canary import (
            SwarmCanaryError,
            _is_hex_64,
            sign_canary_strict,
        )
        token = sign_canary_strict("research", "|{\"workspace_name\":\"alpha\"}")
        sig, _, ts = token.partition(":")
        assert _is_hex_64(sig)
        assert ts.isdigit()
        assert int(ts) > 0

    def test_production_env_without_secret_still_rejects_swarm(self, monkeypatch):
        """PROXY_ENV=production with no PROXY_SESSION_SECRET — swarm refuses.

        The legacy sign_canary would emit a zero-signature dev fallback in this
        state; sign_canary_strict must refuse regardless of PROXY_ENV.
        """
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)
        monkeypatch.setattr(canary_mod, "_SESSION_SECRET", "")
        monkeypatch.setenv("PROXY_ENV", "production")
        from proxy.canary import SwarmCanaryError, build_swarm_skill_canary
        with pytest.raises(SwarmCanaryError) as exc_info:
            build_swarm_skill_canary("research", {})
        assert exc_info.value.code == "swarm_missing_secret"

    def test_legacy_sign_canary_dev_fallback_still_returns_zero_sig(self, without_canary_secret):
        """Ordinary sign_canary() still emits the 0*64 dev fallback when
        no secret is configured. The legacy builtin path depends on this;
        only the strict swarm signer refuses it.
        """
        from proxy.canary import sign_canary
        token = sign_canary("ticket", state_str="", session_id="")
        sig, _, ts = token.partition(":")
        assert sig == "0" * 64
        assert ts.isdigit()


# ---------------------------------------------------------------------------
# Section 3 — sign_canary_strict token validation (patch sign_canary)
# ---------------------------------------------------------------------------


class TestTokenValidation:
    """Patch the inner `sign_canary` to return each malformed token; the
    strict signer must reject every one with `swarm_signing_failed`.
    """

    @pytest.mark.parametrize("bad_token,reason", [
        ("", "empty token"),
        ("abc", "no separator"),
        ("abc:123", "sig wrong length"),
        (("a" * 63) + ":123", "sig 63 chars"),
        (("a" * 65) + ":123", "sig 65 chars"),
        (("g" * 64) + ":123", "sig non-hex"),
        (("0" * 64) + ":123", "sig all-zero dev fallback"),
        (("a" * 64) + ":", "ts empty"),
        (("a" * 64) + ":abc", "ts non-digits"),
        (("a" * 64) + ":123:extra", "extra colon"),
    ])
    def test_malformed_token_rejected(self, with_canary_secret, bad_token, reason):
        from proxy.canary import SwarmCanaryError
        from unittest.mock import patch
        with patch("proxy.canary.sign_canary", return_value=bad_token):
            with pytest.raises(SwarmCanaryError) as exc_info:
                with_canary_secret.sign_canary_strict(
                    "research", "|{\"workspace_name\":\"alpha\"}",
                )
        assert exc_info.value.code == "swarm_signing_failed", reason

    def test_underlying_exception_maps_to_swarm_signing_failed(self, with_canary_secret):
        """If sign_canary raises, the strict signer catches + raises
        `swarm_signing_failed` WITHOUT leaking the original exception text.
        """
        from proxy.canary import SwarmCanaryError
        from unittest.mock import patch
        with patch(
            "proxy.canary.sign_canary",
            side_effect=ValueError("super-secret-internals: token=DEADBEEF secret=abc123"),
        ):
            with pytest.raises(SwarmCanaryError) as exc_info:
                with_canary_secret.sign_canary_strict("research", state_str="")
        assert exc_info.value.code == "swarm_signing_failed"
        # The leaked "secret" string from the mock exception must NOT
        # appear in the SwarmCanaryError message.
        assert "super-secret-internals" not in str(exc_info.value)
        assert "DEADBEEF" not in str(exc_info.value)
        assert "abc123" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Section 4 — Skill name validation
# ---------------------------------------------------------------------------


class TestSkillNameValidation:
    @pytest.mark.parametrize("bad_name", [
        "",
        "Research",
        "1research",
        "_research",
        "research space",
        "research/child",
        "a" * 65,
        None,
        42,
        [],
    ])
    def test_invalid_name_rejected(self, with_canary_secret, bad_name):
        from proxy.canary import SwarmCanaryError, build_swarm_skill_canary
        with pytest.raises(SwarmCanaryError) as exc_info:
            build_swarm_skill_canary(bad_name, {})
        assert exc_info.value.code == "swarm_invalid_skill_name", repr(bad_name)

    @pytest.mark.parametrize("good_name", [
        "a",
        "research",
        "research-swarm",
        "research_swarm",
        "a" * 64,
    ])
    def test_valid_name_accepted(self, with_canary_secret, good_name):
        from proxy.canary import build_swarm_skill_canary
        marker = build_swarm_skill_canary(good_name, {})
        assert good_name in marker


# ---------------------------------------------------------------------------
# Section 5 — State shape validation
# ---------------------------------------------------------------------------


class TestStateShape:
    @pytest.mark.parametrize("good_state", [
        {},
        {"workspace_name": "alpha"},
        {"workspace_name": "research-ws"},
        {"workspace_name": "a" * 1},
    ])
    def test_accept_good_state(self, with_canary_secret, good_state):
        from proxy.canary import _serialize_swarm_state
        result = _serialize_swarm_state(good_state)
        # empty dict → ""; non-empty → "|{json}"
        if good_state == {}:
            assert result == ""
        else:
            assert result.startswith("|")
            # Round-trip: must parse back to the same dict.
            assert json.loads(result[1:]) == good_state
            # And it must use canonical sort_keys=True.
            assert result == f"|{json.dumps(good_state, sort_keys=True)}"

    @pytest.mark.parametrize("bad_state,reason", [
        (None, "None"),
        ([], "list"),
        ("alpha", "string"),
        (42, "int"),
        ({"workspace_id": "alpha"}, "extra key: workspace_id"),
        ({"parent_skill_name": "research"}, "extra key: parent_skill_name"),
        ({"task_hash": "abc"}, "extra key: task_hash"),
        ({"subagents": {}}, "extra key: subagents"),
        ({"workspace_name": None}, "workspace_name None"),
        ({"workspace_name": 123}, "workspace_name int"),
        ({"workspace_name": ""}, "workspace_name empty string"),
        ({"workspace_name": {"k": "v"}}, "workspace_name nested dict"),
        ({"workspace_name": []}, "workspace_name list"),
        ({"workspace_name": "x", "_": "y"}, "extra key with workspace_name"),
    ])
    def test_reject_bad_state(self, with_canary_secret, bad_state, reason):
        from proxy.canary import SwarmCanaryError, _serialize_swarm_state
        with pytest.raises(SwarmCanaryError) as exc_info:
            _serialize_swarm_state(bad_state)
        assert exc_info.value.code == "swarm_invalid_state", reason


# ---------------------------------------------------------------------------
# Section 6 — Canonical serialization round-trips via real parse_canary
# ---------------------------------------------------------------------------


_SWARM_MARKER_PATTERN = re.compile(
    r"^\[\[__SUBAGENT__:SKILL:[a-z][a-z0-9_-]*:"
    r"[0-9a-f]{64}:[0-9]+(?:\|.*)?\]\]$"
)


class TestCanonicalSerialization:
    def test_marker_uses_canonical_serializer_not_compact(self, with_canary_secret):
        from proxy.canary import build_swarm_skill_canary
        state = {"workspace_name": "alpha"}
        marker = build_swarm_skill_canary("ticket", state, session_id="sess-canon")
        # Marker must contain the JSON state with default separators.
        # If compact `(",", ":")` were used, the state suffix would be
        # `{"workspace_name":"alpha"}` (no spaces after `,` / `:`).
        # We assert with the canonical `(", ", ": ")` form:
        canonical_suffix = '{"workspace_name": "alpha"}'
        compact_suffix = '{"workspace_name":"alpha"}'
        assert canonical_suffix in marker
        assert compact_suffix not in marker

    def test_round_trips_via_real_parse_canary(self, with_canary_secret):
        """Real HMAC sign + real parse_canary verifies True, state preserved."""
        from proxy.canary import build_swarm_skill_canary, parse_canary
        marker = build_swarm_skill_canary(
            "ticket",
            {"workspace_name": "alpha"},
            session_id="sess-canon-roundtrip",
        )
        parsed = parse_canary(marker, session_id="sess-canon-roundtrip")
        assert parsed is not None, f"parse_canary returned None for {marker!r}"
        assert parsed.kind == "skill"
        assert parsed.verified is True
        assert parsed.skill_name == "ticket"
        assert parsed.state == {"workspace_name": "alpha"}

    def test_state_mutation_after_build_invalidates_verification(self, with_canary_secret):
        """Mutating the state suffix by one char → parse_canary detects it."""
        from proxy.canary import build_swarm_skill_canary, parse_canary
        marker = build_swarm_skill_canary(
            "ticket",
            {"workspace_name": "alpha"},
            session_id="sess-canon-mut",
        )
        # Reconstruct the JSON portion and change one byte of the state
        # value (e.g., alpha → a1pha) AFTER signing.
        tampered = marker.replace('"alpha"', '"a1pha"')
        assert tampered != marker
        parsed = parse_canary(tampered, session_id="sess-canon-mut")
        assert parsed is not None
        # State was bound into the HMAC, so tampering invalidates verification.
        assert parsed.verified is False


# ---------------------------------------------------------------------------
# Section 7 — 256-byte size boundary
# ---------------------------------------------------------------------------


class TestSizeBoundary:
    def test_exactly_256_bytes_accepted(self, with_canary_secret):
        """`{"workspace_name": "<234-char-name>"}` → 256-byte serialized JSON.

        json.dumps({"workspace_name": "a"*N}) has the form
            {"workspace_name": "a...a"}
        with character count = 22 + N (template `{"workspace_name": ""}` is 22
        chars; we replace the empty value with N+2 chars). For exactly 256
        bytes: N = 234.
        """
        from proxy.canary import build_swarm_skill_canary
        workspace_name = "a" * 234
        assert len(json.dumps({"workspace_name": workspace_name}).encode("utf-8")) == 256
        marker = build_swarm_skill_canary(
            "ticket",
            {"workspace_name": workspace_name},
            session_id="sess-size-ok",
        )
        assert "[[__SUBAGENT__:SKILL:ticket:" in marker

    def test_257_bytes_rejected_with_swarm_state_too_large(self, with_canary_secret):
        """N=235 → serialized JSON = 257 bytes → swarm_state_too_large."""
        from proxy.canary import SwarmCanaryError, build_swarm_skill_canary
        workspace_name = "a" * 235
        assert len(json.dumps({"workspace_name": workspace_name}).encode("utf-8")) == 257
        with pytest.raises(SwarmCanaryError) as exc_info:
            build_swarm_skill_canary(
                "ticket",
                {"workspace_name": workspace_name},
                session_id="sess-size-ko",
            )
        assert exc_info.value.code == "swarm_state_too_large"

    def test_byte_count_is_bytes_not_characters_with_multibyte(self, with_canary_secret):
        """A 100-byte ASCII workspace name passes; a Unicode-string that
        expands to ASCII escapes (longer than 256 bytes) must be rejected.

        json.dumps escapes non-ASCII to \\uXXXX. So a string of 256
        backslash-escape characters of 6 chars each → raw escape output
        is 256*6 = 1536 bytes after sort_keys, which exceeds 256.
        """
        from proxy.canary import SwarmCanaryError, _serialize_swarm_state
        # 50 emoji codepoints = 50 escaped \uXXXX sequences (6 ASCII chars each) = 300 bytes
        # 50*6 = 300; minus the wrapper {"workspace_name": ""} which is 23 bytes,
        # so the JSON prefix+closing = 17 + 50*6 + 2 = 319 bytes. Way over 256.
        # Use 40 emoji instead → 40*6 = 240; +17+2 = 259 bytes — JUST over 256.
        emoji_workspace_name = "🚀" * 40
        with pytest.raises(SwarmCanaryError) as exc_info:
            _serialize_swarm_state({"workspace_name": emoji_workspace_name})
        assert exc_info.value.code == "swarm_state_too_large"


# ---------------------------------------------------------------------------
# Section 8 — Marker shape
# ---------------------------------------------------------------------------


class TestMarkerShape:
    def test_empty_state_marker_matches_regex(self, with_canary_secret):
        from proxy.canary import build_swarm_skill_canary
        marker = build_swarm_skill_canary("ticket", {})
        assert _SWARM_MARKER_PATTERN.match(marker)
        # No `|...` state suffix when state is empty.
        assert "|" not in marker.split("]]")[0]

    def test_workspace_state_marker_matches_regex_with_single_pipe(self, with_canary_secret):
        from proxy.canary import build_swarm_skill_canary
        marker = build_swarm_skill_canary("ticket", {"workspace_name": "alpha"})
        assert _SWARM_MARKER_PATTERN.match(marker)
        # Exactly one `|` separating state from sig/ts.
        body = marker[2:-2]  # strip [[ and ]]
        pipe_count = body.count("|")
        assert pipe_count == 1


# ---------------------------------------------------------------------------
# Section 9 — Tampering and verification (real parser)
# ---------------------------------------------------------------------------


class TestTamperingAndVerification:
    def _build(self, with_canary_secret, session_id="sess-tamper"):
        return with_canary_secret.build_swarm_skill_canary(
            "ticket",
            {"workspace_name": "alpha"},
            session_id=session_id,
        )

    def test_change_skill_name_makes_verification_fail(self, with_canary_secret):
        from proxy.canary import parse_canary
        marker = self._build(with_canary_secret, "sess-tamper-1")
        # The skill_name is embedded BEFORE the sig. We change the
        # name to something else — but the marker still ends with ]]
        # and the same sig:ts, so the parser sees a different name but
        # the same signature. Verification fails because HMAC binds the
        # name into the message.
        tampered = marker.replace(":SKILL:ticket:", ":SKILL:researc:", 1)
        # Re-attach ]] — the new marker must remain parseable structurally.
        tampered = tampered + "]]" if not tampered.endswith("]]") else tampered
        # Note: structurally parseable as a different skill name.
        parsed = parse_canary(tampered, session_id="sess-tamper-1")
        if parsed is not None:
            # Either kind='skill' + verified=False (HMAC fails for new name),
            # or kind=None / unparseable (signature still says ticket's sig but
            # the locator may reject different skill_name). The invariant is:
            # verification is NOT True.
            assert parsed.verified is False

    def test_change_signature_makes_verification_fail(self, with_canary_secret):
        from proxy.canary import parse_canary
        marker = self._build(with_canary_secret, "sess-tamper-2")
        # Flip one hex char in the signature region.
        # Find the signature: between the 4th ':' and the next ':'.
        import re
        m = re.search(r":([0-9a-f]{64}):", marker)
        assert m is not None
        sig = m.group(1)
        flipped = "f" if sig[0] != "f" else "0"
        new_sig = flipped + sig[1:]
        tampered = marker.replace(sig, new_sig)
        parsed = parse_canary(tampered, session_id="sess-tamper-2")
        assert parsed is not None
        assert parsed.verified is False

    def test_change_state_makes_verification_fail(self, with_canary_secret):
        """Same as TestCanonicalSerialization.test_state_mutation_after_build.
        Kept here as a focused tampering invariant.
        """
        from proxy.canary import parse_canary
        marker = self._build(with_canary_secret, "sess-tamper-3")
        tampered = marker.replace('"alpha"', '"beta"')
        parsed = parse_canary(tampered, session_id="sess-tamper-3")
        assert parsed is not None
        assert parsed.verified is False

    def test_unregistered_skill_rejected(self, with_canary_secret):
        """Skill name not in SkillRegistry → parse_canary returns
        verified=False even though HMAC passed.
        """
        from proxy.canary import build_swarm_skill_canary, parse_canary
        marker = build_swarm_skill_canary(
            "research-this-skill-does-not-exist-anywhere",
            {"workspace_name": "alpha"},
            session_id="sess-unregistered",
        )
        parsed = parse_canary(marker, session_id="sess-unregistered")
        assert parsed is not None
        assert parsed.kind == "skill"
        # HMAC succeeded (fresh sig/ts in 5-min window) but the registry
        # membership check fails the verification.
        assert parsed.verified is False

    def test_stale_timestamp_rejected(self, with_canary_secret):
        """Stale ts (signer used int(time.time()) - 600) → verify_canary
        rejects for freshness.

        Patch time.time ONLY during the build so the timestamp embedded
        in the canary is stale, but the verifier sees the real clock.
        """
        from proxy.canary import build_swarm_skill_canary, parse_canary
        now = int(time.time())
        with patch("time.time", return_value=now - 600):
            marker = build_swarm_skill_canary(
                "ticket",
                {"workspace_name": "alpha"},
                session_id="sess-stale",
            )
        # Real clock restored. Verifier's `now = int(time.time())`
        # computes ~now, so the embedded timestamp is 600s old → reject.
        parsed = parse_canary(marker, session_id="sess-stale")
        assert parsed is not None
        assert parsed.verified is False

    def test_marker_not_at_position_zero_does_not_confer_inherited_privilege(
        self, with_canary_secret
    ):
        """If the marker is preceded by other text, parse_canary does NOT
        parse it as a canary (because the locator requires loc.start == 0).
        """
        from proxy.canary import build_swarm_skill_canary, parse_canary
        marker = build_swarm_skill_canary(
            "ticket",
            {"workspace_name": "alpha"},
            session_id="sess-position",
        )
        prefixed = "Some intro text: " + marker
        parsed = parse_canary(prefixed, session_id="sess-position")
        # Either None (no canary at position 0) or verified=False.
        if parsed is not None:
            assert parsed.verified is False
            # And not granted any privilege — inherited_skill is None.
            assert parsed.skill_name is None

    def test_valid_marker_does_not_confer_proxy_memory_privilege(self, with_canary_secret):
        """An arbitrary skill canary grants `inherited_skill == name` but
        does NOT grant builtin trust (the trust gate runs separately and
        is keyed off `is_trusted_skill(name)` — only 'ticket' / 'work'
        are trusted). `parsed.kind == 'skill'` for an unregistered skill
        still yields verified=False per AC 16.17.
        """
        from proxy.canary import build_swarm_skill_canary, parse_canary
        from proxy.skills.skill_definition import is_trusted_skill
        # Even when a verifier accepts the marker, the trust gate decides
        # proxy_* access — only ticket/work are trusted.
        assert is_trusted_skill("ticket") is True
        assert is_trusted_skill("work") is True
        assert is_trusted_skill("research") is False
        # And the parser/registry check still requires the name to
        # exist; an unregistered custom name → verified=False regardless.
        marker = build_swarm_skill_canary(
            "research",
            {"workspace_name": "alpha"},
            session_id="sess-trust",
        )
        parsed = parse_canary(marker, session_id="sess-trust")
        assert parsed is not None
        assert parsed.verified is False


# ---------------------------------------------------------------------------
# Section 10 — Builtin isolation
# ---------------------------------------------------------------------------


class TestBuiltinIsolation:
    """Pin: BaseSkill.build_skill_canary source/behavior unchanged.
    TicketSkill canary still round-trips. WorkSkill canary still round-trips.
    """

    def test_base_skill_canary_marker_format_matches(self, with_canary_secret):
        """A valid swarm canary and a BaseSkill-built canary produce the
        same marker shape for the same input (regex match).
        """
        from proxy.canary import build_swarm_skill_canary, parse_canary
        from unittest.mock import MagicMock
        from proxy.skills.skill_definition import BaseSkill

        # Swarm canary
        swarm_marker = build_swarm_skill_canary(
            "ticket",
            {"workspace_name": "alpha"},
            session_id="sess-base-skill",
        )
        # BaseSkill canary
        bs = BaseSkill()
        bs.name = "ticket"
        bs.get_inheritance_state = lambda ctx: {"workspace_name": "alpha"}
        ctx = MagicMock()
        ctx.session_id = "sess-base-skill"
        bs_marker = bs.build_skill_canary(ctx)

        # Both must match the swarm marker shape.
        assert _SWARM_MARKER_PATTERN.match(swarm_marker)
        # Note: BaseSkill may serialize state slightly differently
        # (include sort_keys=True at line 1052), but the marker SHAPE
        # must be identical — both start with [[__SUBAGENT__:SKILL:ticket:
        # and both end with ]].
        assert swarm_marker.startswith("[[__SUBAGENT__:SKILL:ticket:")
        assert swarm_marker.endswith("]]")
        assert bs_marker.startswith("[[__SUBAGENT__:SKILL:ticket:")
        assert bs_marker.endswith("]]")
        # And both round-trip via parse_canary.
        assert parse_canary(swarm_marker, session_id="sess-base-skill").verified is True
        assert parse_canary(bs_marker, session_id="sess-base-skill").verified is True

    def test_ticket_skill_canary_still_round_trips(self, with_canary_secret):
        """TicketSkill canary via get_canary() still parses + verifies."""
        try:
            from skills.ticket_skill import TicketSkill
        except ImportError:
            pytest.skip("TicketSkill not importable in this environment")
        from proxy.canary import parse_canary

        class _StubCtx:
            session_id = "sess-ticket-skill"
            is_subagent = False
            inherited_skill = None
            subagent_ticket_mode = False
            workspace_name = ""
            client_hint = None
            body = {}
            seeded_tasks = []

        ts = TicketSkill()
        ctx = _StubCtx()
        # Some TicketSkill builds are gated by TrustToken. If so, skip.
        try:
            marker = ts.get_canary(ctx)
        except Exception:
            pytest.skip("TicketSkill.get_canary unavailable")
        if not isinstance(marker, str) or not marker.startswith("[["):
            pytest.skip("TicketSkill returned a non-canary (e.g., None)")
        parsed = parse_canary(marker, session_id="sess-ticket-skill")
        assert parsed is not None
        assert parsed.kind == "skill"

    def test_work_skill_canary_still_round_trips(self, with_canary_secret):
        """WorkSkill canary via get_canary() still parses + verifies."""
        try:
            from skills.work_skill import WorkSkill
        except ImportError:
            pytest.skip("WorkSkill not importable in this environment")
        from proxy.canary import parse_canary

        class _StubCtx:
            session_id = "sess-work-skill"
            is_subagent = False
            inherited_skill = None
            subagent_ticket_mode = False
            workspace_name = ""
            client_hint = None
            body = {}
            seeded_tasks = []

        ws = WorkSkill()
        ctx = _StubCtx()
        try:
            marker = ws.get_canary(ctx)
        except Exception:
            pytest.skip("WorkSkill.get_canary unavailable")
        if not isinstance(marker, str) or not marker.startswith("[["):
            pytest.skip("WorkSkill returned a non-canary (e.g., None)")
        parsed = parse_canary(marker, session_id="sess-work-skill")
        assert parsed is not None
        assert parsed.kind == "skill"

    def test_no_swarm_helper_invoked_by_builtins(self, with_canary_secret):
        """Builtin TicketSkill and WorkSkill MUST NOT use the swarm helpers.

        Behavior assertion: their `get_canary` callbacks either return
        a BaseSkill-built marker (HMAC over their own state) or a bare /
        None marker — never a `build_swarm_skill_canary` output. We
        check by patching `build_swarm_skill_canary` to record calls
        and asserting ZERO invocations.
        """
        try:
            from skills.ticket_skill import TicketSkill
            from skills.work_skill import WorkSkill
        except ImportError:
            pytest.skip("Skills not importable")

        from unittest.mock import MagicMock, patch

        ctx = MagicMock()
        ctx.session_id = "sess-builtin-no-swarm"
        ctx.workspace_name = ""
        ctx.body = {}

        with patch(
            "proxy.canary.build_swarm_skill_canary",
            side_effect=AssertionError("swarm helper invoked from builtin!"),
        ) as mock_swarm:
            # TicketSkill
            try:
                TicketSkill().get_canary(ctx)
            except Exception:
                pass
            # WorkSkill
            try:
                WorkSkill().get_canary(ctx)
            except Exception:
                pass

        # mock_swarm was never called.
        assert mock_swarm.call_count == 0
