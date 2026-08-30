"""Canary parsing and injection for sub-agent handling.

Implements §11.1-§11.2 of the skill-system-spec.

Canary formats (HMAC-signed only):
- [[__SUBAGENT__:SKILL:<name>:<64-hex-sig>:<ts>]] or
  [[__SUBAGENT__:SKILL:<name>:<64-hex-sig>:<ts>|<json>]]
- [[__SUBAGENT__:WORKSPACE:<name>:<64-hex-sig>:<ts>]]  (workspace inherits SKILL)
- [[__SUBAGENT__]] → maps to None (no skill inheritance; bare marker)

Verification model
------------------
Canary signatures are HMAC-signed with ``PROXY_SESSION_SECRET`` to prevent
forged canary injection. Every canary MUST carry a valid 64-character hex
HMAC signature; the parser rejects bare (unsigned) formats. The HMAC
message is ``f"{skill_name}:{state_str}:{timestamp_str}"`` where
``skill_name`` is the literal ``workspace:<name>`` for workspace canaries
and ``state_str`` is the JSON inheritance state (or empty when no state
is bound). State binding (B-3) prevents post-capture mutation of
``workspace_name``/``task_hash``. ``session_id`` is NOT bound — parent
and sub-agent have different session_ids by design, so binding to either
would break the legitimate production flow (BLOCKER). Replay protection
relies on the 5-minute freshness window. Verification is constant-time
via ``hmac.compare_digest`` and rejects canaries older than 5 minutes.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from proxy.types import SubagentIngress
from proxy.utils import extract_text_content


@dataclass
class ParsedCanary:
    skill: str
    timestamp: int
    sig: str
    is_dev_fallback: bool = False


@dataclass
class CanaryParse:
    """Result of parsing a canary marker at the start of a text string.

    This is the SINGLE primitive produced by :func:`parse_canary`; it
    replaces the three previously-divergent parsers
    (the legacy ``parse_skill_canary``, the ad-hoc ``find("]]")``
    branches in ``detect_subagent_ingress``, and the bare-marker
    handling).

    Fields:
        kind:           "bare" | "skill" | "workspace".
        remaining:      text after the canary, leading newlines stripped.
                        For a structurally-malformed canary this is the
                        original text (no reliable close was found).
        verified:       True for bare (always). For skill/workspace, True
                        ONLY if the structural parse succeeded AND the HMAC
                        verifies AND (skill only) the skill is registered.
        skill_name:     parsed skill name; set only when kind="skill" and
                        verified=True. None otherwise.
        workspace_name: parsed workspace name; set only when kind="workspace"
                        and verified=True. None otherwise.
        state:          inheritance state dict; set only when kind="skill",
                        verified=True, and a ``|{json}`` state was present.
        is_dev_fallback: True for dev-mode (no PROXY_SESSION_SECRET) verifies
                        (H1 cache round-trip); False for HMAC-verified. Only
                        set on verified skill/workspace parses; bare/malformed
                        canaries always get False.
    """
    kind: str
    remaining: str
    verified: bool
    skill_name: Optional[str] = None
    workspace_name: Optional[str] = None
    state: Optional[dict] = None
    is_dev_fallback: bool = False


# Backward-compatible module-level cache for PROXY_SESSION_SECRET.
#
# Design rationale:
# - _get_session_secret() reads fresh from os.environ first to avoid stale
#   values in long-running processes.
# - Falls back to this module-level cache so existing tests can manipulate
#   it directly via monkey-patching (e.g., _SESSION_SECRET.value = "...").
#
# Thread-safety: NOT safe for runtime secret rotation. If the secret changes
# at runtime, process restart is required. This is an intentional trade-off
# for test compatibility and simplicity.
_SESSION_SECRET = os.environ.get("PROXY_SESSION_SECRET", "")


def _get_session_secret() -> str:
    """Fetch PROXY_SESSION_SECRET from environment (avoids module-level caching).

    Returns the env value if set, otherwise falls back to the module-level
    _SESSION_SECRET cache (for backward compatibility with tests).
    """
    return os.environ.get("PROXY_SESSION_SECRET") or _SESSION_SECRET


def sign_canary(skill_name: str, state_str: str = "", session_id: str = "") -> str:
    """Generate HMAC signature with timestamp and state binding.

    HMAC input: ``f"{skill_name}:{state_str}:{timestamp}"``

    The state is bound into the signature to prevent post-capture mutation
    of workspace_name/task_hash (B-3). The 5-minute freshness window
    prevents replay attacks — ``session_id`` is NOT bound because parent
    and sub-agent have different session_ids by design (the sub-agent's
    first user message is the parent's tool-call prompt, which hashes to
    a different session_id), and binding to either one would break the
    legitimate production flow (BLOCKER). ``session_id`` is accepted as
    a cache-scope key (see §5.4) for backward compatibility with older callers.

    Empty ``state_str`` produces a 3-field message with an empty middle
    field (``f"{skill_name}::{timestamp}"``) — there is no legacy 2-field
    HMAC; canaries have a 5-minute TTL so cross-version replay is not a
    concern.
    """
    if not _get_session_secret():
        return f"{'0' * 64}:{int(time.time())}"
    timestamp = str(int(time.time()))
    message = f"{skill_name}:{state_str}:{timestamp}"
    sig = hmac.new(
        _get_session_secret().encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{sig}:{timestamp}"


def verify_canary(
    token: str,
    state_str: str = "",
    session_id: str = "",
    skill: Optional[str] = None,
) -> Optional[ParsedCanary]:
    """Verify a canary token and return the parsed payload, or None if invalid.

    Token formats (two variants, both HMAC-signed):

    - SKILL: ``f"{skill_name}:{sig}:{timestamp}"`` (3-part)
    - WORKSPACE: ``f"workspace:{name}:{sig}:{timestamp}"`` (4-part)

    For both formats ``sig`` is always the full 64-hex-char (256-bit) HMAC
    and ``timestamp`` is a unix-seconds integer. The HMAC message is
    computed over the full ``skill_name`` (including the ``workspace:``
    prefix for workspace canaries) so a single secret signs both formats.

    Binding (B-3): the HMAC input is ``f"{skill_name}:{state_str}:{timestamp}"``
    so the signature cannot be replayed with mutated state. Empty
    ``state_str`` produces a 3-field message with an empty middle field
    (``f"{skill_name}::{ts}"``) — there is no legacy 2-field HMAC.
    ``session_id`` is accepted as a cache-scope key (see §5.4) for backward
    compatibility — parent and sub-agent have different session_ids by
    design, so binding to either would break the legitimate flow. Replay
    protection is via the 5-minute freshness window.

    Returns ``None`` if the token is malformed, the HMAC fails verification,
    or the timestamp is older than the 5-minute freshness window.
    """
    if not token or ":" not in token:
        return None
    parts = token.split(":")
    if len(parts) == 3:
        # SKILL format: skill_name:sig:timestamp
        skill_name, sig, timestamp_str = parts
        # R4-CRIT-2: validate skill name format and length
        if not re.match(r'^[a-z][a-z0-9_-]*$', skill_name) or len(skill_name) > 64:
            return None
    elif len(parts) == 4:
        # WORKSPACE format: workspace:name:sig:timestamp
        prefix, name, sig, timestamp_str = parts
        if prefix != "workspace":
            return None  # Unknown prefix
        skill_name = f"workspace:{name}"
        # R4-CRIT-2: validate workspace name format and length
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*$', name) or len(name) > 64:
            return None
    elif len(parts) == 2:
        first, second = parts
        # Two possible 2-part formats:
        # 1. skill_name:sig — dev-mode shorthand (unreachable from parse_canary)
        # 2. sig:timestamp — sign_canary() return value (direct round-trip)
        if len(first) == 64 and all(c in '0123456789abcdef' for c in first.lower()):
            # sig:timestamp format — sign_canary return value
            sig, timestamp_str = first, second
            skill_name = skill or "unknown"
        else:
            # skill_name:sig format — dev-mode shorthand
            from proxy.assertions import is_dev_env
            if not is_dev_env():
                return None  # Production requires full 3-part format
            skill_name, sig = first, second
            if not re.match(r'^[a-z][a-z0-9_-]*$', skill_name) or len(skill_name) > 64:
                return None
            timestamp_str = str(int(time.time()))  # Use current time as fallback
    else:
        return None  # Malformed

    if not (len(sig) == 64 and all(c in '0123456789abcdef' for c in sig.lower())):
        # In dev mode with no secret, accept non-empty sigs that aren't
        # 64-char HMAC (e.g., short hashes from clients that don't have
        # the signing secret). Production still requires full 64-char sig.
        from proxy.assertions import is_dev_env
        if not is_dev_env() or not sig:
            return None  # Malformed: middle field is not a 64-char HMAC

    try:
        timestamp = int(timestamp_str)
    except ValueError:
        return None

    # Timestamp freshness check (5-minute / 300s window).
    # Enforced even when PROXY_SESSION_SECRET is unset: in dev mode we
    # accept unsigned canaries but still require freshness.
    now = int(time.time())
    # R4-HIGH-1: asymmetric clock skew — future tolerance 30s, past tolerance 300s
    CLOCK_SKEW_TOLERANCE = 30
    if timestamp > now + CLOCK_SKEW_TOLERANCE or now - timestamp > 300:
        logger.warning("verify_canary: timestamp out of range (now=%d, ts=%d)", now, timestamp)
        return None

    # R4-CRIT-3: fail-closed in production — only allow dev fallback in dev mode.
    # Use the shared is_dev_env() helper so "dev"/"development" (case-insensitive)
    # are all accepted — consistent with assert_session_secret_configured().
    if not _get_session_secret():
        from proxy.assertions import is_dev_env
        if not is_dev_env():
            logger.error("PROXY_SESSION_SECRET is not set in production — rejecting canary")
            return None
        return ParsedCanary(
            skill=skill_name,
            timestamp=timestamp,
            sig=sig,
            is_dev_fallback=True,
        )

    # Constant-time HMAC comparison.
    # B-3: state_str is bound into the HMAC input (so state mutation fails
    # verification). session_id is NOT bound (BLOCKER fix — parent and
    # sub-agent have different session_ids by design, so binding would break
    # the legitimate production flow). Empty state_str produces a 3-field
    # message with an empty middle field — there is no legacy 2-field HMAC;
    # canaries have a 5-minute TTL so cross-version replay is not a concern.
    message = f"{skill_name}:{state_str}:{timestamp_str}"
    expected_sig = hmac.new(
        _get_session_secret().encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None

    return ParsedCanary(
        skill=skill_name,
        timestamp=timestamp,
        sig=sig,
        is_dev_fallback=False,
    )


# ---------------------------------------------------------------------------
# STRICT SWARM-ONLY CANARY PRIMITIVE
# ---------------------------------------------------------------------------
# This block is the swarm canary path. It is additive and isolates from
# `sign_canary` / `verify_canary` / `parse_canary` so that legacy
# `^^ticket` / `^^work` behavior remains byte-identical.
#
# Invariants:
#   - `swarm_missing_secret` fires before any HMAC work when the secret
#     is unset/empty (no dev fallback for swarm).
#   - `swarm_signing_failed` is the only "anything went wrong during
#     signing or token validation" code — covers malformed tokens,
#     zero-signature, hex-mismatch, exception-in-sign_canary.
#   - `swarm_state_too_large` enforces the 256-byte cap on the EXACT
#     serialized state JSON bytes (no leading `|`), matching the
#     canonical serializer the verifier uses.
#   - `swarm_invalid_skill_name` and `swarm_invalid_state` document
#     defensive-rejection codes (cheap to emit, impossible to misuse
#     silently).
#   - Every failure raises `SwarmCanaryError`; the builder never
#     returns '', None, a bare marker, a workspace marker, or a
#     zero-signature marker.
# ---------------------------------------------------------------------------

_SWARM_MAX_SERIALIZED_STATE_BYTES: int = 256
_SWARM_NAME_REGEX: "re.Pattern[str]" = re.compile(r"^[a-z][a-z0-9_-]*$")
_SWARM_NAME_MAX_LEN: int = 64


class SwarmCanaryError(RuntimeError):
    """Strict swarm-canary construction failure with a stable code.

    Codes (defined here plus prompt-required additions):
      - ``swarm_missing_secret``        — PROXY_SESSION_SECRET unset/empty.
      - ``swarm_signing_failed``        — token malformed OR HMAC rejected
                                         OR underlying sign_canary raised.
      - ``swarm_state_too_large``       — serialized state JSON > 256 UTF-8 bytes.
      - ``swarm_invalid_skill_name``    — name fails ``^[a-z][a-z0-9_-]*$``
                                         or exceeds 64 chars.
      - ``swarm_invalid_state``         — state shape rejects workspace
                                         invariants.

    Messages are fixed and contain NO secret material, no canary text,
    no signature, no serialized state, no session_id, no exception text.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_HEX_DIGITS: "frozenset[str]" = frozenset("0123456789abcdef")


def _is_hex_64(s: str) -> bool:
    """Return True iff `s` is exactly 64 lowercase/uppercase hex chars."""
    if len(s) != 64:
        return False
    return all(c in _HEX_DIGITS for c in s.lower())


def sign_canary_strict(
    skill_name: str,
    state_str: str = "",
    session_id: str = "",
) -> str:
    """Swarm-only signer (spec §12.6).

    Delegates to :func:`sign_canary` for the actual HMAC and then
    validates the returned token shape. Fails closed:

      - missing/empty ``PROXY_SESSION_SECRET`` → ``swarm_missing_secret``.
      - token malformed (separator missing, sig wrong length, sig not hex,
        sig is the all-zero dev fallback, ts empty / non-digits)
        → ``swarm_signing_failed``.
      - underlying ``sign_canary`` exception → ``swarm_signing_failed``
        (no leak of the original exception text).

    This is THE only signer authorised to mint SKILL canaries for a
    swarm-capable custom-skill parent. It never returns the legacy
    ``{'0'*64}:{ts}`` dev fallback. Built-in ``^^ticket`` / ``^^work``
    continue to call ``sign_canary`` directly, untouched.
    """
    secret = _get_session_secret()
    if not secret:
        raise SwarmCanaryError(
            "swarm_missing_secret",
            "Swarm canary signing requires PROXY_SESSION_SECRET",
        )

    try:
        token = sign_canary(
            skill_name,
            state_str=state_str,
            session_id=session_id,
        )
    except Exception:
        # Underlying signer raised — never leak the original text.
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Internal signer raised an unexpected exception",
        )

    # sign_canary() returns "<sig>:<ts>" (or the dev fallback
    # "0"*64":<ts>" — which we will reject here).
    if not isinstance(token, str):
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Signer did not return a string token",
        )

    sig, separator, ts = token.partition(":")
    if not separator:
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Signer token missing the signature/timestamp separator",
        )
    if not ts:
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Signer token has an empty timestamp",
        )
    if not ts.isdigit():
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Signer token timestamp is not decimal digits",
        )
    # Reject the legacy all-zero dev fallback (this is the only place
    # we re-detect it; sign_canary emits it when secret is empty, but
    # we already gated that above, so any zero-sig token reaching here
    # is a malformed caller — reject).
    if sig == "0" * 64:
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Signer returned the zero-signature dev fallback",
        )
    if not _is_hex_64(sig):
        raise SwarmCanaryError(
            "swarm_signing_failed",
            "Signer returned a non-hex or wrong-length signature",
        )
    return token


def _serialize_swarm_state(state: "Mapping[str, str]") -> str:
    """Serialize a swarm canary state into the canonical ``state_str``.

    Accepts exactly one of:
      - empty mapping → returns ``""``.
      - ``{"workspace_name": <non-empty string>}`` → returns
        ``"|" + json.dumps(state, sort_keys=True)``.

    Rejected (all ``swarm_invalid_state``):
      - None / list / tuple / str / non-mapping values.
      - non-empty mapping missing ``workspace_name``.
      - ``workspace_name`` not a string (int, None, dict, list).
      - ``workspace_name`` empty string (caller must use ``{}`` for
        default workspace; a degenerate empty string is treated as a
        caller bug rather than silently normalised — clean break per
        spec §12.3).
      - Any key other than ``workspace_name``
        (rejects ``workspace_id``, ``parent_skill_name``, ``task_hash``,
        ``subagents``, arbitrary caller keys).
      - Nested non-string values inside the workspace_name key.

    Returns ``"|{json}"`` where the JSON is ``json.dumps(state,
    sort_keys=True)`` with default separators (matches verifier
    exactly at line 488).

    Size check is on the EXACT serialized JSON bytes (``state_json``),
    BEFORE the leading ``|`` is prepended — that mirrors the verifier
    byte-for-byte and prevents the off-by-one at the 256-byte boundary.
    The boundary is bytes, not characters: ``ensure_ascii=True`` in
    json.dumps means non-ASCII workspace names escape to ``\\uXXXX``
    (always ASCII in the output), so byte count == character count
    for the JSON portion.
    """
    if not isinstance(state, dict):
        # Empty dict is allowed (default workspace).
        if state == {}:
            return ""
        raise SwarmCanaryError(
            "swarm_invalid_state",
            "State must be an object",
        )
    if not state:
        return ""

    # Reject any key other than `workspace_name`. This explicitly
    # closes the §12.3 identity leak from parent_skill_name / task_hash
    # / workspace_id / subagents that earlier v2 drafts allowed.
    if set(state.keys()) != {"workspace_name"}:
        raise SwarmCanaryError(
            "swarm_invalid_state",
            "State must contain exactly the workspace_name key",
        )

    workspace_name = state["workspace_name"]
    if not isinstance(workspace_name, str):
        raise SwarmCanaryError(
            "swarm_invalid_state",
            "workspace_name must be a string",
        )
    if workspace_name == "":
        # Clean break: degenerate workspace_name. Callers must pass {}
        # for default workspace, not {"workspace_name": ""}.
        raise SwarmCanaryError(
            "swarm_invalid_state",
            "workspace_name must be non-empty (use {} for default workspace)",
        )

    state_json = json.dumps(state, sort_keys=True)

    # Size check is on the EXACT serialized bytes (no `|` prefix).
    serialized_bytes = state_json.encode("utf-8")
    if len(serialized_bytes) > _SWARM_MAX_SERIALIZED_STATE_BYTES:
        raise SwarmCanaryError(
            "swarm_state_too_large",
            "Swarm canary state exceeds 256 UTF-8 bytes",
        )
    return f"|{state_json}"


def build_swarm_skill_canary(
    skill_name: str,
    state: "Mapping[str, str]",
    session_id: str = "",
) -> str:
    """Strict swarm canary builder (spec §12.1, §12.6).

    Returns the canonical SKILL canary marker
    ``[[__SUBAGENT__:SKILL:<name>:<sig>:<ts>{state_str}]]`` matching the
    ``BaseSkill.build_skill_canary`` marker byte-for-byte. The marker
    shape is identical so the existing parser/verifier can round-trip
    it.

    Failure modes (all raise :class:`SwarmCanaryError`):
      - ``swarm_invalid_skill_name`` — name fails the regex or > 64 chars.
      - ``swarm_invalid_state`` / ``swarm_state_too_large`` — state
        validation failures.
      - ``swarm_missing_secret`` — secret absent.
      - ``swarm_signing_failed`` — HMAC token malformed / zero-sig /
        sign_canary raised.

    The builder NEVER returns '', None, a bare marker, a workspace
    marker, or a zero-signature marker. Every failure raises.
    """
    if not isinstance(skill_name, str) or not _SWARM_NAME_REGEX.match(skill_name):
        raise SwarmCanaryError(
            "swarm_invalid_skill_name",
            "Skill name must match ^[a-z][a-z0-9_-]*$",
        )
    if len(skill_name) > _SWARM_NAME_MAX_LEN:
        raise SwarmCanaryError(
            "swarm_invalid_skill_name",
            "Skill name exceeds 64 characters",
        )

    state_str = _serialize_swarm_state(state)

    token = sign_canary_strict(
        skill_name,
        state_str=state_str,
        session_id=session_id,
    )

    # Marker shape — byte-identical to BaseSkill.build_skill_canary.
    return f"[[__SUBAGENT__:SKILL:{skill_name}:{token}{state_str}]]"


# Canary marker constants (STY-005)
_CANARY_SKILL_PREFIX = "[[__SUBAGENT__:SKILL:"
_CANARY_WORKSPACE_PREFIX = "[[__SUBAGENT__:WORKSPACE:"
_CANARY_BASE = "[[__SUBAGENT__]]"

# Max workspace name length — must stay in sync with skills.workspace._MAX_WORKSPACE_NAME_LENGTH
_MAX_WORKSPACE_NAME_LENGTH = 64

logger = logging.getLogger(__name__)

# Position-based parser (handles canary states containing ']')
#
# The previous regex-based stripper used `(?:\|[^\]]+)?` for the SKILL canary
# state, which cannot match states that contain ']' (e.g. JSON arrays like
# `[{"short_hash": "abc"}]`). The new parser uses `json.JSONDecoder().raw_decode()`
# for JSON state and anchors on the `:64hex:digits` sig:ts format so canary
# states containing ']' are located correctly (the canary-stacking production
# bug). Both `parse_canary` and `_strip_existing_canaries` share this locator.
_CANARY_OPEN = "[[__SUBAGENT__"
# Anchoring on sig:ts avoids ambiguity where a hex-only "name" field would be
# misclassified as the skill name.
_RE_SKILL_SIG_ANCHOR = re.compile(r':([a-f0-9]{64}):(\d+)')




@dataclass
class _CanaryLoc:
    """Structural parse result from :func:`_locate_canary`.

    ``kind`` is ``"bare"`` | ``"skill"`` | ``"workspace"`` for a complete
    canary, or ``None`` for a malformed prefix-only match. ``name``,
    ``sig_ts``, and ``state`` are populated only for complete canaries
    (SKILL/WORKSPACE). ``start``/``end`` are inclusive/exclusive indices
    into the source text.
    """
    start: int
    end: int
    kind: Optional[str]
    name: Optional[str] = None
    sig_ts: Optional[str] = None
    state: Optional[dict] = None


def _locate_canary(text: str, start: int = 0) -> Optional[_CanaryLoc]:
    """Find the next canary marker at or after ``start`` (SOLE position-walker).

    Used by both :func:`parse_canary` and :func:`_strip_existing_canaries`.
    Position-based extraction uses ``_RE_SKILL_SIG_ANCHOR`` and
    ``json.JSONDecoder().raw_decode()`` for JSON state, so canary states
    containing ``]`` or ``|`` are located correctly.

    For a malformed canary (prefix matched but no valid close), returns
    a ``_CanaryLoc(kind=None)`` whose span is the prefix bytes alone.
    """
    idx = text.find(_CANARY_OPEN, start)
    if idx == -1:
        return None

    open_len = len(_CANARY_OPEN)

    # Bare [[__SUBAGENT__]] — literal 2-char close.
    if text.startswith("]]", idx + open_len):
        return _CanaryLoc(start=idx, end=idx + open_len + 2, kind="bare")

    if text.startswith(":SKILL:", idx + open_len):
        return _locate_typed_canary(
            text, idx, idx + open_len + len(":SKILL:"), "skill",
            name_regex=r'^[a-z][a-z0-9_-]*$', max_name_len=64,
            has_state=True, check_dunder=True,
        )

    if text.startswith(":WORKSPACE:", idx + open_len):
        return _locate_typed_canary(
            text, idx, idx + open_len + len(":WORKSPACE:"), "workspace",
            name_regex=r'^[a-zA-Z0-9][a-zA-Z0-9-]*$',
            max_name_len=_MAX_WORKSPACE_NAME_LENGTH,
            has_state=False, check_dunder=False,
        )

    # Unknown canary type — return just the prefix bytes.
    return _CanaryLoc(start=idx, end=idx + open_len, kind=None)


def _locate_typed_canary(
    text: str,
    idx: int,
    name_start: int,
    kind: str,
    *,
    name_regex: str,
    max_name_len: int,
    has_state: bool,
    check_dunder: bool,
) -> _CanaryLoc:
    """Structural parse of a SKILL/WORKSPACE canary (parameterized by kind)."""
    n = len(text)
    open_len = len(_CANARY_OPEN)
    malformed = lambda: _CanaryLoc(start=idx, end=idx + open_len, kind=None)

    m = _RE_SKILL_SIG_ANCHOR.search(text, name_start)
    if not m:
        return malformed()
    colon_pos = text.rfind(":", name_start, m.start() + 1)
    if colon_pos < name_start:
        return malformed()
    name = text[name_start:colon_pos]
    if not re.match(name_regex, name) or len(name) > max_name_len:
        return malformed()
    if check_dunder and (name.startswith('__') or name.endswith('__')):
        return malformed()

    sig_ts_start = colon_pos + 1
    i = m.end()

    # Optional `|state` (only when has_state=True).
    if has_state and i < n and text[i] == "|":
        sig_ts = text[sig_ts_start:i]  # exclude state
        i += 1
        if i < n and text[i] == "{":
            try:
                state, json_end = json.JSONDecoder().raw_decode(text, i)
            except (json.JSONDecodeError, ValueError):
                return malformed()
            if not isinstance(state, dict) or not text.startswith("]]", json_end):
                return malformed()
            return _CanaryLoc(
                start=idx, end=json_end + 2, kind=kind,
                name=name, sig_ts=sig_ts, state=state,
            )
        end = text.find("]]", i)
        if end == -1:
            return malformed()
        return _CanaryLoc(
            start=idx, end=end + 2, kind=kind,
            name=name, sig_ts=sig_ts, state=None,
        )

    # No state — must end with ]].
    if not text.startswith("]]", i):
        return malformed()
    return _CanaryLoc(
        start=idx, end=i + 2, kind=kind,
        name=name, sig_ts=text[sig_ts_start:i], state=None,
    )


def parse_canary(text: str, *, session_id: str = "") -> Optional[CanaryParse]:
    """Parse a canary marker at the START of ``text``.

    Returns a :class:`CanaryParse`, or ``None`` if no canary marker
    (bare / SKILL / WORKSPACE prefix) is present at the start of ``text``.
    This is the ONLY ``None`` case — a canary that is present but
    HMAC-failed / unregistered / structurally malformed is returned with
    ``verified=False`` (graceful degradation: the caller still downgrades
    to sub-agent context, but grants no privilege). Replaces the legacy
    ``parse_skill_canary`` plus the ad-hoc ``find("]]")`` branches and
    bare-marker handling that previously lived in
    ``detect_subagent_ingress``.

    Per kind, ``verified=True`` requires:
      - ``"bare"``: always (no HMAC).
      - ``"skill"``: structural parse + name regex + HMAC + skill registry.
        Order: HMAC verify THEN registry check.
      - ``"workspace"``: structural parse + name regex/length + HMAC.
        NO registry check.

    ``remaining`` is the text after the canary close (leading newlines
    stripped). For a malformed canary, ``remaining`` is the original text.

    ``session_id`` is a cache-scope key (§5.4) — NOT bound to HMAC
    (parent and sub-agent have different session_ids by design).
    """
    if not isinstance(text, str):
        return None

    loc = _locate_canary(text, 0)
    if loc is None or loc.start != 0:
        return None

    # Malformed prefix: prefix matched but body parse failed. Preserve
    # the original "kind=skill/workspace, verified=False" contract so
    # detect_subagent_ingress still sees the canary (is_subagent=True)
    # but grants no privilege.
    if loc.kind is None:
        if text.startswith(_CANARY_SKILL_PREFIX):
            return CanaryParse(kind="skill", remaining=text, verified=False)
        if text.startswith(_CANARY_WORKSPACE_PREFIX):
            return CanaryParse(kind="workspace", remaining=text, verified=False)
        return None

    stripped_remaining = text[loc.end:].lstrip('\n')

    if loc.kind == "bare":
        return CanaryParse(kind="bare", remaining=stripped_remaining, verified=True)

    return _verify_canary(loc, stripped_remaining, session_id=session_id)


# Per-kind config for the body verifier. SKILL canaries carry optional JSON
# state and require a registry check; WORKSPACE canaries do not.
_CANARY_VERIFY_CFG = {
    "skill": {"token_prefix": "", "registry_check": True},
    "workspace": {"token_prefix": "workspace:", "registry_check": False},
}


def _verify_canary(loc: _CanaryLoc, stripped_remaining: str, *, session_id: str = "") -> CanaryParse:
    """HMAC verify (and optional registry check) for a SKILL/WORKSPACE canary."""
    cfg = _CANARY_VERIFY_CFG[loc.kind]

    # Reconstruct state_str deterministically (sort_keys=True) so the HMAC
    # matches the signer — Python dict insertion order is not contractual.
    state_str = ""
    if loc.state:
        try:
            state_str = f"|{json.dumps(loc.state, sort_keys=True)}"
        except (TypeError, ValueError):
            logger.warning("Cannot serialise inheritance state for HMAC binding — failing verification")
            return CanaryParse(kind=loc.kind, remaining=stripped_remaining, verified=False)

    token = f"{cfg['token_prefix']}{loc.name}:{loc.sig_ts}"

    # ── Session-scoped canary verification cache (§5.4) ──────────────────
    # Caches verified canaries keyed by (session_id, sig_hash, state_hash,
    # secret_hash) so long-running sub-agents (>5 min) don't lose their
    # ticket mode when the HMAC freshness window expires.
    cache_key = None
    if session_id:
        sig_part = loc.sig_ts.rsplit(":", 1)[0] if ":" in (loc.sig_ts or "") else (loc.sig_ts or "")
        sig_hash = hashlib.sha256(sig_part.encode()).hexdigest()
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        secret = _get_session_secret()
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()[:16] if secret else "dev"
        cache_key = (session_id, sig_hash, state_hash, secret_hash)

        import proxy.state as state
        with state.canary_verify_cache_lock:
            entry = state.canary_verify_cache.get(cache_key)
            if entry is not None:
                cached_verified, cached_skill_name, cached_state, cached_dev_fallback, cached_ts = entry
                now = time.time()
                if now - cached_ts <= state._CANARY_VERIFY_CACHE_TTL:
                    # F4: LRU — touch entry on hit so it's the freshest.
                    state.canary_verify_cache.move_to_end(cache_key)
                    # C1: return the CACHED state verbatim — do NOT re-parse
                    # loc.state. A mutated state suffix produces a different
                    # state_hash → cache miss → full HMAC verify rejects it.
                    return CanaryParse(
                        kind=loc.kind,
                        remaining=stripped_remaining,
                        verified=cached_verified,
                        skill_name=cached_skill_name if loc.kind == "skill" else None,
                        workspace_name=loc.name if loc.kind == "workspace" else None,
                        state=cached_state,
                        is_dev_fallback=cached_dev_fallback,
                    )
                # expired entry — evict
                state.canary_verify_cache.pop(cache_key, None)

    # HMAC verify (session_id is used as a cache-scope key (§5.4) — NOT
    # bound to HMAC (BLOCKER fix: parent and sub-agent have different
    # session_ids by design)).
    pc = verify_canary(token, state_str=state_str, session_id=session_id)
    if pc is None:
        logger.warning("Canary signature verification failed for %s '%s'", loc.kind, loc.name)
        return CanaryParse(kind=loc.kind, remaining=stripped_remaining, verified=False)

    if cfg["registry_check"]:
        # SEC-006: registry check (skill only, AFTER HMAC).
        from proxy.skills.skill_registry import SkillRegistry
        if SkillRegistry().get(loc.name) is None:
            logger.warning("Unregistered skill '%s' in canary — rejected", loc.name)
            return CanaryParse(kind=loc.kind, remaining=stripped_remaining, verified=False)

    # Populate cache on successful verification (HMAC OR dev-fallback — H1).
    if cache_key is not None:
        import copy
        import proxy.state as state
        with state.canary_verify_cache_lock:
            while len(state.canary_verify_cache) >= state._CANARY_VERIFY_CACHE_MAX:
                state.canary_verify_cache.popitem(last=False)
            state.canary_verify_cache[cache_key] = (
                True,
                loc.name if loc.kind == "skill" else None,
                copy.deepcopy(loc.state) if loc.state else None,
                pc.is_dev_fallback,
                time.time(),
            )

    if loc.kind == "workspace":
        return CanaryParse(
            kind="workspace",
            remaining=stripped_remaining,
            verified=True,
            workspace_name=loc.name,
            is_dev_fallback=pc.is_dev_fallback,
        )
    return CanaryParse(
        kind="skill",
        remaining=stripped_remaining,
        verified=True,
        skill_name=loc.name,
        state=loc.state,
        is_dev_fallback=pc.is_dev_fallback,
    )


def _ingress_from_parse(p: CanaryParse) -> tuple:
    """Translate a :class:`CanaryParse` into SubagentIngress field values.

    Returns ``(subagent_ticket_mode, inherited_skill, workspace_name, task_hash)``.
    Privilege (skill/workspace inheritance) is granted ONLY when
    ``verified``; bare or unverified canaries yield no privilege
    (graceful degradation — the caller still marks ``is_subagent=True``).
    """
    if p.kind == "skill" and p.verified:
        ws = p.state.get("workspace_name", "") if p.state else ""
        th = p.state.get("task_hash") if p.state else None
        return (p.skill_name == "ticket", p.skill_name, ws, th)
    if p.kind == "workspace" and p.verified:
        return (False, None, p.workspace_name or "", None)
    # bare, OR skill/workspace with verified=False
    return (False, None, "", None)


def detect_subagent_ingress(body: dict, *, session_id: str = "") -> SubagentIngress:
    """Detect sub-agent ingress from a parsed request body.

    Inspects messages[0].content for [[__SUBAGENT__:SKILL:...]],
    [[__SUBAGENT__:WORKSPACE:...]], or bare [[__SUBAGENT__]] markers,
    strips the prefix, and sets is_subagent=True with the parsed
    inherited_skill / workspace_name.

    Side effects: mutates messages[0]['content'] in place to strip
    the canary prefix. Caller is responsible for thread-safety.

    Security: skill/workspace INHERITANCE is gated on the HMAC
    signature inside the canary (verified by ``parse_canary`` — a
    canary present but unverified still sets ``is_subagent=True`` but
    grants no privilege).

    ``session_id`` is a cache-scope key (§5.4) — NOT bound to HMAC
    (parent and sub-agent have different session_ids by design).
    """
    result = SubagentIngress()
    messages = body.get("messages") or []

    # DIAG (redacted): log structural fields only at DEBUG. NEVER log the
    # first_user content / prompt / signature / session_id / canary markers.
    _first_user_len = 0
    for _em in messages:
        if isinstance(_em, dict) and _em.get("role") == "user":
            _fcontent = extract_text_content(_em.get("content", ""))
            if isinstance(_fcontent, str):
                _first_user_len = len(_fcontent)
            break
    logger.debug(
        "detect_ingress raw_msg_count=%d first_content_len=%d code=ok",
        len(messages),
        _first_user_len,
    )

    for _msg_idx, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        content = extract_text_content(m.get("content", ""))

        # SINGLE parser path. parse_canary returns None only when NO canary
        # marker is at the start; a present-but-unverified canary yields a
        # CanaryParse with verified=False (graceful degradation).
        p = parse_canary(content, session_id=session_id)
        # Bounded bare-recovery (max 2 iterations = 1 recovery level):
        # a client may prepend [[__SUBAGENT__]] before the real skill/workspace
        # canary. If the first parse is bare and a second canary follows,
        # promote to the second parse. Do NOT loop further (pins current
        # single-level recovery behaviour).
        if p is not None and p.kind == "bare":
            p2 = parse_canary(p.remaining, session_id=session_id)
            if p2 is not None:
                logger.debug(
                    "detect_subagent_ingress: bare canary followed by %s canary — promoting (index=%d)",
                    p2.kind, _msg_idx,
                )
                p = p2

        if p is None:
            logger.debug("detect_subagent_ingress: no canary prefix matched on first user message (index=%d)", _msg_idx)
            break  # not a canary at all — leave the message untouched

        # LOAD-BEARING INVARIANT: is_subagent is set whenever a canary was
        # found (parse is not None), REGARDLESS of verified. A forged/expired
        # canary still downgrades to sub-agent context (loses tools); only
        # privilege inheritance is gated on `verified`.
        result.is_subagent = True
        ticket_mode, inherited_skill, ws, th = _ingress_from_parse(p)
        result.subagent_ticket_mode = ticket_mode
        result.inherited_skill = inherited_skill
        result.workspace_name = ws
        result.task_hash = th

        # Mutate the message content (replace the list entry — matches prior
        # behaviour; do NOT switch to in-place mutation).
        _m_copy = dict(m)
        _m_copy["content"] = p.remaining
        messages[_msg_idx] = _m_copy
        logger.debug(
            "detect_subagent_ingress: matched %s canary (index=%d, verified=%s)",
            p.kind, _msg_idx, p.verified,
        )
        break  # only check the first user message

    if result.is_subagent:
        logger.debug(
            "Sub-agent detected via [[__SUBAGENT__]] canary (ticket_mode=%s)",
            result.subagent_ticket_mode,
        )

    # DEBUG: log the final SubagentIngress fields.
    logger.debug(
        "detect_subagent_ingress: returning SubagentIngress(is_subagent=%s, subagent_ticket_mode=%s, inherited_skill=%r, workspace_name=%r)",
        result.is_subagent, result.subagent_ticket_mode, result.inherited_skill, result.workspace_name,
    )

    return result


def _strip_existing_canaries(text: str) -> str:
    """Remove all canary markers to prevent stacking.

    Per §11.3: handles all formats to prevent nested canary stacking.
    Uses :func:`_locate_canary` (the shared position-based parser) so that
    canary states containing ``]`` — e.g. JSON arrays in the inheritance
    state — are stripped correctly.
    """
    out = []
    i = 0
    n = len(text)
    open_len = len(_CANARY_OPEN)
    while i < n:
        loc = _locate_canary(text, i)
        if loc is None:
            out.append(text[i:])
            break
        if loc.start > i:
            out.append(text[i:loc.start])
        # Malformed prefix (kind=None) → keep those bytes, advance past.
        if loc.kind is None:
            out.append(text[loc.start:loc.end])
        # Valid canary → skip it entirely (don't append).
        i = loc.end
    return "".join(out)


def _inject_canary_into_task_calls(
    data: dict,
    canary: str,
    *,
    request_ctx: Optional[object] = None,
    default_mode_name: Optional[str] = None,
) -> dict:
    """Inject canary into prompt argument of task tool calls.

    Per §11.3: iterates tool_calls, finds 'task' function calls,
    strips existing canary markers, and prepends new canary to prompt.

    Per-task override via ``subagent_mode`` arg (US-017c extension):
      - ``"ticket"``: use the default canary (no escalation needed)
      - ``"work"``: when default_mode_name is "ticket", escalate to a work canary
        built via ``WorkSkill.get_canary``. Otherwise use the default canary
        (parent already in work mode — no escalation needed).
      - any other value (or missing): use the default canary; invalid values
        log a WARNING and fall back to the default.

    Args:
        data: The response data dict with choices[0].message.tool_calls
        canary: The canary string to inject (e.g., "[[__SUBAGENT__:SKILL:ticket]]")
        request_ctx: Optional RequestContext — used to build a SkillContext for
            ``WorkSkill.get_canary`` when escalating to a work sub-agent.
            When None, a minimal SkillContext is built (no task_hash, no
            workspace — produces a bare work canary).
        default_mode_name: Optional parent mode name (e.g., "ticket", "work").
            When "ticket" and subagent_mode="work" is requested, the canary
            is escalated. When "work" (or any other value), no escalation.

    Returns:
        Modified data dict with canary prepended to task prompts.
    """
    if not isinstance(data, dict) or "choices" not in data:
        return data

    try:
        native_tc_list = data.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
    except (IndexError, KeyError, TypeError):
        return data

    # M4 #4: count one escalation per RESPONSE, not per tool call.
    # The first matched work-mode tc escalates and consumes one rate-limit
    # slot (via _resolve_task_canary's real check() — see its docstring);
    # subsequent work-mode tcs in the same response fall back to the
    # default canary without re-calling the function (which would
    # otherwise double-count slots). The escalation_already_done flag is
    # the per-response gate.
    escalation_already_done = False
    injected_count = 0
    for tc in native_tc_list:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        if func.get("name") != "task":
            continue
        args_raw = func.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        prompt = args.get("prompt", "")
        if not isinstance(prompt, str):
            continue

        if escalation_already_done:
            # M4 #4: the response already escalated once. Don't
            # re-call _resolve_task_canary (which would re-commit a
            # slot). Just use the default canary.
            chosen_canary = canary
            escalation_kind = None
        else:
            chosen_canary, escalation_kind = _resolve_task_canary(
                default_canary=canary,
                args=args,
                default_mode_name=default_mode_name,
                request_ctx=request_ctx,
            )
            if escalation_kind == "work":
                escalation_already_done = True

        prompt = _strip_existing_canaries(prompt)
        args["prompt"] = chosen_canary + prompt
        func["arguments"] = json.dumps(args)
        injected_count += 1
        if escalation_kind:
            _request_id = getattr(request_ctx, "request_id", "") if request_ctx is not None else ""
            logger.warning(
                "canary_subagent_mode_escalation "
                "subagent_mode=%r default_mode=%r kind=%s",
                args.get("subagent_mode"),
                default_mode_name,
                escalation_kind,
                extra={"request_id": _request_id},
            )

    if injected_count:
        _request_id = getattr(request_ctx, "request_id", "") if request_ctx is not None else ""
        logger.debug(
            "canary_inject task_count=%d request_id=%s code=ok",
            injected_count,
            _request_id,
            extra={"request_id": _request_id},
        )

    return data


def _resolve_task_canary(
    *,
    default_canary: str,
    args: dict,
    default_mode_name: Optional[str],
    request_ctx: Optional[object],
) -> tuple[str, Optional[str]]:
    """Resolve which canary to inject for a single task tool call.

    Returns (canary, escalation_kind) where escalation_kind is one of:
      - ``None``: default canary used (no override or no escalation needed)
      - ``"work"``: escalated to a work canary (parent was ticket)
      - ``"rate_limited"``: ticket->work escalation refused by per-session rate
        limiter (B-5); falls back to the default canary so the sub-agent
        inherits the parent's ticket mode instead of receiving work tools
      - ``"invalid"``: invalid subagent_mode value, fell back to default

    This function is THE rate-limit boundary for ticket->work escalation.
    On a successful escalation (work canary built), it commits one slot
    via ``state.escalation_rate_limiter.check()`` so the per-session
    5/hr budget is honored for direct callers too. The build-failure
    path does NOT consume a slot (M4 #3 invariant): if
    ``_build_work_canary_for_request`` returns ``None``, the function
    returns the default canary without calling ``check()``.

    The per-response cap (M4 #4 — one slot per response even when the
    LLM emits multiple task tool calls) is enforced by the CALLER
    (``_inject_canary_into_task_calls``) via its
    ``escalation_already_done`` flag, which gates whether this function
    is invoked for subsequent tcs in the same response. Direct callers
    of this function (unit tests) get one commit per call.
    """
    subagent_mode = args.get("subagent_mode")
    if not isinstance(subagent_mode, str) or not subagent_mode:
        return default_canary, None
    if subagent_mode == "ticket":
        return default_canary, None
    if subagent_mode == "work":
        if default_mode_name != "ticket":
            return default_canary, None
        work_canary = _build_work_canary_for_request(request_ctx)
        if work_canary:
            _session_id = ""
            if request_ctx is not None:
                _session_id = getattr(request_ctx, "session_id", "") or ""
            if _session_id:
                # M4 #3 + M4-3 followup: commit the slot via the real
                # check() so direct callers (unit tests) and the
                # production path both honor the 5/hr cap. check()
                # under _lock atomically peeks + appends; if denied,
                # the slot stays free and we fall back to the default
                # canary. The build above already returned a truthy
                # canary, so the commit here is well-defined.
                import proxy.state as state
                _rl = state.escalation_rate_limiter.check(_session_id)
                if not _rl.allowed:
                    _req_id = getattr(request_ctx, "request_id", None) if request_ctx is not None else None
                    _extra = {"request_id": _req_id} if _req_id else {}
                    logger.warning(
                        "canary: subagent_mode=work escalation rate-limited "
                        "(session=%s, retry_after=%.1fs) — falling back to default canary",
                        _session_id[:8], _rl.retry_after,
                        extra=_extra,
                    )
                    return default_canary, "rate_limited"
            return work_canary, "work"
        return default_canary, None
    _request_id = getattr(request_ctx, "request_id", None) if request_ctx is not None else None
    if _request_id:
        logger.warning(
            "canary_subagent_mode_invalid subagent_mode=%r allowed=ticket,work",
            subagent_mode,
            extra={"request_id": _request_id},
        )
    else:
        logger.warning(
            "canary_subagent_mode_invalid subagent_mode=%r allowed=ticket,work",
            subagent_mode,
        )
    return default_canary, "invalid"


class _SwarmTransformError(Exception):
    """Internal swarm transform failure (mapped to StageAbort at the call site).

    Carries a stable ``code`` from the runtime ``swarm_*`` namespace.
    The caller's job is to translate this into a
    ``StageAbort((swarm_error(...), status))``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def transform_swarm_task_calls(
    response_data: dict,
    *,
    canary: str,
    max_task_calls: int,
) -> dict:
    """Atomic detached transform of native task tool calls in a swarm response (spec §11.1).

    Algorithm — TWO phases:

    Phase 1 (read-only validation + precomputation):
      1. Read response_data["choices"][0]["message"]["tool_calls"].
      2. For each entry, identify native task calls (function.name == "task").
      3. Non-task sibling entries are recorded for byte-for-byte preservation.
      4. **BEFORE** argument validation: count all entries with function.name == "task".
         If count > max_task_calls, raise _SwarmTransformError(code="swarm_max_calls_exceeded").
         The count mutation happens first; no per-call validation precedes it.
      5. For each native task entry, validate structure (shape checks).
      6. Strip existing canaries from the prompt using _strip_existing_canaries.
      7. Precompute new_prompt = canary + stripped_prompt and new_arguments string.

    Phase 2 (commit, only after every call validates):
      1. transformed = copy.deepcopy(response_data).
      2. Apply every precomputed new_arguments string into the copy.
      3. Return transformed.

    Invariants:
      - Source response_data is byte-for-byte unchanged.
      - Failure on the second or later task leaves the first call and source unchanged.
      - Non-task sibling entries preserve object content + ordering.
      - Empty task list returns identity == source dict.
      - Do not shallow-copy nested structures.
    """
    if not isinstance(response_data, dict):
        return response_data

    # Phase 1: gather validation data + precompute new argument strings.
    try:
        choices = response_data.get("choices") or []
        if not choices:
            return response_data
        first = choices[0]
        if not isinstance(first, dict):
            return response_data
        message = first.get("message") or {}
        if not isinstance(message, dict):
            return response_data
        tc_list = message.get("tool_calls") or []
        if not isinstance(tc_list, list):
            return response_data
    except (IndexError, KeyError, TypeError, AttributeError):
        return response_data

    if not tc_list:
        return response_data

    # Step 4 — count FIRST, before any per-call validation.
    raw_count = 0
    for tc in tc_list:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if isinstance(func, dict) and func.get("name") == "task":
            raw_count += 1
    if raw_count > max_task_calls:
        raise _SwarmTransformError(
            "swarm_max_calls_exceeded",
            f"Task call count {raw_count} exceeds max {max_task_calls}",
        )

    # Step 5 + 6 + 7 — validate every task entry and precompute new arguments.
    # precomputed_args: list of (tc_index, new_arguments_string_or_None) preserving
    # tool_calls order. None for non-task entries (untouched).
    precomputed_args: list = [None] * len(tc_list)

    for idx, tc in enumerate(tc_list):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not (isinstance(func, dict) and func.get("name") == "task"):
            continue

        # Shape validation. Any structural failure aborts the whole response.
        tc_id = tc.get("id")
        if not isinstance(tc_id, str) or not tc_id:
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call missing string id",
            )
        tc_type = tc.get("type")
        if tc_type is not None and tc_type != "function":
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call type must be 'function' or absent",
            )
        if not isinstance(func, dict):
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call function must be an object",
            )
        if func.get("name") != "task":
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call function.name must be 'task'",
            )

        args_raw = func.get("arguments")
        if not isinstance(args_raw, str):
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call arguments must be a JSON string",
            )
        try:
            decoded = json.loads(args_raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call arguments JSON is malformed",
            )
        if not isinstance(decoded, dict):
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call decoded arguments must be an object",
            )

        prompt = decoded.get("prompt")
        if not isinstance(prompt, str):
            raise _SwarmTransformError(
                "swarm_malformed_task_call",
                "Task tool call prompt must be a string",
            )

        stripped = _strip_existing_canaries(prompt)
        new_prompt = canary + stripped
        new_decoded = dict(decoded)
        new_decoded["prompt"] = new_prompt
        precomputed_args[idx] = (new_decoded, json.dumps(new_decoded, ensure_ascii=False))

    # Phase 2: commit — deepcopy first, then apply precomputed values.
    import copy as _copy
    transformed = _copy.deepcopy(response_data)
    try:
        t_first = transformed["choices"][0]
        t_message = t_first["message"]
        t_tc_list = t_message["tool_calls"]
    except (IndexError, KeyError, TypeError):
        # The deepcopy shouldn't change shape but be defensive.
        raise _SwarmTransformError(
            "swarm_malformed_task_call",
            "Deep-copied response shape changed unexpectedly",
        )

    for idx, pre in enumerate(precomputed_args):
        if pre is None:
            continue
        _, new_args_str = pre
        try:
            tc = t_tc_list[idx]
        except IndexError:
            break
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")
        if not isinstance(func, dict):
            continue
        func["arguments"] = new_args_str

    return transformed


def _build_work_canary_for_request(request_ctx: Optional[object]) -> Optional[str]:
    """Build a work canary using ``WorkSkill.get_canary`` with parent's state.

    Uses ``SkillContext.from_request_context`` (the canonical factory used by
    skill_dispatcher / skill_prompts / dispatch.py) to build the base context
    from ``request_ctx``, then overlays the parent's active ``ModeState.skill_state``
    (e.g., ``task_hash``) so the escalated work sub-agent inherits the parent's
    skill-level data.

    Note: per Tier-1 reviewer audit, the previous implementation read
    ``request_ctx.skill_state`` and ``ctx._mode.skill_state`` -- both dead code
    (``RequestContext`` has no ``skill_state`` field; ``ctx._mode`` is a
    ``Mode`` enum, not a ``ModeState``). This implementation correctly reads
    the parent's active ``ModeState`` via ``agent.snapshot_mode()``.

    Note: ``agent`` is gated by ``isinstance(agent, Agent)`` rather than
    ``hasattr(agent, "snapshot_mode")``. The latter is unsafe because
    ``MagicMock`` auto-vivifies any attribute and reports ``hasattr`` as
    True -- a bare ``MagicMock`` ``ctx.agent`` would silently fall through
    to a fake ``snapshot_mode`` returning a ``MagicMock`` whose
    ``skill_state`` ``getattr`` is also truthy, producing a canary with
    bogus state. ``isinstance`` rejects ``MagicMock`` (it is not an ``Agent``
    subclass) so the production fallback path ``agent_registry.get_or_load``
    runs instead.
    """
    from dataclasses import replace

    from proxy.agent.agent import Agent
    from proxy.skills.skill_registry import SkillContext

    if request_ctx is None:
        skill_ctx = SkillContext(session_id="", agent_id="")
    else:
        skill_ctx = SkillContext.from_request_context(request_ctx)

        try:
            agent = getattr(request_ctx, "agent", None)
            if agent is None:
                from proxy.agent.agent_registry import registry as _agent_registry
                agent = _agent_registry.get_or_load(request_ctx.session_id)

            if agent is not None and isinstance(agent, Agent):
                ms = agent.snapshot_mode()
                if ms is not None:
                    ms_skill_state = getattr(ms, "skill_state", None)
                    if ms_skill_state:
                        skill_ctx = replace(
                            skill_ctx,
                            skill_state=dict(ms_skill_state),
                        )
        except Exception:
            pass

    try:
        from skills.work_skill import WorkSkill
        return WorkSkill().get_canary(skill_ctx)
    except Exception as exc:
        logger.warning(
            "canary: WorkSkill.get_canary failed, falling back to default canary: %s",
            exc,
            extra={"request_id": getattr(request_ctx, "request_id", None) if request_ctx else None},
        )
        return None
