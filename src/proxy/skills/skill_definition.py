# Re-export ModeState from proxy.state (third canonical home, Phase 2A).
# ModeState is defined in proxy.agent.types; proxy.state re-exports it;
# skill_definition re-exports from proxy.state to establish the import chain.
from proxy.state import ModeState

# Standard library
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API surface (HIGH fix 8: __all__ export list).
# ---------------------------------------------------------------------------
# Pin the names that downstream modules may import via
# `from proxy.skills.skill_definition import ...`. `from X import *` is
# not used internally, but the list also documents the module's public
# surface for static analysis tools (mypy strict, pyright).
__all__ = [
    "SkillDefinition",
    "BaseSkill",
    "SkillRequestContext",
    "ModeState",
    "ProxyAccessLevel",
    "TRUSTED_BUILTIN_SKILLS",
    "is_trusted_skill",
    "SkillRegistrationError",
    "SkillValidationError",
    "TrustToken",
    "SubagentPolicy",
    "_held_trust_token",
    "_TRUSTED_CALL_SITES",
    "_is_exempt_from_callsite_check",
]


# ---------------------------------------------------------------------------
# TrustToken — moved from skill_registry (Phase 1b)
# Re-exported here so callers can `from proxy.skills.skill_definition import TrustToken`
# without a circular dependency on skill_registry.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Confused-deputy defense (R3-HIGH-2 fix).
# ---------------------------------------------------------------------------
# `for_trusted_caller()` historically checked only that the immediate
# caller's module `__name__` was in `TRUSTED_TRUST_CALLERS` (and, after
# R3-HIGH-1, that the caller's `f_code.co_filename` matched that module's
# `__file__` on `sys.modules`). Neither check stops a WRAPPER function
# defined inside a TRUSTED production module from minting a token on
# behalf of an untrusted caller: the wrapper's frame has the trusted
# module's `__name__` AND the trusted module's `co_filename`, so both
# gates pass.
#
# This allowlist pins the call site to a known-safe entry-point FUNCTION
# name. A wrapper with a different `co_name` in the same trusted module
# is rejected. Adding a new production call site is a security-relevant
# change that MUST add the entry-point function name here.
#
# Test modules (`tests.*`) are exempt: they need to mint tokens from
# arbitrary test function names, and they are already trusted by being
# in `TRUSTED_TRUST_CALLERS`. The exemption keeps the production
# confused-deputy defense without breaking test ergonomics.
_TRUSTED_CALL_SITES: frozenset[str] = frozenset({
    "populate_builtins",
})


def _is_exempt_from_callsite_check(caller_name: str) -> bool:
    """Return True if `caller_name` is exempt from the call-site allowlist.

    Test modules (`tests.*`) are exempt — they are trusted via
    `TRUSTED_TRUST_CALLERS` and need to call `for_trusted_caller()` from
    arbitrary test function names. Production modules (`proxy.*`,
    `__main__`, etc.) are subject to the `_TRUSTED_CALL_SITES` check.

    Exposed as a module-level function (not an inline check) so tests can
    patch it to simulate a non-exempt production caller.
    """
    return caller_name.startswith("tests.")


class TrustToken:
    """Unforgeable trust marker for `register_skill(..., trusted_registrant=...)`.

    The token cannot be self-asserted. The only way to obtain a TRUSTED
    instance is via `TrustToken.for_trusted_caller()`, which inspects
    the call stack to verify the caller's module `__name__` is in
    `TRUSTED_TRUST_CALLERS`. Otherwise the only public value is
    `TrustToken.UNTRUSTED`.

    Security: the constructor is private (`_verified` is a positional
    arg with a leading underscore) so external callers cannot mint
    their own TRUSTED tokens.

    Confused-deputy defense (R3-HIGH-2): `for_trusted_caller()` ALSO
    requires that the immediate caller's function name (`co_name`) is in
    `_TRUSTED_CALL_SITES` when the caller is a PRODUCTION trusted module.
    Test modules are exempt (see `_is_exempt_from_callsite_check`). This
    stops a wrapper function in a trusted production module from minting
    a token on behalf of an untrusted caller.
    """

    __slots__ = ("_verified",)

    UNTRUSTED: "TrustToken"

    def __init__(self, _verified: bool) -> None:
        # HIGH-1 R1 fix: constructor guard. The only legitimate callers
        # of `TrustToken(_verified=...)` are inside this module:
        #   - `TrustToken.UNTRUSTED = TrustToken(_verified=False)` at
        #     module load time (line below).
        #   - `TrustToken.for_trusted_caller()` → `cls(_verified=True)`
        #     (defined in this module).
        # External callers (e.g., a test that does
        # `TrustToken(_verified=True)` directly) are now blocked with
        # `PermissionError`. The only public surface for obtaining a
        # TRUSTED token from outside this module is
        # `TrustToken.for_trusted_caller()`, which itself enforces the
        # `TRUSTED_TRUST_CALLERS` allowlist.
        #
        # R3-HIGH-1 fix: synthetic-frame defense. `__name__` alone is
        # forgeable via `exec(code, globals_with_fake___name__)`. A
        # synthetic frame, however, CANNOT simultaneously spoof
        # `f_code.co_filename` to match the real module on `sys.modules`.
        # Comparing the live `sys.modules['proxy.skills.skill_definition'].__file__`
        # to the caller's `f_code.co_filename` catches this attack.
        import inspect
        import sys as _sys
        caller_frame = inspect.stack()[1]
        try:
            caller_module = caller_frame.frame.f_globals.get("__name__", "")
            if caller_module != "proxy.skills.skill_definition":
                raise PermissionError(
                    "TrustToken(): cannot instantiate with _verified. "
                    "Use TrustToken.for_trusted_caller() (for trusted callers) "
                    "or TrustToken.UNTRUSTED (for untrusted callers)."
                )
            # R3-HIGH-1 fix: file-mismatch check. A genuine caller in
            # proxy.skills.skill_definition has f_code.co_filename equal
            # to that module's __file__ on sys.modules. A synthetic
            # frame forged via exec() cannot satisfy both conditions.
            real_mod = _sys.modules.get("proxy.skills.skill_definition")
            if real_mod is None or getattr(real_mod, "__file__", None) != caller_frame.frame.f_code.co_filename:
                raise PermissionError(
                    "TrustToken(): caller module file mismatch (synthetic frame?)"
                )
        finally:
            del caller_frame
        # Use object.__setattr__ to bypass the post-construction
        # `__setattr__` guard added below (the guard blocks ALL
        # post-construction writes to `_verified`, including the
        # assignment in this constructor). The constructor itself is
        # already gated by the inspect.stack() check above (only callers
        # in `proxy.skills.skill_definition` may instantiate), so using
        # the bypass here is safe.
        object.__setattr__(self, "_verified", bool(_verified))

    @property
    def is_trusted(self) -> bool:
        return self._verified

    @classmethod
    def for_trusted_caller(cls) -> "TrustToken":
        """Mint a TRUSTED token iff the immediate caller's module is trusted.

        Confused-deputy defense (R3-HIGH-2): additionally requires that
        the immediate caller's function name (`co_name`) is in
        `_TRUSTED_CALL_SITES` when the caller is a PRODUCTION trusted
        module. Test modules are exempt. A wrapper function defined
        inside a trusted module but with a function name NOT in the
        allowlist is rejected — this stops the wrapper from minting a
        token on behalf of an untrusted caller.
        """
        # Deferred import to avoid circular dependency with skill_registry.
        # TRUSTED_TRUST_CALLERS lives in skill_registry to keep the security
        # boundary definition in one place.
        from proxy.skills.skill_registry import TRUSTED_TRUST_CALLERS
        import inspect
        import sys as _sys
        try:
            caller_frame = inspect.stack()[1]
            caller_name = caller_frame.frame.f_globals.get("__name__", "")
            caller_func = caller_frame.frame.f_code.co_name
        finally:
            del caller_frame
        if caller_name not in TRUSTED_TRUST_CALLERS:
            raise PermissionError(
                f"TrustToken.for_trusted_caller(): caller {caller_name!r} "
                f"is not in TRUSTED_TRUST_CALLERS"
            )
        # R3-HIGH-2 fix: confused-deputy defense. Even if the caller's
        # module is trusted, the caller's FUNCTION must be a known-safe
        # entry point (in `_TRUSTED_CALL_SITES`). A wrapper function in
        # the same trusted module has the same `__name__` and the same
        # `co_filename` as the legit entry point, so the existing checks
        # do NOT catch it. The `co_name` allowlist does. Test modules are
        # exempt (they call from arbitrary test function names).
        if not _is_exempt_from_callsite_check(caller_name):
            if caller_func not in _TRUSTED_CALL_SITES:
                raise PermissionError(
                    f"TrustToken.for_trusted_caller(): caller function "
                    f"{caller_func!r} in trusted module {caller_name!r} "
                    f"is not in _TRUSTED_CALL_SITES (confused-deputy?)"
                )
        # R3-HIGH-1 fix: synthetic-frame defense (symmetric to __init__).
        # The allowlist check on `caller_name` alone is forgeable via
        # `exec(code, globals_with_fake___name__)`. A genuine caller in
        # one of the trusted modules has f_code.co_filename equal to
        # that module's __file__ on sys.modules. Mismatch = synthetic
        # frame, reject.
        try:
            caller_frame2 = inspect.stack()[1]
            real_mod = _sys.modules.get(caller_name)
            if real_mod is None or getattr(real_mod, "__file__", None) != caller_frame2.frame.f_code.co_filename:
                raise PermissionError(
                    "TrustToken.for_trusted_caller(): caller module file "
                    "mismatch (synthetic frame?)"
                )
        finally:
            del caller_frame2
        return cls(_verified=True)

    def __repr__(self) -> str:
        return f"TrustToken(is_trusted={self._verified})"

    def __setattr__(self, name, value):
        # Prevent post-construction mutation of `_verified`.
        # Without this guard, an attacker could do
        # `token = TrustToken.UNTRUSTED; token._verified = True` to forge a
        # TRUSTED token and bypass the trust gate.
        if name == "_verified":
            raise AttributeError(
                "TrustToken._verified is immutable (use TrustToken.UNTRUSTED or "
                "TrustToken.for_trusted_caller())"
            )
        object.__setattr__(self, name, value)


TrustToken.UNTRUSTED = TrustToken(_verified=False)


# ---------------------------------------------------------------------------
# §7.1 — SubagentPolicy (custom-skill research swarms spec)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubagentPolicy:
    """Immutable per-skill sub-agent spawning policy (spec §7.1).

    Constructed only by the custom-skill loader. The published value
    is carried into SkillDefinition.subagents and referenced per-request
    via RequestContext.swarm_policy. Runtime code MUST NOT reopen the
    JSON file, reparse a raw dictionary, or independently validate the
    nested schema (spec §7.1 / AC 16.7).
    """
    tool_whitelist: frozenset[str]
    max_task_calls_per_request: int


# ---------------------------------------------------------------------------
# HIGH fix 4 (R2-6): thread-local holding the current thread's TrustToken.
# ---------------------------------------------------------------------------
# The HIGH R2-6 trust gate at `SkillDefinition` construction reads this
# thread-local. If a TRUSTED `TrustToken` is held AND the new
# `SkillDefinition`'s name is in `TRUSTED_BUILTIN_SKILLS`, construction
# proceeds. Otherwise, construction raises `SkillRegistrationError`.
#
# DESIGN NOTE: this thread-local is the construction-time gate. The
# registry-level gate (`SkillRegistry.register_skill(trusted_registrant=...)`)
# is a separate mechanism (it inspects the call stack). Both gates
# defend the same trust boundary; the construction-time gate is the
# first line of defense (it stops a privileged name from ever being
# instantiated in a context that doesn't hold a token).
_trust_token_local = threading.local()


# ---------------------------------------------------------------------------
# §3.1 — TRUSTED_BUILTIN_SKILLS and is_trusted_skill
# ---------------------------------------------------------------------------
# Module-level trust gate (CAT-3, server-side). The set of skill names that
# are considered "trusted builtin" for proxy_* access control. Modifying this
# set requires BOTH a code change AND a PR review (spec §14 row "Trusted
# builtin set"). `workspace` is REMOVED from the set (R2-Issue 3, option (b)) —
# workspace routing is parser-only, not a registered skill.
TRUSTED_BUILTIN_SKILLS: frozenset[str] = frozenset({"ticket", "work"})


def is_trusted_skill(skill_name: str) -> bool:
    """Return True iff `skill_name` is in TRUSTED_BUILTIN_SKILLS.

    The trust gate is computed at access time from the server-side
    constant. A user cannot forge `is_builtin=True` on a SkillDefinition
    (the field no longer exists; `is_builtin` is a read-only property
    derived from this set).

    This helper is intentionally PUBLIC (no leading underscore) so that
    security-critical call sites MUST import it explicitly at the top
    of the module. A previous draft left this as a private
    `_is_trusted_skill`, which surfaced as a `NameError` at first call
    rather than at module-import time (CRIT-S.2 / N4 CRIT fix).
    """
    return skill_name in TRUSTED_BUILTIN_SKILLS


# ---------------------------------------------------------------------------
# §3.7 — SkillRegistrationError
# ---------------------------------------------------------------------------
class SkillRegistrationError(PermissionError):
    """Raised when a skill registration is denied by the CAT-3 trust-gate.

    Subclass of PermissionError so existing `except PermissionError`
    blocks continue to work. Use this class for narrow catches where
    only skill-registration denials should be handled.
    """
    pass


class SkillValidationError(ValueError):
    """Raised by a skill to signal an expected, user-correctable activation
    failure that should reach the client with a specific HTTP status (e.g.
    an unknown task hash -> 404).

    The ``http_status`` attribute selects the response code (default 404).
    Subclass of ValueError so legacy ``except ValueError`` blocks still
    catch it; dispatch.py checks for this type FIRST to honor
    ``http_status`` before falling through to the generic 500 path for
    genuinely unexpected ValueErrors.

    Reserved for *expected, user-correctable* errors (the user passed a bad
    hash, a missing arg, etc.). Unexpected ValueErrors still map to 500.
    """

    def __init__(self, message: str = "", *, http_status: int = 404) -> None:
        super().__init__(message)
        self.http_status = http_status


# ---------------------------------------------------------------------------
# §3.2 — ProxyAccessLevel enum
# ---------------------------------------------------------------------------
class ProxyAccessLevel(Enum):
    """Per-skill access level for proxy_* tools (R1 CRIT-3 fix).

    Resolved at runtime by `check_sub_agent_proxy_access` (Phase 1c) to
    decide whether a sub-agent in a given skill can call `proxy_ticket`
    or `proxy_work`. The skill owns its own access level; the resolver
    consults the registry, not a hardcoded `if name == "ticket"`.
    """
    NEVER = "never"           # proxy_* not in tool filter
    READ_ONLY = "read_only"   # list_tasks, get_task, append_review_note only
    FULL = "full"             # all actions


# ---------------------------------------------------------------------------
# §3.6 — _held_trust_token context manager
# ---------------------------------------------------------------------------
from contextlib import contextmanager


@contextmanager
def _held_trust_token(token: "TrustToken"):
    """Hold a TrustToken for the duration of a block.

    Used when constructing SkillDefinition instances for trusted builtins
    (e.g., in populate_builtins).  SkillDefinition.__post_init__ checks a
    thread-local to determine whether a trusted name is being registered.
    """
    old = getattr(_trust_token_local, "token", None)
    _trust_token_local.token = token
    try:
        yield token
    finally:
        if old is None:
            try:
                del _trust_token_local.token
            except AttributeError:
                pass
        else:
            _trust_token_local.token = old


# ---------------------------------------------------------------------------
# §3.4 — SkillRequestContext dataclass
# ---------------------------------------------------------------------------
@dataclass
class SkillRequestContext:
    """Carries per-request skill data (R1 HIGH-2 fix, R1-HIGH-5 permanent name).

    Built once per request from the `RequestContext` (the pipeline-side
    object). The pipeline does not pass the `RequestContext` into skill
    callables — it converts to a `SkillRequestContext` so the skill's call
    surface is decoupled from the pipeline's request shape. Skills that
    need raw body access go through `ctx.body`; skills that need the
    active `ModeState` go through `ctx.mode_state`.

    R1-HIGH-5 fix: this class is named `SkillRequestContext` (NOT
    `SkillContext`) **permanently**, not just for the 1a → 1d window.
    The legacy `SkillContext` in `src/proxy/skills/skill_registry.py`
    (lines ~77-185) is the OLD request-shape context; it is still
    constructed by `SkillContext.from_request_context` but new code
    MUST use `SkillRequestContext`. The two classes coexist under
    disambiguated names.
    """
    session_id: str
    is_subagent: bool
    inherited_skill: Optional[str]
    subagent_ticket_mode: bool
    workspace_name: str
    client_hint: Optional[str]
    body: dict
    seeded_tasks: list = field(default_factory=list)
    mode_state: Optional[ModeState] = None

    @classmethod
    def from_request_context(cls, ctx) -> "SkillRequestContext":
        """Build a `SkillRequestContext` from a pipeline `RequestContext`.

        Uses `getattr` with defaults so the helper is robust to the
        pipeline adding new fields without forcing a 1a → 1c refactor.
        The fields listed in the `getattr` chain are the contract —
        `RequestContext` MUST expose them (it does today; the spec
        §6 `from_request_context` definition at lines 109-119 is the
        canonical pipeline-side contract).

        R1-HIGH-2 fix: `mode_state` is forwarded via
        `getattr(ctx, 'mode_state', None)` so a skill that needs the
        active `ModeState` (TTL, allowed_tool_names, sub_agent_tool_filter)
        can read it through `SkillRequestContext.mode_state` without
        reaching back into the pipeline's `RequestContext`. The
        default `None` keeps the helper robust to pipeline
        configurations that do not populate `mode_state` (e.g.,
        pre-skill-stage requests).
        """
        return cls(
            session_id=ctx.session_id,
            is_subagent=getattr(ctx, 'is_subagent', False),
            inherited_skill=getattr(ctx, 'inherited_skill', None),
            subagent_ticket_mode=getattr(ctx, 'subagent_ticket_mode', False),
            workspace_name=getattr(ctx, 'workspace_name', ''),
            client_hint=getattr(ctx, 'client_hint', None),
            body=ctx.body,
            seeded_tasks=getattr(ctx, 'seeded_tasks', []),
            mode_state=getattr(ctx, 'mode_state', None),
        )


# ---------------------------------------------------------------------------
# §3.4a — Capability table data model (C7a: single source of truth for
# tool visibility/allowed/actions). Defined BEFORE SkillDefinition because
# the `capability_table` field annotation references CapabilityTable.
# ---------------------------------------------------------------------------
# ---- C7a: Capability table data model (frozen, pure addition) ----

@dataclass(frozen=True)
class ToolCapability:
    """A single tool's capability for a specific role in a specific skill.

    The structural invariant (enforced in CapabilityTable.__post_init__):
    visible=True ∧ allowed=False (without actions_allowed) is ILLEGAL —
    that's the bait-loop bug class (model sees a tool it can never call).
    """
    tool: str
    role: str  # "main" | "subagent"
    visible: bool
    allowed: bool
    actions_allowed: Optional[frozenset] = None  # None = all actions
    deny_message: Optional[str] = None  # LLM-actionable message when denied


@dataclass(frozen=True)
class CapabilityTable:
    """Immutable table of tool capabilities for a skill.

    Single source of truth for BUILTIN skills (ticket, work). Custom skills
    use legacy ProxyAccessLevel + check_action fallback paths (deferred per
    plan §7.2 — custom-skill table generation from tool_whitelist is a
    future enhancement).

    Lookup methods derive filter, injection, guard, and prompt information
    from this table — making the P1/P2/P5 bug class (visible-but-denied,
    promised-but-filtered, injected-but-blocked) structurally impossible
    for builtin skills.
    """
    entries: tuple

    def __post_init__(self):
        # Structural invariant: visible=True ∧ allowed=False (without
        # actions_allowed) is ILLEGAL — that's the bait-loop bug.
        for cap in self.entries:
            if cap.visible and not cap.allowed and not cap.actions_allowed:
                raise ValueError(
                    f"Bait-loop invariant violation: {cap.tool}/{cap.role} "
                    f"is visible=True but allowed=False with no actions_allowed. "
                    f"This means the model sees a tool it can never call. "
                    f"Set visible=False or provide actions_allowed."
                )

    def for_role(self, role):
        """Return {tool_name: capability} for the given role."""
        return {cap.tool: cap for cap in self.entries if cap.role == role}

    def visible_tools(self, role):
        """Return the set of visible tool names for the role."""
        return frozenset(
            cap.tool for cap in self.entries
            if cap.role == role and cap.visible
        )

    def check_action(self, tool, action, role):
        """Check if a specific action on a tool is allowed.

        Returns None if allowed, deny_message if denied.

        C7.5 fix: when `allowed=False` is combined with `actions_allowed`,
        the table means "allow ONLY these specific actions, deny everything
        else" (partial-allow exception). This was previously a hole —
        `allowed=False` short-circuited to deny_message without consulting
        actions_allowed. Behavior-neutral for all builtin tables (they
        use `allowed=True` with `actions_allowed` or `allowed=False`
        without `actions_allowed`).
        """
        caps = self.for_role(role)
        cap = caps.get(tool)
        if cap is None:
            return f"Tool '{tool}' is not available in this skill mode."
        if not cap.allowed:
            # Partial-allow exception: if actions_allowed is set, allow
            # those specific actions even though the tool-level allowed=False.
            if cap.actions_allowed and action in cap.actions_allowed:
                return None  # This specific action IS allowed
            return cap.deny_message or f"Tool '{tool}' is not allowed."
        # allowed=True
        if cap.actions_allowed is not None and action not in cap.actions_allowed:
            return cap.deny_message or (
                f"Action '{action}' is not allowed for tool '{tool}'. "
                f"Allowed actions: {sorted(cap.actions_allowed)}."
            )
        return None  # allowed

    def is_visible(self, tool, role):
        caps = self.for_role(role)
        cap = caps.get(tool)
        return cap.visible if cap else False

    def injectable_proxy_tools(self, role):
        """Return visible proxy_* tool names for injection."""
        return frozenset(
            cap.tool for cap in self.entries
            if cap.role == role and cap.visible and cap.tool.startswith("proxy_")
        )


# ---- C7a: 4 builtin tables transcribing CURRENT behavior (post-C1–C6.5) ----
# These match the current TicketSkill/WorkSkill filter methods + skill_guard
# READ_ONLY set + tool_guard allowlists exactly. Consistency tests verify.

TICKET_CAPABILITY_TABLE = CapabilityTable(entries=(
    # --- Ticket MAIN ---
    ToolCapability("read", "main", True, True),
    ToolCapability("glob", "main", True, True),
    ToolCapability("grep", "main", True, True),
    ToolCapability("search", "main", True, True),
    ToolCapability("task", "main", True, True),
    ToolCapability("webfetch", "main", True, True),
    # Decomposed ticket tools (Slice 8b — cutover, visible=True)
    ToolCapability("proxy_ticket_list_tasks", "main", True, True),
    ToolCapability("proxy_ticket_get_task", "main", True, True),
    ToolCapability("proxy_ticket_create_tasks", "main", False, False, deny_message="Ticket MAIN cannot create tasks directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_update_task", "main", False, False, deny_message="Ticket MAIN cannot update tasks directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_edit", "main", False, False, deny_message="Ticket MAIN cannot edit tasks directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_set_parent", "main", False, False, deny_message="Ticket MAIN cannot set parent directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_clear_parent", "main", False, False, deny_message="Ticket MAIN cannot clear parent directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_clear_blocked_by", "main", False, False, deny_message="Ticket MAIN cannot clear blocked_by directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_set_priority", "main", False, False, deny_message="Ticket MAIN cannot set priority directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_ticket_append_review_note", "main", False, False, deny_message="Ticket MAIN cannot append review notes directly. Delegate to sub-agents via the task tool."),
    ToolCapability("proxy_work", "main", False, False,
                   deny_message="proxy_work is blocked in ticket mode. Delegate work operations to sub-agents with subagent_mode: 'work'."),

    # --- Ticket SUB ---
    ToolCapability("read", "subagent", True, True),
    ToolCapability("glob", "subagent", True, True),
    ToolCapability("grep", "subagent", True, True),
    ToolCapability("search", "subagent", True, True),
    ToolCapability("webfetch", "subagent", True, True),
    # Decomposed ticket tools (Slice 8b — cutover, visible=True)
    ToolCapability("proxy_ticket_list_tasks", "subagent", True, True),
    ToolCapability("proxy_ticket_get_task", "subagent", True, True),
    ToolCapability("proxy_ticket_create_tasks", "subagent", True, True),
    ToolCapability("proxy_ticket_update_task", "subagent", True, True),
    ToolCapability("proxy_ticket_edit", "subagent", True, True),
    ToolCapability("proxy_ticket_set_parent", "subagent", True, True),
    ToolCapability("proxy_ticket_clear_parent", "subagent", True, True),
    ToolCapability("proxy_ticket_clear_blocked_by", "subagent", True, True),
    ToolCapability("proxy_ticket_set_priority", "subagent", True, True),
    ToolCapability("proxy_ticket_append_review_note", "subagent", True, True),
    ToolCapability("proxy_work", "subagent", False, False,
                   deny_message="Ticket sub-agents cannot call proxy_work."),
    ToolCapability("task", "subagent", False, False,
                   deny_message="Sub-agents cannot spawn sub-sub-agents."),
    ToolCapability("question", "subagent", False, False,
                   deny_message="Not available to sub-agents."),
))


WORK_CAPABILITY_TABLE = CapabilityTable(entries=(
    # --- Work MAIN ---
    ToolCapability("read", "main", True, True),
    ToolCapability("write", "main", True, True),
    ToolCapability("edit", "main", True, True),
    ToolCapability("bash", "main", True, True),
    ToolCapability("glob", "main", True, True),
    ToolCapability("grep", "main", True, True),
    ToolCapability("search", "main", True, True),
    ToolCapability("task", "main", True, True),
    # Decomposed ticket tools (Slice 8b — cutover, visible=True for read/append)
    ToolCapability("proxy_ticket_list_tasks", "main", True, True),
    ToolCapability("proxy_ticket_get_task", "main", True, True),
    ToolCapability("proxy_ticket_append_review_note", "main", True, True),
    ToolCapability("proxy_ticket_create_tasks", "main", False, False, deny_message="Work MAIN cannot create tasks. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_update_task", "main", False, False, deny_message="Work MAIN cannot update tasks. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_edit", "main", False, False, deny_message="Work MAIN cannot edit tasks. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_set_parent", "main", False, False, deny_message="Work MAIN cannot set parent. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_clear_parent", "main", False, False, deny_message="Work MAIN cannot clear parent. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_clear_blocked_by", "main", False, False, deny_message="Work MAIN cannot clear blocked_by. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_set_priority", "main", False, False, deny_message="Work MAIN cannot set priority. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_work", "main", True, True,
                   actions_allowed=None),  # all 7 actions

    # --- Work SUB ---
    ToolCapability("read", "subagent", True, True),
    ToolCapability("write", "subagent", True, True),
    ToolCapability("edit", "subagent", True, True),
    ToolCapability("bash", "subagent", True, True),
    ToolCapability("glob", "subagent", True, True),
    ToolCapability("grep", "subagent", True, True),
    ToolCapability("search", "subagent", True, True),
    # Decomposed ticket tools (Slice 8b — cutover, visible=True for read/append)
    ToolCapability("proxy_ticket_list_tasks", "subagent", True, True),
    ToolCapability("proxy_ticket_get_task", "subagent", True, True),
    ToolCapability("proxy_ticket_append_review_note", "subagent", True, True),
    ToolCapability("proxy_ticket_create_tasks", "subagent", False, False, deny_message="Work sub-agents cannot create tasks. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_update_task", "subagent", False, False, deny_message="Work sub-agents cannot update tasks. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_edit", "subagent", False, False, deny_message="Work sub-agents cannot edit tasks. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_set_parent", "subagent", False, False, deny_message="Work sub-agents cannot set parent. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_clear_parent", "subagent", False, False, deny_message="Work sub-agents cannot clear parent. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_clear_blocked_by", "subagent", False, False, deny_message="Work sub-agents cannot clear blocked_by. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_ticket_set_priority", "subagent", False, False, deny_message="Work sub-agents cannot set priority. Use proxy_work for task lifecycle."),
    ToolCapability("proxy_work", "subagent", True, True,
                   actions_allowed=frozenset({"report_bug", "review_bug", "validate_bug", "resolve_bug", "handoff_task"}),
                   deny_message="Work mode forbids proxy_work lifecycle actions (start_work, stop_work, complete_work)."),
    ToolCapability("task", "subagent", False, False,
                   deny_message="Sub-agents cannot spawn sub-sub-agents."),
    ToolCapability("question", "subagent", False, False,
                   deny_message="Not available to sub-agents."),
))


# ---------------------------------------------------------------------------
# §3.5 — SkillDefinition dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)  # R3-CRIT-2 fix: SkillDefinition is immutable.
# `frozen=True` blocks whole-attribute reassignment (`sd.name = "x"` raises
# FrozenInstanceError) but does NOT block the `is_builtin` @property (a
# property is not a field assignment) and does NOT block the `__post_init__`
# from running. The `is_builtin` field is intentionally NOT a dataclass
# field; the cache immutability test (§5 test 17) relies on the cached
# SkillDefinition being immutable so that callers cannot mutate
# `sd.allowed_tool_names` after `BaseSkill.to_skill_definition()` has
# returned it.
class SkillDefinition:
    """Single source of truth for skill behavior (R1 HIGH-1 fix).

    Splits the data (identity, tool filters, access levels, TTL) from
    the behavior (10 callable fields for prompts, lifecycle, dispatch,
    canary). The 10 callable fields are optional; the resolver
    (`skill_resolver.py`, Phase 1c) checks each one before invoking.

    CRIT (CAT-3): `is_builtin` is NOT a field — it is a read-only
    `@property` computed from `self.name in TRUSTED_BUILTIN_SKILLS`.
    A user-defined skill cannot forge builtin status; passing
    `is_builtin=True` (or `=False`) to the constructor raises
    `TypeError: __init__() got an unexpected keyword argument
    'is_builtin'`. The contract is pinned by
    `test_skill_definition_rejects_is_builtin_kwarg` (§5 test 1).

    Three-state `allowed_tool_names` (Wave 1, spec §6):

      - ``None`` — unrestricted. Reserved for explicitly trusted
        Python-defined skills only (TicketSkill, WorkSkill, and any
        builtin). The custom-skill loader (Agent 2) NEVER passes
        ``None``; it passes ``frozenset()`` for empty whitelists.
      - ``frozenset()`` — deny all tools. The custom-skill loader
        uses this when ``tool_whitelist`` is empty (fail-closed
        per spec §4.4). The LLM receives no tools.
      - Non-empty ``frozenset`` — exact allowlist. The custom-skill
        loader validates every name against
        ``ToolRegistry.get_tool_names()`` and rejects the skill
        if any name is unknown.

    All callers (LLM-facing tool visibility, intercepted memory-tool
    guards, main-agent mode, mode snapshots, tests, docs) MUST use this
    same three-state semantics. No truthiness check may conflate
    ``None`` with ``frozenset()`` — they are distinct sentinels with
    opposite intent (unrestricted vs. deny-all).

    `sub_agent_tool_filter` is NOT three-state — it is always a
    ``frozenset[str]`` (empty or non-empty) because the sub-agent
    filter is only consulted when the active skill's
    `allowed_tool_names` is itself non-empty (otherwise no tools reach
    the sub-agent either).
    """

    # ---- Identity ----
    name: str
    trigger: str
    aliases: tuple = ()
    # Wave 1: short human-readable description (≤160 chars). Surfaced by
    # `^^help` and the JSON-skill loader (Agent 2 parses it from JSON;
    # Agent 4 sets it on builtin skill classes).
    description: str = ""

    # ---- Tool filters (Wave 1: three-state policy — see class docstring) ----
    allowed_tool_names: Optional[frozenset[str]] = None
    # sub_agent_tool_filter stays as frozenset — see class docstring.
    sub_agent_tool_filter: frozenset = field(default_factory=frozenset)

    # ---- Sub-agent permissions (R1 CRIT-3 fix: derived, not duplicated) ----
    sub_agent_proxy_ticket_access: ProxyAccessLevel = ProxyAccessLevel.NEVER
    sub_agent_proxy_work_access: ProxyAccessLevel = ProxyAccessLevel.NEVER

    # ---- TTL and displacement (R3 CRIT-2: fallback DELETED in C3) ----
    ttl_seconds: int = 300
    fallback_skill: Optional[str] = None  # None=default; no builtin sets a fallback

    # ---- Cross-cutting config ----
    fire_and_forget: bool = False  # US-017a: ticket mode is non-interactive
    blocked_tools_for_subagent: frozenset = field(
        default_factory=lambda: frozenset({"question"})
    )

    # ---- Action-level restrictions (R3 CRIT-4 fix: applies to ALL agents) ----
    check_action: Optional[Callable] = None  # MAIN + sub when in this mode

    # ---- Call limits (R3 MED-16 fix: per-skill overrides) ----
    call_limits: dict = field(default_factory=dict)  # {"proxy_ticket": 50, "task": 3}

    # ---- C7a: capability table (single source of truth for tool visibility) ----
    capability_table: Optional[CapabilityTable] = None

    # ---- Behavior (R1 HIGH-1 fix: pure behavior, not data) ----
    get_system_prompt: Optional[Callable[[SkillRequestContext], str]] = None
    get_subagent_prompt: Optional[Callable[[SkillRequestContext], str]] = None
    on_activate: Optional[Callable[[SkillRequestContext, ModeState], None]] = None
    on_deactivate: Optional[Callable[[SkillRequestContext, ModeState], None]] = None
    # R4 HIGH-1 fix: state only (no behavior); canary verification lives in
    # proxy/canary.py (Phase 2), which calls this hook to obtain the canary
    # marker, not to assert any TTL/secret state. Returns Optional[str].
    # R1+R4 fix: TicketSkill/WorkSkill return `str` (the canary marker, e.g.
    # "[[__SUBAGENT__:SKILL:ticket:<sig>]]"), not `dict` — earlier typing was a lie.
    get_canary: Optional[Callable[["SkillRequestContext"], Optional[str]]] = None
    parse_subcommand: Optional[Callable[[str, str], Tuple[Optional[str], str]]] = None
    handle_direct: Optional[Callable[["SkillRequestContext", str, str], Optional[dict]]] = None
    post_process: Optional[Callable[["SkillRequestContext", dict], dict]] = None  # R1 HIGH-1 fix: returns dict (assigned back by mode.apply_post_process, see proxy/mode.py:162).
    should_deactivate: Optional[Callable[[SkillRequestContext, str], bool]] = None
    get_activation_rewrite: Optional[Callable[[SkillRequestContext], str]] = None

    # ---- Subagent spawning policy (custom-skill research swarms spec §7.1) ----
    # `None` means "this skill does not spawn sub-agents" (the default for
    # every existing builtin). A non-None value freezes the per-skill
    # `tool_whitelist` and per-request `max_task_calls_per_request` bound
    # the loader validated at parse time. The runtime consults this via
    # RequestContext.swarm_policy; it MUST NOT reparse the JSON file.
    subagents: Optional["SubagentPolicy"] = None

    @property
    def is_builtin(self) -> bool:
        """Server-side computed trust flag (CAT-3 fix).

        Returns True iff `self.name` is in `TRUSTED_BUILTIN_SKILLS`. The
        property is read-only — there is no setter, so a user-defined
        skill cannot forge builtin status. The check runs at access
        time, so updates to `TRUSTED_BUILTIN_SKILLS` (e.g., via
        `monkeypatch.setattr` in tests) take effect immediately.

        SECURITY (N3 LOW, upgraded to MED): this property is for
        INTROSPECTION and REPORTING. Security checks (registry
        registration, sub-agent proxy access) MUST consult
        `is_trusted_skill(self.name)` directly — the public
        `is_builtin` property could be shadowed by a subclass. The
        `test_skill_definition_public_surface_matches_helper` test
        (§5 test 3) pins the contract that the property and the
        helper agree for every name.
        """
        return is_trusted_skill(self.name)

    def get_tool_filter(
        self, ctx: Optional[SkillRequestContext] = None
    ) -> Optional[frozenset[str]]:
        """Return the tool filter for this skill (Phase 1b compat shim).

        Matches the `BaseSkill.get_tool_filter(ctx)` signature so call
        sites like `mode.py:295` (which call
        `skill.get_tool_filter(skill_ctx)`) work uniformly for both
        the pre-1a `BaseSkill` instances and the post-1a
        `SkillDefinition` instances stored in the registry. The
        `ctx` parameter is accepted but ignored — the
        `SkillDefinition.allowed_tool_names` field is the single
        source of truth, set at construction time (via
        `BaseSkill.to_skill_definition()`) and immutable afterwards
        (the class is `@dataclass(frozen=True)`).

        Three-state semantics (Wave 1, spec §6) for the returned value:
          - ``None``           — unrestricted (no filtering).
          - ``frozenset()``    — deny all tools.
          - Non-empty frozenset — exact allowlist.

        The annotation matches the field type on
        `SkillDefinition.allowed_tool_names` (C7 fix: was previously
        declared as ``frozenset[str]``, which silently lied about the
        ``None`` case and caused `tool_cache.py` to coerce ``None`` to
        deny-all via ``if tool_filter:``).
        """
        return self.allowed_tool_names

    # CRIT fix 2 (R2-3): __setattr__ override to enforce immutability outside
    # `__post_init__`. The `@dataclass(frozen=True)` only blocks whole-attribute
    # reassignment via the default frozen behavior. This override ensures that
    # ANY write to an instance attribute (including `self.x = y` in
    # non-`__post_init__` methods, or external `sd.x = y` calls) raises
    # `FrozenInstanceError`. Writes are allowed only when `_in_init` is True.
    # The flag is set/cleared via `object.__setattr__` in `__post_init__`
    # (which bypasses this override). The class-level default `_in_init = False`
    # means external code cannot sneak a write in between `__init__` field
    # assignment and `__post_init__` — the dataclass-generated `__init__` uses
    # `object.__setattr__` directly (bypassing this override), then
    # `__post_init__` flips the flag on via `object.__setattr__`.
    #
    # NOTE: this `__setattr__` is attached to the class AFTER the
    # `@dataclass(frozen=True)` decorator runs (see the assignment below
    # the class body). The dataclass decorator refuses to apply its
    # generated `__setattr__` if one is already present in the class
    # body, so the override is applied via a post-class patch.

    def __post_init__(self):
        """Validate identity fields at construction time (R1 LOW-3, R4 CRIT-4, CRIT R1-4, HIGH R2-6).

        Uses `raise ValueError(...)` (NOT `assert`) because `assert` is
        disabled by `python -O` and would silently skip validation in
        optimised runs (production deployments). The regexes are the
        canonical skill-name and trigger patterns used throughout the
        codebase (`SKILL_NAME_RE` in `proxy/canary.py`).

        CRIT fix 2 (R2-3): the `_in_init` flag is set via
        `object.__setattr__` to bypass the `__setattr__` override. The
        `try/finally` ensures the flag is reset to False even when
        validation raises, so the half-constructed instance (if any code
        catches and inspects it) reports a frozen state.

        HIGH fix 7: explicit empty-string guards run BEFORE the regex
        checks. The regexes already reject empty strings (the `^[a-z]`
        anchor requires a leading letter), but the explicit guards
        produce a clearer error message ("name must not be empty" vs.
        "Invalid name: ''") and short-circuit before the regex
        machinery.

        CRIT fix 3 (R1-4): alias validation runs AFTER the name/trigger
        regexes. (a) `alias` must match `^[a-z][a-z0-9_-]*$` (the same
        pattern as the name); (b) `alias != self.name`; (c) no
        duplicate aliases. The pre-existing `alias == self.trigger`
        check is preserved.

        HIGH fix 4 (R2-6): trust-gate check runs AFTER all regex
        validation. If the new `SkillDefinition`'s name is in
        `TRUSTED_BUILTIN_SKILLS` AND no TRUSTED `TrustToken` is held on
        the current thread (`_trust_token_local.token`), construction
        raises `SkillRegistrationError`. `TrustToken` is imported lazily
        to avoid the circular import: `skill_registry.py` imports from
        `skill_definition.py` at module load.
        """
        # CRIT fix 2: flip the gate ON via object.__setattr__ to bypass
        # the __setattr__ override. The try/finally guarantees the flag
        # is reset to False even if validation raises.
        object.__setattr__(self, "_in_init", True)
        try:
            # HIGH fix 7: empty name/trigger guard (clearer error than
            # the regex; also short-circuits before regex compilation).
            # `trigger == "^^"` (prefix only, no name part) is treated
            # as "empty" for the same reason — it cannot match the
            # trigger regex and the dedicated empty-message is clearer
            # than the generic "Invalid trigger" regex error.
            if not self.name:
                raise ValueError("SkillDefinition name must not be empty")
            if not self.trigger or self.trigger == "^^":
                raise ValueError("SkillDefinition trigger must not be empty")
            if not re.match(r'^\^\^[a-z][a-z0-9_-]*$', self.trigger):
                raise ValueError(f"Invalid trigger: {self.trigger!r}")
            if not re.match(r'^[a-z][a-z0-9_-]*$', self.name):
                raise ValueError(f"Invalid name: {self.name!r}")
            # CRIT fix 3 (R1-4) + round-trip fix: alias validation.
            # Aliases may be either bare names (programmatic BaseSkill
            # subclasses — `aliases = ("a1",)`) or ^^-prefixed dispatch
            # triggers (JSON-loaded custom skills — `aliases: ["^^t"]`).
            # Both forms normalize to a bare token for duplicate / name
            # collision checks. The regex accepts:
            #   - `^[a-z][a-z0-9_-]*$` (bare)
            #   - `^\^\^[a-z][a-z0-9_-]*$` (dispatch prefix)
            # and rejects uppercase, special chars, leading-digit,
            # empty strings (all caught by the `[a-z]` first-char anchor).
            trigger_bare = self.trigger[2:] if self.trigger.startswith("^^") else self.trigger
            for alias in self.aliases:
                m = re.match(r'^(?:\^\^)?([a-z][a-z0-9_-]*)$', alias)
                if not m:
                    raise ValueError(f"Invalid alias: {alias!r}")
                alias_bare = m.group(1)
                if alias_bare == self.name:
                    raise ValueError(f"Alias {alias!r} duplicates name")
                if alias_bare == trigger_bare:
                    raise ValueError(f"Alias {alias!r} duplicates trigger")
            if len(self.aliases) != len(set(self.aliases)):
                raise ValueError(f"Duplicate aliases: {self.aliases!r}")
            if is_trusted_skill(self.name):
                held = getattr(_trust_token_local, "token", None)
                if held is None or not isinstance(held, TrustToken) or not held.is_trusted:
                    raise SkillRegistrationError(
                        f"Cannot construct SkillDefinition for trusted "
                        f"builtin {self.name!r} without a held TRUSTED "
                        f"TrustToken on the current thread"
                    )
        finally:
            object.__setattr__(self, "_in_init", False)


# CRIT fix 2 (R2-3): attach the `__setattr__` override AFTER the
# `@dataclass(frozen=True)` decorator has run. The dataclass decorator
# raises `TypeError: Cannot overwrite attribute __setattr__` if the
# class body already defines `__setattr__`, so the override is
# installed via a post-class assignment.
def _skill_definition_setattr(self, name, value):
    """`__setattr__` for `SkillDefinition`: frozen except during `__post_init__`.

    Reads the `_in_init` flag (set via `object.__setattr__` in
    `__post_init__`, which bypasses this override). If the flag is
    True, the write is allowed (this is how `__post_init__` mutates
    `_in_init` itself and any other internal state). Otherwise, raise
    `FrozenInstanceError` — the same exception the default dataclass
    `frozen=True` `__setattr__` raises, so existing `except
    FrozenInstanceError` blocks continue to work.
    """
    if getattr(self, "_in_init", False):
        object.__setattr__(self, name, value)
    else:
        from dataclasses import FrozenInstanceError
        raise FrozenInstanceError(
            f"cannot assign to field {name!r}"
        )


SkillDefinition.__setattr__ = _skill_definition_setattr
del _skill_definition_setattr


# ---------------------------------------------------------------------------
# §3.6 — BaseSkill thin wrapper
# ---------------------------------------------------------------------------
class BaseSkill:
    """Thin wrapper around `SkillDefinition` for backward compat (R2 CRIT-2, R5 CRIT-1).

    The legacy `Skill` Protocol in `src/proxy/skills/skill_registry.py`
    (pre-Phase 1a) defined the 16 method stubs (get_system_prompt,
    get_subagent_prompt, on_activate, etc.) that this NEW `BaseSkill`
    replaces. The 28+ test sites that subclass that `BaseSkill` will be
    migrated in Phase 1d to subclass the NEW `BaseSkill` defined here.

    In 1a, this `BaseSkill` is added but NOT YET wired to any
    production code. Its only consumer is `to_skill_definition()`,
    which converts an instance (or a subclass instance) into the
    canonical `SkillDefinition` for downstream use.

    R1 CRIT-3 fix: the type annotations on the method stubs use
    `SkillRequestContext` (the new 1a name). The pre-1a `BaseSkill`
    in `skill_registry.py` continues to use the old `SkillContext`
    type. The two coexist; the 1d migration unifies them.
    """

    # Subclass-overridable class-level attributes (pre-1a BaseSkill
    # pattern). Default to None / empty so a minimal subclass works.
    name: str = ""
    trigger: str = ""
    aliases: tuple = ()

    # HIGH fix 6: `_definition` is an immutable cache. Class-level
    # default of `None` means `__init__` does NOT need to set it
    # (skipping the assignment avoids the `__setattr__` block below).
    # After the first call to `to_skill_definition()`, the cache is
    # installed via `object.__setattr__` (which bypasses this
    # override); any subsequent instance-level assignment to
    # `_definition` is rejected.
    _definition: Optional[SkillDefinition] = None

    def __init__(self):
        # HIGH fix 6: no-op. `_definition` is a class-level default
        # (see above). `to_skill_definition` installs the cache via
        # `object.__setattr__` to bypass the `__setattr__` block.
        pass

    # HIGH fix 5/6: __setattr__ / __delattr__ overrides enforce the
    # `BaseSkill` contract:
    #   - the cached `_definition` is immutable once set (no
    #     instance-level reassignment, no deletion);
    #   - the `name` / `trigger` identity attributes cannot be
    #     changed on an instance after `to_skill_definition()` has
    #     been called.
    # This pins the identity-stability guarantee relied on by
    # `SkillRegistry.register_skill` and the cache-immutability test
    # (§5 test 17: cache is frozen at first call).
    #
    # Class-level attribute assignment (e.g., `MySubclass.name = "x"`)
    # is NOT affected — `__setattr__` only fires for instance
    # attribute writes. So the test 17 pattern `S.name = "ticket"`
    # (class attribute mutation) continues to work.
    def __setattr__(self, name, value):
        if name == "_definition":
            # HIGH fix 5/6: the first instance-level assignment to
            # `_definition` is allowed (this is how `to_skill_definition`
            # initializes the cache; `to_skill_definition` uses
            # `object.__setattr__` to bypass this override, but external
            # callers that want to set the cache directly are also
            # accommodated for backwards compatibility with test 32).
            # Any SUBSEQUENT instance-level assignment is blocked.
            if "_definition" not in self.__dict__:
                object.__setattr__(self, name, value)
                return
            raise AttributeError(
                "Cannot reassign _definition: cache is frozen at first call"
            )
        if name in ("name", "trigger"):
            current = getattr(self, "_definition", None)
            if current is not None:
                raise AttributeError(
                    f"Cannot set {name!r} after to_skill_definition() was called"
                )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == "_definition":
            raise AttributeError(
                "Cannot delete _definition: cache is frozen at first call"
            )
        object.__delattr__(self, name)

    def get_tool_filter(self, ctx: Optional[SkillRequestContext]) -> Optional[frozenset[str]]:
        """Default: unrestricted (None).

        Wave 1 (spec §6) three-state semantics:

          - ``None`` (this default) — unrestricted. The BaseSkill
            semantic is "unrestricted unless overridden" — subclasses
            that want a stricter policy must override this method
            and return either ``frozenset()`` (deny all) or a
            non-empty frozenset (exact allowlist).
          - ``frozenset()`` — deny all tools.
          - Non-empty ``frozenset`` — exact allowlist.

        Under the old semantics (pre-Wave 1) this method returned
        ``frozenset()`` which meant "no restriction" (fail-open). The
        new semantics make ``None`` the unrestricted sentinel and
        ``frozenset()`` the deny-all sentinel — a clean break per the
        spec, no backwards compatibility shim.

        Pre-1a `BaseSkill` callers always pass `None` for `ctx` (the
        old `SkillContext` was a richer object that this stub does
        not have). The wrapper treats `None` as "no context" and
        returns the unrestricted sentinel.
        """
        return None

    def get_system_prompt(self, ctx: SkillRequestContext) -> str:
        return ""

    def get_subagent_prompt(self, ctx: SkillRequestContext) -> Optional[str]:
        return None

    def on_activate(self, ctx: SkillRequestContext, mode_state: ModeState) -> None:
        pass

    def on_deactivate(self, ctx: SkillRequestContext, mode_state: ModeState) -> None:
        pass

    def get_canary(self, ctx: SkillRequestContext) -> Optional[str]:
        return None

    def build_skill_canary(self, ctx: SkillRequestContext) -> str:
        """Shared canary builder for ticket/work skills.

        Serializes inheritance state, signs via sign_canary, and wraps
        in [[__SUBAGENT__:SKILL:<name>:<sig>:<state>]] marker format.
        Returns a truthy string in both production (real HMAC) and dev
        mode (placeholder).
        """
        import json as _json
        from proxy.canary import sign_canary as _sign_canary

        state = self.get_inheritance_state(ctx)
        state_str = ""
        if state:
            try:
                # sort_keys=True is mandatory — the verify side re-serialises
                # the parsed state dict with the same flag so HMAC inputs match.
                state_str = f"|{_json.dumps(state, sort_keys=True)}"
            except (TypeError, ValueError):
                pass
        sig = _sign_canary(
            self.name,
            state_str=state_str,
            session_id=getattr(ctx, "session_id", ""),
        )
        return f"[[__SUBAGENT__:SKILL:{self.name}:{sig}{state_str}]]"

    def parse_subcommand(self, text: str, session_id: str = "") -> Tuple[Optional[str], str]:
        return (None, "")

    def handle_direct(self, ctx: SkillRequestContext, action: str, arg: str) -> Optional[dict]:
        return None

    def post_process(self, ctx: SkillRequestContext, response: dict) -> dict:
        """Post-process the LLM response. Default: passthrough (return unchanged).

        R5-PHASE-1D fix: the NEW BaseSkill default was changed to `pass`
        (returning None), but that breaks the `mode.py:465` call site
        `data = skill.post_process(skill_ctx, data)` (which expects the
        response dict back). The OLD BaseSkill default was a passthrough
        (`return response`). Phase 1d restores the OLD contract so the
        legacy pipeline continues to work.
        """
        return response

    def should_deactivate(self, ctx: SkillRequestContext, finish_reason: str) -> bool:
        """Canonical lifecycle for builtin skills (spec §7.4).

        Returns True to clear the active mode (COMPLETE or FAILED),
        False to keep the mode alive (CONTINUE — tool_calls or length
        truncation).

        Mapping (matches `proxy.mode.classify_turn_response`):
        - "tool_calls" or "length" → False (CONTINUE, mode retained)
        - "content_filter" → True (FAILED, mode cleared) — covered by
          the default `return True` branch below.
        - "stop" → True (COMPLETE, mode cleared)
        - None → True (no termination signal — treated as COMPLETE)
        - any other non-null reason → True (FAILED, mode cleared)
        """
        if finish_reason in ("tool_calls", "length"):
            return False  # CONTINUE
        if finish_reason is None:
            return True  # No termination signal -> treat as COMPLETE
        # stop, content_filter, and any other reason -> clear
        return True

    def get_activation_rewrite(self, ctx: SkillRequestContext) -> str:
        return ""

    def to_skill_definition(self) -> SkillDefinition:
        """Convert this `BaseSkill` instance to a `SkillDefinition`.

        Builds the `SkillDefinition` from class-level attributes and
        method references. Caches the result in `self._definition`
        **frozen at first call** — the first invocation constructs the
        `SkillDefinition` and writes it to `self._definition`; every
        subsequent call returns the SAME object. The cache is
        deliberately not invalidated: `SkillDefinition` is
        `@dataclass(frozen=True)`, so a caller cannot mutate the cached
        object, and downstream registry/storage logic relies on
        identity-stability (`SkillRegistry.register_skill` keys on
        `skill.name` and the cached definition is what gets stored;
        re-computing on every call would defeat the identity check).
        The cache-immutability test (§5 test 17) pins the contract
        that `is_builtin` does NOT change after
        `to_skill_definition()` is called — meaning a caller cannot
        construct a `BaseSkill` with `name="ticket"`, get the
        definition, then mutate the class attribute to `name="attack"`
        and get a re-evaluated `is_builtin=False`. Once frozen, the
        definition is what it is.

        R3-CRIT-2 fix: raises `ValueError` if `self.name` or
        `self.trigger` is empty (the class-level defaults are
        `name=""` and `trigger=""` so a minimal subclass would
        otherwise produce a `SkillDefinition(name="", trigger="")`
        that bypasses the regex validator and registers as a
        nameless skill). The guard runs BEFORE the cache check so
        that a misconfigured BaseSkill fails loudly on the FIRST
        call, not silently on every subsequent call (which would
        return the same broken cached object).

        The `is_builtin` field is intentionally NOT passed (CAT-3
        fix) — the `SkillDefinition.is_builtin` property is computed
        from `self.name` via `TRUSTED_BUILTIN_SKILLS`.

        R1 CRIT-1 fix: all 23 `SkillDefinition` fields are copied via
        `getattr(self, ..., <default>)` so that subclasses which
        override only a subset of attributes still produce a
        fully-populated `SkillDefinition`. The defaults match the
        `SkillDefinition` field defaults defined in §3.5 — any
        default change in §3.5 MUST be mirrored here. Pinning test:
        `test_base_skill_to_skill_definition_round_trip_all_23_fields`
        (§5 test 16).

        CRIT R1-1 fix (overrides the previous `vars(type(self)).get()`
        approach): the 10 behavior fields (`check_action`,
        `get_system_prompt`, `get_subagent_prompt`, `on_activate`,
        `on_deactivate`, `get_canary`, `parse_subcommand`,
        `handle_direct`, `post_process`, `should_deactivate`,
        `get_activation_rewrite`) are now read via `getattr(self, ...)`
        and any returned bound method is unwrapped via `__func__` to
        preserve identity stability. The previous
        `vars(type(self)).get()` approach bypassed the descriptor
        protocol (returning the raw function for class attributes) but
        BROKE instance methods — `vars(type(self))` only contains class
        attributes, so a method set via `self.x = func` in `__init__`
        would resolve to `None` here. The new approach supports BOTH
        class attributes and instance attributes. Pinned by
        `test_base_skill_to_skill_definition_round_trip_all_23_fields`
        (§5 test 16).

        R1 HIGH-2 fix: `sub_agent_tool_filter` consults
        `get_sub_agent_tool_filter(...)` when the subclass defines
        it; otherwise it falls back to `get_tool_filter(...)`. The
        previous draft unconditionally used `get_tool_filter(None)`
        for both `allowed_tool_names` and `sub_agent_tool_filter`,
        collapsing the two filter fields into a single value — that
        contract error made the sub-agent tool filter always equal
        the parent tool filter, defeating the sub-agent restriction
        purpose. Subclasses that distinguish the two (e.g., the 1d
        migration of `TicketSkill`) define `get_sub_agent_tool_filter`
        as a separate method.

        R2-HIGH-5 fix: an explicit `or frozenset()` guard coerces
        `None` → `frozenset()` for both tool filter fields. The
        OLD `Skill` Protocol's `get_tool_filter` returned `None` as
        the "no restriction" sentinel; the new
        `BaseSkill.get_tool_filter` returns `frozenset()` for the same
        semantics. Coercing at the conversion boundary keeps the
        field types clean without forcing the old `BaseSkill` callers
        to change their `get_tool_filter` return type.

        WAVE 1 update (spec §6 three-state semantics):
        - `BaseSkill.get_tool_filter` now returns ``None`` (unrestricted)
          instead of ``frozenset()`` — see the method docstring.
        - `allowed_tool_names=self.get_tool_filter(None) or None` at the
          call site below coerces any spurious empty frozenset to
          ``None`` (unrestricted), preserving the BaseSkill default
          semantic. Subclasses that return a non-empty frozenset
          (TicketSkill, WorkSkill) keep their exact allowlist.
        - `sub_agent_tool_filter` stays as a real frozenset (NOT
          three-state) — see SkillDefinition class docstring.
        """
        # R3-CRIT-3 fix: BaseSkill class-level defaults are name=""
        # and trigger="". A minimal subclass that forgets to set
        # either would otherwise construct a SkillDefinition(name="",
        # trigger="") and silently bypass the regex validator. Fail
        # loudly here so the misconfiguration is caught at first
        # call, not at registry-registration time.
        if not self.name or not self.trigger:
            raise ValueError(
                "BaseSkill must set name and trigger before calling "
                "to_skill_definition()"
            )
        if self._definition is not None:
            return self._definition
        # R2-HIGH-5 fix: coerce None → frozenset() for the SUB-AGENT
        # filter (sub_agent_tool_filter stays as a real frozenset per
        # the SkillDefinition class docstring). The MAIN filter (allowed_tool_names)
        # is now three-state — the `or None` at the call site below
        # handles the BaseSkill default.
        main_filter = self.get_tool_filter(None) or frozenset()
        if hasattr(self, 'get_sub_agent_tool_filter'):
            sub_filter = self.get_sub_agent_tool_filter(None) or frozenset()
        else:
            # Inherit main filter; coerce None → frozenset() because
            # sub_agent_tool_filter is NOT three-state (must be a real
            # frozenset when inherited).
            sub_filter = main_filter if main_filter is not None else frozenset()
        # HIGH fix 6: build the SkillDefinition into a local variable,
        # then install it via `object.__setattr__` to bypass the
        # `BaseSkill.__setattr__` override (which blocks ALL
        # instance-level assignments to `_definition`). The cache is
        # identity-stable: subsequent calls return this exact object.
        _sd = SkillDefinition(
            name=self.name,
            trigger=self.trigger,
            aliases=getattr(self, 'aliases', ()),
            # Wave 1: short human-readable description, surfaced by
            # `^^help` and the custom JSON loader (Agent 2). Default
            # empty string keeps the dataclass constructor backwards-
            # compatible for callers that do not pass it.
            description=getattr(self, 'description', ""),
            # Wave 1: three-state policy. The `or None` coerces any
            # spurious empty frozenset from `get_tool_filter` to None
            # (= unrestricted), matching the BaseSkill default semantic.
            # See SkillDefinition.allowed_tool_names docstring + spec §6.
            allowed_tool_names=main_filter or None,
            sub_agent_tool_filter=sub_filter,
            # R1 CRIT-3 fix: sub-agent permissions are derived, not duplicated.
            # Default NEVER; subclasses override.
            sub_agent_proxy_ticket_access=getattr(
                self, 'sub_agent_proxy_ticket_access', ProxyAccessLevel.NEVER
            ),
            sub_agent_proxy_work_access=getattr(
                self, 'sub_agent_proxy_work_access', ProxyAccessLevel.NEVER
            ),
            # R3 CRIT-2: fallback DELETED in C3 (field retained for custom-skill JSON loader).
            ttl_seconds=getattr(self, 'ttl_seconds', 300),
            fallback_skill=getattr(self, 'fallback_skill', None),
            # R1 CRIT-1 fix: cross-cutting config (R3 US-017a, R3 MED-16).
            fire_and_forget=getattr(self, 'fire_and_forget', False),
            blocked_tools_for_subagent=getattr(
                self, 'blocked_tools_for_subagent',
                frozenset({"question"})
            ),
            # R3 CRIT-4 fix: action-level restriction applies to ALL agents.
            # CRIT R1-1 fix: use getattr + _unwrap_bound_method instead of
            # `vars(type(self)).get()`. The old approach bypassed the
            # descriptor protocol (returning the raw function for class
            # attributes) but BROKE instance methods — `vars(type(self))`
            # only contains class attributes, so a method set via
            # `self.x = func` in `__init__` would resolve to `None` here.
            # The new approach supports BOTH class attributes and instance
            # attributes. For bound methods, `_unwrap_bound_method` peels
            # off the binding via `__func__` to preserve identity
            # stability for the round-trip test.
            check_action=_unwrap_bound_method(
                getattr(self, 'check_action', None)
            ),
            # R3 MED-16 fix: per-skill call limits.
            call_limits=getattr(self, 'call_limits', {}),
            # C7a: capability table (single source of truth for tool visibility).
            capability_table=getattr(self, 'capability_table', None),
            # Behavior (R1 HIGH-1 fix: pure behavior, not data).
            # Same CRIT R1-1 fix as check_action above.
            get_system_prompt=_unwrap_bound_method(
                getattr(self, 'get_system_prompt', None)
            ),
            get_subagent_prompt=_unwrap_bound_method(
                getattr(self, 'get_subagent_prompt', None)
            ),
            on_activate=_unwrap_bound_method(
                getattr(self, 'on_activate', None)
            ),
            on_deactivate=_unwrap_bound_method(
                getattr(self, 'on_deactivate', None)
            ),
            get_canary=_unwrap_bound_method(
                getattr(self, 'get_canary', None)
            ),
            parse_subcommand=_unwrap_bound_method(
                getattr(self, 'parse_subcommand', None)
            ),
            handle_direct=_unwrap_bound_method(
                getattr(self, 'handle_direct', None)
            ),
            post_process=_unwrap_bound_method(
                getattr(self, 'post_process', None)
            ),
            should_deactivate=_unwrap_bound_method(
                getattr(self, 'should_deactivate', None)
            ),
            get_activation_rewrite=_unwrap_bound_method(
                getattr(self, 'get_activation_rewrite', None)
            ),
            # Subagent spawning policy (custom-skill research swarms §7.1).
            # Defaults to None so existing builtin BaseSkill subclasses
            # (TicketSkill, WorkSkill, WorkspaceSkill) keep their current
            # "no sub-agent spawn" behavior unchanged. The custom-skill
            # loader sets this field when a JSON skill declares a
            # structurally-valid `subagents` block.
            subagents=getattr(self, 'subagents', None),
        )
        object.__setattr__(self, "_definition", _sd)
        return self._definition


# ---------------------------------------------------------------------------
# CRIT R1-1 fix: helper to unwrap bound methods.
# ---------------------------------------------------------------------------
def _unwrap_bound_method(value):
    """Return the underlying function for a bound method, else `value` unchanged.

    `getattr(instance, 'attr')` on a class attribute that is a function
    returns a BOUND METHOD (the function bound to the instance), not
    the raw function. The cached `SkillDefinition` should hold the raw
    function so identity checks in tests pass:

        class Full(BaseSkill):
            get_system_prompt = sentinel_prompt  # a lambda
        skill = Full()
        sd = skill.to_skill_definition()
        assert sd.get_system_prompt is sentinel_prompt  # the raw lambda

    `getattr(skill, 'get_system_prompt')` returns a bound method whose
    `__func__` is `sentinel_prompt`. This helper peels off the
    binding. Pinned by
    `test_base_skill_to_skill_definition_round_trip_all_23_fields`.

    R5-PHASE-1D fix: heuristics-aware unwrap. When the bound method is
    an instance method (the unbound function takes `self` as the first
    positional parameter), we KEEP the bound method so that downstream
    callers (e.g. `mode.py:506: skill.get_canary(skill_ctx)`) can
    invoke it as `bound_method(skill_ctx)` — Python auto-binds `self`
    to the SkillDefinition instance. The legacy pipeline (mode.py,
    Phase 2 dispatch) expects this calling convention; Phase 3a will
    refactor callers to use the new SkillDefinition explicitly.

    For class-attribute functions (lambdas, plain functions assigned
    at class scope that take exactly one positional arg), the bound
    method's first positional arg would be `self`, but the original
    function takes only `ctx`. Unwrapping exposes the raw lambda so
    `sd.get_canary is sentinel_canary` holds (round-trip test).

    The heuristic: if the unbound function takes `self` as its first
    parameter, treat it as an instance method and KEEP the binding.
    Otherwise, treat it as a class-attribute function and unwrap.
    """
    if value is None or not hasattr(value, '__func__'):
        return value
    raw = value.__func__
    # Inspect the unbound function signature. If first param is `self`,
    # this is an instance method — keep the bound method so callers
    # can invoke `bound(skill_ctx)` (Python binds `self` automatically).
    try:
        import inspect
        sig = inspect.signature(raw)
        params = list(sig.parameters.values())
        if params and params[0].name == 'self':
            return value  # instance method — keep bound
    except (ValueError, TypeError):
        pass
    return raw
