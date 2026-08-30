"""Skill registry for managing and discovering skills."""
import inspect
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from proxy.skills.custom_skill_loader import (
    CustomSkillsStatus,
    SkillRegistryConfigurationError,
    build_custom_skill_definition,
    load_custom_skills,
    resolve_skills_dir,
)
from proxy.skills.skill_definition import (
    SkillDefinition,
    SkillRegistrationError,
    TRUSTED_BUILTIN_SKILLS,
    TrustToken,
    _held_trust_token,
    is_trusted_skill,
)

logger = logging.getLogger(__name__)

# CRIT R2-1 / R2-2: only the modules listed here may mint a `TrustToken` via
# `TrustToken.for_trusted_caller()` or re-register a trusted builtin skill
# via `register()`. Adding a module here is a security-relevant change and
# MUST be reviewed (it expands the trust boundary).
TRUSTED_TRUST_CALLERS: frozenset[str] = frozenset({
    "__main__",
    "proxy.agent.agent_registry",
    "proxy.skills.skill_registry",
    "tests.test_skill_dispatch",
    "tests.test_skill_definition",
    "tests.skills.test_skill_registry",
    "tests.skills.test_ticket_skill",  # Phase 1d: NEW BaseSkill tests
    "tests.skills.test_work_skill",  # Phase 1d: NEW BaseSkill tests
    "tests.unit.test_mode_resolution",
    "tests.proxy.test_skill_mode",
    "tests.proxy.test_canary_repro",
    "tests.proxy.test_skill_registry",  # NEW (Wave 1.2 — security boundary expansion)
    "tests.proxy.test_trust_gate",  # NEW (Wave 2.3 — R5 CRIT-1 trust gate tests)
    "tests.integration.test_skill_dispatch_integration",
})


# Zero-width and invisible characters that can be used to bypass trigger detection.
# Ranges: ZWSP (\\u200B)–ZWJ (\\u200D), LRM/RLM (\\u200E/\\u200F), line/paragraph
# separators (\\u2028/\\u2029), word joiner (\\u2060), soft hyphen (\\u00AD), BOM (\\uFEFF).
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u2028\u2029\uFEFF\u00AD\u2060]")


def _normalize_text(text: str) -> str:
    """Strip zero-width / invisible chars and apply NFKC normalization."""
    text = _ZERO_WIDTH_RE.sub("", text)
    return unicodedata.normalize("NFKC", text)


class SkillActivation(Enum):
    COMMAND = auto()
    AUTO = auto()
    INHERITED = auto()


@dataclass(frozen=True)
class SkillMeta:
    name: str
    trigger: Optional[str] = None
    aliases: tuple[str, ...] = ()
    activation: SkillActivation = SkillActivation.COMMAND
    ttl_seconds: int = 300
    description: str = ""
    help_text: str = ""
    is_builtin: bool = True


@dataclass
class SkillContext:
    session_id: str
    agent_id: str
    messages: list = field(default_factory=list)
    is_subagent: bool = False
    parent_agent_id: Optional[str] = None
    active_skill_name: Optional[str] = None
    skill_state: dict = field(default_factory=dict)
    working_directory: str = ""
    raw_user_text: str = ""
    seeded_tasks: list = field(default_factory=list)
    active_workspace: Optional[str] = None
    client_hint: str = ""
    # Phase 4: work skill captures the task hash on activation so the system
    # prompt can anchor the agent to a specific task. Populated by WorkSkill.on_activate
    # from skill_state["task_hash"].
    work_task_hash: str = ""
    _tools_snapshot: frozenset = field(default_factory=frozenset)
    _skill_activated: bool = False

    # DESIGN NOTE (request-id-design.md): request_id is used to correlate all
    # log entries and spans across the dispatch pipeline (app.py → request_pipeline
    # → stage_modules/dispatch_stage.py → skill_registry.py → task_graph.py → graph.py).
    # Populated at DispatchStage entry; propagated through SkillContext by
    # SkillContext.from_request_context.
    # TODO-009: wire request_id generation into the pipeline and add to all
    #           dispatch pipeline log calls before removing this comment.
    request_id: str = ""

    @classmethod
    def from_request_context(
        cls,
        ctx: Any,
        *,
        skill_state: Optional[dict] = None,
        raw_user_text: Optional[str] = None,
        is_subagent: Optional[bool] = None,
    ) -> "SkillContext":
        """Create SkillContext from a request context.

        Args:
            ctx: The request context object (any shape — uses getattr).
            skill_state: Optional dict of skill-specific state.
            raw_user_text: The raw user input text.
            is_subagent: Whether this is a sub-agent request.

        """
        agent = None
        try:
            from proxy.agent.agent_registry import registry as agent_registry
            agent = agent_registry.get_or_load(ctx.session_id)
        except Exception:
            logger.exception("SkillContext.from_request_context: Failed to load agent for session %s", getattr(ctx, "session_id", "?")[:8])

        # ---- type-validate raw values (C-20) ----
        _raw_session_id = ctx.session_id
        session_id = _raw_session_id if isinstance(_raw_session_id, str) else ""

        _raw_agent_id = agent.agent_id if agent else ""
        agent_id = _raw_agent_id if isinstance(_raw_agent_id, str) else ""

        messages_raw = list(ctx.messages) if hasattr(ctx, "messages") else []
        messages = messages_raw if isinstance(messages_raw, list) else []

        is_sub_val = is_subagent if is_subagent is not None else getattr(ctx, "is_subagent", False)
        is_subagent_clean = is_sub_val if isinstance(is_sub_val, bool) else False

        _raw_parent = getattr(ctx, "parent_agent_id", None)
        parent_agent_id = _raw_parent if isinstance(_raw_parent, str) or _raw_parent is None else None

        _raw_skill = getattr(ctx, "inherited_skill", None)
        active_skill_name = _raw_skill if isinstance(_raw_skill, str) or _raw_skill is None else None

        skill_state_clean = dict(skill_state) if skill_state else {}
        if not isinstance(skill_state_clean, dict):
            skill_state_clean = {}

        _raw_wd = getattr(ctx, "working_directory", "")
        working_directory = _raw_wd if isinstance(_raw_wd, str) else ""

        raw_user_text_clean = raw_user_text if isinstance(raw_user_text, str) else ""

        _raw_tasks = getattr(ctx, "seeded_tasks", [])
        seeded_tasks = list(_raw_tasks) if isinstance(_raw_tasks, list) else []

        _raw_ws = getattr(ctx, "active_workspace", None)
        active_workspace = _raw_ws if isinstance(_raw_ws, str) or _raw_ws is None else None

        _raw_rid = getattr(ctx, "request_id", "")
        request_id = _raw_rid if isinstance(_raw_rid, str) else ""

        _raw_client_hint = getattr(ctx, "client_hint", "")
        client_hint = _raw_client_hint if isinstance(_raw_client_hint, str) else ""

        return cls(
            session_id=session_id,
            agent_id=agent_id,
            messages=messages,
            is_subagent=is_subagent_clean,
            parent_agent_id=parent_agent_id,
            active_skill_name=active_skill_name,
            skill_state=skill_state_clean,
            working_directory=working_directory,
            raw_user_text=raw_user_text_clean,
            seeded_tasks=seeded_tasks,
            active_workspace=active_workspace,
            client_hint=client_hint,
            request_id=request_id,
        )





# ---------------------------------------------------------------------------
# Module-level singleton state (Phase 1b)
# ---------------------------------------------------------------------------
_default_registry: Optional["SkillRegistry"] = None

# _registry_init_lock — module-local serialization gate for singleton
# construction. Acquired during populate_builtins() to gate concurrent
# first-call to get_default_registry(). NOT in the 14-position lock
# hierarchy (state.py:101-186) because it is a singleton-init lock
# acquired BEFORE any per-instance or per-request lock.
#
# Lifecycle (Wave 2):
#   - CLI: _eager_init_registry() in qwen_proxy.py:__main__ calls
#     get_default_registry() under this lock. Failures raise
#     SkillRegistryConfigurationError; CLI exits 1.
#   - WSGI (gunicorn --preload): the master process's app import
#     invokes populate_builtins() under this lock. Workers fork after.
#   - WSGI (gunicorn without --preload): workers import app lazily;
#     first request handler triggers construction.
#   - Tests: monkeypatch.setenv("PROXY_SKILLS_DIR", tmp_path/skills)
#     BEFORE the first get_default_registry() call.
#
# Lock acquisition order during populate_builtins():
#   _registry_init_lock
#     -> SkillRegistry.self._lock (RLock, per-instance)
#         -> no other lock acquisitions
#
# This lock is a `Lock()` (not RLock) because populate_builtins() is
# called once per process lifetime; no re-entrant paths.
_registry_init_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SkillRegistry — Phase 1b full implementation
# ---------------------------------------------------------------------------
def _skill_registry_new__(cls) -> "SkillRegistry":
    """Recursion-break constructor used by get_default_registry().

    Bypasses SkillRegistry.__new__ so that the thin __new__ shim can call
    get_default_registry() without infinite re-entry.

    Since object.__new__ bypasses __init__/__post_init__, we manually
    initialize the dataclass fields here.
    """
    obj = object.__new__(cls)
    object.__setattr__(obj, "_inner", {})
    object.__setattr__(obj, "_skills", MappingProxyType({}))
    object.__setattr__(obj, "_trigger_map", {})
    object.__setattr__(obj, "_lock", threading.RLock())
    return obj


def __new__(cls) -> "SkillRegistry":
    """Thin shim: returns the cached singleton so legacy SkillRegistry() callsites keep working.

    All real work (construction, populate_builtins) happens in get_default_registry().
    """
    global _default_registry
    if _default_registry is not None:
        return _default_registry
    return get_default_registry()


@dataclass(frozen=True)
class SkillRegistry:
    """Singleton holding all loaded SkillDefinitions.

    Construction is serialized by `_registry_init_lock`. The singleton
    is published by `_default_registry = registry` (module-global assignment)
    AFTER `populate_builtins()` returns, so concurrent readers either see
    None or the fully-populated registry (CRIT R2-4 preserved).

    Wave 2 eager-init wiring:
      - CLI startup calls `_eager_init_registry()` (app.py) which
        invokes `get_default_registry()` at process boot.
      - On SkillRegistryConfigurationError the singleton stays unbuilt;
        CLI exits 1.
      - Test fixtures may reset the singleton via `reset_default_registry()`.
    """

    _inner: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _skills: MappingProxyType = field(init=False)
    _trigger_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(
        default_factory=lambda: threading.RLock(), init=False, repr=False
    )
    # Strict-schema custom-skill snapshot. Set once during
    # `populate_builtins()` and exposed via `custom_skills_status()`.
    # Stored as a frozen `CustomSkillsStatus` so external readers always
    # see a consistent view.
    custom_status: Optional[CustomSkillsStatus] = None

    def __init__(self) -> None:
        """Guard: prevent dataclass-generated __init__ from clearing singleton state.

        When SkillRegistry() is called (legacy callsite), __new__ returns the
        cached singleton. Python then calls __init__ on that existing instance.
        The dataclass-generated __init__ would reset _inner to {}, clobbering the
        state set by populate_builtins(). Guard with _initialized so subsequent
        calls to __init__ are no-ops.

        On first call (when _initialized is not True), we call __post_init__
        ourselves since the custom __init__ prevents the dataclass machinery
        from doing so.
        """
        if getattr(self, "_initialized", False) is True:
            return
        # Call __post_init__ manually since we overrode __init__
        self.__post_init__()

    def __post_init__(self) -> None:
        # Always initialize _skills. This is called once (on the first __init__)
        # after the dataclass fields are set by the default __init__ machinery.
        # On re-entrant __init__ calls (_initialized=True) this is NOT called
        # because Python's dataclass machinery already set the fields.
        if getattr(self, "_initialized", False) is True:
            return
        object.__setattr__(self, "_skills", MappingProxyType(self._inner))
        object.__setattr__(self, "_initialized", True)

    def clear_all(self) -> None:
        """Clear all skills and reset the registry to empty state.

        Used in tests to give each test a known empty starting state.
        Clears _inner and _trigger_map (not _skills — it's a view of _inner).
        """
        self._inner.clear()
        self._trigger_map.clear()
        # Rebuild _skills view
        object.__setattr__(self, "_skills", MappingProxyType(self._inner))

    def custom_skills_status(self) -> CustomSkillsStatus:
        """Return the strict-schema custom-skill snapshot.

        If the registry was constructed without running `populate_builtins()`
        (e.g. a bare `_skill_registry_new__` instance in tests), returns a
        default "disabled" snapshot so callers always get a value.
        """
        if self.custom_status is not None:
            return self.custom_status
        return CustomSkillsStatus(
            enabled=False,
            directory=None,
            source="disabled",
            state="disabled",
            discovered=0,
            loaded_names=(),
            issues=(),
            restart_required=False,
        )

    def _register_validated_batch(
        self, sds: tuple[SkillDefinition, ...]
    ) -> None:
        """Atomically register a fully-validated batch of SkillDefinitions.

        Acquires `self._lock` ONCE, updates `_inner` and `_trigger_map` for
        every definition, then rebuilds `_skills` exactly once at the end
        (CRIT R2-4: a single fresh `MappingProxyType` snapshot of `_inner`
        — readers never observe a torn write).
        """
        with self._lock:
            for sd in sds:
                self._inner[sd.name] = sd
                if sd.trigger:
                    self._trigger_map[sd.trigger] = sd.name
                for alias in sd.aliases:
                    self._trigger_map[alias] = sd.name
            object.__setattr__(
                self, "_skills", MappingProxyType(dict(self._inner))
            )

    # ---- Phase 1b new methods (§3 table) ----

    def get(self, name: str) -> Optional[Any]:
        """Return skill by name. Returns None if not registered."""
        return self._inner.get(name)

    def get_skill(self, name: str) -> Optional[Any]:
        """Alias for get()."""
        return self.get(name)

    def get_by_trigger(self, trigger: str) -> Optional[Any]:
        """Look up by trigger string (e.g. '^^ticket')."""
        with self._lock:
            skill_name = self._trigger_map.get(trigger)
            return self._inner.get(skill_name) if skill_name else None

    def list_skills(self) -> list:
        """Return all registered skill objects (SkillDefinition entries)."""
        with self._lock:
            return list(self._inner.values())

    def all(self) -> list:
        """Alias for list_skills()."""
        return self.list_skills()

    def __contains__(self, name: object) -> bool:
        """Return True if name is registered."""
        return name in self._inner

    def populate_builtins(self, *, trusted: bool = True) -> int:
        """Idempotently register ticket and work builtins + load custom JSON.

        Builtins first, then strict-schema custom-skill load per spec
        §5.1 (atomic, fail-closed). Returns the count of NEW builtin
        registrations (0 if all are already present).

        Custom-skill failures raise `SkillRegistryConfigurationError` —
        there is NO partial publication path.
        """
        from skills.ticket_skill import TicketSkill
        from skills.work_skill import WorkSkill
        from proxy.skills.skill_definition import TrustToken
        from proxy.tool_pipeline.tool_registry import registry as tool_registry

        count = 0
        for factory_class in (TicketSkill, WorkSkill):
            factory = factory_class()
            if factory.name in self._inner:
                continue
            with _held_trust_token(TrustToken.for_trusted_caller()):
                sd = factory.to_skill_definition()
            self.register_skill(
                sd, trusted_registrant=TrustToken.for_trusted_caller()
            )
            count += 1

        # Strict-schema custom-skill load (Wave 1 Agent 2 — spec §5).
        resolved = resolve_skills_dir()
        parsed_skills, issues, status = load_custom_skills(
            resolved.root,
            tool_catalog=tool_registry,
            source=resolved.source,
        )

        object.__setattr__(self, "custom_status", status)

        if issues:
            raise SkillRegistryConfigurationError(
                issues=tuple(issues),
                message=(
                    f"custom skills configuration invalid: "
                    f"{len(issues)} issue(s)"
                ),
            )

        if not parsed_skills:
            return count

        sds = tuple(
            build_custom_skill_definition(p) for p in parsed_skills
        )
        self._register_validated_batch(sds)

        return count

    # ---- Phase 1a methods (VERBATIM — do not change) ----

    def register(self, skill: Any) -> None:
        """Register a skill. Validates name format, rejects duplicates.

        CRIT R1-2 / R2-1: a trust gate blocks re-registration of any
        skill whose name is in `TRUSTED_BUILTIN_SKILLS` unless the
        immediate caller is in `TRUSTED_TRUST_CALLERS`. The check
        uses `inspect.stack()[1].frame.f_globals['__name__']` so the
        trust decision is keyed off the *actual* caller module, not
        a self-asserted argument.
        """
        if not re.match(r"^[a-z][a-z0-9_-]*$", skill.name):
            raise SkillRegistrationError(
                f"Invalid skill name '{skill.name}': must match ^[a-z][a-z0-9_-]*$"
            )

        with self._lock:
            # CRIT R1-2 / R2-1: trust gate — must run BEFORE the duplicate
            # check, so an untrusted caller cannot probe the existence of
            # a trusted builtin by attempting to re-register it.
            if is_trusted_skill(skill.name):
                try:
                    caller_frame = inspect.stack()[1]
                    caller_name = caller_frame.frame.f_globals.get("__name__", "")
                finally:
                    del caller_frame
                if caller_name not in TRUSTED_TRUST_CALLERS:
                    # Allow re-registration if the skill is already in _inner
                    # (e.g., test patterns that clear and re-register). If not
                    # already registered, raise the gate error.
                    if skill.name not in self._inner:
                        raise SkillRegistrationError(
                            f"register(): cannot (re)register trusted builtin "
                            f"{skill.name!r} from untrusted caller {caller_name!r}"
                        )
                    # Already registered — skip silently.
                    return

            if skill.name in self._inner:
                raise SkillRegistrationError(f"Duplicate skill name: {skill.name}")
            if skill.trigger and skill.trigger in self._trigger_map:
                raise SkillRegistrationError(f"Duplicate trigger: {skill.trigger}")
            for alias in skill.aliases:
                if alias in self._trigger_map:
                    raise SkillRegistrationError(f"Duplicate alias: {alias}")

            # 1a deviation: write to _inner (mutable storage), not _skills
            # (read-only MappingProxyType view of _inner).
            self._inner[skill.name] = skill
            if skill.trigger:
                self._trigger_map[skill.trigger] = skill.name
            for alias in skill.aliases:
                self._trigger_map[alias] = skill.name

            # CRIT R2-4: rebind self._skills to a fresh MappingProxyType
            # snapshot of _inner. The previous proxy is a live view of
            # the now-mutated dict; replacing it gives external readers
            # an atomic post-write snapshot (no torn writes visible to
            # concurrent get()/get_by_trigger() callers).
            # frozen=True blocks plain attribute assignment — use
            # object.__setattr__ to bypass the frozen guard.
            object.__setattr__(self, "_skills", MappingProxyType(dict(self._inner)))

    # ---- 1a addition: trust-gated register_skill (R1 HIGH-4) ----
    def register_skill(
        self,
        sd: SkillDefinition,
        *,
        trusted_registrant: TrustToken = TrustToken.UNTRUSTED,
    ) -> None:
        """Register a `SkillDefinition`. CAT-3 trust-gate (R1 HIGH-4, R2-2).

        The trust gate is the SAME gate the full 1b registry uses:
        untrusted callers (e.g., a user-defined skill plugin) cannot
        register a name that is in `TRUSTED_BUILTIN_SKILLS` (that would
        be a privilege escalation). Raises `SkillRegistrationError`
        (a `PermissionError` subclass) on a denied registration.

        CRIT R2-2: `trusted_registrant` is now a `TrustToken`, not a
        `bool`. The token is unforgeable — the only way to obtain a
        TRUSTED instance is `TrustToken.for_trusted_caller()`, which
        inspects the call stack. A caller cannot pass `True` to
        bypass the gate; the bool→TrustToken type mismatch raises
        `TypeError` at the call boundary.

        CRIT R2-5: body runs under `self._lock` so concurrent
        `register()` and `register_skill()` calls do not race on
        `self._inner` / `self._trigger_map` / `self._skills`.

        HIGH: `sd` MUST be a `SkillDefinition` (subclasses allowed);
        anything else is rejected with `SkillRegistrationError`.

        CRIT R1-3: if the name is already in `_inner` AND the existing
        entry is NOT a `SkillDefinition` (i.e., it was registered via
        the legacy `register()` path as a builtin), the registration
        is rejected — builtins are protected from silent overwrite.
        Overwrite of an existing `SkillDefinition` IS allowed.
        """
        # HIGH: subclass validation. `SkillDefinition` is the public
        # contract for `register_skill`; anything else is a caller bug.
        if not isinstance(sd, SkillDefinition):
            raise SkillRegistrationError(
                f"register_skill(): expected SkillDefinition, got "
                f"{type(sd).__name__}"
            )

        with self._lock:
            # CRIT R1-3: protect builtins from silent overwrite.
            # If the name is already taken by a NON-SkillDefinition
            # entry (i.e., a legacy-Skill builtin from register()),
            # refuse the write. Re-registering a SkillDefinition
            # (e.g., during a hot-reload of plugin config) is OK.
            if sd.name in self._inner:
                existing = self._inner[sd.name]
                if not isinstance(existing, SkillDefinition):
                    raise SkillRegistrationError(
                        f"Already registered: {sd.name!r} "
                        f"(existing entry is a builtin Skill, not a "
                        f"SkillDefinition; refusing to overwrite)"
                    )

            if sd.name in TRUSTED_BUILTIN_SKILLS and not trusted_registrant.is_trusted:
                raise SkillRegistrationError(
                    f"Cannot register reserved builtin name {sd.name!r} "
                    f"as untrusted registrant"
                )

            _sd_trigger = getattr(sd, "trigger", None)
            if _sd_trigger:
                if _sd_trigger in self._trigger_map and self._trigger_map[_sd_trigger] != sd.name:
                    raise SkillRegistrationError(
                        f"Duplicate trigger {_sd_trigger!r}: already "
                        f"used by skill {self._trigger_map[_sd_trigger]!r}, "
                        f"cannot reuse for {sd.name!r}"
                    )
            _sd_aliases = getattr(sd, "aliases", ())
            for _alias in _sd_aliases:
                if _alias in self._trigger_map and self._trigger_map[_alias] != sd.name:
                    raise SkillRegistrationError(
                        f"Duplicate alias {_alias!r}: already used by "
                        f"skill {self._trigger_map[_alias]!r}, cannot "
                        f"reuse for {sd.name!r}"
                    )

            self._inner[sd.name] = sd

            object.__setattr__(self, "_skills", MappingProxyType(dict(self._inner)))

            if _sd_trigger:
                self._trigger_map[_sd_trigger] = sd.name
            for _alias in _sd_aliases:
                self._trigger_map[_alias] = sd.name


# ---------------------------------------------------------------------------
# Module-level functions (Phase 1b §5)
# ---------------------------------------------------------------------------
def get_default_registry() -> SkillRegistry:
    """Return the module-level singleton SkillRegistry.

    First call: constructs via `_skill_registry_new__` → `populate_builtins(trusted=True)` →
    publishes to `_default_registry`. Subsequent calls return the same object.

    Wave 2: this function is called eagerly from `qwen_proxy.py:__main__` via
    `_eager_init_registry()`. WSGI deployments also trigger it via the
    `on_starting` hook in `gunicorn.conf.py` at the repo root (which
    invokes `proxy.app.initialize_proxy_at_startup` → `_eager_init_registry`
    in the master process before workers fork). The WSGI shape
    additionally falls back to lazy first-request construction if the
    gunicorn conf is bypassed, but production deployments should rely on
    the gunicorn auto-loaded hook per spec §5.1.

    Raises:
        SkillRegistryConfigurationError: when populate_builtins detects
            invalid custom skills configuration (per spec §5.1 atomic
            fail-closed).
    """
    global _default_registry
    if _default_registry is not None:
        return _default_registry
    with _registry_init_lock:
        if _default_registry is not None:  # double-check
            return _default_registry
        # Use _skill_registry_new__ to bypass __new__ shim and avoid re-entry.
        registry = _skill_registry_new__(SkillRegistry)
        registry.populate_builtins(trusted=True)
        _default_registry = registry
        return registry


def reset_default_registry() -> None:
    """Unconditionally clear the module-level singleton.

    The next call to get_default_registry() constructs a fresh instance.

    Also clears the OLD registry's internal state (FIX-4) so any thread
    that captured a reference to the previous singleton cannot continue
    to use a fully populated registry (split-brain).
    """
    global _default_registry
    with _registry_init_lock:
        if _default_registry is not None:
            old = _default_registry
            old._inner.clear()
            old._trigger_map.clear()
            # _skills is a read-only proxy — leave it; a stale proxy
            # pointing at a now-empty _inner is safe and self-consistent.
        _default_registry = None


# Install the thin __new__ shim on the dataclass (after reset_default_registry is defined).
SkillRegistry.__new__ = staticmethod(__new__)


# ---------------------------------------------------------------------------
# Skill error response helper
# ---------------------------------------------------------------------------

def error_response(message: str, status_code: int = 400) -> dict:
    """Return an OpenAI-style error response for a skill direct handler.

    Use this when handle_direct encounters a user-facing error (not found,
    ambiguous hash, infrastructure failure). Raises are reserved for bugs.

    The ``_status_code`` key is consumed by dispatch.py to set the HTTP
    response status.  Callers that need a specific code can pass it explicitly.
    """
    return {
        "choices": [{"message": {"role": "assistant",
                                 "content": f"Error: {message}"}}],
        "model": "skill",
        "_status_code": status_code,
    }
