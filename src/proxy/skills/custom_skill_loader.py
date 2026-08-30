"""Strict-schema loader for JSON-defined custom skills.

Implements the canonical schema and loader contract defined in
`docs/specs/custom-skills-remediation.md` §4-§5.

Design summary:
- Atomic, fail-closed: any single file with a validation issue fails the
  ENTIRE custom configuration. No partial publication.
- Strict schema: unknown top-level keys, reserved names, cross-skill
  collisions, oversized inputs, and oversized alias/tool sets are
  REJECTED (not silently dropped, not warned).
- Tool-policy validation: every entry in `tool_whitelist` must be a
  canonical tool name from the `ToolRegistry` catalog; `proxy_ticket`,
  `proxy_work`, and `task` are ALWAYS rejected for custom skills
  regardless of author intent.
- Safe I/O: file reads go through `proxy.skills.path_utils.read_contained_regular_file`
  (Agent 1's safe-read helper) for both JSON definitions and prompt files.
- No hot reload: this loader runs once at startup via `populate_builtins()`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

from proxy.skills.path_utils import (
    SkillPathError,
    get_skills_dir,
    read_contained_regular_file,
)
from proxy.skills.skill_definition import SubagentPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — single source of truth (spec §4.6 + §4.4 + §4.5)
# ---------------------------------------------------------------------------

MAX_SKILL_DEFINITION_BYTES: int = 64 * 1024
MAX_SKILL_PROMPT_BYTES: int = 256 * 1024
MAX_CUSTOM_SKILL_FILES: int = 256
MAX_CUSTOM_SKILL_TTL_SECONDS: int = 86_400
MIN_CUSTOM_SKILL_TTL_SECONDS: int = 1
MAX_ALIASES_PER_SKILL: int = 16
MAX_TOOLS_PER_WHITELIST: int = 64
MAX_DESCRIPTION_LENGTH: int = 160

# Permanently denied to all custom skills regardless of opt-in (spec §0.1).
PERMANENTLY_DENIED_FOR_CUSTOM: frozenset[str] = frozenset(
    {"proxy_ticket", "proxy_work"}
)
# Conditionally admitted — only when the same JSON object also declares
# a structurally-valid `subagents` policy and the deployment flag is "1".
# See `_check_tool_whitelist` and AC 16.16.
CONDITIONALLY_ADMITTED_FOR_CUSTOM: frozenset[str] = frozenset({"task"})

# Reserved names — builtins already occupy these in `TRUSTED_BUILTIN_SKILLS`.
RESERVED_BUILTIN_NAMES: frozenset[str] = frozenset({"ticket", "work"})

# Reserved triggers — the command dispatcher already owns these.
RESERVED_COMMAND_TRIGGERS: frozenset[str] = frozenset(
    {"^^help", "^^status", "^^workspace", "^^ws"}
)

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
TRIGGER_PATTERN = re.compile(r"^\^\^[a-z][a-z0-9_-]*$")
FILENAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*\.json$")

# Per spec §4.1-§4.3 — these are the ONLY top-level keys the loader accepts.
ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "trigger",
        "description",
        "prompt",
        "prompt_file",
        "tool_whitelist",
        "aliases",
        "ttl_seconds",
        "subagents",
    }
)

DEFAULT_TTL_SECONDS: int = 300

# UTF-8 BOM. Rejected — we do not silently switch to utf-8-sig (spec §5.5).
_UTF8_BOM = b"\xef\xbb\xbf"


# ---------------------------------------------------------------------------
# Immutable result types (spec §5.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedCustomSkill:
    """The successful parse of a single JSON skill definition.

    Built before any registration-side concern. The build step then
    converts this to a `SkillDefinition`.
    """

    name: str
    trigger: str
    description: str
    aliases: tuple[str, ...]
    prompt_text: str
    allowed_tool_names: frozenset[str]
    ttl_seconds: int
    source_file: str  # basename only — never absolute
    subagents: Optional[SubagentPolicy] = None


@dataclass(frozen=True)
class SkillLoadIssue(Exception):
    """A single validation failure tied to a specific file.

    `file` is the basename of the offending file (never absolute) so the
    diagnostics are safe to log in user-facing surfaces. Subclasses
    `Exception` so per-file validation paths can `raise SkillLoadIssue(...)`
    and the aggregate caller can catch + accumulate.
    """

    file: str
    code: str
    message: str


@dataclass(frozen=True)
class CustomSkillsStatus:
    """The custom-skills snapshot consumed by `^^status` and `/health`.

    Set once during `populate_builtins()` and exposed via
    `SkillRegistry.custom_skills_status()`.
    """

    enabled: bool
    directory: Optional[str]  # None when disabled
    source: str  # "env" | "default" | "disabled"
    state: str  # "ok" | "degraded" | "disabled"
    discovered: int
    loaded_names: tuple[str, ...]
    issues: tuple[SkillLoadIssue, ...]
    restart_required: bool
    swarm_enabled: bool = False
    swarm_capable_skills: tuple[str, ...] = ()


class SkillRegistryConfigurationError(RuntimeError):
    """Raised when the custom-skill configuration is invalid.

    Atomic: any single failing file produces this error with the FULL
    aggregated issue list. No partial custom set is published.
    """

    def __init__(
        self,
        issues: tuple[SkillLoadIssue, ...],
        message: str = "",
    ) -> None:
        self.issues = issues
        super().__init__(
            message or f"custom skills configuration invalid: {len(issues)} issue(s)"
        )


class LoadResult(NamedTuple):
    """The output of `load_custom_skills`."""

    parsed_skills: tuple[ParsedCustomSkill, ...]
    issues: tuple[SkillLoadIssue, ...]
    status: CustomSkillsStatus


# ---------------------------------------------------------------------------
# Path / repo-root helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedSkillsDir:
    """The PROXY_SKILLS_DIR resolution outcome.

    `root` is None when custom loading is disabled (env empty string) OR
    when env is unset and the default `<repo>/skills` is missing. When
    `enabled` is True, `root` is always set.
    """

    root: Optional[Path]
    source: str  # "env" | "default" | "disabled"
    enabled: bool


def resolve_skills_dir() -> ResolvedSkillsDir:
    """Resolve the configured skills directory per spec §5.1 / §5.2.

    Membership of `PROXY_SKILLS_DIR` in `os.environ` (NOT truthiness) is
    the gate:

    | Env state                           | Result                          |
    |-------------------------------------|---------------------------------|
    | unset                               | source="default", path=default  |
    | empty string                        | source="disabled", path=None    |
    | set, path exists and is a dir       | source="env", path=resolved     |
    | set, path missing                   | FATAL SkillRegistryConfigurationError |
    | set, path exists but is not a dir   | FATAL SkillRegistryConfigurationError |

    For source="default", the on-disk state is NOT checked here — the
    loader (`load_custom_skills`) treats a missing default as a healthy
    empty configuration per spec §5.1 ("Missing default <repo>/skills
    directory: healthy; builtins only"). The split is intentional:
    `resolve_skills_dir` reports the structural resolution; the loader
    decides fatal-vs-healthy based on `source`.
    """
    if "PROXY_SKILLS_DIR" in os.environ:
        env_val = os.environ["PROXY_SKILLS_DIR"]
        if env_val == "":
            return ResolvedSkillsDir(None, "disabled", False)
        # env-set: on-disk validity is REQUIRED and is fatal on failure.
        resolved = get_skills_dir()
        if resolved is None:
            # Defensive: get_skills_dir() returns None only for env=""
            # which the branch above already handled.
            return ResolvedSkillsDir(None, "disabled", False)
        if not resolved.exists():
            raise SkillRegistryConfigurationError(
                issues=(),
                message=(
                    f"PROXY_SKILLS_DIR does not exist "
                    f"(resolved to {resolved})"
                ),
            )
        if not resolved.is_dir():
            raise SkillRegistryConfigurationError(
                issues=(),
                message=(
                    f"PROXY_SKILLS_DIR is not a directory "
                    f"(resolved to {resolved})"
                ),
            )
        return ResolvedSkillsDir(resolved, "env", True)

    # Env unset → default. Do NOT raise on a missing default; the loader
    # treats that as a healthy builtins-only state per spec §5.1.
    resolved = get_skills_dir()
    if resolved is None:
        # Defensive: get_skills_dir() returns the default repo path when
        # env is unset, so this branch is unreachable in practice.
        return ResolvedSkillsDir(None, "disabled", False)
    return ResolvedSkillsDir(resolved, "default", True)


# ---------------------------------------------------------------------------
# Tool catalog adapter — bridges to either current or Agent 3 API
# ---------------------------------------------------------------------------


def _tool_catalog_names(tool_catalog: Any) -> frozenset[str]:
    """Return the canonical tool-name set from the given catalog.

    Uses Agent 3's `get_tool_names()` method.
    """
    if hasattr(tool_catalog, "get_tool_names"):
        try:
            names = tool_catalog.get_tool_names()
            return frozenset(names) if not isinstance(names, frozenset) else names
        except Exception:
            pass
    # Current API fallback.
    if hasattr(tool_catalog, "_tools"):
        try:
            return frozenset(tool_catalog._tools.keys())
        except Exception:
            pass
    raise RuntimeError(
        "tool_catalog has no known tool-naming API "
        "(expected get_tool_names() or _tools dict)"
    )


def _tool_catalog_is_admin(tool_catalog: Any, name: str) -> bool:
    """Return True iff `name` is a proxy admin tool in the catalog."""
    if hasattr(tool_catalog, "is_proxy_admin_tool"):
        try:
            return bool(tool_catalog.is_proxy_admin_tool(name))
        except Exception:
            pass
    try:
        from proxy.tool_pipeline.tool_registry import _PROXY_ADMIN_TOOLS  # type: ignore
        return name in _PROXY_ADMIN_TOOLS
    except ImportError:
        pass
    return name in {"proxy_ticket", "proxy_work"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_description(raw: Any, *, file: str) -> Optional[SkillLoadIssue]:
    """Validate `description` per spec §4.1."""
    if not isinstance(raw, str):
        return SkillLoadIssue(
            file=file,
            code="description_not_string",
            message="description must be a string",
        )
    if not raw.strip():
        return SkillLoadIssue(
            file=file,
            code="description_empty",
            message="description must not be empty",
        )
    if len(raw) > MAX_DESCRIPTION_LENGTH:
        return SkillLoadIssue(
            file=file,
            code="description_too_long",
            message=(
                f"description is {len(raw)} chars, exceeds "
                f"limit {MAX_DESCRIPTION_LENGTH}"
            ),
        )
    # Reject control characters below \x20 except \t, \n, \r.
    for ch in raw:
        cp = ord(ch)
        if cp < 0x20 and cp not in (0x09, 0x0A, 0x0D):
            return SkillLoadIssue(
                file=file,
                code="description_control_chars",
                message=(
                    f"description contains prohibited control character "
                    f"U+{cp:04X}"
                ),
            )
        cat = unicodedata.category(ch)
        if cat.startswith("C") and cp >= 0x20:
            return SkillLoadIssue(
                file=file,
                code="description_control_chars",
                message=(
                    f"description contains prohibited Unicode "
                    f"category {cat!r} (U+{cp:04X})"
                ),
            )
    return None


def _check_name(raw: Any, *, expected_stem: str, file: str) -> Optional[SkillLoadIssue]:
    if not isinstance(raw, str):
        return SkillLoadIssue(
            file=file, code="name_not_string", message="name must be a string"
        )
    if not NAME_PATTERN.match(raw):
        return SkillLoadIssue(
            file=file,
            code="name_invalid_format",
            message=f"name {raw!r} must match ^[a-z][a-z0-9_-]*$",
        )
    if raw != expected_stem:
        return SkillLoadIssue(
            file=file,
            code="name_filename_mismatch",
            message=(
                f"name {raw!r} must equal filename stem {expected_stem!r}"
            ),
        )
    if raw in RESERVED_BUILTIN_NAMES:
        return SkillLoadIssue(
            file=file,
            code="reserved_name",
            message=f"name {raw!r} is reserved (built-in skill)",
        )
    return None


def _check_trigger(
    raw: Any, *, expected: str, file: str
) -> Optional[SkillLoadIssue]:
    if not isinstance(raw, str):
        return SkillLoadIssue(
            file=file,
            code="trigger_not_string",
            message="trigger must be a string",
        )
    if not TRIGGER_PATTERN.match(raw):
        return SkillLoadIssue(
            file=file,
            code="trigger_invalid_format",
            message=(
                f"trigger {raw!r} must match ^\\^\\^[a-z][a-z0-9_-]*$"
            ),
        )
    if raw != expected:
        return SkillLoadIssue(
            file=file,
            code="trigger_name_mismatch",
            message=(
                f"trigger {raw!r} must equal '^^' + name {expected[2:]!r}"
            ),
        )
    if raw in RESERVED_COMMAND_TRIGGERS:
        return SkillLoadIssue(
            file=file,
            code="reserved_trigger",
            message=f"trigger {raw!r} is reserved (built-in command)",
        )
    return None


def _check_aliases(
    raw: Any, *, primary_trigger: str, file: str
) -> Optional[SkillLoadIssue]:
    if not isinstance(raw, list):
        return SkillLoadIssue(
            file=file,
            code="aliases_not_list",
            message="aliases must be a list",
        )
    if len(raw) > MAX_ALIASES_PER_SKILL:
        return SkillLoadIssue(
            file=file,
            code="aliases_too_many",
            message=(
                f"aliases has {len(raw)} entries, exceeds "
                f"limit {MAX_ALIASES_PER_SKILL}"
            ),
        )
    reserved_triggers = RESERVED_COMMAND_TRIGGERS | {
        f"^^{name}" for name in RESERVED_BUILTIN_NAMES
    }
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, str):
            return SkillLoadIssue(
                file=file,
                code="alias_not_string",
                message=f"aliases[{i}] must be a string",
            )
        if not TRIGGER_PATTERN.match(entry):
            return SkillLoadIssue(
                file=file,
                code="alias_invalid_format",
                message=(
                    f"aliases[{i}] {entry!r} must match "
                    f"^\\^\\^[a-z][a-z0-9_-]*$"
                ),
            )
        if entry == primary_trigger:
            return SkillLoadIssue(
                file=file,
                code="alias_equals_trigger",
                message=f"aliases[{i}] {entry!r} equals primary trigger",
            )
        if entry in reserved_triggers:
            return SkillLoadIssue(
                file=file,
                code="alias_reserved",
                message=(
                    f"aliases[{i}] {entry!r} is reserved "
                    f"(built-in trigger or command trigger)"
                ),
            )
        if entry in seen:
            return SkillLoadIssue(
                file=file,
                code="duplicate_alias",
                message=f"aliases[{i}] {entry!r} duplicates a prior alias",
            )
        seen.add(entry)
    return None


def _tool_catalog_names_listing(tool_catalog: Any) -> str:
    """Helper that returns a sorted, comma-joined listing of known tool names."""
    try:
        names = sorted(_tool_catalog_names(tool_catalog))
    except Exception:
        return "<unavailable>"
    if len(names) > 20:
        return ", ".join(names[:20]) + ", ..."
    return ", ".join(names)


def _check_tool_whitelist(
    raw: Any,
    *,
    tool_catalog: Any,
    file: str,
    subagents_present: bool,
) -> Optional[SkillLoadIssue]:
    """Validate the top-level `tool_whitelist` per spec §0.1 + §4.4.

    `subagents_present` is True iff the same JSON object declares a
    `subagents` key (regardless of whether the nested object is
    structurally valid). When `task` is in the whitelist and
    `subagents_present` is False, we emit `swarm_subagents_missing`
    instead of `tool_whitelist_denied` — that error code belongs to the
    canonical permanently-denied set. When `subagents_present` is True,
    the `task` entry receives a provisional pass through this check; the
    post-validation swarm gate (`_check_swarm_outer`) and contradiction
    check decide final admission.
    """
    if raw is None:
        return SkillLoadIssue(
            file=file,
            code="tool_whitelist_missing",
            message=(
                "tool_whitelist is required and must be a non-empty list "
                "(spec §4.4 — missing list falls open to allow-all)"
            ),
        )
    if not isinstance(raw, list):
        return SkillLoadIssue(
            file=file,
            code="tool_whitelist_not_list",
            message="tool_whitelist must be a list",
        )
    if len(raw) > MAX_TOOLS_PER_WHITELIST:
        return SkillLoadIssue(
            file=file,
            code="tool_whitelist_too_many",
            message=(
                f"tool_whitelist has {len(raw)} entries, exceeds "
                f"limit {MAX_TOOLS_PER_WHITELIST}"
            ),
        )
    known = _tool_catalog_names(tool_catalog)
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, str):
            return SkillLoadIssue(
                file=file,
                code="tool_whitelist_entry_not_string",
                message=f"tool_whitelist[{i}] must be a string",
            )
        if not entry:
            return SkillLoadIssue(
                file=file,
                code="tool_whitelist_entry_empty",
                message=f"tool_whitelist[{i}] must be non-empty",
            )
        if entry in seen:
            return SkillLoadIssue(
                file=file,
                code="duplicate_tool",
                message=f"tool_whitelist[{i}] {entry!r} duplicates a prior entry",
            )
        seen.add(entry)
        if entry in PERMANENTLY_DENIED_FOR_CUSTOM:
            return SkillLoadIssue(
                file=file,
                code="tool_whitelist_denied",
                message=(
                    f"tool_whitelist[{i}] {entry!r} is forbidden "
                    f"for custom skills (proxy_ticket and proxy_work "
                    f"are reserved for trusted builtins)"
                ),
            )
        if entry in CONDITIONALLY_ADMITTED_FOR_CUSTOM:
            # `task` — only admissible when a structurally-valid
            # `subagents` policy is present in the same JSON object.
            if not subagents_present:
                return SkillLoadIssue(
                    file=file,
                    code="swarm_subagents_missing",
                    message=(
                        f"tool_whitelist[{i}] {entry!r} requires a "
                        f"structurally-valid `subagents` policy in the "
                        f"same JSON object"
                    ),
                )
            # Provisional pass — skip the unknown_tool check.
            # The post-validation swarm gate decides final admission.
            continue
        if entry not in known:
            return SkillLoadIssue(
                file=file,
                code="unknown_tool",
                message=(
                    f"tool_whitelist[{i}] {entry!r} is not a known tool "
                    f"(canonical names: {_tool_catalog_names_listing(tool_catalog)})"
                ),
            )
    return None


def _check_ttl(raw: Any, *, file: str) -> Optional[SkillLoadIssue]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return SkillLoadIssue(
            file=file,
            code="ttl_bool_rejected",
            message=(
                "ttl_seconds must be a non-bool int "
                "(isinstance(True, int) is True; bool is not accepted)"
            ),
        )
    if not isinstance(raw, int):
        return SkillLoadIssue(
            file=file,
            code="ttl_not_int",
            message=f"ttl_seconds must be int, got {type(raw).__name__}",
        )
    if raw < MIN_CUSTOM_SKILL_TTL_SECONDS:
        return SkillLoadIssue(
            file=file,
            code="ttl_too_small",
            message=(
                f"ttl_seconds={raw} is below minimum "
                f"{MIN_CUSTOM_SKILL_TTL_SECONDS}"
            ),
        )
    if raw > MAX_CUSTOM_SKILL_TTL_SECONDS:
        return SkillLoadIssue(
            file=file,
            code="ttl_too_large",
            message=(
                f"ttl_seconds={raw} exceeds maximum "
                f"{MAX_CUSTOM_SKILL_TTL_SECONDS}"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Swarm-specific constants and helpers (spec §0.1, §7, §10)
# ---------------------------------------------------------------------------

# Tools that custom-skill CHILD sub-agents can never use. Universally denied
# regardless of operator opt-in:
# - `task` — universal no-nesting (§2)
# - `question` — sub-agents do not prompt the operator
# - `proxy_ticket` / `proxy_work` — proxy memory admin surfaces, CAT-3 gate
SWARM_CHILD_FORBIDDEN: frozenset[str] = frozenset(
    {"task", "question", "proxy_ticket", "proxy_work"}
)

# Bounds for `subagents.max_task_calls_per_request` (spec §9).
MIN_SWARM_TASK_CALLS_PER_REQUEST: int = 1
MAX_SWARM_TASK_CALLS_PER_REQUEST: int = 16

# Closed nested schema (spec §7). Reject any other top-level key inside
# the `subagents` object.
ALLOWED_SUBAGENT_KEYS: frozenset[str] = frozenset(
    {"tool_whitelist", "max_task_calls_per_request"}
)


def _parse_subagent_policy(
    raw: Any,
    *,
    tool_catalog: Any,
    file: str,
) -> tuple[Optional[SubagentPolicy], tuple[SkillLoadIssue, ...]]:
    """Parse the nested `subagents` object into a SubagentPolicy.

    Validation order (spec §10):
      1. `tool_whitelist` — must be a list of canonical tool names; no
         duplicates; no SWARM_CHILD_FORBIDDEN entries; no unknown tools.
      2. `max_task_calls_per_request` — non-bool int in [1, 16].

    All validation issues are accumulated and returned together. The
    caller decides whether to aggregate or short-circuit.

    Returns (None, ()) when `raw` is missing OR not a dict — the caller
    emits `swarm_subagents_missing` or `swarm_subagents_not_object` in
    that case. This helper assumes `raw` is a dict when called.
    """
    if not isinstance(raw, dict):
        return None, ()

    issues: list[SkillLoadIssue] = []

    # Closed-schema gate first (spec §7 — no `enabled`, `children`, etc.).
    unknown_keys = set(raw.keys()) - ALLOWED_SUBAGENT_KEYS
    if unknown_keys:
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_subagents_unknown_key",
                message=(
                    f"unknown subagents keys: {sorted(unknown_keys)!r} "
                    f"(allowed: {sorted(ALLOWED_SUBAGENT_KEYS)})"
                ),
            )
        )

    # Phase 1 — tool_whitelist.
    child_tw = raw.get("tool_whitelist")
    child_tw_valid = True
    child_tw_set: set[str] = set()
    if "tool_whitelist" not in raw or child_tw is None:
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_tool_whitelist_missing",
                message="subagents.tool_whitelist is required",
            )
        )
        child_tw_valid = False
    elif not isinstance(child_tw, list):
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_tool_whitelist_not_list",
                message="subagents.tool_whitelist must be a list",
            )
        )
        child_tw_valid = False
    else:
        known = _tool_catalog_names(tool_catalog)
        seen: set[str] = set()
        for i, entry in enumerate(child_tw):
            if not isinstance(entry, str) or not entry:
                issues.append(
                    SkillLoadIssue(
                        file=file,
                        code="swarm_tool_whitelist_entry_invalid",
                        message=(
                            f"subagents.tool_whitelist[{i}] must be a "
                            f"non-empty string"
                        ),
                    )
                )
                child_tw_valid = False
                continue
            if entry in seen:
                issues.append(
                    SkillLoadIssue(
                        file=file,
                        code="swarm_duplicate_tool",
                        message=(
                            f"subagents.tool_whitelist[{i}] {entry!r} "
                            f"duplicates a prior entry"
                        ),
                    )
                )
                child_tw_valid = False
                continue
            seen.add(entry)
            if entry in SWARM_CHILD_FORBIDDEN:
                issues.append(
                    SkillLoadIssue(
                        file=file,
                        code="swarm_child_forbidden_tool",
                        message=(
                            f"subagents.tool_whitelist[{i}] {entry!r} "
                            f"is forbidden in sub-agent context"
                        ),
                    )
                )
                child_tw_valid = False
                continue
            if entry not in known:
                issues.append(
                    SkillLoadIssue(
                        file=file,
                        code="swarm_unknown_tool",
                        message=(
                            f"subagents.tool_whitelist[{i}] {entry!r} "
                            f"is not a known tool "
                            f"(canonical names: {_tool_catalog_names_listing(tool_catalog)})"
                        ),
                    )
                )
                child_tw_valid = False
                continue
            child_tw_set.add(entry)

    # Phase 2 — max_task_calls_per_request.
    max_calls_present = "max_task_calls_per_request" in raw
    max_calls = raw.get("max_task_calls_per_request")
    max_calls_valid = True
    if not max_calls_present or max_calls is None:
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_max_calls_missing",
                message=(
                    "subagents.max_task_calls_per_request is required"
                ),
            )
        )
        max_calls_valid = False
    elif isinstance(max_calls, bool):
        # bool must be rejected BEFORE the int check (isinstance(True, int) is True).
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_max_calls_not_integer",
                message=(
                    "subagents.max_task_calls_per_request must be a "
                    "non-bool int (bool is not accepted)"
                ),
            )
        )
        max_calls_valid = False
    elif not isinstance(max_calls, int):
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_max_calls_not_integer",
                message=(
                    f"subagents.max_task_calls_per_request must be int, "
                    f"got {type(max_calls).__name__}"
                ),
            )
        )
        max_calls_valid = False
    elif (
        max_calls < MIN_SWARM_TASK_CALLS_PER_REQUEST
        or max_calls > MAX_SWARM_TASK_CALLS_PER_REQUEST
    ):
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_max_calls_out_of_range",
                message=(
                    f"subagents.max_task_calls_per_request={max_calls} "
                    f"must be in [{MIN_SWARM_TASK_CALLS_PER_REQUEST}, "
                    f"{MAX_SWARM_TASK_CALLS_PER_REQUEST}]"
                ),
            )
        )
        max_calls_valid = False

    if issues:
        return None, tuple(issues)

    # All validation passed — construct the policy (requires SubagentPolicy).
    assert child_tw_valid and max_calls_valid  # type narrow
    assert isinstance(max_calls, int) and not isinstance(max_calls, bool)
    return (
        SubagentPolicy(
            tool_whitelist=frozenset(child_tw_set),
            max_task_calls_per_request=max_calls,
        ),
        (),
    )


def _check_swarm_outer(*, file: str) -> tuple[SkillLoadIssue, ...]:
    """Return per-file swarm outer issues (flag + secret).

    Both checks fire independently — a file may report BOTH
    `swarm_flag_disabled` AND `swarm_missing_secret`.
    """
    issues: list[SkillLoadIssue] = []
    if os.environ.get("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS") != "1":
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_flag_disabled",
                message=(
                    "subagents policy present but "
                    "PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS != '1'"
                ),
            )
        )
    if not os.environ.get("PROXY_SESSION_SECRET"):
        issues.append(
            SkillLoadIssue(
                file=file,
                code="swarm_missing_secret",
                message=(
                    "subagents policy present but "
                    "PROXY_SESSION_SECRET is unset or empty"
                ),
            )
        )
    return tuple(issues)


def _check_unknown_keys(data: dict, *, file: str) -> Optional[SkillLoadIssue]:
    extras = set(data.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if extras:
        return SkillLoadIssue(
            file=file,
            code="unknown_field",
            message=(
                f"unknown top-level keys: {sorted(extras)!r} "
                f"(allowed: {sorted(ALLOWED_TOP_LEVEL_KEYS)})"
            ),
        )
    return None


def _decode_strict_utf8(raw_bytes: bytes, *, file: str, kind: str) -> str:
    """Decode bytes as strict UTF-8, rejecting BOMs."""
    if raw_bytes.startswith(_UTF8_BOM):
        raise SkillLoadIssue(
            file=file,
            code=f"{kind}_bom",
            message=f"{kind} file begins with UTF-8 BOM (rejected per spec §5.5)",
        )
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillLoadIssue(
            file=file,
            code=f"{kind}_invalid_utf8",
            message=(
                f"{kind} file is not valid UTF-8: {exc.reason} "
                f"at byte {exc.start}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# File discovery + parsing
# ---------------------------------------------------------------------------


def discover_custom_skill_files(root: Path) -> tuple[Path, ...]:
    """Return sorted candidate JSON files in `root`."""
    if not isinstance(root, Path):
        raise TypeError(f"root must be a Path, got {type(root).__name__}")
    if not root.is_dir():
        raise ValueError(f"root {root!r} is not a directory")

    try:
        out: list[Path] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_file():
                continue
            if not FILENAME_PATTERN.match(entry.name):
                continue
            out.append(entry)
    except OSError as exc:
        raise SkillRegistryConfigurationError(
            issues=(),
            message=f"Cannot read skills directory {root}: {exc}",
        ) from exc
    if len(out) > MAX_CUSTOM_SKILL_FILES:
        raise SkillLoadIssue(
            file=root.name,
            code="too_many_files",
            message=(
                f"{len(out)} candidate files found, exceeds limit "
                f"{MAX_CUSTOM_SKILL_FILES}"
            ),
        )
    return tuple(out)


def _safe_read_bytes(root: Path, relative_path: str, *, max_bytes: int) -> bytes:
    """Read a regular file inside `root` via Agent 1's safe-read helper.

    Returns raw bytes (NOT decoded). Caller handles decoding.

    The safe-read helper enforces the spec §5.4 invariants (containment,
    regular-file, size, no symlinks, no `..`) and raises `SkillPathError`
    for any violation. The caller (parse_custom_skill_file) catches
    `SkillPathError` and translates to a per-file `SkillLoadIssue` —
    this translation inspects the message to recover a specific code
    (e.g. `definition_bom` vs `definition_too_large`).
    """
    text = read_contained_regular_file(
        root, relative_path, max_bytes=max_bytes
    )
    return text.encode("utf-8")


_BOM_HINT = "BOM"
_UTF8_HINT = "UTF-8"
_SIZE_HINT = "exceeds limit"


def _classify_read_error(exc: "SkillPathError") -> str:
    """Map a SkillPathError message to a loader-specific code suffix."""
    msg = str(exc)
    if _BOM_HINT in msg:
        return "bom"
    if _UTF8_HINT in msg:
        return "invalid_utf8"
    if _SIZE_HINT in msg:
        return "too_large"
    return "unreadable"


def parse_custom_skill_file(
    path: Path, root: Path, tool_catalog: Any
) -> tuple[Optional[ParsedCustomSkill], tuple[SkillLoadIssue, ...]]:
    """Parse + validate one custom-skill JSON file.

    Returns:
        (parsed, ()) on success.
        (None, (issue,)) on single-issue failure.
        (None, (issue1, issue2, ...)) when multiple issues aggregate.

    All validation issues are reported via the return tuple — this
    function NEVER raises `SkillLoadIssue`. The caller accumulates
    per-file issues and decides atomic publication.
    """
    file_basename = path.name
    expected_stem = path.stem

    try:
        raw_bytes = _safe_read_bytes(
            root, file_basename, max_bytes=MAX_SKILL_DEFINITION_BYTES
        )
    except SkillPathError as exc:
        return None, (
            SkillLoadIssue(
                file=file_basename,
                code=f"definition_{_classify_read_error(exc)}",
                message=str(exc),
            ),
        )
    except (OSError, ValueError) as exc:
        return None, (
            SkillLoadIssue(
                file=file_basename,
                code="definition_unreadable",
                message=str(exc),
            ),
        )

    try:
        text = _decode_strict_utf8(
            raw_bytes, file=file_basename, kind="definition"
        )
    except SkillLoadIssue as exc:
        return None, (exc,)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, (
            SkillLoadIssue(
                file=file_basename,
                code="invalid_json",
                message=(
                    f"not valid JSON: {exc.msg} at line {exc.lineno} "
                    f"col {exc.colno}"
                ),
            ),
        )

    if not isinstance(data, dict):
        return None, (
            SkillLoadIssue(
                file=file_basename,
                code="not_object",
                message=(
                    f"top-level must be an object, got {type(data).__name__}"
                ),
            ),
        )

    unknown_issue = _check_unknown_keys(data, file=file_basename)
    if unknown_issue is not None:
        return None, (unknown_issue,)

    name = data.get("name")
    name_issue = _check_name(
        name, expected_stem=expected_stem, file=file_basename
    )
    if name_issue is not None:
        return None, (name_issue,)
    assert isinstance(name, str)

    trigger = data.get("trigger")
    expected_trigger = "^^" + name
    trigger_issue = _check_trigger(
        trigger, expected=expected_trigger, file=file_basename
    )
    if trigger_issue is not None:
        return None, (trigger_issue,)
    assert isinstance(trigger, str)

    desc_issue = _check_description(
        data.get("description"), file=file_basename
    )
    if desc_issue is not None:
        return None, (desc_issue,)

    # Pre-tool-check — record whether `subagents` is present in the dict.
    # The whitelist validator consults this to decide whether `task` gets
    # a provisional pass or a swarm_subagents_missing reject.
    # `subagents_present` must distinguish "key absent" (default — no
    # policy) from "key present but value is null/string/list/etc."
    # (malformed — must emit swarm_subagents_not_object, not
    # swarm_subagents_missing). Using `"subagents" in data` keys on
    # presence, not on non-None values.
    subagents_present = "subagents" in data
    subagents_raw = data.get("subagents")

    tw_issue = _check_tool_whitelist(
        data.get("tool_whitelist"),
        tool_catalog=tool_catalog,
        file=file_basename,
        subagents_present=subagents_present,
    )
    if tw_issue is not None:
        return None, (tw_issue,)

    has_prompt = "prompt" in data
    has_prompt_file = "prompt_file" in data
    if has_prompt and has_prompt_file:
        return None, (
            SkillLoadIssue(
                file=file_basename,
                code="both_prompt_sources",
                message=(
                    "exactly one of 'prompt' or 'prompt_file' is required "
                    "(both defined — ambiguous precedence)"
                ),
            ),
        )
    if not has_prompt and not has_prompt_file:
        return None, (
            SkillLoadIssue(
                file=file_basename,
                code="missing_prompt",
                message=(
                    "exactly one of 'prompt' or 'prompt_file' is required"
                ),
            ),
        )

    if has_prompt:
        prompt_raw = data["prompt"]
        if not isinstance(prompt_raw, str):
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code="prompt_not_string",
                    message="prompt must be a string",
                ),
            )
        if not prompt_raw:
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code="prompt_empty",
                    message="prompt must be non-empty",
                ),
            )
        prompt_text = prompt_raw
    else:
        pf_rel = data["prompt_file"]
        if not isinstance(pf_rel, str) or not pf_rel:
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code="prompt_file_not_string",
                    message="prompt_file must be a non-empty string",
                ),
            )
        if Path(pf_rel).is_absolute():
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code="prompt_file_absolute",
                    message=(
                        f"prompt_file {pf_rel!r} must be a relative path "
                        f"inside the skills directory"
                    ),
                ),
            )
        try:
            pf_bytes = _safe_read_bytes(
                root, pf_rel, max_bytes=MAX_SKILL_PROMPT_BYTES
            )
        except SkillPathError as exc:
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code=f"prompt_{_classify_read_error(exc)}",
                    message=f"prompt_file {pf_rel!r}: {exc}",
                ),
            )
        except (ValueError, OSError) as exc:
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code="prompt_file_unreadable",
                    message=(
                        f"prompt_file {pf_rel!r} cannot be read: {exc}"
                    ),
                ),
            )
        try:
            prompt_text = _decode_strict_utf8(
                pf_bytes,
                file=pf_rel,
                kind="prompt",
            )
        except SkillLoadIssue as exc:
            return None, (exc,)
        if not prompt_text:
            return None, (
                SkillLoadIssue(
                    file=file_basename,
                    code="prompt_file_empty",
                    message=(
                        f"prompt_file {pf_rel!r} decoded to empty string"
                    ),
                ),
            )

    aliases_raw = data.get("aliases", [])
    alias_issue = _check_aliases(
        aliases_raw, primary_trigger=trigger, file=file_basename
    )
    if alias_issue is not None:
        return None, (alias_issue,)
    assert isinstance(aliases_raw, list)
    aliases = tuple(a for a in aliases_raw if isinstance(a, str))

    ttl = data.get("ttl_seconds", DEFAULT_TTL_SECONDS)
    ttl_issue = _check_ttl(ttl, file=file_basename)
    if ttl_issue is not None:
        return None, (ttl_issue,)
    assert isinstance(ttl, int) and not isinstance(ttl, bool)
    final_ttl = ttl

    assert isinstance(data["tool_whitelist"], list)
    allowed_tool_names = frozenset(
        t for t in data["tool_whitelist"] if isinstance(t, str)
    )

    description = data["description"]
    assert isinstance(description, str)

    # ---- Swarm policy parsing ----
    parsed_subagent_policy: Optional[SubagentPolicy] = None
    swarm_issues: list[SkillLoadIssue] = []

    if subagents_present:
        if not isinstance(subagents_raw, dict):
            swarm_issues.append(
                SkillLoadIssue(
                    file=file_basename,
                    code="swarm_subagents_not_object",
                    message="subagents must be an object",
                )
            )
        else:
            # Nested policy parsing — returns (None, issues) on failure.
            parsed_subagent_policy, nested_issues = _parse_subagent_policy(
                subagents_raw,
                tool_catalog=tool_catalog,
                file=file_basename,
            )
            swarm_issues.extend(nested_issues)

            # Contradiction: subagents present but parent lacks "task" —
            # regardless of whether the nested object was structurally
            # valid. The parent tool whitelist is the gate.
            if "task" not in allowed_tool_names:
                swarm_issues.append(
                    SkillLoadIssue(
                        file=file_basename,
                        code="swarm_parent_task_missing",
                        message=(
                            "subagents is present but parent tool_whitelist "
                            "does not include 'task'"
                        ),
                    )
                )

    # Outer flag + secret checks — both fire independently when a
    # structurally-valid policy exists. Skip when nested parsing failed
    # (those issues already rejected the file).
    if parsed_subagent_policy is not None:
        swarm_issues.extend(_check_swarm_outer(file=file_basename))

    if swarm_issues:
        return None, tuple(swarm_issues)

    return (
        ParsedCustomSkill(
            name=name,
            trigger=trigger,
            description=description,
            aliases=aliases,
            prompt_text=prompt_text,
            allowed_tool_names=allowed_tool_names,
            ttl_seconds=final_ttl,
            source_file=file_basename,
            subagents=parsed_subagent_policy,
        ),
        (),
    )


# ---------------------------------------------------------------------------
# Aggregate loader entry point
# ---------------------------------------------------------------------------


def load_custom_skills(
    root: Optional[Path],
    tool_catalog: Any,
    *,
    source: str = "env",
) -> LoadResult:
    """Discover + parse + validate every JSON skill under `root`.

    Behavior:
        - root is None → returns a `disabled` status with no work done.
        - source=="default" and root does not exist → returns a HEALTHY
          empty status (per spec §5.1: missing default <repo>/skills is
          a builtins-only healthy state, NOT a fatal error).
        - source=="default" and root exists but is not a dir → FATAL
          (operator's misconfiguration to surface — someone created a
          file at <repo>/skills).
        - source=="env" (or any non-"default") and root is missing or
          not a dir → FATAL SkillRegistryConfigurationError.
        - any file has issues → returns the FULL aggregate list.

    The `source` keyword identifies where the directory came from
    ("env", "default", or "disabled") and is recorded in the returned
    status snapshot.
    """
    if root is None:
        return LoadResult(
            parsed_skills=(),
            issues=(),
            status=CustomSkillsStatus(
                enabled=False,
                directory=None,
                source=source if source in {"env", "default"} else "disabled",
                state="disabled",
                discovered=0,
                loaded_names=(),
                issues=(),
                restart_required=False,
                swarm_enabled=False,
                swarm_capable_skills=(),
            ),
        )

    if not root.exists():
        if source == "default":
            # Spec §5.1: "Missing default <repo>/skills directory: healthy;
            # builtins only." The operator has not overridden the default,
            # so the absence of the default directory is NOT an error —
            # it just means no custom skills. Report as state="ok"
            # (no issues), loaded_names=(), enabled=True (the default
            # IS configured; the absence is on-disk, not in-config).
            return LoadResult(
                parsed_skills=(),
                issues=(),
                status=CustomSkillsStatus(
                    enabled=True,
                    directory=str(root),
                    source="default",
                    state="ok",
                    discovered=0,
                    loaded_names=(),
                    issues=(),
                    restart_required=False,
                    swarm_enabled=False,
                    swarm_capable_skills=(),
                ),
            )
        # source != "default" — the operator explicitly set
        # PROXY_SKILLS_DIR to a path that doesn't exist → fatal.
        raise SkillRegistryConfigurationError(
            issues=(),
            message=f"PROXY_SKILLS_DIR does not exist (resolved to {root})",
        )
    if not root.is_dir():
        # Path exists but is a file (not a directory). For any source
        # this is operator misconfiguration — surface it.
        raise SkillRegistryConfigurationError(
            issues=(),
            message=f"PROXY_SKILLS_DIR is not a directory (resolved to {root})",
        )

    try:
        candidate_files = discover_custom_skill_files(root)
    except SkillLoadIssue as exc:
        return LoadResult(
            parsed_skills=(),
            issues=(exc,),
            status=CustomSkillsStatus(
                enabled=True,
                directory=str(root),
                source=source,
                state="degraded",
                discovered=0,
                loaded_names=(),
                issues=(exc,),
                restart_required=True,
                swarm_enabled=False,
                swarm_capable_skills=(),
            ),
        )

    issues: list[SkillLoadIssue] = []
    parsed: list[ParsedCustomSkill] = []
    seen_names: dict[str, str] = {}
    seen_triggers: dict[str, str] = {}

    for path in candidate_files:
        skill, file_issues = parse_custom_skill_file(path, root, tool_catalog)
        if skill is None:
            issues.extend(file_issues)
            continue

        if skill.name in seen_names:
            issues.append(
                SkillLoadIssue(
                    file=skill.source_file,
                    code="duplicate_name",
                    message=(
                        f"name {skill.name!r} duplicates file "
                        f"{seen_names[skill.name]!r}"
                    ),
                )
            )
            continue

        if skill.trigger in seen_triggers:
            issues.append(
                SkillLoadIssue(
                    file=skill.source_file,
                    code="duplicate_trigger",
                    message=(
                        f"trigger {skill.trigger!r} duplicates "
                        f"file {seen_triggers[skill.trigger]!r}"
                    ),
                )
            )
            continue

        alias_clash = next(
            (a for a in skill.aliases if a in seen_triggers),
            None,
        )
        if alias_clash is not None:
            issues.append(
                SkillLoadIssue(
                    file=skill.source_file,
                    code="duplicate_alias_cross_file",
                    message=(
                        f"alias {alias_clash!r} already used by "
                        f"file {seen_triggers[alias_clash]!r}"
                    ),
                )
            )
            continue

        seen_names[skill.name] = skill.source_file
        seen_triggers[skill.trigger] = skill.source_file
        for a in skill.aliases:
            seen_triggers[a] = skill.source_file
        parsed.append(skill)

    if issues:
        return LoadResult(
            parsed_skills=(),
            issues=tuple(issues),
            status=CustomSkillsStatus(
                enabled=True,
                directory=str(root),
                source=source,
                state="degraded",
                discovered=len(candidate_files),
                loaded_names=(),
                issues=tuple(issues),
                restart_required=True,
                swarm_enabled=False,
                swarm_capable_skills=(),
            ),
        )

    # Compute swarm-capable summary on the success path. A skill is
    # swarm-capable iff its `subagents` policy is structurally valid
    # (parsed_subagent_policy is not None). We surface the CANONICAL
    # names, sorted alphabetically for stable ordering.
    swarm_capable = tuple(
        sorted(s.name for s in parsed if s.subagents is not None)
    )
    return LoadResult(
        parsed_skills=tuple(parsed),
        issues=(),
        status=CustomSkillsStatus(
            enabled=True,
            directory=str(root),
            source=source,
            state="ok",
            discovered=len(candidate_files),
            loaded_names=tuple(s.name for s in parsed),
            issues=(),
            restart_required=False,
            swarm_enabled=bool(swarm_capable),
            swarm_capable_skills=swarm_capable,
        ),
    )


# ---------------------------------------------------------------------------
# SkillDefinition builder
# ---------------------------------------------------------------------------


def _default_get_system_prompt(_ctx: Any, *, prompt_text: str) -> str:
    return prompt_text


def _default_get_subagent_prompt(_ctx: Any, *, prompt_text: str) -> str:
    return prompt_text


def _default_parse_subcommand(
    text: str,
    *,
    name: str,
    trigger: str,
    aliases: tuple[str, ...],
) -> Tuple[Optional[str], str]:
    """Direct-dispatch parse: accept text starting with the trigger or
    any alias. Returns ("__direct__", text) to route via the dispatcher.
    """
    candidates = [trigger] + list(aliases)
    if any(text.startswith(c) for c in candidates if c):
        return ("__direct__", text)
    return (None, text)


def _default_handle_direct(_ctx: Any, _mode: str, _action: str) -> Optional[dict]:
    return None


def _default_post_process(_ctx: Any, response: dict, *, name: str) -> dict:
    return response


def _default_should_deactivate(_ctx: Any, finish_reason: str, *, name: str) -> bool:
    """Canonical lifecycle for JSON-loaded custom skills (spec §7.4).

    Uses the finish_reason-only classification (the SkillDefinition
    should_deactivate signature only takes finish_reason, not the full
    response data). For CONTINUE detection with non-null tool_calls,
    a subclass may override.

    Mapping (matches `proxy.mode.classify_turn_response`):
    - "tool_calls" or "length" → False (CONTINUE, mode retained)
    - "content_filter" → True (FAILED, mode cleared)
    - "stop" or None → True (COMPLETE, mode cleared)
    - any other non-null reason → True (FAILED, mode cleared)
    """
    if finish_reason in ("tool_calls", "length"):
        return False  # CONTINUE
    if finish_reason == "content_filter":
        return True  # FAILED
    if finish_reason == "stop" or finish_reason is None:
        return True  # COMPLETE
    return True  # unknown non-null reason -> FAILED


def _default_get_canary(_ctx: Any) -> Optional[str]:
    return None


def _default_get_activation_rewrite(
    ctx: Any,
    *,
    prompt_text: str,
    trigger: str,
    aliases: tuple[str, ...],
) -> str:
    """Strip the trigger/alias prefix from the user message and emit
    `User request: <args>`. Empty if no args.
    """
    raw = getattr(ctx, "raw_user_text", "") or ""
    candidates = [trigger] + list(aliases)
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        if cand and raw.startswith(cand):
            arg = raw[len(cand):].strip()
            if arg:
                return f"User request: {arg}"
            return ""
    return ""


def build_custom_skill_definition(parsed: ParsedCustomSkill) -> Any:
    """Construct the `SkillDefinition` for one parsed custom skill.

    Installs the standard direct-dispatch callbacks. Custom JSON skills
    never spawn sub-agents (no canary), never implement handle_direct,
    and use the per-turn deactivation contract.
    """
    from proxy.skills.skill_definition import SkillDefinition  # local import

    name = parsed.name
    trigger = parsed.trigger
    aliases = parsed.aliases
    prompt_text = parsed.prompt_text
    allowed_tool_names = parsed.allowed_tool_names
    ttl_seconds = parsed.ttl_seconds
    description = parsed.description

    return SkillDefinition(
        name=name,
        trigger=trigger,
        aliases=aliases,
        description=description,
        allowed_tool_names=allowed_tool_names,
        ttl_seconds=ttl_seconds,
        subagents=parsed.subagents,
        get_system_prompt=lambda ctx, _p=prompt_text: _default_get_system_prompt(
            ctx, prompt_text=_p
        ),
        get_subagent_prompt=lambda ctx, _p=prompt_text: _default_get_subagent_prompt(
            ctx, prompt_text=_p
        ),
        parse_subcommand=lambda text, _name=name, _t=trigger, _a=aliases: _default_parse_subcommand(
            text, name=_name, trigger=_t, aliases=_a
        ),
        handle_direct=lambda ctx, mode, action: _default_handle_direct(ctx, mode, action),
        post_process=lambda ctx, response, _name=name: _default_post_process(
            ctx, response, name=_name
        ),
        should_deactivate=lambda ctx, fr, _name=name: _default_should_deactivate(
            ctx, fr, name=_name
        ),
        get_canary=lambda ctx: _default_get_canary(ctx),
        get_activation_rewrite=lambda ctx, _p=prompt_text, _t=trigger, _a=aliases: _default_get_activation_rewrite(
            ctx, prompt_text=_p, trigger=_t, aliases=_a
        ),
    )


__all__ = [
    "ALLOWED_TOP_LEVEL_KEYS",
    "CONDITIONALLY_ADMITTED_FOR_CUSTOM",
    "CustomSkillsStatus",
    "LoadResult",
    "MAX_ALIASES_PER_SKILL",
    "MAX_CUSTOM_SKILL_FILES",
    "MAX_CUSTOM_SKILL_TTL_SECONDS",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_SKILL_DEFINITION_BYTES",
    "MAX_SKILL_PROMPT_BYTES",
    "MAX_TOOLS_PER_WHITELIST",
    "MIN_CUSTOM_SKILL_TTL_SECONDS",
    "PERMANENTLY_DENIED_FOR_CUSTOM",
    "ParsedCustomSkill",
    "RESERVED_BUILTIN_NAMES",
    "RESERVED_COMMAND_TRIGGERS",
    "ResolvedSkillsDir",
    "SkillLoadIssue",
    "SkillRegistryConfigurationError",
    "build_custom_skill_definition",
    "discover_custom_skill_files",
    "load_custom_skills",
    "parse_custom_skill_file",
    "resolve_skills_dir",
]