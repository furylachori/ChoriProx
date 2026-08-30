"""Standalone validator for the custom-skills configuration.

Per spec §8.4, this CLI shares ``custom_skill_loader.load_custom_skills``
with runtime startup so schema parsing and validation logic is not duplicated.
Operators can invoke it in CI to gate config changes, or in production to
inspect custom-skill misconfiguration without starting Flask.

Usage:
    python -m proxy.skills.validate_custom_skills --dir /etc/choriprox/skills --format json
    python -m proxy.skills.validate_custom_skills

Exit codes:
    0 — valid configuration, disabled loading, or missing default directory
    1 — one or more skill validation failures
    2 — invocation or directory-access failure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from proxy.skills.custom_skill_loader import (
    SkillRegistryConfigurationError,
    load_custom_skills,
    resolve_skills_dir,
)
from proxy.skills.path_utils import _REPO_ROOT
from proxy.tool_pipeline.tool_registry import registry as tool_catalog


def _issue_to_dict(issue: Any) -> dict[str, str]:
    return {
        "file": str(getattr(issue, "file", "")),
        "code": str(getattr(issue, "code", "")),
        "message": str(getattr(issue, "message", "")),
    }


def _write_json(output: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(output, indent=2))
    sys.stdout.write("\n")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _resolve_directory(args_dir: str | None) -> tuple[Path | None, str]:
    """Resolve the target directory per spec §5.2."""
    if args_dir is not None:
        if args_dir == "":
            raise SkillRegistryConfigurationError(
                issues=(),
                message="Specified skills directory is empty",
            )
        path = _resolve_path(args_dir)
        if not path.exists() or not path.is_dir():
            raise SkillRegistryConfigurationError(
                issues=(),
                message=(
                    "Specified skills directory does not exist or is not a "
                    f"directory: {path}"
                ),
            )
        return path, "env"

    resolved = resolve_skills_dir()
    return resolved.root, resolved.source


def _directory_for_error(args_dir: str | None) -> str | None:
    if args_dir is not None:
        return str(_resolve_path(args_dir)) if args_dir else None
    if "PROXY_SKILLS_DIR" not in os.environ:
        return None
    env_value = os.environ["PROXY_SKILLS_DIR"]
    if env_value == "":
        return None
    return str(_resolve_path(env_value))


def _source_for_error(args_dir: str | None) -> str:
    if args_dir is not None:
        return "error"
    if "PROXY_SKILLS_DIR" in os.environ:
        return "env"
    return "default"


def _fatal_output(
    *,
    directory: str | None,
    source: str,
    exc: SkillRegistryConfigurationError,
) -> dict[str, Any]:
    issues = [_issue_to_dict(issue) for issue in exc.issues]
    return {
        "status": "fatal",
        "directory": directory,
        "source": source if source in {"env", "default", "disabled"} else "error",
        "state": "error",
        "discovered": 0,
        "loaded": 0,
        "loaded_names": [],
        "valid": [],
        "invalid": issues,
        "issues": issues,
        "error": str(exc),
    }


def _success_output(
    *,
    directory: Path | None,
    source: str,
    parsed_skills: Any,
    issues: Any,
    status: Any,
) -> tuple[dict[str, Any], int]:
    valid = [
        {"file": skill.source_file, "name": skill.name, "trigger": skill.trigger}
        for skill in parsed_skills
    ]
    invalid = [_issue_to_dict(issue) for issue in issues]
    if invalid:
        output_status = "invalid"
        exit_code = 1
    else:
        output_status = "ok"
        exit_code = 0
    output = {
        "status": output_status,
        "directory": str(directory) if directory is not None else status.directory,
        "source": source,
        "state": status.state,
        "discovered": status.discovered,
        "loaded": len(status.loaded_names),
        "loaded_names": list(status.loaded_names),
        "valid": valid,
        "invalid": invalid,
        "issues": invalid,
    }
    return output, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="proxy.skills.validate_custom_skills",
        description="Validate the custom-skills configuration (spec §8.4).",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Skills directory (default: PROXY_SKILLS_DIR or <repo>/skills)",
    )
    parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format (only JSON supported per spec)",
    )
    args = parser.parse_args(argv)

    try:
        directory, source = _resolve_directory(args.dir)
    except SkillRegistryConfigurationError as exc:
        _write_json(
            _fatal_output(
                directory=_directory_for_error(args.dir),
                source=_source_for_error(args.dir),
                exc=exc,
            )
        )
        return 2

    try:
        parsed_skills, issues, status = load_custom_skills(
            directory,
            tool_catalog=tool_catalog,
            source=source,
        )
    except SkillRegistryConfigurationError as exc:
        _write_json(
            _fatal_output(
                directory=str(directory) if directory is not None else None,
                source=source,
                exc=exc,
            )
        )
        return 2

    output, exit_code = _success_output(
        directory=directory,
        source=status.source,
        parsed_skills=parsed_skills,
        issues=issues,
        status=status,
    )
    _write_json(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
