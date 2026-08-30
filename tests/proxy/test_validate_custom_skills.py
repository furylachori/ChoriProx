"""End-to-end tests for the standalone validator CLI (spec §8.4)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_validator(args, env_overrides=None, timeout=30):
    """Invoke ``python -m proxy.skills.validate_custom_skills`` as a subprocess."""
    cmd = [sys.executable, "-m", "proxy.skills.validate_custom_skills", *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_skill(skills_dir: Path, name: str, **overrides) -> None:
    payload = {
        "name": name,
        "trigger": f"^^{name}",
        "description": f"Description for {name}",
        "prompt": f"Prompt for {name}.",
        "tool_whitelist": ["read"],
    }
    payload.update(overrides)
    (skills_dir / f"{name}.json").write_text(json.dumps(payload))


def test_validator_valid_dir_returns_0(tmp_path):
    """A directory with one valid JSON skill returns exit 0."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(
        skills_dir,
        "review",
        description="Code review",
        prompt="Review the code.",
        tool_whitelist=["read", "grep"],
    )
    result = _run_validator(["--dir", str(skills_dir), "--format", "json"])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert output["state"] == "ok"
    assert output["discovered"] == 1
    assert output["loaded"] == 1
    assert output["valid"] == [
        {"file": "review.json", "name": "review", "trigger": "^^review"}
    ]
    assert output["loaded_names"] == ["review"]


def test_validator_invalid_skill_returns_1(tmp_path):
    """A directory with an invalid JSON skill returns exit 1."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "broken.json").write_text(json.dumps({
        "name": "broken",
        "trigger": "^^broken",
        "prompt": "test",
        "tool_whitelist": ["read"],
    }))
    result = _run_validator(["--dir", str(skills_dir), "--format", "json"])
    assert result.returncode == 1, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "invalid"
    assert output["state"] == "degraded"
    assert output["loaded"] == 0
    assert len(output["invalid"]) >= 1
    assert output["invalid"] == output["issues"]


def test_validator_dir_missing_returns_2(tmp_path):
    """An explicit --dir pointing at a nonexistent path returns exit 2."""
    missing = tmp_path / "nonexistent"
    result = _run_validator(["--dir", str(missing), "--format", "json"])
    assert result.returncode == 2, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "fatal"
    assert output["state"] == "error"
    assert output["source"] == "error"


def test_validator_env_missing_returns_2(tmp_path):
    """A missing PROXY_SKILLS_DIR with no --dir returns exit 2."""
    missing = tmp_path / "nonexistent"
    result = _run_validator([], env_overrides={"PROXY_SKILLS_DIR": str(missing)})
    assert result.returncode == 2, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "fatal"
    assert output["source"] == "env"
    assert output["state"] == "error"


def test_validator_default_missing_returns_0():
    """PROXY_SKILLS_DIR unset + missing default returns exit 0."""
    result = _run_validator([], env_overrides={"PROXY_SKILLS_DIR": None})
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert output["source"] == "default"
    assert output["state"] == "ok"
    assert output["discovered"] == 0
    assert output["loaded_names"] == []


def test_validator_empty_env_disables_returns_0():
    """PROXY_SKILLS_DIR empty disables custom loading with exit 0."""
    result = _run_validator([], env_overrides={"PROXY_SKILLS_DIR": ""})
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert output["source"] == "disabled"
    assert output["state"] == "disabled"
    assert output["directory"] is None
    assert output["discovered"] == 0
    assert output["loaded_names"] == []


def test_validator_dir_argument_overrides_env(tmp_path):
    """--dir overrides a bad PROXY_SKILLS_DIR env var."""
    nonexistent = tmp_path / "nonexistent"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "ok")
    result = _run_validator(
        ["--dir", str(skills_dir), "--format", "json"],
        env_overrides={"PROXY_SKILLS_DIR": str(nonexistent)},
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert output["loaded_names"] == ["ok"]


def test_validator_real_examples_dir_returns_0():
    """The shipped examples/skills/ parses cleanly with exit 0."""
    examples_dir = _REPO_ROOT / "examples" / "skills"
    if not examples_dir.exists():
        pytest.skip("examples/skills not present")
    result = _run_validator(["--dir", str(examples_dir), "--format", "json"])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    assert output["status"] == "ok"
    assert output["discovered"] >= 1
    assert {"build", "plan", "test"}.issubset(set(output["loaded_names"]))


def test_validator_uses_runtime_loader():
    """Validator must call load_custom_skills (shared parser invariant per spec §8.4)."""
    import inspect

    from proxy.skills import validate_custom_skills

    src = inspect.getsource(validate_custom_skills)
    assert "load_custom_skills" in src
    assert "from proxy.skills.custom_skill_loader import" in src


def test_validator_unknown_format_returns_error():
    """--format other than json returns an argparse error (exit 2)."""
    result = _run_validator(["--format", "yaml"], env_overrides={"PROXY_SKILLS_DIR": ""})
    assert result.returncode == 2
    assert result.stdout == ""


def test_validator_unreadable_dir_returns_2(tmp_path):
    """An unreadable directory returns exit 2 with JSON output (no traceback)."""
    if sys.platform == "win32":
        pytest.skip("POSIX permission test")
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "ok")
    skills_dir.chmod(0o000)
    try:
        result = _run_validator(["--dir", str(skills_dir), "--format", "json"])
        if result.returncode == 0:
            pytest.skip("Running as root; permission test not effective")
        assert result.returncode == 2, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["status"] == "fatal"
        assert "Traceback" not in result.stderr
    finally:
        skills_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# Swarm validator tests
# ---------------------------------------------------------------------------


def _write_swarm_skill(
    skills_dir: Path,
    name: str,
    *,
    with_subagents: bool = True,
    child_whitelist: list = None,
    max_calls: int = 4,
    parent_whitelist: list = None,
) -> None:
    """Helper: write a swarm-capable custom skill JSON file.

    Defaults produce a structurally-valid swarm config. Set
    `with_subagents=False` to omit the nested `subagents` key.
    """
    if parent_whitelist is None:
        parent_whitelist = ["read", "task"]
    if child_whitelist is None:
        child_whitelist = ["read"]
    payload = {
        "name": name,
        "trigger": f"^^{name}",
        "description": f"Swarm skill {name}",
        "prompt": f"Do {name} work.",
        "tool_whitelist": parent_whitelist,
    }
    if with_subagents:
        payload["subagents"] = {
            "tool_whitelist": child_whitelist,
            "max_task_calls_per_request": max_calls,
        }
    (skills_dir / f"{name}.json").write_text(json.dumps(payload))


def test_validator_swarm_no_flag_returns_1_swarm_flag_disabled(tmp_path):
    """Valid swarm + no flag → exit 1 with swarm_flag_disabled in output.

    Uses a structurally-valid subagents config so the loader reaches
    the post-validation swarm gate. Requires Agent 2's SubagentPolicy
    to actually be constructible — currently expected to fail until
    Agent 2 lands.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_swarm_skill(skills_dir, "research")
    result = _run_validator(
        ["--dir", str(skills_dir), "--format", "json"],
        env_overrides={
            "PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS": None,
            "PROXY_SESSION_SECRET": "x" * 32,
        },
    )
    # Until Agent 2's SubagentPolicy lands, this will raise TypeError
    # at SubagentPolicy construction time and the validator subprocess
    # crashes with a non-zero exit. Pin the expected post-Agent-2 shape.
    assert result.returncode == 1, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    codes = [i["code"] for i in output["issues"]]
    assert "swarm_flag_disabled" in codes


def test_validator_swarm_no_secret_returns_1_swarm_missing_secret(tmp_path):
    """Valid swarm + flag but no secret → exit 1 with swarm_missing_secret.

    Same Agent 2 dependency as the flag-disabled case.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_swarm_skill(skills_dir, "research")
    result = _run_validator(
        ["--dir", str(skills_dir), "--format", "json"],
        env_overrides={
            "PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS": "1",
            "PROXY_SESSION_SECRET": None,
        },
    )
    assert result.returncode == 1, f"stderr: {result.stderr}"
    output = json.loads(result.stdout)
    codes = [i["code"] for i in output["issues"]]
    assert "swarm_missing_secret" in codes


def test_validator_prompt_bom_returns_1_with_aggregated_issue(tmp_path):
    """Validator CLI handles prompt_file BOM via aggregated issue tuple."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "bom.md").write_bytes(b"\xef\xbb\xbf" + b"content")
    (skills_dir / "ok.json").write_text(json.dumps({
        "name": "ok", "trigger": "^^ok",
        "description": "d", "prompt_file": "bom.md",
        "tool_whitelist": ["read"],
    }))
    result = _run_validator(["--dir", str(skills_dir), "--format", "json"])
    assert result.returncode == 1
    output = json.loads(result.stdout)
    codes = [i["code"] for i in output["issues"]]
    assert "prompt_bom" in codes
