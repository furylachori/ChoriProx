"""Wave 2 readiness tests — eager registry init + CLI exit on bad config.

Spec §11 Wave 2 covers:
  - Eager startup initialization
  - Readiness/health snapshot
  - Atomic-publication tests

These tests use subprocess to invoke qwen_proxy.py:__main__ and verify
the eager-init failure path. In-process tests for the lazy path live in
tests/proxy/test_skill_registry.py and tests/unit/test_custom_skill_startup_contract.py.

The CLI subprocess hangs on memory-sync workers unless PROXY_SOTA_API_KEY
is set to a stub value (per src/devtools/AGENTS.md "non-daemon memory-sync workers").

NOTE on hangs: the CLI registers atexit handlers (_graceful_shutdown,
_shutdown_logging) that block on SQLite/thread-pool cleanup when the test
session holds data/memory.db. Even with PROXY_TEST_DB pointing to a fresh
path, the atexit handlers log "Graceful shutdown initiated" and then
block on step 1 (in-flight drain) or step 2 (thread pool). We catch
TimeoutExpired and inspect the captured stdout/stderr — the eager-init
diagnostic is printed BEFORE atexit fires, so a TimeoutExpired that
preceded the diagnostic means eager init succeeded (proxy is still
running). A TimeoutExpired that FOLLOWS the diagnostic means eager init
failed (sys.exit(1) was called, atexit hung). The diagnostic-string
presence is the load-bearing assertion; the exit code is best-effort.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SubprocessOutcome:
    """Result of a CLI invocation, tolerant of atexit hangs."""
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool


def _run_qwen_proxy_cli(env_overrides: dict, timeout: int = 5) -> SubprocessOutcome:
    """Run `python3 -m qwen_proxy` in a subprocess, tolerate atexit hangs.

    The eager-init diagnostic (`[STARTUP] Custom skills configuration invalid`)
    is emitted via `print(..., flush=True)` on stdout BEFORE `sys.exit(1)`.
    The atexit handlers may hang in this test environment; we SIGKILL after
    `timeout` seconds and capture whatever was printed.
    """
    env = os.environ.copy()
    env["PROXY_SOTA_API_KEY"] = "sk-stub-test-key"
    # Isolate the subprocess DB so its WAL doesn't conflict with the parent
    # test session's data/memory.db.
    env["PROXY_TEST_DB"] = env.get("PROXY_TEST_DB", "/tmp/choriprox_subproc_test.db")
    env.update(env_overrides)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "qwen_proxy"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SubprocessOutcome(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        # At least one of stdout/stderr was captured. communicate() may
        # also have been called implicitly by the timeout.
        return SubprocessOutcome(
            returncode=None,
            stdout=(exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr=(exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            timed_out=True,
        )


def test_cli_fails_fast_on_bad_proxy_skills_dir(tmp_path):
    """`python3 -m qwen_proxy` prints the eager-init diagnostic when
    PROXY_SKILLS_DIR points to a non-existent path.

    Wave 2 eager init (spec §5.1): a configured skills directory that
    does not exist on disk is a fatal SkillRegistryConfigurationError.
    The CLI logs the structured per-file diagnostic list and exits 1
    (via sys.exit(1) inside the except block). The atexit handlers may
    hang in the subprocess; the load-bearing assertion is that the
    diagnostic string appears in the captured output BEFORE the
    subprocess hangs or exits.
    """
    outcome = _run_qwen_proxy_cli(
        {"PROXY_SKILLS_DIR": "/nonexistent/path/that/does/not/exist"},
        timeout=8,
    )
    combined = (outcome.stdout + outcome.stderr).lower()
    assert "custom skills configuration invalid" in combined, (
        f"Expected 'custom skills configuration invalid' diagnostic in "
        f"captured output.\nstdout:\n{outcome.stdout}\nstderr:\n{outcome.stderr}"
    )
    # Best-effort exit-code check: if the subprocess exited cleanly (not
    # killed by timeout), the exit code MUST be non-zero. If it was killed
    # by timeout (atexit hang), we accept that the diagnostic proved the
    # failure mode but we couldn't observe the exit code.
    if not outcome.timed_out:
        assert outcome.returncode != 0, (
            f"Expected non-zero exit; got {outcome.returncode}.\n"
            f"stdout:\n{outcome.stdout}\nstderr:\n{outcome.stderr}"
        )


def test_cli_succeeds_with_valid_empty_dir(tmp_path):
    """`python3 -m qwen_proxy` proceeds past eager init when PROXY_SKILLS_DIR
    points to a valid empty directory on disk.

    Spec §5.1: env-set + exists + is dir → load (empty set is healthy).
    We use tmp_path so we know the dir is empty and exists. The proxy
    will eventually call app.run() which blocks until the timeout fires;
    the load-bearing assertion is that the eager-init diagnostic is NOT
    printed.
    """
    empty_dir = tmp_path / "empty_skills"
    empty_dir.mkdir()
    outcome = _run_qwen_proxy_cli(
        {"PROXY_SKILLS_DIR": str(empty_dir)},
        timeout=5,
    )
    combined = (outcome.stdout + outcome.stderr).lower()
    assert "custom skills configuration invalid" not in combined, (
        f"Eager init failed unexpectedly. stdout:\n{outcome.stdout}\n"
        f"stderr:\n{outcome.stderr}"
    )


def test_cli_succeeds_with_explicit_empty_string(tmp_path):
    """`python3 -m qwen_proxy` proceeds past eager init when PROXY_SKILLS_DIR=""
    (per spec §5.2 'Empty string disables custom-skill loading explicitly')."""
    outcome = _run_qwen_proxy_cli({"PROXY_SKILLS_DIR": ""}, timeout=5)
    combined = (outcome.stdout + outcome.stderr).lower()
    assert "custom skills configuration invalid" not in combined, (
        f"Eager init failed with PROXY_SKILLS_DIR=''. stdout:\n{outcome.stdout}\n"
        f"stderr:\n{outcome.stderr}"
    )


def test_import_does_not_trigger_eager_init(tmp_path):
    """Importing qwen_proxy with PROXY_SKILLS_DIR=/nonexistent does NOT
    construct the singleton at import time.

    Pinning contract: `_eager_init_registry()` is only called from
    `qwen_proxy.py:__main__`, NEVER at module import. So `import qwen_proxy`
    must succeed even with a bad PROXY_SKILLS_DIR. The eager init fires
    only when running `python -m qwen_proxy`.

    This is the contract Wave 2 tests rely on — tests that import qwen_proxy
    BEFORE setting PROXY_SKILLS_DIR must not trigger eager init.
    """
    import importlib
    import sys as _sys

    env_save = os.environ.copy()
    try:
        os.environ["PROXY_SKILLS_DIR"] = "/nonexistent/path/that/does/not/exist"
        # Force a fresh import to bypass any cached singleton.
        for mod_name in list(_sys.modules.keys()):
            if mod_name.startswith("proxy.") or mod_name in ("qwen_proxy",):
                del _sys.modules[mod_name]

        # qwen_proxy at import time does NOT call _eager_init_registry.
        # So `import qwen_proxy` should succeed even with bad config.
        # The eager init only fires from `if __name__ == "__main__":`.
        # This test pins that contract.
        import qwen_proxy  # should NOT raise
        # Eager init must NOT trigger at module-import time — it only fires from __main__.
        import proxy.skills.skill_registry as sr_module
        assert sr_module._default_registry is None, (
            "Eager init was triggered at qwen_proxy import time; "
            "should only fire from __main__."
        )
    finally:
        os.environ.clear()
        os.environ.update(env_save)