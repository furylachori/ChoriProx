"""Pin `proxy.state.MAX_CALLS_PER_ITERATION == 50` (default) and assert the
three use sites agree on the value.

Use sites (off-limits to this test — owned by other sub-agents):
  - src/memory/tool_loop.py:155            — `max_calls_per_iteration: int = 50`
  - src/proxy/pipeline/tool_interception.py — `max_calls_per_iteration: int = 50`
  - src/proxy/tool_pipeline/tool_pipeline.py:44  — `max_calls_per_iteration: int = 50`

If any use site drifts away from `state.MAX_CALLS_PER_ITERATION`, this test
fails with a clear diff so the divergence surfaces during review instead of
silently at runtime.

The env-override tests run a subprocess with `PROXY_MAX_CALLS_PER_ITERATION`
set, then import `proxy.state` inside that subprocess and print the value.
This avoids `importlib.reload(proxy.state)` — which would rebuild the
module's `_HTTP_SESSION` and other singletons that other tests in the
same worker depend on.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_USE_SITES = [
    "src/memory/tool_loop.py",
    "src/proxy/pipeline/tool_interception.py",
    "src/proxy/tool_pipeline/tool_pipeline.py",
]


def _read_use_site_default(repo_root: str, relpath: str) -> int | str | None:
    """Extract `max_calls_per_iteration: int = N` from the file.

    Accepts either an integer literal (``= 50``) or a reference to the
    state constant (``= _state.MAX_CALLS_PER_ITERATION``).

    Returns None if the parameter is no longer present. Returns the string
    ``"STATE"`` if the default is wired to the central constant, which
    always tracks ``proxy.state.MAX_CALLS_PER_ITERATION``.
    """
    path = os.path.join(repo_root, relpath)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    match = re.search(
        r"max_calls_per_iteration\s*:\s*int\s*=\s*([^\s,]+)",
        src,
    )
    if match is None:
        return None
    raw = match.group(1).rstrip(",)")
    if raw.isdigit():
        return int(raw)
    if "MAX_CALLS_PER_ITERATION" in raw:
        return "STATE"
    return None


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _probe_state_value(env_value: str | None) -> int:
    """Run a subprocess that imports proxy.state and prints MAX_CALLS_PER_ITERATION.

    Returns the int value. Any non-numeric output or non-zero exit is a test
    failure surfaced via the return value (caller asserts).
    """
    env = os.environ.copy()
    if env_value is None:
        env.pop("PROXY_MAX_CALLS_PER_ITERATION", None)
    else:
        env["PROXY_MAX_CALLS_PER_ITERATION"] = env_value
    env["PYTHONPATH"] = os.path.join(_REPO_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, proxy.state as s; "
         "sys.stdout.write(str(s.MAX_CALLS_PER_ITERATION))"],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Subprocess failed (rc={result.returncode}): {result.stderr}"
    )
    return int(result.stdout.strip())


# ---------------------------------------------------------------------------
# Default value pinning
# ---------------------------------------------------------------------------


class TestMaxCallsPerIterationDefault:
    def test_default_is_50(self):
        value = _probe_state_value(env_value=None)
        assert value == 50, (
            f"proxy.state.MAX_CALLS_PER_ITERATION must default to 50; "
            f"got {value}"
        )

    def test_default_is_positive_and_within_range(self):
        value = _probe_state_value(env_value=None)
        assert 1 <= value <= 200, (
            f"MAX_CALLS_PER_ITERATION={value} must be within [1, 200]"
        )


# ---------------------------------------------------------------------------
# Env var override (subprocess-isolated to avoid module-reload side effects)
# ---------------------------------------------------------------------------


class TestMaxCallsPerIterationEnvOverride:
    def test_env_var_overrides_default(self):
        value = _probe_state_value(env_value="10")
        assert value == 10, (
            f"PROXY_MAX_CALLS_PER_ITERATION=10 must yield state value 10; "
            f"got {value}"
        )

    def test_env_var_clamped_to_max(self):
        value = _probe_state_value(env_value="999")
        assert value == 200, (
            f"Out-of-range env var (999) must clamp to 200; got {value}"
        )

    def test_env_var_clamped_to_min(self):
        value = _probe_state_value(env_value="0")
        assert value == 1, (
            f"Out-of-range env var (0) must clamp to 1; got {value}"
        )


# ---------------------------------------------------------------------------
# Use-site parity
# ---------------------------------------------------------------------------


class TestUseSitesAgreeWithStateConstant:
    def test_use_sites_match(self):
        """Each call site that exposes ``max_calls_per_iteration`` must use
        the same default as ``state.MAX_CALLS_PER_ITERATION`` (or wire to it
        via ``_state.MAX_CALLS_PER_ITERATION``).

        We read the file directly rather than importing it: the use sites
        are owned by other agents and may not be importable in this test's
        process (Neo4j / network init at module load). A static read keeps
        this test self-contained.
        """
        default = _probe_state_value(env_value=None)
        for relpath in _USE_SITES:
            site_default = _read_use_site_default(_REPO_ROOT, relpath)
            assert site_default is not None, (
                f"{relpath} no longer exposes `max_calls_per_iteration: int = N` "
                f"— either the parameter was renamed or the constant was "
                f"inlined. Update this test to match."
            )
            if site_default == "STATE":
                # Wired to the central constant — guaranteed to track.
                continue
            assert site_default == default, (
                f"{relpath} defaults max_calls_per_iteration={site_default} "
                f"but state.MAX_CALLS_PER_ITERATION={default}. Drift between "
                f"the central constant and a call-site default silently "
                f"changes loop behavior depending on which path is hit — "
                f"pin them together, either by switching the default to "
                f"`_state.MAX_CALLS_PER_ITERATION` or by updating both literals."
            )
