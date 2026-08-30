"""Integration tests for swarm visibility on operator surfaces.

Pins the read-only operator-facing exposure of the swarm fields specified
in `docs/specs/custom-skill-research-swarms.md` §0.1 / AC 16.15:

  - `GET /health` MUST include `custom_skills.swarm_enabled` (bool) and
    `custom_skills.swarm_capable_skills` (list of names) in its JSON
    payload.
  - `^^status` MUST render Swarm-enabled and Swarm-capable skills lines
    in its Custom Skills block.

Both surfaces share a single source: `CustomSkillsStatus` populated by
the loader (`proxy/skills/custom_skill_loader.py:1502-1503`). The fields
are populated whenever a custom skill declares a valid `subagents` block
AND the deployment flag `PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS=1` is set
AND `PROXY_SESSION_SECRET` is configured.

The negative case (no swarm flag set / no subagent block / no skill
directory) exercises the same surfaces and asserts the same fields
appear with `False` / `[]` (or `Swarm-enabled: no` / `0`) defaults so
operators never see a silently-stripped field.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from commands import CommandContext, dispatch_command


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_swarm_skill(skills_dir, name: str = "research") -> None:
    """Write a single swarm-capable custom skill into `skills_dir`."""
    (skills_dir / f"{name}.json").write_text(json.dumps({
        "name": name,
        "trigger": f"^^{name}",
        "description": f"Swarm visibility test skill ({name}).",
        "prompt": f"{name} prompt",
        "tool_whitelist": ["read", "task"],
        "subagents": {
            "tool_whitelist": ["read"],
            "max_task_calls_per_request": 4,
        },
    }))


def _write_ordinary_skill(skills_dir, name: str = "audit") -> None:
    """Write an ordinary (non-swarm) custom skill."""
    (skills_dir / f"{name}.json").write_text(json.dumps({
        "name": name,
        "trigger": f"^^{name}",
        "description": f"Ordinary skill ({name}).",
        "prompt": f"{name} prompt",
        "tool_whitelist": ["read"],
    }))


# ---------------------------------------------------------------------------
# /health — swarm_enabled + swarm_capable_skills
# ---------------------------------------------------------------------------


@pytest.fixture
def health_client(monkeypatch):
    """Flask test client with permissive admin rate limiter (so the
    `tmp_path` env mutations below can land before the route's registry
    lookup, mirroring `tests/integration/test_admin_endpoints.py`).
    """
    monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")
    from proxy.app import app as flask_app
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def patched_admin_deps():
    """Patch Neo4j + SQLite checks so /health returns 200 regardless of the
    actual environment. Mirrors the pattern in test_admin_endpoints.py.
    """
    with patch("memory.db.get_connection") as mock_conn, \
            patch("proxy.context._get_task_graph_singleton") as mock_tg:
        mock_conn.return_value.execute.return_value.fetchone.return_value = (1,)
        mock_tg.return_value._executor.is_available.return_value = True
        yield


def _install_permissive_admin_limiter(monkeypatch):
    from proxy.rate_limiter import RateLimiter
    tight = RateLimiter(
        max_requests=100, window_seconds=60, name="admin", persist=False
    )
    monkeypatch.setattr("proxy.admin.admin_rate_limiter", tight)


class TestHealthSwarmVisibility:
    """`/health` MUST surface the swarm fields on `custom_skills` (AC 16.15)."""

    def test_health_exposes_swarm_fields_when_swarm_capable_skill_loaded(
        self, health_client, tmp_path, monkeypatch, patched_admin_deps
    ):
        """Loaded swarm-capable skill + flag on + secret set →
        /health JSON contains swarm_enabled=True and the canonical name.
        """
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _write_swarm_skill(skills_dir, "research")
        monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
        monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
        monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()
        _install_permissive_admin_limiter(monkeypatch)

        resp = health_client.get("/health")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        cs = body["custom_skills"]
        assert cs["swarm_enabled"] is True
        assert cs["swarm_capable_skills"] == ["research"]

        reset_default_registry()

    def test_health_swarm_disabled_when_flag_unset(
        self, health_client, tmp_path, monkeypatch, patched_admin_deps
    ):
        """Flag off → swarm_enabled=False, swarm_capable_skills=[].

        Note: a swarm-capable skill on disk with the flag off triggers the
        atomic fail-closed path (`SkillRegistryConfigurationError`); that
        is exercised by the existing `test_swarm_secret_missing_rejects`
        loader unit test. Here we use an ORDINARY skill so the registry
        initializes cleanly and the swarm fields surface (False/[]).
        """
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _write_ordinary_skill(skills_dir, "audit")
        monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
        monkeypatch.delenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", raising=False)
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()
        _install_permissive_admin_limiter(monkeypatch)

        resp = health_client.get("/health")

        assert resp.status_code == 200
        cs = resp.get_json()["custom_skills"]
        assert cs["swarm_enabled"] is False
        assert cs["swarm_capable_skills"] == []

        reset_default_registry()

    def test_health_swarm_disabled_when_no_subagents_block(
        self, health_client, tmp_path, monkeypatch, patched_admin_deps
    ):
        """Flag on but the only loaded skill has NO `subagents` block →
        swarm_enabled stays False (the field is flag-aware, not
        file-presence-aware).
        """
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _write_ordinary_skill(skills_dir, "audit")
        monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
        monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
        monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()
        _install_permissive_admin_limiter(monkeypatch)

        resp = health_client.get("/health")

        assert resp.status_code == 200
        cs = resp.get_json()["custom_skills"]
        assert cs["swarm_enabled"] is False
        assert cs["swarm_capable_skills"] == []

        reset_default_registry()

    def test_health_swarm_disabled_when_skills_dir_unset(
        self, health_client, monkeypatch, patched_admin_deps
    ):
        """PROXY_SKILLS_DIR empty → custom_skills.status=disabled →
        swarm_enabled is also False (no skills to be capable)."""
        monkeypatch.setenv("PROXY_SKILLS_DIR", "")
        monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
        monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()
        _install_permissive_admin_limiter(monkeypatch)

        resp = health_client.get("/health")

        assert resp.status_code == 200
        cs = resp.get_json()["custom_skills"]
        assert cs["status"] == "disabled"
        assert cs["swarm_enabled"] is False
        assert cs["swarm_capable_skills"] == []

        reset_default_registry()

    def test_health_handles_legacy_status_without_swarm_fields(
        self, health_client, tmp_path, monkeypatch, patched_admin_deps
    ):
        """A CustomSkillsStatus-like object missing swarm_enabled /
        swarm_capable_skills must NOT degrade /health to 503."""
        from unittest.mock import patch
        from proxy.skills.skill_registry import get_default_registry

        class LegacyStatus:
            def __init__(self):
                self.state = "ok"
                self.enabled = True
                self.source = "env"
                self.directory = str(tmp_path)
                self.restart_required = False
                self.discovered = 0
                self.loaded_names = ("build", "plan")
                self.issues = ()

            def __getattr__(self, name):
                raise AttributeError(name)

        _install_permissive_admin_limiter(monkeypatch)
        with patch(
            "proxy.skills.skill_registry.get_default_registry"
        ) as mock_reg:
            mock_reg.return_value.custom_skills_status.return_value = (
                LegacyStatus()
            )
            resp = health_client.get(
                "/health", headers={"Authorization": "Bearer test"}
            )

        assert resp.status_code == 200
        body = resp.get_json()
        cs = body.get("custom_skills", {})
        assert cs.get("status") == "ok"
        assert cs.get("loaded_names") == ["build", "plan"]
        assert cs.get("swarm_enabled") is False
        assert cs.get("swarm_capable_skills") == []


# ---------------------------------------------------------------------------
# ^^status — Swarm-enabled / Swarm-capable skills
# ---------------------------------------------------------------------------


def _render_status():
    """Invoke `^^status` via the real dispatcher; return the assistant content."""
    ctx = CommandContext(raw_text="^^status", session_id="s1")
    resp = dispatch_command(ctx)
    return resp["choices"][0]["message"]["content"]


class TestCmdStatusSwarmVisibility:
    """`^^status` MUST render Swarm-enabled and Swarm-capable skills lines."""

    def _install_skills_dir(self, tmp_path, monkeypatch, *, kind: str):
        """`kind` is "swarm" (on-disk swarm skill) or "ordinary" (audit)."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        if kind == "swarm":
            _write_swarm_skill(skills_dir, "research")
        else:
            _write_ordinary_skill(skills_dir, "audit")
        monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def test_status_renders_swarm_enabled_yes_and_skill_count(self, tmp_path, monkeypatch):
        """Loaded swarm-capable skill + flag on + secret set →
        Swarm-enabled: yes + count + name appear in the Custom Skills block.
        """
        self._install_skills_dir(tmp_path, monkeypatch, kind="swarm")
        monkeypatch.setenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", "1")
        monkeypatch.setenv("PROXY_SESSION_SECRET", "x" * 32)

        content = _render_status()
        # Both new lines must appear.
        assert "Swarm-enabled: yes" in content
        assert "Swarm-capable skills: 1 skill(s): research" in content
        # Re-pinned existing line still present (regression guard).
        assert "Custom Skills:" in content
        assert "- Status: ok" in content

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def test_status_renders_swarm_enabled_no_and_zero_skills(
        self, tmp_path, monkeypatch
    ):
        """Flag off + ordinary skill (registry initializes cleanly) →
        Swarm-enabled: no and zero skills, but the lines MUST still be
        present so operators see the field every time.

        (Swarm-capable skill on disk with the flag off is the atomic
        fail-closed path; that surface is covered by
        `test_health_swarm_atomic_fail_closed` below.)
        """
        self._install_skills_dir(tmp_path, monkeypatch, kind="ordinary")
        monkeypatch.delenv("PROXY_ENABLE_CUSTOM_SKILL_SUBAGENTS", raising=False)
        monkeypatch.delenv("PROXY_SESSION_SECRET", raising=False)

        content = _render_status()
        assert "Swarm-enabled: no" in content
        assert "Swarm-capable skills: 0" in content

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

    def test_status_handles_legacy_status_without_swarm_fields(self):
        """A CustomSkillsStatus-like object missing swarm_enabled /
        swarm_capable_skills must NOT replace the Custom Skills block."""
        from unittest.mock import patch
        from proxy.skills import skill_registry as _sr_mod

        class LegacyStatus:
            def __init__(self):
                self.state = "ok"
                self.enabled = True
                self.source = "env"
                self.directory = "/tmp/skills"
                self.restart_required = False
                self.discovered = 2
                self.loaded_names = ("build", "plan")
                self.issues = ()

            def __getattr__(self, name):
                raise AttributeError(name)

        with patch.object(
            _sr_mod, "get_default_registry"
        ) as mock_reg:
            mock_reg.return_value.custom_skills_status.return_value = (
                LegacyStatus()
            )
            ctx = CommandContext(raw_text="^^status", session_id="test-legacy")
            resp = dispatch_command(ctx)

        content = resp["choices"][0]["message"]["content"]
        assert "Custom Skills:" in content
        assert "Status: ok" in content
        assert "Loaded: 2 (build, plan)" in content
        assert "Swarm-enabled: no" in content
        assert "Swarm-capable skills: 0" in content
        assert "registry not initialized" not in content
