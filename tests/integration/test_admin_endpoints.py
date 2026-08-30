"""Integration tests for admin/utility endpoints.

Covers:
- US-062: /health accessible without auth, returns Neo4j + SQLite status
- /health is rate-limited by admin_rate_limiter
- US-063: /v1/models works when PROXY_API_KEY IS set

These endpoints live in proxy.admin (admin blueprint) and are registered
on the Flask app in proxy.app. The /v1/models and /health routes are
explicitly allowlisted by the fail-closed @before_request hook in
proxy.app, so they remain reachable when PROXY_API_KEY is unset.
"""

from unittest.mock import MagicMock, patch

import pytest

import proxy.state as state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """Flask test client with PROXY_API_KEY set so non-allowlisted paths
    would 500 (not relevant here — we only hit /health and /v1/models).
    """
    monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")
    from proxy.app import app as flask_app

    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def reset_admin_rate_limiter():
    """Reset admin_rate_limiter state between tests to avoid cross-test
    bleed from the persistent SQLite-backed entries.
    """
    state.admin_rate_limiter.reset()
    yield
    state.admin_rate_limiter.reset()


# ---------------------------------------------------------------------------
# US-062: /health returns Neo4j + SQLite status, no auth required
# ---------------------------------------------------------------------------


def _mock_sqlite_connection():
    manager = MagicMock()
    conn = manager.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = (1,)
    return manager, conn


class TestHealthEndpoint:
    """Tests for GET /health (US-062)."""

    def test_health_accessible_without_auth(self, monkeypatch, reset_admin_rate_limiter):
        """Health endpoint is allowlisted in the fail-closed check, so
        callers without PROXY_API_KEY can still reach it. Verifies the
        endpoint is reachable from a non-localhost caller with no
        credentials when PROXY_API_KEY is unset.
        """
        monkeypatch.delenv("PROXY_API_KEY", raising=False)
        from proxy.app import app as flask_app
        flask_app.config["TESTING"] = True

        mock_connection, _ = _mock_sqlite_connection()

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True

            with flask_app.test_client() as c:
                resp = c.get(
                    "/health",
                    environ_overrides={"REMOTE_ADDR": "10.0.0.1"},
                )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body is not None
        assert body["status"] == "ok"
        assert body["neo4j"] == "ok"
        assert body["sqlite"] == "ok"

    def test_health_reports_neo4j_down(self, client, reset_admin_rate_limiter):
        """When the task graph's Neo4j executor is unavailable, /health
        reports neo4j='down' and returns 503 Service Unavailable.
        """
        mock_connection, _ = _mock_sqlite_connection()

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = False

            resp = client.get("/health")

        assert resp.status_code == 503
        body = resp.get_json()
        assert body["neo4j"] == "degraded"
        assert body["status"] == "degraded"

    def test_health_reports_sqlite_down(self, client, reset_admin_rate_limiter):
        """When the SQLite connection raises, /health reports sqlite='down'
        and returns 503.
        """
        mock_connection, conn = _mock_sqlite_connection()
        conn.execute.side_effect = RuntimeError("db closed")

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True
            resp = client.get("/health")

        assert resp.status_code == 503
        body = resp.get_json()
        assert body["sqlite"] == "down"
        assert body["status"] == "unhealthy"
        mock_connection.return_value.__exit__.assert_called_once()

    def test_health_reports_neo4j_exception_as_down(self, client, reset_admin_rate_limiter):
        """When _get_task_graph_singleton() itself raises, /health reports
        neo4j='down' and 503 — the broad except in admin.health classifies
        any exception as down, not degraded.
        """
        mock_connection, _ = _mock_sqlite_connection()

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.side_effect = RuntimeError("singleton init failed")

            resp = client.get("/health")

        assert resp.status_code == 503
        body = resp.get_json()
        assert body["neo4j"] == "down"
        assert body["status"] == "unhealthy"

    def test_health_both_deps_ok_returns_200(self, client, reset_admin_rate_limiter):
        """Happy path: both Neo4j and SQLite healthy → 200 with status='ok'."""
        mock_connection, _ = _mock_sqlite_connection()

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True

            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.get_json()
        # Wave 2: /health now also reports a `custom_skills` block. Assert
        # only the documented dependency fields; do not pin the whole dict.
        assert body["status"] == "ok"
        assert body["neo4j"] == "ok"
        assert body["sqlite"] == "ok"

    def test_health_returns_sqlite_connection_to_pool(
        self,
        client,
        reset_admin_rate_limiter,
    ):
        """US-062: repeated real SQLite health checks return every checkout."""
        from memory.db import init_db
        import memory.db_pool as db_pool

        init_db()

        with patch(
            "proxy.context._get_task_graph_singleton"
        ) as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True

            for request_number in range(60):
                resp = client.get("/health")
                assert resp.status_code == 200, (
                    f"request {request_number + 1} returned "
                    f"{resp.status_code}: {resp.get_json()!r}"
                )
                body = resp.get_json()
                assert body["sqlite"] == "ok"

                pool = db_pool._pool
                assert pool is not None
                assert len(pool._checked_out) == 0, (
                    f"request {request_number + 1} leaked "
                    f"{len(pool._checked_out)} SQLite checkout(s)"
                )

        pool = db_pool._pool
        assert pool is not None
        with pool.connection() as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1
        assert len(pool._checked_out) == 0

    def test_health_is_rate_limited(self, client, monkeypatch):
        """The admin_rate_limiter caps /health per IP.

        Prod runs this limiter at 1000/min (commit 872964c raised it from
        30 to stop sub-agent fan-out self-DoS — see the "rate-limiter
        self-DoS" landmine in AGENTS.md). 1000 is unreachable in test-time,
        so we inject a tight limiter (3/60s) for this test only.

        The patch targets proxy.admin.admin_rate_limiter — admin.py binds
        the name via `from proxy.state import ...` at import time, so the
        route reads it from proxy.admin's namespace, not proxy.state.
        monkeypatch auto-restores the binding, so no state leaks to other
        tests; prod state.py is untouched.
        """
        from proxy.rate_limiter import RateLimiter
        tight = RateLimiter(max_requests=3, window_seconds=60, name="admin_test", persist=False)
        monkeypatch.setattr("proxy.admin.admin_rate_limiter", tight)

        mock_connection, _ = _mock_sqlite_connection()

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True

            for _ in range(3):
                ok = client.get("/health")
                assert ok.status_code == 200, (
                    f"Expected 200 within rate-limit budget, got {ok.status_code}"
                )

            limited = client.get("/health")

        assert limited.status_code == 429
        body = limited.get_json()
        assert body is not None
        assert body.get("error") == "rate_limited"
        assert "retry_after" in body
        assert isinstance(body["retry_after"], int)
        assert body["retry_after"] >= 0
        assert mock_connection.call_count == 3, (
            f"3 successful requests should invoke SQLite checkout exactly "
            f"3 times; the 4th (rate-limited) request must not. "
            f"Got {mock_connection.call_count} calls."
        )

    def test_health_reports_custom_skills_block_when_enabled(
        self, client, tmp_path, monkeypatch, reset_admin_rate_limiter
    ):
        """When PROXY_SKILLS_DIR points to a valid dir with one valid JSON,
        /health reports loaded_names correctly (Wave 2, spec §8.3).
        """
        import json as _json

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "foo.json").write_text(_json.dumps({
            "name": "foo",
            "trigger": "^^foo",
            "description": "Test skill",
            "prompt": "Test prompt",
            "tool_whitelist": ["read"],
        }))
        monkeypatch.setenv("PROXY_SKILLS_DIR", str(skills_dir))

        # Reset the singleton so the new env takes effect on the next
        # get_default_registry() call inside the /health handler.
        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

        # Bypass admin rate limiter with a permissive per-test limiter.
        from proxy.rate_limiter import RateLimiter
        tight = RateLimiter(
            max_requests=100, window_seconds=60, name="admin", persist=False
        )
        monkeypatch.setattr("proxy.admin.admin_rate_limiter", tight)

        mock_connection, _ = _mock_sqlite_connection()

        # Patch Neo4j + SQLite so the response is deterministic 200 — the
        # custom_skills block is what we're asserting on; the rest mirrors
        # the established pattern from the other 5 tests in this class.
        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True

            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.get_json()
        assert "custom_skills" in body
        cs = body["custom_skills"]
        assert cs["status"] == "ok"
        assert cs["loaded"] == 1
        assert "foo" in cs["loaded_names"]

    def test_health_reports_custom_skills_disabled_when_env_empty(
        self, client, monkeypatch, reset_admin_rate_limiter
    ):
        """When PROXY_SKILLS_DIR="", /health reports custom_skills
        status=disabled (spec §8.3 + §5.2 row 3).
        """
        monkeypatch.setenv("PROXY_SKILLS_DIR", "")

        from proxy.skills.skill_registry import reset_default_registry
        reset_default_registry()

        from proxy.rate_limiter import RateLimiter
        tight = RateLimiter(
            max_requests=100, window_seconds=60, name="admin", persist=False
        )
        monkeypatch.setattr("proxy.admin.admin_rate_limiter", tight)

        mock_connection, _ = _mock_sqlite_connection()

        with patch("memory.db.connection", mock_connection), \
             patch("proxy.context._get_task_graph_singleton") as mock_tg:
            mock_tg.return_value._executor.is_available.return_value = True

            resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.get_json()
        cs = body["custom_skills"]
        assert cs["status"] == "disabled"
        assert cs["enabled"] is False
        assert cs["loaded"] == 0
        assert cs["discovered"] == 0
        assert cs["loaded_names"] == []


# ---------------------------------------------------------------------------
# US-063: /v1/models works when PROXY_API_KEY IS set
# ---------------------------------------------------------------------------


class TestV1ModelsEndpoint:
    """Tests for GET /v1/models (US-063)."""

    def test_v1_models_returns_model_list(self, monkeypatch, tmp_path):
        """When PROXY_API_KEY is set, /v1/models returns the registered
        models list. The endpoint is always allowlisted (no auth required),
        but the spec for this test is the positive case where the proxy
        is fully configured.
        """
        monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")

        config = {
            "models": {
                "kimi": {
                    "backend": "http://localhost:7777/v1",
                    "tool_parser": "openai",
                    "context_limit": 262144,
                    "name": "ChoriProx Kimi",
                    "api_key_env": "TEST_API_KEY",
                    "backend_model": "opencode-go/kimi-k2.6",
                },
                "qwen3-coder": {
                    "backend": "http://localhost:7777/v1",
                    "tool_parser": "xml",
                    "context_limit": 32000,
                    "name": "ChoriProx Qwen3 Coder",
                },
            },
            "default": {
                "backend": "http://localhost:7777/v1",
                "tool_parser": "openai",
                "context_limit": 32000,
            },
        }
        cfg_path = tmp_path / "proxy_models.json"
        import json as _json
        cfg_path.write_text(_json.dumps(config))

        from clients.model_registry import ModelRegistry

        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        monkeypatch.setattr("proxy.app._model_registry", reg)
        monkeypatch.setattr("proxy.model_registry._model_registry", reg)
        monkeypatch.setattr("qwen_proxy._model_registry", reg)
        monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
        monkeypatch.setattr("proxy.app.USE_ADAPTERS", True)

        state.admin_rate_limiter.reset()

        from proxy.app import app as flask_app
        flask_app.config["TESTING"] = True

        with flask_app.test_client() as c:
            resp = c.get(
                "/v1/models",
                environ_overrides={"REMOTE_ADDR": "10.0.0.1"},
            )

        state.admin_rate_limiter.reset()

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["object"] == "list"
        model_ids = {m["id"] for m in body["data"]}
        assert "qwen3-coder" in model_ids
        assert "proxy-default" in model_ids

    def test_v2_models_returns_same_list(self, monkeypatch, tmp_path):
        """The blueprint registers /v1/models and /v2/models to the same
        handler. Both must return the model list.
        """
        monkeypatch.setenv("PROXY_API_KEY", "test-proxy-key")

        config = {
            "models": {
                "kimi": {
                    "backend": "http://localhost:7777/v1",
                    "tool_parser": "openai",
                    "context_limit": 262144,
                    "name": "ChoriProx Kimi",
                    "api_key_env": "TEST_API_KEY",
                    "backend_model": "opencode-go/kimi-k2.6",
                },
            },
            "default": {
                "backend": "http://localhost:7777/v1",
                "tool_parser": "openai",
                "context_limit": 32000,
            },
        }
        cfg_path = tmp_path / "proxy_models.json"
        import json as _json
        cfg_path.write_text(_json.dumps(config))

        from clients.model_registry import ModelRegistry

        reg = ModelRegistry(config_path="proxy_models.json", base_dir=str(tmp_path))
        monkeypatch.setattr("proxy.app._model_registry", reg)
        monkeypatch.setattr("proxy.model_registry._model_registry", reg)
        monkeypatch.setattr("qwen_proxy._model_registry", reg)
        monkeypatch.setattr("qwen_proxy.USE_ADAPTERS", True)
        monkeypatch.setattr("proxy.app.USE_ADAPTERS", True)

        state.admin_rate_limiter.reset()

        from proxy.app import app as flask_app
        flask_app.config["TESTING"] = True

        with flask_app.test_client() as c:
            resp = c.get(
                "/v2/models",
                environ_overrides={"REMOTE_ADDR": "10.0.0.1"},
            )

        state.admin_rate_limiter.reset()

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["object"] == "list"
        assert "qwen3-coder" in {m["id"] for m in body["data"]}
