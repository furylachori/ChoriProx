# tests/unit/clients/test_proxy_models_json.py
"""Regression tests for the on-disk src/proxy_models.json file.

These tests load the ACTUAL configuration file (not a synthesized one)
to catch JSON syntax errors and missing-model regressions at CI time.

Background: commit d873719 introduced a JSON syntax error (missing
closing braces) that broke proxy startup. The proxy booted with zero
models and every request fell through to a dead default backend. The
silent-fallback path in ModelRegistry._load() made this invisible
until the user hit the proxy. This file pins the config so that
future edits cannot reintroduce the regression.
"""

import json
import os
from pathlib import Path

import pytest

from clients.model_registry import ModelRegistry


REPO_ROOT = Path(__file__).resolve().parents[3]
PROXY_MODELS_PATH = REPO_ROOT / "src" / "proxy_models.json"


REQUIRED_MODELS = {
    "qwen3-coder",
    "mini-m3",
    "stepfun",
    "glm",
    "deepseek-pro",
}


class TestProxyModelsJsonOnDisk:
    """Load the real on-disk proxy_models.json and assert invariants."""

    def test_file_exists(self):
        assert PROXY_MODELS_PATH.exists(), (
            f"Expected proxy_models.json at {PROXY_MODELS_PATH}"
        )

    def test_file_parses_as_valid_json(self):
        with open(PROXY_MODELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_loads_at_least_18_models(self):
        with open(PROXY_MODELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["models"]) >= 18, (
            f"Expected at least 18 models, got {len(data['models'])}"
        )

    def test_required_models_present(self):
        with open(PROXY_MODELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        missing = REQUIRED_MODELS - set(data["models"].keys())
        assert not missing, f"Missing required models: {missing}"

    def test_every_model_has_backend_and_tool_parser(self):
        with open(PROXY_MODELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for name, cfg in data["models"].items():
            assert isinstance(cfg, dict), f"{name} config is not a dict"
            backend = cfg.get("backend", "")
            assert isinstance(backend, str) and backend, (
                f"{name} has empty or missing backend"
            )
            tool_parser = cfg.get("tool_parser", "")
            assert isinstance(tool_parser, str) and tool_parser, (
                f"{name} has empty or missing tool_parser"
            )

    def test_default_section_present(self):
        with open(PROXY_MODELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert "default" in data
        assert isinstance(data["default"], dict)
        assert data["default"].get("backend")
        assert data["default"].get("tool_parser")

    def test_model_registry_loads_real_file_without_error(self):
        reg = ModelRegistry(
            config_path="proxy_models.json",
            base_dir=str(REPO_ROOT / "src"),
        )
        loaded = dict(reg.list_models())
        assert len(loaded) >= 18
        for name in REQUIRED_MODELS:
            assert name in loaded, f"ModelRegistry did not load required model {name}"
            cfg = loaded[name]
            assert cfg.backend, f"{name} has empty backend after load"
            assert cfg.tool_parser, f"{name} has empty tool_parser after load"


class TestLoaderRaisesOnMalformedJson:
    """The hardened loader must raise on JSON parse errors."""

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "proxy_models.json"
        path.write_text("{this is not valid json")
        with pytest.raises(json.JSONDecodeError):
            ModelRegistry(base_dir=str(tmp_path))

    def test_truncated_json_raises(self, tmp_path):
        path = tmp_path / "proxy_models.json"
        path.write_text('{"models": {"foo": {"backend":')
        with pytest.raises(json.JSONDecodeError):
            ModelRegistry(base_dir=str(tmp_path))

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "proxy_models.json"
        path.write_text("")
        with pytest.raises(json.JSONDecodeError):
            ModelRegistry(base_dir=str(tmp_path))

    def test_missing_file_silent_for_test_compat(self, tmp_path):
        """OSError (file missing) must remain silent so unit tests can
        instantiate ModelRegistry without a config file present."""
        reg = ModelRegistry(base_dir=str(tmp_path))
        assert reg._models == {}
        assert reg._default.backend == "http://localhost:7777"

    def test_schema_warnings_still_emitted(self, tmp_path, caplog):
        """Non-dict JSON, models-as-list etc. still warn (not raise).

        A parse error is a different class from a schema mismatch:
        a list at the top level is valid JSON, just structurally wrong.
        We only fail loud on parse errors, per the bug policy.
        """
        import logging
        path = tmp_path / "proxy_models.json"
        path.write_text("[1, 2, 3]")
        with caplog.at_level(logging.WARNING):
            reg = ModelRegistry(base_dir=str(tmp_path))
        assert "must be a JSON object" in caplog.text
        assert reg._models == {}
