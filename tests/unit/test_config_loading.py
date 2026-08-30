"""Config loading integration tests.

Verify adapters load real JSON/YAML/markdown files from disk.
"""

import json
import pytest
import os

from clients.kilocode import KiloCodeAdapter
from clients.opencode import OpenCodeAdapter
from clients.interface import ConfigLoader


class TestConfigLoading:
    def test_kilocode_loads_json_config(self, tmp_path):
        agent_dir = tmp_path / ".kilo" / "agent"
        agent_dir.mkdir(parents=True)
        config_path = agent_dir / "default.json"
        config_path.write_text(
            json.dumps({"name": "my-agent", "model": "gpt-4", "temperature": 0.5}),
            encoding="utf-8",
        )

        adapter = KiloCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))

        assert cfg.name == "my-agent"
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.5

    def test_kilocode_skips_malformed_json(self, tmp_path):
        agent_dir = tmp_path / ".kilo" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "bad.json").write_text("not json", encoding="utf-8")
        (agent_dir / "good.json").write_text(
            json.dumps({"name": "good-agent", "model": "gpt-3.5"}), encoding="utf-8"
        )

        adapter = KiloCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))

        assert cfg.name == "good-agent"
        assert cfg.model == "gpt-3.5"

    def test_kilocode_fallback_when_no_config(self, tmp_path):
        adapter = KiloCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))
        assert cfg.name == "kilocode"
        assert cfg.model == ""
        assert cfg.temperature == 0.1

    def test_opencode_loads_markdown_frontmatter(self, tmp_path):
        opencode_dir = tmp_path / ".opencode"
        opencode_dir.mkdir()
        md_path = opencode_dir / "agent.md"
        md_path.write_text(
            "---\nname: opencode-agent\nmodel: claude-3\ntemperature: 0.7\n---\n\n# Agent\n",
            encoding="utf-8",
        )

        adapter = OpenCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))

        assert cfg.name == "opencode-agent"
        assert cfg.model == "claude-3"
        assert cfg.temperature == 0.7

    def test_opencode_skips_malformed_frontmatter(self, tmp_path):
        opencode_dir = tmp_path / ".opencode"
        opencode_dir.mkdir()
        # This is genuinely invalid YAML and will raise yaml.YAMLError
        (opencode_dir / "bad.md").write_text(
            "---\n[unclosed: bracket\n---\n", encoding="utf-8"
        )
        (opencode_dir / "good.md").write_text(
            "---\nname: good-agent\nmodel: gpt-4\n---\n", encoding="utf-8"
        )

        adapter = OpenCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))

        assert cfg.name == "good-agent"
        assert cfg.model == "gpt-4"

    def test_opencode_fallback_when_no_config(self, tmp_path):
        adapter = OpenCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))
        assert cfg.name == "opencode"
        assert cfg.model == ""
        assert cfg.temperature == 0.1

    def test_kilocode_ignores_non_json_files(self, tmp_path):
        agent_dir = tmp_path / ".kilo" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "notes.txt").write_text("hello", encoding="utf-8")
        (agent_dir / "config.json").write_text(
            json.dumps({"name": "txt-ignore"}), encoding="utf-8"
        )

        adapter = KiloCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))
        assert cfg.name == "txt-ignore"

    def test_opencode_ignores_non_md_files(self, tmp_path):
        opencode_dir = tmp_path / ".opencode"
        opencode_dir.mkdir()
        (opencode_dir / "readme.txt").write_text("hello", encoding="utf-8")
        (opencode_dir / "agent.md").write_text(
            "---\nname: md-only\n---\n", encoding="utf-8"
        )

        adapter = OpenCodeAdapter()
        cfg = adapter.load_config(str(tmp_path))
        assert cfg.name == "md-only"

    def test_config_loader_real_filesystem(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "test.txt").write_text("hello", encoding="utf-8")

        loader = ConfigLoader()
        assert loader.is_dir(str(sub))
        assert loader.is_file(str(sub / "test.txt"))
        assert "test.txt" in loader.list_dir(str(sub))
        assert loader.read_file(str(sub / "test.txt")) == "hello"
        assert loader.path_join(str(sub), "test.txt") == str(sub / "test.txt")

    def test_kilocode_load_config_uses_injected_loader(self, tmp_path):
        agent_dir = tmp_path / ".kilo" / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "cfg.json").write_text(
            json.dumps({"name": "injected"}), encoding="utf-8"
        )

        class SpyLoader(ConfigLoader):
            def read_file(self, path: str) -> str:
                self.last_path = path
                return super().read_file(path)

        spy = SpyLoader()
        adapter = KiloCodeAdapter()
        cfg = adapter.load_config(str(tmp_path), loader=spy)

        assert cfg.name == "injected"
        assert hasattr(spy, "last_path")
