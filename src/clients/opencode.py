import json
import logging
import re
from clients.interface import ClientAdapter, CapabilitySet, ClientConfig, ConfigLoader
from clients.capabilities import OPENCODE_CAPABILITIES

logger = logging.getLogger(__name__)


class OpenCodeAdapter(ClientAdapter):
    name = "opencode"

    def get_capabilities(self) -> CapabilitySet:
        return OPENCODE_CAPABILITIES

    def get_tool_names(self) -> set[str]:
        from proxy.tool_pipeline.tool_registry import registry

        return registry.get_default_tools()

    def normalize_tool_args(self, tool_name: str, arguments: dict) -> dict:
        # no-op: schema-driven normalization in proxy.tool_parsing.normalize supersedes this.
        return dict(arguments)

    def load_config(
        self, project_root: str, loader: ConfigLoader = None
    ) -> ClientConfig:
        import yaml

        loader = loader or ConfigLoader()
        agent_dir = loader.path_join(project_root, ".opencode")
        if loader.is_dir(agent_dir):
            for f in sorted(loader.list_dir(agent_dir)):
                if f.endswith(".md") and loader.is_file(loader.path_join(agent_dir, f)):
                    try:
                        content = loader.read_file(loader.path_join(agent_dir, f))
                        match = re.match(r"^---\r?\n(.*?)\r?\n---", content, re.DOTALL)
                        if match:
                            fm = yaml.safe_load(match.group(1))
                            if isinstance(fm, dict):
                                return ClientConfig(
                                    name=fm.get("name", f.replace(".md", "")),
                                    model=fm.get("model", ""),
                                    temperature=fm.get("temperature", 0.1),
                                )
                    except (OSError, yaml.YAMLError, AttributeError, TypeError) as e:
                        logger.warning(f"Failed to load OpenCode config: {e}")
        return ClientConfig(name="opencode")

    def get_system_prompt(self, context: dict) -> str:
        return "You are an AI coding assistant with access to tools.\nUse tools when needed. Keep responses concise."
