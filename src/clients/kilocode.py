import json
import logging
import os
from clients.interface import ClientAdapter, CapabilitySet, ClientConfig, ConfigLoader
from clients.capabilities import KILOCODE_CAPABILITIES

logger = logging.getLogger(__name__)


class KiloCodeAdapter(ClientAdapter):
    name = "kilocode"

    def get_capabilities(self) -> CapabilitySet:
        return KILOCODE_CAPABILITIES

    def get_tool_names(self) -> set[str]:
        from proxy.tool_pipeline.tool_registry import registry

        return registry.get_default_tools()

    def normalize_tool_args(self, tool_name: str, arguments: dict) -> dict:
        arguments = dict(arguments)
        mappings = {
            "read": {"filePath": "path"},
            "write": {"filePath": "path"},
            "edit": {
                "filePath": "path",
                "oldString": "old_string",
                "newString": "new_string",
            },
        }
        if tool_name in mappings:
            for old, new in mappings[tool_name].items():
                if old in arguments:
                    arguments[new] = arguments.pop(old)
        return arguments

    def denormalize_tool_args(self, tool_name: str, arguments: dict) -> dict:
        arguments = dict(arguments)
        mappings = {
            "read": {"path": "filePath"},
            "write": {"path": "filePath"},
            "edit": {
                "path": "filePath",
                "old_string": "oldString",
                "new_string": "newString",
            },
        }
        if tool_name in mappings:
            for old, new in mappings[tool_name].items():
                if old in arguments:
                    arguments[new] = arguments.pop(old)
        return arguments

    def load_config(
        self, project_root: str, loader: ConfigLoader = None
    ) -> ClientConfig:
        loader = loader or ConfigLoader()
        agent_dir = loader.path_join(project_root, ".kilo", "agent")
        if loader.is_dir(agent_dir):
            for f in sorted(loader.list_dir(agent_dir)):
                if f.endswith(".json"):
                    try:
                        raw = json.loads(
                            loader.read_file(loader.path_join(agent_dir, f))
                        )
                        return ClientConfig(
                            name=raw.get("name", "unknown"),
                            model=raw.get("model", ""),
                            temperature=raw.get("temperature", 0.1),
                        )
                    except (json.JSONDecodeError, AttributeError, TypeError) as e:
                        logger.warning(f"Failed to load KiloCode config: {e}")
        return ClientConfig(name="kilocode")

    def get_system_prompt(self, context: dict) -> str:
        return "You are an AI assistant with access to tools.\nUse tools when needed. Keep responses concise."
