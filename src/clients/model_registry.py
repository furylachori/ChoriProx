import os
import re
import json
import logging
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


ALLOWED_SCHEMES = {"http", "https"}
MAX_NAME_LENGTH = 128
NAME_PATTERN = re.compile(r"^[\w\s\-./]+$")


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    tool_parser: str
    name: str = ""
    context_limit: int = 32000
    supports_vision: bool = False
    supports_parallel_tools: bool = True
    api_key_env: str = ""
    backend_model: str = ""
    ignore_subagents: bool = False
    memory_enabled: bool = True
    memory_injection_enabled: bool = False
    raw_exchange_count: int = 0  # 0 = auto-derived from context_limit
    budget_ratio: Optional[float] = None  # per-model override for CONTEXT_BUDGET_RATIO


class ToolParser(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def parse(self, content: str) -> list[dict]: ...


class OpenAIToolParser(ToolParser):
    name = "openai"

    def parse(self, content: str) -> list[dict]:
        return []


class QwenXMLToolParser(ToolParser):
    name = "xml"

    def parse(self, content: str) -> list[dict]:
        from proxy.tool_parsing.parsers import (
            parse_all_tools_from_content as _parse_all,
        )
        return _parse_all(content)


class ToolParserRegistry:
    def __init__(self):
        self._parser_classes: dict[str, Callable[[], ToolParser]] = {
            "openai": OpenAIToolParser,
            "xml": QwenXMLToolParser,
        }
        # Pre-instantiate all parsers (thread-safe, no warm-up race)
        self._parser_instances: dict[str, ToolParser] = {}
        for parser_name, parser_cls in self._parser_classes.items():
            try:
                self._parser_instances[parser_name] = parser_cls()
            except Exception as e:
                logger.warning(f"Failed to instantiate parser '{parser_name}': {e}")

    def get(self, parser_name: str, strict: bool = False) -> ToolParser:
        cls = self._parser_classes.get(parser_name)
        if cls is None:
            if strict:
                raise KeyError(f"Unknown parser: {parser_name}")
            logger.warning(f"Unknown parser '{parser_name}', falling back to 'xml'")
            return self._parser_instances.get("xml", self._parser_classes["xml"]())
        instance = self._parser_instances.get(parser_name)
        if instance is None:
            # Fallback: instantiate on demand if pre-instantiation failed
            try:
                instance = cls()
                self._parser_instances[parser_name] = instance
            except Exception as e:
                logger.warning(f"Failed to instantiate parser '{parser_name}': {e}")
                return self._parser_instances.get("xml", cls())
        return instance


class ModelRegistry:
    def __init__(self, config_path: str = "proxy_models.json", base_dir: str = ""):
        self._models: dict[str, ModelConfig] = {}
        self._default = ModelConfig(
            backend="http://localhost:7777",
            tool_parser="xml",
            context_limit=32000,
        )
        # NOTE: Default parser changed from "openai" to "xml" in Phase 3.
        # All models without an explicit config entry will now attempt XML parsing.
        # This matches the behavior of parse_all_tools_from_content() used by the proxy.
        self._config_path = (
            os.path.join(base_dir, config_path) if base_dir else config_path
        )
        self._load(self._config_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Model registry {path} must be a JSON object")
            return

        known = {f.name for f in fields(ModelConfig)}

        models_data = data.get("models", {})
        if not isinstance(models_data, dict):
            logger.warning(
                f"Model registry 'models' must be an object, got {type(models_data).__name__}"
            )
            return

        for name, cfg in models_data.items():
            if isinstance(cfg, dict):
                # Validate backend is a string
                backend = cfg.get("backend", "")
                if not isinstance(backend, str):
                    logger.warning(
                        f"Invalid backend type for model '{name}': {type(backend).__name__}"
                    )
                    continue
                parsed = urlparse(backend)
                if parsed.scheme not in ALLOWED_SCHEMES:
                    logger.warning(
                        f"Invalid scheme for model '{name}': {parsed.scheme}"
                    )
                    continue
                if not parsed.netloc:
                    logger.warning(f"Empty netloc for model '{name}'")
                    continue

                # Validate name field
                model_name = cfg.get("name", "")
                if model_name and (
                    not isinstance(model_name, str)
                    or len(model_name) > MAX_NAME_LENGTH
                    or not NAME_PATTERN.match(model_name)
                ):
                    logger.warning(f"Invalid name for model '{name}': {model_name}")
                    cfg = dict(cfg)  # make mutable copy
                    cfg["name"] = ""

                filtered = {k: v for k, v in cfg.items() if k in known}
                try:
                    cfg_obj = ModelConfig(**filtered)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Invalid model config for {name}: {e}")
                    continue

                if cfg_obj.backend.startswith("https") and not cfg_obj.backend_model:
                    logger.warning(
                        f"Remote model '{name}' has no backend_model set — model rewriting disabled"
                    )
                if cfg_obj.api_key_env and cfg_obj.api_key_env not in os.environ:
                    logger.warning(
                        f"Model '{name}' references env var '{cfg_obj.api_key_env}' which is not set"
                    )

                self._models[name] = cfg_obj

        default_cfg = data.get("default")
        if isinstance(default_cfg, dict):
            filtered = {k: v for k, v in default_cfg.items() if k in known}
            try:
                self._default = ModelConfig(**filtered)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid default model config: {e}")

    def get(self, model_name: str) -> ModelConfig:
        if model_name is None:
            return self._default
        if model_name in self._models:
            return self._models[model_name]
        # Only prefix-match keys ending with '*' (explicit wildcard)
        matches = [
            (prefix, cfg)
            for prefix, cfg in self._models.items()
            if prefix.endswith("*") and model_name.startswith(prefix.rstrip("*"))
        ]
        if matches:
            return max(matches, key=lambda x: len(x[0]))[1]
        # Case-insensitive fallback
        lower = model_name.lower()
        for key, cfg in self._models.items():
            if key.lower() == lower:
                return cfg
        # Normalize hyphens/underscores as equivalent so "deepseek-v4-pro" matches "deepseek_v4_pro"
        normalized = lower.replace("-", "_")
        for key, cfg in self._models.items():
            if key.lower().replace("-", "_") == normalized:
                return cfg
        # Match against backend_model values — clients often send the backend model name directly
        for key, cfg in self._models.items():
            if cfg.backend_model and cfg.backend_model.lower() == lower:
                return cfg
        if normalized:
            for key, cfg in self._models.items():
                if cfg.backend_model and cfg.backend_model.lower().replace("-", "_") == normalized:
                    return cfg
        return self._default

    def list_models(self) -> list[tuple[str, ModelConfig]]:
        """Return list of (model_key, config) pairs for registered models."""
        return list(self._models.items())

    def get_default(self) -> ModelConfig:
        """Return the default fallback model config."""
        return self._default

    def auto_detect(self, request_body: dict) -> ModelConfig:
        model_name = request_body.get("model", "")
        return self.get(model_name)
