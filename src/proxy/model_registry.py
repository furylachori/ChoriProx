"""Model resolution, URL construction, adapter management."""
import os
import re
import logging
import json
import hashlib
from urllib.parse import urlparse, urlunparse

from proxy.utils import extract_text_content

from proxy.state import (
    context_config_lock,
    model_config_cache,
)

logger = logging.getLogger(__name__)

CONTEXT_CONFIG = {
    "default": {
        "max_context_tokens": 32000,
        "safe_context_tokens": 19500,
        "optimal_context_tokens": 15000,
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": 8,
    },
    "qwen3-coder": {
        "max_context_tokens": 32000,
        "safe_context_tokens": 19500,
        "optimal_context_tokens": 15000,
        "tokens_per_char": 0.25,
        "chars_per_token": 4,
        "context_reduction_ratio": 0.75,
        "raw_exchange_count": 8,
    },
}

ENABLE_TOKEN_CALCULATION = True
TEST_MODE = False
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_/][a-zA-Z0-9_\-./]{0,63}$")
MAX_MODEL_CACHE_SIZE = 100
MAX_MESSAGE_CONTENT_SIZE = 100 * 1024
MAX_TOTAL_CONTENT_SIZE = 500 * 1024
MAX_RESPONSE_TOKENS = 1200

# Adapter globals
_model_registry = None
_tool_parser_registry = None
ACTIVE_CLIENT = None
USE_ADAPTERS = True


def _sanitize_url(url: str) -> str:
    """Redact userinfo and query params from URLs for logging."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc
        if "@" in netloc:
            netloc = netloc.split("@", 1)[1]
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))
    except ValueError:
        return "[redacted]"


def is_valid_model_name(name: str) -> bool:
    return bool(name and len(name) <= 64 and MODEL_NAME_PATTERN.match(name))


def resolve_session_id(body: dict, client_hint: str = "") -> str:
    """Derive session ID from first 10 user messages. Deterministic, stable.

    NOTE (C-25): session_id is content-derived and therefore rotatable — an attacker
    can change their first message to get a new session_id. This makes per-session
    rate limiting trivially bypassable. The skill rate limit in dispatch.py mitigates
    this with a secondary IP-based bucket; any future per-session limits should
    follow the same dual-key pattern or use proper authentication.
    """
    msgs = body.get("messages", [])
    user_contents = []
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = extract_text_content(msg.get("content", ""))
            if content:
                user_contents.append(content)
            if len(user_contents) >= 10:
                break
    if user_contents:
        combined = "|".join(user_contents) + ":" + client_hint
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    return hashlib.sha256(
        f"{json.dumps(body, sort_keys=True)}:{client_hint}".encode()
    ).hexdigest()[:16]


def extract_session_id(body):
    """DEPRECATED: Use resolve_session_id instead."""
    return resolve_session_id(body)


def resolve_model_route(model_name: str, registry) -> "ModelRoute":
    """Resolve model config, upstream URL, and tool parser.

    Validates URL scheme (http/https only) to prevent SSRF.
    Raises ValueError for invalid model names or URL schemes.
    """
    from proxy.types import ModelRoute
    from proxy.model_registry import _tool_parser_registry

    if not is_valid_model_name(model_name):
        raise ValueError(f"Invalid model name: {model_name}")

    model_cfg = registry.get(model_name)

    backend = model_cfg.backend.rstrip("/")
    if backend.endswith("/v1/chat/completions"):
        upstream_url = backend
    elif backend.endswith("/v1"):
        upstream_url = f"{backend}/chat/completions"
    else:
        # Append /v1/chat/completions for bare endpoints (default Ollama-style),
        # but preserve custom paths like /chat/completions on DeepSeek.
        parsed_backend = urlparse(backend)
        if parsed_backend.path in ("", "/"):
            upstream_url = f"{backend}/v1/chat/completions"
        else:
            upstream_url = backend

    parsed = urlparse(upstream_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid upstream URL scheme '{parsed.scheme}' for model '{model_name}'. "
            f"Only http and https are allowed."
        )

    tool_parser = None
    is_remote = True

    tpr = _tool_parser_registry
    if tpr and model_cfg:
        tool_parser = tpr.get(model_cfg.tool_parser)

    return ModelRoute(model_cfg, upstream_url, tool_parser, is_remote)


def calculate_text_from_tokens(token_count, model_name=None):
    """Calculate approximate text length from token count."""
    if not ENABLE_TOKEN_CALCULATION:
        return 0
    config = get_context_config(model_name)
    return int(token_count * config["chars_per_token"])


def _build_context_config(model_name, model_cfg, context_limit_override):
    """Compute a fresh context-config dict for `model_name`/`model_cfg`.

    Does NOT read or write model_config_cache — callers (only get_context_config)
    own cache lifecycle. When `context_limit_override` is None, the real model
    window is used. When set, the budget is clamped to min(override, real) via
    resolve_effective_context_limit and returned fresh (never cached, to prevent
    cross-session leak).
    """
    try:
        if model_cfg is not None:
            cfg = model_cfg
        else:
            reg = globals().get("_model_registry")
            if reg is not None:
                cfg = reg.get(model_name)
            else:
                raise AttributeError("_model_registry not available")
        import proxy.state as _state
        ratio = getattr(cfg, 'budget_ratio', None)
        if ratio is None:
            ratio = getattr(_state, 'CONTEXT_BUDGET_RATIO', 0.75)
        # R2 #4 (per-model budget_ratio clamp): per-model `cfg.budget_ratio`
        # is clamped to the same [0.1, 0.95] range as the
        # `CONTEXT_BUDGET_RATIO` state fallback so a model config with
        # budget_ratio near 1.0 cannot remove the entire discount
        # cushion (the F2 audit tail). The clamp itself was added by
        # R1-A W1-C; this comment makes the F2 audit trail explicit
        # so future readers do not regress it.
        ratio = max(0.1, min(0.95, ratio))
        if context_limit_override is not None:
            from proxy.context import resolve_effective_context_limit
            effective = resolve_effective_context_limit(cfg, context_limit_override)
            return {
                "max_context_tokens": effective,
                "safe_context_tokens": int(effective * ratio),
                "optimal_context_tokens": int(effective * 0.47),
                "tokens_per_char": 0.25,
                "chars_per_token": 4,
                "context_reduction_ratio": 0.75,
                "raw_exchange_count": cfg.raw_exchange_count if cfg.raw_exchange_count > 0 else max(5, effective // 4000),
            }
        return {
            "max_context_tokens": cfg.context_limit,
            "safe_context_tokens": int(cfg.context_limit * ratio),
            "optimal_context_tokens": int(cfg.context_limit * 0.47),
            "tokens_per_char": 0.25,
            "chars_per_token": 4,
            "context_reduction_ratio": 0.75,
            "raw_exchange_count": cfg.raw_exchange_count if cfg.raw_exchange_count > 0 else max(5, cfg.context_limit // 4000),
        }
    except (AttributeError, KeyError, TypeError):
        return CONTEXT_CONFIG.get(
            model_name,
            CONTEXT_CONFIG.get(
                "default",
                {
                    "max_context_tokens": 32000,
                    "safe_context_tokens": 19500,
                    "optimal_context_tokens": 15000,
                    "tokens_per_char": 0.25,
                    "chars_per_token": 4,
                    "context_reduction_ratio": 0.75,
                    "raw_exchange_count": max(5, 32000 // 4000),
                },
            ),
        )


def get_context_config(model_name=None, model_cfg=None, context_limit_override=None):
    """Get context configuration for the given model.

    Override path returns a fresh, un-cached config (no eviction, no cache
    write) so per-session overrides cannot leak across sessions. Non-override
    path uses the LRU cache as before.
    """
    if not model_name:
        model_name = "default"

    # Override path: bypass the cache entirely (no eviction, no write).
    if context_limit_override is not None:
        return _build_context_config(model_name, model_cfg, context_limit_override)

    with context_config_lock:
        if model_name in model_config_cache:
            model_config_cache.move_to_end(model_name)
            return model_config_cache[model_name][0]

        if len(model_config_cache) >= MAX_MODEL_CACHE_SIZE:
            evicted_key, _ = model_config_cache.popitem(last=False)
            logger.info(f"Evicted model: {evicted_key}")

    config = _build_context_config(model_name, model_cfg, None)
    with context_config_lock:
        model_config_cache[model_name] = (config, None)
    return config


def get_safe_context_limit(model_name=None, model_cfg=None):
    """Get the safe context limit for the model."""
    if not model_name:
        model_name = "default"

    with context_config_lock:
        if (
            model_name in model_config_cache
            and model_config_cache[model_name][1] is not None
        ):
            return model_config_cache[model_name][1]

    config = get_context_config(model_name, model_cfg=model_cfg)
    limits = {
        "max_tokens": config["max_context_tokens"],
        "safe_tokens": config["safe_context_tokens"],
        "optimal_tokens": config["optimal_context_tokens"],
        "safe_mb": calculate_text_from_tokens(config["safe_context_tokens"])
        / (1024 * 1024),
    }

    with context_config_lock:
        model_config_cache[model_name] = (config, limits)

    return limits


def _initialize_adapters():
    """Run adapter initialization. Called at module load time."""
    global _model_registry, _tool_parser_registry, ACTIVE_CLIENT, USE_ADAPTERS

    try:
        from clients import register_client, detect_client, get_client
        from clients.kilocode import KiloCodeAdapter
        from clients.opencode import OpenCodeAdapter
        from clients.interface import DetectionError, RegistryError
        from clients.model_registry import ModelRegistry, ToolParserRegistry

        USE_ADAPTERS = os.environ.get("PROXY_USE_ADAPTERS", "1") != "0"

        if USE_ADAPTERS:
            try:
                register_client(KiloCodeAdapter())
                register_client(OpenCodeAdapter())

                _model_registry = ModelRegistry(
                    config_path="proxy_models.json",
                    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                )
                _tool_parser_registry = ToolParserRegistry()

                _project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
                try:
                    ACTIVE_CLIENT = detect_client(_project_root)
                    logger.info(f"Active client detected: {ACTIVE_CLIENT.name}")
                except DetectionError as _e:
                    logger.info(f"No client detected, using defaults: {_e}")
                    ACTIVE_CLIENT = None
                except Exception as _e:
                    logger.info(f"Client detection error: {_e}")
                    ACTIVE_CLIENT = None
            except json.JSONDecodeError as _e:
                logger.error(
                    "FATAL: proxy_models.json is malformed: %s. Refusing to start.",
                    _e,
                    exc_info=True,
                )
                raise
            except Exception as _e:
                logger.error("Failed to initialize adapters: %s", _e, exc_info=True)
                USE_ADAPTERS = False
                _model_registry = None
                _tool_parser_registry = None
                ACTIVE_CLIENT = None
        else:
            ACTIVE_CLIENT = None
    except ImportError as _e:
        logger.info(f"Clients layer not available, using defaults: {_e}")
        USE_ADAPTERS = False
        _model_registry = None
        _tool_parser_registry = None
        ACTIVE_CLIENT = None


_initialize_adapters()
