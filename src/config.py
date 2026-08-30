import json
import logging
import os

logger = logging.getLogger("config")


# Default heartbeat interval (seconds) for SSE keepalive comments emitted by
# the streaming delivery generator. Validated by _resolve_heartbeat_interval
# below: must be a positive int/float; zero and negatives are rejected with
# a WARN log and a fall-back to this default.
DEFAULT_HEARTBEAT_INTERVAL = 5.0


DEFAULT_CONFIG = {
    "security": {"config_auth_enabled": False, "config_api_key_env": None},
    "context": {
        "enable_token_calculation": True,
        "max_message_content_size_kb": 100,
        "max_total_content_size_kb": 500,
    },
    "upstream": {
        "timeout": 1800,
        "timeout_multiplier": 2.0,
        "retry_max_attempts": 5,
        "retry_base_delay": 2.0,
        "retry_max_delay": 30.0,
    },
    "streaming": {
        "heartbeat_interval": DEFAULT_HEARTBEAT_INTERVAL,
    },
}


def _resolve_heartbeat_interval(value):
    """Validate the configured heartbeat interval.

    Accepts positive int / float seconds. Rejects zero and negatives with a
    WARNING log and falls back to ``DEFAULT_HEARTBEAT_INTERVAL``. Falls back
    silently to the default for non-numeric values (defensive: malformed
    input MUST NOT take down the streaming path).
    """
    if isinstance(value, bool):
        # bool is a subclass of int; reject True/False explicitly to avoid
        # accidentally treating it as numeric.
        logger.warning(
            "streaming.heartbeat_interval=%r is not a positive numeric; "
            "falling back to default %s",
            value, DEFAULT_HEARTBEAT_INTERVAL,
        )
        return DEFAULT_HEARTBEAT_INTERVAL
    if not isinstance(value, (int, float)):
        logger.warning(
            "streaming.heartbeat_interval=%r is not a positive numeric; "
            "falling back to default %s",
            value, DEFAULT_HEARTBEAT_INTERVAL,
        )
        return DEFAULT_HEARTBEAT_INTERVAL
    if value <= 0:
        logger.warning(
            "streaming.heartbeat_interval=%r is non-positive; "
            "falling back to default %s",
            value, DEFAULT_HEARTBEAT_INTERVAL,
        )
        return DEFAULT_HEARTBEAT_INTERVAL
    return float(value)


def load_config(path="proxy_config.json"):
    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
            for section in user:
                if section in config:
                    config[section].update(user[section])
                else:
                    config[section] = dict(user[section])
    for section in config:
        for key in config[section]:
            env_key = f"PROXY_{section.upper()}_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                if isinstance(config[section][key], bool):
                    config[section][key] = env_val.lower() == "true"
                elif config[section][key] is None or isinstance(
                    config[section][key], str
                ):
                    config[section][key] = env_val if env_val != "" else None
                else:
                    # Coerce numeric/typed env values. If the user provided
                    # something that can't be coerced (e.g. PROXY_FOO_BAR=junk
                    # for an int slot), surface it as a WARNING and fall back
                    # to the default rather than crashing config load.
                    try:
                        config[section][key] = type(config[section][key])(env_val)
                    except (ValueError, TypeError):
                        logger.warning(
                            "%s=%r could not be coerced to %s; "
                            "falling back to default %r",
                            env_key, env_val,
                            type(config[section][key]).__name__,
                            config[section][key],
                        )
    # Validate the streaming heartbeat interval AFTER env-var overrides so a
    # misconfigured PROXY_STREAMING_HEARTBEAT_INTERVAL=0 is rejected rather
    # than baked into CONFIG. The validation lives in load_config rather than
    # the streaming call site so the warning is emitted exactly once at boot,
    # not per-request.
    streaming_section = config.get("streaming", {})
    streaming_section["heartbeat_interval"] = _resolve_heartbeat_interval(
        streaming_section.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)
    )
    config["streaming"] = streaming_section
    return config


CONFIG = load_config()
