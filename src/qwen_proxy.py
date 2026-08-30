"""Backward-compatible shim. Import from proxy.app instead."""
import sys

from proxy.app import app
from proxy.types import BackendError
from proxy.auth import (
    _sanitize_error_body, _SENSITIVE_KEYS,
    AuthConfigError, detect_malicious_content, INJECTION_PATTERNS,
)
from proxy.context import (
    select_context, calculate_tokens_from_text,
    calculate_tokens_from_request_body, calculate_text_from_tokens,
)
from proxy.streaming import stream_response, make_sse_chunk
from proxy.tool_pipeline.tool_registry import registry
from proxy.tool_parsing import (
    parse_yaml_tool_blocks, parse_all_tools_from_content,
    parse_tool_from_content, make_tool_calls,
    normalize_tool_arguments, provide_defaults_for_optional_params,
    OUR_SYSTEM_PROMPT,
)
from proxy.tool_cache import (
    validate_tool_schema, cache_tool_schema, MAX_TOOL_CACHE_SIZE,
)
from proxy.model_registry import (
    get_context_config, get_safe_context_limit,
    CONTEXT_CONFIG, ENABLE_TOKEN_CALCULATION, TEST_MODE,
    MODEL_NAME_PATTERN, MAX_MODEL_CACHE_SIZE,
    MAX_MESSAGE_CONTENT_SIZE, MAX_TOTAL_CONTENT_SIZE, MAX_RESPONSE_TOKENS,
    _model_registry, _tool_parser_registry, ACTIVE_CLIENT, USE_ADAPTERS,
    is_valid_model_name, resolve_session_id, extract_session_id,
    _sanitize_url, resolve_model_route,
)
from proxy.http_client import get_http_session
from proxy.state import _HTTP_SESSION, request_counter, request_counter_lock
from proxy.session import (
    CLEANUP_INTERVAL, SESSION_TTL, SUMMARY_PROMPT,
    get_chat_summary, cleanup_old_sessions,
    search_chat, get_last_n_chats,
)

if __name__ == "__main__":
    from proxy.app import _eager_init_registry, initialize_proxy_at_startup
    from proxy.skills.skill_registry import SkillRegistryConfigurationError
    try:
        _eager_init_registry()
    except SkillRegistryConfigurationError as exc:
        print(f"[STARTUP] Custom skills configuration invalid: {exc}", flush=True)
        for issue in exc.issues:
            print(f"[STARTUP]   {issue.file}: {issue.code} — {issue.message}", flush=True)
        sys.exit(1)

    try:
        initialize_proxy_at_startup()
    except Exception as e:
        print(f"[MEMORY] Database initialization failed: {e}", flush=True)
    app.run(host="0.0.0.0", port=6666, debug=False, threaded=True)
