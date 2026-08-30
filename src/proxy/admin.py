"""Flask Blueprint — admin/utility routes extracted from app.py."""

import os
import json
import hmac

from flask import Blueprint, request, jsonify, Response, current_app

import proxy.state as state
from proxy.state import (
    AppState,
    _HTTP_SESSION,
    CONFIG,
    chat_summaries,
    chat_histories,
    chat_summaries_lock,
    chat_histories_lock,
    tool_cache,
    tool_cache_lock,
    admin_rate_limiter,
)
from proxy.context import (
    calculate_tokens_from_text,
    get_context_config,
    get_safe_context_limit,
    ENABLE_TOKEN_CALCULATION,
)
from proxy.model_registry import (
    USE_ADAPTERS,
    _model_registry,
    _sanitize_url,
)
from proxy.tool_cache import MAX_TOOL_CACHE_SIZE
from proxy.session import get_chat_summary
from proxy.rate_limit_response import rate_limited_response

admin_bp = Blueprint("admin", __name__)


def _get_client_ip():
    """Lazy import to avoid circular dependency with app.py."""
    from proxy.app import get_client_ip
    return get_client_ip()


def _validate_json_request(expected_fields):
    """Validate JSON body has required fields with correct types.

    Returns (data, None) on success, (None, error_response) on failure.
    """
    data = request.get_json(silent=True)
    if data is None:
        return None, (jsonify({"error": "Invalid JSON body"}), 400)
    for field, field_type in expected_fields.items():
        if field not in data:
            return None, (jsonify({"error": f"Missing required field: {field}"}), 400)
        if not isinstance(data[field], field_type):
            return None, (jsonify({"error": f"Field {field} must be {field_type.__name__}"}), 400)
    return data, None


def check_config_auth():
    """Check config endpoint authentication. Returns None if OK, error response if denied."""
    if not CONFIG:
        return None

    if not CONFIG.get("security", {}).get("config_auth_enabled"):
        return None
    key_env_var = CONFIG["security"].get("config_api_key_env")
    if not key_env_var:
        return None
    expected_key = os.environ.get(key_env_var)
    if not expected_key:
        return jsonify({"error": "Config API key not configured on server"}), 500
    provided_key = request.headers.get("X-Config-Key")
    if not provided_key:
        return jsonify({"error": "Missing X-Config-Key header"}), 401
    if not hmac.compare_digest(provided_key, expected_key):
        return jsonify({"error": "Invalid config API key"}), 403
    return None


@admin_bp.route("/v1/models", methods=["GET"])
@admin_bp.route("/v2/models", methods=["GET"])
def proxy_models():
    """Return registered models from ModelRegistry ONLY."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    models = []

    if USE_ADAPTERS:
        reg = _model_registry
        if reg:
            for model_key, cfg in reg.list_models():
                display_name = cfg.name if cfg.name else model_key
                models.append(
                    {
                        "id": model_key,
                        "name": display_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "proxy",
                        "permission": [],
                        "root": model_key,
                        "parent": None,
                    }
                )
            default_cfg = reg.get_default()
            models.append(
                {
                    "id": "proxy-default",
                    "name": default_cfg.name or "default",
                    "object": "model",
                    "created": 0,
                    "owned_by": "proxy",
                    "permission": [],
                    "root": "proxy-default",
                    "parent": None,
                }
            )
    else:
        try:
            app_state = current_app.config["APP_STATE"]
            base_url = app_state.llama_url.rsplit("/", 3)[0]
            response = _HTTP_SESSION.get(
                f"{base_url}/v1/models", timeout=10, allow_redirects=False
            )
            return Response(response.content, content_type="application/json")
        except Exception:
            return jsonify({"error": "Backend unavailable"}), 500

    return jsonify({"object": "list", "data": models})


@admin_bp.route("/health", methods=["GET"])
def health():
    """Health check with Neo4j, SQLite, and custom-skills dependency checks."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    status = {"neo4j": "unknown", "sqlite": "unknown", "status": "ok"}
    http_code = 200

    try:
        from proxy.context import _get_task_graph_singleton
        tg = _get_task_graph_singleton()
        if tg._executor.is_available():
            status["neo4j"] = "ok"
        else:
            status["neo4j"] = "degraded"
            status["status"] = "degraded"
            http_code = 503
    except Exception:
        status["neo4j"] = "down"
        status["status"] = "unhealthy"
        http_code = 503

    try:
        from memory.db import connection
        with connection() as conn:
            conn.execute("SELECT 1")
        status["sqlite"] = "ok"
    except Exception:
        status["sqlite"] = "down"
        status["status"] = "unhealthy"
        http_code = 503

    # Custom skills (Wave 2) — read CustomSkillsStatus from the singleton
    # registry. If the singleton isn't constructed (lazy init), the loader's
    # resolve_skills_dir() runs and returns a safe default; never raises.
    try:
        from proxy.skills.skill_registry import get_default_registry
        registry = get_default_registry()
        cs_status = registry.custom_skills_status()
        swarm_enabled = bool(getattr(cs_status, "swarm_enabled", False))
        swarm_capable_skills = tuple(
            getattr(cs_status, "swarm_capable_skills", ()) or ()
        )
        custom_skills_block = {
            "status": cs_status.state,
            "enabled": cs_status.enabled,
            "source": cs_status.source,
            "directory": cs_status.directory,
            "restart_required": cs_status.restart_required,
            "discovered": cs_status.discovered,
            "loaded": len(cs_status.loaded_names),
            "loaded_names": list(cs_status.loaded_names),
            # Swarm operator-visibility (spec AC 16.15).
            "swarm_enabled": swarm_enabled,
            "swarm_capable_skills": list(swarm_capable_skills),
        }
        # Per spec §8.3: do not expose prompt contents or file contents on
        # unauthenticated /health. basename-only file paths in issues.
        if cs_status.issues:
            custom_skills_block["skipped"] = len(cs_status.issues)
            custom_skills_block["issues"] = [
                {"file": i.file.split("/")[-1].split("\\")[-1], "code": i.code}
                for i in cs_status.issues
            ]
        status["custom_skills"] = custom_skills_block
        # Wave 1's strict loader is atomic-fail-closed: a running process
        # has state in {"ok", "disabled"}. The "degraded" branch is a
        # future-state belt-and-suspenders per spec §8.3.
        if cs_status.state not in ("ok", "disabled"):
            status["status"] = "degraded"
            http_code = 503
    except Exception as exc:
        # If registry init raised SkillRegistryConfigurationError, the
        # singleton is None and the lazy path also raises — surface it
        # honestly.
        status["custom_skills"] = {
            "status": "error",
            "enabled": False,
            "source": "unknown",
            "directory": None,
            "error": type(exc).__name__,
        }
        status["status"] = "degraded"
        http_code = 503

    # Transparent request deduplication (CP4-5) — surface DedupStore stats.
    # stats() snapshots pending/completed entry counts, stored bytes, and the
    # hit/miss/eviction/conflict counters. Never raises into /health.
    try:
        status["dedup"] = state.dedup_store.stats()
    except Exception:
        status["dedup"] = {"error": "unavailable"}

    return jsonify(status), http_code


@admin_bp.route("/v1/context/config", methods=["GET"])
@admin_bp.route("/v2/context/config", methods=["GET"])
def context_config():
    """Get context configuration for debugging and monitoring."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = check_config_auth()
    if auth_result:
        return auth_result
    model_name = request.args.get("model", "default")
    limits = get_safe_context_limit(model_name)

    return jsonify(
        {
            "model": model_name,
            "max_tokens": limits["max_tokens"],
            "safe_tokens": limits["safe_tokens"],
            "status": "operational",
        }
    )


@admin_bp.route("/v1/context/config/toggle", methods=["POST"])
@admin_bp.route("/v2/context/config/toggle", methods=["POST"])
def toggle_token_calculation():
    """Enable/disable dynamic token calculation."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = check_config_auth()
    if auth_result:
        return auth_result

    data, error = _validate_json_request({"enabled": bool})
    if error:
        return error

    ENABLE_TOKEN_CALCULATION.enabled = data["enabled"]

    return jsonify(
        {
            "token_calculation_enabled": ENABLE_TOKEN_CALCULATION.enabled,
            "message": f"Token calculation {'enabled' if ENABLE_TOKEN_CALCULATION.enabled else 'disabled'}",
        }
    )


@admin_bp.route("/v1/context/sessions", methods=["GET"])
@admin_bp.route("/v2/context/sessions", methods=["GET"])
def list_sessions():
    """List all active sessions with statistics."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = check_config_auth()
    if auth_result:
        return auth_result
    sessions = []
    tokens_per_char = get_context_config()["tokens_per_char"]
    with chat_summaries_lock:
        session_data = [
            (
                sid,
                len(chat_summaries.get(sid, "")),
                calculate_tokens_from_text(chat_summaries.get(sid, ""), tokens_per_char),
            )
            for sid in chat_summaries.keys()
        ]
    with chat_histories_lock:
        for sid, summary_length, summary_tokens in session_data:
            sessions.append(
                {
                    "session_id": sid,
                    "summary_length": summary_length,
                    "history_messages": len(chat_histories.get(sid, [])),
                    "summary_tokens": summary_tokens,
                }
            )

    return jsonify({"total_sessions": len(sessions), "sessions": sessions})


@admin_bp.route("/v1/context/session/<session_id>", methods=["GET"])
@admin_bp.route("/v2/context/session/<session_id>", methods=["GET"])
def get_session(session_id):
    """Get details for a specific session."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = check_config_auth()
    if auth_result:
        return auth_result

    summary = get_chat_summary(session_id)
    with chat_histories_lock:
        history = chat_histories.get(session_id, [])
        if not history:
            # Not in cache — try reloading from SQLite
            from proxy.session import reload_session_from_db

            history = reload_session_from_db(session_id)
        if not history and not summary:
            return jsonify({"error": "Session not found"}), 404

    return jsonify(
        {
            "session_id": session_id,
            "summary": summary,
            "history": history,
            "summary_tokens": calculate_tokens_from_text(summary, get_context_config()["tokens_per_char"]),
        }
    )


@admin_bp.route("/v1/tools/cache", methods=["GET"])
@admin_bp.route("/v2/tools/cache", methods=["GET"])
def tool_cache_status():
    """Get tool cache statistics and status."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = check_config_auth()
    if auth_result:
        return auth_result
    with tool_cache_lock:
        cached_tool_hashes = list(tool_cache.keys())
        tool_count_per_hash = {
            hash_key: len(tools) for hash_key, tools in tool_cache.items()
        }
    return jsonify(
        {
            "cache_size": state.tool_cache_size,
            "max_cache_size": MAX_TOOL_CACHE_SIZE,
            "cache_hits": state.tool_cache_hits,
            "cache_misses": state.tool_cache_misses,
            "hit_ratio": state.tool_cache_hits / (state.tool_cache_hits + state.tool_cache_misses)
            if (state.tool_cache_hits + state.tool_cache_misses) > 0
            else 0,
            "cached_tool_hashes": cached_tool_hashes,
            "tool_count_per_hash": tool_count_per_hash,
        }
    )


@admin_bp.route("/v1/tools/cache/clear", methods=["POST"])
@admin_bp.route("/v2/tools/cache/clear", methods=["POST"])
def clear_tool_cache():
    """Clear the tool cache."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = check_config_auth()
    if auth_result:
        return auth_result

    with tool_cache_lock:
        cleared_count = len(tool_cache)
        tool_cache.clear()
        state.tool_cache_size = 0
        state.tool_cache_hits = 0
        state.tool_cache_misses = 0

    return jsonify(
        {
            "message": f"Cleared {cleared_count} tool definitions from cache",
            "cleared_count": cleared_count,
        }
    )


@admin_bp.route("/v1/tools/cache/<tool_hash>", methods=["GET"])
@admin_bp.route("/v2/tools/cache/<tool_hash>", methods=["GET"])
def get_cached_tools(tool_hash):
    """Get cached tool definitions by hash."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    with tool_cache_lock:
        if tool_hash in tool_cache:
            return jsonify(
                {
                    "hash": tool_hash,
                    "tools": tool_cache[tool_hash],
                    "tool_count": len(tool_cache[tool_hash]),
                }
            )
        else:
            return jsonify({"error": "Tool hash not found in cache"}), 404


def require_debug_auth():
    """Require authentication for debug endpoints.

    FAIL-CLOSED: If no auth configured (CONFIG_AUTH_PASSWORD is empty/unset),
    debug endpoints return 403 Forbidden.

    Only explicitly configured deployments can access debug data.
    """
    import base64

    password = os.environ.get("CONFIG_AUTH_PASSWORD", "")
    if not password:
        return jsonify({"error": "Debug API disabled: CONFIG_AUTH_PASSWORD not set"}), 403

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return jsonify({"error": "Authorization required"}), 401

    # Timing-safe comparison
    expected = base64.b64encode(f"admin:{password}".encode()).decode()
    provided = auth_header[6:]  # strip "Basic "
    if not hmac.compare_digest(expected, provided):
        return jsonify({"error": "Invalid credentials"}), 401

    # Auth passed
    return None


@admin_bp.route("/v1/debug/agents", methods=["GET"])
@admin_bp.route("/v2/debug/agents", methods=["GET"])
def debug_list_agents():
    """List all active agents."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = require_debug_auth()
    if auth_result is not None:
        return auth_result

    try:
        from proxy.agent.agent_registry import registry

        agents = []
        with registry._lock:
            for agent in registry._agents.values():
                agents.append(
                    {
                        "agent_id": agent.agent_id,
                        "session_id": agent.session_id,
                        "parent_id": agent.parent.agent_id if agent.parent else None,
                        "is_ticket_mode": agent.is_ticket_mode,
                        "message_count": len(agent.messages),
                    }
                )

        return jsonify({"agents": agents, "count": len(agents)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/v1/debug/agents/<agent_id>", methods=["GET"])
@admin_bp.route("/v2/debug/agents/<agent_id>", methods=["GET"])
def debug_get_agent(agent_id):
    """Get agent details including tree position."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = require_debug_auth()
    if auth_result is not None:
        return auth_result

    try:
        from proxy.agent.agent_registry import registry

        tree = registry.get_tree(agent_id)
        if "error" in tree:
            return jsonify(tree), 404

        return jsonify(tree)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/v1/debug/agents/<agent_id>/history", methods=["GET"])
@admin_bp.route("/v2/debug/agents/<agent_id>/history", methods=["GET"])
def debug_get_agent_history(agent_id):
    """Get message history for an agent."""
    rl_result = admin_rate_limiter.check(_get_client_ip())
    if not rl_result.allowed:
        body, status = rate_limited_response(int(rl_result.retry_after))
        return jsonify(body), status
    auth_result = require_debug_auth()
    if auth_result is not None:
        return auth_result

    try:
        from proxy.agent.agent_registry import registry

        agent = registry.get_or_load(agent_id)
        if agent is None:
            return jsonify({"error": "Agent not found"}), 404

        messages = agent.messages

        # Include provenance if available
        result = []
        for msg in messages:
            msg_copy = dict(msg)
            prov = msg.get("_provenance")
            if prov:
                msg_copy["_provenance"] = prov
            result.append(msg_copy)

        return jsonify(
            {
                "agent_id": agent_id,
                "message_count": len(result),
                "messages": result,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
