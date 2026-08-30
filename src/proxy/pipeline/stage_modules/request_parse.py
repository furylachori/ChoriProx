"""parse_request — Flask request parsing → RequestContext (runs before the pipeline)."""

import logging
import os
import uuid

from proxy.pipeline.exceptions import StageAbort
from proxy.types import InvalidRequestError, RequestContext

logger = logging.getLogger("proxy.pipeline.stages")

_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
_VALID_CONTENT_PART_TYPES = frozenset({"text", "image_url"})


def parse_request(flask_request) -> RequestContext:
    """Parse and validate incoming Flask request. Returns RequestContext.

    Raises ValueError on validation failure (caught by caller, returns 400).
    """
    body = flask_request.get_json(silent=True)
    if body is None:
        raise InvalidRequestError(
            param="body",
            code="invalid_body",
            message="Request body must be a JSON object",
        )
    if not isinstance(body, dict):
        raise InvalidRequestError(
            param="body",
            code="invalid_body",
            message="Request body must be a JSON object",
        )

    # DEBUG: log the first user message content (truncated) to verify whether
    # the canary text is actually present in the sub-agent's request body in
    # production. Critical for tracing the sub-agent canary detection path.
    try:
        from proxy.utils import extract_text_content as _dbg_extract
        _first_user_content = ""
        for _m in body.get("messages", []) or []:
            if isinstance(_m, dict) and _m.get("role") == "user":
                _first_user_content = _dbg_extract(_m.get("content", ""))
                break
        logger.debug(
            "parse_request: first user message (truncated 200) = %r",
            _first_user_content[:200] if isinstance(_first_user_content, str) else _first_user_content,
        )
    except Exception as _dbg_exc:
        logger.debug("parse_request: failed to log first user message: %s", _dbg_exc)

    model_name = body.get("model", "")
    if not isinstance(model_name, str):
        raise InvalidRequestError(
            param="model",
            code="invalid_model",
            message="Model must be a string",
        )

    if "/" in model_name:
        model_name = model_name.rsplit("/", 1)[-1]
    model_name = model_name.replace(" ", "_")

    wants_stream = body.get("stream", False)

    client_hint = flask_request.headers.get(
        "X-Session-Client", flask_request.remote_addr or "unknown"
    )

    from proxy.model_registry import resolve_session_id
    session_id = resolve_session_id(body, client_hint)

    # DEBUG: log caller_session_id (and note that _agent_meta extraction was
    # removed in refactor da6202d — only the structured _agent_metadata path
    # is gone; the text canary is the production signal).
    logger.debug("parse_request: caller_session_id=%r", session_id)
    logger.debug("parse_request: _agent_meta is no longer extracted (text canary is the in-band signal)")

    from proxy.session_tokens import validate_session_token
    _session_token = flask_request.headers.get("X-Session-Token") or body.get("_session_token", "")
    _secret_configured = bool(os.environ.get("PROXY_SESSION_SECRET"))

    _restored_session_id = None
    if _secret_configured:
        if _session_token:
            _token_payload = validate_session_token(_session_token)
            if not _token_payload:
                logger.error("Invalid session token from client=%s", client_hint)
                raise StageAbort(({"error": "invalid_session_token", "message": "Session token is invalid or expired"}, 401))
            if _token_payload.get("client_id") != client_hint:
                logger.error("Session token client_id mismatch: token_client=%s, hint=%s",
                              _token_payload.get("client_id"), client_hint)
                raise StageAbort(({"error": "client_mismatch", "message": "Session token client_id does not match"}, 403))
            _restored_session_id = _token_payload.get("session_id")
        else:
            logger.info("No session token provided — using content-derived session_id for bootstrapping (client=%s)", client_hint)
    else:
        if _session_token:
            logger.warning("Session token provided but PROXY_SESSION_SECRET not configured — ignoring token (client=%s)", client_hint)

    _validation_errors: list[tuple] = []

    if "stream" in body and not isinstance(body["stream"], bool):
        _validation_errors.append(
            ("stream", "invalid_stream", "stream must be a boolean")
        )

    _raw_messages = body.get("messages", None)
    if not isinstance(_raw_messages, list):
        _validation_errors.append(
            ("messages", "invalid_messages", "messages must be a list")
        )
        _raw_messages = []
    elif len(_raw_messages) == 0:
        _validation_errors.append(
            ("messages", "invalid_messages", "messages must be a non-empty list")
        )
    else:
        for _idx, _msg in enumerate(_raw_messages):
            _path = f"messages[{_idx}]"
            if not isinstance(_msg, dict):
                _validation_errors.append(
                    (_path, "invalid_message", f"{_path} must be an object")
                )
                continue
            _role = _msg.get("role")
            if not isinstance(_role, str):
                _validation_errors.append(
                    (f"{_path}.role", "invalid_role", f"{_path}.role must be a string")
                )
            elif _role not in _VALID_ROLES:
                _validation_errors.append(
                    (f"{_path}.role", "invalid_role_value",
                     f"{_path}.role must be one of system/user/assistant/tool (got {_role!r})")
                )

            if "content" in _msg:
                _content = _msg["content"]
                if _content is not None and not isinstance(_content, str):
                    if isinstance(_content, list):
                        for _part_idx, _part in enumerate(_content):
                            if not isinstance(_part, dict) or not isinstance(_part.get("type"), str):
                                _validation_errors.append(
                                    (f"{_path}.content[{_part_idx}]", "invalid_content",
                                     f"{_path}.content[{_part_idx}] must be an object with a string type")
                                )
                            elif _part.get("type") not in _VALID_CONTENT_PART_TYPES:
                                _validation_errors.append(
                                    (f"{_path}.content[{_part_idx}]", "invalid_content",
                                     f"{_path}.content[{_part_idx}].type must be one of text/image_url "
                                     f"(got {_part.get('type')!r})")
                                )
                    else:
                        _validation_errors.append(
                            (f"{_path}.content", "invalid_content",
                             f"{_path}.content must be a string, null, or a list of content parts")
                        )

            if "tool_calls" in _msg and not isinstance(_msg["tool_calls"], list):
                _validation_errors.append(
                    (f"{_path}.tool_calls", "invalid_tool_calls",
                     f"{_path}.tool_calls must be a list")
                )

            if _role == "tool":
                _tcid = _msg.get("tool_call_id")
                if not isinstance(_tcid, str) or not _tcid:
                    _validation_errors.append(
                        (f"{_path}.tool_call_id", "missing_tool_call_id",
                         f"{_path} (role=tool) requires a non-empty tool_call_id")
                    )

    if _validation_errors:
        _first_param = _validation_errors[0][0]
        _first_code = _validation_errors[0][1]
        _summary = "; ".join(f"{e[0]}: {e[2]}" for e in _validation_errors[:3])
        if len(_validation_errors) > 3:
            _summary += f" (+{len(_validation_errors) - 3} more)"
        raise InvalidRequestError(
            param=_first_param,
            code=_first_code,
            message=_summary,
        )

    messages = _raw_messages

    from proxy.canary import detect_subagent_ingress

    # session_id is used as a cache-scope key (§5.4) — NOT bound to HMAC
    # (BLOCKER fix: parent and sub-agent have different session_ids by design).
    ingress = detect_subagent_ingress(body, session_id=session_id)
    is_subagent = ingress.is_subagent
    _ticket_mode_subagent = ingress.subagent_ticket_mode
    ctx_inherited_skill = ingress.inherited_skill
    ctx_inherited_task_hash = ingress.task_hash
    _workspace_name = ingress.workspace_name

    # CP2A: restore the continuity session_id embedded in the (HMAC-verified)
    # session token. Gated on `not is_subagent` so a sub-agent that echoes the
    # parent's token does NOT collapse onto the parent's cache/history scope
    # (parent and sub-agent have different session_ids by design — see
    # detect_subagent_ingress docstring).
    if _restored_session_id and not is_subagent:
        session_id = _restored_session_id

    # DEBUG: log the sub-agent ingress fields right after detection.
    logger.debug(
        "parse_request: detect_subagent_ingress returned: is_subagent=%s, subagent_ticket_mode=%s, inherited_skill=%r, workspace_name=%r, task_hash=%r",
        is_subagent, _ticket_mode_subagent, ctx_inherited_skill, _workspace_name, ctx_inherited_task_hash,
    )

    # Determine unattended mode: API/fire-and-forget clients can set this explicitly,
    # otherwise default to False (interactive assumption for safety).
    # Sub-agents are always unattended (they have no human partner).
    unattended = body.get("unattended", False)
    if is_subagent:
        unattended = True

    ctx = RequestContext(
        body, model_name, wants_stream, session_id, client_hint,
        messages, flask_request.remote_addr or "", is_subagent,
        unattended=unattended,
        subagent_ticket_mode=_ticket_mode_subagent,
        inherited_skill=ctx_inherited_skill,
        workspace_name=_workspace_name,
        inherited_task_hash=ctx_inherited_task_hash,
    )
    ctx.request_id = str(uuid.uuid4())[:12]
    from proxy.event_log import set_request_context
    set_request_context(ctx.request_id, ctx.session_id)

    # DEBUG: log the same fields from the constructed RequestContext.
    logger.debug(
        "parse_request: RequestContext constructed: is_subagent=%s, subagent_ticket_mode=%s, inherited_skill=%r, workspace_name=%r, request_id=%s",
        ctx.is_subagent, ctx.subagent_ticket_mode, ctx.inherited_skill, ctx.workspace_name, ctx.request_id,
    )

    return ctx
