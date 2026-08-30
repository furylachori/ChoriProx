"""Extracted tool interception loop for memory tools.

Extracted from qwen_proxy.py in Phase 3. Handles the re-query loop
where memory tools are intercepted and executed locally before
sending results back to the LLM.
"""

import copy
import json
import re
import threading
import time
import logging
import uuid
import requests
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from proxy.utils import safe_content as _safe_content

logger = logging.getLogger(__name__)


TERMINATION_MODEL_COMPLETED = "model_completed"
TERMINATION_ITERATION_LIMIT = "iteration_limit"
TERMINATION_TIME_LIMIT = "time_limit"
TERMINATION_REQUERY_FAILED = "requery_failed"
TERMINATION_PASSTHROUGH_REQUIRED = "passthrough_required"
TERMINATION_INTERNAL_ERROR = "internal_error"
TERMINATION_VALIDATION_RETRY_EXHAUSTED = "validation_retry_exhausted"
TERMINATION_AUTHORIZATION_DENIED = "authorization_denied"

_BRACKET_LEDGER_RE = re.compile(r"^\s*\[[^\]\n]+:[^\]\n]+\]")

_LEDGER_MAX_CHARS = 2000
_LEDGER_MAX_RECORDS = 12
_LEDGER_RECORD_LINE_MAX = 180
_FALLBACK_MAX_CHARS = 2000


class ToolCallGuardException(Exception):
    pass


def _cleanup_orphaned_tool_results(messages: list) -> list:
    valid_tool_call_ids = set()
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tc in msg["tool_calls"]:
                if tc.get("id"):
                    valid_tool_call_ids.add(tc["id"])

    present_result_ids = set()
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            present_result_ids.add(msg.get("tool_call_id"))

    cleaned = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") not in valid_tool_call_ids:
            continue
        cleaned.append(msg)

    final = []
    for msg in cleaned:
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            surviving = [
                tc for tc in msg["tool_calls"]
                if tc.get("id") in present_result_ids
            ]
            if len(surviving) < len(msg["tool_calls"]):
                if surviving:
                    new_msg = dict(msg)
                    new_msg["tool_calls"] = surviving
                    final.append(new_msg)
                else:
                    new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    final.append(new_msg)
            else:
                final.append(msg)
        else:
            final.append(msg)
    return final


def _trim_working_messages_by_tokens(
    working_messages: list,
    tokens_per_char: float,
    max_tokens: int,
) -> list:
    from proxy.context import calculate_tokens_from_text

    system_msgs = [m for m in working_messages if m.get("role") == "system"]
    non_system = [m for m in working_messages if m.get("role") != "system"]
    if not non_system:
        return system_msgs

    last_user_idx = None
    for i, m in enumerate(non_system):
        if m.get("role") == "user":
            last_user_idx = i
    budget = max_tokens
    if last_user_idx is not None:
        last_user_msg = non_system[last_user_idx]
        last_user_tokens = calculate_tokens_from_text(
            json.dumps(last_user_msg.get("content", ""), default=str),
            tokens_per_char,
        )
        budget -= last_user_tokens

    kept = []
    for rev_idx, msg in enumerate(reversed(non_system)):
        original_idx = len(non_system) - 1 - rev_idx
        if original_idx == last_user_idx:
            kept.insert(0, msg)
            continue
        if msg.get("_sub_agent_return"):
            kept.insert(0, msg)
            continue
        msg_tokens = calculate_tokens_from_text(
            json.dumps(msg.get("content", ""), default=str),
            tokens_per_char,
        )
        if budget - msg_tokens > 0:
            kept.insert(0, msg)
            budget -= msg_tokens

    return system_msgs + kept


def _enforce_working_messages_count_cap(
    working_messages: list,
    max_messages: int = 69,
) -> list:
    system_msgs = [m for m in working_messages if m.get("role") == "system"]
    non_system = [m for m in working_messages if m.get("role") != "system"]
    last_user = None
    for i in range(len(non_system) - 1, -1, -1):
        if non_system[i].get("role") == "user":
            last_user = non_system.pop(i)
            break
    sentinels = [m for m in non_system if m.get("_sub_agent_return")]
    non_sentinels = [m for m in non_system if not m.get("_sub_agent_return")]
    reserved = len(system_msgs) + (1 if last_user else 0) + len(sentinels)
    keep_count = max(1, max_messages - reserved)
    recent = non_sentinels[-keep_count:]
    if last_user and last_user not in recent:
        recent.append(last_user)
    return system_msgs + recent + sentinels


def _record_canonical_envelope(
    result: Any,
    tool_name: str,
    tool_args: Optional[dict],
    tool_call_id: Optional[str],
    iteration: int,
    attempt: int,
) -> dict:
    """Build a canonical execution record from a tool's raw return.

    The raw return is one of:
      - canonical action envelope (has "ok" or "status") — preserved
        with its declared operation/status/error/data fields
      - dict with top-level "error" key — canonicalised to error
      - legacy success dict (results/total_matches) — recorded as
        success with data=result so the closure ledger sees the counts
      - string/None/legacy error — recorded as error with an actionable
        message; the operation is derived from tool_args.action when
        present, otherwise the tool name.

    Always returns a dict with tool, operation, tool_call_id, iteration,
    attempt, status, error, data — preserving actionable detail so the
    deterministic-fallback renderer and the compact ledger see it.
    """
    op: Optional[str] = None
    status: str = "error"
    err_obj: Optional[dict] = None
    data_obj: Optional[dict] = None

    if isinstance(result, dict):
        if "operation" in result and isinstance(result.get("operation"), str):
            op = result["operation"]
        elif isinstance(tool_args, dict) and isinstance(tool_args.get("action"), str):
            op = tool_args["action"]
        else:
            from memory.tools.dispatch import TICKET_TOOL_OPERATIONS
            op = TICKET_TOOL_OPERATIONS.get(tool_name, tool_name)

        if "status" in result and isinstance(result.get("status"), str):
            status = result["status"]
        elif result.get("ok") is True:
            status = "success"
        elif result.get("ok") is False:
            status = "error"
        elif "error" in result:
            status = "error"
        else:
            status = "success"

        raw_err = result.get("error")
        if isinstance(raw_err, dict):
            err_obj = raw_err
        elif isinstance(raw_err, str) and raw_err:
            err_obj = {
                "code": "tool_error",
                "message": raw_err[:200],
                "retryable": False,
                "category": "validation",
            }
        elif status == "error":
            msg = result.get("message") or result.get("error_message")
            if not isinstance(msg, str):
                msg = str(result)[:200] if result else "tool reported no detail"
            err_obj = {
                "code": result.get("code") or "tool_error",
                "message": msg[:200],
                "retryable": bool(result.get("retryable", False)),
                "category": result.get("category") or "internal",
            }

        data_obj = result.get("data")
        if data_obj is None and status == "success" and err_obj is None:
            data_obj = result
    else:
        from memory.tools.dispatch import TICKET_TOOL_OPERATIONS
        op = TICKET_TOOL_OPERATIONS.get(tool_name, tool_name)
        err_obj = {
            "code": "tool_execution_failed",
            "message": (str(result) if result is not None else "no detail")[:200],
            "retryable": False,
            "category": "internal",
        }
        status = "error"

    return {
        "tool": tool_name,
        "operation": op,
        "tool_call_id": tool_call_id,
        "iteration": iteration,
        "attempt": attempt,
        "status": status,
        "error": err_obj,
        "data": data_obj,
    }


def _is_empty_memory_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    status = result.get("status")
    if status in ("success", "partial"):
        data = result.get("data")
        if not isinstance(data, dict):
            return True
        if data.get("total_matches", 0) == 0 and "total_matches" in data:
            return True
        if "results" in data and len(data.get("results", [])) == 0:
            return True
        return False
    if status == "error":
        return True
    if isinstance(result.get("error"), dict):
        return True
    if result.get("error"):
        return True
    if "total_matches" in result:
        return result.get("total_matches", 0) == 0
    if "facts" in result:
        return (
            len(result.get("facts", [])) == 0
            and len(result.get("events", [])) == 0
            and len(result.get("instructions", [])) == 0
        )
    if "messages" in result:
        return len(result.get("messages", [])) == 0
    if "results" in result:
        return len(result.get("results", [])) == 0
    return False


CONTEXT_TOKEN_LIMIT = 26000
try:
    import proxy.state as _state
    USABLE_CONTEXT_RATIO = getattr(_state, 'CONTEXT_BUDGET_RATIO', 0.75)
except ImportError:
    USABLE_CONTEXT_RATIO = 0.75


@dataclass
class ToolLoopResult:
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    finish_reason: str = "stop"
    messages_for_history: list = field(default_factory=list)
    execution_records: list = field(default_factory=list)
    needs_closure: bool = False
    termination_reason: str = TERMINATION_MODEL_COMPLETED
    closure_source: Optional[str] = None


def _is_bracket_ledger_content(content: Optional[str]) -> bool:
    if not content or not isinstance(content, str):
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return bool(_BRACKET_LEDGER_RE.match(stripped))
    return False


def _truncate_for_ledger(text: str, limit: int = _LEDGER_RECORD_LINE_MAX) -> str:
    text = str(text) if text is not None else ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_ledger_record(record: Any) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    tool = record.get("tool") or "unknown"
    operation = record.get("operation") or "unknown"
    status = record.get("status") or "unknown"
    attempt = record.get("attempt")
    iteration = record.get("iteration")
    line_parts = [f"[{tool}:{operation}:{status}]"]
    if isinstance(attempt, int):
        line_parts.append(f"attempt={attempt}")
    if isinstance(iteration, int):
        line_parts.append(f"iter={iteration}")
    base = " ".join(line_parts)

    detail_bits = []
    err = record.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        message = err.get("message")
        if code:
            detail_bits.append(f"code={code}")
        if message:
            detail_bits.append(_truncate_for_ledger(message, 140))

    data = record.get("data")
    if isinstance(data, dict):
        created = data.get("created")
        if isinstance(created, list):
            for entry in created[:3]:
                if not isinstance(entry, dict):
                    continue
                short_hash = entry.get("short_hash")
                priority = entry.get("priority")
                name = entry.get("name")
                tag_bits = []
                if short_hash:
                    tag_bits.append(f"hash={short_hash}")
                if priority:
                    tag_bits.append(f"priority={priority}")
                if name:
                    tag_bits.append(f"name={_truncate_for_ledger(name, 60)}")
                if tag_bits:
                    detail_bits.append("created(" + ", ".join(tag_bits) + ")")
    if detail_bits:
        return _truncate_for_ledger(base + " " + "; ".join(detail_bits))
    return _truncate_for_ledger(base)


def build_compact_execution_ledger(
    execution_records: Any,
    max_chars: int = _LEDGER_MAX_CHARS,
    max_records: int = _LEDGER_MAX_RECORDS,
) -> str:
    if not execution_records or not isinstance(execution_records, list):
        return "(no operations recorded)"

    total = len(execution_records)
    window = execution_records[-max_records:] if total > max_records else list(execution_records)

    lines = []
    for rec in window:
        formatted = _format_ledger_record(rec)
        if formatted:
            lines.append(formatted)
    if not lines:
        return "(no operations recorded)"

    header = f"Recorded {total} operation(s); showing last {len(lines)}."
    body = "\n".join(lines)
    out = f"{header}\n{body}"
    if len(out) > max_chars:
        out = out[: max(0, max_chars - 1)].rstrip() + "…"
    return out


def _classify_outcome(records: Any) -> str:
    if not records or not isinstance(records, list):
        return "FAILED"

    confirmed_success = 0
    confirmed_error = 0
    confirmed_partial = 0
    confirmed_not_executed = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        status = rec.get("status")
        if status == "success":
            confirmed_success += 1
        elif status == "error":
            confirmed_error += 1
        elif status == "partial":
            confirmed_partial += 1
        elif status == "not_executed":
            confirmed_not_executed += 1

    if (
        confirmed_error == 0
        and confirmed_partial == 0
        and confirmed_not_executed == 0
        and confirmed_success > 0
    ):
        return "COMPLETE"
    if confirmed_error == 0 and confirmed_success == 0 and confirmed_partial == 0:
        return "FAILED"
    if confirmed_error > 0 and confirmed_success == 0 and confirmed_partial == 0:
        return "FAILED"
    if confirmed_not_executed > 0 and confirmed_success > 0:
        return "PARTIAL"
    if confirmed_error > 0 and confirmed_success > 0:
        return "PARTIAL"
    if confirmed_partial > 0:
        return "PARTIAL"
    return "PARTIAL"


def _safe_short(s: Any, limit: int) -> str:
    s = "" if s is None else str(s)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def _confirmed_ops_prose(records: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(records, list):
        return out
    for rec in records:
        if not isinstance(rec, dict):
            continue
        status = rec.get("status")
        if status not in ("success", "partial"):
            continue
        tool = rec.get("tool") or "unknown"
        operation = rec.get("operation") or "unknown"
        data = rec.get("data")
        if not isinstance(data, dict):
            out.append(
                f"- {tool} {operation} (status={status})"
            )
            continue
        created = data.get("created")
        if isinstance(created, list) and created:
            for entry in created[:5]:
                if not isinstance(entry, dict):
                    continue
                bits = []
                if entry.get("short_hash"):
                    bits.append(f"hash={entry['short_hash']}")
                if entry.get("priority"):
                    bits.append(f"priority={entry['priority']}")
                if entry.get("name"):
                    bits.append(f"name={_safe_short(entry['name'], 60)}")
                if bits:
                    out.append(
                        f"- created {tool}:{operation} (" + ", ".join(bits) + ")"
                    )
            continue
        results = data.get("results")
        if isinstance(results, list) and results:
            count = len(results)
            out.append(
                f"- {tool} {operation} returned {count} result(s)"
            )
            continue
        total_matches = data.get("total_matches")
        if isinstance(total_matches, int):
            out.append(
                f"- {tool} {operation} returned {total_matches} match(es)"
            )
            continue
        out.append(f"- {tool} {operation} (status={status})")
    return out


def _failure_prose(records: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(records, list):
        return out
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "error":
            continue
        tool = rec.get("tool") or "unknown"
        operation = rec.get("operation") or "unknown"
        attempt = rec.get("attempt")
        err = rec.get("error")
        code = err.get("code") if isinstance(err, dict) else None
        message = err.get("message") if isinstance(err, dict) else None
        head = f"- {tool} {operation} failed"
        if isinstance(attempt, int):
            head += f" (attempt {attempt})"
        if code:
            head += f" code={code}"
        out.append(head)
        if message:
            out.append(f"  - reason: {_safe_short(message, 200)}")
    return out


def _not_executed_prose(records: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(records, list):
        return out
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "not_executed":
            continue
        tool = rec.get("tool") or "unknown"
        operation = rec.get("operation") or "unknown"
        err = rec.get("error")
        code = err.get("code") if isinstance(err, dict) else None
        message = err.get("message") if isinstance(err, dict) else None
        if code and ("retry" in code or "no_retry" in code):
            reason = _safe_short(message, 140) if message else code
            out.append(
                f"- {tool} {operation} not executed ({reason})"
            )
        elif code == "iteration_limit":
            out.append(
                f"- {tool} {operation} not executed (iteration limit reached)"
            )
        else:
            out.append(f"- {tool} {operation} not executed")
    return out


def _truncate_report(prefix: str, body_lines: list[str], max_chars: int) -> tuple[str, int]:
    body = "" if not body_lines else ("\n" + "\n".join(body_lines))
    full = prefix + body
    if len(full) <= max_chars:
        return full, 0
    omitted_marker = " (omitted)"
    suffix = "…"
    keep_budget = max_chars - len(prefix) - len(suffix) - len(omitted_marker) - 2
    if keep_budget < 0:
        return (prefix[:max_chars] if len(prefix) > max_chars else prefix), 0
    kept: list[str] = []
    used = 0
    for line in body_lines:
        line_len = len(line) + 1
        if used + line_len > keep_budget:
            break
        kept.append(line)
        used += line_len
    omitted = len(body_lines) - len(kept)
    out = prefix + ("\n" + "\n".join(kept) if kept else "")
    if omitted > 0:
        out += f"{omitted_marker}{suffix}"
    if len(out) > max_chars:
        out = out[:max_chars - len(suffix)] + suffix
    return out, omitted


def render_deterministic_fallback(
    execution_records: Any,
    termination_reason: str,
) -> str:
    outcome = _classify_outcome(execution_records)
    confirmed = _confirmed_ops_prose(execution_records)
    not_executed = _not_executed_prose(execution_records)
    failures = _failure_prose(execution_records)
    body_lines: list[str] = []
    if confirmed:
        body_lines.append("Confirmed operations:")
        body_lines.extend(confirmed)
    if not_executed:
        if body_lines:
            body_lines.append("")
        body_lines.append("Not executed:")
        body_lines.extend(not_executed)
    if failures:
        if body_lines:
            body_lines.append("")
        body_lines.append("Errors:")
        body_lines.extend(failures)
    if not body_lines:
        body_lines.append(
            "No confirmed operations were recorded before the tool loop terminated."
        )
    body_lines.append(
        f"Termination: {termination_reason or TERMINATION_INTERNAL_ERROR}."
    )
    full, _omitted = _truncate_report(
        f"{outcome} Report:",
        body_lines,
        _FALLBACK_MAX_CHARS,
    )
    return full


def _sanitize_closure_request_messages(messages: list) -> list:
    """Deep-copy + strip dangling tool_calls so the closure request is
    protocol-valid. When an assistant message has substantive content
    (text or reasoning) but ALL its tool_calls are dangling, the message
    is kept with ``tool_calls`` removed so its text is preserved.
    """
    if not isinstance(messages, list) or not messages:
        return []

    sanitized: list = []
    for m in messages:
        if isinstance(m, dict):
            sanitized.append(copy.deepcopy(m))

    valid_tool_call_ids: set = set()
    for msg in sanitized:
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict) and tc.get("id"):
                    valid_tool_call_ids.add(tc["id"])

    present_result_ids = {
        m.get("tool_call_id") for m in sanitized
        if m.get("role") == "tool" and m.get("tool_call_id")
    }

    def _has_substantive_content(msg):
        text = msg.get("content")
        if isinstance(text, str) and text.strip():
            return True
        if msg.get("reasoning_content") or msg.get("reasoning"):
            return True
        return False

    out: list = []
    for msg in sanitized:
        if msg.get("role") == "tool" and msg.get("tool_call_id") not in valid_tool_call_ids:
            continue
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            surviving = [
                tc for tc in msg["tool_calls"]
                if isinstance(tc, dict) and tc.get("id") in present_result_ids
            ]
            new_msg = copy.deepcopy(msg)
            if surviving:
                new_msg["tool_calls"] = surviving
                out.append(new_msg)
            elif _has_substantive_content(msg):
                new_msg.pop("tool_calls", None)
                out.append(new_msg)
            continue
        out.append(msg)
    return out


def _build_sanitized_history_view(working_messages: list, original_len: int) -> list:
    """Deep-copy the working messages and sanitize before persisting
    into chat history. Strips dangling unexecuted assistant tool_calls
    (those without paired tool results) and preserves safe substantive
    text content. The closure-private user instruction is never added
    by this function — callers must not append the instruction message
    before calling.
    """
    if not isinstance(working_messages, list):
        return []
    return _sanitize_closure_request_messages(working_messages[original_len:])


def _call_closure(
    working_messages: list,
    llama_url: str,
    body: dict,
    llama_body: dict,
    model_cfg: Any,
    auth_headers: Optional[dict],
    ledger: str,
    termination_reason: str,
    max_total_time: float,
    tool_loop_start: float,
    timeout_cap: float = 30.0,
    parse_tools_fn=None,
) -> Optional[str]:
    from proxy.requery import closure_finalize

    try:
        outcome = closure_finalize(
            working_messages=working_messages,
            llama_url=llama_url,
            body=body,
            llama_body=llama_body,
            model_cfg=model_cfg,
            auth_headers=auth_headers,
            ledger=ledger,
            termination_reason=termination_reason,
            timeout_cap=timeout_cap,
            tool_loop_start=tool_loop_start,
            max_total_time=max_total_time,
            parse_tools_fn=parse_tools_fn,
        )
    except Exception as e:
        logger.warning("[TOOL LOOP] closure_finalize raised: %s", e)
        return None

    if outcome is None or not getattr(outcome, "success", False):
        return None
    content = getattr(outcome, "content", None)
    return content if isinstance(content, str) else None


def _resolve_closure(
    candidate: ToolLoopResult,
    *,
    working_messages: list,
    llama_url: str,
    body: dict,
    llama_body: dict,
    model_cfg: Any,
    auth_headers: Optional[dict],
    tool_loop_start: float,
    max_total_time: float,
    termination_reason: str,
    parse_tools_fn=None,
) -> ToolLoopResult:
    if not candidate.needs_closure:
        return candidate

    ledger = build_compact_execution_ledger(candidate.execution_records)
    safe_msgs = _sanitize_closure_request_messages(working_messages)

    closure_attempted = _call_closure(
        working_messages=safe_msgs,
        llama_url=llama_url,
        body=body,
        llama_body=llama_body,
        model_cfg=model_cfg,
        auth_headers=auth_headers,
        ledger=ledger,
        termination_reason=termination_reason,
        max_total_time=max_total_time,
        tool_loop_start=tool_loop_start,
        parse_tools_fn=parse_tools_fn,
    )

    if closure_attempted:
        return ToolLoopResult(
            content=closure_attempted,
            tool_calls=None,
            finish_reason="stop",
            messages_for_history=candidate.messages_for_history,
            execution_records=candidate.execution_records,
            needs_closure=False,
            termination_reason=termination_reason,
            closure_source="llm",
        )

    try:
        final_content = render_deterministic_fallback(
            candidate.execution_records, termination_reason
        )
    except Exception:
        final_content = (
            f"FAILED The tool loop terminated before a completion "
            "report could be rendered."
        )
    if not final_content.startswith(("COMPLETE", "PARTIAL", "FAILED")):
        final_content = "FAILED " + final_content

    return ToolLoopResult(
        content=final_content,
        tool_calls=None,
        finish_reason="stop",
        messages_for_history=candidate.messages_for_history,
        execution_records=candidate.execution_records,
        needs_closure=False,
        termination_reason=termination_reason,
        closure_source="fallback",
    )


def _clean_close_skip_eligible(
    content: Optional[str],
    termination_reason: str,
    new_tool_calls: Optional[list],
) -> bool:
    """Return True iff the loop ended on a clean happy path so the
    dedicated closure call is unnecessary.

    Per the user's Slice 3 mandate: closure is skipped ONLY when ALL are
    true:
      - final iteration had no tool calls
      - content is non-empty / substantive
      - content does not begin with the bracket-ledger pattern
      - termination_reason is model_completed
    """
    if termination_reason != TERMINATION_MODEL_COMPLETED:
        return False
    if new_tool_calls:
        return False
    if not content or not content.strip():
        return False
    if _is_bracket_ledger_content(content):
        return False
    return True


def _build_candidate_result(
    *,
    content: Optional[str],
    tool_calls: Optional[list],
    finish_reason: str,
    messages_for_history: list,
    execution_records: list,
    termination_reason: str,
    needs_closure: bool,
) -> ToolLoopResult:
    return ToolLoopResult(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        messages_for_history=messages_for_history,
        execution_records=execution_records,
        needs_closure=needs_closure,
        termination_reason=termination_reason,
    )


def run_memory_tool_loop(
    tool_calls_list: list,
    working_messages: list,
    tools: list,
    model: str,
    session_id: Optional[str],
    llama_url: str,
    body: dict,
    llama_body: dict,
    chat_histories: dict,
    chat_histories_lock: threading.Lock,
    client_hint: str = "",
    execute_fn: Optional[Callable] = None,
    make_tool_calls_fn: Optional[Callable] = None,
    parse_tools_fn: Optional[Callable] = None,
    http_session: requests.Session = None,
    max_iterations: int = 5,
    max_total_time: float = 900.0,
    max_result_size: int = 16 * 1024,
    max_calls_per_iteration: int = 50,
    auth_headers: Optional[dict] = None,
    model_cfg=None,
    existing_tool_ids: Optional[list] = None,
) -> ToolLoopResult:
    from proxy.tool_pipeline.tool_registry import registry as _tool_reg
    _memory_tool_names = _tool_reg.get_memory_tool_names()

    if execute_fn is None:
        from memory.tools import execute_memory_tool
        execute_fn = execute_memory_tool
    if make_tool_calls_fn is None:
        from proxy.tool_parsing import make_tool_calls as _default_mtc
        import functools as _functools
        from proxy.tool_parsing import build_valid_params_by_tool as _bvp
        _vpbt = _bvp(tools)
        make_tool_calls_fn = _functools.partial(_default_mtc, valid_params_by_tool=_vpbt)
    if parse_tools_fn is None:
        from proxy.tool_parsing.parsers import (
            parse_all_tools_from_content as parse_tools_fn,
        )

    from proxy.tool_pipeline.tool_limits import SemanticRetryTracker
    retry_tracker = SemanticRetryTracker()

    working_messages = copy.deepcopy(list(working_messages))

    execution_records: list = []

    has_memory_tools = any(
        tc.get("name") in _memory_tool_names for tc in tool_calls_list
    )
    if not has_memory_tools:
        return _build_candidate_result(
            content=None,
            tool_calls=make_tool_calls_fn(tool_calls_list, existing_ids=existing_tool_ids)
            if make_tool_calls_fn
            else None,
            finish_reason="tool_calls" if tool_calls_list else "stop",
            messages_for_history=[],
            execution_records=execution_records,
            termination_reason=(
                TERMINATION_PASSTHROUGH_REQUIRED if tool_calls_list
                else TERMINATION_MODEL_COMPLETED
            ),
            needs_closure=False,
        )

    content = None
    original_tool_calls = []
    for msg in reversed(working_messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            content = _safe_content(msg.get("content"))
            original_tool_calls = msg.get("tool_calls", [])
            break

    original_tool_calls_raw = []
    for tc in original_tool_calls:
        _tc_name = tc.get("function", {}).get("name")
        if not _tc_name:
            continue
        _tc_args = tc.get("function", {}).get("arguments", {})
        if isinstance(_tc_args, str):
            try:
                _tc_args = json.loads(_tc_args)
            except (json.JSONDecodeError, ValueError):
                _tc_args = {}
        original_tool_calls_raw.append({"name": _tc_name, "arguments": _tc_args})

    if content is None:
        content = ""

    _requery_failed = False
    raw_results = []
    _attempt_counts: dict = {}
    _raw_tool_id_map = {}
    requery_content = None
    iteration = 0
    tool_loop_start = time.time()
    final_tool_calls = None
    last_upstream_finish_reason = None
    termination_reason = TERMINATION_MODEL_COMPLETED
    saw_passthrough_path = False
    new_tool_calls: list = []
    _any_retry_rejected = False

    while iteration < max_iterations:
        elapsed = time.time() - tool_loop_start
        if elapsed > max_total_time:
            termination_reason = TERMINATION_TIME_LIMIT
            break

        if not tool_calls_list:
            break

        memory_calls = [
            tc for tc in tool_calls_list if tc.get("name") in _memory_tool_names
        ][:max_calls_per_iteration]
        passthrough_calls = [
            tc for tc in tool_calls_list if tc.get("name") not in _memory_tool_names
        ]

        if not memory_calls:
            openai_tool_calls = make_tool_calls_fn(tool_calls_list)
            if openai_tool_calls:
                final_tool_calls = openai_tool_calls
                saw_passthrough_path = True
            break

        pre_exec_len = len(working_messages)
        _any_retry_rejected = False
        try:
            memory_tool_results = []
            original_tc_id_map = {}
            if _raw_tool_id_map:
                original_tc_id_map = dict(_raw_tool_id_map)
            else:
                for otc in original_tool_calls:
                    otc_name = otc.get("function", {}).get("name") or otc.get("name")
                    otc_args = otc.get("function", {}).get("arguments") or otc.get(
                        "arguments", "{}"
                    )
                    if isinstance(otc_args, dict):
                        otc_args = json.dumps(otc_args, sort_keys=True)
                    key = (otc_name, otc_args)
                    if otc.get("id"):
                        original_tc_id_map[key] = otc.get("id")
            for tc in memory_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("arguments", {})
                if isinstance(tool_args, str):
                    _raw_args_str = tool_args
                    try:
                        tool_args = json.loads(tool_args)
                    except (json.JSONDecodeError, ValueError):
                        tool_call_id = original_tc_id_map.get((tool_name, _raw_args_str))
                        from proxy.tool_pipeline.action_envelope import error_envelope
                        _malformed_envelope = error_envelope(
                            tool=tool_name,
                            operation=None,
                            code="malformed_arguments",
                            message="proxy_ticket arguments must be a JSON object",
                            session_id=session_id or "",
                        )
                        raw_results.append(_malformed_envelope)
                        execution_records.append({
                            "tool": tool_name,
                            "operation": None,
                            "tool_call_id": tool_call_id,
                            "iteration": iteration + 1,
                            "attempt": 1,
                            "status": "error",
                            "error": _malformed_envelope.get("error"),
                            "data": None,
                        })
                        if tool_call_id:
                            _malformed_str = json.dumps(_malformed_envelope, default=str)
                            if len(_malformed_str) > max_result_size:
                                _malformed_str = _malformed_str[:max_result_size] + "\n[TRUNCATED]"
                            memory_tool_results.append({
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "content": _malformed_str,
                            })
                        else:
                            logger.warning(
                                "[TOOL LOOP] Skipping malformed-JSON result for %s — no matching tool_call_id",
                                tool_name,
                            )
                        _op_retry = tool_name
                        retry_tracker.record(tool_name, _op_retry, _malformed_envelope, {})
                        continue

                _pre_args_str = (
                    tool_args
                    if isinstance(tool_args, str)
                    else json.dumps(tool_args, sort_keys=True, default=str)
                )
                _pre_key = (tool_name, _pre_args_str)
                tool_call_id = original_tc_id_map.get(_pre_key)

                _op_retry = SemanticRetryTracker._derive_operation(tool_name, tool_args)
                rejection = retry_tracker.check(tool_name, _op_retry, tool_args)
                if rejection is not None:
                    rej_code, rej_category = rejection
                    from proxy.tool_pipeline.action_envelope import error_envelope as _rej_env_fn
                    _rejection_envelope = _rej_env_fn(
                        tool=tool_name,
                        operation=_op_retry,
                        code=rej_code,
                        message=f"Retry budget exhausted ({rej_category})",
                        session_id=session_id or "",
                    )
                    _any_retry_rejected = True
                    if rej_category == "validation":
                        termination_reason = TERMINATION_VALIDATION_RETRY_EXHAUSTED
                    elif rej_category == "authorization":
                        termination_reason = TERMINATION_AUTHORIZATION_DENIED
                    else:
                        termination_reason = TERMINATION_INTERNAL_ERROR
                    raw_results.append(_rejection_envelope)
                    execution_records.append({
                        "tool": tool_name,
                        "operation": _op_retry,
                        "tool_call_id": tool_call_id,
                        "iteration": iteration + 1,
                        "attempt": None,
                        "status": "not_executed",
                        "error": _rejection_envelope.get("error"),
                        "data": None,
                    })
                    _rej_str = (
                        json.dumps(_rejection_envelope, default=str)
                        if isinstance(_rejection_envelope, dict)
                        else str(_rejection_envelope)
                    )
                    if len(_rej_str) > max_result_size:
                        _rej_str = (
                            _rej_str[:max_result_size]
                            + f"\n[TRUNCATED: {len(_rej_str)} bytes exceeds {max_result_size} byte limit]"
                        )
                    if tool_call_id:
                        memory_tool_results.append(
                            {
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "content": _rej_str,
                            }
                        )
                    else:
                        logger.warning(
                            "[TOOL LOOP] Skipping rejected-retry result for %s — no matching tool_call_id in assistant message",
                            tool_name,
                        )
                    continue

                try:
                    agent_name = tool_name.split(":")[0].lower() if ":" in tool_name else tool_name.lower()
                    if agent_name in ("agent", "task", "ticket"):
                        raise ToolCallGuardException(
                            f"Tool call self-reference blocked: {tool_name}."
                        )
                    if (tool_name in ("search", "recall", "proxy_work") or tool_name.startswith("proxy_ticket")) and session_id:
                        if not tool_args.get("session_id"):
                            tool_args["session_id"] = session_id
                    if (tool_name == "proxy_work" or tool_name.startswith("proxy_ticket")) and client_hint:
                        if not tool_args.get("client_hint"):
                            tool_args["client_hint"] = client_hint
                    try:
                        result = execute_fn(tool_name, tool_args)
                        from memory.db import log_tool_call
                        log_tool_call(
                            session_id=session_id or "unknown",
                            tool_name=tool_name,
                            arguments=tool_args,
                            result_status=(
                                "error" if (isinstance(result, dict) and result.get("status") == "error")
                                else "partial" if (isinstance(result, dict) and result.get("status") == "partial")
                                else "success"
                            ),
                            result_summary=str(result)[:200],
                        )
                    except Exception as e:
                        from proxy.tool_pipeline.action_envelope import error_envelope
                        result = error_envelope(
                            tool=tool_name,
                            operation=None,
                            code="tool_execution_failed",
                            message=f"Tool execution failed: {type(e).__name__}: {str(e)[:200]}",
                            session_id=session_id or "",
                        )
                        from memory.db import log_tool_call
                        log_tool_call(
                            session_id=session_id or "unknown",
                            tool_name=tool_name,
                            arguments=tool_args,
                            result_status="error",
                            result_summary=str(result)[:200],
                        )
                except ToolCallGuardException as e:
                    from proxy.tool_pipeline.action_envelope import error_envelope
                    result = error_envelope(
                        tool=tool_name,
                        operation=None,
                        code="tool_execution_failed",
                        message=str(e),
                        session_id=session_id or "",
                    )

                raw_results.append(result)

                _op = result.get("operation") if isinstance(result, dict) else None
                _attempt_key = (tool_name, _op)
                _attempt_counts[_attempt_key] = _attempt_counts.get(_attempt_key, 0) + 1

                execution_records.append(
                    _record_canonical_envelope(
                        result=result,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_call_id=tool_call_id,
                        iteration=iteration + 1,
                        attempt=_attempt_counts[_attempt_key],
                    )
                )

                retry_tracker.record(tool_name, _op_retry, result, tool_args)

                result_str = (
                    json.dumps(result, default=str)
                    if isinstance(result, dict)
                    else str(result)
                )
                if len(result_str) > max_result_size:
                    truncated = result_str[:max_result_size]
                    result_str = (
                        truncated
                        + f"\n[TRUNCATED: {len(result_str)} bytes exceeds {max_result_size} byte limit]"
                    )

                if tool_call_id:
                    memory_tool_results.append(
                        {
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "content": result_str,
                        }
                    )
                else:
                    logger.warning(
                        "[TOOL LOOP] Preserving result for %s with synthetic tool_call_id — no matching ID in assistant message",
                        tool_name,
                    )
                    synthetic_id = f"synth_{tool_name}_{int(time.time() * 1_000_000)}"
                    synthetic_assistant = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": synthetic_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else "{}",
                            },
                        }],
                    }
                    working_messages.append(synthetic_assistant)
                    memory_tool_results.append(
                        {
                            "tool_call_id": synthetic_id,
                            "role": "tool",
                            "content": result_str,
                        }
                    )

            for result in memory_tool_results:
                working_messages.append(result)

            if _any_retry_rejected:
                break

            if len(working_messages) > 10:
                try:
                    from proxy.context import calculate_tokens_from_text, get_context_config

                    tokens_per_char = get_context_config()["tokens_per_char"]
                    total_tokens = sum(
                        calculate_tokens_from_text(
                            json.dumps(m.get("content", ""), default=str), tokens_per_char
                        )
                        for m in working_messages
                        if m.get("content")
                    )
                    max_tokens = int(CONTEXT_TOKEN_LIMIT * USABLE_CONTEXT_RATIO)
                    if total_tokens > max_tokens:
                        working_messages = _trim_working_messages_by_tokens(
                            working_messages, tokens_per_char, max_tokens
                        )
                        working_messages = _cleanup_orphaned_tool_results(working_messages)
                except ImportError:
                    pass

                if len(working_messages) > 69:
                    working_messages = _enforce_working_messages_count_cap(
                        working_messages, max_messages=69
                    )
                    working_messages = _cleanup_orphaned_tool_results(working_messages)

            working_messages = _cleanup_orphaned_tool_results(working_messages)

            if passthrough_calls:
                logger.info(
                    f"[TOOL LOOP] Discarding {len(passthrough_calls)} non-memory tool calls"
                )

            from proxy.requery import requery_llm

            requery_result = requery_llm(
                working_messages=working_messages,
                llama_body=llama_body,
                body=body,
                model_cfg=model_cfg,
                llama_url=llama_url,
                auth_headers=auth_headers,
                tool_loop_start=tool_loop_start,
                max_total_time=max_total_time,
                passthrough_calls=passthrough_calls,
                parse_tools_fn=parse_tools_fn,
                make_tool_calls_fn=make_tool_calls_fn,
                _raw_tool_id_map=_raw_tool_id_map,
            )

            _requery_failed = requery_result.requery_failed
            if _requery_failed:
                termination_reason = TERMINATION_REQUERY_FAILED
                break

            requery_data     = requery_result.requery_data
            working_messages = requery_result.working_messages
            _raw_tool_id_map = requery_result._raw_tool_id_map
            new_tool_calls   = requery_result.new_tool_calls
            requery_content  = requery_result.requery_content
            should_continue  = requery_result.should_continue

            if requery_result.upstream_finish_reason is not None:
                last_upstream_finish_reason = requery_result.upstream_finish_reason

            if should_continue:
                tool_calls_list = new_tool_calls
                iteration += 1
                continue

            if new_tool_calls:
                remaining_passthrough = [
                    tc for tc in new_tool_calls if tc.get("name") not in _memory_tool_names
                ]
                if remaining_passthrough:
                    openai_tool_calls = make_tool_calls_fn(new_tool_calls)
                    if openai_tool_calls:
                        final_tool_calls = openai_tool_calls
                        content = requery_content if requery_content else None
                        saw_passthrough_path = True
                        termination_reason = TERMINATION_PASSTHROUGH_REQUIRED
                    break

            content = requery_content if requery_content else None
            break
        except Exception as e:
            working_messages = working_messages[:pre_exec_len]
            logger.error(f"[TOOL LOOP] Transactional rollback due to error: {e}")
            termination_reason = TERMINATION_INTERNAL_ERROR
            _requery_failed = True
            break

        iteration += 1

    if iteration >= max_iterations and termination_reason == TERMINATION_MODEL_COMPLETED:
        termination_reason = TERMINATION_ITERATION_LIMIT
        last_upstream_finish_reason = None
        if requery_content:
            content = requery_content
        for tc in (new_tool_calls or []):
            tc_name = tc.get("name")
            if tc_name not in _memory_tool_names:
                continue
            tc_args = tc.get("arguments", {})
            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except (json.JSONDecodeError, ValueError):
                    tc_args = {}
            from memory.tools.dispatch import TICKET_TOOL_OPERATIONS
            _op = (
                tc_args.get("action")
                if isinstance(tc_args, dict) and isinstance(tc_args.get("action"), str)
                else None
            ) or TICKET_TOOL_OPERATIONS.get(tc_name, tc_name)
            execution_records.append(
                {
                    "tool": tc_name,
                    "operation": _op,
                    "tool_call_id": None,
                    "iteration": max_iterations + 1,
                    "attempt": None,
                    "status": "not_executed",
                    "error": {
                        "code": "iteration_limit",
                        "message": "Operation not executed — iteration limit reached",
                        "retryable": False,
                        "category": "infrastructure",
                    },
                    "data": None,
                }
            )

    original_len = len(llama_body.get("messages", []))
    if _requery_failed:
        messages_for_history = []
    else:
        messages_for_history = _build_sanitized_history_view(
            working_messages, original_len
        )

    if _requery_failed:
        fallback_calls = [
            tc for tc in tool_calls_list if tc.get("name") not in _memory_tool_names
        ]
        if not fallback_calls:
            fallback_calls = [
                tc for tc in original_tool_calls_raw
                if tc.get("name") not in _memory_tool_names
            ] if original_tool_calls_raw else []

        if fallback_calls and make_tool_calls_fn:
            fallback_openai = make_tool_calls_fn(fallback_calls)
            return _build_candidate_result(
                content=None,
                tool_calls=fallback_openai,
                finish_reason="tool_calls",
                messages_for_history=messages_for_history,
                execution_records=execution_records,
                termination_reason=TERMINATION_PASSTHROUGH_REQUIRED,
                needs_closure=False,
            )

        candidate = _build_candidate_result(
            content=None,
            tool_calls=None,
            finish_reason="stop",
            messages_for_history=messages_for_history,
            execution_records=execution_records,
            termination_reason=termination_reason or TERMINATION_REQUERY_FAILED,
            needs_closure=True,
        )
        return _resolve_closure(
            candidate,
            working_messages=working_messages,
            llama_url=llama_url,
            body=body,
            llama_body=llama_body,
            model_cfg=model_cfg,
            auth_headers=auth_headers,
            tool_loop_start=tool_loop_start,
            max_total_time=max_total_time,
            termination_reason=termination_reason or TERMINATION_REQUERY_FAILED,
            parse_tools_fn=parse_tools_fn,
        )

    from proxy.tool_pipeline.tool_pipeline import TRUNCATION_REASONS

    if final_tool_calls:
        _final_finish_reason = "tool_calls"
    elif last_upstream_finish_reason in TRUNCATION_REASONS:
        _final_finish_reason = last_upstream_finish_reason
    else:
        _final_finish_reason = "stop"

    if saw_passthrough_path:
        return _build_candidate_result(
            content=content,
            tool_calls=final_tool_calls,
            finish_reason=_final_finish_reason,
            messages_for_history=messages_for_history,
            execution_records=execution_records,
            termination_reason=TERMINATION_PASSTHROUGH_REQUIRED,
            needs_closure=False,
        )

    if _clean_close_skip_eligible(
        content=content,
        termination_reason=termination_reason,
        new_tool_calls=new_tool_calls,
    ):
        return _build_candidate_result(
            content=content,
            tool_calls=final_tool_calls,
            finish_reason=_final_finish_reason,
            messages_for_history=messages_for_history,
            execution_records=execution_records,
            termination_reason=TERMINATION_MODEL_COMPLETED,
            needs_closure=False,
        )

    candidate = _build_candidate_result(
        content=content,
        tool_calls=final_tool_calls,
        finish_reason=_final_finish_reason,
        messages_for_history=messages_for_history,
        execution_records=execution_records,
        termination_reason=termination_reason,
        needs_closure=True,
    )
    return _resolve_closure(
        candidate,
        working_messages=working_messages,
        llama_url=llama_url,
        body=body,
        llama_body=llama_body,
        model_cfg=model_cfg,
        auth_headers=auth_headers,
        tool_loop_start=tool_loop_start,
        max_total_time=max_total_time,
        termination_reason=termination_reason,
        parse_tools_fn=parse_tools_fn,
    )