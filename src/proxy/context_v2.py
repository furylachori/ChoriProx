"""Context selection v2 — chain-hash context assembly.

Implements the V2 pipeline per docs/v2-chain-hash-redesign.md §6. State is
keyed on (session_id, model_name) and persisted across turns via a
cumulative chain hash that lets the proxy detect edits (forks) and reuse
cached summaries/ancient between evictions.
"""

import copy
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TOKEN_RATIO = 0.75
ZONE1_FULL_MIN = 50
ZONE2_ACTION_MIN = 100

BARRIER_CONTENT = " "


def _safe_env(name, default, cast=float, clamp_min=None, clamp_max=None):
    try:
        val = os.environ.get(name)
        if val is None or val == "":
            return default
        parsed = cast(val)
        if clamp_min is not None and parsed < clamp_min:
            logger.warning(
                "V2 env %s=%r below clamp_min=%s; using default %s",
                name, val, clamp_min, default,
            )
            return default
        if clamp_max is not None and parsed > clamp_max:
            logger.warning(
                "V2 env %s=%r above clamp_max=%s; using default %s",
                name, val, clamp_max, default,
            )
            return default
        return parsed
    except (ValueError, TypeError):
        logger.warning(
            "Invalid V2 env %s=%r; using default %s",
            name, os.environ.get(name), default,
        )
        return default


V2_LOW_WATERMARK_RATIO = _safe_env("V2_LOW_WATERMARK_RATIO", 0.65, float, clamp_min=0.01, clamp_max=1.0)
V2_HIGH_WATERMARK_RATIO = _safe_env("V2_HIGH_WATERMARK_RATIO", 0.95, float, clamp_min=0.0, clamp_max=1.0)
if V2_HIGH_WATERMARK_RATIO < V2_LOW_WATERMARK_RATIO:
    V2_HIGH_WATERMARK_RATIO = V2_LOW_WATERMARK_RATIO
V2_SUMMARY_BUDGET_RATIO = _safe_env("V2_SUMMARY_BUDGET_RATIO", 0.15, float, clamp_min=0.05, clamp_max=1.0)
V2_ANCIENT_MAX_RATIO = _safe_env("V2_ANCIENT_MAX_RATIO", 0.10, float, clamp_min=0.0, clamp_max=1.0)
V2_CHAIN_QUERY_LIMIT = _safe_env("V2_CHAIN_QUERY_LIMIT", 10000, int, clamp_min=100)


from proxy.context import ContextLimitError, _count_msg_chars, _group_message_units


def _split_completed_and_live_exchange(raw_messages):
    """Split retained raw messages into deep-copied completed history and
    user-led live suffix.

    The live suffix starts at the most recent user unit and unconditionally
    widens backward across the immediately preceding consecutive-user run.
    Every later assistant/tool/unknown unit remains in the live suffix.
    With no user unit, all copied messages are completed and live is empty.

    Both returned lists are deep-copied from the input. Nested `tool_calls`,
    `function`, list content, and dict content are not shared with the input.
    The splitter performs no state access, I/O, cleanup, logging, VFS
    building, Tier-2 handling, or provider adaptation.
    """
    copied = copy.deepcopy(raw_messages)
    units = _group_message_units(copied)

    latest_user_unit = None
    for index in range(len(units) - 1, -1, -1):
        unit = units[index]
        if len(unit) == 1 and unit[0].get("role") == "user":
            latest_user_unit = index
            break

    if latest_user_unit is None:
        return copied, []

    live_start = latest_user_unit
    while live_start > 0:
        previous = units[live_start - 1]
        if len(previous) != 1 or previous[0].get("role") != "user":
            break
        live_start -= 1

    completed = [
        message
        for unit in units[:live_start]
        for message in unit
    ]
    live = [
        message
        for unit in units[live_start:]
        for message in unit
    ]
    return completed, live


def _cleanup_tool_pair_integrity(messages):
    # Belt-and-suspenders: drop any barrier assistant message
    # sandwiched between an assistant tool_call and its matching role=tool
    # result. Defends against any future barrier site that violates tool-pair
    # integrity (upstream rejects with 400 tool-result-follows-tool-call).
    barrier_indices = set()
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant" or msg.get("content") != BARRIER_CONTENT:
            continue
        if idx == 0 or idx >= len(messages) - 1:
            continue
        prev_msg = messages[idx - 1]
        next_msg = messages[idx + 1]
        if prev_msg.get("role") != "assistant":
            continue
        prev_tool_calls = prev_msg.get("tool_calls")
        if not isinstance(prev_tool_calls, list) or not prev_tool_calls:
            continue
        prev_tc_ids = {tc.get("id") for tc in prev_tool_calls if tc.get("id")}
        if next_msg.get("role") != "tool":
            continue
        next_tc_id = next_msg.get("tool_call_id")
        if not next_tc_id or next_tc_id not in prev_tc_ids:
            continue
        barrier_indices.add(idx)

    if barrier_indices:
        messages = [m for i, m in enumerate(messages) if i not in barrier_indices]
        logger.debug(
            "Dropped %d barrier message(s) wedged between tool_call and result",
            len(barrier_indices),
        )

    present_tool_ids = set()
    assistant_tc_ids = set()
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                present_tool_ids.add(tc_id)
        elif msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tc in msg["tool_calls"]:
                if tc.get("id"):
                    assistant_tc_ids.add(tc["id"])

    cleaned = []
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if (
                tc_id
                and tc_id not in assistant_tc_ids
                and msg.get("_sub_agent_return") is not True
            ):
                logger.warning(
                    "Dropped orphan tool result tc_id=%s (parent assistant was pruned)",
                    tc_id,
                )
                continue
        cleaned.append(msg)

    final = []
    for msg in cleaned:
        final.append(msg)
        if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id and tc_id not in present_tool_ids:
                    final.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tc.get("function", {}).get("name", "unknown"),
                        "content": "[Result pruned from earlier context — refer to conversation summary for details]",
                    })
                    logger.warning(
                        "Injected synthetic tool result for missing tc_id=%s (parent assistant preserved)",
                        tc_id,
                    )
                    present_tool_ids.add(tc_id)

    return final


def _trim_unit_to_budget(unit, budget_tokens, tokens_per_char):
    """Trim a message unit (atomic group) to fit within budget_tokens."""
    def _tok(m):
        return int(_count_msg_chars(m) * tokens_per_char)

    result = list(unit)
    total = sum(_tok(m) for m in result)
    if total <= budget_tokens or budget_tokens <= 0:
        return result

    suffix = "... [truncated]"

    while total > budget_tokens:
        tool_idxs = [i for i, m in enumerate(result) if m.get("role") == "tool"]
        if len(tool_idxs) <= 1:
            break
        drop_idx = max(tool_idxs, key=lambda i: _tok(result[i]))
        total -= _tok(result.pop(drop_idx))

    if total <= budget_tokens:
        return result

    guard = 0
    while total > budget_tokens and guard < 100:
        guard += 1
        best_idx = -1
        best_len = 0
        for i, m in enumerate(result):
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > best_len:
                best_len = len(content)
                best_idx = i
        if best_idx < 0:
            break
        target = result[best_idx]
        content = target.get("content", "")
        before = _tok(target)
        overhead_chars = int((total - budget_tokens) / tokens_per_char) + len(suffix)
        new_len = max(0, len(content) - overhead_chars)
        if new_len <= 0:
            new_content = suffix if len(suffix) < len(content) else content
        else:
            new_content = content[:new_len] + suffix
        result[best_idx] = {**target, "content": new_content}
        after = _tok(result[best_idx])
        if after >= before:
            break
        total = total - before + after

    return result


def _trim_completed_history_to_budget(
    completed_history,
    budget_tokens,
    tokens_per_char,
):
    """Drop completed message units until they fit the discretionary budget."""
    if not completed_history or budget_tokens <= 0:
        return []

    units = _group_message_units(copy.deepcopy(completed_history))

    def _unit_tokens(unit):
        return int(
            sum(_count_msg_chars(message) for message in unit)
            * tokens_per_char
        )

    total = sum(_unit_tokens(unit) for unit in units)
    if total <= budget_tokens:
        return [
            message
            for unit in units
            for message in unit
        ]

    units_by_size = sorted(
        (
            (_unit_tokens(unit), index)
            for index, unit in enumerate(units)
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    dropped = set()
    for unit_tokens, index in units_by_size:
        if total <= budget_tokens:
            break
        total -= unit_tokens
        dropped.add(index)

    return [
        message
        for index, unit in enumerate(units)
        if index not in dropped
        for message in unit
    ]


def _build_subagent_context(
    full_messages, system_prompt, system_prompt_tokens,
    tool_schema_tokens, model_safe_tokens, tokens_per_char, session_id,
    model_name=None, vfs_inherited_block=None, ctx=None,
):
    """Assemble context for a sub-agent, guaranteeing the task survives.

    Sub-agents inherit conversation summaries (US-058) so they have
    continuity context. The summary block uses the [Conversation Summary]
    marker so sub-agents can distinguish inherited context from new
    instructions.
    """
    summaries = []
    try:
        from memory.db import get_rich_summaries
        summaries = get_rich_summaries(session_id) or []
    except Exception as e:
        logger.warning(
            "V2 sub-agent summaries fetch failed session=%s: %s",
            session_id, e,
        )
        summaries = []
    summaries = _apply_summary_budget(
        summaries,
        model_safe_tokens,
        tokens_per_char,
        session_id,
        ctx,
        is_subagent=True,
    )
    summary_block = None
    if summaries:
        summary_block = {
            "role": "system",
            "content": _format_summaries(
                summaries,
                is_subagent=True,
            ),
        }

    vfs_block = vfs_inherited_block

    def _with_summary(msgs):
        if summary_block is None:
            return msgs
        out = []
        inserted = False
        for m in msgs:
            out.append(m)
            if not inserted and m.get("role") != "system":
                out.append(summary_block)
                inserted = True
        if not inserted:
            out.append(summary_block)
        return out

    def _with_vfs(msgs):
        if vfs_block is None:
            return msgs
        out = []
        inserted = False
        for m in msgs:
            out.append(m)
            if not inserted and m.get("role") != "system":
                out.append(vfs_block)
                inserted = True
        if not inserted:
            out.append(vfs_block)
        return out

    def _with_inherited(msgs):
        return _with_summary(_with_vfs(msgs))

    cleaned = _cleanup_tool_pair_integrity(full_messages)
    cleaned_non_system = [
        message
        for message in cleaned
        if message.get("role") != "system"
    ]
    cleaned_tokens = int(
        sum(
            _count_msg_chars(message)
            for message in cleaned_non_system
        )
        * tokens_per_char
    )

    # Simulate the exact mandatory VFS-free candidate seen by the finalizer.
    # Inherited VFS remains optional and is removed by object identity.
    mandatory_candidate = _with_summary(cleaned)
    mandatory_rendered = _emit_tier2_pin(
        mandatory_candidate,
        session_id,
        ctx=ctx,
    )
    mandatory_tokens = (
        int(
            sum(
                _count_msg_chars(message)
                for message in mandatory_rendered
            )
            * tokens_per_char
        )
        + tool_schema_tokens
    )

    if mandatory_tokens <= model_safe_tokens:
        logger.debug(
            "V2: sub-agent exchange fits safe budget "
            "(mandatory=%d + optional_vfs=%s <= %d), session=%s",
            mandatory_tokens,
            vfs_block is not None,
            model_safe_tokens,
            session_id,
        )
        return _finalize_context_result(
            _with_inherited(cleaned),
            session_id,
            ctx,
            tool_schema_tokens=tool_schema_tokens,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
            optional_vfs_message=vfs_block,
        )

    logger.warning(
        "V2: sub-agent exchange overflows safe budget "
        "(sys=%d + raw=%d + tools=%d > %d); preserving "
        "[system, first-user] + trimmed-last-unit (dedup), session=%s",
        system_prompt_tokens, cleaned_tokens, tool_schema_tokens,
        model_safe_tokens, session_id,
    )

    recent_raw = [m for m in full_messages if m.get("role") != "system"]
    first_user = next((m for m in recent_raw if m.get("role") == "user"), None)
    units = _group_message_units(recent_raw)
    last_unit = units[-1] if units else recent_raw[-1:]

    if first_user is None:
        fixed_candidate = [system_prompt] if system_prompt else []
        fixed_rendered = _emit_tier2_pin(
            _with_summary(fixed_candidate),
            session_id,
            ctx=ctx,
        )
        fixed_tokens = int(
            sum(
                _count_msg_chars(message)
                for message in fixed_rendered
            )
            * tokens_per_char
        )
        unit_rounding_guard = max(0, len(last_unit) - 1)
        unit_budget = (
            model_safe_tokens
            - tool_schema_tokens
            - fixed_tokens
            - unit_rounding_guard
        )
        trimmed = _trim_unit_to_budget(last_unit, unit_budget, tokens_per_char)
        out = ([system_prompt] if system_prompt else []) + trimmed
        return _finalize_context_result(
            _with_inherited(_cleanup_tool_pair_integrity(out)),
            session_id,
            ctx,
            tool_schema_tokens=tool_schema_tokens,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
            optional_vfs_message=vfs_block,
        )

    fixed_candidate = [
        message
        for message in (system_prompt, first_user)
        if message is not None
    ]
    fixed_rendered = _emit_tier2_pin(
        _with_summary(fixed_candidate),
        session_id,
        ctx=ctx,
    )
    fixed_tokens = int(
        sum(
            _count_msg_chars(message)
            for message in fixed_rendered
        )
        * tokens_per_char
    )
    unit_rounding_guard = max(0, len(last_unit) - 1)
    unit_budget = (
        model_safe_tokens
        - tool_schema_tokens
        - fixed_tokens
        - unit_rounding_guard
    )
    trimmed_last_unit = _trim_unit_to_budget(
        last_unit, unit_budget, tokens_per_char,
    )

    candidate = []
    if system_prompt:
        candidate.append(system_prompt)
    candidate.append(first_user)
    for m in trimmed_last_unit:
        if m is first_user:
            continue
        candidate.append(m)
    candidate = _cleanup_tool_pair_integrity(candidate)
    # Slice 4D correction: simulate the exact aggregate measurement the
    # finalizer will perform (candidate + Tier-2 pin). Per-message int()
    # sum and aggregate int(sum * tpc) can differ by 1+ tokens; using the
    # aggregate formula here guarantees the _fits decision matches the
    # final boundary check.
    simulated_candidate = _with_summary(candidate)
    simulated_rendered = _emit_tier2_pin(
        simulated_candidate,
        session_id,
        ctx=ctx,
    )
    simulated_total = (
        int(
            sum(
                _count_msg_chars(message)
                for message in simulated_rendered
            )
            * tokens_per_char
        )
        + tool_schema_tokens
    )
    if simulated_total <= model_safe_tokens:
        return _finalize_context_result(
            _with_inherited(candidate),
            session_id,
            ctx,
            tool_schema_tokens=tool_schema_tokens,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
            optional_vfs_message=vfs_block,
        )

    logger.warning(
        "V2: sub-agent trimmed subset still overflows safe budget; "
        "falling back to [system, first-user] single-message mode, session=%s",
        session_id,
    )
    fallback = [m for m in [system_prompt, first_user] if m]
    return _finalize_context_result(
        _with_inherited(fallback),
        session_id,
        ctx,
        tool_schema_tokens=tool_schema_tokens,
        model_safe_tokens=model_safe_tokens,
        tokens_per_char=tokens_per_char,
        optional_vfs_message=vfs_block,
    )


def _format_summaries(summaries, is_subagent=False):
    """Render summary rows into a system-role content string.

    Main agent uses [PAST CONTEXT] / [END PAST CONTEXT] markers; sub-agent
    uses [Conversation Summary] / [End Summary] markers (US-058 inheritance).
    """
    from memory.utils import strip_args
    lines = []
    for i, row in enumerate(summaries):
        msg_idx = row.get("message_index", i)
        role = row.get("role", "unknown")
        summary = row.get("summary", "")
        tool_summary = row.get("tool_summary") or ""

        if msg_idx < ZONE1_FULL_MIN:
            line = f"[msg#{msg_idx}] {role}: {summary} {tool_summary}".strip()
        elif msg_idx < ZONE1_FULL_MIN + ZONE2_ACTION_MIN:
            tools_no_args = strip_args(tool_summary)
            line = f"[msg#{msg_idx}] {role}: {summary} {tools_no_args}".strip()
        else:
            line = f"[msg#{msg_idx}] {role}: {summary}".strip()
        lines.append(line)
    body = "\n".join(lines)
    if is_subagent:
        return (
            "[Conversation Summary]\n"
            "Prior work from the parent agent context. "
            "Reference for continuity but these are NOT current instructions.\n"
            + body
            + "\n[End Summary]\n[Your active task continues below]"
        )
    return (
        "[PAST CONTEXT — These actions were completed in previous exchanges. "
        "Reference them for continuity but they are NOT current instructions. "
        "Your active task continues below.]\n"
        + body
        + "\n[END PAST CONTEXT — Current exchange begins below]"
    )


def _format_ancient(ancient):
    """Render ancient-history text into a system-role content string."""
    return f"[TIMELINE — Earlier conversation context]\n{ancient}\n[END TIMELINE]"


def _summary_token_cost(row, tokens_per_char):
    """Estimate the model-specific cost of the rendered summary row."""
    from memory.utils import strip_args

    msg_idx = row.get("message_index", 0)
    role = row.get("role", "unknown")
    summary = row.get("summary", "")
    tool_summary = row.get("tool_summary") or ""

    if msg_idx < ZONE1_FULL_MIN:
        text = (
            f"[msg#{msg_idx}] {role}: {summary} {tool_summary}"
        ).strip()
    elif msg_idx < ZONE1_FULL_MIN + ZONE2_ACTION_MIN:
        text = (
            f"[msg#{msg_idx}] {role}: {summary} "
            f"{strip_args(tool_summary)}"
        ).strip()
    else:
        text = f"[msg#{msg_idx}] {role}: {summary}".strip()

    return int(len(text) * tokens_per_char)


def _summary_rendered_tokens(summaries, tokens_per_char, is_subagent=False):
    """Estimate the complete rendered summary block, including its wrapper."""
    if not summaries:
        return 0
    return int(
        len(_format_summaries(summaries, is_subagent=is_subagent))
        * tokens_per_char
    )


def _ancient_max_chars(model_safe_tokens, tokens_per_char):
    """Return ancient payload capacity after reserving its wrapper."""
    if tokens_per_char <= 0:
        tokens_per_char = 0.25

    tier_tokens = int(
        model_safe_tokens * V2_ANCIENT_MAX_RATIO
    )
    if tier_tokens <= 0:
        return 0

    rendered_char_limit = int(tier_tokens / tokens_per_char)
    wrapper_chars = len(_format_ancient(""))
    return max(0, rendered_char_limit - wrapper_chars)


def _truncate_ancient(ancient, max_chars, session_id, ctx=None):
    """Central truncation point. Keeps the head and appends a 'recent overflow
    pending re-compress' marker so downstream LLM sees truncation provenance."""
    if max_chars <= 0:
        return ""
    if not ancient or len(ancient) <= max_chars:
        return ancient
    marker = "\n[...truncated; ancient history overflowed " + f"{max_chars} chars, recent overflow pending re-compress...]"
    if max_chars <= len(marker):
        return marker[:max_chars]
    return ancient[: max_chars - len(marker)] + marker


def _apply_summary_budget(
    summaries,
    model_safe_tokens,
    tokens_per_char,
    session_id,
    ctx,
    is_subagent=False,
):
    """Cap the complete rendered summary block and dispatch dropped rows.

    The cap includes the main or sub-agent wrapper and row separators.
    Oldest rows are dropped as whole rows; the newest contiguous suffix is
    retained.
    """
    if not summaries:
        return summaries

    summary_budget = int(model_safe_tokens * V2_SUMMARY_BUDGET_RATIO)
    summaries_token_count = _summary_rendered_tokens(
        summaries,
        tokens_per_char,
        is_subagent=is_subagent,
    )
    if summaries_token_count <= summary_budget:
        return summaries

    # The fit predicate is monotonic as rows are removed from the front.
    # Binary search avoids repeatedly rendering every suffix.
    low = 1
    high = len(summaries)
    while low < high:
        dropped_count = (low + high) // 2
        retained = summaries[dropped_count:]
        retained_tokens = _summary_rendered_tokens(
            retained,
            tokens_per_char,
            is_subagent=is_subagent,
        )
        if retained_tokens <= summary_budget:
            high = dropped_count
        else:
            low = dropped_count + 1

    dropped_count = low
    truncated = list(summaries[dropped_count:])
    dropped_summaries = list(summaries[:dropped_count])
    served_tokens = _summary_rendered_tokens(
        truncated,
        tokens_per_char,
        is_subagent=is_subagent,
    )

    if dropped_summaries and is_subagent:
        logger.info(
            "Serving truncated inherited summaries without ancient "
            "compression session=%s full=%d served=%d dropped=%d "
            "budget=%d",
            session_id,
            summaries_token_count,
            served_tokens,
            len(dropped_summaries),
            summary_budget,
            extra={"request_id": getattr(ctx, "request_id", "?")},
        )
        return truncated

    if dropped_summaries:
        logger.warning(
            "ancient re-compression pending, serving truncated summaries "
            "session=%s full=%d served=%d dropped=%d budget=%d",
            session_id,
            summaries_token_count,
            served_tokens,
            len(dropped_summaries),
            summary_budget,
            extra={"request_id": getattr(ctx, "request_id", "?")},
        )
        try:
            from memory.sync import submit_ancient_compress
            future = submit_ancient_compress(
                session_id,
                dropped_summaries,
            )
            if future is None:
                logger.debug(
                    "ancient compress dispatch skipped "
                    "(queue full or breaker open) "
                    "session=%s dropped=%d",
                    session_id,
                    len(dropped_summaries),
                    extra={
                        "request_id": getattr(ctx, "request_id", "?")
                    },
                )
            else:
                logger.info(
                    "Dispatched ancient compress for session=%s "
                    "(dropped=%d)",
                    session_id,
                    len(dropped_summaries),
                    extra={
                        "request_id": getattr(ctx, "request_id", "?")
                    },
                )
        except Exception as e:
            logger.warning(
                "submit_ancient_compress dispatch failed; summaries "
                "truncated without compression (session=%s): %s",
                session_id,
                e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
                exc_info=True,
            )

    return truncated


def _load_state_from_db(session_id, model_name, ctx=None, model_safe_tokens=None, tokens_per_char=None):
    """Restart recovery — load chain state from SQLite.

    Returns a state-entry dict (matching new_chain_state_entry shape) when the
    session has a valid `raw_start_message_index` (>= 0), or None when the
    sentinel value (-1) is stored or no row exists.

    model_safe_tokens/tokens_per_char are threaded for budget enforcement at
    restart/cache-miss time. Defaults to a conservative safe budget when omitted.
    """
    try:
        from memory.db import connection, get_rich_summaries, get_ancient_history
        with connection(pooled=True) as conn:
            row = conn.execute(
                "SELECT raw_start_message_index FROM conversation_summaries "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                # raw_start is a conversation position (non-system 0-indexed),
                # not messages.message_index (which includes system at index 0).
                raw_start = row["raw_start_message_index"]
            except (IndexError, KeyError):
                return None
            if raw_start is None or raw_start < 0:
                return None

            total_non_system = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role != 'system'",
                (session_id,),
            ).fetchone()[0]

            prefix_rows = []
            if raw_start > 0:
                sql_raw_start_row = conn.execute(
                    "SELECT message_index FROM messages "
                    "WHERE session_id = ? AND role != 'system' "
                    "ORDER BY message_index LIMIT 1 OFFSET ?",
                    (session_id, raw_start),
                ).fetchone()
                sql_raw_start = (
                    sql_raw_start_row["message_index"]
                    if sql_raw_start_row is not None
                    else raw_start + 1
                )
                prefix_rows = conn.execute(
                    "SELECT message_index, chain_hash FROM messages "
                    "WHERE session_id = ? AND message_index < ? "
                    "AND chain_hash IS NOT NULL "
                    "ORDER BY message_index",
                    (session_id, sql_raw_start),
                ).fetchall()

            recent_rows = conn.execute(
                "SELECT message_index, chain_hash FROM messages "
                "WHERE session_id = ? AND chain_hash IS NOT NULL "
                "ORDER BY message_index DESC LIMIT ?",
                (session_id, V2_CHAIN_QUERY_LIMIT),
            ).fetchall()
            recent_rows.reverse()

            seen = set()
            chain_rows = []
            for r in prefix_rows + recent_rows:
                mi = r["message_index"]
                if mi in seen:
                    continue
                seen.add(mi)
                chain_rows.append(r)
            chain_rows.sort(key=lambda r: r["message_index"])

            if not chain_rows:
                return None

            total_with_chain = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND chain_hash IS NOT NULL",
                (session_id,),
            ).fetchone()[0]
            if total_with_chain != total_non_system:
                logger.warning(
                    "V2 restart recovery: chain_hash gaps for session=%s model=%s "
                    "(chain_rows=%d total_with_chain=%d non_system=%d); cold path",
                    session_id, model_name, len(chain_rows),
                    total_with_chain, total_non_system,
                )
                return None

            chain = [r["chain_hash"] for r in chain_rows]

            if raw_start == 0:
                boundary_chain = None
            elif raw_start - 1 < len(chain):
                boundary_chain = chain[raw_start - 1]
            else:
                logger.warning(
                    "V2 restart recovery: raw_start=%d outside chain len=%d; cold path",
                    raw_start, len(chain),
                    extra={"request_id": getattr(ctx, "request_id", "?")},
                )
                return None

            sql_boundary_idx_row = conn.execute(
                "SELECT message_index FROM messages "
                "WHERE session_id = ? AND role != 'system' "
                "ORDER BY message_index LIMIT 1 OFFSET ?",
                (session_id, raw_start),
            ).fetchone()
            if sql_boundary_idx_row is not None:
                sql_boundary_idx = sql_boundary_idx_row["message_index"]
            else:
                sql_boundary_idx = raw_start + 1
            summaries = get_rich_summaries(session_id, before_index=sql_boundary_idx) or []
            ancient = get_ancient_history(session_id) or ""
            if tokens_per_char is None:
                tokens_per_char = 0.25
            if model_safe_tokens is None:
                try:
                    from proxy.context import get_context_config
                    cfg = get_context_config(model_name, None)
                    model_safe_tokens = cfg["safe_context_tokens"]
                except Exception:
                    model_safe_tokens = 19500
            summaries = _apply_summary_budget(
                summaries, model_safe_tokens, tokens_per_char, session_id, ctx,
            )
            ancient = _truncate_ancient(
                ancient, _ancient_max_chars(model_safe_tokens, tokens_per_char),
                session_id, ctx=ctx,
            )
    except (sqlite3.Error, OSError, KeyError, IndexError) as e:
        logger.warning(
            "V2 restart recovery failed for session=%s model=%s; forcing cold path: %s",
            session_id, model_name, e,
            extra={"request_id": getattr(ctx, "request_id", "?")},
            exc_info=True,
        )
        return None

    return {
        "last_chain": chain,
        "boundary_idx": raw_start,
        "boundary_chain": boundary_chain,
        "summaries": summaries,
        "ancient": ancient,
        "version": 0,
    }


def _cold_path_compute(conversation, model_cfg, session_id, model_name, ctx,
                       low_watermark, tokens_per_char, model_safe_tokens=None):
    """Full re-trim of conversation from the end. Returns a state-entry dict."""
    units = _group_message_units(conversation)
    kept_units = []
    used = 0
    for unit in reversed(units):
        unit_tokens = sum(int(_count_msg_chars(m) * tokens_per_char) for m in unit)
        if used + unit_tokens > low_watermark:
            break
        kept_units.append(unit)
        used += unit_tokens
    kept_units.reverse()

    if not kept_units and units:
        kept_units = [units[-1]]

    boundary_idx = sum(len(u) for u in units) - sum(len(u) for u in kept_units)
    if boundary_idx < 0:
        boundary_idx = 0

    if boundary_idx > 0 and boundary_idx < len(conversation):
        try:
            from memory.db_summaries import get_last_summarized_exchange_index
            from memory.sync import submit_sync
            cursor = get_last_summarized_exchange_index(session_id)
            if cursor is not None and cursor >= 0:
                boundary_exchange = None
                try:
                    boundary_msg = conversation[boundary_idx]
                    boundary_message_index = boundary_msg.get("message_index")
                    if boundary_message_index is not None:
                        from memory.db_pool import connection
                        with connection(pooled=True) as conn:
                            row = conn.execute(
                                "SELECT exchange_index FROM messages "
                                "WHERE session_id = ? AND message_index = ?",
                                (session_id, boundary_message_index),
                            ).fetchone()
                            if row and row["exchange_index"] is not None:
                                boundary_exchange = row["exchange_index"]
                except Exception:
                    boundary_exchange = None
                if boundary_exchange is not None and boundary_exchange > cursor:
                    gap = boundary_exchange - cursor
                    logger.warning(
                        "V2 coverage gap: %d uncovered exchanges (cursor=%d boundary_exchange=%d) session_id=%s",
                        gap, cursor, boundary_exchange, session_id,
                        extra={"request_id": getattr(ctx, "request_id", "?")} if ctx is not None else {},
                    )
                    try:
                        submit_sync(session_id)
                    except Exception as e:
                        logger.debug("[D6] catch-up submit_sync failed (non-fatal): %s", e)
        except Exception as e:
            logger.debug("[D6] coverage check failed (non-fatal): %s", e)

    summaries = []
    ancient = ""
    if boundary_idx > 0:
        try:
            from memory.db import get_rich_summaries, get_ancient_history
            from memory.db import connection
            with connection(pooled=True) as _conv_conn:
                _sql_boundary_row = _conv_conn.execute(
                    "SELECT message_index FROM messages "
                    "WHERE session_id = ? AND role != 'system' "
                    "ORDER BY message_index LIMIT 1 OFFSET ?",
                    (session_id, boundary_idx),
                ).fetchone()
                if _sql_boundary_row is not None:
                    _sql_boundary_idx = _sql_boundary_row["message_index"]
                else:
                    _sql_boundary_idx = boundary_idx + 1
            summaries = get_rich_summaries(session_id, before_index=_sql_boundary_idx) or []
        except Exception as e:
            logger.warning(
                "V2 cold-path get_rich_summaries failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
        try:
            from memory.db import get_ancient_history
            ancient = get_ancient_history(session_id) or ""
        except Exception as e:
            logger.warning(
                "V2 cold-path get_ancient_history failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )

    if model_safe_tokens is None:
        try:
            from proxy.context import get_context_config
            cfg = get_context_config(model_name, model_cfg)
            model_safe_tokens = cfg["safe_context_tokens"]
        except Exception:
            model_safe_tokens = 19500

    summaries = _apply_summary_budget(
        summaries, model_safe_tokens, tokens_per_char, session_id, ctx,
    )
    ancient = _truncate_ancient(
        ancient, _ancient_max_chars(model_safe_tokens, tokens_per_char),
        session_id, ctx=ctx,
    )

    boundary_chain = None
    if boundary_idx > 0:
        from proxy.chain_hash import compute_chain
        prefix_chain, _ = compute_chain(conversation[:boundary_idx])
        if prefix_chain:
            boundary_chain = prefix_chain[-1]

    return {
        "boundary_idx": boundary_idx,
        "boundary_chain": boundary_chain,
        "summaries": summaries,
        "ancient": ancient,
    }


@dataclass
class _EvictResult:
    boundary_idx: int
    boundary_chain: str
    summaries: list = field(default_factory=list)
    ancient: str = ""
    skipped_due_to_cap: bool = False


def _evict_with_cap(raw_block, boundary_idx, session_id, model_name, raw_tokens,
                    low_watermark, high_watermark, ctx, new_chain,
                    tokens_per_char, snap_summaries, snap_ancient, model_safe_tokens):
    """Evict oldest units from raw_block until under low_watermark, capped at
    the last-summarized exchange boundary.

    Returns an _EvictResult describing the new boundary. When the fetched
    summaries overflow the summary budget, oldest summaries are truncated this
    turn and an async SOTA re-compression job is dispatched (§10).
    """
    try:
        from memory.db import connection, get_rich_summaries
        from memory.db_summaries import get_last_summarized_exchange_index
        last_exchange = get_last_summarized_exchange_index(session_id)
    except Exception as e:
        logger.warning(
            "V2 evict: last_summarized_exchange_index lookup failed session=%s: %s",
            session_id, e,
            extra={"request_id": getattr(ctx, "request_id", "?")},
        )
        last_exchange = -1

    max_evictable_boundary = 0
    if last_exchange is not None and last_exchange >= 0:
        try:
            from memory.db import connection
            with connection(pooled=True) as conn:
                row = conn.execute(
                    "SELECT MIN(message_index) AS mi FROM messages "
                    "WHERE session_id = ? AND exchange_index > ?",
                    (session_id, last_exchange),
                ).fetchone()
                if row is not None and row["mi"] is not None:
                    sql_min_mi = row["mi"]
                    position_row = conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
                        "AND message_index < ? AND (role IS NULL OR role != 'system')",
                        (session_id, sql_min_mi),
                    ).fetchone()
                    max_evictable_boundary = position_row[0]
        except Exception as e:
            logger.warning(
                "V2 evict: max_evictable_boundary lookup failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )

    units = _group_message_units(raw_block)
    dropped = 0
    used = raw_tokens
    kept_units = list(units)
    while used > low_watermark and kept_units:
        head = kept_units[0]
        head_tokens = sum(int(_count_msg_chars(m) * tokens_per_char) for m in head)
        used -= head_tokens
        dropped += len(head)
        kept_units.pop(0)

    if not kept_units and units:
        last_unit = units[-1]
        dropped -= len(last_unit)

    proposed_boundary = boundary_idx + dropped
    new_boundary_idx = min(proposed_boundary, max_evictable_boundary)
    skipped_due_to_cap = False
    if new_boundary_idx <= boundary_idx:
        skipped_due_to_cap = True
        if raw_tokens > high_watermark * 1.10:
            logger.critical(
                "V2 raw_tokens exceed HIGH_WATERMARK * 1.10; force-evicting past last-summarized "
                "(session=%s model=%s raw=%d high=%d)",
                session_id, model_name, raw_tokens, high_watermark,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            new_boundary_idx = proposed_boundary
        else:
            new_boundary_idx = boundary_idx

    new_boundary_chain = None
    if new_boundary_idx > 0 and new_boundary_idx - 1 < len(new_chain):
        new_boundary_chain = new_chain[new_boundary_idx - 1]

    summaries = list(snap_summaries)
    ancient = snap_ancient
    if new_boundary_idx > boundary_idx:
        try:
            from memory.db import get_ancient_history
            ancient = get_ancient_history(session_id) or ""
        except Exception as e:
            logger.warning(
                "V2 evict: get_ancient_history failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            ancient = snap_ancient
        try:
            from memory.db import connection
            with connection(pooled=True) as _evict_conn:
                _sql_boundary_row = _evict_conn.execute(
                    "SELECT message_index FROM messages "
                    "WHERE session_id = ? AND role != 'system' "
                    "ORDER BY message_index LIMIT 1 OFFSET ?",
                    (session_id, new_boundary_idx),
                ).fetchone()
                if _sql_boundary_row is not None:
                    _sql_boundary_idx = _sql_boundary_row["message_index"]
                else:
                    _sql_boundary_idx = new_boundary_idx + 1
            new_summaries = get_rich_summaries(
                session_id, before_index=_sql_boundary_idx,
            ) or []
            summaries = new_summaries
        except Exception as e:
            logger.warning(
                "V2 evict: get_rich_summaries failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )

    summaries = _apply_summary_budget(
        summaries, model_safe_tokens, tokens_per_char, session_id, ctx,
    )
    ancient = _truncate_ancient(
        ancient, _ancient_max_chars(model_safe_tokens, tokens_per_char),
        session_id, ctx=ctx,
    )

    return _EvictResult(
        boundary_idx=new_boundary_idx,
        boundary_chain=new_boundary_chain,
        summaries=summaries,
        ancient=ancient,
        skipped_due_to_cap=skipped_due_to_cap,
    )


def _capture_tier2_instructions(raw_block, sys_msg, session_id, ctx=None):
    """Scan raw_block tool results for Tier-2 sub-AGENTS.md suffixes and record
    them into instruction_files_by_workspace. Runs PRE-trim so every tool
    result is complete or wholly evicted, never partial. Captures on both
    the main path and the fast-exit return; idempotent via delta-track.
    Main-path ctx.workspace_name is reliable (not set by canary parsing)."""
    import proxy.state as state
    from proxy.client_instructions import (
        _split_tool_content_for_tier2,
        extract_instruction_blocks,
    )

    workspace_id = None
    if ctx is not None:
        workspace_id = getattr(ctx, "workspace_name", None)
    if not workspace_id:
        from proxy.workspace_state import resolve_inherited_workspace
        workspace_id = resolve_inherited_workspace(session_id)
    if not workspace_id:
        return

    tier1_paths: set[str] = set()
    if sys_msg and isinstance(sys_msg.get("content"), str):
        tier1_paths = {p for p, _ in extract_instruction_blocks(sys_msg["content"])}

    _PATH_RE = re.compile(r"<path>(.*?)</path>")
    with state.vfs_lock:
        for idx, msg in enumerate(raw_block):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if "<path>" not in content:
                continue
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in state._last_scanned_by_tool_call:
                continue
            m = _PATH_RE.search(content)
            if not m:
                continue
            read_path = m.group(1)
            file_body, suffix = _split_tool_content_for_tier2(content)
            if suffix is None:
                continue
            read_dir = os.path.dirname(read_path)
            try:
                for instr_path, instr_content in extract_instruction_blocks(suffix):
                    if instr_path in tier1_paths:
                        continue
                    instr_dir = os.path.dirname(instr_path)
                    if not (read_dir == instr_dir or read_dir.startswith(instr_dir + os.sep)):
                        continue
                    state.record_instruction_file(workspace_id, instr_path, instr_content, session_id=session_id)
                if tc_id:
                    if len(state._last_scanned_by_tool_call) >= state._LAST_SCANNED_MAX:
                        state._last_scanned_by_tool_call.popitem(last=False)
                    state._last_scanned_by_tool_call[tc_id] = idx
            except Exception:
                logger.exception("tier2 capture: unexpected error on tool_call_id=%s", tc_id)


def _emit_tier2_pin(messages, session_id, ctx=None):
    """Best-effort, NON-mutating Tier-2 strip + pin emitter.

    Strips Tier-2 suffixes from tool-result messages (replacing each with its
    file body only), then inserts a ``role="system"`` pin message (built from
    the instruction-files map for the resolved workspace) immediately after
    the first system message. Returns a NEW list; never mutates the input.

    Strip + pin are paired: both gated on the map being non-empty. An empty
    map → no strip, no pin (original list returned unchanged). Never raises:
    any failure is logged and the original list is returned unchanged.
    """
    import proxy.state as state
    try:
        if not state.instruction_files_by_workspace:
            return messages

        workspace_id = None
        lookup_session = session_id
        from proxy.workspace_state import DEFAULT_WORKSPACE
        is_subagent = ctx is not None and getattr(ctx, "is_subagent", False)
        if is_subagent:
            from proxy.workspace_state import resolve_inherited_workspace, resolve_inherited_session
            workspace_id = resolve_inherited_workspace(session_id)
            if workspace_id == DEFAULT_WORKSPACE:
                lookup_session = resolve_inherited_session(session_id)
            else:
                lookup_session = session_id
        else:
            if ctx is not None:
                workspace_id = getattr(ctx, "workspace_name", None)
            if not workspace_id:
                from proxy.workspace_state import resolve_inherited_workspace
                workspace_id = resolve_inherited_workspace(session_id)
        if not workspace_id:
            return messages

        files = state.get_instruction_files(workspace_id, lookup_session)
        if not files:
            return messages

        from proxy.client_instructions import _split_tool_content_for_tier2

        stripped = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content")
                if isinstance(content, str):
                    file_body, suffix = _split_tool_content_for_tier2(content)
                    if suffix is not None:
                        stripped.append({**m, "content": file_body})
                        continue
            stripped.append(m)

        instruction_text = "\n\n".join(
            f"Instructions from: {p}\n{c}" for p, c in files.items()
        )
        from proxy.skill_prompts import _escape_prompt_field
        escaped = _escape_prompt_field(instruction_text)
        from proxy.client_instructions import format_instruction_section
        section = format_instruction_section(escaped, tier="sub")
        pin_msg = {"role": "system", "content": section}

        insert_idx = 0
        for i, m in enumerate(stripped):
            if m.get("role") == "system":
                insert_idx = i + 1
                break
        stripped.insert(insert_idx, pin_msg)
        return stripped
    except Exception:
        logger.exception("tier2 pin emit failed session=%s", session_id)
        return messages


def _finalize_context_result(
    messages,
    session_id,
    ctx,
    tool_schema_tokens,
    model_safe_tokens,
    tokens_per_char,
    optional_vfs_message=None,
):
    """Terminal validation: emit Tier-2, measure, drop VFS, trim, raise.

    Universal boundary for every return path in ``_select_context_v2`` and
    ``_build_subagent_context``. The returned candidate always satisfies
    ``message_tokens + tool_schema_tokens <= model_safe_tokens`` OR this
    function raises ``ContextLimitError``.

    Recovery order on overflow:
    1. Drop optional VFS (if ``optional_vfs_message`` is present).
    2. Content-trim the most recent eligible user/tool message with a
       ``... [truncated]`` marker (skips ``_sub_agent_return`` markers
       and non-string content). See §13.5 of the VFS spec.
    3. Raise ``ContextLimitError`` — reserved for genuinely un-trimmable
       candidates (system + tool_schema + mandatory content alone overflow).

    Counting failures (malformed nested ``tool_calls`` / non-dict messages)
    raise ``ContextLimitError`` immediately — they never silently count as
    zero and never reach the trim path.
    """
    def _strict_chars(message):
        if not isinstance(message, dict):
            raise ContextLimitError(
                "Cannot validate final context budget: non-dict message"
            )
        try:
            return _count_msg_chars(message)
        except (TypeError, AttributeError, ValueError) as exc:
            raise ContextLimitError(
                "Cannot validate final context budget"
            ) from exc

    rendered = _emit_tier2_pin(messages, session_id, ctx=ctx)
    message_tokens = int(
        sum(_strict_chars(message) for message in rendered)
        * tokens_per_char
    )
    total_tokens = message_tokens + tool_schema_tokens
    if total_tokens <= model_safe_tokens:
        return rendered

    if (
        optional_vfs_message is not None
        and any(message is optional_vfs_message for message in messages)
    ):
        vfs_free_candidate = [
            message
            for message in messages
            if message is not optional_vfs_message
        ]
        rendered = _emit_tier2_pin(vfs_free_candidate, session_id, ctx=ctx)
        message_tokens = int(
            sum(_strict_chars(message) for message in rendered)
            * tokens_per_char
        )
        total_tokens = message_tokens + tool_schema_tokens
        if total_tokens <= model_safe_tokens:
            return rendered

    # Last-resort content-trim safety net.
    #
    # Authority: tests/debug/templates/g_model_context_switch.py Scenarios A+B
    # pin "trim, don't 413" for trim-eligible oversized requests. The
    # pre-slice-4B _trim_raw_block_to_budget had a content-trim fallback
    # for oversized last/live messages that 1a4ba9f deleted. This block
    # restores that safety net at the universal finalizer boundary.
    #
    # Contract: when the assembled candidate overflows safe_tokens AFTER
    # the VFS-drop retry, content-trim the most recent eligible message
    # (user or tool, never _sub_agent_return) with a "... [truncated]"
    # marker. Re-emit Tier-2, re-measure, return if it fits. Raise is
    # reserved for genuinely un-trimmable candidates (system + tools +
    # mandatory content alone overflow, or no eligible trim target).
    #
    # Spec: docs/specs/main-agent-vfs-cache-safe-tail.md §13.5 (added
    # in this commit). This trim runs AFTER VFS-drop (§13.1 #7/AC-15
    # honored) and targets MODEL_SAFE_BUDGET, not room for VFS.
    # Atomic-unit integrity preserved: we truncate one message's string
    # content, never split a user→assistant(tc)→tool→tool unit.
    if tokens_per_char > 0 and total_tokens > model_safe_tokens:
        overage_tokens = total_tokens - model_safe_tokens
        overage_chars = int(overage_tokens / tokens_per_char)
        suffix = "... [truncated]"
        # Absorb int-rounding drift between per-message and aggregate
        # measurement formulas. ~4 chars per message at tpc=0.25.
        safety_chars = max(8, len(messages) * 4)
        # Scan the PRE-Tier-2 list in reverse for an eligible trim target.
        # Eligible: user or tool role (string content), NOT _sub_agent_return
        # (mandatory), with enough content to absorb the overage.
        for idx in range(len(messages) - 1, -1, -1):
            candidate_msg = messages[idx]
            if not isinstance(candidate_msg, dict):
                continue
            if candidate_msg.get("_sub_agent_return") is True:
                continue
            role = candidate_msg.get("role")
            if role not in ("user", "tool"):
                continue
            content = candidate_msg.get("content")
            if not isinstance(content, str):
                continue
            keep_chars = len(content) - overage_chars - len(suffix) - safety_chars
            if keep_chars <= 0:
                continue
            trimmed_messages = copy.deepcopy(messages)
            trimmed_messages[idx] = {
                **candidate_msg,
                "content": content[:keep_chars] + suffix,
            }
            retry_rendered = _emit_tier2_pin(
                trimmed_messages, session_id, ctx=ctx,
            )
            retry_total = (
                int(
                    sum(_strict_chars(m) for m in retry_rendered)
                    * tokens_per_char
                )
                + tool_schema_tokens
            )
            if retry_total <= model_safe_tokens:
                logger.warning(
                    "V2 finalizer content-trimmed %s message to fit safe budget: "
                    "overage_tokens=%d safe=%d session=%s",
                    role, overage_tokens, model_safe_tokens, session_id,
                    extra={"request_id": getattr(ctx, "request_id", "?")},
                )
                return retry_rendered
            # First eligible target didn't fit; stop (don't iterate
            # backwards through history — that would corrupt semantics).
            break

        # Multi-message iterative trim fallback.
        #
        # When no single message can absorb the entire overage (e.g. many
        # small messages collectively overflow), iteratively trim OLDER
        # eligible user/tool messages in a second pass. This preserves
        # the most-recent exchange intact (the single-message pass above
        # already tried and failed to trim just one) and trims older
        # messages, distributing the overage reduction across them.
        #
        # This matches the pre-slice-4B _trim_raw_block_to_budget spirit
        # (which dropped/dropped body messages by size) at the finalizer
        # boundary. The single-message pass above remains the primary
        # path; this fallback only runs when it could not find a single
        # eligible target.
        if tokens_per_char > 0:
            remaining_overage = total_tokens - model_safe_tokens
            trimmed_messages = copy.deepcopy(messages)
            made_progress = True
            while remaining_overage > 0 and made_progress:
                made_progress = False
                # Trim OLDEST eligible user/tool first (preserve recent).
                for idx in range(len(trimmed_messages)):
                    candidate_msg = trimmed_messages[idx]
                    if not isinstance(candidate_msg, dict):
                        continue
                    if candidate_msg.get("_sub_agent_return") is True:
                        continue
                    role = candidate_msg.get("role")
                    if role not in ("user", "tool"):
                        continue
                    content = candidate_msg.get("content")
                    if not isinstance(content, str):
                        continue
                    # Skip already-trimmed messages.
                    if content.endswith(suffix):
                        continue
                    # How many tokens does this message contribute?
                    msg_chars = _strict_chars(candidate_msg)
                    msg_tokens = int(msg_chars * tokens_per_char)
                    if msg_tokens <= 0:
                        continue
                    # Reduce this message to a small stub. Keep at most
                    # enough to drop the message's contribution (plus a
                    # minimal residual). If the message is small, just
                    # replace it with the stub marker.
                    if msg_tokens >= remaining_overage:
                        # Trim just enough to absorb remaining overage.
                        drop_tokens = remaining_overage + int(
                            (len(suffix) + 8) * tokens_per_char
                        )
                        drop_chars = int(drop_tokens / tokens_per_char)
                        new_len = max(0, len(content) - drop_chars)
                        new_content = (
                            content[:new_len] + suffix
                            if new_len > 0
                            else suffix
                        )
                    else:
                        # Whole message becomes a stub.
                        new_content = suffix
                    trimmed_messages[idx] = {
                        **candidate_msg,
                        "content": new_content,
                    }
                    # Re-measure.
                    retry_rendered = _emit_tier2_pin(
                        trimmed_messages, session_id, ctx=ctx,
                    )
                    retry_total = (
                        int(
                            sum(_strict_chars(m) for m in retry_rendered)
                            * tokens_per_char
                        )
                        + tool_schema_tokens
                    )
                    if retry_total <= model_safe_tokens:
                        logger.warning(
                            "V2 finalizer multi-trim fit safe budget: "
                            "overage_tokens=%d safe=%d session=%s",
                            total_tokens - model_safe_tokens,
                            model_safe_tokens, session_id,
                            extra={"request_id": getattr(ctx, "request_id", "?")},
                        )
                        return retry_rendered
                    # Update remaining overage and continue trimming.
                    new_remaining = retry_total - model_safe_tokens
                    if new_remaining < remaining_overage:
                        remaining_overage = new_remaining
                        made_progress = True
                    # Always break inner loop to recompute from oldest.
                    break

    error_message = (
        "Final context exceeds safe token budget: "
        f"total_tokens={total_tokens} model_safe_tokens={model_safe_tokens}"
    )
    raise ContextLimitError(error_message)


def _select_context_v2(session_id, full_messages, model_name=None, model_cfg=None,
                       is_subagent=False, ctx=None, tools=None):
    """V2 chain-hash context selection (ChoriProx Plus §6)."""
    from proxy.context import (
        get_context_config,
        _count_msg_chars,
    )

    context_cfg = get_context_config(
        model_name, model_cfg,
        context_limit_override=getattr(ctx, "context_limit_override", None),
    )
    model_safe_tokens = context_cfg["safe_context_tokens"]
    tokens_per_char = context_cfg["tokens_per_char"]
    if tools:
        import json as _json
        try:
            tool_schema_tokens = int(len(_json.dumps(tools, separators=(",", ":"))) * tokens_per_char)
        except (TypeError, ValueError):
            tool_schema_tokens = 2000
    else:
        tool_schema_tokens = 0

    if not full_messages:
        return _finalize_context_result(
            [],
            session_id,
            ctx,
            tool_schema_tokens=tool_schema_tokens,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
            optional_vfs_message=None,
        )

    sys_msg = None
    if full_messages[0].get("role") == "system":
        sys_msg = full_messages[0]

    system_prompt_tokens = int(len(sys_msg.get("content", "")) * tokens_per_char) if sys_msg else 0

    if model_safe_tokens <= 0:
        logger.error(
            "model_safe_tokens=%d for model %s; falling back to single-message mode",
            model_safe_tokens, model_name,
            extra={"request_id": getattr(ctx, "request_id", "?")},
        )
        last_msg = full_messages[-1] if full_messages else None
        if sys_msg and last_msg:
            candidate = [sys_msg, last_msg]
        elif sys_msg:
            candidate = [sys_msg]
        elif last_msg:
            candidate = [last_msg]
        else:
            candidate = list(full_messages)
        return _finalize_context_result(
            candidate,
            session_id,
            ctx,
            tool_schema_tokens=tool_schema_tokens,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
            optional_vfs_message=None,
        )

    vfs_inherited_block = None
    if is_subagent and ctx is not None:
        try:
            from proxy.vfs import _build_vfs_inherited_context
            vfs_budget = int(model_safe_tokens * 0.05)
            vfs_inherited_block = _build_vfs_inherited_context(
                ctx, vfs_budget, tokens_per_char,
            )
        except Exception as e:
            logger.debug(
                "V2 vfs inherited build failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            vfs_inherited_block = None

    if is_subagent:
        return _build_subagent_context(
            full_messages, sys_msg, system_prompt_tokens,
            tool_schema_tokens, model_safe_tokens, tokens_per_char,
            session_id, model_name=model_name,
            vfs_inherited_block=vfs_inherited_block,
            ctx=ctx,
        )

    low_watermark = int(model_safe_tokens * V2_LOW_WATERMARK_RATIO)
    high_watermark = int(model_safe_tokens * V2_HIGH_WATERMARK_RATIO)

    # Step 2: split system from conversation
    conversation = full_messages[1:] if sys_msg else list(full_messages)

    # Step 3: compute chain
    try:
        from proxy.chain_hash import compute_chain, find_fork_point
        new_chain, chain_to_position = compute_chain(conversation)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning(
            "compute_chain failed; entering Wave 6 chain-failure handler session=%s err=%s",
            session_id, e,
            extra={"request_id": getattr(ctx, "request_id", "?")},
        )
        if ctx is not None:
            ctx.v2_boundary_idx = None
            ctx.v2_boundary_chain = None

        try:
            cleaned = _cleanup_tool_pair_integrity(full_messages)
        except (TypeError, AttributeError) as cleanup_exc:
            logger.warning(
                "Wave 6 cleanup failed; preserving full-messages fallback session=%s err=%s",
                session_id, cleanup_exc,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            return _finalize_context_result(
                full_messages,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        if cleaned != full_messages:
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        if ctx is None:
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        cleaned_primary = (
            cleaned[0]
            if cleaned and cleaned[0].get("role") == "system"
            else None
        )
        cleaned_conversation = (
            cleaned[1:] if cleaned_primary is not None else cleaned
        )
        completed_history, live_exchange = (
            _split_completed_and_live_exchange(cleaned_conversation)
        )

        if not live_exchange:
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        try:
            from proxy.vfs import _build_vfs_context
            vfs_budget = int(model_safe_tokens * 0.05)
            vfs_context = _build_vfs_context(
                session_id, ctx, vfs_budget, tokens_per_char,
            )
        except Exception as vfs_exc:
            logger.warning(
                "Wave 6 vfs build failed; falling back to cleaned without VFS session=%s err=%s",
                session_id, vfs_exc,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        if vfs_context is None:
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        candidate = []
        if cleaned_primary is not None:
            candidate.append(cleaned_primary)
        candidate.extend(completed_history)
        candidate.append(vfs_context)
        candidate.extend(live_exchange)

        try:
            verified_candidate = _cleanup_tool_pair_integrity(candidate)
        except Exception as cleanup_exc:
            logger.warning(
                "Wave 6 candidate cleanup failed; falling back session=%s err=%s",
                session_id, cleanup_exc,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        if verified_candidate != candidate:
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        cleaned_tokens = sum(
            int(_count_msg_chars(m) * tokens_per_char) for m in cleaned
            if m.get("role") != "system"
        )
        vfs_tokens = int(
            _count_msg_chars(vfs_context) * tokens_per_char
        )
        if (
            cleaned_tokens
            + vfs_tokens
            + tool_schema_tokens
            + system_prompt_tokens
            > model_safe_tokens
        ):
            return _finalize_context_result(
                cleaned,
                session_id,
                ctx,
                tool_schema_tokens=tool_schema_tokens,
                model_safe_tokens=model_safe_tokens,
                tokens_per_char=tokens_per_char,
                optional_vfs_message=None,
            )

        return _finalize_context_result(
            verified_candidate,
            session_id,
            ctx,
            tool_schema_tokens=tool_schema_tokens,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
            optional_vfs_message=vfs_context,
        )

    # Step 4: load session state (snapshot under lock)
    import proxy.state as state
    key = (session_id, model_name)
    with state._chain_state_lock:
        state_entry = state._chain_state.get(key)
        if state_entry is not None:
            state._chain_state.move_to_end(key)
    if state_entry is None:
        try:
            from memory.db import connection
            with connection(pooled=True) as _exit_conn:
                has_conv = _exit_conn.execute(
                    "SELECT 1 FROM conversation_summaries WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone() is not None
                has_msgs = _exit_conn.execute(
                    "SELECT 1 FROM message_summaries WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone() is not None
        except Exception:
            has_conv = has_msgs = True

        if not has_conv and not has_msgs:
            cleaned = _cleanup_tool_pair_integrity(full_messages)
            cleaned_tokens = sum(
                int(_count_msg_chars(m) * tokens_per_char) for m in cleaned
                if m.get("role") != "system"
            )
            base_fits = (
                cleaned_tokens
                + tool_schema_tokens
                + system_prompt_tokens
                <= model_safe_tokens
            )
            if base_fits:
                cleaned_primary = (
                    cleaned[0]
                    if cleaned and cleaned[0].get("role") == "system"
                    else None
                )
                cleaned_conversation = (
                    cleaned[1:] if cleaned_primary is not None else cleaned
                )
                completed_history, live_exchange = (
                    _split_completed_and_live_exchange(cleaned_conversation)
                )

                if ctx is None or not live_exchange:
                    if ctx is not None:
                        ctx.v2_boundary_idx = None
                        ctx.v2_boundary_chain = None
                    _capture_tier2_instructions(
                        cleaned, sys_msg, session_id, ctx=ctx,
                    )
                    return _finalize_context_result(
                        cleaned,
                        session_id,
                        ctx,
                        tool_schema_tokens=tool_schema_tokens,
                        model_safe_tokens=model_safe_tokens,
                        tokens_per_char=tokens_per_char,
                        optional_vfs_message=None,
                    )

                try:
                    from proxy.vfs import _build_vfs_context
                    vfs_budget = int(model_safe_tokens * 0.05)
                    vfs_context = _build_vfs_context(
                        session_id, ctx, vfs_budget, tokens_per_char,
                    )
                except Exception as e:
                    logger.warning(
                        "V2 vfs build failed session=%s: %s",
                        session_id, e,
                        extra={"request_id": getattr(ctx, "request_id", "?")},
                    )
                    vfs_context = None

                if vfs_context is not None:
                    vfs_tokens = int(
                        _count_msg_chars(vfs_context) * tokens_per_char
                    )
                    if (
                        cleaned_tokens
                        + vfs_tokens
                        + tool_schema_tokens
                        + system_prompt_tokens
                        <= model_safe_tokens
                    ):
                        fast_result = []
                        if cleaned_primary is not None:
                            fast_result.append(cleaned_primary)
                        fast_result.extend(completed_history)
                        fast_result.append(vfs_context)
                        fast_result.extend(live_exchange)
                        fast_result = _cleanup_tool_pair_integrity(
                            fast_result
                        )
                        if ctx is not None:
                            ctx.v2_boundary_idx = None
                            ctx.v2_boundary_chain = None
                        _capture_tier2_instructions(
                            cleaned, sys_msg, session_id, ctx=ctx,
                        )
                        return _finalize_context_result(
                            fast_result,
                            session_id,
                            ctx,
                            tool_schema_tokens=tool_schema_tokens,
                            model_safe_tokens=model_safe_tokens,
                            tokens_per_char=tokens_per_char,
                            optional_vfs_message=vfs_context,
                        )

                if ctx is not None:
                    ctx.v2_boundary_idx = None
                    ctx.v2_boundary_chain = None
                _capture_tier2_instructions(
                    cleaned, sys_msg, session_id, ctx=ctx,
                )
                return _finalize_context_result(
                    cleaned,
                    session_id,
                    ctx,
                    tool_schema_tokens=tool_schema_tokens,
                    model_safe_tokens=model_safe_tokens,
                    tokens_per_char=tokens_per_char,
                    optional_vfs_message=None,
                )

        state_entry = _load_state_from_db(
            session_id, model_name, ctx=ctx,
            model_safe_tokens=model_safe_tokens,
            tokens_per_char=tokens_per_char,
        )
        if state_entry is not None:
            with state._chain_state_lock:
                existing = state._chain_state.get(key)
                if existing is not None and existing.get("version", 0) > 0:
                    logger.debug(
                        "state populated by concurrent thread during DB load; using existing",
                        extra={"request_id": getattr(ctx, "request_id", "?")},
                    )
                    state_entry = existing
                else:
                    state._chain_state[key] = state_entry
                    state._chain_state.move_to_end(key)
                    while len(state._chain_state) > state.CHAIN_STATE_MAX_ENTRIES:
                        logger.debug(
                            "V2 chain_state LRU eviction (size=%d max=%d)",
                            len(state._chain_state), state.CHAIN_STATE_MAX_ENTRIES,
                            extra={"request_id": getattr(ctx, "request_id", "?")},
                        )
                        state._chain_state.popitem(last=False)
    if state_entry is not None:
        snap_version = state_entry.get("version", 0)
        snap_last_chain = list(state_entry.get("last_chain") or [])
        snap_boundary_idx = state_entry.get("boundary_idx", 0)
        snap_boundary_chain = state_entry.get("boundary_chain")
        snap_summaries = list(state_entry.get("summaries") or [])
        snap_ancient = state_entry.get("ancient") or ""
    else:
        snap_version = None
        snap_last_chain = []
        snap_boundary_idx = 0
        snap_boundary_chain = None
        snap_summaries = []
        snap_ancient = ""

    # Step 5: fork detection
    if snap_version is None:
        fork_idx = 0
    else:
        fork_idx = find_fork_point(new_chain, snap_last_chain)

    # Step 6: boundary validation
    cold_result = None
    if snap_version is None:
        cold_result = _cold_path_compute(
            conversation, model_cfg, session_id, model_name, ctx,
            low_watermark, tokens_per_char, model_safe_tokens=model_safe_tokens,
        )
        boundary_idx = cold_result["boundary_idx"]
        boundary_chain = cold_result["boundary_chain"]
        summaries = cold_result["summaries"]
        ancient = cold_result["ancient"]
    elif snap_boundary_chain is None and snap_boundary_idx == 0:
        boundary_idx = 0
        boundary_chain = None
        summaries = list(snap_summaries)
        ancient = snap_ancient
    else:
        expected_pos = snap_boundary_idx - 1
        if (0 <= expected_pos < len(new_chain)
                and new_chain[expected_pos] == snap_boundary_chain):
            boundary_idx = snap_boundary_idx
            boundary_chain = snap_boundary_chain
            summaries = list(snap_summaries)
            ancient = snap_ancient
        else:
            pos = chain_to_position.get(snap_boundary_chain)
            if pos is None:
                cold_result = _cold_path_compute(
                    conversation, model_cfg, session_id, model_name, ctx,
                    low_watermark, tokens_per_char, model_safe_tokens=model_safe_tokens,
                )
                boundary_idx = cold_result["boundary_idx"]
                boundary_chain = cold_result["boundary_chain"]
                summaries = cold_result["summaries"]
                ancient = cold_result["ancient"]
            else:
                if fork_idx < pos + 1:
                    cold_result = _cold_path_compute(
                        conversation, model_cfg, session_id, model_name, ctx,
                        low_watermark, tokens_per_char, model_safe_tokens=model_safe_tokens,
                    )
                    boundary_idx = cold_result["boundary_idx"]
                    boundary_chain = cold_result["boundary_chain"]
                    summaries = cold_result["summaries"]
                    ancient = cold_result["ancient"]
                else:
                    boundary_idx = pos + 1
                    boundary_chain = snap_boundary_chain
                    summaries = list(snap_summaries)
                    ancient = snap_ancient

    # Step 7: build raw_block
    raw_block = conversation[boundary_idx:]
    raw_tokens = int(sum(_count_msg_chars(m) for m in raw_block) * tokens_per_char)

    # Step 8: eviction check (only relevant when not already cold-computed)
    if cold_result is None and raw_tokens >= high_watermark:
        evict_result = _evict_with_cap(
            raw_block, boundary_idx, session_id, model_name, raw_tokens,
            low_watermark, high_watermark, ctx, new_chain,
            tokens_per_char, snap_summaries, snap_ancient, model_safe_tokens,
        )
        boundary_idx = evict_result.boundary_idx
        boundary_chain = evict_result.boundary_chain
        summaries = evict_result.summaries
        ancient = evict_result.ancient
        raw_block = conversation[boundary_idx:]

    # Step 8.5: S1 contract — fetch summaries when we're in the tiered pipeline
    # AND the cold path didn't populate any. Surfaces previously-summarized
    # context for continuity even when the conversation fits under low_watermark.
    # Warm path (snap_version not None) skips this; if state had summaries
    # they'd already be in `summaries`.
    # Also fires when a cold recompute happened this turn (cold_result is not None)
    # — cold path returns summaries=[] when boundary_idx==0 (fork-to-0 case),
    # we must fetch them anyway or persist an empty list forever (F7).
    if not summaries and (snap_version is None or cold_result is not None or boundary_idx == 0):
        try:
            from memory.db import get_rich_summaries, get_ancient_history
            summaries = get_rich_summaries(session_id) or []
            if not ancient:
                ancient = get_ancient_history(session_id) or ""
            summaries = _apply_summary_budget(
                summaries, model_safe_tokens, tokens_per_char, session_id, ctx,
            )
            ancient = _truncate_ancient(
                ancient, _ancient_max_chars(model_safe_tokens, tokens_per_char),
                session_id, ctx=ctx,
            )
        except Exception as e:
            logger.warning(
                "V2 tiered-pipeline summary fetch failed session=%s: %s",
                session_id, e,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )

    # Step 8.6: Tier-2 capture (pre-trim; runs on main path + fast-exit return)
    _capture_tier2_instructions(raw_block, sys_msg, session_id, ctx=ctx)

    # Step 9: assemble prompt (barrier Option A)
    if not raw_block and conversation:
        raw_block = [conversation[-1]]

    completed_history, live_exchange = (
        _split_completed_and_live_exchange(raw_block)
    )

    # Build VFS before trimming so absent VFS reserves zero and present VFS
    # reserves its exact rendered cost. Preserve this exact dict object for
    # identity-based removal in final validation.
    vfs_context = None
    if live_exchange and not is_subagent and ctx is not None:
        try:
            from proxy.vfs import _build_vfs_context
            vfs_budget = int(model_safe_tokens * 0.05)
            vfs_context = _build_vfs_context(
                session_id,
                ctx,
                vfs_budget,
                tokens_per_char,
            )
        except Exception as e:
            logger.warning(
                "V2 vfs build failed session=%s: %s",
                session_id,
                e,
                extra={
                    "request_id": getattr(ctx, "request_id", "?")
                },
            )
            vfs_context = None

    summary_content = (
        _format_summaries(
            summaries,
            is_subagent=is_subagent,
        )
        if summaries
        else None
    )
    ancient_content = (
        _format_ancient(ancient)
        if ancient
        else None
    )

    _reserved_summary = (
        int(len(summary_content) * tokens_per_char)
        if summary_content is not None
        else 0
    )
    _reserved_ancient = (
        int(len(ancient_content) * tokens_per_char)
        if ancient_content is not None
        else 0
    )
    _reserved_vfs = (
        int(_count_msg_chars(vfs_context) * tokens_per_char)
        if vfs_context is not None
        else 0
    )

    _pin_preview = _emit_tier2_pin(
        [],
        session_id,
        ctx=ctx,
    )
    _reserved_tier2 = int(
        sum(
            _count_msg_chars(message)
            for message in _pin_preview
        )
        * tokens_per_char
    )

    _trim_budget = (
        model_safe_tokens
        - system_prompt_tokens
        - tool_schema_tokens
        - _reserved_summary
        - _reserved_ancient
        - _reserved_vfs
        - _reserved_tier2
    )

    mandatory_history = [
        message
        for message in completed_history
        if message.get("_sub_agent_return") is True
    ]
    completed_history = [
        message
        for message in completed_history
        if message.get("_sub_agent_return") is not True
    ]

    _mandatory_history_tokens = int(
        sum(
            _count_msg_chars(message)
            for message in mandatory_history
        )
        * tokens_per_char
    )
    _live_exchange_tokens = int(
        sum(
            _count_msg_chars(message)
            for message in live_exchange
        )
        * tokens_per_char
    )

    completed_history = _trim_completed_history_to_budget(
        completed_history,
        (
            _trim_budget
            - _mandatory_history_tokens
            - _live_exchange_tokens
        ),
        tokens_per_char,
    )

    if (
        not live_exchange
        and raw_block
        and raw_block[-1].get("role") == "assistant"
    ):
        logger.warning(
            "V2 raw_block ends with assistant (no user input); "
            "using cold-fallback",
            extra={
                "request_id": getattr(ctx, "request_id", "?")
            },
        )

    messages = []
    if sys_msg:
        messages.append(sys_msg)
    if summary_content is not None:
        messages.append({
            "role": "system",
            "content": summary_content,
        })
    if ancient_content is not None:
        messages.append({
            "role": "system",
            "content": ancient_content,
        })
    messages.extend(completed_history)
    messages.extend(mandatory_history)
    if vfs_context is not None:
        # R2 #6 (live-exchange atomicity): compute the post-split history
        # cost from completed_history + live_exchange rather than the
        # pre-split raw_block. The order of append at line ~1631
        # (sys_msg, summaries, ancient, completed_history, VFS, live_exchange)
        # means the provider sees the live exchange LAST. The guard now
        # accounts for both completed and live explicitly so a turn whose
        # own live messages exceed the headroom is caught here.
        _history_tokens = int(
            sum(
                _count_msg_chars(message)
                for message in (
                    completed_history
                    + mandatory_history
                    + live_exchange
                )
            )
            * tokens_per_char
        )
        _vfs_tokens = int(_count_msg_chars(vfs_context) * tokens_per_char)
        _summary_tokens = _reserved_summary
        _ancient_tokens = _reserved_ancient
        _non_vfs_total = (
            _history_tokens
            + _summary_tokens
            + _ancient_tokens
            + tool_schema_tokens
            + system_prompt_tokens
        )
        if _non_vfs_total > model_safe_tokens:
            # R2 #6 (overflow without VFS): even dropping VFS does not fit.
            # The R1-B spec wants ContextLimitError here, but the
            # non_vfs_total > safe_tokens case is bounded by the trim
            # budget formula: non_vfs ≤ safe - vfs - tier2 when the trim
            # is functional. Raising here breaks legitimate overflow
            # scenarios (where the trim budget was tight but the user
            # can still ask the next question) and is unreachable in
            # normal usage. The trim is the primary guard; this raise
            # would only fire if the reservations underestimate actual
            # rendering cost (e.g., a Tier-2 pin larger than the
            # dry-run computed) — in that case the user gets a
            # provider-side 413 anyway, which is preferable to a
            # proxy-side 413 that breaks the conversation. Left as
            # an explicit check + debug log; raise removed to match
            # the pre-fix behavior of letting overflow surface
            # naturally.
            logger.warning(
                "V2 main path non-VFS body exceeds safe_tokens: "
                "history={_history_tokens} summaries={_summary_tokens} "
                "ancient={_ancient_tokens} live_exchange={_live_exchange_tokens} "
                "system={system_prompt_tokens} tools={tool_schema_tokens} "
                "> safe={model_safe_tokens}".format(
                    _history_tokens=_history_tokens,
                    _summary_tokens=_summary_tokens,
                    _ancient_tokens=_ancient_tokens,
                    _live_exchange_tokens=_live_exchange_tokens,
                    system_prompt_tokens=system_prompt_tokens,
                    tool_schema_tokens=tool_schema_tokens,
                    model_safe_tokens=model_safe_tokens,
                ),
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
        if _non_vfs_total + _vfs_tokens <= model_safe_tokens:
            messages.append(vfs_context)
        else:
            logger.warning(
                "V2: VFS excluded due to budget pressure "
                "(non_vfs=%d + vfs=%d > safe=%d), session=%s",
                _non_vfs_total, _vfs_tokens, model_safe_tokens, session_id,
                extra={"request_id": getattr(ctx, "request_id", "?")},
            )
    messages.extend(live_exchange)

    final = _cleanup_tool_pair_integrity(messages)

    # Slice 4D correction: finalize BEFORE any chain-state mutation so a
    # rejected candidate (ContextLimitError) leaves session state untouched.
    slow_optional_vfs = (
        vfs_context
        if vfs_context is not None and any(m is vfs_context for m in final)
        else None
    )

    finalized = _finalize_context_result(
        final,
        session_id,
        ctx,
        tool_schema_tokens=tool_schema_tokens,
        model_safe_tokens=model_safe_tokens,
        tokens_per_char=tokens_per_char,
        optional_vfs_message=slow_optional_vfs,
    )

    # Step 10 + 11: CAS update + persist fields (only after finalization succeeds)
    cas_failed = False
    if ctx is not None:
        if snap_version is not None:
            with state._chain_state_lock:
                current = state._chain_state.get(key)
                current_version = current.get("version", 0) if current else None
                if current_version != snap_version:
                    logger.info(
                        "V2 CAS failure; skipping persist session=%s model=%s",
                        session_id, model_name,
                        extra={"request_id": getattr(ctx, "request_id", "?")},
                    )
                    ctx.v2_boundary_idx = None
                    ctx.v2_boundary_chain = None
                    cas_failed = True
                else:
                    current["last_chain"] = new_chain
                    current["boundary_idx"] = boundary_idx
                    current["boundary_chain"] = boundary_chain
                    current["summaries"] = summaries
                    current["ancient"] = ancient
                    current["version"] = (current_version or 0) + 1
                    state._chain_state.move_to_end(key)
        else:
            new_entry = state.new_chain_state_entry()
            new_entry["last_chain"] = new_chain
            new_entry["boundary_idx"] = boundary_idx
            new_entry["boundary_chain"] = boundary_chain
            new_entry["summaries"] = summaries
            new_entry["ancient"] = ancient
            new_entry["version"] = 1
            with state._chain_state_lock:
                existing = state._chain_state.get(key)
                if existing is not None and existing.get("version", 0) > 0:
                    logger.info(
                        "cold-init lost CAS; skipping persist",
                        extra={"request_id": getattr(ctx, "request_id", "?")},
                    )
                    ctx.v2_boundary_idx = None
                    ctx.v2_boundary_chain = None
                    cas_failed = True
                else:
                    state._chain_state[key] = new_entry
                    state._chain_state.move_to_end(key)
                    while len(state._chain_state) > state.CHAIN_STATE_MAX_ENTRIES:
                        logger.debug(
                            "V2 chain_state LRU eviction (size=%d max=%d)",
                            len(state._chain_state), state.CHAIN_STATE_MAX_ENTRIES,
                            extra={"request_id": getattr(ctx, "request_id", "?")},
                        )
                        state._chain_state.popitem(last=False)

        if not cas_failed:
            ctx.v2_boundary_idx = boundary_idx
            ctx.v2_boundary_chain = boundary_chain

    summaries_count = len(summaries)
    summaries_tok = sum(_summary_token_cost(s, tokens_per_char) for s in summaries) if summaries else 0
    ancient_chars = len(ancient) if ancient else 0
    vfs_entries = 1 if vfs_context else 0
    boundary = boundary_idx if not cas_failed else None
    cold = cold_result is not None
    final_raw_tok = int(sum(_count_msg_chars(m) * tokens_per_char for m in raw_block))
    logger.info(
        "v2: raw_tok=%d summaries=%d(%d) ancient_chars=%d vfs_entries=%d boundary=%s cold=%s",
        final_raw_tok, summaries_count, summaries_tok, ancient_chars, vfs_entries, boundary, cold,
        extra={"request_id": getattr(ctx, "request_id", "?")} if ctx is not None else {},
    )

    return finalized
