import logging
import os
import re
import threading
import time
import uuid

from memory.sota_client import (
    SyncResult,
    SOTATimeoutError,
    SOTAAPIError,
    SOTAParseError,
    _validate_extraction_items,
    _sanitize_text,
)

from memory.persistence import (
    _store_entities_and_facts,
    _store_memories,
    _get_graph,
)

from memory.batch_manager import (
    _prepare_extraction_batch,
    _update_batch_status,
    _update_batch_extraction,
    _set_batch_processing,
    _update_batch_status_if,
    _claim_batch_for_extraction,
    _raw_sql,
)

from memory._db import get_db as _db

logger = logging.getLogger("memory.sync")


# Bounded queue for ancient-compress dispatches. The underlying
# ThreadPoolExecutor has an unbounded internal queue, so under a sustained
# eviction burst + slow SOTA the queue grows without bound (2GB+ observed in
# round 4 review). This counter caps pending dispatches at 100. Excess
# dispatches are dropped (caller logs at debug); the truncated summaries
# remain in DB and will be re-dispatched on the next eviction.
_ANCIENT_COMPRESS_MAX_PENDING = 100
_ancient_compress_pending = 0
_ancient_compress_lock = threading.Lock()


def _exchange_range_to_message_range(session_id, start_exchange, end_exchange):
    """Map exchange index range to message index range."""
    from memory.db_pool import connection
    with connection(pooled=True) as conn:
        row = conn.execute(
            "SELECT MIN(message_index) as start_idx, MAX(message_index) as end_idx "
            "FROM messages WHERE session_id = ? AND exchange_index >= ? AND exchange_index <= ?",
            (session_id, start_exchange, end_exchange)
        ).fetchone()
        return row["start_idx"], row["end_idx"]


def _get_exchange_for_message(session_id, message_index):
    """Look up exchange_index for a given message_index.
    
    Returns -1 if message not found or exchange_index is NULL.
    """
    from memory.db_pool import connection
    with connection(pooled=True) as conn:
        row = conn.execute(
            "SELECT exchange_index FROM messages WHERE session_id = ? AND message_index = ?",
            (session_id, message_index)
        ).fetchone()
        if row and row["exchange_index"] is not None:
            return row["exchange_index"]
        return -1


MAX_SYNC_BACKLOG = 100  # If more uncached exchanges than this, only process newest N

# ── In-memory reasoning cache ─────────────────────────────────────────
# Reasoning content exists only momentarily during SSE streaming.
# We stash it here so the async sync loop can pick it up later.
# Guards: max 500 entries (LRU eviction), entries auto-expire after 10 min.
# Entries are consumed (popped) when _build_previous_context reads them.
import threading as _thr

_sync_locks: dict = {}
_sync_locks_lock = _thr.Lock()


def _get_sync_lock(session_id):
    with _sync_locks_lock:
        if session_id not in _sync_locks:
            _sync_locks[session_id] = _thr.Lock()
        return _sync_locks[session_id]


_ancient_locks: dict = {}
_ancient_locks_guard = _thr.Lock()


def _get_ancient_lock(session_id):
    """Background-pool coordination lock (NOT part of the global #0–#13 hierarchy).

    Local locks scoped to this module — deliberately outside the global
    proxy lock hierarchy defined in ``src/proxy/AGENTS.md``. The global
    hierarchy is #0–#13 (14 positions); this per-session lock is module-
    local state managed entirely inside ``sync.py`` and never participates
    in cross-module lock ordering.

    Acquired by background workers (running in _summary_pool / _sync_pool,
    off the request path) to serialize per-session ancient compress across
    read→SOTA→write. Never acquired by request-path code.
    """
    with _ancient_locks_guard:
        if session_id not in _ancient_locks:
            _ancient_locks[session_id] = _thr.Lock()
        return _ancient_locks[session_id]


_reasoning_cache: dict[str, dict[int, str]] = {}  # {session_id: {msg_idx: reasoning}}
_reasoning_cache_ts: dict[str, dict[int, float]] = {}  # {session_id: {msg_idx: inserted_at}}
_reasoning_cache_lock = _thr.Lock()
_MAX_REASONING_ENTRIES = 500
_REASONING_TTL_SECONDS = 600  # 10 min

def _cache_reasoning(session_id: str, msg_idx: int, reasoning: str):
    """Store reasoning content for later consumption by summarization."""
    if not reasoning or not reasoning.strip():
        return
    with _reasoning_cache_lock:
        session_cache = _reasoning_cache.setdefault(session_id, {})
        ts_cache = _reasoning_cache_ts.setdefault(session_id, {})
        now = time.time()
        # LRU eviction: if too many entries across all sessions, drop oldest
        total = sum(len(v) for v in _reasoning_cache.values())
        if total >= _MAX_REASONING_ENTRIES:
            # Drop the oldest entry from the largest session
            largest_session = max(_reasoning_cache, key=lambda k: len(_reasoning_cache[k]))
            oldest_idx = min(_reasoning_cache[largest_session].keys())
            del _reasoning_cache[largest_session][oldest_idx]
            _reasoning_cache_ts.get(largest_session, {}).pop(oldest_idx, None)
            if not _reasoning_cache[largest_session]:
                del _reasoning_cache[largest_session]
                _reasoning_cache_ts.pop(largest_session, None)
        # TTL eviction: drop expired entries from this session before adding
        expired = [idx for idx, ts in ts_cache.items() if now - ts > _REASONING_TTL_SECONDS]
        for idx in expired:
            session_cache.pop(idx, None)
            ts_cache.pop(idx, None)
            total -= 1
        session_cache[msg_idx] = reasoning
        ts_cache[msg_idx] = now


def _pop_reasoning(session_id: str, msg_idx: int) -> str:
    """Consume reasoning content for a message (pop so it's freed). Also cleans expired entries."""
    with _reasoning_cache_lock:
        # TTL-based eviction: drop entries older than _REASONING_TTL_SECONDS
        now = time.time()
        expired_sessions = []
        for sid, ts_cache in _reasoning_cache_ts.items():
            expired = [idx for idx, ts in ts_cache.items() if now - ts > _REASONING_TTL_SECONDS]
            for idx in expired:
                _reasoning_cache.get(sid, {}).pop(idx, None)
                ts_cache.pop(idx, None)
            if not ts_cache:
                expired_sessions.append(sid)
        for sid in expired_sessions:
            _reasoning_cache_ts.pop(sid, None)
            _reasoning_cache.pop(sid, None)
        # LRU size-cap eviction
        total = sum(len(v) for v in _reasoning_cache.values())
        if total > _MAX_REASONING_ENTRIES:
            oldest_to_drop = total - _MAX_REASONING_ENTRIES
            sessions_by_size = sorted(_reasoning_cache.keys(), key=lambda k: len(_reasoning_cache[k]), reverse=True)
            for sid in sessions_by_size:
                if oldest_to_drop <= 0:
                    break
                cache = _reasoning_cache.get(sid, {})
                to_drop = min(oldest_to_drop, len(cache))
                keys = sorted(cache.keys())
                for k in keys[:to_drop]:
                    del cache[k]
                    _reasoning_cache_ts.get(sid, {}).pop(k, None)
                    oldest_to_drop -= 1
                if not cache:
                    del _reasoning_cache[sid]
                    _reasoning_cache_ts.pop(sid, None)

        session_cache = _reasoning_cache.get(session_id, {})
        _reasoning_cache_ts.get(session_id, {}).pop(msg_idx, None)
        return session_cache.pop(msg_idx, "")


def _build_previous_context(rows, target_msg_index, max_chars=20000):
    """Build a '[Previous Context]' block from messages preceding target_msg_index.

    Walks backwards from the message just before the target, accumulating
    raw content until we hit max_chars (~5K tokens). Returns a formatted string
    suitable for prepending to the summarization user prompt.

    Also checks the in-memory reasoning cache for each message and includes
    reasoning content if available.

    Args:
        rows: list of dicts with keys (message_index, role, content), sorted ASC
        target_msg_index: the message being summarized — we collect only rows
                          with message_index < target_msg_index
        max_chars: character budget for the context block (default 20K ≈ 5K tokens)

    Returns:
        Formatted string like:
            [MSG #5 | user]: <content>
            [MSG #6 | assistant]:  thinking <reasoning> response <content>
        Or empty string if no preceding messages exist.
    """
    # Find the position of target in the sorted rows list
    preceding = [r for r in rows if r["message_index"] < target_msg_index]
    if not preceding:
        return ""

    # Walk backwards from the most recent preceding message, accumulating chars
    context_parts = []
    total_chars = 0
    # Pull session_id from the first row (all rows are same session)
    session_id_for_reasoning = rows[0]["session_id"] if rows else ""

    for r in reversed(preceding):
        role = r["role"]
        content = r["content"] or ""
        # Skip system messages in context (no narrative value)
        if role and role.lower().strip() == "system":
            continue

        msg_idx = r["message_index"]

        # Check reasoning cache for this message (only for assistant messages)
        reasoning = ""
        if role and role.lower().strip() == "assistant":
            reasoning = _pop_reasoning(session_id_for_reasoning, msg_idx)

        if reasoning:
            line = f"[MSG #{msg_idx} | {role}]:  thinking {reasoning[:3000]}... response {content}"
        else:
            line = f"[MSG #{msg_idx} | {role}]: {content}"

        # Truncate individual message if it alone would blow the budget
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[:remaining] + "..."
        context_parts.append(line)
        total_chars += len(line)

    # Reverse back to chronological order (oldest first)
    return "\n\n".join(reversed(context_parts))


def _identify_uncached_exchanges(session_id, start_idx, end_idx):
    """Return (uncached, all_rows) tuple for uncached exchanges.

    all_rows is the pre-fetched message snapshot for the range, reused by
    _summarize_uncached_exchanges to avoid N+1 redundant queries.
    """
    import hashlib
    import json

    from memory.db_pool import immediate_transaction
    initial_rows: list = []
    with immediate_transaction(pooled=True) as conn:
        rows = conn.execute(
            """SELECT m.session_id, m.message_index, m.role, m.content, m.content_hash,
                      m.tool_calls_json, m.tool_call_id, m.exchange_index
               FROM messages m
               WHERE m.session_id = ? AND m.message_index >= ? AND m.message_index <= ?
               ORDER BY m.message_index""",
            (session_id, start_idx, end_idx)
        ).fetchall()
        initial_rows = list(rows)  # capture before the inner block overwrites `rows`

        # Group by stored exchange_index (Phase 0 fix ensures non-NULL for new messages)
        exchanges = {}
        null_count = 0
        for row in rows:
            ei = row["exchange_index"]
            if ei is None:
                null_count += 1
                continue  # skip rows without exchange_index
            if ei not in exchanges:
                exchanges[ei] = []
            exchanges[ei].append(row)
        
        if null_count > 0:
            logger.warning("[SYNC] %d messages have NULL exchange_index in session %s", null_count, session_id)

        # ── Filter to uncached exchanges ────────────────────────────────────────
        hash_updates = []
        exchange_data = []
        for exchange in exchanges.values():
            first_row = exchange[0]
            user_role = first_row["role"]
            user_content = first_row["content"]

            if user_role != "user":
                continue
            if not user_content or not isinstance(user_content, str):
                continue

            if len(user_content.strip()) < 20:
                logger.debug(
                    "[SYNC] Exchange starting at msg #%d skipped: user message too short (%d chars) session_id=%s",
                    first_row["message_index"], len(user_content.strip()), session_id
                )
                continue

            content_hash = first_row["content_hash"]
            if not content_hash:
                content_hash = hashlib.sha256(user_content.encode()).hexdigest()[:16]
                hash_updates.append((content_hash, session_id, first_row["message_index"]))

            exchange_idx = first_row["exchange_index"]
            exchange_data.append((exchange, exchange_idx, content_hash))

        if hash_updates:
            chunk_size = 500
            for i in range(0, len(hash_updates), chunk_size):
                chunk = hash_updates[i:i + chunk_size]
                conn.executemany(
                    "UPDATE messages SET content_hash = ? WHERE session_id = ? AND message_index = ?",
                    chunk
                )

        lookup_keys = [(session_id, exchange[0]["exchange_index"], content_hash)
                       for exchange, _, content_hash in exchange_data
                       if content_hash]

        existing_summaries = {}
        if lookup_keys:
            chunk_size = 500
            for i in range(0, len(lookup_keys), chunk_size):
                chunk = lookup_keys[i:i + chunk_size]
                placeholders = ",".join(["(?, ?, ?)"] * len(chunk))
                params = []
                for sid, exidx, chash in chunk:
                    params.extend([sid, exidx, chash])

                rows = conn.execute(
                    f"""SELECT session_id, exchange_index, content_hash 
                        FROM message_summaries 
                        WHERE (session_id, exchange_index, content_hash) IN (VALUES {placeholders})""",
                    params
                ).fetchall()

                for row in rows:
                    existing_summaries[(row[0], row[1], row[2])] = True

        uncached = []
        for exchange, exchange_idx, content_hash in exchange_data:
            skip_dedup = not content_hash
            if content_hash and not skip_dedup:
                ex_idx = exchange[0]["exchange_index"]
                if existing_summaries.get((session_id, ex_idx, content_hash)):
                    logger.debug(
                        "[SYNC] Exchange %d already summarized (hash match), skipping session_id=%s",
                        ex_idx, session_id
                    )
                    continue

            uncached.append((exchange, exchange_idx, content_hash))

    # M5-4: ensure the rows are accessible by column name in downstream
    # code. SQLite rows support both index and column-name access, but
    # the conversion ensures the type contract is explicit and future
    # SQLite driver changes (e.g. a `Row` class without __contains__)
    # don't break `_build_previous_context` which uses `r["message_index"]`.
    # Use the captured initial_rows (the inner block overwrites `rows`).
    rows = [dict(r) for r in initial_rows]

    # Apply backlog cap
    if len(uncached) > MAX_SYNC_BACKLOG:
        logger.info(
            "[SYNC] %d uncached exchanges exceeds backlog cap %d, processing oldest %d session_id=%s",
            len(uncached), MAX_SYNC_BACKLOG, MAX_SYNC_BACKLOG, session_id,
        )
        uncached = uncached[:MAX_SYNC_BACKLOG]

    return uncached, rows


def _summarize_uncached_exchanges(session_id, sota_client, start_exchange, end_exchange):
    """Summarize user↔assistant exchanges that don't have a matching content-hash summary.

    Groups consecutive rows from the messages table into exchanges, where each exchange
    starts at role=user and includes all following rows (assistant, tool) until the next
    role=user or end-of-range. One summary is produced per exchange, keyed to the user
    message's message_index.

    Returns dict: {"stored": int, "found_uncached": bool, "new_cursor": int}
      stored — count of NEW summaries stored this cycle
      found_uncached — True if any uncached exchanges were identified (before summarization)
      new_cursor — highest exchange_index that is safe to advance to (the
        highest CONTIGUOUS successful prefix of `uncached`, counted from index 0).
        When any exchange in the middle fails, the cursor cannot cross it —
        failed exchanges stay in the unprocessed window for the next run.
        The caller passes this to set_last_summarized_exchange_index so the
        next sync cycle retries from the correct boundary.

        M5-4: previously, the caller advanced the cursor to `end_exchange` on
        partial success (`new_count > 0`), which permanently skipped failed
        exchanges above the cursor. This returns the new semantic: cursor
        only advances to the highest CONTIGUOUS successful prefix.
    """
    import json

    start_idx, end_idx = _exchange_range_to_message_range(session_id, start_exchange, end_exchange)
    if start_idx is None:
        return {"stored": 0, "found_uncached": False, "new_cursor": end_exchange}

    uncached, all_rows = _identify_uncached_exchanges(session_id, start_idx, end_idx)

    if not uncached:
        return {"stored": 0, "found_uncached": False, "new_cursor": end_exchange}

    if len(all_rows) > 50000:
        logger.warning(
            "[SYNC] Large message range: %d rows in sync for session %s",
            len(all_rows), session_id,
        )

    # Per-session lock and summarization loop
    lock = _get_sync_lock(session_id)

    stored = 0
    failed_exchanges = []  # Track exchanges that failed to summarize (SOTA errors)
    success_flags = [True] * len(uncached)  # parallel to uncached[]; False where SOTA failed
    for i, (exchange, exchange_idx, content_hash) in enumerate(uncached):
        with lock:
            # Build content block for this single exchange
            exchange_lines = []
            exchange_end_idx = max(row["message_index"] for row in exchange)

            for row in exchange:
                role = row["role"]
                content = row["content"] or ""
                safe_content = content.replace("[", "\\[")
                tc_id = row["tool_call_id"] or "?"
                safe_tc_id = str(tc_id or "").replace("[", "\\[")

                if role == "user":
                    exchange_lines.append(f"[user]: {safe_content}")
                elif role == "assistant":
                    tc_json = row["tool_calls_json"]
                    if tc_json:
                        if isinstance(tc_json, str):
                            try:
                                tc_json = json.loads(tc_json)
                            except Exception:
                                tc_json = None
                        if tc_json:
                            tool_names = [tc.get("function", {}).get("name", "?") for tc in tc_json]
                            exchange_lines.append(f"[assistant tool_calls]: {', '.join(tool_names)}")
                    if content:
                        exchange_lines.append(f"[assistant]: {safe_content}")
                elif role == "tool":
                    truncated = safe_content[:300]
                    if len(safe_content) > 300:
                        truncated += "... [truncated]"
                    sanitized = re.sub(r'/[^\s]{2,}(?:\.\w{1,6})?', '[path]', truncated)
                    exchange_lines.append(f"[tool {safe_tc_id}]: {sanitized}")

            turn_content = "\n".join(exchange_lines)

            # all_rows is a consistent snapshot from _identify_uncached_exchanges (fetched once)
            previous_context = _build_previous_context(all_rows, exchange[0]["message_index"], max_chars=20000)

            try:
                result = sota_client.summarize_message(
                    session_id, exchange_idx, turn_content, "exchange", None,
                    previous_messages_text=previous_context,
                )
                if result and result.get("summary"):
                    ex_idx = exchange[0]["exchange_index"]
                    _db().store_message_summary(
                        session_id, exchange_idx, "user",
                        result["summary"],
                        result.get("tool_summary", ""),
                        content_hash,
                        model_used=sota_client.model,
                        token_estimate=result.get("token_estimate", 0),
                        end_message_index=exchange_end_idx,
                    )
                    stored += 1
                    logger.info(
                        "[SYNC] Stored per-exchange summary for exchange at exchange #%d → #%d session_id=%s",
                        exchange_idx, exchange_end_idx, session_id,
                    )
            except (SOTATimeoutError, SOTAAPIError):
                logger.warning(
                    "[SYNC] Exchange summarization SOTA error for exchange #%d→#%d — skipping, will retry on next sync session_id=%s",
                    exchange_idx, exchange_end_idx, session_id, exc_info=True,
                )
                # Mark this index as failed (success_flags[i] = False). The
                # cursor below will only advance through contiguous successes.
                success_flags[i] = False
                failed_exchange = (exchange_idx, exchange_end_idx)
                failed_exchanges.append(failed_exchange)
                continue
            except SOTAParseError:
                logger.warning(
                    "[SYNC] Exchange summarization parse error for exchange #%d→#%d session_id=%s",
                    exchange_idx, exchange_end_idx, session_id, exc_info=True,
                )
                success_flags[i] = False
                failed_exchanges.append((exchange_idx, exchange_end_idx))
            except Exception as e:
                logger.warning(
                    "[SYNC] Exchange summarization failed for exchange #%d→#%d: %s session_id=%s",
                    exchange_idx, exchange_end_idx, e, session_id,
                )
                success_flags[i] = False
                failed_exchanges.append((exchange_idx, exchange_end_idx))

    # M5-4: compute the safe cursor advance.
    #
    # Semantic (a) — strict contiguous: the cursor advances only through a
    # CONTIGUOUS run of successes starting at index 0. The first failed
    # exchange blocks the cursor at the index BEFORE the failure.
    #
    # Example: 5 exchanges; #2 fails. Cursor advances to exchange_index
    # of index 1 (the highest contiguous success). #2, #3, #4 stay in
    # the unprocessed window for the next run.
    #
    # Example: 5 exchanges; ALL fail. success_flags = [F,F,F,F,F]. The
    # cursor does NOT advance — `end_exchange` of the new_cursor would
    # equal the input start_exchange - 1, so the caller must not call
    # set_last_summarized_exchange_index with a value < the current cursor.
    # We return start_exchange (the caller's input) which means "no
    # progress".
    if stored == 0:
        new_cursor = start_exchange
    else:
        # Find the longest prefix of all-True success_flags.
        new_cursor_in_batch = 0
        for ok in success_flags:
            if ok:
                new_cursor_in_batch += 1
            else:
                break
        if new_cursor_in_batch == 0:
            # Defensive: stored > 0 but no successes? Should not happen,
            # but stay safe.
            new_cursor = start_exchange
        else:
            # Map batch index back to absolute exchange_index. uncached[i]
            # is a (rows, exchange_idx, content_hash) tuple — index 1 is
            # the exchange_idx.
            new_cursor = uncached[new_cursor_in_batch - 1][1]

    # Surface failed exchanges for operators; the next sync cycle will
    # retry them since the cursor stops at the first failure.
    if failed_exchanges:
        logger.warning(
            "[SYNC] %d/%d exchanges failed to summarize (will retry next cycle) session_id=%s failed=%s",
            len(failed_exchanges), len(uncached), session_id, failed_exchanges,
        )
    return {
        "stored": stored,
        "found_uncached": True,
        "failed_exchanges": failed_exchanges,
        "new_cursor": new_cursor,
    }


def _handle_sync_error(e, session_id, batch_id, error_label, log_level="error"):
    """Handle sync error with appropriate logging and status update.
    Args:
        e: The exception
        session_id: Session ID for logging
        batch_id: Batch ID for status update (may be None)
        error_label: Value for error field in DB ("timeout", str(e), "parse_error", "unexpected")
        log_level: "warning", "error", or "exception" (exception includes traceback)
    """
    log_fn = getattr(logger, log_level)
    log_fn(
        f"[SYNC] Failed for session {session_id} batch {batch_id}: {error_label} - {e}",
        exc_info=(log_level == "exception"),
    )

    if batch_id:
        try:
            logger.info(f"Updating batch {batch_id} status to 'failed' ({error_label})")
            _update_batch_status(batch_id, "failed", error=error_label)
            logger.info(
                f"Successfully updated batch {batch_id} to 'failed' ({error_label})"
            )
        except Exception as update_error:
            logger.error(
                f"Failed to update batch {batch_id} status to 'failed': {update_error}",
                exc_info=True,
            )


_SOTA_API_KEY = os.environ.get("PROXY_SOTA_API_KEY", "")
_SOTA_ENDPOINT = os.environ.get("PROXY_SOTA_ENDPOINT", "https://api.stepfun.ai/step_plan/v1")
_SOTA_MODEL = os.environ.get("PROXY_SOTA_MODEL", "step-3.7-flash")
# Let SOTAClient resolve its own default; pass None for default path
_SOTA_TIMEOUT = int(os.environ.get("PROXY_SOTA_TIMEOUT", "180"))

# D2: SOTA input cap = 50% of SOTA model context (StepFun = 256,000)
SOTA_MODEL_CTX = 256_000
SOTA_INPUT_CAP_PCT = 0.50
SOTA_MAX_INPUT_TOKENS = int(SOTA_MODEL_CTX * SOTA_INPUT_CAP_PCT)
SOTA_MAX_INPUT_CHARS = SOTA_MAX_INPUT_TOKENS * 4  # ~409,600 chars at 4 chars/token
SOTA_MAX_HISTORY_MESSAGES = 5   # Upper bound for summary history window
SAFE_CONTEXT_TOKENS = 204_800  # 80% of 256K StepFun window

_sota_client = None

from memory.executors import submit_sync as _exec_submit_sync
from memory.executors import submit_summary as _exec_submit_summary
from memory.db_summaries import get_last_summarized_exchange_index as _get_summarization_cursor
from memory.db_summaries import set_last_summarized_exchange_index as _set_summarization_cursor


def _get_sota():
    global _sota_client
    if _sota_client is None:
        from memory.sota_client import SOTAClient

        _sota_client = SOTAClient(
            api_key=_SOTA_API_KEY,
            endpoint=_SOTA_ENDPOINT,
            model=_SOTA_MODEL,
            timeout=_SOTA_TIMEOUT,
        )
    return _sota_client


# NOTE: unused after two_stage removal
def _get_recent_messages(session_id, limit=None):
    """Get recent messages from SQLite for summary generation.
    D3: If limit is None, use SOTA_MAX_HISTORY_MESSAGES as default.
    """
    from memory.db_pool import connection

    if limit is None:
        limit = SOTA_MAX_HISTORY_MESSAGES

    with connection(pooled=True) as conn:
        try:
            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ? ORDER BY message_index DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []


def _resolve_safe_context_tokens(model_name):
    """Resolve safe context token count from model config or fallback."""
    if model_name:
        try:
            from proxy.context import get_context_config
            _cfg = get_context_config(model_name)
            return _cfg["safe_context_tokens"]
        except Exception:
            return SAFE_CONTEXT_TOKENS
    return SAFE_CONTEXT_TOKENS


SYNC_BATCH_SIZE_EXCHANGES = 10


def _determine_sync_range(session_id, _recovery_start=None, _recovery_end=None):
    """Determine which exchanges to process this sync cycle.

    Reads the summarization cursor from conversation_summaries.last_exchange_index
    (the cursor the summarization phase itself maintains via
    set_last_summarized_exchange_index) and picks up on the next cycle.
    First-run cursor is None → bootstrap from 0.

    Returns:
        dict with keys: start_exchange, end_exchange, start_idx, end_idx, new_messages
        OR None if no new exchanges to process.
    """
    if _recovery_start is not None and _recovery_end is not None:
        # Recovery path: convert message indices to exchange indices
        start_exchange = _get_exchange_for_message(session_id, _recovery_start)
        end_exchange = _get_exchange_for_message(session_id, _recovery_end)
        from memory.db import get_messages_in_range
        new_messages = get_messages_in_range(session_id, _recovery_start, _recovery_end)
        if not new_messages:
            logger.warning("[SYNC RECOVERY] No messages found in range %d-%d for session %s (purged?)",
                           _recovery_start, _recovery_end, session_id)
            return None

        return {
            'start_exchange': start_exchange,
            'end_exchange': end_exchange,
            'start_idx': _recovery_start,
            'end_idx': _recovery_end,
            'new_messages': new_messages,
            'last_synced': None,
        }

    # Normal path: use exchange indices directly
    from memory.db import get_messages_in_range, get_max_exchange_index
    last_synced = _get_summarization_cursor(session_id)
    if last_synced is None:
        last_synced = -1
    max_exchange = get_max_exchange_index(session_id)
    start_exchange = last_synced + 1
    end_exchange = min(max_exchange, start_exchange + SYNC_BATCH_SIZE_EXCHANGES - 1)

    if start_exchange > max_exchange:
        return None

    start_idx, end_idx = _exchange_range_to_message_range(session_id, start_exchange, end_exchange)
    if start_idx is None or end_idx is None:
        logger.warning("[SYNC] No messages for exchange range %d-%d session_id=%s",
                       start_exchange, end_exchange, session_id)
        return None

    new_messages = get_messages_in_range(session_id, start_idx, end_idx)
    if not new_messages:
        return None

    return {
        'start_exchange': start_exchange,
        'end_exchange': end_exchange,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'new_messages': new_messages,
        'last_synced': last_synced,
    }


def _run_summarization_phase(session_id, start_exchange, end_exchange, end_idx, last_synced=None):
    """Run per-exchange summarization. Returns the summarization result dict.

    The returned dict has keys:
        stored: int — number of summaries successfully written
        found_uncached: bool — True if there were exchanges needing summarization
    """
    summarization_result = _summarize_uncached_exchanges(session_id, _get_sota(), start_exchange, end_exchange)
    new_count = summarization_result["stored"]
    found_uncached = summarization_result["found_uncached"]
    new_cursor = summarization_result.get("new_cursor", end_exchange)
    if new_count > 0:
        logger.info("[SYNC] Summarized %d new exchanges for session %s", new_count, session_id)
    current_cursor = last_synced if last_synced is not None else start_exchange - 1
    if found_uncached:
        failed_exchanges = summarization_result.get("failed_exchanges") or []
        if new_count > 0 and new_cursor > current_cursor:
            _db().set_last_summarized_exchange_index(session_id, new_cursor)
            logger.info(
                "[SYNC] Advanced last_summarized_exchange_index to %d (was %d, end_exchange=%d, failed_in_batch=%d) session_id=%s",
                new_cursor, current_cursor, end_exchange, len(failed_exchanges), session_id,
            )
        else:
            logger.warning(
                "[SYNC] Found uncached exchanges but no cursor advance (new_count=%d, new_cursor=%d, current_cursor=%d, failed=%d) — will retry session_id=%s",
                new_count, new_cursor, current_cursor, len(failed_exchanges), session_id,
            )
    elif last_synced is not None and new_cursor > last_synced:
        _db().set_last_summarized_exchange_index(session_id, new_cursor)
        logger.info(
            "[SYNC] Cursor advanced over fully-cached window from %d to %d (end_exchange=%d) session_id=%s",
            last_synced, new_cursor, end_exchange, session_id,
        )
    return summarization_result


def _run_extraction_phase(session_id, batch_id, was_previously_failed):
    """Run extraction from summaries. Returns extraction dict or None (on skip/failure)."""
    try:
        summary_batch = _db().get_rich_summaries(
            session_id,
            before_index=None
        )
        if summary_batch:
            summary_batch = summary_batch[-SYNC_BATCH_SIZE_EXCHANGES:]
    except Exception as e:
        logger.warning("[SYNC] Failed to get summary batch: %s", e)
        summary_batch = []

    MIN_SUMMARIES_FOR_EXTRACTION = 3
    meaningful = [s for s in summary_batch if s.get("summary", "").strip()
                  and s["summary"] not in ("(no response)", "(empty message)", "(parse error)")]
    if len(meaningful) < MIN_SUMMARIES_FOR_EXTRACTION:
        logger.info("[SYNC] Only %d meaningful summaries (< %d), skipping extraction session_id=%s",
                   len(meaningful), MIN_SUMMARIES_FOR_EXTRACTION, session_id)
        _update_batch_status(batch_id, "skipped")
        return None

    extraction = None
    if summary_batch:
        try:
            extraction = _get_sota().extract_and_summarize_from_summaries(
                session_id, summary_batch
            )
            logger.info("[SYNC] Extraction from %d summaries produced %d entities, %d facts session_id=%s",
                       len(summary_batch),
                       len(extraction.get("entities", [])),
                       len(extraction.get("facts", [])), session_id)
        except Exception as e:
            logger.warning("[SYNC] Extraction from summaries failed: %s session_id=%s", e, session_id)
            if was_previously_failed:
                _update_batch_status(batch_id, "skipped", error=f"extraction_error: {e}")
                logger.warning("[SYNC] Batch %s previously failed, second extraction failure — permanently skipping", batch_id)
            else:
                _update_batch_status(batch_id, "failed", error=f"extraction_error: {e}")
            return None

    if extraction is None:
        extraction = {"extraction_timestamp": "", "entities": [], "facts": []}

    return extraction


def _persist_extraction_results(session_id, extraction, batch_id, end_idx, end_exchange=None):
    """Persist extraction to SQLite + Neo4j. Returns True on success, False to stop."""
    # M5-3a Window 2: claim the batch for extraction ONLY if it's still pending.
    # A late-arriving worker (e.g., recovery from a crashed sync) that finds
    # status='completed' must NOT re-extract and write duplicates. The claim
    # is a conditional UPDATE that returns True only if the WHERE clause
    # matched a row in status='pending'.
    if not _claim_batch_for_extraction(batch_id):
        logger.info(
            "Batch %s already extracted/completed/failed by another worker; persist is a no-op session_id=%s",
            batch_id, session_id,
        )
        return False

    has_entities = bool(extraction.get("entities"))
    has_facts = bool(extraction.get("facts"))
    has_events = bool(extraction.get("events", []))
    has_instructions = bool(extraction.get("instructions", []))
    has_tasks = bool(extraction.get("tasks", []))
    has_any = has_entities or has_facts or has_events or has_instructions or has_tasks

    if extraction and has_any:
        update_ok = _update_batch_extraction(batch_id, extraction, "extracted")
        if update_ok is None:
            logger.error("[SYNC] Failed to persist extraction for batch %s session_id=%s", batch_id, session_id)
            _update_batch_status(batch_id, "failed", error="extraction_persist_failed")
            return False

        persist_errors = []

        persist_result = _store_entities_and_facts(session_id, extraction)
        logger.info("[SYNC] Entity/fact persistence: %s session_id=%s", persist_result, session_id)

        if persist_result["errors"]:
            persist_errors.extend(persist_result["errors"][:3])

        memory_stored = _store_memories(
            session_id,
            events=extraction.get("events", []),
            instructions=extraction.get("instructions", []),
            tasks=extraction.get("tasks", []),
        )

        memories_requested = (
            len(extraction.get("events", []))
            + len(extraction.get("instructions", []))
            + len(extraction.get("tasks", []))
        )
        if memories_requested > 0 and not memory_stored:
            persist_errors.append("neo4j_failed")
            logger.warning(
                "[SYNC] Memory persistence failed: %d memories requested, 0 stored for batch %s session_id=%s",
                memories_requested, batch_id, session_id,
            )

        if persist_errors:
            error_msg = "; ".join(persist_errors)
            _update_batch_status(batch_id, "failed", error=error_msg)
            logger.error("[SYNC] Persistence had errors, batch %s marked failed: %s", batch_id, error_msg)
            return False

        # M5-3b: extraction must NOT advance the cursor past a
        # summarization gap. If exchanges 8..14 are queued for
        # extraction but summarization is stuck at 7, jumping the
        # cursor to 14 would drop the failed 8..14 from the next
        # _identify_uncached_exchanges output forever. Mirror of the
        # M5-4 / P1.4 fix on the summarization path.
        summarization_cursor = _get_summarization_cursor(session_id)
        if summarization_cursor is not None and summarization_cursor < end_exchange:
            proposed_cursor = summarization_cursor
            logger.info(
                "[SYNC] Extraction at %d but summarization cursor stuck at %d; "
                "capping extraction cursor at %d to preserve the summarization gap session_id=%s",
                end_exchange, summarization_cursor, proposed_cursor, session_id,
            )
        else:
            proposed_cursor = end_exchange
        current_cursor = summarization_cursor or 0
        new_cursor = max(current_cursor, proposed_cursor)
        _set_summarization_cursor(session_id, new_cursor)
        logger.info(
            "[SYNC] Advanced last_summarized_exchange_index to %d for session %s (post-extraction)",
            new_cursor, session_id,
        )

        logger.info("Extraction stored for batch %s (%d entities, %d facts, %d events, %d instructions, %d tasks)",
                   batch_id,
                   len(extraction.get("entities", [])),
                   len(extraction.get("facts", [])),
                   len(extraction.get("events", [])),
                   len(extraction.get("instructions", [])),
                   len(extraction.get("tasks", [])))
    else:
        logger.info("No entities/facts/events/instructions/tasks extracted for batch %s session_id=%s", batch_id, session_id)

    # M5-3a: conditional UPDATE for the terminal "completed" claim.
    # The atomicity gate is "I'm about to mark complete" — if another
    # worker already marked it complete, rowcount=0 and we return False
    # (so the caller doesn't think we did the work).
    claimed = _update_batch_status_if(
        batch_id,
        from_status=("extracted", "processing"),
        to_status="completed",
    )
    if not claimed:
        logger.info(
            "Batch %s already completed by another worker; persist was a no-op session_id=%s",
            batch_id, session_id,
        )
        return False
    return True


def _maybe_compress_summaries(session_id, safe_context_tokens):
    """Check if compression is needed and compress ancient history."""
    all_summaries = None
    try:
        all_summaries = _db().get_rich_summaries(session_id)
        summary_count = len(all_summaries) if all_summaries else 0
        existing_ancient = _db().get_ancient_history(session_id)
        ancient_len = len(existing_ancient) if existing_ancient else 0
        logger.info(
            "[SYNC] _maybe_compress_summaries: session=%s summaries=%d ancient_history_len=%d safe_tokens=%d",
            session_id, summary_count, ancient_len, safe_context_tokens,
        )
    except Exception as e:
        logger.debug("[SYNC] Diagnostic logging failed (non-fatal): %s", e)

    RICH_SUMMARY_BUDGET = int(safe_context_tokens * 0.25)
    RICH_SUMMARY_THRESHOLD = int(RICH_SUMMARY_BUDGET * 0.75)

    try:
        if all_summaries is None:
            all_summaries = _db().get_rich_summaries(session_id)
        total_summary_tokens = sum(
            s.get("rendered_full_tokens", 0) for s in all_summaries
        )
        if total_summary_tokens > RICH_SUMMARY_THRESHOLD or len(all_summaries) > 50:
            logger.info("[SYNC] Rich summary budget exceeded (%d tokens > %d threshold), triggering compression",
                       total_summary_tokens, RICH_SUMMARY_THRESHOLD)

            evict_count = max(10, len(all_summaries) // 5)
            summaries_to_evict = all_summaries[:evict_count]

            ancient_lock = _get_ancient_lock(session_id)
            with ancient_lock:
                existing_ancient = _db().get_ancient_history(session_id) or ""

                try:
                    from memory.sota_client import CompressionError
                    from memory.db_pool import transaction as db_transaction

                    new_ancient = _get_sota().compress_ancient_history(
                        session_id, existing_ancient, summaries_to_evict
                    )

                    evicted_indices = [s["exchange_index"] for s in summaries_to_evict]

                    with db_transaction() as conn:
                        _db().update_ancient_history(session_id, new_ancient, conn=conn)
                        if evicted_indices:
                            chunk_size = 500
                            for i in range(0, len(evicted_indices), chunk_size):
                                chunk = evicted_indices[i:i + chunk_size]
                                placeholders = ",".join(["?"] * len(chunk))
                                conn.execute(
                                    f"DELETE FROM message_summaries WHERE session_id = ? AND exchange_index IN ({placeholders})",
                                    [session_id] + chunk
                                )
                        logger.info("[SYNC] Compressed %d summaries into ancient history, deleted evicted", len(evicted_indices))
                    try:
                        from proxy.state import invalidate_chain_state_for_session
                        invalidate_chain_state_for_session(session_id)
                    except ImportError:
                        pass
                    except Exception as e:
                        logger.warning(
                            "Ancient compressed successfully but chain-state invalidation "
                            "failed: %s session_id=%s", e, session_id,
                        )
                except CompressionError as e:
                    logger.warning("[SYNC] Compression failed, keeping existing summaries: %s", e)
                except Exception as e:
                    logger.error("[SYNC] Compression failed unexpectedly: %s", e)
    except Exception as e:
        logger.warning("[SYNC] Compression check failed (non-fatal): %s", e)


def sync_exchange(session_id, messages=None, _recovery_start=None, _recovery_end=None):
    """Run sync pipeline: extraction + summary for new exchanges since last sync."""
    safe_context_tokens = _resolve_safe_context_tokens(None)
    logger.info("[SYNC] Starting sync for session %s", session_id)
    batch_id = None
    range_result = _determine_sync_range(session_id, _recovery_start, _recovery_end)
    if range_result is None:
        return

    start_exchange = range_result['start_exchange']
    end_exchange = range_result['end_exchange']
    start_idx = range_result['start_idx']
    end_idx = range_result['end_idx']
    new_messages = range_result['new_messages']
    last_synced = range_result.get('last_synced')
    try:
        _run_summarization_phase(session_id, start_exchange, end_exchange, end_idx, last_synced)
    except (SOTATimeoutError, SOTAAPIError) as e:
        logger.warning("[SYNC] Summarization SOTA error (will retry): %s session_id=%s", e, session_id)
        # Do NOT return early — continue to extraction phase for successfully
        # summarized exchanges. The failed exchanges will be retried on next sync.
        # Returning early leaves later exchange batches unprocessed → context gaps.
    except Exception as e:
        logger.warning("[SYNC] Summarization step failed: %s session_id=%s", e, session_id)
        # Continue to extraction phase — partial success is better than no progress.
        # Returning early would leave later exchanges unsummarized.
    try:
        batch_result = _prepare_extraction_batch(
            session_id,
            start_exchange=start_exchange,
            end_exchange=end_exchange,
            start_message_index=start_idx,
            end_message_index=end_idx,
        )
        if batch_result is None:
            return
        batch_id, was_previously_failed = batch_result
        extraction = _run_extraction_phase(session_id, batch_id, was_previously_failed)
        if extraction is None:
            return
        if not _persist_extraction_results(session_id, extraction, batch_id, end_idx, end_exchange):
            return
        _maybe_compress_summaries(session_id, safe_context_tokens)
    except SOTATimeoutError as e:
        logger.debug("[SYNC DEBUG] SOTATimeoutError for session %s batch %s: %s", session_id, batch_id, e)
        _handle_sync_error(e, session_id, batch_id, "timeout", "warning")
    except SOTAAPIError as e:
        logger.debug("[SYNC DEBUG] SOTAAPIError for session %s batch %s: %s", session_id, batch_id, e)
        _handle_sync_error(e, session_id, batch_id, str(e), "error")
    except SOTAParseError as e:
        logger.debug("[SYNC DEBUG] SOTAParseError for session %s batch %s: %s", session_id, batch_id, e)
        _handle_sync_error(e, session_id, batch_id, "parse_error", "error")
    except Exception as e:
        logger.debug("[SYNC DEBUG] Unexpected exception for session %s batch %s: %s", session_id, batch_id, e)
        _handle_sync_error(e, session_id, batch_id, "unexpected", "exception")


def init_db():
    """Initialize both SQLite and Neo4j schemas. Call once at proxy startup."""
    from memory.db import init_db as db_init

    db_init()
    try:
        _get_graph().init_schema()
    except Exception as e:
        logger.error("Neo4j schema init failed: %s", e)

    from memory.query_executor import get_shared_executor_from_env
    from memory.migrations.runner import MigrationRunner
    from memory.migrations import BackfillParentTaskUuid, VerifyBackfill, RemoveProjectRelationships, GraphNativeTasks, EnsureGraphNative, AgentTaskAssignmentMigration, WorkSessionMigration, BugTrackingMigration, WorkspaceBackfill, StripWorkspacePrefix, EntityDedupConstraint

    executor = get_shared_executor_from_env()
    if executor is None:
        logger.error("Neo4j executor init failed — Neo4j likely down")
        return

    runner = MigrationRunner(executor)
    runner.register(BackfillParentTaskUuid())
    runner.register(VerifyBackfill())
    runner.register(RemoveProjectRelationships())
    runner.register(GraphNativeTasks())
    runner.register(EnsureGraphNative())
    runner.register(AgentTaskAssignmentMigration())
    runner.register(WorkSessionMigration())
    runner.register(BugTrackingMigration())
    runner.register(WorkspaceBackfill())
    runner.register(StripWorkspacePrefix())
    runner.register(EntityDedupConstraint())

    try:
        runner.run_pending()
    except Exception as e:
        logger.error("Migration runner failed: %s", e)


def reconcile_sessions(session_id=None):
    """Detect SQLite↔Neo4j data divergence per session.

    Returns dict with per-session entity/fact counts from both stores.
    Does NOT auto-repair — just reports discrepancies.

    Args:
        session_id: Optional session to check. If None, all sessions are checked.

    Returns:
        dict with shape:
            {
                "sessions": {
                    "<sid>": {
                        "sqlite": int (entity count),
                        "neo4j": int | -1 on error (entity count),
                        "sqlite_facts": int,
                        "neo4j_facts": int | -1 on error,
                        "divergent": bool,
                        "orphan": bool (True if in Neo4j but missing from SQLite),
                    },
                    ...
                },
                "neo4j_available": bool (only present if Neo4j was unavailable),
            }
    """
    from memory.db_pool import connection

    results = {"sessions": {}}

    with connection(pooled=True) as conn:
        if session_id:
            entity_rows = conn.execute(
                "SELECT session_id, COUNT(*) FROM entities WHERE session_id = ? GROUP BY session_id",
                (session_id,),
            ).fetchall()
        else:
            entity_rows = conn.execute(
                "SELECT session_id, COUNT(*) FROM entities GROUP BY session_id"
            ).fetchall()
        sqlite_counts = {row[0]: row[1] for row in entity_rows}

        if session_id:
            fact_rows = conn.execute(
                "SELECT session_id, COUNT(*) FROM facts WHERE session_id = ? GROUP BY session_id",
                (session_id,),
            ).fetchall()
        else:
            fact_rows = conn.execute(
                "SELECT session_id, COUNT(*) FROM facts GROUP BY session_id"
            ).fetchall()
        sqlite_fact_counts = {row[0]: row[1] for row in fact_rows}

    g = _get_graph()
    if not g.is_available():
        logger.warning("reconcile_sessions: Neo4j unavailable, skipping")
        results["neo4j_available"] = False
        return results

    for sid, sqlite_count in sqlite_counts.items():
        try:
            neo_count = g._executor.execute_read(
                "MATCH (e:Entity {session_id: $sid}) RETURN count(e) AS cnt",
                {"sid": sid},
            )
            neo_count_val = neo_count[0]["cnt"] if neo_count else 0
        except Exception as e:
            logger.warning("reconcile_sessions: Neo4j entity query failed for %s: %s", sid, e)
            neo_count_val = -1

        results["sessions"][sid] = {
            "sqlite": sqlite_count,
            "neo4j": neo_count_val,
            "sqlite_facts": sqlite_fact_counts.get(sid, 0),
            "neo4j_facts": 0,
            "divergent": neo_count_val >= 0 and neo_count_val != sqlite_count,
            "orphan": False,
        }

    for sid, sqlite_fact_count in sqlite_fact_counts.items():
        if sid not in results["sessions"]:
            results["sessions"][sid] = {
                "sqlite": 0, "neo4j": 0,
                "sqlite_facts": sqlite_fact_count, "neo4j_facts": 0,
                "divergent": False, "orphan": False,
            }
        try:
            neo_fact_rows = g._executor.execute_read(
                "MATCH (s:Entity {session_id: $sid})-[r:RELATES_TO]->(t:Entity {session_id: $sid}) "
                "RETURN count(r) AS cnt",
                {"sid": sid},
            )
            neo_fact_count = neo_fact_rows[0]["cnt"] if neo_fact_rows else 0
        except Exception as e:
            logger.warning("reconcile_sessions: Neo4j fact query failed for %s: %s", sid, e)
            neo_fact_count = -1

        results["sessions"][sid]["neo4j_facts"] = neo_fact_count
        if neo_fact_count >= 0 and neo_fact_count != sqlite_fact_count:
            results["sessions"][sid]["divergent"] = True
            logger.warning(
                "[RECONCILE] session=%s facts divergent sqlite=%d neo4j=%d",
                sid[:8], sqlite_fact_count, neo_fact_count,
            )

    try:
        all_neo_sessions = g._executor.execute_read(
            "MATCH (e:Entity) RETURN DISTINCT e.session_id AS sid"
        )
        for rec in all_neo_sessions:
            sid = rec.get("sid") if isinstance(rec, dict) else None
            if not sid:
                continue
            if sid not in results["sessions"]:
                try:
                    orphan_count_rows = g._executor.execute_read(
                        "MATCH (e:Entity {session_id: $sid}) RETURN count(e) AS cnt",
                        {"sid": sid},
                    )
                    orphan_count = orphan_count_rows[0]["cnt"] if orphan_count_rows else 0
                except Exception as e:
                    logger.warning("reconcile_sessions: orphan count failed for %s: %s", sid, e)
                    orphan_count = -1
                results["sessions"][sid] = {
                    "sqlite": 0, "neo4j": orphan_count,
                    "sqlite_facts": 0, "neo4j_facts": 0,
                    "divergent": True, "orphan": True,
                }
                logger.warning(
                    "[RECONCILE] orphan Neo4j session=%s entities=%d",
                    sid[:8], orphan_count,
                )
    except Exception as e:
        logger.warning("reconcile_sessions: orphan detection failed: %s", e)

    divergent_count = sum(1 for s in results["sessions"].values() if s["divergent"])
    logger.info(
        "[RECONCILE] %d sessions checked, %d divergent",
        len(results["sessions"]), divergent_count,
    )
    return results


def submit_sync(session_id):
    """Submit a sync job to the thread pool."""
    return _exec_submit_sync(sync_exchange, session_id)


def submit_ancient_compress(session_id, summary_payload):
    """Dispatch async SOTA job to re-compress overflow summaries into ancient.

    Parallel to submit_sync (line ~1223). Fire-and-forget — returns immediately,
    runs in background. M5-1: routed through the SUMMARY pool (not the
    SYNC pool) so ancient-compress jobs don't block sync_exchange workers
    during eviction bursts. Ancient-compress is fundamentally a
    summarization task (re-compresses older exchanges into "ancient"
    history), and the summary pool is a separate 4-worker pool
    (executors.py:50-56) that doesn't share a worker queue with sync.

    Never spawns raw threading.Thread (HIGH 7 fix).

    On completion, writes compressed ancient to conversation_summaries.ancient_history
    for the session_id row.

    If circuit breaker is open (SOTA unavailable), returns immediately without
    dispatching — proxy continues with truncated summaries (non-fatal).

    Bounded queue: max 100 pending dispatches. Excess calls return None without
    dispatching; the truncated summaries remain in DB and will be re-dispatched
    on the next eviction (preventing unbounded memory growth under sustained load).

    Args:
        session_id: Session identifier.
        summary_payload: List of summary dicts to re-compress into ancient.
    """
    if not summary_payload:
        return None

    try:
        from memory.sota_channel import _sota_breaker
        if not _sota_breaker.can_execute():
            logger.info(
                "[SYNC] SOTA circuit breaker open, skipping ancient compress dispatch session_id=%s",
                session_id,
            )
            return None
    except ImportError:
        logger.debug("[SYNC] sota_channel unavailable, dispatching ancient compress anyway session_id=%s", session_id)

    global _ancient_compress_pending
    with _ancient_compress_lock:
        if _ancient_compress_pending >= _ANCIENT_COMPRESS_MAX_PENDING:
            logger.debug(
                "ancient compress queue full (%d pending); skipping dispatch "
                "(truncated summaries stay in DB for next eviction) session_id=%s",
                _ancient_compress_pending, session_id,
            )
            return None
        _ancient_compress_pending += 1

    def _wrapped():
        try:
            _ancient_compress_worker(session_id, summary_payload)
        finally:
            global _ancient_compress_pending
            with _ancient_compress_lock:
                _ancient_compress_pending -= 1

    try:
        return _exec_submit_summary(_wrapped)
    except Exception as e:
        with _ancient_compress_lock:
            _ancient_compress_pending -= 1
        logger.warning(
            "[SYNC] Failed to dispatch ancient compress (pool submit failed); "
            "counter decremented session_id=%s: %s",
            session_id, e,
            exc_info=True,
        )
        return None


def _ancient_compress_worker(session_id, summary_payload):
    """Worker for submit_ancient_compress — runs in bounded pool thread.

    Compresses summary_payload into ancient history via SOTA and persists result.
    All errors are logged (non-fatal) — the proxy continues with truncated summaries.
    """
    if not summary_payload:
        return
    try:
        ancient_lock = _get_ancient_lock(session_id)
        with ancient_lock:
            existing_ancient = _db().get_ancient_history(session_id) or ""
            new_ancient = _get_sota().compress_ancient_history(
                session_id, existing_ancient, summary_payload
            )
            from memory.db_pool import transaction as db_transaction
            with db_transaction() as conn:
                _db().update_ancient_history(session_id, new_ancient, conn=conn)
                for s in summary_payload:
                    _db().delete_message_summary(session_id, s["exchange_index"], conn=conn)
                logger.info(
                    "[SYNC] Async ancient compress completed for session_id=%s (deleted %d summaries)",
                    session_id, len(summary_payload),
                )
            try:
                from proxy.state import invalidate_chain_state_for_session
                invalidate_chain_state_for_session(session_id)
            except ImportError:
                pass
    except (SOTATimeoutError, SOTAAPIError) as e:
        logger.warning(
            "[SYNC] Ancient compress SOTA error for session_id=%s: %s",
            session_id, e,
        )
    except SOTAParseError as e:
        logger.warning(
            "[SYNC] Ancient compress parse error for session_id=%s: %s",
            session_id, e,
        )
    except Exception as e:
        logger.warning(
            "[SYNC] Ancient compress failed for session_id=%s: %s",
            session_id, e,
            exc_info=True,
        )


def _cleanup_sync_locks(stale_session_ids: set) -> int:
    """Remove sync locks for sessions that have been evicted from cache.

    Called from session.py cleanup_old_sessions() to prevent lock dict growth.
    Also removes stale ancient locks for the same sessions.
    """
    cleaned = 0
    with _sync_locks_lock:
        for sid in stale_session_ids:
            if _sync_locks.pop(sid, None) is not None:
                cleaned += 1
    if cleaned:
        logger.debug("Cleaned %d stale sync locks", cleaned)
    _cleanup_ancient_locks(stale_session_ids)
    return cleaned


def _cleanup_ancient_locks(stale_session_ids: set) -> int:
    cleaned = 0
    with _ancient_locks_guard:
        for sid in stale_session_ids:
            lock = _ancient_locks.get(sid)
            if lock is not None and not lock.locked():
                del _ancient_locks[sid]
                cleaned += 1
    if cleaned:
        logger.debug("Cleaned %d stale ancient locks", cleaned)
    return cleaned
