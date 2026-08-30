import os
import re
import json
import logging
from datetime import datetime, timezone, timedelta

from openai import OpenAI

from memory.prompts import extraction
from proxy.utils import _now_iso

logger = logging.getLogger("memory.sota_client")

_TOPIC_KEY_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+$")
_VALID_CONFIDENCE = {"high", "medium", "low"}
_VALID_TASK_STATUS = {"pending", "in_progress", "completed", "cancelled"}

# Import from _sota_rate_limiter.py
from memory._sota_rate_limiter import SOTA_MAX_INPUT_CHARS

# Import from sota_channel.py
from memory.sota_channel import (
    SOTAChannel,
    SOTATimeoutError,
    SOTAAPIError,
    SOTARateLimitError,
    SOTAParseError,
    _sanitize_text,
    _validate_content,
    parse_retry_after,
)

from memory.sota_channel import SOTAMode, SOTAResult


class CompressionError(Exception):
    """Raised when ancient history compression produces invalid output."""
    pass


class SyncResult:
    def __init__(self, facts=None, events=None, instructions=None, tasks=None, summary=""):
        self.facts = facts or []
        self.events = events or []
        self.instructions = instructions or []
        self.tasks = tasks or []
        self.summary = summary or ""


__SOTA_SUBMIT_SYNC_TOOL__ = {
    "type": "function",
    "function": {
        "name": "__sota_submit_sync__",
        "description": "Submit extracted entities and facts from conversation summaries",
        "parameters": {
            "type": "object",
            "properties": {
                "extraction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 timestamp of extraction"
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 2, "maxLength": 200},
                            "type": {
                                "type": "string",
                                "enum": ["Person", "Organization", "Project", "Technology", "File", "Concept", "Objective", "Bug", "Fix", "Task"]
                            },
                            "summary": {"type": "string", "minLength": 10, "maxLength": 2000}
                        },
                        "required": ["name", "type", "summary"]
                    }
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_entity": {"type": "string"},
                            "target_entity": {"type": "string"},
                            "relationship_type": {
                                "type": "string",
                                "enum": ["USES", "CONTAINS", "CREATES", "SUPERSEDES", "DOCUMENTS", "CAUSES", "RESOLVES", "DEPENDS_ON", "PRECEDES", "ALTERNATIVE_TO"]
                            },
                            "fact": {"type": "string", "minLength": 10, "maxLength": 2000},
                            "valid_at": {"type": ["string", "null"]},
                            "invalid_at": {"type": ["string", "null"]}
                        },
                        "required": ["source_entity", "target_entity", "relationship_type", "fact"]
                    }
                }

            },
            "required": ["extraction_timestamp", "entities", "facts"]
        }
    }
}


__SOTA_SUMMARIZE_MESSAGE_TOOL__ = {
    "type": "function",
    "function": {
        "name": "submit_summary",
        "description": "Submit a single exchange summary for storage in the memory system",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "3-6 sentence detailed narrative. Cover all applicable: WHAT (action/intent), DECISIONS (technical choices), FILES (files referenced/changed), TOOLS (tools used), OUTCOME (result/next step). Self-contained so it makes sense without context."
                },
                "tool_summary": {
                    "type": "string",
                    "description": "Summary of tools mentioned or used. Format: [tools: action1: arg1, action2: arg2]. For bash/exec/shell/eval, NEVER include command text — use [tools: bash] only. If no tools, use empty string."
                },
                "token_estimate": {
                    "type": "integer",
                    "description": "Estimated number of tokens in the summary text (not including tool_summary). Rough estimate: word_count / 0.75."
                }
            },
            "required": ["summary", "tool_summary", "token_estimate"]
        }
    }
}


def strip_code_blocks(text):
    text = re.sub(r"```[\s\S]*?```", "[CODE BLOCK REMOVED]", text)
    text = re.sub(r"`[^`]+`", "[CODE]", text)
    return text


def strip_tool_blocks(text: str) -> str:
    """Remove <TOOL>...</TOOL> blocks. Handles case, attributes, unclosed tags."""
    from memory.sota_channel import _TOOL_RE, _TOOL_UNCLOSED_RE
    text = _TOOL_RE.sub("", text)
    if _TOOL_UNCLOSED_RE.search(text):
        text = _TOOL_UNCLOSED_RE.sub("", text)
    return text


def _validate_confidence(item):
    """Normalize confidence to lowercase. Default to medium if missing/invalid."""
    conf = item.get("confidence", "")
    if not isinstance(conf, str) or conf.lower() not in _VALID_CONFIDENCE:
        item["confidence"] = "medium"
    else:
        item["confidence"] = conf.lower()


def _validate_topic_key(item):
    """Validate topic_key format (category/subcategory). Default to misc/unknown if invalid."""
    key = item.get("topic_key", "")
    if not isinstance(key, str) or not _TOPIC_KEY_RE.match(key):
        item["topic_key"] = "misc/unknown"


def _validate_timestamp(item):
    """Validate ISO-8601 timestamp. Default to now if invalid."""
    ts = item.get("timestamp", "")
    if not isinstance(ts, str) or not ts:
        item["timestamp"] = _now_iso()
        return
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        item["timestamp"] = _now_iso()


def _validate_task_status(item):
    """Normalize task status. Default to pending if missing/invalid."""
    _STATUS_NORMALIZE = {"completed": "complete", "cancelled": "invalid"}
    status = item.get("status", "")
    normalized = status.lower() if isinstance(status, str) else ""
    item["status"] = _STATUS_NORMALIZE.get(normalized, normalized)
    if not isinstance(item["status"], str) or item["status"] not in _VALID_TASK_STATUS:
        item["status"] = "pending"


def _validate_fact_item(item):
    """Validate a fact item: content, topic_key, confidence in one pass."""
    _validate_content(item)
    _validate_topic_key(item)
    _validate_confidence(item)


def _validate_event_item(item):
    """Validate an event item: content, timestamp, confidence in one pass."""
    _validate_content(item)
    _validate_timestamp(item)
    _validate_confidence(item)


def _validate_instruction_item(item):
    """Validate an instruction item: content, topic_key, confidence in one pass."""
    _validate_content(item)
    _validate_topic_key(item)
    _validate_confidence(item)


def _validate_task_item(item):
    """Validate a task item: content, status in one pass."""
    _validate_content(item)
    _validate_task_status(item)


def _validate_extraction_items(args):
    """Validate and sanitize all extraction items in one pass per item.

    Dispatches to per-item-type validators to avoid calling the same
    sub-validators repeatedly in inner loops (N+1 call reduction).
    """
    for fact in args.get("facts", []):
        _validate_fact_item(fact)
    for event in args.get("events", []):
        _validate_event_item(event)
    for instruction in args.get("instructions", []):
        _validate_instruction_item(instruction)
    for task in args.get("tasks", []):
        _validate_task_item(task)


def resolve_temporal_reference(text, current_datetime):
    now = current_datetime

    text = re.sub(
        r"\byesterday\b", (now - timedelta(days=1)).strftime("%Y-%m-%d"), text
    )
    text = re.sub(r"\btoday\b", now.strftime("%Y-%m-%d"), text)
    text = re.sub(
        r"\blast week\b", (now - timedelta(weeks=1)).strftime("week of %Y-%m-%d"), text
    )
    text = re.sub(
        r"\b(\d+) days? ago\b",
        lambda m: (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d"),
        text,
    )
    text = re.sub(r"\bthis morning\b", now.strftime("%Y-%m-%d morning"), text)

    return text


class SOTAClient:
    def __init__(
        self,
        api_key=None,
        endpoint=None,
        model=None,
        timeout=None,
    ):
        import os

        api_key = api_key or os.environ.get("PROXY_SOTA_API_KEY", "")
        base_url = endpoint or os.environ.get(
            "PROXY_SOTA_ENDPOINT", "https://api.stepfun.ai/step_plan/v1"
        )
        model = model or os.environ.get("PROXY_SOTA_MODEL", "step-3.7-flash")
        if timeout is None:
            timeout = int(os.environ.get("PROXY_SOTA_TIMEOUT", "180"))
        # SOTA extraction prompts (StepFun/MiniMax) may need long timeouts;
        # floor at 180 only when resolved from environment / hardcoded defaults
        timeout = max(timeout, 180)

        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self.model = model

    def sync(self, messages, current_summary="", **kwargs):
        return self.extract_and_summarize(messages, current_summary=current_summary, **kwargs)

    def extract_and_summarize(self, messages, current_summary="", **kwargs):
        iso_timestamp = kwargs.get("iso_timestamp", "")
        conversation_start = kwargs.get("conversation_start", "")
        conversation_end = kwargs.get("conversation_end", "")
        existing_memories = kwargs.get("existing_memories", "none")
        session_id = kwargs.get("session_id", "")

        stripped_messages = []
        for msg in messages:
            if msg.get("content") and isinstance(msg["content"], str):
                cleaned = strip_tool_blocks(msg["content"])
                cleaned = strip_code_blocks(cleaned)
                if msg.get("role") == "assistant" and not cleaned.strip() and "tool_calls" not in msg:
                    continue
                stripped_messages.append({**msg, "content": cleaned})
            else:
                stripped_messages.append(msg)

        user_prompt = extraction.build_unified_prompt(
            stripped_messages,
            current_summary=current_summary,
            iso_timestamp=iso_timestamp,
            conversation_start=conversation_start,
            conversation_end=conversation_end,
            existing_memories=existing_memories,
        )

        if len(user_prompt) > SOTA_MAX_INPUT_CHARS:
            logger.warning(
                "Unified prompt too large (%d chars, cap %d). Truncating messages.",
                len(user_prompt), int(SOTA_MAX_INPUT_CHARS),
            )
            trunc_msgs = list(stripped_messages)
            while len(user_prompt) > SOTA_MAX_INPUT_CHARS and len(trunc_msgs) > 1:
                trunc_msgs = trunc_msgs[1:]
                user_prompt = extraction.build_unified_prompt(
                    trunc_msgs,
                    current_summary=current_summary,
                    iso_timestamp=iso_timestamp,
                    conversation_start=conversation_start,
                    conversation_end=conversation_end,
                    existing_memories=existing_memories,
                )

        logger.info(
            "SOTA unified payload: %d chars (%.1f KB), %d messages",
            len(user_prompt), len(user_prompt) / 1024, len(stripped_messages),
        )

        # ── SOTA channel call (replaces 5-attempt retry loop) ────────────────
        channel = SOTAChannel(
            mode=SOTAMode.EXTRACTION,
            system_prompt=extraction.UNIFIED_SYSTEM_PROMPT,
            tools=[__SOTA_SUBMIT_SYNC_TOOL__],
            tool_choice={"type": "function", "function": {"name": "__sota_submit_sync__"}},
            temperature=0.2,
            max_tokens=65536,
            reasoning_effort="low",
            parse_fn=None,
            model=self.model,
        )
        result = channel.ask(user_prompt, session_id=session_id)

        # ── Post-parse validation and sanitization ────────────────────────────
        args = result.parsed

        if not isinstance(args, dict):
            raise SOTAParseError(f"Expected dict from SOTA, got {type(args).__name__}")

        for key in ["facts", "events", "instructions", "tasks"]:
            if key not in args:
                args[key] = []
            elif not isinstance(args[key], list):
                args[key] = []

        _validate_extraction_items(args)
        logger.info("[SOTA] Extractions call latency: %.1fms session_id=%s", result.elapsed_ms, session_id)

        raw_summary = args.get("summary", "")
        summary_text = _sanitize_text(raw_summary if isinstance(raw_summary, str) else "")

        if result.usage:
            reasoning_tokens = getattr(result.usage, "completion_tokens_details", None) or getattr(result.usage, "prompt_tokens_details", None)
            if reasoning_tokens:
                logger.info("SOTA unified reasoning tokens: %s", reasoning_tokens)

        return SyncResult(
            facts=args.get("facts", []),
            events=args.get("events", []),
            instructions=args.get("instructions", []),
            tasks=args.get("tasks", []),
            summary=summary_text,
        )

    def summarize_message(self, session_id, exchange_index, raw_content, role, tool_calls_json=None, previous_messages_text=""):
        """Summarize a single exchange (user message + assistant/tool turns). Returns dict with summary, tool_summary, token_estimate.

        Calls SOTA API with SUMMARIZE_MESSAGE_SYSTEM_PROMPT + SUMMARIZE_MESSAGE_INSTRUCTIONS. Enforces 512-token hard cap.
        Uses 429-aware retry with jitter (from Phase 0 infrastructure).
        """
        from memory.prompts.summarize_message import SUMMARIZE_MESSAGE_SYSTEM_PROMPT, SUMMARIZE_MESSAGE_INSTRUCTIONS, sanitize_tool_calls

        if not raw_content:
            return {"summary": "(empty message)", "tool_summary": "", "token_estimate": 1}

        # Sanitize tool calls before sending to SOTA
        safe_tool_calls = sanitize_tool_calls(tool_calls_json) if tool_calls_json else None

        # ── Build user prompt ──────────────────────────────────────────
        # Structure (per design spec):
        # 1. Previous history (old→new), tagged as context — do NOT summarize
        # 2. Divider
        # 3. THE MESSAGE TO SUMMARIZE — the target, included here and ONLY here
        # 4. Tool calls (if any)

        user_content = ""
        if previous_messages_text:
            user_content += (
                "[Conversation History — for context only, do NOT summarize]\n"
                f"{previous_messages_text}\n"
                "---\n\n"
            )

        user_content += (
            SUMMARIZE_MESSAGE_INSTRUCTIONS
            + "\n\nTHE MESSAGE TO SUMMARIZE:\n"
            f"Role: {role}\n"
            f"Content: {raw_content[:8000]}"
        )
        if safe_tool_calls:
            user_content += f"\nTool calls: {json.dumps(safe_tool_calls)[:4000]}"

        # ── SOTA channel call (replaces 5-attempt retry loop) ────────────────
        channel = SOTAChannel(
            mode=SOTAMode.SUMMARIZATION,
            system_prompt=SUMMARIZE_MESSAGE_SYSTEM_PROMPT,
            tools=[__SOTA_SUMMARIZE_MESSAGE_TOOL__],
            tool_choice={"type": "function", "function": {"name": "submit_summary"}},
            temperature=0.1,
            max_tokens=65536,
            reasoning_effort="low",
            parse_fn=None,  # use result.raw for multi-path fallback in caller
            model=self.model,
        )
        result = channel.ask(user_content, session_id=session_id)
        logger.info("SOTA summarize_message: %.1fms (attempt %d) session_id=%s exchange_index=%d", result.elapsed_ms, result.attempts, session_id, exchange_index)

        # ── Parse response (replaces lines 547-570) ─────────────────────────
        # With tools set, mode branching always returns a dict (tool args) or
        # raises SOTAParseError. String-typed raw is unreachable.
        raw = result.raw
        if isinstance(raw, dict):
            parsed_result = raw
        else:
            # Should not reach here; treat as parse error
            logger.warning(
                "summarize_message: unexpected raw type %s session_id=%s",
                type(raw).__name__, session_id
            )
            parsed_result = {"summary": "(parse error)", "tool_summary": "", "token_estimate": 0}

        if not isinstance(parsed_result, dict):
            summary = parsed_result if isinstance(parsed_result, str) else "(parse error)"
            tool_summary = ""
            token_estimate = 1
        else:
            summary = parsed_result.get("summary", "")
            tool_summary = parsed_result.get("tool_summary", "")
            token_estimate = parsed_result.get("token_estimate", 0)

        # Hard cap at 512 tokens (spec §5.1)
        if summary and len(summary.split()) / 0.75 > 512:
            max_words = int(512 * 0.75)
            words = summary.split()[:max_words]
            summary = " ".join(words) + "..."
            logger.warning(
                "Summarize_message output truncated to ~512 tokens session_id=%s",
                session_id
            )

        return {
            "summary": summary,
            "tool_summary": tool_summary,
            "token_estimate": token_estimate,
        }

    def _call_compressor(self, session_id, existing_ancient_history, summaries_to_evict):
        """Call SOTA to compress a batch of old summaries into a timeline narrative."""
        from memory.prompts.compress_history import COMPRESS_HISTORY_SYSTEM_PROMPT

        summaries_text = ""
        for s in summaries_to_evict:
            idx = s.get("exchange_index", s.get("message_index", "?"))
            role = s.get("role", "unknown")
            summary = s.get("summary", "")
            summaries_text += f"[exchange#{idx}] {role}: {summary}\n"

        target_tokens = min(2000, len(summaries_text.split()) // 5)
        max_tokens_val = max(32768, target_tokens * 2)

        system_prompt = COMPRESS_HISTORY_SYSTEM_PROMPT.format(
            target_tokens=target_tokens, max_tokens=max_tokens_val
        )

        existing_text = existing_ancient_history or "(new session)"

        user_prompt = (
            f"Existing timeline:\n{existing_text}\n\n"
            f"Old summaries to incorporate:\n{summaries_text}\n\n"
            f"Produce updated timeline:"
        )

        # ── SOTA channel call (replaces 5-attempt retry loop) ────────────────
        # Build a new channel per call since system_prompt and max_tokens are dynamic
        channel = SOTAChannel(
            mode=SOTAMode.COMPRESSION,
            system_prompt=system_prompt,
            tools=None,
            tool_choice=None,
            temperature=0.2,
            max_tokens=max_tokens_val,
            reasoning_effort="low",
            parse_fn=None,
            model=self.model,
        )
        result = channel.ask(user_prompt, session_id=session_id)
        return result.raw or ""

    def compress_ancient_history(self, session_id, existing_ancient_history, summaries_to_evict):
        """Compress old exchange summaries into updated ancient history timeline.

        Returns the new ancient_history text. Raises CompressionError on failure.
        The caller is responsible for write-verify-delete transaction.
        """
        if not summaries_to_evict:
            return existing_ancient_history or ""

        new_ancient = self._call_compressor(session_id, existing_ancient_history, summaries_to_evict)

        if not new_ancient or len(new_ancient.strip()) < 200:
            raise CompressionError("Compressor returned empty or trivial output")

        evicted_text = " ".join(s.get("summary", "") for s in summaries_to_evict[:10])
        keywords = sorted(set(w.lower() for w in evicted_text.split() if len(w) > 3))[:10]

        if keywords and not any(kw in new_ancient.lower() for kw in keywords):
            raise CompressionError("Compressor output lacks keywords from evicted summaries")

        matched = sum(1 for kw in keywords if kw in new_ancient.lower()) if keywords else 0
        if keywords and matched < min(3, len(keywords) * 0.5):
            raise CompressionError(
                f"Compressor output matched only {matched}/{len(keywords)} keywords; insufficient coverage"
            )

        return new_ancient

    def extract_and_summarize_from_summaries(self, session_id, summary_batch):
        """Extract entities and facts from a batch of per-exchange summaries.

        Input: ~4K tokens of summary text (not the raw chat messages).
        Output: dict with extraction_timestamp, entities, facts.
        Uses the new entity-relationship tool schema.
        """
        from memory.prompts.extract_from_summaries import EXTRACT_FROM_SUMMARIES_PROMPT

        if not summary_batch:
            return {"extraction_timestamp": _now_iso(), "entities": [], "facts": []}

        summaries_text = ""
        for s in summary_batch:
            idx = s.get("exchange_index", s.get("message_index", "?"))
            role = s.get("role", "unknown")
            summary = s.get("summary", "")
            summaries_text += f"[exchange#{idx}] {role}: {summary}\n"

        user_prompt = f"Extract entities and facts from these exchange summaries:\n\n{summaries_text}"

        iso_timestamp = _now_iso()

        max_retries = 2
        for attempt in range(max_retries + 1):
            extraction = None
            # ── SOTA channel call (replaces 5-attempt inner retry loop) ───────
            channel = SOTAChannel(
                mode=SOTAMode.EXTRACTION,
                system_prompt=EXTRACT_FROM_SUMMARIES_PROMPT,
                tools=[__SOTA_SUBMIT_SYNC_TOOL__],
                tool_choice={"type": "function", "function": {"name": "__sota_submit_sync__"}},
                temperature=0.2,
                max_tokens=65536,
                reasoning_effort="low",
                parse_fn=None,
                model=self.model,
            )
            result = channel.ask(user_prompt, session_id=session_id)
            args = result.parsed  # dict from mode branching (tool args or cleaned content JSON)

            if not isinstance(args, dict):
                if attempt < max_retries:
                    continue
                raise SOTAParseError(f"Expected dict from extraction, got {type(args).__name__}")

            from memory.extraction_validator import validate_extraction
            extraction, errors = validate_extraction(args)

            if not errors:
                return extraction

            if attempt < max_retries:
                logger.warning("Extraction validation failed (attempt %d): %s. Retrying. session_id=%s", attempt + 1, errors, session_id)
                error_text = "\n".join(errors[:5])
                user_prompt = f"{user_prompt}\n\n[Previous attempt had errors: {error_text}. Please fix and resubmit.]"
                continue
            else:
                logger.error("Extraction validation failed after %d attempts: %s session_id=%s", max_retries + 1, errors, session_id)
                raise SOTAParseError(
                    f"Extraction validation failed after {max_retries + 1} attempts: {'; '.join(errors)}"
                )

        return {"extraction_timestamp": iso_timestamp, "entities": [], "facts": []}