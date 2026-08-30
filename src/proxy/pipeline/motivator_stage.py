"""MotivatorStage — injects context-usage reminders into sub-agent prompts.

First consumer of the Prompt Injection Layer (PromptInjector, ThresholdEngine,
AgentTelemetry, ConditionalStage). Reminds sub-agents of their original task
when context is filling up, helping them stay focused and return results
before the context budget is exhausted.

Trigger is based on the high-watermark of request body size vs the model's context window
(rough heuristic: 1 token ≈ 4 chars). Fires at 50/75/90% of context.
"""
import copy
import json
import logging
import os

from proxy.pipeline.conditional_stage import ConditionalStage
from proxy.pipeline.prompt_injector import PromptInjector

logger = logging.getLogger(__name__)

MOTIVATOR_ENABLED = os.environ.get("PROXY_MOTIVATOR_ENABLED", "1") == "1"
# Default fallback if neither config override nor model_cfg are available.
# 4_000_000 chars ≈ 1M tokens (the mini-m3 model window). Most modern
# models have 100K-1M+ tokens; this default assumes a large-window model.
try:
    CONTEXT_WINDOW_CHARS = int(os.environ.get("PROXY_MOTIVATOR_CONTEXT_WINDOW_CHARS", "4000000"))
except (ValueError, TypeError):
    CONTEXT_WINDOW_CHARS = 4_000_000

THRESHOLD_MESSAGES = {
    50: "You've used about half of the context. Stay focused on the core task.",
    75: "Context is filling up. Prioritize completing your task over additional exploration.",
    90: "Context is nearly full. Return your results now; further tool calls may exceed the budget.",
}


def _estimate_chars(messages):
    """Estimate total char count of all messages in the array.

    Rough heuristic: 1 token ≈ 4 chars (English text). Handles string and
    multimodal (list-of-parts) content.
    """
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", "")) + len(part.get("input_text", ""))
                elif isinstance(part, str):
                    total += len(part)
    return total


class MotivatorStage(ConditionalStage):
    """Injects context-usage reminders into sub-agent prompts.

    Only active when:
    - PROXY_MOTIVATOR_ENABLED=1
    - Request is from a sub-agent (ctx.is_subagent)
    - Pre-optimization messages are available (ctx.messages_original)
    """

    def should_run(self, ctx) -> bool:
        return (
            MOTIVATOR_ENABLED
            and getattr(ctx, "is_subagent", False)
            and getattr(ctx, "messages_original", None) is not None
        )

    def _execute(self, ctx):
        from proxy.agent.agent_registry import registry

        agent = registry.get_or_load(ctx.session_id)
        if not agent:
            return None

        if not getattr(ctx, "body", None):
            return None

        telemetry = agent.telemetry

        config = getattr(agent, 'config', None)
        # Resolve the context window for the actual model being used.
        # Priority: explicit config override → model_cfg.context_limit →
        # env/default fallback. This ensures the Motivator reminder fires
        # based on the model's REAL capacity (e.g. 1M tokens for mini-m3),
        # not an arbitrary static default.
        explicit_window = getattr(config, 'motivator_context_window_chars', None) if config else None
        model_cfg = getattr(ctx, 'model_cfg', None)
        model_window_tokens = getattr(model_cfg, 'context_limit', None) if model_cfg else None

        if explicit_window is not None:
            context_window = explicit_window
        elif model_window_tokens is not None and model_window_tokens > 0:
            # Convert token limit to chars (1 token ≈ 4 chars heuristic)
            context_window = model_window_tokens * 4
        else:
            context_window = CONTEXT_WINDOW_CHARS

        thresholds = getattr(config, 'motivator_thresholds', (50, 75, 90)) if config else (50, 75, 90)
        if agent.threshold_engine.thresholds != sorted(set(thresholds)):
            from proxy.pipeline.threshold_engine import ThresholdEngine
            agent.threshold_engine = ThresholdEngine(thresholds=thresholds)

        telemetry.capture_task(ctx.messages_original)
        telemetry.increment_turn()

        if telemetry.turn_count == 1:
            agent.threshold_engine.reset()

        # Track high-watermark of request body size (latest request IS the cumulative context)
        messages_for_size = ctx.body.get("messages", [])
        telemetry.record_request_chars(_estimate_chars(messages_for_size))

        pct = telemetry.get_context_pct(context_window)
        threshold = agent.threshold_engine.check(pct)

        if threshold is not None:
            task_summary = telemetry.original_task or "(task not captured)"
            msg = THRESHOLD_MESSAGES.get(threshold, "Stay focused.")
            label = {50: "Context Advisory", 75: "Context Warning", 90: "Context Critical"}.get(threshold, "Context Advisory")
            reminder = (
                f"[MOTIVATOR — {label}]\n"
                f"Original task: \"{task_summary}\"\n"
                f"Context usage: {pct:.0f}% of {context_window // 4} tokens\n"
                f"{msg}\n"
                f"[END MOTIVATOR]"
            )
            messages = ctx.body.get("messages", [])
            injected = PromptInjector.prepend_to_last_user_message(messages, reminder)
            if injected:
                ctx.body["messages"] = messages
                ctx.messages = copy.deepcopy(messages)
                logger.info(
                    "[MOTIVATOR] %d%% threshold fired session=%s pct=%.1f",
                    threshold,
                    ctx.session_id,
                    pct,
                    extra={"request_id": getattr(ctx, "request_id", "?")},
                )
            else:
                logger.debug(
                    "[MOTIVATOR] no user message to inject into, session=%s",
                    ctx.session_id,
                    extra={"request_id": getattr(ctx, "request_id", "?")},
                )
        # Turn-2 early reminder: reinforce task before context pruning kicks in
        if telemetry.turn_count == 2:
            task_summary = telemetry.original_task or ""
            # Only fire for non-trivial tasks (> 100 chars)
            if len(task_summary) > 100:
                reminder = (
                    f"[MOTIVATOR — Early Reminder]\n"
                    f"Your task: \"{task_summary[:200]}\"\n"
                    f"Stay focused and return results when you have enough information.\n"
                    f"[END MOTIVATOR]"
                )
                messages = ctx.body.get("messages", [])
                injected = PromptInjector.prepend_to_last_user_message(messages, reminder)
                if injected:
                    ctx.body["messages"] = messages
                    ctx.messages = copy.deepcopy(messages)
                    logger.info(
                        "[MOTIVATOR] Turn-2 reminder session=%s",
                        ctx.session_id,
                        extra={"request_id": getattr(ctx, "request_id", "?")},
                    )
        return None
