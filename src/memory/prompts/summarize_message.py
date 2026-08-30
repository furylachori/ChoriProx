SENSITIVE_TOOL_NAMES = {"bash", "shell", "exec", "eval"}


def sanitize_tool_calls(tool_calls_json):
    """Replace sensitive tool args with [REDACTED] before sending to SOTA."""
    if not tool_calls_json:
        return tool_calls_json
    for tc in tool_calls_json:
        name = tc.get("function", {}).get("name", "")
        if name in SENSITIVE_TOOL_NAMES:
            tc["function"]["arguments"] = "[REDACTED]"
    return tool_calls_json


# ── Part 1: System prompt (role + task, upfront) ─────────────────────
SUMMARIZE_MESSAGE_SYSTEM_PROMPT = r"""
You are a session analyst. Your task: summarize the exchange in the user input
marked "THE MESSAGE TO SUMMARIZE:" at the very end — treating it as one unit
containing a user request, any intermediate tool calls and results, and the final
assistant response. Everything before the "---" divider is previous conversation history provided for context only. Do NOT summarize the history.
""".strip()

# ── Part 2: Instructions (injected right before the target message) ──
SUMMARIZE_MESSAGE_INSTRUCTIONS = r"""
Your summary MUST be a detailed 3-6 sentence narrative covering these dimensions (when applicable):
1. WHAT — What action did the user take or what question did they ask?
2. DECISIONS — What technical decisions were made (e.g., approach chosen, library selected)?
3. FILES — What files were read, created, modified, or deleted?
4. TOOLS — What tools were invoked and for what purpose?
5. TASK THREAD — (When applicable) What is the user's overarching goal across this exchange?
   What step are we on? What was completed vs. what is still in progress?
   Frame it so someone reading ONLY this summary knows what the user
   is trying to accomplish and where things stand.
   Skip this dimension for ad-hoc questions that don't relate to a known task, ticket, or feature.
6. OUTCOME — What was the result? What is the immediate next step or open question?

The message may include tool calls and results between the user request and final
response. Summarize the entire exchange as one unit.

Rules:
- The summary MUST make sense even if tool metadata is stripped later (self-contained).
- Do not include raw code in the summary. Reference files by name only.
- For bash/exec/shell/eval, NEVER include the command text. Use [tools: bash] only.
- For any tool that receives credentials, secrets, or destructive commands,
  NEVER include the argument values. Use [tools: TOOL_NAME] only.
- Only include facts explicitly stated in the message. Never infer, assume,
  or fabricate actions, files, or decisions.

Call the submit_summary tool with your result.
""".strip()
