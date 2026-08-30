"""Shared utilities for the memory system."""


def strip_args(tool_summary: str) -> str:
    """Strip argument values from a tool_summary string, keeping only tool names.

    Used for positional zone degradation in context assembly (ChoriProx Plus §6.3).

    Input:  "[tools: bash: rm -rf /, write_file: foo.txt: content...]"
    Output: "[tools: bash, write_file]"

    If the input doesn't match the expected format, return it unchanged.
    """
    if not tool_summary or not tool_summary.startswith("[tools:") or not tool_summary.endswith("]"):
        return tool_summary
    inner = tool_summary[7:-1].strip()
    if not inner:
        return "[tools:]"
    actions = [t.split(":")[0].strip() for t in inner.split(",") if t.strip()]
    return "[tools: " + ", ".join(actions) + "]"
