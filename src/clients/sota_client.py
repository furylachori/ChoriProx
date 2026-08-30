"""SOTA client adapter — thin wrapper around memory.sota_channel.

This module exists to keep the enhancer/ layer dependency direction clean:
enhancer/ imports from clients/, never from memory/. clients/ is the only
package allowed to reach into memory/ for SOTA transport details.

Pre-existing SOTAChannel/SOTAMode usage elsewhere (memory/sync.py,
memory/sota_client.py) keeps its direct memory.* imports — those callers
live in memory/ and have no layering concern.
"""

from typing import Optional


def call_sota_extraction(
    prompt: str,
    system_prompt: str = "",
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    session_id: str = "enhancer",
) -> str:
    """Call SOTA in EXTRACTION mode and return the parsed string content.

    Thin wrapper around SOTAChannel — callers (notably enhancer/) should
    use this rather than importing SOTAChannel directly.
    """
    from memory.sota_channel import SOTAChannel
    from memory._sota_http import SOTAMode

    channel = SOTAChannel(
        mode=SOTAMode.EXTRACTION,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = channel.ask(prompt, session_id=session_id)
    return str(result.parsed).strip()