"""§9.8 — compute_chain equivalence: pin (role=system) does NOT alter the
computed chain prefix for non-system messages.

Per spec §7: "V2 chain cache | None. Pin is role='system' → compute_chain
skips it." This is the cheaper proxy for "pin doesn't bust the V2 chain cache."

We assert compute_chain([sys_msg, pin_msg]) covers sys_msg identically to
compute_chain([sys_msg]) — i.e. the pin (a system message with the
<CLIENT_SUB_INSTRUCTIONS> content) is transparent to the chain hash because
compute_chain skips ALL role=system messages.
"""
import pytest

from proxy.chain_hash import compute_chain
from proxy.client_instructions import format_instruction_section
from proxy.skill_prompts import _escape_prompt_field


def _make_pin_msg(content="<CLIENT_SUB_INSTRUCTIONS>dummy</CLIENT_SUB_INSTRUCTIONS>") -> dict:
    """Build a system-role pin message with the given inner content."""
    escaped = _escape_prompt_field(content)
    section = format_instruction_section(escaped, tier="sub")
    return {"role": "system", "content": section}


def test_compute_chain_empty():
    chain, pos = compute_chain([])
    assert chain == []
    assert pos == {}


def test_compute_chain_single_non_system():
    msgs = [{"role": "user", "content": "hello"}]
    chain, pos = compute_chain(msgs)
    assert len(chain) == 1
    assert chain[0] in pos
    assert pos[chain[0]] == 0


def test_compute_chain_skips_system_messages():
    """System messages are skipped entirely — they don't appear in the chain."""
    sys_msg = {"role": "system", "content": "you are a helpful assistant"}
    user_msg = {"role": "user", "content": "do the thing"}

    chain_sys_only, _ = compute_chain([sys_msg])
    assert chain_sys_only == []

    chain_user_only, pos_user = compute_chain([user_msg])
    assert len(chain_user_only) == 1

    chain_both, pos_both = compute_chain([sys_msg, user_msg])
    assert chain_both == chain_user_only
    assert pos_both == pos_user


def test_compute_chain_pin_does_not_bust_cache():
    """The core §9.8 invariant: adding a <CLIENT_SUB_INSTRUCTIONS> pin as a
    system message does NOT alter the chain hash of the non-system messages.

    If compute_chain did NOT skip system messages, the pin's content would
    change the cumulative hash and bust the V2 chain cache on every turn
    after a Tier-2 capture. The pin MUST be transparent.
    """
    sys_msg = {"role": "system", "content": "operational directives"}
    user_msg = {"role": "user", "content": "please read the config"}
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"filePath": "/x"}'},
            }
        ],
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "<path>/x</path><type>file</type><content>body</content>",
    }

    pin_msg = _make_pin_msg()

    # Baseline chain WITHOUT the pin
    base_msgs = [sys_msg, user_msg, assistant_msg, tool_msg]
    base_chain, base_pos = compute_chain(base_msgs)

    # Chain WITH the pin inserted after sys_msg (mirrors _emit_tier2_pin behavior)
    pinned_msgs = [sys_msg, pin_msg, user_msg, assistant_msg, tool_msg]
    pinned_chain, pinned_pos = compute_chain(pinned_msgs)

    assert base_chain == pinned_chain, (
        f"pin busts cache: base={base_chain} != pinned={pinned_chain}"
    )
    assert base_pos == pinned_pos


def test_compute_chain_pin_content_irrelevant():
    """Even a large, realistic Tier-2 pin content must be transparent."""
    sys_msg = {"role": "system", "content": "Tier-1 root instructions"}
    user_msg = {"role": "user", "content": "read src/proxy/sub/route.py"}
    tool_msg = {
        "role": "tool",
        "tool_call_id": "tc_tier2",
        "content": (
            "<path>/repo/project/src/proxy/sub/route.py</path>"
            "<type>file</type><content>\n# file body\n</content>"
            "<system-reminder>\n"
            "Instructions from: /repo/project/src/proxy/sub/AGENTS.md\n"
            "# Sub AGENTS.md\n"
            "Instructions from: /repo/project/src/proxy/AGENTS.md\n"
            "# Parent AGENTS.md\n"
            "</system-reminder>"
        ),
    }

    pin_content = (
        '<CLIENT_SUB_INSTRUCTIONS scope="subfolder-convention">\n'
        "Instructions from: /repo/project/src/proxy/sub/AGENTS.md\n"
        "# large sub-AGENTS.md content (~17 KB)\n"
        "Instructions from: /repo/project/src/proxy/AGENTS.md\n"
        "# large parent-AGENTS.md content (~17 KB)\n"
        "</CLIENT_SUB_INSTRUCTIONS>"
    )
    pin_msg = _make_pin_msg(pin_content)

    base = [sys_msg, user_msg, tool_msg]
    pinned = [sys_msg, pin_msg, user_msg, tool_msg]

    base_chain, _ = compute_chain(base)
    pinned_chain, _ = compute_chain(pinned)

    assert base_chain == pinned_chain


def test_compute_chain_multiple_system_messages_all_skipped():
    """Any number of system messages (including multiple pins) are transparent."""
    sys1 = {"role": "system", "content": "root instructions"}
    sys2 = {"role": "system", "content": "another system msg"}
    pin = _make_pin_msg()
    user = {"role": "user", "content": "hi"}
    assistant = {"role": "assistant", "content": "hello!"}

    base, _ = compute_chain([sys1, user, assistant])
    with_extra_sys, _ = compute_chain([sys1, sys2, user, assistant])
    with_pin, _ = compute_chain([sys1, pin, sys2, user, assistant])

    assert base == with_extra_sys
    assert base == with_pin
