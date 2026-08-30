import html
import pytest
from unittest.mock import patch, MagicMock

from commands import (
    is_command,
    parse_command,
    CommandContext,
    make_command_response,
    dispatch_command,
    COMMAND_REGISTRY,
    COMMAND_HELP,
    register,
)


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------

def test_is_command_true():
    assert is_command("^^help") is True


def test_is_command_false():
    assert is_command("help") is False


def test_is_command_prefix_only():
    assert is_command("^^") is False


# ---------------------------------------------------------------------------
# 2. Parsing
# ---------------------------------------------------------------------------

def test_parse_command_basic():
    assert parse_command("^^help") == ("^^help", [])


def test_parse_command_with_args():
    assert parse_command("^^status detail") == ("^^status", ["detail"])


# ---------------------------------------------------------------------------
# 3. Context
# ---------------------------------------------------------------------------

def test_command_context_construction():
    ctx = CommandContext(raw_text="^^help", session_id="abc")
    assert ctx.raw_text == "^^help"
    assert ctx.session_id == "abc"


# ---------------------------------------------------------------------------
# 4. Response
# ---------------------------------------------------------------------------

def test_make_command_response_format():
    resp = make_command_response("hello")
    assert "id" in resp
    assert resp["object"] == "chat.completion"
    assert "created" in resp
    assert resp["model"] == "skill"
    assert len(resp["choices"]) == 1
    choice = resp["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"
    assert "usage" in resp


def test_make_command_response_raw_content():
    """Commands return raw content without HTML escaping."""
    raw = "<script>alert('xss')</script>"
    resp = make_command_response(raw)
    content = resp["choices"][0]["message"]["content"]
    # Content should be returned as-is (no escaping)
    assert "<script>" in content
    assert content == "<script>alert('xss')</script>"


# ---------------------------------------------------------------------------
# 5. Registry
# ---------------------------------------------------------------------------

def test_register_adds_to_registry():
    def dummy(ctx, args):
        return make_command_response("dummy")

    register("^^test_reg", "test help")(dummy)
    try:
        assert COMMAND_REGISTRY["^^test_reg"] is dummy
        assert COMMAND_HELP["^^test_reg"] == "test help"
    finally:
        del COMMAND_REGISTRY["^^test_reg"]
        del COMMAND_HELP["^^test_reg"]


def test_register_duplicate_overwrites():
    def first(ctx, args):
        return make_command_response("first")

    def second(ctx, args):
        return make_command_response("second")

    register("^^test_dup", "first help")(first)
    register("^^test_dup", "second help")(second)
    try:
        assert COMMAND_REGISTRY["^^test_dup"] is second
        assert COMMAND_HELP["^^test_dup"] == "second help"
    finally:
        del COMMAND_REGISTRY["^^test_dup"]
        del COMMAND_HELP["^^test_dup"]


# ---------------------------------------------------------------------------
# 6. Dispatch
# ---------------------------------------------------------------------------

def test_dispatch_known_command():
    ctx = CommandContext(raw_text="^^help", session_id="s1")
    resp = dispatch_command(ctx)
    assert "choices" in resp


def test_dispatch_unknown_command():
    ctx = CommandContext(raw_text="^^unknown", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Unknown command" in content


# ---------------------------------------------------------------------------
# 7. Built-ins
# ---------------------------------------------------------------------------

def test_cmd_help_lists_commands():
    """^^help lists built-in commands. Pin COMMAND_HELP coverage (regression)."""
    ctx = CommandContext(raw_text="^^help", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "^^help" in content
    assert "^^status" in content


def _install_registry_with_custom_skill():
    """Build a fresh SkillRegistry with builtins + a wave5custom skill and install it.

    Returns (registry_module, registry_instance). Caller must call
    `reset_default_registry()` in a finally block to undo the singleton
    install — leaking it would corrupt later tests.
    """
    import proxy.skills.skill_registry as sr_module
    from proxy.skills.skill_registry import (
        SkillRegistry,
        _skill_registry_new__,
        reset_default_registry,
    )
    from proxy.skills.skill_definition import TrustToken
    from proxy.skills.custom_skill_loader import (
        ParsedCustomSkill,
        build_custom_skill_definition,
    )

    reset_default_registry()
    reg = _skill_registry_new__(SkillRegistry)
    reg.populate_builtins(trusted=True)

    parsed = ParsedCustomSkill(
        name="wave5custom",
        trigger="^^wave5custom",
        description="WAVE5_UNIQUE_TEST_DESCRIPTION_ZZZ",
        aliases=("^^wc",),
        prompt_text="test prompt",
        allowed_tool_names=frozenset({"read"}),
        ttl_seconds=300,
        source_file="wave5custom.json",
    )
    sd = build_custom_skill_definition(parsed)
    reg.register_skill(sd, trusted_registrant=TrustToken.UNTRUSTED)

    # SkillRegistry is a frozen dataclass, so set the module-level
    # singleton (what `get_default_registry()` reads), NOT an instance
    # attribute via object.__setattr__.
    sr_module._default_registry = reg
    return sr_module, reg


def test_cmd_help_lists_real_registry_skills_section():
    """^^help MUST enumerate real SkillRegistry entries with [builtin]/[custom] tags and aliases.

    Replaces the weak `test_cmd_help_lists_commands`. Asserts the Skills
    section comes from the actual registry (a UNIQUE description proves
    the registry was read, not a hard-coded stub).
    """
    from proxy.skills.skill_registry import reset_default_registry
    try:
        _install_registry_with_custom_skill()
        ctx = CommandContext(raw_text="^^help", session_id="s1")
        resp = dispatch_command(ctx)
        content = resp["choices"][0]["message"]["content"]
        # Section headers per spec §8.1
        assert "Available Commands:" in content
        assert "Commands:" in content
        assert "Skills:" in content
        # Builtin tag
        assert "^^ticket" in content
        assert "[builtin]" in content
        # Custom tag
        assert "^^wave5custom" in content
        assert "[custom]" in content
        # Real registry read — unique description from the actual skill
        assert "WAVE5_UNIQUE_TEST_DESCRIPTION_ZZZ" in content
        # Alias rendering
        assert "aliases: ^^wc" in content
    finally:
        reset_default_registry()


def test_cmd_help_does_not_duplicate_ticket_in_commands_section():
    """^^ticket is a SKILL, not a command. It must NOT appear in the Commands section."""
    from proxy.skills.skill_registry import reset_default_registry
    try:
        _install_registry_with_custom_skill()
        ctx = CommandContext(raw_text="^^help", session_id="s1")
        resp = dispatch_command(ctx)
        content = resp["choices"][0]["message"]["content"]
        # Find the Commands and Skills sections.
        cmd_section_start = content.index("Commands:")
        skills_section_start = content.index("Skills:")
        cmd_section = content[cmd_section_start:skills_section_start]
        # ^^ticket must NOT appear in the Commands section.
        assert "^^ticket" not in cmd_section
        # ^^ticket MUST appear in the Skills section (from registry).
        skills_section = content[skills_section_start:]
        assert "^^ticket" in skills_section
    finally:
        reset_default_registry()


def test_cmd_help_workspace_in_commands_not_skills():
    """^^workspace is a command (workspace_commands.py), not a registered skill."""
    from proxy.skills.skill_registry import reset_default_registry
    try:
        _install_registry_with_custom_skill()
        ctx = CommandContext(raw_text="^^help", session_id="s1")
        resp = dispatch_command(ctx)
        content = resp["choices"][0]["message"]["content"]
        # ^^workspace IS in Commands (via @register in workspace_commands.py).
        assert "^^workspace" in content
        # ^^workspace is NOT in Skills (workspace is not in SkillRegistry).
        if "Skills:" in content:
            skills_start = content.index("Skills:")
            skills_section = content[skills_start:]
            assert "^^workspace" not in skills_section
    finally:
        reset_default_registry()


@patch.multiple(
    "proxy.state",
    request_counter=42,
    tool_cache_size=5,
    tool_cache_hits=10,
    tool_cache_misses=3,
    chat_histories={"sess1": []},
    create=True,
)
@patch("proxy.model_registry._model_registry")
def test_cmd_status_shows_stats(mock_registry, **_state_mocks):
    mock_registry.list_models.return_value = ["model1", "model2"]
    ctx = CommandContext(raw_text="^^status", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Models loaded: 2" in content
    assert "Total requests: 42" in content
    assert "Tool schemas cached: 5" in content
    assert "77% hit rate" in content


@patch.multiple(
    "proxy.state",
    request_counter=0,
    tool_cache_size=0,
    tool_cache_hits=0,
    tool_cache_misses=0,
    chat_histories={},
    create=True,
)
@patch("proxy.model_registry._model_registry", None)
def test_cmd_status_no_registry(**_state_mocks):
    ctx = CommandContext(raw_text="^^status", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Models loaded: 0" in content


# ---------------------------------------------------------------------------
# 6b. Wave 2 — Custom Skills block (spec §8.2)
# ---------------------------------------------------------------------------


@patch("proxy.model_registry._model_registry")
@patch.multiple(
    "proxy.state",
    request_counter=42,
    tool_cache_size=8,
    tool_cache_hits=10,
    tool_cache_misses=5,
    chat_histories={"s1": []},
    log_overflow_count=0,
)
@patch("proxy.skills.skill_registry.get_default_registry")
def test_cmd_status_shows_custom_skills_block(
    mock_get_default_registry,
    mock_registry,
    **_state_mocks,
):
    """cmd_status MUST render all 6 spec §8.2 custom_skills fields.

    Pinned regression: Status / Loading / Directory / Discovered / Loaded /
    Restart. Each field is asserted as a substring of the rendered output.
    """
    from proxy.skills.custom_skill_loader import CustomSkillsStatus
    fake_status = CustomSkillsStatus(
        enabled=True,
        directory="/etc/choriprox/skills",
        source="env",
        state="ok",
        discovered=3,
        loaded_names=("build", "plan", "review"),
        issues=(),
        restart_required=True,
    )
    fake_registry = MagicMock()
    fake_registry.custom_skills_status.return_value = fake_status
    mock_get_default_registry.return_value = fake_registry
    mock_registry.list_models.return_value = []

    ctx = CommandContext(raw_text="^^status", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Custom Skills:" in content
    assert "Status: ok" in content
    assert "Loading: enabled" in content
    assert "Directory: /etc/choriprox/skills" in content
    assert "Discovered: 3" in content
    assert "Loaded: 3 (build, plan, review)" in content
    assert "Restart: required for changes to take effect" in content


@patch("proxy.model_registry._model_registry")
@patch("proxy.skills.skill_registry.get_default_registry")
def test_cmd_status_handles_registry_failure_gracefully(
    mock_get_default_registry,
    mock_registry,
):
    """If get_default_registry raises, cmd_status reports unavailable and doesn't crash."""
    mock_get_default_registry.side_effect = Exception("registry not ready")
    mock_registry.list_models.return_value = []

    ctx = CommandContext(raw_text="^^status", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Custom Skills: unavailable" in content


# ---------------------------------------------------------------------------
# 8. Feature flag
# ---------------------------------------------------------------------------

def test_feature_flag_enabled(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "^^help"}],
        },
    )
    data = resp.get_json()
    content = data["choices"][0]["message"]["content"]
    assert "Available Commands" in content


def test_feature_flag_disabled(client):
    from clients.orchestrator_config import ProxyFeatures

    disabled = ProxyFeatures(intercepts_commands=False)
    with patch("proxy.app._FEATURES", disabled):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "^^help"}],
            },
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("model") != "^^command"


# ---------------------------------------------------------------------------
# 9. Interception
# ---------------------------------------------------------------------------

def test_user_message_triggers_command(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "^^help"}],
        },
    )
    data = resp.get_json()
    assert data["model"] == "^^command"
    content = data["choices"][0]["message"]["content"]
    assert "^^help" in content


def test_system_message_skipped(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "^^help"},
            ],
        },
    )
    data = resp.get_json()
    assert data.get("model") != "^^command"


def test_empty_messages_skipped(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [],
        },
    )
    assert resp.status_code == 400
    error = resp.get_json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages"
    assert error["code"] == "invalid_messages"
    assert "message" in error


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

def test_prefix_only_passthrough(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "^^ "}],
        },
    )
    data = resp.get_json()
    assert data.get("model") != "^^command"


def test_unicode_command():
    ctx = CommandContext(raw_text="^^héllo", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Unknown command" in content


def test_whitespace_padding():
    assert parse_command("  ^^status  ") == ("^^status", [])


def test_uppercase_help():
    # EC #2: `^^HELP` lowercased to `^^help` by parse_command -> matches registry
    ctx = CommandContext(raw_text="^^HELP", session_id="s1")
    resp = dispatch_command(ctx)
    content = resp["choices"][0]["message"]["content"]
    assert "Available Commands" in content


def test_non_prefix_command():
    # EC #4: `multiple ^^help` — doesn't start with "^^"
    assert is_command("multiple ^^help") is False


def test_tool_role_skipped(client):
    # EC #5: tool role missing tool_call_id now triggers structured 400 per spec §7.1
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "^^help"},
            ],
        },
    )
    assert resp.status_code == 400
    error = resp.get_json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages[1].tool_call_id"
    assert error["code"] == "missing_tool_call_id"
    assert "message" in error


def test_multi_content_text_part(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "^^help"}]},
            ],
        },
    )
    data = resp.get_json()
    assert data.get("model") == "^^command"


def test_empty_content_passthrough(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": ""}],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("model") != "^^command"


def test_content_null_passthrough(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": None}],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("model") != "^^command"


def test_content_int_passthrough(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": 42}],
        },
    )
    data = resp.get_json()
    assert data.get("model") != "^^command"


def test_content_empty_list_passthrough(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": []}],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("model") != "^^command"
