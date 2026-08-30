from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def test_no_default_tool_names_literal_in_src():
    """Pin the consolidation: DEFAULT_TOOL_NAMES literal must not
    appear in src/ after Phase 3 deletion. All consumers go through
    ToolRegistry.get_default_tools()."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "DEFAULT_TOOL_NAMES", "src/"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.stdout.strip() == "", (
        f"DEFAULT_TOOL_NAMES literal still exists in src/. All consumers "
        f"must use ToolRegistry.get_default_tools(). Hits:\n{result.stdout}"
    )


def test_no_memory_tool_names_literal_in_src():
    """Pin the consolidation: MEMORY_TOOL_NAMES literal must not
    appear in src/ after Phase 3 deletion. All consumers go through
    ToolRegistry.get_memory_tool_names()."""
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "MEMORY_TOOL_NAMES", "src/"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.stdout.strip() == "", (
        f"MEMORY_TOOL_NAMES literal still exists in src/. All consumers "
        f"must use ToolRegistry.get_memory_tool_names(). Hits:\n{result.stdout}"
    )


def test_tool_registry_default_tools_matches_old_literal():
    """Verify the registry produces the same 13 tools the old
    DEFAULT_TOOL_NAMES literal had. Prevents silent tool-set drift
    during the consolidation."""
    from proxy.tool_pipeline.tool_registry import registry

    # 12 tools (was 13 in the old flat list — 'question' moved to
    # PASSTHROUGH in the registry, which is a refinement not a regression).
    expected = {
        "bash", "read", "glob", "grep", "edit", "write", "webfetch",
        "MiniMax_web_search", "suggest", "task", "recall", "search",
    }
    actual = registry.get_default_tools()
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"ToolRegistry.get_default_tools() drifted from the old "
        f"DEFAULT_TOOL_NAMES literal. Missing: {missing}. Extra: {extra}."
    )


def test_tool_registry_memory_tools_matches_old_literal():
    """Verify the registry produces the same 4 tools the old
    MEMORY_TOOL_NAMES literal had."""
    from proxy.tool_pipeline.tool_registry import registry

    expected = {
        "recall", "search", "proxy_ticket", "proxy_work",
        "proxy_ticket_list_tasks", "proxy_ticket_get_task",
        "proxy_ticket_create_tasks", "proxy_ticket_update_task",
        "proxy_ticket_edit", "proxy_ticket_set_parent",
        "proxy_ticket_clear_parent", "proxy_ticket_clear_blocked_by",
        "proxy_ticket_set_priority", "proxy_ticket_append_review_note",
    }
    actual = registry.get_memory_tool_names()
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"ToolRegistry.get_memory_tool_names() drifted from the old "
        f"MEMORY_TOOL_NAMES literal. Missing: {missing}. Extra: {extra}."
    )
