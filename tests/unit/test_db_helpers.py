import hashlib
import os

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_test_db():
    os.environ["PROXY_TEST_DB"] = ":memory:"
    from memory.db import init_db, get_connection

    init_db()
    yield
    conn = get_connection()
    conn.execute("DELETE FROM message_summaries")
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM conversation_summaries")
    conn.execute("DELETE FROM extraction_batches")
    conn.commit()
    os.environ.pop("PROXY_TEST_DB", None)


# ---------------------------------------------------------------------------
# Imports (after fixture so env is set before init_db runs in fixture)
# ---------------------------------------------------------------------------
from memory.db import (  # noqa: E402
    TOKEN_RATIO,
    compute_rendered_tokens,
    delete_message_summary,
    get_connection,
    get_message_content_hash,
    get_message_summary_by_hash,
    store_message,
    store_message_summary,
    update_ancient_history,
)


# ============================================================================
# TOKEN_RATIO
# ============================================================================


def test_token_ratio():
    assert TOKEN_RATIO == 0.75


# ============================================================================
# compute_rendered_tokens
# ============================================================================


def _compute_hash(content):
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def test_compute_rendered_tokens_positive_integers():
    full, action, narrative = compute_rendered_tokens(
        0, "assistant", "user asks about file", ""
    )
    assert isinstance(full, int) and full > 0
    assert isinstance(action, int) and action > 0
    assert isinstance(narrative, int) and narrative > 0


def test_compute_rendered_tokens_hierarchy():
    """full > action > narrative when tool args contain multi-word values."""
    tool_summary = "[tools: search:query hello world, find:pattern foo bar]"
    full, action, narrative = compute_rendered_tokens(
        1, "assistant", "user asks about", tool_summary
    )
    assert full > action > narrative


def test_compute_rendered_tokens_empty_tool_summary():
    """Empty tool_summary does not crash; all three token counts are valid."""
    full, action, narrative = compute_rendered_tokens(
        5, "user", "hello world", ""
    )
    assert full >= action >= narrative >= 0


def test_compute_rendered_tokens_non_tool_summary_string():
    """A plain string tool_summary (no [tools: prefix) is treated as-is."""
    full, action, narrative = compute_rendered_tokens(
        2, "assistant", "summary", "plain tool text"
    )
    assert full == action  # non-matching prefix → tools_no_args unchanged
    assert action >= narrative


# ============================================================================
# store_message_summary
# ============================================================================


def test_store_message_summary_success():
    assert store_message_summary(
        "sid-1", 0, "user", "a summary", "", "hash-abc"
    )


def test_store_message_summary_rendered_tokens_populated():
    sid, idx = "sid-tokens", 42
    summary = "user asks about the database"
    tool_summary = "[tools: search:query hello world, find:pattern foo bar]"
    store_message_summary(sid, idx, "assistant", summary, tool_summary, "hash-t")

    conn = get_connection()
    row = conn.execute(
        """SELECT rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens
           FROM message_summaries WHERE session_id = ? AND exchange_index = ?""",
        (sid, idx),
    ).fetchone()

    assert row is not None
    assert row["rendered_full_tokens"] > 0
    assert row["rendered_action_tokens"] > 0
    assert row["rendered_narrative_tokens"] > 0
    assert row["rendered_full_tokens"] >= row["rendered_action_tokens"] >= row["rendered_narrative_tokens"]


def test_store_message_summary_upsert_different_hash():
    sid, idx = "sid-upsert", 0
    h1, h2 = "aaaa1111", "bbbb2222"

    assert store_message_summary(sid, idx, "user", "sum v1", "", h1)
    result = get_message_summary_by_hash(sid, idx, h1)
    assert result is not None
    assert result["content_hash"] == h1

    # Upsert with different hash → row updated
    assert store_message_summary(sid, idx, "assistant", "sum v2", "", h2)

    # Old hash no longer matches
    assert get_message_summary_by_hash(sid, idx, h1) is None
    # New hash matches
    result = get_message_summary_by_hash(sid, idx, h2)
    assert result is not None
    assert result["content_hash"] == h2


def test_store_message_summary_same_hash_no_change():
    """Same session+index+hast → WHERE clause prevents update but returns True."""
    sid, idx = "sid-samehash", 0
    h = "samehash01"

    assert store_message_summary(sid, idx, "user", "sum", "", h)
    assert store_message_summary(sid, idx, "user", "changed", "", h)

    result = get_message_summary_by_hash(sid, idx, h)
    assert result is not None
    assert result["content_hash"] == h


# ============================================================================
# get_message_summary_by_hash
# ============================================================================


def test_get_message_summary_by_hash_finds_existing():
    sid, idx, h = "sid-find", 5, "hash-find"
    store_message_summary(sid, idx, "user", "s", "", h)
    result = get_message_summary_by_hash(sid, idx, h)
    assert result is not None
    assert result["content_hash"] == h


def test_get_message_summary_by_hash_nonexistent():
    result = get_message_summary_by_hash("no-session", 0, "no-hash")
    assert result is None


def test_get_message_summary_by_hash_partial_match():
    """Only matches when session, index, and hash ALL match."""
    sid, idx, h = "sid-partial", 3, "hash-partial"
    store_message_summary(sid, idx, "user", "s", "", h)

    assert get_message_summary_by_hash("wrong-sid", idx, h) is None
    assert get_message_summary_by_hash(sid, 999, h) is None
    assert get_message_summary_by_hash(sid, idx, "wrong-hash") is None
    assert get_message_summary_by_hash(sid, idx, h) is not None


# ============================================================================
# get_message_content_hash
# ============================================================================


def test_get_message_content_hash_existing():
    sid, idx = "sid-msghash", 0
    content = "hello world"
    store_message(sid, idx, "user", content=content)

    expected = _compute_hash(content)
    result = get_message_content_hash(sid, idx)
    assert result == expected


def test_get_message_content_hash_nonexistent():
    assert get_message_content_hash("no-sid", 0) is None


def test_get_message_content_hash_wrong_index():
    sid = "sid-wrongidx"
    store_message(sid, 0, "user", content="data")
    assert get_message_content_hash(sid, 99) is None


# ============================================================================
# update_ancient_history
# ============================================================================


def test_update_ancient_history_conn_none():
    """When conn is None, function auto-commits."""
    sid = "sid-ahtest-auto"
    assert update_ancient_history(sid, "era: Romans")

    from memory.db import get_ancient_history

    assert get_ancient_history(sid) == "era: Romans"


def test_update_ancient_history_conn_provided():
    """When conn is provided, caller controls commit."""
    sid = "sid-ahtest-manual"
    conn = get_connection()

    # Pass conn → function does NOT commit
    assert update_ancient_history(sid, "era: Greeks", conn=conn)

    # With pooled connections, uncommitted data on conn is invisible to
    # a separate pooled connection used by get_ancient_history().
    from memory.db import get_ancient_history

    # After manual commit, data is visible to all connections
    conn.commit()
    assert get_ancient_history(sid) == "era: Greeks"


# ============================================================================
# delete_message_summary
# ============================================================================


def test_delete_message_summary_conn_none():
    sid, idx, h = "sid-del-auto", 0, "hash-del"
    store_message_summary(sid, idx, "user", "sum", "", h)
    assert get_message_summary_by_hash(sid, idx, h) is not None

    assert delete_message_summary(sid, idx) is True
    assert get_message_summary_by_hash(sid, idx, h) is None


def test_delete_message_summary_conn_provided():
    sid, idx, h = "sid-del-manual", 0, "hash-del2"
    store_message_summary(sid, idx, "user", "sum", "", h)

    conn = get_connection()
    assert delete_message_summary(sid, idx, conn=conn) is True

    # After commit, delete is visible to all pooled connections
    conn.commit()
    assert get_message_summary_by_hash(sid, idx, h) is None


# ============================================================================
# Pool path-change regression (Option B fix)
# ============================================================================


def test_pool_recreates_when_path_changes():
    """Regression guard: the global pool must be recreated when a different
    path is requested (e.g., a test fixture set PROXY_TEST_DB after the pool
    was already initialized with a different path).

    Without this fix, tests that set PROXY_TEST_DB=":memory:" after the pool
    was bound to data/memory.db would see stale data from the file DB.
    """
    import memory.db
    from memory.db import _ensure_pool, init_db

    _ensure_pool(":memory:")
    init_db()
    pool_before = memory.db._pool
    assert pool_before is not None
    assert pool_before._db_path == ":memory:"

    alt_path = "/tmp/_test_pool_recreate_alt.db"
    try:
        _ensure_pool(alt_path)
        pool_after = memory.db._pool
        assert pool_after is not None
        assert pool_after is not pool_before, "Pool should be recreated when path changes"
        # Pool stores the canonical path. Compare against the canonical
        # form to handle symlinks (e.g., /tmp → /private/tmp on macOS).
        canonical_alt = os.path.realpath(os.path.abspath(alt_path))
        assert pool_after._db_path == canonical_alt
    finally:
        if os.path.exists(alt_path):
            os.unlink(alt_path)
        # Restore pool to :memory: with schema before fixture teardown
        _ensure_pool(":memory:")
        init_db()


def test_pool_reused_when_path_unchanged():
    """Calling get_connection() repeatedly with the same path must reuse the pool."""
    import memory.db
    from memory.db import _ensure_pool

    _ensure_pool(":memory:")
    pool_first = memory.db._pool
    _ensure_pool(":memory:")
    pool_second = memory.db._pool
    assert pool_first is pool_second
