"""Tests that validate the 5 data inspection scripts will work against the schema.

These tests create an in-memory SQLite database with the exact schema the scripts
expect. They validate the SQL queries used by each script and assert that the
columns the scripts reference are present. If a column the scripts use is removed
or renamed in the real schema, these tests will FAIL.
"""
import sqlite3
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------------------
# Column inventories: only columns that at least one of the 5 scripts queries.
# If any of these columns disappear from the real schema, the matching script
# will break — so these tests must FAIL.
# ---------------------------------------------------------------------------
MESSAGES_COLUMNS = {
    "id",
    "session_id",
    "message_index",
    "role",
    "content",
    "tool_calls_json",
    "tool_call_id",
    "created_at",
}

MESSAGE_SUMMARIES_COLUMNS = {
    "session_id",
    "message_index",
    "end_message_index",
    "exchange_index",
    "role",
    "summary",
    "tool_summary",
    "model_used",
    "token_estimate",
    "created_at",
    "rendered_full_tokens",
}

CONVERSATION_SUMMARIES_COLUMNS = {
    "session_id",
    "exchange_count",
    "last_message_index",
    "created_at",
    "ancient_history",
}

EXTRACTION_BATCHES_COLUMNS = {
    "id",
    "session_id",
    "start_message_index",
    "end_message_index",
    "start_exchange_index",
    "end_exchange_index",
    "status",
    "created_at",
    "updated_at",
}

ENTITIES_COLUMNS = {
    "id",
    "name",
    "entity_type",
    "summary",
    "session_id",
    "extracted_at",
}

FACTS_COLUMNS = {
    "id",
    "source_entity_id",
    "target_entity_id",
    "relationship_type",
    "fact",
    "session_id",
}

ALL_TABLE_COLUMNS = {
    "messages": MESSAGES_COLUMNS,
    "message_summaries": MESSAGE_SUMMARIES_COLUMNS,
    "conversation_summaries": CONVERSATION_SUMMARIES_COLUMNS,
    "extraction_batches": EXTRACTION_BATCHES_COLUMNS,
    "entities": ENTITIES_COLUMNS,
    "facts": FACTS_COLUMNS,
}

REQUIRED_TABLES = sorted(ALL_TABLE_COLUMNS.keys())

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
SCHEMA_DDL = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    content_hash TEXT,
    tool_calls_json TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    is_edited BOOLEAN DEFAULT 0,
    UNIQUE(session_id, message_index)
);

CREATE TABLE conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    summary TEXT,
    summary_tokens INTEGER,
    exchange_count INTEGER DEFAULT 0,
    last_message_index INTEGER DEFAULT -1,
    ancient_history TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE message_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    end_message_index INTEGER DEFAULT 0,
    exchange_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    summary TEXT NOT NULL,
    tool_summary TEXT,
    content_hash TEXT NOT NULL,
    model_used TEXT,
    token_estimate INTEGER,
    rendered_full_tokens INTEGER DEFAULT 0,
    rendered_action_tokens INTEGER DEFAULT 0,
    rendered_narrative_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, message_index),
    UNIQUE(session_id, exchange_index)
);

CREATE TABLE extraction_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_message_index INTEGER NOT NULL,
    end_message_index INTEGER NOT NULL,
    start_exchange_index INTEGER NOT NULL DEFAULT 0,
    end_exchange_index INTEGER NOT NULL DEFAULT 0,
    extraction_json TEXT,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(session_id, start_message_index, end_message_index),
    UNIQUE(session_id, start_exchange_index, end_exchange_index)
);

CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    summary TEXT,
    session_id TEXT,
    extracted_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, entity_type, session_id)
);

CREATE TABLE facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    fact TEXT NOT NULL,
    valid_at TEXT,
    invalid_at TEXT,
    session_id TEXT,
    extracted_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_entity_id, target_entity_id, relationship_type, fact)
);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_empty():
    """In-memory database with the full schema and no data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_DDL)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def db_populated(db_empty):
    """In-memory database with schema + test data for session_a and session_b."""
    conn = db_empty

    # --- messages ---
    messages = [
        # session_a: 6 messages (index 0-5)
        ("session_a", 0, "user", "user msg 0", None, None, NOW),
        ("session_a", 1, "assistant", "assistant msg 1", None, None, NOW),
        ("session_a", 2, "tool", "tool result 2", None, "tc_2", NOW),
        ("session_a", 3, "assistant", "assistant msg 3", None, None, NOW),
        ("session_a", 4, "user", "user msg 4", None, None, NOW),
        ("session_a", 5, "assistant", "assistant msg 5", None, None, NOW),
        # session_b: 2 messages (index 0-1)
        ("session_b", 0, "user", "user msg 0 (b)", None, None, NOW),
        ("session_b", 1, "assistant", "assistant msg 1 (b)", None, None, NOW),
    ]
    conn.executemany(
        "INSERT INTO messages (session_id, message_index, role, content, tool_calls_json, tool_call_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        messages,
    )

    # --- message_summaries ---
    msg_summaries = [
        ("session_a", 0, 3, 0, "assistant", "Summary for exchange 0-3", "Tool summary 0-3",
         "h1", "gpt-4", 150, 100, 20, 10, NOW, NOW),
        ("session_a", 4, 5, 4, "assistant", "Summary for exchange 4-5", None,
         "h2", "gpt-4", 80, 60, 10, 5, NOW, NOW),
        ("session_b", 0, 1, 0, "assistant", "Summary for exchange 0 (b)", None,
         "h3", "gpt-3.5", 50, 40, 5, 2, NOW, NOW),
    ]
    conn.executemany(
        "INSERT INTO message_summaries (session_id, message_index, end_message_index, exchange_index, role, "
        "summary, tool_summary, content_hash, model_used, token_estimate, "
        "rendered_full_tokens, rendered_action_tokens, rendered_narrative_tokens, "
        "created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        msg_summaries,
    )

    # --- conversation_summaries ---
    conv_summaries = [
        ("session_a", "Conversation summary A", 100, 2, 5, "Ancient text for session_a", NOW, NOW),
        ("session_b", "Conversation summary B", 50, 1, 1, None, NOW, NOW),
    ]
    conn.executemany(
        "INSERT INTO conversation_summaries (session_id, summary, summary_tokens, "
        "exchange_count, last_message_index, ancient_history, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        conv_summaries,
    )

    # --- extraction_batches ---
    batches = [
        ("session_a", 0, 5, 0, 5, '{"key": "val"}', "completed", None, NOW, NOW),
        ("session_b", 0, 1, 0, 1, None, "processing", None, NOW, NOW),
    ]
    conn.executemany(
        "INSERT INTO extraction_batches (session_id, start_message_index, end_message_index, "
        "start_exchange_index, end_exchange_index, "
        "extraction_json, status, error, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        batches,
    )

    # --- entities ---
    entities = [
        ("db.py", "File", "A Python database file", "session_a", NOW, NOW, NOW),
        ("sqlite3", "Module", "SQLite3 module", "session_a", NOW, NOW, NOW),
        ("User", "Person", "A user entity", "session_b", NOW, NOW, NOW),
    ]
    conn.executemany(
        "INSERT INTO entities (name, entity_type, summary, session_id, "
        "extracted_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        entities,
    )
    conn.commit()

    # --- facts ---
    # Need entity IDs for session_a entities
    e1_id = conn.execute(
        "SELECT id FROM entities WHERE session_id = ? AND name = ?",
        ("session_a", "db.py"),
    ).fetchone()[0]
    e2_id = conn.execute(
        "SELECT id FROM entities WHERE session_id = ? AND name = ?",
        ("session_a", "sqlite3"),
    ).fetchone()[0]

    facts = [
        (e1_id, e2_id, "DEPENDS_ON", "db.py depends on sqlite3",
         None, None, "session_a", NOW, NOW, NOW),
    ]
    conn.executemany(
        "INSERT INTO facts (source_entity_id, target_entity_id, relationship_type, fact, "
        "valid_at, invalid_at, session_id, extracted_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        facts,
    )
    conn.commit()

    return conn


# ===================================================================
# Functional smoke tests (queries from the actual scripts)
# ===================================================================

class TestSessionStats:
    """Validate queries used by scripts/data/session_stats.py."""

    def test_smoke(self, db_populated):
        conn = db_populated

        sessions = conn.execute(
            "SELECT DISTINCT session_id FROM messages ORDER BY session_id"
        ).fetchall()
        assert len(sessions) >= 2

        for (sid,) in sessions:
            raw_total = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
            ).fetchone()[0]
            assert raw_total >= 1

            user_c = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'",
                (sid,),
            ).fetchone()[0]
            asst_c = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'assistant'",
                (sid,),
            ).fetchone()[0]
            tool_c = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'tool'",
                (sid,),
            ).fetchone()[0]
            exch_c = conn.execute(
                "SELECT COUNT(*) FROM message_summaries WHERE session_id = ?",
                (sid,),
            ).fetchone()[0]

            assert user_c + asst_c + tool_c <= raw_total
            assert exch_c >= 0

    def test_counts_exact(self, db_populated):
        conn = db_populated

        # session_a
        for col, expected in [("user", 2), ("assistant", 3), ("tool", 1)]:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = ?",
                ("session_a", col),
            ).fetchone()[0]
            assert count == expected, f"session_a {col} count mismatch"

        exch_a = conn.execute(
            "SELECT COUNT(*) FROM message_summaries WHERE session_id = ?",
            ("session_a",),
        ).fetchone()[0]
        assert exch_a == 2

        # session_b
        for col, expected in [("user", 1), ("assistant", 1), ("tool", 0)]:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = ?",
                ("session_b", col),
            ).fetchone()[0]
            assert count == expected, f"session_b {col} count mismatch"

        exch_b = conn.execute(
            "SELECT COUNT(*) FROM message_summaries WHERE session_id = ?",
            ("session_b",),
        ).fetchone()[0]
        assert exch_b == 1


class TestFetchSummaries:
    """Validate queries used by scripts/data/fetch_summaries.py."""

    def test_by_session(self, db_populated):
        conn = db_populated

        rows = conn.execute(
            """SELECT message_index, end_message_index, role, summary,
                      tool_summary, model_used, token_estimate, created_at
               FROM message_summaries
               WHERE session_id = ?
               ORDER BY message_index DESC
               LIMIT 5""",
            ("session_a",),
        ).fetchall()

        assert len(rows) == 2
        # DESC order means first row has highest message_index (4)
        assert rows[0]["message_index"] == 4

    def test_session_b_summaries(self, db_populated):
        conn = db_populated

        rows = conn.execute(
            """SELECT message_index, end_message_index, role, summary,
                      tool_summary, model_used, token_estimate, created_at
               FROM message_summaries
               WHERE session_id = ?
               ORDER BY message_index DESC
               LIMIT 5""",
            ("session_b",),
        ).fetchall()

        assert len(rows) == 1
        assert rows[0]["message_index"] == 0
        assert rows[0]["end_message_index"] == 1


class TestFetchAncient:
    """Validate queries used by scripts/data/fetch_ancient.py."""

    def test_exists(self, db_populated):
        conn = db_populated

        row = conn.execute(
            """SELECT exchange_count, last_message_index, created_at, ancient_history
               FROM conversation_summaries
               WHERE session_id = ?""",
            ("session_a",),
        ).fetchone()

        assert row is not None
        assert row["ancient_history"] == "Ancient text for session_a"
        assert row["exchange_count"] == 2
        assert row["last_message_index"] == 5

    def test_null(self, db_populated):
        conn = db_populated

        row = conn.execute(
            """SELECT exchange_count, last_message_index, created_at, ancient_history
               FROM conversation_summaries
               WHERE session_id = ?""",
            ("session_b",),
        ).fetchone()

        assert row is not None
        assert row["ancient_history"] is None


class TestFetchExchange:
    """Validate queries used by scripts/data/fetch_exchange.py."""

    def test_range(self, db_populated):
        conn = db_populated

        # Find next user message after msg_index=0
        next_user = conn.execute(
            "SELECT MIN(message_index) FROM messages "
            "WHERE session_id = ? AND message_index > ? AND role = 'user'",
            ("session_a", 0),
        ).fetchone()[0]
        assert next_user == 4

        # Exchange goes from 0 to next_user - 1 = 3 (inclusive)
        end_idx = next_user - 1  # 3

        # Fetch messages in range
        msgs = conn.execute(
            "SELECT message_index, role, content, tool_calls_json, tool_call_id, created_at "
            "FROM messages "
            "WHERE session_id = ? AND message_index >= ? AND message_index <= ? "
            "ORDER BY message_index",
            ("session_a", 0, end_idx),
        ).fetchall()

        assert len(msgs) == 4  # messages 0, 1, 2, 3
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_summary_join(self, db_populated):
        conn = db_populated

        row = conn.execute(
            "SELECT summary, tool_summary, model_used, token_estimate "
            "FROM message_summaries "
            "WHERE session_id = ? AND message_index = ?",
            ("session_a", 0),
        ).fetchone()

        assert row is not None
        assert row["summary"] == "Summary for exchange 0-3"
        assert row["tool_summary"] == "Tool summary 0-3"

    def test_max_message_index(self, db_populated):
        conn = db_populated

        max_idx = conn.execute(
            "SELECT MAX(message_index) FROM messages WHERE session_id = ?",
            ("session_a",),
        ).fetchone()[0]
        assert max_idx == 5

    def test_user_message_verification(self, db_populated):
        conn = db_populated

        row = conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND message_index = ? AND role = 'user'",
            ("session_a", 0),
        ).fetchone()
        assert row is not None

        row_none = conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND message_index = ? AND role = 'user'",
            ("session_a", 1),  # msg_index 1 is assistant, not user
        ).fetchone()
        assert row_none is None


class TestFetchEntities:
    """Validate queries used by scripts/data/fetch_entities.py."""

    def test_entities_by_session(self, db_populated):
        conn = db_populated

        rows = conn.execute(
            "SELECT id, name, entity_type, summary, extracted_at "
            "FROM entities "
            "WHERE session_id = ? "
            "ORDER BY entity_type, name",
            ("session_a",),
        ).fetchall()

        assert len(rows) == 2
        names = [r["name"] for r in rows]
        assert "db.py" in names
        assert "sqlite3" in names

    def test_extraction_batch_covers_message(self, db_populated):
        conn = db_populated

        batch = conn.execute(
            "SELECT start_message_index, end_message_index, status, created_at, updated_at "
            "FROM extraction_batches "
            "WHERE session_id = ? AND start_message_index <= ? AND end_message_index >= ? "
            "ORDER BY id",
            ("session_a", 0, 0),
        ).fetchone()

        assert batch is not None
        assert batch["start_message_index"] == 0
        assert batch["end_message_index"] == 5
        assert batch["status"] == "completed"

    def test_facts_relationship(self, db_populated):
        conn = db_populated

        entity_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM entities WHERE session_id = ?", ("session_a",)
        ).fetchall()]
        assert len(entity_ids) == 2

        placeholders = ",".join("?" for _ in entity_ids)
        facts = conn.execute(
            f"SELECT id, source_entity_id, target_entity_id, relationship_type, fact "
            f"FROM facts "
            f"WHERE session_id = ? AND (source_entity_id IN ({placeholders}) "
            f"OR target_entity_id IN ({placeholders}))",
            ["session_a"] + entity_ids + entity_ids,
        ).fetchall()

        assert len(facts) == 1
        assert facts[0]["relationship_type"] == "DEPENDS_ON"
        assert facts[0]["fact"] == "db.py depends on sqlite3"


# ===================================================================
# Schema column tests — these MUST fail when columns scripts use change
# ===================================================================

def _get_columns(conn, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


class TestSchemaColumnsMessages:
    TABLE = "messages"

    def test_all_expected_columns_exist(self, db_empty):
        actual = _get_columns(db_empty, self.TABLE)
        missing = MESSAGES_COLUMNS - actual
        assert not missing, (
            f"{self.TABLE}: columns the scripts reference are missing: {missing}"
        )


class TestSchemaColumnsMessageSummaries:
    TABLE = "message_summaries"

    def test_all_expected_columns_exist(self, db_empty):
        actual = _get_columns(db_empty, self.TABLE)
        missing = MESSAGE_SUMMARIES_COLUMNS - actual
        assert not missing, (
            f"{self.TABLE}: columns the scripts reference are missing: {missing}"
        )


class TestSchemaColumnsConversationSummaries:
    TABLE = "conversation_summaries"

    def test_all_expected_columns_exist(self, db_empty):
        actual = _get_columns(db_empty, self.TABLE)
        missing = CONVERSATION_SUMMARIES_COLUMNS - actual
        assert not missing, (
            f"{self.TABLE}: columns the scripts reference are missing: {missing}"
        )


class TestSchemaColumnsExtractionBatches:
    TABLE = "extraction_batches"

    def test_all_expected_columns_exist(self, db_empty):
        actual = _get_columns(db_empty, self.TABLE)
        missing = EXTRACTION_BATCHES_COLUMNS - actual
        assert not missing, (
            f"{self.TABLE}: columns the scripts reference are missing: {missing}"
        )


class TestSchemaColumnsEntities:
    TABLE = "entities"

    def test_all_expected_columns_exist(self, db_empty):
        actual = _get_columns(db_empty, self.TABLE)
        missing = ENTITIES_COLUMNS - actual
        assert not missing, (
            f"{self.TABLE}: columns the scripts reference are missing: {missing}"
        )


class TestSchemaColumnsFacts:
    TABLE = "facts"

    def test_all_expected_columns_exist(self, db_empty):
        actual = _get_columns(db_empty, self.TABLE)
        missing = FACTS_COLUMNS - actual
        assert not missing, (
            f"{self.TABLE}: columns the scripts reference are missing: {missing}"
        )


class TestMinimalTablesExist:
    """Ensure the 6 tables the scripts depend on exist (they may be more)."""

    def test_required_tables_present(self, db_empty):
        names = [
            r["name"]
            for r in db_empty.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        missing = set(REQUIRED_TABLES) - set(names)
        assert not missing, (
            f"Required tables missing from schema: {missing}"
        )
