"""
Test database helper for sync pipeline tests.

Provides functions to initialize, query, and clean up the database
(data/memory.db) used by the proxy. Since the proxy runs as a separate
process, both the test and proxy must use the same default database path.
"""

import os
import sqlite3
import time
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_REQUIRED_TABLES = {"extraction_batches", "conversation_summaries", "messages"}


def _db_has_required_tables(path: str) -> bool:
    """Return True if the SQLite file at *path* contains all required tables."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        return _REQUIRED_TABLES.issubset(tables)
    except Exception:
        return False


# Use PROXY_TEST_DB if set and valid, else memory_test.db if valid, else memory.db
_env_db = os.environ.get("PROXY_TEST_DB", "")
_test_db = str(_DATA_DIR / "memory_test.db")
_prod_db = str(_DATA_DIR / "memory.db")

if _env_db and os.path.exists(_env_db) and _db_has_required_tables(_env_db):
    TEST_DB_PATH = _env_db
elif os.path.exists(_test_db) and _db_has_required_tables(_test_db):
    TEST_DB_PATH = _test_db
else:
    TEST_DB_PATH = _prod_db

print(f"[TEST DB] Env PROXY_TEST_DB={os.environ.get('PROXY_TEST_DB', '<unset>')}", flush=True)
print(f"[TEST DB] Prod DB: {_prod_db} exists={os.path.exists(_prod_db)} has_tables={_db_has_required_tables(_prod_db) if os.path.exists(_prod_db) else 'N/A'}", flush=True)
print(f"[TEST DB] Test DB: {_test_db} exists={os.path.exists(_test_db)} has_tables={_db_has_required_tables(_test_db) if os.path.exists(_test_db) else 'N/A'}", flush=True)
print(f"[TEST DB] >>> Selected: {TEST_DB_PATH}", flush=True)

from memory.db import SCHEMA_SQL


def init_test_db():
    """Initialize test database. The proxy creates it when it starts."""
    # Just verify the database exists
    if os.path.exists(TEST_DB_PATH):
        print(f"[TEST DB] Database exists at {TEST_DB_PATH}", flush=True)
    else:
        print(
            f"[TEST DB] WARNING: Database not found at {TEST_DB_PATH}. Is proxy running?",
            flush=True,
        )


def get_test_connection():
    """Get connection to test database."""
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def check_sync_completed(session_id, timeout=60):
    """
    Check if sync completed for a session in the test database.

    Returns:
        tuple: (sqlite_ok, neo4j_ok, batch_status)
    """
    print(f"[TEST DB] Checking sync for session {session_id}", flush=True)

    for i in range(timeout * 10):  # Check every 100ms
        try:
            conn = get_test_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, status FROM extraction_batches WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                batch_id = row["id"]
                status = row["status"]
                print(
                    f"[TEST DB] Found batch {batch_id} with status '{status}'",
                    flush=True,
                )

                if status == "completed":
                    return True, True, status
                elif status in ["extracted", "neo4j_completed"]:
                    return True, False, status
                elif status in ["failed"]:
                    return False, False, status
                elif status in ["neo4j_failed"]:
                    # SQLite extraction completed, Neo4j store failed
                    # Treat as success for SQLite verification
                    return True, False, status
                # status == "pending": keep polling, sync still running
            else:
                print(
                    f"[TEST DB] No batch found yet (attempt {i + 1}/{timeout * 10})",
                    flush=True,
                )

        except Exception as e:
            print(f"[TEST DB] Error: {e}", flush=True)

        time.sleep(0.1)

    return False, False, "timeout"


def cleanup_test_db():
    """Clean up test database - truncate tables instead of deleting file."""
    try:
        conn = get_test_connection()
        conn.execute("DELETE FROM extraction_batches")
        conn.execute("DELETE FROM conversation_summaries")
        conn.execute("DELETE FROM messages")
        conn.commit()
        conn.close()
        print(f"[TEST DB] Cleaned up database (truncated tables)", flush=True)
    except Exception as e:
        print(f"[TEST DB] Warning: Could not clean up database: {e}", flush=True)
