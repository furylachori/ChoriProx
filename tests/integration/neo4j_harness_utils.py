"""Pure utilities used by the Neo4j test harness.

Kept separate from ``neo4j_harness`` so the helpers can be unit-tested
without standing up a Neo4j driver or pulling in pytest fixtures.
"""

from __future__ import annotations

import hashlib
import re


def safe_db_name(test_name: str) -> str:
    """Convert a test node ID into a valid, unique Neo4j database name.

    Neo4j database names allow only ``[a-zA-Z0-9.-]``, must be 3-63 chars,
    and must start with a letter. The path is sanitized to allowed chars and
    an 8-char MD5 digest of the FULL test name is appended to guarantee
    uniqueness — truncating the path prefix would collide tests in the same
    class (the bug that caused CREATE OR REPLACE to destroy concurrent test
    databases).
    """
    sanitized = re.sub(r"[^a-zA-Z0-9.-]", "-", test_name).strip("-").lower()
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "t-" + sanitized
    digest = hashlib.md5(test_name.encode()).hexdigest()[:8]
    prefix = sanitized[:53]
    name = f"{prefix}-{digest}"
    if len(name) > 63:
        name = name[:63]
    return name