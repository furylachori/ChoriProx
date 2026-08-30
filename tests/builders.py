"""Test data builder functions — plain dicts with sensible defaults.

Each builder returns a dict matching the production schema (Neo4j node
properties or SQLite row columns).  Override only what's test-specific
via positional/keyword args; pass **kwargs to add arbitrary extra fields.
"""

import uuid as _uuid_mod
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_task(uuid=None, short_hash=None, name="Test Task", status="open",
              assigned_to=None, workspace_id="ws-1", priority="medium",
              summary="A test task", created_at=None, **kwargs):
    """Return a task dict matching Neo4j task node properties."""
    if uuid is None:
        uuid = f"entity-{_uuid_mod.uuid4().hex}"
    if short_hash is None:
        short_hash = uuid.replace("entity-", "")[:8]
    if created_at is None:
        created_at = _now_iso()
    d = {
        "uuid": uuid,
        "short_hash": short_hash,
        "name": name,
        "status": status,
        "assigned_agent_id": assigned_to,
        "workspace_id": workspace_id,
        "priority": priority,
        "summary": summary,
        "created_at": created_at,
        "deleted_at": None,
        "status_changed_at": created_at,
        "started_at": None,
        "completed_at": None,
        "session_id": "test-session",
        "planning_info": None,
        "tag": None,
    }
    d.update(kwargs)
    return d


def make_bug(uuid=None, short_hash=None, title="Test Bug", status="open",
             reported_by=None, workspace_id="ws-1", severity="medium",
             description="A test bug", **kwargs):
    """Return a bug dict matching Neo4j Bug node properties."""
    if uuid is None:
        uuid = f"bug-{_uuid_mod.uuid4().hex}"
    if short_hash is None:
        short_hash = uuid.replace("bug-", "")[:8]
    if reported_by is None:
        reported_by = "agent-1"
    d = {
        "uuid": uuid,
        "short_hash": short_hash,
        "title": title,
        "status": status,
        "reported_by_agent": reported_by,
        "workspace_id": workspace_id,
        "severity": severity,
        "description": description,
        "max_reviews": 2,
        "review_count": 0,
        "validated_at": None,
        "dismissal_reason": None,
        "created_at": _now_iso(),
    }
    d.update(kwargs)
    return d


def make_review(uuid=None, bug_uuid=None, reviewer_agent_id=None,
                verdict="confirmed", content="Looks like a bug", **kwargs):
    """Return a review dict matching Neo4j Review node properties."""
    if uuid is None:
        uuid = f"review-{_uuid_mod.uuid4().hex}"
    if bug_uuid is None:
        bug_uuid = f"bug-{_uuid_mod.uuid4().hex}"
    if reviewer_agent_id is None:
        reviewer_agent_id = "agent-1"
    d = {
        "uuid": uuid,
        "bug_uuid": bug_uuid,
        "reviewer_agent_id": reviewer_agent_id,
        "verdict": verdict,
        "proposal": content,
        "confidence_score": 0.9,
        "created_at": _now_iso(),
    }
    d.update(kwargs)
    return d


def make_work_session(uuid=None, task_short_hash=None, agent_id=None,
                      status="active", started_at=None, **kwargs):
    """Return a work session dict matching Neo4j WorkSession node properties."""
    if uuid is None:
        uuid = f"ws-{_uuid_mod.uuid4().hex}"
    if task_short_hash is None:
        task_short_hash = _uuid_mod.uuid4().hex[:8]
    if agent_id is None:
        agent_id = "agent-1"
    if started_at is None:
        started_at = _now_iso()
    d = {
        "uuid": uuid,
        "task_short_hash": task_short_hash,
        "agent_id": agent_id,
        "session_id": "test-session",
        "status": status,
        "started_at": started_at,
        "ended_at": None,
        "turn_count": 0,
        "transferred_from_agent": None,
        "git_commit_sha": None,
    }
    d.update(kwargs)
    return d


def make_message(role="user", content="Hello", exchange_index=0,
                 session_id="test-session", message_index=None,
                 tool_calls=None, tool_call_id=None, **kwargs):
    """Return a message dict matching SQLite messages table schema."""
    if message_index is None:
        message_index = exchange_index * 2
    d = {
        "id": message_index + 1,
        "session_id": session_id,
        "message_index": message_index,
        "role": role,
        "content": content,
        "content_hash": None,
        "tool_calls_json": None,
        "tool_call_id": tool_call_id,
        "created_at": _now_iso(),
        "updated_at": None,
        "is_edited": 0,
        "exchange_index": exchange_index,
    }
    if tool_calls is not None:
        import json
        d["tool_calls_json"] = json.dumps(tool_calls)
    d.update(kwargs)
    return d


def make_tool_call(name="read", arguments='{"file": "test.py"}',
                   result="file contents", tool_call_id=None, **kwargs):
    """Return a tool_call dict for building message histories with tool calls."""
    if tool_call_id is None:
        tool_call_id = f"call_{_uuid_mod.uuid4().hex[:12]}"
    d = {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
        "result": result,
    }
    d.update(kwargs)
    return d


def make_memory(memory_type="fact", content="The sky is blue",
                confidence="high", source="test", **kwargs):
    """Return a memory dict matching the memory injection schema."""
    d = {
        "id": f"mem-{_uuid_mod.uuid4().hex}",
        "memory_type": memory_type,
        "content": content,
        "confidence": confidence,
        "source": source,
        "topic_key": None,
        "session_id": "test-session",
        "created_at": _now_iso(),
        "valid_at": None,
        "invalid_at": None,
        "superseded_by": None,
        "last_accessed_at": None,
        "access_count": 0,
    }
    d.update(kwargs)
    return d


# ---------------------------------------------------------------------------
# Smoke tests — verify each builder produces a valid dict with expected keys
# ---------------------------------------------------------------------------

def test_make_task_defaults():
    t = make_task()
    assert t["status"] == "open"
    assert t["priority"] == "medium"
    assert len(t["short_hash"]) == 8
    assert t["uuid"].startswith("entity-")
    assert t["deleted_at"] is None


def test_make_task_overrides():
    t = make_task(name="Custom", priority="high", extra_field=42)
    assert t["name"] == "Custom"
    assert t["priority"] == "high"
    assert t["extra_field"] == 42


def test_make_bug_defaults():
    b = make_bug()
    assert b["status"] == "open"
    assert b["severity"] == "medium"
    assert b["uuid"].startswith("bug-")
    assert b["review_count"] == 0


def test_make_review_defaults():
    r = make_review()
    assert r["verdict"] == "confirmed"
    assert r["confidence_score"] == 0.9
    assert r["uuid"].startswith("review-")


def test_make_work_session_defaults():
    ws = make_work_session()
    assert ws["status"] == "active"
    assert ws["agent_id"] == "agent-1"
    assert ws["turn_count"] == 0
    assert ws["uuid"].startswith("ws-")


def test_make_message_defaults():
    m = make_message()
    assert m["role"] == "user"
    assert m["content"] == "Hello"
    assert m["session_id"] == "test-session"
    assert m["exchange_index"] == 0


def test_make_message_with_tool_calls():
    import json
    calls = [make_tool_call()]
    m = make_message(role="assistant", tool_calls=calls)
    parsed = json.loads(m["tool_calls_json"])
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "read"


def test_make_tool_call_defaults():
    tc = make_tool_call()
    assert tc["function"]["name"] == "read"
    assert tc["id"].startswith("call_")


def test_make_memory_defaults():
    mem = make_memory()
    assert mem["memory_type"] == "fact"
    assert mem["confidence"] == "high"
    assert mem["content"] == "The sky is blue"
    assert mem["id"].startswith("mem-")
