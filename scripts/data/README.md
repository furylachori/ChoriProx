# Data Inspection Scripts

These scripts inspect the ChoriProx memory SQLite database (`data/memory.db`). Use them
to explore sessions, examine exchange-level summaries and raw messages, review ancient
(rolled-up) history, and inspect extracted entities and facts.

## Scripts

| Script | Usage | Description |
|---|---|---|
| `session_stats.py` | `python session_stats.py` | Print per-session statistics (message counts, exchanges, entities, tokens). |
| `fetch_summaries.py` | `python fetch_summaries.py SESSION_ID [--last N]` | Fetch the last N per-exchange summaries for a session. |
| `fetch_ancient.py` | `python fetch_ancient.py SESSION_ID` | Fetch the ancient (rolled-up) history summary for a session. |
| `fetch_exchange.py` | `python fetch_exchange.py SESSION_ID MSG_INDEX` | Fetch raw messages and the summary for a single exchange. |
| `fetch_entities.py` | `python fetch_entities.py SESSION_ID MSG_INDEX` | Fetch all entities (and their facts) extracted for a session, keyed by an exchange. |

---

### `session_stats.py`

Print an overview of every session in the database: total messages broken down by role
(user, assistant, tool), number of exchanges and summaries, last message index, whether
ancient history exists, entity count, and total token estimate. Sessions are sorted by
raw message count (descending).

**Usage:**

```
python scripts/data/session_stats.py
```

**Example:**

```
python scripts/data/session_stats.py
```

**Output:** One block per session showing counts and metadata. Example excerpt:

```
=== Session Statistics ===
Total sessions: 3

Session: abc-123
  Raw messages:    142
    user:           38
    assistant:      38
    tool:           66
  Exchanges:        37
  Summaries:        37
  Last msg index:   141
  Ancient history:  yes
  Entities:         12
  Total tokens:     45200
```

---

### `fetch_summaries.py`

Retrieve the most recent N per-exchange summaries for a given session. Each summary
covers one user→assistant exchange and includes the compressed conversation text,
tool-call summary, model used, and token estimate.

**Usage:**

```
python scripts/data/fetch_summaries.py SESSION_ID [--last N]
```

- `SESSION_ID` — the session identifier (string).
- `--last N` — number of most recent summaries to return (default: 5).

**Example:**

```
python scripts/data/fetch_summaries.py abc-123 --last 3
```

**Output:** Summaries printed oldest-first. Each shows the exchange range
(message indices), role, model, tokens, and the summary text.

---

### `fetch_ancient.py`

Fetch the "ancient history" for a session — a single rolled-up summary of all exchanges
that were evicted from the active context window. This is the long-term compressed
memory the system carries forward.

**Usage:**

```
python scripts/data/fetch_ancient.py SESSION_ID
```

- `SESSION_ID` — the session identifier.

**Example:**

```
python scripts/data/fetch_ancient.py abc-123
```

**Output:** Session metadata (exchange count, last message index, creation time)
followed by the ancient history text. Prints `No ancient history for this session.`
if none exists yet.

---

### `fetch_exchange.py`

Retrieve the raw messages and associated summary for a single exchange. An exchange
starts at the given user message index and ends at the next user message (or the last
message in the session). Content longer than 500 characters is truncated.

**Usage:**

```
python scripts/data/fetch_exchange.py SESSION_ID MESSAGE_INDEX
```

- `SESSION_ID` — the session identifier.
- `MESSAGE_INDEX` — the message index of the **user** message that starts the exchange.

**Example:**

```
python scripts/data/fetch_exchange.py abc-123 42
```

**Output:** The exchange range, then each raw message (index, role, timestamp, content,
tool calls), followed by the summary (model, tokens, summary text, tool summary).

---

### `fetch_entities.py`

Fetch all entities extracted for a session, keyed by a specific exchange. Displays the
extraction batch that covers the given message index, then lists every entity with its
type and summary, plus any relationships (facts) between them.

**Usage:**

```
python scripts/data/fetch_entities.py SESSION_ID MESSAGE_INDEX
```

- `SESSION_ID` — the session identifier.
- `MESSAGE_INDEX` — a message index within the exchange of interest (used to locate
  the extraction batch).

**Example:**

```
python scripts/data/fetch_entities.py abc-123 42
```

**Output:** The extraction batch range and status, then each entity with type, summary,
and directional facts (e.g., `→ works_at Acme Corp (organization)`).

---

## Prerequisites

- **Python 3** — the `sqlite3` module is in the standard library (no extra packages needed).
- The database file **`data/memory.db`** must exist in the project root. All scripts
  resolve it relative to their own location (`scripts/data/`). If it is missing, each
  script reports "Database not found" and exits.

## Example Workflow

A typical investigation of a session might go like this:

```bash
# 1. Get a bird's-eye view of all sessions
python scripts/data/session_stats.py

# 2. Pick a session of interest (e.g., "abc-123") and check its recent summaries
python scripts/data/fetch_summaries.py abc-123 --last 5

# 3. See if any long-term ancient history has accumulated
python scripts/data/fetch_ancient.py abc-123

# 4. Drill into a specific exchange (user message #42) to see raw messages + summary
python scripts/data/fetch_exchange.py abc-123 42

# 5. Inspect entities and facts extracted in the context of that same exchange
python scripts/data/fetch_entities.py abc-123 42
```
