# Test Suite

> For test patterns (US-XXX traceability, builder usage, layer choice, flaky-test handling, common pitfalls), see [`docs/TEST_PATTERNS.md`](../docs/TEST_PATTERNS.md).

## Directory Structure

```
tests/
├── unit/           # Layer 1: 11 tests, 100% MOCKED
├── integration/    # Layer 2: 19 tests, MOCKED LLM
├── e2e/            # Layer 3: 9 tests, REAL LLM
├── fixtures/       # Shared test fixtures (conftest.py, prompts, etc.)
├── deprecated/     # Deprecated test files (preserved for reference)
└── README.md       # This file
```

## Test Counts

| Layer | Count | Description |
|-------|-------|-------------|
| Unit | 11 | 100% MOCKED - fast, no external dependencies |
| Integration | 19 | MOCKED LLM - tests proxy integration |
| E2E | 9 | REAL LLM - full system validation |

## Running Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run all integration tests
pytest tests/integration/ -v

# Run all E2E tests
pytest tests/e2e/ -v

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/unit/test_smart_validation_unit.py -v
```

## E2E Real Tests (`tests/e2e_real/`)

The E2E real tests validate the full sync pipeline with real SOTA API calls and Neo4j.
They require a running proxy, llama-server, Neo4j, and the SOTA API key.

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEO4J_AUTH` | Neo4j auth credentials | `neo4j:password` |
| `PROXY_SOTA_API_KEY` | SOTA API key for memory extraction | `sk-...` |
| `PROXY_SOTA_ENDPOINT` | SOTA API endpoint | `https://api.minimax.io/v1` |
| `PROXY_SOTA_MODEL` | SOTA model name | `MiniMax-M2.7` |
| `PROXY_TEST_DB` | (Optional) Test database path for isolation | `data/memory_test.db` |

### Running E2E Real Tests

**Step 1**: Start Neo4j:
```bash
# macOS with Homebrew
neo4j start

# Verify it's running
neo4j status

# Set auth (or use NEO4J_AUTH env var)
NEO4J_AUTH="neo4j:your-neo4j-password"
```

**Step 2**: Start the proxy with test database:
```bash
cd /path/to/documents/qwen-adapter

PROXY_TEST_DB="data/memory_test.db" \
NEO4J_AUTH="neo4j:your-neo4j-password" \
PROXY_SOTA_API_KEY="sk-..." \
PROXY_SOTA_ENDPOINT="https://api.minimax.io/v1" \
PROXY_SOTA_MODEL="MiniMax-M2.7" \
PYTHONPATH=src python3 src/qwen_proxy.py
```

**Step 3**: Run the tests:
```bash
PYTHONPATH=src \
NEO4J_AUTH="neo4j:your-neo4j-password" \
PROXY_SOTA_API_KEY="sk-..." \
PROXY_SOTA_ENDPOINT="https://api.minimax.io/v1" \
PROXY_SOTA_MODEL="MiniMax-M2.7" \
python3 -m pytest tests/e2e_real/ -v
```

### Test Database Isolation

E2E real tests use `PROXY_TEST_DB` to isolate test data from the production database.
The proxy reads `PROXY_TEST_DB` at runtime and writes to that path instead of `data/memory.db`.
Test fixtures automatically set this path and clean up after each test.

## E2E Memory Test Setup

E2E memory tests require Neo4j to be running:

```bash
# Start Neo4j (Docker)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 neo4j:5.0.0

# Verify connection
python -c "from neo4j import GraphDatabase; print('Neo4j connection OK')"
```

## Golden Dataset Update Workflow

When tool schemas or behavior changes:

1. **Unit tests** validate constant values (e.g., `OPENCODE_TOOL_NAMES`)
2. **Integration tests** validate parsing logic against fixtures
3. **E2E tests** validate against golden conversation traces

To update golden datasets:
```bash
# Run E2E tests in capture mode
GOLDEN_UPDATE=1 pytest tests/e2e/ -v

# Review changes
git diff tests/e2e/fixtures/
```

## Dependency Map

```
tests/
├── conftest.py          # Shared fixtures (OPENCODE_TOOL_NAMES, etc.)
├── fixtures/
│   ├── prompts/          # E2E test prompts
│   └── golden/          # Expected outputs for regression testing
├── unit/
│   ├── test_keyword_analyzer_unit.py
│   ├── test_proxy_recovery_unit.py
│   ├── test_smart_validation_unit.py
│   └── ...              # 11 unit tests
├── integration/
│   ├── test_keyword_analyzer_integration.py
│   ├── test_proxy_validation_integration.py
│   └── ...              # 19 integration tests
└── e2e/
    ├── test_e2e_context_preservation.py
    ├── test_e2e_error_recovery.py
    ├── test_e2e_tool_selection.py
    └── ...              # 9 E2E tests
```

## Layer Definitions

### Layer 1: Unit Tests
- Test individual functions in isolation
- Mock all external dependencies (LLM, Neo4j, file system)
- Fast execution (~seconds total)

### Layer 2: Integration Tests
- Test component interactions
- Mock LLM responses, real or mock Neo4j
- Validate parsing, schema checking, tool selection logic

### Layer 3: E2E Tests
- Test full system with real LLM
- Require running llama-server and Neo4j
- Validate end-to-end tool selection and execution

## Pass Criteria

| Criterion | Definition |
|-----------|------------|
| Correct Tool Call | Schema validity + Canonical naming + Semantic validity + Required params 100% + Param types correct |
| OUTPUT Rules | finish_reason consistency + JSON-parseable args + schema valid + allowed tool set |
| E2E | Tool selection EXACT MATCH + Schema 100% + Required param accuracy 100% + File state verified |
| Memory Tools | All 6 memory tools parseable + params normalized + context injection verified |
| OPENCODE_TOOL_NAMES | 12 tools (6 original + 6 memory), validated by unit test |