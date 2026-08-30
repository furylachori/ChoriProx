.PHONY: help test test-unit test-integration test-proxy test-e2e test-e2e-real test-clients test-root coverage install-test-deps check debug-templates

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
WORKERS ?= 4

help:
	@echo "Test targets:"
	@echo "  test              Run unit + integration + proxy + tests/ root tests in parallel (-n $(WORKERS))"
	@echo "  test-unit         Run unit tests only (-n $(WORKERS))"
	@echo "  test-integration  Run integration tests only (-n $(WORKERS))"
	@echo "  test-proxy        Run proxy-layer tests only (-n $(WORKERS))"
	@echo "  test-e2e          Run mocked e2e tests only (-n $(WORKERS))"
	@echo "  test-e2e-real     Run real LLM e2e tests (-n 2)"
	@echo "  test-clients      Run all clients layer tests"
	@echo "  test-root         Run tests/ root only (skips unit/integration/proxy/e2e/e2e_real subtrees)"
	@echo "  coverage          Run coverage with floor gates: clients 90 / memory 56 / proxy 66 / overall 63 + 8 per-file gates"
	@echo "  install-test-deps Install test dependencies"
	@echo ""
	@echo "  Worker count: $(WORKERS) (override with make test WORKERS=8)"

install-test-deps:
	$(PYTHON) -m pip install -e ".[test]"

test-unit:
	$(PYTEST) tests/unit -m "not e2e_real" -n $(WORKERS) --dist=loadfile -v

# Integration tests target the isolated neo4j-test container (Enterprise Edition,
# bolt://localhost:7688) so they can CREATE DATABASE per-test and NEVER touch the
# prod graph on bolt://localhost:7687. Start it with:
#   docker compose -f docker-compose.test.yml up -d neo4j-test
test-integration:
	docker compose -f docker-compose.test.yml up -d --wait neo4j-test 2>/dev/null || true
	PROXY_NEO4J_BOLT_URL=bolt://localhost:7688 NEO4J_AUTH=neo4j:testpassword \
	$(PYTEST) tests/integration -m "not e2e_real" -n $(WORKERS) --dist=loadfile -v

test-proxy:
	$(PYTEST) tests/proxy -m "not e2e_real" -n $(WORKERS) --dist=loadfile -v

test-e2e:
	$(PYTEST) tests/e2e -m "not e2e_real" -n $(WORKERS) --dist=loadfile -v

test-e2e-real:
	$(PYTEST) tests/e2e_real -m e2e_real -n 2 --dist=loadfile -v

test-clients:
	$(PYTEST) tests/unit/clients -n $(WORKERS) -v

# tests/ root contains 140+ top-level test files (tests/test_*.py) that
# pre-existed alongside the unit/integration/proxy subtree targets. This
# target runs ONLY the root tests (skips the per-layer subtrees that have
# their own targets above). Using the test_*.py glob avoids collecting
# from tests/load, tests/memory, tests/skills (which are orphan dirs not
# part of the unit/integration/proxy layers).
test-root:
	$(PYTEST) tests/test_*.py -n $(WORKERS) --dist=loadfile -v

test: test-unit test-integration test-proxy test-root

coverage:
	@rm -f .coverage .coverage.*
	$(PYTEST) tests/unit tests/integration -n $(WORKERS) --dist=loadfile --cov=src --cov-branch --cov-report= --cov-config=pyproject.toml
	@coverage combine --quiet || true
	@echo "=== src/clients/ coverage (target: 90%) ==="
	coverage report --include='src/clients/*' --fail-under=90
	@echo "=== src/memory/ coverage floor (current ~58%, gate 56) ==="
	coverage report --include='src/memory/*' --fail-under=56
	@echo "=== src/proxy/ coverage floor (current ~68%, gate 66) ==="
	coverage report --include='src/proxy/*' --fail-under=66
	@echo "=== Per-file gates (stable, well-tested modules) ==="
	coverage report --include='src/memory/extraction_validator.py' --fail-under=96
	coverage report --include='src/memory/circuit_breaker.py' --fail-under=90
	coverage report --include='src/memory/security.py' --fail-under=88
	coverage report --include='src/memory/bug_tracker.py' --fail-under=83
	coverage report --include='src/memory/db_recovery.py' --fail-under=45
	coverage report --include='src/proxy/chat_history.py' --fail-under=90
	coverage report --include='src/proxy/auth.py' --fail-under=89
	coverage report --include='src/proxy/rate_limiter.py' --fail-under=89
	coverage report --include='src/proxy/vfs.py' --fail-under=87
	@echo "=== Overall coverage (target: 70%) ==="
	coverage report --fail-under=63
	coverage html

check:
	$(PYTHON) scripts/check_mock_drift.py
	$(PYTHON) scripts/check_imports.py

# Run the debug_session reproducer templates (tests/debug/templates/*.py).
# Each template exits 0 on all-ok / 1 on any failure via _finish() in
# _helpers.py. Templates use `sys.path.insert(0, os.path.dirname(__file__))`
# to import _helpers (NOT the Makefile's PYTHONPATH). Each template MUST
# run as a separate process (canary.py:98 caches _SESSION_SECRET at import).
debug-templates:
	@for f in tests/debug/templates/[a-z]*.py; do \
		echo "=== $$f ==="; \
		PYTHONPATH=src PROXY_SESSION_SECRET="" $(PYTHON) $$f || exit 1; \
	done
