# Block: Test Structure

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 9. Known Limitations

1. **Streaming architecture** -- force_stream=False remains. Real streaming deferred.
2. **denormalize_tool_args()** -- minimal implementation. Full inverse mappings deferred.
3. **Per-model system prompts** -- hardcoded per-client. Model-specific branching deferred.
4. **PluginAdapter** -- returns None. Kilocode plugin integration deferred.
5. **Tool parser wiring** -- QwenXMLToolParser delegates to proxy.tool_parsing.parsers.parse_all_tools_from_content (canonical parser; 7 extractors superset of legacy 5).
6. **Route table** -- no protocol detection yet. Header-based detection deferred.
7. **Re-query loop bypass** -- run_memory_tool_loop constructs requery_body directly,
   bypassing adapter preprocessing (system prompt, tool filtering, context optimization).
   This is a high-risk integration point for Phase 2 wiring.
8. **Package namespace** -- src/clients/ installs as top-level clients package.
   Restructuring under src/chori_proxy/ deferred.
9. **MEMORY_TOOL_DEFINITIONS move completed** -- now in memory/tools.py:429, single source of truth.
10. **Doc consolidation done** -- all specs merged into this file.

---

## 10. Test Structure

### Unit Tests (tests/unit/)
- tests/unit/clients/ -- 9 test files covering all clients/ modules
  - test_interface.py, test_registry.py, test_kilocode.py, test_opencode.py
  - test_capabilities.py, test_orchestrator_config.py, test_model_registry.py
  - test_init.py, test_edge_cases.py
- tests/unit/test_tool_loop.py -- tool loop extraction tests
- tests/unit/test_proxy_recovery_unit.py -- recovery system tests
- tests/unit/test_context_management_unit.py -- context optimization tests
- tests/unit/test_smart_validation_unit.py -- validation tests
- tests/unit/test_tool_optimizer_unit.py -- tool caching tests
- tests/unit/test_parsing_unit.py -- XML parser tests
- tests/unit/test_streaming_unit.py -- streaming tests
- tests/unit/test_argument_normalization_unit.py -- adapter arg mapping tests
- tests/unit/test_memory_tool_parsing_unit.py -- memory tool parsing tests
- tests/unit/test_schema_validation_unit.py -- schema validation tests
- tests/unit/test_error_handling_unit.py -- error handling tests
- tests/unit/test_opencode_tool_names_unit.py -- tool name sync tests
- tests/unit/test_memory_tool_names_sync.py -- memory tool name consistency

### Integration Tests (tests/integration/)
- tests/integration/clients/ -- 7 test files
  - test_adapter_wiring.py, test_config_loading.py, test_registry_detection.py
  - test_regression_prevention.py, test_thread_safety.py, test_import_isolation.py
  - test_wiring_coverage.py
- 14 integration test files covering:
  - argument normalization, tool selection, protocol translation
  - context management, memory context injection, SSE format
  - error recovery, proxy validation, finish reason consistency
  - SOTA client integration, parameter correctness, file state

### E2E Tests (tests/e2e/)
- 10 test files covering:
  - simple operations, tool chains, tool selection, memory tools
  - file operations, error recovery, context preservation
  - proxy fidelity, parameter correctness, negative/invalid tools

### E2E Real Tests (tests/e2e_real/)
- 3 test files for real LLM testing:
  - test_real_llm_complexity.py, test_real_llm_tool_chain.py
  - test_real_llm_tool_selection.py, test_db.py

**Total: 676 tests pass**
