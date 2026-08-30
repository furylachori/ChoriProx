# Block: Validation Code

> Part of [IMPLEMENTATION_SPEC](../IMPLEMENTATION_SPEC.md). See also [Spec](../spec/02-architecture.md)

## 6. Smart Proxy Validation

**File:** src/smart_proxy_validation.py (285 lines)

The SmartProxyValidation class provides semantic analysis of malformed tool calls.

### 6.1 Pattern Detection

4 problem patterns with regex matching, risk levels, and semantic analysis:
- missing_description (low risk, auto_fix)
- missing_required_params (medium risk, clarification)
- parameter_type_mismatch (medium risk, clarification)
- invalid_tool_name (high risk, progressive_assistance)

### 6.2 Auto-Fix Strategy

Low-risk patterns: auto-fix by injecting description from DESCRIPTION_MAPPING.
Medium-risk: generate clarification request with schema.
High-risk: progressive assistance (clarification -> detailed examples -> alternative suggestions).

### 6.3 Progressive Assistance Levels

- First failure: clarification request with current vs expected format
- Second failure: detailed correct-call examples
- Third+ failure: alternative tool suggestions

Thread-safe via RLock (smart_proxy_validation.py:66).
