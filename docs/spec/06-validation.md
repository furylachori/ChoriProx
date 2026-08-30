# Component: Validation

> Part of [SPEC](../SPEC.md). See also [Implementation](../implementation/06-validation.md)

Intercepts malformed tool calls at the proxy layer and applies auto-fix or clarification.

## Detection Patterns

`PROBLEM_PATTERNS` maps regex patterns to tools and fix types. Known aliases (e.g. `execute_command` → `bash`) trigger suggestions; missing `description` fields trigger auto-fix.

## Auto-Fix Strategy

- **Safe auto-fixes**: Missing `description` (generated from tool name), tool name substitution, missing optional fields filled with defaults.
- **Clarification requests**: Parameter structure mismatches, wrong parameter names, type conversion ambiguities.

## Implementation Flow

```
LLM Request → Proxy → Detect Malformed Tool Call
  ├─ Safe Pattern   → Auto-Fix → Forward Fixed Call
  └─ Risky Pattern  → Clarification Request → LLM Fix → Forward
```

## Error Recovery

Progressive assistance: simple clarification on first failure, detailed examples on repeated failure, alternative tool suggestions on persistent failure. Clarification templates use schema snippets and current arguments.

## Metrics

| Metric | Target |
|--------|--------|
| E2E test pass rate | 100% |
| Context overhead reduction | 80% (8,400 → ~1,000 tokens) |
| Malformed call resolution | ≤ 2 iterations |
| Tool call success rate | 95%+ |
