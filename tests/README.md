# Tests

**Scaffold only.** Automated tests will be added alongside code.

Prioritize tests for:

- Exact source-file and manifest integrity checks.
- Client splitting (assets do not accidentally cross splits).
- Time leakage (all features available at prediction time).
- RUL target generation and Azure label horizon boundaries.
- Client-local normalization without hidden shared statistics.
- FL model-update shape, aggregation, and held-out isolation.
- Knowledge graph evidence provenance and unsupported LLM claim handling.

A README here is **not** a passing test suite. Do not claim CI or unit-test coverage until tests exist and run successfully.
