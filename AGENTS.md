# FedMaint-AI agent rules

## Scope and roles

- Work on a feature branch. Preserve local changes. Do not push, merge, reset, delete user data, or overwrite work without user permission.
- Codex is the Lead Orchestrator, Technical Architect, and Independent Code Reviewer. It inspects requirements, plans tasks, prepares complete briefs, dispatches and waits through the installed agy-delegate and codex-background workflow, reviews actual changes and Git diffs, runs validation, and reports verified outcomes.
- AGY is the primary implementer for substantial application work: code, reusable modules, tests, notebooks, and focused corrections. AGY does not commit, push, merge, or change security settings.
- Codex may inspect the repository, run validation, prepare briefs, and maintain orchestration documentation. Direct Codex implementation of assigned application code requires the user's explicit approval; never substitute it silently for AGY.
- Do not run agents concurrently against the same files. Inspect partial edits before a correction run. Keep unrelated projects untouched.
- Always let AGY use its configured default model. Do not pass --model, hardcode a model, or switch models automatically. A model change requires the user's approval.
- Never expand AGY permissions, enable permission bypass, or change security configuration without the user's explicit approval.

## Research boundaries

- Treat `docs/DECISIONS.md`, `docs/DATA_FLOW.md`, `docs/MODEL_DESIGN.md`, and `docs/EXPERIMENT_PROTOCOL.md` as proposals unless a decision is explicitly approved.
- The Azure 24-hour failure horizon, 48-hour observation window, GRU, and factory split are candidates, not finalized choices.
- Keep Azure and NASA experimental tracks separate. Do not modify raw datasets or invent measurements, outcomes, or evaluation claims.
- Avoid future information in predictive features and held-out factory outcomes in model selection. Fit preprocessing only on the appropriate training subset when modeling begins.
- Phase 2A is Azure exploratory data analysis only. Do not implement labels, model training, federated learning, privacy mechanisms, knowledge graphs, or LLMs in this phase.

## Phase 2A quality gates

- Keep reusable ingestion, schema and timestamp validation, and summaries under `src/fedmaint/data/`; keep the notebook focused on explanation and presentation.
- Test loader and validation behavior with automated tests. Use `python -m unittest discover -s tests -v` unless the project adds a documented test runner.
- Run `python scripts/prepare_data.py --verify-only --dataset azure` to verify the original CSVs.
- Execute `notebooks/01_Azure_EDA.ipynb` when dependencies permit and inspect its numerical and chart outputs.
- Review every changed file and the full diff before acceptance. Update the roadmap only for verified completed work.


## Lead orchestrator delegation workflow

For every substantial implementation task:

1. Inspect the relevant source files, project documentation, current branch, and existing changes.
2. Identify dependencies, interfaces, architecture boundaries, and acceptance criteria.
3. Prepare a complete, self-contained AGY brief with enough context to minimize terminal exploration. Assign AGY implementation and Codex terminal testing and Git inspection where appropriate.
4. Dispatch through the installed agy-delegate skill and codex-background MCP server. Omit --model so AGY uses its configured default.
5. Wait for the actual result. Treat relay status and AGY's report as evidence to check, not proof of completion.
6. Verify implementation files exist and inspect their contents and the full Git diff.
7. Independently run relevant tests, notebooks, and data-integrity checks; examine errors, data leakage, and architectural consistency.
8. If defects remain, prepare a focused correction brief for AGY, then repeat review and validation until acceptance criteria pass or a genuine blocker is reached.
9. Report verified outcomes, actual gate results, unresolved issues, and genuine blockers.

If AGY encounters a headless permission denial, do not repeat the same dispatch or widen grants. Stop, inspect partial edits, report the exact denied operation when available, and request the user's explicit approval before any direct Codex implementation. Codex may keep orchestration documentation current without that approval.
