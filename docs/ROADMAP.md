# FedMaint-AI — Project Roadmap

> **Status: working plan, not a report of completed experiments.** This plan follows the nine development stages in the project proposal and adds Phase 0 for repository setup. See [DECISIONS.md](DECISIONS.md) for approved vs unapproved design choices.

## Scope

Two **independent experimental tracks** share reusable infrastructure but not one combined prediction target:

- **Track A (primary): Microsoft Azure Predictive Maintenance.** Approved Phase 2B target: binary failure in `(t, t+24h]`, past-only 48h telemetry window, non-IID client partition (model2, model3, model4), and held-out model1. Focus: failure prediction, local maintenance KG, and grounded explanations.
- **Track B: NASA C-MAPSS.** Target: engine Remaining Useful Life (RUL) in operating cycles. Focus: time-series prediction and cross-domain FL generalization.

The datasets simulate machines and engine domains, **not multiple real organizations**. "Factories" refers to simulated federated clients.

## Phases and Completion Gates

| Phase | Name | Dependencies | Deliverables | Exit criteria | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | Repository foundation | Uploaded datasets | Folder structure; preparation script; design/roadmap documentation | Tracked folders and documents exist; links resolve | **Completed (scaffold only)** |
| 1 | Requirements and architecture | Phase 0 | Frozen prediction tasks, client definitions, split policy, initial ontology, privacy threat model, interfaces | Team reviews and records decisions in DECISIONS.md | **Approved for Phase 2B; future tracks in progress** |
| 2A | Azure Data Understanding & EDA | Phase 1 | Reusable loader, schema and timestamp validation, exploratory data analysis summaries | Reproducible data profiling and automated tests pass | **Completed** |
| 2B | Azure Leakage-Safe Preprocessing | Phase 2A | Episode collapse, blackout filtering, past-only window generation, client partitioning, per-window normalization, frozen config | Synthetic unit tests pass; inspectable client/split summaries; notebook executes on Azure data | **Completed (verified)** |
| 3 | Predictive baselines | Phase 2B | Matched-model local-only and centralized experiments, first sequence model | Training and evaluation repeat from configs; baseline results logged | **Not started** |
| 4 | Federated learning | Phases 2–3 | FedAvg and FedProx training loops, per-client results, communication tracking | Reproducible FL runs and like-for-like comparison to baselines | **Not started** |
| 5 | Privacy evaluation | Phase 4 | One chosen privacy mechanism (DP or secure aggregation), threat model, risk discussion | Technique implemented and evaluated; trade-offs/limitations reported | **Not started** |
| 6 | Maintenance knowledge graphs | Phases 1–2 | Shared ontology; client-local KG construction; provenance-aware retrieval | Retrieval returns traceable facts without invented causal relationships | **Not started** |
| 7 | LLM-based explainability | Phases 3 and 6 | Explanation pipeline grounded in model output and local KG evidence | Grounded vs ungrounded explanation evaluation and error analysis | **Not started** |
| 8 | Evaluation and ablations | Phases 3–7 | Cross-client/generalization results, metrics, ablations, privacy/communication analysis, multiple seeds | All required comparisons and limitations documented | **Not started** |
| 9 | Integration and final report | Phase 8 | End-to-end research demo (if feasible), reproducibility guide, report | Demo and claims match actual experimental evidence | **Not started** |

Phase 6 can progress **in parallel** with Phases 3–5 after the data and ontology contracts are stable. Phase 7 requires model predictions and a working graph retriever.

## Immediate Tasks (Phase 2B Verification & Phase 3 Preparation)
- [x] Freeze Azure Phase 2B prediction unit, horizon, observation window, and blackout rule.
- [x] Implement leak-free episode collapse, past-only windowing, and per-window median/IQR normalization.
- [x] Enforce model1 held-out client isolation (test only).
- [x] Implement synthetic test suite and reproducible preprocessing notebook.
- [x] Codex independent verification of Phase 2B gates on real data.
- [ ] Select and benchmark matched predictive backbone (Phase 3).
- [ ] Formulate privacy threat model (Phase 5).
