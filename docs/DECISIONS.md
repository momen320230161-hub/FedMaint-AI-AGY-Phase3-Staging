# Architecture and Research Decision Log

This document makes assumptions explicit. **Proposal** is not the same as **approved** or **implemented**.

| ID | Decision | Current position | Status |
| --- | --- | --- | --- |
| D-001 | Dataset tracks | Azure classification main track; NASA C-MAPSS separate RUL benchmark | Proposed |
| D-002 | Azure label & windowing | Binary failure in `(t, t+24h]`; past-only `[t-48h, t)` telemetry; `[t-48h, t]` blackout; simultaneous failure collapse; right-censored rejection; no future events in samples | **Approved (Phase 2B)** |
| D-003 | Azure participating FL clients | Group known machines by model2, model3, model4 as 3 non-IID clients | **Approved (Phase 2B)** |
| D-004 | Unseen factory | All 16 model1 machines strictly withheld from train/val; final test evaluation only | **Approved (Phase 2B)** |
| D-005 | Predictive backbone | GRU sequence encoder, dataset-specific classifier/RUL head; matched within each strategy | **Candidate** — not benchmarked |
| D-006 | Privacy mechanism | DP or secure aggregation; threat model first; avoid claiming ordinary FL guarantees privacy | Open |
| D-007 | Graph architecture | Shared ontology, client-local event facts, provenance-aware retrieval; no initial GNN requirement | **Candidate** |
| D-008 | LLM evidence policy | Model output + verified local KG facts; compare grounded vs ungrounded | Required, hosting/rubric open |
| D-009 | Dataset separation | Do not merge Azure and NASA feature/label tables or reuse their model weights | Proposed |
| D-010 | NASA windows | 30 cycles as first tested window length; per-cycle RUL reconstruction | **Candidate** |
| D-011 | Azure window normalization | Per-window median and IQR (linear interpolation); exact zero IQR fallback 1.0; raw summaries retained | **Approved (Phase 2B)** |
| D-012 | Client-count study | Repartition training machines into additional whole-machine clients as a secondary experiment | **Candidate** |
| D-013 | Azure temporal splits | Half-open intervals: train `[2015-01-01 06:00, 2015-09-01 06:00)`, val `[2015-09-01 06:00, 2015-11-01 06:00)`, test `[2015-11-01 06:00, 2016-01-01 06:00)`; strict boundary membership | **Approved (Phase 2B)** |
| D-014 | Feature scope & causality | Primary features strictly telemetry-only (`volt`, `rotate`, `pressure`, `vibration`); no same-hour maintenance/error joins | **Approved (Phase 2B)** |
| D-015 | Window overlap semantics | Adjacent positive hourly windows overlap heavily (up to 23 hours); positive windows are NOT independent failure episodes | **Approved (Phase 2B)** |

---

## Approved Phase 2B Decision Records

### D-002 — Azure Prediction Unit, Observation Window, Blackout, and Label Horizon
- **Status:** Approved (Phase 2B)
- **Context:** Define a leakage-free prediction task and observation structure for Azure PdM failure prediction.
- **Decision:**
  1. **Prediction Unit:** One machine at hourly prediction timestamp $t$.
  2. **Observation History:** Exactly 48 hourly readings in $[t-48\text{h}, t)$, i.e. $t-48\text{h}, \dots, t-1\text{h}$. No reading at $t$ or later is included in predictors.
  3. **Label Horizon:** Binary $y = 1$ iff at least one distinct failure/replacement episode occurs in $(t, t+24\text{ hours}]$. $y = 0$ only when the full 24-hour horizon is observable without event.
  4. **Right-Censored Rejection:** If a machine's telemetry coverage ends before $t+24\text{h}$, the outcome horizon is not fully observable. The window is marked right-censored and rejected; it is never silently labeled negative.
  5. **Episode Collapse:** Collapse all component failure records sharing `machineID` and exact timestamp into one single episode. Repeated timestamps at different hours remain distinct episodes.
  6. **Blackout Assumption:** Exclude candidate prediction time $t$ if any failure episode occurs at $t$ or in the prior 48 hours ($[t-48\text{h}, t]$ inclusive).
  7. **Future Event Isolation:** Exported training samples (`AzureSample`) contain only past normalized features, binary target label, raw sensor summaries, and observation intervals. Future event records (component identities, failure timestamps) are strictly omitted from sample payloads; episode auditing is retained exclusively in summary statistics.
- **Semantics & Limitations:** This blackout is an explicit research assumption representing post-failure repair and recovery time, not a physical failure onset. Failures in Azure PdM represent recorded replacement events.
- **Impact:** Implemented in `src/fedmaint/data/episodes.py` and `src/fedmaint/data/pipeline.py`.

### D-003 & D-004 — Participating FL Clients and Held-Out Factory (model1)
- **Status:** Approved (Phase 2B)
- **Context:** Establish non-IID simulated clients and a strict unseen-domain evaluation partition.
- **Decision:**
  1. Participating federated clients are grouped strictly by machine model: `client_model2` (17 machines), `client_model3` (35 machines), and `client_model4` (32 machines). These participate in `train`, `val`, and `test`.
  2. Held-out factory: all 16 machines belonging to `model1` (`heldout_model1`) are strictly withheld from global training, validation, hyperparameter tuning, feature selection, threshold selection, and adaptation.
  3. `heldout_model1` is evaluated exclusively on its final `test` split. Querying it for `train` or `val` raises `HoldoutLeakageError`.
  4. Any unexpected machine model not defined in configuration is strictly rejected with `ValueError`.
- **Limitations:** These represent simulated non-IID machine groupings, not distinct commercial organizations.

### D-011 — Per-Window Normalization (Median and IQR)
- **Status:** Approved (Phase 2B)
- **Context:** Normalize sensor readings across heterogeneous machines without global scalers, future leakage, or cross-client parameter sharing.
- **Decision:**
  1. Compute median and Interquartile Range ($\text{IQR} = Q_{75} - Q_{25}$) strictly from the 48 past observations of that window.
  2. Quartile interpolation: deterministic linear method ($R-7$ / NumPy default).
  3. Safe scale fallback: when $\text{IQR} == 0.0$ (exact zero fallback, e.g. constant sensor series), scale defaults to 1.0.
  4. Raw sensor summaries (`median`, `iqr`, `scale`, `q25`, `q75`) are preserved in every sample alongside normalized values.
  5. Strict nonfinite checking: nonfinite raw, quantile, or normalized values are rejected deterministically.
  6. Zero population or cross-client scalers.

### D-013 — Temporal Splits and Boundary Rules
- **Status:** Approved (Phase 2B)
- **Context:** Enforce strict temporal ordering and prevent horizon crossing across splits.
- **Decision:**
  1. Half-open intervals:
     - `train`: `[2015-01-01 06:00:00, 2015-09-01 06:00:00)`
     - `val`: `[2015-09-01 06:00:00, 2015-11-01 06:00:00)`
     - `test`: `[2015-11-01 06:00:00, 2016-01-01 06:00:00)`
  2. Sample inclusion rule: a sample belongs to split $[T_{\text{start}}, T_{\text{end}})$ iff:
     $$t - 48\text{h} \ge T_{\text{start}} \quad \text{and} \quad t < T_{\text{end}} \quad \text{and} \quad t + 24\text{h} < T_{\text{end}} \text{ (strict)}$$
  3. Right-censored times (including $t \ge T_{\text{end}} - 24\text{h}$ and dataset end) are strictly excluded.

### D-014 — Primary Telemetry-Only Feature Scope
- **Status:** Approved (Phase 2B)
- **Context:** Avoid circularity or data leakage from same-hour event logging.
- **Decision:** Primary pipeline features are strictly the 4 telemetry sensors (`volt`, `rotate`, `pressure`, `vibration`), shape $48 \times 4$. Maintenance and error tables are not joined into primary predictors.

### D-015 — Overlap Semantics of Positive Hourly Windows
- **Status:** Approved (Phase 2B)
- **Context:** Prevent misinterpreting positive window counts as independent failure events.
- **Decision:** Consecutive hourly candidate windows for the same machine overlap heavily (sharing 47 of 48 observation hours). A single failure episode generates up to 24 consecutive positive hourly windows. Positive windows must not be treated as independent failure occurrences in evaluation or loss weighting.

---

## Unapproved Proposals (Phase 1 / Future Phases)

The following remain **unapproved candidates or proposals** and must not be treated as implemented:
- **D-001 & D-009:** Final NASA C-MAPSS protocol and cross-track benchmarking rules.
- **D-005:** GRU backbone vs alternative sequence architectures (Phase 3).
- **D-006:** Differential Privacy vs Secure Aggregation threat model and budget (Phase 5).
- **D-007:** Knowledge graph schema, ontology, and triple extraction (Phase 6).
- **D-008:** LLM grounded explanation hosting, prompt architecture, and evaluation rubric (Phase 7).
- **D-010:** NASA 30-cycle observation window and piecewise RUL capping.
- **D-012:** Secondary client-count partitioning study.
