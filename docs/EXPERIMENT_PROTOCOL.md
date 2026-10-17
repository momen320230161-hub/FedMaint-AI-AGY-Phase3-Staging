# Experiment Protocol

> **Notice:** This document distinguishes **Approved Phase 2B protocol specifications** from **unapproved downstream proposals** (model backbones, FL coordination, privacy mechanisms, Knowledge Graphs, and LLM explainability). Downstream items remain candidate proposals until their respective phases.

---

## 1. Approved Phase 2B Azure Data Preparation Protocol

### A. Prediction Task & Target Label Definition
- **Prediction Unit:** One machine at hourly prediction timestamp $t$.
- **Binary Target ($y$):**
  - $y = 1$ if at least one distinct recorded failure/replacement episode occurs in $(t, t+24\text{ hours}]$.
  - $y = 0$ only if the full 24-hour horizon is observed with no failure episode.
- **Right-Censored Horizon Rejection:**
  - If a machine's telemetry ends before $t+24\text{h}$, the future outcome horizon is unobservable. The candidate window is rejected as `right_censored` and is **never** emitted as negative $y=0$.
- **Episode Collapse:**
  - Multiple component failure records sharing `machineID` and exact timestamp collapse into a single failure episode.
  - Repeated failure timestamps at different hours remain distinct episodes.
- **Blackout Exclusion Assumption:**
  - A candidate prediction time $t$ is excluded if any failure episode occurs at $t$ or in the prior 48 hours ($[t-48\text{h}, t]$ inclusive).
  - *Research assumption note:* This blackout reflects maintenance downtime and component replacement recovery in Azure PdM, not a physical failure onset. Failures represent recorded replacement events.
- **Future Event Payload Isolation:**
  - Exported training samples (`AzureSample`) strictly omit future event records (failure components, failure timestamps). They contain normalized features, binary target label, raw sensor summaries, and observation intervals only.
  - Episode auditing and attribution are retained exclusively in dataset-level summary statistics.
- **Multiple Failures in Horizon:**
  - If multiple distinct failure episodes fall within $(t, t+24\text{h}]$, the target remains binary $y = 1$.
- **Window Overlap Semantics:**
  - Adjacent hourly positive windows overlap heavily (sharing up to 47 of 48 observation hours).
  - A single failure episode typically produces up to 24 consecutive positive hourly windows.
  - **Positive windows are NOT independent failure episodes.** Evaluation must distinguish window-level metrics from episode-level detection rates.

### B. Observation Window & Feature Scope
- **Past-Only Window:** Exactly 48 hourly readings in $[t-48\text{h}, t)$ (i.e. $t-48\text{h}, \dots, t-1\text{h}$).
- **Zero Lookahead:** No reading at $t$ or later is included in predictive features.
- **Primary Features:** Telemetry-only (`volt`, `rotate`, `pressure`, `vibration`), shape $48 \times 4$.
- **Event Join Prohibition:** Maintenance (`PdM_maint.csv`) and error (`PdM_errors.csv`) records are **never** joined into primary predictors at or before $t$.

### C. Half-Open Temporal Splits
- Half-open intervals $[T_{\text{start}}, T_{\text{end}})$:
  - `train`: `[2015-01-01 06:00:00, 2015-09-01 06:00:00)` (243 days, 5,832 hours)
  - `val`: `[2015-09-01 06:00:00, 2015-11-01 06:00:00)` (61 days, 1,464 hours)
  - `test`: `[2015-11-01 06:00:00, 2016-01-01 06:00:00)` (61 days, 1,464 hours)
- **Membership Boundary Condition:** A sample belongs to split $[T_{\text{start}}, T_{\text{end}})$ iff:
  $$t - 48\text{h} \ge T_{\text{start}} \quad \text{and} \quad t < T_{\text{end}} \quad \text{and} \quad t + 24\text{h} < T_{\text{end}} \text{ (strict)}$$
- Both observation history and label horizon are strictly contained within the split; no crossing interval boundaries. Right-censored times (including $t \ge T_{\text{end}} - 24\text{h}$) are excluded.

### D. Client Partitioning & Held-Out Factory Isolation
- **Participating Federated Clients:**
  - `client_model2` (17 machines, model2): participating in `train`, `val`, `test`.
  - `client_model3` (35 machines, model3): participating in `train`, `val`, `test`.
  - `client_model4` (32 machines, model4): participating in `train`, `val`, `test`.
- **Strict Held-Out Factory (model1):**
  - All 16 machines with `model1` are assigned to `heldout_model1`.
  - **No train or val participation:** held-out data contributes zero training updates, validation checks, hyperparameter tuning, feature selection, or adaptation.
  - Querying `heldout_model1` for `train` or `val` raises `HoldoutLeakageError`.
  - Evaluated exclusively on its final `test` split.
  - Unexpected machine models not defined in configuration are rejected with `ValueError`.
  - *Domain shift note:* The four machine models simulate non-IID client heterogeneity, not separate physical factories.

### E. Deterministic Per-Window Normalization
- Normalization is computed per-window and per-sensor using the median and Interquartile Range ($\text{IQR} = Q_{75} - Q_{25}$) of the 48 past observations only.
- **Quartile Interpolation:** Deterministic linear interpolation ($R-7$ / NumPy default).
- **Exact Zero IQR Fallback:** If $\text{IQR} == 0.0$ (constant sensor series), scale defaults to 1.0.
- **Summary Retention:** Raw summaries (`median`, `iqr`, `scale`, `q25`, `q75`) are preserved in every sample so absolute sensor levels remain accessible.
- **Strict Nonfinite Rejection:** Nonfinite raw, quantile, or normalized values are rejected deterministically.
- **No Population Scalers:** Zero global cross-client or population fitting.

---

## 2. Unapproved Downstream Proposals (Phases 3–8)

The following remain **proposals and candidates**, pending formal review and implementation in future phases:

### A. Predictive Backbone (Phase 3 Candidate)
- Proposed: GRU sequence encoder with classification head for Azure, regression head for NASA.
- Baseline comparisons: Local-only vs. Centralized reference vs. FedAvg vs. FedProx.
- Status: **Candidate — not benchmarked.**

### B. Privacy Mechanism (Phase 5 Proposal)
- Techniques: Differential Privacy (DP-SGD) or Secure Aggregation.
- Status: **Open — threat model required before claiming privacy.**

### C. Maintenance Knowledge Graph (Phase 6 Candidate)
- Candidate: Local graph retrieval of machine event facts for LLM grounding.
- Status: **Candidate — ontology and schemas pending.**

### D. LLM Grounded Explanations (Phase 7 Proposal)
- Candidate: Grounded explanation generation evaluated with citation accuracy vs ungrounded LLM.
- Status: **Required, hosting and rubric open.**

### E. NASA C-MAPSS Track (Separate Benchmark)
- RUL regression benchmark across operating cycles.
- Status: **Separate experimental track — unfinalized candidate.**
