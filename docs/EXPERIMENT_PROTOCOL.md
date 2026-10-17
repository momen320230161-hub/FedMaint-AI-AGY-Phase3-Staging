# Experiment Protocol

> **Notice:** This document defines the **Approved Phase 2B Azure Data Preparation Protocol** and **Approved Phase 3 Azure Baseline Experiment Protocol**. Downstream components (Phases 4–8: Federated Learning, Differential Privacy, Knowledge Graph construction, and LLM explainability) remain candidate proposals until their respective phases.

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

### E. Deterministic Per-Window Normalization
- Normalization is computed per-window and per-sensor using the median and Interquartile Range ($\text{IQR} = Q_{75} - Q_{25}$) of the 48 past observations only.
- **Quartile Interpolation:** Deterministic linear interpolation ($R-7$ / NumPy default).
- **Exact Zero IQR Fallback:** If $\text{IQR} == 0.0$ (constant sensor series), scale defaults to 1.0.
- **Summary Retention:** Raw summaries (`median`, `iqr`, `scale`, `q25`, `q75`) are preserved in every sample.
- **No Population Scalers:** Zero global cross-client or population fitting.

---

## 2. Approved Phase 3 Azure Baseline Experiment Protocol

Implemented in `src/fedmaint/experiments/phase3_azure.py`, `src/fedmaint/models/`, `src/fedmaint/training/`, `src/fedmaint/evaluation/`, `scripts/run_phase3_azure.py`, and `notebooks/03_Azure_Modeling.ipynb`.

### A. Evaluated Model Architectures

1. **`AzureGRU` (Primary Sequential Backbone):**
   - Inputs: Normalized sequences $[B, 48, 4]$ and signed log1p transformed historical summaries $[B, 8]$.
   - Recurrent Layer: 1-layer PyTorch GRU, `hidden_size=32`, `dropout=0.0`.
   - Classification Head: Linear projection on concatenated 40-dim representation $[h_n[0] \,\|\, \text{summaries}] \to [B, 1]$ logit.
   - Loss: Unweighted `BCEWithLogitsLoss`.

2. **`AzureLogisticRegression` (Tabular Linear Baseline):**
   - Inputs: 20 causal tabular features per window (median, IQR, last raw reading, last-minus-first raw delta, and population standard deviation across 48 raw readings for each sensor).
   - Algorithm: Scikit-learn `LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)`.
   - Scaler: Train-fitted `TabularStandardScaler`.

3. **`AzureMLP` (Tabular Neural Baseline):**
   - Inputs: 20 causal tabular features, standardized via train-fitted `TabularStandardScaler`.
   - Architecture: Linear $[20 \to 32] \to \text{ReLU} \to$ Linear $[32 \to 1]$ logit.
   - Optimization: Same neural budget as `AzureGRU`.

### B. Training Regimes & Checkpoint Rules

- **Local Regime:**
  - Trains 3 independent models per seed (one each for `client_model2`, `client_model3`, `client_model4`).
  - Training partition: Own client's `train` split only.
  - Validation Checkpoint: Evaluated on own client's `val` split. Best epoch selected by **validation Average Precision (AP)** with strict improvement (preserving the earliest epoch on ties).
  - Validation Threshold: Selected on own client's `val` split by sweeping candidate thresholds to maximize F1 score. Ties resolved by choosing the highest threshold.
  - Test Evaluation: Evaluated on own client's `test` split, plus separate zero-shot evaluation on `heldout_model1` `test` split.

- **Centralized Regime (Non-Private Reference):**
  - Trains 1 pooled model per seed across all three participating clients (`client_model2`, `client_model3`, `client_model4`).
  - Training partition: Concatenated participating client `train` splits.
  - Validation Checkpoint: Monitored on the **unweighted macro validation AP** across participating clients:
    $$\text{Macro AP}_{\text{val}} = \frac{1}{3} \left( \text{AP}_{\text{model2, val}} + \text{AP}_{\text{model3, val}} + \text{AP}_{\text{model4, val}} \right)$$
  - Validation Threshold: Selected on pooled participating client `val` splits by maximizing F1 score (highest threshold on tie).
  - Test Evaluation: Evaluated on per-client `test` splits, pooled participating `test`, macro `test`, and separate zero-shot on `heldout_model1` `test` split.
  - *Reference Note:* The centralized regime serves as a non-private reference (never a guaranteed upper bound); it does not satisfy federated privacy constraints.

### C. Dual Evaluation Framework

Evaluation reports both window-level discrimination and operational event-level detection rates.

#### 1. Window-Level Metrics
- Computed at the frozen validation threshold on test partitions:
  - **Average Precision (AP / PR-AUC):** Primary metric, computed via step interpolation matching scikit-learn.
  - **ROC-AUC:** Area under the ROC curve.
  - **F1 Score, Precision, Recall:** Threshold-dependent classification scores.
  - **Confusion Matrix:** Counts of true positives ($TP$), false positives ($FP$), false negatives ($FN$), and true negatives ($TN$).
- *Absence Handling:* If a test partition contains only one class, AP and ROC-AUC return `None` gracefully.

#### 2. Event-Level Metrics (Operational Failure Detection)
Implemented in `src/fedmaint/evaluation/events.py`.
- **24-Hour Alert Cooldown:** When an alert is emitted for machine $m$ at time $t$ ($\hat{p} \ge \tau$), subsequent positive predictions for machine $m$ are suppressed for 24 hours ($[t, t+24\text{h}]$). This represents standard operational maintenance dispatch and prevents alert fatigue.
- **Chronological Episode Matching:**
  - Each emitted alert is matched to at most one earliest unmatched future collapsed failure episode occurring strictly within $(t, t+24\text{h}]$.
  - Each failure episode matches at most one alert.
- **Reported Event Metrics:**
  - **Total Episodes:** All failure episodes within the split interval for cohort machines.
  - **Eligible Episodes:** Episodes observable with valid preceding prediction opportunities ($t - 48\text{h} \ge T_{\text{start}}$ and $t + 24\text{h} < T_{\text{end}}$).
  - **Ineligible / Unobservable Episodes:** Episodes occurring too close to split boundaries or inside blackouts.
  - **Detected Episodes:** Eligible episodes successfully matched by an alert.
  - **Event Recall:** $\frac{\text{detected\_episodes}}{\text{eligible\_episodes}}$.
  - **True Alerts ($TA$) & False Alerts ($FA$):** Alerts matching vs not matching an episode.
  - **Eligible Machine-Days:** $\frac{\text{distinct valid prediction opportunities}}{24}$.
  - **False Alert Rate (FAR):** False alerts per 100 eligible machine-days:
    $$\text{FAR} = \frac{\text{false\_alerts}}{\text{eligible\_machine\_days} / 100}$$
  - **Lead Hours:** Distribution of $(\text{episode\_timestamp} - \text{alert\_time})$ and mean lead hours.

### D. Machine-Cluster Bootstrap Uncertainty Estimation
Implemented in `src/fedmaint/evaluation/bootstrap.py`.
- **Method:** Resamples entire machines (clusters) with replacement using an explicit random seed.
- **Optimized Algorithm:** Pre-sorts and vectorizes once across window samples, weighting clusters by resampled machine multiplicities to eliminate redundant window duplications, with exact tie handling and pure-Python fallback when NumPy is unavailable.
- **Multiplicity Preservation:** All windows and failure events of a resampled machine appear with multiplicity equal to the number of times that machine is drawn.
- **Intervals Reported:** 95% percentile confidence intervals ($[Q_{2.5}, Q_{97.5}]$) for AP, ROC-AUC, F1, precision, recall, event recall, FAR per 100 machine-days, and mean lead hours.
- **Draw Counts:**
  - Full mode: 1000 bootstrap draws.
  - Smoke mode: 30 bootstrap draws (labeled plumbing only).
- Returns `None` for CI bounds if valid draws fall below `min_valid_draws = 10`.

### E. Experiment Execution Modes

| Attribute | Smoke Mode (`--mode smoke`) | Full Mode (`--mode full`) |
| --- | --- | --- |
| **Seeds** | `[17]` | `[17, 23, 41]` |
| **Max Epochs** | 2 | 25 |
| **Early Stopping Patience** | 2 | 5 |
| **Bootstrap Draws** | 30 | 1000 |
| **Sample Bounding** | Stratified bounded subset ($\le 60$ samples/split) | Full dataset partitions |
| **Cache Rebuild** | Builds/reuses full memmap cache, then takes bounded in-memory subset | Full memmapped dataset cache |
| **Representative?** | **No (plumbing verification only)** | **Yes (scientific baseline evaluation)** |

### F. Reproducibility Commands

#### 1. Command-Line Interface (CLI)
Run from the repository root:
```bash
# Rapid plumbing verification (smoke mode, automatically uses data/Azure Data if present):
python scripts/run_phase3_azure.py --mode smoke

# Full multi-seed scientific baseline run:
python scripts/run_phase3_azure.py --mode full

# Targeted run for specific models, regimes, and device:
python scripts/run_phase3_azure.py --mode smoke --models gru --regimes local central --device cpu

# Parameterized data directory and output directory:
python scripts/run_phase3_azure.py --mode full --data-dir "data/Azure Data" --output-dir "outputs/phase3"
```
*Note on `--data-dir`:* The CLI automatically defaults to `data/Azure Data` relative to the repository root when present. If the raw CSV directory is not found and `--data-dir` is not explicitly provided, a clear actionable error is raised.

#### 2. Interactive Jupyter Notebook
Open and execute:
```bash
jupyter notebook notebooks/03_Azure_Modeling.ipynb
```
- Defaults safely to `smoke` mode.
- Safely skips execution without errors when raw CSVs are absent in data-free environments.
- Resolves data directory automatically to `data/Azure Data` when present.
- Full mode requires an explicit parameter change (`CONFIRM_FULL_RUN = True`).

### G. Expected Outputs & Provenance Tracking

All runs write structured outputs to `--output-dir` (default: `outputs/phase3`):

1. **`results.json`:** Comprehensive JSON serialization of `Phase3ExperimentReport`:
   - `pipeline_name`, `phase`, `mode`, `is_representative`, `is_plumbing_only`
   - `config_hash` (deterministic SHA-256 of `configs/phase3_azure.json`)
   - `git_revision` (HEAD commit SHA)
   - `created_at` (UTC ISO timestamp)
   - `results`: List of `SingleRunResult` records including all window metrics, event metrics, bootstrap CIs, per-seed training telemetry (optimizer steps, samples processed, best epoch), model fitting runtime (`fit_elapsed_seconds`), per-run duration (`total_elapsed_seconds`, measuring model fit plus validation/test/bootstrap only), and zero-shot `heldout_model1` evaluation.
   - `summary_table`: Formatted tabular rows including both end-to-end `runtime_s` and `fit_time_s`.
2. **`summary.txt`:** Plain-text formatted table comparing models, regimes, clients, seeds, validation F1, test AP/F1/recall/FAR, held-out AP/F1/recall, end-to-end `Run(s)`, and model `Fit(s)`.
3. **Memmapped Caches (`<output_dir>/cache/`):**
   - Zero-copy memmaps for `sequences.npy` $[N, 48, 4]$, `summaries.npy` $[N, 8]$, `tabular.npy` $[N, 20]$, `labels.npy` $[N]$, `metadata_machine_ids.npy` $[N]$, `metadata_timestamps.npy` $[N]$.
   - Manifests with SHA-256 hashes and atomic `.completed.marker` files.

### H. Telemetry & Compute Matching
- Measures true peak memory and process RSS via `psutil` or platform APIs.
- Accurately tracks per-run runtime: run duration (`total_elapsed_seconds`, labeled `Run(s)`) measuring model fit plus validation/test/bootstrap only (since `run_start` begins after `load_phase3_datasets`, excluding raw data loading/cache build), alongside dedicated model fit duration (`fit_elapsed_seconds`, labeled `Fit(s)`).
- Logs cumulative optimizer steps and samples presented per client and regime.
- These compute figures form the baseline against which Phase 4 Federated Learning communication and computation trade-offs will be evaluated.

---

## 3. Unapproved Downstream Proposals (Phases 4–8)

The following remain **proposals and candidates**, pending formal review and implementation in future phases:

### A. Federated Learning (Phase 4 Proposal)
- Proposed: FedAvg (sample-count weighted aggregation) and FedProx (local proximal term $\frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$).
- Communication tracking: Bytes exchanged, round counts, and per-client convergence curves.
- Status: **Candidate — to be implemented in Phase 4.**

### B. Privacy Mechanism (Phase 5 Proposal)
- Proposed: Differential Privacy (DP-SGD) or Secure Aggregation.
- Status: **Open — formal threat model required before claiming privacy guarantees.**

### C. Maintenance Knowledge Graph (Phase 6 Candidate)
- Candidate: Local graph retrieval of machine event facts for LLM grounding.
- Status: **Candidate — ontology and schemas pending.**

### D. LLM Grounded Explanations (Phase 7 Proposal)
- Candidate: Grounded explanation generation evaluated with citation accuracy vs ungrounded LLM.
- Status: **Required — hosting and evaluation rubric open.**

### E. NASA C-MAPSS Track (Separate Benchmark)
- RUL regression benchmark across operating cycles.
- Status: **Separate experimental track — unfinalized candidate.**
