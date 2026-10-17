# Experiment Configurations

This directory holds **versioned experiment settings** and pipeline configurations for FedMaint-AI.

## Frozen Phase 2B Configuration

### `azure_preprocessing.json`
The single source of truth for Phase 2B Azure PdM data preparation, time splitting, client partitioning, and feature extraction:

- **Temporal Splits (half-open `[start, end)`):**
  - `train`: `[2015-01-01 06:00:00, 2015-09-01 06:00:00)`
  - `val`: `[2015-09-01 06:00:00, 2015-11-01 06:00:00)`
  - `test`: `[2015-11-01 06:00:00, 2016-01-01 06:00:00)`
  - Split boundary rule: a sample at prediction time $t$ belongs to split $[T_{\text{start}}, T_{\text{end}})$ if and only if:
    $$t - 48\text{h} \ge T_{\text{start}} \quad \text{and} \quad t < T_{\text{end}} \quad \text{and} \quad t + 24\text{h} < T_{\text{end}} \text{ (strict)}$$
- **Windowing and Horizons:**
  - Cadence: 1 hour (`cadence_hours = 1`)
  - Past observation window: exactly 48 hourly readings in $[t-48\text{h}, t)$
  - Label horizon: $(t, t+24\text{h}]$
  - Blackout interval: $[t-48\text{h}, t]$ inclusive (if a failure episode occurs within this window, the sample is excluded)
- **Features and Normalization:**
  - Sensors: `["volt", "rotate", "pressure", "vibration"]` (fixed order, shape $48 \times 4$)
  - Primary features only (telemetry-only; no same-hour maintenance or error joins)
  - Normalization: per-window median and IQR ($Q_{75} - Q_{25}$) computed from past 48 observations only
  - Quartile interpolation: linear ($R-7$ / NumPy default)
  - Safe scale fallback: 1.0 when $\text{IQR} = 0.0$
  - Raw summaries (`median`, `iqr`, `scale`, `q25`, `q75`) are retained in every sample
- **Client Partitioning and Holdout Isolation:**
  - Participating clients: `client_model2`, `client_model3`, `client_model4` (allowed splits: `train`, `val`, `test`)
  - Held-out factory: `heldout_model1` (all 16 machines with model1)
  - Strict holdout policy: `heldout_model1` is accessible **only** for final `test` evaluation; querying it for `train` or `val` raises `HoldoutLeakageError`
  - No global cross-client scalers or population fitting
- **Labels and Episode Collapse:**
  - Binary target $y = 1$ if at least one failure episode occurs in $(t, t+24\text{h}]$, else $0$
  - Simultaneous failure records sharing `machineID` and exact timestamp are collapsed into a single episode
  - Adjacent positive hourly windows overlap heavily (up to 23 hours) and are **not** independent failure episodes

### Python Usage
```python
from fedmaint.data import load_azure_preprocessing_config

config = load_azure_preprocessing_config()
print(config.history_hours)  # 48
print(config.horizon_hours)  # 24
print(config.sensor_names)   # ('volt', 'rotate', 'pressure', 'vibration')
```

## Future Configuration Files (Phase 3+)
- `azure_baseline.yaml` — local/centralized baseline settings
- `azure_fedavg.yaml`, `azure_fedprox.yaml` — matched federated experiments
- `nasa_rul.yaml` — NASA RUL benchmarking
- `privacy_*.yaml` — mechanism and budget settings
