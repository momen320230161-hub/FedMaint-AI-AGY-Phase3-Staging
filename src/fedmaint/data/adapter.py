"""Model-ready data adapter for Phase 3 Azure PdM.

This module adapts Phase 2B AzureSample streams into model-ready batches:
- Sequences: [N, 48, 4] float32 past-only normalized sensor readings in canonical order
  (volt, rotate, pressure, vibration).
- Summaries: [N, 8] float32 deterministic signed_log1p of historical median and IQR per sensor.
- Tabular: [N, 20] float32 historical features per sensor: median, IQR, last RAW reading,
  last-minus-first RAW reading, and population standard deviation over 48 RAW readings.
- Labels: [N] binary classification targets in (0, 1).
- Metadata: strictly separated SampleMetadata instances preserving tracking identifiers
  without predictive leakage.
- Strict float32 finiteness and overflow validation.
- Memory-bounded streaming batch generation without unbounded AzureSample materialization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

from .normalization import NonfiniteValueError
from .pipeline import AzureSample

DEFAULT_SENSORS: Tuple[str, ...] = ("volt", "rotate", "pressure", "vibration")
FLOAT32_MAX: float = 3.4028234663852886e+38
FLOAT32_MIN: float = -3.4028234663852886e+38

TABULAR_FEATURE_NAMES: Tuple[str, ...] = (
    "volt_median",
    "volt_iqr",
    "volt_last_raw",
    "volt_last_minus_first_raw",
    "volt_std_raw",
    "rotate_median",
    "rotate_iqr",
    "rotate_last_raw",
    "rotate_last_minus_first_raw",
    "rotate_std_raw",
    "pressure_median",
    "pressure_iqr",
    "pressure_last_raw",
    "pressure_last_minus_first_raw",
    "pressure_std_raw",
    "vibration_median",
    "vibration_iqr",
    "vibration_last_raw",
    "vibration_last_minus_first_raw",
    "vibration_std_raw",
)

SUMMARY_FEATURE_NAMES: Tuple[str, ...] = (
    "volt_median_signed_log1p",
    "volt_iqr_signed_log1p",
    "rotate_median_signed_log1p",
    "rotate_iqr_signed_log1p",
    "pressure_median_signed_log1p",
    "pressure_iqr_signed_log1p",
    "vibration_median_signed_log1p",
    "vibration_iqr_signed_log1p",
)


def signed_log1p(x: float) -> float:
    """Compute deterministic signed log1p: sign(x) * ln(1 + |x|).

    Validates that the input and transformed value are finite and fit within
    IEEE 754 single precision float32 bounds.

    Args:
        x: Float value to transform.

    Returns:
        Transformed float.

    Raises:
        NonfiniteValueError: If x is NaN, Inf, or overflows float32.
    """
    try:
        fx = float(x)
    except (TypeError, ValueError) as exc:
        raise NonfiniteValueError(f"Cannot convert value to float: {x}") from exc

    if not math.isfinite(fx):
        raise NonfiniteValueError(f"Nonfinite value in signed_log1p: {fx}")

    if fx > FLOAT32_MAX or fx < FLOAT32_MIN:
        raise NonfiniteValueError(f"Value overflows float32 range before signed_log1p: {fx}")

    if fx == 0.0:
        return 0.0

    try:
        val = math.copysign(math.log1p(abs(fx)), fx)
    except OverflowError as exc:
        raise NonfiniteValueError(f"signed_log1p overflowed: {exc}") from exc

    if not math.isfinite(val) or val > FLOAT32_MAX or val < FLOAT32_MIN:
        raise NonfiniteValueError(f"signed_log1p produced nonfinite/overflow value: {val}")

    return val


def reconstruct_raw_series(
    features: Sequence[Sequence[float]],
    raw_summaries: Dict[str, Dict[str, float]],
    sensors: Sequence[str] = DEFAULT_SENSORS,
) -> Dict[str, List[float]]:
    """Reconstruct 48 raw sensor readings from normalized features and summaries.

    Uses Phase 2B scale and median: raw = normalized * scale + median.

    Args:
        features: 48 timesteps of normalized readings in canonical sensor order.
        raw_summaries: Dict mapping sensor name to summary stats containing 'median' and 'scale'.
        sensors: Ordered tuple of sensor names.

    Returns:
        Dict mapping sensor name to list of 48 reconstructed raw readings.

    Raises:
        NonfiniteValueError: If reconstructed raw reading is nonfinite or overflows float32.
    """
    if len(features) != 48:
        raise ValueError(f"Expected 48 timesteps of features; got {len(features)}")

    reconstructed: Dict[str, List[float]] = {}
    for col_idx, s in enumerate(sensors):
        if s not in raw_summaries:
            raise ValueError(f"Missing summary for sensor '{s}' in raw_summaries")

        s_stats = raw_summaries[s]
        median = float(s_stats["median"])
        # Phase 2B retains scale; fallback to IQR or 1.0 if scale not explicitly keyed
        scale = float(s_stats.get("scale", s_stats["iqr"] if s_stats.get("iqr", 0.0) > 0.0 else 1.0))

        if not math.isfinite(median) or not math.isfinite(scale) or scale <= 0.0:
            raise NonfiniteValueError(f"Invalid summary stats for sensor '{s}': median={median}, scale={scale}")

        series: List[float] = []
        for step_idx in range(48):
            norm_val = float(features[step_idx][col_idx])
            if not math.isfinite(norm_val):
                raise NonfiniteValueError(f"Nonfinite normalized value for '{s}' at step {step_idx}: {norm_val}")

            try:
                raw_val = norm_val * scale + median
            except OverflowError as exc:
                raise NonfiniteValueError(
                    f"Raw reconstruction overflowed for sensor '{s}' at step {step_idx}: {exc}"
                ) from exc

            if not math.isfinite(raw_val) or raw_val > FLOAT32_MAX or raw_val < FLOAT32_MIN:
                raise NonfiniteValueError(
                    f"Reconstructed raw value for sensor '{s}' at step {step_idx} overflows float32: {raw_val}"
                )
            series.append(raw_val)

        reconstructed[s] = series

    return reconstructed


def extract_tabular_features(
    raw_series: Dict[str, List[float]],
    raw_summaries: Dict[str, Dict[str, float]],
    sensors: Sequence[str] = DEFAULT_SENSORS,
) -> List[float]:
    """Extract 20 fixed historical tabular features from reconstructed raw series.

    Order for each sensor (5 features x 4 sensors = 20 features):
      1. median
      2. IQR
      3. last RAW reading (timestep 47)
      4. last-minus-first RAW reading (timestep 47 - timestep 0)
      5. population standard deviation over 48 RAW readings

    Args:
        raw_series: Reconstructed raw readings for each sensor (48 steps).
        raw_summaries: Retained Phase 2B sensor summaries.
        sensors: Ordered tuple of sensor names.

    Returns:
        List of 20 float values.
    """
    tabular: List[float] = []
    for s in sensors:
        s_stats = raw_summaries[s]
        median = float(s_stats["median"])
        iqr = float(s_stats["iqr"])
        readings = raw_series[s]

        last_raw = readings[47]
        first_raw = readings[0]
        last_minus_first = last_raw - first_raw

        # Population standard deviation over 48 readings (denominator N = 48)
        mean_raw = sum(readings) / 48.0
        var_raw = sum((x - mean_raw) ** 2 for x in readings) / 48.0
        std_raw = math.sqrt(max(0.0, var_raw))

        for feat_name, feat_val in (
            ("median", median),
            ("iqr", iqr),
            ("last_raw", last_raw),
            ("last_minus_first_raw", last_minus_first),
            ("std_raw", std_raw),
        ):
            if not math.isfinite(feat_val) or feat_val > FLOAT32_MAX or feat_val < FLOAT32_MIN:
                raise NonfiniteValueError(
                    f"Tabular feature '{s}_{feat_name}' overflows float32 or is nonfinite: {feat_val}"
                )
            tabular.append(feat_val)

    if len(tabular) != 20:
        raise ValueError(f"Expected 20 tabular features; got {len(tabular)}")
    return tabular


def extract_summary_features(
    raw_summaries: Dict[str, Dict[str, float]],
    sensors: Sequence[str] = DEFAULT_SENSORS,
) -> List[float]:
    """Extract 8 historical summary features after signed_log1p.

    Order: for each sensor in canonical order:
      1. signed_log1p(median)
      2. signed_log1p(iqr)

    Args:
        raw_summaries: Retained Phase 2B sensor summaries.
        sensors: Canonical sensor names.

    Returns:
        List of 8 float values.
    """
    summaries: List[float] = []
    for s in sensors:
        s_stats = raw_summaries[s]
        median = float(s_stats["median"])
        iqr = float(s_stats["iqr"])

        trans_med = signed_log1p(median)
        trans_iqr = signed_log1p(iqr)

        summaries.append(trans_med)
        summaries.append(trans_iqr)

    if len(summaries) != 8:
        raise ValueError(f"Expected 8 summary features; got {len(summaries)}")
    return summaries


@dataclass(frozen=True)
class SampleMetadata:
    """Tracking metadata strictly isolated from predictive features.

    Guarantees that machine IDs, client names, split boundaries, and timestamps
    are never leaked into neural or linear model inputs.
    """
    machine_id: int
    prediction_time: datetime
    client_name: str
    split_name: str
    observation_start: datetime
    observation_end: datetime
    horizon_start: datetime
    horizon_end: datetime


@dataclass
class ModelReadyBatch:
    """A bounded batch of model-ready arrays and tracking metadata."""
    sequences: Any  # shape: [N, 48, 4], float32
    summaries: Any  # shape: [N, 8], float32
    labels: Any     # shape: [N], binary integer
    tabular: Any    # shape: [N, 20], float32
    metadata: Tuple[SampleMetadata, ...]

    def __len__(self) -> int:
        return len(self.metadata)


def adapt_sample(
    sample: AzureSample,
    sensors: Sequence[str] = DEFAULT_SENSORS,
) -> Tuple[List[List[float]], List[float], int, List[float], SampleMetadata]:
    """Adapt a single AzureSample into verified model-ready components.

    Validates:
      - 48 timesteps and 4 sensors matching canonical order.
      - Binary integer label in {0, 1}.
      - IEEE float32 finiteness and overflow boundaries.
      - Complete separation of metadata from predictors.

    Returns:
        (sequence_48x4, summaries_8, binary_label, tabular_20, metadata)
    """
    # 1. Validate sensor ordering and shape
    if tuple(sample.sensor_names) != tuple(sensors):
        raise ValueError(
            f"Sensor mismatch or ordering error: sample has {sample.sensor_names}, expected {sensors}"
        )
    if len(sample.features) != 48:
        raise ValueError(f"Expected 48 timesteps in sample features; got {len(sample.features)}")
    for t_idx, step_row in enumerate(sample.features):
        if len(step_row) != len(sensors):
            raise ValueError(f"Step {t_idx} has {len(step_row)} sensors; expected {len(sensors)}")
        for col_idx, v in enumerate(step_row):
            fv = float(v)
            if not math.isfinite(fv) or fv > FLOAT32_MAX or fv < FLOAT32_MIN:
                raise NonfiniteValueError(
                    f"Feature value at step {t_idx}, col {col_idx} overflows float32 or is nonfinite: {fv}"
                )

    # 2. Validate binary label
    if sample.label not in (0, 1) or isinstance(sample.label, bool) or not isinstance(sample.label, int):
        raise ValueError(f"Invalid label {sample.label}; expected binary integer 0 or 1")

    # 3. Reconstruct raw sensor readings
    raw_series = reconstruct_raw_series(sample.features, sample.raw_summaries, sensors=sensors)

    # 4. Extract tabular features [20]
    tabular = extract_tabular_features(raw_series, sample.raw_summaries, sensors=sensors)

    # 5. Extract summary features [8]
    summaries = extract_summary_features(sample.raw_summaries, sensors=sensors)

    # 6. Extract isolated tracking metadata
    meta = SampleMetadata(
        machine_id=int(sample.machine_id),
        prediction_time=sample.prediction_time,
        client_name=str(sample.client_name),
        split_name=str(sample.split_name),
        observation_start=sample.observation_start,
        observation_end=sample.observation_end,
        horizon_start=sample.horizon_start,
        horizon_end=sample.horizon_end,
    )

    return sample.features, summaries, sample.label, tabular, meta


def iter_batches(
    samples: Iterable[AzureSample],
    batch_size: int = 32,
    *,
    drop_last: bool = False,
    sensors: Sequence[str] = DEFAULT_SENSORS,
) -> Iterator[ModelReadyBatch]:
    """Yield bounded ModelReadyBatch instances without materializing full sample stream.

    Args:
        samples: Iterable yielding AzureSample instances.
        batch_size: Maximum number of samples per batch (must be >= 1).
        drop_last: If True, omit the final batch if it has fewer than batch_size samples.
        sensors: Canonical sensor names and ordering.

    Yields:
        ModelReadyBatch instances.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")

    batch_seq: List[List[List[float]]] = []
    batch_sum: List[List[float]] = []
    batch_lbl: List[int] = []
    batch_tab: List[List[float]] = []
    batch_meta: List[SampleMetadata] = []

    def _build_batch() -> ModelReadyBatch:
        if _HAS_NUMPY:
            seq_arr = np.array(batch_seq, dtype=np.float32)
            sum_arr = np.array(batch_sum, dtype=np.float32)
            lbl_arr = np.array(batch_lbl, dtype=np.int64)
            tab_arr = np.array(batch_tab, dtype=np.float32)
        else:
            seq_arr = batch_seq
            sum_arr = batch_sum
            lbl_arr = batch_lbl
            tab_arr = batch_tab

        return ModelReadyBatch(
            sequences=seq_arr,
            summaries=sum_arr,
            labels=lbl_arr,
            tabular=tab_arr,
            metadata=tuple(batch_meta),
        )

    for sample in samples:
        seq, summ, lbl, tab, meta = adapt_sample(sample, sensors=sensors)
        batch_seq.append(seq)
        batch_sum.append(summ)
        batch_lbl.append(lbl)
        batch_tab.append(tab)
        batch_meta.append(meta)

        if len(batch_meta) == batch_size:
            yield _build_batch()
            batch_seq.clear()
            batch_sum.clear()
            batch_lbl.clear()
            batch_tab.clear()
            batch_meta.clear()

    if batch_meta and not drop_last:
        yield _build_batch()


class AzureModelReadyAdapter:
    """Configurable adapter providing streaming batch adaptation for Azure PdM."""

    def __init__(
        self,
        batch_size: int = 32,
        *,
        drop_last: bool = False,
        sensors: Sequence[str] = DEFAULT_SENSORS,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sensors = tuple(sensors)

    def adapt_sample(
        self,
        sample: AzureSample,
    ) -> Tuple[List[List[float]], List[float], int, List[float], SampleMetadata]:
        return adapt_sample(sample, sensors=self.sensors)

    def iter_batches(
        self,
        samples: Iterable[AzureSample],
    ) -> Iterator[ModelReadyBatch]:
        return iter_batches(
            samples,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            sensors=self.sensors,
        )
