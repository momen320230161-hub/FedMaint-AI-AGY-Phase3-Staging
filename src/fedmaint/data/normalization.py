"""Deterministic per-window sensor normalization using median and IQR.

This module implements:
- Past-only per-window, per-sensor median and IQR (Interquartile Range).
- Deterministic linear quartile interpolation (matching NumPy default linear method).
- Exact zero IQR fallback (scale = 1.0) when IQR == 0.0 to prevent division by zero.
- Strict rejection of nonfinite raw, quantile, and normalized values (NaN, Inf, or OverflowError).
- Retention of raw-level sensor summaries (median, IQR, scale, Q25, Q75) in each sample.
- Zero cross-client or population fitting: normalizers are entirely local to each 48-hour window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

DEFAULT_SENSORS: Tuple[str, ...] = ("volt", "rotate", "pressure", "vibration")
ZERO_IQR_SCALE: float = 1.0


class NonfiniteValueError(ValueError):
    """Encountered a NaN, Inf, or overflowed value in sensor telemetry or computed features."""


@dataclass(frozen=True)
class SensorSummary:
    """Summary statistics retained for a single sensor within a 48-hour window."""
    median: float
    q25: float
    q75: float
    iqr: float
    scale: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
            "scale": self.scale,
        }


def compute_quantile(values: Sequence[float], q: float, *, method: str = "linear") -> float:
    """Compute percentile q in [0, 1] using deterministic linear interpolation.

    Matches standard linear interpolation (NumPy method='linear' / R-7):
        index = q * (N - 1)
        result = x[i] + (index - i) * (x[i+1] - x[i])

    Args:
        values: Sequence of numbers (will be sorted).
        q: Quantile in [0.0, 1.0].
        method: Interpolation method (currently 'linear').

    Returns:
        Interpolated quantile value.
    """
    if not values:
        raise ValueError("Cannot compute quantile of empty sequence.")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile q must be between 0.0 and 1.0; got {q}")
    if method != "linear":
        raise ValueError(f"Unsupported quantile method '{method}'; use 'linear'.")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        res = sorted_vals[0]
        if not math.isfinite(res):
            raise NonfiniteValueError(f"Quantile value is nonfinite: {res}")
        return res

    index = q * (n - 1)
    i = int(math.floor(index))
    fraction = index - i

    try:
        if i >= n - 1:
            res = sorted_vals[-1]
        else:
            res = sorted_vals[i] + fraction * (sorted_vals[i + 1] - sorted_vals[i])
    except OverflowError as exc:
        raise NonfiniteValueError(f"Quantile interpolation overflowed: {exc}") from exc

    if not math.isfinite(res):
        raise NonfiniteValueError(f"Quantile computation resulted in nonfinite value: {res}")
    return res


def compute_sensor_stats(
    values: Sequence[float],
    *,
    safe_scale: float = ZERO_IQR_SCALE,
) -> SensorSummary:
    """Compute deterministic median, Q25, Q75, IQR, and effective scale for a sensor series.

    Under approved design, exact zero IQR (iqr == 0.0) triggers fallback to safe_scale (default 1.0).

    Args:
        values: Sequence of sensor float readings for a 48-hour window.
        safe_scale: Fallback scale when IQR == 0.0 (default 1.0).

    Returns:
        SensorSummary with median, q25, q75, iqr, and scale.
    """
    for v in values:
        if not math.isfinite(v):
            raise NonfiniteValueError(f"Nonfinite raw sensor value encountered: {v}")

    try:
        median = compute_quantile(values, 0.5, method="linear")
        q25 = compute_quantile(values, 0.25, method="linear")
        q75 = compute_quantile(values, 0.75, method="linear")
        iqr = q75 - q25
    except OverflowError as exc:
        raise NonfiniteValueError(f"Sensor quantile or IQR computation overflowed: {exc}") from exc

    if not math.isfinite(iqr):
        raise NonfiniteValueError(f"Computed IQR is nonfinite: {iqr}")

    if iqr == 0.0:
        scale = safe_scale
    elif iqr < 0.0:
        raise ValueError(f"Invalid negative IQR: {iqr}")
    else:
        scale = iqr

    if not math.isfinite(scale) or scale <= 0.0:
        raise NonfiniteValueError(f"Computed scale is invalid or nonfinite: {scale}")

    return SensorSummary(median=median, q25=q25, q75=q75, iqr=iqr, scale=scale)


def normalize_window(
    telemetry_rows: Sequence[Dict[str, object]],
    *,
    sensors: Sequence[str] = DEFAULT_SENSORS,
    safe_scale: float = ZERO_IQR_SCALE,
) -> Tuple[List[List[float]], Dict[str, SensorSummary]]:
    """Normalize a 48-step hourly observation window per sensor using median and IQR.

    Args:
        telemetry_rows: Exactly 48 consecutive hourly telemetry records ordered chronologically.
        sensors: Sensor names in fixed canonical order.
        safe_scale: Safe scale when IQR == 0.0 (default 1.0).

    Returns:
        A tuple of:
          - normalized_matrix: list of 48 timesteps, each containing normalized float values
            in canonical sensor order: (val - median) / scale.
          - summaries: dict mapping sensor name to its SensorSummary (preserving raw levels).
    """
    if len(telemetry_rows) != 48:
        raise ValueError(f"Expected exactly 48 hourly telemetry records; got {len(telemetry_rows)}")

    # Extract per-sensor sequences and validate raw finiteness
    series_by_sensor: Dict[str, List[float]] = {s: [] for s in sensors}
    for row in telemetry_rows:
        for s in sensors:
            val = float(row[s])  # type: ignore
            if not math.isfinite(val):
                raise NonfiniteValueError(f"Nonfinite raw sensor value for '{s}': {val}")
            series_by_sensor[s].append(val)

    # Compute summary stats and scale per sensor
    summaries: Dict[str, SensorSummary] = {}
    for s in sensors:
        summaries[s] = compute_sensor_stats(series_by_sensor[s], safe_scale=safe_scale)

    # Build 48 x num_sensors normalized matrix and verify normalized finiteness
    normalized_matrix: List[List[float]] = []
    for step_idx in range(len(telemetry_rows)):
        step_features: List[float] = []
        for s in sensors:
            val = series_by_sensor[s][step_idx]
            stat = summaries[s]
            try:
                norm_val = (val - stat.median) / stat.scale
            except OverflowError as exc:
                raise NonfiniteValueError(
                    f"Normalization overflowed for sensor '{s}' at step {step_idx}: {exc}"
                ) from exc
            if not math.isfinite(norm_val):
                raise NonfiniteValueError(f"Nonfinite normalized value {norm_val} for sensor '{s}' at step {step_idx}")
            step_features.append(norm_val)
        normalized_matrix.append(step_features)

    return normalized_matrix, summaries
