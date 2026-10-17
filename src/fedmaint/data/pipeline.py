"""End-to-end leakage-safe data preparation pipeline for Phase 2B Azure PdM.

This module coordinates:
- Chronological hourly candidate window generation across half-open splits.
- Rejection of missing, irregular, duplicate, nonfinite, right-censored, or blacked-out windows.
- Past-only per-window median and IQR normalization (exact zero fallback).
- Client partitioning and strict held-out client isolation.
- Full telemetry is indexed in memory for fast evaluation; actual peak memory is unmeasured.
- Provably unified deterministic validity decision between summary and sample generation.
- Detailed, inspectable summary statistics and rejection tracking without feature materialization during summary.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from .azure import DEFAULT_AZURE_DIR, iter_azure_rows, load_machines
from .config import Phase2BConfig, load_azure_preprocessing_config
from .episodes import (
    FailureEpisode,
    collapse_failure_episodes,
    is_in_blackout,
    label_window,
)
from .normalization import (
    DEFAULT_SENSORS,
    NonfiniteValueError,
    SensorSummary,
    normalize_window,
)
from .splits import (
    ClientPartition,
    HoldoutLeakageError,
    SplitBounds,
    get_client_partitions,
    get_default_splits,
)

DataDir = Union[str, Path]
Record = Dict[str, object]


class PipelineRejectionReason:
    """Standardized rejection reason strings."""
    OUTSIDE_SPLIT = "outside_split"
    BLACKOUT = "blackout"
    MISSING_TELEMETRY = "missing_telemetry"
    IRREGULAR_INTERVAL = "irregular_interval"
    NONFINITE_VALUES = "nonfinite_values"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    RIGHT_CENSORED = "right_censored"


@dataclass(frozen=True)
class AzureSample:
    """A single leakage-safe prediction window sample.

    Contains feature matrix, binary target label, raw summaries, and observation intervals.
    Future event records (failure types, timestamps) are strictly omitted from training samples.
    """
    machine_id: int
    prediction_time: datetime
    client_name: str
    split_name: str
    label: int
    features: List[List[float]]  # Shape: (48 timesteps, num_sensors)
    sensor_names: Tuple[str, ...]
    raw_summaries: Dict[str, Dict[str, float]]
    observation_start: datetime  # t - 48h
    observation_end: datetime    # t - 1h
    horizon_start: datetime      # t
    horizon_end: datetime        # t + 24h


@dataclass
class _MachineTelemetryIndex:
    """Indexed telemetry data for one machine providing interval validation."""
    records: List[Record]
    dt_first_idx: Dict[datetime, int]
    dt_last_idx: Dict[datetime, int]
    prefix_duplicates: List[int]
    prefix_nonfinites: List[int]
    max_dt: Optional[datetime]


@dataclass
class PipelineData:
    """In-memory indexed dataset container allowing reuse across pipeline operations."""
    machines: Dict[int, Record]
    episodes: Dict[int, List[FailureEpisode]]
    indices: Dict[int, _MachineTelemetryIndex]


def _build_machine_index(
    raw_records: Sequence[Record],
    sensors: Sequence[str],
) -> _MachineTelemetryIndex:
    """Build index arrays for a machine's telemetry."""
    if not raw_records:
        return _MachineTelemetryIndex([], {}, {}, [0], [0], None)

    # Sort records chronologically
    sorted_records = sorted(raw_records, key=lambda r: r["datetime"])  # type: ignore
    dt_first_idx: Dict[datetime, int] = {}
    dt_last_idx: Dict[datetime, int] = {}
    prefix_dup = [0]
    prefix_nonfinite = [0]
    seen_dts: Set[datetime] = set()

    for idx, r in enumerate(sorted_records):
        ts: datetime = r["datetime"]  # type: ignore
        is_dup = 1 if ts in seen_dts else 0
        seen_dts.add(ts)

        if ts not in dt_first_idx:
            dt_first_idx[ts] = idx
        dt_last_idx[ts] = idx

        # Check nonfinite sensors
        is_nonfinite = 0
        for s in sensors:
            v = r.get(s)
            if v is None:
                is_nonfinite = 1
                break
            try:
                fv = float(v)  # type: ignore
                if not math.isfinite(fv):
                    is_nonfinite = 1
                    break
            except (ValueError, TypeError):
                is_nonfinite = 1
                break

        prefix_dup.append(prefix_dup[-1] + is_dup)
        prefix_nonfinite.append(prefix_nonfinite[-1] + is_nonfinite)

    max_dt = sorted_records[-1]["datetime"] if sorted_records else None
    return _MachineTelemetryIndex(
        records=sorted_records,
        dt_first_idx=dt_first_idx,
        dt_last_idx=dt_last_idx,
        prefix_duplicates=prefix_dup,
        prefix_nonfinites=prefix_nonfinite,
        max_dt=max_dt,
    )


def load_pipeline_data(
    data_dir: Optional[DataDir] = None,
    config: Optional[Phase2BConfig] = None,
    *,
    telemetry_rows: Optional[Iterable[Record]] = None,
    failure_rows: Optional[Iterable[Record]] = None,
    machine_rows: Optional[Iterable[Record]] = None,
) -> PipelineData:
    """Load and index pipeline data into memory for fast, repeated evaluation.

    Memory behavior: full telemetry is indexed in memory; actual peak memory is unmeasured.

    Args:
        data_dir: Directory containing Azure CSV files.
        config: Phase2BConfig instance (if None, loaded from configs/azure_preprocessing.json).
        telemetry_rows: Optional in-memory telemetry records.
        failure_rows: Optional in-memory failure records.
        machine_rows: Optional in-memory machine records.

    Returns:
        PipelineData container.
    """
    cfg = config if config is not None else load_azure_preprocessing_config()
    sensors = cfg.sensor_names

    if machine_rows is not None:
        machines = {int(r["machineID"]): r for r in machine_rows}
    elif data_dir is not None:
        machines = load_machines(data_dir)
    else:
        raise ValueError("Must provide either data_dir or in-memory machine_rows.")

    if failure_rows is not None:
        episodes = collapse_failure_episodes(failure_rows)
    elif data_dir is not None:
        episodes = collapse_failure_episodes(iter_azure_rows("failures", data_dir))
    else:
        episodes = {}

    grouped_telemetry: Dict[int, List[Record]] = defaultdict(list)
    source_telemetry = (
        telemetry_rows if telemetry_rows is not None else iter_azure_rows("telemetry", data_dir)  # type: ignore
    )
    for row in source_telemetry:
        m_id = int(row["machineID"])
        grouped_telemetry[m_id].append(row)

    indices: Dict[int, _MachineTelemetryIndex] = {}
    for m_id, rows in grouped_telemetry.items():
        indices[m_id] = _build_machine_index(rows, sensors)

    return PipelineData(machines=machines, episodes=episodes, indices=indices)


def _check_candidate_window(
    m_index: _MachineTelemetryIndex,
    m_episodes: List[FailureEpisode],
    t: datetime,
    history_hours: int,
    horizon_hours: int,
    sensors: Sequence[str],
    safe_scale: float,
) -> Tuple[bool, Optional[str], Optional[List[List[float]]], Optional[Dict[str, SensorSummary]]]:
    """Validate candidate time t and compute normalized features in one provably consistent step.

    Returns:
        (is_accepted, rejection_reason, normalized_features, summaries)
    """
    # 1. Right-censored check: telemetry must cover through t + horizon_hours
    if m_index.max_dt is None or m_index.max_dt < t + timedelta(hours=horizon_hours):
        return False, PipelineRejectionReason.RIGHT_CENSORED, None, None

    # 2. Blackout check: [t - history_hours, t] inclusive
    if is_in_blackout(m_episodes, t, history_hours=history_hours):
        return False, PipelineRejectionReason.BLACKOUT, None, None

    # 3. Telemetry history check: [t - history_hours, t)
    t_start = t - timedelta(hours=history_hours)
    t_end = t - timedelta(hours=1)

    if t_start not in m_index.dt_first_idx or t_end not in m_index.dt_last_idx:
        return False, PipelineRejectionReason.MISSING_TELEMETRY, None, None

    idx_start = m_index.dt_first_idx[t_start]
    idx_end = m_index.dt_last_idx[t_end]

    # Verify duplicate presence across entire interval, including both endpoints
    dup_count = m_index.prefix_duplicates[idx_end + 1] - m_index.prefix_duplicates[idx_start]
    if dup_count > 0:
        return False, PipelineRejectionReason.DUPLICATE_TIMESTAMP, None, None

    # Verify raw nonfinite presence in interval
    nonfinite_count = m_index.prefix_nonfinites[idx_end + 1] - m_index.prefix_nonfinites[idx_start]
    if nonfinite_count > 0:
        return False, PipelineRejectionReason.NONFINITE_VALUES, None, None

    # Verify exact step count and hourly spacing
    if (idx_end - idx_start) != (history_hours - 1):
        return False, PipelineRejectionReason.MISSING_TELEMETRY, None, None

    actual_start_dt = m_index.records[idx_start]["datetime"]
    actual_end_dt = m_index.records[idx_end]["datetime"]
    if actual_end_dt - actual_start_dt != timedelta(hours=history_hours - 1):
        return False, PipelineRejectionReason.IRREGULAR_INTERVAL, None, None

    # 4. Computed nonfinite check: normalize the 48 rows directly.
    # This guarantees generator and summary share the identical validity decision across all inputs.
    history_rows = m_index.records[idx_start : idx_end + 1]
    try:
        normalized_features, summaries = normalize_window(
            history_rows, sensors=sensors, safe_scale=safe_scale
        )
    except NonfiniteValueError:
        return False, PipelineRejectionReason.NONFINITE_VALUES, None, None

    return True, None, normalized_features, summaries


def generate_azure_windows(
    data_dir: Optional[DataDir] = None,
    client_name: str = "client_model2",
    split_name: str = "train",
    config: Optional[Phase2BConfig] = None,
    *,
    pipeline_data: Optional[PipelineData] = None,
    telemetry_rows: Optional[Iterable[Record]] = None,
    failure_rows: Optional[Iterable[Record]] = None,
    machine_rows: Optional[Iterable[Record]] = None,
) -> Iterator[AzureSample]:
    """Generate leakage-safe, normalized observation windows for a specific client and split.

    Args:
        data_dir: Path to directory containing Azure CSVs (optional if pipeline_data passed).
        client_name: Name of client partition (e.g. 'client_model2', 'heldout_model1').
        split_name: Name of split ('train', 'val', or 'test').
        config: Phase2BConfig instance (if None, loaded from configs/azure_preprocessing.json).
        pipeline_data: Preloaded PipelineData container (optional).
        telemetry_rows: Optional in-memory telemetry records.
        failure_rows: Optional in-memory failure records.
        machine_rows: Optional in-memory machine records.

    Yields:
        AzureSample instances in deterministic order (by machineID, then prediction time t).
    """
    cfg = config if config is not None else load_azure_preprocessing_config()
    data = pipeline_data or load_pipeline_data(
        data_dir=data_dir,
        config=cfg,
        telemetry_rows=telemetry_rows,
        failure_rows=failure_rows,
        machine_rows=machine_rows,
    )

    partitions = get_client_partitions(data.machines, config=cfg)
    if client_name not in partitions:
        raise ValueError(f"Unknown client '{client_name}'; available: {tuple(partitions.keys())}")

    partition = partitions[client_name]
    # Enforce holdout leakage guardrail
    partition.validate_split_access(split_name)

    split_bounds = cfg.splits.get(split_name)
    if split_bounds is None:
        raise ValueError(f"Unknown split '{split_name}'; available: {tuple(cfg.splits.keys())}")

    sensors = cfg.sensor_names
    safe_scale = cfg.zero_iqr_fallback_scale
    history_hours = cfg.history_hours
    horizon_hours = cfg.horizon_hours
    cadence = cfg.cadence_hours

    # Candidate prediction times satisfy:
    # t - history_hours >= split_start and t < split_end and t + horizon_hours < split_end (strict)
    earliest_t = split_bounds.start + timedelta(hours=history_hours)
    latest_t = split_bounds.end - timedelta(hours=horizon_hours + 1)

    for machine_id in sorted(partition.machine_ids):
        m_episodes = data.episodes.get(machine_id, [])
        m_index = data.indices.get(machine_id)
        if m_index is None:
            continue

        current_t = earliest_t
        while current_t <= latest_t:
            is_valid, _, normalized_features, summaries = _check_candidate_window(
                m_index, m_episodes, current_t, history_hours, horizon_hours, sensors, safe_scale
            )
            if not is_valid:
                current_t += timedelta(hours=cadence)
                continue

            assert normalized_features is not None and summaries is not None
            raw_sums = {s: summaries[s].to_dict() for s in sensors}

            # Target label y in (current_t, current_t + horizon_hours]
            y = label_window(m_episodes, current_t, horizon_hours=horizon_hours)

            sample = AzureSample(
                machine_id=machine_id,
                prediction_time=current_t,
                client_name=client_name,
                split_name=split_name,
                label=y,
                features=normalized_features,
                sensor_names=tuple(sensors),
                raw_summaries=raw_sums,
                observation_start=current_t - timedelta(hours=history_hours),
                observation_end=current_t - timedelta(hours=1),
                horizon_start=current_t,
                horizon_end=current_t + timedelta(hours=horizon_hours),
            )
            yield sample
            current_t += timedelta(hours=cadence)


def summarize_azure_pipeline(
    data_dir: Optional[DataDir] = None,
    config: Optional[Phase2BConfig] = None,
    *,
    pipeline_data: Optional[PipelineData] = None,
    telemetry_rows: Optional[Iterable[Record]] = None,
    failure_rows: Optional[Iterable[Record]] = None,
    machine_rows: Optional[Iterable[Record]] = None,
) -> Dict[str, Any]:
    """Compute detailed summary statistics and rejection tracking across all clients and splits.

    Uses the exact same validity check as generate_azure_windows, ensuring perfect consistency.

    Returns:
        Structured dictionary with client/split breakdown of candidate, accepted, rejected,
        positive/negative window counts, and episode breakdown.
    """
    cfg = config if config is not None else load_azure_preprocessing_config()
    data = pipeline_data or load_pipeline_data(
        data_dir=data_dir,
        config=cfg,
        telemetry_rows=telemetry_rows,
        failure_rows=failure_rows,
        machine_rows=machine_rows,
    )
    partitions = get_client_partitions(data.machines, config=cfg)

    sensors = cfg.sensor_names
    safe_scale = cfg.zero_iqr_fallback_scale
    history_hours = cfg.history_hours
    horizon_hours = cfg.horizon_hours
    cadence = cfg.cadence_hours

    report: Dict[str, Any] = {
        "pipeline": cfg.pipeline_name,
        "phase": cfg.phase,
        "splits": {name: {"start": str(s.start), "end": str(s.end)} for name, s in cfg.splits.items()},
        "clients": {},
    }

    for client_name in sorted(partitions):
        partition = partitions[client_name]
        client_summary: Dict[str, Any] = {
            "model": partition.model,
            "is_held_out": partition.is_held_out,
            "machine_count": len(partition.machine_ids),
            "allowed_splits": list(partition.allowed_splits),
            "splits": {},
        }

        for split_name in sorted(cfg.splits):
            if split_name not in partition.allowed_splits:
                continue

            split_bounds = cfg.splits[split_name]
            earliest_t = split_bounds.start + timedelta(hours=history_hours)
            latest_t = split_bounds.end - timedelta(hours=horizon_hours + 1)

            candidate_count = 0
            accepted_count = 0
            positives = 0
            negatives = 0
            rejection_counts: Counter = Counter()

            all_episodes_in_split: List[FailureEpisode] = []
            labeled_episode_set: Set[Tuple[int, datetime]] = set()

            for machine_id in partition.machine_ids:
                m_episodes = data.episodes.get(machine_id, [])
                m_index = data.indices.get(machine_id)

                for ep in m_episodes:
                    if split_bounds.contains(ep.timestamp):
                        all_episodes_in_split.append(ep)

                if m_index is None:
                    # Machine missing telemetry entirely
                    current_t = earliest_t
                    while current_t <= latest_t:
                        candidate_count += 1
                        rejection_counts[PipelineRejectionReason.MISSING_TELEMETRY] += 1
                        current_t += timedelta(hours=cadence)
                    continue

                current_t = earliest_t
                while current_t <= latest_t:
                    candidate_count += 1

                    is_valid, reason, _, _ = _check_candidate_window(
                        m_index, m_episodes, current_t, history_hours, horizon_hours, sensors, safe_scale
                    )
                    if not is_valid:
                        assert reason is not None
                        rejection_counts[reason] += 1
                        current_t += timedelta(hours=cadence)
                        continue

                    accepted_count += 1
                    y = label_window(m_episodes, current_t, horizon_hours=horizon_hours)
                    if y == 1:
                        positives += 1
                        # Attribute to episodes falling in (current_t, current_t + horizon_hours]
                        t_end_horiz = current_t + timedelta(hours=horizon_hours)
                        for ep in m_episodes:
                            if current_t < ep.timestamp <= t_end_horiz:
                                labeled_episode_set.add((ep.machine_id, ep.timestamp))
                    else:
                        negatives += 1

                    current_t += timedelta(hours=cadence)

            # Analyze honest reasons for unlabelable episodes
            unlabelable_breakdown: Counter = Counter()
            for ep in all_episodes_in_split:
                key = (ep.machine_id, ep.timestamp)
                if key in labeled_episode_set:
                    continue

                m_index = data.indices.get(ep.machine_id)
                m_eps = data.episodes.get(ep.machine_id, [])

                # An episode at E is targetable by prediction times t in [E - horizon_hours, E - 1h]
                reasons_seen = set()
                potential_t = ep.timestamp - timedelta(hours=horizon_hours)
                while potential_t < ep.timestamp:
                    if potential_t < earliest_t or potential_t > latest_t:
                        reasons_seen.add("split_boundary")
                    elif m_index is None:
                        reasons_seen.add("missing_telemetry")
                    else:
                        is_v, r, _, _ = _check_candidate_window(
                            m_index, m_eps, potential_t, history_hours, horizon_hours, sensors, safe_scale
                        )
                        if not is_v and r is not None:
                            reasons_seen.add(r)
                    potential_t += timedelta(hours=cadence)

                if "blackout" in reasons_seen and "split_boundary" not in reasons_seen:
                    unlabelable_breakdown["blackout"] += 1
                elif "split_boundary" in reasons_seen:
                    unlabelable_breakdown["split_boundary"] += 1
                elif "right_censored" in reasons_seen:
                    unlabelable_breakdown["right_censored"] += 1
                elif "missing_telemetry" in reasons_seen:
                    unlabelable_breakdown["missing_telemetry"] += 1
                elif "nonfinite_values" in reasons_seen:
                    unlabelable_breakdown["nonfinite_values"] += 1
                elif "duplicate_timestamp" in reasons_seen:
                    unlabelable_breakdown["duplicate_timestamp"] += 1
                else:
                    unlabelable_breakdown["other"] += 1

            unlabelable_count = len(all_episodes_in_split) - len(labeled_episode_set)
            prevalence = (positives / accepted_count * 100.0) if accepted_count > 0 else 0.0

            # Verify candidate = accepted + rejected
            total_rejected = sum(rejection_counts.values())
            assert candidate_count == accepted_count + total_rejected

            client_summary["splits"][split_name] = {
                "candidate_windows": candidate_count,
                "accepted_windows": accepted_count,
                "rejected_windows": total_rejected,
                "rejections_by_reason": dict(rejection_counts),
                "positive_windows": positives,
                "negative_windows": negatives,
                "positive_prevalence_pct": round(prevalence, 4),
                "total_episodes_in_split": len(all_episodes_in_split),
                "labelable_episodes": len(labeled_episode_set),
                "unlabelable_episodes": unlabelable_count,
                "unlabelable_breakdown": dict(unlabelable_breakdown),
            }

        report["clients"][client_name] = client_summary

    return report
