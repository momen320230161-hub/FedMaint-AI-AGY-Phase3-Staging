"""Machine-cluster bootstrap uncertainty estimation for Phase 3 Azure PdM.

Specification:
- Resamples entire machines with replacement using an explicit random seed.
- Preserves cluster multiplicity: all windows/events of a resampled machine appear with
  multiplicity equal to the number of times that machine was drawn.
- Computes metrics on resampled machine bundles.
- Reports percentile confidence intervals (lower, upper), point estimates, and number of valid draws.
- Returns None for CI bounds when there is insufficient data (< 2 machines or < min_valid_draws).
- Deterministic and reproducible across all operating environments.
"""
from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from fedmaint.data.adapter import SampleMetadata
from fedmaint.data.config import Phase2BConfig, load_azure_preprocessing_config
from fedmaint.data.episodes import FailureEpisode
from fedmaint.data.normalization import compute_quantile
from fedmaint.data.pipeline import AzureSample
from fedmaint.data.splits import HoldoutLeakageError, SplitBounds, get_default_splits
from .events import EventMetrics, _parse_predictions, evaluate_event_metrics
from .window import WindowMetrics, evaluate_window_metrics

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False


@dataclass(frozen=True)
class BootstrapCI:
    """Confidence interval from machine-cluster bootstrap."""
    metric_name: str
    point_estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    confidence_level: float
    num_draws: int
    num_valid_draws: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize confidence interval to a dictionary."""
        return {
            "metric_name": self.metric_name,
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "num_draws": self.num_draws,
            "num_valid_draws": self.num_valid_draws,
        }


def _compute_ci_bounds(
    valid_values: List[float],
    confidence_level: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute lower and upper percentile bootstrap CI bounds."""
    if not valid_values:
        return None, None
    alpha = 1.0 - confidence_level
    q_lower = alpha / 2.0
    q_upper = 1.0 - q_lower

    ci_lower = compute_quantile(valid_values, q_lower, method="linear")
    ci_upper = compute_quantile(valid_values, q_upper, method="linear")
    return float(ci_lower), float(ci_upper)


def bootstrap_window_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metadata: Sequence[Any],
    threshold: float = 0.5,
    *,
    num_draws: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_valid_draws: int = 10,
) -> Dict[str, BootstrapCI]:
    """Calculate machine-cluster bootstrap confidence intervals for window-level metrics.

    Optimized algorithm:
    - Pre-sorts and vectorizes once across windows without expanding duplicates.
    - Uses per-machine multiplicities/weights for resampled clusters.
    - Respects tied scores and exact existing metric semantics (AP, ROC-AUC, Precision, Recall, F1).
    - Preserves exact seeded machine draw sequence and cluster multiplicity.
    - Pure-Python safe fallback when optional NumPy is unavailable.

    Args:
        y_true: Ground truth binary targets in {0, 1}.
        y_score: Continuous prediction scores.
        metadata: SampleMetadata sequence containing machine_id.
        threshold: Frozen decision threshold.
        num_draws: Number of bootstrap draws (default 1000).
        confidence_level: Confidence level in (0, 1) (default 0.95).
        seed: Explicit random seed for deterministic resampling.
        min_valid_draws: Minimum valid draws required to report CI bounds.

    Returns:
        Dictionary mapping metric_name -> BootstrapCI.
    """
    if len(y_true) != len(y_score) or len(y_true) != len(metadata):
        raise ValueError("Lengths of y_true, y_score, and metadata must match.")
    if num_draws < 1:
        raise ValueError(f"num_draws must be >= 1; got {num_draws}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be between 0 and 1; got {confidence_level}")

    extracted_m_ids: List[int] = []
    extracted_y: List[int] = []
    extracted_s: List[float] = []

    for yt, s, m in zip(y_true, y_score, metadata):
        m_id = getattr(m, "machine_id", None) or (m.get("machine_id") if isinstance(m, dict) else None)
        if m_id is None:
            raise ValueError(f"Metadata object missing machine_id: {m}")
        extracted_m_ids.append(int(m_id))
        extracted_y.append(int(yt))
        extracted_s.append(float(s))

    machine_ids = sorted(set(extracted_m_ids))
    point_wm = evaluate_window_metrics(y_true, y_score, threshold=threshold)

    metric_names = ("average_precision", "roc_auc", "precision", "recall", "f1")

    # Insufficient data check (< 2 machines)
    if len(machine_ids) < 2:
        return {
            name: BootstrapCI(
                metric_name=name,
                point_estimate=getattr(point_wm, name),
                ci_lower=None,
                ci_upper=None,
                confidence_level=confidence_level,
                num_draws=num_draws,
                num_valid_draws=0,
            )
            for name in metric_names
        }

    n_machines = len(machine_ids)
    mach_to_idx = {m_id: i for i, m_id in enumerate(machine_ids)}
    n_samples = len(extracted_y)

    # Precompute per-machine counts at frozen threshold
    m_tp = [0] * n_machines
    m_fp = [0] * n_machines
    m_fn = [0] * n_machines
    m_tn = [0] * n_machines

    f_threshold = float(threshold)
    for y, s, m_id in zip(extracted_y, extracted_s, extracted_m_ids):
        idx = mach_to_idx[m_id]
        pred = 1 if s >= f_threshold else 0
        if y == 1:
            if pred == 1:
                m_tp[idx] += 1
            else:
                m_fn[idx] += 1
        else:
            if pred == 1:
                m_fp[idx] += 1
            else:
                m_tn[idx] += 1

    rng = random.Random(seed)
    draw_metrics: Dict[str, List[Optional[float]]] = {name: [] for name in metric_names}

    if _HAS_NUMPY and np is not None:
        np_m_tp = np.array(m_tp, dtype=np.int64)
        np_m_fp = np.array(m_fp, dtype=np.int64)
        np_m_fn = np.array(m_fn, dtype=np.int64)
        np_m_tn = np.array(m_tn, dtype=np.int64)

        arr_s = np.array(extracted_s, dtype=np.float64)
        arr_y = np.array(extracted_y, dtype=np.int64)
        arr_m = np.array([mach_to_idx[m] for m in extracted_m_ids], dtype=np.int64)

        # Sort descending by score; on ties stable sort preserves relative indexing
        sort_indices = np.argsort(arr_s, kind="stable")[::-1]
        sorted_s = arr_s[sort_indices]
        sorted_y = arr_y[sort_indices]
        sorted_m = arr_m[sort_indices]

        # Identify distinct score group boundaries (tie ends)
        diff_mask = sorted_s[1:] != sorted_s[:-1]
        tie_ends = np.concatenate([np.where(diff_mask)[0], np.array([n_samples - 1], dtype=np.int64)])

        for _ in range(num_draws):
            # Resample machines with replacement using exact seeded sequence
            sampled_machines = [rng.choice(machine_ids) for _ in range(n_machines)]
            mach_counts = np.bincount([mach_to_idx[m] for m in sampled_machines], minlength=n_machines)

            # Threshold metrics
            tp = int(np.dot(mach_counts, np_m_tp))
            fp = int(np.dot(mach_counts, np_m_fp))
            fn = int(np.dot(mach_counts, np_m_fn))
            tn = int(np.dot(mach_counts, np_m_tn))

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

            n_pos = tp + fn
            n_neg = fp + tn

            if n_pos == 0 or n_neg == 0:
                ap = None
                roc_auc = None
            else:
                weights = mach_counts[sorted_m]
                w_pos = weights * sorted_y
                w_neg = weights - w_pos
                cum_pos = np.cumsum(w_pos)
                cum_neg = np.cumsum(w_neg)

                tp_curve = cum_pos[tie_ends]
                fp_curve = cum_neg[tie_ends]

                delta_tp = np.diff(np.insert(tp_curve, 0, 0))
                delta_fp = np.diff(np.insert(fp_curve, 0, 0))

                # AP calculation matching scikit-learn / pure Python
                sum_tp_fp = tp_curve + fp_curve
                nonzero_mask = sum_tp_fp > 0
                prec_curve = np.zeros_like(sum_tp_fp, dtype=np.float64)
                prec_curve[nonzero_mask] = tp_curve[nonzero_mask] / sum_tp_fp[nonzero_mask]

                rec_curve = tp_curve / float(n_pos)
                delta_rec = np.diff(np.insert(rec_curve, 0, 0.0))
                ap = float(np.sum(delta_rec * prec_curve))

                # ROC-AUC calculation matching Mann-Whitney U / pure Python
                total_samples = float(n_pos + n_neg)
                w_curve = delta_tp + delta_fp
                avg_ranks = total_samples - sum_tp_fp + 0.5 * (w_curve + 1.0)
                rank_sum_pos = np.sum(delta_tp * avg_ranks)
                u_stat = rank_sum_pos - 0.5 * float(n_pos * (n_pos + 1))
                roc_auc = float(u_stat / float(n_pos * n_neg))

            draw_metrics["average_precision"].append(ap)
            draw_metrics["roc_auc"].append(roc_auc)
            draw_metrics["precision"].append(prec)
            draw_metrics["recall"].append(rec)
            draw_metrics["f1"].append(f1)
    else:
        # Pure Python safe fallback when optional NumPy is unavailable
        indexed_samples = sorted(
            zip(extracted_s, extracted_y, [mach_to_idx[m] for m in extracted_m_ids]),
            key=lambda item: item[0],
            reverse=True,
        )

        groups: List[Tuple[Dict[int, int], Dict[int, int]]] = []
        curr_score: Optional[float] = None
        curr_pos: Dict[int, int] = defaultdict(int)
        curr_neg: Dict[int, int] = defaultdict(int)

        for s, y, m_idx in indexed_samples:
            if curr_score is None:
                curr_score = s
            elif s != curr_score:
                groups.append((dict(curr_pos), dict(curr_neg)))
                curr_score = s
                curr_pos = defaultdict(int)
                curr_neg = defaultdict(int)

            if y == 1:
                curr_pos[m_idx] += 1
            else:
                curr_neg[m_idx] += 1

        if curr_score is not None:
            groups.append((dict(curr_pos), dict(curr_neg)))

        for _ in range(num_draws):
            sampled_machines = [rng.choice(machine_ids) for _ in range(n_machines)]
            mach_counts = [0] * n_machines
            for m in sampled_machines:
                mach_counts[mach_to_idx[m]] += 1

            tp = sum(c * val for c, val in zip(mach_counts, m_tp))
            fp = sum(c * val for c, val in zip(mach_counts, m_fp))
            fn = sum(c * val for c, val in zip(mach_counts, m_fn))
            tn = sum(c * val for c, val in zip(mach_counts, m_tn))

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

            n_pos = tp + fn
            n_neg = fp + tn

            if n_pos == 0 or n_neg == 0:
                ap = None
                roc_auc = None
            else:
                cum_tp = 0
                cum_fp = 0
                prev_rec = 0.0
                ap = 0.0
                rank_sum_pos = 0.0
                total_samples = float(n_pos + n_neg)

                for pos_dict, neg_dict in groups:
                    d_tp = 0
                    for m_idx, cnt in pos_dict.items():
                        c = mach_counts[m_idx]
                        if c:
                            d_tp += c * cnt

                    d_fp = 0
                    for m_idx, cnt in neg_dict.items():
                        c = mach_counts[m_idx]
                        if c:
                            d_fp += c * cnt

                    if d_tp == 0 and d_fp == 0:
                        continue

                    cum_tp += d_tp
                    cum_fp += d_fp

                    curr_prec = cum_tp / (cum_tp + cum_fp)
                    curr_rec = cum_tp / n_pos
                    delta_rec = curr_rec - prev_rec
                    ap += delta_rec * curr_prec
                    prev_rec = curr_rec

                    w = d_tp + d_fp
                    avg_rank = total_samples - (cum_tp + cum_fp) + 0.5 * (w + 1.0)
                    rank_sum_pos += d_tp * avg_rank

                u_stat = rank_sum_pos - 0.5 * float(n_pos * (n_pos + 1))
                roc_auc = float(u_stat / float(n_pos * n_neg))

            draw_metrics["average_precision"].append(ap)
            draw_metrics["roc_auc"].append(roc_auc)
            draw_metrics["precision"].append(prec)
            draw_metrics["recall"].append(rec)
            draw_metrics["f1"].append(f1)

    # Calculate CIs
    results: Dict[str, BootstrapCI] = {}
    for name in metric_names:
        valid_vals = [v for v in draw_metrics[name] if v is not None and math.isfinite(v)]
        num_valid = len(valid_vals)

        if num_valid >= min_valid_draws:
            ci_lower, ci_upper = _compute_ci_bounds(valid_vals, confidence_level)
        else:
            ci_lower, ci_upper = None, None

        results[name] = BootstrapCI(
            metric_name=name,
            point_estimate=getattr(point_wm, name),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            num_draws=num_draws,
            num_valid_draws=num_valid,
        )

    return results


def bootstrap_event_ci(
    episodes: Union[Sequence[FailureEpisode], Dict[int, Sequence[FailureEpisode]]],
    predictions: Sequence[Any],
    machine_ids: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
    *,
    split: Optional[Union[SplitBounds, Tuple[datetime, datetime]]] = None,
    split_name: Optional[str] = None,
    config: Optional[Phase2BConfig] = None,
    num_draws: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_valid_draws: int = 10,
) -> Dict[str, BootstrapCI]:
    """Calculate machine-cluster bootstrap confidence intervals for event-level metrics.

    Args:
        episodes: Sequence of FailureEpisode objects or mapping of machine_id -> episodes.
        predictions: Prediction opportunities sequence.
        machine_ids: Required explicit cohort of machine IDs in client partition.
        scores: Optional continuous prediction scores.
        threshold: Frozen decision threshold.
        split: Explicit SplitBounds instance or (split_start, split_end) tuple.
        split_name: Explicit split name ('test', 'val', 'train').
        config: Optional Phase2BConfig instance.
        num_draws: Number of bootstrap draws (default 1000).
        confidence_level: Confidence level in (0, 1) (default 0.95).
        seed: Explicit random seed.
        min_valid_draws: Minimum valid draws required.

    Returns:
        Dictionary mapping metric_name -> BootstrapCI.
    """
    if machine_ids is None or len(machine_ids) == 0:
        raise ValueError("machine_ids is required: must specify client partition cohort machines.")

    cohort = tuple(sorted(set(int(m) for m in machine_ids)))
    cohort_set = set(cohort)

    if split is None and split_name is None:
        raise ValueError("Must explicitly specify either 'split' (SplitBounds or (start, end)) or a known 'split_name'.")

    cfg = config if config is not None else load_azure_preprocessing_config()
    split_bounds_map = get_default_splits(cfg)

    split_start: datetime
    split_end: datetime
    target_split_name: Optional[str] = None

    if split is not None:
        if isinstance(split, SplitBounds):
            split_start = split.start
            split_end = split.end
            target_split_name = split.name
        elif isinstance(split, (tuple, list)) and len(split) == 2:
            split_start = split[0]
            split_end = split[1]
            target_split_name = split_name
        else:
            raise TypeError(f"Invalid split specification: {split}")
    else:
        assert split_name is not None
        if split_name not in split_bounds_map:
            raise ValueError(f"Unknown split name '{split_name}'; approved splits: {tuple(split_bounds_map.keys())}")
        sb = split_bounds_map[split_name]
        split_start = sb.start
        split_end = sb.end
        target_split_name = split_name

    history_hours = cfg.history_hours
    horizon_hours = cfg.horizon_hours

    parsed_preds = _parse_predictions(predictions, scores)
    for m_id, pred_t, s, p_split in parsed_preds:
        if m_id not in cohort_set:
            raise ValueError(f"Prediction for machineID {m_id} is outside the specified cohort {cohort}.")
        if p_split is not None and target_split_name is not None and p_split != target_split_name:
            raise HoldoutLeakageError(
                f"Prediction record split '{p_split}' does not match evaluation split '{target_split_name}'."
            )

    # Point estimate on full cohort
    point_em = evaluate_event_metrics(
        episodes=episodes,
        predictions=parsed_preds,
        machine_ids=cohort,
        threshold=threshold,
        split=(split_start, split_end),
        split_name=target_split_name,
        config=cfg,
    )

    metric_names = ("event_recall", "false_alerts_per_100_machine_days", "mean_lead_hours")

    # Insufficient data check (< 2 machines in cohort)
    if len(cohort) < 2:
        return {
            name: BootstrapCI(
                metric_name=name,
                point_estimate=getattr(point_em, name),
                ci_lower=None,
                ci_upper=None,
                confidence_level=confidence_level,
                num_draws=num_draws,
                num_valid_draws=0,
            )
            for name in metric_names
        }

    # Group predictions by machine: strict Phase 2B boundary rule
    machine_opps: Dict[int, Dict[datetime, float]] = {m_id: {} for m_id in cohort}
    for m_id, pred_t, s, _ in parsed_preds:
        if (
            pred_t - timedelta(hours=history_hours) >= split_start
            and pred_t < split_end
            and pred_t + timedelta(hours=horizon_hours) < split_end
        ):
            machine_opps[m_id][pred_t] = s

    # Group episodes by machine
    eps_by_machine: Dict[int, List[FailureEpisode]] = {m_id: [] for m_id in cohort}
    if isinstance(episodes, dict):
        for m_id, m_eps in episodes.items():
            im_id = int(m_id)
            if im_id in cohort_set:
                eps_by_machine[im_id].extend(m_eps)
    else:
        for ep in episodes:
            im_id = int(ep.machine_id)
            if im_id in cohort_set:
                eps_by_machine[im_id].append(ep)

    for m_id in eps_by_machine:
        eps_by_machine[m_id].sort(key=lambda ep: ep.timestamp)

    # Precompute single-machine metrics to preserve cluster multiplicity without inter-instance interference
    m_opp_count: Dict[int, int] = {}
    m_eligible_count: Dict[int, int] = {}
    m_true_alerts: Dict[int, int] = {}
    m_false_alerts: Dict[int, int] = {}
    m_detected_count: Dict[int, int] = {}
    m_lead_hours: Dict[int, List[float]] = {}

    for m_id in cohort:
        opps = machine_opps[m_id]
        sorted_opp_times = sorted(opps.keys())
        m_opp_count[m_id] = len(sorted_opp_times)

        split_eps = [ep for ep in eps_by_machine[m_id] if split_start <= ep.timestamp < split_end]

        # Determine eligibility
        eligible_eps: List[FailureEpisode] = []
        for ep in split_eps:
            t_min = ep.timestamp - timedelta(hours=horizon_hours)
            t_max = ep.timestamp
            idx_start = bisect.bisect_left(sorted_opp_times, t_min)
            idx_end = bisect.bisect_left(sorted_opp_times, t_max)

            if idx_end > idx_start:
                eligible_eps.append(ep)

        m_eligible_count[m_id] = len(eligible_eps)

        # Alerts with 24h cooldown
        alerts: List[datetime] = []
        last_alert_t: Optional[datetime] = None
        for opp_t in sorted_opp_times:
            if opps[opp_t] >= threshold:
                if last_alert_t is None or opp_t >= last_alert_t + timedelta(hours=horizon_hours):
                    alerts.append(opp_t)
                    last_alert_t = opp_t

        # Match alerts to eligible episodes
        unmatched = list(eligible_eps)
        true_alerts = 0
        false_alerts = 0
        leads: List[float] = []

        for alt in alerts:
            horiz_end = alt + timedelta(hours=horizon_hours)
            matched_ep = None
            for ep in unmatched:
                if alt < ep.timestamp <= horiz_end:
                    matched_ep = ep
                    break
            if matched_ep is not None:
                true_alerts += 1
                leads.append((matched_ep.timestamp - alt).total_seconds() / 3600.0)
                unmatched.remove(matched_ep)
            else:
                false_alerts += 1

        m_true_alerts[m_id] = true_alerts
        m_false_alerts[m_id] = false_alerts
        m_detected_count[m_id] = true_alerts
        m_lead_hours[m_id] = leads

    rng = random.Random(seed)
    n_cohort = len(cohort)

    draw_metrics: Dict[str, List[Optional[float]]] = {name: [] for name in metric_names}

    for _ in range(num_draws):
        sampled_machines = [rng.choice(cohort) for _ in range(n_cohort)]

        draw_opps = sum(m_opp_count[m] for m in sampled_machines)
        draw_days = draw_opps / 24.0
        draw_eligible = sum(m_eligible_count[m] for m in sampled_machines)
        draw_detected = sum(m_detected_count[m] for m in sampled_machines)
        draw_false_alerts = sum(m_false_alerts[m] for m in sampled_machines)
        draw_leads = [lh for m in sampled_machines for lh in m_lead_hours[m]]

        draw_recall = (draw_detected / draw_eligible) if draw_eligible > 0 else None
        draw_far = ((draw_false_alerts / draw_days) * 100.0) if draw_days > 0 else None
        draw_lead = (sum(draw_leads) / len(draw_leads)) if draw_leads else None

        draw_metrics["event_recall"].append(draw_recall)
        draw_metrics["false_alerts_per_100_machine_days"].append(draw_far)
        draw_metrics["mean_lead_hours"].append(draw_lead)

    results: Dict[str, BootstrapCI] = {}
    for name in metric_names:
        valid_vals = [v for v in draw_metrics[name] if v is not None and math.isfinite(v)]
        num_valid = len(valid_vals)

        if num_valid >= min_valid_draws:
            ci_lower, ci_upper = _compute_ci_bounds(valid_vals, confidence_level)
        else:
            ci_lower, ci_upper = None, None

        results[name] = BootstrapCI(
            metric_name=name,
            point_estimate=getattr(point_em, name),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            num_draws=num_draws,
            num_valid_draws=num_valid,
        )

    return results
