"""Event-level evaluation, alert generation, and failure episode matching for Phase 3 Azure PdM.

Specification:
- Inputs explicit collapsed FailureEpisode records, eligible prediction metadata/scores,
  required cohort machine_ids, frozen threshold, and explicit split interval.
- Enforces strict Phase 2B boundary rule:
    t - 48h >= split.start AND t < split.end AND t + 24h < split.end (STRICT).
- Scopes evaluation to the explicit cohort machine_ids; rejects prediction records outside cohort
  and prediction records with mismatched split metadata.
- Counts all collapsed episodes within [split.start, split.end) for cohort machines,
  properly retaining machines with zero valid prediction opportunities as ineligible.
- Generates positive alerts chronologically per machine, suppressing subsequent alerts for
  24 hours after each emitted alert.
- Matches each alert to at most one earliest unmatched future episode with event timestamp
  strictly in (t, t+24h]; each episode matches at most one alert.
- Reports total episodes, eligible, ineligible/unobservable separately.
- Reports detected eligible episodes, event recall, lead hours for matched events, distinct
  true/false alert counts, and false alerts per 100 eligible machine-days
  (where eligible machine-days = distinct valid prediction opportunities / 24).
- Avoids treating overlapping positive windows as independent failure events.
- Defines robust behavior for multiple close episodes, exact boundary conditions, and zero denominators.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from fedmaint.data.adapter import SampleMetadata
from fedmaint.data.config import Phase2BConfig, load_azure_preprocessing_config
from fedmaint.data.episodes import FailureEpisode
from fedmaint.data.pipeline import AzureSample
from fedmaint.data.splits import HoldoutLeakageError, SplitBounds, get_default_splits


@dataclass(frozen=True)
class EventMatch:
    """A match between an emitted alert and a future failure episode."""
    machine_id: int
    alert_time: datetime
    episode_timestamp: datetime
    lead_hours: float
    components: Tuple[str, ...]


@dataclass(frozen=True)
class EventMetrics:
    """Event-level evaluation metrics with 24-hour alert cooldown and chronological episode matching."""
    total_episodes: int
    eligible_episodes: int
    ineligible_episodes: int
    detected_episodes: int
    missed_episodes: int
    event_recall: Optional[float]
    true_alerts: int
    false_alerts: int
    total_alerts: int
    eligible_machine_days: float
    false_alerts_per_100_machine_days: Optional[float]
    lead_hours: Tuple[float, ...]
    mean_lead_hours: Optional[float]

    @property
    def unobservable_episodes(self) -> int:
        """Alias for ineligible_episodes (unobservable within split)."""
        return self.ineligible_episodes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a dictionary."""
        return {
            "total_episodes": self.total_episodes,
            "eligible_episodes": self.eligible_episodes,
            "ineligible_episodes": self.ineligible_episodes,
            "unobservable_episodes": self.unobservable_episodes,
            "detected_episodes": self.detected_episodes,
            "missed_episodes": self.missed_episodes,
            "event_recall": self.event_recall,
            "true_alerts": self.true_alerts,
            "false_alerts": self.false_alerts,
            "total_alerts": self.total_alerts,
            "eligible_machine_days": self.eligible_machine_days,
            "false_alerts_per_100_machine_days": self.false_alerts_per_100_machine_days,
            "lead_hours": list(self.lead_hours),
            "mean_lead_hours": self.mean_lead_hours,
        }


def _parse_predictions(
    predictions: Sequence[Any],
    scores: Optional[Sequence[float]] = None,
) -> List[Tuple[int, datetime, float, Optional[str]]]:
    """Parse various prediction input formats into canonical (machine_id, pred_time, score, split_name)."""
    parsed: List[Tuple[int, datetime, float, Optional[str]]] = []

    if scores is not None:
        if len(predictions) != len(scores):
            raise ValueError(
                f"Length mismatch: len(predictions)={len(predictions)} != len(scores)={len(scores)}"
            )
        for p, s in zip(predictions, scores):
            score = float(s)
            if isinstance(p, (SampleMetadata, AzureSample)):
                parsed.append((int(p.machine_id), p.prediction_time, score, p.split_name))
            elif isinstance(p, dict):
                parsed.append((int(p["machine_id"]), p["prediction_time"], score, p.get("split_name")))
            elif isinstance(p, (tuple, list)) and len(p) >= 2:
                parsed.append((int(p[0]), p[1], score, p[2] if len(p) > 2 else None))
            elif hasattr(p, "machine_id") and hasattr(p, "prediction_time"):
                parsed.append((int(p.machine_id), p.prediction_time, score, getattr(p, "split_name", None)))
            else:
                raise TypeError(f"Unsupported prediction record format with separate scores: {type(p)}")
    else:
        for p in predictions:
            if isinstance(p, dict):
                score = float(p.get("score", p.get("y_score", p.get("probability", 0.0))))
                parsed.append((int(p["machine_id"]), p["prediction_time"], score, p.get("split_name")))
            elif isinstance(p, (tuple, list)):
                if len(p) == 2:
                    meta, s = p
                    score = float(s)
                    if hasattr(meta, "machine_id") and hasattr(meta, "prediction_time"):
                        parsed.append((int(meta.machine_id), meta.prediction_time, score, getattr(meta, "split_name", None)))
                    elif isinstance(meta, dict):
                        parsed.append((int(meta["machine_id"]), meta["prediction_time"], score, meta.get("split_name")))
                    else:
                        parsed.append((int(meta[0]), meta[1], score, None))
                elif len(p) >= 3:
                    parsed.append((int(p[0]), p[1], float(p[2]), p[3] if len(p) > 3 else None))
                else:
                    raise ValueError(f"Unexpected tuple length in predictions: {p}")
            elif hasattr(p, "machine_id") and hasattr(p, "prediction_time") and hasattr(p, "score"):
                parsed.append((int(p.machine_id), p.prediction_time, float(p.score), getattr(p, "split_name", None)))
            else:
                raise TypeError(f"Cannot parse prediction record: {p}")

    return parsed


def evaluate_event_metrics(
    episodes: Union[Sequence[FailureEpisode], Dict[int, Sequence[FailureEpisode]]],
    predictions: Sequence[Any],
    machine_ids: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
    *,
    split: Optional[Union[SplitBounds, Tuple[datetime, datetime]]] = None,
    split_name: Optional[str] = None,
    config: Optional[Phase2BConfig] = None,
) -> EventMetrics:
    """Evaluate event-level detection rates and false alert frequency for Azure PdM.

    Args:
        episodes: Sequence of FailureEpisode objects or mapping of machine_id -> episodes.
        predictions: Sequence of prediction opportunities (SampleMetadata, AzureSample, or tuples).
        machine_ids: Required explicit cohort of machine IDs in client partition.
        scores: Optional sequence of continuous prediction scores matching predictions.
        threshold: Frozen decision threshold for positive alerts.
        split: Explicit SplitBounds instance or (split_start, split_end) tuple defining the split.
        split_name: Explicit split name ('test', 'val', 'train') to look up default split bounds.
        config: Optional Phase2BConfig instance.

    Returns:
        EventMetrics instance.

    Raises:
        ValueError: If machine_ids is missing/empty, split is unspecified, or predictions contain
                    machine IDs outside the cohort.
        HoldoutLeakageError: If predictions contain metadata with mismatched split_name.
    """
    if machine_ids is None or len(machine_ids) == 0:
        raise ValueError("machine_ids is required: must specify client partition cohort machines.")

    cohort = tuple(sorted(set(int(m) for m in machine_ids)))
    cohort_set = set(cohort)

    # Require explicit split bounds or known split_name; do not infer from score timestamps
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

    # Parse predictions and reject predictions outside cohort or with mismatched split metadata
    parsed_preds = _parse_predictions(predictions, scores)
    for m_id, pred_t, s, p_split in parsed_preds:
        if m_id not in cohort_set:
            raise ValueError(f"Prediction for machineID {m_id} is outside the specified cohort {cohort}.")
        if p_split is not None and target_split_name is not None and p_split != target_split_name:
            raise HoldoutLeakageError(
                f"Prediction record split '{p_split}' does not match evaluation split '{target_split_name}'."
            )

    # Organize predictions by machine for all cohort machines:
    # Phase 2B boundary condition:
    # t - 48h >= split.start AND t < split.end AND t + 24h < split.end (STRICT)
    machine_opps: Dict[int, Dict[datetime, float]] = {m_id: {} for m_id in cohort}
    for m_id, pred_t, s, _ in parsed_preds:
        if (
            pred_t - timedelta(hours=history_hours) >= split_start
            and pred_t < split_end
            and pred_t + timedelta(hours=horizon_hours) < split_end
        ):
            machine_opps[m_id][pred_t] = s

    # Total valid prediction opportunities across cohort machines
    num_valid_opps = sum(len(opps) for opps in machine_opps.values())
    eligible_machine_days = num_valid_opps / 24.0

    # Filter episode records to this cohort
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

    total_episodes_count = 0
    eligible_episodes_count = 0
    ineligible_episodes_count = 0
    detected_episodes_count = 0

    all_matched_lead_hours: List[float] = []
    total_true_alerts = 0
    total_false_alerts = 0

    # Evaluate EVERY machine in cohort; machines with zero valid opportunities are preserved
    for m_id in cohort:
        opps = machine_opps[m_id]
        sorted_opp_times = sorted(opps.keys())
        m_episodes = eps_by_machine[m_id]

        # 1. Scope episodes to split: split_start <= timestamp < split_end
        split_episodes = [ep for ep in m_episodes if split_start <= ep.timestamp < split_end]
        total_episodes_count += len(split_episodes)

        # 2. Determine eligibility for each episode:
        # An episode at t_E is eligible iff there exists at least one prediction opportunity t
        # in [t_E - 24h, t_E) whose complete horizon is observable within split (t + 24h < split_end).
        m_eligible_episodes: List[FailureEpisode] = []
        m_ineligible_episodes: List[FailureEpisode] = []

        for ep in split_episodes:
            t_min = ep.timestamp - timedelta(hours=horizon_hours)
            t_max = ep.timestamp

            # Binary search for opportunities in [t_min, t_max)
            idx_start = bisect.bisect_left(sorted_opp_times, t_min)
            idx_end = bisect.bisect_left(sorted_opp_times, t_max)

            # Any opportunity in sorted_opp_times already satisfies t + 24h < split_end
            if idx_end > idx_start:
                m_eligible_episodes.append(ep)
            else:
                m_ineligible_episodes.append(ep)

        eligible_episodes_count += len(m_eligible_episodes)
        ineligible_episodes_count += len(m_ineligible_episodes)

        # 3. Generate positive alerts with 24-hour suppression cooldown
        m_alerts: List[datetime] = []
        last_alert_time: Optional[datetime] = None
        for opp_t in sorted_opp_times:
            score = opps[opp_t]
            if score >= threshold:
                if last_alert_time is None or opp_t >= last_alert_time + timedelta(hours=horizon_hours):
                    m_alerts.append(opp_t)
                    last_alert_time = opp_t

        # 4. Match alerts to episodes:
        # Each alert matches at most one earliest unmatched future episode in (t, t+24h];
        # Each episode matches at most one alert.
        unmatched_episodes = list(m_eligible_episodes)
        for alert_t in m_alerts:
            horizon_end = alert_t + timedelta(hours=horizon_hours)

            matched_ep: Optional[FailureEpisode] = None
            for ep in unmatched_episodes:
                # Strictly after alert_t and up to alert_t + 24h: (alert_t, alert_t + 24h]
                if alert_t < ep.timestamp <= horizon_end:
                    matched_ep = ep
                    break

            if matched_ep is not None:
                lead_h = (matched_ep.timestamp - alert_t).total_seconds() / 3600.0
                all_matched_lead_hours.append(lead_h)
                total_true_alerts += 1
                detected_episodes_count += 1
                unmatched_episodes.remove(matched_ep)
            else:
                total_false_alerts += 1

    missed_episodes_count = eligible_episodes_count - detected_episodes_count
    total_alerts_count = total_true_alerts + total_false_alerts

    # Event recall with zero-denominator handling
    if eligible_episodes_count > 0:
        event_recall: Optional[float] = detected_episodes_count / eligible_episodes_count
    else:
        event_recall = None

    # False alerts per 100 machine-days with zero-denominator handling
    if eligible_machine_days > 0:
        far_per_100_days: Optional[float] = (total_false_alerts / eligible_machine_days) * 100.0
    else:
        far_per_100_days = None

    # Lead hours summary
    if all_matched_lead_hours:
        mean_lead: Optional[float] = sum(all_matched_lead_hours) / len(all_matched_lead_hours)
    else:
        mean_lead = None

    return EventMetrics(
        total_episodes=total_episodes_count,
        eligible_episodes=eligible_episodes_count,
        ineligible_episodes=ineligible_episodes_count,
        detected_episodes=detected_episodes_count,
        missed_episodes=missed_episodes_count,
        event_recall=event_recall,
        true_alerts=total_true_alerts,
        false_alerts=total_false_alerts,
        total_alerts=total_alerts_count,
        eligible_machine_days=float(eligible_machine_days),
        false_alerts_per_100_machine_days=far_per_100_days,
        lead_hours=tuple(all_matched_lead_hours),
        mean_lead_hours=mean_lead,
    )
