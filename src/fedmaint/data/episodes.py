"""Failure episode aggregation, blackout exclusion, and window labeling.

This module enforces the approved Phase 2B failure episode rules:
- Collapse all component failure records sharing machineID and exact timestamp into one episode.
- Repeated timestamps at different hours remain distinct episodes.
- Blackout exclusion: exclude prediction time t if any episode occurs at t or in the prior 48h ([t-48h, t] inclusive).
- Labeling: binary y=1 iff at least one distinct failure episode occurs in (t, t+24h].
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

Record = Dict[str, object]


@dataclass(frozen=True)
class FailureEpisode:
    """A distinct machine failure replacement episode.

    Multiple component failures recorded for the same machine at the exact same
    timestamp are collapsed into a single episode with multiple components.
    """
    machine_id: int
    timestamp: datetime
    components: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("FailureEpisode must contain at least one failed component.")


def collapse_failure_episodes(rows: Iterable[Record]) -> Dict[int, List[FailureEpisode]]:
    """Collapse failure records sharing machineID and exact timestamp into distinct episodes.

    Args:
        rows: Iterable of validated failure records with keys 'machineID', 'datetime', 'failure'.

    Returns:
        Dictionary mapping machineID to a sorted list of FailureEpisode objects.
    """
    grouped: Dict[Tuple[int, datetime], set] = defaultdict(set)
    for row in rows:
        machine_id = int(row["machineID"])
        timestamp = row["datetime"]
        if not isinstance(timestamp, datetime):
            raise TypeError(f"Expected datetime for record 'datetime', got {type(timestamp)}")
        comp = str(row["failure"])
        grouped[(machine_id, timestamp)].add(comp)

    by_machine: Dict[int, List[FailureEpisode]] = defaultdict(list)
    for (machine_id, timestamp), comps in grouped.items():
        sorted_comps = tuple(sorted(comps))
        by_machine[machine_id].append(
            FailureEpisode(machine_id=machine_id, timestamp=timestamp, components=sorted_comps)
        )

    for machine_id in by_machine:
        by_machine[machine_id].sort(key=lambda ep: ep.timestamp)

    return dict(by_machine)


def is_in_blackout(
    episodes: Sequence[FailureEpisode],
    t: datetime,
    *,
    history_hours: int = 48,
) -> bool:
    """Check if prediction time t falls within an episode blackout window.

    A time t is excluded if any recorded episode occurs at t or in the prior 48h,
    i.e., within [t - history_hours, t] inclusive.

    Args:
        episodes: Chronologically sorted failure episodes for the machine.
        t: Candidate prediction timestamp.
        history_hours: Blackout window duration in hours (default 48).

    Returns:
        True if t is blacked out by an episode, False otherwise.
    """
    if not episodes:
        return False
    window_start = t - timedelta(hours=history_hours)
    # Check if any episode timestamp E satisfies window_start <= E <= t
    # Using bisect on timestamps
    timestamps = [ep.timestamp for ep in episodes]
    idx = bisect.bisect_left(timestamps, window_start)
    if idx < len(timestamps) and timestamps[idx] <= t:
        return True
    return False


def get_episodes_in_blackout(
    episodes: Sequence[FailureEpisode],
    t: datetime,
    *,
    history_hours: int = 48,
) -> List[FailureEpisode]:
    """Return all failure episodes falling in the blackout window [t - history_hours, t]."""
    if not episodes:
        return []
    window_start = t - timedelta(hours=history_hours)
    return [ep for ep in episodes if window_start <= ep.timestamp <= t]


def label_window(
    episodes: Sequence[FailureEpisode],
    t: datetime,
    *,
    horizon_hours: int = 24,
) -> int:
    """Compute binary failure label y for prediction time t.

    y = 1 if at least one failure episode occurs strictly after t and up to t + horizon_hours,
    i.e., in (t, t + horizon_hours]. Otherwise, y = 0.

    Args:
        episodes: Chronologically sorted failure episodes for the machine.
        t: Prediction timestamp.
        horizon_hours: Future observation horizon in hours (default 24).

    Returns:
        1 if failure in (t, t + horizon_hours], 0 otherwise.
    """
    if not episodes:
        return 0
    window_end = t + timedelta(hours=horizon_hours)
    timestamps = [ep.timestamp for ep in episodes]
    idx = bisect.bisect_right(timestamps, t)
    if idx < len(timestamps) and timestamps[idx] <= window_end:
        return 1
    return 0


def get_episodes_in_horizon(
    episodes: Sequence[FailureEpisode],
    t: datetime,
    *,
    horizon_hours: int = 24,
) -> List[FailureEpisode]:
    """Return all failure episodes falling in the label horizon (t, t + horizon_hours]."""
    if not episodes:
        return []
    window_end = t + timedelta(hours=horizon_hours)
    return [ep for ep in episodes if t < ep.timestamp <= window_end]
