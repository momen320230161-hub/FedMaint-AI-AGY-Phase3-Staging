"""Validation threshold selection with strict leakage prevention for Phase 3 Azure PdM.

Specification:
- Validates that ALL input metadata belong to participating clients and split_name="val".
- Rejects missing metadata, training, test, or held-out client (model1) data with HoldoutLeakageError.
- Evaluates candidate thresholds maximizing F1 in O(N log N) time via cumulative sweep.
- In case of ties, deterministically chooses the highest threshold.
- Returns frozen threshold and validation score.
- Scopes globally for central/shared models and locally per participating client.
- Test paths must never invoke threshold selection.
"""
from __future__ import annotations

import bisect
import collections.abc
import math
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from fedmaint.data.adapter import SampleMetadata
from fedmaint.data.splits import HoldoutLeakageError

PARTICIPATING_CLIENTS: frozenset[str] = frozenset({"client_model2", "client_model3", "client_model4"})
HELDOUT_CLIENTS: frozenset[str] = frozenset({"heldout_model1", "model1"})


class ThresholdResult(NamedTuple):
    """Result of validation threshold selection."""
    threshold: float
    score: float
    metric: str = "f1"

    @property
    def frozen_threshold(self) -> float:
        """Alias for the selected frozen decision threshold."""
        return self.threshold

    @property
    def val_score(self) -> float:
        """Alias for the validation F1 score achieved at this threshold."""
        return self.score


def validate_threshold_metadata(meta: Any) -> None:
    """Validate that input metadata belongs strictly to participating clients and the 'val' split.

    Args:
        meta: A SampleMetadata instance, dict, or object with client_name and split_name attributes.

    Raises:
        HoldoutLeakageError: If metadata specifies a split other than 'val', or a held-out /
                             non-participating client.
    """
    if meta is None:
        raise HoldoutLeakageError("Metadata item cannot be None.")

    # 1. Extract split_name
    split_name: Optional[str] = None
    if isinstance(meta, dict):
        split_name = meta.get("split_name") or meta.get("split")
    else:
        split_name = getattr(meta, "split_name", None) or getattr(meta, "split", None)

    if split_name != "val":
        raise HoldoutLeakageError(
            f"Threshold selection strictly requires split_name='val', got '{split_name}'. "
            f"Training and test partitions must never be used for threshold search."
        )

    # 2. Extract client_name
    client_name: Optional[str] = None
    if isinstance(meta, dict):
        client_name = meta.get("client_name") or meta.get("client")
        is_held_out = bool(meta.get("is_held_out", False))
    else:
        client_name = getattr(meta, "client_name", None) or getattr(meta, "client", None)
        is_held_out = bool(getattr(meta, "is_held_out", False))

    if (
        is_held_out
        or client_name is None
        or client_name in HELDOUT_CLIENTS
        or (isinstance(client_name, str) and client_name.startswith("heldout"))
        or client_name not in PARTICIPATING_CLIENTS
    ):
        raise HoldoutLeakageError(
            f"Metadata client '{client_name}' (held_out={is_held_out}) is not an approved "
            f"participating client ({sorted(PARTICIPATING_CLIENTS)})."
        )


def select_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metadata: Sequence[Any],
    *,
    candidate_thresholds: Optional[Sequence[float]] = None,
) -> ThresholdResult:
    """Select decision threshold maximizing F1 score on validation data in O(N log N) time.

    Validates that all input samples have per-sample metadata belonging to participating clients
    and the 'val' split. Evaluates all unique candidate scores, choosing the highest threshold on ties.

    Args:
        y_true: Ground truth binary targets in {0, 1}.
        y_score: Continuous prediction scores or predicted probabilities.
        metadata: Required per-sample metadata sequence for validation scope check.
        candidate_thresholds: Optional explicit candidate thresholds (defaults to all unique scores).

    Returns:
        ThresholdResult with selected frozen threshold and validation F1 score.

    Raises:
        HoldoutLeakageError: If metadata specifies non-val split or heldout client.
        ValueError: If metadata is missing, inputs are empty, or lengths do not match.
    """
    if metadata is None:
        raise ValueError("metadata is required: threshold selection strictly requires per-sample validation metadata.")

    if not isinstance(metadata, (list, tuple, collections.abc.Sequence)):
        raise ValueError("metadata must be a sequence of per-sample metadata matching y_true length.")

    if len(metadata) != len(y_true):
        raise ValueError(
            f"Metadata length ({len(metadata)}) does not match y_true length ({len(y_true)})."
        )

    # Validate all per-sample metadata upfront
    for m in metadata:
        validate_threshold_metadata(m)

    # Length and value checks
    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} != len(y_score)={len(y_score)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot select threshold on empty validation set.")

    # Convert to pure Python floats/ints
    y_t = [int(y) for y in y_true]
    y_s = [float(s) for s in y_score]

    for y in y_t:
        if y not in (0, 1):
            raise ValueError(f"Ground truth target must be binary integer 0 or 1; got {y}")
    for s in y_s:
        if not math.isfinite(s):
            raise ValueError(f"Prediction score must be finite float; got {s}")

    # Group counts by unique score for O(N log N) cumulative sweep
    score_pos_counts: Dict[float, int] = defaultdict(int)
    score_neg_counts: Dict[float, int] = defaultdict(int)
    for yt, s in zip(y_t, y_s):
        if yt == 1:
            score_pos_counts[s] += 1
        else:
            score_neg_counts[s] += 1

    asc_scores = sorted(score_pos_counts.keys() | score_neg_counts.keys())
    m_scores = len(asc_scores)

    # Prefix sums of positive and negative counts in ascending score order
    # prefix_tp[i] = count of positives with score < asc_scores[i]
    prefix_tp = [0]
    prefix_fp = [0]
    for s in asc_scores:
        prefix_tp.append(prefix_tp[-1] + score_pos_counts[s])
        prefix_fp.append(prefix_fp[-1] + score_neg_counts[s])

    total_pos = prefix_tp[-1]
    total_neg = prefix_fp[-1]

    # Helper to evaluate (tp, fp, fn, f1) for any threshold c under >= semantics
    def _evaluate_at_threshold(c: float) -> float:
        # idx is index of first score >= c
        idx = bisect.bisect_left(asc_scores, c)
        tp = total_pos - prefix_tp[idx]
        fp = total_neg - prefix_fp[idx]
        fn = prefix_tp[idx]

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

    # Determine candidates
    if candidate_thresholds is not None:
        if len(candidate_thresholds) == 0:
            raise ValueError("candidate_thresholds cannot be empty.")
        candidates: List[float] = []
        for c in candidate_thresholds:
            fc = float(c)
            if not math.isfinite(fc):
                raise ValueError(f"Explicit candidate threshold must be finite float; got {c}")
            candidates.append(fc)
        # Sort candidates descending for fast tie-breaking
        sorted_candidates = sorted(set(candidates), reverse=True)
    else:
        # All unique scores descending
        sorted_candidates = sorted(asc_scores, reverse=True)

    best_threshold: Optional[float] = None
    best_f1: Optional[float] = None

    for cand in sorted_candidates:
        f1 = _evaluate_at_threshold(cand)

        if best_threshold is None:
            best_threshold = cand
            best_f1 = f1
        else:
            assert best_f1 is not None
            if f1 > best_f1 and not math.isclose(f1, best_f1, rel_tol=1e-9, abs_tol=1e-9):
                best_f1 = f1
                best_threshold = cand
            elif math.isclose(f1, best_f1, rel_tol=1e-9, abs_tol=1e-9):
                # Tie breaking: choose highest threshold
                if cand > best_threshold:
                    best_threshold = cand
                    best_f1 = f1

    assert best_threshold is not None and best_f1 is not None
    return ThresholdResult(threshold=float(best_threshold), score=float(best_f1), metric="f1")


def select_client_thresholds(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metadata: Sequence[Any],
    *,
    candidate_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, ThresholdResult]:
    """Select optimal per-client thresholds maximizing local validation F1 for each participating client.

    Validates that all samples belong to 'val' split and participating clients.

    Args:
        y_true: Validation ground truth targets.
        y_score: Validation continuous prediction scores.
        metadata: Validation sample metadata containing client_name and split_name.
        candidate_thresholds: Optional candidate thresholds.

    Returns:
        Dictionary mapping client_name -> ThresholdResult for each participating client.
    """
    if metadata is None:
        raise ValueError("metadata is required for client threshold selection.")

    if len(y_true) != len(metadata) or len(y_true) != len(y_score):
        raise ValueError("Lengths of y_true, y_score, and metadata must match.")

    # Validate all metadata upfront
    for m in metadata:
        validate_threshold_metadata(m)

    by_client: Dict[str, Tuple[List[int], List[float], List[Any]]] = {}
    for yt, score, m in zip(y_true, y_score, metadata):
        c_name = getattr(m, "client_name", None) or (m.get("client_name") if isinstance(m, dict) else None)
        if c_name is None:
            raise ValueError(f"Metadata object missing client_name: {m}")
        if c_name not in by_client:
            by_client[c_name] = ([], [], [])
        by_client[c_name][0].append(yt)
        by_client[c_name][1].append(score)
        by_client[c_name][2].append(m)

    results: Dict[str, ThresholdResult] = {}
    for c_name in sorted(by_client):
        c_y_true, c_y_score, c_meta = by_client[c_name]
        results[c_name] = select_threshold(
            c_y_true,
            c_y_score,
            metadata=c_meta,
            candidate_thresholds=candidate_thresholds,
        )

    return results
