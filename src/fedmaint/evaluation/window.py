"""Window-level binary classification evaluation for Phase 3 Azure PdM.

Specification:
- Scikit-learn average_precision_score (primary) and roc_auc_score.
- Precision, recall, F1, and confusion matrix counts (tp, fp, fn, tn) at a frozen decision threshold.
- Returns None for AP and ROC-AUC when a class is absent (single-class subsets), plus class support counts.
- Pure Python numerical fallback for AP and ROC-AUC when scikit-learn is not installed.
- Multi-client reporting: evaluates each participating client, pooled participating test,
  unweighted macro of defined client metrics, and held-out client (model1) separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from sklearn.metrics import average_precision_score as _sklearn_ap
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc
    _SKLEARN_AVAILABLE = True
except ImportError:
    _sklearn_ap = None  # type: ignore
    _sklearn_roc_auc = None  # type: ignore
    _SKLEARN_AVAILABLE = False

from fedmaint.data.adapter import SampleMetadata
from .threshold import HELDOUT_CLIENTS, PARTICIPATING_CLIENTS


@dataclass(frozen=True)
class WindowMetrics:
    """Window-level classification metrics at a frozen decision threshold."""
    average_precision: Optional[float]  # primary metric; None when class absent
    roc_auc: Optional[float]            # None when class absent
    precision: float
    recall: float
    f1: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_pos: int
    n_neg: int
    n_total: int

    @property
    def pr_auc(self) -> Optional[float]:
        """Alias for average_precision (primary metric)."""
        return self.average_precision

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a dictionary."""
        return {
            "average_precision": self.average_precision,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "n_total": self.n_total,
        }


def _pure_python_average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Compute step average precision in pure Python matching scikit-learn definition."""
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0 or n_pos == len(y_true):
        return None

    # Sort descending by score; on tie preserve stable index
    indexed_scores = sorted(enumerate(zip(y_true, y_score)), key=lambda item: item[1][1], reverse=True)

    # Scikit-learn groups identical thresholds
    distinct_thresholds: Dict[float, List[int]] = {}
    for _, (yt, s) in indexed_scores:
        if s not in distinct_thresholds:
            distinct_thresholds[s] = []
        distinct_thresholds[s].append(yt)

    # Walk through distinct thresholds in descending order
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0

    for thresh in sorted(distinct_thresholds.keys(), reverse=True):
        labels_at_thresh = distinct_thresholds[thresh]
        for yt in labels_at_thresh:
            if yt == 1:
                tp += 1
            else:
                fp += 1

        curr_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        curr_recall = tp / n_pos

        delta_recall = curr_recall - prev_recall
        ap += delta_recall * curr_precision
        prev_recall = curr_recall

    return float(ap)


def _pure_python_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Compute Mann-Whitney U rank ROC-AUC in pure Python matching scikit-learn definition."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Rank all scores with average rank for ties
    sorted_items = sorted(enumerate(zip(y_true, y_score)), key=lambda item: item[1][1])
    n = len(sorted_items)

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and sorted_items[j][1][1] == sorted_items[i][1][1]:
            j += 1
        # Assign average 1-based rank to tie group [i, j-1]
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_items[k][0]] = avg_rank
        i = j

    rank_sum_pos = sum(r for r, y in zip(ranks, y_true) if y == 1)
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def evaluate_window_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> WindowMetrics:
    """Compute window-level classification metrics at a frozen decision threshold.

    Args:
        y_true: Ground truth binary targets in {0, 1}.
        y_score: Continuous prediction scores or predicted probabilities.
        threshold: Frozen decision threshold for binary classification.

    Returns:
        WindowMetrics dataclass instance.
    """
    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_score)={len(y_score)}")

    n_pos = 0
    n_neg = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    y_t: List[int] = []
    y_s: List[float] = []

    for y, s in zip(y_true, y_score):
        yt = int(y)
        fs = float(s)
        if yt not in (0, 1):
            raise ValueError(f"Ground truth target must be binary integer 0 or 1; got {yt}")
        if not math.isfinite(fs):
            raise ValueError(f"Prediction score must be finite float; got {fs}")

        y_t.append(yt)
        y_s.append(fs)

        if yt == 1:
            n_pos += 1
        else:
            n_neg += 1

        pred = 1 if fs >= threshold else 0
        if yt == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                tn += 1

    n_total = n_pos + n_neg

    # Precision, recall, and F1 at frozen threshold
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

    # AP and ROC-AUC are undefined (None) when either class is absent
    if n_pos == 0 or n_neg == 0:
        ap: Optional[float] = None
        roc_auc: Optional[float] = None
    else:
        if _SKLEARN_AVAILABLE:
            try:
                ap = float(_sklearn_ap(y_t, y_s))
            except (ValueError, TypeError):
                ap = _pure_python_average_precision(y_t, y_s)
            try:
                roc_auc = float(_sklearn_roc_auc(y_t, y_s))
            except (ValueError, TypeError):
                roc_auc = _pure_python_roc_auc(y_t, y_s)
        else:
            ap = _pure_python_average_precision(y_t, y_s)
            roc_auc = _pure_python_roc_auc(y_t, y_s)

    return WindowMetrics(
        average_precision=ap,
        roc_auc=roc_auc,
        precision=prec,
        recall=rec,
        f1=f1,
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_pos=n_pos,
        n_neg=n_neg,
        n_total=n_total,
    )


@dataclass(frozen=True)
class WindowReport:
    """Multi-client window evaluation report separating participating from held-out clients."""
    clients: Dict[str, WindowMetrics]
    pooled_participating: Optional[WindowMetrics]
    macro_participating: Dict[str, Optional[float]]
    heldout: Dict[str, WindowMetrics]
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to a dictionary."""
        return {
            "clients": {k: v.to_dict() for k, v in self.clients.items()},
            "pooled_participating": self.pooled_participating.to_dict() if self.pooled_participating else None,
            "macro_participating": self.macro_participating,
            "heldout": {k: v.to_dict() for k, v in self.heldout.items()},
            "threshold": self.threshold,
        }


def evaluate_window_report(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metadata: Sequence[Any],
    threshold: float,
    *,
    split_filter: Optional[str] = "test",
) -> WindowReport:
    """Compute comprehensive window-level evaluation report across federated clients.

    Reports:
      1. Each participating client separately.
      2. Pooled participating clients test metrics.
      3. Unweighted macro average of defined client metrics across participating clients.
      4. Held-out client (heldout_model1) evaluated completely separately on its test partition.

    Args:
        y_true: Ground truth targets.
        y_score: Continuous prediction scores.
        metadata: Sample metadata sequence with client_name and split_name.
        threshold: Frozen decision threshold.
        split_filter: Split to evaluate on (default 'test'; None evaluates all provided samples).

    Returns:
        WindowReport dataclass instance.
    """
    if len(y_true) != len(y_score) or len(y_true) != len(metadata):
        raise ValueError("Lengths of y_true, y_score, and metadata must match.")

    by_client: Dict[str, Tuple[List[int], List[float]]] = {}
    for yt, score, m in zip(y_true, y_score, metadata):
        s_name = getattr(m, "split_name", None) or (m.get("split_name") if isinstance(m, dict) else None)
        if split_filter is not None and s_name != split_filter:
            continue

        c_name = getattr(m, "client_name", None) or (m.get("client_name") if isinstance(m, dict) else None)
        if c_name is None:
            raise ValueError(f"Metadata object missing client_name: {m}")

        if c_name not in by_client:
            by_client[c_name] = ([], [])
        by_client[c_name][0].append(int(yt))
        by_client[c_name][1].append(float(score))

    participating_metrics: Dict[str, WindowMetrics] = {}
    heldout_metrics: Dict[str, WindowMetrics] = {}

    pooled_true: List[int] = []
    pooled_score: List[float] = []

    for c_name in sorted(by_client):
        c_yt, c_score = by_client[c_name]
        is_held_out = c_name in HELDOUT_CLIENTS or c_name.startswith("heldout")
        wm = evaluate_window_metrics(c_yt, c_score, threshold=threshold)

        if is_held_out:
            heldout_metrics[c_name] = wm
        elif c_name in PARTICIPATING_CLIENTS:
            participating_metrics[c_name] = wm
            pooled_true.extend(c_yt)
            pooled_score.extend(c_score)
        else:
            raise ValueError(
                f"Unknown or unauthorized client '{c_name}'. Approved participating clients: "
                f"{sorted(PARTICIPATING_CLIENTS)}; held-out clients: {sorted(HELDOUT_CLIENTS)}."
            )

    pooled_wm = evaluate_window_metrics(pooled_true, pooled_score, threshold=threshold) if pooled_true else None

    # Macro unweighted average across participating clients for defined metrics only
    macro_metrics: Dict[str, Optional[float]] = {}
    for metric_name in ("average_precision", "roc_auc", "precision", "recall", "f1"):
        values = []
        for wm in participating_metrics.values():
            val = getattr(wm, metric_name)
            if val is not None and math.isfinite(val):
                values.append(val)
        macro_metrics[metric_name] = (sum(values) / len(values)) if values else None

    return WindowReport(
        clients=participating_metrics,
        pooled_participating=pooled_wm,
        macro_participating=macro_metrics,
        heldout=heldout_metrics,
        threshold=float(threshold),
    )
