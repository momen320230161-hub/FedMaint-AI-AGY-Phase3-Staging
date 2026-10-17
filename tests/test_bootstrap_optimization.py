"""Synthetic equivalence and unit tests for optimized machine-cluster bootstrap.

Verifies:
- Numerical equivalence against an independent brute-force resampling reference
  (which duplicates all machine windows and invokes evaluate_window_metrics per draw).
- Exact agreement across all metrics: average_precision, roc_auc, precision, recall, f1.
- Exact agreement on point_estimate, ci_lower, ci_upper, num_draws, and num_valid_draws.
- Respect of tied continuous scores (within-class and cross-class ties).
- Handling of class imbalance and single-class bootstrap draws (where AP/ROC are undefined).
- Exact behavior when valid draws fall below min_valid_draws.
- Pure-Python fallback numerical equivalence when NumPy is unavailable.
- Deterministic draw sequence with random seeds.
- All tests run completely data-free using synthetic in-memory samples.
"""
from __future__ import annotations

import collections
import math
from pathlib import Path
import random
import sys
import unittest
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fedmaint.evaluation.bootstrap import (
    BootstrapCI,
    _compute_ci_bounds,
    bootstrap_window_ci,
)
import fedmaint.evaluation.bootstrap as bootstrap_mod
from fedmaint.evaluation.window import evaluate_window_metrics


def brute_force_bootstrap_window_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metadata: Sequence[Any],
    threshold: float = 0.5,
    *,
    num_draws: int = 100,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_valid_draws: int = 10,
) -> Dict[str, BootstrapCI]:
    """Independent brute-force reference implementation.

    Explicitly duplicates machine windows across resampled clusters and computes
    metrics by invoking evaluate_window_metrics on the combined arrays.
    """
    extracted_m_ids: List[int] = []
    extracted_y: List[int] = []
    extracted_s: List[float] = []

    for yt, s, m in zip(y_true, y_score, metadata):
        m_id = getattr(m, "machine_id", None) or (m.get("machine_id") if isinstance(m, dict) else None)
        if m_id is None:
            raise ValueError(f"Missing machine_id: {m}")
        extracted_m_ids.append(int(m_id))
        extracted_y.append(int(yt))
        extracted_s.append(float(s))

    machine_ids = sorted(set(extracted_m_ids))
    point_wm = evaluate_window_metrics(y_true, y_score, threshold=threshold)
    metric_names = ("average_precision", "roc_auc", "precision", "recall", "f1")

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

    machine_indices = collections.defaultdict(list)
    for idx, m_id in enumerate(extracted_m_ids):
        machine_indices[m_id].append(idx)

    rng = random.Random(seed)
    n_machines = len(machine_ids)

    draw_metrics: Dict[str, List[Optional[float]]] = {name: [] for name in metric_names}

    for _ in range(num_draws):
        sampled_machines = [rng.choice(machine_ids) for _ in range(n_machines)]
        boot_y: List[int] = []
        boot_s: List[float] = []

        for m_id in sampled_machines:
            for idx in machine_indices[m_id]:
                boot_y.append(extracted_y[idx])
                boot_s.append(extracted_s[idx])

        wm = evaluate_window_metrics(boot_y, boot_s, threshold=threshold)
        draw_metrics["average_precision"].append(wm.average_precision)
        draw_metrics["roc_auc"].append(wm.roc_auc)
        draw_metrics["precision"].append(wm.precision)
        draw_metrics["recall"].append(wm.recall)
        draw_metrics["f1"].append(wm.f1)

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


class BootstrapOptimizationEquivalenceTests(unittest.TestCase):
    """Test suite comparing optimized bootstrap_window_ci with brute-force reference."""

    def setUp(self) -> None:
        # Construct synthetic multi-machine dataset with tied scores and class imbalance
        # 5 machines with varying sizes and distributions
        self.y_true: List[int] = []
        self.y_score: List[float] = []
        self.metadata: List[Dict[str, int]] = []

        # Machine 1: 30 windows, 5 positives, contains tied scores
        m1_scores = [0.90, 0.85, 0.85, 0.70, 0.65] + [0.40] * 5 + [0.25] * 10 + [0.10] * 10
        m1_labels = [1, 1, 0, 1, 0] + [1] * 2 + [0] * 3 + [0] * 20
        for yt, s in zip(m1_labels, m1_scores):
            self.y_true.append(yt)
            self.y_score.append(s)
            self.metadata.append({"machine_id": 101})

        # Machine 2: 20 windows, 2 positives, ties with machine 1 scores (0.85, 0.40, 0.10)
        m2_scores = [0.85, 0.75, 0.60] + [0.40] * 7 + [0.10] * 10
        m2_labels = [1, 0, 1] + [0] * 17
        for yt, s in zip(m2_labels, m2_scores):
            self.y_true.append(yt)
            self.y_score.append(s)
            self.metadata.append({"machine_id": 102})

        # Machine 3: 40 windows, 6 positives, high and low scores
        m3_scores = [0.95, 0.90, 0.80, 0.70, 0.70, 0.55] + [0.35] * 14 + [0.05] * 20
        m3_labels = [1, 1, 1, 0, 1, 1] + [1] * 1 + [0] * 13 + [0] * 20
        for yt, s in zip(m3_labels, m3_scores):
            self.y_true.append(yt)
            self.y_score.append(s)
            self.metadata.append({"machine_id": 103})

        # Machine 4: 25 windows, 1 positive (sparse)
        m4_scores = [0.65] + [0.30] * 12 + [0.15] * 12
        m4_labels = [1] + [0] * 24
        for yt, s in zip(m4_labels, m4_scores):
            self.y_true.append(yt)
            self.y_score.append(s)
            self.metadata.append({"machine_id": 104})

        # Machine 5: 15 windows, 0 positives (purely negative machine)
        m5_scores = [0.45] * 5 + [0.20] * 5 + [0.05] * 5
        m5_labels = [0] * 15
        for yt, s in zip(m5_labels, m5_scores):
            self.y_true.append(yt)
            self.y_score.append(s)
            self.metadata.append({"machine_id": 105})

    def _assert_results_equivalent(
        self,
        actual: Dict[str, BootstrapCI],
        expected: Dict[str, BootstrapCI],
        places: int = 6,
    ) -> None:
        """Helper to assert exact or near-identical equivalence between bootstrap outputs."""
        self.assertEqual(set(actual.keys()), set(expected.keys()))

        for name in actual:
            act = actual[name]
            exp = expected[name]

            self.assertEqual(act.metric_name, exp.metric_name)
            self.assertEqual(act.num_draws, exp.num_draws)
            self.assertEqual(act.num_valid_draws, exp.num_valid_draws)

            if exp.point_estimate is None:
                self.assertIsNone(act.point_estimate)
            else:
                self.assertIsNotNone(act.point_estimate)
                assert act.point_estimate is not None
                self.assertAlmostEqual(act.point_estimate, exp.point_estimate, places=places)

            if exp.ci_lower is None:
                self.assertIsNone(act.ci_lower)
            else:
                self.assertIsNotNone(act.ci_lower)
                assert act.ci_lower is not None
                self.assertAlmostEqual(act.ci_lower, exp.ci_lower, places=places)

            if exp.ci_upper is None:
                self.assertIsNone(act.ci_upper)
            else:
                self.assertIsNotNone(act.ci_upper)
                assert act.ci_upper is not None
                self.assertAlmostEqual(act.ci_upper, exp.ci_upper, places=places)

    def test_equivalence_vectorized_numpy_path(self) -> None:
        """Verify optimized NumPy path produces equivalent results to brute force with ties."""
        num_draws = 40
        seed = 42
        threshold = 0.50

        expected = brute_force_bootstrap_window_ci(
            self.y_true,
            self.y_score,
            self.metadata,
            threshold=threshold,
            num_draws=num_draws,
            seed=seed,
            min_valid_draws=5,
        )

        actual = bootstrap_window_ci(
            self.y_true,
            self.y_score,
            self.metadata,
            threshold=threshold,
            num_draws=num_draws,
            seed=seed,
            min_valid_draws=5,
        )

        self._assert_results_equivalent(actual, expected)

    def test_equivalence_pure_python_fallback_path(self) -> None:
        """Verify pure-Python fallback path produces equivalent results to brute force."""
        num_draws = 40
        seed = 42
        threshold = 0.50

        expected = brute_force_bootstrap_window_ci(
            self.y_true,
            self.y_score,
            self.metadata,
            threshold=threshold,
            num_draws=num_draws,
            seed=seed,
            min_valid_draws=5,
        )

        orig_has_numpy = bootstrap_mod._HAS_NUMPY
        try:
            bootstrap_mod._HAS_NUMPY = False
            actual = bootstrap_window_ci(
                self.y_true,
                self.y_score,
                self.metadata,
                threshold=threshold,
                num_draws=num_draws,
                seed=seed,
                min_valid_draws=5,
            )
        finally:
            bootstrap_mod._HAS_NUMPY = orig_has_numpy

        self._assert_results_equivalent(actual, expected)

    def test_sparse_positives_and_single_class_draws(self) -> None:
        """Verify exact handling when resampled machine draws contain only 1 class."""
        # Only machine 1 has 1 positive; machines 2 and 3 have 0 positives
        y_true = [1] + [0] * 9 + [0] * 10 + [0] * 10
        y_score = [0.8] + [0.2] * 9 + [0.3] * 10 + [0.1] * 10
        metadata = [{"machine_id": 1}] * 10 + [{"machine_id": 2}] * 10 + [{"machine_id": 3}] * 10

        num_draws = 30
        seed = 99
        # Many draws will only pick machines 2 and 3, yielding 0 positives (single-class)
        expected = brute_force_bootstrap_window_ci(
            y_true,
            y_score,
            metadata,
            threshold=0.5,
            num_draws=num_draws,
            seed=seed,
            min_valid_draws=10,
        )

        actual = bootstrap_window_ci(
            y_true,
            y_score,
            metadata,
            threshold=0.5,
            num_draws=num_draws,
            seed=seed,
            min_valid_draws=10,
        )

        self._assert_results_equivalent(actual, expected)

    def test_insufficient_machines_boundary(self) -> None:
        """Verify single-machine partition returns None bounds and 0 valid draws."""
        y_true = [1, 0, 1, 0]
        y_score = [0.8, 0.2, 0.7, 0.3]
        metadata = [{"machine_id": 10}] * 4

        actual = bootstrap_window_ci(y_true, y_score, metadata, num_draws=20)
        for ci in actual.values():
            self.assertIsNone(ci.ci_lower)
            self.assertIsNone(ci.ci_upper)
            self.assertEqual(ci.num_valid_draws, 0)

    def test_seed_determinism(self) -> None:
        """Verify identical seed produces bit-identical outputs and different seeds differ."""
        r1 = bootstrap_window_ci(self.y_true, self.y_score, self.metadata, num_draws=30, seed=777)
        r2 = bootstrap_window_ci(self.y_true, self.y_score, self.metadata, num_draws=30, seed=777)
        r3 = bootstrap_window_ci(self.y_true, self.y_score, self.metadata, num_draws=30, seed=888)

        for name in r1:
            self.assertEqual(r1[name].ci_lower, r2[name].ci_lower)
            self.assertEqual(r1[name].ci_upper, r2[name].ci_upper)

        # Different seed should yield different bounds
        has_diff = any(r1[name].ci_lower != r3[name].ci_lower for name in r1 if r1[name].ci_lower is not None)
        self.assertTrue(has_diff)


if __name__ == "__main__":
    unittest.main()
