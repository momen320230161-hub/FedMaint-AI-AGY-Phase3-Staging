"""Comprehensive synthetic unit tests for Phase 3 Azure PdM evaluation module.

Tests verify:
- Exact AP, ROC-AUC, precision, recall, F1, and confusion matrix counts (tp, fp, fn, tn).
- Single-class subsets return None for AP and ROC-AUC, plus accurate class support counts.
- Threshold selection requires per-sample validation metadata; missing metadata raises ValueError.
- Strict threshold validation split scope: rejects non-val splits and heldout client (model1) with HoldoutLeakageError.
- Threshold F1 maximization and deterministic highest-threshold choice on ties.
- Equivalence of O(N log N) threshold sweep against brute-force search on fixtures with equal scores.
- Event evaluation requires explicit cohort machine_ids and explicit split bounds/name.
- Event 24h horizon exact boundary behavior: open on left (t, ...], closed on right (..., t+24h].
- Strict Phase 2B boundary rule: t - 48h >= split.start AND t < split.end AND t + 24h < split.end (STRICT).
- Predictions at exact split end (t + 24h == split.end) strictly excluded.
- Cohort scoping: zero-opportunity cohort machines retain all episodes as ineligible (not dropped).
- Cross-cohort contamination rejected with ValueError; mismatched split metadata rejected with HoldoutLeakageError.
- Alert generation 24-hour suppression cooldown preventing overlapping alert floods.
- Simultaneous failure records collapsed into a single episode matching at most one alert.
- Multiple close episodes and greedy chronological one-to-one alert-episode matching.
- Ineligible/unobservable episode attribution (outside observable horizon or prior to 48h history).
- Zero-denominator handling for event recall, false alert rates, and lead hours.
- Deterministic machine-cluster bootstrap uncertainty estimation with cluster multiplicity preservation.
- Insufficient data (< 2 machines) returning None for bootstrap confidence intervals.
- Multi-client window evaluation reporting: participating pooled, unweighted macro, heldout isolation,
  and unknown client rejection with ValueError.
"""
from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.adapter import SampleMetadata
from fedmaint.data.episodes import FailureEpisode
from fedmaint.data.splits import HoldoutLeakageError, SplitBounds
from fedmaint.evaluation import (
    BootstrapCI,
    EventMatch,
    EventMetrics,
    ThresholdResult,
    WindowMetrics,
    WindowReport,
    bootstrap_event_ci,
    bootstrap_window_ci,
    evaluate_event_metrics,
    evaluate_window_metrics,
    evaluate_window_report,
    select_client_thresholds,
    select_threshold,
    validate_threshold_metadata,
)


def _make_val_meta(
    machine_id: int = 1,
    client_name: str = "client_model2",
    prediction_time: datetime = datetime(2015, 9, 10, 6),
) -> SampleMetadata:
    """Helper to construct a valid validation metadata item."""
    return SampleMetadata(
        machine_id=machine_id,
        prediction_time=prediction_time,
        client_name=client_name,
        split_name="val",
        observation_start=prediction_time - timedelta(hours=48),
        observation_end=prediction_time - timedelta(hours=1),
        horizon_start=prediction_time,
        horizon_end=prediction_time + timedelta(hours=24),
    )


class WindowMetricsTests(unittest.TestCase):
    """Synthetic unit tests for window-level binary classification metrics."""

    def test_exact_window_metrics(self):
        """Verify exact precision, recall, F1, AP, ROC-AUC, and confusion counts."""
        y_true = [1, 1, 0, 0, 1, 0]
        y_score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
        threshold = 0.5

        # At threshold 0.5:
        # Predictions: [1, 1, 1, 0, 0, 0]
        # TP = 2 (indices 0, 1), FP = 1 (index 2), FN = 1 (index 4), TN = 2 (indices 3, 5)
        wm = evaluate_window_metrics(y_true, y_score, threshold=threshold)

        self.assertEqual(wm.tp, 2)
        self.assertEqual(wm.fp, 1)
        self.assertEqual(wm.fn, 1)
        self.assertEqual(wm.tn, 2)
        self.assertEqual(wm.n_pos, 3)
        self.assertEqual(wm.n_neg, 3)
        self.assertEqual(wm.n_total, 6)

        expected_prec = 2.0 / 3.0
        expected_rec = 2.0 / 3.0
        expected_f1 = 2.0 / 3.0

        self.assertAlmostEqual(wm.precision, expected_prec, places=6)
        self.assertAlmostEqual(wm.recall, expected_rec, places=6)
        self.assertAlmostEqual(wm.f1, expected_f1, places=6)

        self.assertIsNotNone(wm.average_precision)
        self.assertIsNotNone(wm.roc_auc)
        self.assertAlmostEqual(wm.pr_auc, wm.average_precision)
        self.assertAlmostEqual(wm.roc_auc, 7.0 / 9.0, places=5)

    def test_one_class_subsets_return_none_and_support_counts(self):
        """Verify undefined (None) AP and ROC-AUC when a class is absent, preserving >= semantics."""
        # 1. All-zero targets
        # Under >= 0.5 semantics, scores [0.1, 0.3, 0.5, 0.7, 0.9] yield:
        # 0.5, 0.7, 0.9 are >= 0.5 -> FP = 3
        # 0.1, 0.3 are < 0.5 -> TN = 2
        y_all_zeros = [0, 0, 0, 0, 0]
        scores_zeros = [0.1, 0.3, 0.5, 0.7, 0.9]

        wm_zeros = evaluate_window_metrics(y_all_zeros, scores_zeros, threshold=0.5)
        self.assertIsNone(wm_zeros.average_precision)
        self.assertIsNone(wm_zeros.roc_auc)
        self.assertIsNone(wm_zeros.pr_auc)
        self.assertEqual(wm_zeros.n_pos, 0)
        self.assertEqual(wm_zeros.n_neg, 5)
        self.assertEqual(wm_zeros.n_total, 5)
        self.assertEqual(wm_zeros.tp, 0)
        self.assertEqual(wm_zeros.fn, 0)
        self.assertEqual(wm_zeros.fp, 3)
        self.assertEqual(wm_zeros.tn, 2)

        # 2. All-one targets
        # Under >= 0.5 semantics, scores [0.2, 0.4, 0.6, 0.8] yield:
        # 0.6, 0.8 are >= 0.5 -> TP = 2
        # 0.2, 0.4 are < 0.5 -> FN = 2
        y_all_ones = [1, 1, 1, 1]
        scores_ones = [0.2, 0.4, 0.6, 0.8]

        wm_ones = evaluate_window_metrics(y_all_ones, scores_ones, threshold=0.5)
        self.assertIsNone(wm_ones.average_precision)
        self.assertIsNone(wm_ones.roc_auc)
        self.assertEqual(wm_ones.n_pos, 4)
        self.assertEqual(wm_ones.n_neg, 0)
        self.assertEqual(wm_ones.n_total, 4)
        self.assertEqual(wm_ones.tp, 2)
        self.assertEqual(wm_ones.fn, 2)
        self.assertEqual(wm_ones.fp, 0)
        self.assertEqual(wm_ones.tn, 0)


class ThresholdSelectionTests(unittest.TestCase):
    """Synthetic unit tests for validation threshold selection, metadata enforcement, and leakage guardrails."""

    def test_missing_metadata_rejected(self):
        """Calling select_threshold without metadata must raise ValueError."""
        with self.assertRaises(ValueError):
            select_threshold([1, 0], [0.8, 0.2], metadata=None)  # type: ignore

    def test_metadata_validation_rejects_non_val_splits(self):
        """Threshold search must strictly reject training and test splits with HoldoutLeakageError."""
        train_meta = SampleMetadata(
            machine_id=1,
            prediction_time=datetime(2015, 3, 1, 6),
            client_name="client_model2",
            split_name="train",
            observation_start=datetime(2015, 2, 27, 6),
            observation_end=datetime(2015, 3, 1, 5),
            horizon_start=datetime(2015, 3, 1, 6),
            horizon_end=datetime(2015, 3, 2, 6),
        )
        with self.assertRaises(HoldoutLeakageError):
            validate_threshold_metadata(train_meta)

        with self.assertRaises(HoldoutLeakageError):
            select_threshold([1], [0.8], metadata=[train_meta])

        test_meta = SampleMetadata(
            machine_id=1,
            prediction_time=datetime(2015, 11, 10, 6),
            client_name="client_model2",
            split_name="test",
            observation_start=datetime(2015, 11, 8, 6),
            observation_end=datetime(2015, 11, 10, 5),
            horizon_start=datetime(2015, 11, 10, 6),
            horizon_end=datetime(2015, 11, 11, 6),
        )
        with self.assertRaises(HoldoutLeakageError):
            validate_threshold_metadata(test_meta)

        with self.assertRaises(HoldoutLeakageError):
            select_threshold([1], [0.8], metadata=[test_meta])

    def test_metadata_validation_rejects_heldout_client(self):
        """Threshold search must strictly reject held-out client (model1) data with HoldoutLeakageError."""
        heldout_meta = SampleMetadata(
            machine_id=99,
            prediction_time=datetime(2015, 9, 10, 6),
            client_name="heldout_model1",
            split_name="val",
            observation_start=datetime(2015, 9, 8, 6),
            observation_end=datetime(2015, 9, 10, 5),
            horizon_start=datetime(2015, 9, 10, 6),
            horizon_end=datetime(2015, 9, 11, 6),
        )
        with self.assertRaises(HoldoutLeakageError):
            validate_threshold_metadata(heldout_meta)

        with self.assertRaises(HoldoutLeakageError):
            select_threshold([1], [0.8], metadata=[heldout_meta])

    def test_metadata_validation_accepts_participating_val(self):
        """Threshold search accepts participating clients in val split."""
        val_meta = _make_val_meta(machine_id=20, client_name="client_model3")
        validate_threshold_metadata(val_meta)

    def test_threshold_f1_maximization_and_tie_breaking(self):
        """Verify F1 maximization over unique scores and highest-threshold selection on ties."""
        y_true = [1, 0, 1, 0]
        y_score = [0.8, 0.6, 0.4, 0.2]
        meta = [_make_val_meta(i + 1) for i in range(4)]

        # At 0.8: TP=1, FP=0, FN=1 -> F1 ≈ 0.6667
        # At 0.6: TP=1, FP=1, FN=1 -> F1 = 0.5
        # At 0.4: TP=2, FP=1, FN=0 -> F1 = 4/5 = 0.8 (optimal)
        # At 0.2: TP=2, FP=2, FN=0 -> F1 ≈ 0.6667
        res = select_threshold(y_true, y_score, metadata=meta)
        self.assertAlmostEqual(res.threshold, 0.4)
        self.assertAlmostEqual(res.score, 0.8)

        # Tie breaking test:
        # Both 0.3 and 0.5 produce TP=2, FP=1, FN=0 -> F1 = 0.8.
        # Tie rule: choose HIGHEST candidate threshold (0.5)
        y_tie = [1, 1, 0, 0]
        scores_tie = [0.9, 0.8, 0.35, 0.6]
        meta_tie = [_make_val_meta(i + 1) for i in range(4)]
        res_tie = select_threshold(
            y_tie,
            scores_tie,
            metadata=meta_tie,
            candidate_thresholds=[0.3, 0.5],
        )
        self.assertEqual(res_tie.threshold, 0.5)
        self.assertAlmostEqual(res_tie.score, 0.8)

    def test_o_n_log_n_sweep_matches_brute_force_equivalence(self):
        """Verify O(N log N) cumulative sweep matches brute force search on fixtures with equal scores."""
        def _brute_force_select(yt, ys):
            unique_candidates = sorted(set(ys), reverse=True)
            best_t = None
            best_f1 = None
            for c in unique_candidates:
                tp = sum(1 for y, s in zip(yt, ys) if s >= c and y == 1)
                fp = sum(1 for y, s in zip(yt, ys) if s >= c and y == 0)
                fn = sum(1 for y, s in zip(yt, ys) if s < c and y == 1)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

                if best_t is None:
                    best_t = c
                    best_f1 = f
                else:
                    if f > best_f1 and not math.isclose(f, best_f1, rel_tol=1e-9, abs_tol=1e-9):
                        best_f1 = f
                        best_t = c
                    elif math.isclose(f, best_f1, rel_tol=1e-9, abs_tol=1e-9):
                        if c > best_t:
                            best_t = c
                            best_f1 = f
            return best_t, best_f1

        # Fixture with repeated scores and varying balances
        test_cases = [
            ([1, 1, 0, 0, 1, 0, 1], [0.8, 0.8, 0.5, 0.5, 0.5, 0.2, 0.1]),
            ([0, 0, 0, 1, 1], [0.9, 0.9, 0.7, 0.7, 0.3]),
            ([1, 0, 1, 0, 1, 0], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
            ([1, 0, 1, 1, 0, 0, 1], [0.95, 0.75, 0.75, 0.6, 0.6, 0.3, 0.1]),
        ]

        for yt, ys in test_cases:
            meta = [_make_val_meta(i + 1) for i in range(len(yt))]
            fast_res = select_threshold(yt, ys, metadata=meta)
            bf_t, bf_f1 = _brute_force_select(yt, ys)

            self.assertAlmostEqual(fast_res.threshold, bf_t, places=7)
            self.assertAlmostEqual(fast_res.score, bf_f1, places=7)


class EventEvaluationTests(unittest.TestCase):
    """Synthetic unit tests for event-level evaluation, alert cooldown, and matching."""

    def setUp(self):
        self.split_start = datetime(2015, 11, 1, 6, 0, 0)
        self.split_end = datetime(2016, 1, 1, 6, 0, 0)
        self.split = SplitBounds(name="test", start=self.split_start, end=self.split_end)
        self.cohort = [1]

    def test_missing_machine_ids_or_split_raises_error(self):
        """Calling evaluate_event_metrics without machine_ids or split raises ValueError."""
        # Missing machine_ids
        with self.assertRaises(ValueError):
            evaluate_event_metrics(episodes=[], predictions=[], machine_ids=[], split=self.split)

        # Missing split and split_name
        with self.assertRaises(ValueError):
            evaluate_event_metrics(episodes=[], predictions=[], machine_ids=[1])

    def test_cross_cohort_contamination_and_mismatched_split_rejection(self):
        """Predictions outside cohort or with mismatched split metadata must be rejected."""
        # Prediction machineID 99 outside cohort [1, 2]
        with self.assertRaises(ValueError):
            evaluate_event_metrics(
                episodes=[],
                predictions=[(99, datetime(2015, 11, 10, 6), 0.9)],
                machine_ids=[1, 2],
                split=self.split,
            )

        # Prediction record with split_name="val" evaluated on test split
        bad_split_meta = SampleMetadata(
            machine_id=1,
            prediction_time=datetime(2015, 11, 10, 6),
            client_name="client_model2",
            split_name="val",  # Wrong split
            observation_start=datetime(2015, 11, 8, 6),
            observation_end=datetime(2015, 11, 10, 5),
            horizon_start=datetime(2015, 11, 10, 6),
            horizon_end=datetime(2015, 11, 11, 6),
        )
        with self.assertRaises(HoldoutLeakageError):
            evaluate_event_metrics(
                episodes=[],
                predictions=[bad_split_meta],
                machine_ids=[1],
                scores=[0.9],
                split=self.split,
            )

    def test_event_24h_horizon_open_closed_endpoints(self):
        """Verify (t, t+24h] open at t, closed at t+24h, and exact lead hours."""
        t_alert = datetime(2015, 11, 10, 10, 0, 0)

        # 1. Episode at exact t_alert: NOT in (t_alert, t_alert+24h] -> False alert
        ep_at_t = FailureEpisode(machine_id=1, timestamp=t_alert, components=("comp1",))
        m1 = evaluate_event_metrics(
            episodes=[ep_at_t],
            predictions=[(1, t_alert, 0.9)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertEqual(m1.detected_episodes, 0)
        self.assertEqual(m1.true_alerts, 0)
        self.assertEqual(m1.false_alerts, 1)

        # 2. Episode at t_alert + 1 hour: strictly inside -> True alert, lead = 1.0 hour
        ep_plus_1h = FailureEpisode(machine_id=1, timestamp=t_alert + timedelta(hours=1), components=("comp1",))
        m2 = evaluate_event_metrics(
            episodes=[ep_plus_1h],
            predictions=[(1, t_alert, 0.9)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertEqual(m2.detected_episodes, 1)
        self.assertEqual(m2.true_alerts, 1)
        self.assertEqual(m2.false_alerts, 0)
        self.assertEqual(m2.lead_hours, (1.0,))
        self.assertEqual(m2.mean_lead_hours, 1.0)

        # 3. Episode at exact right boundary t_alert + 24 hours: inside -> True alert, lead = 24.0 hours
        ep_at_24h = FailureEpisode(machine_id=1, timestamp=t_alert + timedelta(hours=24), components=("comp1",))
        m3 = evaluate_event_metrics(
            episodes=[ep_at_24h],
            predictions=[(1, t_alert, 0.9)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertEqual(m3.detected_episodes, 1)
        self.assertEqual(m3.true_alerts, 1)
        self.assertEqual(m3.lead_hours, (24.0,))

        # 4. Episode at t_alert + 24h + 1s: outside -> False alert
        ep_after_24h = FailureEpisode(
            machine_id=1, timestamp=t_alert + timedelta(hours=24, seconds=1), components=("comp1",)
        )
        m4 = evaluate_event_metrics(
            episodes=[ep_after_24h],
            predictions=[(1, t_alert, 0.9)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertEqual(m4.detected_episodes, 0)
        self.assertEqual(m4.true_alerts, 0)
        self.assertEqual(m4.false_alerts, 1)

    def test_strict_phase2b_boundary_condition(self):
        """Verify strict rule: t-48h >= split.start AND t < split.end AND t+24h < split.end (STRICT)."""
        # A prediction opportunity where t + 24h == split_end is strictly invalid
        exact_end_t = self.split_end - timedelta(hours=24)
        m_strict = evaluate_event_metrics(
            episodes=[],
            predictions=[(1, exact_end_t, 0.9)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        # Because t + 24h == split_end, it is not strictly < split_end, so 0 opportunities accepted
        self.assertEqual(m_strict.eligible_machine_days, 0.0)
        self.assertEqual(m_strict.total_alerts, 0)

        # A prediction opportunity 1 hour earlier satisfies strict t + 24h < split_end
        valid_t = self.split_end - timedelta(hours=25)
        m_valid = evaluate_event_metrics(
            episodes=[],
            predictions=[(1, valid_t, 0.9)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertAlmostEqual(m_valid.eligible_machine_days, 1.0 / 24.0)
        self.assertEqual(m_valid.total_alerts, 1)

    def test_zero_opportunity_machine_in_cohort_retains_episodes_as_ineligible(self):
        """Machines in cohort with zero opportunities must not drop their episodes; counted as ineligible."""
        # Cohort has Machine 1 (has predictions) and Machine 2 (zero valid prediction opportunities)
        cohort = [1, 2]
        pred_t = datetime(2015, 11, 10, 6)
        ep1 = FailureEpisode(machine_id=1, timestamp=pred_t + timedelta(hours=5), components=("comp1",))
        ep2 = FailureEpisode(machine_id=2, timestamp=datetime(2015, 11, 15, 12), components=("comp3",))

        # Only Machine 1 has a prediction opportunity
        preds = [(1, pred_t, 0.9)]

        metrics = evaluate_event_metrics(
            episodes=[ep1, ep2],
            predictions=preds,
            machine_ids=cohort,
            threshold=0.5,
            split=self.split,
        )

        # Total episodes in split = 2 (ep1 for machine 1, ep2 for machine 2)
        self.assertEqual(metrics.total_episodes, 2)
        # Machine 1 has an opportunity covering ep1 -> eligible = 1
        self.assertEqual(metrics.eligible_episodes, 1)
        # Machine 2 has ZERO opportunities -> ep2 is retained as ineligible = 1
        self.assertEqual(metrics.ineligible_episodes, 1)
        self.assertEqual(metrics.unobservable_episodes, 1)
        # Detected: ep1 is detected
        self.assertEqual(metrics.detected_episodes, 1)
        self.assertAlmostEqual(metrics.event_recall, 1.0)

    def test_alert_suppression_cooldown_prevents_flood(self):
        """Positive scores in [t, t+24h) are suppressed after an emitted alert."""
        start_t = datetime(2015, 11, 10, 0, 0, 0)
        preds = [(1, start_t + timedelta(hours=i), 0.9) for i in range(48)]
        ep = FailureEpisode(machine_id=1, timestamp=start_t + timedelta(hours=10), components=("comp2",))

        metrics = evaluate_event_metrics(
            episodes=[ep],
            predictions=preds,
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )

        self.assertEqual(metrics.total_alerts, 2)
        self.assertEqual(metrics.true_alerts, 1)
        self.assertEqual(metrics.false_alerts, 1)
        self.assertEqual(metrics.detected_episodes, 1)

    def test_simultaneous_failures_collapsed(self):
        """Simultaneous failure records collapse into 1 episode and match only 1 alert."""
        pred_t = datetime(2015, 11, 10, 6, 0, 0)
        fail_t = pred_t + timedelta(hours=5)

        collapsed_ep = FailureEpisode(machine_id=1, timestamp=fail_t, components=("comp1", "comp2", "comp4"))

        metrics = evaluate_event_metrics(
            episodes=[collapsed_ep],
            predictions=[(1, pred_t, 0.85)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )

        self.assertEqual(metrics.total_episodes, 1)
        self.assertEqual(metrics.eligible_episodes, 1)
        self.assertEqual(metrics.detected_episodes, 1)
        self.assertEqual(metrics.true_alerts, 1)
        self.assertEqual(metrics.false_alerts, 0)

    def test_multiple_episodes_and_one_to_one_matching(self):
        """Alert matches earliest unmatched episode; each alert and episode match at most once."""
        t1 = datetime(2015, 11, 10, 6, 0, 0)
        t2 = t1 + timedelta(hours=24)

        epA = FailureEpisode(machine_id=1, timestamp=t1 + timedelta(hours=5), components=("comp1",))
        epB = FailureEpisode(machine_id=1, timestamp=t1 + timedelta(hours=10), components=("comp2",))

        preds = [(1, t1, 0.9), (1, t2, 0.9)]

        metrics = evaluate_event_metrics(
            episodes=[epA, epB],
            predictions=preds,
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )

        self.assertEqual(metrics.detected_episodes, 1)
        self.assertEqual(metrics.missed_episodes, 1)
        self.assertEqual(metrics.true_alerts, 1)
        self.assertEqual(metrics.false_alerts, 1)
        self.assertEqual(metrics.lead_hours, (5.0,))

    def test_no_eligible_events_and_zero_denominators(self):
        """Zero denominator handling for event recall and false alert rates."""
        m_zero_eps = evaluate_event_metrics(
            episodes=[],
            predictions=[(1, datetime(2015, 11, 10, 6), 0.1)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertEqual(m_zero_eps.total_episodes, 0)
        self.assertEqual(m_zero_eps.eligible_episodes, 0)
        self.assertIsNone(m_zero_eps.event_recall)
        self.assertEqual(m_zero_eps.false_alerts_per_100_machine_days, 0.0)

        # Episode before history (at split_start + 10h) is ineligible/unobservable
        unobservable_ep = FailureEpisode(
            machine_id=1, timestamp=self.split_start + timedelta(hours=10), components=("comp1",)
        )
        valid_opp_t = self.split_start + timedelta(hours=50)
        m_unobs = evaluate_event_metrics(
            episodes=[unobservable_ep],
            predictions=[(1, valid_opp_t, 0.8)],
            machine_ids=self.cohort,
            threshold=0.5,
            split=self.split,
        )
        self.assertEqual(m_unobs.total_episodes, 1)
        self.assertEqual(m_unobs.eligible_episodes, 0)
        self.assertEqual(m_unobs.ineligible_episodes, 1)
        self.assertEqual(m_unobs.unobservable_episodes, 1)
        self.assertIsNone(m_unobs.event_recall)


class MachineClusterBootstrapTests(unittest.TestCase):
    """Synthetic unit tests for deterministic machine-cluster bootstrap uncertainty."""

    def test_window_bootstrap_determinism_and_multiplicity(self):
        """Verify deterministic CI calculation and cluster multiplicity preservation."""
        y_true: List[int] = []
        y_score: List[float] = []
        metadata: List[SampleMetadata] = []

        start_t = datetime(2015, 11, 10, 6, 0, 0)
        for m_id in (1, 2, 3):
            for i in range(10):
                yt = 1 if (m_id == 1 and i < 5) else 0
                s = 0.8 if yt == 1 else 0.2
                t = start_t + timedelta(hours=i)
                y_true.append(yt)
                y_score.append(s)
                metadata.append(
                    SampleMetadata(
                        machine_id=m_id,
                        prediction_time=t,
                        client_name="client_model2",
                        split_name="test",
                        observation_start=t - timedelta(hours=48),
                        observation_end=t - timedelta(hours=1),
                        horizon_start=t,
                        horizon_end=t + timedelta(hours=24),
                    )
                )

        ci_run1 = bootstrap_window_ci(y_true, y_score, metadata, threshold=0.5, num_draws=50, seed=123)
        ci_run2 = bootstrap_window_ci(y_true, y_score, metadata, threshold=0.5, num_draws=50, seed=123)

        for metric in ("f1", "precision", "recall"):
            self.assertEqual(ci_run1[metric].point_estimate, ci_run2[metric].point_estimate)
            self.assertEqual(ci_run1[metric].ci_lower, ci_run2[metric].ci_lower)
            self.assertEqual(ci_run1[metric].ci_upper, ci_run2[metric].ci_upper)
            self.assertIsNotNone(ci_run1[metric].ci_lower)
            self.assertIsNotNone(ci_run1[metric].ci_upper)
            self.assertLessEqual(ci_run1[metric].ci_lower, ci_run1[metric].ci_upper)

    def test_insufficient_data_returns_none_ci(self):
        """When fewer than 2 machines exist, bootstrap CI bounds must return None."""
        y_true = [1, 0, 1]
        y_score = [0.8, 0.2, 0.7]
        metadata = [
            SampleMetadata(
                machine_id=1,
                prediction_time=datetime(2015, 11, 10, 6 + i),
                client_name="client_model2",
                split_name="test",
                observation_start=datetime(2015, 11, 8, 6 + i),
                observation_end=datetime(2015, 11, 10, 5 + i),
                horizon_start=datetime(2015, 11, 10, 6 + i),
                horizon_end=datetime(2015, 11, 11, 6 + i),
            )
            for i in range(3)
        ]

        ci = bootstrap_window_ci(y_true, y_score, metadata, threshold=0.5, num_draws=20)
        for metric, res in ci.items():
            self.assertIsNone(res.ci_lower)
            self.assertIsNone(res.ci_upper)
            self.assertEqual(res.num_valid_draws, 0)

    def test_event_bootstrap_multiplicity_and_determinism(self):
        """Verify event bootstrap resamples machine bundles and reproduces results."""
        t = datetime(2015, 11, 10, 6, 0, 0)
        cohort = [10, 20, 30]
        split = SplitBounds("test", datetime(2015, 11, 1, 6), datetime(2016, 1, 1, 6))

        preds = []
        for m_id in cohort:
            for i in range(24):
                preds.append((m_id, t + timedelta(hours=i), 0.9 if i == 0 else 0.1))

        episodes = [
            FailureEpisode(machine_id=10, timestamp=t + timedelta(hours=5), components=("comp1",)),
            FailureEpisode(machine_id=20, timestamp=t + timedelta(hours=12), components=("comp2",)),
        ]

        ci1 = bootstrap_event_ci(episodes, preds, machine_ids=cohort, split=split, threshold=0.5, num_draws=30, seed=999)
        ci2 = bootstrap_event_ci(episodes, preds, machine_ids=cohort, split=split, threshold=0.5, num_draws=30, seed=999)

        self.assertEqual(ci1["event_recall"].point_estimate, ci2["event_recall"].point_estimate)
        self.assertEqual(ci1["event_recall"].ci_lower, ci2["event_recall"].ci_lower)
        self.assertEqual(ci1["event_recall"].ci_upper, ci2["event_recall"].ci_upper)


class MultiClientReportTests(unittest.TestCase):
    """Synthetic unit tests for multi-client window evaluation reporting and heldout separation."""

    def test_window_report_separates_participating_from_heldout(self):
        """Verify participating clients are pooled/macro-averaged while heldout is reported strictly separately."""
        t = datetime(2015, 11, 10, 6, 0, 0)

        metadata = [
            SampleMetadata(1, t, "client_model2", "test", t - timedelta(hours=48), t - timedelta(hours=1), t, t + timedelta(hours=24)),
            SampleMetadata(1, t + timedelta(hours=1), "client_model2", "test", t - timedelta(hours=47), t, t + timedelta(hours=1), t + timedelta(hours=25)),
            SampleMetadata(2, t, "client_model3", "test", t - timedelta(hours=48), t - timedelta(hours=1), t, t + timedelta(hours=24)),
            SampleMetadata(2, t + timedelta(hours=1), "client_model3", "test", t - timedelta(hours=47), t, t + timedelta(hours=1), t + timedelta(hours=25)),
            SampleMetadata(99, t, "heldout_model1", "test", t - timedelta(hours=48), t - timedelta(hours=1), t, t + timedelta(hours=24)),
            SampleMetadata(99, t + timedelta(hours=1), "heldout_model1", "test", t - timedelta(hours=47), t, t + timedelta(hours=1), t + timedelta(hours=25)),
        ]

        y_true = [1, 0, 1, 0, 0, 1]
        y_score = [0.9, 0.2, 0.8, 0.1, 0.3, 0.7]

        report = evaluate_window_report(y_true, y_score, metadata, threshold=0.5, split_filter="test")

        self.assertIn("client_model2", report.clients)
        self.assertIn("client_model3", report.clients)
        self.assertNotIn("heldout_model1", report.clients)
        self.assertIn("heldout_model1", report.heldout)
        self.assertIsNotNone(report.pooled_participating)
        self.assertEqual(report.pooled_participating.n_total, 4)
        self.assertIn("f1", report.macro_participating)
        self.assertIsNotNone(report.macro_participating["f1"])

    def test_window_report_rejects_unknown_client(self):
        """Passing metadata with an unknown/unapproved client must raise ValueError."""
        t = datetime(2015, 11, 10, 6, 0, 0)
        metadata = [
            SampleMetadata(1, t, "client_model99", "test", t - timedelta(hours=48), t - timedelta(hours=1), t, t + timedelta(hours=24)),
        ]
        with self.assertRaises(ValueError):
            evaluate_window_report([1], [0.9], metadata, threshold=0.5, split_filter="test")


if __name__ == "__main__":
    unittest.main()
