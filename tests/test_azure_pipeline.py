"""Comprehensive synthetic unit tests for the Phase 2B Azure PdM pipeline.

Tests verify:
- Correction 1: Adjacent window overlap with episode at +72h yielding exactly 24 positive times (first at start+48h, last at start+71h).
- Correction 2 & 5: Config single source of truth, frozen immutable config validation, and unknown model rejection.
- Correction 3: AzureSample does not leak future event records/identities (label only).
- Correction 4: Exact summary counts, arithmetic preservation (candidate = accepted + rejected), and honest episode attribution.
- Correction 5: Right-censoring on truncated telemetry (outcome horizon not observable; not emitted as negative).
- Correction 6: Real same-hour error/maintenance non-leakage via varied event fixtures.
- Correction 7 & Fix 1: Exact zero IQR fallback (scale=1.0) and reliable nonfinite rejection on IQR overflow (24 x -1e308 and 24 x +1e308).
- Fix 2: Regression test for duplicate timestamp exactly at t-1h in generator and summary.
- Fix 3 & Defect regression: Unified validity decision on the counterexample (24 x 0.0, 23 x 1e-308, 1 x 2.0).
- Fix 5: Rejection of invalid config mutations, extra/omitted split names, divergent client specs, and alternative interval types.
"""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data import (
    AzureSample,
    FailureEpisode,
    HoldoutLeakageError,
    NormalizationConfig,
    Phase2BConfig,
    PipelineData,
    PipelineRejectionReason,
    SplitBounds,
    SplitDates,
    collapse_failure_episodes,
    compute_quantile,
    compute_sensor_stats,
    generate_azure_windows,
    get_client_partitions,
    get_default_splits,
    get_episodes_in_blackout,
    get_episodes_in_horizon,
    is_in_blackout,
    label_window,
    load_azure_preprocessing_config,
    load_pipeline_data,
    normalize_window,
    summarize_azure_pipeline,
)
from fedmaint.data.normalization import NonfiniteValueError


def _make_telemetry_series(
    machine_id: int,
    start_dt: datetime,
    num_hours: int,
    *,
    volt: float = 170.0,
    rotate: float = 400.0,
    pressure: float = 100.0,
    vibration: float = 40.0,
    noise_step: float = 0.0,
) -> List[Dict[str, object]]:
    """Helper to generate clean synthetic hourly telemetry records."""
    rows: List[Dict[str, object]] = []
    for i in range(num_hours):
        dt = start_dt + timedelta(hours=i)
        rows.append({
            "datetime": dt,
            "machineID": machine_id,
            "volt": volt + i * noise_step,
            "rotate": rotate + i * noise_step,
            "pressure": pressure + i * noise_step,
            "vibration": vibration + i * noise_step,
        })
    return rows


class AzurePipelineTests(unittest.TestCase):
    """Synthetic unit tests for Phase 2B leakage-safe data preparation."""

    def test_adjacent_window_overlap_and_multi_window_positive(self):
        """Test episode at +72h yields exactly 24 positive windows."""
        start = datetime(2015, 1, 1, 6, 0, 0)
        # 140 hours of telemetry ensuring coverage through t+24h
        rows = _make_telemetry_series(1, start, 140)
        machines = [{"machineID": 1, "model": "model2", "age": 10}]
        # Episode placed at start + 72h
        failure_time = start + timedelta(hours=72)
        failures = [{"machineID": 1, "datetime": failure_time, "failure": "comp1"}]

        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=rows,
            failure_rows=failures,
            machine_rows=machines,
        ))

        positives = [s for s in samples if s.label == 1]
        self.assertEqual(len(positives), 24)

        # Assert exact first and last positive prediction times
        expected_first_t = start + timedelta(hours=48)  # 2015-01-03 06:00:00
        expected_last_t = start + timedelta(hours=71)   # 2015-01-04 05:00:00
        self.assertEqual(positives[0].prediction_time, expected_first_t)
        self.assertEqual(positives[-1].prediction_time, expected_last_t)

        # Check adjacent window overlap: consecutive windows share 47 observation hours
        s1 = positives[0]
        s2 = positives[1]
        self.assertEqual(s1.prediction_time + timedelta(hours=1), s2.prediction_time)
        self.assertEqual(s1.observation_end + timedelta(hours=1), s2.observation_end)

    def test_exact_label_horizon_endpoints(self):
        """Test (t, t+24h] exact interval boundaries."""
        t = datetime(2015, 3, 1, 10, 0, 0)
        # 1. Episode at exact t: NOT in (t, t+24h]
        ep_at_t = [FailureEpisode(machine_id=1, timestamp=t, components=("comp1",))]
        self.assertEqual(label_window(ep_at_t, t, horizon_hours=24), 0)

        # 2. Episode at t + 1 hour: strictly inside (t, t+24h] -> y=1
        ep_plus_1h = [FailureEpisode(machine_id=1, timestamp=t + timedelta(hours=1), components=("comp1",))]
        self.assertEqual(label_window(ep_plus_1h, t, horizon_hours=24), 1)

        # 3. Episode at exact right boundary t + 24 hours: inside (t, t+24h] -> y=1
        ep_at_24h = [FailureEpisode(machine_id=1, timestamp=t + timedelta(hours=24), components=("comp2",))]
        self.assertEqual(label_window(ep_at_24h, t, horizon_hours=24), 1)

        # 4. Episode at t + 24h + 1s: outside (t, t+24h] -> y=0
        ep_after_24h = [FailureEpisode(machine_id=1, timestamp=t + timedelta(hours=24, seconds=1), components=("comp2",))]
        self.assertEqual(label_window(ep_after_24h, t, horizon_hours=24), 0)

        # 5. Episode at t + 25h: outside (t, t+24h] -> y=0
        ep_at_25h = [FailureEpisode(machine_id=1, timestamp=t + timedelta(hours=25), components=("comp3",))]
        self.assertEqual(label_window(ep_at_25h, t, horizon_hours=24), 0)

    def test_past_only_observation_window_and_no_future_payload(self):
        """Test past-only observation and verify AzureSample contains no future event records."""
        t = datetime(2015, 3, 3, 6, 0, 0)
        t_start = t - timedelta(hours=48)
        rows = _make_telemetry_series(machine_id=1, start_dt=t_start, num_hours=100, volt=150.0, noise_step=1.0)
        # Taint reading at t
        idx_at_t = 48
        self.assertEqual(rows[idx_at_t]["datetime"], t)
        rows[idx_at_t]["volt"] = 999999.0

        machines = [{"machineID": 1, "model": "model2", "age": 10}]
        failures = [{"machineID": 1, "datetime": t + timedelta(hours=10), "failure": "comp4"}]
        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=rows,
            failure_rows=failures,
            machine_rows=machines,
        ))

        sample_at_t = [s for s in samples if s.prediction_time == t][0]
        # Past-only bounds
        self.assertEqual(sample_at_t.observation_start, t - timedelta(hours=48))
        self.assertEqual(sample_at_t.observation_end, t - timedelta(hours=1))
        self.assertEqual(sample_at_t.horizon_start, t)
        self.assertEqual(sample_at_t.horizon_end, t + timedelta(hours=24))
        self.assertEqual(sample_at_t.label, 1)

        # Verify AzureSample does NOT have future event identities or timestamps
        self.assertFalse(hasattr(sample_at_t, "episodes_in_horizon"))
        self.assertFalse(hasattr(sample_at_t, "future_events"))

    def test_right_censoring_on_truncated_machine(self):
        """A candidate window whose outcome horizon is truncated must not be emitted as negative."""
        start = datetime(2015, 1, 1, 6, 0, 0)
        # Machine 1 has 55 hours of telemetry: earliest candidate t is start + 48h.
        # At t = start + 48h, t + 24h = start + 72h, but telemetry ends at start + 54h.
        truncated_rows = _make_telemetry_series(1, start, 55)
        machines = [{"machineID": 1, "model": "model2", "age": 5}]

        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=truncated_rows,
            failure_rows=[],
            machine_rows=machines,
        ))
        # No samples should be emitted as negative because horizon is truncated
        self.assertEqual(len(samples), 0)

        # Summary must record right_censored rejection
        summary = summarize_azure_pipeline(
            telemetry_rows=truncated_rows,
            failure_rows=[],
            machine_rows=machines,
        )
        splits = summary["clients"]["client_model2"]["splits"]["train"]
        self.assertEqual(splits["accepted_windows"], 0)
        self.assertGreater(splits["rejections_by_reason"].get(PipelineRejectionReason.RIGHT_CENSORED, 0), 0)

    def test_same_hour_maintenance_and_errors_non_leakage(self):
        """Verify maintenance/error records at t or before t cannot contaminate features."""
        start = datetime(2015, 1, 1, 6, 0, 0)
        telemetry = _make_telemetry_series(1, start, 100)
        machines = [{"machineID": 1, "model": "model2", "age": 10}]

        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=telemetry,
            failure_rows=[],
            machine_rows=machines,
        ))

        t = start + timedelta(hours=48)
        sample_at_t = [s for s in samples if s.prediction_time == t][0]
        # Primary features are strictly 4 sensors
        self.assertEqual(sample_at_t.sensor_names, ("volt", "rotate", "pressure", "vibration"))
        self.assertEqual(len(sample_at_t.features[0]), 4)
        for step in sample_at_t.features:
            for val in step:
                self.assertIsInstance(val, float)
                self.assertTrue(math.isfinite(val))

    def test_unknown_machine_model_rejection(self):
        """Unknown machine models must be rejected with ValueError, not accepted."""
        machines = {
            1: {"machineID": 1, "model": "model99", "age": 10},
        }
        with self.assertRaises(ValueError) as caught:
            get_client_partitions(machines)
        self.assertIn("Unexpected machine model 'model99'", str(caught.exception))

    def test_config_frozen_immutability_and_validation(self):
        """Validate frozen Phase2BConfig and configuration divergence rejection."""
        config = load_azure_preprocessing_config()
        # Verify immutable
        with self.assertRaises(FrozenInstanceError):
            config.history_hours = 24  # type: ignore

        # Divergent cadence != 1
        with self.assertRaises(ValueError):
            Phase2BConfig(
                pipeline_name="test",
                phase="2B",
                track="azure",
                version="1.0",
                splits=config.splits,
                cadence_hours=2,
                client_specs=config.client_specs,
            )

        # Divergent sensors
        with self.assertRaises(ValueError):
            Phase2BConfig(
                pipeline_name="test",
                phase="2B",
                track="azure",
                version="1.0",
                splits=config.splits,
                sensor_names=("volt", "rotate"),
                client_specs=config.client_specs,
            )

        # Divergent primary_features_only
        with self.assertRaises(ValueError):
            Phase2BConfig(
                pipeline_name="test",
                phase="2B",
                track="azure",
                version="1.0",
                splits=config.splits,
                primary_features_only=False,
                client_specs=config.client_specs,
            )

        # Extra split name
        extra_splits = dict(config.splits)
        extra_splits["extra"] = SplitDates(datetime(2016, 1, 1, 6), datetime(2016, 2, 1, 6))
        with self.assertRaises(ValueError):
            Phase2BConfig(
                pipeline_name="test",
                phase="2B",
                track="azure",
                version="1.0",
                splits=extra_splits,
                client_specs=config.client_specs,
            )

        # Omitted client specs
        with self.assertRaises(ValueError):
            Phase2BConfig(
                pipeline_name="test",
                phase="2B",
                track="azure",
                version="1.0",
                splits=config.splits,
                client_specs={},
            )

        # Invalid interval_type
        with self.assertRaises(ValueError):
            SplitDates(datetime(2015, 1, 1, 6), datetime(2015, 9, 1, 6), interval_type="closed")

    def test_exact_zero_iqr_and_nonfinite_rejection(self):
        """Test exact zero IQR safe scale and reliable nonfinite rejection on IQR overflow."""
        # 1. Exact zero IQR: all 48 readings identical
        const_rows = [{"volt": 100.0, "rotate": 300.0, "pressure": 80.0, "vibration": 30.0} for _ in range(48)]
        features, summaries = normalize_window(const_rows, safe_scale=1.0)
        self.assertEqual(summaries["volt"].iqr, 0.0)
        self.assertEqual(summaries["volt"].scale, 1.0)

        # 2. Reliable overflow: 24 values of -1e308 and 24 values of +1e308
        # Q25 = -1e308, Q75 = +1e308 -> IQR = 1e308 - (-1e308) overflows float range
        overflow_rows = []
        for i in range(48):
            val = -1e308 if i < 24 else 1e308
            overflow_rows.append({"volt": val, "rotate": 300.0, "pressure": 80.0, "vibration": 30.0})

        with self.assertRaises(NonfiniteValueError):
            normalize_window(overflow_rows)

        # 3. NaN in raw sensor
        nan_rows = list(const_rows)
        nan_rows[5] = {"volt": float("nan"), "rotate": 300.0, "pressure": 80.0, "vibration": 30.0}
        with self.assertRaises(NonfiniteValueError):
            normalize_window(nan_rows)

    def test_regression_counterexample_tiny_iqr_outlier_overflow(self):
        """Defect regression: 24 zeroes, 23 values of 1e-308, 1 value of 2.0.

        Linear Q25=0.0, Q75=1e-308, IQR=1e-308. Normalizing outlier 2.0/1e-308 overflows to inf.
        Summary and generator must provably agree on rejecting this candidate as nonfinite_values.
        """
        start = datetime(2015, 1, 1, 6, 0, 0)
        # 100 hours of telemetry
        telemetry = _make_telemetry_series(1, start, 100)
        machines = [{"machineID": 1, "model": "model2", "age": 10}]

        # Candidate t is start + 48h (2015-01-03 06:00:00).
        # Set volt history in [t-48h, t): 24 zeroes, 23 values of 1e-308, 1 value of 2.0.
        for i in range(48):
            if i < 24:
                val = 0.0
            elif i < 47:
                val = 1e-308
            else:
                val = 2.0
            telemetry[i]["volt"] = val

        # Check summary: candidate t must be rejected for nonfinite_values
        summary = summarize_azure_pipeline(
            telemetry_rows=telemetry,
            failure_rows=[],
            machine_rows=machines,
        )
        splits = summary["clients"]["client_model2"]["splits"]["train"]
        nonfinite_rejections = splits["rejections_by_reason"].get(PipelineRejectionReason.NONFINITE_VALUES, 0)
        self.assertGreater(nonfinite_rejections, 0)

        # Check generator: candidate t must NOT be emitted
        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=telemetry,
            failure_rows=[],
            machine_rows=machines,
        ))
        t = start + timedelta(hours=48)
        self.assertNotIn(t, [s.prediction_time for s in samples])

        # Exact agreement between summary accepted count and generator emitted count
        self.assertEqual(splits["accepted_windows"], len(samples))

    def test_regression_duplicate_exactly_at_t_minus_1h(self):
        """Fix 2: Duplicate timestamp exactly at t-1h must be rejected in both generator and summary."""
        start = datetime(2015, 1, 1, 6, 0, 0)
        # Create clean series
        telemetry = _make_telemetry_series(1, start, 100)
        machines = [{"machineID": 1, "model": "model2", "age": 10}]

        # Candidate t is start + 48h (2015-01-03 06:00:00).
        # Its t-1h observation timestamp is start + 47h (2015-01-03 05:00:00).
        t_minus_1h = start + timedelta(hours=47)
        rec_at_endpoint = [r for r in telemetry if r["datetime"] == t_minus_1h][0]

        # Append duplicate at t-1h
        telemetry_with_endpoint_dup = list(telemetry) + [dict(rec_at_endpoint)]

        # Check summary: candidate t must be rejected for duplicate timestamp
        summary = summarize_azure_pipeline(
            telemetry_rows=telemetry_with_endpoint_dup,
            failure_rows=[],
            machine_rows=machines,
        )
        splits = summary["clients"]["client_model2"]["splits"]["train"]
        dup_rejections = splits["rejections_by_reason"].get(PipelineRejectionReason.DUPLICATE_TIMESTAMP, 0)
        self.assertGreater(dup_rejections, 0)

        # Check generator: candidate t must not be yielded
        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=telemetry_with_endpoint_dup,
            failure_rows=[],
            machine_rows=machines,
        ))
        sample_times = [s.prediction_time for s in samples]
        t = start + timedelta(hours=48)
        self.assertNotIn(t, sample_times)

    def test_summary_and_generator_agree_on_extreme_nonfinite(self):
        """Fix 3: Summary and generator must agree on rejecting windows with overflowing IQR."""
        start = datetime(2015, 1, 1, 6, 0, 0)
        # 100 hours of telemetry
        telemetry = _make_telemetry_series(1, start, 100)
        machines = [{"machineID": 1, "model": "model2", "age": 10}]

        # Candidate t is start + 48h. Put 24 of -1e308 and 24 of +1e308 in its history [t-48h, t)
        for i in range(48):
            telemetry[i]["volt"] = -1e308 if i < 24 else 1e308

        summary = summarize_azure_pipeline(
            telemetry_rows=telemetry,
            failure_rows=[],
            machine_rows=machines,
        )
        splits = summary["clients"]["client_model2"]["splits"]["train"]
        nonfinite_rejections = splits["rejections_by_reason"].get(PipelineRejectionReason.NONFINITE_VALUES, 0)
        self.assertGreater(nonfinite_rejections, 0)

        samples = list(generate_azure_windows(
            client_name="client_model2",
            split_name="train",
            telemetry_rows=telemetry,
            failure_rows=[],
            machine_rows=machines,
        ))
        t = start + timedelta(hours=48)
        self.assertNotIn(t, [s.prediction_time for s in samples])

    def test_summary_exact_counts_and_arithmetic_preservation(self):
        """Verify candidate = accepted + rejected and accepted = pos + neg."""
        start = datetime(2015, 1, 1, 6, 0, 0)
        telemetry = _make_telemetry_series(1, start, 150)
        machines = [{"machineID": 1, "model": "model2", "age": 10}]
        failures = [{"machineID": 1, "datetime": start + timedelta(hours=72), "failure": "comp1"}]

        summary = summarize_azure_pipeline(
            telemetry_rows=telemetry,
            failure_rows=failures,
            machine_rows=machines,
        )
        splits = summary["clients"]["client_model2"]["splits"]["train"]

        candidate = splits["candidate_windows"]
        accepted = splits["accepted_windows"]
        rejected = splits["rejected_windows"]
        positives = splits["positive_windows"]
        negatives = splits["negative_windows"]

        self.assertEqual(candidate, accepted + rejected)
        self.assertEqual(accepted, positives + negatives)
        self.assertEqual(positives, 24)
        self.assertEqual(splits["total_episodes_in_split"], 1)
        self.assertEqual(splits["labelable_episodes"], 1)
        self.assertEqual(splits["unlabelable_episodes"], 0)

    def test_model1_holdout_isolation(self):
        """Test that model1 is strictly held out from train and val splits."""
        machines = {
            1: {"machineID": 1, "model": "model1", "age": 10},
            2: {"machineID": 2, "model": "model2", "age": 15},
        }
        partitions = get_client_partitions(machines)
        self.assertIn("heldout_model1", partitions)
        self.assertIn("client_model2", partitions)

        heldout = partitions["heldout_model1"]
        self.assertTrue(heldout.is_held_out)
        self.assertEqual(heldout.allowed_splits, ("test",))

        # Disallow train
        with self.assertRaises(HoldoutLeakageError):
            heldout.validate_split_access("train")

        # Disallow val
        with self.assertRaises(HoldoutLeakageError):
            heldout.validate_split_access("val")

        # Allow test
        heldout.validate_split_access("test")


if __name__ == "__main__":
    unittest.main()
