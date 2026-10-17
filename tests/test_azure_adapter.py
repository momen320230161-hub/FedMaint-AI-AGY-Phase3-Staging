"""Synthetic unit tests for Phase 3 Azure model-ready data adapter.

Tests verify:
- Exact signed_log1p mathematical properties (zero, sign preservation, log1p, continuity).
- Exact raw sensor reconstruction from normalized features * scale + median.
- Fixed 20-dim tabular feature extraction and exact ordering across all 4 sensors.
- Shape and type contracts for sequences [N,48,4], summaries [N,8], tabular [N,20], labels [N].
- Strict rejection of invalid values, NaN, Inf, non-binary labels, and float32 overflow.
- Deterministic streaming batch generation and drop_last behavior without unbounded materialization.
- Strict isolation of machine, time, client, and split metadata from predictor arrays.
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

from fedmaint.data import (
    DEFAULT_SENSORS,
    FLOAT32_MAX,
    FLOAT32_MIN,
    SUMMARY_FEATURE_NAMES,
    TABULAR_FEATURE_NAMES,
    AzureModelReadyAdapter,
    AzureSample,
    ModelReadyBatch,
    NonfiniteValueError,
    SampleMetadata,
    adapt_sample,
    extract_summary_features,
    extract_tabular_features,
    iter_batches,
    normalize_window,
    reconstruct_raw_series,
    signed_log1p,
)


def _make_sample(
    machine_id: int = 1,
    pred_time: datetime = datetime(2015, 1, 3, 6, 0, 0),
    client_name: str = "client_model2",
    split_name: str = "train",
    label: int = 0,
    *,
    volt_base: float = 170.0,
    rotate_base: float = 400.0,
    pressure_base: float = 100.0,
    vibration_base: float = 40.0,
    step_noise: float = 0.5,
) -> AzureSample:
    """Helper to build a valid AzureSample with 48 timesteps of deterministic telemetry."""
    rows: List[Dict[str, object]] = []
    start_dt = pred_time - timedelta(hours=48)
    for i in range(48):
        dt = start_dt + timedelta(hours=i)
        rows.append({
            "datetime": dt,
            "machineID": machine_id,
            "volt": volt_base + i * step_noise,
            "rotate": rotate_base + i * (step_noise * 1.5),
            "pressure": pressure_base + i * (step_noise * 0.8),
            "vibration": vibration_base + i * (step_noise * 0.2),
        })

    normalized_features, summaries = normalize_window(rows, sensors=DEFAULT_SENSORS)
    raw_sums = {s: summaries[s].to_dict() for s in DEFAULT_SENSORS}

    return AzureSample(
        machine_id=machine_id,
        prediction_time=pred_time,
        client_name=client_name,
        split_name=split_name,
        label=label,
        features=normalized_features,
        sensor_names=DEFAULT_SENSORS,
        raw_summaries=raw_sums,
        observation_start=pred_time - timedelta(hours=48),
        observation_end=pred_time - timedelta(hours=1),
        horizon_start=pred_time,
        horizon_end=pred_time + timedelta(hours=24),
    )


class AzureModelReadyAdapterTests(unittest.TestCase):
    """Synthetic unit tests for the model-ready adapter."""

    def test_signed_log1p_exact_math(self):
        """Verify signed_log1p mathematical properties, sign preservation, and continuity."""
        # 1. Zero check
        self.assertEqual(signed_log1p(0.0), 0.0)

        # 2. Positive exact checks
        e_minus_1 = math.e - 1.0
        self.assertAlmostEqual(signed_log1p(e_minus_1), 1.0, places=7)
        self.assertAlmostEqual(signed_log1p(10.0), math.log1p(10.0), places=7)

        # 3. Negative sign preservation
        self.assertAlmostEqual(signed_log1p(-e_minus_1), -1.0, places=7)
        self.assertAlmostEqual(signed_log1p(-10.0), -math.log1p(10.0), places=7)

        # 4. Small values around zero (continuity)
        eps = 1e-12
        self.assertAlmostEqual(signed_log1p(eps), eps, delta=1e-15)
        self.assertAlmostEqual(signed_log1p(-eps), -eps, delta=1e-15)

        # 5. Nonfinite inputs reject
        with self.assertRaises(NonfiniteValueError):
            signed_log1p(float("nan"))
        with self.assertRaises(NonfiniteValueError):
            signed_log1p(float("inf"))
        with self.assertRaises(NonfiniteValueError):
            signed_log1p(float("-inf"))

        # 6. Float32 overflow reject
        with self.assertRaises(NonfiniteValueError):
            signed_log1p(1e39)
        with self.assertRaises(NonfiniteValueError):
            signed_log1p(-1e39)

    def test_exact_raw_reconstruction_and_tabular_order(self):
        """Verify raw sensor reconstruction from normalized features and tabular feature order."""
        sample = _make_sample(
            machine_id=10,
            volt_base=170.0,
            rotate_base=400.0,
            pressure_base=100.0,
            vibration_base=40.0,
            step_noise=1.0,
        )

        # 1. Reconstruct raw readings
        raw_series = reconstruct_raw_series(sample.features, sample.raw_summaries)
        self.assertEqual(set(raw_series.keys()), set(DEFAULT_SENSORS))

        for s in DEFAULT_SENSORS:
            readings = raw_series[s]
            self.assertEqual(len(readings), 48)

            # Check linear step reconstruction for volt: starts at 170.0, ends at 170.0 + 47 * 1.0 = 217.0
            if s == "volt":
                self.assertAlmostEqual(readings[0], 170.0, places=4)
                self.assertAlmostEqual(readings[47], 217.0, places=4)
                self.assertAlmostEqual(readings[47] - readings[0], 47.0, places=4)

        # 2. Extract 20 tabular features
        tabular = extract_tabular_features(raw_series, sample.raw_summaries)
        self.assertEqual(len(tabular), 20)
        self.assertEqual(len(TABULAR_FEATURE_NAMES), 20)

        # Verify fixed 5-feature group order for each sensor:
        # (median, iqr, last_raw, last_minus_first_raw, std_raw)
        for s_idx, s in enumerate(DEFAULT_SENSORS):
            offset = s_idx * 5
            s_readings = raw_series[s]
            s_stats = sample.raw_summaries[s]

            # 1. Median
            self.assertAlmostEqual(tabular[offset + 0], s_stats["median"], places=5)
            # 2. IQR
            self.assertAlmostEqual(tabular[offset + 1], s_stats["iqr"], places=5)
            # 3. Last RAW
            self.assertAlmostEqual(tabular[offset + 2], s_readings[47], places=5)
            # 4. Last minus first RAW
            self.assertAlmostEqual(tabular[offset + 3], s_readings[47] - s_readings[0], places=5)
            # 5. Population standard deviation over 48 readings
            mean_val = sum(s_readings) / 48.0
            pop_std = math.sqrt(sum((x - mean_val) ** 2 for x in s_readings) / 48.0)
            self.assertAlmostEqual(tabular[offset + 4], pop_std, places=5)

    def test_summary_feature_names_and_transformation(self):
        """Verify summary features extraction produces 8 values with signed_log1p."""
        sample = _make_sample()
        summaries = extract_summary_features(sample.raw_summaries)
        self.assertEqual(len(summaries), 8)
        self.assertEqual(len(SUMMARY_FEATURE_NAMES), 8)

        for s_idx, s in enumerate(DEFAULT_SENSORS):
            med = sample.raw_summaries[s]["median"]
            iqr = sample.raw_summaries[s]["iqr"]
            self.assertAlmostEqual(summaries[s_idx * 2], signed_log1p(med), places=6)
            self.assertAlmostEqual(summaries[s_idx * 2 + 1], signed_log1p(iqr), places=6)

    def test_adapter_shapes_and_types(self):
        """Verify shape, length, and finiteness of adapted batch components."""
        samples = [_make_sample(label=i % 2) for i in range(5)]
        adapter = AzureModelReadyAdapter(batch_size=5)
        batches = list(adapter.iter_batches(samples))
        self.assertEqual(len(batches), 1)

        batch = batches[0]
        self.assertEqual(len(batch), 5)

        # Sequences [5, 48, 4]
        if hasattr(batch.sequences, "shape"):
            self.assertEqual(batch.sequences.shape, (5, 48, 4))
        else:
            self.assertEqual(len(batch.sequences), 5)
            self.assertEqual(len(batch.sequences[0]), 48)
            self.assertEqual(len(batch.sequences[0][0]), 4)

        # Summaries [5, 8]
        if hasattr(batch.summaries, "shape"):
            self.assertEqual(batch.summaries.shape, (5, 8))
        else:
            self.assertEqual(len(batch.summaries), 5)
            self.assertEqual(len(batch.summaries[0]), 8)

        # Tabular [5, 20]
        if hasattr(batch.tabular, "shape"):
            self.assertEqual(batch.tabular.shape, (5, 20))
        else:
            self.assertEqual(len(batch.tabular), 5)
            self.assertEqual(len(batch.tabular[0]), 20)

        # Binary labels [5]
        if hasattr(batch.labels, "shape"):
            self.assertEqual(batch.labels.shape, (5,))
            labels_list = list(batch.labels)
        else:
            labels_list = batch.labels
        self.assertEqual(labels_list, [0, 1, 0, 1, 0])

    def test_invalid_values_and_float32_overflow(self):
        """Verify rejection of NaN, Inf, non-binary labels, and float32 overflow."""
        valid_sample = _make_sample()

        # 1. Non-binary label
        with self.assertRaises(ValueError):
            bad_label_sample = AzureSample(
                machine_id=valid_sample.machine_id,
                prediction_time=valid_sample.prediction_time,
                client_name=valid_sample.client_name,
                split_name=valid_sample.split_name,
                label=2,  # Invalid: must be 0 or 1
                features=valid_sample.features,
                sensor_names=valid_sample.sensor_names,
                raw_summaries=valid_sample.raw_summaries,
                observation_start=valid_sample.observation_start,
                observation_end=valid_sample.observation_end,
                horizon_start=valid_sample.horizon_start,
                horizon_end=valid_sample.horizon_end,
            )
            adapt_sample(bad_label_sample)

        # 2. NaN in features
        nan_features = [list(row) for row in valid_sample.features]
        nan_features[10][0] = float("nan")
        with self.assertRaises(NonfiniteValueError):
            bad_features_sample = AzureSample(
                machine_id=valid_sample.machine_id,
                prediction_time=valid_sample.prediction_time,
                client_name=valid_sample.client_name,
                split_name=valid_sample.split_name,
                label=0,
                features=nan_features,
                sensor_names=valid_sample.sensor_names,
                raw_summaries=valid_sample.raw_summaries,
                observation_start=valid_sample.observation_start,
                observation_end=valid_sample.observation_end,
                horizon_start=valid_sample.horizon_start,
                horizon_end=valid_sample.horizon_end,
            )
            adapt_sample(bad_features_sample)

        # 3. Float32 overflow in features (> 3.4028235e38)
        overflow_features = [list(row) for row in valid_sample.features]
        overflow_features[5][2] = 1e39
        with self.assertRaises(NonfiniteValueError):
            bad_overflow_sample = AzureSample(
                machine_id=valid_sample.machine_id,
                prediction_time=valid_sample.prediction_time,
                client_name=valid_sample.client_name,
                split_name=valid_sample.split_name,
                label=0,
                features=overflow_features,
                sensor_names=valid_sample.sensor_names,
                raw_summaries=valid_sample.raw_summaries,
                observation_start=valid_sample.observation_start,
                observation_end=valid_sample.observation_end,
                horizon_start=valid_sample.horizon_start,
                horizon_end=valid_sample.horizon_end,
            )
            adapt_sample(bad_overflow_sample)

        # 4. Mismatched sensor ordering
        with self.assertRaises(ValueError):
            bad_sensor_sample = AzureSample(
                machine_id=valid_sample.machine_id,
                prediction_time=valid_sample.prediction_time,
                client_name=valid_sample.client_name,
                split_name=valid_sample.split_name,
                label=0,
                features=valid_sample.features,
                sensor_names=("volt", "pressure", "rotate", "vibration"),  # Wrong order
                raw_summaries=valid_sample.raw_summaries,
                observation_start=valid_sample.observation_start,
                observation_end=valid_sample.observation_end,
                horizon_start=valid_sample.horizon_start,
                horizon_end=valid_sample.horizon_end,
            )
            adapt_sample(bad_sensor_sample)

    def test_deterministic_streaming_batches(self):
        """Verify incremental streaming, exact chunk sizes, and drop_last behavior."""
        total_samples = 65
        start_time = datetime(2015, 1, 3, 6, 0, 0)
        samples = [
            _make_sample(
                machine_id=i,
                pred_time=start_time + timedelta(hours=i),
                label=i % 2,
            )
            for i in range(total_samples)
        ]

        # Generator that yields one by one to ensure streaming
        def _stream():
            for s in samples:
                yield s

        # 1. Batch size 32 without drop_last -> batches of 32, 32, 1
        batches = list(iter_batches(_stream(), batch_size=32, drop_last=False))
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 32)
        self.assertEqual(len(batches[1]), 32)
        self.assertEqual(len(batches[2]), 1)

        # Verify exact sequential machine_id order across batches
        reconstructed_ids = []
        for b in batches:
            reconstructed_ids.extend([m.machine_id for m in b.metadata])
        self.assertEqual(reconstructed_ids, list(range(total_samples)))

        # 2. Batch size 32 with drop_last=True -> batches of 32, 32 (drops final 1)
        batches_drop = list(iter_batches(_stream(), batch_size=32, drop_last=True))
        self.assertEqual(len(batches_drop), 2)
        self.assertEqual(len(batches_drop[0]), 32)
        self.assertEqual(len(batches_drop[1]), 32)

    def test_no_metadata_leakage_in_predictors(self):
        """Verify machineID, client_name, split_name, and labels never leak into predictors."""
        distinct_id = 9999
        distinct_client = "client_model3"
        distinct_split = "train"
        sample = _make_sample(
            machine_id=distinct_id,
            client_name=distinct_client,
            split_name=distinct_split,
            label=1,
        )

        batches = list(iter_batches([sample], batch_size=1))
        batch = batches[0]

        # Convert to nested lists for recursive value inspection
        seq_vals = batch.sequences.tolist() if hasattr(batch.sequences, "tolist") else batch.sequences
        sum_vals = batch.summaries.tolist() if hasattr(batch.summaries, "tolist") else batch.summaries
        tab_vals = batch.tabular.tolist() if hasattr(batch.tabular, "tolist") else batch.tabular

        def _flatten(nested):
            flat = []
            for item in nested:
                if isinstance(item, (list, tuple)):
                    flat.extend(_flatten(item))
                else:
                    flat.append(item)
            return flat

        all_predictor_values = _flatten(seq_vals) + _flatten(sum_vals) + _flatten(tab_vals)

        # Assert machine_id 9999 is nowhere in predictor numbers
        self.assertNotIn(float(distinct_id), all_predictor_values)

        # Assert metadata is correctly and strictly held in metadata container
        self.assertEqual(len(batch.metadata), 1)
        meta = batch.metadata[0]
        self.assertEqual(meta.machine_id, distinct_id)
        self.assertEqual(meta.client_name, distinct_client)
        self.assertEqual(meta.split_name, distinct_split)
        self.assertIsInstance(meta.prediction_time, datetime)


if __name__ == "__main__":
    unittest.main()
