"""Synthetic tests for Phase 3 model-ready cache bridge.

Validates:
- Exact count, canonical shapes [N,48,4], [N,8], [N,20], [N], chronological order, and reopen.
- Memory-bounded lazy metadata Sequence behavior (non-resident SampleMetadata objects).
- Strict holdout guardrails: no train/val cache for held-out client (model1) with HoldoutLeakageError.
- Stale manifest rejection (config hash, source hash, code version mismatch).
- Crash-safe / partial-cache behavior without deleting arbitrary paths.
- Zero-copy numpy memmaps without copying full arrays into memory.
- Unchanged labels and model-ready features vs Phase 2B generator.
- Single shared PipelineData across all clients and splits.
- Explicit cohort machine IDs stored and exposed on ClientDataset.
- Safe test/heldout scoring interface without train/val leakage validation.
"""
from __future__ import annotations

import collections.abc
import json
import math
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from fedmaint.data.adapter import SampleMetadata, adapt_sample
from fedmaint.data.config import Phase2BConfig, load_azure_preprocessing_config
from fedmaint.data.model_cache import (
    CODE_VERSION,
    CacheIncompleteError,
    CacheManifest,
    CacheStaleError,
    LazyMetadataSequence,
    build_all_model_caches,
    build_client_split_cache,
    compute_config_hash,
    get_cache_dir,
    is_cache_valid,
    load_cached_client_dataset,
)
from fedmaint.data.pipeline import (
    PipelineData,
    generate_azure_windows,
    load_pipeline_data,
    summarize_azure_pipeline,
)
from fedmaint.data.splits import (
    HoldoutLeakageError,
    get_client_partitions,
)
from fedmaint.training.engine import ClientDataset


def create_synthetic_pipeline_data(config: Phase2BConfig) -> PipelineData:
    """Create in-memory synthetic pipeline data covering train, val, and test splits."""
    machine_rows = [
        {"machineID": 1, "model": "model2", "age": 10},
        {"machineID": 2, "model": "model3", "age": 5},
        {"machineID": 3, "model": "model4", "age": 7},
        {"machineID": 4, "model": "model1", "age": 12},  # held-out client
    ]

    # Telemetry intervals around each split start to yield accepted windows
    intervals: List[Tuple[datetime, datetime]] = [
        # Train interval: split start is 2015-01-01 06:00
        (datetime(2015, 1, 1, 6, 0, 0), datetime(2015, 1, 4, 12, 0, 0)),
        # Val interval: split start is 2015-09-01 06:00
        (datetime(2015, 9, 1, 6, 0, 0), datetime(2015, 9, 4, 12, 0, 0)),
        # Test interval: split start is 2015-11-01 06:00
        (datetime(2015, 11, 1, 6, 0, 0), datetime(2015, 11, 4, 12, 0, 0)),
    ]

    telemetry_rows: List[Dict[str, object]] = []
    for m in machine_rows:
        m_id = m["machineID"]
        for start_dt, end_dt in intervals:
            curr = start_dt
            step = 0
            while curr <= end_dt:
                telemetry_rows.append({
                    "machineID": m_id,
                    "datetime": curr,
                    "volt": 170.0 + math.sin(step * 0.1) * 5.0,
                    "rotate": 440.0 + math.cos(step * 0.1) * 10.0,
                    "pressure": 100.0 + (step % 5) * 2.0,
                    "vibration": 40.0 + (step % 3) * 1.5,
                })
                curr += timedelta(hours=1)
                step += 1

    # Failure episodes to generate positive labels
    failure_rows = [
        # Machine 1 failure during train outcome horizon (t=2015-01-03 06:00 -> horizon is (t, t+24h])
        {"machineID": 1, "datetime": datetime(2015, 1, 3, 14, 0, 0), "failure": "comp1"},
        # Machine 2 failure during val
        {"machineID": 2, "datetime": datetime(2015, 9, 3, 14, 0, 0), "failure": "comp2"},
        # Machine 4 (heldout) failure during test
        {"machineID": 4, "datetime": datetime(2015, 11, 3, 14, 0, 0), "failure": "comp3"},
    ]

    return load_pipeline_data(
        config=config,
        telemetry_rows=telemetry_rows,
        failure_rows=failure_rows,
        machine_rows=machine_rows,
    )


class TestModelCache(unittest.TestCase):
    """Synthetic test suite for model_cache.py and ClientDataset cache bridge."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="fedmaint_cache_test_")
        self.config = load_azure_preprocessing_config()
        self.pipeline_data = create_synthetic_pipeline_data(self.config)

    def tearDown(self) -> None:
        import gc
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exact_count_shape_order_reopen(self) -> None:
        """Verify exact sample count, canonical shapes, chronological order, and reopen."""
        client_name = "client_model2"
        split_name = "train"

        # Check expected count from Phase 2B summary
        summary = summarize_azure_pipeline(config=self.config, pipeline_data=self.pipeline_data)
        expected_n = summary["clients"][client_name]["splits"][split_name]["accepted_windows"]
        self.assertGreater(expected_n, 0, "Synthetic test data must yield at least one accepted window.")

        # Build cache
        cache_path = build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )
        self.assertTrue((cache_path / "completed.marker").is_file())
        self.assertTrue((cache_path / "manifest.json").is_file())

        # Reopen via load_cached_client_dataset
        dataset = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            config=self.config,
            validate=True,
        )

        self.assertEqual(len(dataset), expected_n)
        self.assertEqual(dataset.client_name, client_name)
        self.assertEqual(dataset.split_name, split_name)

        # Shapes and dtypes
        self.assertEqual(dataset.sequences.shape, (expected_n, 48, 4))
        self.assertEqual(dataset.sequences.dtype, np.float32)
        self.assertEqual(dataset.summaries.shape, (expected_n, 8))
        self.assertEqual(dataset.summaries.dtype, np.float32)
        self.assertEqual(dataset.tabular.shape, (expected_n, 20))
        self.assertEqual(dataset.tabular.dtype, np.float32)
        self.assertEqual(dataset.labels.shape, (expected_n,))
        self.assertEqual(dataset.labels.dtype, np.int64)

        # Finiteness
        self.assertTrue(np.all(np.isfinite(dataset.sequences)))
        self.assertTrue(np.all(np.isfinite(dataset.summaries)))
        self.assertTrue(np.all(np.isfinite(dataset.tabular)))
        self.assertTrue(np.all((dataset.labels == 0) | (dataset.labels == 1)))

        # Chronological ordering check
        timestamps = [m.prediction_time for m in dataset.metadata]
        for i in range(1, len(timestamps)):
            self.assertGreaterEqual(timestamps[i], timestamps[i - 1])

    def test_lazy_metadata_behavior(self) -> None:
        """Verify LazyMetadataSequence is memory-bounded and generates SampleMetadata on demand."""
        client_name = "client_model2"
        split_name = "train"

        build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        dataset = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            config=self.config,
        )

        # Must be LazyMetadataSequence and collections.abc.Sequence
        self.assertIsInstance(dataset.metadata, LazyMetadataSequence)
        self.assertIsInstance(dataset.metadata, collections.abc.Sequence)
        # Must NOT be a materialized tuple or list
        self.assertNotIsInstance(dataset.metadata, (tuple, list))

        self.assertEqual(len(dataset.metadata), len(dataset))

        # Access index 0 produces SampleMetadata on access
        meta0 = dataset.metadata[0]
        self.assertIsInstance(meta0, SampleMetadata)
        self.assertEqual(meta0.client_name, client_name)
        self.assertEqual(meta0.split_name, split_name)
        self.assertEqual(meta0.observation_start, meta0.prediction_time - timedelta(hours=48))
        self.assertEqual(meta0.observation_end, meta0.prediction_time - timedelta(hours=1))
        self.assertEqual(meta0.horizon_start, meta0.prediction_time)
        self.assertEqual(meta0.horizon_end, meta0.prediction_time + timedelta(hours=24))

        # Negative indexing
        meta_last = dataset.metadata[-1]
        self.assertEqual(meta_last, dataset.metadata[len(dataset) - 1])

        # Out of bounds raises IndexError
        with self.assertRaises(IndexError):
            _ = dataset.metadata[len(dataset)]
        with self.assertRaises(IndexError):
            _ = dataset.metadata[-len(dataset) - 1]

        # Slicing returns a sliced LazyMetadataSequence
        sliced = dataset.metadata[0:2]
        self.assertIsInstance(sliced, LazyMetadataSequence)
        self.assertEqual(len(sliced), min(2, len(dataset)))
        self.assertEqual(sliced[0], meta0)

    def test_no_heldout_train_val_cache(self) -> None:
        """Verify held-out client (model1) strictly forbids train and val cache building and loading."""
        # Building heldout train/val must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            build_client_split_cache(
                output_root=self.temp_dir,
                client_name="heldout_model1",
                split_name="train",
                pipeline_data=self.pipeline_data,
                config=self.config,
            )

        with self.assertRaises(HoldoutLeakageError):
            build_client_split_cache(
                output_root=self.temp_dir,
                client_name="heldout_model1",
                split_name="val",
                pipeline_data=self.pipeline_data,
                config=self.config,
            )

        # Loading heldout train/val must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            load_cached_client_dataset(
                output_root=self.temp_dir,
                client_name="heldout_model1",
                split_name="train",
                config=self.config,
            )

        with self.assertRaises(HoldoutLeakageError):
            load_cached_client_dataset(
                output_root=self.temp_dir,
                client_name="heldout_model1",
                split_name="val",
                config=self.config,
            )

        # Building heldout test split must succeed
        heldout_test_path = build_client_split_cache(
            output_root=self.temp_dir,
            client_name="heldout_model1",
            split_name="test",
            pipeline_data=self.pipeline_data,
            config=self.config,
        )
        self.assertTrue((heldout_test_path / "completed.marker").is_file())

        # Loading heldout test split must succeed
        heldout_ds = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name="heldout_model1",
            split_name="test",
            config=self.config,
        )
        self.assertEqual(heldout_ds.client_name, "heldout_model1")
        self.assertEqual(heldout_ds.split_name, "test")

    def test_stale_manifest_rejection(self) -> None:
        """Verify stale manifests (config hash, source hash, code version) are rejected."""
        client_name = "client_model2"
        split_name = "train"

        cache_path = build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )
        manifest_file = cache_path / "manifest.json"

        # 1. Config hash mismatch
        with manifest_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        orig_config_hash = data["config_hash"]
        data["config_hash"] = "tampered_config_hash_value"
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        self.assertFalse(is_cache_valid(self.temp_dir, client_name, split_name, config=self.config))
        with self.assertRaises(CacheStaleError):
            load_cached_client_dataset(self.temp_dir, client_name, split_name, config=self.config)

        # 2. Source manifest hash mismatch
        data["config_hash"] = orig_config_hash
        data["source_manifest_hash"] = "stale_source_hash"
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        self.assertFalse(is_cache_valid(
            self.temp_dir, client_name, split_name, config=self.config, source_manifest_hash="expected_hash"
        ))
        with self.assertRaises(CacheStaleError):
            load_cached_client_dataset(
                self.temp_dir, client_name, split_name, config=self.config, source_manifest_hash="expected_hash"
            )

        # 3. Code version mismatch
        data["source_manifest_hash"] = "staging_synthetic_manifest_untracked"
        data["code_version"] = "0.0.1_outdated"
        with manifest_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        with self.assertRaises(CacheStaleError):
            load_cached_client_dataset(self.temp_dir, client_name, split_name, config=self.config)

    def test_partial_and_incomplete_cache_behavior(self) -> None:
        """Verify incomplete cache is refused, and rebuilds safely without deleting arbitrary paths."""
        client_name = "client_model2"
        split_name = "train"

        cache_path = build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        # Put a canary file in the output root and a subfolder to ensure safe cleanup
        canary_root = Path(self.temp_dir) / "canary_root.txt"
        canary_root.write_text("must_not_be_deleted", encoding="utf-8")

        # 1. Missing completed.marker
        marker_file = cache_path / "completed.marker"
        marker_file.unlink()

        self.assertFalse(is_cache_valid(self.temp_dir, client_name, split_name, config=self.config))
        with self.assertRaises(CacheIncompleteError):
            load_cached_client_dataset(self.temp_dir, client_name, split_name, config=self.config)

        # 2. Missing manifest.json
        manifest_file = cache_path / "manifest.json"
        manifest_file.unlink()

        with self.assertRaises(CacheIncompleteError):
            load_cached_client_dataset(self.temp_dir, client_name, split_name, config=self.config)

        # 3. Missing array file
        seq_file = cache_path / "sequences.npy"
        seq_file.unlink()

        with self.assertRaises(CacheIncompleteError):
            load_cached_client_dataset(self.temp_dir, client_name, split_name, config=self.config)

        # 4. Rebuild from incomplete cache
        rebuilt_path = build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
            force_rebuild=True,
        )
        self.assertTrue((rebuilt_path / "completed.marker").is_file())
        self.assertTrue((rebuilt_path / "manifest.json").is_file())
        self.assertTrue((rebuilt_path / "sequences.npy").is_file())

        # Canary file in output root MUST still exist (no arbitrary path deletion)
        self.assertTrue(canary_root.is_file(), "Arbitrary path was deleted during cache rebuild.")
        self.assertEqual(canary_root.read_text(encoding="utf-8"), "must_not_be_deleted")

    def test_no_copying_full_arrays(self) -> None:
        """Verify memmaps back arrays and minibatches are sliced without loading full arrays."""
        client_name = "client_model2"
        split_name = "train"

        build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        dataset = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            config=self.config,
            mmap_mode="r",
        )

        # Ensure underlying arrays are memmaps
        self.assertIsInstance(dataset.sequences, np.memmap)
        self.assertIsInstance(dataset.summaries, np.memmap)
        self.assertIsInstance(dataset.tabular, np.memmap)
        self.assertIsInstance(dataset.labels, np.memmap)

        # Read-only check
        self.assertFalse(dataset.sequences.flags.writeable)
        self.assertFalse(dataset.tabular.flags.writeable)

        # Slicing a minibatch creates a sub-array without altering base memmap
        slice_seq = dataset.sequences[0:2]
        self.assertEqual(slice_seq.shape, (2, 48, 4))

    def test_unchanged_labels_and_features_vs_phase2b_generator(self) -> None:
        """Verify cache exactly matches direct Phase 2B AzureSample generator and adapter."""
        client_name = "client_model2"
        split_name = "train"

        build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        dataset = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            config=self.config,
        )

        direct_samples = list(generate_azure_windows(
            pipeline_data=self.pipeline_data,
            client_name=client_name,
            split_name=split_name,
            config=self.config,
        ))

        self.assertEqual(len(dataset), len(direct_samples))

        direct_seqs = []
        direct_sums = []
        direct_lbls = []
        direct_tabs = []
        direct_metas = []

        for s in direct_samples:
            seq, summ, lbl, tab, meta = adapt_sample(s, sensors=self.config.sensor_names)
            direct_seqs.append(seq)
            direct_sums.append(summ)
            direct_lbls.append(lbl)
            direct_tabs.append(tab)
            direct_metas.append(meta)

        # Exact label matching
        np.testing.assert_array_equal(dataset.labels, np.array(direct_lbls, dtype=np.int64))

        # Exact feature matching
        np.testing.assert_allclose(dataset.sequences, np.array(direct_seqs, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(dataset.summaries, np.array(direct_sums, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(dataset.tabular, np.array(direct_tabs, dtype=np.float32), atol=1e-6)

        # Exact metadata matching on access
        for idx in range(len(dataset)):
            cached_meta = dataset.metadata[idx]
            exp_meta = direct_metas[idx]
            self.assertEqual(cached_meta.machine_id, exp_meta.machine_id)
            self.assertEqual(cached_meta.prediction_time, exp_meta.prediction_time)
            self.assertEqual(cached_meta.client_name, exp_meta.client_name)
            self.assertEqual(cached_meta.split_name, exp_meta.split_name)
            self.assertEqual(cached_meta.observation_start, exp_meta.observation_start)
            self.assertEqual(cached_meta.observation_end, exp_meta.observation_end)
            self.assertEqual(cached_meta.horizon_start, exp_meta.horizon_start)
            self.assertEqual(cached_meta.horizon_end, exp_meta.horizon_end)

    def test_one_shared_pipeline_data(self) -> None:
        """Verify build_all_model_caches operates from one shared PipelineData across clients/splits."""
        all_caches = build_all_model_caches(
            output_root=self.temp_dir,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        # Expected 10 partitions: 3 participating * 3 splits + 1 held-out * 1 test split
        expected_keys = {
            ("client_model2", "train"),
            ("client_model2", "val"),
            ("client_model2", "test"),
            ("client_model3", "train"),
            ("client_model3", "val"),
            ("client_model3", "test"),
            ("client_model4", "train"),
            ("client_model4", "val"),
            ("client_model4", "test"),
            ("heldout_model1", "test"),
        }
        self.assertEqual(set(all_caches.keys()), expected_keys)

        # Ensure heldout train/val are strictly absent
        self.assertNotIn(("heldout_model1", "train"), all_caches)
        self.assertNotIn(("heldout_model1", "val"), all_caches)

        for key, path in all_caches.items():
            self.assertTrue(path.is_dir())
            self.assertTrue((path / "completed.marker").is_file())

    def test_explicit_cohort_machine_ids(self) -> None:
        """Verify explicit machine cohort IDs from get_client_partitions are exposed."""
        client_name = "client_model2"
        split_name = "train"

        build_client_split_cache(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        dataset = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name=client_name,
            split_name=split_name,
            config=self.config,
        )

        partitions = get_client_partitions(self.pipeline_data.machines, config=self.config)
        expected_cohort = partitions[client_name].machine_ids

        self.assertEqual(dataset.cohort_machine_ids, expected_cohort)
        self.assertEqual(dataset.machine_ids, expected_cohort)

    def test_test_heldout_scoring_without_train_val_leakage_validation(self) -> None:
        """Verify ClientDataset exists for test/heldout scoring but never triggers train/val validation."""
        build_client_split_cache(
            output_root=self.temp_dir,
            client_name="heldout_model1",
            split_name="test",
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        # Loading test/heldout with validate=True must NOT fail (train/val validation is skipped for test)
        heldout_ds = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name="heldout_model1",
            split_name="test",
            config=self.config,
            validate=True,
        )
        self.assertGreater(len(heldout_ds), 0)

        # Directly invoking ds.validate() on test data would raise HoldoutLeakageError per engine.py
        with self.assertRaises(HoldoutLeakageError):
            heldout_ds.validate()

    def test_summary_computed_once_on_build_and_zero_on_reuse(self) -> None:
        """Verify summarize_azure_pipeline is called exactly once when building all partitions,
        and zero times when valid caches are reused."""
        import fedmaint.data.model_cache as mc

        summary_call_count = [0]
        real_summarize = mc.summarize_azure_pipeline

        def counting_summarize(*args, **kwargs):
            summary_call_count[0] += 1
            return real_summarize(*args, **kwargs)

        mc.summarize_azure_pipeline = counting_summarize
        try:
            # Pass 1: Build all 10 partitions from scratch; summary should be called exactly once
            all_caches = mc.build_all_model_caches(
                output_root=self.temp_dir,
                pipeline_data=self.pipeline_data,
                config=self.config,
            )
            self.assertEqual(len(all_caches), 10)
            self.assertEqual(
                summary_call_count[0], 1,
                f"Expected exactly 1 summary computation for all 10 partitions; got {summary_call_count[0]}."
            )

            # Pass 2: Re-run with valid existing caches; summary should be called zero additional times
            reused_caches = mc.build_all_model_caches(
                output_root=self.temp_dir,
                pipeline_data=self.pipeline_data,
                config=self.config,
                force_rebuild=False,
            )
            self.assertEqual(len(reused_caches), 10)
            self.assertEqual(
                summary_call_count[0], 1,
                f"Expected zero additional summary calls when caches are valid; total calls: {summary_call_count[0]}."
            )
        finally:
            mc.summarize_azure_pipeline = real_summarize

    def test_timezone_aware_utc_timestamp_in_manifest_and_marker(self) -> None:
        """Verify manifest created_at and completed.marker use timezone-aware UTC, while prediction time is naive."""
        cache_path = build_client_split_cache(
            output_root=self.temp_dir,
            client_name="client_model2",
            split_name="train",
            pipeline_data=self.pipeline_data,
            config=self.config,
        )

        manifest_file = cache_path / "manifest.json"
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        created_at_str = manifest_data["created_at"]
        dt = datetime.fromisoformat(created_at_str)
        self.assertIsNotNone(dt.tzinfo, "Manifest created_at must be timezone-aware UTC.")

        marker_file = cache_path / "completed.marker"
        marker_text = marker_file.read_text(encoding="utf-8")
        self.assertTrue(marker_text.startswith("COMPLETED "))
        marker_dt_str = marker_text.strip().split(" ", 1)[1]
        marker_dt = datetime.fromisoformat(marker_dt_str)
        self.assertIsNotNone(marker_dt.tzinfo, "Completion marker timestamp must be timezone-aware UTC.")

        # Ensure prediction timestamps remain naive (Phase 2B requirement)
        dataset = load_cached_client_dataset(
            output_root=self.temp_dir,
            client_name="client_model2",
            split_name="train",
            config=self.config,
        )
        sample_meta = dataset.metadata[0]
        self.assertIsNone(sample_meta.prediction_time.tzinfo, "Prediction time must remain naive.")


if __name__ == "__main__":
    unittest.main()
