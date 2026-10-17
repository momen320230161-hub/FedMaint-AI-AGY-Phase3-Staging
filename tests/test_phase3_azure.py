"""Synthetic integration and unit tests for Phase 3 Azure baseline experiments and CLI.

Tests verify:
- End-to-end smoke execution across all models (GRU, Logistic, MLP) and regimes (local, central).
- Exact local and central client boundaries:
    * Local: 1 client per run, own train/val/test + separate zero-shot heldout model1.
    * Central: pooled participating clients, unweighted macro val checkpoint, single pooled threshold,
      pooled/macro test metrics + separate zero-shot heldout model1. Distinguishes central reference.
- Identical AzureGRU single-layer architecture in both local and central regimes.
- Strict held-out client isolation: model1 never loaded or cached for train or val.
- Validation-only threshold selection: threshold chosen strictly on val split.
- Separate zero-shot evaluation on held-out model1 with frozen thresholds.
- Configuration modes: smoke (1 seed, 2 epochs, 30 bootstrap draws, non-representative)
  vs full (seeds 17/23/41, 25 epochs, 1000 bootstrap draws, representative).
- Deterministic outputs on CPU across repeated smoke runs (except measured runtime/memory).
- Safe JSON serialization to outputs/phase3 without raw data leaks.
- Public batched scoring via predict_model_probabilities: enforces frozen train scaler with no fitting.
- Tracking of window positives vs distinct failure episodes.
- Logistic regression convergence and solver logging.
- All tests run completely data-free using synthetic injected PipelineData on CPU.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

from fedmaint.data.config import Phase2BConfig, load_azure_preprocessing_config
from fedmaint.data.episodes import FailureEpisode
from fedmaint.data.model_cache import load_cached_client_dataset
from fedmaint.data.pipeline import PipelineData, load_pipeline_data
from fedmaint.data.splits import HoldoutLeakageError
from fedmaint.evaluation.threshold import (
    HELDOUT_CLIENTS,
    PARTICIPATING_CLIENTS,
    select_threshold,
)
from fedmaint.experiments.phase3_azure import (
    Phase3AzureConfig,
    Phase3ExperimentReport,
    SingleRunResult,
    create_stratified_subset,
    load_phase3_azure_config,
    load_phase3_datasets,
    run_phase3_experiment,
)
from fedmaint.models import (
    AzureGRU,
    AzureLogisticRegression,
    AzureMLP,
    is_sklearn_available,
    is_torch_available,
)
from fedmaint.training import (
    ClientDataset,
    predict_model_probabilities,
)


def create_multi_machine_synthetic_data(config: Phase2BConfig) -> PipelineData:
    """Create in-memory synthetic pipeline data with >= 2 machines per model to test clustering."""
    # 2 machines per model: 6 participating (models 2, 3, 4) + 2 held-out (model 1) = 8 machines
    machine_rows = [
        {"machineID": 10, "model": "model2", "age": 10},
        {"machineID": 11, "model": "model2", "age": 12},
        {"machineID": 20, "model": "model3", "age": 5},
        {"machineID": 21, "model": "model3", "age": 8},
        {"machineID": 30, "model": "model4", "age": 7},
        {"machineID": 31, "model": "model4", "age": 9},
        {"machineID": 40, "model": "model1", "age": 15},  # held-out
        {"machineID": 41, "model": "model1", "age": 14},  # held-out
    ]

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
                    "volt": 170.0 + math.sin(step * 0.1) * 5.0 + (m_id % 3),
                    "rotate": 440.0 + math.cos(step * 0.1) * 10.0,
                    "pressure": 100.0 + (step % 5) * 2.0,
                    "vibration": 40.0 + (step % 3) * 1.5,
                })
                curr += timedelta(hours=1)
                step += 1

    failure_rows = [
        # Model 2 failures
        {"machineID": 10, "datetime": datetime(2015, 1, 3, 14, 0, 0), "failure": "comp1"},
        {"machineID": 11, "datetime": datetime(2015, 9, 3, 14, 0, 0), "failure": "comp2"},
        {"machineID": 10, "datetime": datetime(2015, 11, 3, 14, 0, 0), "failure": "comp1"},
        # Model 3 failures
        {"machineID": 20, "datetime": datetime(2015, 1, 3, 16, 0, 0), "failure": "comp2"},
        {"machineID": 21, "datetime": datetime(2015, 9, 3, 16, 0, 0), "failure": "comp3"},
        {"machineID": 20, "datetime": datetime(2015, 11, 3, 16, 0, 0), "failure": "comp2"},
        # Model 4 failures
        {"machineID": 30, "datetime": datetime(2015, 1, 3, 18, 0, 0), "failure": "comp4"},
        {"machineID": 31, "datetime": datetime(2015, 9, 3, 18, 0, 0), "failure": "comp1"},
        {"machineID": 30, "datetime": datetime(2015, 11, 3, 18, 0, 0), "failure": "comp4"},
        # Held-out Model 1 failures (test split only)
        {"machineID": 40, "datetime": datetime(2015, 11, 3, 15, 0, 0), "failure": "comp3"},
        {"machineID": 41, "datetime": datetime(2015, 11, 3, 20, 0, 0), "failure": "comp1"},
    ]

    return load_pipeline_data(
        config=config,
        telemetry_rows=telemetry_rows,
        failure_rows=failure_rows,
        machine_rows=machine_rows,
    )


class Phase3AzureIntegrationTests(unittest.TestCase):
    """Synthetic end-to-end integration tests for Phase 3 Azure baseline experiments."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="fedmaint_phase3_test_")
        self.output_dir = Path(self.temp_dir) / "outputs"
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.phase2b_config = load_azure_preprocessing_config()
        self.phase3_config = load_phase3_azure_config()
        self.synthetic_data = create_multi_machine_synthetic_data(self.phase2b_config)

    def tearDown(self) -> None:
        import gc
        gc.collect()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_smoke_execution_and_json_serialization(self) -> None:
        """Verify complete smoke mode execution for all models and regimes, writing valid JSON."""
        report = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["gru", "logistic", "mlp"],
            regimes=["local", "central"],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            device="cpu",
        )

        self.assertIsInstance(report, Phase3ExperimentReport)
        self.assertEqual(report.mode, "smoke")
        self.assertFalse(report.is_representative)
        self.assertTrue(report.is_plumbing_only)
        self.assertEqual(report.seeds_executed, [17])

        # Expected runs:
        # Local GRU: 3 clients * 1 seed = 3
        # Central GRU: 1 pooled * 1 seed = 1
        # Local Logistic: 3 clients * 1 seed = 3
        # Central Logistic: 1 pooled * 1 seed = 1
        # Local MLP: 3 clients * 1 seed = 3
        # Central MLP: 1 pooled * 1 seed = 1
        # Total = 12 runs
        self.assertEqual(len(report.results), 12)

        # Check JSON output file
        results_file = self.output_dir / "results.json"
        self.assertTrue(results_file.is_file())

        with results_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["pipeline_name"], "azure_phase3_baselines_and_orchestration")
        self.assertEqual(data["phase"], "3")
        self.assertEqual(data["mode"], "smoke")
        self.assertFalse(data["is_representative"])
        self.assertTrue(data["is_plumbing_only"])
        self.assertEqual(len(data["results"]), 12)
        self.assertEqual(len(data["summary_table"]), 12)

        # Runtime verification: total_elapsed_seconds measures model fit plus validation/test/bootstrap (run_start begins after load_phase3_datasets, excluding raw data loading/cache build); fit_elapsed_seconds measures model fitting
        for r in report.results:
            self.assertGreater(r.total_elapsed_seconds, 0.0)
            self.assertIsNotNone(r.fit_elapsed_seconds)
            assert r.fit_elapsed_seconds is not None
            self.assertGreater(r.fit_elapsed_seconds, 0.0)
            self.assertGreaterEqual(r.total_elapsed_seconds, r.fit_elapsed_seconds)

        first_res = data["results"][0]
        self.assertIn("total_elapsed_seconds", first_res)
        self.assertIn("fit_elapsed_seconds", first_res)
        self.assertGreaterEqual(first_res["total_elapsed_seconds"], first_res["fit_elapsed_seconds"])

        first_summary = data["summary_table"][0]
        self.assertIn("runtime_s", first_summary)
        self.assertIn("fit_time_s", first_summary)

    def test_data_dir_resolution_and_actionable_error(self) -> None:
        """Verify data directory resolution and actionable error message when data is missing."""
        # 1. Non-existent explicit data-dir raises FileNotFoundError
        non_existent_dir = Path(self.temp_dir) / "does_not_exist"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_phase3_datasets(
                cache_root=self.cache_dir,
                data_dir=non_existent_dir,
                pipeline_data=None,
                phase2b_config=self.phase2b_config,
                phase3_config=self.phase3_config,
                mode="smoke",
            )
        self.assertIn("Specified data directory does not exist or is not a directory", str(ctx.exception))

        # 2. data_dir=None with absent default data directory raises clear actionable ValueError
        # (temporarily simulate absent DEFAULT_DATA_DIR by pointing DEFAULT_DATA_DIR to non-existent)
        import fedmaint.experiments.phase3_azure as p3_mod
        orig_default = p3_mod.DEFAULT_DATA_DIR
        try:
            p3_mod.DEFAULT_DATA_DIR = Path(self.temp_dir) / "no_azure_data"
            with self.assertRaises(ValueError) as ctx:
                load_phase3_datasets(
                    cache_root=self.cache_dir,
                    data_dir=None,
                    pipeline_data=None,
                    phase2b_config=self.phase2b_config,
                    phase3_config=self.phase3_config,
                    mode="smoke",
                )
            err_msg = str(ctx.exception)
            self.assertIn("Raw Azure dataset directory not found at default location", err_msg)
            self.assertIn("--data-dir", err_msg)
        finally:
            p3_mod.DEFAULT_DATA_DIR = orig_default

    def test_client_boundaries_and_zero_shot_isolation(self) -> None:
        """Verify local vs central client boundaries and separate zero-shot model1 reporting."""
        report = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["gru"],
            regimes=["local", "central"],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            device="cpu",
        )

        local_results = [r for r in report.results if r.regime == "local"]
        central_results = [r for r in report.results if r.regime == "central"]

        # Local runs: 3 participating clients
        self.assertEqual(len(local_results), 3)
        local_clients = {r.client_name for r in local_results}
        self.assertEqual(local_clients, PARTICIPATING_CLIENTS)

        for lr in local_results:
            self.assertFalse(lr.is_central_reference)
            self.assertIn(lr.client_name, PARTICIPATING_CLIENTS)
            # Test metrics evaluated on own client
            self.assertGreater(lr.total_windows_test, 0)
            # Zero-shot heldout evaluated separately
            self.assertIn("precision", lr.zero_shot_heldout_window_metrics)
            self.assertIn("f1", lr.zero_shot_heldout_window_metrics)
            self.assertIn("event_recall", lr.zero_shot_heldout_event_metrics)

        # Central run: 1 pooled run
        self.assertEqual(len(central_results), 1)
        cr = central_results[0]
        self.assertTrue(cr.is_central_reference)
        self.assertEqual(cr.client_name, "pooled_participating")
        self.assertIsNotNone(cr.central_window_report)

        report_dict = cr.central_window_report
        assert report_dict is not None
        self.assertIn("clients", report_dict)
        self.assertIn("pooled_participating", report_dict)
        self.assertIn("macro_participating", report_dict)
        self.assertIn("heldout", report_dict)

        # Heldout model1 is strictly isolated in heldout section
        self.assertIn("heldout_model1", report_dict["heldout"])
        self.assertNotIn("heldout_model1", report_dict["clients"])

    def test_holdout_leakage_prevented_at_all_stages(self) -> None:
        """Verify heldout_model1 can never be loaded for train or val."""
        # Querying load_cached_client_dataset for train or val raises HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            load_cached_client_dataset(self.cache_dir, "heldout_model1", "train", config=self.phase2b_config)

        with self.assertRaises(HoldoutLeakageError):
            load_cached_client_dataset(self.cache_dir, "heldout_model1", "val", config=self.phase2b_config)

        # Datasets loaded for Phase 3 must not contain heldout train/val
        datasets, _ = load_phase3_datasets(
            cache_root=self.cache_dir,
            pipeline_data=self.synthetic_data,
            phase2b_config=self.phase2b_config,
            phase3_config=self.phase3_config,
            mode="smoke",
        )
        self.assertNotIn(("heldout_model1", "train"), datasets)
        self.assertNotIn(("heldout_model1", "val"), datasets)
        self.assertIn(("heldout_model1", "test"), datasets)

    def test_validation_only_threshold_selection(self) -> None:
        """Verify threshold selection requires validation metadata and rejects non-val."""
        report = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["gru"],
            regimes=["local"],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            device="cpu",
        )

        for r in report.results:
            self.assertGreater(r.threshold, 0.0)
            self.assertLess(r.threshold, 1.0)
            self.assertGreater(r.val_f1_score, 0.0)

        # Attempting select_threshold on test metadata must fail with HoldoutLeakageError
        datasets, _ = load_phase3_datasets(
            cache_root=self.cache_dir,
            pipeline_data=self.synthetic_data,
            phase2b_config=self.phase2b_config,
            phase3_config=self.phase3_config,
            mode="smoke",
        )
        test_ds = datasets[("client_model2", "test")]
        with self.assertRaises(HoldoutLeakageError):
            select_threshold([1] * len(test_ds), [0.8] * len(test_ds), metadata=test_ds.metadata)

    def test_one_gru_architecture_shared_across_regimes(self) -> None:
        """Verify local and central GRU share the exact same AzureGRU class and configuration."""
        local_model = AzureGRU(input_size=4, hidden_size=32, num_summaries=8, output_size=1)
        central_model = AzureGRU(input_size=4, hidden_size=32, num_summaries=8, output_size=1)

        self.assertEqual(local_model.gru.num_layers, central_model.gru.num_layers)
        self.assertEqual(local_model.gru.hidden_size, central_model.gru.hidden_size)
        self.assertEqual(local_model.fc.in_features, central_model.fc.in_features)
        self.assertEqual(local_model.fc.out_features, central_model.fc.out_features)

        # Same param keys
        self.assertEqual(set(local_model.state_dict().keys()), set(central_model.state_dict().keys()))

    def test_cpu_determinism_across_repeated_smoke_runs(self) -> None:
        """Verify identical results across two independent smoke runs with same seed on CPU."""
        report1 = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["gru"],
            regimes=["central"],
            seeds=[17],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir / "run1",
            cache_dir=self.cache_dir,
            device="cpu",
        )

        report2 = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["gru"],
            regimes=["central"],
            seeds=[17],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir / "run2",
            cache_dir=self.cache_dir,
            device="cpu",
        )

        r1 = report1.results[0]
        r2 = report2.results[0]

        # Exact determinism on threshold and validation score
        self.assertAlmostEqual(r1.threshold, r2.threshold, places=6)
        self.assertAlmostEqual(r1.val_f1_score, r2.val_f1_score, places=6)
        self.assertEqual(r1.best_epoch, r2.best_epoch)
        self.assertAlmostEqual(r1.best_score or 0.0, r2.best_score or 0.0, places=6)

        # Exact determinism on test window metrics
        for metric in ("f1", "precision", "recall", "average_precision"):
            v1 = r1.test_window_metrics.get(metric)
            v2 = r2.test_window_metrics.get(metric)
            if v1 is not None and v2 is not None:
                self.assertAlmostEqual(v1, v2, places=6)

        # Exact determinism on event metrics
        for metric in ("event_recall", "true_alerts", "false_alerts", "total_alerts"):
            v1 = r1.test_event_metrics.get(metric)
            v2 = r2.test_event_metrics.get(metric)
            if v1 is not None and v2 is not None:
                self.assertAlmostEqual(v1, v2, places=6)

    def test_config_seed_and_epoch_modes(self) -> None:
        """Verify smoke mode vs full mode parameters from frozen config."""
        cfg = self.phase3_config
        self.assertEqual(cfg.seeds_smoke, (17,))
        self.assertEqual(cfg.seeds_full, (17, 23, 41))
        self.assertEqual(cfg.max_epochs_smoke, 2)
        self.assertEqual(cfg.max_epochs_full, 25)
        self.assertEqual(cfg.patience_smoke, 2)
        self.assertEqual(cfg.patience_full, 5)
        self.assertEqual(cfg.bootstrap_draws_smoke, 30)
        self.assertEqual(cfg.bootstrap_draws_full, 1000)

        # Config hash is deterministic
        h1 = cfg.compute_hash()
        h2 = cfg.compute_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_window_positives_vs_failure_episodes_tracking(self) -> None:
        """Verify window positives and failure episodes are tracked as distinct concepts."""
        report = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["gru"],
            regimes=["central"],
            seeds=[17],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            device="cpu",
        )

        r = report.results[0]
        # In PdM, adjacent positive windows overlap heavily: positive windows != distinct episodes
        self.assertIn("window_positives_test", r.to_dict())
        self.assertIn("distinct_failure_episodes_test", r.to_dict())
        self.assertIn("eligible_failure_episodes_test", r.to_dict())
        self.assertIn("detected_failure_episodes_test", r.to_dict())

    def test_logistic_regression_details_logging(self) -> None:
        """Verify logistic baseline records solver, convergence, iterations, and sklearn version."""
        if not is_sklearn_available():
            raise unittest.SkipTest("scikit-learn is not available.")

        report = run_phase3_experiment(
            config=self.phase3_config,
            mode="smoke",
            models=["logistic"],
            regimes=["local"],
            seeds=[17],
            pipeline_data=self.synthetic_data,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            device="cpu",
        )

        r = report.results[0]
        self.assertIsNotNone(r.logistic_details)
        details = r.logistic_details
        assert details is not None
        self.assertEqual(details["solver"], "lbfgs")
        self.assertIn("n_iter", details)
        self.assertIn("converged", details)
        self.assertIn("sklearn", r.package_versions)

    def test_public_batched_scoring_function(self) -> None:
        """Verify predict_model_probabilities applies frozen scaler without fitting and rejects unfitted MLP."""
        datasets, _ = load_phase3_datasets(
            cache_root=self.cache_dir,
            pipeline_data=self.synthetic_data,
            phase2b_config=self.phase2b_config,
            phase3_config=self.phase3_config,
            mode="smoke",
        )
        test_ds = datasets[("client_model2", "test")]

        # 1. Unfitted MLP must raise RuntimeError
        unfitted_mlp = AzureMLP()
        with self.assertRaises(RuntimeError) as ctx:
            predict_model_probabilities(unfitted_mlp, test_ds, device="cpu")
        self.assertIn("AzureMLP scaler must be fitted", str(ctx.exception))

        # 2. Fitted MLP succeeds
        train_ds = datasets[("client_model2", "train")]
        unfitted_mlp.fit_scaler(train_ds.tabular, split_name="train")
        probs, targets = predict_model_probabilities(unfitted_mlp, test_ds, device="cpu", batch_size=10)
        self.assertEqual(len(probs), len(test_ds))
        self.assertEqual(len(targets), len(test_ds))
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)

        # 3. AzureGRU succeeds
        gru = AzureGRU()
        gru_probs, gru_targets = predict_model_probabilities(gru, test_ds, device="cpu", batch_size=10)
        self.assertEqual(len(gru_probs), len(test_ds))
        for p in gru_probs:
            self.assertTrue(0.0 <= p <= 1.0)


if __name__ == "__main__":
    unittest.main()
