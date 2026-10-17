"""Synthetic unit tests for Phase 3 Azure neural training engine.

Tests verify:
- Split and client leakage rejection: non-train in train, non-val in val, test data,
  and held-out client (model1) data rejected with HoldoutLeakageError or ValueError.
- Regime validation: local requires exactly 1 matching client; central requires all 3.
- AzureGRU single-layer architecture shared, trained, and weights transferred via get/set_weights.
- Loss and optimizer updates: BCEWithLogitsLoss unweighted, AdamW updates model parameters.
- Checkpoint restoration and earliest tie preservation: strict improvement preserves earliest epoch.
- Patience and early stopping: terminates after frozen patience epochs without strict improvement.
- Local vs central macro AP selection with unequal client sizes: macro average weights clients equally.
- Single-class validation subset fails explicitly with informative ValueError.
- Deterministic repeated seed on CPU: identical weights and history across identical seeds regardless of prior RNG state.
- Optimizer step and sample presentation counts: exact cumulative tracking for FL matching.
- Batch shape and float32 finiteness validation: rejects NaN, Inf, and incorrect tensor shapes.
- AzureMLP baseline training with training-fitted tabular scaler:
  - Scaler fitted in memory-bounded streaming passes on training partition only.
  - Frozen train-fitted scaler applied to all train and val inference minibatches.
  - Model receives scaled features (centered ~ 0), not raw values.
- Approved seeds enforcement ([17, 23, 41]) and smoke test override flag.
- Memory metric semantics: true OS peak working set vs current RSS, plus hardware and package logging.
- Explicit skipping when PyTorch is not available.
- All tests are completely synthetic and use no raw data.
"""
from __future__ import annotations

import copy
import math
import sys
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

from fedmaint.data.adapter import NonfiniteValueError, SampleMetadata
from fedmaint.data.splits import HoldoutLeakageError
from fedmaint.evaluation.threshold import HELDOUT_CLIENTS, PARTICIPATING_CLIENTS
from fedmaint.models import (
    AzureGRU,
    AzureMLP,
    TabularStandardScaler,
    is_torch_available,
)
from fedmaint.training import (
    APPROVED_SEEDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    AzureNeuralTrainer,
    ClientDataset,
    EpochLog,
    TrainingConfig,
    TrainingResult,
    fit_tabular_scaler_streaming,
    get_environment_info,
    get_memory_metrics,
    get_peak_memory_mb,
    train_azure_model,
)

if is_torch_available():
    import torch


def _make_metadata(
    n_samples: int,
    client_name: str,
    split_name: str,
    start_time: datetime = datetime(2015, 3, 1, 6),
) -> List[SampleMetadata]:
    """Helper to construct synthetic SampleMetadata sequence."""
    metadata: List[SampleMetadata] = []
    for i in range(n_samples):
        t = start_time + timedelta(hours=i)
        metadata.append(
            SampleMetadata(
                machine_id=100 + (i % 5),
                prediction_time=t,
                client_name=client_name,
                split_name=split_name,
                observation_start=t - timedelta(hours=48),
                observation_end=t - timedelta(hours=1),
                horizon_start=t,
                horizon_end=t + timedelta(hours=24),
            )
        )
    return metadata


def _make_synthetic_gru_data(
    n_samples: int,
    client_name: str,
    split_name: str,
    *,
    seed: int = 42,
    pos_ratio: float = 0.5,
) -> ClientDataset:
    """Helper to construct synthetic GRU sequences, summaries, binary labels, and metadata."""
    rng = np.random.RandomState(seed) if _HAS_NUMPY else None
    metadata = _make_metadata(n_samples, client_name, split_name)

    n_pos = max(1, min(n_samples - 1, int(round(n_samples * pos_ratio))))
    labels_list = [1] * n_pos + [0] * (n_samples - n_pos)
    if rng is not None:
        rng.shuffle(labels_list)
        labels = np.array(labels_list, dtype=np.int64)
        sequences = rng.randn(n_samples, 48, 4).astype(np.float32)
        summaries = rng.randn(n_samples, 8).astype(np.float32)
    else:
        labels = labels_list
        sequences = [[[0.1] * 4 for _ in range(48)] for _ in range(n_samples)]
        summaries = [[0.2] * 8 for _ in range(n_samples)]

    return ClientDataset(
        client_name=client_name,
        split_name=split_name,
        labels=labels,
        metadata=metadata,
        sequences=sequences,
        summaries=summaries,
    )


def _make_synthetic_mlp_data(
    n_samples: int,
    client_name: str,
    split_name: str,
    *,
    seed: int = 42,
    pos_ratio: float = 0.5,
) -> ClientDataset:
    """Helper to construct synthetic MLP tabular features, binary labels, and metadata."""
    rng = np.random.RandomState(seed) if _HAS_NUMPY else None
    metadata = _make_metadata(n_samples, client_name, split_name)

    n_pos = max(1, min(n_samples - 1, int(round(n_samples * pos_ratio))))
    labels_list = [1] * n_pos + [0] * (n_samples - n_pos)
    if rng is not None:
        rng.shuffle(labels_list)
        labels = np.array(labels_list, dtype=np.int64)
        tabular = rng.randn(n_samples, 20).astype(np.float32)
    else:
        labels = labels_list
        tabular = [[0.5] * 20 for _ in range(n_samples)]

    return ClientDataset(
        client_name=client_name,
        split_name=split_name,
        labels=labels,
        metadata=metadata,
        tabular=tabular,
    )


class SplitAndClientLeakageTests(unittest.TestCase):
    """Synthetic unit tests verifying strict fail-closed leakage prevention."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_rejects_test_split_in_train_or_val(self):
        """Passing test split data to train or val raises HoldoutLeakageError."""
        train_bad = _make_synthetic_gru_data(10, "client_model2", "test")
        val_good = _make_synthetic_gru_data(10, "client_model2", "val")
        model = AzureGRU()

        with self.assertRaises(HoldoutLeakageError):
            train_azure_model(model, train_data=train_bad, val_data=val_good, regime="local")

        train_good = _make_synthetic_gru_data(10, "client_model2", "train")
        val_bad = _make_synthetic_gru_data(10, "client_model2", "test")

        with self.assertRaises(HoldoutLeakageError):
            train_azure_model(model, train_data=train_good, val_data=val_bad, regime="local")

    def test_rejects_split_mismatch_across_train_and_val(self):
        """Passing 'val' split in train_data or 'train' split in val_data raises HoldoutLeakageError."""
        model = AzureGRU()

        train_val = _make_synthetic_gru_data(10, "client_model2", "val")
        val_ok = _make_synthetic_gru_data(10, "client_model2", "val")
        with self.assertRaises(HoldoutLeakageError):
            train_azure_model(model, train_data=train_val, val_data=val_ok, regime="local")

        train_ok = _make_synthetic_gru_data(10, "client_model2", "train")
        val_train = _make_synthetic_gru_data(10, "client_model2", "train")
        with self.assertRaises(HoldoutLeakageError):
            train_azure_model(model, train_data=train_ok, val_data=val_train, regime="local")

    def test_rejects_heldout_client_model1(self):
        """Passing heldout_model1 into train_data or val_data strictly raises HoldoutLeakageError."""
        model = AzureGRU()
        train_good = _make_synthetic_gru_data(10, "client_model2", "train")
        val_good = _make_synthetic_gru_data(10, "client_model2", "val")

        heldout_train = _make_synthetic_gru_data(10, "heldout_model1", "train")
        with self.assertRaises(HoldoutLeakageError):
            train_azure_model(model, train_data=heldout_train, val_data=val_good, regime="local")

        heldout_val = _make_synthetic_gru_data(10, "heldout_model1", "val")
        with self.assertRaises(HoldoutLeakageError):
            train_azure_model(model, train_data=train_good, val_data=heldout_val, regime="local")

    def test_rejects_missing_metadata_and_unapproved_client(self):
        """Missing metadata or unknown client raises ValueError."""
        model = AzureGRU()
        train_good = _make_synthetic_gru_data(10, "client_model2", "train")
        val_good = _make_synthetic_gru_data(10, "client_model2", "val")

        train_bad_meta = _make_synthetic_gru_data(10, "client_model2", "train")
        train_bad_meta.metadata = None  # type: ignore
        with self.assertRaises(ValueError):
            train_azure_model(model, train_data=train_bad_meta, val_data=val_good, regime="local")

        unknown_train = _make_synthetic_gru_data(10, "client_model99", "train")
        with self.assertRaises(ValueError):
            train_azure_model(model, train_data=unknown_train, val_data=val_good, regime="local")

    def test_local_regime_requires_exactly_one_matching_client(self):
        """Local regime requires exactly 1 participating client and same client in train and val."""
        model = AzureGRU()
        c2_train = _make_synthetic_gru_data(10, "client_model2", "train")
        c2_val = _make_synthetic_gru_data(10, "client_model2", "val")
        c3_val = _make_synthetic_gru_data(10, "client_model3", "val")

        with self.assertRaises(ValueError):
            train_azure_model(model, train_data=c2_train, val_data=c3_val, regime="local")

        c3_train = _make_synthetic_gru_data(10, "client_model3", "train")
        with self.assertRaises(ValueError):
            train_azure_model(
                model,
                train_data={"client_model2": c2_train, "client_model3": c3_train},
                val_data={"client_model2": c2_val},
                regime="local",
            )

    def test_central_regime_requires_all_three_participating_clients(self):
        """Central regime requires exactly client_model2, client_model3, and client_model4."""
        model = AzureGRU()
        c2_train = _make_synthetic_gru_data(10, "client_model2", "train")
        c3_train = _make_synthetic_gru_data(10, "client_model3", "train")
        c4_train = _make_synthetic_gru_data(10, "client_model4", "train")

        c2_val = _make_synthetic_gru_data(10, "client_model2", "val")
        c3_val = _make_synthetic_gru_data(10, "client_model3", "val")
        c4_val = _make_synthetic_gru_data(10, "client_model4", "val")

        with self.assertRaises(ValueError):
            train_azure_model(
                model,
                train_data={"client_model2": c2_train, "client_model3": c3_train},
                val_data={"client_model2": c2_val, "client_model3": c3_val, "client_model4": c4_val},
                regime="central",
            )

        with self.assertRaises(ValueError):
            train_azure_model(
                model,
                train_data={"client_model2": c2_train, "client_model3": c3_train, "client_model4": c4_train},
                val_data={"client_model3": c3_val, "client_model4": c4_val},
                regime="central",
            )


class AzureGRUTrainingArchitectureTests(unittest.TestCase):
    """Synthetic unit tests for AzureGRU architecture and parameter sharing."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_exact_gru_architecture_and_fedavg_transfer(self):
        """Verify AzureGRU 1-layer, no dropout, 40-dim head, and parameter extraction/transfer."""
        model1 = AzureGRU(input_size=4, hidden_size=32, num_summaries=8, output_size=1)
        self.assertEqual(model1.gru.num_layers, 1)
        self.assertTrue(model1.gru.batch_first)
        self.assertEqual(model1.gru.dropout, 0.0)
        self.assertEqual(model1.fc.in_features, 40)
        self.assertEqual(model1.fc.out_features, 1)

        train_data = _make_synthetic_gru_data(20, "client_model2", "train", seed=1)
        val_data = _make_synthetic_gru_data(20, "client_model2", "val", seed=2)

        config = TrainingConfig(regime="local", max_epochs=1, batch_size=10, seed=17)
        res = train_azure_model(model1, train_data=train_data, val_data=val_data, config=config)
        self.assertEqual(len(res.history), 1)

        weights = model1.get_weights()
        model2 = AzureGRU()
        model2.set_weights(weights)

        device = next(model1.parameters()).device
        model2.to(device)
        seq = torch.randn(4, 48, 4, device=device)
        summ = torch.randn(4, 8, device=device)
        out1 = model1(seq, summ)
        out2 = model2(seq, summ)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


class OptimizationAndLossUpdateTests(unittest.TestCase):
    """Synthetic unit tests verifying loss computation and AdamW parameter updates."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_loss_and_optimizer_updates_parameters(self):
        """Running training for 1 epoch updates model parameters from initial state."""
        model = AzureGRU()

        train_data = _make_synthetic_gru_data(30, "client_model2", "train", seed=10)
        val_data = _make_synthetic_gru_data(20, "client_model2", "val", seed=20)

        config = TrainingConfig(regime="local", max_epochs=1, batch_size=10, lr=1e-3, seed=17)
        trainer = AzureNeuralTrainer(model, config=config)

        # Before training, reinitialize parameters under seed=17 so we capture initial state
        trainer._reset_model_parameters()
        initial_params = [p.detach().clone() for p in model.parameters()]

        result = trainer.train(train_data, val_data)

        self.assertEqual(len(result.history), 1)
        log = result.history[0]
        self.assertTrue(math.isfinite(log.train_loss))
        self.assertGreater(log.train_loss, 0.0)
        self.assertEqual(log.optimizer_steps, 3)
        self.assertEqual(log.samples_processed, 30)

        updated_params = list(model.parameters())
        changed = False
        for p_init, p_upd in zip(initial_params, updated_params):
            if not torch.allclose(p_init, p_upd):
                changed = True
                break
        self.assertTrue(changed, "Model parameters should update after training.")


class CheckpointRestorationAndPatienceTests(unittest.TestCase):
    """Synthetic unit tests for early stopping, patience, and earliest tie preservation."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_checkpoint_restores_best_model_and_earliest_tie(self):
        """Verify strict improvement: tie preserves earliest epoch and restores its weights."""
        model = AzureGRU()

        train_data = _make_synthetic_gru_data(20, "client_model2", "train", seed=1)
        val_data = _make_synthetic_gru_data(20, "client_model2", "val", seed=2)

        original_predict = AzureNeuralTrainer._predict_dataset

        call_count = [0]
        def mock_predict(self_trainer, ds):
            call_count[0] += 1
            if call_count[0] == 1:
                return [0.9, 0.8, 0.7, 0.2], [1, 1, 0, 0]
            elif call_count[0] == 2:
                return [0.9, 0.8, 0.7, 0.2], [1, 1, 0, 0]
            else:
                return [0.2, 0.8, 0.9, 0.1], [1, 1, 0, 0]

        AzureNeuralTrainer._predict_dataset = mock_predict
        try:
            config = TrainingConfig(regime="local", max_epochs=3, batch_size=10, patience=5, seed=17)
            result = train_azure_model(model, train_data, val_data, config=config)

            self.assertEqual(result.best_epoch, 1)
            self.assertFalse(result.early_stopped)
        finally:
            AzureNeuralTrainer._predict_dataset = original_predict

    def test_patience_terminates_training_early(self):
        """Verify training stops after exactly patience epochs without strict improvement."""
        model = AzureGRU()
        train_data = _make_synthetic_gru_data(20, "client_model2", "train", seed=1)
        val_data = _make_synthetic_gru_data(20, "client_model2", "val", seed=2)

        original_predict = AzureNeuralTrainer._predict_dataset

        call_count = [0]
        def mock_predict(self_trainer, ds):
            call_count[0] += 1
            if call_count[0] == 1:
                return [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]
            else:
                return [0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]

        AzureNeuralTrainer._predict_dataset = mock_predict
        try:
            config = TrainingConfig(regime="local", max_epochs=10, batch_size=10, patience=2, seed=17)
            result = train_azure_model(model, train_data, val_data, config=config)

            self.assertEqual(result.best_epoch, 1)
            self.assertTrue(result.early_stopped)
            self.assertEqual(len(result.history), 3)
        finally:
            AzureNeuralTrainer._predict_dataset = original_predict


class LocalVsCentralMacroAPTests(unittest.TestCase):
    """Synthetic unit tests for unweighted macro AP validation selection with unequal client sizes."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_macro_ap_unweighted_across_unequal_client_sizes(self):
        """Central validation score is unweighted macro average, distinct from pooled sample-weighted AP."""
        model = AzureGRU()

        train_dict = {
            "client_model2": _make_synthetic_gru_data(10, "client_model2", "train", seed=1),
            "client_model3": _make_synthetic_gru_data(20, "client_model3", "train", seed=2),
            "client_model4": _make_synthetic_gru_data(30, "client_model4", "train", seed=3),
        }

        val_dict = {
            "client_model2": _make_synthetic_gru_data(4, "client_model2", "val", seed=10),
            "client_model3": _make_synthetic_gru_data(8, "client_model3", "val", seed=20),
            "client_model4": _make_synthetic_gru_data(20, "client_model4", "val", seed=30),
        }

        original_predict = AzureNeuralTrainer._predict_dataset

        def mock_predict(self_trainer, ds):
            if ds.client_name == "client_model2":
                return [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]
            elif ds.client_name == "client_model3":
                return [0.2, 0.8, 0.9, 0.1], [1, 1, 0, 0]
            else:
                return [0.1, 0.2, 0.7, 0.8], [1, 0, 0, 1]

        AzureNeuralTrainer._predict_dataset = mock_predict
        try:
            config = TrainingConfig(regime="central", max_epochs=1, batch_size=10, seed=17)
            result = train_azure_model(model, train_data=train_dict, val_data=val_dict, config=config)

            log = result.history[0]
            self.assertIn("client_model2", log.client_ap)
            self.assertIn("client_model3", log.client_ap)
            self.assertIn("client_model4", log.client_ap)

            ap2 = log.client_ap["client_model2"]
            ap3 = log.client_ap["client_model3"]
            ap4 = log.client_ap["client_model4"]

            expected_macro = (ap2 + ap3 + ap4) / 3.0
            self.assertAlmostEqual(log.macro_ap, expected_macro, places=6)
            self.assertAlmostEqual(log.val_score, expected_macro, places=6)
            self.assertAlmostEqual(result.best_score, expected_macro, places=6)

            config_local = TrainingConfig(regime="local", max_epochs=1, batch_size=10, seed=17)
            model_local = AzureGRU()
            res_local = train_azure_model(
                model_local,
                train_data=train_dict["client_model2"],
                val_data=val_dict["client_model2"],
                config=config_local,
            )
            self.assertAlmostEqual(res_local.best_score, ap2, places=6)
        finally:
            AzureNeuralTrainer._predict_dataset = original_predict

    def test_single_class_validation_fails_closed(self):
        """Single-class validation subset where AP is undefined raises ValueError explicitly."""
        model = AzureGRU()
        train_data = _make_synthetic_gru_data(10, "client_model2", "train")

        val_all_zeros = _make_synthetic_gru_data(10, "client_model2", "val")
        if _HAS_NUMPY:
            val_all_zeros.labels = np.zeros(10, dtype=np.int64)
        else:
            val_all_zeros.labels = [0] * 10

        config = TrainingConfig(regime="local", max_epochs=1, batch_size=10)
        with self.assertRaises(ValueError) as ctx:
            train_azure_model(model, train_data=train_data, val_data=val_all_zeros, config=config)

        self.assertIn("Average Precision is undefined", str(ctx.exception))


class DeterminismAndStepCountTests(unittest.TestCase):
    """Synthetic unit tests for CPU determinism, RNG state reset, and compute tracking."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_deterministic_repeated_seed_on_cpu(self):
        """Training twice with the same seed on CPU produces identical losses and weights."""
        train_data1 = _make_synthetic_gru_data(30, "client_model2", "train", seed=100)
        val_data1 = _make_synthetic_gru_data(20, "client_model2", "val", seed=200)

        # Run 1 with seed=17 on CPU
        config1 = TrainingConfig(regime="local", max_epochs=2, batch_size=10, seed=17, device="cpu")
        model1 = AzureGRU()
        res1 = train_azure_model(model1, train_data1, val_data1, config=config1)

        # Run 2 with seed=17 on CPU
        train_data2 = _make_synthetic_gru_data(30, "client_model2", "train", seed=100)
        val_data2 = _make_synthetic_gru_data(20, "client_model2", "val", seed=200)
        config2 = TrainingConfig(regime="local", max_epochs=2, batch_size=10, seed=17, device="cpu")
        model2 = AzureGRU()
        res2 = train_azure_model(model2, train_data2, val_data2, config=config2)

        for log1, log2 in zip(res1.history, res2.history):
            self.assertAlmostEqual(log1.train_loss, log2.train_loss, places=6)
            self.assertAlmostEqual(log1.val_score, log2.val_score, places=6)

        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        for k in weights1:
            self.assertTrue(torch.allclose(weights1[k], weights2[k], atol=1e-6))

        # Run 3 with seed=23 should produce different weights
        config3 = TrainingConfig(regime="local", max_epochs=2, batch_size=10, seed=23, device="cpu")
        model3 = AzureGRU()
        res3 = train_azure_model(model3, train_data1, val_data1, config=config3)
        weights3 = model3.get_weights()
        is_different = not torch.allclose(weights1["fc.weight"], weights3["fc.weight"], atol=1e-4)
        self.assertTrue(is_different)

    def test_deterministic_reinitialization_across_different_prior_rng(self):
        """Verify models initialized under different global RNG states produce identical weights under same seed."""
        train_data = _make_synthetic_gru_data(30, "client_model2", "train", seed=10)
        val_data = _make_synthetic_gru_data(20, "client_model2", "val", seed=20)

        # Pre-seed Run A with arbitrary RNG disturbance
        torch.manual_seed(999999)
        _ = torch.randn(10000)
        modelA = AzureGRU()
        initial_pre_A = {k: v.clone() for k, v in modelA.get_weights().items()}

        # Pre-seed Run B with a completely different sequence
        torch.manual_seed(111111)
        _ = torch.randn(50000)
        modelB = AzureGRU()
        initial_pre_B = {k: v.clone() for k, v in modelB.get_weights().items()}

        # Verify their pre-construction weights differ
        self.assertFalse(torch.allclose(initial_pre_A["fc.weight"], initial_pre_B["fc.weight"]))

        # Train both models under frozen seed=17 on CPU
        configA = TrainingConfig(regime="local", max_epochs=2, batch_size=10, seed=17, device="cpu", reset_parameters=True)
        resA = train_azure_model(modelA, train_data, val_data, config=configA)

        configB = TrainingConfig(regime="local", max_epochs=2, batch_size=10, seed=17, device="cpu", reset_parameters=True)
        resB = train_azure_model(modelB, train_data, val_data, config=configB)

        # After fresh training reset under seed=17, final weights must match bitwise
        weightsA = modelA.get_weights()
        weightsB = modelB.get_weights()
        for k in weightsA:
            self.assertTrue(torch.allclose(weightsA[k], weightsB[k], atol=1e-7))

        # Histories must match
        for logA, logB in zip(resA.history, resB.history):
            self.assertAlmostEqual(logA.train_loss, logB.train_loss, places=6)
            self.assertAlmostEqual(logA.val_score, logB.val_score, places=6)

    def test_approved_seeds_enforcement_and_override(self):
        """Verify only approved seeds [17, 23, 41] are accepted unless allow_seed_override=True."""
        model = AzureGRU()
        train_data = _make_synthetic_gru_data(10, "client_model2", "train")
        val_data = _make_synthetic_gru_data(10, "client_model2", "val")

        for approved in APPROVED_SEEDS:
            config = TrainingConfig(regime="local", max_epochs=1, batch_size=10, seed=approved)
            trainer = AzureNeuralTrainer(model, config=config)
            self.assertEqual(trainer.config.seed, approved)

        # Non-approved seed raises ValueError
        bad_config = TrainingConfig(regime="local", max_epochs=1, batch_size=10, seed=999, allow_seed_override=False)
        with self.assertRaises(ValueError) as ctx:
            AzureNeuralTrainer(model, config=bad_config)
        self.assertIn("approved protocol seeds", str(ctx.exception))

        # Smoke override permits non-protocol seed
        override_config = TrainingConfig(regime="local", max_epochs=1, batch_size=10, seed=999, allow_seed_override=True)
        trainer_override = AzureNeuralTrainer(model, config=override_config)
        self.assertEqual(trainer_override.config.seed, 999)

    def test_cumulative_optimizer_steps_and_sample_counts(self):
        """Verify exact cumulative optimizer steps and sample counts across epochs."""
        n_samples = 45
        batch_size = 20
        train_data = _make_synthetic_gru_data(n_samples, "client_model2", "train", seed=5)
        val_data = _make_synthetic_gru_data(10, "client_model2", "val", seed=6)

        config = TrainingConfig(regime="local", max_epochs=3, batch_size=batch_size, seed=17)
        model = AzureGRU()
        res = train_azure_model(model, train_data, val_data, config=config)

        self.assertEqual(len(res.history), 3)

        self.assertEqual(res.history[0].optimizer_steps, 3)
        self.assertEqual(res.history[0].samples_processed, 45)

        self.assertEqual(res.history[1].optimizer_steps, 6)
        self.assertEqual(res.history[1].samples_processed, 90)

        self.assertEqual(res.history[2].optimizer_steps, 9)
        self.assertEqual(res.history[2].samples_processed, 135)

        self.assertEqual(res.total_optimizer_steps, 9)
        self.assertEqual(res.total_samples_processed, 135)

    def test_memory_metrics_semantics_and_environment_logging(self):
        """Verify true peak memory vs current RSS semantics and hardware/dependency logging."""
        model = AzureGRU()
        train_data = _make_synthetic_gru_data(20, "client_model2", "train")
        val_data = _make_synthetic_gru_data(10, "client_model2", "val")

        config = TrainingConfig(regime="local", max_epochs=1, batch_size=10, seed=17)
        res = train_azure_model(model, train_data, val_data, config=config)

        log = res.history[0]
        # Check both fields exist in dataclass and dict
        self.assertIn("peak_memory_mb", log.to_dict())
        self.assertIn("process_rss_mb", log.to_dict())

        # Environment info checks
        self.assertIsNotNone(res.environment_info)
        env = res.environment_info
        self.assertIn("python", env)
        self.assertIn("platform", env)
        self.assertIn("cpu", env)
        self.assertIn("torch", env)
        self.assertIn("numpy", env)
        self.assertIn("sklearn", env)
        self.assertIn("torch", res.package_versions)


class BatchShapeAndNonfiniteValidationTests(unittest.TestCase):
    """Synthetic unit tests for input shape validation and float32 finiteness."""

    def test_invalid_tensor_shapes_rejected(self):
        """Mismatched sequence, summary, or tabular shapes raise ValueError."""
        if _HAS_NUMPY:
            bad_seq = np.zeros((10, 30, 4), dtype=np.float32)
            good_sum = np.zeros((10, 8), dtype=np.float32)
            lbls = np.zeros(10, dtype=np.int64)
        else:
            bad_seq = [[[0.0] * 4 for _ in range(30)] for _ in range(10)]
            good_sum = [[0.0] * 8 for _ in range(10)]
            lbls = [0] * 10

        meta = _make_metadata(10, "client_model2", "train")
        ds = ClientDataset("client_model2", "train", lbls, meta, sequences=bad_seq, summaries=good_sum)
        with self.assertRaises(ValueError):
            ds.validate()

        if _HAS_NUMPY:
            good_seq = np.zeros((10, 48, 4), dtype=np.float32)
            bad_sum = np.zeros((10, 5), dtype=np.float32)
        else:
            good_seq = [[[0.0] * 4 for _ in range(48)] for _ in range(10)]
            bad_sum = [[0.0] * 5 for _ in range(10)]

        ds_bad_sum = ClientDataset("client_model2", "train", lbls, meta, sequences=good_seq, summaries=bad_sum)
        with self.assertRaises(ValueError):
            ds_bad_sum.validate()

        if _HAS_NUMPY:
            bad_tab = np.zeros((10, 15), dtype=np.float32)
        else:
            bad_tab = [[0.0] * 15 for _ in range(10)]

        ds_bad_tab = ClientDataset("client_model2", "train", lbls, meta, tabular=bad_tab)
        with self.assertRaises(ValueError):
            ds_bad_tab.validate()

    def test_nonfinite_values_rejected(self):
        """NaN and Inf values in sequences, summaries, or tabular raise NonfiniteValueError."""
        meta = _make_metadata(5, "client_model2", "train")
        lbls = [0] * 5

        if _HAS_NUMPY:
            nan_seq = np.zeros((5, 48, 4), dtype=np.float32)
            nan_seq[0, 0, 0] = np.nan
            good_sum = np.zeros((5, 8), dtype=np.float32)
        else:
            nan_seq = [[[float("nan")] * 4 for _ in range(48)] for _ in range(5)]
            good_sum = [[0.0] * 8 for _ in range(5)]

        ds_nan = ClientDataset("client_model2", "train", lbls, meta, sequences=nan_seq, summaries=good_sum)
        with self.assertRaises(NonfiniteValueError):
            ds_nan.validate()

        if _HAS_NUMPY:
            good_seq = np.zeros((5, 48, 4), dtype=np.float32)
            inf_sum = np.zeros((5, 8), dtype=np.float32)
            inf_sum[1, 1] = np.inf
        else:
            good_seq = [[[0.0] * 4 for _ in range(48)] for _ in range(5)]
            inf_sum = [[float("inf")] * 8 for _ in range(5)]

        ds_inf = ClientDataset("client_model2", "train", lbls, meta, sequences=good_seq, summaries=inf_sum)
        with self.assertRaises(NonfiniteValueError):
            ds_inf.validate()


class AzureMLPBaselineTrainingTests(unittest.TestCase):
    """Synthetic unit tests for AzureMLP baseline training with tabular features."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch is not available in the current environment.")

    def test_mlp_training_with_scaler_fit(self):
        """AzureMLP baseline trains successfully on 20-dim tabular features and fits scaler on train."""
        mlp = AzureMLP(input_dim=20, hidden_dim=32, output_dim=1)
        self.assertFalse(mlp.scaler.is_fitted)

        train_data = _make_synthetic_mlp_data(30, "client_model2", "train", seed=11)
        val_data = _make_synthetic_mlp_data(20, "client_model2", "val", seed=22)

        config = TrainingConfig(regime="local", max_epochs=2, batch_size=10, seed=17)
        result = train_azure_model(mlp, train_data=train_data, val_data=val_data, config=config)

        self.assertTrue(mlp.scaler.is_fitted)
        self.assertEqual(len(result.history), 2)
        self.assertTrue(math.isfinite(result.history[0].train_loss))
        self.assertTrue(math.isfinite(result.history[0].val_score))

    def test_mlp_receives_scaled_features_with_frozen_train_stats(self):
        """Verify AzureMLP receives scaled features rather than raw values, and validation uses frozen train stats."""
        mlp = AzureMLP(input_dim=20, hidden_dim=32, output_dim=1)

        # Nonzero mean and unequal scales in training data
        n_train = 60
        raw_train = []
        rng = np.random.RandomState(42) if _HAS_NUMPY else None

        for i in range(n_train):
            row = [50.0] * 20
            if rng is not None:
                row[0] = 150.0 + rng.randn() * 25.0
                row[1] = -80.0 + rng.randn() * 10.0
            else:
                row[0] = 150.0 + (i - 30) * 1.0
                row[1] = -80.0 + (i - 30) * 0.5
            raw_train.append(row)

        train_data = ClientDataset(
            client_name="client_model2",
            split_name="train",
            labels=[i % 2 for i in range(n_train)],
            metadata=_make_metadata(n_train, "client_model2", "train"),
            tabular=np.array(raw_train, dtype=np.float32) if _HAS_NUMPY else raw_train,
        )

        # Validation data with distinct distribution, but feature 0 has raw value equal to training mean (150.0)
        n_val = 10
        raw_val = []
        for i in range(n_val):
            row = [500.0] * 20
            row[0] = 150.0
            raw_val.append(row)

        val_data = ClientDataset(
            client_name="client_model2",
            split_name="val",
            labels=[i % 2 for i in range(n_val)],
            metadata=_make_metadata(n_val, "client_model2", "val"),
            tabular=np.array(raw_val, dtype=np.float32) if _HAS_NUMPY else raw_val,
        )

        captured_inputs = []

        def hook_fn(module, input_tuple):
            captured_inputs.append(input_tuple[0].detach().cpu().clone())

        handle = mlp.fc1.register_forward_pre_hook(hook_fn)
        try:
            config = TrainingConfig(regime="local", max_epochs=1, batch_size=30, seed=17)
            _ = train_azure_model(mlp, train_data=train_data, val_data=val_data, config=config)

            # Scaler fitted strictly on train
            self.assertTrue(mlp.scaler.is_fitted)
            self.assertAlmostEqual(mlp.scaler.mean_[0], 150.0, delta=5.0)
            self.assertGreater(mlp.scaler.scale_[0], 10.0)

            # Train minibatch input to fc1 must be standardized near 0, NOT raw 150.0
            train_batch_0 = captured_inputs[0]
            mean_f0_received = float(train_batch_0[:, 0].mean().item())
            self.assertAlmostEqual(mean_f0_received, 0.0, delta=0.5)
            self.assertLess(abs(mean_f0_received), 10.0)

            # Val minibatch input: feature 0 (raw 150.0) scaled by train stats must be near 0.0
            val_batch = captured_inputs[-1]
            val_f0_received = float(val_batch[0, 0].item())
            self.assertAlmostEqual(val_f0_received, 0.0, delta=0.2)

            # Val feature 2 (raw 500.0) scaled by train stats (mean~50, std~1) must be > 100
            val_f2_received = float(val_batch[0, 2].item())
            self.assertGreater(val_f2_received, 100.0)
        finally:
            handle.remove()

    def test_streaming_scaler_fit_rejects_leakage(self):
        """fit_tabular_scaler_streaming rejects validation or test datasets with HoldoutLeakageError."""
        scaler = TabularStandardScaler()
        val_data = _make_synthetic_mlp_data(10, "client_model2", "val")

        with self.assertRaises(HoldoutLeakageError):
            fit_tabular_scaler_streaming(scaler, [val_data], split_name="val")

        with self.assertRaises(HoldoutLeakageError):
            fit_tabular_scaler_streaming(scaler, [val_data], split_name="train")


if __name__ == "__main__":
    unittest.main()
