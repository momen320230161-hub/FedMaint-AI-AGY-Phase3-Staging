"""Synthetic unit tests for Phase 3 models, scalers, and federated interfaces.

Tests verify:
- TabularStandardScaler strictly enforces training-only fitting and rejects val/test splits.
- Scaler handles zero-variance features with safe fallback scale (1.0).
- Calling transform() before fit() raises RuntimeError.
- AzureLogisticRegression fits scaler only on caller-provided training partition.
- Logistic regression decision function, probability, and class prediction shapes.
- Logistic regression get_weights() and set_weights() enable FedAvg parameter transfer.
- AzureGRU single-layer architecture (input_size=4, hidden_size=32, no dropout, summary concat).
- AzureGRU forward output logit shape [batch_size, 1].
- AzureGRU get_weights() and set_weights() enable FedAvg / FedProx parameter sharing.
- AzureMLP architecture (20 -> 32 -> ReLU -> 1 logit) and output shape [batch_size, 1].
- AzureMLP train-fitted scaler interface and FedAvg parameter transfer.
- Explicit skipping when PyTorch or scikit-learn dependencies are not present.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.splits import HoldoutLeakageError
from fedmaint.models import (
    AzureGRU,
    AzureLogisticRegression,
    AzureMLP,
    TabularStandardScaler,
    is_sklearn_available,
    is_torch_available,
)

if is_torch_available():
    import torch


class TabularStandardScalerTests(unittest.TestCase):
    """Synthetic unit tests for TabularStandardScaler fit scope and leakage guardrails."""

    def test_fit_scope_rejects_validation_and_test(self):
        """Verify scaler refuses to fit on non-training partitions."""
        scaler = TabularStandardScaler()
        train_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        # Fitting with split_name=None or omitting split_name must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit(train_data, split_name=None)
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit(train_data)

        # fit_transform with split_name=None or omitting must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit_transform(train_data, split_name=None)
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit_transform(train_data)

        # Fitting on validation split must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit(train_data, split_name="val")
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit_transform(train_data, split_name="val")

        # Fitting on test split must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit(train_data, split_name="test")

        # Fitting on held-out split must raise HoldoutLeakageError
        with self.assertRaises(HoldoutLeakageError):
            scaler.fit(train_data, split_name="heldout_model1")

        # Fitting on train split must succeed
        scaler.fit(train_data, split_name="train")
        self.assertTrue(scaler.is_fitted)

        # fit_transform on train split must succeed
        transformed = scaler.fit_transform(train_data, split_name="train")
        self.assertEqual(len(transformed), len(train_data))

    def test_transform_before_fit_raises_error(self):
        """Calling transform() on unfitted scaler must raise RuntimeError."""
        scaler = TabularStandardScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform([[1.0, 2.0]])

    def test_zero_variance_fallback(self):
        """Features with zero variance (constant values) must fallback to scale=1.0."""
        scaler = TabularStandardScaler()
        # Col 0 has zero variance (constant 5.0); col 1 has variance
        data = [[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]]
        scaler.fit(data, split_name="train")

        self.assertAlmostEqual(scaler.mean_[0], 5.0)
        self.assertEqual(scaler.scale_[0], 1.0)  # Safe zero-variance scale fallback

        transformed = scaler.transform(data)
        # Scaled col 0 should be (5.0 - 5.0) / 1.0 = 0.0 without division by zero
        for row in transformed:
            self.assertEqual(row[0], 0.0)


class AzureLogisticRegressionTests(unittest.TestCase):
    """Synthetic unit tests for scikit-learn Logistic Regression wrapper."""

    def setUp(self):
        if not is_sklearn_available():
            raise unittest.SkipTest("scikit-learn is not available in the current environment.")

    def test_fit_scope_and_prediction(self):
        """Verify logistic regression fits only on train partition and predicts logits/probabilities."""
        # 10 synthetic samples with 20 tabular features
        X_train = [[float(i + j * 0.1) for j in range(20)] for i in range(10)]
        y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        model = AzureLogisticRegression(max_iter=100)

        # Reject fitting with split_name=None
        with self.assertRaises(HoldoutLeakageError):
            model.fit(X_train, y_train, split_name=None)

        # Reject fitting on validation split
        with self.assertRaises(HoldoutLeakageError):
            model.fit(X_train, y_train, split_name="val")

        # Fit on training split
        model.fit(X_train, y_train, split_name="train")
        self.assertTrue(model.is_fitted)

        # Prediction shapes
        X_eval = [[float(i + j * 0.2) for j in range(20)] for i in range(4)]
        probs = model.predict_proba(X_eval)
        self.assertEqual(probs.shape, (4, 2))

        logits = model.predict_logits(X_eval)
        self.assertEqual(logits.shape, (4,))

        preds = model.predict(X_eval)
        self.assertEqual(preds.shape, (4,))
        for p in preds:
            self.assertIn(p, (0, 1))

    def test_weights_get_and_set_for_fedavg(self):
        """Verify get_weights() and set_weights() enable FedAvg parameter transfer."""
        X_train = [[float(i + j * 0.1) for j in range(20)] for i in range(10)]
        y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        model1 = AzureLogisticRegression(max_iter=100)
        model1.fit(X_train, y_train, split_name="train")
        weights1 = model1.get_weights()

        self.assertIn("coef", weights1)
        self.assertIn("intercept", weights1)
        self.assertEqual(weights1["coef"].shape, (1, 20))
        self.assertEqual(weights1["intercept"].shape, (1,))

        # Inject weights into fresh un-fitted model2
        model2 = AzureLogisticRegression(max_iter=100)
        model2.scaler.fit(X_train, split_name="train")  # Scaler must match
        model2.set_weights(weights1)

        # Predictions from model2 must match model1 exactly
        X_test = [[1.0] * 20]
        self.assertAlmostEqual(model1.predict_logits(X_test)[0], model2.predict_logits(X_test)[0], places=5)


class AzureGRUTests(unittest.TestCase):
    """Synthetic unit tests for AzureGRU PyTorch backbone."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch ('torch') is not available in the current environment.")

    def test_gru_architecture_and_output_shape(self):
        """Verify single-layer GRU, no dropout, summary concat, and logit output shape [B, 1]."""
        model = AzureGRU(input_size=4, hidden_size=32, num_summaries=8, output_size=1)

        # Verify architecture attributes
        self.assertEqual(model.gru.num_layers, 1)
        self.assertTrue(model.gru.batch_first)
        self.assertEqual(model.gru.dropout, 0.0)
        self.assertEqual(model.fc.in_features, 40)  # 32 hidden + 8 summaries
        self.assertEqual(model.fc.out_features, 1)

        # Forward pass with dummy batch of size 3
        batch_size = 3
        sequences = torch.randn(batch_size, 48, 4)
        summaries = torch.randn(batch_size, 8)

        logits = model(sequences, summaries)
        self.assertEqual(logits.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_gru_input_validation(self):
        """Verify shape mismatches in sequences or summaries raise ValueError."""
        model = AzureGRU()
        batch_size = 2

        # Wrong sequence length (e.g. 30 instead of 48)
        with self.assertRaises(ValueError):
            model(torch.randn(batch_size, 30, 4), torch.randn(batch_size, 8))

        # Wrong summary length (e.g. 5 instead of 8)
        with self.assertRaises(ValueError):
            model(torch.randn(batch_size, 48, 4), torch.randn(batch_size, 5))

        # Batch size mismatch between sequences and summaries
        with self.assertRaises(ValueError):
            model(torch.randn(batch_size, 48, 4), torch.randn(batch_size + 1, 8))

    def test_gru_fedavg_weights_transfer(self):
        """Verify get_weights() and set_weights() allow parameter sharing for FedAvg."""
        model1 = AzureGRU()
        weights1 = model1.get_weights()
        self.assertIn("gru.weight_ih_l0", weights1)
        self.assertIn("fc.weight", weights1)

        model2 = AzureGRU()
        model2.set_weights(weights1)

        seq = torch.randn(2, 48, 4)
        summ = torch.randn(2, 8)
        out1 = model1(seq, summ)
        out2 = model2(seq, summ)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


class AzureMLPTests(unittest.TestCase):
    """Synthetic unit tests for AzureMLP PyTorch baseline."""

    def setUp(self):
        if not is_torch_available():
            raise unittest.SkipTest("PyTorch ('torch') is not available in the current environment.")

    def test_mlp_architecture_and_output_shape(self):
        """Verify 20 -> 32 -> ReLU -> 1 logit architecture."""
        model = AzureMLP(input_dim=20, hidden_dim=32, output_dim=1)

        self.assertEqual(model.fc1.in_features, 20)
        self.assertEqual(model.fc1.out_features, 32)
        self.assertEqual(model.fc2.in_features, 32)
        self.assertEqual(model.fc2.out_features, 1)

        # Forward pass
        batch_size = 4
        x = torch.randn(batch_size, 20)
        logits = model(x)
        self.assertEqual(logits.shape, (batch_size, 1))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mlp_scaler_interface_rejects_validation_fit(self):
        """Verify MLP's train-fitted scaler interface enforces train split scope."""
        model = AzureMLP()
        dummy_data = [[1.0] * 20]

        with self.assertRaises(HoldoutLeakageError):
            model.fit_scaler(dummy_data, split_name=None)

        with self.assertRaises(HoldoutLeakageError):
            model.fit_scaler(dummy_data, split_name="val")

        model.fit_scaler(dummy_data, split_name="train")
        self.assertTrue(model.scaler.is_fitted)

    def test_mlp_fedavg_weights_transfer(self):
        """Verify get_weights() and set_weights() enable FedAvg parameter transfer."""
        model1 = AzureMLP()
        weights1 = model1.get_weights()

        model2 = AzureMLP()
        model2.set_weights(weights1)

        x = torch.randn(2, 20)
        out1 = model1(x)
        out2 = model2(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
