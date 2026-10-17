"""Leakage-safe tabular feature scaler for Phase 3 models.

Guarantees:
- Scaler can only be fitted on training partition ('train').
- Attempting to fit on validation, test, or held-out splits raises HoldoutLeakageError or ValueError.
- Calling transform() before fit() raises RuntimeError.
- Zero standard deviation falls back to scale = 1.0 (preventing division by zero).
- Supports both NumPy arrays and standard Python nested lists.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

from ..data.splits import HoldoutLeakageError


class TabularStandardScaler:
    """Standardizes 20-dim tabular features with strict split leakage prevention."""

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-8):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.mean_: Optional[List[float]] = None
        self.scale_: Optional[List[float]] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: Any,
        split_name: Optional[str] = None,
    ) -> "TabularStandardScaler":
        """Compute the mean and standard deviation on training features only.

        Args:
            X: Training feature matrix [N, D] (NumPy array or list of lists).
            split_name: Name of the split. Must be strictly 'train'. Any other value,
                including None, 'val', 'test', or held-out partitions is rejected.

        Returns:
            Fitted scaler instance.

        Raises:
            HoldoutLeakageError: If split_name is not strictly 'train'.
        """
        if split_name != "train":
            raise HoldoutLeakageError(
                f"Data leakage guardrail violation: feature scalers must only be fitted "
                f"on the 'train' partition with explicit split_name='train'. Attempted to fit with split_name={split_name!r}."
            )

        rows = self._to_matrix(X)
        if not rows:
            raise ValueError("Cannot fit scaler on empty feature matrix.")

        n_samples = len(rows)
        n_features = len(rows[0])
        self.n_features_in_ = n_features

        means: List[float] = [0.0] * n_features
        for row in rows:
            if len(row) != n_features:
                raise ValueError(f"Inconsistent feature row length: expected {n_features}, got {len(row)}")
            for j in range(n_features):
                v = float(row[j])
                if not math.isfinite(v):
                    raise ValueError(f"Nonfinite feature value at col {j}: {v}")
                means[j] += v

        for j in range(n_features):
            means[j] /= float(n_samples)

        scales: List[float] = [1.0] * n_features
        if self.with_std:
            for j in range(n_features):
                var = sum((float(row[j]) - means[j]) ** 2 for row in rows) / float(n_samples)
                std = math.sqrt(max(0.0, var))
                if std < self.eps:
                    scales[j] = 1.0  # Safe zero-variance fallback
                else:
                    scales[j] = std

        self.mean_ = means if self.with_mean else [0.0] * n_features
        self.scale_ = scales if self.with_std else [1.0] * n_features
        self.is_fitted = True
        return self

    def transform(self, X: Any) -> Any:
        """Apply fitted standardization to X: (X - mean) / scale.

        Args:
            X: Feature matrix [N, D] to standardize.

        Returns:
            Standardized features matching input container type (NumPy array or nested list).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TabularStandardScaler is not fitted. Call fit() on training partition first.")

        is_np_input = _HAS_NUMPY and isinstance(X, np.ndarray)
        rows = self._to_matrix(X)
        if not rows:
            return np.empty((0, self.n_features_in_), dtype=np.float32) if is_np_input else []

        n_features = self.n_features_in_
        transformed_rows: List[List[float]] = []
        for row_idx, row in enumerate(rows):
            if len(row) != n_features:
                raise ValueError(f"Feature count mismatch at row {row_idx}: expected {n_features}, got {len(row)}")
            trans_row: List[float] = []
            for j in range(n_features):
                v = float(row[j])
                scaled_v = (v - self.mean_[j]) / self.scale_[j]
                trans_row.append(scaled_v)
            transformed_rows.append(trans_row)

        if is_np_input:
            return np.array(transformed_rows, dtype=np.float32)
        return transformed_rows

    def fit_transform(self, X: Any, split_name: Optional[str] = None) -> Any:
        """Fit scaler on training data and return transformed training data."""
        self.fit(X, split_name=split_name)
        return self.transform(X)

    @staticmethod
    def _to_matrix(X: Any) -> List[List[float]]:
        if _HAS_NUMPY and isinstance(X, np.ndarray):
            return X.tolist()  # type: ignore
        return [[float(v) for v in row] for row in X]
