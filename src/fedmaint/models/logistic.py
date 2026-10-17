"""Scikit-learn Logistic Regression wrapper for Azure PdM 20-dim tabular features.

Specification:
- Scikit-learn LogisticRegression classifier on 20 tabular features.
- Strict train-only fitted scaler interface; never fits on validation, test, or heldout.
- Parameter extraction and injection (coef, intercept) for FedAvg/FedProx aggregation.
- Optional scikit-learn import handling with graceful error reporting.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    LogisticRegression = None  # type: ignore
    _SKLEARN_AVAILABLE = False

from .scaler import TabularStandardScaler


def is_sklearn_available() -> bool:
    """Return True if scikit-learn is available in the current environment."""
    return _SKLEARN_AVAILABLE


class AzureLogisticRegression:
    """Leakage-safe scikit-learn Logistic Regression wrapper on 20 tabular features."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        scaler: Optional[TabularStandardScaler] = None,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "AzureLogisticRegression requires scikit-learn ('sklearn'). "
                "Please install scikit-learn to use this model."
            )

        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.scaler = scaler or TabularStandardScaler()

        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver,
        )
        self.is_fitted: bool = False

    def fit(
        self,
        X: Any,
        y: Any,
        split_name: Optional[str] = "train",
    ) -> "AzureLogisticRegression":
        """Fit scaler and logistic regression strictly on caller-provided training partition.

        Args:
            X: Training tabular features [N, 20].
            y: Binary training labels [N].
            split_name: Must be strictly 'train'. Any other value including None, 'val', or 'test' is rejected.

        Returns:
            Fitted wrapper instance.
        """
        # Fit scaler strictly on training partition
        X_scaled = self.scaler.fit_transform(X, split_name=split_name)

        if _HAS_NUMPY:
            y_arr = np.asarray(y)
        else:
            y_arr = list(y)

        self.classifier.fit(X_scaled, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> Any:
        """Predict binary class labels (0 or 1) using training-fitted scaler."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities [N, 2] using training-fitted scaler."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def predict_logits(self, X: Any) -> Any:
        """Compute decision function / raw logit values using training-fitted scaler."""
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        return self.classifier.decision_function(X_scaled)

    def get_weights(self) -> Dict[str, Any]:
        """Extract coefficients and intercept for FedAvg parameter sharing."""
        self._check_fitted()
        return {
            "coef": self.classifier.coef_.copy(),
            "intercept": self.classifier.intercept_.copy(),
        }

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Set model parameters from aggregated global weights."""
        if _HAS_NUMPY:
            self.classifier.coef_ = np.array(weights["coef"], copy=True)
            self.classifier.intercept_ = np.array(weights["intercept"], copy=True)
            self.classifier.classes_ = np.array([0, 1])
            self.classifier.n_features_in_ = int(self.classifier.coef_.shape[1])
        else:
            self.classifier.coef_ = weights["coef"]
            self.classifier.intercept_ = weights["intercept"]
            self.classifier.classes_ = [0, 1]  # type: ignore
            self.classifier.n_features_in_ = len(weights["coef"][0])
        self.is_fitted = True


    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("AzureLogisticRegression is not fitted. Call fit() with training data first.")
