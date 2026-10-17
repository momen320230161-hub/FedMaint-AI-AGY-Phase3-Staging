"""PyTorch MLP baseline for Azure PdM 20-dim tabular features.

Specification:
- Architecture: 20 -> 32 -> ReLU -> 1 logit.
- Train-fitted scaler interface for tabular feature standardization.
- Never fit scaler on validation, test, or held-out partitions.
- PyTorch optional import handling.
- Architecture shareable across local, central, FedAvg, and FedProx training.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _TORCH_AVAILABLE = False

from .scaler import TabularStandardScaler


if _TORCH_AVAILABLE:
    class AzureMLP(nn.Module):
        """Two-layer MLP for Azure PdM tabular features (20 -> 32 -> ReLU -> 1 logit)."""

        def __init__(
            self,
            input_dim: int = 20,
            hidden_dim: int = 32,
            output_dim: int = 1,
            scaler: Optional[TabularStandardScaler] = None,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.scaler = scaler or TabularStandardScaler()

            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Float tensor of shape [batch_size, 20].

            Returns:
                Logits tensor of shape [batch_size, 1].
            """
            if x.ndim != 2 or x.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected tabular features tensor of shape [B, {self.input_dim}]; got {list(x.shape)}"
                )

            h = self.relu(self.fc1(x))
            logits = self.fc2(h)
            return logits

        def fit_scaler(self, X_train: Any, split_name: Optional[str] = "train") -> None:
            """Fit tabular scaler strictly on training partition."""
            self.scaler.fit(X_train, split_name=split_name)

        def transform_features(self, X: Any) -> Any:
            """Apply training-fitted standardization to features."""
            return self.scaler.transform(X)

        def get_weights(self) -> Dict[str, torch.Tensor]:
            """Extract detached parameter tensors for federated aggregation."""
            return {name: param.detach().clone() for name, param in self.state_dict().items()}

        def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
            """Load aggregated global parameter tensors."""
            self.load_state_dict(weights)

else:
    class AzureMLP:  # type: ignore
        """Placeholder class when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AzureMLP requires PyTorch ('torch'). Please install PyTorch to use this model."
            )
