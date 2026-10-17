"""PyTorch GRU backbone for Azure PdM sequence and summary classification.

Specification:
- One GRU layer, input_size=4, hidden_size=32, batch_first=True.
- No GRU dropout.
- Concatenates final hidden state (32-dim) with 8 transformed summaries (8-dim) -> 40-dim.
- One linear output head producing 1 logit per sample.
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


def is_torch_available() -> bool:
    """Return True if PyTorch is importable in the current environment."""
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:
    class AzureGRU(nn.Module):
        """Single-layer GRU with concatenated historical summary head for Azure PdM."""

        def __init__(
            self,
            input_size: int = 4,
            hidden_size: int = 32,
            num_summaries: int = 8,
            output_size: int = 1,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_summaries = num_summaries
            self.output_size = output_size

            # PyTorch 1-layer GRU has no inter-layer dropout
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            # 32 hidden + 8 summaries = 40 features into linear head
            self.fc = nn.Linear(hidden_size + num_summaries, output_size)

        def forward(
            self,
            sequences: torch.Tensor,
            summaries: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                sequences: Tensor of shape [batch_size, 48, 4] with normalized sensor history.
                summaries: Tensor of shape [batch_size, 8] with transformed historical summaries.

            Returns:
                Logits tensor of shape [batch_size, 1].
            """
            if sequences.ndim != 3 or sequences.shape[1] != 48 or sequences.shape[2] != self.input_size:
                raise ValueError(
                    f"Expected sequences tensor of shape [B, 48, {self.input_size}]; got {list(sequences.shape)}"
                )
            if summaries.ndim != 2 or summaries.shape[1] != self.num_summaries:
                raise ValueError(
                    f"Expected summaries tensor of shape [B, {self.num_summaries}]; got {list(summaries.shape)}"
                )
            if sequences.shape[0] != summaries.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: sequences has {sequences.shape[0]}, summaries has {summaries.shape[0]}"
                )

            # sequences: [B, 48, 4]
            # out: [B, 48, 32], h_n: [1, B, 32]
            _, h_n = self.gru(sequences)
            last_hidden = h_n[0]  # [B, 32]

            # Concatenate final hidden state with 8 transformed summaries -> [B, 40]
            combined = torch.cat([last_hidden, summaries], dim=-1)

            # Linear logit output -> [B, 1]
            logits = self.fc(combined)
            return logits

        def get_weights(self) -> Dict[str, torch.Tensor]:
            """Extract detached parameter tensors for federated aggregation."""
            return {name: param.detach().clone() for name, param in self.state_dict().items()}

        def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
            """Load aggregated global parameter tensors."""
            self.load_state_dict(weights)

else:
    class AzureGRU:  # type: ignore
        """Placeholder class when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AzureGRU requires PyTorch ('torch'). Please install PyTorch to use this model."
            )
