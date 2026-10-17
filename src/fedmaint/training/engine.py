"""Reusable PyTorch neural training engine for Azure PdM predictive backbones.

Specification:
- Supports AzureGRU ([N, 48, 4] normalized sequence + [N, 8] signed_log1p summaries)
  and AzureMLP ([N, 20] training-scaled tabular baseline).
- Strict split and client metadata validation for EVERY sample: rejects missing metadata,
  split_name != "train" for training, split_name != "val" for validation, test data,
  and held-out client (model1) with HoldoutLeakageError or ValueError.
- Regimes:
  - 'local': requires exactly 1 participating client in train and val, and same client.
  - 'central': requires all 3 participating clients (model2, model3, model4) in train and val.
- Checkpoint selection based on validation Average Precision:
  - Local score = own-client val AP.
  - Central score = unweighted macro AP across model2/3/4 val.
  - Fails explicitly if AP is undefined due to single-class client validation.
  - Strict improvement (no update on tie) preserves earliest epoch.
  - Deep copy state_dict of best model; restores best weights before return.
  - Early stopping patience = 5.
- Training defaults:
  - Unweighted BCEWithLogitsLoss, no oversampling.
  - AdamW optimizer (lr=1e-3, weight_decay=1e-4), constant learning rate.
  - Batch size 512, max_epochs 25, patience 5, seeds [17, 23, 41].
  - Deterministic shuffling and seeding across Python, NumPy, torch, CUDA.
  - Fresh training parameter reinitialization under seed; preserves future FL continuation when disabled.
  - Device CPU fallback.
  - Float32 finiteness and shape validation.
- Scaling behavior:
  - Train-fitted TabularStandardScaler computed via memory-bounded streaming passes over training data only.
  - Same frozen train-fitted scaler applied to every MLP train minibatch and validation inference minibatch.
  - Never fits on validation, test, or held-out partitions.
- Memory & Compute:
  - Accepts NumPy arrays, memmaps, or torch tensors per client with metadata stored separately.
  - Batching slices minibatches without copying entire pooled arrays to GPU.
  - Tracks cumulative optimizer steps and sample presentations for later FL compute matching.
  - Exposes epoch logs with loss, APs, steps, samples, seed, device, time, true peak memory, and current RSS.
  - Records hardware (CPU, GPU) and package versions (torch, numpy, sklearn, python).
"""
from __future__ import annotations

import collections.abc
import copy
import math
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _TORCH_AVAILABLE = False

try:
    import sklearn
    _SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None  # type: ignore
    _SKLEARN_AVAILABLE = False

from ..data.adapter import (
    FLOAT32_MAX,
    FLOAT32_MIN,
    ModelReadyBatch,
    NonfiniteValueError,
    SampleMetadata,
)
from ..data.splits import HoldoutLeakageError
from ..evaluation.threshold import HELDOUT_CLIENTS, PARTICIPATING_CLIENTS
from ..evaluation.window import evaluate_window_metrics
from ..models.gru import AzureGRU, is_torch_available
from ..models.mlp import AzureMLP
from ..models.scaler import TabularStandardScaler

APPROVED_SEEDS: Tuple[int, ...] = (17, 23, 41)
DEFAULT_BATCH_SIZE: int = 512
DEFAULT_MAX_EPOCHS: int = 25
DEFAULT_PATIENCE: int = 5
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4


@dataclass
class TrainingConfig:
    """Frozen training hyperparameters and engine configuration."""
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    seed: int = 17
    regime: str = "local"  # 'local' or 'central'
    device: Optional[str] = None  # None selects cuda if available, else cpu
    deterministic: bool = True
    fit_mlp_scaler: bool = True
    reset_parameters: bool = True  # Reinitialize model parameters under seed for fresh training; False for FL continuation
    allow_seed_override: bool = False  # Permit non-protocol seeds for smoke testing

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1; got {self.max_epochs}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1; got {self.patience}")
        if self.lr <= 0.0 or not math.isfinite(self.lr):
            raise ValueError(f"lr must be positive finite float; got {self.lr}")
        if self.weight_decay < 0.0 or not math.isfinite(self.weight_decay):
            raise ValueError(f"weight_decay must be non-negative finite float; got {self.weight_decay}")
        if self.regime not in ("local", "central"):
            raise ValueError(f"regime must be 'local' or 'central'; got {self.regime!r}")


@dataclass(frozen=True)
class EpochLog:
    """Telemetry and metric log for a single training epoch."""
    epoch: int
    train_loss: float
    client_ap: Dict[str, float]
    macro_ap: float
    val_score: float
    optimizer_steps: int
    samples_processed: int
    seed: int
    device: str
    elapsed_seconds: float
    peak_memory_mb: Optional[float]
    process_rss_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "client_ap": dict(self.client_ap),
            "macro_ap": self.macro_ap,
            "val_score": self.val_score,
            "optimizer_steps": self.optimizer_steps,
            "samples_processed": self.samples_processed,
            "seed": self.seed,
            "device": self.device,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "process_rss_mb": self.process_rss_mb,
        }


@dataclass
class TrainingResult:
    """Summary of training outcome, best checkpoint, compute tracking, and epoch logs."""
    best_epoch: int
    best_score: float
    best_state_dict: Dict[str, Any]
    history: List[EpochLog]
    total_optimizer_steps: int
    total_samples_processed: int
    total_elapsed_seconds: float
    package_versions: Dict[str, Optional[str]]
    regime: str
    seed: int
    device: str
    early_stopped: bool
    environment_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "history": [log.to_dict() for log in self.history],
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_samples_processed": self.total_samples_processed,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "package_versions": self.package_versions,
            "environment_info": self.environment_info,
            "regime": self.regime,
            "seed": self.seed,
            "device": self.device,
            "early_stopped": self.early_stopped,
        }


class ClientDataset:
    """Client partition container holding arrays, memmaps, or tensors and tracking metadata."""

    def __init__(
        self,
        client_name: str,
        split_name: str,
        labels: Any,
        metadata: Sequence[Any],
        *,
        sequences: Optional[Any] = None,
        summaries: Optional[Any] = None,
        tabular: Optional[Any] = None,
        machine_ids: Optional[Tuple[int, ...]] = None,
    ):
        self.client_name = str(client_name)
        self.split_name = str(split_name)
        self.labels = labels
        if metadata is not None:
            if isinstance(metadata, tuple):
                self.metadata = metadata
            elif isinstance(metadata, collections.abc.Sequence) and not isinstance(metadata, (list, str, bytes)):
                self.metadata = metadata
            else:
                self.metadata = tuple(metadata)
        else:
            self.metadata = None
        self.sequences = sequences
        self.summaries = summaries
        self.tabular = tabular
        self.machine_ids = tuple(machine_ids) if machine_ids is not None else None
        self.cohort_machine_ids = self.machine_ids

    def __len__(self) -> int:
        if self.labels is not None:
            return len(self.labels)
        if self.metadata is not None:
            return len(self.metadata)
        if self.sequences is not None:
            return len(self.sequences)
        if self.tabular is not None:
            return len(self.tabular)
        return 0

    def validate(self, expected_split: Optional[str] = None) -> None:
        """Validate shapes, float32 finiteness, binary labels, and sample metadata."""
        n_samples = len(self)
        if n_samples == 0:
            raise ValueError(f"ClientDataset for '{self.client_name}' is empty.")

        if self.metadata is None:
            raise ValueError(f"ClientDataset for '{self.client_name}' is missing required metadata.")

        if len(self.metadata) != n_samples:
            raise ValueError(
                f"Metadata count ({len(self.metadata)}) does not match sample count ({n_samples}) "
                f"for client '{self.client_name}'."
            )

        # 1. Validate every sample metadata item
        req_split = expected_split or self.split_name
        for idx, m in enumerate(self.metadata):
            if m is None:
                raise ValueError(f"Sample metadata at index {idx} is None for client '{self.client_name}'.")

            m_split = getattr(m, "split_name", None) or (m.get("split_name") if isinstance(m, dict) else None)
            m_client = getattr(m, "client_name", None) or (m.get("client_name") if isinstance(m, dict) else None)

            if m_split is None:
                raise ValueError(f"Metadata at index {idx} missing split_name: {m}")
            if m_client is None:
                raise ValueError(f"Metadata at index {idx} missing client_name: {m}")

            # Disallow test split
            if m_split == "test":
                raise HoldoutLeakageError(
                    f"Data leakage violation: sample {idx} has split_name='test'. "
                    f"Test data must never enter training or validation."
                )

            # Check split match
            if req_split is not None and m_split != req_split:
                raise HoldoutLeakageError(
                    f"Data leakage violation: sample {idx} has split_name='{m_split}', "
                    f"expected '{req_split}'."
                )

            # Check heldout client
            if (
                m_client in HELDOUT_CLIENTS
                or (isinstance(m_client, str) and m_client.startswith("heldout"))
            ):
                raise HoldoutLeakageError(
                    f"Held-out client data violation: sample {idx} belongs to '{m_client}'. "
                    f"Held-out client data must never enter training or validation."
                )

            # Check participating client
            if m_client not in PARTICIPATING_CLIENTS:
                raise ValueError(
                    f"Unauthorized client '{m_client}' at index {idx}. Approved participating clients: "
                    f"{sorted(PARTICIPATING_CLIENTS)}."
                )

            # Check consistency with container client_name
            if m_client != self.client_name:
                raise ValueError(
                    f"Metadata client mismatch at index {idx}: sample has '{m_client}', "
                    f"container is '{self.client_name}'."
                )

        # 2. Validate labels
        lbls = self.labels
        if hasattr(lbls, "tolist"):
            lbl_list = lbls.tolist()
        else:
            lbl_list = list(lbls)

        if len(lbl_list) != n_samples:
            raise ValueError(f"Labels length ({len(lbl_list)}) does not match sample count ({n_samples}).")

        for idx, y in enumerate(lbl_list):
            if y not in (0, 1) or isinstance(y, bool):
                raise ValueError(f"Invalid label at index {idx}: {y}. Expected binary 0 or 1.")

        # 3. Validate features
        if self.sequences is not None:
            if len(self.sequences) != n_samples:
                raise ValueError(f"Sequences length ({len(self.sequences)}) does not match samples ({n_samples}).")
            if hasattr(self.sequences, "shape"):
                if len(self.sequences.shape) != 3 or self.sequences.shape[1] != 48 or self.sequences.shape[2] != 4:
                    raise ValueError(
                        f"Expected sequences shape [N, 48, 4]; got {list(self.sequences.shape)}"
                    )
            _validate_finite_array(self.sequences, "sequences")

        if self.summaries is not None:
            if len(self.summaries) != n_samples:
                raise ValueError(f"Summaries length ({len(self.summaries)}) does not match samples ({n_samples}).")
            if hasattr(self.summaries, "shape"):
                if len(self.summaries.shape) != 2 or self.summaries.shape[1] != 8:
                    raise ValueError(
                        f"Expected summaries shape [N, 8]; got {list(self.summaries.shape)}"
                    )
            _validate_finite_array(self.summaries, "summaries")

        if self.tabular is not None:
            if len(self.tabular) != n_samples:
                raise ValueError(f"Tabular length ({len(self.tabular)}) does not match samples ({n_samples}).")
            if hasattr(self.tabular, "shape"):
                if len(self.tabular.shape) != 2 or self.tabular.shape[1] != 20:
                    raise ValueError(
                        f"Expected tabular shape [N, 20]; got {list(self.tabular.shape)}"
                    )
            _validate_finite_array(self.tabular, "tabular")

    @classmethod
    def from_batches(
        cls,
        batches: Sequence[ModelReadyBatch],
        client_name: str,
        split_name: str,
    ) -> "ClientDataset":
        """Assemble a ClientDataset from adapted ModelReadyBatch instances."""
        if not batches:
            raise ValueError("Cannot create ClientDataset from empty batches sequence.")

        all_meta: List[SampleMetadata] = []
        all_lbls: List[int] = []
        seq_list: List[Any] = []
        sum_list: List[Any] = []
        tab_list: List[Any] = []

        for b in batches:
            all_meta.extend(b.metadata)
            if hasattr(b.labels, "tolist"):
                all_lbls.extend(b.labels.tolist())
            else:
                all_lbls.extend(b.labels)

            seq_list.append(b.sequences)
            sum_list.append(b.summaries)
            tab_list.append(b.tabular)

        if _HAS_NUMPY:
            sequences = np.concatenate(seq_list, axis=0) if seq_list[0] is not None else None
            summaries = np.concatenate(sum_list, axis=0) if sum_list[0] is not None else None
            tabular = np.concatenate(tab_list, axis=0) if tab_list[0] is not None else None
            labels = np.array(all_lbls, dtype=np.int64)
        else:
            sequences = [item for sub in seq_list for item in sub] if seq_list[0] is not None else None
            summaries = [item for sub in sum_list for item in sub] if sum_list[0] is not None else None
            tabular = [item for sub in tab_list for item in sub] if tab_list[0] is not None else None
            labels = all_lbls

        dataset = cls(
            client_name=client_name,
            split_name=split_name,
            labels=labels,
            metadata=all_meta,
            sequences=sequences,
            summaries=summaries,
            tabular=tabular,
        )
        dataset.validate()
        return dataset


def _validate_finite_array(arr: Any, name: str) -> None:
    """Validate that array/tensor values are finite float32."""
    if _TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        if not torch.isfinite(arr).all():
            raise NonfiniteValueError(f"Tensor '{name}' contains NaN or Inf.")
        max_v = float(torch.max(torch.abs(arr)).item())
        if max_v > FLOAT32_MAX:
            raise NonfiniteValueError(f"Tensor '{name}' overflows float32 bounds: max abs={max_v}")
        return

    if _HAS_NUMPY and isinstance(arr, np.ndarray):
        if not np.all(np.isfinite(arr)):
            raise NonfiniteValueError(f"NumPy array '{name}' contains NaN or Inf.")
        max_v = float(np.max(np.abs(arr)))
        if max_v > FLOAT32_MAX:
            raise NonfiniteValueError(f"NumPy array '{name}' overflows float32 bounds: max abs={max_v}")
        return

    # Nested list fallback
    def _check_nested(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for item in v:
                _check_nested(item)
        else:
            fv = float(v)
            if not math.isfinite(fv):
                raise NonfiniteValueError(f"Value in '{name}' is nonfinite: {fv}")
            if fv > FLOAT32_MAX or fv < FLOAT32_MIN:
                raise NonfiniteValueError(f"Value in '{name}' overflows float32 range: {fv}")

    _check_nested(arr)


def fit_tabular_scaler_streaming(
    scaler: TabularStandardScaler,
    datasets: Iterable[ClientDataset],
    split_name: str = "train",
    eps: float = 1e-8,
) -> TabularStandardScaler:
    """Fit TabularStandardScaler across client datasets in memory-bounded streaming passes.

    Validates that split_name is strictly 'train'. Never fits on validation, test,
    or held-out partitions. Avoids large concatenated arrays.

    Args:
        scaler: TabularStandardScaler instance to fit.
        datasets: Iterable of training ClientDataset partitions.
        split_name: Must be strictly 'train'.
        eps: Zero-variance fallback threshold.

    Returns:
        Fitted TabularStandardScaler instance.

    Raises:
        HoldoutLeakageError: If split_name or any dataset's split_name is not strictly 'train'.
        ValueError: If datasets are empty or feature shape is not 20.
        NonfiniteValueError: If any tabular feature is nonfinite.
    """
    if split_name != "train":
        raise HoldoutLeakageError(
            f"Data leakage guardrail violation: feature scalers must only be fitted "
            f"on the 'train' partition with explicit split_name='train'. Attempted with split_name={split_name!r}."
        )

    ds_list = list(datasets)
    for ds in ds_list:
        if ds.split_name != "train":
            raise HoldoutLeakageError(
                f"Data leakage guardrail violation: dataset '{ds.client_name}' has split_name='{ds.split_name}', "
                f"expected 'train'."
            )

    n_features = 20
    n_total_samples = sum(len(d) for d in ds_list)
    if n_total_samples == 0:
        raise ValueError("Cannot fit scaler on empty datasets.")

    # Pass 1: compute sums for mean
    sums = [0.0] * n_features
    for ds in ds_list:
        tab = ds.tabular
        if tab is None:
            raise ValueError(f"Dataset '{ds.client_name}' has no tabular features.")
        if _HAS_NUMPY and isinstance(tab, np.ndarray):
            if tab.ndim != 2 or tab.shape[1] != n_features:
                raise ValueError(f"Expected tabular shape [N, {n_features}]; got {list(tab.shape)}")
            if not np.all(np.isfinite(tab)):
                raise NonfiniteValueError("Nonfinite tabular values encountered during scaler fit.")
            col_sums = np.sum(tab, axis=0, dtype=np.float64)
            for j in range(n_features):
                sums[j] += float(col_sums[j])
        else:
            for row in tab:
                if len(row) != n_features:
                    raise ValueError(f"Expected {n_features} features; got {len(row)}")
                for j in range(n_features):
                    v = float(row[j])
                    if not math.isfinite(v):
                        raise NonfiniteValueError(f"Nonfinite tabular value at feature {j}: {v}")
                    sums[j] += v

    means = [s / float(n_total_samples) for s in sums]

    # Pass 2: compute sum of squared differences for variance/std
    sq_diff_sums = [0.0] * n_features
    for ds in ds_list:
        tab = ds.tabular
        if _HAS_NUMPY and isinstance(tab, np.ndarray):
            diff = tab - np.asarray(means, dtype=np.float64)
            sq_sums = np.sum(diff * diff, axis=0, dtype=np.float64)
            for j in range(n_features):
                sq_diff_sums[j] += float(sq_sums[j])
        else:
            for row in tab:
                for j in range(n_features):
                    diff = float(row[j]) - means[j]
                    sq_diff_sums[j] += diff * diff

    scales = [1.0] * n_features
    for j in range(n_features):
        var = sq_diff_sums[j] / float(n_total_samples)
        std = math.sqrt(max(0.0, var))
        if std < eps:
            scales[j] = 1.0  # safe zero-variance fallback
        else:
            scales[j] = std

    scaler.mean_ = means
    scaler.scale_ = scales
    scaler.n_features_in_ = n_features
    scaler.is_fitted = True
    return scaler


def get_memory_metrics(device: Optional[Any] = None) -> Tuple[Optional[float], Optional[float]]:
    """Measure (peak_memory_mb, process_rss_mb) without mislabeling current RSS as peak.

    Returns:
        (peak_memory_mb, process_rss_mb)
        - peak_memory_mb: True OS peak working set (Windows peak_wset) or peak RSS (Unix ru_maxrss)
          or CUDA max_memory_allocated if on GPU.
        - process_rss_mb: Current resident set size at call time.
    """
    peak_mb: Optional[float] = None
    rss_mb: Optional[float] = None

    # 1. GPU peak if on CUDA
    if _TORCH_AVAILABLE and device is not None and getattr(device, "type", None) == "cuda":
        try:
            if torch.cuda.is_available():
                peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        except Exception:
            pass

    # 2. psutil metrics: RSS and Windows peak_wset
    try:
        import psutil  # type: ignore
        proc = psutil.Process()
        mem_info = proc.memory_info()
        rss_mb = float(mem_info.rss / (1024.0 * 1024.0))
        # Windows true peak working set
        if hasattr(mem_info, "peak_wset"):
            peak_wset = float(mem_info.peak_wset / (1024.0 * 1024.0))
            if peak_mb is None or peak_wset > peak_mb:
                peak_mb = peak_wset
    except Exception:
        pass

    # 3. Unix resource ru_maxrss
    if peak_mb is None:
        try:
            import resource  # type: ignore
            ru = resource.getrusage(resource.RUSAGE_SELF)
            val = float(ru.ru_maxrss)
            if sys.platform == "darwin":
                val = val / (1024.0 * 1024.0)
            else:
                val = val / 1024.0
            peak_mb = val
        except Exception:
            pass

    return peak_mb, rss_mb


def get_peak_memory_mb(device: Optional[Any] = None) -> Optional[float]:
    """Compatibility alias returning true peak memory in MB where measurable."""
    peak_mb, _ = get_memory_metrics(device=device)
    return peak_mb


def get_environment_info(device: Optional[Any] = None) -> Dict[str, Any]:
    """Record environment, hardware, and dependency versions."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "cpu": platform.processor() or platform.machine(),
        "torch": getattr(torch, "__version__", None) if _TORCH_AVAILABLE else None,
        "numpy": getattr(np, "__version__", None) if _HAS_NUMPY else None,
        "sklearn": getattr(sklearn, "__version__", None) if _SKLEARN_AVAILABLE else None,
        "cuda_available": torch.cuda.is_available() if _TORCH_AVAILABLE else False,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None) if _TORCH_AVAILABLE else None,
        "gpu": None,
    }
    if _TORCH_AVAILABLE and device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        try:
            info["gpu"] = torch.cuda.get_device_name(device)
        except Exception:
            pass
    return info


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU & CUDA) deterministically."""
    random.seed(seed)
    if _HAS_NUMPY:
        np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
            except Exception:
                pass


def validate_training_inputs(
    train_data: Dict[str, ClientDataset],
    val_data: Dict[str, ClientDataset],
    regime: str,
) -> None:
    """Validate all client partitions, split names, and per-sample metadata."""
    if regime not in ("local", "central"):
        raise ValueError(f"regime must be 'local' or 'central'; got {regime!r}")

    if not train_data:
        raise ValueError("train_data cannot be empty.")
    if not val_data:
        raise ValueError("val_data cannot be empty.")

    # 1. Validate every train dataset
    for c_name, ds in train_data.items():
        if (
            c_name in HELDOUT_CLIENTS
            or (isinstance(c_name, str) and c_name.startswith("heldout"))
        ):
            raise HoldoutLeakageError(
                f"Held-out client '{c_name}' cannot be passed into train_data."
            )
        if c_name not in PARTICIPATING_CLIENTS:
            raise ValueError(
                f"Client '{c_name}' in train_data is not in approved participating clients "
                f"{sorted(PARTICIPATING_CLIENTS)}."
            )
        if ds.client_name != c_name:
            raise ValueError(
                f"Dataset client_name '{ds.client_name}' does not match dictionary key '{c_name}'."
            )
        if ds.split_name != "train":
            raise HoldoutLeakageError(
                f"Dataset for '{c_name}' in train_data has split_name='{ds.split_name}', expected 'train'."
            )
        ds.validate(expected_split="train")

    # 2. Validate every val dataset
    for c_name, ds in val_data.items():
        if (
            c_name in HELDOUT_CLIENTS
            or (isinstance(c_name, str) and c_name.startswith("heldout"))
        ):
            raise HoldoutLeakageError(
                f"Held-out client '{c_name}' cannot be passed into val_data."
            )
        if c_name not in PARTICIPATING_CLIENTS:
            raise ValueError(
                f"Client '{c_name}' in val_data is not in approved participating clients "
                f"{sorted(PARTICIPATING_CLIENTS)}."
            )
        if ds.client_name != c_name:
            raise ValueError(
                f"Dataset client_name '{ds.client_name}' does not match dictionary key '{c_name}'."
            )
        if ds.split_name != "val":
            raise HoldoutLeakageError(
                f"Dataset for '{c_name}' in val_data has split_name='{ds.split_name}', expected 'val'."
            )
        ds.validate(expected_split="val")

    # 3. Regime checks
    if regime == "local":
        if len(train_data) != 1:
            raise ValueError(
                f"Local regime requires exactly 1 participating client in train_data; "
                f"got {len(train_data)} ({list(train_data.keys())})."
            )
        if len(val_data) != 1:
            raise ValueError(
                f"Local regime requires exactly 1 participating client in val_data; "
                f"got {len(val_data)} ({list(val_data.keys())})."
            )
        train_client = next(iter(train_data.keys()))
        val_client = next(iter(val_data.keys()))
        if train_client != val_client:
            raise ValueError(
                f"Local regime requires the same client in train and val; "
                f"got train='{train_client}' and val='{val_client}'."
            )
    elif regime == "central":
        train_clients = set(train_data.keys())
        val_clients = set(val_data.keys())
        if train_clients != PARTICIPATING_CLIENTS:
            raise ValueError(
                f"Central regime requires all 3 participating clients {sorted(PARTICIPATING_CLIENTS)} "
                f"in train_data; got {sorted(train_clients)}."
            )
        if val_clients != PARTICIPATING_CLIENTS:
            raise ValueError(
                f"Central regime requires all 3 participating clients {sorted(PARTICIPATING_CLIENTS)} "
                f"in val_data; got {sorted(val_clients)}."
            )


class _PooledClientIndex:
    """Routes global indices across client datasets without copying entire pooled arrays."""

    def __init__(self, datasets: Sequence[ClientDataset]):
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.total_length = sum(self.lengths)
        self.offsets = [0]
        for l in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + l)

    def __len__(self) -> int:
        return self.total_length

    def get_batch(
        self,
        batch_indices: Sequence[int],
        is_mlp: bool,
    ) -> Tuple[Any, Any, Any]:
        """Slice a minibatch from underlying client datasets.

        Returns:
            (batch_inputs_tuple_or_tensor, batch_labels_tensor, batch_size)
        """
        b_size = len(batch_indices)
        if is_mlp:
            batch_tab = np.empty((b_size, 20), dtype=np.float32) if _HAS_NUMPY else [None] * b_size
        else:
            batch_seq = np.empty((b_size, 48, 4), dtype=np.float32) if _HAS_NUMPY else [None] * b_size
            batch_sum = np.empty((b_size, 8), dtype=np.float32) if _HAS_NUMPY else [None] * b_size

        batch_lbl = np.empty(b_size, dtype=np.float32) if _HAS_NUMPY else [None] * b_size

        # Group indices by client dataset for fast chunk slicing
        from bisect import bisect_right
        by_client: Dict[int, List[Tuple[int, int]]] = {}
        for pos, g_idx in enumerate(batch_indices):
            c_idx = bisect_right(self.offsets, g_idx) - 1
            l_idx = g_idx - self.offsets[c_idx]
            if c_idx not in by_client:
                by_client[c_idx] = []
            by_client[c_idx].append((pos, l_idx))

        for c_idx, items in by_client.items():
            ds = self.datasets[c_idx]
            positions = [item[0] for item in items]
            local_indices = [item[1] for item in items]

            # Labels
            if hasattr(ds.labels, "__getitem__"):
                c_lbls = ds.labels[local_indices] if _HAS_NUMPY and isinstance(ds.labels, np.ndarray) else [ds.labels[i] for i in local_indices]
            else:
                c_lbls = [ds.labels[i] for i in local_indices]

            for p, val in zip(positions, c_lbls):
                batch_lbl[p] = float(val)

            if is_mlp:
                c_tab = ds.tabular[local_indices] if _HAS_NUMPY and isinstance(ds.tabular, np.ndarray) else [ds.tabular[i] for i in local_indices]
                for p, val in zip(positions, c_tab):
                    batch_tab[p] = val
            else:
                c_seq = ds.sequences[local_indices] if _HAS_NUMPY and isinstance(ds.sequences, np.ndarray) else [ds.sequences[i] for i in local_indices]
                c_sum = ds.summaries[local_indices] if _HAS_NUMPY and isinstance(ds.summaries, np.ndarray) else [ds.summaries[i] for i in local_indices]
                for p, s_val, m_val in zip(positions, c_seq, c_sum):
                    batch_seq[p] = s_val
                    batch_sum[p] = m_val

        if _HAS_NUMPY:
            lbl_tensor = torch.from_numpy(batch_lbl).float()
            if is_mlp:
                return batch_tab, lbl_tensor, b_size
            else:
                seq_tensor = torch.from_numpy(batch_seq).float()
                sum_tensor = torch.from_numpy(batch_sum).float()
                return (seq_tensor, sum_tensor), lbl_tensor, b_size
        else:
            lbl_tensor = torch.tensor(batch_lbl, dtype=torch.float32)
            if is_mlp:
                return batch_tab, lbl_tensor, b_size
            else:
                seq_tensor = torch.tensor(batch_seq, dtype=torch.float32)
                sum_tensor = torch.tensor(batch_sum, dtype=torch.float32)
                return (seq_tensor, sum_tensor), lbl_tensor, b_size


class AzureNeuralTrainer:
    """Reusable PyTorch neural network training engine for Azure PdM."""

    def __init__(
        self,
        model: Any,
        config: Optional[TrainingConfig] = None,
        *,
        regime: str = "local",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = 17,
        device: Optional[str] = None,
        reset_parameters: bool = True,
        allow_seed_override: bool = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch ('torch') is required for AzureNeuralTrainer. Please install torch."
            )

        self.model = model
        self.config = config or TrainingConfig(
            regime=regime,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            reset_parameters=reset_parameters,
            allow_seed_override=allow_seed_override,
        )

        # Enforce approved protocol seeds unless explicitly overridden
        if self.config.seed not in APPROVED_SEEDS and not self.config.allow_seed_override:
            raise ValueError(
                f"Seed {self.config.seed} is not in approved protocol seeds {list(APPROVED_SEEDS)}. "
                f"Set allow_seed_override=True to permit non-protocol seeds for smoke testing."
            )

        # Detect model type
        self.is_mlp = isinstance(model, AzureMLP) or (hasattr(model, "fc1") and not hasattr(model, "gru"))

        # Device selection: caller device -> cuda if available -> cpu fallback
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def _reset_model_parameters(self) -> None:
        """Reset model parameters under current seed for fresh Phase 3 training."""
        if isinstance(self.model, AzureGRU):
            self.model.gru.reset_parameters()
            self.model.fc.reset_parameters()
        elif isinstance(self.model, AzureMLP):
            self.model.fc1.reset_parameters()
            self.model.fc2.reset_parameters()
        else:
            for module in self.model.modules():
                if module is not self.model and hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def _apply_mlp_scaler(self, x: Any) -> torch.Tensor:
        """Apply train-fitted scaler to a 20-dim tabular minibatch without fitting.

        Args:
            x: Raw tabular minibatch (NumPy array, torch Tensor, or list of rows).

        Returns:
            Standardized torch.Tensor of shape [batch_size, 20], float32.
        """
        scaler = getattr(self.model, "scaler", None)
        if scaler is None or not getattr(scaler, "is_fitted", False):
            raise RuntimeError(
                "AzureMLP scaler must be fitted on training partition before transforming features."
            )

        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            mean = torch.tensor(scaler.mean_, dtype=x.dtype, device=x.device)
            scale = torch.tensor(scaler.scale_, dtype=x.dtype, device=x.device)
            return (x - mean) / scale
        elif _HAS_NUMPY and isinstance(x, np.ndarray):
            mean = np.asarray(scaler.mean_, dtype=np.float32)
            scale = np.asarray(scaler.scale_, dtype=np.float32)
            trans = (x - mean) / scale
            return torch.from_numpy(trans).float()
        else:
            trans_rows = []
            for row in x:
                trans_rows.append([(float(row[j]) - scaler.mean_[j]) / scaler.scale_[j] for j in range(20)])
            return torch.tensor(trans_rows, dtype=torch.float32)

    def train(
        self,
        train_data: Union[ClientDataset, Dict[str, ClientDataset]],
        val_data: Union[ClientDataset, Dict[str, ClientDataset]],
    ) -> TrainingResult:
        """Run bounded neural training loop with AP early stopping and checkpoint restoration.

        Args:
            train_data: Dict mapping participating client name -> ClientDataset, or single ClientDataset.
            val_data: Dict mapping participating client name -> ClientDataset, or single ClientDataset.

        Returns:
            TrainingResult dataclass with history, metrics, steps, and restored best state_dict.
        """
        # Normalize inputs to dict
        if isinstance(train_data, ClientDataset):
            train_dict = {train_data.client_name: train_data}
        else:
            train_dict = dict(train_data)

        if isinstance(val_data, ClientDataset):
            val_dict = {val_data.client_name: val_data}
        else:
            val_dict = dict(val_data)

        # 1. Validate partitions, splits, and per-sample metadata
        validate_training_inputs(train_dict, val_dict, regime=self.config.regime)

        # 2. MLP scaler fitting strictly on training partitions via memory-bounded streaming
        if self.is_mlp and self.config.fit_mlp_scaler and hasattr(self.model, "scaler"):
            if not getattr(self.model.scaler, "is_fitted", False):
                fit_tabular_scaler_streaming(
                    self.model.scaler,
                    train_dict.values(),
                    split_name="train",
                )

        # 3. Setup reproducible seeds BEFORE model parameter reinitialization
        set_reproducible_seed(self.config.seed, deterministic=self.config.deterministic)

        # 4. Fresh Phase 3 training: reinitialize model parameters under seed
        # (Preserves FL continuation semantics when reset_parameters=False)
        if self.config.reset_parameters:
            self._reset_model_parameters()

        # 5. Setup unweighted BCEWithLogitsLoss and AdamW AFTER parameter reinitialization
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # 6. Build training index
        if self.config.regime == "local":
            local_client = next(iter(train_dict.keys()))
            train_pool = _PooledClientIndex([train_dict[local_client]])
        else:
            # Ordered model2, model3, model4
            sorted_clients = [c for c in ("client_model2", "client_model3", "client_model4") if c in train_dict]
            train_pool = _PooledClientIndex([train_dict[c] for c in sorted_clients])

        n_train = len(train_pool)
        batch_size = self.config.batch_size

        # Tracking state
        history: List[EpochLog] = []
        best_score: float = -1.0
        best_epoch: int = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        patience_counter: int = 0
        early_stopped: bool = False

        total_optimizer_steps: int = 0
        total_samples_processed: int = 0
        start_time = time.time()

        # Deterministic generator for shuffling
        shuffler = random.Random(self.config.seed)

        # Environment & Package versions
        package_versions: Dict[str, Optional[str]] = {
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None),
            "numpy": getattr(np, "__version__", None) if _HAS_NUMPY else None,
            "sklearn": getattr(sklearn, "__version__", None) if _SKLEARN_AVAILABLE else None,
        }
        env_info = get_environment_info(self.device)

        # 7. Epoch training loop
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()
            self.model.train()

            # Deterministic epoch permutation
            epoch_indices = list(range(n_train))
            # Seeded shuffle per epoch
            shuffler.shuffle(epoch_indices)

            epoch_loss_sum: float = 0.0
            epoch_samples: int = 0

            # Iterate minibatches
            for b_start in range(0, n_train, batch_size):
                b_indices = epoch_indices[b_start : b_start + batch_size]
                b_inp, b_lbl, b_len = train_pool.get_batch(b_indices, is_mlp=self.is_mlp)

                # Transfer minibatch slice to device
                b_lbl = b_lbl.to(self.device).view(-1, 1)

                optimizer.zero_grad()
                if self.is_mlp:
                    # Apply train-fitted scaler to MLP minibatch
                    scaled_inp = self._apply_mlp_scaler(b_inp)
                    logits = self.model(scaled_inp.to(self.device))
                else:
                    b_seq, b_sum = b_inp
                    logits = self.model(b_seq.to(self.device), b_sum.to(self.device))

                loss = criterion(logits, b_lbl)
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                epoch_loss_sum += loss_val * b_len
                epoch_samples += b_len
                total_optimizer_steps += 1
                total_samples_processed += b_len

            train_loss = epoch_loss_sum / epoch_samples if epoch_samples > 0 else 0.0

            # 8. Evaluate validation AP for each validation client
            self.model.eval()
            client_ap: Dict[str, float] = {}

            with torch.no_grad():
                for c_name, v_ds in val_dict.items():
                    c_probs, c_targets = self._predict_dataset(v_ds)
                    wm = evaluate_window_metrics(c_targets, c_probs, threshold=0.5)

                    if wm.average_precision is None:
                        raise ValueError(
                            f"Validation Average Precision is undefined for client '{c_name}' "
                            f"due to single-class validation data (positives={wm.n_pos}, negatives={wm.n_neg})."
                        )
                    client_ap[c_name] = float(wm.average_precision)

            # Compute validation score
            if self.config.regime == "local":
                local_client = next(iter(val_dict.keys()))
                val_score = client_ap[local_client]
                macro_ap = val_score
            else:
                # Unweighted macro AP across participating clients (model2, model3, model4)
                participating_aps = [client_ap[c] for c in PARTICIPATING_CLIENTS if c in client_ap]
                macro_ap = sum(participating_aps) / float(len(participating_aps))
                val_score = macro_ap

            elapsed_epoch = time.time() - epoch_start
            peak_mem, rss_mem = get_memory_metrics(self.device)

            log_entry = EpochLog(
                epoch=epoch,
                train_loss=train_loss,
                client_ap=client_ap,
                macro_ap=macro_ap,
                val_score=val_score,
                optimizer_steps=total_optimizer_steps,
                samples_processed=total_samples_processed,
                seed=self.config.seed,
                device=str(self.device),
                elapsed_seconds=elapsed_epoch,
                peak_memory_mb=peak_mem,
                process_rss_mb=rss_mem,
            )
            history.append(log_entry)

            # 9. Checkpoint selection: strict improvement (no update on tie)
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    early_stopped = True
                    break

        total_elapsed = time.time() - start_time

        # 10. Restore best model weights before return
        self.model.load_state_dict(best_state_dict)

        return TrainingResult(
            best_epoch=best_epoch,
            best_score=best_score,
            best_state_dict=best_state_dict,
            history=history,
            total_optimizer_steps=total_optimizer_steps,
            total_samples_processed=total_samples_processed,
            total_elapsed_seconds=total_elapsed,
            package_versions=package_versions,
            regime=self.config.regime,
            seed=self.config.seed,
            device=str(self.device),
            early_stopped=early_stopped,
            environment_info=env_info,
        )

    def _predict_dataset(self, dataset: ClientDataset) -> Tuple[List[float], List[int]]:
        """Run batched forward inference on dataset and return (probabilities, binary_targets)."""
        return predict_model_probabilities(
            self.model,
            dataset,
            batch_size=self.config.batch_size,
            device=self.device,
        )


def predict_model_probabilities(
    model: Any,
    dataset: ClientDataset,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[Union[str, Any]] = None,
) -> Tuple[List[float], List[int]]:
    """Run public batched forward inference on a ClientDataset and return (probabilities, binary_targets).

    Supports AzureGRU, AzureMLP, and AzureLogisticRegression.
    For AzureMLP, applies the frozen train-fitted scaler without fitting.
    Evaluates in minibatches to avoid large GPU memory allocations.

    Args:
        model: Trained AzureGRU, AzureMLP, or AzureLogisticRegression instance.
        dataset: ClientDataset instance to score.
        batch_size: Minibatch size for inference (default 512).
        device: Optional PyTorch device or device string (e.g. 'cpu', 'cuda').

    Returns:
        Tuple of (predicted_probabilities, ground_truth_targets) as Python lists.

    Raises:
        RuntimeError: If AzureMLP scaler is unfitted or missing.
        ImportError: If PyTorch (for neural models) or scikit-learn (for logistic) is missing.
        ValueError: If dataset is empty or features are invalid.
    """
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError(f"Cannot predict on empty dataset for client '{dataset.client_name}'.")

    # 1. Scikit-learn Logistic Regression
    if hasattr(model, "predict_proba") and not hasattr(model, "forward"):
        if dataset.tabular is None:
            raise ValueError(f"Dataset '{dataset.client_name}' missing tabular features for logistic prediction.")
        all_probs: List[float] = []
        for b_start in range(0, n_samples, batch_size):
            b_end = min(b_start + batch_size, n_samples)
            chunk = dataset.tabular[b_start:b_end]
            p = model.predict_proba(chunk)
            if hasattr(p, "shape") and len(p.shape) == 2 and p.shape[1] >= 2:
                all_probs.extend(float(v) for v in p[:, 1])
            elif isinstance(p, list):
                all_probs.extend(float(row[1]) for row in p)
            else:
                all_probs.extend(float(v) for v in p)

        if hasattr(dataset.labels, "tolist"):
            all_targets = [int(y) for y in dataset.labels.tolist()]
        else:
            all_targets = [int(y) for y in dataset.labels]
        return all_probs, all_targets

    # 2. PyTorch Neural Models (AzureGRU, AzureMLP)
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch ('torch') is required for neural model scoring.")

    # Determine device
    if device is not None:
        torch_device = torch.device(device)
    elif hasattr(model, "parameters"):
        try:
            torch_device = next(model.parameters()).device
        except StopIteration:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(torch_device)
    model.eval()

    is_mlp = isinstance(model, AzureMLP) or (hasattr(model, "fc1") and not hasattr(model, "gru"))

    if is_mlp:
        scaler = getattr(model, "scaler", None)
        if scaler is None or not getattr(scaler, "is_fitted", False):
            raise RuntimeError(
                "AzureMLP scaler must be fitted on training partition before transforming features."
            )
        mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=torch_device)
        scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=torch_device)

    index_pool = _PooledClientIndex([dataset])
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for b_start in range(0, n_samples, batch_size):
            b_indices = list(range(b_start, min(b_start + batch_size, n_samples)))
            b_inp, b_lbl, _ = index_pool.get_batch(b_indices, is_mlp=is_mlp)

            if is_mlp:
                if _TORCH_AVAILABLE and isinstance(b_inp, torch.Tensor):
                    inp_t = b_inp.float().to(torch_device)
                elif _HAS_NUMPY and isinstance(b_inp, np.ndarray):
                    inp_t = torch.from_numpy(b_inp).float().to(torch_device)
                else:
                    inp_t = torch.tensor(b_inp, dtype=torch.float32, device=torch_device)
                scaled_inp = (inp_t - mean_t) / scale_t
                logits = model(scaled_inp)
            else:
                b_seq, b_sum = b_inp
                logits = model(b_seq.to(torch_device), b_sum.to(torch_device))

            probs = torch.sigmoid(logits).view(-1).cpu().tolist()
            targets = [int(y) for y in b_lbl.tolist()]

            all_probs.extend(probs)
            all_targets.extend(targets)

    return all_probs, all_targets


def train_azure_model(
    model: Any,
    train_data: Union[ClientDataset, Dict[str, ClientDataset]],
    val_data: Union[ClientDataset, Dict[str, ClientDataset]],
    config: Optional[TrainingConfig] = None,
    **kwargs: Any,
) -> TrainingResult:
    """Convenience helper to initialize AzureNeuralTrainer and execute training."""
    trainer = AzureNeuralTrainer(model=model, config=config, **kwargs)
    return trainer.train(train_data=train_data, val_data=val_data)
