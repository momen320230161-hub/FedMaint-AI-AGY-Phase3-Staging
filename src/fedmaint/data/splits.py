"""Dataset splits and simulated non-IID client partitioning for Azure PdM.

This module enforces:
- Derivation of temporal split intervals and client mappings from Phase2BConfig as single source of truth.
- Strict half-open temporal intervals for train, val, and test.
- The sample inclusion boundary rule:
    t - 48h >= split_start and t < split_end and t + 24h < split_end (strict).
- Separation of participating clients (model2, model3, model4) from held-out client (model1).
- Rejection of any unexpected machine models.
- HoldoutLeakageError when any attempt is made to request train or val splits for model1.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .config import Phase2BConfig, load_azure_preprocessing_config

Record = Dict[str, object]


class HoldoutLeakageError(ValueError):
    """Raised when an operation attempts to include held-out data in training or validation."""


@dataclass(frozen=True)
class SplitBounds:
    """Half-open interval [start, end) defining a dataset split."""
    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"Split start {self.start} must be strictly before end {self.end}")

    def is_eligible_prediction_time(
        self,
        t: datetime,
        *,
        history_hours: int = 48,
        horizon_hours: int = 24,
    ) -> bool:
        """Check if prediction time t satisfies all split boundary rules.

        Conditions:
          1. t - history_hours >= split.start (entire history inside split)
          2. t < split.end
          3. t + horizon_hours < split.end (strict: entire horizon inside split)
        """
        history_start = t - timedelta(hours=history_hours)
        horizon_end = t + timedelta(hours=horizon_hours)
        return (history_start >= self.start) and (t < self.end) and (horizon_end < self.end)


@dataclass(frozen=True)
class ClientPartition:
    """Metadata defining a client partition."""
    name: str
    model: str
    machine_ids: Tuple[int, ...]
    is_held_out: bool = False
    allowed_splits: Tuple[str, ...] = ("train", "val", "test")

    def validate_split_access(self, split_name: str) -> None:
        """Ensure the requested split is permitted for this client partition.

        Raises:
            HoldoutLeakageError if requesting a disallowed split (e.g. train/val on held-out client).
        """
        if split_name not in self.allowed_splits:
            raise HoldoutLeakageError(
                f"Client '{self.name}' (model {self.model}, held_out={self.is_held_out}) "
                f"does not permit split '{split_name}'. Allowed splits: {self.allowed_splits}"
            )


def get_default_splits(config: Optional[Phase2BConfig] = None) -> Dict[str, SplitBounds]:
    """Return split intervals derived from Phase2BConfig."""
    cfg = config if config is not None else load_azure_preprocessing_config()
    return {
        name: SplitBounds(name=name, start=s.start, end=s.end)
        for name, s in cfg.splits.items()
    }


def get_client_partitions(
    machines: Dict[int, Record],
    config: Optional[Phase2BConfig] = None,
) -> Dict[str, ClientPartition]:
    """Group machines by model into participating clients and the held-out client.

    Derives allowed models and partition roles directly from Phase2BConfig. Rejects any unexpected
    machine model not defined in configuration.

    Args:
        machines: Mapping of machineID -> machine record (containing 'model').
        config: Phase2BConfig instance (if None, loaded from configs/azure_preprocessing.json).

    Returns:
        Mapping of client name -> ClientPartition.
    """
    cfg = config if config is not None else load_azure_preprocessing_config()
    approved_models = set(cfg.participating_models) | {cfg.held_out_model}

    by_model: Dict[str, List[int]] = {m: [] for m in approved_models}
    for machine_id, rec in machines.items():
        model = str(rec["model"])
        if model not in approved_models:
            raise ValueError(
                f"Unexpected machine model '{model}' for machineID {machine_id}. "
                f"Approved models from configuration are: {sorted(approved_models)}"
            )
        by_model[model].append(int(machine_id))

    partitions: Dict[str, ClientPartition] = {}

    # If client_specs is populated in config, derive directly from client_specs
    if cfg.client_specs:
        for c_name, c_spec in cfg.client_specs.items():
            model = c_spec["model"]
            ids = tuple(sorted(by_model.get(model, [])))
            is_held_out = (c_spec.get("role") == "held_out_evaluation") or (model == cfg.held_out_model)
            allowed = tuple(c_spec.get("allowed_splits", ("test",) if is_held_out else ("train", "val", "test")))
            partitions[c_name] = ClientPartition(
                name=c_name,
                model=model,
                machine_ids=ids,
                is_held_out=is_held_out,
                allowed_splits=allowed,
            )
    else:
        # Fallback to model-based naming
        for model in sorted(by_model):
            ids = tuple(sorted(by_model[model]))
            if model == cfg.held_out_model:
                partitions[f"heldout_{model}"] = ClientPartition(
                    name=f"heldout_{model}",
                    model=model,
                    machine_ids=ids,
                    is_held_out=True,
                    allowed_splits=("test",),
                )
            else:
                partitions[f"client_{model}"] = ClientPartition(
                    name=f"client_{model}",
                    model=model,
                    machine_ids=ids,
                    is_held_out=False,
                    allowed_splits=("train", "val", "test"),
                )

    return partitions
