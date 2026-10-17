"""Reproducible end-to-end experiment orchestration for Phase 3 Azure PdM.

Specification:
- Orchestrates local-only and centralized reference GRU backbones, plus Logistic Regression
  and MLP baselines on 20 causal tabular features.
- Single source of truth configuration frozen in configs/phase3_azure.json.
- Reuses ONE Phase 2B PipelineData instance and calls build_all_model_caches once across clients/splits.
- Data-free synthetic integration via caller-injected PipelineData.
- Enforces strict split and client isolation before training and scoring:
    * Participating clients (model2, model3, model4): train, val, test.
    * Held-out client (model1): test ONLY. No train/val loading, tuning, or threshold search.
- Stratified bounded subsetting for --mode smoke: deterministic, bounded sample count,
  stratified across classes, marked non-representative and plumbing-only.
- Public batched scoring via predict_model_probabilities: applies frozen train-fitted scaler
  without fitting; does not leak features or call private methods.
- Local GRU: one independent model per participating client per seed, checkpoint selected on own
  val AP, val-only F1 threshold, evaluated on own test split and separate zero-shot model1 test.
- Central GRU: pooled participating train, checkpoint selected on unweighted macro participant
  val AP, single pooled participant val threshold, evaluated on per-client test, pooled test,
  macro test, and zero-shot model1 test. Distinguishes central non-private reference.
- Baselines:
    * Logistic Regression: 20 tabular features, train-fitted scaler, val threshold, no epochs,
      records solver, iterations, convergence, sklearn version.
    * MLP: 20 tabular features, train-fitted scaler (local vs pooled central), same neural budget
      as GRU (AdamW 1e-3, wd 1e-4, batch 512, max 25, patience 5), val AP checkpoint, val threshold.
- Evaluation:
    * Window-level metrics and WindowReport at frozen validation threshold.
    * Event-level metrics with 24h alert cooldown and chronological episode matching.
    * Machine-cluster bootstrap CIs (smoke: 30 draws marked plumbing only; full: >=1000 draws).
    * Explicit tracking of window positives vs distinct failure episodes.
- Telemetry & Provenance:
    * Per-seed optimizer steps, samples processed, epochs, runtime, true peak memory, current RSS,
      package versions, device info, config hash, Git commit SHA.
    * Structured JSON serialization output under caller-specified output directory.
"""
from __future__ import annotations

import collections.abc
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import sklearn
    _SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None  # type: ignore
    _SKLEARN_AVAILABLE = False

from ..data.adapter import SampleMetadata
from ..data.config import Phase2BConfig, load_azure_preprocessing_config
from ..data.episodes import FailureEpisode
from ..data.model_cache import (
    ALLOWED_CLIENT_SPLITS,
    build_all_model_caches,
    get_cache_dir,
    load_cached_client_dataset,
)
from ..data.pipeline import PipelineData, load_pipeline_data
from ..data.splits import HoldoutLeakageError, get_client_partitions
from ..evaluation.bootstrap import (
    BootstrapCI,
    bootstrap_event_ci,
    bootstrap_window_ci,
)
from ..evaluation.events import EventMetrics, evaluate_event_metrics
from ..evaluation.threshold import (
    HELDOUT_CLIENTS,
    PARTICIPATING_CLIENTS,
    ThresholdResult,
    select_threshold,
    validate_threshold_metadata,
)
from ..evaluation.window import (
    WindowMetrics,
    WindowReport,
    evaluate_window_metrics,
    evaluate_window_report,
)
from ..models.gru import AzureGRU
from ..models.logistic import AzureLogisticRegression
from ..models.mlp import AzureMLP
from ..models.scaler import TabularStandardScaler
from ..training.engine import (
    APPROVED_SEEDS,
    AzureNeuralTrainer,
    ClientDataset,
    TrainingConfig,
    TrainingResult,
    get_environment_info,
    get_memory_metrics,
    predict_model_probabilities,
    set_reproducible_seed,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "phase3_azure.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "Azure Data"


def get_git_revision() -> Optional[str]:
    """Retrieve Git commit SHA from repository filesystem if available, without subprocess."""
    try:
        git_dir = Path(__file__).resolve().parents[3] / ".git"
        if not git_dir.exists():
            return None
        head_file = git_dir / "HEAD"
        if not head_file.is_file():
            return None
        content = head_file.read_text(encoding="utf-8").strip()
        if content.startswith("ref:"):
            ref_path = content[4:].strip()
            target = git_dir / ref_path
            if target.is_file():
                return target.read_text(encoding="utf-8").strip()
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith(("#", "^")):
                        parts = line.strip().split()
                        if len(parts) == 2 and parts[1] == ref_path:
                            return parts[0]
        else:
            return content
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class Phase3AzureConfig:
    """Frozen Phase 3 Azure experiment configuration."""
    pipeline_name: str
    phase: str
    track: str
    version: str
    description: str
    raw_config: Dict[str, Any]
    seeds_full: Tuple[int, ...] = (17, 23, 41)
    seeds_smoke: Tuple[int, ...] = (17,)
    max_epochs_full: int = 25
    max_epochs_smoke: int = 2
    patience_full: int = 5
    patience_smoke: int = 2
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    bootstrap_draws_full: int = 1000
    bootstrap_draws_smoke: int = 30
    smoke_bounded_subset_max_samples: int = 60

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase3AzureConfig":
        training_cfg = data.get("training", {})
        eval_cfg = data.get("evaluation", {})
        boot_cfg = eval_cfg.get("bootstrap", {})
        modes_cfg = data.get("modes", {})
        smoke_cfg = modes_cfg.get("smoke", {})
        full_cfg = modes_cfg.get("full", {})

        return cls(
            pipeline_name=str(data.get("pipeline_name", "azure_phase3_baselines_and_orchestration")),
            phase=str(data.get("phase", "3")),
            track=str(data.get("track", "azure")),
            version=str(data.get("version", "1.0.0")),
            description=str(data.get("description", "")),
            raw_config=data,
            seeds_full=tuple(full_cfg.get("seeds", training_cfg.get("seeds", [17, 23, 41]))),
            seeds_smoke=tuple(smoke_cfg.get("seeds", [17])),
            max_epochs_full=int(full_cfg.get("max_epochs", training_cfg.get("max_epochs", 25))),
            max_epochs_smoke=int(smoke_cfg.get("max_epochs", 2)),
            patience_full=int(full_cfg.get("patience", training_cfg.get("patience", 5))),
            patience_smoke=int(smoke_cfg.get("patience", 2)),
            batch_size=int(training_cfg.get("batch_size", 512)),
            lr=float(training_cfg.get("lr", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
            bootstrap_draws_full=int(boot_cfg.get("num_draws_full", 1000)),
            bootstrap_draws_smoke=int(boot_cfg.get("num_draws_smoke", 30)),
            smoke_bounded_subset_max_samples=int(smoke_cfg.get("bounded_subset_max_samples", 60)),
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Phase3AzureConfig":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 digest of configuration."""
        encoded = json.dumps(self.raw_config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_phase3_azure_config(config_path: Optional[Union[str, Path]] = None) -> Phase3AzureConfig:
    """Load Phase3AzureConfig from file or default path."""
    target_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    return Phase3AzureConfig.from_file(target_path)


def create_stratified_subset(dataset: ClientDataset, max_samples: int = 60) -> ClientDataset:
    """Extract a deterministic, bounded stratified subset of a ClientDataset for smoke testing.

    Ensures that if both positive and negative samples are present, both classes are preserved
    so validation average precision, F1, and threshold selection remain well-defined.
    Preserves chronological order of selected samples.

    Args:
        dataset: The ClientDataset to subset.
        max_samples: Maximum number of samples to retain.

    Returns:
        Subsetted ClientDataset instance.
    """
    n = len(dataset)
    if n <= max_samples:
        return dataset

    labels = dataset.labels
    if hasattr(labels, "tolist"):
        lbl_list = labels.tolist()
    else:
        lbl_list = list(labels)

    pos_indices = [i for i, y in enumerate(lbl_list) if y == 1]
    neg_indices = [i for i, y in enumerate(lbl_list) if y == 0]

    if pos_indices and neg_indices:
        target_pos = min(len(pos_indices), max(2, max_samples // 4))
        target_neg = min(len(neg_indices), max_samples - target_pos)
        selected_indices = sorted(pos_indices[:target_pos] + neg_indices[:target_neg])
    elif pos_indices:
        selected_indices = pos_indices[:max_samples]
    else:
        selected_indices = neg_indices[:max_samples]

    sub_labels = dataset.labels[selected_indices] if hasattr(dataset.labels, "__getitem__") else [lbl_list[i] for i in selected_indices]
    sub_metadata = [dataset.metadata[i] for i in selected_indices]

    sub_seq = dataset.sequences[selected_indices] if dataset.sequences is not None else None
    sub_sum = dataset.summaries[selected_indices] if dataset.summaries is not None else None
    sub_tab = dataset.tabular[selected_indices] if dataset.tabular is not None else None

    return ClientDataset(
        client_name=dataset.client_name,
        split_name=dataset.split_name,
        labels=sub_labels,
        metadata=sub_metadata,
        sequences=sub_seq,
        summaries=sub_sum,
        tabular=sub_tab,
        machine_ids=dataset.cohort_machine_ids,
    )


def load_phase3_datasets(
    cache_root: Union[str, Path],
    *,
    pipeline_data: Optional[PipelineData] = None,
    data_dir: Optional[Union[str, Path]] = None,
    phase2b_config: Optional[Phase2BConfig] = None,
    phase3_config: Optional[Phase3AzureConfig] = None,
    force_rebuild: bool = False,
    mode: str = "smoke",
) -> Tuple[Dict[Tuple[str, str], ClientDataset], PipelineData]:
    """Build and load all Phase 3 datasets with strict client/split boundaries and held-out isolation.

    Enforces:
    - Calls build_all_model_caches once across all partitions using ONE shared PipelineData.
    - Loads memmapped ClientDataset for participating model2/3/4 (train, val, test) and heldout model1 (test only).
    - Enforces HoldoutLeakageError if heldout client is ever requested for train or val.
    - Validates sample counts against summary.
    - If mode == 'smoke', applies deterministic stratified bounding to each partition.

    Returns:
        Tuple of (datasets_dict, shared_pipeline_data)
    """
    cfg2b = phase2b_config or load_azure_preprocessing_config()
    cfg3 = phase3_config or load_phase3_azure_config()

    if pipeline_data is not None:
        shared_data = pipeline_data
    else:
        if data_dir is not None:
            resolved_dir = Path(data_dir)
            if not resolved_dir.is_dir():
                raise FileNotFoundError(
                    f"Specified data directory does not exist or is not a directory: {resolved_dir}. "
                    "Please provide a valid path via --data-dir or ensure the raw Azure CSV directory exists."
                )
        elif DEFAULT_DATA_DIR.is_dir():
            resolved_dir = DEFAULT_DATA_DIR
        else:
            raise ValueError(
                f"Raw Azure dataset directory not found at default location '{DEFAULT_DATA_DIR}'. "
                f"Must provide either data_dir or in-memory machine_rows. "
                f"Please pass --data-dir pointing to your raw Azure CSV files (e.g. --data-dir 'data/Azure Data')."
            )
        shared_data = load_pipeline_data(data_dir=resolved_dir, config=cfg2b)

    # 1. Build all caches once
    build_all_model_caches(
        output_root=cache_root,
        pipeline_data=shared_data,
        config=cfg2b,
        force_rebuild=force_rebuild,
    )

    # 2. Strict held-out leakage check: attempt to load heldout train/val must raise HoldoutLeakageError
    for disallowed_split in ("train", "val"):
        try:
            load_cached_client_dataset(cache_root, "heldout_model1", disallowed_split, config=cfg2b)
            raise AssertionError(f"Security failure: heldout_model1 was loaded for '{disallowed_split}'!")
        except HoldoutLeakageError:
            pass  # Expected and verified

    # 3. Load all allowed partitions
    loaded_datasets: Dict[Tuple[str, str], ClientDataset] = {}
    for client_name, splits in ALLOWED_CLIENT_SPLITS.items():
        for split_name in splits:
            ds = load_cached_client_dataset(
                output_root=cache_root,
                client_name=client_name,
                split_name=split_name,
                config=cfg2b,
                validate=False,
            )
            if len(ds) == 0:
                raise ValueError(f"Partition {client_name}/{split_name} is empty.")

            if mode == "smoke":
                ds = create_stratified_subset(ds, max_samples=cfg3.smoke_bounded_subset_max_samples)

            loaded_datasets[(client_name, split_name)] = ds

    return loaded_datasets, shared_data


def _safe_serialize(val: Any) -> Any:
    """Recursively convert NumPy, dataclass, tuple, and datetime values to JSON-serializable types."""
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return _safe_serialize(asdict(val))
    if isinstance(val, dict):
        return {str(k): _safe_serialize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set, frozenset)):
        return [_safe_serialize(v) for v in val]
    if _HAS_NUMPY:
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val) if math.isfinite(val) else None
        if isinstance(val, np.ndarray):
            return _safe_serialize(val.tolist())
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    return val


@dataclass
class SingleRunResult:
    """Complete results, evaluation metrics, and provenance for a single model run.

    Note on timing metrics:
        total_elapsed_seconds: Elapsed duration of the individual run (model fit plus
            validation/test/bootstrap evaluation). Does NOT include raw dataset loading or
            cache construction, since run_start begins after load_phase3_datasets.
        fit_elapsed_seconds: Dedicated duration spent fitting the model (excluding
            validation threshold search, test scoring, and bootstrap CI estimation).
    """
    model_name: str
    regime: str
    client_name: str
    seed: int
    mode: str
    is_representative: bool
    is_plumbing_only: bool
    is_central_reference: bool
    threshold: float
    val_f1_score: float
    best_epoch: Optional[int]
    best_score: Optional[float]
    total_optimizer_steps: int
    total_samples_processed: int
    total_elapsed_seconds: float
    peak_memory_mb: Optional[float]
    process_rss_mb: Optional[float]
    device: str
    package_versions: Dict[str, Optional[str]]
    environment_info: Optional[Dict[str, Any]]
    config_hash: str
    git_revision: Optional[str]
    # Counts
    window_positives_test: int
    window_negatives_test: int
    total_windows_test: int
    distinct_failure_episodes_test: int
    eligible_failure_episodes_test: int
    detected_failure_episodes_test: int
    # Metrics
    test_window_metrics: Dict[str, Any]
    test_event_metrics: Dict[str, Any]
    test_window_ci: Dict[str, Any]
    test_event_ci: Dict[str, Any]
    # Zero-shot heldout model1 metrics
    zero_shot_heldout_window_metrics: Dict[str, Any]
    zero_shot_heldout_event_metrics: Dict[str, Any]
    zero_shot_heldout_window_ci: Dict[str, Any]
    zero_shot_heldout_event_ci: Dict[str, Any]
    # Optional central multi-client report
    central_window_report: Optional[Dict[str, Any]] = None
    logistic_details: Optional[Dict[str, Any]] = None
    fit_elapsed_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return _safe_serialize(asdict(self))


def evaluate_model_partition(
    model: Any,
    dataset: ClientDataset,
    threshold: float,
    all_episodes: Dict[int, List[FailureEpisode]],
    *,
    num_bootstrap_draws: int,
    seed: int,
    device: Optional[Any] = None,
    phase2b_config: Optional[Phase2BConfig] = None,
    is_smoke: bool = False,
) -> Tuple[WindowMetrics, EventMetrics, Dict[str, BootstrapCI], Dict[str, BootstrapCI], int, int, int, int]:
    """Evaluate window metrics, event metrics, and bootstrap CIs on a test partition."""
    cfg2b = phase2b_config or load_azure_preprocessing_config()
    probs, targets = predict_model_probabilities(model, dataset, device=device)

    # 1. Window metrics
    wm = evaluate_window_metrics(targets, probs, threshold=threshold)

    # 2. Event metrics
    cohort = dataset.cohort_machine_ids or ()
    cohort_eps = [ep for m in cohort for ep in all_episodes.get(m, [])]
    em = evaluate_event_metrics(
        episodes=cohort_eps,
        predictions=dataset.metadata,
        machine_ids=cohort,
        scores=probs,
        threshold=threshold,
        split_name=dataset.split_name,
        config=cfg2b,
    )

    # 3. Bootstrap CIs
    w_ci = bootstrap_window_ci(
        targets, probs, dataset.metadata, threshold=threshold, num_draws=num_bootstrap_draws, seed=seed
    )
    e_ci = bootstrap_event_ci(
        cohort_eps, dataset.metadata, machine_ids=cohort, scores=probs, threshold=threshold,
        split_name=dataset.split_name, config=cfg2b, num_draws=num_bootstrap_draws, seed=seed
    )

    win_pos = wm.n_pos
    win_neg = wm.n_neg
    ep_total = em.total_episodes
    ep_eligible = em.eligible_episodes
    ep_detected = em.detected_episodes

    return wm, em, w_ci, e_ci, win_pos, win_neg, ep_total, ep_eligible, ep_detected  # type: ignore


def run_local_gru(
    datasets: Dict[Tuple[str, str], ClientDataset],
    pipeline_data: PipelineData,
    phase3_config: Phase3AzureConfig,
    phase2b_config: Phase2BConfig,
    *,
    seeds: Sequence[int],
    mode: str = "smoke",
    device: Optional[str] = None,
) -> List[SingleRunResult]:
    """Execute Local GRU training and evaluation across participating clients and seeds."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch ('torch') is required for AzureGRU.")

    results: List[SingleRunResult] = []
    config_hash = phase3_config.compute_hash()
    git_rev = get_git_revision()
    is_smoke = (mode == "smoke")
    max_epochs = phase3_config.max_epochs_smoke if is_smoke else phase3_config.max_epochs_full
    patience = phase3_config.patience_smoke if is_smoke else phase3_config.patience_full
    num_draws = phase3_config.bootstrap_draws_smoke if is_smoke else phase3_config.bootstrap_draws_full

    heldout_test_ds = datasets[("heldout_model1", "test")]

    for seed in seeds:
        for client_name in sorted(PARTICIPATING_CLIENTS):
            train_ds = datasets[(client_name, "train")]
            val_ds = datasets[(client_name, "val")]
            test_ds = datasets[(client_name, "test")]

            run_start = time.time()
            # Fresh GRU instance initialized under seed
            model = AzureGRU(input_size=4, hidden_size=32, num_summaries=8, output_size=1)

            t_fit_start = time.time()
            t_cfg = TrainingConfig(
                regime="local",
                batch_size=phase3_config.batch_size,
                max_epochs=max_epochs,
                patience=patience,
                lr=phase3_config.lr,
                weight_decay=phase3_config.weight_decay,
                seed=seed,
                device=device,
                allow_seed_override=True,
            )
            trainer = AzureNeuralTrainer(model=model, config=t_cfg)
            train_result = trainer.train(train_data=train_ds, val_data=val_ds)
            fit_elapsed = time.time() - t_fit_start

            # Validation-only threshold selection
            val_probs, val_targets = predict_model_probabilities(model, val_ds, device=trainer.device)
            thresh_res = select_threshold(val_targets, val_probs, metadata=val_ds.metadata)
            threshold = thresh_res.threshold
            val_score = thresh_res.score

            # Test evaluation on own client
            wm, em, w_ci, e_ci, w_pos, w_neg, ep_tot, ep_elig, ep_det = evaluate_model_partition(
                model=model,
                dataset=test_ds,
                threshold=threshold,
                all_episodes=pipeline_data.episodes,
                num_bootstrap_draws=num_draws,
                seed=seed,
                device=trainer.device,
                phase2b_config=phase2b_config,
                is_smoke=is_smoke,
            )

            # Separate zero-shot evaluation on heldout model1
            h_wm, h_em, h_wci, h_eci, _, _, _, _, _ = evaluate_model_partition(
                model=model,
                dataset=heldout_test_ds,
                threshold=threshold,
                all_episodes=pipeline_data.episodes,
                num_bootstrap_draws=num_draws,
                seed=seed,
                device=trainer.device,
                phase2b_config=phase2b_config,
                is_smoke=is_smoke,
            )

            peak_mem, rss_mem = get_memory_metrics(trainer.device)
            total_elapsed = time.time() - run_start

            run_entry = SingleRunResult(
                model_name="gru",
                regime="local",
                client_name=client_name,
                seed=seed,
                mode=mode,
                is_representative=not is_smoke,
                is_plumbing_only=is_smoke,
                is_central_reference=False,
                threshold=threshold,
                val_f1_score=val_score,
                best_epoch=train_result.best_epoch,
                best_score=train_result.best_score,
                total_optimizer_steps=train_result.total_optimizer_steps,
                total_samples_processed=train_result.total_samples_processed,
                total_elapsed_seconds=total_elapsed,
                peak_memory_mb=peak_mem,
                process_rss_mb=rss_mem,
                device=str(trainer.device),
                package_versions=train_result.package_versions,
                environment_info=train_result.environment_info,
                config_hash=config_hash,
                git_revision=git_rev,
                window_positives_test=w_pos,
                window_negatives_test=w_neg,
                total_windows_test=w_pos + w_neg,
                distinct_failure_episodes_test=ep_tot,
                eligible_failure_episodes_test=ep_elig,
                detected_failure_episodes_test=ep_det,
                test_window_metrics=wm.to_dict(),
                test_event_metrics=em.to_dict(),
                test_window_ci={k: v.to_dict() for k, v in w_ci.items()},
                test_event_ci={k: v.to_dict() for k, v in e_ci.items()},
                zero_shot_heldout_window_metrics=h_wm.to_dict(),
                zero_shot_heldout_event_metrics=h_em.to_dict(),
                zero_shot_heldout_window_ci={k: v.to_dict() for k, v in h_wci.items()},
                zero_shot_heldout_event_ci={k: v.to_dict() for k, v in h_eci.items()},
                fit_elapsed_seconds=fit_elapsed,
            )
            results.append(run_entry)

    return results


def run_central_gru(
    datasets: Dict[Tuple[str, str], ClientDataset],
    pipeline_data: PipelineData,
    phase3_config: Phase3AzureConfig,
    phase2b_config: Phase2BConfig,
    *,
    seeds: Sequence[int],
    mode: str = "smoke",
    device: Optional[str] = None,
) -> List[SingleRunResult]:
    """Execute Centralized Reference GRU training and evaluation across seeds."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch ('torch') is required for AzureGRU.")

    results: List[SingleRunResult] = []
    config_hash = phase3_config.compute_hash()
    git_rev = get_git_revision()
    is_smoke = (mode == "smoke")
    max_epochs = phase3_config.max_epochs_smoke if is_smoke else phase3_config.max_epochs_full
    patience = phase3_config.patience_smoke if is_smoke else phase3_config.patience_full
    num_draws = phase3_config.bootstrap_draws_smoke if is_smoke else phase3_config.bootstrap_draws_full

    train_dict = {c: datasets[(c, "train")] for c in sorted(PARTICIPATING_CLIENTS)}
    val_dict = {c: datasets[(c, "val")] for c in sorted(PARTICIPATING_CLIENTS)}
    test_dict = {c: datasets[(c, "test")] for c in sorted(PARTICIPATING_CLIENTS)}
    heldout_test_ds = datasets[("heldout_model1", "test")]

    for seed in seeds:
        run_start = time.time()
        model = AzureGRU(input_size=4, hidden_size=32, num_summaries=8, output_size=1)

        t_fit_start = time.time()
        t_cfg = TrainingConfig(
            regime="central",
            batch_size=phase3_config.batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lr=phase3_config.lr,
            weight_decay=phase3_config.weight_decay,
            seed=seed,
            device=device,
            allow_seed_override=True,
        )
        trainer = AzureNeuralTrainer(model=model, config=t_cfg)
        train_result = trainer.train(train_data=train_dict, val_data=val_dict)
        fit_elapsed = time.time() - t_fit_start

        # Single pooled participant validation threshold
        pooled_val_targets: List[int] = []
        pooled_val_probs: List[float] = []
        pooled_val_meta: List[SampleMetadata] = []

        for c in sorted(PARTICIPATING_CLIENTS):
            c_probs, c_targets = predict_model_probabilities(model, val_dict[c], device=trainer.device)
            pooled_val_probs.extend(c_probs)
            pooled_val_targets.extend(c_targets)
            pooled_val_meta.extend(val_dict[c].metadata)

        thresh_res = select_threshold(pooled_val_targets, pooled_val_probs, metadata=pooled_val_meta)
        threshold = thresh_res.threshold
        val_score = thresh_res.score

        # Multi-client report over participating test datasets + heldout
        all_test_targets: List[int] = []
        all_test_probs: List[float] = []
        all_test_meta: List[SampleMetadata] = []

        for c in sorted(PARTICIPATING_CLIENTS):
            c_test_ds = test_dict[c]
            c_probs, c_targets = predict_model_probabilities(model, c_test_ds, device=trainer.device)
            all_test_probs.extend(c_probs)
            all_test_targets.extend(c_targets)
            all_test_meta.extend(c_test_ds.metadata)

        # Heldout scoring
        h_probs, h_targets = predict_model_probabilities(model, heldout_test_ds, device=trainer.device)
        all_test_probs.extend(h_probs)
        all_test_targets.extend(h_targets)
        all_test_meta.extend(heldout_test_ds.metadata)

        window_report = evaluate_window_report(
            all_test_targets, all_test_probs, all_test_meta, threshold=threshold, split_filter="test"
        )

        # Pooled participating event metrics and bootstrap CIs
        pooled_cohort = tuple(sorted(set(m for c in PARTICIPATING_CLIENTS for m in test_dict[c].cohort_machine_ids)))
        pooled_eps = [ep for m in pooled_cohort for ep in pipeline_data.episodes.get(m, [])]
        participating_meta = [m for c in sorted(PARTICIPATING_CLIENTS) for m in test_dict[c].metadata]
        participating_probs = [s for s, m in zip(all_test_probs, all_test_meta) if m.client_name in PARTICIPATING_CLIENTS]

        pooled_em = evaluate_event_metrics(
            episodes=pooled_eps,
            predictions=participating_meta,
            machine_ids=pooled_cohort,
            scores=participating_probs,
            threshold=threshold,
            split_name="test",
            config=phase2b_config,
        )

        participating_targets = [y for y, m in zip(all_test_targets, all_test_meta) if m.client_name in PARTICIPATING_CLIENTS]
        pooled_w_ci = bootstrap_window_ci(
            participating_targets, participating_probs, participating_meta,
            threshold=threshold, num_draws=num_draws, seed=seed
        )
        pooled_e_ci = bootstrap_event_ci(
            pooled_eps, participating_meta, machine_ids=pooled_cohort,
            scores=participating_probs, threshold=threshold,
            split_name="test", config=phase2b_config, num_draws=num_draws, seed=seed
        )

        # Heldout event metrics and bootstrap CIs
        h_cohort = heldout_test_ds.cohort_machine_ids or ()
        h_eps = [ep for m in h_cohort for ep in pipeline_data.episodes.get(m, [])]
        h_em = evaluate_event_metrics(
            episodes=h_eps,
            predictions=heldout_test_ds.metadata,
            machine_ids=h_cohort,
            scores=h_probs,
            threshold=threshold,
            split_name="test",
            config=phase2b_config,
        )
        h_wm = window_report.heldout.get("heldout_model1") or evaluate_window_metrics(h_targets, h_probs, threshold=threshold)
        h_wci = bootstrap_window_ci(h_targets, h_probs, heldout_test_ds.metadata, threshold=threshold, num_draws=num_draws, seed=seed)
        h_eci = bootstrap_event_ci(h_eps, heldout_test_ds.metadata, machine_ids=h_cohort, scores=h_probs, threshold=threshold, split_name="test", config=phase2b_config, num_draws=num_draws, seed=seed)

        peak_mem, rss_mem = get_memory_metrics(trainer.device)
        pooled_wm = window_report.pooled_participating or evaluate_window_metrics(participating_targets, participating_probs, threshold=threshold)
        total_elapsed = time.time() - run_start

        run_entry = SingleRunResult(
            model_name="gru",
            regime="central",
            client_name="pooled_participating",
            seed=seed,
            mode=mode,
            is_representative=not is_smoke,
            is_plumbing_only=is_smoke,
            is_central_reference=True,
            threshold=threshold,
            val_f1_score=val_score,
            best_epoch=train_result.best_epoch,
            best_score=train_result.best_score,
            total_optimizer_steps=train_result.total_optimizer_steps,
            total_samples_processed=train_result.total_samples_processed,
            total_elapsed_seconds=total_elapsed,
            peak_memory_mb=peak_mem,
            process_rss_mb=rss_mem,
            device=str(trainer.device),
            package_versions=train_result.package_versions,
            environment_info=train_result.environment_info,
            config_hash=config_hash,
            git_revision=git_rev,
            window_positives_test=pooled_wm.n_pos,
            window_negatives_test=pooled_wm.n_neg,
            total_windows_test=pooled_wm.n_total,
            distinct_failure_episodes_test=pooled_em.total_episodes,
            eligible_failure_episodes_test=pooled_em.eligible_episodes,
            detected_failure_episodes_test=pooled_em.detected_episodes,
            test_window_metrics=pooled_wm.to_dict(),
            test_event_metrics=pooled_em.to_dict(),
            test_window_ci={k: v.to_dict() for k, v in pooled_w_ci.items()},
            test_event_ci={k: v.to_dict() for k, v in pooled_e_ci.items()},
            zero_shot_heldout_window_metrics=h_wm.to_dict(),
            zero_shot_heldout_event_metrics=h_em.to_dict(),
            zero_shot_heldout_window_ci={k: v.to_dict() for k, v in h_wci.items()},
            zero_shot_heldout_event_ci={k: v.to_dict() for k, v in h_eci.items()},
            central_window_report=window_report.to_dict(),
            fit_elapsed_seconds=fit_elapsed,
        )
        results.append(run_entry)

    return results


def run_local_logistic(
    datasets: Dict[Tuple[str, str], ClientDataset],
    pipeline_data: PipelineData,
    phase3_config: Phase3AzureConfig,
    phase2b_config: Phase2BConfig,
    *,
    seeds: Sequence[int],
    mode: str = "smoke",
) -> List[SingleRunResult]:
    """Execute Local Logistic Regression baselines on 20 causal tabular features."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn ('sklearn') is required for AzureLogisticRegression.")

    results: List[SingleRunResult] = []
    config_hash = phase3_config.compute_hash()
    git_rev = get_git_revision()
    is_smoke = (mode == "smoke")
    num_draws = phase3_config.bootstrap_draws_smoke if is_smoke else phase3_config.bootstrap_draws_full
    heldout_test_ds = datasets[("heldout_model1", "test")]

    package_versions = {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", None) if _TORCH_AVAILABLE else None,
        "numpy": getattr(np, "__version__", None) if _HAS_NUMPY else None,
        "sklearn": getattr(sklearn, "__version__", None) if _SKLEARN_AVAILABLE else None,
    }

    for seed in seeds:
        for client_name in sorted(PARTICIPATING_CLIENTS):
            train_ds = datasets[(client_name, "train")]
            val_ds = datasets[(client_name, "val")]
            test_ds = datasets[(client_name, "test")]

            run_start = time.time()
            model = AzureLogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")

            t_fit_start = time.time()
            model.fit(train_ds.tabular, train_ds.labels, split_name="train")
            fit_elapsed = time.time() - t_fit_start

            n_iter = int(model.classifier.n_iter_[0]) if hasattr(model.classifier, "n_iter_") else 1
            converged = bool(n_iter < model.max_iter)

            # Val threshold
            val_probs, val_targets = predict_model_probabilities(model, val_ds)
            thresh_res = select_threshold(val_targets, val_probs, metadata=val_ds.metadata)
            threshold = thresh_res.threshold
            val_score = thresh_res.score

            # Test evaluation
            wm, em, w_ci, e_ci, w_pos, w_neg, ep_tot, ep_elig, ep_det = evaluate_model_partition(
                model=model,
                dataset=test_ds,
                threshold=threshold,
                all_episodes=pipeline_data.episodes,
                num_bootstrap_draws=num_draws,
                seed=seed,
                phase2b_config=phase2b_config,
                is_smoke=is_smoke,
            )

            # Heldout evaluation
            h_wm, h_em, h_wci, h_eci, _, _, _, _, _ = evaluate_model_partition(
                model=model,
                dataset=heldout_test_ds,
                threshold=threshold,
                all_episodes=pipeline_data.episodes,
                num_bootstrap_draws=num_draws,
                seed=seed,
                phase2b_config=phase2b_config,
                is_smoke=is_smoke,
            )

            peak_mem, rss_mem = get_memory_metrics(None)
            total_elapsed = time.time() - run_start

            run_entry = SingleRunResult(
                model_name="logistic",
                regime="local",
                client_name=client_name,
                seed=seed,
                mode=mode,
                is_representative=not is_smoke,
                is_plumbing_only=is_smoke,
                is_central_reference=False,
                threshold=threshold,
                val_f1_score=val_score,
                best_epoch=None,
                best_score=None,
                total_optimizer_steps=n_iter,
                total_samples_processed=len(train_ds),
                total_elapsed_seconds=total_elapsed,
                peak_memory_mb=peak_mem,
                process_rss_mb=rss_mem,
                device="cpu",
                package_versions=package_versions,
                environment_info=get_environment_info(None),
                config_hash=config_hash,
                git_revision=git_rev,
                window_positives_test=w_pos,
                window_negatives_test=w_neg,
                total_windows_test=w_pos + w_neg,
                distinct_failure_episodes_test=ep_tot,
                eligible_failure_episodes_test=ep_elig,
                detected_failure_episodes_test=ep_det,
                test_window_metrics=wm.to_dict(),
                test_event_metrics=em.to_dict(),
                test_window_ci={k: v.to_dict() for k, v in w_ci.items()},
                test_event_ci={k: v.to_dict() for k, v in e_ci.items()},
                zero_shot_heldout_window_metrics=h_wm.to_dict(),
                zero_shot_heldout_event_metrics=h_em.to_dict(),
                zero_shot_heldout_window_ci={k: v.to_dict() for k, v in h_wci.items()},
                zero_shot_heldout_event_ci={k: v.to_dict() for k, v in h_eci.items()},
                logistic_details={
                    "solver": model.solver,
                    "max_iter": model.max_iter,
                    "n_iter": n_iter,
                    "converged": converged,
                    "C": model.C,
                },
                fit_elapsed_seconds=fit_elapsed,
            )
            results.append(run_entry)

    return results


def run_central_logistic(
    datasets: Dict[Tuple[str, str], ClientDataset],
    pipeline_data: PipelineData,
    phase3_config: Phase3AzureConfig,
    phase2b_config: Phase2BConfig,
    *,
    seeds: Sequence[int],
    mode: str = "smoke",
) -> List[SingleRunResult]:
    """Execute Centralized Reference Logistic Regression on pooled 20 tabular features."""
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn ('sklearn') is required for AzureLogisticRegression.")

    results: List[SingleRunResult] = []
    config_hash = phase3_config.compute_hash()
    git_rev = get_git_revision()
    is_smoke = (mode == "smoke")
    num_draws = phase3_config.bootstrap_draws_smoke if is_smoke else phase3_config.bootstrap_draws_full
    heldout_test_ds = datasets[("heldout_model1", "test")]

    package_versions = {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", None) if _TORCH_AVAILABLE else None,
        "numpy": getattr(np, "__version__", None) if _HAS_NUMPY else None,
        "sklearn": getattr(sklearn, "__version__", None) if _SKLEARN_AVAILABLE else None,
    }

    test_dict = {c: datasets[(c, "test")] for c in sorted(PARTICIPATING_CLIENTS)}

    # Pool training tabular data
    train_tabs = [datasets[(c, "train")].tabular for c in sorted(PARTICIPATING_CLIENTS)]
    train_lbls = [datasets[(c, "train")].labels for c in sorted(PARTICIPATING_CLIENTS)]

    if _HAS_NUMPY:
        pooled_train_X = np.concatenate(train_tabs, axis=0)
        pooled_train_y = np.concatenate(train_lbls, axis=0)
    else:
        pooled_train_X = [row for tab in train_tabs for row in tab]
        pooled_train_y = [y for lbl in train_lbls for y in lbl]

    for seed in seeds:
        run_start = time.time()
        model = AzureLogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")

        t_fit_start = time.time()
        model.fit(pooled_train_X, pooled_train_y, split_name="train")
        fit_elapsed = time.time() - t_fit_start

        n_iter = int(model.classifier.n_iter_[0]) if hasattr(model.classifier, "n_iter_") else 1
        converged = bool(n_iter < model.max_iter)

        # Pooled validation threshold
        pooled_val_probs: List[float] = []
        pooled_val_targets: List[int] = []
        pooled_val_meta: List[SampleMetadata] = []

        for c in sorted(PARTICIPATING_CLIENTS):
            v_ds = datasets[(c, "val")]
            c_probs, c_targets = predict_model_probabilities(model, v_ds)
            pooled_val_probs.extend(c_probs)
            pooled_val_targets.extend(c_targets)
            pooled_val_meta.extend(v_ds.metadata)

        thresh_res = select_threshold(pooled_val_targets, pooled_val_probs, metadata=pooled_val_meta)
        threshold = thresh_res.threshold
        val_score = thresh_res.score

        # Multi-client report over participating test datasets + heldout
        all_test_targets: List[int] = []
        all_test_probs: List[float] = []
        all_test_meta: List[SampleMetadata] = []

        for c in sorted(PARTICIPATING_CLIENTS):
            c_test_ds = test_dict[c]
            c_probs, c_targets = predict_model_probabilities(model, c_test_ds)
            all_test_probs.extend(c_probs)
            all_test_targets.extend(c_targets)
            all_test_meta.extend(c_test_ds.metadata)

        h_probs, h_targets = predict_model_probabilities(model, heldout_test_ds)
        all_test_probs.extend(h_probs)
        all_test_targets.extend(h_targets)
        all_test_meta.extend(heldout_test_ds.metadata)

        window_report = evaluate_window_report(
            all_test_targets, all_test_probs, all_test_meta, threshold=threshold, split_filter="test"
        )

        pooled_cohort = tuple(sorted(set(m for c in PARTICIPATING_CLIENTS for m in test_dict[c].cohort_machine_ids)))
        pooled_eps = [ep for m in pooled_cohort for ep in pipeline_data.episodes.get(m, [])]
        participating_meta = [m for c in sorted(PARTICIPATING_CLIENTS) for m in test_dict[c].metadata]
        participating_probs = [s for s, m in zip(all_test_probs, all_test_meta) if m.client_name in PARTICIPATING_CLIENTS]
        participating_targets = [y for y, m in zip(all_test_targets, all_test_meta) if m.client_name in PARTICIPATING_CLIENTS]

        pooled_em = evaluate_event_metrics(
            episodes=pooled_eps,
            predictions=participating_meta,
            machine_ids=pooled_cohort,
            scores=participating_probs,
            threshold=threshold,
            split_name="test",
            config=phase2b_config,
        )

        pooled_w_ci = bootstrap_window_ci(
            participating_targets, participating_probs, participating_meta,
            threshold=threshold, num_draws=num_draws, seed=seed
        )
        pooled_e_ci = bootstrap_event_ci(
            pooled_eps, participating_meta, machine_ids=pooled_cohort,
            scores=participating_probs, threshold=threshold,
            split_name="test", config=phase2b_config, num_draws=num_draws, seed=seed
        )

        h_cohort = heldout_test_ds.cohort_machine_ids or ()
        h_eps = [ep for m in h_cohort for ep in pipeline_data.episodes.get(m, [])]
        h_em = evaluate_event_metrics(
            episodes=h_eps,
            predictions=heldout_test_ds.metadata,
            machine_ids=h_cohort,
            scores=h_probs,
            threshold=threshold,
            split_name="test",
            config=phase2b_config,
        )
        h_wm = window_report.heldout.get("heldout_model1") or evaluate_window_metrics(h_targets, h_probs, threshold=threshold)
        h_wci = bootstrap_window_ci(h_targets, h_probs, heldout_test_ds.metadata, threshold=threshold, num_draws=num_draws, seed=seed)
        h_eci = bootstrap_event_ci(h_eps, heldout_test_ds.metadata, machine_ids=h_cohort, scores=h_probs, threshold=threshold, split_name="test", config=phase2b_config, num_draws=num_draws, seed=seed)

        peak_mem, rss_mem = get_memory_metrics(None)
        pooled_wm = window_report.pooled_participating or evaluate_window_metrics(participating_targets, participating_probs, threshold=threshold)
        total_elapsed = time.time() - run_start

        run_entry = SingleRunResult(
            model_name="logistic",
            regime="central",
            client_name="pooled_participating",
            seed=seed,
            mode=mode,
            is_representative=not is_smoke,
            is_plumbing_only=is_smoke,
            is_central_reference=True,
            threshold=threshold,
            val_f1_score=val_score,
            best_epoch=None,
            best_score=None,
            total_optimizer_steps=n_iter,
            total_samples_processed=len(pooled_train_X),
            total_elapsed_seconds=total_elapsed,
            peak_memory_mb=peak_mem,
            process_rss_mb=rss_mem,
            device="cpu",
            package_versions=package_versions,
            environment_info=get_environment_info(None),
            config_hash=config_hash,
            git_revision=git_rev,
            window_positives_test=pooled_wm.n_pos,
            window_negatives_test=pooled_wm.n_neg,
            total_windows_test=pooled_wm.n_total,
            distinct_failure_episodes_test=pooled_em.total_episodes,
            eligible_failure_episodes_test=pooled_em.eligible_episodes,
            detected_failure_episodes_test=pooled_em.detected_episodes,
            test_window_metrics=pooled_wm.to_dict(),
            test_event_metrics=pooled_em.to_dict(),
            test_window_ci={k: v.to_dict() for k, v in pooled_w_ci.items()},
            test_event_ci={k: v.to_dict() for k, v in pooled_e_ci.items()},
            zero_shot_heldout_window_metrics=h_wm.to_dict(),
            zero_shot_heldout_event_metrics=h_em.to_dict(),
            zero_shot_heldout_window_ci={k: v.to_dict() for k, v in h_wci.items()},
            zero_shot_heldout_event_ci={k: v.to_dict() for k, v in h_eci.items()},
            central_window_report=window_report.to_dict(),
            logistic_details={
                "solver": model.solver,
                "max_iter": model.max_iter,
                "n_iter": n_iter,
                "converged": converged,
                "C": model.C,
            },
            fit_elapsed_seconds=fit_elapsed,
        )
        results.append(run_entry)

    return results


def run_local_mlp(
    datasets: Dict[Tuple[str, str], ClientDataset],
    pipeline_data: PipelineData,
    phase3_config: Phase3AzureConfig,
    phase2b_config: Phase2BConfig,
    *,
    seeds: Sequence[int],
    mode: str = "smoke",
    device: Optional[str] = None,
) -> List[SingleRunResult]:
    """Execute Local MLP baseline on 20 causal tabular features."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch ('torch') is required for AzureMLP.")

    results: List[SingleRunResult] = []
    config_hash = phase3_config.compute_hash()
    git_rev = get_git_revision()
    is_smoke = (mode == "smoke")
    max_epochs = phase3_config.max_epochs_smoke if is_smoke else phase3_config.max_epochs_full
    patience = phase3_config.patience_smoke if is_smoke else phase3_config.patience_full
    num_draws = phase3_config.bootstrap_draws_smoke if is_smoke else phase3_config.bootstrap_draws_full
    heldout_test_ds = datasets[("heldout_model1", "test")]

    for seed in seeds:
        for client_name in sorted(PARTICIPATING_CLIENTS):
            train_ds = datasets[(client_name, "train")]
            val_ds = datasets[(client_name, "val")]
            test_ds = datasets[(client_name, "test")]

            run_start = time.time()
            model = AzureMLP(input_dim=20, hidden_dim=32, output_dim=1)

            t_fit_start = time.time()
            t_cfg = TrainingConfig(
                regime="local",
                batch_size=phase3_config.batch_size,
                max_epochs=max_epochs,
                patience=patience,
                lr=phase3_config.lr,
                weight_decay=phase3_config.weight_decay,
                seed=seed,
                device=device,
                fit_mlp_scaler=True,
                allow_seed_override=True,
            )
            trainer = AzureNeuralTrainer(model=model, config=t_cfg)
            train_result = trainer.train(train_data=train_ds, val_data=val_ds)
            fit_elapsed = time.time() - t_fit_start

            val_probs, val_targets = predict_model_probabilities(model, val_ds, device=trainer.device)
            thresh_res = select_threshold(val_targets, val_probs, metadata=val_ds.metadata)
            threshold = thresh_res.threshold
            val_score = thresh_res.score

            wm, em, w_ci, e_ci, w_pos, w_neg, ep_tot, ep_elig, ep_det = evaluate_model_partition(
                model=model,
                dataset=test_ds,
                threshold=threshold,
                all_episodes=pipeline_data.episodes,
                num_bootstrap_draws=num_draws,
                seed=seed,
                device=trainer.device,
                phase2b_config=phase2b_config,
                is_smoke=is_smoke,
            )

            h_wm, h_em, h_wci, h_eci, _, _, _, _, _ = evaluate_model_partition(
                model=model,
                dataset=heldout_test_ds,
                threshold=threshold,
                all_episodes=pipeline_data.episodes,
                num_bootstrap_draws=num_draws,
                seed=seed,
                device=trainer.device,
                phase2b_config=phase2b_config,
                is_smoke=is_smoke,
            )

            peak_mem, rss_mem = get_memory_metrics(trainer.device)
            total_elapsed = time.time() - run_start

            run_entry = SingleRunResult(
                model_name="mlp",
                regime="local",
                client_name=client_name,
                seed=seed,
                mode=mode,
                is_representative=not is_smoke,
                is_plumbing_only=is_smoke,
                is_central_reference=False,
                threshold=threshold,
                val_f1_score=val_score,
                best_epoch=train_result.best_epoch,
                best_score=train_result.best_score,
                total_optimizer_steps=train_result.total_optimizer_steps,
                total_samples_processed=train_result.total_samples_processed,
                total_elapsed_seconds=total_elapsed,
                peak_memory_mb=peak_mem,
                process_rss_mb=rss_mem,
                device=str(trainer.device),
                package_versions=train_result.package_versions,
                environment_info=train_result.environment_info,
                config_hash=config_hash,
                git_revision=git_rev,
                window_positives_test=w_pos,
                window_negatives_test=w_neg,
                total_windows_test=w_pos + w_neg,
                distinct_failure_episodes_test=ep_tot,
                eligible_failure_episodes_test=ep_elig,
                detected_failure_episodes_test=ep_det,
                test_window_metrics=wm.to_dict(),
                test_event_metrics=em.to_dict(),
                test_window_ci={k: v.to_dict() for k, v in w_ci.items()},
                test_event_ci={k: v.to_dict() for k, v in e_ci.items()},
                zero_shot_heldout_window_metrics=h_wm.to_dict(),
                zero_shot_heldout_event_metrics=h_em.to_dict(),
                zero_shot_heldout_window_ci={k: v.to_dict() for k, v in h_wci.items()},
                zero_shot_heldout_event_ci={k: v.to_dict() for k, v in h_eci.items()},
                fit_elapsed_seconds=fit_elapsed,
            )
            results.append(run_entry)

    return results


def run_central_mlp(
    datasets: Dict[Tuple[str, str], ClientDataset],
    pipeline_data: PipelineData,
    phase3_config: Phase3AzureConfig,
    phase2b_config: Phase2BConfig,
    *,
    seeds: Sequence[int],
    mode: str = "smoke",
    device: Optional[str] = None,
) -> List[SingleRunResult]:
    """Execute Centralized Reference MLP on pooled 20 tabular features."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch ('torch') is required for AzureMLP.")

    results: List[SingleRunResult] = []
    config_hash = phase3_config.compute_hash()
    git_rev = get_git_revision()
    is_smoke = (mode == "smoke")
    max_epochs = phase3_config.max_epochs_smoke if is_smoke else phase3_config.max_epochs_full
    patience = phase3_config.patience_smoke if is_smoke else phase3_config.patience_full
    num_draws = phase3_config.bootstrap_draws_smoke if is_smoke else phase3_config.bootstrap_draws_full

    train_dict = {c: datasets[(c, "train")] for c in sorted(PARTICIPATING_CLIENTS)}
    val_dict = {c: datasets[(c, "val")] for c in sorted(PARTICIPATING_CLIENTS)}
    test_dict = {c: datasets[(c, "test")] for c in sorted(PARTICIPATING_CLIENTS)}
    heldout_test_ds = datasets[("heldout_model1", "test")]

    for seed in seeds:
        run_start = time.time()
        model = AzureMLP(input_dim=20, hidden_dim=32, output_dim=1)

        t_fit_start = time.time()
        t_cfg = TrainingConfig(
            regime="central",
            batch_size=phase3_config.batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lr=phase3_config.lr,
            weight_decay=phase3_config.weight_decay,
            seed=seed,
            device=device,
            fit_mlp_scaler=True,
            allow_seed_override=True,
        )
        trainer = AzureNeuralTrainer(model=model, config=t_cfg)
        train_result = trainer.train(train_data=train_dict, val_data=val_dict)
        fit_elapsed = time.time() - t_fit_start

        # Single pooled participant validation threshold
        pooled_val_targets: List[int] = []
        pooled_val_probs: List[float] = []
        pooled_val_meta: List[SampleMetadata] = []

        for c in sorted(PARTICIPATING_CLIENTS):
            c_probs, c_targets = predict_model_probabilities(model, val_dict[c], device=trainer.device)
            pooled_val_probs.extend(c_probs)
            pooled_val_targets.extend(c_targets)
            pooled_val_meta.extend(val_dict[c].metadata)

        thresh_res = select_threshold(pooled_val_targets, pooled_val_probs, metadata=pooled_val_meta)
        threshold = thresh_res.threshold
        val_score = thresh_res.score

        all_test_targets: List[int] = []
        all_test_probs: List[float] = []
        all_test_meta: List[SampleMetadata] = []

        for c in sorted(PARTICIPATING_CLIENTS):
            c_test_ds = test_dict[c]
            c_probs, c_targets = predict_model_probabilities(model, c_test_ds, device=trainer.device)
            all_test_probs.extend(c_probs)
            all_test_targets.extend(c_targets)
            all_test_meta.extend(c_test_ds.metadata)

        h_probs, h_targets = predict_model_probabilities(model, heldout_test_ds, device=trainer.device)
        all_test_probs.extend(h_probs)
        all_test_targets.extend(h_targets)
        all_test_meta.extend(heldout_test_ds.metadata)

        window_report = evaluate_window_report(
            all_test_targets, all_test_probs, all_test_meta, threshold=threshold, split_filter="test"
        )

        pooled_cohort = tuple(sorted(set(m for c in PARTICIPATING_CLIENTS for m in test_dict[c].cohort_machine_ids)))
        pooled_eps = [ep for m in pooled_cohort for ep in pipeline_data.episodes.get(m, [])]
        participating_meta = [m for c in sorted(PARTICIPATING_CLIENTS) for m in test_dict[c].metadata]
        participating_probs = [s for s, m in zip(all_test_probs, all_test_meta) if m.client_name in PARTICIPATING_CLIENTS]
        participating_targets = [y for y, m in zip(all_test_targets, all_test_meta) if m.client_name in PARTICIPATING_CLIENTS]

        pooled_em = evaluate_event_metrics(
            episodes=pooled_eps,
            predictions=participating_meta,
            machine_ids=pooled_cohort,
            scores=participating_probs,
            threshold=threshold,
            split_name="test",
            config=phase2b_config,
        )

        pooled_w_ci = bootstrap_window_ci(
            participating_targets, participating_probs, participating_meta,
            threshold=threshold, num_draws=num_draws, seed=seed
        )
        pooled_e_ci = bootstrap_event_ci(
            pooled_eps, participating_meta, machine_ids=pooled_cohort,
            scores=participating_probs, threshold=threshold,
            split_name="test", config=phase2b_config, num_draws=num_draws, seed=seed
        )

        h_cohort = heldout_test_ds.cohort_machine_ids or ()
        h_eps = [ep for m in h_cohort for ep in pipeline_data.episodes.get(m, [])]
        h_em = evaluate_event_metrics(
            episodes=h_eps,
            predictions=heldout_test_ds.metadata,
            machine_ids=h_cohort,
            scores=h_probs,
            threshold=threshold,
            split_name="test",
            config=phase2b_config,
        )
        h_wm = window_report.heldout.get("heldout_model1") or evaluate_window_metrics(h_targets, h_probs, threshold=threshold)
        h_wci = bootstrap_window_ci(h_targets, h_probs, heldout_test_ds.metadata, threshold=threshold, num_draws=num_draws, seed=seed)
        h_eci = bootstrap_event_ci(h_eps, heldout_test_ds.metadata, machine_ids=h_cohort, scores=h_probs, threshold=threshold, split_name="test", config=phase2b_config, num_draws=num_draws, seed=seed)

        peak_mem, rss_mem = get_memory_metrics(trainer.device)
        pooled_wm = window_report.pooled_participating or evaluate_window_metrics(participating_targets, participating_probs, threshold=threshold)
        total_elapsed = time.time() - run_start

        run_entry = SingleRunResult(
            model_name="mlp",
            regime="central",
            client_name="pooled_participating",
            seed=seed,
            mode=mode,
            is_representative=not is_smoke,
            is_plumbing_only=is_smoke,
            is_central_reference=True,
            threshold=threshold,
            val_f1_score=val_score,
            best_epoch=train_result.best_epoch,
            best_score=train_result.best_score,
            total_optimizer_steps=train_result.total_optimizer_steps,
            total_samples_processed=train_result.total_samples_processed,
            total_elapsed_seconds=total_elapsed,
            peak_memory_mb=peak_mem,
            process_rss_mb=rss_mem,
            device=str(trainer.device),
            package_versions=train_result.package_versions,
            environment_info=train_result.environment_info,
            config_hash=config_hash,
            git_revision=git_rev,
            window_positives_test=pooled_wm.n_pos,
            window_negatives_test=pooled_wm.n_neg,
            total_windows_test=pooled_wm.n_total,
            distinct_failure_episodes_test=pooled_em.total_episodes,
            eligible_failure_episodes_test=pooled_em.eligible_episodes,
            detected_failure_episodes_test=pooled_em.detected_episodes,
            test_window_metrics=pooled_wm.to_dict(),
            test_event_metrics=pooled_em.to_dict(),
            test_window_ci={k: v.to_dict() for k, v in pooled_w_ci.items()},
            test_event_ci={k: v.to_dict() for k, v in pooled_e_ci.items()},
            zero_shot_heldout_window_metrics=h_wm.to_dict(),
            zero_shot_heldout_event_metrics=h_em.to_dict(),
            zero_shot_heldout_window_ci={k: v.to_dict() for k, v in h_wci.items()},
            zero_shot_heldout_event_ci={k: v.to_dict() for k, v in h_eci.items()},
            central_window_report=window_report.to_dict(),
            fit_elapsed_seconds=fit_elapsed,
        )
        results.append(run_entry)

    return results


@dataclass
class Phase3ExperimentReport:
    """Consolidated report across models, regimes, and seeds."""
    pipeline_name: str
    phase: str
    mode: str
    is_representative: bool
    is_plumbing_only: bool
    config_hash: str
    git_revision: Optional[str]
    created_at: str
    output_dir: str
    models_executed: List[str]
    regimes_executed: List[str]
    seeds_executed: List[int]
    results: List[SingleRunResult]
    summary_table: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return _safe_serialize(asdict(self))

    def write_json(self, output_path: Union[str, Path]) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p


def build_summary_table(results: Sequence[SingleRunResult]) -> List[Dict[str, Any]]:
    """Generate concise tabular comparison for reporting."""
    table = []
    for r in results:
        wm = r.test_window_metrics
        em = r.test_event_metrics
        h_wm = r.zero_shot_heldout_window_metrics
        h_em = r.zero_shot_heldout_event_metrics
        table.append({
            "model": r.model_name,
            "regime": r.regime,
            "client": r.client_name,
            "seed": r.seed,
            "threshold": round(r.threshold, 4),
            "val_f1": round(r.val_f1_score, 4),
            "test_ap": round(wm.get("average_precision", 0.0) or 0.0, 4) if wm.get("average_precision") is not None else None,
            "test_f1": round(wm.get("f1", 0.0), 4),
            "test_event_recall": round(em.get("event_recall", 0.0) or 0.0, 4) if em.get("event_recall") is not None else None,
            "test_far_per_100_days": round(em.get("false_alerts_per_100_machine_days", 0.0) or 0.0, 2) if em.get("false_alerts_per_100_machine_days") is not None else None,
            "heldout_ap": round(h_wm.get("average_precision", 0.0) or 0.0, 4) if h_wm.get("average_precision") is not None else None,
            "heldout_f1": round(h_wm.get("f1", 0.0), 4),
            "heldout_event_recall": round(h_em.get("event_recall", 0.0) or 0.0, 4) if h_em.get("event_recall") is not None else None,
            "is_central_ref": r.is_central_reference,
            "runtime_s": round(r.total_elapsed_seconds, 2),
            "fit_time_s": round(r.fit_elapsed_seconds, 2) if r.fit_elapsed_seconds is not None else None,
        })
    return table


def run_phase3_experiment(
    *,
    config: Optional[Union[str, Path, Phase3AzureConfig]] = None,
    mode: str = "smoke",
    models: Optional[Sequence[str]] = None,
    regimes: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    pipeline_data: Optional[PipelineData] = None,
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    force_rebuild_cache: bool = False,
) -> Phase3ExperimentReport:
    """Execute end-to-end Phase 3 Azure baseline experiments across models and regimes.

    Args:
        config: Phase3AzureConfig instance or path to config JSON.
        mode: 'smoke' (one seed, 2 epochs, bounded subset, plumbing only) or 'full'.
        models: Subset of ('gru', 'logistic', 'mlp', 'all'). Defaults to 'all'.
        regimes: Subset of ('local', 'central', 'all'). Defaults to 'all'.
        seeds: Optional override list of seeds. Defaults to config seeds for mode.
        pipeline_data: Optional caller-injected PipelineData for data-free testing.
        data_dir: Optional directory with raw Azure CSVs (used if pipeline_data is None).
        output_dir: Directory where results JSON will be written (default outputs/phase3).
        cache_dir: Directory where memmaps will be stored (default output_dir/cache).
        device: Torch device (e.g. 'cpu', 'cuda').
        force_rebuild_cache: Force rebuild of memmap caches.

    Returns:
        Phase3ExperimentReport containing results, metrics, summary table, and provenance.
    """
    if mode not in ("smoke", "full"):
        raise ValueError(f"mode must be 'smoke' or 'full'; got {mode!r}")

    if isinstance(config, Phase3AzureConfig):
        cfg3 = config
    elif config is not None:
        cfg3 = Phase3AzureConfig.from_file(config)
    else:
        cfg3 = load_phase3_azure_config()

    cfg2b = load_azure_preprocessing_config()

    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    c_dir = Path(cache_dir) if cache_dir is not None else (out_dir / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)

    # Determine seeds
    if seeds is not None:
        active_seeds = list(seeds)
    elif mode == "smoke":
        active_seeds = list(cfg3.seeds_smoke)
    else:
        active_seeds = list(cfg3.seeds_full)

    # Determine models
    if models is None or "all" in models:
        active_models = ["gru", "logistic", "mlp"]
    else:
        active_models = [m.lower() for m in models]

    # Determine regimes
    if regimes is None or "all" in regimes:
        active_regimes = ["local", "central"]
    else:
        active_regimes = [r.lower() for r in regimes]

    # Load datasets and shared PipelineData
    datasets, shared_pipeline_data = load_phase3_datasets(
        cache_root=c_dir,
        pipeline_data=pipeline_data,
        data_dir=data_dir,
        phase2b_config=cfg2b,
        phase3_config=cfg3,
        force_rebuild=force_rebuild_cache,
        mode=mode,
    )

    all_results: List[SingleRunResult] = []

    # 1. GRU
    if "gru" in active_models:
        if "local" in active_regimes:
            gru_local_results = run_local_gru(
                datasets=datasets,
                pipeline_data=shared_pipeline_data,
                phase3_config=cfg3,
                phase2b_config=cfg2b,
                seeds=active_seeds,
                mode=mode,
                device=device,
            )
            all_results.extend(gru_local_results)

        if "central" in active_regimes:
            gru_central_results = run_central_gru(
                datasets=datasets,
                pipeline_data=shared_pipeline_data,
                phase3_config=cfg3,
                phase2b_config=cfg2b,
                seeds=active_seeds,
                mode=mode,
                device=device,
            )
            all_results.extend(gru_central_results)

    # 2. Logistic Regression
    if "logistic" in active_models:
        if "local" in active_regimes:
            log_local_results = run_local_logistic(
                datasets=datasets,
                pipeline_data=shared_pipeline_data,
                phase3_config=cfg3,
                phase2b_config=cfg2b,
                seeds=active_seeds,
                mode=mode,
            )
            all_results.extend(log_local_results)

        if "central" in active_regimes:
            log_central_results = run_central_logistic(
                datasets=datasets,
                pipeline_data=shared_pipeline_data,
                phase3_config=cfg3,
                phase2b_config=cfg2b,
                seeds=active_seeds,
                mode=mode,
            )
            all_results.extend(log_central_results)

    # 3. MLP
    if "mlp" in active_models:
        if "local" in active_regimes:
            mlp_local_results = run_local_mlp(
                datasets=datasets,
                pipeline_data=shared_pipeline_data,
                phase3_config=cfg3,
                phase2b_config=cfg2b,
                seeds=active_seeds,
                mode=mode,
                device=device,
            )
            all_results.extend(mlp_local_results)

        if "central" in active_regimes:
            mlp_central_results = run_central_mlp(
                datasets=datasets,
                pipeline_data=shared_pipeline_data,
                phase3_config=cfg3,
                phase2b_config=cfg2b,
                seeds=active_seeds,
                mode=mode,
                device=device,
            )
            all_results.extend(mlp_central_results)

    summary_table = build_summary_table(all_results)
    is_smoke = (mode == "smoke")

    report = Phase3ExperimentReport(
        pipeline_name=cfg3.pipeline_name,
        phase=cfg3.phase,
        mode=mode,
        is_representative=not is_smoke,
        is_plumbing_only=is_smoke,
        config_hash=cfg3.compute_hash(),
        git_revision=get_git_revision(),
        created_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(out_dir),
        models_executed=active_models,
        regimes_executed=active_regimes,
        seeds_executed=active_seeds,
        results=all_results,
        summary_table=summary_table,
    )

    # Serialize results.json
    results_json_path = out_dir / "results.json"
    report.write_json(results_json_path)

    return report
