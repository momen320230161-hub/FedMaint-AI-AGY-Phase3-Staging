"""Model-ready cache bridge for Phase 3 Azure PdM.

This module coordinates:
- Pre-computing and caching model-ready features from Phase 2B AzureSample streams.
- Zero-copy memmap backing using numpy .npy open_memmap for:
    * Sequences: [N, 48, 4] float32
    * Summaries: [N, 8] float32
    * Tabular: [N, 20] float32
    * Labels: [N] int64
    * Compact metadata: machine IDs [N] int32, timestamps [N] int64
- Memory-bounded lazy metadata Sequence generating SampleMetadata on access without
  materializing hundreds of thousands of Python objects resident in memory.
- Strict client and split partition enforcement:
    * Participating clients (client_model2, client_model3, client_model4): train, val, test
    * Held-out client (heldout_model1): test ONLY
    * Rejection of any train or val caching/loading for held-out client with HoldoutLeakageError
- Deterministic cache path derived strictly from caller output_root and client/split.
- Atomic completion marker and manifest tracking config hash, source data manifest hash,
  code/protocol version, shapes, and counts.
- Safe cleanup of stale or incomplete caches without recursive deletion of arbitrary paths.
- Conversion to existing Phase 3 ClientDataset interface without copying full arrays.
- Storage of explicit cohort machine IDs from get_client_partitions for downstream evaluation.
"""
from __future__ import annotations

import collections.abc
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    from numpy.lib.format import open_memmap
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    open_memmap = None  # type: ignore
    _HAS_NUMPY = False

from .adapter import DEFAULT_SENSORS, SampleMetadata, adapt_sample
from .azure import DEFAULT_AZURE_DIR
from .config import DEFAULT_CONFIG_PATH, Phase2BConfig, load_azure_preprocessing_config
from .pipeline import (
    AzureSample,
    PipelineData,
    generate_azure_windows,
    load_pipeline_data,
    summarize_azure_pipeline,
)
from .splits import (
    ClientPartition,
    HoldoutLeakageError,
    get_client_partitions,
)
from ..training.engine import ClientDataset

CODE_VERSION: str = "1.0.0"

DEFAULT_SOURCE_MANIFEST_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "manifest.json"
)

ALLOWED_CLIENT_SPLITS: Dict[str, Tuple[str, ...]] = {
    "client_model2": ("train", "val", "test"),
    "client_model3": ("train", "val", "test"),
    "client_model4": ("train", "val", "test"),
    "heldout_model1": ("test",),
}

KNOWN_CACHE_FILES: Tuple[str, ...] = (
    "sequences.npy",
    "summaries.npy",
    "tabular.npy",
    "labels.npy",
    "metadata_machine_ids.npy",
    "metadata_timestamps.npy",
    "manifest.json",
    "manifest.json.tmp",
    "completed.marker",
)


class CacheError(Exception):
    """Base exception for model cache errors."""


class CacheIncompleteError(CacheError):
    """Raised when a cached dataset is incomplete, corrupted, or missing completion markers."""


class CacheStaleError(CacheError):
    """Raised when a cached dataset does not match the active config, code, or source version."""


class LazyMetadataSequence(collections.abc.Sequence):
    """Memory-bounded, read-only Sequence generating SampleMetadata on access.

    Avoids materializing hundreds of thousands of resident Python SampleMetadata objects.
    Reconstructs SampleMetadata on demand using compact machine IDs and prediction timestamps
    with known partition metadata and Phase 2B parameters.
    """

    def __init__(
        self,
        machine_ids: Any,
        timestamps: Any,
        client_name: str,
        split_name: str,
        *,
        history_hours: int = 48,
        horizon_hours: int = 24,
    ) -> None:
        self._machine_ids = machine_ids
        self._timestamps = timestamps
        self.client_name = str(client_name)
        self.split_name = str(split_name)
        self.history_hours = int(history_hours)
        self.horizon_hours = int(horizon_hours)

        if len(self._machine_ids) != len(self._timestamps):
            raise ValueError(
                f"Length mismatch: machine_ids ({len(self._machine_ids)}) != "
                f"timestamps ({len(self._timestamps)})"
            )
        self._length = len(self._machine_ids)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Union[int, slice]) -> Union[SampleMetadata, "LazyMetadataSequence"]:
        if isinstance(index, slice):
            return LazyMetadataSequence(
                machine_ids=self._machine_ids[index],
                timestamps=self._timestamps[index],
                client_name=self.client_name,
                split_name=self.split_name,
                history_hours=self.history_hours,
                horizon_hours=self.horizon_hours,
            )

        idx = int(index)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {index} out of range for sequence of length {self._length}")

        m_id = int(self._machine_ids[idx])
        ts_sec = int(self._timestamps[idx])
        pred_dt = datetime(1970, 1, 1) + timedelta(seconds=ts_sec)

        return SampleMetadata(
            machine_id=m_id,
            prediction_time=pred_dt,
            client_name=self.client_name,
            split_name=self.split_name,
            observation_start=pred_dt - timedelta(hours=self.history_hours),
            observation_end=pred_dt - timedelta(hours=1),
            horizon_start=pred_dt,
            horizon_end=pred_dt + timedelta(hours=self.horizon_hours),
        )

    def __iter__(self) -> Iterator[SampleMetadata]:
        for i in range(self._length):
            yield self[i]  # type: ignore

    def __repr__(self) -> str:
        return (
            f"LazyMetadataSequence(client_name={self.client_name!r}, split_name={self.split_name!r}, "
            f"length={self._length})"
        )


@dataclass(frozen=True)
class CacheManifest:
    """Metadata manifest describing a cached client split."""
    format_version: str
    client_name: str
    split_name: str
    cohort_machine_ids: Tuple[int, ...]
    config_hash: str
    source_manifest_hash: str
    code_version: str
    created_at: str
    counts: Dict[str, int]
    shapes: Dict[str, List[int]]
    dtypes: Dict[str, str]
    completed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["cohort_machine_ids"] = list(self.cohort_machine_ids)
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheManifest":
        return cls(
            format_version=str(data.get("format_version", "1.0.0")),
            client_name=str(data["client_name"]),
            split_name=str(data["split_name"]),
            cohort_machine_ids=tuple(data.get("cohort_machine_ids", ())),
            config_hash=str(data["config_hash"]),
            source_manifest_hash=str(data["source_manifest_hash"]),
            code_version=str(data["code_version"]),
            created_at=str(data["created_at"]),
            counts=dict(data.get("counts", {})),
            shapes=dict(data.get("shapes", {})),
            dtypes=dict(data.get("dtypes", {})),
            completed=bool(data.get("completed", False)),
        )


def compute_config_hash(config: Phase2BConfig) -> str:
    """Compute deterministic SHA-256 digest of frozen Phase2BConfig."""
    payload = {
        "pipeline_name": config.pipeline_name,
        "phase": config.phase,
        "track": config.track,
        "version": config.version,
        "history_hours": config.history_hours,
        "horizon_hours": config.horizon_hours,
        "cadence_hours": config.cadence_hours,
        "sensor_names": list(config.sensor_names),
        "participating_models": sorted(config.participating_models),
        "held_out_model": config.held_out_model,
        "zero_iqr_fallback_scale": config.zero_iqr_fallback_scale,
        "splits": {
            k: {
                "start": v.start.strftime("%Y-%m-%d %H:%M:%S"),
                "end": v.end.strftime("%Y-%m-%d %H:%M:%S"),
                "interval_type": v.interval_type,
            }
            for k, v in sorted(config.splits.items())
        },
        "clients": {
            k: {
                "model": v.get("model"),
                "role": v.get("role"),
                "allowed_splits": list(v.get("allowed_splits", ())),
            }
            for k, v in sorted(config.client_specs.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 digest of a local file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute hash for missing file: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_cache_dir(output_root: Union[str, Path], client_name: str, split_name: str) -> Path:
    """Derive canonical cache directory path for a client and split."""
    return Path(output_root) / str(client_name) / str(split_name)


def _resolve_source_manifest_hash(
    source_manifest_path: Optional[Union[str, Path]] = None,
) -> str:
    """Resolve SHA-256 of data manifest, or fallback to fixed synthetic hash when no manifest exists."""
    if source_manifest_path is not None:
        return compute_file_sha256(source_manifest_path)
    if DEFAULT_SOURCE_MANIFEST_PATH.is_file():
        return compute_file_sha256(DEFAULT_SOURCE_MANIFEST_PATH)
    # Staging environment without raw data or manifest
    return "staging_synthetic_manifest_untracked"


def _safe_clean_cache_dir(cache_dir: Path) -> None:
    """Clean up only known cache files in cache_dir without deleting arbitrary paths."""
    if not cache_dir.is_dir():
        return
    import gc
    gc.collect()
    for child in cache_dir.iterdir():
        if child.is_file() and (child.name in KNOWN_CACHE_FILES or child.name.endswith(".tmp")):
            try:
                child.unlink()
            except OSError:
                pass


def is_cache_valid(
    output_root: Union[str, Path],
    client_name: str,
    split_name: str,
    *,
    config: Optional[Phase2BConfig] = None,
    source_manifest_hash: Optional[str] = None,
    code_version: str = CODE_VERSION,
) -> bool:
    """Check whether a cached partition is complete, readable, and matches all hashes."""
    cache_dir = get_cache_dir(output_root, client_name, split_name)
    if not cache_dir.is_dir():
        return False

    marker_path = cache_dir / "completed.marker"
    manifest_path = cache_dir / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        return False

    required_files = (
        cache_dir / "sequences.npy",
        cache_dir / "summaries.npy",
        cache_dir / "tabular.npy",
        cache_dir / "labels.npy",
        cache_dir / "metadata_machine_ids.npy",
        cache_dir / "metadata_timestamps.npy",
    )
    if any(not f.is_file() for f in required_files):
        return False

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            m_dict = json.load(f)
        manifest = CacheManifest.from_dict(m_dict)
    except Exception:
        return False

    if not manifest.completed:
        return False
    if manifest.code_version != code_version:
        return False
    if manifest.client_name != client_name or manifest.split_name != split_name:
        return False

    cfg = config if config is not None else load_azure_preprocessing_config()
    if manifest.config_hash != compute_config_hash(cfg):
        return False

    expected_src_hash = (
        source_manifest_hash
        if source_manifest_hash is not None
        else _resolve_source_manifest_hash()
    )
    if manifest.source_manifest_hash != expected_src_hash:
        return False

    return True


def build_client_split_cache(
    output_root: Union[str, Path],
    client_name: str,
    split_name: str,
    *,
    pipeline_data: Optional[PipelineData] = None,
    config: Optional[Phase2BConfig] = None,
    data_dir: Optional[Union[str, Path]] = None,
    source_manifest_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
    expected_count: Optional[int] = None,
) -> Path:
    """Build memory-efficient numpy memmaps for an allowed client and split.

    Enforces:
    - Participating model2/3/4 allowed for train/val/test; heldout_model1 allowed for test only.
    - HoldoutLeakageError when attempting to build train/val cache for held-out client.
    - Exact count, finite values, and canonical shapes asserted against Phase 2B summary.
    - Cache manifest and completion marker written atomically upon success.
    - Rebuilds stale or incomplete caches cleanly without recursive deletion of arbitrary paths.
    """
    if not _HAS_NUMPY:
        raise ImportError("NumPy is required to build model cache.")

    cfg = config if config is not None else load_azure_preprocessing_config()

    # 1. Enforce holdout leakage guardrail
    if client_name == "heldout_model1" or client_name.startswith("heldout") or client_name == "model1":
        if split_name in ("train", "val"):
            raise HoldoutLeakageError(
                f"Held-out client '{client_name}' cannot be cached for '{split_name}'. "
                f"Held-out client data must never enter training or validation."
            )

    allowed_splits = ALLOWED_CLIENT_SPLITS.get(client_name)
    if allowed_splits is None or split_name not in allowed_splits:
        raise ValueError(
            f"Client '{client_name}' split '{split_name}' is not allowed. "
            f"Allowed partitions: {ALLOWED_CLIENT_SPLITS}"
        )

    cache_dir = get_cache_dir(output_root, client_name, split_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_hash = _resolve_source_manifest_hash(source_manifest_path)

    # Check if existing cache is valid
    if not force_rebuild and is_cache_valid(
        output_root,
        client_name,
        split_name,
        config=cfg,
        source_manifest_hash=src_hash,
    ):
        return cache_dir

    data = pipeline_data or load_pipeline_data(data_dir=data_dir, config=cfg)
    partitions = get_client_partitions(data.machines, config=cfg)
    if client_name not in partitions:
        raise ValueError(f"Unknown client '{client_name}'; available: {tuple(partitions.keys())}")

    partition = partitions[client_name]
    partition.validate_split_access(split_name)
    cohort_machine_ids = tuple(sorted(partition.machine_ids))

    # Clean stale/incomplete files safely
    _safe_clean_cache_dir(cache_dir)

    # Get expected accepted count from provided mapping or Phase 2B summary
    if expected_count is not None:
        expected_n = int(expected_count)
    else:
        summary = summarize_azure_pipeline(config=cfg, pipeline_data=data)
        client_summary = summary["clients"][client_name]
        expected_n = int(client_summary["splits"][split_name]["accepted_windows"])

    seq_path = cache_dir / "sequences.npy"
    sum_path = cache_dir / "summaries.npy"
    tab_path = cache_dir / "tabular.npy"
    lbl_path = cache_dir / "labels.npy"
    mid_path = cache_dir / "metadata_machine_ids.npy"
    ts_path = cache_dir / "metadata_timestamps.npy"

    # Allocate memmaps
    seq_mmap = open_memmap(seq_path, mode="w+", dtype=np.float32, shape=(expected_n, 48, 4))
    sum_mmap = open_memmap(sum_path, mode="w+", dtype=np.float32, shape=(expected_n, 8))
    tab_mmap = open_memmap(tab_path, mode="w+", dtype=np.float32, shape=(expected_n, 20))
    lbl_mmap = open_memmap(lbl_path, mode="w+", dtype=np.int64, shape=(expected_n,))
    mid_mmap = open_memmap(mid_path, mode="w+", dtype=np.int32, shape=(expected_n,))
    ts_mmap = open_memmap(ts_path, mode="w+", dtype=np.int64, shape=(expected_n,))

    actual_count = 0
    pos_count = 0
    neg_count = 0

    for sample in generate_azure_windows(
        pipeline_data=data,
        client_name=client_name,
        split_name=split_name,
        config=cfg,
    ):
        if actual_count >= expected_n:
            raise ValueError(
                f"Generator produced more samples than expected ({expected_n}) for {client_name}/{split_name}."
            )

        seq, summ, lbl, tab, _ = adapt_sample(sample, sensors=cfg.sensor_names)

        seq_mmap[actual_count] = seq
        sum_mmap[actual_count] = summ
        lbl_mmap[actual_count] = lbl
        tab_mmap[actual_count] = tab
        mid_mmap[actual_count] = sample.machine_id
        ts_mmap[actual_count] = int((sample.prediction_time - datetime(1970, 1, 1)).total_seconds())

        if lbl == 1:
            pos_count += 1
        else:
            neg_count += 1

        actual_count += 1

    # Assert exact count
    if actual_count != expected_n:
        raise ValueError(
            f"Exact sample count mismatch for {client_name}/{split_name}: "
            f"generator produced {actual_count}, summary reported {expected_n}."
        )

    # Assert finite values and binary labels
    if expected_n > 0:
        if not np.all(np.isfinite(seq_mmap)):
            raise ValueError(f"Nonfinite values detected in sequences for {client_name}/{split_name}.")
        if not np.all(np.isfinite(sum_mmap)):
            raise ValueError(f"Nonfinite values detected in summaries for {client_name}/{split_name}.")
        if not np.all(np.isfinite(tab_mmap)):
            raise ValueError(f"Nonfinite values detected in tabular for {client_name}/{split_name}.")
        if not np.all((lbl_mmap == 0) | (lbl_mmap == 1)):
            raise ValueError(f"Non-binary label detected in labels for {client_name}/{split_name}.")

    # Flush and close memmaps
    seq_mmap.flush()
    sum_mmap.flush()
    tab_mmap.flush()
    lbl_mmap.flush()
    mid_mmap.flush()
    ts_mmap.flush()
    del seq_mmap, sum_mmap, tab_mmap, lbl_mmap, mid_mmap, ts_mmap

    # Build manifest
    manifest = CacheManifest(
        format_version="1.0.0",
        client_name=client_name,
        split_name=split_name,
        cohort_machine_ids=cohort_machine_ids,
        config_hash=compute_config_hash(cfg),
        source_manifest_hash=src_hash,
        code_version=CODE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        counts={
            "total": actual_count,
            "positives": pos_count,
            "negatives": neg_count,
        },
        shapes={
            "sequences": [actual_count, 48, 4],
            "summaries": [actual_count, 8],
            "tabular": [actual_count, 20],
            "labels": [actual_count],
            "metadata_machine_ids": [actual_count],
            "metadata_timestamps": [actual_count],
        },
        dtypes={
            "sequences": "float32",
            "summaries": "float32",
            "tabular": "float32",
            "labels": "int64",
            "metadata_machine_ids": "int32",
            "metadata_timestamps": "int64",
        },
        completed=True,
    )

    # Write manifest atomically via .tmp
    tmp_manifest = cache_dir / "manifest.json.tmp"
    manifest_path = cache_dir / "manifest.json"
    with tmp_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    tmp_manifest.replace(manifest_path)

    # Write completion marker
    marker_path = cache_dir / "completed.marker"
    marker_path.write_text(f"COMPLETED {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")

    return cache_dir


def build_all_model_caches(
    output_root: Union[str, Path],
    *,
    pipeline_data: Optional[PipelineData] = None,
    config: Optional[Phase2BConfig] = None,
    data_dir: Optional[Union[str, Path]] = None,
    source_manifest_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
) -> Dict[Tuple[str, str], Path]:
    """Build model caches for all allowed clients and splits from ONE shared PipelineData.

    Computes the Phase 2B pipeline summary ONCE across all clients/splits when any cache
    needs rebuilding, skipping summary computation entirely when all caches are valid.
    Never fits on or caches held-out train or val partitions.
    """
    cfg = config if config is not None else load_azure_preprocessing_config()
    src_hash = _resolve_source_manifest_hash(source_manifest_path)

    all_partitions: List[Tuple[str, str]] = [
        (c_name, s_name)
        for c_name, splits in ALLOWED_CLIENT_SPLITS.items()
        for s_name in splits
    ]

    # Check which partitions actually need rebuilding
    needs_rebuild = []
    for c_name, s_name in all_partitions:
        if force_rebuild or not is_cache_valid(
            output_root,
            c_name,
            s_name,
            config=cfg,
            source_manifest_hash=src_hash,
        ):
            needs_rebuild.append((c_name, s_name))

    # If all existing caches are valid and no rebuild forced, return immediately with zero summary work
    if not needs_rebuild and not force_rebuild:
        return {
            (c_name, s_name): get_cache_dir(output_root, c_name, s_name)
            for c_name, s_name in all_partitions
        }

    # Data is needed to build missing or rebuilt caches
    shared_data = pipeline_data or load_pipeline_data(data_dir=data_dir, config=cfg)

    # Compute Phase 2B summary ONCE from shared PipelineData
    summary = summarize_azure_pipeline(config=cfg, pipeline_data=shared_data)
    expected_counts: Dict[Tuple[str, str], int] = {}
    for c_name in summary.get("clients", {}):
        for s_name, s_stats in summary["clients"][c_name].get("splits", {}).items():
            expected_counts[(c_name, s_name)] = int(s_stats["accepted_windows"])

    results: Dict[Tuple[str, str], Path] = {}
    for c_name, s_name in all_partitions:
        exp_n = expected_counts.get((c_name, s_name))
        c_path = build_client_split_cache(
            output_root=output_root,
            client_name=c_name,
            split_name=s_name,
            pipeline_data=shared_data,
            config=cfg,
            source_manifest_path=source_manifest_path,
            force_rebuild=force_rebuild,
            expected_count=exp_n,
        )
        results[(c_name, s_name)] = c_path

    return results


def load_cached_client_dataset(
    output_root: Union[str, Path],
    client_name: str,
    split_name: str,
    *,
    config: Optional[Phase2BConfig] = None,
    source_manifest_hash: Optional[str] = None,
    mmap_mode: str = "r",
    validate: bool = False,
) -> ClientDataset:
    """Load cached partition as ClientDataset with memmaps and lazy metadata Sequence.

    Args:
        output_root: Caller-specified cache root directory.
        client_name: Name of client partition (e.g. 'client_model2', 'heldout_model1').
        split_name: Name of split ('train', 'val', or 'test').
        config: Optional Phase2BConfig instance.
        source_manifest_hash: Optional source manifest hash for staleness validation.
        mmap_mode: NumPy memmap mode (default 'r' read-only).
        validate: If True, validate train/val partitions against leakage guardrails.
            Never calls train/val validation on test or held-out partitions.

    Returns:
        ClientDataset instance holding memmaps and LazyMetadataSequence.
    """
    if not _HAS_NUMPY:
        raise ImportError("NumPy is required to load cached ClientDataset.")

    # 1. Enforce holdout leakage guardrail
    if client_name == "heldout_model1" or client_name.startswith("heldout") or client_name == "model1":
        if split_name in ("train", "val"):
            raise HoldoutLeakageError(
                f"Held-out client '{client_name}' cannot be loaded for '{split_name}'. "
                f"Held-out client data must never enter training or validation."
            )

    allowed_splits = ALLOWED_CLIENT_SPLITS.get(client_name)
    if allowed_splits is None or split_name not in allowed_splits:
        raise ValueError(
            f"Client '{client_name}' split '{split_name}' is not allowed. "
            f"Allowed partitions: {ALLOWED_CLIENT_SPLITS}"
        )

    cache_dir = get_cache_dir(output_root, client_name, split_name)
    if not cache_dir.is_dir():
        raise CacheIncompleteError(f"Cache directory does not exist: {cache_dir}")

    marker_path = cache_dir / "completed.marker"
    manifest_path = cache_dir / "manifest.json"
    if not marker_path.is_file():
        raise CacheIncompleteError(f"Cache in {cache_dir} is incomplete: missing completed.marker")
    if not manifest_path.is_file():
        raise CacheIncompleteError(f"Cache in {cache_dir} is incomplete: missing manifest.json")

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = CacheManifest.from_dict(json.load(f))
    except Exception as exc:
        raise CacheIncompleteError(f"Corrupt manifest in {cache_dir}: {exc}") from exc

    if not manifest.completed:
        raise CacheIncompleteError(f"Manifest indicates incomplete cache in {cache_dir}")

    # Staleness checks
    if manifest.code_version != CODE_VERSION:
        raise CacheStaleError(
            f"Stale cache code_version: cached {manifest.code_version!r}, expected {CODE_VERSION!r}"
        )

    cfg = config if config is not None else load_azure_preprocessing_config()
    cfg_hash = compute_config_hash(cfg)
    if manifest.config_hash != cfg_hash:
        raise CacheStaleError(
            f"Stale cache config: config hash mismatch ({manifest.config_hash} != {cfg_hash})"
        )

    expected_src_hash = (
        source_manifest_hash
        if source_manifest_hash is not None
        else _resolve_source_manifest_hash()
    )
    if manifest.source_manifest_hash != expected_src_hash:
        raise CacheStaleError(
            f"Stale cache source data: source manifest hash mismatch "
            f"({manifest.source_manifest_hash} != {expected_src_hash})"
        )

    # Open memmaps
    seq_mmap = open_memmap(cache_dir / "sequences.npy", mode=mmap_mode)
    sum_mmap = open_memmap(cache_dir / "summaries.npy", mode=mmap_mode)
    tab_mmap = open_memmap(cache_dir / "tabular.npy", mode=mmap_mode)
    lbl_mmap = open_memmap(cache_dir / "labels.npy", mode=mmap_mode)
    mid_mmap = open_memmap(cache_dir / "metadata_machine_ids.npy", mode=mmap_mode)
    ts_mmap = open_memmap(cache_dir / "metadata_timestamps.npy", mode=mmap_mode)

    lazy_meta = LazyMetadataSequence(
        machine_ids=mid_mmap,
        timestamps=ts_mmap,
        client_name=client_name,
        split_name=split_name,
        history_hours=cfg.history_hours,
        horizon_hours=cfg.horizon_hours,
    )

    dataset = ClientDataset(
        client_name=client_name,
        split_name=split_name,
        labels=lbl_mmap,
        metadata=lazy_meta,
        sequences=seq_mmap,
        summaries=sum_mmap,
        tabular=tab_mmap,
        machine_ids=manifest.cohort_machine_ids,
    )

    # "For test/heldout, ClientDataset exists for scoring but never call train/val validation on it."
    if validate:
        if split_name != "test" and not (client_name == "heldout_model1" or client_name.startswith("heldout")):
            dataset.validate(expected_split=split_name)

    return dataset
