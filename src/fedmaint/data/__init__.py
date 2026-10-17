"""FedMaint-AI data loading, validation, preprocessing, and window generation."""
from __future__ import annotations

# Phase 2A exports: validated loader and EDA
from .azure import (
    DEFAULT_AZURE_DIR,
    AzureDataError,
    iter_azure_rows,
    load_machines,
    validate_azure_files,
)
from .eda import analyze_azure

# Phase 2B exports: leakage-safe data preparation pipeline
from .config import (
    NormalizationConfig,
    Phase2BConfig,
    SplitDates,
    load_azure_preprocessing_config,
)
from .episodes import (
    FailureEpisode,
    collapse_failure_episodes,
    get_episodes_in_blackout,
    get_episodes_in_horizon,
    is_in_blackout,
    label_window,
)
from .normalization import (
    NonfiniteValueError,
    SensorSummary,
    compute_quantile,
    compute_sensor_stats,
    normalize_window,
)
from .splits import (
    ClientPartition,
    HoldoutLeakageError,
    SplitBounds,
    get_client_partitions,
    get_default_splits,
)
from .pipeline import (
    AzureSample,
    PipelineData,
    PipelineRejectionReason,
    generate_azure_windows,
    load_pipeline_data,
    summarize_azure_pipeline,
)
from .adapter import (
    DEFAULT_SENSORS,
    FLOAT32_MAX,
    FLOAT32_MIN,
    SUMMARY_FEATURE_NAMES,
    TABULAR_FEATURE_NAMES,
    AzureModelReadyAdapter,
    ModelReadyBatch,
    SampleMetadata,
    adapt_sample,
    extract_summary_features,
    extract_tabular_features,
    iter_batches,
    reconstruct_raw_series,
    signed_log1p,
)
from .model_cache import (
    ALLOWED_CLIENT_SPLITS,
    CODE_VERSION,
    CacheError,
    CacheIncompleteError,
    CacheManifest,
    CacheStaleError,
    LazyMetadataSequence,
    build_all_model_caches,
    build_client_split_cache,
    compute_config_hash,
    get_cache_dir,
    is_cache_valid,
    load_cached_client_dataset,
)

__all__ = [
    # Phase 2A
    "DEFAULT_AZURE_DIR",
    "AzureDataError",
    "iter_azure_rows",
    "load_machines",
    "validate_azure_files",
    "analyze_azure",
    # Phase 2B config
    "NormalizationConfig",
    "Phase2BConfig",
    "SplitDates",
    "load_azure_preprocessing_config",
    # Phase 2B episodes
    "FailureEpisode",
    "collapse_failure_episodes",
    "get_episodes_in_blackout",
    "get_episodes_in_horizon",
    "is_in_blackout",
    "label_window",
    # Phase 2B normalization
    "NonfiniteValueError",
    "SensorSummary",
    "compute_quantile",
    "compute_sensor_stats",
    "normalize_window",
    # Phase 2B splits
    "ClientPartition",
    "HoldoutLeakageError",
    "SplitBounds",
    "get_client_partitions",
    "get_default_splits",
    # Phase 2B pipeline
    "AzureSample",
    "PipelineData",
    "PipelineRejectionReason",
    "generate_azure_windows",
    "load_pipeline_data",
    "summarize_azure_pipeline",
    # Phase 3 adapter
    "DEFAULT_SENSORS",
    "FLOAT32_MAX",
    "FLOAT32_MIN",
    "SUMMARY_FEATURE_NAMES",
    "TABULAR_FEATURE_NAMES",
    "AzureModelReadyAdapter",
    "ModelReadyBatch",
    "SampleMetadata",
    "adapt_sample",
    "extract_summary_features",
    "extract_tabular_features",
    "iter_batches",
    "reconstruct_raw_series",
    "signed_log1p",
    # Phase 3 model-ready cache bridge
    "ALLOWED_CLIENT_SPLITS",
    "CODE_VERSION",
    "CacheError",
    "CacheIncompleteError",
    "CacheManifest",
    "CacheStaleError",
    "LazyMetadataSequence",
    "build_all_model_caches",
    "build_client_split_cache",
    "compute_config_hash",
    "get_cache_dir",
    "is_cache_valid",
    "load_cached_client_dataset",
]
