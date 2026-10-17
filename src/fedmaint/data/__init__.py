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
]
