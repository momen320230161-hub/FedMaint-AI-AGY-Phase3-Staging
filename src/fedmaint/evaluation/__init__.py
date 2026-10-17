"""Phase 3 evaluation: threshold selection, window/event metrics, and machine-cluster bootstrap."""
from __future__ import annotations

from .bootstrap import (
    BootstrapCI,
    bootstrap_event_ci,
    bootstrap_window_ci,
)
from .events import (
    EventMatch,
    EventMetrics,
    evaluate_event_metrics,
)
from .threshold import (
    HELDOUT_CLIENTS,
    PARTICIPATING_CLIENTS,
    ThresholdResult,
    select_client_thresholds,
    select_threshold,
    validate_threshold_metadata,
)
from .window import (
    WindowMetrics,
    WindowReport,
    evaluate_window_metrics,
    evaluate_window_report,
)

__all__ = [
    # Threshold selection
    "ThresholdResult",
    "select_threshold",
    "select_client_thresholds",
    "validate_threshold_metadata",
    "PARTICIPATING_CLIENTS",
    "HELDOUT_CLIENTS",
    # Window-level metrics
    "WindowMetrics",
    "WindowReport",
    "evaluate_window_metrics",
    "evaluate_window_report",
    # Event-level metrics
    "EventMatch",
    "EventMetrics",
    "evaluate_event_metrics",
    # Machine-cluster bootstrap
    "BootstrapCI",
    "bootstrap_window_ci",
    "bootstrap_event_ci",
]
