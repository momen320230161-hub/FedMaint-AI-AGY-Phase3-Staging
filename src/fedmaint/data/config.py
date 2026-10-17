"""Configuration loading and dataclasses for Phase 2B Azure PdM pipeline.

This module exposes the frozen Phase 2B parameters as validated dataclasses,
loading from configs/azure_preprocessing.json as the single source of truth.
All approved Phase 2B protocol values are strictly validated and immutable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "azure_preprocessing.json"

FROZEN_SENSORS: Tuple[str, ...] = ("volt", "rotate", "pressure", "vibration")
FROZEN_PARTICIPATING_MODELS: Tuple[str, ...] = ("model2", "model3", "model4")
FROZEN_HELDOUT_MODEL: str = "model1"

FROZEN_SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test")
FROZEN_SPLIT_DATES: Dict[str, Tuple[datetime, datetime]] = {
    "train": (datetime(2015, 1, 1, 6, 0, 0), datetime(2015, 9, 1, 6, 0, 0)),
    "val": (datetime(2015, 9, 1, 6, 0, 0), datetime(2015, 11, 1, 6, 0, 0)),
    "test": (datetime(2015, 11, 1, 6, 0, 0), datetime(2016, 1, 1, 6, 0, 0)),
}

FROZEN_CLIENT_SPECS: Dict[str, Dict[str, object]] = {
    "client_model2": {
        "model": "model2",
        "role": "participating_client",
        "allowed_splits": ("train", "val", "test"),
    },
    "client_model3": {
        "model": "model3",
        "role": "participating_client",
        "allowed_splits": ("train", "val", "test"),
    },
    "client_model4": {
        "model": "model4",
        "role": "participating_client",
        "allowed_splits": ("train", "val", "test"),
    },
    "heldout_model1": {
        "model": "model1",
        "role": "held_out_evaluation",
        "allowed_splits": ("test",),
    },
}


@dataclass(frozen=True)
class SplitDates:
    """Temporal boundary for a dataset split."""
    start: datetime
    end: datetime
    interval_type: str = "half_open_left_closed_right_open"

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(f"Split start {self.start} must be strictly before end {self.end}")
        if self.interval_type != "half_open_left_closed_right_open":
            raise ValueError(
                f"Approved interval_type is 'half_open_left_closed_right_open'; got '{self.interval_type}'"
            )

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls in [start, end)."""
        return self.start <= timestamp < self.end


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalization settings for sensor windows."""
    method: str = "per_window_median_iqr"
    quartile_interpolation: str = "linear"
    zero_iqr_fallback_scale: float = 1.0
    retain_raw_summaries: bool = True

    def __post_init__(self) -> None:
        if self.method != "per_window_median_iqr":
            raise ValueError(f"Approved normalization method is 'per_window_median_iqr'; got '{self.method}'")
        if self.quartile_interpolation != "linear":
            raise ValueError(f"Approved quartile_interpolation is 'linear'; got '{self.quartile_interpolation}'")
        if self.zero_iqr_fallback_scale != 1.0:
            raise ValueError(
                f"Approved zero_iqr_fallback_scale is 1.0; got {self.zero_iqr_fallback_scale}"
            )
        if not self.retain_raw_summaries:
            raise ValueError("Approved protocol requires retain_raw_summaries=True")


@dataclass(frozen=True)
class Phase2BConfig:
    """Validated Phase 2B data preparation configuration."""
    pipeline_name: str
    phase: str
    track: str
    version: str
    splits: Dict[str, SplitDates]
    history_hours: int = 48
    horizon_hours: int = 24
    cadence_hours: int = 1
    sensor_names: Tuple[str, ...] = FROZEN_SENSORS
    primary_features_only: bool = True
    include_events_in_primary: bool = False
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    participating_models: Tuple[str, ...] = FROZEN_PARTICIPATING_MODELS
    held_out_model: str = FROZEN_HELDOUT_MODEL
    zero_iqr_fallback_scale: float = 1.0
    client_specs: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Strict validation of frozen approved values
        if self.history_hours != 48:
            raise ValueError(f"Approved protocol requires history_hours=48; got {self.history_hours}")
        if self.horizon_hours != 24:
            raise ValueError(f"Approved protocol requires horizon_hours=24; got {self.horizon_hours}")
        if self.cadence_hours != 1:
            raise ValueError(f"Approved protocol requires cadence_hours=1; got {self.cadence_hours}")
        if self.sensor_names != FROZEN_SENSORS:
            raise ValueError(
                f"Approved sensor order is {FROZEN_SENSORS}; got {self.sensor_names}"
            )
        if not self.primary_features_only:
            raise ValueError("Approved protocol requires primary_features_only=True")
        if self.include_events_in_primary:
            raise ValueError("Approved protocol requires include_events_in_primary=False")
        if set(self.participating_models) != set(FROZEN_PARTICIPATING_MODELS):
            raise ValueError(
                f"Approved participating models are {FROZEN_PARTICIPATING_MODELS}; got {self.participating_models}"
            )
        if self.held_out_model != FROZEN_HELDOUT_MODEL:
            raise ValueError(f"Approved held_out_model is '{FROZEN_HELDOUT_MODEL}'; got '{self.held_out_model}'")

        # Validate exact frozen split set (no missing, no extra)
        if set(self.splits.keys()) != set(FROZEN_SPLIT_NAMES):
            raise ValueError(
                f"Approved splits must be exactly {FROZEN_SPLIT_NAMES}; got {tuple(self.splits.keys())}"
            )

        # Validate exact frozen split dates and non-overlapping half-open intervals
        for s_name, (expected_start, expected_end) in FROZEN_SPLIT_DATES.items():
            split = self.splits[s_name]
            if split.start != expected_start or split.end != expected_end:
                raise ValueError(
                    f"Split '{s_name}' must be [{expected_start}, {expected_end}); got [{split.start}, {split.end})"
                )

        train = self.splits["train"]
        val = self.splits["val"]
        test = self.splits["test"]

        if train.end > val.start:
            raise ValueError(f"Train split end {train.end} overlaps val split start {val.start}")
        if val.end > test.start:
            raise ValueError(f"Val split end {val.end} overlaps test split start {test.start}")

        # Validate exact approved client specifications set
        if set(self.client_specs.keys()) != set(FROZEN_CLIENT_SPECS.keys()):
            raise ValueError(
                f"Approved client_specs must be exactly {tuple(FROZEN_CLIENT_SPECS.keys())}; got {tuple(self.client_specs.keys())}"
            )

        for c_name, expected in FROZEN_CLIENT_SPECS.items():
            actual = self.client_specs[c_name]
            if actual.get("model") != expected["model"]:
                raise ValueError(
                    f"Client '{c_name}' must specify model '{expected['model']}'; got '{actual.get('model')}'"
                )
            if actual.get("role") != expected["role"]:
                raise ValueError(
                    f"Client '{c_name}' must specify role '{expected['role']}'; got '{actual.get('role')}'"
                )
            actual_allowed = tuple(actual.get("allowed_splits", ()))
            expected_allowed = expected["allowed_splits"]
            if actual_allowed != expected_allowed:
                raise ValueError(
                    f"Client '{c_name}' must permit splits {expected_allowed}; got {actual_allowed}"
                )

    @classmethod
    def from_dict(cls, data: dict) -> Phase2BConfig:
        """Construct a validated Phase2BConfig from a raw dictionary."""
        splits_raw = data.get("splits", {})
        splits: Dict[str, SplitDates] = {}
        for name, spec in splits_raw.items():
            start = datetime.strptime(spec["start"], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(spec["end"], "%Y-%m-%d %H:%M:%S")
            splits[name] = SplitDates(
                start=start,
                end=end,
                interval_type=spec.get("interval_type", "half_open_left_closed_right_open"),
            )

        windowing = data.get("windowing", {})
        features = data.get("features", {})
        norm_raw = features.get("normalization", {})
        normalization = NormalizationConfig(
            method=norm_raw.get("method", "per_window_median_iqr"),
            quartile_interpolation=norm_raw.get("quartile_interpolation", "linear"),
            zero_iqr_fallback_scale=float(norm_raw.get("zero_iqr_fallback_scale", 1.0)),
            retain_raw_summaries=bool(norm_raw.get("retain_raw_summaries", True)),
        )

        clients = data.get("clients", {})
        participating = tuple(spec["model"] for spec in clients.get("participating", {}).values())
        held_out_specs = list(clients.get("held_out", {}).values())
        held_out = held_out_specs[0]["model"] if held_out_specs else FROZEN_HELDOUT_MODEL

        client_specs = {}
        client_specs.update(clients.get("participating", {}))
        client_specs.update(clients.get("held_out", {}))

        return cls(
            pipeline_name=data.get("pipeline_name", "azure_leakage_safe_data_preparation"),
            phase=data.get("phase", "2B"),
            track=data.get("track", "azure"),
            version=data.get("version", "1.0.0"),
            splits=splits,
            history_hours=int(windowing.get("history_hours", 48)),
            horizon_hours=int(windowing.get("horizon_hours", 24)),
            cadence_hours=int(windowing.get("cadence_hours", 1)),
            sensor_names=tuple(features.get("sensor_names", FROZEN_SENSORS)),
            primary_features_only=bool(features.get("primary_features_only", True)),
            include_events_in_primary=bool(features.get("include_events_in_primary", False)),
            normalization=normalization,
            participating_models=participating or FROZEN_PARTICIPATING_MODELS,
            held_out_model=held_out,
            zero_iqr_fallback_scale=normalization.zero_iqr_fallback_scale,
            client_specs=client_specs,
        )


def load_azure_preprocessing_config(path: Optional[Union[str, Path]] = None) -> Phase2BConfig:
    """Load and parse the Phase 2B frozen configuration from JSON.

    If path is None, DEFAULT_CONFIG_PATH (configs/azure_preprocessing.json) is used.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Phase 2B configuration not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Phase2BConfig.from_dict(data)
