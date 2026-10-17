"""Reusable descriptive analysis of validated Azure PdM source records.

All statistics are exploratory counts and summaries, not future-failure labels.
The telemetry reader remains incremental; only exact-row keys and a bounded
plotting sample are retained alongside aggregate statistics.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .azure import DEFAULT_AZURE_DIR, iter_azure_rows

DataDir = Union[str, Path]
SENSORS = ("volt", "rotate", "pressure", "vibration")
COLUMNS = {
    "machines": ("machineID", "model", "age"),
    "telemetry": ("datetime", "machineID", *SENSORS),
    "errors": ("datetime", "machineID", "errorID"),
    "maint": ("datetime", "machineID", "comp"),
    "failures": ("datetime", "machineID", "failure"),
}
CATEGORY = {"errors": "errorID", "maint": "comp", "failures": "failure"}


def analyze_azure(data_dir: DataDir = DEFAULT_AZURE_DIR, *, sample_size: int = 10000, seed: int = 0) -> dict:
    """Profile the five source tables without materializing telemetry rows.

    Exact duplicate detection retains one key per distinct row. Plot samples
    use deterministic reservoir sampling; means, ranges and correlations use
    every telemetry record. Missing-field counts are zero for validated rows:
    the loader raises instead of silently admitting a missing required value.
    """
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")

    rng = random.Random(seed)
    tables: Dict[str, dict] = {}
    by_model: Counter = Counter()
    age_by_model: Dict[str, List[int]] = {}
    category_counts: Dict[str, Counter] = {name: Counter() for name in CATEGORY}
    monthly_counts: Dict[str, Counter] = {}
    sample: List[Tuple[float, float, float, float]] = []
    sums = [0.0] * len(SENSORS)
    products = [[0.0] * len(SENSORS) for _ in SENSORS]
    minima = [math.inf] * len(SENSORS)
    maxima = [-math.inf] * len(SENSORS)

    for table, columns in COLUMNS.items():
        seen = set()
        missing = {column: 0 for column in columns}
        duplicates = 0
        count = 0
        earliest: Optional[datetime] = None
        latest: Optional[datetime] = None
        months: Counter = Counter()

        for row in iter_azure_rows(table, data_dir):
            count += 1
            key = tuple(row[column] for column in columns)
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
            for column in columns:
                if row[column] is None or row[column] == "":
                    missing[column] += 1

            if table == "machines":
                model = str(row["model"])
                by_model[model] += 1
                age_by_model.setdefault(model, []).append(int(row["age"]))
                continue

            timestamp = row["datetime"]
            if earliest is None or timestamp < earliest:
                earliest = timestamp
            if latest is None or timestamp > latest:
                latest = timestamp
            months[timestamp.strftime("%Y-%m")] += 1

            if table == "telemetry":
                values = tuple(float(row[name]) for name in SENSORS)
                for index, value in enumerate(values):
                    sums[index] += value
                    minima[index] = min(minima[index], value)
                    maxima[index] = max(maxima[index], value)
                    for other, other_value in enumerate(values):
                        products[index][other] += value * other_value
                if len(sample) < sample_size:
                    sample.append(values)
                else:
                    selected = rng.randrange(count)
                    if selected < sample_size:
                        sample[selected] = values
            else:
                category_counts[table][str(row[CATEGORY[table]])] += 1

        tables[table] = {
            "rows": count,
            "columns": list(columns),
            "missing_values": missing,
            "exact_duplicate_rows": duplicates,
            "first_timestamp": earliest,
            "last_timestamp": latest,
        }
        if table != "machines":
            monthly_counts[table] = months

    telemetry_count = tables["telemetry"]["rows"]
    sensor_stats: Dict[str, dict] = {}
    correlation: Dict[str, Dict[str, Optional[float]]] = {}
    deviations: List[float] = []
    for index, name in enumerate(SENSORS):
        if telemetry_count:
            mean = sums[index] / telemetry_count
            variance = max(0.0, products[index][index] / telemetry_count - mean * mean)
            deviation = math.sqrt(variance)
            sensor_stats[name] = {
                "count": telemetry_count,
                "mean": mean,
                "std_population": deviation,
                "min": minima[index],
                "max": maxima[index],
            }
        else:
            mean = deviation = None
            sensor_stats[name] = {
                "count": 0, "mean": None, "std_population": None, "min": None, "max": None
            }
        deviations.append(deviation if deviation is not None else 0.0)

    for index, name in enumerate(SENSORS):
        correlation[name] = {}
        for other, other_name in enumerate(SENSORS):
            if telemetry_count and deviations[index] and deviations[other]:
                covariance = (
                    products[index][other] / telemetry_count
                    - (sums[index] / telemetry_count) * (sums[other] / telemetry_count)
                )
                correlation[name][other_name] = max(-1.0, min(1.0, covariance / (deviations[index] * deviations[other])))
            else:
                correlation[name][other_name] = None

    return {
        "tables": tables,
        "machines_by_model": dict(sorted(by_model.items())),
        "ages_by_model": age_by_model,
        "sensor_stats": sensor_stats,
        "sensor_correlation": correlation,
        "telemetry_plot_sample": {
            name: [values[index] for values in sample] for index, name in enumerate(SENSORS)
        },
        "event_counts": {table: dict(sorted(counts.items())) for table, counts in category_counts.items()},
        "monthly_counts": {
            table: dict(sorted(counts.items())) for table, counts in monthly_counts.items()
        },
    }
