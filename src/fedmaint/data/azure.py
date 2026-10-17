"""Validated, streaming readers for the committed Azure PdM CSV tables.

This module loads source records only. It does not create prediction features or labels.
"""
from __future__ import annotations

import csv
import math
import re
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

DataDir = Union[str, Path]
Record = Dict[str, object]

DEFAULT_AZURE_DIR = Path(__file__).resolve().parents[3] / "data" / "Azure Data"

_SCHEMAS = {
    "telemetry": ("PdM_telemetry.csv", ("datetime", "machineID", "volt", "rotate", "pressure", "vibration")),
    "errors": ("PdM_errors.csv", ("datetime", "machineID", "errorID")),
    "maint": ("PdM_maint.csv", ("datetime", "machineID", "comp")),
    "failures": ("PdM_failures.csv", ("datetime", "machineID", "failure")),
    "machines": ("PdM_machines.csv", ("machineID", "model", "age")),
}
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\Z")
_INTEGER = re.compile(r"[0-9]+\Z")


class AzureDataError(ValueError):
    """A source CSV has an invalid schema, record, or machine reference."""


def _error(filename: str, row_number: int, message: str) -> AzureDataError:
    return AzureDataError(f"{filename}: CSV row {row_number}: {message}")


def _raw_rows(path: Path, columns: Tuple[str, ...]) -> Iterator[Tuple[int, Dict[str, str]]]:
    filename = path.name
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise AzureDataError(f"{filename}: cannot open {path}: {exc}") from exc
    with handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise _error(filename, 1, "missing header") from exc
        except (csv.Error, UnicodeError) as exc:
            raise _error(filename, 1, f"invalid header: {exc}") from exc
        if len(header) != len(columns) or set(header) != set(columns):
            raise _error(filename, 1, f"expected columns {columns}; found {tuple(header)}")
        row_number = 1
        while True:
            try:
                values = next(reader)
            except StopIteration:
                break
            except (csv.Error, UnicodeError) as exc:
                raise _error(filename, row_number + 1, f"invalid CSV: {exc}") from exc
            row_number += 1
            if len(values) != len(header):
                raise _error(filename, row_number, f"expected {len(header)} fields; found {len(values)}")
            yield row_number, dict(zip(header, values))


def _integer(value: str, filename: str, row_number: int, field: str, *, positive: bool) -> int:
    cleaned = value.strip()
    if not _INTEGER.fullmatch(cleaned):
        raise _error(filename, row_number, f"{field} must be an integer")
    number = int(cleaned)
    if number < (1 if positive else 0):
        rule = "positive" if positive else "nonnegative"
        raise _error(filename, row_number, f"{field} must be {rule}")
    return number


def _timestamp(value: str, filename: str, row_number: int) -> datetime:
    cleaned = value.strip()
    if not _TIMESTAMP.fullmatch(cleaned):
        raise _error(filename, row_number, "datetime must use YYYY-MM-DD HH:MM:SS")
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise _error(filename, row_number, f"invalid datetime {cleaned!r}") from exc


def _float(value: str, filename: str, row_number: int, field: str) -> float:
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise _error(filename, row_number, f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise _error(filename, row_number, f"{field} must be a finite number")
    return number


def _text(value: str, filename: str, row_number: int, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise _error(filename, row_number, f"{field} must not be empty")
    return cleaned


def iter_azure_rows(table: str, data_dir: DataDir = DEFAULT_AZURE_DIR) -> Iterator[Record]:
    """Yield typed rows from one Azure table, checking schemas and machine references.

    Supported table names: telemetry, errors, maint, failures, machines. Validation
    occurs as rows are consumed. Only the machines lookup and current row are kept
    in memory for non-machine tables.
    """
    if table not in _SCHEMAS:
        raise ValueError(f"unknown Azure table {table!r}; choose from {tuple(_SCHEMAS)}")
    filename, columns = _SCHEMAS[table]
    directory = Path(data_dir)
    machine_ids: Optional[set] = None
    if table != "machines":
        machine_ids = set(load_machines(directory))
    seen_ids = set()
    with closing(_raw_rows(directory / filename, columns)) as rows:
        for row_number, raw in rows:
            machine_id = _integer(raw["machineID"], filename, row_number, "machineID", positive=True)
            if table == "machines":
                if machine_id in seen_ids:
                    raise _error(filename, row_number, f"duplicate machineID {machine_id}")
                seen_ids.add(machine_id)
                yield {
                    "machineID": machine_id,
                    "model": _text(raw["model"], filename, row_number, "model"),
                    "age": _integer(raw["age"], filename, row_number, "age", positive=False),
                }
                continue
            if machine_id not in machine_ids:
                raise _error(filename, row_number, f"machineID {machine_id} is absent from PdM_machines.csv")
            timestamp = _timestamp(raw["datetime"], filename, row_number)
            record: Record = {"datetime": timestamp, "machineID": machine_id}
            if table == "telemetry":
                for field in ("volt", "rotate", "pressure", "vibration"):
                    record[field] = _float(raw[field], filename, row_number, field)
            else:
                field = {"errors": "errorID", "maint": "comp", "failures": "failure"}[table]
                record[field] = _text(raw[field], filename, row_number, field)
            yield record


def load_machines(data_dir: DataDir = DEFAULT_AZURE_DIR) -> Dict[int, Record]:
    """Load the small machine lookup, rejecting duplicate machine IDs."""
    return {int(row["machineID"]): row for row in iter_azure_rows("machines", data_dir)}


def validate_azure_files(data_dir: DataDir = DEFAULT_AZURE_DIR) -> Dict[str, int]:
    """Validate all five source tables without materializing telemetry records."""
    directory = Path(data_dir)
    counts = {"machines": len(load_machines(directory))}
    for table in ("telemetry", "errors", "maint", "failures"):
        counts[table] = sum(1 for _ in iter_azure_rows(table, directory))
    return counts
