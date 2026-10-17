#!/usr/bin/env python3
"""Verify uploaded datasets in-place or extract and verify local source archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
LOCATIONS = {
    "azure": [ROOT / "data" / "Azure Data", ROOT / "data" / "raw" / "azure"],
    "nasa": [ROOT / "data" / "CMAPSSData", ROOT / "data" / "raw" / "nasa"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path, dataset: str) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if dataset == "azure":
            reader = csv.reader(handle)
            next(reader, None)  # Header, not a record.
            return sum(1 for row in reader if row)
        return sum(1 for line in handle if line.strip())


def validate(directory: Path, expected: dict, dataset: str) -> bool:
    success = True
    print(f"\nChecking {dataset.upper()} at {directory}")
    for name, metadata in expected.items():
        path = directory / name
        if not path.is_file():
            print(f"  MISSING {name}", file=sys.stderr)
            success = False
            continue
        records = row_count(path, dataset)
        digest = sha256(path)
        # A tracked Azure CSV may use LF instead of the source ZIP's CRLF.
        accepted = {metadata["sha256"]}
        if "repository_sha256" in metadata:
            accepted.add(metadata["repository_sha256"])
        ok = records == metadata["rows"] and digest in accepted
        status = "OK" if ok else "MISMATCH"
        print(f"  {status} {name}: records={records}")
        if not ok:
            print(f"    Actual SHA-256: {digest}", file=sys.stderr)
            print(f"    Expected records: {metadata['rows']}", file=sys.stderr)
            success = False
    return success


def extract(archive: Path, destination: Path, expected: dict, expected_archive_hash: str) -> None:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if sha256(archive) != expected_archive_hash:
        raise ValueError(
            f"{archive}: ZIP SHA-256 differs from the original supplied archive. "
            "Use the files already in this repository, or check archive provenance."
        )
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        members = {Path(item.filename).name: item for item in zf.infolist() if not item.is_dir()}
        missing = set(expected) - members.keys()
        if missing:
            raise ValueError(f"Missing expected files in {archive}: {', '.join(sorted(missing))}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in expected:
            with zf.open(members[name]) as source, (destination / name).open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
            print(f"Extracted: {destination / name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without extraction")
    parser.add_argument("--dataset", choices=("all", "azure", "nasa"), default="all", help="Select dataset to check")
    parser.add_argument("--azure-zip", type=Path, help="Optional original Azure source ZIP")
    parser.add_argument("--nasa-zip", type=Path, help="Optional original NASA source ZIP")
    args = parser.parse_args()
    if args.verify_only and (args.azure_zip or args.nasa_zip):
        parser.error("--verify-only cannot be combined with ZIP extraction arguments")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archives = {"azure": args.azure_zip, "nasa": args.nasa_zip}
    selected = ("azure", "nasa") if args.dataset == "all" else (args.dataset,)
    if args.dataset != "all" and any(archives[name] for name in archives if name not in selected):
        parser.error("A ZIP option does not match --dataset")
    if not args.verify_only and any(archives.values()):
        selected = tuple(name for name in selected if archives[name] is not None)
    success = True
    for name in selected:
        archive = archives[name]
        if archive:
            folder = ROOT / "data" / "raw" / name
            extract(archive, folder, manifest[name]["files"], manifest[name]["archive_sha256"])
        else:
            folder = next((location for location in LOCATIONS[name] if location.is_dir()), None)
            if folder is None:
                print(f"MISSING {name}: expected one of: {', '.join(str(p) for p in LOCATIONS[name])}", file=sys.stderr)
                success = False
                continue
        success = validate(folder, manifest[name]["files"], name) and success
    print("\nPASS: all selected datasets match the manifest." if success else "\nFAIL: dataset verification failed.")
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
