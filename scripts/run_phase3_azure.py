#!/usr/bin/env python3
"""CLI for Phase 3 Azure PdM baseline experiment orchestration.

Usage:
    python scripts/run_phase3_azure.py --mode smoke
    python scripts/run_phase3_azure.py --mode full
    python scripts/run_phase3_azure.py --mode smoke --models gru --regimes central --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project src to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.experiments.phase3_azure import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    Phase3AzureConfig,
    load_phase3_azure_config,
    run_phase3_experiment,
)


def format_summary_table(rows: list[dict]) -> str:
    """Format tabular summary rows into a clean readable string table."""
    if not rows:
        return "No results."
    lines = []
    lines.append("-" * 130)
    lines.append(
        f"{'Model':<10} {'Regime':<9} {'Client':<22} {'Seed':<5} {'Thresh':<7} {'Val F1':<7} "
        f"{'Test AP':<8} {'Test F1':<8} {'Test Rec':<9} {'Test FAR':<9} "
        f"{'Hold AP':<8} {'Hold F1':<8} {'Hold Rec':<9} {'Run(s)':<8} {'Fit(s)':<8}"
    )
    lines.append("-" * 130)
    for r in rows:
        t_ap = f"{r['test_ap']:.4f}" if r.get("test_ap") is not None else "N/A"
        t_rec = f"{r['test_event_recall']:.4f}" if r.get("test_event_recall") is not None else "N/A"
        t_far = f"{r['test_far_per_100_days']:.1f}" if r.get("test_far_per_100_days") is not None else "N/A"
        h_ap = f"{r['heldout_ap']:.4f}" if r.get("heldout_ap") is not None else "N/A"
        h_rec = f"{r['heldout_event_recall']:.4f}" if r.get("heldout_event_recall") is not None else "N/A"
        run_s = f"{r['runtime_s']:.1f}" if r.get("runtime_s") is not None else "N/A"
        fit_s = f"{r['fit_time_s']:.1f}" if r.get("fit_time_s") is not None else "N/A"

        lines.append(
            f"{r['model']:<10} {r['regime']:<9} {r['client']:<22} {r['seed']:<5} {r['threshold']:<7.4f} {r['val_f1']:<7.4f} "
            f"{t_ap:<8} {r['test_f1']:<8.4f} {t_rec:<9} {t_far:<9} "
            f"{h_ap:<8} {r['heldout_f1']:<8.4f} {h_rec:<9} {run_s:<8} {fit_s:<8}"
        )
    lines.append("-" * 130)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 Azure baseline experiments CLI for local and centralized GRU, Logistic, and MLP."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Experiment mode: 'smoke' (1 seed, 2 epochs, bounded subset, plumbing only) or 'full' (all approved data and seeds).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to Phase 3 Azure configuration JSON (default: configs/phase3_azure.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output JSON and summary (default: outputs/phase3).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Root directory for memmapped model cache (default: <output-dir>/cache).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to raw Azure CSV files directory (default: 'data/Azure Data' if present).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device: 'cpu', 'cuda', or specific device name.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override random seeds (default: from config mode).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=("gru", "logistic", "mlp", "all"),
        default=["all"],
        help="Models to execute (default: all).",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        nargs="+",
        choices=("local", "central", "all"),
        default=["all"],
        help="Training regimes to execute (default: all).",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Force rebuild of all memmapped caches.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FedMaint-AI Phase 3 Azure Experiment Orchestration")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == "smoke":
        print("  NOTICE: Running in SMOKE mode: plumbing verification only; NOT representative.")
    print(f"Config: {args.config}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    try:
        report = run_phase3_experiment(
            config=args.config,
            mode=args.mode,
            models=args.models,
            regimes=args.regimes,
            seeds=args.seeds,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            force_rebuild_cache=args.force_rebuild_cache,
        )
    except Exception as exc:
        print(f"\nERROR: Experiment execution failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    formatted_table = format_summary_table(report.summary_table)
    print("\nSummary Results:")
    print(formatted_table)

    results_path = Path(report.output_dir) / "results.json"
    summary_path = Path(report.output_dir) / "summary.txt"

    summary_path.write_text(
        f"FedMaint-AI Phase 3 Azure PdM Experiment Summary\n"
        f"Mode: {args.mode} (representative={report.is_representative})\n"
        f"Config Hash: {report.config_hash}\n"
        f"Git Revision: {report.git_revision}\n"
        f"Created At: {report.created_at}\n\n"
        f"{formatted_table}\n",
        encoding="utf-8",
    )

    print(f"\nResults successfully written to:")
    print(f"  JSON:    {results_path}")
    print(f"  Summary: {summary_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
