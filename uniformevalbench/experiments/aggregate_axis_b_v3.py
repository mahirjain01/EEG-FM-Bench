#!/usr/bin/env python3
"""
Aggregate Axis B v3 results and write the summary JSON + print ranked table.

Primary metric: test_metrics_bestval (best-val epoch on clean validation split).
CDR is reported as a diagnostic column; it is NOT used in the composite ranking.

Usage
-----
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/aggregate_axis_b_v3.py

Output
------
    results/axis_b_v3/axis_b_v3_summary.json
    Prints ranked table to stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from uniformevalbench.metrics.chance_corrected import chance_correct, DATASET_N_CLASSES

RESULTS_DIR = _REPO / "uniformevalbench/experiments/results/axis_b_v3"
PILOT_DIR   = _REPO / "uniformevalbench/experiments/results/pilot/per_run"
PER_RUN_DIR = RESULTS_DIR / "per_run"
SUMMARY_OUT = RESULTS_DIR / "axis_b_v3_summary.json"

MODELS    = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve"]
DATASETS  = ["bcic_2a", "hmc", "adftd", "motor_mv_img",
             "siena_scalp", "workload", "epilepsy_mimickers"]
SEEDS     = [42, 123, 7]
DROPOUT_P = [0.1, 0.25, 0.4]


def _tag(p):
    return f"p{int(round(p * 100)):03d}"


def _load(path):
    try:
        return json.loads(Path(path).read_text()) if Path(path).exists() else None
    except Exception:
        return None


def _clean_ba(model, dataset, seed):
    """Best-val balanced accuracy from the clean Axis A FBP run."""
    d = _load(PILOT_DIR / f"lp_{model}_{dataset}_s{seed}.json")
    if d and d.get("status") == "complete":
        bv = d.get("test_metrics_bestval") or d.get("test_metrics")
        if bv:
            return bv.get("balanced_acc")
    return None


def _corrupt_ba(model, dataset, seed, p):
    """Best-val balanced accuracy from the v3 corrupted run."""
    d = _load(PER_RUN_DIR / f"b3_{model}_{dataset}_s{seed}_{_tag(p)}.json")
    if d and d.get("status") == "complete":
        bv = d.get("test_metrics_bestval")
        if bv:
            return bv.get("balanced_acc")
    return None


def main():
    summary = {}

    for model in MODELS:
        b_cells, cdr_cells = [], []
        per_dataset = {}

        for dataset in DATASETS:
            n_cls = DATASET_N_CLASSES.get(dataset)
            if n_cls is None:
                continue

            clean_vals  = [_clean_ba(model, dataset, s)   for s in SEEDS]
            clean_valid = [v for v in clean_vals if v is not None]
            ba_clean    = float(np.mean(clean_valid)) if clean_valid else None

            dataset_b, dataset_cdr = [], []
            degradation = {"clean": ba_clean}

            for p in DROPOUT_P:
                corrupt_vals  = [_corrupt_ba(model, dataset, s, p) for s in SEEDS]
                corrupt_valid = [v for v in corrupt_vals if v is not None]
                if not corrupt_valid:
                    degradation[f"p{_tag(p)}"] = None
                    continue

                ba_corrupt = float(np.mean(corrupt_valid))
                degradation[f"p{_tag(p)}"] = round(ba_corrupt, 4)

                cc = max(0.0, min(1.0, chance_correct(ba_corrupt, n_cls)))
                dataset_b.append(cc)
                b_cells.append(cc)

                if ba_clean and ba_clean > 0:
                    cdr = ba_corrupt / ba_clean
                    dataset_cdr.append(cdr)
                    cdr_cells.append(cdr)

            per_dataset[dataset] = {
                "ba_clean":     round(ba_clean, 4) if ba_clean else None,
                "B_dataset":    round(float(np.mean(dataset_b)),   4) if dataset_b   else None,
                "CDR_dataset":  round(float(np.mean(dataset_cdr)), 4) if dataset_cdr else None,
                "degradation":  degradation,
            }

        summary[model] = {
            "B_m":   round(float(np.mean(b_cells)),   4) if b_cells   else None,
            "CDR_m": round(float(np.mean(cdr_cells)), 4) if cdr_cells else None,
            "per_dataset": per_dataset,
            "n_cells": len(b_cells),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
    print(f"Written: {SUMMARY_OUT}")

    # Print ranked table
    print(f"\n{'Model':<12} {'B_m (robust util)':>20} {'CDR_m (diagnostic)':>20} {'n_cells':>10}")
    print("-" * 65)
    ranked = sorted(
        summary.items(),
        key=lambda kv: kv[1]["B_m"] if kv[1]["B_m"] is not None else -1,
        reverse=True,
    )
    for model, s in ranked:
        b   = f"{s['B_m']:.4f}"    if s["B_m"]   is not None else "N/A"
        cdr = f"{s['CDR_m']:.4f}"  if s["CDR_m"] is not None else "N/A"
        print(f"{model:<12} {b:>20} {cdr:>20} {s['n_cells']:>10}")

    print()
    print("Primary metric: B_m (chance-corrected robust utility, used in UEB-General ranking)")
    print("CDR is a diagnostic: it can reward models that are near-chance on clean data.")


if __name__ == "__main__":
    main()
