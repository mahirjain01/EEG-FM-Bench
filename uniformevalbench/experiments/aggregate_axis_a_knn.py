#!/usr/bin/env python3
"""
UniformEvalBench — Axis A kNN aggregation.

Reads all per-run kNN JSONs from results/axis_a/knn/per_run/, pairs each run
with its FBP counterpart from results/pilot/per_run/, and produces:

  1. Per-dataset table: kNN@20 mean ± std vs FBP mean ± std per model
  2. Model leaderboard: ranked by mean kNN@20 across all datasets
  3. Kendall τ between kNN@20 and FBP rankings per dataset (validates FBP
     as a proxy for representation quality)
  4. k-sensitivity table: max − min balanced acc across k ∈ {5,10,20,50}
     per (model, dataset) — shows the single hyperparameter barely matters
  5. results/axis_a/knn_summary.json with all of the above

Usage
-----
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/aggregate_axis_a_knn.py
    python uniformevalbench/experiments/aggregate_axis_a_knn.py --dataset bcic_2a
    python uniformevalbench/experiments/aggregate_axis_a_knn.py --latex
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "uniformevalbench" / "experiments" / "results"
KNN_DIR      = RESULTS_ROOT / "axis_a" / "knn" / "per_run"
FBP_DIR      = RESULTS_ROOT / "pilot" / "per_run"
OUT_PATH     = RESULTS_ROOT / "axis_a" / "knn_summary.json"

MODELS   = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve", "moment"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload",
            "epilepsy_mimickers"]  # 7 datasets — epilepsy_mimickers MOMENT-excluded
SEEDS    = [42, 7, 123]
K_VALUES = [5, 10, 20, 50]
K_PRIMARY = 20

# Human-readable dataset labels
DATASET_LABELS = {
    "bcic_2a":            "BCIC-IV-2a",
    "hmc":                "HMC",
    "adftd":              "ADFTD",
    "motor_mv_img":       "Motor-MI",
    "siena_scalp":        "Siena",
    "workload":           "Workload",
    "epilepsy_mimickers": "Epil-Mim",
}

MODEL_LABELS = {
    "eegpt":   "EEGPT",
    "labram":  "LaBraM",
    "cbramod": "CBraMod",
    "biot":    "BIOT",
    "csbrain": "CSBrain",
    "reve":    "REVE",
    "moment":  "MOMENT",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_knn_runs() -> dict:
    """
    Returns nested dict: runs[model][dataset][seed] = result dict (or None).
    """
    runs: dict = {m: {d: {} for d in DATASETS} for m in MODELS}
    for path in sorted(KNN_DIR.glob("knn_*.json")):
        try:
            j = json.loads(path.read_text())
        except Exception:
            continue
        m, d, s = j.get("model"), j.get("dataset"), j.get("seed")
        if m in runs and d in runs[m]:
            runs[m][d][s] = j
    return runs


def _load_fbp_runs() -> dict:
    """
    Returns nested dict: fbp[model][dataset][seed] = result dict (or None).
    """
    fbp: dict = {m: {d: {} for d in DATASETS} for m in MODELS}
    for path in sorted(FBP_DIR.glob("lp_*.json")):
        try:
            j = json.loads(path.read_text())
        except Exception:
            continue
        m, d, s = j.get("model"), j.get("dataset"), j.get("seed")
        if m in fbp and d in fbp.get(m, {}) and j.get("mode") == "lp":
            fbp[m][d][s] = j
    return fbp


def _extract_knn_primary(result: dict | None) -> float | None:
    if result is None or result.get("status") != "complete":
        return None
    v = result.get("knn_primary")
    return float(v) if v is not None else None


def _extract_fbp(result: dict | None) -> float | None:
    if result is None or result.get("status") != "complete":
        return None
    bv = result.get("test_metrics_bestval") or result.get("test_metrics") or {}
    v = bv.get("balanced_acc")
    return float(v) if v is not None else None


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    if len(valid) == 1:
        return valid[0], None
    return float(np.mean(valid)), float(np.std(valid, ddof=1))


# ---------------------------------------------------------------------------
# Per-dataset aggregation
# ---------------------------------------------------------------------------

def aggregate_per_dataset(knn_runs: dict, fbp_runs: dict) -> dict:
    """
    Returns: {dataset: {model: {"knn_mean", "knn_std", "fbp_mean", "fbp_std",
                                 "knn_per_seed", "fbp_per_seed",
                                 "k_sensitivity_mean"}}}
    """
    per_dataset: dict = {}
    for dataset in DATASETS:
        per_dataset[dataset] = {}
        for model in MODELS:
            knn_vals = [_extract_knn_primary(knn_runs[model][dataset].get(s)) for s in SEEDS]
            fbp_vals = [_extract_fbp(fbp_runs[model][dataset].get(s)) for s in SEEDS]

            knn_mean, knn_std = _mean_std(knn_vals)
            fbp_mean, fbp_std = _mean_std(fbp_vals)

            # k-sensitivity: max − min across k values, averaged over seeds
            k_sens_vals = []
            for s in SEEDS:
                r = knn_runs[model][dataset].get(s)
                if r and r.get("status") == "complete":
                    pk = r.get("knn_per_k", {})
                    valid_k = [v for v in pk.values() if v is not None]
                    if valid_k:
                        k_sens_vals.append(max(valid_k) - min(valid_k))
            k_sens_mean = float(np.mean(k_sens_vals)) if k_sens_vals else None

            per_dataset[dataset][model] = {
                "knn_mean":        knn_mean,
                "knn_std":         knn_std,
                "fbp_mean":        fbp_mean,
                "fbp_std":         fbp_std,
                "knn_per_seed":    {str(s): _extract_knn_primary(knn_runs[model][dataset].get(s)) for s in SEEDS},
                "fbp_per_seed":    {str(s): _extract_fbp(fbp_runs[model][dataset].get(s)) for s in SEEDS},
                "k_sensitivity":   k_sens_mean,
            }
    return per_dataset


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(per_dataset: dict) -> list[dict]:
    """
    Rank models by mean kNN@20 balanced accuracy, macro-averaged across all
    datasets for which results are available.

    Returns list of {model, knn_macro, fbp_macro, n_datasets} sorted desc by knn_macro.
    """
    rows = []
    for model in MODELS:
        knn_vals, fbp_vals = [], []
        for dataset in DATASETS:
            cell = per_dataset[dataset][model]
            if cell["knn_mean"] is not None:
                knn_vals.append(cell["knn_mean"])
            if cell["fbp_mean"] is not None:
                fbp_vals.append(cell["fbp_mean"])
        rows.append({
            "model":       model,
            "knn_macro":   float(np.mean(knn_vals)) if knn_vals else None,
            "fbp_macro":   float(np.mean(fbp_vals)) if fbp_vals else None,
            "n_datasets":  len(knn_vals),
        })
    rows.sort(key=lambda r: r["knn_macro"] or -1, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Kendall τ between kNN and FBP per dataset
# ---------------------------------------------------------------------------

def compute_kendall_tau(per_dataset: dict) -> dict:
    """
    For each dataset, rank models by kNN@20 and by FBP, then compute Kendall τ.

    Returns {dataset: {"tau", "p_value", "n_models"}} plus a "mean_tau" key.
    """
    results: dict = {}
    taus = []

    for dataset in DATASETS:
        cells = per_dataset[dataset]
        knn_scores, fbp_scores = [], []
        for model in MODELS:
            knn_m = cells[model]["knn_mean"]
            fbp_m = cells[model]["fbp_mean"]
            if knn_m is not None and fbp_m is not None:
                knn_scores.append(knn_m)
                fbp_scores.append(fbp_m)

        if len(knn_scores) < 3:
            results[dataset] = {"tau": None, "p_value": None, "n_models": len(knn_scores)}
            continue

        tau, p = kendalltau(fbp_scores, knn_scores)
        results[dataset] = {
            "tau":      float(tau),
            "p_value":  float(p),
            "n_models": len(knn_scores),
        }
        taus.append(tau)

    results["mean_tau"] = float(np.mean(taus)) if taus else None
    return results


# ---------------------------------------------------------------------------
# k-sensitivity summary
# ---------------------------------------------------------------------------

def summarize_k_sensitivity(per_dataset: dict) -> dict:
    """
    Returns {model: {dataset: k_sensitivity_mean}}, plus global mean.
    """
    summary: dict = {}
    all_sens = []
    for model in MODELS:
        summary[model] = {}
        for dataset in DATASETS:
            v = per_dataset[dataset][model]["k_sensitivity"]
            summary[model][dataset] = v
            if v is not None:
                all_sens.append(v)
    summary["global_mean"] = float(np.mean(all_sens)) if all_sens else None
    return summary


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None, prec: int = 4) -> str:
    return f"{v:.{prec}f}" if v is not None else "   —  "


def print_per_dataset_table(per_dataset: dict, dataset_filter: str | None = None, latex: bool = False):
    datasets = [dataset_filter] if dataset_filter else DATASETS
    for dataset in datasets:
        label = DATASET_LABELS.get(dataset, dataset)
        print(f"\n{'─'*70}")
        print(f"  {label} — kNN@{K_PRIMARY} vs FBP (mean ± std over 3 seeds)")
        print(f"{'─'*70}")

        header = f"  {'Model':<12}  {'kNN@20':>8}  {'± std':>6}  {'FBP':>8}  {'± std':>6}  {'k-sens':>7}"
        print(header)
        print(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}")

        for model in MODELS:
            c = per_dataset[dataset][model]
            km, ks = c["knn_mean"], c["knn_std"]
            fm, fs = c["fbp_mean"], c["fbp_std"]
            ksens  = c["k_sensitivity"]
            print(f"  {MODEL_LABELS.get(model, model):<12}  {_fmt(km)}  "
                  f"{_fmt(ks, 3):>6}  {_fmt(fm)}  {_fmt(fs, 3):>6}  {_fmt(ksens, 4):>7}")

        if latex:
            print(f"\n  LaTeX snippet for {label}:")
            print(f"  \\begin{{tabular}}{{lrrrrrr}}")
            print(f"  Model & kNN@20 & ±std & FBP & ±std & k-sens \\\\\\midrule")
            for model in MODELS:
                c = per_dataset[dataset][model]
                km, ks = c["knn_mean"], c["knn_std"]
                fm, fs = c["fbp_mean"], c["fbp_std"]
                ksens  = c["k_sensitivity"]
                print(f"  {MODEL_LABELS.get(model, model)} & {_fmt(km)} & {_fmt(ks, 3)} & "
                      f"{_fmt(fm)} & {_fmt(fs, 3)} & {_fmt(ksens, 4)} \\\\")
            print(f"  \\end{{tabular}}")


def print_leaderboard(leaderboard: list[dict], latex: bool = False):
    print(f"\n{'='*60}")
    print(f"  Model Leaderboard — ranked by macro-avg kNN@{K_PRIMARY}")
    print(f"{'='*60}")
    header = f"  {'Rank':>4}  {'Model':<10}  {'kNN@20 macro':>13}  {'FBP macro':>10}  {'Δ (kNN−FBP)':>12}  {'N datasets':>10}"
    print(header)
    print(f"  {'─'*4}  {'─'*10}  {'─'*13}  {'─'*10}  {'─'*12}  {'─'*10}")
    for row in leaderboard:
        km  = row["knn_macro"]
        fm  = row["fbp_macro"]
        delta = (km - fm) if km is not None and fm is not None else None
        print(f"  {row['rank']:>4}  {MODEL_LABELS.get(row['model'], row['model']):<10}  "
              f"{_fmt(km):>13}  {_fmt(fm):>10}  {_fmt(delta):>12}  {row['n_datasets']:>10}")

    if latex:
        print(f"\n  LaTeX leaderboard:")
        print(f"  \\begin{{tabular}}{{rlrrrr}}")
        print(f"  Rank & Model & kNN@20 & FBP & $\\Delta$ & N \\\\\\midrule")
        for row in leaderboard:
            km = row["knn_macro"]
            fm = row["fbp_macro"]
            delta = (km - fm) if km is not None and fm is not None else None
            print(f"  {row['rank']} & {MODEL_LABELS.get(row['model'], row['model'])} & "
                  f"{_fmt(km)} & {_fmt(fm)} & {_fmt(delta)} & {row['n_datasets']} \\\\")
        print(f"  \\end{{tabular}}")


def print_kendall_tau(tau_results: dict, latex: bool = False):
    print(f"\n{'='*60}")
    print("  Kendall τ: kNN@20 vs FBP rankings per dataset")
    print(f"  (τ > 0.7 validates FBP as a proxy for kNN)")
    print(f"{'='*60}")
    header = f"  {'Dataset':<16}  {'τ':>7}  {'p':>8}  {'N':>4}  {'Agreement?':>10}"
    print(header)
    print(f"  {'─'*16}  {'─'*7}  {'─'*8}  {'─'*4}  {'─'*10}")

    for dataset in DATASETS:
        r = tau_results.get(dataset, {})
        tau = r.get("tau")
        p   = r.get("p_value")
        n   = r.get("n_models", 0)
        agree = ("YES" if tau is not None and tau >= 0.7 else
                 "no"  if tau is not None else
                 "  —")
        label = DATASET_LABELS.get(dataset, dataset)
        print(f"  {label:<16}  {_fmt(tau, 3):>7}  "
              f"{'<.001' if (p is not None and p < 0.001) else _fmt(p, 3):>8}  "
              f"{n:>4}  {agree:>10}")

    mean_tau = tau_results.get("mean_tau")
    print(f"\n  Mean τ across datasets: {_fmt(mean_tau, 3)}")
    if mean_tau is not None:
        conclusion = (
            "FBP is a valid proxy for kNN ranking (τ ≥ 0.7 confirmed)"
            if mean_tau >= 0.7 else
            "FBP diverges from kNN ranking — kNN@20 should be sole primary"
        )
        print(f"  → {conclusion}")


def print_k_sensitivity(k_sens: dict, latex: bool = False):
    print(f"\n{'='*60}")
    print("  k-sensitivity: max − min balanced_acc across k ∈ {5,10,20,50}")
    print("  (low = kNN@20 result robust to choice of k)")
    print(f"{'='*60}")
    datasets = [d for d in DATASETS if any(
        k_sens.get(m, {}).get(d) is not None for m in MODELS
    )]
    header = f"  {'Model':<10}  " + "  ".join(f"{DATASET_LABELS.get(d, d):>9}" for d in datasets)
    print(header)
    print(f"  {'─'*10}  " + "  ".join("─"*9 for _ in datasets))
    for model in MODELS:
        row = f"  {MODEL_LABELS.get(model, model):<10}  "
        row += "  ".join(f"{_fmt(k_sens.get(model, {}).get(d), 4):>9}" for d in datasets)
        print(row)
    global_mean = k_sens.get("global_mean")
    print(f"\n  Global mean k-sensitivity: {_fmt(global_mean, 4)}")
    if global_mean is not None and global_mean < 0.02:
        print(f"  → k-sensitivity < 0.02 — kNN@20 result is robust to k choice")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate Axis A kNN results")
    ap.add_argument("--dataset", default=None, help="Show per-dataset table for one dataset only")
    ap.add_argument("--latex",   action="store_true", help="Also print LaTeX snippets")
    ap.add_argument("--no-save", action="store_true", help="Don't write summary JSON")
    args = ap.parse_args()

    if not KNN_DIR.exists() or not any(KNN_DIR.glob("knn_*.json")):
        print(f"No kNN results found in {KNN_DIR}")
        print("Run axis_a_knn.py first.")
        sys.exit(1)

    print(f"Loading kNN results from {KNN_DIR} ...")
    knn_runs = _load_knn_runs()
    print(f"Loading FBP results from {FBP_DIR} ...")
    fbp_runs = _load_fbp_runs()

    per_dataset  = aggregate_per_dataset(knn_runs, fbp_runs)
    leaderboard  = build_leaderboard(per_dataset)
    tau_results  = compute_kendall_tau(per_dataset)
    k_sens       = summarize_k_sensitivity(per_dataset)

    print_per_dataset_table(per_dataset, args.dataset, args.latex)
    print_leaderboard(leaderboard, args.latex)
    print_kendall_tau(tau_results, args.latex)
    print_k_sensitivity(k_sens, args.latex)

    if not args.no_save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "per_dataset":  per_dataset,
            "leaderboard":  leaderboard,
            "kendall_tau":  tau_results,
            "k_sensitivity": k_sens,
        }
        OUT_PATH.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written → {OUT_PATH}")


if __name__ == "__main__":
    main()
