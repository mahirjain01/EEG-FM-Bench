#!/usr/bin/env python3
"""
UniformEvalBench — Phase 0 pilot aggregation.

Reads all 54 per-run JSONs, computes per-(model, dataset, mode) mean/std over 3 seeds,
AdaptGap and NormAdaptGain per (model, dataset), and Kendall's tau between LP and FT
model rankings per dataset. Writes a summary JSON and prints a human-readable table.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("/home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench/uniformevalbench/experiments/results/pilot/per_run")
OUTPUT = Path("/home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench/uniformevalbench/experiments/results/pilot/kendall_tau_summary.json")

MODELS = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve", "moment"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload"]
MODES = ["lp", "ft"]


def load_runs():
    # Returns dict of {(model, dataset, mode): {"last": [accs], "bestval": [accs]}}
    by_cell = defaultdict(lambda: {"last": [], "bestval": []})
    for p in sorted(RESULTS_DIR.glob("*.json")):
        j = json.loads(p.read_text())
        if j.get("status") != "complete":
            continue
        # Prefer best-val if available, fall back to legacy test_metrics
        bv = (j.get("test_metrics_bestval") or {}).get("balanced_acc")
        last = (j.get("test_metrics_last") or j.get("test_metrics") or {}).get("balanced_acc")
        key = (j["model"], j["dataset"], j["mode"])
        if bv is not None:
            by_cell[key]["bestval"].append(float(bv))
        if last is not None:
            by_cell[key]["last"].append(float(last))
    return by_cell


def compute_summary(by_cell, selector):
    """selector in {'last', 'bestval'}. Returns (per_dataset, taus, cell_mean, cell_std)."""
    cell_mean = {}
    cell_std = {}
    for k, v in by_cell.items():
        xs = v[selector]
        if not xs:
            continue
        cell_mean[k] = float(np.mean(xs))
        cell_std[k] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

    per_dataset = {}
    taus = []
    spearman_per_dataset = {}

    for dataset in DATASETS:
        lp_by_model = {m: cell_mean.get((m, dataset, "lp")) for m in MODELS}
        ft_by_model = {m: cell_mean.get((m, dataset, "ft")) for m in MODELS}
        lp_std = {m: cell_std.get((m, dataset, "lp"), 0.0) for m in MODELS}
        ft_std = {m: cell_std.get((m, dataset, "ft"), 0.0) for m in MODELS}

        valid_models = [m for m in MODELS if lp_by_model[m] is not None and ft_by_model[m] is not None]
        if len(valid_models) < 2:
            continue

        lp_rank = sorted(valid_models, key=lambda m: lp_by_model[m], reverse=True)
        ft_rank = sorted(valid_models, key=lambda m: ft_by_model[m], reverse=True)

        lp_vals = [lp_by_model[m] for m in valid_models]
        ft_vals = [ft_by_model[m] for m in valid_models]
        tau_res = stats.kendalltau(lp_vals, ft_vals)
        spear = stats.spearmanr(lp_vals, ft_vals)

        per_model = {}
        for m in valid_models:
            lp = lp_by_model[m]
            ft = ft_by_model[m]
            gap = ft - lp
            norm_gain = gap / (1 - lp) if abs(1 - lp) > 1e-9 else None
            per_model[m] = {
                "lp_mean": lp,
                "lp_std": lp_std[m],
                "ft_mean": ft,
                "ft_std": ft_std[m],
                "adaptgap": gap,
                "norm_adapt_gain": norm_gain,
            }

        per_dataset[dataset] = {
            "kendall_tau": float(tau_res.statistic) if hasattr(tau_res, "statistic") else float(tau_res[0]),
            "kendall_p": float(tau_res.pvalue) if hasattr(tau_res, "pvalue") else float(tau_res[1]),
            "spearman_rho": float(spear.statistic) if hasattr(spear, "statistic") else float(spear[0]),
            "lp_ranking": lp_rank,
            "ft_ranking": ft_rank,
            "per_model": per_model,
        }
        spearman_per_dataset[dataset] = per_dataset[dataset]["spearman_rho"]
        taus.append(per_dataset[dataset]["kendall_tau"])

    return per_dataset, taus, spearman_per_dataset


def bootstrap_ci(taus, n_boot=10000, seed=0):
    if len(taus) <= 1:
        return None, None
    rng = np.random.default_rng(seed)
    tau_arr = np.array(taus)
    boot_means = [float(np.mean(rng.choice(tau_arr, size=len(tau_arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def print_summary(label, per_dataset, taus, spearman_per_dataset):
    print("=" * 78)
    print(f"UniformEvalBench Pilot — {label}")
    print("=" * 78)
    for dataset in DATASETS:
        if dataset not in per_dataset:
            continue
        d = per_dataset[dataset]
        print(f"\n>>> Dataset: {dataset}")
        print(f"    Kendall's tau (LP vs FT rankings): {d['kendall_tau']:+.4f}   (p={d['kendall_p']:.4f})")
        print(f"    Spearman's rho:                    {d['spearman_rho']:+.4f}")
        print(f"    LP ranking (best -> worst): {d['lp_ranking']}")
        print(f"    FT ranking (best -> worst): {d['ft_ranking']}")
        print(f"    {'model':<10} {'LP mean':>10} {'LP std':>8}  {'FT mean':>10} {'FT std':>8}  {'AdaptGap':>10}  {'NormAGain':>10}")
        for m in MODELS:
            if m not in d["per_model"]:
                continue
            pm = d["per_model"][m]
            nag = pm["norm_adapt_gain"]
            nag_s = f"{nag:+.4f}" if nag is not None else "   n/a"
            print(f"    {m:<10} {pm['lp_mean']:>10.4f} {pm['lp_std']:>8.4f}  {pm['ft_mean']:>10.4f} {pm['ft_std']:>8.4f}  {pm['adaptgap']:>+10.4f}  {nag_s:>10}")

    ci_low, ci_high = bootstrap_ci(taus)
    print("\n" + "-" * 78)
    print(f"[{label}] mean Kendall's tau across {len(taus)} datasets: {float(np.mean(taus)):+.4f}")
    print(f"[{label}] median Kendall's tau:                  {float(np.median(taus)):+.4f}")
    if ci_low is not None:
        print(f"[{label}] mean tau bootstrap 95% CI:            [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"[{label}] Claim 1 pass (mean tau < 0.7):        {bool(float(np.mean(taus)) < 0.7)}")


def main():
    by_cell = load_runs()

    # Compute both summaries
    pd_last, taus_last, spear_last = compute_summary(by_cell, "last")
    pd_bv, taus_bv, spear_bv = compute_summary(by_cell, "bestval")

    ci_low_last, ci_high_last = bootstrap_ci(taus_last)
    ci_low_bv, ci_high_bv = bootstrap_ci(taus_bv)

    summary = {
        "last_epoch": {
            "per_dataset": pd_last,
            "overall": {
                "mean_kendall_tau": float(np.mean(taus_last)) if taus_last else None,
                "median_kendall_tau": float(np.median(taus_last)) if taus_last else None,
                "mean_kendall_tau_ci95_low": ci_low_last,
                "mean_kendall_tau_ci95_high": ci_high_last,
                "spearman_rho_per_dataset": spear_last,
                "n_tasks": len(taus_last),
                "claim1_pass_mean_tau_below_0_7": bool(float(np.mean(taus_last)) < 0.7) if taus_last else None,
            },
        },
        "bestval": {
            "per_dataset": pd_bv,
            "overall": {
                "mean_kendall_tau": float(np.mean(taus_bv)) if taus_bv else None,
                "median_kendall_tau": float(np.median(taus_bv)) if taus_bv else None,
                "mean_kendall_tau_ci95_low": ci_low_bv,
                "mean_kendall_tau_ci95_high": ci_high_bv,
                "spearman_rho_per_dataset": spear_bv,
                "n_tasks": len(taus_bv),
                "claim1_pass_mean_tau_below_0_7": bool(float(np.mean(taus_bv)) < 0.7) if taus_bv else None,
            },
        },
        "counts": {
            f"{k[0]}_{k[1]}_{k[2]}": {"last": len(v["last"]), "bestval": len(v["bestval"])}
            for k, v in sorted(by_cell.items())
        },
    }
    OUTPUT.write_text(json.dumps(summary, indent=2))

    print_summary("LAST-EPOCH metrics", pd_last, taus_last, spear_last)
    print()
    print_summary("BEST-VAL-SELECTED metrics", pd_bv, taus_bv, spear_bv)

    # Side-by-side comparison table
    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE: last-epoch vs best-val mean Kendall's tau per dataset")
    print("=" * 78)
    print(f"  {'dataset':<16} {'tau_last':>10} {'tau_bv':>10}  {'Δ(bv-last)':>12}")
    for ds in DATASETS:
        tl = pd_last.get(ds, {}).get("kendall_tau")
        tb = pd_bv.get(ds, {}).get("kendall_tau")
        if tl is None or tb is None:
            continue
        print(f"  {ds:<16} {tl:>+10.4f} {tb:>+10.4f}  {tb - tl:>+12.4f}")

    mean_last = float(np.mean(taus_last)) if taus_last else float("nan")
    mean_bv = float(np.mean(taus_bv)) if taus_bv else float("nan")
    print(f"\n  {'OVERALL mean':<16} {mean_last:>+10.4f} {mean_bv:>+10.4f}  {mean_bv - mean_last:>+12.4f}")
    print(f"Summary JSON written to: {OUTPUT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
