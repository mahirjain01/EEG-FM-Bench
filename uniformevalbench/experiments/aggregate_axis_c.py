#!/usr/bin/env python3
"""
UniformEvalBench — Axis C NAS aggregator.

Reads all per-run NAS JSONs from results/axis_c/per_run/ and produces:
  - Per-(model, dataset) NAS: mean ± std over 3 seeds, per baseline
  - Per-model mean NAS over datasets × seeds
  - NAS leaderboard table (primary = phase_shuffled baseline)
  - Baseline sensitivity check (how much do zero/mean/phase_shuffled differ?)

Claim 3 validity check (partial Spearman with cross-subject FBP) is NOT
computed here — it requires cross-subject FBP data that is a separate axis.
This aggregator reports the NAS values; the validity check is logged as pending.

Writes: results/axis_c/nas_summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS  = Path(__file__).resolve().parent / "results"
INPUT    = RESULTS / "axis_c" / "per_run"
OUTPUT   = RESULTS / "axis_c" / "nas_summary.json"

MODELS   = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload"]
SEEDS    = [42, 123, 7]
BASELINES = ["zero", "mean", "phase_shuffled"]


def load_runs() -> dict:
    """Returns nested dict [model][dataset][seed] = {bl: nas, ...}."""
    table: dict = defaultdict(lambda: defaultdict(dict))
    n_loaded = 0
    for p in sorted(INPUT.glob("nas_*.json")):
        j = json.loads(p.read_text())
        if j.get("status") != "complete":
            continue
        m, ds, seed = j["model"], j["dataset"], j["seed"]
        by_bl = {}
        for bl in BASELINES:
            rec = (j.get("nas_by_baseline") or {}).get(bl)
            if rec and rec.get("nas") is not None:
                by_bl[bl] = float(rec["nas"])
        if by_bl:
            table[m][ds][seed] = by_bl
            n_loaded += 1
    print(f"Loaded {n_loaded} complete NAS records.")
    return table


def summarise(table: dict) -> dict:
    per_model_ds: dict = {}           # [model][dataset] → {bl: mean, std, n}
    per_model_all: dict = {}          # [model] → {bl: mean, std} over all (ds, seed)

    for m in MODELS:
        per_model_ds[m] = {}
        per_bl_model: dict[str, list[float]] = defaultdict(list)

        for ds in DATASETS:
            per_bl_ds: dict[str, list[float]] = defaultdict(list)
            for seed in SEEDS:
                cell = table.get(m, {}).get(ds, {}).get(seed, {})
                for bl in BASELINES:
                    if bl in cell:
                        per_bl_ds[bl].append(cell[bl])
                        per_bl_model[bl].append(cell[bl])

            per_model_ds[m][ds] = {
                bl: {
                    "mean": float(np.mean(v)) if v else None,
                    "std":  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "n":    len(v),
                }
                for bl, v in per_bl_ds.items()
            }

        per_model_all[m] = {
            bl: {
                "mean": float(np.mean(v)) if v else None,
                "std":  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n":    len(v),
            }
            for bl, v in per_bl_model.items()
        }

    return per_model_ds, per_model_all


def baseline_sensitivity(per_model_all: dict) -> dict:
    """
    For each model, report range across baselines as a sensitivity metric.
    Small range → result is stable across baselines.
    """
    sensitivity = {}
    for m in MODELS:
        vals = {bl: (per_model_all[m].get(bl) or {}).get("mean")
                for bl in BASELINES}
        valid = [v for v in vals.values() if v is not None]
        sensitivity[m] = {
            "nas_by_baseline": vals,
            "range":  float(max(valid) - min(valid)) if len(valid) > 1 else None,
            "max_bl": max(vals, key=lambda k: vals[k] or -1) if valid else None,
        }
    return sensitivity


def print_report(per_model_ds: dict, per_model_all: dict, sensitivity: dict):
    print("=" * 80)
    print("UniformEvalBench — Axis C: Neurophysiological Alignment Score (NAS)")
    print("Primary baseline: phase_shuffled  |  Secondary: zero, mean")
    print("=" * 80)

    BL = "phase_shuffled"

    # Per-model summary
    print(f"\n{'model':<10} {'mean NAS':>10} {'std NAS':>9} {'n cells':>8}  "
          f"(primary = {BL} baseline)")
    for m in MODELS:
        rec = per_model_all[m].get(BL)
        if rec and rec["mean"] is not None:
            print(f"{m:<10} {rec['mean']:>10.4f} {rec['std']:>9.4f} {rec['n']:>8}")
        else:
            print(f"{m:<10} {'n/a':>10}")

    # Per-(model, dataset) heatmap
    print(f"\nPer-(model, dataset) NAS [{BL} baseline]:")
    header = f"  {'':10}" + "".join(f"{ds:>15}" for ds in DATASETS)
    print(header)
    for m in MODELS:
        row = f"  {m:<10}"
        for ds in DATASETS:
            rec = per_model_ds[m].get(ds, {}).get(BL)
            if rec and rec["mean"] is not None:
                row += f"{rec['mean']:>15.4f}"
            else:
                row += f"{'n/a':>15}"
        print(row)

    # Baseline sensitivity
    print(f"\nBaseline sensitivity (range across zero/mean/phase_shuffled):")
    print(f"  {'model':<10} {'zero':>10} {'mean':>10} {'phase_shuf':>12} {'range':>8}")
    for m in MODELS:
        s = sensitivity[m]
        vals = s["nas_by_baseline"]
        parts = " ".join(
            f"{vals.get(bl, None):>10.4f}" if vals.get(bl) is not None else f"{'n/a':>10}"
            for bl in ["zero", "mean", "phase_shuffled"]
        )
        rng = f"{s['range']:>8.4f}" if s["range"] is not None else f"{'n/a':>8}"
        print(f"  {m:<10} {parts} {rng}")

    # Claim 3 note
    print("\n" + "-" * 80)
    print("Claim 3 validity check (partial Spearman NAS vs CrossSubj-FBP | FBP):")
    print("  → PENDING: cross-subject FBP data required (separate experimental axis).")
    print("  Marginal Spearman(NAS, FBP) can be computed from pilot FBP values above.")
    print("=" * 80)


def main():
    if not INPUT.exists() or not any(INPUT.glob("nas_*.json")):
        print(f"No NAS result files found at {INPUT}.")
        print("Run `python axis_c_nas.py` first.")
        return

    table = load_runs()
    per_model_ds, per_model_all = summarise(table)
    sensitivity = baseline_sensitivity(per_model_all)

    summary = {
        "per_model":         per_model_all,
        "per_model_dataset": per_model_ds,
        "sensitivity":       sensitivity,
        "claim3": {
            "validity_check": "pending_cross_subject_FBP",
            "note": "Partial Spearman(NAS, CrossSubj-FBP | FBP) requires cross-subject experiment (Axis C validity check, §9.4).",
        },
        "primary_baseline":  "phase_shuffled",
        "schema_version":    1,
        "models":            MODELS,
        "datasets":          DATASETS,
        "seeds":             SEEDS,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))

    print_report(per_model_ds, per_model_all, sensitivity)
    print(f"\nSummary JSON written to: {OUTPUT}")


if __name__ == "__main__":
    main()
