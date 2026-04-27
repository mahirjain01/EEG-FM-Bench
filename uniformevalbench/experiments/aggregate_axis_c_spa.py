#!/usr/bin/env python3
"""
UniformEvalBench — Axis C SPA aggregator.

Reads all per-run SPA JSONs from results/axis_c/spa/per_run/ and produces:
  - Per-(model, dataset) SPA: mean ± std over 3 seeds, per band R²
  - Per-model mean SPA over datasets × seeds
  - SPA leaderboard table
  - Per-band breakdown (which frequency bands are best encoded)

Writes: results/axis_c/spa_summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS  = Path(__file__).resolve().parent / "results"
INPUT    = RESULTS / "axis_c" / "spa" / "per_run"
OUTPUT   = RESULTS / "axis_c" / "spa_summary.json"

MODELS   = ["eegpt", "labram", "cbramod", "csbrain", "reve"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload", "epilepsy_mimickers"]
SEEDS    = [42, 123, 7]
BANDS    = ["delta", "theta", "alpha", "beta", "gamma"]


def load_runs() -> dict:
    """Returns nested dict [model][dataset][seed] = {spa, r2_per_band, ...}."""
    table: dict = defaultdict(lambda: defaultdict(dict))
    n_loaded = 0
    for p in sorted(INPUT.glob("spa_*.json")):
        j = json.loads(p.read_text())
        if j.get("status") != "complete":
            continue
        m, ds, seed = j["model"], j["dataset"], j["seed"]
        spa = j.get("spa")
        if spa is None:
            continue
        table[m][ds][seed] = {
            "spa":         float(spa),
            "r2_per_band": j.get("r2_per_band"),
            "n_samples":   j.get("n_samples"),
            "emb_dim":     j.get("embedding_dim"),
        }
        n_loaded += 1
    print(f"Loaded {n_loaded} complete SPA records.")
    return table


def summarise(table: dict) -> tuple[dict, dict]:
    per_model_ds: dict = {}   # [model][dataset] → {mean, std, n, r2_per_band}
    per_model_all: dict = {}  # [model] → {mean, std} over all (ds, seed)

    for m in MODELS:
        per_model_ds[m] = {}
        all_spa: list[float] = []
        all_bands: list[list[float]] = []

        for ds in DATASETS:
            ds_spa: list[float] = []
            ds_bands: list[list[float]] = []

            for seed in SEEDS:
                cell = table.get(m, {}).get(ds, {}).get(seed)
                if cell is None:
                    continue
                ds_spa.append(cell["spa"])
                all_spa.append(cell["spa"])
                if cell.get("r2_per_band"):
                    ds_bands.append(cell["r2_per_band"])
                    all_bands.append(cell["r2_per_band"])

            mean_r2_bands = (
                list(np.mean(ds_bands, axis=0).tolist()) if ds_bands else None
            )
            per_model_ds[m][ds] = {
                "spa_mean":    float(np.mean(ds_spa)) if ds_spa else None,
                "spa_std":     float(np.std(ds_spa, ddof=1)) if len(ds_spa) > 1 else 0.0,
                "n":           len(ds_spa),
                "r2_per_band": mean_r2_bands,
            }

        mean_r2_model = (
            list(np.mean(all_bands, axis=0).tolist()) if all_bands else None
        )
        per_model_all[m] = {
            "spa_mean":    float(np.mean(all_spa)) if all_spa else None,
            "spa_std":     float(np.std(all_spa, ddof=1)) if len(all_spa) > 1 else 0.0,
            "n":           len(all_spa),
            "r2_per_band": mean_r2_model,
        }

    return per_model_ds, per_model_all


def print_report(per_model_ds: dict, per_model_all: dict):
    print("=" * 80)
    print("UniformEvalBench — Axis C: Spectral Prior Alignment (SPA)")
    print("R²: ridge regression from frozen embeddings → [δ, θ, α, β, γ] band powers")
    print("Models: eegpt, labram, cbramod, csbrain, reve  (BIOT excluded)")
    print("=" * 80)

    # Leaderboard
    sorted_models = sorted(
        [m for m in MODELS if per_model_all[m]["spa_mean"] is not None],
        key=lambda m: per_model_all[m]["spa_mean"],
        reverse=True,
    )
    print(f"\n{'model':<10} {'mean SPA':>10} {'std SPA':>9} {'n cells':>8}")
    for m in sorted_models:
        rec = per_model_all[m]
        print(f"{m:<10} {rec['spa_mean']:>10.4f} {rec['spa_std']:>9.4f} {rec['n']:>8}")

    # Per-(model, dataset) heatmap
    print(f"\nPer-(model, dataset) SPA (mean over 3 seeds):")
    header = f"  {'':10}" + "".join(f"{ds[:12]:>14}" for ds in DATASETS)
    print(header)
    for m in MODELS:
        row = f"  {m:<10}"
        for ds in DATASETS:
            rec = per_model_ds[m].get(ds, {})
            v = rec.get("spa_mean")
            row += f"{v:>14.4f}" if v is not None else f"{'n/a':>14}"
        print(row)

    # Per-band R² breakdown
    print(f"\nPer-band R² breakdown (mean over all datasets × seeds):")
    band_header = f"  {'model':<10}" + "".join(f"{b:>8}" for b in BANDS)
    print(band_header)
    for m in MODELS:
        bands = per_model_all[m].get("r2_per_band")
        row = f"  {m:<10}"
        if bands:
            row += "".join(f"{v:>8.4f}" for v in bands)
        else:
            row += "  n/a"
        print(row)

    print("=" * 80)


def main():
    if not INPUT.exists() or not any(INPUT.glob("spa_*.json")):
        print(f"No SPA result files found at {INPUT}.")
        print("Run `python axis_c_spa.py` first.")
        return

    table = load_runs()
    per_model_ds, per_model_all = summarise(table)

    summary = {
        "per_model":         per_model_all,
        "per_model_dataset": per_model_ds,
        "band_names":        BANDS,
        "models":            MODELS,
        "datasets":          DATASETS,
        "seeds":             SEEDS,
        "schema_version":    1,
        "note": (
            "SPA = R² of RidgeCV(frozen_embedding → 5 relative band powers). "
            "BIOT excluded (STFT tokenizer trivially encodes bands). "
            "Higher = embedding linearly encodes spectral content."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))

    print_report(per_model_ds, per_model_all)
    print(f"\nSummary JSON written to: {OUTPUT}")


if __name__ == "__main__":
    main()
