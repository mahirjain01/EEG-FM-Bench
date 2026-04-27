#!/usr/bin/env python3
"""Render Axis B v2 figures from cdr_v2_summary.json + per-run JSONs.

Outputs:
  figures/axis_b_v2_degradation_curves.png     - 6-panel grid (one per dataset), 6 models each
  figures/axis_b_v2_cdr_bars.png                - mean CDR per model with std error bars
  figures/axis_b_v2_cdr_heatmap.png             - per-(model, dataset) CDR heatmap
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("/home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench/uniformevalbench/experiments/results/axis_b_v2")
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODELS = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload"]
LEVELS = [0.0, 0.1, 0.25, 0.4]

# Colorblind-friendly palette
COLORS = {
    "eegpt":   "#1f77b4",
    "labram":  "#ff7f0e",
    "cbramod": "#2ca02c",
    "biot":    "#d62728",
    "csbrain": "#9467bd",
    "reve":    "#8c564b",
}
MARKERS = {"eegpt":"o","labram":"s","cbramod":"^","biot":"D","csbrain":"P","reve":"X"}

summary = json.loads((RESULTS / "cdr_v2_summary.json").read_text())
per_model = summary["per_model"]


# --- Figure 1: per-dataset degradation curves (6 subplots) ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
axes = axes.ravel()
for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    for m in MODELS:
        curves = per_model[m].get("curves_by_dataset", {}).get(ds, {})
        if not curves:
            continue
        xs = [p for p in LEVELS if str(p) in curves or p in curves]
        means, stds = [], []
        for p in LEVELS:
            c = curves.get(p) or curves.get(str(p))
            if c is None:
                continue
            means.append(c["mean"])
            stds.append(c["std"])
        xs = [p for p in LEVELS if (curves.get(p) or curves.get(str(p))) is not None]
        ax.errorbar(xs, means, yerr=stds, marker=MARKERS[m], color=COLORS[m],
                    label=m.upper(), capsize=2, linewidth=1.2, markersize=5)
    ax.set_title(ds, fontsize=11)
    ax.axhline(0.5, linestyle=":", color="grey", linewidth=0.8, label=None)
    ax.set_xticks(LEVELS)
    ax.grid(alpha=0.3)
    if idx % 3 == 0:
        ax.set_ylabel("Test balanced_acc")
    if idx >= 3:
        ax.set_xlabel("Channel-dropout p (test-only)")
# Single legend above
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Axis B v2 — Channel-dropout degradation curves (FBP, test-only, last-epoch)", y=1.06, fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "axis_b_v2_degradation_curves.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG_DIR / 'axis_b_v2_degradation_curves.png'}")


# --- Figure 2: mean CDR bar chart ---
fig, ax = plt.subplots(figsize=(7.5, 4.2))
means = [per_model[m]["mean_cdr"] for m in MODELS]
stds = [per_model[m]["std_cdr"] for m in MODELS]
x = np.arange(len(MODELS))
bars = ax.bar(x, means, yerr=stds, capsize=4,
              color=[COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.8)
for i, m in enumerate(MODELS):
    ax.text(i, means[i] + stds[i] + 0.01, f"{means[i]:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in MODELS])
ax.set_ylabel("Mean CDR (test-only, last-epoch)")
ax.axhline(1.0, linestyle="--", color="grey", linewidth=0.8)
ax.set_ylim(0.5, 1.15)
ax.set_title(
    f"Axis B v2 — Channel Degradation Resilience\n"
    f"variance ratio (CDR / clean-FBP) = {summary['claim2']['variance_ratio']:.3f} "
    f"(< 1.5: Claim 2 FAILS)",
    fontsize=11,
)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "axis_b_v2_cdr_bars.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG_DIR / 'axis_b_v2_cdr_bars.png'}")


# --- Figure 3: CDR heatmap per (model, dataset) ---
# Compute per-(model, ds) CDR = mean over seeds of (mean over p>0 of acc(p)/acc(0))
import collections
per_model_ds_cdr = collections.defaultdict(lambda: collections.defaultdict(list))
for rec in summary["cdr_records"]:
    per_model_ds_cdr[rec["model"]][rec["dataset"]].append(rec["cdr"])

grid = np.full((len(MODELS), len(DATASETS)), np.nan)
for i, m in enumerate(MODELS):
    for j, ds in enumerate(DATASETS):
        vals = per_model_ds_cdr.get(m, {}).get(ds, [])
        if vals:
            grid[i, j] = np.mean(vals)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
vmin, vmax = 0.65, 1.05
im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels(DATASETS, rotation=25, ha="right")
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([m.upper() for m in MODELS])
for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        v = grid[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="red")
        else:
            color = "white" if v < 0.85 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=color)
fig.colorbar(im, ax=ax, label="CDR (mean over seeds)")
ax.set_title("Axis B v2 — per-(model, dataset) CDR (test-only dropout, last-epoch)")
fig.tight_layout()
fig.savefig(FIG_DIR / "axis_b_v2_cdr_heatmap.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG_DIR / 'axis_b_v2_cdr_heatmap.png'}")

print("all figures written to", FIG_DIR)
