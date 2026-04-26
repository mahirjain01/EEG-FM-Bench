#!/usr/bin/env python3
"""
Axis C SPA — Figure generation for paper.

Produces four publication-quality figures saved to
EEGBench-OpenDeployableSystem/figures/axis_c/

  fig1_spa_leaderboard.pdf/.png
  fig2_spa_heatmap.pdf/.png
  fig3_spa_bands.pdf/.png
  fig4_spa_crossaxis.pdf/.png

Usage:
    python plot_axis_c_spa.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ─── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
SPA_SUMMARY  = REPO_ROOT / "uniformevalbench" / "experiments" / "results" / "axis_c" / "spa_summary.json"
OUT_DIR      = REPO_ROOT / "EEGBench-OpenDeployableSystem" / "figures" / "axis_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── data ─────────────────────────────────────────────────────────────────────
summary = json.loads(SPA_SUMMARY.read_text())

MODELS   = ["reve", "eegpt", "csbrain", "cbramod", "labram"]  # SPA order
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload"]
BANDS    = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_LABELS = ["δ (0.5-4)", "θ (4-8)", "α (8-13)", "β (13-30)", "γ (30-45)"]
DS_LABELS   = ["BCIC-2A", "HMC", "ADFTD", "Motor-MV", "Siena", "Workload"]
MODEL_LABELS = {"reve": "REVE", "eegpt": "EEGPT", "csbrain": "CSBrain",
                "cbramod": "CBraMod", "labram": "LaBraM"}

# model colours — consistent across all figures
MODEL_COLORS = {
    "reve":    "#2166ac",   # blue
    "eegpt":   "#4dac26",   # green
    "csbrain": "#d6604d",   # salmon-red
    "cbramod": "#f4a582",   # peach
    "labram":  "#9970ab",   # purple
}

# ─── cross-axis data (hand-computed from pilot + CDR summaries) ───────────────
FBP_VALUES = {"reve": 0.5921, "eegpt": 0.5899, "csbrain": 0.5508, "labram": 0.5413, "cbramod": 0.4922}
CDR_VALUES = {"reve": 0.8468, "eegpt": 0.8696, "csbrain": 0.9778, "labram": 0.9665, "cbramod": 0.9488}

# ─── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.linewidth":  0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.05,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — SPA Leaderboard (horizontal bar + scatter with error bars)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_leaderboard():
    per_model = summary["per_model"]

    means = [per_model[m]["spa_mean"] for m in MODELS]
    stds  = [per_model[m]["spa_std"]  for m in MODELS]
    labels = [MODEL_LABELS[m] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    y = np.arange(len(MODELS))
    bars = ax.barh(y, means, xerr=stds, color=colors, height=0.55,
                   error_kw=dict(ecolor="#333333", capsize=4, elinewidth=1.2),
                   alpha=0.88, zorder=3)

    # annotate each bar
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.004, y[i], f"{m:.3f}", va="center", ha="left",
                fontsize=9.5, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean SPA  (R², mean ± std over 6 datasets × 3 seeds)", fontsize=10)
    ax.set_xlim(0.35, 1.02)
    ax.axvline(0.5, color="#aaaaaa", lw=0.7, ls="--", zorder=1, label="R²=0.5")
    ax.axvline(0.8, color="#888888", lw=0.7, ls=":",  zorder=1, label="R²=0.8")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.7)
    ax.set_title("Spectral Prior Alignment (SPA) — Model Leaderboard\n"
                 "(BIOT excluded: STFT tokenizer trivially encodes frequency bands)",
                 fontsize=10, pad=8)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.invert_yaxis()

    fig.tight_layout()
    save(fig, "fig1_spa_leaderboard")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — SPA Heatmap (model × dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_heatmap():
    per_ds = summary["per_model_dataset"]

    matrix = np.zeros((len(MODELS), len(DATASETS)))
    for i, m in enumerate(MODELS):
        for j, ds in enumerate(DATASETS):
            v = (per_ds[m].get(ds) or {}).get("spa_mean")
            matrix[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 3.6))

    cmap = LinearSegmentedColormap.from_list(
        "spa", ["#f7f7f7", "#92c5de", "#2166ac"], N=256
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.3, vmax=1.0)

    # annotate cells
    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            v = matrix[i, j]
            if not np.isnan(v):
                txt_color = "white" if v > 0.75 else "#333333"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9.5, color=txt_color, fontweight="bold" if v > 0.88 else "normal")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DS_LABELS, fontsize=10.5)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("SPA (mean R²)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_title("Axis C — SPA per (Model, Dataset)  [mean over 3 seeds]",
                 fontsize=11, pad=8)

    # draw thin grid lines
    for x in np.arange(-0.5, len(DATASETS), 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, len(MODELS), 1):
        ax.axhline(y, color="white", lw=0.8)

    fig.tight_layout()
    save(fig, "fig2_spa_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-band R² breakdown
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_bands():
    per_model = summary["per_model"]

    # shape [n_models, n_bands]
    band_matrix = np.array([
        per_model[m].get("r2_per_band") or [np.nan]*5
        for m in MODELS
    ])

    fig, ax = plt.subplots(figsize=(8, 3.8))

    n_bands  = len(BANDS)
    n_models = len(MODELS)
    width    = 0.14
    x        = np.arange(n_bands)

    for i, m in enumerate(MODELS):
        offset = (i - n_models/2 + 0.5) * width
        vals = band_matrix[i]
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                      alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=10.5)
    ax.set_ylabel("R² (Ridge regression)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--", zorder=1)
    ax.legend(loc="lower center", ncol=5, fontsize=9, framealpha=0.8,
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Per-Band R² Breakdown  [mean over all datasets × seeds]",
                 fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    save(fig, "fig3_spa_bands")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cross-axis: FBP vs SPA (bubble size = CDR)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_crossaxis():
    per_model = summary["per_model"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    panels = [
        ("FBP", FBP_VALUES, "Mean FBP (frozen probe accuracy)"),
        ("CDR", CDR_VALUES, "Mean CDR (channel degradation resilience)"),
    ]

    for ax, (x_label, x_vals, x_axlabel) in zip(axes, panels):
        for m in MODELS:
            spa = per_model[m]["spa_mean"]
            spa_std = per_model[m]["spa_std"]
            xv  = x_vals[m]

            ax.scatter(xv, spa, s=220, color=MODEL_COLORS[m],
                       zorder=4, alpha=0.9, edgecolors="white", linewidths=0.8)
            ax.errorbar(xv, spa, yerr=spa_std, fmt="none",
                        ecolor=MODEL_COLORS[m], elinewidth=1.2, capsize=3, alpha=0.7)
            ax.annotate(MODEL_LABELS[m],
                        xy=(xv, spa), xytext=(5, 4), textcoords="offset points",
                        fontsize=9, color=MODEL_COLORS[m], fontweight="bold")

        # regression line
        xs = np.array([x_vals[m] for m in MODELS])
        ys = np.array([per_model[m]["spa_mean"] for m in MODELS])
        z  = np.polyfit(xs, ys, 1)
        xl = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
        ax.plot(xl, np.polyval(z, xl), color="#888888", lw=1.2, ls="--", zorder=1)

        # Spearman rho annotation
        from scipy.stats import spearmanr
        rho, _ = spearmanr(xs, ys)
        ax.text(0.97, 0.05, f"Spearman ρ = {rho:+.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5, color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#cccccc"))

        ax.set_xlabel(x_axlabel, fontsize=10)
        ax.set_ylabel("SPA (mean R²)", fontsize=10)
        ax.set_title(f"SPA vs {x_label}", fontsize=11)
        ax.grid(alpha=0.2)

    fig.suptitle("Cross-Axis Analysis: Spectral Prior Alignment vs FBP and CDR",
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    save(fig, "fig4_spa_crossaxis")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Dataset-level SPA variation (line chart per dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_dataset_profiles():
    per_ds = summary["per_model_dataset"]

    fig, ax = plt.subplots(figsize=(8.5, 4))

    for m in MODELS:
        vals = [
            (per_ds[m].get(ds) or {}).get("spa_mean") or np.nan
            for ds in DATASETS
        ]
        ax.plot(range(len(DATASETS)), vals, marker="o", markersize=7,
                color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                lw=1.8, alpha=0.9)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DS_LABELS, fontsize=10.5)
    ax.set_ylabel("SPA (mean R² over 3 seeds)", fontsize=10)
    ax.set_ylim(0.25, 1.02)
    ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--", label="R²=0.5")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8, ncol=2)
    ax.set_title("SPA Per-Dataset Profiles — Spectral Encoding Across Tasks",
                 fontsize=11, pad=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    save(fig, "fig5_spa_dataset_profiles")


if __name__ == "__main__":
    print("Generating Axis C SPA figures...")
    fig1_leaderboard()
    fig2_heatmap()
    fig3_bands()
    fig4_crossaxis()
    fig5_dataset_profiles()
    print(f"\nAll figures saved to: {OUT_DIR}")
