"""
UniformEvalBench composite ranking.

Three-layer ranking system
--------------------------

Layer 1 — Per-axis scores (primary scientific results)
    A_m  = mean_d [ cc(kNN@20_{m,d}) ]          Axis A: representation quality
    B_m  = mean_{d,p} [ cc(BA^corrupt_{m,d,p}) ] Axis B: robust utility
    C_m  = mean_d [ SPA_{m,d} ]                  Axis C: spectral grounding
    CDR_m = mean_{d,p} [ BA^corrupt / BA^clean ]  Diagnostic only, not ranked

Layer 2 — Balanced general-purpose score (default leaderboard)
    RepCore_m      = 0.75 * A_m + 0.25 * C_m
    UEB-General_m  = (RepCore_m + ε)^0.6 * (B_m + ε)^0.4 − ε    ε = 1e-4

    Geometric mean structure penalises catastrophic failure on either dimension.
    A model excellent on representations but collapsing under channel dropout
    cannot average its way to the top.

Layer 3 — Application profiles (additive, four presets)
    Representation:       0.85*A + 0.15*C
    Clinical deployment:  0.45*B + 0.30*A + 0.15*C + 0.10*CDR
    Neurophysiology:      0.55*C + 0.35*A + 0.10*B
    Balanced general:     UEB-General (default)

Why chance-correct?
    Raw BA is not cross-dataset comparable: 0.55 on a binary task ≠ 0.55 on
    5-class. Chance correction maps all datasets to the same [0,1] scale where
    0 = chance and 1 = perfect.  Prior work (EEG-FM-Bench) reports raw BA and
    therefore cannot aggregate across paradigms without this artefact.

Why geometric mean for UEB-General?
    Arithmetic mean allows a model to win by dominating one axis while failing
    another. The geometric mean (0.6/0.4 exponents matching the A/B importance
    ratio) penalises cross-axis imbalance.  A model with RepCore=0.9 and B=0.01
    scores (0.9006)^0.6 * (0.0101)^0.4 ≈ 0.09 — the collapsed robustness
    propagates through.

Missing values
    Models excluded from an axis (BIOT from SPA; MOMENT from robustness) receive
    NaN for that axis.  UEB-General is NaN if either component is NaN.  These
    models appear in the extended reference leaderboard but not the core one.

Usage
-----
    from uniformevalbench.metrics.ranking import load_all_scores, compute_composite, print_leaderboard

    scores = load_all_scores()          # reads result JSONs from disk
    df     = compute_composite(scores)  # adds RepCore, UEB-General, profiles
    print_leaderboard(df)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import warnings

import numpy as np

from .chance_corrected import chance_correct, DATASET_N_CLASSES

# ---------------------------------------------------------------------------
# Paths (relative to repo root — adjust if the metrics module is moved)
# ---------------------------------------------------------------------------

_HERE       = Path(__file__).resolve().parent
_REPO_ROOT  = _HERE.parent.parent
_RESULTS    = _REPO_ROOT / "uniformevalbench" / "experiments" / "results"

KNN_DIR     = _RESULTS / "axis_a" / "knn" / "per_run"
PILOT_DIR   = _RESULTS / "pilot" / "per_run"
B_V3_DIR    = _RESULTS / "axis_b_v3" / "per_run"
SPA_DIR     = _RESULTS / "axis_c" / "spa" / "per_run"

# Composite constant
_EPS = 1e-4

# ---------------------------------------------------------------------------
# Models and datasets in scope
# ---------------------------------------------------------------------------

MODELS_CORE = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve"]
MODELS_REF  = ["moment"]            # reference models (may have partial axes)
SEEDS       = [42, 123, 7]
DROPOUT_P   = [0.1, 0.25, 0.4]

# BIOT is excluded from SPA (STFT tokeniser trivially encodes band powers)
SPA_EXCLUDED = {"biot"}

# Datasets used in each axis
AXIS_A_DATASETS = [
    "bcic_2a", "hmc", "adftd", "motor_mv_img",
    "siena_scalp", "workload", "epilepsy_mimickers",
]
AXIS_B_DATASETS = [
    "bcic_2a", "hmc", "adftd", "motor_mv_img",
    "siena_scalp", "workload", "epilepsy_mimickers",
]
AXIS_C_DATASETS = [
    "bcic_2a", "hmc", "adftd", "motor_mv_img",
    "siena_scalp", "workload", "epilepsy_mimickers",
]

# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _knn_ba(model: str, dataset: str, seed: int) -> Optional[float]:
    """kNN@20 balanced accuracy for one cell."""
    d = _load_json(KNN_DIR / f"knn_{model}_{dataset}_s{seed}.json")
    if d is None or d.get("status") != "complete":
        return None
    return d.get("knn_primary")  # kNN@20


def _clean_ba(model: str, dataset: str, seed: int) -> Optional[float]:
    """Best-val balanced accuracy from Axis A FBP (clean baseline for CDR)."""
    # Try pilot/per_run first (lp_ prefix), then axis_a per_run
    for prefix, directory in [("lp", PILOT_DIR), ("lp", _RESULTS / "axis_a" / "per_run")]:
        d = _load_json(directory / f"{prefix}_{model}_{dataset}_s{seed}.json")
        if d and d.get("status") == "complete":
            bv = d.get("test_metrics_bestval") or d.get("test_metrics")
            if bv:
                return bv.get("balanced_acc")
    return None


def _corrupt_ba(model: str, dataset: str, seed: int, p: float) -> Optional[float]:
    """Best-val balanced accuracy from Axis B v3 corrupted run.

    Priority: test_metrics_bestval (best-val epoch selection on clean val)
              test_metrics_last    (last epoch, always written by sweep script)
              test_metrics         (legacy fallback)
    """
    tag = f"p{int(round(p * 100)):03d}"
    d = _load_json(B_V3_DIR / f"b3_{model}_{dataset}_s{seed}_{tag}.json")
    if d is None or d.get("status") != "complete":
        return None
    bv = (d.get("test_metrics_bestval")
          or d.get("test_metrics_last")
          or d.get("test_metrics"))
    return bv.get("balanced_acc") if bv else None


def _spa(model: str, dataset: str, seed: int) -> Optional[float]:
    """SPA score (mean R² over 5 bands) for one cell."""
    if model in SPA_EXCLUDED:
        return float("nan")
    d = _load_json(SPA_DIR / f"spa_{model}_{dataset}_s{seed}.json")
    if d is None or d.get("status") != "complete":
        return None
    return d.get("spa")


# ---------------------------------------------------------------------------
# Per-axis score computation
# ---------------------------------------------------------------------------

def _axis_a_score(model: str) -> float:
    """
    A_m = mean_d [ cc(kNN@20_{m,d}) ],  clipped to [0,1] per dataset before mean.
    """
    values = []
    for dataset in AXIS_A_DATASETS:
        n_cls = DATASET_N_CLASSES.get(dataset)
        if n_cls is None:
            continue
        per_seed = [_knn_ba(model, dataset, s) for s in SEEDS]
        valid = [v for v in per_seed if v is not None]
        if not valid:
            continue
        ba = float(np.mean(valid))
        cc = max(0.0, min(1.0, chance_correct(ba, n_cls)))
        values.append(cc)
    return float(np.mean(values)) if values else float("nan")


def _axis_b_scores(model: str) -> tuple[float, float]:
    """
    B_m   = mean_{d,p} [ cc(BA^corrupt_{m,d,p}) ]   robust utility (ranking score)
    CDR_m = mean_{d,p} [ BA^corrupt / BA^clean ]     diagnostic ratio

    Returns (B_m, CDR_m).  Both are NaN if no corrupted runs exist.
    """
    b_vals, cdr_vals = [], []
    for dataset in AXIS_B_DATASETS:
        n_cls = DATASET_N_CLASSES.get(dataset)
        if n_cls is None:
            continue
        clean_seed = [_clean_ba(model, dataset, s) for s in SEEDS]
        clean_valid = [v for v in clean_seed if v is not None]
        if not clean_valid:
            continue
        ba_clean = float(np.mean(clean_valid))

        for p in DROPOUT_P:
            corrupt_seed = [_corrupt_ba(model, dataset, s, p) for s in SEEDS]
            valid = [v for v in corrupt_seed if v is not None]
            if not valid:
                continue
            ba_corrupt = float(np.mean(valid))
            cc = max(0.0, min(1.0, chance_correct(ba_corrupt, n_cls)))
            b_vals.append(cc)
            if ba_clean > 0:
                cdr_vals.append(ba_corrupt / ba_clean)

    b_m   = float(np.mean(b_vals))   if b_vals   else float("nan")
    cdr_m = float(np.mean(cdr_vals)) if cdr_vals else float("nan")
    return b_m, cdr_m


def _axis_c_score(model: str) -> float:
    """C_m = mean_d [ SPA_{m,d} ].  NaN for SPA-excluded models (BIOT)."""
    if model in SPA_EXCLUDED:
        return float("nan")
    values = []
    for dataset in AXIS_C_DATASETS:
        per_seed = [_spa(model, dataset, s) for s in SEEDS]
        valid = [v for v in per_seed if v is not None and not np.isnan(v)]
        if not valid:
            continue
        values.append(float(np.mean(valid)))
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_axis_scores(models: list[str] | None = None) -> dict[str, dict]:
    """
    Compute per-axis scores for all models.

    Returns a dict keyed by model name:
        {
            "A":   float | nan,   # chance-corrected kNN@20
            "B":   float | nan,   # chance-corrected robust utility
            "CDR": float | nan,   # CDR retention ratio (diagnostic)
            "C":   float | nan,   # SPA mean R²
        }
    """
    if models is None:
        models = MODELS_CORE + MODELS_REF
    scores = {}
    for m in models:
        b_m, cdr_m = _axis_b_scores(m)
        scores[m] = {
            "A":   _axis_a_score(m),
            "B":   b_m,
            "CDR": cdr_m,
            "C":   _axis_c_score(m),
        }
    return scores


def compute_composite(scores: dict[str, dict]) -> dict[str, dict]:
    """
    Add RepCore, UEB-General, and four application profile scores to each model's
    score dict.  Input is the dict returned by compute_axis_scores().

    Returns an augmented copy — original dict is not modified.
    """
    result = {}
    for m, s in scores.items():
        A, B, C = s["A"], s["B"], s["C"]
        entry = dict(s)

        # RepCore: down-weights C to avoid double-counting (A and C share ~81% variance)
        if not (np.isnan(A) or np.isnan(C)):
            entry["RepCore"] = 0.75 * A + 0.25 * C
        else:
            entry["RepCore"] = float("nan")

        # UEB-General: geometric mean that punishes catastrophic failure
        rc, b = entry["RepCore"], B
        if not (np.isnan(rc) or np.isnan(b)):
            entry["UEB_General"] = (
                (rc + _EPS) ** 0.6 * (b + _EPS) ** 0.4 - _EPS
            )
        else:
            entry["UEB_General"] = float("nan")

        # Application profiles
        CDR = s["CDR"]

        def safe(w_a, w_b, w_c):
            if np.isnan(A) or np.isnan(B) or np.isnan(C):
                return float("nan")
            return w_a * A + w_b * B + w_c * C

        def safe_cdr(w_a, w_b, w_c, w_cdr):
            if np.isnan(A) or np.isnan(B) or np.isnan(C) or np.isnan(CDR):
                return float("nan")
            return w_a * A + w_b * B + w_c * C + w_cdr * CDR

        entry["profile_representation"] = (
            0.85 * A + 0.15 * C if not (np.isnan(A) or np.isnan(C)) else float("nan")
        )
        # Clinical: absolute robust utility dominates; CDR captures brittleness signal
        entry["profile_clinical"]       = safe_cdr(0.30, 0.45, 0.15, 0.10)
        entry["profile_neurophysiology"]= safe(0.35, 0.10, 0.55)
        entry["profile_balanced"]       = entry["UEB_General"]

        result[m] = entry
    return result


def load_all_scores(models: list[str] | None = None) -> dict[str, dict]:
    """Convenience: compute_axis_scores + compute_composite in one call."""
    return compute_composite(compute_axis_scores(models))


def print_leaderboard(scores: dict[str, dict], sort_by: str = "UEB_General") -> None:
    """Print a formatted leaderboard table sorted by `sort_by`."""
    cols = ["A", "B", "CDR", "C", "RepCore", "UEB_General"]
    header = f"{'Model':<12}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))

    def _sort_key(item):
        v = item[1].get(sort_by, float("nan"))
        return -v if not np.isnan(v) else float("inf")

    for model, s in sorted(scores.items(), key=_sort_key):
        row = f"{model:<12}"
        for c in cols:
            v = s.get(c, float("nan"))
            row += f"{v:>12.4f}" if not np.isnan(v) else f"{'N/A':>12}"
        print(row)
