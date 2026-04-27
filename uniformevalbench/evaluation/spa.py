"""
Axis C — Spectral Prior Alignment (SPA, pure, model-agnostic).

SPA measures whether a frozen encoder's representations track physiologically
meaningful spectral features, independently of classification performance.

Method
------
1. Collect (embedding, raw_eeg) pairs from the frozen encoder.
2. Compute relative band powers [δ, θ, α, β, γ] per sample from raw EEG.
3. Fit RidgeCV(embedding → band_powers) with 5-fold CV.
4. Report mean R² across the 5 bands as the SPA score.

A model scores high on SPA if its representations encode frequency-band
information even when trained purely for downstream classification.  Models
that learn clinically meaningful spectral patterns should score higher than
models that learn task-specific shortcuts.

BIOT is excluded: its STFT tokeniser trivially encodes band powers by
construction, making SPA uninformative as a discriminative test.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model_adapter import ModelAdapter

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS.keys())


# ---------------------------------------------------------------------------
# Band power computation
# ---------------------------------------------------------------------------

def compute_band_powers(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Relative band powers for one EEG sample.

    Parameters
    ----------
    x     : [C, T] float — raw EEG
    sfreq : sampling rate in Hz

    Returns
    -------
    [5] float — relative power in [δ, θ, α, β, γ], sums to 1
    """
    from scipy.signal import welch

    f, psd = welch(x, fs=sfreq, nperseg=min(256, x.shape[-1]))
    psd = psd.mean(axis=0)  # average over channels → scalar spectrum

    powers = []
    for lo, hi in BANDS.values():
        mask = (f >= lo) & (f < hi)
        powers.append(float(psd[mask].sum()) if mask.any() else 0.0)

    total = sum(powers) + 1e-12
    return np.array([p / total for p in powers], dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding + band power collection
# ---------------------------------------------------------------------------

def collect_embeddings_and_bands(
    adapter:     ModelAdapter,
    loader:      DataLoader,
    sfreq:       float,
    max_batches: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Run adapter.encode() and compute band powers in one pass.

    The raw EEG is read from batch["data"] ([B, C, T]).

    Returns
    -------
    embeddings  : [N, D] float32, or None on failure
    band_powers : [N, 5] float32, or None on failure
    """
    all_emb: list[np.ndarray] = []
    all_bp:  list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_raw = batch.get("data")
        if x_raw is None:
            continue

        try:
            emb = adapter.encode(batch)      # [B, D]
        except Exception as exc:
            print(f"  [spa batch {batch_idx}] encode failed: {exc}")
            continue

        if not isinstance(emb, np.ndarray):
            emb = emb.detach().float().cpu().numpy()
        all_emb.append(emb)

        x_np = (x_raw.float().cpu().numpy()
                if isinstance(x_raw, torch.Tensor)
                else np.asarray(x_raw, dtype=np.float32))
        for i in range(x_np.shape[0]):
            all_bp.append(compute_band_powers(x_np[i], sfreq))

    if not all_emb or not all_bp:
        return None, None

    emb = np.concatenate(all_emb, axis=0)
    bps = np.stack(all_bp, axis=0)
    n = min(len(emb), len(bps))
    print(f"  spa: {n} samples, emb_dim={emb.shape[1]}")
    return emb[:n], bps[:n]


# ---------------------------------------------------------------------------
# Ridge regression fit
# ---------------------------------------------------------------------------

def compute_spa_score(
    embeddings:  np.ndarray,
    band_powers: np.ndarray,
) -> dict:
    """
    Fit RidgeCV(embedding → band_powers) and return per-band and mean R².

    Parameters
    ----------
    embeddings  : [N, D]
    band_powers : [N, 5]
    """
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    n = embeddings.shape[0]
    if n < 20:
        return {"spa": None, "r2_per_band": None, "n_samples": n,
                "error": "too_few_samples"}

    X = StandardScaler().fit_transform(embeddings)
    Y = band_powers  # already relative (sums to 1)

    alphas = np.logspace(-3, 4, 20)
    ridge  = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X, Y)
    # Out-of-sample predictions via cross_val_predict (same 5 folds, best alpha).
    # RidgeCV.fit() selects alpha_; Ridge(alpha_) re-fits each fold for prediction.
    Y_pred = cross_val_predict(Ridge(alpha=ridge.alpha_), X, Y, cv=5)

    r2_per_band = [
        float(max(0.0, r2_score(Y[:, i], Y_pred[:, i])))
        for i in range(Y.shape[1])
    ]

    return {
        "spa":           float(np.mean(r2_per_band)),
        "r2_per_band":   r2_per_band,
        "band_names":    BAND_NAMES,
        "n_samples":     int(n),
        "embedding_dim": int(embeddings.shape[1]),
        "ridge_alpha":   float(ridge.alpha_),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_spa(
    adapter:     ModelAdapter,
    loader:      DataLoader,
    sfreq:       float,
    max_batches: int | None = None,
) -> dict:
    """
    Full Axis C evaluation for one (model, dataset, seed) cell.

    Freezes the encoder, collects embeddings and band powers,
    fits ridge regression, returns the result dict.
    """
    adapter.freeze_encoder()

    emb, bps = collect_embeddings_and_bands(adapter, loader, sfreq, max_batches)
    if emb is None:
        return {"status": "failed", "error": "no_embeddings_collected"}

    result = compute_spa_score(emb, bps)
    result["status"] = "complete"

    spa = result.get("spa")
    if spa is not None:
        print(f"  SPA={spa:.4f}  (per band: {[f'{v:.3f}' for v in result['r2_per_band']]})")

    return result
