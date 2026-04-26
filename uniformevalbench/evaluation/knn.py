"""
Axis A — kNN@20 evaluation (pure, model-agnostic).

Usage
-----
    from uniformevalbench.adapters.eegpt import EEGPTAdapter
    from uniformevalbench.evaluation.knn import run_knn

    adapter = EEGPTAdapter.from_run_json(run_json_path)
    result  = run_knn(adapter, train_loader, test_loader)
    # result["knn_primary"] → balanced accuracy at k=20

Why kNN?
--------
kNN@20 has exactly one parameter (k) and makes no assumptions about the
decision boundary.  It directly tests whether same-class samples cluster in
the frozen embedding space — the same evaluation adopted by DINOv2 (Oquab et
al., 2023).  FBP requires choosing head architecture, optimiser, scheduler,
and training duration, any of which can systematically favour one model's
geometry over another's.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..model_adapter import ModelAdapter

K_DEFAULT = [5, 10, 20, 50]
K_PRIMARY = 20


# ---------------------------------------------------------------------------
# Embedding collection
# ---------------------------------------------------------------------------

def collect_embeddings(
    adapter: ModelAdapter,
    loader: DataLoader,
    split_name: str = "",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Run adapter.encode() over every batch in loader.

    Returns
    -------
    embeddings : [N, D] float32, or None on failure
    labels     : [N] int32, or None on failure
    """
    all_emb: list[np.ndarray] = []
    all_lbl: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader):
        try:
            emb = adapter.encode(batch)          # [B, D]
        except Exception as exc:
            print(f"  [{split_name} batch {batch_idx}] encode failed: {exc}")
            continue

        if not isinstance(emb, np.ndarray):
            emb = emb.detach().float().cpu().numpy()
        all_emb.append(emb)

        lbl = batch.get("label")
        if lbl is None:
            print(f"  [{split_name} batch {batch_idx}] no 'label' key in batch")
            all_emb.pop()
            continue
        if isinstance(lbl, Tensor):
            lbl = lbl.cpu().numpy().astype(np.int32)
        else:
            lbl = np.asarray(lbl, dtype=np.int32)
        all_lbl.append(lbl)

    if not all_emb:
        return None, None

    emb = np.concatenate(all_emb, axis=0)
    lbl = np.concatenate(all_lbl, axis=0)
    n = min(len(emb), len(lbl))
    if split_name:
        print(f"  {split_name}: {n} samples, emb_dim={emb.shape[1]}")
    return emb[:n], lbl[:n]


# ---------------------------------------------------------------------------
# Cosine kNN classifier
# ---------------------------------------------------------------------------

def cosine_knn(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    test_emb:  np.ndarray,
    test_lbl:  np.ndarray,
    k_values:  list[int] = K_DEFAULT,
) -> dict:
    """
    L2-normalise then cosine kNN for each k in k_values.

    Returns a dict with:
        knn_per_k      : {str(k): balanced_accuracy}
        knn_primary    : balanced accuracy at K_PRIMARY (k=20)
        k_sensitivity  : max − min balanced acc across valid k values
        n_train / n_test / embedding_dim
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score

    def _l2(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-12)

    X_train = _l2(train_emb.astype(np.float32))
    X_test  = _l2(test_emb.astype(np.float32))

    per_k: dict[int, float | None] = {}
    for k in k_values:
        if k >= len(X_train):
            per_k[k] = None
            continue
        clf = KNeighborsClassifier(
            n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1
        )
        clf.fit(X_train, train_lbl)
        preds = clf.predict(X_test)
        per_k[k] = float(balanced_accuracy_score(test_lbl, preds))

    valid = [v for v in per_k.values() if v is not None]
    return {
        "knn_per_k":     {str(k): v for k, v in per_k.items()},
        "knn_primary":   per_k.get(K_PRIMARY),
        "k_primary":     K_PRIMARY,
        "k_sensitivity": float(max(valid) - min(valid)) if len(valid) > 1 else 0.0,
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "embedding_dim": int(train_emb.shape[1]),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_knn(
    adapter:      ModelAdapter,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    k_values:     list[int] = K_DEFAULT,
) -> dict:
    """
    Full Axis A evaluation for one (model, dataset, seed) cell.

    Freezes the encoder, collects embeddings from both splits,
    runs cosine kNN, returns the result dict.
    """
    adapter.freeze_encoder()

    print("  collecting train embeddings...", flush=True)
    train_emb, train_lbl = collect_embeddings(adapter, train_loader, "train")
    if train_emb is None:
        return {"status": "failed", "error": "no_train_embeddings"}

    print("  collecting test embeddings...", flush=True)
    test_emb, test_lbl = collect_embeddings(adapter, test_loader, "test")
    if test_emb is None:
        return {"status": "failed", "error": "no_test_embeddings"}

    print(f"  running cosine kNN at k={k_values}...", flush=True)
    result = cosine_knn(train_emb, train_lbl, test_emb, test_lbl, k_values)

    primary = result.get("knn_primary")
    sens    = result.get("k_sensitivity", 0.0)
    if primary is not None:
        print(f"  kNN@{K_PRIMARY}={primary:.4f}  k-sensitivity={sens:.4f}")

    result["status"] = "complete"
    return result
