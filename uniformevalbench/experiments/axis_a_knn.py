#!/usr/bin/env python3
"""
UniformEvalBench — Axis A: kNN@20 Evaluation.

kNN is the primary metric for frozen encoder quality in this benchmark.

Why kNN, not FBP
----------------
FBP (frozen-backbone probe) still requires training a classifier head, which
introduces at least 8 hyperparameters: head architecture (hidden dims, dropout),
optimiser (LR, weight decay), scheduler (warmup, cosine/onecycle), training
duration (epochs), and label smoothing.  Any of these can systematically favour
one model's representation geometry over another's.

kNN has exactly one parameter: k (a count, not a gradient).  It makes no
assumptions about the decision boundary.  It directly asks: "do same-class
samples cluster together in the frozen embedding space?"  That is the purest
measure of representation quality, and is the evaluation adopted by DINOv2
(Oquab et al., 2023) for the same reason.

Method
------
1.  Load the FBP checkpoint from pilot_sweep (lp mode).
2.  Run the frozen encoder over the train split → collect embeddings + labels.
3.  Run the frozen encoder over the test split  → collect embeddings + labels.
4.  L2-normalise all embeddings (enables cosine similarity via dot product).
5.  For k ∈ {5, 10, 20, 50}: cosine kNN + majority vote → balanced accuracy.
6.  Primary metric: k=20.  Sensitivity across k is also reported.

Usage
-----
    source ~/arvasu/ndx-pipeline/venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/axis_a_knn.py
    python uniformevalbench/experiments/axis_a_knn.py --model eegpt --dataset bcic_2a
    python uniformevalbench/experiments/axis_a_knn.py --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from uniformevalbench.adapters import build_adapter, ADAPTER_REGISTRY
from uniformevalbench.adapters._viz_bridge import _find_ckpt
from uniformevalbench.evaluation.knn import run_knn

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

RESULTS_ROOT = REPO_ROOT / "uniformevalbench" / "experiments" / "results"
PILOT_DIR    = RESULTS_ROOT / "pilot" / "per_run"
OUTPUT_DIR   = RESULTS_ROOT / "axis_a" / "knn" / "per_run"

MODELS   = list(ADAPTER_REGISTRY)   # eegpt, labram, cbramod, biot, csbrain, reve
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img",
            "siena_scalp", "workload", "epilepsy_mimickers"]
SEEDS    = [42, 123, 7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_failed(path: Path, model, dataset, seed, source, error):
    path.write_text(json.dumps({
        "experiment_id": f"knn_{model}_{dataset}_s{seed}",
        "model": model, "dataset": dataset, "seed": seed,
        "status": "failed", "error": error, "source_run": source,
    }, indent=2))


# ---------------------------------------------------------------------------
# Per-run processor
# ---------------------------------------------------------------------------

def _process_run(run_json: Path, force: bool = False) -> str:
    j = json.loads(run_json.read_text())

    if j.get("status") != "complete":
        return "skip-incomplete"
    if j.get("mode") != "lp":
        return "skip-not-lp"

    model   = j["model"]
    dataset = j["dataset"]
    seed    = j["seed"]

    if model not in MODELS or dataset not in DATASETS or seed not in SEEDS:
        return "skip-outofscope"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"knn_{model}_{dataset}_s{seed}.json"

    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        if existing.get("status") == "complete":
            return "skip-already"

    ckpt = _find_ckpt(j["log_path"], model, dataset)
    if ckpt is None or not ckpt.exists():
        _write_failed(out_path, model, dataset, seed, run_json.name,
                      f"checkpoint not found (expected {ckpt})")
        return "no-ckpt"

    print(f"\n>>> kNN: {model}/{dataset}/s{seed}")
    print(f"    ckpt: {ckpt}")

    adapter = None
    try:
        adapter = build_adapter(model, run_json)

        train_loader = adapter.create_dataloader(dataset, "finetune", "train")
        test_loader  = adapter.create_dataloader(dataset, "finetune", "test")

        knn_result = run_knn(adapter, train_loader, test_loader)

        primary = knn_result.get("knn_primary")
        if primary is not None:
            print(f"    kNN@20={primary:.4f}")

        out = {
            "experiment_id": f"knn_{model}_{dataset}_s{seed}",
            "model": model, "dataset": dataset, "seed": seed,
            **knn_result,
            "source_run": run_json.name,
        }
        out_path.write_text(json.dumps(out, indent=2))
        return "done"

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write_failed(out_path, model, dataset, seed, run_json.name, str(exc))
        return "failed"

    finally:
        del adapter
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main — one subprocess per model for GPU memory isolation
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default=None)
    ap.add_argument("--dataset",    default=None)
    ap.add_argument("--seed",       type=int, default=None)
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--force",      action="store_true")
    ap.add_argument("--in-process", action="store_true",
                    help="Run in current process (used by per-model subprocess)")
    args = ap.parse_args()

    run_jsons = sorted(PILOT_DIR.glob("lp_*.json"))

    if args.model:
        run_jsons = [p for p in run_jsons if f"_{args.model}_" in p.name]
    if args.dataset:
        run_jsons = [p for p in run_jsons if f"_{args.dataset}_" in p.name]
    if args.seed is not None:
        run_jsons = [p for p in run_jsons if p.name.endswith(f"_s{args.seed}.json")]

    run_jsons = [p for p in run_jsons if any(f"_{m}_" in p.name for m in MODELS)]
    print(f"Found {len(run_jsons)} pilot LP runs to evaluate.")

    if args.dry_run:
        for p in run_jsons:
            j = json.loads(p.read_text())
            if j.get("status") != "complete" or j.get("mode") != "lp":
                continue
            ckpt = _find_ckpt(j["log_path"], j["model"], j["dataset"])
            status = "OK" if (ckpt and ckpt.exists()) else "NO_CKPT"
            print(f"  [{status}] {p.name}  ckpt={ckpt}")
        return

    if args.in_process:
        stats: dict[str, int] = {}
        for p in run_jsons:
            r = _process_run(p, args.force)
            stats[r] = stats.get(r, 0) + 1
        print("\n=== kNN in-process complete ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
        return

    # Default: one subprocess per model for GPU memory isolation
    import os
    models_seen: set[str] = set()
    models_to_run: list[str] = []
    for p in run_jsons:
        j = json.loads(p.read_text())
        if j.get("status") == "complete" and j.get("mode") == "lp":
            m = j["model"]
            if m in MODELS and m not in models_seen:
                models_seen.add(m)
                models_to_run.append(m)

    for model in models_to_run:
        print(f"\n{'='*60}\n=== subprocess: model={model} ===\n{'='*60}")
        cmd = [sys.executable, __file__, "--model", model, "--in-process"]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.force:
            cmd.append("--force")
        proc = subprocess.run(cmd, env=os.environ.copy())
        if proc.returncode != 0:
            print(f"  [WARNING] subprocess for {model} exited {proc.returncode}")

    print("\n=== All kNN subprocesses complete ===")


if __name__ == "__main__":
    main()
