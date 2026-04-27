#!/usr/bin/env python3
"""
UniformEvalBench — Axis C: Spectral Prior Alignment (SPA).

SPA tests whether frozen encoder representations track physiologically
meaningful spectral features, independently of classification accuracy.

A model scores high on SPA when its embeddings encode EEG frequency-band
information (δ, θ, α, β, γ) — bands that define clinically recognised
brain states — even when trained purely for downstream classification.
Models that learn clinically meaningful spectral patterns should score higher
than models that learn task-specific shortcuts.

Method
------
1.  Load the FBP checkpoint from pilot_sweep (lp mode).
2.  Collect (embedding, raw_eeg) pairs from the frozen encoder on test data.
3.  Compute relative band powers per sample via Welch PSD.
4.  Fit RidgeCV(embedding → band_powers) with 5-fold CV.
5.  Report mean R² across the 5 bands as the SPA score.

BIOT exclusion
--------------
BIOT uses an STFT tokeniser that trivially encodes band powers by construction.
Including it would inflate SPA without measuring anything meaningful about
representation quality.  BIOT is excluded from Axis C.

Usage
-----
    source ~/arvasu/ndx-pipeline/venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/axis_c_spa.py
    python uniformevalbench/experiments/axis_c_spa.py --model eegpt --dataset bcic_2a
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from uniformevalbench.adapters import build_adapter, ADAPTER_REGISTRY
from uniformevalbench.adapters._viz_bridge import _find_ckpt
from uniformevalbench.evaluation.spa import run_spa

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

RESULTS_ROOT = REPO_ROOT / "uniformevalbench" / "experiments" / "results"
PILOT_DIR    = RESULTS_ROOT / "pilot" / "per_run"
OUTPUT_DIR   = RESULTS_ROOT / "axis_c" / "spa" / "per_run"

# BIOT excluded: STFT tokeniser trivially encodes band powers
SPA_EXCLUDED = {"biot"}
MODELS       = [m for m in ADAPTER_REGISTRY if m not in SPA_EXCLUDED]

DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img",
            "siena_scalp", "workload", "epilepsy_mimickers"]
SEEDS    = [42, 123, 7]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_failed(path, model, dataset, seed, source, error):
    path.write_text(json.dumps({
        "experiment_id": f"spa_{model}_{dataset}_s{seed}",
        "model": model, "dataset": dataset, "seed": seed,
        "status": "failed", "error": error, "source_run": source,
    }, indent=2))


# ---------------------------------------------------------------------------
# Per-run processor
# ---------------------------------------------------------------------------

def _process_run(run_json: Path, max_batches: int, force: bool) -> str:
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
    out_path = OUTPUT_DIR / f"spa_{model}_{dataset}_s{seed}.json"

    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        if existing.get("status") == "complete":
            return "skip-already"

    ckpt = _find_ckpt(j["log_path"], model, dataset)
    if ckpt is None or not ckpt.exists():
        _write_failed(out_path, model, dataset, seed, run_json.name,
                      f"checkpoint not found (expected {ckpt})")
        return "no-ckpt"

    print(f"\n>>> SPA: {model}/{dataset}/s{seed}")
    print(f"    ckpt: {ckpt}")

    adapter = None
    try:
        adapter = build_adapter(model, run_json, batch_size=16)

        # sfreq from config; fall back to 256 Hz
        from omegaconf import OmegaConf
        file_cfg = OmegaConf.load(j["config_path"])
        sfreq = float(file_cfg.get("data", {}).get("fs", 256))
        print(f"    sfreq={sfreq} Hz")

        test_loader = adapter.create_dataloader(dataset, "finetune", "test")

        spa_result = run_spa(adapter, test_loader, sfreq,
                             max_batches=max_batches)

        out = {
            "experiment_id": f"spa_{model}_{dataset}_s{seed}",
            "model":   model,
            "dataset": dataset,
            "seed":    seed,
            **spa_result,
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
    ap.add_argument("--model",       default=None)
    ap.add_argument("--dataset",     default=None)
    ap.add_argument("--seed",        type=int, default=None)
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--force",       action="store_true")
    ap.add_argument("--max-batches", type=int, default=50,
                    help="Max test batches per run (default 50, ~800 samples at bs=16)")
    ap.add_argument("--in-process",  action="store_true",
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
    print(f"Found {len(run_jsons)} pilot LP runs to evaluate (BIOT excluded).")

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
            r = _process_run(p, args.max_batches, args.force)
            stats[r] = stats.get(r, 0) + 1
        print("\n=== Axis C SPA in-process complete ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
        return

    # Default: one subprocess per model for GPU memory isolation
    models_to_run: list[str] = []
    seen: set[str] = set()
    for p in run_jsons:
        j = json.loads(p.read_text())
        if j.get("status") == "complete" and j.get("mode") == "lp":
            m = j["model"]
            if m in MODELS and m not in seen:
                seen.add(m)
                models_to_run.append(m)

    for model in models_to_run:
        print(f"\n{'='*60}\n=== subprocess: model={model} ===\n{'='*60}")
        cmd = [sys.executable, __file__,
               "--model", model,
               "--max-batches", str(args.max_batches),
               "--in-process"]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.force:
            cmd.append("--force")
        proc = subprocess.run(cmd, env=os.environ.copy())
        if proc.returncode != 0:
            print(f"  [WARNING] subprocess for {model} exited {proc.returncode}")

    print("\n=== All SPA subprocesses complete ===")


if __name__ == "__main__":
    main()
