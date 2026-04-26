#!/usr/bin/env python3
"""
UniformEvalBench — Axis B: Channel Dropout Robustness.

Protocol
--------
Train on CLEAN data.  Validate on CLEAN data.  Corrupt TEST split only.

Protocol details:
  - Val-split stays clean: UE_TEST_ONLY_DROPOUT env var gates dropout to
    split == 'test' only. Best-val epoch selection is independent of p.
  - BIOT's data pipeline is routed through AbstractDatasetAdapter.__getitem__()
    so the dropout hook fires correctly.
  - MOMENT excluded (OOM on long-segment datasets; general TS-FM, not EEG-FM)

Metric reporting
----------------
Best-val epoch = argmax(eval balanced_acc, earliest-epoch tiebreak) on CLEAN val.
Last-epoch metrics are also written to the JSON as a secondary record.

Usage
-----
    source ~/arvasu/ndx-pipeline/venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/axis_b_v3_sweep.py [--model M] [--dataset D] [--seed S]

Results → uniformevalbench/experiments/results/axis_b_v3/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_builder import build_run_config, write_config
from run_utils import (
    REPO_ROOT, build_env, spawn_training_run, run_sweep,
    parse_last_metrics, TEST_METRICS_RE, EVAL_METRICS_RE,
)

# ---------------------------------------------------------------------------
# Sweep scope  (MOMENT excluded — OOM on long-segment datasets)
# ---------------------------------------------------------------------------

MODELS    = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve"]
DATASETS  = [
    "bcic_2a", "hmc", "adftd",
    "motor_mv_img", "siena_scalp", "workload",
    "epilepsy_mimickers",   # added in v3; was missing from v2
]
SEEDS     = [42, 123, 7]
DROPOUT_P = [0.1, 0.25, 0.4]   # p=0 (clean baseline) reused from Axis A FBP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RUN_ROOT     = Path("/mnt/eegfmbench/runs/uniformeval/axis_b_v3")
RESULTS_ROOT = REPO_ROOT / "uniformevalbench/experiments/results/axis_b_v3"
CONFIG_ROOT  = REPO_ROOT / "uniformevalbench/experiments/configs/axis_b_v3_generated"
LOG_ROOT     = REPO_ROOT / "uniformevalbench/experiments/logs/axis_b_v3"

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def _tag(p: float) -> str:
    return f"p{int(round(p * 100)):03d}"


def run_one(model: str, dataset: str, seed: int, p: float) -> dict:
    """Execute one (model, dataset, seed, dropout_p) v3 robustness run."""
    exp_name = f"b3_{model}_{dataset}_s{seed}_{_tag(p)}"

    config, _ = build_run_config(model, dataset, seed, mode="lp",
                                  run_root=RUN_ROOT, exp_name=exp_name)
    cfg_path    = CONFIG_ROOT  / f"{exp_name}.yaml"
    log_path    = LOG_ROOT     / f"{exp_name}.log"
    result_path = RESULTS_ROOT / "per_run" / f"{exp_name}.json"

    write_config(config, cfg_path)

    # v3 protocol: UE_TEST_ONLY_DROPOUT=1 now means test split ONLY (not val).
    # Val stays clean → best-val epoch selection is unaffected by p.
    env = build_env(
        run_root=RUN_ROOT,
        extra={
            "UE_CHANNEL_DROPOUT_P":    f"{p:.3f}",
            "UE_CHANNEL_DROPOUT_SEED": str(seed),
            "UE_TEST_ONLY_DROPOUT":    "1",
        },
    )

    status, rc, elapsed = spawn_training_run(exp_name, model, cfg_path, env, log_path)

    stdout = log_path.read_text(errors="ignore")
    test_last = parse_last_metrics(stdout, TEST_METRICS_RE)
    eval_last = parse_last_metrics(stdout, EVAL_METRICS_RE)

    # bestval fields are placeholders; filled inline by parse_log_bestval() below.
    test_bv       = None
    bestval_epoch = None
    last_epoch    = None

    if test_last:
        print(f"  {exp_name}  status={status}  wall={elapsed:.0f}s  "
              f"test_bal_acc_last={test_last['balanced_acc']:.4f}")
    else:
        print(f"  {exp_name}  status={status}  wall={elapsed:.0f}s  no metrics")

    result = {
        "experiment_id":      exp_name,
        "axis":               "B_v3",
        "model":              model,
        "dataset":            dataset,
        "seed":               seed,
        "mode":               "lp",
        "channel_dropout_p":  p,
        "test_only_dropout":  True,
        "val_clean":          True,     # v3 distinguishes from v2
        "status":             status,
        "return_code":        rc,
        "wall_time_sec":      round(elapsed, 1),
        "timestamp":          datetime.now().isoformat(),
        "config_path":        str(cfg_path),
        "log_path":           str(log_path),
        # Primary metric: best-val epoch on clean val
        "test_metrics_bestval": test_bv,
        # Retained for sensitivity / comparison with EEG-FM-Bench last-epoch reporting
        "test_metrics_last":    test_last,
        "eval_metrics_last":    eval_last,
        "bestval_epoch":        bestval_epoch,
        "last_epoch":           last_epoch,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Axis B v3 sweep")
    parser.add_argument("--model",   default=None, help="Run only this model")
    parser.add_argument("--dataset", default=None, help="Run only this dataset")
    parser.add_argument("--seed",    default=None, type=int, help="Run only this seed")
    args = parser.parse_args()

    models   = [args.model]   if args.model   else MODELS
    datasets = [args.dataset] if args.dataset else DATASETS
    seeds    = [args.seed]    if args.seed    else SEEDS

    all_runs = [
        (model, dataset, seed, p)
        for model   in models
        for dataset in datasets
        for seed    in seeds
        for p       in DROPOUT_P
    ]

    n_total = len(MODELS) * len(DATASETS) * len(SEEDS) * len(DROPOUT_P)
    print(f"\nAxis B v3 sweep: {len(all_runs)} runs (full suite = {n_total})")
    print(f"  Models:   {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds:    {seeds}")
    print(f"  Dropout:  {DROPOUT_P} (test-only, val clean)")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    print(f"  Results → {RESULTS_ROOT}")
    print()
    print("  Protocol fix v3:")
    print("    [1] Val split stays clean (best-val epoch independent of p)")
    print("    [2] BIOT dropout hook routed through AbstractDatasetAdapter.__getitem__")
    print()

    def get_exp_name(run):
        model, dataset, seed, p = run
        return f"b3_{model}_{dataset}_s{seed}_{_tag(p)}"

    run_sweep(
        all_runs     = all_runs,
        get_exp_name = get_exp_name,
        results_dir  = RESULTS_ROOT,
        run_fn       = lambda run: run_one(*run),
    )


if __name__ == "__main__":
    main()
