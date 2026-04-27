#!/usr/bin/env python3
"""
UniformEvalBench — Pilot sweep (Axis A FBP + FT).

Runs linear-probe (lp) and fine-tune (ft) training for every
(model, dataset, seed) cell and stores per-run JSONs.

Usage
-----
    source ~/arvasu/ndx-pipeline/venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/pilot_sweep.py [--model M] [--dataset D] [--seed S] [--mode lp|ft]

Results → uniformevalbench/experiments/results/pilot/per_run/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_builder import build_run_config, write_config, PILOT_RUN_ROOT
from run_utils import (
    REPO_ROOT, build_env, spawn_training_run, run_sweep,
    parse_last_metrics, parse_bestval_metrics, TEST_METRICS_RE, EVAL_METRICS_RE,
)

# ---------------------------------------------------------------------------
# Sweep scope
# ---------------------------------------------------------------------------

MODELS = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve", "moment"]
DATASETS = [
    "bcic_2a", "hmc", "adftd",
    "motor_mv_img", "siena_scalp", "workload",
    "epilepsy_mimickers",
]
SEEDS = [42, 123, 7]
MODES = ["lp", "ft"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = REPO_ROOT / "uniformevalbench/experiments/results/pilot"
CONFIG_ROOT  = REPO_ROOT / "uniformevalbench/experiments/configs/pilot_generated"
LOG_ROOT     = REPO_ROOT / "uniformevalbench/experiments/logs/pilot"

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def run_one(model: str, dataset: str, seed: int, mode: str) -> dict:
    """Execute one (model, dataset, seed, mode) pilot run."""
    exp_name = f"{mode}_{model}_{dataset}_s{seed}"

    config, _ = build_run_config(model, dataset, seed, mode=mode,
                                  run_root=PILOT_RUN_ROOT, exp_name=exp_name)
    cfg_path    = CONFIG_ROOT  / f"{exp_name}.yaml"
    log_path    = LOG_ROOT     / f"{exp_name}.log"
    result_path = RESULTS_ROOT / "per_run" / f"{exp_name}.json"

    write_config(config, cfg_path)
    env = build_env(run_root=PILOT_RUN_ROOT)

    status, rc, elapsed = spawn_training_run(exp_name, model, cfg_path, env, log_path)

    stdout = log_path.read_text(errors="ignore")
    test_last = parse_last_metrics(stdout, TEST_METRICS_RE)
    eval_last = parse_last_metrics(stdout, EVAL_METRICS_RE)
    test_bv, eval_bv, bestval_epoch, last_epoch = parse_bestval_metrics(stdout)

    bv_str = f"  bv_ba={test_bv['balanced_acc']:.4f}" if test_bv else ""
    if test_last:
        print(f"  {exp_name}  status={status}  wall={elapsed:.0f}s  "
              f"test_ba_last={test_last['balanced_acc']:.4f}{bv_str}")
    else:
        print(f"  {exp_name}  status={status}  wall={elapsed:.0f}s  no metrics{bv_str}")

    result = {
        "experiment_id":        exp_name,
        "axis":                 "pilot",
        "model":                model,
        "dataset":              dataset,
        "seed":                 seed,
        "mode":                 mode,
        "status":               status,
        "return_code":          rc,
        "wall_time_sec":        round(elapsed, 1),
        "timestamp":            datetime.now().isoformat(),
        "config_path":          str(cfg_path),
        "log_path":             str(log_path),
        # Primary metric: best-val epoch on clean val
        "test_metrics_bestval": test_bv,
        "eval_metrics_bestval": eval_bv,
        # Legacy / sensitivity: last epoch
        "test_metrics_last":    test_last,
        "eval_metrics_last":    eval_last,
        # For compatibility with older aggregation scripts
        "test_metrics":         test_last,
        "eval_metrics":         eval_last,
        "bestval_epoch":        bestval_epoch,
        "last_epoch":           last_epoch,
        "n_epochs_logged":      last_epoch,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pilot sweep (FBP + FT)")
    parser.add_argument("--model",   default=None, help="Run only this model")
    parser.add_argument("--dataset", default=None, help="Run only this dataset")
    parser.add_argument("--seed",    default=None, type=int, help="Run only this seed")
    parser.add_argument("--mode",    default=None, choices=["lp", "ft"], help="Run only this mode")
    args = parser.parse_args()

    models   = [args.model]   if args.model   else MODELS
    datasets = [args.dataset] if args.dataset else DATASETS
    seeds    = [args.seed]    if args.seed    else SEEDS
    modes    = [args.mode]    if args.mode    else MODES

    all_runs = [
        (model, dataset, seed, mode)
        for mode    in modes
        for model   in models
        for dataset in datasets
        for seed    in seeds
    ]

    n_total = len(MODELS) * len(DATASETS) * len(SEEDS) * len(MODES)
    print(f"\nPilot sweep: {len(all_runs)} runs (full suite = {n_total})")
    print(f"  Models:   {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Seeds:    {seeds}")
    print(f"  Modes:    {modes}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    print(f"  Results → {RESULTS_ROOT}")
    print()

    def get_exp_name(run):
        model, dataset, seed, mode = run
        return f"{mode}_{model}_{dataset}_s{seed}"

    run_sweep(
        all_runs     = all_runs,
        get_exp_name = get_exp_name,
        results_dir  = RESULTS_ROOT,
        run_fn       = lambda run: run_one(*run),
    )


if __name__ == "__main__":
    main()
