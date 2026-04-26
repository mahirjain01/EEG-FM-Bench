#!/usr/bin/env python3
"""
UniformEvalBench — Axis A pilot sweep (FBP + FT).

Runs every combination of (model × dataset × seed × mode) and writes one
JSON result per run to results/pilot/per_run/.  Already-completed runs are
skipped automatically, so the script is safe to re-run after interruption.

Modes
-----
  lp  — linear probe: encoder is frozen, only the classifier head is trained
  ft  — full fine-tuning: encoder and head are trained jointly

Usage
-----
    source ~/arvasu/ndx-pipeline/venv/bin/activate
    export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
    cd /home/neurodx/mahir/EVAL_PAPER/EEG-FM-Bench
    python uniformevalbench/experiments/pilot_sweep.py

Results → uniformevalbench/experiments/results/pilot/
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_builder import build_run_config, write_config, PILOT_RUN_ROOT
from run_utils import (
    REPO_ROOT, build_env, spawn_training_run, run_sweep,
    parse_last_metrics, TEST_METRICS_RE, EVAL_METRICS_RE,
)

# ---------------------------------------------------------------------------
# Sweep scope
# ---------------------------------------------------------------------------

MODELS   = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve", "moment"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img",
            "siena_scalp", "workload", "epilepsy_mimickers"]
SEEDS    = [42, 123, 7]
MODES    = ["lp", "ft"]   # lp = frozen probe, ft = full fine-tuning

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = REPO_ROOT / "uniformevalbench/experiments/results/pilot"
CONFIG_ROOT  = REPO_ROOT / "uniformevalbench/experiments/configs/pilot_generated"
LOG_ROOT     = REPO_ROOT / "uniformevalbench/experiments/logs"

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def run_one(model: str, dataset: str, seed: int, mode: str) -> dict:
    """
    Execute one (model, dataset, seed, mode) experiment.

    Generates the YAML config, spawns torchrun, parses the last test metrics
    from stdout, and writes a result JSON. Returns the result dict.
    """
    config, exp_name = build_run_config(model, dataset, seed, mode, run_root=PILOT_RUN_ROOT)

    cfg_path = CONFIG_ROOT / f"{exp_name}.yaml"
    log_path = LOG_ROOT / f"{exp_name}.log"
    result_path = RESULTS_ROOT / "per_run" / f"{exp_name}.json"

    write_config(config, cfg_path)

    env = build_env(run_root=PILOT_RUN_ROOT)
    status, rc, elapsed = spawn_training_run(exp_name, model, cfg_path, env, log_path)

    stdout = log_path.read_text(errors="ignore")
    test_metrics = parse_last_metrics(stdout, TEST_METRICS_RE)
    eval_metrics = parse_last_metrics(stdout, EVAL_METRICS_RE)

    if test_metrics:
        print(f"  STATUS={status}  wall={elapsed:.0f}s  test_bal_acc={test_metrics['balanced_acc']:.4f}")
    else:
        print(f"  STATUS={status}  wall={elapsed:.0f}s  no metrics found (see {log_path})")

    result = {
        "experiment_id": exp_name,
        "model":         model,
        "dataset":       dataset,
        "seed":          seed,
        "mode":          mode,
        "status":        status,
        "return_code":   rc,
        "wall_time_sec": round(elapsed, 1),
        "timestamp":     datetime.now().isoformat(),
        "config_path":   str(cfg_path),
        "log_path":      str(log_path),
        "test_metrics":  test_metrics,
        "eval_metrics":  eval_metrics,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    all_runs = [
        (model, dataset, seed, mode)
        for model   in MODELS
        for dataset in DATASETS
        for seed    in SEEDS
        for mode    in MODES
    ]

    print(f"\nPilot sweep: {len(all_runs)} runs")
    print(f"  Models:   {MODELS}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  Modes:    {MODES}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    print(f"  Results → {RESULTS_ROOT}")
    print()

    def get_exp_name(run):
        model, dataset, seed, mode = run
        return f"{mode}_{model}_{dataset}_s{seed}"

    run_sweep(
        all_runs    = all_runs,
        get_exp_name = get_exp_name,
        results_dir  = RESULTS_ROOT,
        run_fn       = lambda run: run_one(*run),
    )


if __name__ == "__main__":
    main()
