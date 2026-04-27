"""
Shared utilities for all UniformEvalBench sweep scripts.

Covers three concerns:
  1. Environment setup   — build_env(), DATA_ROOT, REPO_ROOT
  2. GPU / subprocess    — count_visible_gpus(), spawn_training_run()
  3. Sweep orchestration — run_sweep() (skip-complete loop + rolling JSON summary)
  4. Metrics parsing     — parse_last_metrics(), TEST_METRICS_RE, EVAL_METRICS_RE
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Canonical repo / data paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("EEGFM_DATA_ROOT", "/mnt/eegfmbench"))

# Venv-pinned torchrun: avoids pyarrow/HF version conflicts when the system
# Python differs from the project venv.  Falls back to bare "torchrun" if the
# venv binary is missing (e.g. CI environments with a different venv layout).
# Override via EEGFM_TORCHRUN env var for machines with a different venv layout.
_VENV_TORCHRUN = Path(os.environ.get(
    "EEGFM_TORCHRUN", "/home/neurodx/arvasu/ndx-pipeline/venv/bin/torchrun"
))
TORCHRUN = str(_VENV_TORCHRUN) if _VENV_TORCHRUN.exists() else "torchrun"
CKPT_ROOT = DATA_ROOT / "FM_checkpoints"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def build_env(run_root: Path, extra: dict | None = None) -> dict:
    """
    Build the environment dict for a training subprocess.

    Starts from the current environment, adds the EEGFM_* path variables that
    the trainer and data loaders read, then merges any caller-supplied extras
    (e.g. UE_CHANNEL_DROPOUT_P for Axis B).
    """
    env = os.environ.copy()
    env.update({
        "EEGFM_PROJECT_ROOT":        str(REPO_ROOT),
        "EEGFM_CONF_ROOT":           str(REPO_ROOT / "assets/conf"),
        "EEGFM_DATABASE_PROC_ROOT":  str(DATA_ROOT / "proc"),
        "EEGFM_DATABASE_CACHE_ROOT": str(DATA_ROOT / "cache"),
        "EEGFM_DATABASE_RAW_ROOT":   str(DATA_ROOT / "raw"),
        "EEGFM_RUN_ROOT":            str(run_root),
        "EEGFM_LOG_ROOT":            str(run_root / "log"),
    })
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def count_visible_gpus() -> int:
    """
    Return how many GPUs are visible via CUDA_VISIBLE_DEVICES.
    Falls back to 1 if the variable is unset or empty.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [x.strip() for x in cvd.split(",") if x.strip()]
    return max(1, len(gpus))


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def spawn_training_run(
    exp_name: str,
    model: str,
    cfg_path: Path,
    env: dict,
    log_path: Path,
    timeout: int = 7200,
) -> tuple[str, int, float]:
    """
    Launch `torchrun baseline_main.py` for a single experiment.

    Uses torchrun with one process per visible GPU (--standalone so no
    rendezvous server is needed). Stdout and stderr both go to log_path.

    Returns
    -------
    status  : 'complete' | 'failed' | 'timeout'
    rc      : process return code (-1 on timeout)
    elapsed : wall-clock seconds
    """
    nproc = count_visible_gpus()
    cmd = [
        TORCHRUN, "--standalone", f"--nproc_per_node={nproc}",
        "main.py",
        f"conf_file={cfg_path}",
        f"model_type={model}",
    ]

    print(f"\n{'='*70}")
    print(f"[{datetime.now().isoformat(timespec='seconds')}] RUNNING: {exp_name}")
    print(f"  gpus={nproc}  cmd: {' '.join(cmd)}")
    print(f"{'='*70}")

    start = time.time()
    try:
        with log_path.open("w") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        status = "complete" if proc.returncode == 0 else "failed"
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        status, rc = "timeout", -1

    return status, rc, time.time() - start


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def run_sweep(
    all_runs: list,
    get_exp_name,
    results_dir: Path,
    run_fn,
) -> None:
    """
    Standard UniformEvalBench sweep loop.

    For each entry in all_runs:
      - Skip if a completed result JSON already exists (resume-safe).
      - Otherwise call run_fn(run) to execute the experiment.
      - Append the result dict to a rolling sweep_summary.json.

    Parameters
    ----------
    all_runs     : list of run descriptors (tuples or any type)
    get_exp_name : callable(run) -> str   — unique name for this run
    results_dir  : directory for per_run/ JSONs and sweep_summary.json
    run_fn       : callable(run) -> dict  — executes one run, returns result dict
    """
    per_run_dir  = results_dir / "per_run"
    summary_path = results_dir / "sweep_summary.json"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    total   = len(all_runs)

    for i, run in enumerate(all_runs, start=1):
        exp_name    = get_exp_name(run)
        result_path = per_run_dir / f"{exp_name}.json"

        if result_path.exists():
            try:
                existing = json.loads(result_path.read_text())
                if existing.get("status") == "complete":
                    print(f"[{i}/{total}] SKIP (complete): {exp_name}")
                    summary.append(existing)
                    summary_path.write_text(json.dumps(summary, indent=2))
                    continue
            except Exception:
                pass  # malformed JSON → rerun

        print(f"\n--- [{i}/{total}] ---")
        result = run_fn(run)
        summary.append(result)
        summary_path.write_text(json.dumps(summary, indent=2))

    n_complete = sum(1 for r in summary if r["status"] == "complete")
    n_fail     = len(summary) - n_complete
    print(f"\n{'='*70}")
    print(f"SWEEP DONE: {n_complete}/{len(summary)} complete, {n_fail} non-complete")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Metrics parsing
# ---------------------------------------------------------------------------

# The trainer emits one log line per epoch per split, e.g.:
#   bcic_2a/test epoch: 29, loss: 0.512, acc: 0.812, balanced_acc: 0.743, auroc: 0.91
# We capture balanced_acc (primary metric) and any trailing fields opportunistically.

TEST_METRICS_RE = re.compile(
    r"(?P<dataset>\w+)/test epoch: (?P<epoch>\d+), "
    r"loss: (?P<loss>[\d.]+), "
    r"acc: (?P<acc>[\d.]+), "
    r"balanced_acc: (?P<bal_acc>[\d.]+)"
)

EVAL_METRICS_RE = re.compile(
    r"(?P<dataset>\w+)/eval epoch: (?P<epoch>\d+), "
    r"loss: (?P<loss>[\d.]+), "
    r"acc: (?P<acc>[\d.]+), "
    r"balanced_acc: (?P<bal_acc>[\d.]+)"
)


def _match_to_dict(m: re.Match, full_text: str) -> dict:
    """Convert a TEST/EVAL_METRICS_RE match to a metrics dict (shared helper)."""
    line_start = full_text.rfind("\n", 0, m.start()) + 1
    line_end   = full_text.find("\n", m.end())
    if line_end == -1:
        line_end = len(full_text)
    full_line = full_text[line_start:line_end]
    def _ex(key: str) -> float | None:
        mm = re.search(rf"{key}: (-?[\d.]+)", full_line)
        return float(mm.group(1)) if mm else None
    return {
        "epoch":        int(m.group("epoch")),
        "loss":         float(m.group("loss")),
        "acc":          float(m.group("acc")),
        "balanced_acc": float(m.group("bal_acc")),
        "cohen_kappa":  _ex("cohen_kappa"),
        "f1":           _ex("f1"),
        "auroc":        _ex("auroc"),
        "auc_pr":       _ex("auc_pr"),
    }


def parse_bestval_metrics(
    stdout_text: str,
) -> tuple[dict | None, dict | None, int | None, int | None]:
    """
    Find the best-val epoch (argmax eval balanced_acc, earliest-epoch tiebreak)
    and return metrics at that epoch.

    Returns (test_bv, eval_bv, bestval_epoch, last_epoch).
    All values are None if the relevant log lines are absent.
    """
    eval_matches = list(EVAL_METRICS_RE.finditer(stdout_text))
    test_matches = list(TEST_METRICS_RE.finditer(stdout_text))

    last_epoch = int(test_matches[-1].group("epoch")) if test_matches else None
    if not eval_matches:
        return None, None, None, last_epoch

    best_epoch, best_ba = None, -1.0
    for m in eval_matches:
        e, ba = int(m.group("epoch")), float(m.group("bal_acc"))
        if ba > best_ba or (ba == best_ba and best_epoch is not None and e < best_epoch):
            best_ba, best_epoch = ba, e

    eval_bv = next(
        (_match_to_dict(m, stdout_text) for m in eval_matches if int(m.group("epoch")) == best_epoch),
        None,
    )
    test_bv = next(
        (_match_to_dict(m, stdout_text) for m in test_matches if int(m.group("epoch")) == best_epoch),
        None,
    )
    return test_bv, eval_bv, best_epoch, last_epoch


def parse_last_metrics(stdout_text: str, regex: re.Pattern) -> dict | None:
    """
    Find the last regex match in stdout_text and return a metrics dict.

    In addition to the captured groups, tries to parse auroc / auc_pr /
    cohen_kappa / f1 from the same log line (binary vs multiclass datasets
    emit different trailing fields).

    Returns None if no match found.
    """
    matches = list(regex.finditer(stdout_text))
    if not matches:
        return None

    m = matches[-1]

    # Grab the full log line that contains the match for opportunistic parsing
    line_start = stdout_text.rfind("\n", 0, m.start()) + 1
    line_end   = stdout_text.find("\n", m.end())
    if line_end == -1:
        line_end = len(stdout_text)
    full_line = stdout_text[line_start:line_end]

    def _extract(key: str) -> float | None:
        mm = re.search(rf"{key}: (-?[\d.]+)", full_line)
        return float(mm.group(1)) if mm else None

    return {
        "epoch":        int(m.group("epoch")),
        "loss":         float(m.group("loss")),
        "acc":          float(m.group("acc")),
        "balanced_acc": float(m.group("bal_acc")),
        "cohen_kappa":  _extract("cohen_kappa"),
        "f1":           _extract("f1"),
        "auroc":        _extract("auroc"),
        "auc_pr":       _extract("auc_pr"),
    }
