#!/usr/bin/env python3
"""
Axis C — Neurophysiological Alignment Score (NAS) sweep.

For each completed pilot LP (frozen-backbone probe) run, load the trained
checkpoint, run Integrated Gradients on the test split with three baselines
(zero, mean, phase_shuffled), and compute:

    NAS(m, t) = 1 - JSD_distance(q_{m,t}, p_t)

where q_{m,t} is the mean |IG| per channel (normalized to a distribution)
and p_t is the spatial prior for task t (from spatial_priors.py).

Primary NAS uses the phase-shuffled baseline (most principled for EEG).
All three baselines are stored for sensitivity analysis (see §9.2).

Usage:
    python axis_c_nas.py [--model MODEL] [--dataset DS] [--seed SEED]
                         [--dry-run] [--force] [--n-steps N] [--max-batches M]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from captum.attr import IntegratedGradients
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR))

from uniformevalbench.utils import ElectrodeSet
from uniformevalbench.adapters._viz_bridge import _build_trainer, _get_model

from spatial_priors import get_prior

RESULTS_ROOT = REPO_ROOT / "uniformevalbench" / "experiments" / "results"
PILOT_DIR    = RESULTS_ROOT / "pilot" / "per_run"
OUTPUT_DIR   = RESULTS_ROOT / "axis_c" / "per_run"

MODELS   = ["eegpt", "labram", "cbramod", "biot", "csbrain", "reve"]
DATASETS = ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload"]
SEEDS    = [42, 123, 7]
BASELINES = ["zero", "mean", "phase_shuffled"]

_electrode_set = ElectrodeSet()


def _ensure_dist():
    """Init a single-process gloo group on a free port (idempotent)."""
    import torch.distributed as _dist
    if _dist.is_initialized():
        return
    import socket, os as _os
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        free_port = _s.getsockname()[1]
    _os.environ["MASTER_ADDR"] = "127.0.0.1"
    _os.environ["MASTER_PORT"] = str(free_port)
    _dist.init_process_group(backend="gloo", rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ckpt(log_path: str, model: str, dataset: str) -> Path | None:
    text = Path(log_path).read_text(errors="ignore")
    m = re.search(r"checkpoint dir: (/[^\s,]+)", text)
    if not m:
        return None
    ckpt_dir = Path(m.group(1))
    return ckpt_dir / "seperated" / dataset / f"{model}_{dataset}_last.pt"


_LONG_CONTEXT_DATASETS = {"motor_mv_img", "siena_scalp"}   # 64ch × 800t or 28ch × 6000t



def _phase_shuffle(x: torch.Tensor) -> torch.Tensor:
    """Preserve per-channel magnitude spectrum; randomize phase. Shape [B, C, T]."""
    X = torch.fft.rfft(x.float(), dim=-1)
    rand_phase = torch.rand_like(X.real) * (2.0 * torch.pi)
    X_shuf = X.abs() * torch.polar(torch.ones_like(X.abs()), rand_phase)
    return torch.fft.irfft(X_shuf, n=x.shape[-1], dim=-1).to(dtype=x.dtype)


def _run_ig_baseline(
    model: torch.nn.Module,
    device,
    loader,
    baseline_type: str,
    n_steps: int,
    max_batches: int,
) -> tuple[np.ndarray | None, list[str] | None]:
    """
    Run IG for one baseline type over the test loader.

    Returns
    -------
    mean_channel_attr : [C] float  — mean |IG| per channel, unnormalized
    ch_names          : list[str]  — channel names (same order as C)
    """
    model.eval()

    all_attrs: list[np.ndarray] = []
    ch_names_out: list[str] | None = None
    current_batch: dict = {}   # mutable reference for forward closure

    def _forward(input_tensor: torch.Tensor) -> torch.Tensor:
        # IG expands batch from B to B×n_steps; expand all other batch tensors too
        ig_batch = input_tensor.shape[0]
        b = {}
        for k, v in current_batch.items():
            if not isinstance(v, torch.Tensor):
                b[k] = v
                continue
            tv = v.to(device)
            if tv.dim() > 0 and tv.shape[0] != ig_batch and ig_batch % tv.shape[0] == 0:
                reps = ig_batch // tv.shape[0]
                tv = tv.repeat(*([reps] + [1] * (tv.dim() - 1)))
            b[k] = tv
        b["data"] = input_tensor
        return model(b)

    ig = IntegratedGradients(_forward)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        x = batch["data"]                       # [B, C, T]
        labels = batch.get("label", batch.get("labels"))
        if labels is None:
            continue
        targets = labels.long().tolist()

        # Baseline tensor
        if baseline_type == "zero":
            bl = torch.zeros_like(x)
        elif baseline_type == "mean":
            bl = x.mean(dim=0, keepdim=True).expand_as(x).clone()
        elif baseline_type == "phase_shuffled":
            bl = _phase_shuffle(x)
        else:
            raise ValueError(baseline_type)

        # Update closure reference BEFORE calling ig.attribute
        current_batch.update(batch)

        try:
            with torch.enable_grad():
                attr = ig.attribute(x, baselines=bl, target=targets, n_steps=n_steps,
                                    internal_batch_size=x.shape[0])
            # attr: [B, C, T] — sum |attr| over time → [B, C]
            ch_attr = attr.abs().sum(dim=-1).detach().cpu().numpy()  # [B, C]
            del attr, bl  # release GPU memory immediately
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            all_attrs.append(ch_attr)

            if ch_names_out is None and "chs" in batch:
                chs_tensor = batch["chs"]
                # batch["chs"] is [B, C] indices into ElectrodeSet.Electrodes
                ch_names_out = _electrode_set.get_electrodes_name(
                    chs_tensor[0].long().tolist()
                )
        except Exception as exc:
            print(f"    [batch {batch_idx}] IG failed: {exc}")
            continue

    if not all_attrs:
        return None, None

    stacked = np.concatenate(all_attrs, axis=0)  # [N, C]
    return stacked.mean(axis=0), ch_names_out    # [C]


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _process_run(run_json: Path, n_steps: int, max_batches: int, force: bool) -> str:
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
    out_path = OUTPUT_DIR / f"nas_{model}_{dataset}_s{seed}.json"
    if out_path.exists() and not force:
        existing = json.loads(out_path.read_text())
        if existing.get("status") == "complete":
            return "skip-already"

    ckpt_path = _find_ckpt(j["log_path"], model, dataset)
    if ckpt_path is None or not ckpt_path.exists():
        _write_failed(out_path, model, dataset, seed, run_json.name,
                      f"checkpoint not found (expected {ckpt_path})")
        return "no-ckpt"

    print(f"\n>>> {model}/{dataset}/s{seed}")
    print(f"    ckpt: {ckpt_path}")

    try:
        import datasets as hf_datasets

        batch_sz = 4 if dataset in _LONG_CONTEXT_DATASETS else 16
        trainer = _build_trainer(
            model_type  = model,
            ckpt_path   = str(ckpt_path),
            config_path = j["config_path"],
            dataset     = dataset,
            seed        = seed,
            batch_size  = batch_sz,
        )
        viz_model = _get_model(trainer)
        loader, _ = trainer.create_single_dataloader(dataset, "finetune", hf_datasets.Split.TEST)

        nas_by_baseline: dict[str, dict | None] = {}
        ch_names_final: list[str] | None = None

        for bl in BASELINES:
            print(f"    baseline={bl} ...", end=" ", flush=True)
            mean_attr, ch_names = _run_ig_baseline(viz_model, trainer.device, loader, bl, n_steps, max_batches)

            if mean_attr is None:
                print("no attributions")
                nas_by_baseline[bl] = None
                continue

            if ch_names_final is None:
                ch_names_final = ch_names

            total = mean_attr.sum()
            if total <= 0:
                print("zero sum")
                nas_by_baseline[bl] = None
                continue

            q = mean_attr / total                           # empirical distribution
            p = get_prior(dataset, ch_names or [])          # spatial prior

            if p is None or len(p) != len(q):
                print("prior mismatch")
                nas_by_baseline[bl] = None
                continue

            js_dist = float(jensenshannon(q, p, base=2))   # sqrt(JSD), in [0, 1]
            nas     = float(1.0 - js_dist)
            print(f"NAS={nas:.4f}  JS_dist={js_dist:.4f}")

            nas_by_baseline[bl] = {
                "nas":                nas,
                "js_distance":        js_dist,
                "n_channels":         int(len(q)),
                "attribution_entropy": float(-np.sum(q * np.log(q + 1e-12))),
            }

        primary = (nas_by_baseline.get("phase_shuffled") or {}).get("nas")

        out = {
            "experiment_id": f"nas_{model}_{dataset}_s{seed}",
            "model":          model,
            "dataset":        dataset,
            "seed":           seed,
            "status":         "complete",
            "nas_primary":    primary,
            "nas_by_baseline": nas_by_baseline,
            "channel_names":  ch_names_final,
            "n_steps_ig":     n_steps,
            "max_batches":    max_batches,
            "source_run":     run_json.name,
        }
        out_path.write_text(json.dumps(out, indent=2))
        return "done"

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write_failed(out_path, model, dataset, seed, run_json.name, str(exc))
        return "failed"

    finally:
        # Explicitly free GPU memory after each run so models don't accumulate
        try:
            del trainer, viz_model
        except Exception:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass  # CUDA context may be poisoned (cudaErrorContained); ignore


def _write_failed(path, model, dataset, seed, source, error):
    path.write_text(json.dumps({
        "experiment_id": f"nas_{model}_{dataset}_s{seed}",
        "model":   model, "dataset": dataset, "seed": seed,
        "status":  "failed", "error": error, "source_run": source,
    }, indent=2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default=None)
    ap.add_argument("--dataset",    default=None)
    ap.add_argument("--seed",       type=int, default=None)
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--force",      action="store_true",
                    help="Recompute even if output already complete")
    ap.add_argument("--n-steps",    type=int, default=50,
                    help="IG integration steps (default 50)")
    ap.add_argument("--max-batches",type=int, default=20,
                    help="Max test batches per run (default 20)")
    ap.add_argument("--subprocess", action="store_true",
                    help="Internal flag: run in single-run subprocess mode")
    args = ap.parse_args()

    run_jsons = sorted(PILOT_DIR.glob("lp_*.json"))

    if args.model:
        run_jsons = [p for p in run_jsons if f"_{args.model}_" in p.name]
    if args.dataset:
        run_jsons = [p for p in run_jsons if f"_{args.dataset}_" in p.name]
    if args.seed is not None:
        run_jsons = [p for p in run_jsons if f"_s{args.seed}.json" == p.name[-len(f"_s{args.seed}.json"):]]

    print(f"Found {len(run_jsons)} pilot LP runs to process.")

    if args.dry_run:
        for p in run_jsons:
            j = json.loads(p.read_text())
            if j.get("status") != "complete" or j.get("mode") != "lp":
                continue
            ckpt = _find_ckpt(j["log_path"], j["model"], j["dataset"])
            status = "OK" if (ckpt and ckpt.exists()) else "NO_CKPT"
            print(f"  [{status}] {p.name}  ckpt={ckpt}")
        return

    # --subprocess: run all filtered runs in-process (used by subprocess dispatcher)
    if args.subprocess:
        stats: dict[str, int] = {}
        for p in run_jsons:
            result = _process_run(p, args.n_steps, args.max_batches, args.force)
            stats[result] = stats.get(result, 0) + 1
        print("\n=== Axis C NAS sweep complete ===")
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")
        return

    # Default: dispatch one subprocess per model to guarantee GPU memory isolation.
    # Each subprocess owns a fresh CUDA context and exits cleanly after its model.
    import subprocess
    import os as _os

    models_to_run: list[str] = []
    seen: set[str] = set()
    for p in run_jsons:
        j = json.loads(p.read_text())
        if j.get("status") == "complete" and j.get("mode") == "lp":
            m = j["model"]
            if m not in seen:
                seen.add(m)
                models_to_run.append(m)

    env = _os.environ.copy()
    all_stats: dict[str, int] = {}

    for m in models_to_run:
        print(f"\n{'='*60}\nDispatching subprocess for model: {m}\n{'='*60}")
        cmd = [
            sys.executable, __file__,
            "--model", m,
            "--n-steps", str(args.n_steps),
            "--max-batches", str(args.max_batches),
            "--subprocess",
        ]
        if args.force:
            cmd.append("--force")
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  [WARNING] subprocess for {m} exited with code {result.returncode}")

    # Tally final results from written JSON files
    for p in run_jsons:
        j_pilot = json.loads(p.read_text())
        if j_pilot.get("status") != "complete" or j_pilot.get("mode") != "lp":
            continue
        m2, ds2, sd2 = j_pilot["model"], j_pilot["dataset"], j_pilot["seed"]
        out_p = OUTPUT_DIR / f"nas_{m2}_{ds2}_s{sd2}.json"
        if out_p.exists():
            j_out = json.loads(out_p.read_text())
            key = j_out.get("status", "unknown")
        else:
            key = "missing"
        all_stats[key] = all_stats.get(key, 0) + 1

    print("\n=== Axis C NAS sweep complete ===")
    for k, v in sorted(all_stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
