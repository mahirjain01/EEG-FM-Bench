"""
UniformEvalBench config builder.

Assembles per-run YAML configs from three sources:
  1. Model architecture defaults  — configs/model_defaults/{model}.yaml
  2. Benchmark-level constants    — defined here (epochs, LR schedules, etc.)
  3. Run-specific values          — passed as arguments to build_run_config()

The generated config is a plain Python dict; callers write it to disk using
yaml.dump() or OmegaConf.save().

Public API
----------
  build_run_config(model, dataset, seed, mode, run_root, exp_name=None)
      -> (config_dict, exp_name)

  MODEL_PRETRAINED   : dict[str, str]   checkpoint paths per model
  REVE_POS_BANK_PATH : str              extra checkpoint for REVE
"""

import yaml
from pathlib import Path

from run_utils import CKPT_ROOT, REPO_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DEFAULTS_DIR = Path(__file__).parent / "configs" / "model_defaults"
PILOT_RUN_ROOT     = Path("/mnt/eegfmbench/runs/uniformeval/pilot")

# ---------------------------------------------------------------------------
# Per-model checkpoint paths
# ---------------------------------------------------------------------------

MODEL_PRETRAINED: dict[str, str] = {
    "eegpt":   str(CKPT_ROOT / "eegpt/eegpt_wrapped.ckpt"),
    "labram":  str(CKPT_ROOT / "labram/labram_wrapped.pt"),
    "cbramod": str(CKPT_ROOT / "cbramod/pretrained_weights.pth"),
    "biot":    str(CKPT_ROOT / "biot/EEG-six-datasets-18-channels.ckpt"),
    "csbrain": str(CKPT_ROOT / "csbrain/csbrain_stripped.pth"),
    "reve":    str(CKPT_ROOT / "reve/reve_base.safetensors"),
    "moment":  str(CKPT_ROOT / "moment/model.safetensors"),
}

# REVE requires a second checkpoint for its position bank
REVE_POS_BANK_PATH: str = str(CKPT_ROOT / "reve/reve_positions.safetensors")

# ---------------------------------------------------------------------------
# Per-model training settings
# ---------------------------------------------------------------------------

# LR schedule per model — some models' validate_config() rejects "onecycle"
LR_SCHEDULE: dict[str, str] = {
    "eegpt":   "onecycle",
    "labram":  "cosine",
    "cbramod": "cosine",
    "biot":    "cosine",
    "csbrain": "cosine",
    "reve":    "reduce_on_plateau",
    "moment":  "cosine",
}

# AMP disabled for BIOT: its STFT (cuFFT) cannot run in bf16
USE_AMP: dict[str, bool] = {
    "eegpt":   True,
    "labram":  True,
    "cbramod": True,
    "biot":    False,
    "csbrain": True,
    "reve":    True,
    "moment":  True,
}

# ---------------------------------------------------------------------------
# Benchmark-level constants
# ---------------------------------------------------------------------------

EPOCHS:      dict[str, int] = {"lp": 30, "ft": 30}
BATCH_SIZE:  int = 32
FS:          int = 200   # target sampling rate (Hz)
NUM_WORKERS: int = 4


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _load_model_defaults(model: str) -> dict:
    """Load model architecture defaults from configs/model_defaults/{model}.yaml."""
    path = MODEL_DEFAULTS_DIR / f"{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No model defaults found for '{model}' at {path}. "
            f"Available: {[p.stem for p in MODEL_DEFAULTS_DIR.glob('*.yaml')]}"
        )
    with path.open() as f:
        return yaml.safe_load(f)


def _make_port(exp_name: str) -> int:
    """Deterministic master port per experiment — avoids collision when queuing runs."""
    return 51103 + (abs(hash(exp_name)) % 1000)


def build_run_config(
    model: str,
    dataset: str,
    seed: int,
    mode: str,
    run_root: Path = PILOT_RUN_ROOT,
    exp_name: str | None = None,
) -> tuple[dict, str]:
    """
    Build a complete training config dict for one experiment run.

    Parameters
    ----------
    model    : model key, e.g. 'eegpt'
    dataset  : dataset key, e.g. 'bcic_2a'
    seed     : random seed
    mode     : 'lp' (frozen encoder) or 'ft' (full fine-tuning)
    run_root : where checkpoints and logs are written (default: pilot run root)
    exp_name : override the auto-generated '{mode}_{model}_{dataset}_s{seed}' name

    Returns
    -------
    config   : dict ready for yaml.dump() or OmegaConf.save()
    exp_name : the experiment name string
    """
    if exp_name is None:
        exp_name = f"{mode}_{model}_{dataset}_s{seed}"

    # Load model-specific architecture params
    model_section = _load_model_defaults(model)

    # Inject the pretrained checkpoint path(s)
    model_section["pretrained_path"] = MODEL_PRETRAINED[model]
    if model == "reve":
        model_section["pos_bank_pretrained_path"] = REVE_POS_BANK_PATH

    epochs = EPOCHS[mode]

    config = {
        "seed":        seed,
        "master_port": _make_port(exp_name),
        "multitask":   False,
        "model_type":  model,
        "fs":          FS,

        "data": {
            "batch_size":  BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "datasets":    {dataset: "finetune"},
        },

        "model": model_section,

        "training": {
            "max_epochs":        epochs,
            "weight_decay":      0.01,
            "max_grad_norm":     3.0,
            "lr_schedule":       LR_SCHEDULE[model],
            "max_lr":            5e-4,
            "encoder_lr_scale":  0.1,
            "warmup_epochs":     2,
            "warmup_scale":      1e-2,
            "pct_start":         0.2,
            "min_lr":            1e-6,
            "freeze_encoder":    (mode == "lp"),
            "use_amp":           USE_AMP[model],
            "label_smoothing":   0.1,
            "lora":              {"use_lora": False},
        },

        "logging": {
            "experiment_name":  exp_name,
            "run_dir":          str(run_root),
            "use_cloud":        False,
            "project":          "uniformevalbench",
            "offline":          True,
            "tags":             ["pilot", model, mode, dataset],
            "log_step_interval": 50,
            "ckpt_interval":    epochs,
        },
    }

    return config, exp_name


def write_config(config: dict, path: Path) -> None:
    """Serialize config dict to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
