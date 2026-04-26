"""
Internal: direct FM trainer loading.

Replaces the ModelRegistry + BaselineVisualizer dependency chain.
Each reference FM adapter calls _build_trainer() which:
  1. Loads config via OmegaConf and a local per-FM map (no ModelRegistry)
  2. Imports the trainer class directly via a local map (no ModelRegistry)
  3. Sets up the trainer in single-GPU eval mode
  4. Loads the FBP checkpoint directly

This is NOT part of the public API.  End users implement ModelAdapter directly.
The 6 reference FMs subclass VizBridgeAdapter to avoid duplicating the
checkpoint-loading and hook logic.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..model_adapter import ModelAdapter

# ---------------------------------------------------------------------------
# Per-FM config and trainer class maps (replaces ModelRegistry)
# ---------------------------------------------------------------------------

_CONFIG_MAP: dict[str, tuple[str, str]] = {
    "eegpt":   ("uniformevalbench.models.eegpt.eegpt_config",    "EegptConfig"),
    "labram":  ("uniformevalbench.models.labram.labram_config",   "LabramConfig"),
    "biot":    ("uniformevalbench.models.biot.biot_config",       "BiotConfig"),
    "cbramod": ("uniformevalbench.models.cbramod.cbramod_config", "CBraModConfig"),
    "csbrain": ("uniformevalbench.models.csbrain.csbrain_config", "CSBrainConfig"),
    "reve":    ("uniformevalbench.models.reve.reve_config",       "ReveConfig"),
}

_TRAINER_MAP: dict[str, tuple[str, str]] = {
    "eegpt":   ("uniformevalbench.models.eegpt.eegpt_trainer",       "EegptTrainer"),
    "labram":  ("uniformevalbench.models.labram.labram_trainer",      "LabramTrainer"),
    "biot":    ("uniformevalbench.models.biot.biot_trainer",          "BiotTrainer"),
    "cbramod": ("uniformevalbench.models.cbramod.cbramod_trainer",    "CBraModTrainer"),
    "csbrain": ("uniformevalbench.models.csbrain.csbrain_trainer",    "CSBrainTrainer"),
    "reve":    ("uniformevalbench.models.reve.reve_trainer",          "REVETrainer"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_repo_on_path() -> Path:
    """Add repo root to sys.path so baseline/* imports work."""
    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _ensure_dist() -> None:
    """Minimal single-GPU dist init (gloo). Some trainers require an active process group."""
    import torch.distributed as dist
    if dist.is_initialized():
        return
    import os, socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _find_ckpt(log_path: str, model: str, dataset: str) -> Path | None:
    text = Path(log_path).read_text(errors="ignore")
    m = re.search(r"checkpoint dir: (/[^\s,]+)", text)
    if not m:
        return None
    ckpt_dir = Path(m.group(1))
    return ckpt_dir / "seperated" / dataset / f"{model}_{dataset}_last.pt"


def _get_class(mapping: dict[str, tuple[str, str]], key: str):
    """Dynamic import via (module_path, class_name) map entry."""
    mod_path, cls_name = mapping[key]
    return getattr(importlib.import_module(mod_path), cls_name)


def _load_cfg(model_type: str, config_path: str, dataset: str, batch_size: int = 128):
    """Merge code defaults ← YAML ← eval overrides. No ModelRegistry needed."""
    from omegaconf import OmegaConf
    cfg_class = _get_class(_CONFIG_MAP, model_type)
    file_cfg  = OmegaConf.load(config_path)
    code_cfg  = OmegaConf.create(cfg_class().model_dump())
    merged    = OmegaConf.merge(code_cfg, file_cfg)
    cfg_dict  = OmegaConf.to_container(merged, resolve=True, throw_on_missing=True)
    cfg       = cfg_class.model_validate(cfg_dict)

    if hasattr(cfg, "model") and hasattr(cfg.model, "pretrained_path"):
        cfg.model.pretrained_path = None
    cfg.data.batch_size  = batch_size
    cfg.data.num_workers = 1
    cfg.data.datasets    = {dataset: "finetune"}
    return cfg


def _get_model(trainer) -> torch.nn.Module:
    """Unwrap DDP wrapper if present."""
    m = trainer.model
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def _build_trainer(
    model_type:  str,
    ckpt_path:   str,
    config_path: str,
    dataset:     str,
    seed:        int,
    batch_size:  int = 128,
):
    """
    Build an eval-mode trainer with FBP checkpoint loaded.
    Replaces build_viz() / BaselineVisualizer.build_model().
    """
    _ensure_repo_on_path()
    _ensure_dist()

    cfg         = _load_cfg(model_type, config_path, dataset, batch_size)
    trainer_cls = _get_class(_TRAINER_MAP, model_type)

    trainer = trainer_cls(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.setup_device(device)
    if hasattr(trainer, "setup_analysis_mode"):
        trainer.setup_analysis_mode()

    ds_name = next(iter(cfg.data.datasets.keys()))
    trainer.collect_dataset_info(mixed=False, ds_name=ds_name)
    trainer.setup_model()

    # Load FBP checkpoint (saved by trainer.save_checkpoint during pilot sweep).
    # Strip DDP "module." prefix that may be present from multi-GPU training.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw   = state.get("model", state)
    raw   = {k.removeprefix("module."): v for k, v in raw.items()}
    m = _get_model(trainer)
    m.load_state_dict(raw, strict=False)

    trainer.model.eval()
    return trainer


# ---------------------------------------------------------------------------
# ModelAdapter backed by a trainer instance
# ---------------------------------------------------------------------------

class VizBridgeAdapter(ModelAdapter):
    """
    ModelAdapter backed by an AbstractTrainer instance.

    encode() captures the pooled representation at the classifier input via a
    forward hook — the same technique used in axis_a_knn.py, factored here so
    knn.py and spa.py share identical logic across all 6 reference FMs.
    """

    MODEL_TYPE: str = ""

    def __init__(self, trainer) -> None:
        self._trainer = trainer

    @property
    def _model(self) -> torch.nn.Module:
        return _get_model(self._trainer)

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def encode(self, batch: dict) -> Tensor:
        """Hook model.classifier, run forward, return [B, D] embedding."""
        device   = self._trainer.device
        captured: list[Tensor] = []

        def _hook(module, inp, out):
            feat = inp[0].detach().float()
            if feat.dim() == 4:
                feat = feat.mean(dim=(1, 2))
            elif feat.dim() == 3:
                feat = feat.mean(dim=1)
            captured.append(feat.cpu())

        handle = self._model.classifier.register_forward_hook(_hook)
        try:
            gpu_batch = {
                k: (v.to(device) if isinstance(v, Tensor) else v)
                for k, v in batch.items()
            }
            with torch.no_grad():
                self._model(gpu_batch)
        finally:
            handle.remove()

        return captured[0] if captured else torch.zeros(0)

    def freeze_encoder(self) -> None:
        for name, p in self._model.named_parameters():
            if "classifier" not in name:
                p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for p in self._model.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Dataloader creation delegates to the trainer
    # ------------------------------------------------------------------

    def create_dataloader(self, dataset: str, config: str, split: str) -> DataLoader:
        import datasets as hf_datasets
        split_map = {
            "train": hf_datasets.Split.TRAIN,
            "valid": hf_datasets.Split.VALIDATION,
            "test":  hf_datasets.Split.TEST,
        }
        hf_split = split_map.get(split, split)
        loader, _ = self._trainer.create_single_dataloader(dataset, config, hf_split)
        return loader

    # ------------------------------------------------------------------
    # Factory — shared across all per-FM adapters
    # ------------------------------------------------------------------

    @classmethod
    def from_run_json(cls, run_json_path, batch_size: int = 128):
        """
        Build an adapter from a completed pilot (lp) run JSON.

        The JSON must contain config_path, log_path, model, dataset, seed.
        """
        import json
        j       = json.loads(Path(run_json_path).read_text())
        model   = j["model"]
        dataset = j["dataset"]
        seed    = int(j["seed"])

        ckpt = _find_ckpt(j["log_path"], model, dataset)
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {model}/{dataset}/s{seed}. "
                f"Expected: {ckpt}"
            )

        trainer = _build_trainer(
            model_type  = model,
            ckpt_path   = str(ckpt),
            config_path = j["config_path"],
            dataset     = dataset,
            seed        = seed,
            batch_size  = batch_size,
        )
        return cls(trainer)
