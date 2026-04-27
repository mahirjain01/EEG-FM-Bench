#!/usr/bin/env python3
"""
UniformEvalBench — training entry point.

Standalone replacement for baseline_main.py.  All imports come from the
bundled uniformevalbench.models.* package; no parent-repo baseline/* needed.

Usage (same interface as baseline_main.py):
    torchrun --standalone --nproc_per_node=N main.py \
        conf_file=<path/to/config.yaml> model_type=<model>

Supported models: eegpt, labram, biot, cbramod, csbrain, reve
"""

import importlib
import os
import sys
from typing import Any

from omegaconf import OmegaConf

from uniformevalbench.models.abstract import ModelRegistry
from uniformevalbench.common.path import get_conf_file_path
from uniformevalbench.utils import setup_yaml


MODEL_SPECS = {
    "eegpt": {
        "config":  ("uniformevalbench.models.eegpt.eegpt_config",    "EegptConfig"),
        "adapter": ("uniformevalbench.models.eegpt.eegpt_adapter",   "EegptDataLoaderFactory"),
        "trainer": ("uniformevalbench.models.eegpt.eegpt_trainer",   "EegptTrainer"),
    },
    "labram": {
        "config":  ("uniformevalbench.models.labram.labram_config",   "LabramConfig"),
        "adapter": ("uniformevalbench.models.labram.labram_adapter",  "LabramDataLoaderFactory"),
        "trainer": ("uniformevalbench.models.labram.labram_trainer",  "LabramTrainer"),
    },
    "biot": {
        "config":  ("uniformevalbench.models.biot.biot_config",       "BiotConfig"),
        "adapter": None,
        "trainer": ("uniformevalbench.models.biot.biot_trainer",      "BiotTrainer"),
    },
    "cbramod": {
        "config":  ("uniformevalbench.models.cbramod.cbramod_config", "CBraModConfig"),
        "adapter": ("uniformevalbench.models.cbramod.cbramod_adapter","CBraModDataLoaderFactory"),
        "trainer": ("uniformevalbench.models.cbramod.cbramod_trainer","CBraModTrainer"),
    },
    "csbrain": {
        "config":  ("uniformevalbench.models.csbrain.csbrain_config", "CSBrainConfig"),
        "adapter": ("uniformevalbench.models.csbrain.csbrain_adapter","CSBrainDataLoaderFactory"),
        "trainer": ("uniformevalbench.models.csbrain.csbrain_trainer","CSBrainTrainer"),
    },
    "reve": {
        "config":  ("uniformevalbench.models.reve.reve_config",       "ReveConfig"),
        "adapter": ("uniformevalbench.models.reve.reve_adapter",      "ReveDataLoaderFactory"),
        "trainer": ("uniformevalbench.models.reve.reve_trainer",      "ReveTrainer"),
    },
}

DEFAULT_SWEEP_METHODS = ("linear_probe", "full_finetune")


def _import_attr(module_path: str, attr_name: str):
    return getattr(importlib.import_module(module_path), attr_name)


def ensure_model_registered(model_type: str):
    if model_type in ModelRegistry.list_models():
        return
    if model_type not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Bundled models: {sorted(MODEL_SPECS)}"
        )
    spec = MODEL_SPECS[model_type]
    config_class  = _import_attr(*spec["config"])
    trainer_class = _import_attr(*spec["trainer"])
    adapter_class = _import_attr(*spec["adapter"]) if spec["adapter"] else None
    ModelRegistry.register_model(model_type, config_class, adapter_class, trainer_class)


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_sweep_methods(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_SWEEP_METHODS)
    if isinstance(value, str):
        methods = [m.strip() for m in value.split(",") if m.strip()]
    else:
        methods = [str(m).strip() for m in value if str(m).strip()]
    seen: list[str] = []
    for m in methods:
        if m not in seen:
            seen.append(m)
    return seen


def _build_validated_config(config_class, merged_config):
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)
    if not cfg.validate_config():
        raise ValueError(f"Invalid configuration for model type: {cfg.model_type}")
    return cfg


def _run_training(config_class, merged_config):
    cfg     = _build_validated_config(config_class, merged_config)
    trainer = ModelRegistry.create_trainer(cfg)
    trainer.run()


def _build_sweep_run_config(base_config, method: str):
    run_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
    if "model" not in run_config or "train_method" not in run_config.model:
        raise ValueError("Sweep mode requires `model.train_method` in the config.")
    run_config.model.train_method = method
    if "logging" in run_config:
        base_name    = run_config.logging.get("experiment_name", run_config.model_type)
        base_run_dir = run_config.logging.get("run_dir")
        tags         = list(run_config.logging.get("tags", []))
        run_config.logging.experiment_name = f"{base_name}_{method}"
        if base_run_dir:
            run_config.logging.run_dir = os.path.join(base_run_dir, "sweep", method)
        for tag in ("sweep", method):
            if tag not in tags:
                tags.append(tag)
        run_config.logging.tags = tags
    return run_config


def _run_sweep(config_class, merged_config, methods: list[str]):
    datasets = list(merged_config.get("data", {}).get("datasets", {}).keys())
    print(f"Sweep: {len(methods)} method(s) × {len(datasets)} dataset(s): {datasets}")
    for idx, method in enumerate(methods, start=1):
        print(f"[{idx}/{len(methods)}] train_method={method}")
        _run_training(config_class, _build_sweep_run_config(merged_config, method))


def main():
    setup_yaml()
    cli_args = OmegaConf.from_cli()

    if "conf_file" not in cli_args:
        raise ValueError("Provide a config file: conf_file=path/to/config.yaml")

    model_type: str = cli_args.get("model_type", None)
    conf_file_path  = get_conf_file_path(cli_args.conf_file)
    file_cfg        = OmegaConf.load(conf_file_path)

    if model_type is None:
        model_type = file_cfg.get("model_type")

    ensure_model_registered(model_type)

    config_class  = ModelRegistry.get_config_class(model_type)
    code_cfg      = OmegaConf.create(config_class().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg, cli_args)
    merged_config.model_type = model_type

    if _is_enabled(merged_config.get("sweep", False)):
        _run_sweep(config_class, merged_config, _parse_sweep_methods(merged_config.get("sweep_methods")))
    else:
        _run_training(config_class, merged_config)


if __name__ == "__main__":
    main()
