#!/usr/bin/env python3
"""
Unified Baseline Model Training Script

This script provides a unified interface for training different baseline models
(EEGPT, LABRAM, etc.) using the abstract class architecture.

Usage:
    python baseline_main.py conf_file=assets/conf/eegpt/eegpt_unified.yaml model_type=eegpt
    python baseline_main.py conf_file=assets/conf/labram/labram_config.yaml model_type=labram

The config file should contain all necessary parameters for training.
The model_type parameter specifies which model architecture to use.
"""

import importlib
import os
import sys
from typing import Any

from omegaconf import OmegaConf

from baseline.abstract.factory import ModelRegistry
from common.path import get_conf_file_path
from common.utils import setup_yaml


MODEL_SPECS = {
    "MANAS": {
        "config": ("baseline.manas.manas_config", "MANASConfig"),
        "adapter": ("baseline.manas.manas_adapter", "MANASDataLoaderFactory"),
        "trainer": ("baseline.manas.manas_trainer", "MANASTrainer"),
    },
    "ndx_mae": {
        "config": ("baseline.ndx_mae_linear_split.ndx_mae_config", "NdxMAEConfig"),
        "adapter": ("baseline.manas.manas_adapter", "MANASDataLoaderFactory"),
        "trainer": ("baseline.ndx_mae_linear_split.ndx_mae_trainer", "NdxMAETrainer"),
    },
    "cosine_split_mae": {
        "config": (
            "baseline.cosine_split_mae.cosine_split_mae_config",
            "CosineSplitMAEConfig",
        ),
        "adapter": ("baseline.manas.manas_adapter", "MANASDataLoaderFactory"),
        "trainer": (
            "baseline.cosine_split_mae.cosine_split_mae_trainer",
            "CosineSplitMAETrainer",
        ),
    },
    "eegpt": {
        "config": ("baseline.eegpt.eegpt_config", "EegptConfig"),
        "adapter": ("baseline.eegpt.eegpt_adapter", "EegptDataLoaderFactory"),
        "trainer": ("baseline.eegpt.eegpt_trainer", "EegptTrainer"),
    },
    "labram": {
        "config": ("baseline.labram.labram_config", "LabramConfig"),
        "adapter": ("baseline.labram.labram_adapter", "LabramDataLoaderFactory"),
        "trainer": ("baseline.labram.labram_trainer", "LabramTrainer"),
    },
    "bendr": {
        "config": ("baseline.bendr.bendr_config", "BendrConfig"),
        "adapter": None,
        "trainer": ("baseline.bendr.bendr_trainer", "BendrTrainer"),
    },
    "biot": {
        "config": ("baseline.biot.biot_config", "BiotConfig"),
        "adapter": None,
        "trainer": ("baseline.biot.biot_trainer", "BiotTrainer"),
    },
    "cbramod": {
        "config": ("baseline.cbramod.cbramod_config", "CBraModConfig"),
        "adapter": ("baseline.cbramod.cbramod_adapter", "CBraModDataLoaderFactory"),
        "trainer": ("baseline.cbramod.cbramod_trainer", "CBraModTrainer"),
    },
    "reve": {
        "config": ("baseline.reve.reve_config", "ReveConfig"),
        "adapter": ("baseline.reve.reve_adapter", "ReveDataLoaderFactory"),
        "trainer": ("baseline.reve.reve_trainer", "ReveTrainer"),
    },
    "csbrain": {
        "config": ("baseline.csbrain.csbrain_config", "CSBrainConfig"),
        "adapter": ("baseline.csbrain.csbrain_adapter", "CSBrainDataLoaderFactory"),
        "trainer": ("baseline.csbrain.csbrain_trainer", "CSBrainTrainer"),
    },
    "mantis": {
        "config": ("baseline.mantis.mantis_config", "MantisConfig"),
        "adapter": ("baseline.mantis.mantis_adapter", "MantisDataLoaderFactory"),
        "trainer": ("baseline.mantis.mantis_trainer", "MantisTrainer"),
    },
    "moment": {
        "config": ("baseline.moment.moment_config", "MomentConfig"),
        "adapter": ("baseline.moment.moment_adapter", "MomentDataLoaderFactory"),
        "trainer": ("baseline.moment.moment_trainer", "MomentTrainer"),
    },
}

DEFAULT_SWEEP_METHODS = ("linear_probe", "full_finetune")


def _import_attr(module_path: str, attr_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def ensure_model_registered(model_type: str):
    """Lazily register the requested model to avoid importing optional deps eagerly."""
    if model_type in ModelRegistry.list_models():
        return

    if model_type not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {sorted(MODEL_SPECS.keys())}"
        )

    spec = MODEL_SPECS[model_type]
    config_class = _import_attr(*spec["config"])
    trainer_class = _import_attr(*spec["trainer"])
    adapter_class = None
    if spec["adapter"] is not None:
        adapter_class = _import_attr(*spec["adapter"])

    ModelRegistry.register_model(
        model_type=model_type,
        config_class=config_class,
        adapter_class=adapter_class,
        trainer_class=trainer_class,
    )


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
        methods = [item.strip() for item in value.split(",") if item.strip()]
    else:
        methods = [str(item).strip() for item in value if str(item).strip()]

    deduped_methods: list[str] = []
    for method in methods:
        if method not in deduped_methods:
            deduped_methods.append(method)
    return deduped_methods


def _build_validated_config(config_class, merged_config):
    cfg_dict = OmegaConf.to_container(merged_config, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg_dict)

    if not cfg.validate_config():
        raise ValueError(f"Invalid configuration for model type: {cfg.model_type}")

    return cfg


def _run_training(config_class, merged_config):
    cfg = _build_validated_config(config_class, merged_config)
    trainer = ModelRegistry.create_trainer(cfg)
    trainer.run()


def _build_sweep_run_config(base_config, method: str):
    run_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))

    if "model" not in run_config or "train_method" not in run_config.model:
        raise ValueError(
            "Sweep mode requires `model.train_method` in the selected model config."
        )

    run_config.model.train_method = method

    if "logging" in run_config:
        base_experiment_name = run_config.logging.get(
            "experiment_name", run_config.model_type
        )
        base_run_dir = run_config.logging.get("run_dir")
        tags = list(run_config.logging.get("tags", []))

        run_config.logging.experiment_name = f"{base_experiment_name}_{method}"
        if base_run_dir:
            run_config.logging.run_dir = os.path.join(base_run_dir, "sweep", method)
        for tag in ("sweep", method):
            if tag not in tags:
                tags.append(tag)
        run_config.logging.tags = tags

    return run_config


def _run_sweep(config_class, merged_config, methods: list[str]):
    datasets = list(merged_config.get("data", {}).get("datasets", {}).keys())
    print(
        "Running sweep with "
        f"{len(methods)} train method(s) across {len(datasets)} dataset(s): {datasets}"
    )

    for idx, method in enumerate(methods, start=1):
        print(f"[{idx}/{len(methods)}] train_method={method}")
        run_config = _build_sweep_run_config(merged_config, method)
        _run_training(config_class, run_config)


def main():
    """Main training function that can handle any registered baseline model."""
    setup_yaml()

    # Parse CLI arguments
    cli_args = OmegaConf.from_cli()

    if "conf_file" not in cli_args:
        raise ValueError("Please provide a config file: conf_file=path/to/config.yaml")

    # Get model type from CLI args or config
    model_type: str = cli_args.get("model_type", None)

    # Load config file
    conf_file_path = get_conf_file_path(cli_args.conf_file)
    file_cfg = OmegaConf.load(conf_file_path)

    if model_type is None:
        model_type = file_cfg.get("model_type")

    ensure_model_registered(model_type)

    # Validate model type
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")

    # Create base config for the specified model type
    config_class = ModelRegistry.get_config_class(model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())

    # Merge configurations: code defaults < file config < CLI args
    merged_config = OmegaConf.merge(code_cfg, file_cfg, cli_args)

    # Ensure model_type is set correctly
    merged_config.model_type = model_type

    sweep_enabled = _is_enabled(merged_config.get("sweep", False))
    sweep_methods = _parse_sweep_methods(merged_config.get("sweep_methods"))

    if sweep_enabled:
        _run_sweep(config_class, merged_config, sweep_methods)
    else:
        _run_training(config_class, merged_config)


def list_available_models():
    """List all available model types."""
    print("Available baseline models:")
    for model_type in sorted(set(ModelRegistry.list_models()) | set(MODEL_SPECS)):
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main() 
