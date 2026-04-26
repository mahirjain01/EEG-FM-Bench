from typing import Dict, List, Literal, Optional

from pydantic import Field

from baseline.abstract.config import (
    AbstractConfig,
    BaseDataArgs,
    BaseLoggingArgs,
    BaseModelArgs,
    BaseTrainingArgs,
)
from common.path import RUN_ROOT


class MANASDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class MANASModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None
    checkpoint_variant: Literal["ndx_legacy", "bench_v2"] = "bench_v2"
    checkpoint_encoder_only: bool = False

    # Data
    fs: int = 200
    patch_size: int = 200
    overlap_size: int = 20

    # Model:
    dim: int = 512
    enc_layers: int = 12
    enc_heads: int = 8
    dec_layers: int = 4
    dec_heads: int = 8

    # Masking
    mask_ratio: float = 0.55
    spat_radius: float = 3.0
    time_radius: float = 3.0

    # Loss
    aux_weight: float = 0.1

    # Coord Encoding
    n_freqs: int = 4

    @property
    def stride(self):
        return self.patch_size - self.overlap_size

    train_method: Literal["linear_probe", "partial_finetune", "full_finetune"] = (
        "linear_probe"
    )


class MANASTrainingArgs(BaseTrainingArgs):
    max_epochs: int = 30

    lr_schedule: str = "cosine"
    max_lr: float = 2e-4
    min_lr: float = 2e-5

    use_amp: bool = True
    freeze_encoder: bool = False


class MANASLoggingArgs(BaseLoggingArgs):
    experiment_name: str = "MANAS"
    run_dir: str = RUN_ROOT

    tags: List[str] = Field(default_factory=lambda: ["MANAS", "EEG-FM-Bench"])

    log_step_interval: int = 5
    ckpt_interval: int = 1


class MANASConfig(AbstractConfig):
    model_type: str = "MANAS"

    data: MANASDataArgs = Field(default_factory=MANASDataArgs)
    model: MANASModelArgs = Field(default_factory=MANASModelArgs)
    training: MANASTrainingArgs = Field(default_factory=MANASTrainingArgs)
    logging: MANASLoggingArgs = Field(default_factory=MANASLoggingArgs)

    def validate_config(self) -> bool:
        return True
