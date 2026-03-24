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


class CosineSplitMAEDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class CosineSplitMAEModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = (
        "/share/tmp/mahir/ckpts/cosine_split/cosine_split_15.pt"
    )
    source_mae_path: str = "/share/tmp/mahir/ckpts/cosine_split/mae_model.py"
    checkpoint_encoder_only: bool = False

    fs: int = 200
    patch_seconds: float = 1.0
    overlap_seconds: float = 0.1

    embed_dim: int = 512
    encoder_depth: int = 22
    encoder_heads: int = 8
    decoder_depth: int = 4
    decoder_heads: int = 8

    mask_ratio: float = 0.55
    aux_loss_weight: float = 0.1
    which_mask: str = "default"
    fuzzy_noise_std: float = 0.0
    spatial_radius_black: float = 3.0
    spatial_radius_fuzzy: float = 4.0
    temporal_radius_black: float = 3.0
    temporal_radius_fuzzy: float = 4.0
    dropout_ratio: float = 0.1
    dropout_radius: float = 3.0
    ema_mix_ratio: float = 0.6
    ema_temperature: float = 2.0
    ema_floor_eps: float = 0.1

    use_pairwise_channel_diffs: bool = False
    n_spatial_coords: int = 3
    posenc_n_freqs: int = 4
    sigma_noise: float = 0.25

    attn_dropout: float = 0.0
    use_flash_attention: bool = True

    train_method: Literal["linear_probe", "partial_finetune", "full_finetune"] = (
        "linear_probe"
    )

    @property
    def dim(self) -> int:
        return self.embed_dim

    @property
    def patch_size(self) -> int:
        return int(round(self.patch_seconds * self.fs))

    @property
    def overlap_size(self) -> int:
        return int(round(self.overlap_seconds * self.fs))


class CosineSplitMAETrainingArgs(BaseTrainingArgs):
    max_epochs: int = 30

    lr_schedule: str = "cosine"
    max_lr: float = 2e-4
    min_lr: float = 2e-5

    use_amp: bool = True
    freeze_encoder: bool = False


class CosineSplitMAELoggingArgs(BaseLoggingArgs):
    experiment_name: str = "cosine_split_mae"
    run_dir: str = RUN_ROOT

    tags: List[str] = Field(
        default_factory=lambda: ["cosine_split_mae", "EEG-FM-Bench"]
    )

    log_step_interval: int = 5
    ckpt_interval: int = 1


class CosineSplitMAEConfig(AbstractConfig):
    model_type: str = "cosine_split_mae"
    fs: int = 200

    data: CosineSplitMAEDataArgs = Field(default_factory=CosineSplitMAEDataArgs)
    model: CosineSplitMAEModelArgs = Field(default_factory=CosineSplitMAEModelArgs)
    training: CosineSplitMAETrainingArgs = Field(
        default_factory=CosineSplitMAETrainingArgs
    )
    logging: CosineSplitMAELoggingArgs = Field(
        default_factory=CosineSplitMAELoggingArgs
    )

    def validate_config(self) -> bool:
        return True
