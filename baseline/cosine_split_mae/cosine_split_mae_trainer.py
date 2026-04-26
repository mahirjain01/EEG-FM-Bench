from __future__ import annotations

from typing import Any

import torch

from baseline.manas.manas_trainer import MANASTrainer
from baseline.cosine_split_mae.model import CosineSplitMAEEncoder


class CosineSplitMAETrainer(MANASTrainer):
    def _build_mae_kwargs(self) -> dict[str, Any]:
        model_cfg = self.config.model
        return {
            "fs": model_cfg.fs,
            "patch_seconds": model_cfg.patch_seconds,
            "overlap_seconds": model_cfg.overlap_seconds,
            "embed_dim": model_cfg.embed_dim,
            "encoder_depth": model_cfg.encoder_depth,
            "encoder_heads": model_cfg.encoder_heads,
            "decoder_depth": model_cfg.decoder_depth,
            "decoder_heads": model_cfg.decoder_heads,
            "mask_ratio": model_cfg.mask_ratio,
            "aux_loss_weight": model_cfg.aux_loss_weight,
            "which_mask": model_cfg.which_mask,
            "fuzzy_noise_std": model_cfg.fuzzy_noise_std,
            "spatial_radius_black": model_cfg.spatial_radius_black,
            "spatial_radius_fuzzy": model_cfg.spatial_radius_fuzzy,
            "temporal_radius_black": model_cfg.temporal_radius_black,
            "temporal_radius_fuzzy": model_cfg.temporal_radius_fuzzy,
            "dropout_ratio": model_cfg.dropout_ratio,
            "dropout_radius": model_cfg.dropout_radius,
            "ema_mix_ratio": model_cfg.ema_mix_ratio,
            "ema_temperature": model_cfg.ema_temperature,
            "ema_floor_eps": model_cfg.ema_floor_eps,
            "use_pairwise_channel_diffs": model_cfg.use_pairwise_channel_diffs,
            "n_spatial_coords": model_cfg.n_spatial_coords,
            "posenc_n_freqs": model_cfg.posenc_n_freqs,
            "sigma_noise": model_cfg.sigma_noise,
            "attn_dropout": model_cfg.attn_dropout,
            "use_flash_attention": model_cfg.use_flash_attention,
        }

    def build_encoder(self) -> torch.nn.Module:
        return CosineSplitMAEEncoder(
            source_mae_path=self.config.model.source_mae_path,
            mae_kwargs=self._build_mae_kwargs(),
        )

    def _setup_encoder(self):
        method = self.config.model.train_method

        model = (
            self.model.module
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else self.model
        )
        if not hasattr(model, "encoder") or not hasattr(model, "classifier"):
            raise AttributeError(
                "cosine_split_mae model must have 'encoder' and 'classifier' attributes."
            )

        encoder = model.encoder
        mae = encoder.mae

        for param in encoder.parameters():
            param.requires_grad = False

        if method == "partial_finetune":
            for layer in mae.encoder.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in mae.encoder.final_norm.parameters():
                param.requires_grad = True

        if method == "full_finetune":
            for param in encoder.parameters():
                param.requires_grad = True

        if hasattr(model, "classifier") and model.classifier is not None:
            for param in model.classifier.parameters():
                param.requires_grad = True
