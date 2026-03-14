import os
from typing import Literal, Optional

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.manas.MAEBottleneckDecoderv2 import MAE, MAEConfig
from baseline.manas.manas_adapter import MANASDataLoaderFactory
from baseline.manas.manas_config import MANASConfig


class MANASUnifiedModel(nn.Module):
    def __init__(self, encoder: MAE, classifier: MultiHeadClassifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        x = batch["data"]
        coords = batch["coords"]
        montage = batch["montage"][0]

        features = self.encoder(x, coords)

        B, C, _ = x.shape
        P = features.shape[1] // C

        features = features.view(B, C, P, -1)
        features = features.permute(0, 2, 1, 3)

        logits = self.classifier(features, montage)
        return logits


class MANASTrainer(AbstractTrainer):
    def __init__(self, config: MANASConfig):
        super().__init__(config)

        self.config = config

        self.dataloader_factory = MANASDataLoaderFactory(
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
        )

        self.encoder: Optional[MAE] = None
        self.classifier: Optional[MultiHeadClassifier] = None
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def compute_num_patches(
        n_timepoints: int, patch_size: int, overlap_size: int
    ) -> int:
        if n_timepoints < patch_size:
            return 0
        return 1 + (n_timepoints - patch_size) // (patch_size - overlap_size)

    def load_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided for loading.")

        if self.encoder is None:
            raise RuntimeError(
                "MANAS encoder must be initialized before loading checkpoint."
            )

        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from: {checkpoint_path}"
            ) from e

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)

        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint loading issues: missing keys: {missing}, unexpected keys: {unexpected}"
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
                "MANAS model must have 'encoder' and 'classifier' attributes."
            )

        encoder: MAE = model.encoder

        for param in encoder.parameters():
            param.requires_grad = False

        if method == "partial_finetune":
            for param in encoder.enc.blocks[-2:].parameters():
                param.requires_grad = True
            for param in encoder.enc.norm.parameters():
                param.requires_grad = True

        if method == "full_finetune":
            for param in encoder.parameters():
                param.requires_grad = True

        if hasattr(model, "classifier") and model.classifier is not None:
            for param in model.classifier.parameters():
                param.requires_grad = True

    def setup_model(self):
        self.encoder = MAE(MAEConfig())

        embed_dim = self.config.model.dim

        head_configs = {
            ds_name: info["n_class"] for ds_name, info in self.ds_info.items()
        }
        head_cfg = self.config.model.classifier_head

        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info["shape_info"].items():
                n_patches = self.compute_num_patches(
                    n_timepoints,
                    int(self.config.model.patch_size),
                    int(self.config.model.overlap_size),
                )
                ds_shape_info[montage_key] = (n_patches, n_channels, embed_dim)

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_info,
        )

        self.load_checkpoint(self.config.model.pretrained_path)
        model = MANASUnifiedModel(encoder=self.encoder, classifier=self.classifier)

        model = self.apply_lora(model)
        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model)

        self.model = model

        return model

    def train_epoch(self, train_loader, train_sampler):
        self._setup_encoder()
        self.config.training.freeze_encoder = (
            self.config.model.train_method == "linear_probe"
        )

        super().train_epoch(train_loader, train_sampler)
