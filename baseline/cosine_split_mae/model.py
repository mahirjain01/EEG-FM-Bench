from __future__ import annotations

import importlib.util
import inspect
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@lru_cache(maxsize=None)
def _load_external_mae_module(source_path: str):
    source_file = Path(source_path).expanduser().resolve()
    if not source_file.is_file():
        raise FileNotFoundError(f"External MAE source file not found: {source_file}")

    spec = importlib.util.spec_from_file_location(
        "baseline_external_cosine_split_mae", source_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {source_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_external_mae_class(source_path: str):
    module = _load_external_mae_module(source_path)
    mae_cls = getattr(module, "MAE", None)
    if mae_cls is None:
        raise AttributeError(f"No MAE class found in external source: {source_path}")
    return mae_cls


def filter_external_mae_kwargs(mae_cls, mae_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs supported by the external MAE constructor."""
    signature = inspect.signature(mae_cls.__init__)
    accepted = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in mae_kwargs.items() if key in accepted}


def _call_pos_enc(mae: nn.Module, xyz: Tensor, num_patches: int) -> Tensor:
    """
    Support both external MAE families:
    - newer split-pos models: pos_enc(xyz, num_patches) -> (B, C, P, D)
    - older att-mask models: prepare_coords(xyz, num_patches) + pos_enc(coords) -> (B, N, D)
    """
    if hasattr(mae, "_build_full_pos_emb"):
        pos_emb = mae._build_full_pos_emb(xyz, num_patches)
        return pos_emb.flatten(1, 2) if pos_emb.ndim == 4 else pos_emb

    pos_signature = inspect.signature(mae.pos_enc.forward)
    pos_params = [
        param
        for name, param in pos_signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    if len(pos_params) >= 2:
        pos_emb = mae.pos_enc(xyz, num_patches)
    else:
        if not hasattr(mae, "prepare_coords"):
            raise AttributeError(
                "External MAE requires prepare_coords(xyz, num_patches) for its "
                "positional encoding path."
            )
        coords = mae.prepare_coords(xyz, num_patches)
        pos_emb = mae.pos_enc(coords)

    return pos_emb.flatten(1, 2) if pos_emb.ndim == 4 else pos_emb


def _call_encoder(mae: nn.Module, x: Tensor, xyz: Tensor, num_patches: int) -> Tensor:
    pos_emb = _call_pos_enc(mae, xyz, num_patches)
    encoder_input = x + pos_emb

    encoder_forward = mae.encoder.forward
    encoder_signature = inspect.signature(encoder_forward)
    supports_spatial_bias = "spatial_bias" in encoder_signature.parameters

    if supports_spatial_bias and hasattr(mae, "_build_spatial_rel_bias"):
        full_mask = torch.ones(
            encoder_input.shape[:2], dtype=torch.bool, device=encoder_input.device
        )
        base_spatial_bias = mae._build_spatial_rel_bias(xyz)
        spatial_bias = mae._gather_visible_spatial_bias(
            base_spatial_bias, full_mask, num_patches
        )
        encoded = mae.encoder(encoder_input, spatial_bias=spatial_bias)
    else:
        encoded = mae.encoder(encoder_input)

    return encoded[0] if isinstance(encoded, tuple) else encoded


class CosineSplitMAEEncoder(nn.Module):
    """
    Encoder-only wrapper around the cosine-split MAE implementation.

    The wrapped module is stored under ``self.mae`` so checkpoints with
    ``mae.*`` keys load directly.
    """

    def __init__(self, source_mae_path: str, mae_kwargs: dict[str, Any]):
        super().__init__()
        self.source_mae_path = os.path.abspath(source_mae_path)

        mae_cls = load_external_mae_class(self.source_mae_path)
        self.mae = mae_cls(**filter_external_mae_kwargs(mae_cls, mae_kwargs))

    def forward(self, x: Tensor, xyz: Tensor) -> Tensor:
        x, xyz = self.mae._to_pairwise_channels(x, xyz)

        patches = x.unfold(-1, self.mae.patch_size, self.mae.step)
        num_patches = patches.shape[2]

        tokens = self.mae.patch_embed.linear(patches).flatten(1, 2)
        return _call_encoder(self.mae, tokens, xyz, num_patches)
