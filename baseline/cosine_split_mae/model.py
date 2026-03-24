from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

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
        self.mae = mae_cls(**mae_kwargs)

    def forward(self, x: Tensor, xyz: Tensor) -> Tensor:
        x, xyz = self.mae._to_pairwise_channels(x, xyz)

        patches = x.unfold(-1, self.mae.patch_size, self.mae.step)
        num_patches = patches.shape[2]

        tokens = self.mae.patch_embed.linear(patches).flatten(1, 2)
        pos_emb = self.mae.pos_enc(xyz, num_patches).flatten(1, 2)

        encoded, _ = self.mae.encoder(tokens + pos_emb)
        return encoded
