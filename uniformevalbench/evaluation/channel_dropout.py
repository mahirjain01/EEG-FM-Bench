"""
Channel dropout transform for Axis B robustness evaluation.

This is a pure function equivalent to the env-var hook in
baseline/abstract/adapter.py (_axis_b_apply_channel_dropout).
New FM authors call this directly instead of setting env vars.
"""

import torch
from torch import Tensor


def drop_channels(x: Tensor, p: float, seed: int = 0) -> Tensor:
    """
    Deterministically zero a fraction of EEG channels per sample.

    Parameters
    ----------
    x    : [B, C, T] or [C, T] float tensor
    p    : fraction in [0, 1) of channels to zero per sample
    seed : base random seed; per-sample seed = seed + sample_index

    Returns
    -------
    Tensor of same shape with floor(C * p) channels zeroed per sample.
    """
    if p <= 0.0:
        return x

    single = x.dim() == 2
    if single:
        x = x.unsqueeze(0)

    B, C, T = x.shape
    n_drop = int(C * p)
    if n_drop == 0:
        return x.squeeze(0) if single else x

    out = x.clone()
    for i in range(B):
        g = torch.Generator()
        g.manual_seed(seed + i)
        drop_idx = torch.randperm(C, generator=g)[:n_drop]
        out[i, drop_idx] = 0.0

    return out.squeeze(0) if single else out
