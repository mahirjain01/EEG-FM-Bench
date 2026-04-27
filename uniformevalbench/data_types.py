from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class EEGSampleBatch:
    """
    Standard batch container for UniformEvalBench evaluations.

    x             : [B, C, T] float32 — raw EEG, C channels, T timepoints
    y             : [B] int64 — class labels
    channel_names : length-C list of electrode names (e.g. ['Fp1', 'Fz', 'Cz'])
    fs            : sampling rate in Hz
    dataset_id    : dataset identifier (e.g. 'bcic_2a')
    split         : 'train' | 'validation' | 'test'
    subject_id    : per-sample subject identifiers, shape [B], if available
    """
    x:             Tensor
    y:             Tensor
    channel_names: list[str]
    fs:            float
    dataset_id:    str
    split:         str
    subject_id:    Optional[list[str]] = None

    def to(self, device) -> EEGSampleBatch:
        return EEGSampleBatch(
            x=self.x.to(device),
            y=self.y.to(device),
            channel_names=self.channel_names,
            fs=self.fs,
            dataset_id=self.dataset_id,
            split=self.split,
            subject_id=self.subject_id,
        )
