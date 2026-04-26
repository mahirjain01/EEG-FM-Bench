"""
ModelAdapter — the only interface a foundation model must implement to be
evaluated by UniformEvalBench.

Quick-start
-----------
Subclass ModelAdapter, implement the three abstract methods, and pass an
instance to the evaluation functions:

    class MyFM(ModelAdapter):
        def encode(self, batch):
            x = batch["data"].to(self.device)   # [B, C, T]
            return self.encoder(x)              # [B, D]

        def freeze_encoder(self):
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        def unfreeze_encoder(self):
            for p in self.encoder.parameters():
                p.requires_grad_(True)

    adapter = MyFM(encoder, device="cuda")

    from uniformevalbench.evaluation.knn import run_knn
    result = run_knn(adapter, train_loader, test_loader)

Axis coverage
-------------
Axis A (kNN@20)         — requires encode()
Axis B (robustness FBP) — requires encode() + freeze/unfreeze
Axis C (SPA)            — requires encode()
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ModelAdapter(ABC):
    """Abstract interface for EEG foundation models in UniformEvalBench."""

    @abstractmethod
    def encode(self, batch: dict) -> Tensor:
        """
        Run the frozen encoder and return pooled embeddings.

        Parameters
        ----------
        batch : dict
            Batch dict from the evaluation dataloader.  The key "data" holds
            the raw EEG tensor [B, C, T]; "label" holds integer class labels.

        Returns
        -------
        Tensor of shape [B, D], float32.
        Do NOT L2-normalise here — evaluation/knn.py handles that.
        """

    @abstractmethod
    def freeze_encoder(self) -> None:
        """Set all encoder parameters to requires_grad=False."""

    @abstractmethod
    def unfreeze_encoder(self) -> None:
        """Restore all encoder parameters to requires_grad=True."""

    def trainable_parameter_groups(self) -> list[dict]:
        """
        Parameter groups for the Axis B FBP optimiser.

        Default behaviour: all parameters with requires_grad=True in one group.
        Override to apply layer-wise learning rates or weight-decay schedules.
        Only needed for Axis B training — Axis A and C never call this.
        """
        params = [p for p in self._all_parameters() if p.requires_grad]
        return [{"params": params}]

    # ------------------------------------------------------------------
    # Optional helpers — subclasses may override but are not required to
    # ------------------------------------------------------------------

    def _all_parameters(self):
        """Yield all torch.nn.Parameter objects owned by this adapter."""
        for attr in vars(self).values():
            if isinstance(attr, torch.nn.Module):
                yield from attr.parameters()
