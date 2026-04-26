from typing import Any, Dict, List

import mne
import numpy as np
import torch
from datasets import Dataset

from baseline.abstract.adapter import AbstractDataLoaderFactory, AbstractDatasetAdapter
from common.utils import ElectrodeSet


class MANASDatasetAdapter(AbstractDatasetAdapter):
    def __init__(
        self,
        dataset: Dataset,
        dataset_names: List[str],
        dataset_configs: List[str],
        target_fs: int = 200,
    ):

        self.model_name = "MANAS"

        self.target_fs = target_fs
        self.electrode_set = ElectrodeSet()

        super().__init__(dataset, dataset_names, dataset_configs)

    def get_supported_channels(self):
        return self.electrode_set.Electrodes

    def _resample(self, data: torch.Tensor, fs: int | None) -> torch.Tensor:
        if fs is None or fs == self.target_fs:
            return data

        _data = data.detach().cpu().numpy().astype(np.float64, copy=False)
        if fs < self.target_fs:
            _data = mne.filter.resample(_data, up=self.target_fs / fs, verbose=False)
        else:
            _data = mne.filter.resample(_data, down=fs / self.target_fs, verbose=False)

        return torch.as_tensor(_data, dtype=torch.float32)

    @staticmethod
    def _build_coords(channel_names: List[str]) -> torch.Tensor:
        mne_info = mne.create_info(ch_names=channel_names, sfreq=1.0, ch_types="eeg")
        raw = mne.io.RawArray(
            np.zeros((len(channel_names), 1)), info=mne_info, verbose=False
        )

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False)

        coords = raw.get_montage().get_positions()["ch_pos"].values()
        _coords = np.asarray(list(coords), dtype=np.float32)

        return 100 * torch.from_numpy(_coords).float()

    @staticmethod
    def _z_score(data: torch.Tensor) -> torch.Tensor:
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        return (data - mean) / (std + 1e-6)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        result = super()._process_sample(sample)

        ch_indices = result["chs"].tolist()
        ch_names = self.electrode_set.get_electrodes_name(ch_indices)

        fs = sample.get("fs")
        if isinstance(fs, torch.Tensor) or isinstance(fs, np.ndarray):
            fs = int(fs.item())

        data = result["data"]
        data = self._resample(data, fs)
        data = self._z_score(data)

        result["coords"] = self._build_coords(ch_names)
        result["data"] = data

        return result


class MANASDataLoaderFactory(AbstractDataLoaderFactory):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        target_fs: int = 200,
    ):
        super().__init__(batch_size, num_workers, seed)

        self.target_fs = target_fs

    def create_adapter(
        self, dataset: Dataset, dataset_names: List[str], dataset_configs: List[str]
    ) -> MANASDatasetAdapter:
        return MANASDatasetAdapter(
            dataset=dataset,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
            target_fs=self.target_fs,
        )
