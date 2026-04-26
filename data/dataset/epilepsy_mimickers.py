import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, Any

import datasets
import mne
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


@dataclass
class EpilepsyMimickersConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "Routine clinical EEG from 80 subjects: 50 confirmed epilepsy and 30 epilepsy mimickers "
        "(non-epileptic conditions presenting similarly to epilepsy). Recorded using the international "
        "10-20 system at 125 Hz with a linked-ear reference. Binary classification: Epileptic vs Mimickers."
    )
    citation: Optional[str] = ""

    filter_notch: float = 50.0

    dataset_name: Optional[str] = 'epilepsy_mimickers'
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
                   'Fp1', 'Fp2',
            'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
                    'O1', 'O2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 10
    suffix_path: str = 'epilepsy_mimickers_dataset'
    scan_sub_dir: str = ''

    category: list[str] = field(default_factory=lambda: ['Epileptic', 'Mimickers'])


class EpilepsyMimickersBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = EpilepsyMimickersConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True),
    ]

    # Explicit reproducible splits — last ~10% per class for test, next ~10% for valid
    _EPILEPTIC_TEST  = {'E046', 'E047', 'E048', 'E049', 'E050'}
    _EPILEPTIC_VALID = {'E041', 'E042', 'E043', 'E044', 'E045'}
    _MIMICKER_TEST   = {'M028', 'M029', 'M030'}
    _MIMICKER_VALID  = {'M025', 'M026', 'M027'}

    def __init__(self, config_name='pretrain', **kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        csv_path = os.path.join(self.config.raw_path, 'ft_data_manifest.csv')
        if not os.path.exists(csv_path):
            # Not available when loading from processed cache
            self.sub_meta: DataFrame = pd.DataFrame()
            return
        self.sub_meta: DataFrame = pd.read_csv(csv_path)

    def _walk_raw_data_files(self):
        raw_data_files = []
        root = self.config.raw_path
        for subdir in ('Epileptic', 'Mimickers'):
            subdir_path = os.path.join(root, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in sorted(os.listdir(subdir_path)):
                if fname.lower().endswith('.edf'):
                    raw_data_files.append(os.path.join(subdir_path, fname))
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        sub_id = os.path.basename(file_path)[:-4]  # "E001" or "M001"
        return {
            'subject': sub_id,
            'session': 1,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        sub_id = info['subject']

        row = self.sub_meta[self.sub_meta['Anonymised_ID'] == sub_id]
        group = row['Class'].iloc[0] if not row.empty else (
            'Epileptic' if sub_id.startswith('E') else 'Mimickers'
        )

        with self._read_raw_data(file_path, preload=False) as raw:
            time = raw.times[-1] + raw.times[1]

        info.update({
            'montage': '10_20',
            'time': time,
            'group': group,
        })
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [('default', 0, -1)]
        # Whole-recording label for each subject
        return [(info['group'], 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        test_subs  = self._EPILEPTIC_TEST  | self._MIMICKER_TEST
        valid_subs = self._EPILEPTIC_VALID | self._MIMICKER_VALID

        df['split'] = 'train'
        df.loc[df['subject'].isin(valid_subs), 'split'] = 'valid'
        if self.config.is_finetune:
            df.loc[df['subject'].isin(test_subs), 'split'] = 'test'
        # In pretrain mode test_ratio=0: test subjects stay in train
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache:
            return self._std_chs_cache[montage]
        chs = self.config.montage[montage]
        chs_std = [self.montage_10_20_replace_dict.get(ch.upper(), ch.upper()) for ch in chs]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(file_path, verbose=verbose, preload=preload)

            channel_mapping = {}
            for ch in raw.ch_names:
                name = ch
                if name.startswith('EEG '):
                    name = name[4:]       # strip 'EEG ' prefix (with space)
                if name.endswith('-REF'):
                    name = name[:-4]      # strip '-REF' suffix
                channel_mapping[ch] = name

            raw.rename_channels(channel_mapping)
            raw.pick(self.config.montage['10_20'])
            return raw


if __name__ == "__main__":
    builder = EpilepsyMimickersBuilder('finetune')
    builder.preproc(n_proc=4)
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
    print(dataset)
