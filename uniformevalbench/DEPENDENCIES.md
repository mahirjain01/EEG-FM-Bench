# External Dependencies

`uniformevalbench/` is self-contained except for one external dependency: the dataset loading layer.

## `data/` — Dataset Loading (Required)

The FM trainers in `uniformevalbench/models/abstract/` call into `data.processor.wrapper` to
load EEG datasets at runtime. This module is NOT included in `uniformevalbench/` because it
pulls in the entire dataset builder tree (20+ datasets, raw BIDS preprocessing pipelines, etc.).

**What it provides:**
- `load_concat_eeg_datasets(cfg, split, ...)` — loads one or more HuggingFace-format EEG datasets
- `get_dataset_n_class(name)` — number of classes for a named dataset
- `get_dataset_montage(name)` — canonical channel list
- `get_dataset_patch_len(name)` — patch length for tokeniser alignment
- `get_dataset_category(name)` — class label names

**Where it is used:**
- `uniformevalbench/models/abstract/trainer.py` — dataset loading at training/eval time
- `uniformevalbench/models/abstract/adapter.py` — dataloader construction
- `uniformevalbench/models/abstract/classifier.py` — montage-aware channel routing
- `uniformevalbench/common/distributed/loader.py` — distributed sampler

**To run the benchmark on your own data:**
You have two options:

1. **Use the EEG-FM-Bench repo** (recommended): Clone the full repo and run
   `uniformevalbench/` from inside it. The `data/` directory will be on `sys.path`
   automatically since the repo root is added by the experiment scripts.

2. **Implement a drop-in replacement**: Provide a `data/processor/wrapper.py` module
   that implements the four functions above for your dataset. The abstract trainer only
   calls these four entry points; nothing else from `data/` is imported.

## Python Packages

Standard scientific Python stack. See the parent repo's `requirements.txt` for pinned versions.

Key packages:
- `torch >= 2.0`, `datasets` (HuggingFace), `omegaconf`, `pydantic`, `captum`
- `scipy`, `numpy`, `pandas`, `scikit-learn`
- `wandb`, `comet_ml` (optional: used by AbstractTrainer for logging; pass `--no-log` to skip)
