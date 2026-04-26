#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/scale10final_checkpoint_sweep.yaml}"
SWEEP_METHODS="${SWEEP_METHODS:-linear_probe,full_finetune}"

PYTHONPATH="$PWD" "$PYTHON_BIN" baseline_main.py \
  conf_file="$CONFIG_PATH" \
  sweep=true \
  sweep_methods="$SWEEP_METHODS"
