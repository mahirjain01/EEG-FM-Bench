#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.." || exit

PYTHON_BIN="${PYTHON_BIN:-python3}"
SWEEP_METHODS="${SWEEP_METHODS:-linear_probe,full_finetune}"

CONFIGS=(
  "configs/cosine_split_mae.yaml"
  "configs/manas.yaml"
  "configs/ndx_mae.yaml"
)

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
fi

for config in "${CONFIGS[@]}"; do
  echo "==> Running sweep for ${config}"
  PYTHONPATH="$PWD" "$PYTHON_BIN" baseline_main.py \
    conf_file="$config" \
    sweep=true \
    sweep_methods="$SWEEP_METHODS"
done
