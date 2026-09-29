#!/usr/bin/env bash
# Run the full crop classification pipeline from data inspection to output figures.
# Usage (from the repository root, with the virtual environment active):
#   bash run_all.sh [path/to/config.yaml]
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"
echo "Using config: ${CONFIG}"

steps=(data_loading eda "train_test_split --compare" preprocessing train evaluate visualization)
for step in "${steps[@]}"; do
  echo
  echo "=============================================="
  echo " python -m src.${step}"
  echo "=============================================="
  # shellcheck disable=SC2086
  python -m src.${step} --config "${CONFIG}"
done

echo
echo "Done. Figures in outputs/figures/, metrics in outputs/metrics/."