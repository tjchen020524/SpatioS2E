#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/hippocampus_spatios2e.yaml}"
CKPT_PATH="outputs/hippocampus_spatios2e/checkpoints/best.pt"
RESULTS_DIR="outputs/hippocampus_spatios2e/results"

spatios2e-compute-residual-scale "${CONFIG_PATH}"
spatios2e-train --config "${CONFIG_PATH}"
spatios2e-eval \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --split test \
  --save-dir "${RESULTS_DIR}"
