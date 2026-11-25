#!/usr/bin/env bash
# Launch M5-TTS training with Accelerate.
# Environment overrides: CONFIG, EXP_DIR, NUM_PROCESSES, CUDA_VISIBLE_DEVICES.
# Extra arguments are forwarded to train.py, e.g. ./train.sh --total_steps 100.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch \
    --num_machines 1 \
    --num_processes "${NUM_PROCESSES:-1}" \
    train.py \
    --config "${CONFIG:-m5/configs/train_example.json}" \
    --exp_dir "${EXP_DIR:-outputs/train_demo}" \
    "$@"
