#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_PROJECT="${WANDB_PROJECT:-mllm-lightning}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-llava-hf/llava-1.5-7b-hf}"
DATASET_DIR="${DATASET_DIR:-data/llava_next_train}"
DATASET="${DATASET:-llava_next_data_train}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/llava-1.5-7b/lora/llava-next-data-streaming}"
RUN_NAME="${RUN_NAME:-llava15-lora-next-data-streaming-8gpu}"
REPORT_TO="${REPORT_TO:-none}"
MAX_STEPS="${MAX_STEPS:-10000}"
VAL_SIZE="${VAL_SIZE:-1000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  src/train.py examples/train_lora/llava15_lora_next_data_streaming.yaml \
  model_name_or_path="${MODEL_NAME_OR_PATH}" \
  dataset_dir="${DATASET_DIR}" \
  dataset="${DATASET}" \
  output_dir="${OUTPUT_DIR}" \
  run_name="${RUN_NAME}" \
  report_to="${REPORT_TO}" \
  max_steps="${MAX_STEPS}" \
  val_size="${VAL_SIZE}" \
  per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  "$@"
