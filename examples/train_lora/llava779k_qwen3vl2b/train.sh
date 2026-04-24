#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

llamafactory-cli train examples/train_lora/llava779k_qwen3vl2b/train.yaml
