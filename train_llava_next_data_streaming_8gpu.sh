#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="examples/train_lora/llava_next_data_streaming_sft.yaml"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export FORCE_TORCHRUN=1
export MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m llamafactory.cli train "${CONFIG}" "$@"
