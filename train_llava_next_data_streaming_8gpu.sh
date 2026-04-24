#!/bin/bash

set -euo pipefail
set -x

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${PROJECT_ROOT}"

CONFIG=${CONFIG:-examples/train_lora/llava_next_data_streaming_sft.yaml}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export FORCE_TORCHRUN=${FORCE_TORCHRUN:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  export NPROC_PER_NODE=${#DEVICES[@]}
fi

if [[ "${NPROC_PER_NODE}" -ne 8 ]]; then
  echo "Warning: NPROC_PER_NODE=${NPROC_PER_NODE}; expected 8 for the default 8-GPU run." >&2
fi

exec llamafactory-cli train "${CONFIG}" "$@"
