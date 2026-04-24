#!/usr/bin/env bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export FORCE_TORCHRUN=1
export PYTHONPATH=src:$PYTHONPATH

python -m llamafactory.cli train examples/train_lora/qwen3vl_2b_stream.yaml
