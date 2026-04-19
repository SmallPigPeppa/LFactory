#!/usr/bin/env bash

WANDB_PROJECT=mllm-lightning python src/train.py examples/train_lora/llava15_lora_mllm_demo.yaml
