#!/bin/bash
FORCE_TORCHRUN=1 PYTHONPATH=src python -m llamafactory.cli train examples/train_lora/llava779k_qwen3vl2b/train.yaml
