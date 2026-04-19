#!/usr/bin/env bash

WANDB_PROJECT=mllm-lightning python src/train.py examples/train_lora/llava15_lora_llava_next_data.yaml \
  dataset_dir=data/llava_779k_demo_10000 \
  dataset=llava_779k_demo_10000_parquet \
  output_dir=saves/llava-1.5-7b/lora/llava-779k-demo-10000 \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=8 \
  num_train_epochs=1.0 \
  learning_rate=2e-4 \
  warmup_ratio=0.03 \
  val_size=0.001 \
  bf16=true \
  fp16=false
