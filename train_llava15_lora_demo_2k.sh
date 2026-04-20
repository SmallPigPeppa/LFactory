#!/usr/bin/env bash

WANDB_PROJECT=CL-debug python src/train.py examples/train_lora/llava15_lora_next_data.yaml \
  dataset_dir=data/llava_779k_demo \
  dataset=demo_2000 \
  output_dir=saves/llava-1.5-7b/lora/llava-779k-demo-2k \
  run_name=llava15-lora-779k-demo-2k \
  report_to=wandb \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=1 \
  num_train_epochs=1.0 \
  learning_rate=2e-4 \
  warmup_ratio=0.03 \
  val_size=0.001 \
  bf16=true \
  fp16=false
