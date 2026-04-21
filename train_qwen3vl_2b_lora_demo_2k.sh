#!/usr/bin/env bash

python src/train.py examples/train_lora/qwen3vl_lora_sft.yaml \
  model_name_or_path=Qwen/Qwen3-VL-2B-Instruct \
  dataset_dir=data/llava_779k_demo \
  dataset=demo_2000 \
  max_samples=2000 \
  output_dir=saves/qwen3-vl-2b/lora/demo-2000 \
  run_name=qwen3-vl-2b-demo-2000 \
  report_to=none \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=8 \
  num_train_epochs=1.0 \
  learning_rate=2e-4 \
  warmup_ratio=0.03 \
  val_size=0.001
