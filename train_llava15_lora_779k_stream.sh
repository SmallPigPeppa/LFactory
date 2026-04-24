#!/usr/bin/env bash

python src/train.py examples/train_lora/llava15_lora_779k_stream.yaml \
  model_name_or_path=llava-hf/llava-1.5-7b-hf \
  dataset_dir=data/llava_779k_stream \
  dataset=llava_779k_train \
  output_dir=saves/llava-1.5-7b/lora/llava-779k-stream \
  run_name=llava15-lora-779k-stream \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=8 \
  learning_rate=2e-4 \
  warmup_ratio=0.03 \
  max_steps=10000 \
  streaming=true \
  val_size=1000 \
  bf16=true \
  fp16=false
