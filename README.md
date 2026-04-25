# LFactory slim 779k trainer

This package is a train-only slim build of the uploaded LFactory/LLaMA-Factory project.
It keeps the training path used by `train_llava779k_qwen3vl2b.sh` and removes unrelated compatibility code.

## Kept

- CLI entry: `python -m llamafactory.cli train ...`
- Stages: `sft` and `pt` only
- Image VLM families/templates: LLaVA, Qwen-VL/Qwen3-VL, InternVL
- 779k-style local parquet datasets declared through `dataset_info.json`
- ShareGPT conversations with image column mapping, tokenized dataset save/load, DDP/torchrun, LoRA/OFT/full/freeze, quantization hooks, `plot_loss`, and WandB reporting

## Removed

- Video/audio multimodal code paths
- Non-parquet/raw/cloud dataset loaders and non-779k formatting branches
- API/chat/WebUI/eval/inference engines
- DPO/KTO/PPO/RM/value-head training paths
- Ray/MCA/HyperParallel/KTransformers/Unsloth/MoD compatibility paths

## Expected command

```bash
bash train_llava779k_qwen3vl2b.sh
```

The example config is under `examples/train_lora/llava779k_qwen3vl2b/`.
