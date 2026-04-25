# LFactory 精简版 779k 训练器

这是基于上传的 LFactory/LLaMA-Factory 项目整理出的 train-only 精简版。
它保留 `train_llava779k_qwen3vl2b.sh` 已经使用的训练路径，并删除无关兼容代码。

## 保留

- CLI entry: `python -m llamafactory.cli train ...`
- Stages: `sft` and `pt` only
- Image VLM families/templates: LLaVA, Qwen-VL/Qwen3-VL, InternVL
- 779k-style local parquet datasets declared through `dataset_info.json`
- ShareGPT conversations with image column mapping, tokenized dataset save/load, DDP/torchrun, LoRA/OFT/full/freeze, quantization hooks, `plot_loss`, and WandB reporting

## 删除

- Video/audio multimodal code paths
- Non-parquet/raw/cloud dataset loaders and non-779k formatting branches
- API/chat/WebUI/eval/inference engines
- DPO/KTO/PPO/RM/value-head training paths
- Ray/MCA/HyperParallel/KTransformers/Unsloth/MoD compatibility paths

## 预期命令

```bash
bash train_llava779k_qwen3vl2b.sh
```

示例配置位于 `examples/train_lora/llava779k_qwen3vl2b/`。
