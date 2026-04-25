# LFactory slim 779k trainer

This build is intentionally narrowed to the training path used by `train_llava779k_qwen3vl2b.sh`.

Kept:

- CLI training entry: `python -m llamafactory.cli train ...`
- stages: `sft`, `pt`
- image VLM families/templates: LLaVA, Qwen-VL/Qwen3-VL, InternVL
- 779k-style ShareGPT parquet loading via `dataset_info.json` + `file_name`
- LoRA/OFT/full/freeze tuning, quantization hooks, DDP/torchrun, plot_loss, tokenized dataset save/load

Removed:

- video/audio multimodal code paths
- Alpaca/OpenAI/raw/cloud dataset formats; only local parquet is supported
- API, chat/inference engines, WebUI, eval, Ray, v1 launcher
- RM/PPO/DPO/KTO/MCA/HyperParallel training stages
- KTransformers, Unsloth, value-head, mixture-of-depths paths

Expected command:

```bash
bash train_llava779k_qwen3vl2b.sh
```
