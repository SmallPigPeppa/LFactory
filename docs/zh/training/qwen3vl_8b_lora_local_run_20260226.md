# Qwen3-VL-8B 本地 LoRA 微调与评测总结（2026-02-26）

## 1. 目标

在当前服务器上使用本地模型 `/root/Qwen3-VL-8B` 启动 LoRA 微调，跑通推理与批量评测流程，并给出可复现配置。

## 2. 环境与算力

- 时间窗口：2026-02-25 至 2026-02-26
- 机器：4 x Tesla V100S 32GB
- Python 环境：`/root/anaconda3/envs/gui_agent`
- 关键包版本（当前）：
  - `torch==2.8.0+cu128`
  - `torchvision==0.23.0+cu128`
  - `torchaudio==2.8.0+cu128`

## 3. 关键问题与处理

- 问题：`torch 2.9.x + Qwen3-VL(Conv3D)` 在 LLaMA-Factory 中被保护性拦截，训练报错。
- 处理：将 PyTorch 相关包降级到 `2.8.0+cu128`，随后训练可正常启动并完成。
- 备注：该改动发生在 `gui_agent` 环境，不属于仓库文件变更。

## 4. 本次新增配置文件

- `examples/train_lora/qwen3vl_lora_sft_local_8b_v100.yaml`
  - 本地模型路径、V100 兼容精度（`fp16`）、LoRA SFT 参数
- `examples/inference/qwen3vl_8b_lora_sft_local.yaml`
  - LoRA 推理配置（CLI/WebChat/API 可复用）
- `examples/extras/nlg_eval/qwen3vl_8b_lora_predict_local.yaml`
  - 批量预测评测配置（`do_predict: true`）

## 5. 训练执行信息

- 训练数据集：`mllm_demo, identity, alpaca_en_demo`
- 主要超参：
  - `num_train_epochs: 3.0`
  - `per_device_train_batch_size: 1`
  - `gradient_accumulation_steps: 8`
  - GPU 数：4
  - 有效总 batch size：32
  - 总步数：105
- 训练输出目录：`saves/qwen3-vl-8b/lora/sft-local`
- 最终指标（训练日志）：
  - `train_loss: 1.0227`
  - `train_runtime: 0:07:39.81`
  - `train_steps_per_second: 0.228`

## 6. 推理与评测结果

- 交互推理配置：
  - `examples/inference/qwen3vl_8b_lora_sft_local.yaml`
- 批量评测（smoke）输出：
  - `saves/qwen3-vl-8b/lora/predict-local-smoke/generated_predictions.jsonl`
  - `saves/qwen3-vl-8b/lora/predict-local-smoke/predict_results.json`
- smoke 指标：
  - `predict_bleu-4: 71.42`
  - `predict_rouge-1: 73.4518`
  - `predict_rouge-2: 60.7942`
  - `predict_rouge-l: 70.8342`

## 7. 训练后清理

已删除中间/失败残留，保留最终可推理 LoRA 文件：

- 删除：
  - 失败训练日志
  - 中间 checkpoint：`saves/qwen3-vl-8b/lora/sft-local/checkpoint-105`
  - `/tmp/torchelastic_*`、`/tmp/pymp-*` 等临时文件
- 释放空间：约 `278,684,536` bytes（约 265.8 MB）

## 8. 推荐复现命令

训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 \
  --master_addr 127.0.0.1 --master_port 55211 \
  src/llamafactory/launcher.py examples/train_lora/qwen3vl_lora_sft_local_8b_v100.yaml
```

CLI 推理：

```bash
llamafactory-cli chat examples/inference/qwen3vl_8b_lora_sft_local.yaml
```

Web 推理：

```bash
llamafactory-cli webchat examples/inference/qwen3vl_8b_lora_sft_local.yaml
```

批量评测：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/nlg_eval/qwen3vl_8b_lora_predict_local.yaml
```

## 9. Git 提交建议（待确认）

### 建议提交（本次新增且可复用）

- `docs/zh/training/qwen3vl_8b_lora_local_run_20260226.md`
- `examples/train_lora/qwen3vl_lora_sft_local_8b_v100.yaml`
- `examples/inference/qwen3vl_8b_lora_sft_local.yaml`
- `examples/extras/nlg_eval/qwen3vl_8b_lora_predict_local.yaml`

### 需要你确认后再提交（当前工作区已存在，但非本次明确产物）

- `data/dataset_info.json`（当前为已修改状态）
- `examples/train_lora/qwen3vl_lora_sft_gui_reason_hq.yaml`（当前为未跟踪状态）
- `AGENTS.md`
- `scripts/gui_agent/`
- `guidao.jpg`

### 不建议提交

- 训练产物目录：`saves/...`
- 运行日志：`logs/...`
- 环境包变更（conda/pip 层面）

