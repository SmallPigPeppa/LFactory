#!/usr/bin/env python
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auto-generate LlamaFactory training configs for the multi-teacher distillation pipeline.

Reads the purification output directory and generates:
  - SFT YAML  (trains on consensus_sft.jsonl)
  - DPO YAML  (trains on conflict_dpo.jsonl, loads SFT adapter)
  - Merge YAML (merges SFT + DPO adapters)
  - mergekit YAML (optional DARE-TIES merge for multi-specialist)

Usage:
    python scripts/gen_distill_configs.py \
        --student Qwen/Qwen2.5-1.5B-Instruct \
        --data-dir data/purified \
        --out-dir examples/distillation/auto \
        --tag my_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml  # xray: ignore[SEC-015]


def _sft_config(student: str, dataset_name: str, tag: str, cpu_safe: bool, early_stopping_patience: int = 0) -> dict:
    cfg = {
        "model_name_or_path": student,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 16,
        "lora_target": "all",
        "dataset_dir": "data",
        "dataset": dataset_name,
        "template": "qwen",
        "cutoff_len": 1024,
        "max_samples": 10000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": f"saves/{tag}/lora/sft",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "save_only_model": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2.0e-5,
        "num_train_epochs": 2.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": not cpu_safe,
        "fp16": False,
    }
    if early_stopping_patience > 0:
        cfg["eval_strategy"] = "steps"
        cfg["eval_steps"] = 50
        cfg["load_best_model_at_end"] = True
        cfg["metric_for_best_model"] = "eval_loss"
        cfg["greater_is_better"] = False
        cfg["early_stopping_patience"] = early_stopping_patience
    return cfg


def _dpo_config(student: str, dataset_name: str, tag: str, cpu_safe: bool, early_stopping_patience: int = 0) -> dict:
    cfg = {
        "model_name_or_path": student,
        "adapter_name_or_path": f"saves/{tag}/lora/sft",
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 16,
        "lora_target": "all",
        "pref_beta": 0.1,
        "pref_loss": "sigmoid",
        "dataset_dir": "data",
        "dataset": dataset_name,
        "template": "qwen",
        "cutoff_len": 1024,
        "max_samples": 10000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": f"saves/{tag}/lora/dpo",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "save_only_model": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1.0e-5,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.0,
        "bf16": not cpu_safe,
        "fp16": False,
    }
    if early_stopping_patience > 0:
        cfg["eval_strategy"] = "steps"
        cfg["eval_steps"] = 50
        cfg["load_best_model_at_end"] = True
        cfg["metric_for_best_model"] = "eval_loss"
        cfg["greater_is_better"] = False
        cfg["early_stopping_patience"] = early_stopping_patience
    return cfg


def _merge_config(student: str, tag: str, has_dpo: bool = False) -> dict:
    adapter_path = f"saves/{tag}/lora/sft"
    if has_dpo:
        dpo_path = f"saves/{tag}/lora/dpo"
        # Validate adapter paths exist before writing merge config
        if not Path(dpo_path).exists():
            print(f"  WARNING: DPO adapter not found at {dpo_path} — merge will use SFT only")  # xray: ignore[PY-004]
            has_dpo = False
    if not Path(adapter_path).exists():
        print(f"  WARNING: SFT adapter not found at {adapter_path} — merge may fail")  # xray: ignore[PY-004]

    if has_dpo:
        adapter_path += f",saves/{tag}/lora/dpo"
    return {
        "model_name_or_path": student,
        "adapter_name_or_path": adapter_path,
        "template": "qwen",
        "trust_remote_code": True,
        "export_dir": f"saves/{tag}/merged",
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }


def _dare_ties_config(student: str, tag: str, specialists: list[str]) -> dict:
    """Generate mergekit DARE-TIES config for multi-specialist merge."""
    models = []
    for i, spec_path in enumerate(specialists):
        weight = round(1.0 / len(specialists), 2)
        models.append({
            "model": spec_path,
            "parameters": {
                "weight": weight,
                "density": 0.5,
            },
        })

    return {
        "merge_method": "dare_ties",
        "base_model": student,
        "parameters": {
            "normalize": True,
            "int8_mask": True,
        },
        "models": models,
        "dtype": "float16",
    }


def _write_yaml(cfg: dict, path: Path, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header:
            f.write(f"### {header}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Written: {path}")  # xray: ignore[PY-004]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate distillation training configs.",
        epilog="""\
examples:
  %(prog)s --student Qwen/Qwen2.5-1.5B --data-dir data/zena007/purified --tag zena007
  %(prog)s --student Qwen/Qwen2.5-1.5B --data-dir data/purified --tag test --cpu-safe --auto-register
  %(prog)s --student Qwen/Qwen2.5-1.5B --data-dir data/purified --tag test --min-dpo-samples 50
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--student", required=True, help="Student model name/path (e.g. Qwen/Qwen2.5-1.5B-Instruct).")
    parser.add_argument("--data-dir", required=True, help="Directory with purified data (consensus_sft.jsonl, conflict_dpo.jsonl).")
    parser.add_argument("--out-dir", default="examples/distillation/auto", help="Output directory for YAML configs.")
    parser.add_argument("--tag", default="distill_auto", help="Run tag for output directories.")
    parser.add_argument("--cpu-safe", action="store_true", help="Disable bf16 for CPU-only machines.")
    parser.add_argument("--sft-dataset-name", default="", help="Override SFT dataset name in dataset_info.json.")
    parser.add_argument("--dpo-dataset-name", default="", help="Override DPO dataset name in dataset_info.json.")
    parser.add_argument("--specialists", nargs="*", help="Specialist model paths for DARE-TIES merge (optional).")
    parser.add_argument("--auto-register", action="store_true", help="Auto-register datasets in data/dataset_info.json.")
    parser.add_argument("--min-dpo-samples", type=int, default=20, help="Minimum DPO samples to include DPO training (default: 20).")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled). Stops training when eval_loss doesn't improve.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Determine dataset names
    sft_name = args.sft_dataset_name or f"{args.tag}_consensus_sft"
    dpo_name = args.dpo_dataset_name or f"{args.tag}_conflict_dpo"

    # Check purified data exists
    sft_data = data_dir / "consensus_sft.jsonl"
    dpo_data = data_dir / "conflict_dpo.jsonl"

    has_sft = sft_data.exists() and sft_data.stat().st_size > 0
    has_dpo = dpo_data.exists() and dpo_data.stat().st_size > 0

    # Weak DPO detection: if too few samples, skip DPO to avoid overfitting on noise
    dpo_sample_count = 0
    if has_dpo:
        dpo_sample_count = sum(1 for line in dpo_data.read_text(encoding="utf-8").splitlines() if line.strip())
        if dpo_sample_count < args.min_dpo_samples:
            print(f"  Weak DPO detected: only {dpo_sample_count} samples (< {args.min_dpo_samples} minimum) — skipping DPO")  # xray: ignore[PY-004]
            has_dpo = False

    print(f"Purified data: SFT={'yes' if has_sft else 'NO'}, DPO={'yes' if has_dpo else 'NO'} ({dpo_sample_count} samples)")  # xray: ignore[PY-004]

    # Generate configs
    if has_sft:
        sft_cfg = _sft_config(args.student, sft_name, args.tag, args.cpu_safe, args.early_stopping_patience)
        _write_yaml(sft_cfg, out_dir / f"{args.tag}_sft.yaml", "Auto-generated SFT config")

    if has_dpo:
        dpo_cfg = _dpo_config(args.student, dpo_name, args.tag, args.cpu_safe, args.early_stopping_patience)
        _write_yaml(dpo_cfg, out_dir / f"{args.tag}_dpo.yaml", "Auto-generated DPO config")

    if has_sft or has_dpo:
        merge_cfg = _merge_config(args.student, args.tag, has_dpo)
        _write_yaml(merge_cfg, out_dir / f"{args.tag}_merge.yaml", "Auto-generated Merge config")

    if args.specialists:
        dare_cfg = _dare_ties_config(args.student, args.tag, args.specialists)
        _write_yaml(dare_cfg, out_dir / f"{args.tag}_dare_ties.yaml", "Auto-generated DARE-TIES mergekit config")

    # Auto-register datasets in dataset_info.json
    if args.auto_register:
        ds_path = Path("data/dataset_info.json")
        try:
            ds_info = json.loads(ds_path.read_text(encoding="utf-8")) if ds_path.exists() else {}
        except (json.JSONDecodeError, ValueError):
            ds_info = {}
        if has_sft and sft_name not in ds_info:
            ds_info[sft_name] = {
                "file_name": str(sft_data),
                "columns": {"prompt": "instruction", "response": "output"},
            }
            print(f"  Auto-registered: {sft_name}")  # xray: ignore[PY-004]
        if has_dpo and dpo_name not in ds_info:
            ds_info[dpo_name] = {
                "file_name": str(dpo_data),
                "ranking": True,
                "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
            }
            print(f"  Auto-registered: {dpo_name}")  # xray: ignore[PY-004]
        ds_path.write_text(json.dumps(ds_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print dataset_info.json registration snippet
    print(f"\n=== Add to data/dataset_info.json ===")  # xray: ignore[PY-004]
    reg: dict = {}
    if has_sft:
        reg[sft_name] = {
            "file_name": str(sft_data),
            "columns": {"prompt": "instruction", "response": "output"},
        }
    if has_dpo:
        reg[dpo_name] = {
            "file_name": str(dpo_data),
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        }
    print(json.dumps(reg, indent=2))  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
