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

"""Bayesian Hyperparameter Search for Student Forge.

Uses Optuna TPE sampler to optimise LoRA rank, learning rate, and epoch count
across Forge Matrix variants. Each Optuna trial runs one complete training variant
and returns the final SFT loss as the objective.

Usage:
    python scripts/bayesian_forge.py \
        --base-matrix data/forge_matrix/zena007_matrix.yaml \
        --tag zena007_bayes \
        --n-trials 20 \
        --study-name zena007_hyperparam \
        --py .venv-py314/Scripts/python.exe

Requires: pip install optuna  (optional dependency)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    raise SystemExit(1)


def _run_single_trial(
    trial_id: str,
    model: str,
    lora_rank: int,
    learning_rate: float,
    num_epochs: int,
    tag: str,
    sft_dataset: str,
    template: str,
    cpu_safe: bool,
    py: str,
) -> float | None:
    """Run a single SFT training variant and return final loss (or None on failure)."""
    import subprocess

    cfg = {
        "model_name_or_path": model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        "dataset_dir": "data",
        "dataset": sft_dataset,
        "template": template,
        "cutoff_len": 1024,
        "max_samples": 10000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": f"saves/{tag}/bayesian/{trial_id}/lora/sft",
        "logging_steps": 5,
        "save_steps": 500,
        "plot_loss": False,
        "overwrite_output_dir": True,
        "save_only_model": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": not cpu_safe,
        "fp16": False,
    }

    yaml_path = Path(f"saves/{tag}/bayesian/{trial_id}/config.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    proc = subprocess.run(
        [py, "-m", "llamafactory.cli", "train", str(yaml_path)],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        return None

    # Read final loss from trainer_log.jsonl
    log_path = Path(cfg["output_dir"]) / "trainer_log.jsonl"
    if not log_path.exists():
        return None

    last_loss = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if "loss" in entry:
            last_loss = entry["loss"]
    return last_loss


def main() -> int:
    if optuna is None:
        print("ERROR: optuna not installed. Run: pip install optuna", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Bayesian hyperparameter search for Student Forge.")
    parser.add_argument("--base-matrix", required=True, help="Base forge matrix YAML (used for model/template/data).")
    parser.add_argument("--tag", default="bayes", help="Run tag for saves directory.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--study-name", default="forge_hyperparam", help="Optuna study name.")
    parser.add_argument("--py", default=".venv-py314/Scripts/python.exe", help="Python interpreter.")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Min learning rate.")
    parser.add_argument("--lr-max", type=float, default=1e-4, help="Max learning rate.")
    parser.add_argument("--rank-choices", type=str, default="8,16,32,64", help="Comma-separated LoRA rank choices.")
    parser.add_argument("--epoch-min", type=int, default=1, help="Min epochs.")
    parser.add_argument("--epoch-max", type=int, default=5, help="Max epochs.")
    parser.add_argument("--timeout", type=int, default=0, help="Total search timeout in seconds (0=unlimited).")
    args = parser.parse_args()

    matrix = yaml.safe_load(Path(args.base_matrix).read_text(encoding="utf-8"))
    # Use first variant's model as default
    first_variant = next(iter(matrix.get("variants", {}).values()), {})
    model = first_variant.get("model", "")
    template = matrix.get("template", "qwen")
    cpu_safe = matrix.get("cpu_safe", True)
    sft_data = matrix.get("sft_data", "")
    tag = args.tag

    # Derive SFT dataset name from tag
    sft_ds_name = f"{tag}_forge_train_sft"

    rank_choices = [int(r.strip()) for r in args.rank_choices.split(",")]

    print(f"=== Bayesian Hyperparameter Search ===")
    print(f"Model:      {model}")
    print(f"Trials:     {args.n_trials}")
    print(f"LR range:   [{args.lr_min}, {args.lr_max}]")
    print(f"Rank opts:  {rank_choices}")
    print(f"Epoch range: [{args.epoch_min}, {args.epoch_max}]")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        lora_rank = trial.suggest_categorical("lora_rank", rank_choices)
        lr = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
        epochs = trial.suggest_int("num_train_epochs", args.epoch_min, args.epoch_max)

        trial_id = f"trial_{trial.number:03d}"
        print(f"\n--- Trial {trial.number}: rank={lora_rank}, lr={lr:.2e}, epochs={epochs} ---")

        t0 = time.time()
        loss = _run_single_trial(
            trial_id=trial_id,
            model=model,
            lora_rank=lora_rank,
            learning_rate=lr,
            num_epochs=epochs,
            tag=tag,
            sft_dataset=sft_ds_name,
            template=template,
            cpu_safe=cpu_safe,
            py=args.py,
        )
        elapsed = time.time() - t0

        if loss is None:
            print(f"  Trial {trial.number} FAILED ({elapsed:.0f}s)")
            raise optuna.TrialPruned()

        print(f"  Trial {trial.number} => loss={loss:.4f} ({elapsed:.0f}s)")
        return loss

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best loss:  {study.best_trial.value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")

    # Save results
    results_path = Path(f"saves/{tag}/bayesian/search_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = {
        "best_trial": study.best_trial.number,
        "best_loss": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    results_path.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
