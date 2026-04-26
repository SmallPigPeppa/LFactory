import json
import math
import os
import time
from pathlib import Path

import torch
from transformers.trainer_utils import IntervalStrategy

from ..extras import logging
from .callbacks import SavePretrainedCallback
from .lightning_compat import CSVLogger, TensorBoardLogger, WandbLogger, pl

logger = logging.get_logger(__name__)


def _report_to_list(report_to):
    if report_to is None:
        return []
    if isinstance(report_to, str):
        if report_to.lower() in {"none", "null", "false"}:
            return []
        return [report_to]
    return list(report_to)


def get_lightning_precision(training_args, finetuning_args):
    if getattr(finetuning_args, "pure_bf16", False):
        return "bf16-true"
    if getattr(training_args, "bf16", False):
        return "bf16-mixed"
    if getattr(training_args, "fp16", False):
        return "16-mixed"
    return "32-true"


def get_lightning_logger(training_args):
    report_to = _report_to_list(getattr(training_args, "report_to", None))
    if not report_to:
        return False
    os.makedirs(training_args.output_dir, exist_ok=True)
    if "wandb" in report_to:
        return WandbLogger(project=os.getenv("WANDB_PROJECT", "llamafactory"), save_dir=training_args.output_dir)
    if "tensorboard" in report_to:
        return TensorBoardLogger(save_dir=training_args.output_dir, name="tensorboard")
    if "csv" in report_to:
        return CSVLogger(save_dir=training_args.output_dir, name="csv_logs")
    return False


def get_resume_ckpt_path(training_args):
    checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    if checkpoint is None:
        return None
    if os.path.isdir(checkpoint):
        ckpt = os.path.join(checkpoint, "lightning.ckpt")
        if os.path.isfile(ckpt):
            return ckpt
        logger.warning_rank0(
            f"resume_from_checkpoint={checkpoint!r} does not contain lightning.ckpt; "
            "Lightning will start a fresh optimizer/scheduler state."
        )
        return None
    if os.path.isfile(checkpoint):
        return checkpoint
    return None


def _interval_strategy(value):
    try:
        return IntervalStrategy(value)
    except Exception:
        return value


def should_validate_during_fit(training_args):
    if not getattr(training_args, "do_eval", False):
        return False
    eval_strategy = _interval_strategy(getattr(training_args, "eval_strategy", getattr(training_args, "evaluation_strategy", "no")))
    return eval_strategy in {IntervalStrategy.STEPS, IntervalStrategy.EPOCH}


def build_lightning_trainer(training_args, finetuning_args, data_module, callbacks=None, enable_validation_during_fit=False):
    callbacks = list(callbacks or [])
    if getattr(training_args, "do_train", False):
        callbacks.append(SavePretrainedCallback(training_args))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = get_lightning_precision(training_args, finetuning_args)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    strategy = "ddp" if world_size > 1 else "auto"
    max_steps = data_module.max_train_steps() if getattr(training_args, "do_train", False) else -1

    eval_strategy = _interval_strategy(getattr(training_args, "eval_strategy", getattr(training_args, "evaluation_strategy", "no")))
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1 if world_size > 1 else "auto",
        strategy=strategy,
        precision=precision,
        max_steps=max_steps,
        max_epochs=-1 if max_steps and max_steps > 0 else int(math.ceil(float(getattr(training_args, "num_train_epochs", 1.0)))),
        accumulate_grad_batches=max(1, int(getattr(training_args, "gradient_accumulation_steps", 1))),
        gradient_clip_val=float(getattr(training_args, "max_grad_norm", 0.0) or 0.0),
        log_every_n_steps=max(1, int(getattr(training_args, "logging_steps", 1) or 1)),
        callbacks=callbacks,
        logger=get_lightning_logger(training_args),
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=False,
        default_root_dir=training_args.output_dir,
        num_sanity_val_steps=0,
    )

    if enable_validation_during_fit and data_module.eval_dataset is not None:
        if eval_strategy == IntervalStrategy.STEPS:
            trainer_kwargs["val_check_interval"] = max(1, int(getattr(training_args, "eval_steps", None) or getattr(training_args, "logging_steps", 1) or 1))
            trainer_kwargs["check_val_every_n_epoch"] = None
        elif eval_strategy == IntervalStrategy.EPOCH:
            trainer_kwargs["check_val_every_n_epoch"] = 1

    return pl.Trainer(**trainer_kwargs)


def _normalise_metrics(metrics):
    if metrics is None:
        return {}
    if isinstance(metrics, list):
        merged = {}
        for idx, item in enumerate(metrics):
            for key, value in (item or {}).items():
                target = key if len(metrics) == 1 else f"{key}_{idx}"
                merged[target] = value
        metrics = merged
    normalised = {}
    for key, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().float().item() if value.numel() == 1 else value.detach().cpu().tolist()
        normalised[key] = value
    return normalised


def log_metrics(split, metrics):
    clean = _normalise_metrics(metrics)
    if clean:
        logger.info_rank0(f"{split} metrics: {json.dumps(clean, ensure_ascii=False, sort_keys=True)}")
    return clean


def save_metrics(training_args, split, metrics):
    if not getattr(training_args, "should_save", True):
        return
    os.makedirs(training_args.output_dir, exist_ok=True)
    clean = _normalise_metrics(metrics)
    path = os.path.join(training_args.output_dir, f"{split}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2, sort_keys=True)
    all_path = os.path.join(training_args.output_dir, "all_results.json")
    all_metrics = {}
    if os.path.exists(all_path):
        try:
            with open(all_path, encoding="utf-8") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}
    all_metrics.update({f"{split}_{k}": v for k, v in clean.items()})
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, sort_keys=True)


def train_with_metrics(trainer, lightning_module, data_module, training_args):
    started_at = time.time()
    original_eval_dataset = data_module.eval_dataset
    original_eval_names = data_module.eval_dataset_names
    if not should_validate_during_fit(training_args):
        data_module.eval_dataset = None
        data_module.eval_dataset_names = []
    try:
        trainer.fit(lightning_module, datamodule=data_module, ckpt_path=get_resume_ckpt_path(training_args))
    finally:
        data_module.eval_dataset = original_eval_dataset
        data_module.eval_dataset_names = original_eval_names
    runtime = max(time.time() - started_at, 1e-8)
    metrics = _normalise_metrics(trainer.callback_metrics)
    metrics.update(
        train_runtime=round(runtime, 4),
        train_steps=int(trainer.global_step),
        epoch=float(trainer.current_epoch),
    )
    train_dataset = data_module.train_dataset
    if train_dataset is not None:
        samples = len(train_dataset)
        metrics["train_samples_per_second"] = round(samples / runtime, 4)
    return metrics


def validate_with_metrics(trainer, lightning_module, data_module):
    results = trainer.validate(lightning_module, datamodule=data_module, verbose=False)
    return _normalise_metrics(results)


def predict_with_outputs(trainer, lightning_module, data_module):
    return trainer.predict(lightning_module, datamodule=data_module, return_predictions=True)


def create_modelcard_and_push(lightning_module, model_args, data_args, training_args, finetuning_args):
    if not getattr(training_args, "do_train", False):
        return
    output_dir = Path(training_args.output_dir)
    if not getattr(training_args, "should_save", True):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ", ".join(data_args.dataset or []) if data_args.dataset is not None else "not specified"
    readme = output_dir / "README.md"
    readme.write_text(
        "---\n"
        "license: other\n"
        f"base_model: {model_args.model_name_or_path}\n"
        "tags:\n"
        "- llama-factory-slim-core\n"
        f"- {finetuning_args.finetuning_type}\n"
        "---\n\n"
        "# Fine-tuned model\n\n"
        f"- Base model: `{model_args.model_name_or_path}`\n"
        f"- Dataset: `{dataset}`\n"
        f"- Fine-tuning type: `{finetuning_args.finetuning_type}`\n"
        "- Trainer: PyTorch Lightning\n",
        encoding="utf-8",
    )
    if getattr(training_args, "push_to_hub", False):
        repo_id = getattr(training_args, "hub_model_id", None) or output_dir.name
        token = getattr(training_args, "hub_token", None)
        private = getattr(training_args, "hub_private_repo", None)
        try:
            lightning_module.model.push_to_hub(repo_id=repo_id, token=token, private=private)
            if lightning_module.tokenizer is not None:
                lightning_module.tokenizer.push_to_hub(repo_id=repo_id, token=token, private=private)
            if lightning_module.processor is not None:
                lightning_module.processor.push_to_hub(repo_id=repo_id, token=token, private=private)
        except Exception as exc:
            logger.warning_rank0(f"Failed to push model to hub: {exc}")
