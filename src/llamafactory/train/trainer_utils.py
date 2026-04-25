# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from typing import Optional

import torch
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from ..extras import logging


from transformers import PreTrainedModel
from ..hparams import DataArguments, FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def create_modelcard_and_push(
    trainer: "Trainer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    kwargs = {
        "tasks": "text-generation",
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["llama-factory-slim", finetuning_args.finetuning_type],
    }
    if data_args.dataset is not None:
        kwargs["dataset"] = data_args.dataset
    if training_args.do_train and training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    elif training_args.do_train:
        Trainer.create_model_card(trainer, license="other", **kwargs)


def _get_decay_parameter_names(model: "PreTrainedModel") -> list[str]:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    return [name for name in decay_parameters if "bias" not in name]


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> "torch.optim.Optimizer":
    default_lr = training_args.learning_rate
    loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    embedding_lr = finetuning_args.loraplus_lr_embedding
    decay_param_names = _get_decay_parameter_names(model)
    param_dict: dict[str, list[torch.nn.Parameter]] = {"lora_a": [], "lora_b": [], "lora_b_nodecay": [], "embedding": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_embedding_B" in name:
            param_dict["embedding"].append(param)
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_param_names:
                param_dict["lora_b"].append(param)
            else:
                param_dict["lora_b_nodecay"].append(param)
        else:
            param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optim_class(
        [
            dict(params=param_dict["lora_a"], lr=default_lr, weight_decay=training_args.weight_decay),
            dict(params=param_dict["lora_b"], lr=loraplus_lr, weight_decay=training_args.weight_decay),
            dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
            dict(params=param_dict["embedding"], lr=embedding_lr, weight_decay=training_args.weight_decay),
        ],
        **optim_kwargs,
    )
    logger.info_rank0(f"Using LoRA+ optimizer with loraplus lr ratio {finetuning_args.loraplus_lr_ratio:.2f}.")
    return optimizer


def create_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Optional["torch.optim.Optimizer"]:
    if finetuning_args.loraplus_lr_ratio is not None:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)
    return None


def create_custom_scheduler(
    training_args: "TrainingArguments",
    num_training_steps: int,
    optimizer: Optional["torch.optim.Optimizer"] = None,
) -> None:
    if training_args.lr_scheduler_type == "warmup_stable_decay":
        num_warmup_steps = training_args.get_warmup_steps(num_training_steps)
        remaining_steps = num_training_steps - num_warmup_steps
        num_stable_steps = remaining_steps // 3
        num_decay_steps = remaining_steps - num_stable_steps
        scheduler_kwargs = training_args.lr_scheduler_kwargs or {}
        scheduler_kwargs.setdefault("num_stable_steps", num_stable_steps)
        scheduler_kwargs.setdefault("num_decay_steps", num_decay_steps)
        training_args.lr_scheduler_kwargs = scheduler_kwargs
