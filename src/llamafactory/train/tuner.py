# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from typing import Any, Optional

import torch.distributed as dist
from transformers import EarlyStoppingCallback

from ..extras import logging
from ..hparams import get_train_args, read_args
from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from .pt import run_pt
from .sft import run_sft


from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks") or []
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())
    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("This slim build only supports `stage: pt` and `stage: sft`.")

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        logger.warning(f"Failed to destroy process group: {exc}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)
    _training_function({"args": args, "callbacks": callbacks or []})
