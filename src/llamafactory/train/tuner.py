import torch.distributed as dist
from transformers import EarlyStoppingCallback

from ..extras import logging
from ..hparams import get_train_args, read_args
from .callbacks import LogCallback, ReporterCallback
from .pt import run_pt
from .sft import run_sft

logger = logging.get_logger(__name__)


def _training_function(config):
    args = config.get("args")
    callbacks = config.get("callbacks") or []
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks.append(LogCallback())
    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Only pt and sft stages are kept.")

    if dist.is_initialized():
        dist.destroy_process_group()


def run_exp(args=None, callbacks=None):
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)
        return
    _training_function({"args": args, "callbacks": callbacks or []})
