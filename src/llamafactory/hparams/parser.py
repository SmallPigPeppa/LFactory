import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES
from ..extras.misc import get_current_device
from .data_args import DataArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments

logger = logging.get_logger(__name__)
_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]


def read_args(args: dict[str, Any] | list[str] | None = None):
    if args is not None:
        return args
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
        config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(config, OmegaConf.from_cli(sys.argv[2:])))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        config = OmegaConf.create(json.load(open(Path(sys.argv[1]).absolute(), encoding="utf-8")))
        return OmegaConf.to_container(OmegaConf.merge(config, OmegaConf.from_cli(sys.argv[2:])))
    return sys.argv[1:]


def _parse_train_args(args=None):
    parser = HfArgumentParser(_TRAIN_ARGS)
    args = read_args(args)
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=True)
    parsed = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    return tuple(parsed[:-1])


def _set_transformers_logging():
    if os.getenv("LLAMAFACTORY_VERBOSITY", "INFO") in {"DEBUG", "INFO"}:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def get_train_args(args=None):
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    if training_args.should_log:
        _set_transformers_logging()

    if finetuning_args.stage not in {"pt", "sft"}:
        raise ValueError("Only pt and sft stages are kept in this slim build.")
    if training_args.do_train and data_args.dataset is None and data_args.tokenized_path is None:
        raise ValueError("dataset or tokenized_path is required for training.")

    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False
    training_args.label_names = training_args.label_names or ["labels"]
    if training_args.parallel_mode == ParallelMode.DISTRIBUTED and training_args.ddp_find_unused_parameters is None:
        logger.info_rank0("Set ddp_find_unused_parameters=False for LoRA DDP training.")
        training_args.ddp_find_unused_parameters = False

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not getattr(training_args, "overwrite_output_dir", False)
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(os.path.isfile(os.path.join(training_args.output_dir, n)) for n in CHECKPOINT_NAMES):
            raise ValueError("Output directory already exists. Set overwrite_output_dir=true or use a new output_dir.")
        training_args.resume_from_checkpoint = last_checkpoint

    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
    else:
        model_args.compute_dtype = torch.float32

    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    logger.info(
        f"Process rank: {training_args.process_index}, world size: {training_args.world_size}, "
        f"device: {training_args.device}, distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED}, "
        f"compute dtype: {model_args.compute_dtype}"
    )
    transformers.set_seed(training_args.seed)
    return model_args, data_args, training_args, finetuning_args, generating_args
