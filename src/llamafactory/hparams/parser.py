# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_npu_available

from ..extras import logging
from ..extras.constants import CHECKPOINT_NAMES
from ..extras.misc import check_dependencies, check_version, get_current_device, is_env_enabled
from .data_args import DataArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments


logger = logging.get_logger(__name__)
check_dependencies()

_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]
_TRAIN_CLS = tuple[ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments]


def read_args(args: dict[str, Any] | list[str] | None = None) -> dict[str, Any] | list[str]:
    if args is not None:
        return args
    if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.create(json.load(Path(sys.argv[1]).absolute()))
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    return sys.argv[1:]


def _parse_args(parser: "HfArgumentParser", args: dict[str, Any] | list[str] | None = None) -> tuple[Any]:
    args = read_args(args)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    if isinstance(args, dict):
        return parser.parse_dict(args, allow_extra_keys=allow_extra_keys)
    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        raise ValueError(f"Some specified arguments are not used by HfArgumentParser: {unknown_args}")
    return tuple(parsed_args)


def _parse_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    return _parse_args(HfArgumentParser(_TRAIN_ARGS), args)


def _set_transformers_logging() -> None:
    if os.getenv("LLAMAFACTORY_VERBOSITY", "INFO") in ["DEBUG", "INFO"]:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _set_env_vars() -> None:
    if is_torch_npu_available():
        torch.npu.set_compile_mode(jit_compile=is_env_enabled("NPU_JIT_COMPILE"))


def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter loading is only valid for LoRA training.")
    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type not in ["lora", "oft"]:
            raise ValueError("Quantization is only compatible with LoRA or OFT training.")
        if finetuning_args.pissa_init:
            raise ValueError("PiSSA initialization is not compatible with quantized training.")
        if model_args.resize_vocab:
            raise ValueError("Cannot resize embeddings of a quantized model.")
        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create a new adapter on a quantized model.")
        if model_args.adapter_name_or_path is not None and len(model_args.adapter_name_or_path) != 1:
            raise ValueError("Quantized model only accepts a single adapter.")


def _check_extra_dependencies(model_args: "ModelArguments", finetuning_args: "FinetuningArguments", training_args: "TrainingArguments") -> None:
    if model_args.enable_liger_kernel:
        check_version("liger-kernel", mandatory=True)
    if finetuning_args.plot_loss:
        check_version("matplotlib", mandatory=True)
    if training_args.deepspeed:
        check_version("deepspeed", mandatory=True)
    if training_args.predict_with_generate:
        check_version("jieba", mandatory=True)
        check_version("nltk", mandatory=True)
        check_version("rouge_chinese", mandatory=True)


def get_train_args(args: dict[str, Any] | list[str] | None = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)

    if training_args.should_log:
        _set_transformers_logging()

    if finetuning_args.stage not in ["pt", "sft"]:
        raise ValueError("Slim build only supports `stage: pt` and `stage: sft`.")
    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` can only be enabled for SFT.")
        if data_args.neat_packing:
            raise ValueError("`neat_packing` can only be enabled for SFT.")
        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError("`train_on_prompt`/`mask_history` can only be enabled for SFT.")
    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
        raise ValueError("Please launch distributed training via `FORCE_TORCHRUN=1`/`torchrun`.")
    if training_args.deepspeed and training_args.parallel_mode != ParallelMode.DISTRIBUTED:
        raise ValueError("Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.")
    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")
    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")
    if (training_args.do_eval or training_args.do_predict or training_args.predict_with_generate) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please provide eval_dataset or set val_size > 0.")
    if training_args.predict_with_generate:
        if is_deepspeed_zero3_enabled():
            raise ValueError("`predict_with_generate` is incompatible with DeepSpeed ZeRO-3.")
        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")
    if training_args.do_train and model_args.quantization_device_map == "auto":
        raise ValueError("Cannot use device map for quantized models in training.")
    if finetuning_args.pissa_init and is_deepspeed_zero3_enabled():
        raise ValueError("PiSSA initialization is incompatible with DeepSpeed ZeRO-3 in this slim build.")
    if finetuning_args.pure_bf16:
        if not (is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())):
            raise ValueError("This device does not support `pure_bf16`.")
        if is_deepspeed_zero3_enabled():
            raise ValueError("`pure_bf16` is incompatible with DeepSpeed ZeRO-3.")
    if training_args.fp8 and model_args.quantization_bit is not None:
        raise ValueError("FP8 training is incompatible with quantization.")

    _verify_model_args(model_args, finetuning_args)
    _check_extra_dependencies(model_args, finetuning_args, training_args)

    if training_args.do_train and finetuning_args.finetuning_type == "lora" and model_args.quantization_bit is None:
        if model_args.resize_vocab and finetuning_args.additional_target is None:
            logger.warning_rank0("Add embedding layers to `additional_target` to train newly added tokens.")
    if training_args.do_train and model_args.quantization_bit is not None and not model_args.upcast_layernorm:
        logger.warning_rank0("We recommend enabling `upcast_layernorm` in quantized training.")
    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning_rank0("We recommend enabling mixed precision training.")

    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False
    if finetuning_args.finetuning_type == "lora":
        training_args.label_names = training_args.label_names or ["labels"]
    if (
        training_args.parallel_mode == ParallelMode.DISTRIBUTED
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.info_rank0("Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.")
        training_args.ddp_find_unused_parameters = False

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not getattr(training_args, "overwrite_output_dir", False)
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(os.path.isfile(os.path.join(training_args.output_dir, n)) for n in CHECKPOINT_NAMES):
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")
        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info_rank0(f"Resuming training from {training_args.resume_from_checkpoint}.")

    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
    model_args.device_map = {"": get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"

    _set_env_vars()
    logger.info(
        f"Process rank: {training_args.process_index}, world size: {training_args.world_size}, "
        f"device: {training_args.device}, distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED}, "
        f"compute dtype: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(training_args.seed)
    return model_args, data_args, training_args, finetuning_args, generating_args
