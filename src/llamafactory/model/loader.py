# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import os
from typing import Any, Optional, TypedDict

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.visual import COMPOSITE_MODELS
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer


from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not a Processor instance. Dropping it.")
        processor = None
    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    init_kwargs = _get_init_kwargs(model_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    model_type = getattr(config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        allowed = ", ".join(sorted(COMPOSITE_MODELS.keys()))
        raise ValueError(f"Slim build only supports LLaVA/Qwen-VL/InternVL image models. Got model_type={model_type!r}; allowed: {allowed}.")
    return config


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    if add_valuehead:
        raise ValueError("This slim build does not include value-head models.")

    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=False)

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    init_kwargs["torch_dtype"] = "auto"
    load_class = AutoModelForImageTextToText if type(config) in AutoModelForImageTextToText._model_mapping.keys() else AutoModelForCausalLM

    if model_args.train_from_scratch:
        model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
    else:
        model = load_class.from_pretrained(**init_kwargs)

    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead=False)
    register_autoclass(config, model, tokenizer)
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"
    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")
    return model
