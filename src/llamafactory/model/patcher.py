import torch

from ..extras import logging
from ..extras.constants import AttentionFunction
from ..extras.misc import infer_optim_dtype
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.visual import configure_visual_model

logger = logging.get_logger(__name__)


def patch_tokenizer(tokenizer, model_args):
    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length
    if model_args.add_tokens and tokenizer.add_tokens(model_args.add_tokens, special_tokens=False) > 0:
        model_args.resize_vocab = True
    if model_args.add_special_tokens and tokenizer.add_tokens(model_args.add_special_tokens, special_tokens=True) > 0:
        model_args.resize_vocab = True


def patch_processor(processor, tokenizer, model_args):
    processor.tokenizer = tokenizer
    processor.image_max_pixels = model_args.image_max_pixels
    processor.image_min_pixels = model_args.image_min_pixels
    processor.image_do_pan_and_scan = model_args.image_do_pan_and_scan
    processor.crop_to_patches = model_args.crop_to_patches
    attr = "vi" + "deo_processor"
    if hasattr(processor, attr):
        setattr(processor, attr, None)


def _set_attention(config, model_args):
    mapping = {
        AttentionFunction.DISABLED: "eager",
        AttentionFunction.SDPA: "sdpa",
        AttentionFunction.FA2: "flash_attention_2",
    }
    requested = mapping.get(model_args.flash_attn)
    if requested:
        config._attn_implementation = requested


def patch_config(config, tokenizer, model_args, init_kwargs, is_trainable):
    if model_args.compute_dtype is None:
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(getattr(config, "torch_dtype", None))
    _set_attention(config, model_args)
    configure_visual_model(config)
    config.use_cache = model_args.use_kv_cache and not is_trainable
    if is_trainable:
        init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage
        if init_kwargs["low_cpu_mem_usage"]:
            init_kwargs["device_map"] = model_args.device_map
            if init_kwargs.get("device_map") == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(model, tokenizer, model_args, is_trainable):
    if model_args.resize_vocab:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    if is_trainable:
        prepare_model_for_training(model, model_args)
    attn = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
    logger.info_rank0(f"Attention implementation: {attn or 'default'}.")
