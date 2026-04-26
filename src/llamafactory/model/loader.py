import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from ..extras import logging
from ..extras.misc import count_parameters
from .adapter import init_adapter
from .model_utils.misc import register_autoclass
from .model_utils.visual import COMPOSITE_MODELS
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer

logger = logging.get_logger(__name__)


def _init_kwargs(model_args):
    return dict(
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.hf_hub_token,
    )


def load_tokenizer(model_args):
    kwargs = _init_kwargs(model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **kwargs,
    )
    patch_tokenizer(tokenizer, model_args)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        **kwargs,
    )
    patch_processor(processor, tokenizer, model_args)
    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **_init_kwargs(model_args))
    model_type = getattr(config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        raise ValueError(f"Only LLaVA/Qwen-VL/InternVL image models are kept; got model_type={model_type!r}.")
    return config


def load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False):
    if add_valuehead:
        raise ValueError("Value-head models were removed from this slim build.")

    init_kwargs = _init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    init_kwargs.update(config=config, pretrained_model_name_or_path=model_args.model_name_or_path, torch_dtype="auto")

    load_class = AutoModelForImageTextToText
    if type(config) not in AutoModelForImageTextToText._model_mapping.keys():
        load_class = AutoModelForCausalLM

    model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code) if model_args.train_from_scratch else load_class.from_pretrained(**init_kwargs)
    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer)
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
    model.train() if is_trainable else model.eval()
    if not is_trainable:
        model.requires_grad_(False)

    trainable, total = count_parameters(model)
    logger.info_rank0(
        f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.4f}"
        if is_trainable else f"all params: {total:,}"
    )
    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")
    return model
