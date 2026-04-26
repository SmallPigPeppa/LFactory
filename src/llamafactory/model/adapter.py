import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from ..extras import logging
from .model_utils.misc import find_all_linear_modules
from .model_utils.visual import get_forbidden_modules, patch_target_modules

logger = logging.get_logger(__name__)


def _cast_trainable_params_to_fp32(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)


def _setup_full_tuning(model, finetuning_args, is_trainable):
    if not is_trainable:
        return model
    forbidden = get_forbidden_modules(model.config, finetuning_args)
    logger.info_rank0("Fine-tuning method: full")
    for name, param in model.named_parameters():
        param.requires_grad_(not any(block in name for block in forbidden))
    return model


def _setup_lora_tuning(model, model_args, finetuning_args, is_trainable):
    if model_args.adapter_name_or_path:
        init_kwargs = dict(
            subfolder=model_args.adapter_folder,
            offload_folder=model_args.offload_folder,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.hf_hub_token,
        )
        adapters = model_args.adapter_name_or_path
        to_merge = adapters[:-1] if is_trainable and not finetuning_args.create_new_adapter else adapters
        resume = adapters[-1] if is_trainable and not finetuning_args.create_new_adapter else None
        for adapter in to_merge:
            model = PeftModel.from_pretrained(model, adapter, **init_kwargs).merge_and_unload()
        if resume is not None:
            model = PeftModel.from_pretrained(model, resume, is_trainable=is_trainable, **init_kwargs)
        logger.info_rank0("Loaded adapter(s): {}".format(",".join(adapters)))
        if is_trainable and resume is not None:
            return model

    if not is_trainable:
        return model

    logger.info_rank0("Fine-tuning method: LoRA")
    if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
        target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
    else:
        target_modules = finetuning_args.lora_target
    target_modules = patch_target_modules(model, finetuning_args, target_modules)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetuning_args.lora_rank,
        target_modules=target_modules,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        use_rslora=finetuning_args.use_rslora,
        use_dora=finetuning_args.use_dora,
        modules_to_save=finetuning_args.additional_target,
    )
    return get_peft_model(model, peft_config)


def init_adapter(config, model, model_args, finetuning_args, is_trainable):
    if finetuning_args.finetuning_type == "full":
        model = _setup_full_tuning(model, finetuning_args, is_trainable)
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(model, model_args, finetuning_args, is_trainable)
    else:
        raise ValueError("Only LoRA and full fine-tuning are kept.")

    if is_trainable and not finetuning_args.pure_bf16:
        logger.info_rank0("Upcasting trainable params to float32.")
        _cast_trainable_params_to_fp32(model)
    return model
