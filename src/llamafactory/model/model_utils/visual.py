from dataclasses import dataclass, field

from ...extras import logging

logger = logging.get_logger(__name__)


@dataclass
class CompositeModel:
    projector_keys: list[str] = field(default_factory=lambda: ["multi_modal_projector"])
    vision_model_keys: list[str] = field(default_factory=lambda: ["vision_tower"])
    language_model_keys: list[str] = field(default_factory=lambda: ["language_model", "lm_head"])
    lora_conflict_keys: list[str] = field(default_factory=list)


COMPOSITE_MODELS = {
    "internvl": CompositeModel(),
    "interns1": CompositeModel(),
    "llava": CompositeModel(),
    "llava_next": CompositeModel(),
    "qwen2_vl": CompositeModel(
        projector_keys=["visual.merger"],
        vision_model_keys=["visual.patch_embed", "visual.blocks"],
        lora_conflict_keys=["patch_embed"],
    ),
    "qwen2_5_vl": CompositeModel(
        projector_keys=["visual.merger"],
        vision_model_keys=["visual.patch_embed", "visual.blocks"],
        lora_conflict_keys=["patch_embed"],
    ),
    "qwen3_vl": CompositeModel(
        projector_keys=["visual.merger"],
        vision_model_keys=["visual.pos_embed", "visual.patch_embed", "visual.blocks", "visual.deepstack_merger_list"],
        lora_conflict_keys=["patch_embed"],
    ),
}


def configure_visual_model(config):
    if getattr(config, "text_config", None) and not getattr(config, "hidden_size", None):
        config.hidden_size = getattr(config.text_config, "hidden_size", None)


def get_forbidden_modules(config, finetuning_args):
    model_type = getattr(config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        return set()
    info = COMPOSITE_MODELS[model_type]
    forbidden = set()
    if finetuning_args.freeze_vision_tower:
        logger.info_rank0(f"Set vision model not trainable: {info.vision_model_keys}.")
        forbidden.update(info.vision_model_keys)
    if finetuning_args.freeze_multi_modal_projector:
        logger.info_rank0(f"Set multimodal projector not trainable: {info.projector_keys}.")
        forbidden.update(info.projector_keys)
    if finetuning_args.freeze_language_model:
        logger.info_rank0(f"Set language model not trainable: {info.language_model_keys}.")
        forbidden.update(info.language_model_keys)
    return forbidden


def patch_target_modules(model, finetuning_args, target_modules):
    model_type = getattr(model.config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        return target_modules
    forbidden = get_forbidden_modules(model.config, finetuning_args)
    forbidden.update(COMPOSITE_MODELS[model_type].lora_conflict_keys)
    modules = []
    for name, _ in model.named_modules():
        if any(target in name for target in target_modules) and not any(block in name for block in forbidden):
            modules.append(name)
    return modules
