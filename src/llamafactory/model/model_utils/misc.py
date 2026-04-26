from ...extras import logging
from .visual import COMPOSITE_MODELS

logger = logging.get_logger(__name__)


def find_all_linear_modules(model, freeze_vision_tower=True):
    model_type = getattr(model.config, "model_type", None)
    forbidden = {"lm_head"}
    if model_type in COMPOSITE_MODELS:
        forbidden.update(COMPOSITE_MODELS[model_type].projector_keys)
        if freeze_vision_tower:
            forbidden.update(COMPOSITE_MODELS[model_type].vision_model_keys)
    names = set()
    for name, module in model.named_modules():
        if any(block in name for block in forbidden):
            continue
        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            names.add(name.split(".")[-1])
    logger.info_rank0("Found linear modules: {}".format(",".join(sorted(names))))
    return sorted(names)


def register_autoclass(config, model, tokenizer):
    return None
