from ...extras import logging

logger = logging.get_logger(__name__)


def prepare_model_for_training(model, model_args):
    model.config.use_cache = False
    if model_args.disable_gradient_checkpointing:
        return
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc})
        logger.info_rank0("Gradient checkpointing enabled.")
