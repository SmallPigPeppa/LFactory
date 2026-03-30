import json
from copy import deepcopy
from typing import Any


def infer_deepspeed_mixed_precision(ds_config: dict[str, Any], bf16: bool) -> str:
    bf16_enabled = ds_config.get("bf16", {}).get("enabled", "auto")

    if bool(bf16_enabled):
        return "bf16"
    if bf16_enabled == "auto":
        return "bf16" if bf16 else "no"
    return "no"


def _unset_hf_deepspeed_config() -> None:
    try:
        from transformers.integrations import unset_hf_deepspeed_config
    except ImportError:
        from transformers.deepspeed import unset_hf_deepspeed_config

    unset_hf_deepspeed_config()


def _load_deepspeed_config(config_file: str) -> dict[str, Any]:
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)


def setup_deepspeed_zero3_model_loading(is_train: bool, dist_config: dict[str, Any] | None, bf16: bool):
    """Enable transformers' ZeRO-3-aware model loading for the current thread."""
    if not is_train or dist_config is None or dist_config.get("name") != "deepspeed":
        return None

    config_file = dist_config.get("config_file")
    if not config_file:
        raise ValueError("DeepSpeed config_file is required in dist_config")

    from accelerate.utils import DeepSpeedPlugin

    try:
        from transformers.integrations import is_deepspeed_zero3_enabled
    except ImportError:
        from transformers.deepspeed import is_deepspeed_zero3_enabled

    # DeepSpeed configs often use "auto" placeholders that only make sense once
    # we know the current runtime batch settings and precision mode.
    ds_config = deepcopy(_load_deepspeed_config(config_file))
    if "gradient_accumulation_steps" not in ds_config or ds_config["gradient_accumulation_steps"] == "auto":
        ds_config["gradient_accumulation_steps"] = 1
    if "train_micro_batch_size_per_gpu" not in ds_config or ds_config["train_micro_batch_size_per_gpu"] == "auto":
        ds_config["train_micro_batch_size_per_gpu"] = 1
    if ds_config.get("train_batch_size") == "auto":
        ds_config.pop("train_batch_size")

    # ZeRO-3 model loading needs concrete fp16/bf16 flags, not "auto".
    ds_config.setdefault("fp16", {})
    ds_config.setdefault("bf16", {})
    if ds_config["bf16"].get("enabled", "auto") == "auto":
        ds_config["bf16"]["enabled"] = bf16
    if ds_config["fp16"].get("enabled", "auto") == "auto":
        ds_config["fp16"]["enabled"] = False

    plugin = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=True)

    if not plugin.hf_ds_config.is_zero3():
        return None

    # Reuse the same precision inference rule as the training-time DeepSpeed path
    # so both model-loading and engine setup stay aligned.
    plugin.set_mixed_precision(infer_deepspeed_mixed_precision(ds_config, bf16=bf16))
    plugin.set_deepspeed_weakref()

    if not is_deepspeed_zero3_enabled():
        raise RuntimeError(
            "DeepSpeed ZeRO-3 model-loading bootstrap failed: transformers still reports zero3 disabled "
            "after constructing HfDeepSpeedConfig. This usually means the runtime is using a different transformers "
            "installation than expected, or the DeepSpeed global state was not established correctly."
        )
    return plugin


def teardown_deepspeed_zero3_model_loading(plugin) -> None:
    if plugin is not None:
        _unset_hf_deepspeed_config()
