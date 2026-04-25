# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Self

import torch
from omegaconf import OmegaConf

from ..extras.constants import AttentionFunction, QuantizationMethod, RopeScaling
from ..extras.logging import get_logger


logger = get_logger(__name__)


@dataclass
class BaseModelArguments:
    model_name_or_path: str | None = field(default=None, metadata={"help": "Model path or hub id."})
    adapter_name_or_path: str | None = field(default=None, metadata={"help": "Comma-separated adapter paths."})
    adapter_folder: str | None = field(default=None, metadata={"help": "Subfolder containing adapter weights."})
    cache_dir: str | None = field(default=None, metadata={"help": "Model/dataset cache dir."})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer when available."})
    resize_vocab: bool = field(default=False, metadata={"help": "Resize token embeddings after adding tokens."})
    split_special_tokens: bool = field(default=False, metadata={"help": "Split special tokens during tokenization."})
    add_tokens: str | None = field(default=None, metadata={"help": "Comma-separated normal tokens to add."})
    add_special_tokens: str | None = field(default=None, metadata={"help": "Comma-separated special tokens to add."})
    new_special_tokens_config: str | None = field(default=None, metadata={"help": "YAML token->description config."})
    init_special_tokens: Literal["noise_init", "desc_init", "desc_init_w_noise"] = field(
        default="noise_init", metadata={"help": "Initialization for newly added special tokens."}
    )
    model_revision: str = field(default="main", metadata={"help": "Model revision."})
    low_cpu_mem_usage: bool = field(default=True, metadata={"help": "Use memory-efficient loading."})
    rope_scaling: RopeScaling | None = field(default=None, metadata={"help": "RoPE scaling strategy."})
    flash_attn: AttentionFunction = field(default=AttentionFunction.AUTO, metadata={"help": "Attention implementation."})
    shift_attn: bool = field(default=False, metadata={"help": "Enable LongLoRA shift short attention."})
    enable_liger_kernel: bool = field(default=False, metadata={"help": "Enable Liger kernel if installed."})
    moe_aux_loss_coef: float | None = field(default=None, metadata={"help": "Auxiliary router loss coefficient."})
    disable_gradient_checkpointing: bool = field(default=False, metadata={"help": "Disable gradient checkpointing."})
    use_reentrant_gc: bool = field(default=True, metadata={"help": "Use reentrant gradient checkpointing."})
    upcast_layernorm: bool = field(default=False, metadata={"help": "Upcast LayerNorm weights to fp32."})
    upcast_lmhead_output: bool = field(default=False, metadata={"help": "Upcast lm_head output to fp32."})
    train_from_scratch: bool = field(default=False, metadata={"help": "Initialize model from config."})
    offload_folder: str = field(default="offload", metadata={"help": "Offload folder for model weights."})
    use_kv_cache: bool = field(default=True, metadata={"help": "Use KV cache when not training."})
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(default="auto", metadata={"help": "Eval dtype."})
    hf_hub_token: str | None = field(default=None, metadata={"help": "HF token."})
    ms_hub_token: str | None = field(default=None, metadata={"help": "ModelScope token."})
    om_hub_token: str | None = field(default=None, metadata={"help": "OpenMind token."})
    print_param_status: bool = field(default=False, metadata={"help": "Print trainable/frozen parameter names."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote model code."})

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")
        if self.adapter_name_or_path is not None:
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]
        if self.add_tokens is not None:
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",") if token.strip()]

        if self.new_special_tokens_config is not None:
            cfg = OmegaConf.load(self.new_special_tokens_config)
            token_descriptions = OmegaConf.to_container(cfg)
            if not isinstance(token_descriptions, dict):
                raise ValueError("new_special_tokens_config must be a token->description mapping.")
            self.add_special_tokens = list(token_descriptions.keys())
            self._special_token_descriptions = token_descriptions
            logger.info_rank0(f"Loaded {len(self.add_special_tokens)} special tokens from {self.new_special_tokens_config}.")
        elif self.add_special_tokens is not None:
            self.add_special_tokens = [token.strip() for token in self.add_special_tokens.split(",") if token.strip()]
            self._special_token_descriptions = None
        else:
            self._special_token_descriptions = None

        if self.init_special_tokens in ["desc_init", "desc_init_w_noise"] and self._special_token_descriptions is None:
            logger.warning_rank0("Description-based token initialization needs new_special_tokens_config; using noise_init.")
            self.init_special_tokens = "noise_init"


@dataclass
class QuantizationArguments:
    quantization_method: QuantizationMethod = field(default=QuantizationMethod.BNB, metadata={"help": "Quantization method."})
    quantization_bit: int | None = field(default=None, metadata={"help": "On-the-fly quantization bits."})
    quantization_type: Literal["fp4", "nf4"] = field(default="nf4", metadata={"help": "bnb 4-bit type."})
    double_quantization: bool = field(default=True, metadata={"help": "Use bnb double quantization."})
    quantization_device_map: Literal["auto"] | None = field(default=None, metadata={"help": "Device map for quantized loading."})


@dataclass
class ProcessorArguments:
    image_max_pixels: int = field(default=768 * 768, metadata={"help": "Maximum image pixels."})
    image_min_pixels: int = field(default=32 * 32, metadata={"help": "Minimum image pixels."})
    image_do_pan_and_scan: bool = field(default=False, metadata={"help": "Use image pan-and-scan where supported."})
    crop_to_patches: bool = field(default=False, metadata={"help": "Crop images to patches for InternVL."})

    def __post_init__(self):
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError("`image_max_pixels` cannot be smaller than `image_min_pixels`.")


@dataclass
class ModelArguments(ProcessorArguments, QuantizationArguments, BaseModelArguments):
    """Model/config/tokenizer arguments for image VLM pt/sft training."""

    compute_dtype: torch.dtype | None = field(default=None, init=False)
    device_map: str | dict[str, Any] | None = field(default=None, init=False)
    model_max_length: int | None = field(default=None, init=False)
    block_diag_attn: bool = field(default=False, init=False)

    def __post_init__(self):
        BaseModelArguments.__post_init__(self)
        ProcessorArguments.__post_init__(self)

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)
        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)
        return result

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("token") else v for k, v in args.items()}
        args["compute_dtype"] = str(self.compute_dtype) if self.compute_dtype is not None else None
        return args
