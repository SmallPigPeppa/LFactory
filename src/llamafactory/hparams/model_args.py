from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from ..extras.constants import AttentionFunction


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default=None)
    adapter_name_or_path: str | None = field(default=None)
    adapter_folder: str | None = field(default=None)
    cache_dir: str | None = field(default=None)
    model_revision: str = field(default="main")
    hf_hub_token: str | None = field(default=None)
    trust_remote_code: bool = field(default=False)
    use_fast_tokenizer: bool = field(default=True)
    split_special_tokens: bool = field(default=False)
    low_cpu_mem_usage: bool = field(default=True)
    train_from_scratch: bool = field(default=False)
    offload_folder: str = field(default="offload")
    resize_vocab: bool = field(default=False)
    add_tokens: str | None = field(default=None)
    add_special_tokens: str | None = field(default=None)
    disable_gradient_checkpointing: bool = field(default=False)
    use_reentrant_gc: bool = field(default=True)
    use_kv_cache: bool = field(default=True)
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(default="auto")
    flash_attn: AttentionFunction = field(default=AttentionFunction.AUTO)
    image_max_pixels: int = field(default=768 * 768)
    image_min_pixels: int = field(default=32 * 32)
    image_do_pan_and_scan: bool = field(default=False)
    crop_to_patches: bool = field(default=False)
    print_param_status: bool = field(default=False)

    compute_dtype: torch.dtype | None = field(default=None, init=False)
    device_map: str | dict[str, Any] | None = field(default=None, init=False)
    model_max_length: int | None = field(default=None, init=False)
    block_diag_attn: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required.")
        if self.adapter_name_or_path:
            self.adapter_name_or_path = [p.strip() for p in self.adapter_name_or_path.split(",") if p.strip()]
        if self.add_tokens:
            self.add_tokens = [t.strip() for t in self.add_tokens.split(",") if t.strip()]
        if self.add_special_tokens:
            self.add_special_tokens = [t.strip() for t in self.add_special_tokens.split(",") if t.strip()]
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError("image_max_pixels cannot be smaller than image_min_pixels.")

    def to_dict(self):
        data = asdict(self)
        data["hf_hub_token"] = "<HF_HUB_TOKEN>" if self.hf_hub_token else None
        data["compute_dtype"] = str(self.compute_dtype)
        return data
