# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, MROPE_MODELS
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image
else:  # pragma: no cover
    Image = None


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from .template import Template


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)
    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    return torch.where(attention_mask_4d, zero_tensor, min_dtype)


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")
        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model
        if self.model is not None and hasattr(self.model, "get_rope_index"):
            self.get_rope_func = self.model.get_rope_index
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index
        else:
            self.get_rope_func = None

    def _inject_dummy_image(self, features: list[dict[str, Any]], images_per_sample: list[list[Any]]) -> bool:
        if self.processor is None or self.template.mm_plugin.image_token is None or sum(len(x) for x in images_per_sample) != 0:
            return False
        if Image is None:
            return False

        fake_image = Image.new("RGB", (64, 64), color=(255, 255, 255))
        fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
        fake_messages = self.template.mm_plugin.process_messages(fake_messages, [fake_image], self.processor)
        fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
        fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
            fake_input_ids, None, [fake_image], self.tokenizer, self.processor
        )
        if len(features) == 0:
            return False
        features[0]["input_ids"] = list(features[0]["input_ids"]) + fake_input_ids
        features[0]["attention_mask"] = list(features[0]["attention_mask"]) + [0] * len(fake_input_ids)
        features[0]["labels"] = list(features[0]["labels"]) + [IGNORE_INDEX] * len(fake_input_ids)
        images_per_sample[0] = [fake_image]
        return True

    def _compute_rope_position_ids(self, features: dict[str, "torch.Tensor"], mm_inputs: dict[str, Any]) -> None:
        if self.get_rope_func is None:
            return
        if features["attention_mask"].sum() == 0:
            features["position_ids"] = torch.zeros((3, *features["input_ids"].shape), dtype=torch.long)
            features["rope_deltas"] = torch.zeros(features["input_ids"].shape[0], dtype=torch.long)
            return

        kwargs = {
            "input_ids": features["input_ids"],
            "image_grid_thw": mm_inputs.get("image_grid_thw"),
            "attention_mask": (features["attention_mask"] >= 1).float(),
        }
        params = inspect.signature(self.get_rope_func).parameters
        if "mm_token_type_ids" in params:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            if image_token_id is None and self.template.mm_plugin.image_token is not None:
                image_token_id = self.tokenizer.convert_tokens_to_ids(self.template.mm_plugin.image_token)
            if image_token_id is not None:
                token_type_ids = torch.zeros_like(features["input_ids"])
                token_type_ids[features["input_ids"] == image_token_id] = 1
                kwargs["mm_token_type_ids"] = token_type_ids

        kwargs = {key: value for key, value in kwargs.items() if key in params}
        features["position_ids"], features["rope_deltas"] = self.get_rope_func(**kwargs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        images_per_sample: list[list[Any]] = []
        for feature in features:
            raw_images = feature.pop("images", None)
            feature.pop("packing_params", None)  # simplified: train_llava779k_qwen3vl2b.sh uses packing=false
            if raw_images is None:
                images_per_sample.append([])
            elif isinstance(raw_images, list):
                images_per_sample.append(raw_images)
            else:
                images_per_sample.append([raw_images])

        has_dummy_image = self._inject_dummy_image(features, images_per_sample)
        batch_images = [image for images in images_per_sample for image in images]
        batch_imglens = [len(images) for images in images_per_sample]
        batch_input_ids = [list(feature["input_ids"]) for feature in features]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(batch_images, batch_imglens, batch_input_ids, self.processor)
        padded_features = super().__call__(features)

        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if self.get_rope_func is not None or model_type in MROPE_MODELS:
            self._compute_rope_position_ids(padded_features, mm_inputs)

        padded_features.update(mm_inputs)
        if has_dummy_image:
            padded_features["has_dummy_image"] = True
        return padded_features


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32
    neat_packing: bool = False

    @staticmethod
    def _unpad_packed_features(features: dict[str, Any]) -> None:
        attention_mask = features.get("attention_mask")
        if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2 or attention_mask.size(0) != 1:
            return
        seq_len = attention_mask.size(1)
        non_padding_indices = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
        if non_padding_indices.numel() == seq_len:
            return
        for key, value in list(features.items()):
            if not torch.is_tensor(value):
                continue
            if key == "position_ids" and value.size(-1) == seq_len:
                features[key] = value.index_select(-1, non_padding_indices)
            elif key in {"input_ids", "labels", "attention_mask", "token_type_ids"} and value.dim() == 2:
                if value.size(0) == 1 and value.size(1) == seq_len:
                    features[key] = value.index_select(1, non_padding_indices)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        has_dummy_image = features.pop("has_dummy_image", False)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        if self.neat_packing and self.attn_implementation == "flash_attention_2":
            if features["input_ids"].shape[0] != 1:
                raise ValueError("batch_size should be 1 for neat packing with flash attention.")
            if not has_dummy_image:
                self._unpad_packed_features(features)
            features["attention_mask"] = None

        for key, value in list(features.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)
        return features
