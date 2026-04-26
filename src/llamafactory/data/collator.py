import inspect
from dataclasses import dataclass
from typing import Any, Literal

import torch
from peft import PeftModel
from PIL import Image
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, MROPE_MODELS


def prepare_4d_attention_mask(attention_mask_with_indices, dtype):
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    non_padding = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    row = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    col = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    visible = (row == col) & non_padding & causal
    return torch.where(visible, torch.tensor(0, dtype=dtype), min_dtype)


@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    template: Any = None
    processor: Any = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("template is required.")
        if isinstance(self.model, PeftModel):
            self.model = self.model.base_model.model
        if self.model is not None and hasattr(self.model, "get_rope_index"):
            self.get_rope_func = self.model.get_rope_index
        elif self.model is not None and hasattr(self.model, "model") and hasattr(self.model.model, "get_rope_index"):
            self.get_rope_func = self.model.model.get_rope_index
        else:
            self.get_rope_func = None

    def _inject_dummy_image(self, features, images_per_sample):
        # Some VLM forward paths expect image tensors even when a batch has only text.
        # Keep the original LLaMA-Factory trick: append an ignored image token span and a 64x64 white image.
        if self.processor is None or self.template.mm_plugin.image_token is None:
            return False
        if sum(len(x) for x in images_per_sample) != 0 or len(features) == 0:
            return False

        fake_image = Image.new("RGB", (64, 64), color=(255, 255, 255))
        fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
        fake_messages = self.template.mm_plugin.process_messages(fake_messages, [fake_image], self.processor)
        fake_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
        fake_ids, _ = self.template.mm_plugin.process_token_ids(fake_ids, None, [fake_image], self.tokenizer, self.processor)

        features[0]["input_ids"] = list(features[0]["input_ids"]) + fake_ids
        features[0]["attention_mask"] = list(features[0]["attention_mask"]) + [0] * len(fake_ids)
        features[0]["labels"] = list(features[0]["labels"]) + [IGNORE_INDEX] * len(fake_ids)
        images_per_sample[0] = [fake_image]
        return True

    def _compute_rope_position_ids(self, features, mm_inputs):
        if self.get_rope_func is None:
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
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        features["position_ids"], features["rope_deltas"] = self.get_rope_func(**kwargs)

    def __call__(self, features):
        text_keys = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids"}
        cleaned, images_per_sample = [], []
        for feature in features:
            feature = dict(feature)
            raw_images = feature.pop("images", None)
            feature.pop("packing_params", None)
            cleaned.append({k: v for k, v in feature.items() if k in text_keys})
            if raw_images is None:
                images_per_sample.append([])
            elif isinstance(raw_images, (list, tuple)):
                images_per_sample.append([img for img in raw_images if img is not None])
            else:
                images_per_sample.append([raw_images])

        has_dummy_image = self._inject_dummy_image(cleaned, images_per_sample)
        batch_images = [img for images in images_per_sample for img in images]
        batch_imglens = [len(images) for images in images_per_sample]
        batch_input_ids = [list(feature["input_ids"]) for feature in cleaned]
        mm_inputs = self.template.mm_plugin.get_mm_inputs(batch_images, batch_imglens, batch_input_ids, self.processor)

        padded = super().__call__(cleaned)
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if self.get_rope_func is not None or model_type in MROPE_MODELS:
            self._compute_rope_position_ids(padded, mm_inputs)
        padded.update(mm_inputs)
        if has_dummy_image:
            padded["has_dummy_image"] = True
        return padded


@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: torch.dtype = torch.float32
    neat_packing: bool = False

    def __call__(self, features):
        features = super().__call__(features)
        has_dummy_image = features.pop("has_dummy_image", False)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)
        if self.neat_packing and self.attn_implementation == "flash_attention_2":
            if features["input_ids"].shape[0] != 1:
                raise ValueError("neat_packing requires batch_size=1.")
            if not has_dummy_image:
                self._unpad_packed_features(features)
            features["attention_mask"] = None
        for key, value in list(features.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                features[key] = value.to(self.compute_dtype)
        return features

    @staticmethod
    def _unpad_packed_features(features):
        attention_mask = features.get("attention_mask")
        if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2 or attention_mask.size(0) != 1:
            return
        non_padding = torch.nonzero(attention_mask[0] != 0, as_tuple=False).flatten()
        seq_len = attention_mask.size(1)
        if non_padding.numel() == seq_len:
            return
        for key, value in list(features.items()):
            if not torch.is_tensor(value):
                continue
            if key == "position_ids" and value.size(-1) == seq_len:
                features[key] = value.index_select(-1, non_padding)
            elif key in {"input_ids", "labels", "attention_mask", "token_type_ids"} and value.dim() == 2 and value.size(1) == seq_len:
                features[key] = value.index_select(1, non_padding)
