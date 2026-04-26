import math
import os
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from transformers.image_utils import get_image_size, make_flat_list_of_images, to_numpy_array

from ..extras.constants import IMAGE_PLACEHOLDER


def _make_batched_images(images, imglens):
    batches, offset = [], 0
    for imglen in imglens:
        batches.append(images[offset : offset + imglen])
        offset += imglen
    return batches


@dataclass
class BasePlugin:
    image_token: str | None = None
    expand_mm_tokens: bool = True
    vision_bos_token: str = ""
    vision_eos_token: str = ""

    def _validate_input(self, processor, images):
        if images and self.image_token is None:
            raise ValueError("This template does not support image input.")
        if self.image_token is not None and images and processor is None:
            raise ValueError("Processor is required for image training.")
        if self.image_token is not None and images and getattr(processor, "image_processor", None) is None:
            raise ValueError("Image processor is required for image training.")

    def _validate_messages(self, messages, images):
        placeholders = sum(m["content"].count(IMAGE_PLACEHOLDER) for m in messages)
        if len(images) != placeholders:
            raise ValueError(f"images={len(images)} but {IMAGE_PLACEHOLDER} placeholders={placeholders}: {messages}")

    def _open_image(self, image):
        if isinstance(image, ImageObject):
            return image
        if isinstance(image, str):
            with Image.open(os.path.expanduser(image)) as img:
                return img.copy()
        if isinstance(image, bytes):
            with Image.open(BytesIO(image)) as img:
                return img.copy()
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                with Image.open(BytesIO(image["bytes"])) as img:
                    return img.copy()
            if image.get("path") is not None:
                with Image.open(os.path.expanduser(image["path"])) as img:
                    return img.copy()
        if hasattr(image, "read"):
            with Image.open(image) as img:
                return img.copy()
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def _preprocess_image(self, image, image_max_pixels, image_min_pixels, **kwargs):
        pixels = image.width * image.height
        if pixels > image_max_pixels:
            scale = math.sqrt(image_max_pixels / pixels)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        pixels = image.width * image.height
        if pixels < image_min_pixels:
            scale = math.sqrt(image_min_pixels / pixels)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        return image.convert("RGB") if image.mode != "RGB" else image

    def _regularize_images(self, images, processor):
        return [
            self._preprocess_image(
                self._open_image(image),
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )
            for image in images
        ]

    def _get_mm_inputs(self, images, processor, imglens=None):
        if not images:
            return {}
        images = self._regularize_images(images, processor)
        if imglens is not None:
            images = _make_batched_images(images, imglens)
        kwargs = {}
        if getattr(processor, "image_do_pan_and_scan", False):
            kwargs.update(
                do_pan_and_scan=True,
                pan_and_scan_min_crop_size=256,
                pan_and_scan_max_num_crops=4,
                pan_and_scan_min_ratio_to_activate=1.2,
            )
        return processor.image_processor(images, return_tensors="pt", **kwargs)

    def process_messages(self, messages, images, processor):
        self._validate_input(processor, images)
        return messages

    def process_token_ids(self, input_ids, labels, images, tokenizer, processor):
        self._validate_input(processor, images)
        return input_ids, labels

    def get_mm_inputs(self, images, imglens, batch_ids, processor):
        self._validate_input(processor, images)
        if processor is None or not images:
            return {}
        return self._get_mm_inputs(images, processor)


@dataclass
class LlavaPlugin(BasePlugin):
    def process_messages(self, messages, images, processor):
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        image_seqlen = 1
        if self.expand_mm_tokens and images:
            mm_inputs = self._get_mm_inputs(images, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                image_seqlen += getattr(processor, "num_additional_image_tokens", 0)
                if getattr(processor, "vision_feature_select_strategy", None) == "default":
                    image_seqlen -= 1
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, (self.image_token or "") * image_seqlen, 1)
            message["content"] = content
        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    def process_messages(self, messages, images, processor):
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        image_sizes = None
        height = width = None
        if self.expand_mm_tokens and images:
            mm_inputs = self._get_mm_inputs(images, processor)
            if "pixel_values" in mm_inputs and "image_sizes" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = 1
                if image_sizes is not None:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if getattr(processor, "vision_feature_select_strategy", None) == "default":
                        image_seqlen -= 1
                content = content.replace(IMAGE_PLACEHOLDER, (self.image_token or "") * image_seqlen, 1)
            message["content"] = content
        return messages


@dataclass
class InternVLPlugin(BasePlugin):
    def _get_mm_inputs(self, images, processor, imglens=None):
        if not images:
            return {}
        kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            kwargs.update(crop_to_patches=True, max_patches=12, min_patches=1)
        images = [
            self._preprocess_image(
                self._open_image(image),
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )
            for image in images
        ]
        image_inputs = processor.image_processor(images=make_flat_list_of_images(images), return_tensors="pt", **kwargs)
        num_patches = image_inputs.pop("num_patches")
        pixel_values = image_inputs.pop("pixel_values")
        patch_ends = np.cumsum(num_patches)
        patches = []
        for i in range(len(images)):
            start = patch_ends[i - 1] if i > 0 else 0
            patches.append(pixel_values[start : patch_ends[i]])
        return {"pixel_values": torch.cat(patches, dim=0), "image_num_patches": num_patches}

    def process_messages(self, messages, images, processor):
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, processor)
        image_num_patches = mm_inputs.get("image_num_patches", [])
        image_seqlen = getattr(processor, "image_seq_length", 1) if self.expand_mm_tokens else 1
        image_idx = 0
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                patches = int(image_num_patches[image_idx]) if self.expand_mm_tokens and len(image_num_patches) else 1
                content = content.replace(IMAGE_PLACEHOLDER, f"<img>{'<IMG_CONTEXT>' * image_seqlen * patches}</img>", 1)
                image_idx += 1
            message["content"] = content
        return messages

    def get_mm_inputs(self, images, imglens, batch_ids, processor):
        self._validate_input(processor, images)
        if processor is None or not images:
            return {}
        mm_inputs = self._get_mm_inputs(images, processor)
        mm_inputs.pop("image_num_patches", None)
        return mm_inputs


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    def _preprocess_image(self, image, **kwargs):
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            image = image.resize((max(image.width, 28), max(image.height, 28)))
        if image.width / image.height > 200:
            image = image.resize((image.height * 180, image.height))
        if image.height / image.width > 200:
            image = image.resize((image.width, image.width * 180))
        return image

    def process_messages(self, messages, images, processor):
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        merge_length = getattr(processor.image_processor, "merge_size", 1) ** 2
        if self.expand_mm_tokens and images:
            image_grid_thw = self._get_mm_inputs(images, processor).get("image_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
        image_idx = 0
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = int(image_grid_thw[image_idx].prod().item() // merge_length) if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{(self.image_token or '') * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                image_idx += 1
            message["content"] = content
        return messages


class Qwen3VLPlugin(Qwen2VLPlugin):
    pass


PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "intern_vl": InternVLPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
}


def get_mm_plugin(name, image_token=None, **kwargs):
    if name not in PLUGINS:
        raise ValueError(f"Unknown multimodal plugin: {name}")
    kept = {k: v for k, v in kwargs.items() if k in {"expand_mm_tokens", "vision_bos_token", "vision_eos_token"}}
    return PLUGINS[name](image_token=image_token, **kept)
