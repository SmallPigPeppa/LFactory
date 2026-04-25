# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any, BinaryIO, Optional, TypedDict, Union

import numpy as np
import torch
from transformers.image_utils import get_image_size, make_flat_list_of_images, to_numpy_array
from typing_extensions import override

from ..extras.constants import IMAGE_PLACEHOLDER
from ..extras.packages import is_pillow_available


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject
else:  # pragma: no cover - image training requires Pillow
    Image = None
    ImageObject = object


from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor

class EncodedImage(TypedDict):
    path: str | None
    bytes: bytes | None

ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]

class RegularizedImageOutput(TypedDict):
    images: list[ImageObject]

class MMProcessor(ProcessorMixin):
    patch_size: int
    image_seq_length: int
    num_additional_image_tokens: int
    vision_feature_select_strategy: str

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        pass


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    batch_images = []
    offset = 0
    for imglen in imglens:
        batch_images.append(images[offset : offset + imglen])
        offset += imglen
    return batch_images


@dataclass
class BasePlugin:
    image_token: str | None = None
    expand_mm_tokens: bool = True
    vision_bos_token: str = ""
    vision_eos_token: str = ""

    def _validate_input(self, processor: Optional["MMProcessor"], images: list["ImageInput"]) -> None:
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This template/model does not support image input.")
        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found. Please check the model processor files.")
        if self.image_token is not None and getattr(processor, "image_processor", None) is None:
            raise ValueError("Image processor was not found. Please check the model processor files.")

    def _validate_messages(self, messages: list[dict[str, str]], images: list["ImageInput"]) -> None:
        num_image_tokens = sum(message["content"].count(IMAGE_PLACEHOLDER) for message in messages)
        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images ({len(images)}) does not match the number of "
                f"{IMAGE_PLACEHOLDER} placeholders ({num_image_tokens}) in {messages}."
            )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * resize_factor)), max(1, int(image.height * resize_factor))))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            image = image.resize((max(1, int(image.width * resize_factor)), max(1, int(image.height * resize_factor))))

        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _open_image(self, image: "ImageInput") -> "ImageObject":
        if Image is None:
            raise RuntimeError("Pillow is required for image training.")

        if isinstance(image, str):
            with Image.open(os.path.expanduser(image)) as img:
                return img.copy()
        if isinstance(image, bytes):
            with Image.open(BytesIO(image)) as img:
                return img.copy()
        if isinstance(image, dict):
            image_bytes = image.get("bytes")
            image_path = image.get("path")
            if image_bytes is not None:
                with Image.open(BytesIO(image_bytes)) as img:
                    return img.copy()
            if image_path is not None:
                with Image.open(os.path.expanduser(image_path)) as img:
                    return img.copy()
            raise ValueError(f"Invalid image dictionary: {image.keys()}.")
        if hasattr(image, "read"):
            with Image.open(image) as img:
                return img.copy()
        if isinstance(image, ImageObject):
            return image
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> "RegularizedImageOutput":
        results = []
        for image in images:
            results.append(self._preprocess_image(self._open_image(image), **kwargs))
        return {"images": results}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        processor: "MMProcessor",
        imglens: list[int] | None = None,
    ) -> dict[str, "torch.Tensor"]:
        mm_inputs: dict[str, Any] = {}
        if len(images) == 0:
            return mm_inputs

        image_processor = getattr(processor, "image_processor")
        images = self._regularize_images(
            images,
            image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
            image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
        )["images"]

        if imglens is not None:
            images = _make_batched_images(images, imglens)

        image_processor_kwargs = {}
        if getattr(processor, "image_do_pan_and_scan", False):
            image_processor_kwargs.update(
                {
                    "do_pan_and_scan": True,
                    "pan_and_scan_min_crop_size": 256,
                    "pan_and_scan_max_num_crops": 4,
                    "pan_and_scan_min_ratio_to_activate": 1.2,
                }
            )

        mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))
        return mm_inputs

    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: list[int] | None,
        images: list["ImageInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], list[int] | None]:
        self._validate_input(processor, images)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        imglens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Any]:
        self._validate_input(processor, images)
        if processor is None:
            return {}
        return self._get_mm_inputs(images, processor)


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        if self.expand_mm_tokens and images:
            mm_inputs = self._get_mm_inputs(images, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                image_seqlen += getattr(processor, "num_additional_image_tokens", 0)
                if getattr(processor, "vision_feature_select_strategy", None) == "default":
                    image_seqlen -= 1
            else:
                image_seqlen = 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
            message["content"] = content.replace("{{image}}", self.image_token or "")
        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
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
                if self.expand_mm_tokens and image_sizes is not None:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if getattr(processor, "vision_feature_select_strategy", None) == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
            message["content"] = content.replace("{{image}}", self.image_token or "")
        return messages


@dataclass
class InternVLPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        processor: "MMProcessor",
        imglens: list[int] | None = None,
    ) -> dict[str, "torch.Tensor"]:
        if len(images) == 0:
            return {}
        image_processor = getattr(processor, "image_processor")
        image_processor_kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            image_processor_kwargs.update({"crop_to_patches": True, "max_patches": 12, "min_patches": 1})

        images = self._regularize_images(
            images,
            image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
            image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
        )["images"]
        images = make_flat_list_of_images(images)
        image_inputs = image_processor(images=images, return_tensors="pt", **image_processor_kwargs)
        image_num_patches = image_inputs.pop("num_patches")
        image_pixel_values = image_inputs.pop("pixel_values")
        image_num_patches_indices = np.cumsum(image_num_patches)
        image_patches = []
        for i in range(len(images)):
            start_index = image_num_patches_indices[i - 1] if i > 0 else 0
            end_index = image_num_patches_indices[i]
            image_patches.append(image_pixel_values[start_index:end_index])
        return {"pixel_values": torch.cat(image_patches, dim=0), "image_num_patches": image_num_patches}

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
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
                num_patches = int(image_num_patches[image_idx]) if self.expand_mm_tokens and len(image_num_patches) else 1
                content = content.replace(IMAGE_PLACEHOLDER, f"<img>{'<IMG_CONTEXT>' * image_seqlen * num_patches}</img>", 1)
                image_idx += 1
            message["content"] = content
        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        imglens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Any]:
        self._validate_input(processor, images)
        if processor is None:
            return {}
        mm_inputs = self._get_mm_inputs(images, processor)
        mm_inputs.pop("image_num_patches", None)
        return mm_inputs


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            image = image.resize((max(image.width, 28), max(image.height, 28)))
        if image.width / image.height > 200:
            image = image.resize((image.height * 180, image.height))
        if image.height / image.width > 200:
            image = image.resize((image.width, image.width * 180))
        return image

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images)
        self._validate_messages(messages, images)
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")
        merge_length = getattr(image_processor, "merge_size", 1) ** 2
        image_idx = 0
        if self.expand_mm_tokens and images:
            image_grid_thw = self._get_mm_inputs(images, processor).get("image_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)

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


@dataclass
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


def get_mm_plugin(name: str, image_token: str | None = None, **kwargs) -> BasePlugin:
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found in slim build.")
    supported_kwargs = {"expand_mm_tokens", "vision_bos_token", "vision_eos_token"}
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}
    return PLUGINS[name](image_token=image_token, **filtered_kwargs)
