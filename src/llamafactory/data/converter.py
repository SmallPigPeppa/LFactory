# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..extras import logging
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .parser import DatasetAttr


logger = logging.get_logger(__name__)


@dataclass
class ShareGPT779KConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_images(self, images: Any) -> list[Any] | None:
        if images is None:
            return None
        if isinstance(images, (str, bytes, dict)) or hasattr(images, "read"):
            images = [images]
        elif not isinstance(images, list):
            images = [images]
        elif len(images) == 0:
            return None
        else:
            images = images[:]

        for i, image in enumerate(images):
            if isinstance(image, str):
                image_path = os.path.join(self.data_args.media_dir, image)
                if os.path.isfile(image_path):
                    images[i] = image_path
        return images

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = example[self.dataset_attr.messages]
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken = True
                break
            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if len(aligned_messages) % 2 != 0:
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken = True

        if broken:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        else:
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_images(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
        }


def align_dataset(
    dataset: "Dataset | IterableDataset",
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> "Dataset | IterableDataset":
    converter = ShareGPT779KConverter(dataset_attr=dataset_attr, data_args=data_args)
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting 779k parquet rows to ShareGPT format",
        )

    return dataset.map(converter, remove_columns=column_names, **kwargs)
