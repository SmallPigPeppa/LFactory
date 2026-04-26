import os
from dataclasses import dataclass
from typing import Any

from ..extras import logging
from .data_utils import Role

logger = logging.get_logger(__name__)


@dataclass
class ShareGPT779KConverter:
    dataset_attr: Any
    data_args: Any

    def _find_images(self, images):
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
                path = os.path.join(self.data_args.media_dir, image)
                if os.path.isfile(path):
                    images[i] = path
        return images

    def __call__(self, example):
        attr = self.dataset_attr
        messages = example[attr.messages]
        tag_mapping = {
            attr.user_tag: Role.USER.value,
            attr.assistant_tag: Role.ASSISTANT.value,
            attr.observation_tag: Role.OBSERVATION.value,
            attr.function_tag: Role.FUNCTION.value,
            attr.system_tag: Role.SYSTEM.value,
        }
        if attr.system_tag and messages and messages[0][attr.role_tag] == attr.system_tag:
            system = messages[0][attr.content_tag]
            messages = messages[1:]
        else:
            system = example[attr.system] if attr.system else ""

        aligned, broken = [], False
        odd_tags = (attr.user_tag, attr.observation_tag)
        even_tags = (attr.assistant_tag, attr.function_tag)
        for turn_idx, message in enumerate(messages):
            valid_tags = odd_tags if turn_idx % 2 == 0 else even_tags
            if message[attr.role_tag] not in valid_tags:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken = True
                break
            aligned.append({"role": tag_mapping[message[attr.role_tag]], "content": message[attr.content_tag]})

        if len(aligned) % 2 != 0:
            broken = True
        prompt, response = ([], []) if broken else (aligned[:-1], aligned[-1:])
        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[attr.tools] if attr.tools else "",
            "_images": self._find_images(example[attr.images]) if attr.images else None,
        }


def align_dataset(dataset, dataset_attr, data_args, training_args):
    converter = ShareGPT779KConverter(dataset_attr, data_args)
    column_names = list(next(iter(dataset)).keys())
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Converting 779k parquet rows",
    )
    return dataset.map(converter, remove_columns=column_names, **kwargs)
