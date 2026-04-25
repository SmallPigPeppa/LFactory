# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import json
import os
from dataclasses import dataclass
from typing import Literal

from ..extras.constants import DATA_CONFIG


@dataclass
class DatasetAttr:
    r"""Dataset metadata for the slim build.

    Only local 779k-style parquet datasets are supported. The dataset should be
    declared in `dataset_info.json` with `file_name`, `formatting=sharegpt`, and
    `columns.messages` / `columns.images`.
    """

    load_from: Literal["file"]
    dataset_name: str
    formatting: Literal["sharegpt"] = "sharegpt"
    split: str = "train"
    num_samples: int | None = None

    messages: str = "conversations"
    images: str = "image"
    system: str | None = None
    tools: str | None = None

    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    observation_tag: str = "observation"
    function_tag: str = "function_call"
    system_tag: str = "system"

    def __repr__(self) -> str:
        return self.dataset_name


def _ensure_parquet_path(name: str) -> None:
    base = os.path.basename(name.rstrip(os.sep))
    # A directory named llava779k under /datasets/parquet is expected and valid.
    # A single file must be .parquet.
    if "." in base and not base.endswith(".parquet"):
        raise ValueError("Slim data loader only supports parquet files or parquet directories.")


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str) -> list[DatasetAttr]:
    if dataset_names is None:
        return []

    config_path = os.path.join(dataset_dir, DATA_CONFIG)
    if not os.path.isfile(config_path):
        raise ValueError(f"Cannot find {DATA_CONFIG} in dataset_dir: {dataset_dir}.")

    with open(config_path, encoding="utf-8") as f:
        dataset_info = json.load(f)

    attrs: list[DatasetAttr] = []
    for dataset_name in dataset_names:
        if dataset_name not in dataset_info:
            raise ValueError(f"Undefined dataset {dataset_name} in {config_path}.")

        info = dataset_info[dataset_name]
        if "file_name" not in info:
            raise ValueError("Slim build requires `file_name` in dataset_info.json.")
        if info.get("formatting", "sharegpt") != "sharegpt":
            raise ValueError("Slim build only supports 779k/sharegpt parquet formatting.")

        file_name = info["file_name"]
        _ensure_parquet_path(file_name)
        columns = info.get("columns", {})
        tags = info.get("tags", {})
        attrs.append(
            DatasetAttr(
                load_from="file",
                dataset_name=file_name,
                formatting="sharegpt",
                split=info.get("split", "train"),
                num_samples=info.get("num_samples"),
                messages=columns.get("messages", "conversations"),
                images=columns.get("images", "image"),
                system=columns.get("system"),
                tools=columns.get("tools"),
                role_tag=tags.get("role_tag", "from"),
                content_tag=tags.get("content_tag", "value"),
                user_tag=tags.get("user_tag", "human"),
                assistant_tag=tags.get("assistant_tag", "gpt"),
                observation_tag=tags.get("observation_tag", "observation"),
                function_tag=tags.get("function_tag", "function_call"),
                system_tag=tags.get("system_tag", "system"),
            )
        )

    return attrs
