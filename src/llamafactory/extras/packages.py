# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version


if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_jieba_available():
    return _is_package_available("jieba")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


def is_pillow_available():
    return _is_package_available("PIL")


def is_rouge_available():
    return _is_package_available("rouge_chinese")


def is_safetensors_available():
    return _is_package_available("safetensors")


@lru_cache
def is_transformers_version_greater_than(content: str):
    return _get_package_version("transformers") >= version.parse(content)


@lru_cache
def is_torch_version_greater_than(content: str):
    return _get_package_version("torch") >= version.parse(content)
