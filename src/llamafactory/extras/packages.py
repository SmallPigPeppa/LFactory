# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import importlib.util


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
