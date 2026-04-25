# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import os
from collections import OrderedDict, defaultdict
from enum import StrEnum, unique

SAFE_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
WEIGHTS_NAME = "pytorch_model.bin"

CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

DATA_CONFIG = "dataset_info.json"
DEFAULT_TEMPLATE = defaultdict(str)
FILEEXT2TYPE = {"parquet": "parquet"}
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
LAYERNORM_NAMES = {"norm", "ln"}
METHODS = ["full", "freeze", "lora", "oft"]
MROPE_MODELS = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
MULTIMODAL_SUPPORTED_MODELS = {"llava", "llava_next", "internvl", "interns1", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
PEFT_METHODS = {"lora", "oft"}
RUNNING_LOG = "running_log.txt"
STAGES_USE_PAIR_DATA = set()
SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}
SUPPORTED_MODELS = OrderedDict()
TRAINER_LOG = "trainer_log.jsonl"
TRAINING_ARGS = "training_args.yaml"
TRAINING_STAGES = {"Supervised Fine-Tuning": "sft", "Pre-Training": "pt"}


class AttentionFunction(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"
    FA3 = "fa3"


class EngineName(StrEnum):
    HF = "huggingface"


class DownloadSource(StrEnum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"


@unique
class QuantizationMethod(StrEnum):
    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"
    MXFP4 = "mxfp4"
    FP8 = "fp8"


class RopeScaling(StrEnum):
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"
