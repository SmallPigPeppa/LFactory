import os
from enum import StrEnum

DATA_CONFIG = "dataset_info.json"
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
RUNNING_LOG = "running_log.txt"
TRAINER_LOG = "trainer_log.jsonl"

CHECKPOINT_NAMES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors.index.json",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "pytorch_model.bin",
}

MROPE_MODELS = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
SUPPORTED_VLM_MODELS = {
    "llava",
    "llava_next",
    "internvl",
    "interns1",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
}


class AttentionFunction(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"
