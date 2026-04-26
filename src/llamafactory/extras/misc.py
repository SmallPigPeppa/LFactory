import os
import socket

import torch
import torch.distributed as dist


def calculate_tps(dataset, metrics, stage):
    effective_token_num = sum(len(data["input_ids"]) for data in dataset) if stage == "sft" else 0
    result = effective_token_num * metrics["epoch"] / metrics["train_runtime"]
    return result / dist.get_world_size() if dist.is_initialized() else result


def count_parameters(model):
    trainable, total = 0, 0
    for param in model.parameters():
        n = param.numel() or getattr(param, "ds_numel", 0)
        total += n
        trainable += n if param.requires_grad else 0
    return trainable, total


def get_current_device():
    return torch.device(f"cuda:{os.getenv('LOCAL_RANK', '0')}") if torch.cuda.is_available() else torch.device("cpu")


def get_device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def has_tokenized_data(path):
    return path is not None and os.path.isdir(path) and len(os.listdir(path)) > 0


def infer_optim_dtype(model_dtype=None):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def is_env_enabled(env_var, default="0"):
    return os.getenv(env_var, default).lower() in {"1", "true", "y", "yes"}


def find_available_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
