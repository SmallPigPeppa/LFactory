# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stateless checkpoint utilities for training resume."""

import glob
import json
import os
import random
import shutil

import numpy as np
import torch

from ...accelerator.helper import DeviceType, get_current_accelerator
from ...utils import logging


logger = logging.get_logger(__name__)

CHECKPOINT_COMPLETE_MARKER = "CHECKPOINT_COMPLETE"


def _parse_checkpoint_step(path: str) -> int:
    """Extract the step number from a checkpoint directory name, or -1 if invalid."""
    try:
        return int(os.path.basename(path).split("-")[-1])
    except ValueError:
        return -1


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest valid checkpoint directory in output_dir."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    ckpt_dirs = [d for d in glob.glob(pattern) if _parse_checkpoint_step(d) >= 0]
    ckpt_dirs.sort(key=_parse_checkpoint_step)
    for d in reversed(ckpt_dirs):
        if os.path.exists(os.path.join(d, CHECKPOINT_COMPLETE_MARKER)):
            return d
    return None


def rotate_checkpoints(output_dir: str, limit: int) -> None:
    """Keep only the latest `limit` complete checkpoints, delete older ones and incomplete leftovers."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    all_dirs = [d for d in glob.glob(pattern) if _parse_checkpoint_step(d) >= 0]
    all_dirs.sort(key=_parse_checkpoint_step)

    complete_dirs = []
    for d in all_dirs:
        if os.path.exists(os.path.join(d, CHECKPOINT_COMPLETE_MARKER)):
            complete_dirs.append(d)
        else:
            shutil.rmtree(d)
            logger.info_rank0(f"Cleaned up incomplete checkpoint: {d}")

    while len(complete_dirs) > limit:
        oldest = complete_dirs.pop(0)
        shutil.rmtree(oldest)
        logger.info_rank0(f"Deleted old checkpoint: {oldest}")


def save_metadata(ckpt_dir: str, **kwargs) -> None:
    """Save training metadata as JSON (rank 0 only)."""
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(kwargs, f, indent=2)


def load_metadata(ckpt_dir: str) -> dict:
    """Load training metadata from a checkpoint directory."""
    with open(os.path.join(ckpt_dir, "metadata.json")) as f:
        return json.load(f)


def _get_accelerator_rng_state():
    """Get RNG state for the current accelerator, device-agnostic."""
    device_type = get_current_accelerator().type
    if device_type == DeviceType.CUDA:
        return torch.cuda.get_rng_state_all()
    elif device_type == DeviceType.NPU:
        return torch.npu.get_rng_state_all()
    elif device_type == DeviceType.XPU:
        return torch.xpu.get_rng_state_all()
    return None


def _set_accelerator_rng_state(state) -> None:
    """Set RNG state for the current accelerator, device-agnostic."""
    if state is None:
        return

    device_type = get_current_accelerator().type
    if device_type == DeviceType.CUDA:
        torch.cuda.set_rng_state_all(state)
    elif device_type == DeviceType.NPU:
        torch.npu.set_rng_state_all(state)
    elif device_type == DeviceType.XPU:
        torch.xpu.set_rng_state_all(state)


def save_rng_state(ckpt_dir: str, rank: int) -> None:
    """Save per-rank RNG states for reproducibility."""
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "accelerator": _get_accelerator_rng_state(),
    }
    torch.save(rng_state, os.path.join(ckpt_dir, f"rng_state_{rank}.pt"))


def load_rng_state(ckpt_dir: str, rank: int) -> None:
    """Restore per-rank RNG states from a checkpoint."""
    path = os.path.join(ckpt_dir, f"rng_state_{rank}.pt")
    if not os.path.exists(path):
        return
    rng_state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(rng_state["torch"])
    _set_accelerator_rng_state(
        rng_state.get("accelerator")
        or (rng_state.get("cuda") if get_current_accelerator().type == DeviceType.CUDA else None)
    )


def mark_checkpoint_complete(ckpt_dir: str) -> None:
    """Write a marker file indicating the checkpoint is fully saved."""
    open(os.path.join(ckpt_dir, CHECKPOINT_COMPLETE_MARKER), "w").close()
