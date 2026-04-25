# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import os
import subprocess
import sys
from copy import deepcopy


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli env: show environment info                      |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)


def _welcome() -> str:
    from .extras.env import VERSION

    return "-" * 58 + "\n" + f"| LFactory slim train-only build, version {VERSION}" + " " * max(0, 4 - len(VERSION)) + "|\n" + "-" * 58


def launch():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command == "help":
        print(USAGE)
        return
    if command == "version":
        print(_welcome())
        return
    if command == "env":
        from .extras.env import print_env
        print_env()
        return
    if command != "train":
        print(f"Unknown or removed command: {command}.\n{USAGE}")
        return

    from .extras import logging
    from .extras.misc import find_available_port, get_device_count, is_env_enabled

    logger = logging.get_logger(__name__)
    if is_env_enabled("FORCE_TORCHRUN") or get_device_count() > 1:
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed task(s) at: {master_addr}:{master_port}")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")
        if rdzv_id is not None:
            rdzv_nnodes = f"{min_nnodes}:{max_nnodes}" if min_nnodes and max_nnodes else nnodes
            cmd = (
                "torchrun --nnodes {rdzv_nnodes} --nproc-per-node {nproc_per_node} "
                "--rdzv-id {rdzv_id} --rdzv-backend c10d --rdzv-endpoint {master_addr}:{master_port} "
                "--max-restarts {max_restarts} {file_name} {args}"
            ).format(
                rdzv_nnodes=rdzv_nnodes,
                nproc_per_node=nproc_per_node,
                rdzv_id=rdzv_id,
                master_addr=master_addr,
                master_port=master_port,
                max_restarts=max_restarts,
                file_name=__file__,
                args=" ".join(sys.argv[1:]),
            )
        else:
            cmd = (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            ).format(
                nnodes=nnodes,
                node_rank=node_rank,
                nproc_per_node=nproc_per_node,
                master_addr=master_addr,
                master_port=master_port,
                file_name=__file__,
                args=" ".join(sys.argv[1:]),
            )
        process = subprocess.run(cmd.split(), env=env, check=True)
        sys.exit(process.returncode)

    from .train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    from llamafactory.train.tuner import run_exp
    run_exp()
