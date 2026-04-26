import os
import subprocess
import sys
from copy import deepcopy


USAGE = """Usage:
  llamafactory-cli train <yaml>
  lmf train <yaml>
"""


def launch():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "help":
        print(USAGE)
        return
    if command != "train":
        print(f"Unknown command: {command}\n{USAGE}")
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
