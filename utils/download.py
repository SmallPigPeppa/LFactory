from pathlib import Path

from huggingface_hub import snapshot_download

WORKERS = 16
BASE_DIR = Path("/ppio_net0/download")


def download_model(repo_id: str, base_dir=BASE_DIR) -> str:
    # Save to: <base_dir>/<repo_name>
    base_dir = Path(base_dir)
    local_dir = base_dir / repo_id.rsplit("/", 1)[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    return snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        max_workers=WORKERS,
    )


if __name__ == "__main__":
    for repo_id in [
        # "Qwen/Qwen3-VL-2B-Instruct",
        # "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "OpenGVLab/InternVL3-2B"
    ]:
        download_model(repo_id)
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    # processor = Qwen3VLProcessor()

    # from transformers.models.qwen3_vl.modular_qwen3_vl import
    # from transformers.models.llava LlavaProcessor
