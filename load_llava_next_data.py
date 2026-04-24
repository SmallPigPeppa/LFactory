#!/usr/bin/env python

import argparse
import inspect
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ID = "lmms-lab/LLaVA-NeXT-Data"
DEFAULT_CACHE_DIR = "/ppio_net0/huggingface"
DEFAULT_EXPECTED_SHARDS = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load LLaVA-NeXT-Data from local cache, or download it first.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    return parser.parse_args()


def find_parquet_shards(cache_dir: Path, repo_id: str, expected_shards: int) -> list[Path]:
    repo_name = repo_id.split("/")[-1].lower()
    grouped: dict[Path, list[Path]] = defaultdict(list)

    for path in cache_dir.rglob("train-*.parquet"):
        if repo_name not in str(path).lower():
            continue

        grouped[path.parent].append(path)

    if not grouped:
        return []

    shards = sorted(max(grouped.values(), key=len))
    if expected_shards > 0 and len(shards) < expected_shards:
        return []

    return shards


def download_dataset(repo_id: str, cache_dir: Path, workers: int, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        allow_patterns=["data/train-*.parquet", "README.md", "dataset_infos.json", ".gitattributes"],
        token=token,
    )
    if "max_workers" in inspect.signature(snapshot_download).parameters:
        kwargs["max_workers"] = workers

    return Path(snapshot_download(**kwargs))


def summarize(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"

    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            output[key] = summarize(item)
        return output

    if isinstance(value, list):
        return [summarize(item) for item in value[:2]]

    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "..."

    return value


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    shards = find_parquet_shards(cache_dir, args.repo_id, args.expected_shards)
    if shards:
        print(f"Found {len(shards)} parquet shards in local cache: {shards[0].parent}")
    else:
        print(f"Dataset is not complete in {cache_dir}. Downloading with {args.workers} workers...")
        snapshot_dir = download_dataset(args.repo_id, cache_dir, args.workers, args.token)
        shards = sorted((snapshot_dir / "data").glob("train-*.parquet"))
        if not shards:
            shards = find_parquet_shards(snapshot_dir, args.repo_id, expected_shards=0)

    if not shards:
        raise RuntimeError(f"No parquet shards found for {args.repo_id} in {cache_dir}.")

    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files={args.split: [str(path) for path in shards]},
        split=args.split,
        streaming=args.streaming,
    )

    print(f"Loaded dataset split={args.split!r}, streaming={args.streaming}.")
    if args.streaming:
        sample = next(iter(dataset))
        print(f"Columns: {list(sample.keys())}")
    else:
        print(f"Rows: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        sample = dataset[0]

    print("First sample:")
    print(summarize(sample))


if __name__ == "__main__":
    main()
