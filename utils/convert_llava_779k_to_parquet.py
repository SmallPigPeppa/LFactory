#!/usr/bin/env python3

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator
from tqdm import tqdm


SKIP_ROW = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LLaVA/LLaVA-NeXT 779K style data to Parquet with raw image bytes embedded."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input_json",
        type=str,
        help="Local annotation file. Supports .json top-level list and .jsonl.",
    )
    source.add_argument(
        "--hf_dataset",
        type=str,
        help="Hugging Face dataset name, e.g. lmms-lab/LLaVA-NeXT-Data.",
    )

    parser.add_argument("--split", type=str, default="train", help="Dataset split for Hugging Face loading.")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for relative local image paths.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write Parquet shards.")
    parser.add_argument("--prefix", type=str, default="train", help="Parquet shard file prefix.")

    parser.add_argument("--image_col", type=str, default="image", help="Input/output image column name.")
    parser.add_argument(
        "--messages_col",
        type=str,
        default="conversations",
        help="Conversation/messages column name for optional LLaMA-Factory dataset_info.json.",
    )
    parser.add_argument(
        "--keep_columns",
        nargs="*",
        default=None,
        help="Optional list of columns to keep. By default all input columns are kept.",
    )

    parser.add_argument("--num_workers", type=int, default=min(32, (os.cpu_count() or 8) * 4))
    parser.add_argument("--batch_size", type=int, default=1024, help="Rows processed by the thread pool at a time.")
    parser.add_argument("--rows_per_file", type=int, default=10000, help="Rows per Parquet shard.")
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Convert at most N rows. Default: all samples.",
    )
    parser.add_argument(
        "--row_error_mode",
        type=str,
        default="error",
        choices=["error", "keep_none", "skip"],
        help="Behavior when row/image normalization fails. `skip` skips the whole row; `keep_none` only applies to missing images.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset for HF mode.",
    )

    parser.add_argument(
        "--write_llamafactory_info",
        action="store_true",
        help="Create/update a LLaMA-Factory dataset_info.json next to output_dir.",
    )
    return parser.parse_args()


def iter_json_or_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError(f"{path}:{line_no} is not a JSON object.")
                yield obj
        return

    try:
        import ijson  # type: ignore

        with path.open("rb") as f:
            for obj in ijson.items(f, "item"):
                if not isinstance(obj, dict):
                    raise TypeError(f"An item in {path} is not a JSON object.")
                yield obj
        return
    except ImportError:
        pass

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{path} must be a JSON list or JSONL file.")

    for obj in data:
        if not isinstance(obj, dict):
            raise TypeError(f"An item in {path} is not a JSON object.")
        yield obj


def iter_hf_dataset(args: argparse.Namespace) -> tuple[Iterable[dict[str, Any]], int | None]:
    try:
        from datasets import Image, load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "HF mode requires `datasets`. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(
        args.hf_dataset,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
    )

    # Make sure HF Image is returned as {"bytes": ..., "path": ...}, not PIL.Image.
    if args.image_col in getattr(ds, "column_names", []):
        try:
            ds = ds.cast_column(args.image_col, Image(decode=False))
        except Exception as exc:
            print(
                f"[WARN] Could not cast column `{args.image_col}` to Image(decode=False): {exc}",
                file=sys.stderr,
            )
            print(
                "[WARN] If this column yields PIL images, original encoding may not be preserved.",
                file=sys.stderr,
            )

    total = len(ds) if hasattr(ds, "__len__") else None
    return ds, total


def limit_iter(it: Iterable[dict[str, Any]], max_samples: int | None) -> Iterator[dict[str, Any]]:
    if max_samples is None:
        yield from it
    else:
        yield from islice(it, max_samples)


def batched(it: Iterable[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(it)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def resolve_image_path(path: str, image_root: str | None) -> str:
    """Resolve image path. Preserve original path in Parquet metadata, read local_path for bytes."""
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    if image_root:
        candidate = os.path.join(os.path.expanduser(image_root), path)
        if os.path.isfile(candidate):
            return candidate
    return path


def read_binary_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def handle_missing_image(message: str, args: argparse.Namespace):
    if args.row_error_mode == "error":
        raise FileNotFoundError(message)
    return SKIP_ROW if args.row_error_mode == "skip" else None


def load_image_from_path(original_path: str, args: argparse.Namespace) -> dict[str, Any] | None | object:
    local_path = resolve_image_path(original_path, args.image_root)
    if os.path.isfile(local_path):
        return {"bytes": read_binary_file(local_path), "path": original_path}
    return handle_missing_image(
        f"Image file not found: {original_path} ; tried: {local_path}",
        args,
    )


def normalize_one_image(image: Any, args: argparse.Namespace) -> dict[str, Any] | None | object:
    if image is None:
        return handle_missing_image("Image value is None.", args)

    if isinstance(image, dict):
        raw_bytes = image.get("bytes")
        image_path = image.get("path") or image.get("file_name") or image.get("filename")
        if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            return {"bytes": raw_bytes, "path": str(image_path) if image_path is not None else None}
        if image_path is not None:
            return load_image_from_path(str(image_path), args)
        raise ValueError(f"Image dict has neither bytes nor path: keys={list(image.keys())}")

    if isinstance(image, (bytes, bytearray)):
        return {"bytes": bytes(image), "path": None}
    if isinstance(image, memoryview):
        return {"bytes": image.tobytes(), "path": None}

    if isinstance(image, (str, os.PathLike)):
        return load_image_from_path(os.fspath(image), args)

    raise TypeError(f"Unsupported image value type: {type(image)!r}")


def normalize_image_value(value: Any, args: argparse.Namespace) -> Any:
    # Support both single-image column `image` and multi-image column `images`.
    if isinstance(value, list):
        out = []
        for item in value:
            normalized = normalize_one_image(item, args)
            if normalized is SKIP_ROW:
                return SKIP_ROW
            out.append(normalized)
        return out
    return normalize_one_image(value, args)


def process_record(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        if args.keep_columns is None:
            out = dict(row)
        else:
            out = {name: row.get(name) for name in args.keep_columns}
            out[args.image_col] = row.get(args.image_col)

        out[args.image_col] = normalize_image_value(row.get(args.image_col), args)
        if out[args.image_col] is SKIP_ROW:
            return None
        return out
    except Exception as exc:
        if args.row_error_mode == "skip":
            row_id = row.get("id", row.get("uid", "<no id>"))
            print(f"[WARN] skip bad row id={row_id}: {exc}", file=sys.stderr)
            return None
        raise


class ShardedParquetWriter:
    def __init__(
        self,
        output_dir: str | Path,
        prefix: str,
        image_col: str,
        rows_per_file: int,
        compression: str | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.image_col = image_col
        self.rows_per_file = rows_per_file
        self.compression = None if compression == "none" else compression

        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        self.pa = pa
        self.pq = pq

        self.schema = None
        self.writer = None
        self.shard_idx = 0
        self.rows_in_current_shard = 0
        self.total_rows = 0
        self.files: list[Path] = []

    def infer_schema(self, rows: list[dict[str, Any]]):
        pa = self.pa
        image_struct = pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ])

        image_is_list = False
        for row in rows:
            value = row.get(self.image_col)
            if isinstance(value, list):
                image_is_list = True
                break
        image_type = pa.list_(image_struct) if image_is_list else image_struct

        rows_without_image = [{k: v for k, v in row.items() if k != self.image_col} for row in rows]
        base_schema = pa.Table.from_pylist(rows_without_image).schema if rows_without_image else pa.schema([])
        base_fields = {field.name: field for field in base_schema}

        # Preserve column order from the first observed rows.
        ordered_names: list[str] = []
        seen = set()
        for row in rows:
            for name in row.keys():
                if name not in seen:
                    ordered_names.append(name)
                    seen.add(name)

        fields = []
        added = set()
        for name in ordered_names:
            if name == self.image_col:
                fields.append(pa.field(name, image_type))
                added.add(name)
            elif name in base_fields:
                fields.append(base_fields[name])
                added.add(name)

        for field in base_schema:
            if field.name not in added:
                fields.append(field)

        if self.image_col not in added:
            fields.append(pa.field(self.image_col, image_type))

        return pa.schema(fields)

    def _open_next_shard(self) -> None:
        path = self.output_dir / f"{self.prefix}-{self.shard_idx:05d}.parquet"
        self.writer = self.pq.ParquetWriter(
            where=str(path),
            schema=self.schema,
            compression=self.compression,
            use_dictionary=True,
        )
        self.files.append(path)
        self.rows_in_current_shard = 0
        self.shard_idx += 1

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self.schema is None:
            self.schema = self.infer_schema(rows)

        offset = 0
        while offset < len(rows):
            if self.writer is None or self.rows_in_current_shard >= self.rows_per_file:
                self.close_current()
                self._open_next_shard()

            take = min(len(rows) - offset, self.rows_per_file - self.rows_in_current_shard)
            subrows = rows[offset: offset + take]
            table = self.pa.Table.from_pylist(subrows, schema=self.schema)
            self.writer.write_table(table)
            self.rows_in_current_shard += take
            self.total_rows += take
            offset += take

    def close_current(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        self.close_current()


def get_llamafactory_dataset_name(args: argparse.Namespace, files_dir: Path) -> str:
    if args.hf_dataset:
        return args.hf_dataset
    return files_dir.name


def write_llamafactory_dataset_info(args: argparse.Namespace, files_dir: Path) -> tuple[Path, str]:
    info_path = files_dir.parent / "dataset_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            try:
                info = json.load(f)
            except json.JSONDecodeError:
                info = {}
    else:
        info = {}

    dataset_name = get_llamafactory_dataset_name(args, files_dir)
    file_name = os.path.relpath(files_dir, info_path.parent)
    info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": args.messages_col,
            "images": args.image_col,
        },
    }

    tmp_path = info_path.with_suffix(info_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(info_path)
    return info_path, dataset_name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_dataset:
        records, total = iter_hf_dataset(args)
    else:
        records = iter_json_or_jsonl(args.input_json)
        total = None

    if args.max_samples is not None:
        total = min(total, args.max_samples) if total is not None else args.max_samples

    sink = ShardedParquetWriter(
        output_dir=output_dir,
        prefix=args.prefix,
        image_col=args.image_col,
        rows_per_file=args.rows_per_file,
        compression=args.compression,
    )

    n_seen = 0
    n_written = 0
    n_skipped = 0
    pbar = tqdm(total=total, desc="convert")

    try:
        limited_records = limit_iter(records, args.max_samples)
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for raw_batch in batched(limited_records, args.batch_size):
                processed_batch = list(executor.map(lambda r: process_record(r, args), raw_batch))
                good_batch = [row for row in processed_batch if row is not None]

                sink.write_rows(good_batch)
                n_seen += len(raw_batch)
                n_written += len(good_batch)
                n_skipped += len(raw_batch) - len(good_batch)
                pbar.update(len(raw_batch))
    finally:
        pbar.close()
        sink.close()

    print(f"Done. seen={n_seen}, written={n_written}, skipped={n_skipped}")
    print(f"Output dir: {output_dir}")
    print(f"Parquet shards: {len(sink.files)}")
    if sink.files:
        print("First shard:", sink.files[0])

    if args.write_llamafactory_info:
        info_path, dataset_name = write_llamafactory_dataset_info(args, output_dir)
        print(f"Updated LLaMA-Factory dataset_info.json: {info_path}")
        print(f"Dataset name: {dataset_name}")


if __name__ == "__main__":
    main()
