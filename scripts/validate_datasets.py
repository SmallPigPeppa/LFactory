#!/usr/bin/env python
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

"""Dataset Validation — safety layer for the distillation pipeline.

Checks for duplicate prompts, eval-train data leakage, category distribution
anomalies, byte-range sanity, and minimum diversity before training starts.

Usage:
    python scripts/validate_datasets.py \\
        --sft-data data/zena007/purified/consensus_sft.jsonl \\
        --dpo-data data/zena007/purified/conflict_dpo.jsonl \\
        --train-data data/zena007/purified/train_sft.jsonl \\
        --probe-data data/zena007/purified/eval_probes.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL rows, skipping malformed lines."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass
    return rows


def _prompt_hash(text: str) -> str:
    """Deterministic hash of prompt text for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _extract_prompt(row: dict) -> str:
    """Extract prompt text from either SFT or DPO format."""
    return row.get("instruction", row.get("prompt", ""))


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

class ValidationReport:
    """Accumulates validation findings."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)

    def print_report(self) -> None:
        print("\n=== Dataset Validation Report ===\n")  # xray: ignore[PY-004]
        for msg in self.info:
            print(f"  INFO: {msg}")  # xray: ignore[PY-004]
        for msg in self.warnings:
            print(f"  WARN: {msg}")  # xray: ignore[PY-004]
        for msg in self.errors:
            print(f"  ERROR: {msg}")  # xray: ignore[PY-004]
        if self.passed:
            print(f"\n  RESULT: PASS ({len(self.warnings)} warnings)")  # xray: ignore[PY-004]
        else:
            print(f"\n  RESULT: FAIL ({len(self.errors)} errors, {len(self.warnings)} warnings)")  # xray: ignore[PY-004]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def check_duplicates(rows: list[dict], dataset_name: str, report: ValidationReport) -> set[str]:
    """Check for duplicate prompts within a dataset. Returns set of prompt hashes."""
    hashes: dict[str, int] = Counter()
    for row in rows:
        h = _prompt_hash(_extract_prompt(row))
        hashes[h] += 1

    duplicates = {h: c for h, c in hashes.items() if c > 1}
    if duplicates:
        total_dupes = sum(c - 1 for c in duplicates.values())
        report.warn(
            f"{dataset_name}: {total_dupes} duplicate prompts found "
            f"({len(duplicates)} unique prompts appear more than once)"
        )
    else:
        report.note(f"{dataset_name}: no duplicate prompts ({len(rows)} unique)")
    return set(hashes.keys())


def check_leakage(
    train_rows: list[dict],
    probe_rows: list[dict],
    report: ValidationReport,
) -> None:
    """Check that eval probes do not appear in training data."""
    train_hashes = {_prompt_hash(_extract_prompt(r)) for r in train_rows}
    probe_hashes = {_prompt_hash(_extract_prompt(r)) for r in probe_rows}

    leaked = train_hashes & probe_hashes
    if leaked:
        report.error(
            f"DATA LEAKAGE: {len(leaked)} eval probes found in training data! "
            f"Eval scores will be inflated. De-duplicate before training."
        )
    else:
        report.note(
            f"No data leakage: 0/{len(probe_hashes)} probes found in "
            f"{len(train_hashes)} training samples"
        )


def check_category_distribution(
    rows: list[dict], dataset_name: str, report: ValidationReport,
    min_per_category: int = 3,
) -> None:
    """Check category distribution for balance and minimum counts."""
    categories: Counter = Counter()
    for row in rows:
        cat = row.get("category", "")
        if not cat:
            sid = row.get("id", "")
            if sid.startswith("tr-detect"):
                cat = "detect"
            elif sid.startswith("tr-"):
                cat = "translation"
            elif sid.startswith("chat-"):
                cat = "chat"
            elif sid.startswith("ocr-"):
                cat = "ocr"
            else:
                cat = "other"
        categories[cat] += 1

    if not categories:
        report.warn(f"{dataset_name}: no categories detected")
        return

    total = sum(categories.values())
    report.note(
        f"{dataset_name}: {len(categories)} categories, {total} total samples"
    )

    # Check for under-represented categories
    for cat, count in categories.most_common():
        pct = count / total * 100
        if count < min_per_category:
            report.warn(
                f"{dataset_name}: category '{cat}' has only {count} samples "
                f"(< {min_per_category} minimum) — low confidence in eval"
            )
        report.note(f"  {cat}: {count} ({pct:.1f}%)")

    # Check for extreme imbalance
    most_common_count = categories.most_common(1)[0][1]
    least_common_count = categories.most_common()[-1][1]
    if most_common_count > 0 and least_common_count > 0:
        imbalance_ratio = most_common_count / least_common_count
        if imbalance_ratio > 10:
            report.warn(
                f"{dataset_name}: extreme category imbalance — "
                f"ratio {imbalance_ratio:.0f}:1 between largest and smallest"
            )


def check_diversity(rows: list[dict], dataset_name: str, report: ValidationReport) -> None:
    """Check that outputs are not all identical (degenerate SFT data)."""
    outputs = [row.get("output", row.get("chosen", "")) for row in rows]
    unique_outputs = len(set(outputs))

    if len(outputs) > 1 and unique_outputs == 1:
        report.error(
            f"{dataset_name}: ALL {len(outputs)} outputs are identical — "
            f"degenerate training data, student will collapse"
        )
    elif len(outputs) > 10 and unique_outputs < len(outputs) * 0.1:
        report.warn(
            f"{dataset_name}: very low output diversity — "
            f"only {unique_outputs}/{len(outputs)} unique outputs ({unique_outputs / len(outputs):.0%})"
        )
    else:
        report.note(
            f"{dataset_name}: output diversity OK "
            f"({unique_outputs}/{len(outputs)} unique, {unique_outputs / max(len(outputs), 1):.0%})"
        )


def check_byte_ranges(rows: list[dict], dataset_name: str, report: ValidationReport) -> None:
    """Sanity-check sample sizes (too short or too long)."""
    if not rows:
        return

    sizes = []
    for row in rows:
        text = _extract_prompt(row) + " " + row.get("output", row.get("chosen", ""))
        sizes.append(len(text.encode("utf-8")))

    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    report.note(
        f"{dataset_name}: byte stats — min={min_size}, avg={avg_size:.0f}, max={max_size}"
    )

    empty_count = sum(1 for s in sizes if s < 10)
    if empty_count > 0:
        report.warn(
            f"{dataset_name}: {empty_count} samples are nearly empty (< 10 bytes)"
        )

    huge_count = sum(1 for s in sizes if s > 50000)
    if huge_count > 0:
        report.warn(
            f"{dataset_name}: {huge_count} samples exceed 50KB — "
            f"may be truncated during training"
        )


def check_dpo_validity(rows: list[dict], report: ValidationReport) -> None:
    """Validate DPO pairs have proper chosen/rejected structure."""
    missing_chosen = 0
    missing_rejected = 0
    identical_pairs = 0

    for row in rows:
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if not chosen.strip():
            missing_chosen += 1
        if not rejected.strip():
            missing_rejected += 1
        if chosen.strip() == rejected.strip() and chosen.strip():
            identical_pairs += 1

    if missing_chosen:
        report.error(f"DPO data: {missing_chosen} rows have empty 'chosen' field")
    if missing_rejected:
        report.error(f"DPO data: {missing_rejected} rows have empty 'rejected' field")
    if identical_pairs:
        report.warn(
            f"DPO data: {identical_pairs} pairs have identical chosen/rejected — "
            f"no learning signal from these"
        )


def check_dataset_info_paths(report: ValidationReport, base_dir: Path = Path("data")) -> None:
    """Verify that all file_name entries in dataset_info.json point to existing files."""
    ds_info_path = base_dir / "dataset_info.json"
    if not ds_info_path.exists():
        report.warn("dataset_info.json not found — cannot verify dataset paths")
        return

    try:
        ds_info = json.loads(ds_info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        report.error("dataset_info.json is malformed JSON")
        return

    for name, config in ds_info.items():
        file_name = config.get("file_name", "")
        if not file_name:
            continue
        resolved = base_dir / file_name
        if not resolved.exists():
            report.error(
                f"dataset_info.json['{name}']: file_name '{file_name}' "
                f"does not exist at {resolved}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(
    sft_path: Path | None = None,
    dpo_path: Path | None = None,
    train_path: Path | None = None,
    probe_path: Path | None = None,
    check_ds_info: bool = True,
) -> ValidationReport:
    """Run all validation checks. Returns a ValidationReport."""
    report = ValidationReport()

    # Load datasets
    sft_rows = _load_jsonl(sft_path) if sft_path else []
    dpo_rows = _load_jsonl(dpo_path) if dpo_path else []
    train_rows = _load_jsonl(train_path) if train_path else []
    probe_rows = _load_jsonl(probe_path) if probe_path else []

    # Check duplicates within each dataset
    if sft_rows:
        check_duplicates(sft_rows, "SFT", report)
        check_diversity(sft_rows, "SFT", report)
        check_byte_ranges(sft_rows, "SFT", report)
        check_category_distribution(sft_rows, "SFT", report)

    if dpo_rows:
        check_duplicates(dpo_rows, "DPO", report)
        check_byte_ranges(dpo_rows, "DPO", report)
        check_dpo_validity(dpo_rows, report)

    # Check train/probe leakage
    if train_rows and probe_rows:
        check_leakage(train_rows, probe_rows, report)
    elif sft_rows and probe_rows:
        # If no separate train file, check SFT vs probes
        check_leakage(sft_rows, probe_rows, report)

    # Check dataset_info.json references
    if check_ds_info:
        check_dataset_info_paths(report)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate distillation datasets before training.")
    parser.add_argument("--sft-data", help="Path to consensus_sft.jsonl")
    parser.add_argument("--dpo-data", help="Path to conflict_dpo.jsonl")
    parser.add_argument("--train-data", help="Path to train_sft.jsonl (post-split)")
    parser.add_argument("--probe-data", help="Path to eval_probes.jsonl")
    parser.add_argument("--no-ds-info", action="store_true", help="Skip dataset_info.json validation")
    parser.add_argument("--json-out", help="Write report as JSON to this path")
    args = parser.parse_args()

    report = validate(
        sft_path=Path(args.sft_data) if args.sft_data else None,
        dpo_path=Path(args.dpo_data) if args.dpo_data else None,
        train_path=Path(args.train_data) if args.train_data else None,
        probe_path=Path(args.probe_data) if args.probe_data else None,
        check_ds_info=not args.no_ds_info,
    )

    report.print_report()

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
