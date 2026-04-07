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

"""Pipeline Preflight — validate environment before generation or training.

Checks manifest files, prompt formats, model file existence, YAML config
validity, disk space, and dependency availability before a long pipeline run.

Usage:
    python scripts/pipeline_preflight.py --manifest teachers.json --prompts prompts.jsonl
    python scripts/pipeline_preflight.py --config examples/distillation/auto/zena007_sft.yaml
    python scripts/pipeline_preflight.py --matrix data/forge_matrix/zena007_matrix.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


class PreflightReport:
    """Accumulates preflight check results."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.ok: list[str] = []

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def good(self, msg: str) -> None:
        self.ok.append(msg)

    def print_report(self) -> None:
        print("\n=== Pipeline Preflight Report ===\n")  # xray: ignore[PY-004]
        for msg in self.ok:
            print(f"  OK:    {msg}")  # xray: ignore[PY-004]
        for msg in self.warnings:
            print(f"  WARN:  {msg}")  # xray: ignore[PY-004]
        for msg in self.errors:
            print(f"  ERROR: {msg}")  # xray: ignore[PY-004]
        if self.passed:
            print(f"\n  RESULT: READY ({len(self.warnings)} warnings)")  # xray: ignore[PY-004]
        else:
            print(f"\n  RESULT: BLOCKED ({len(self.errors)} errors)")  # xray: ignore[PY-004]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "ok": self.ok,
        }


def check_manifest(path: Path, report: PreflightReport) -> list[dict]:
    """Validate teacher manifest JSON."""
    if not path.exists():
        report.error(f"Manifest not found: {path}")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        report.error(f"Manifest is invalid JSON: {exc}")
        return []

    if not isinstance(data, list):
        # Support both list-of-teachers and dict-with-teachers-key formats
        if isinstance(data, dict) and "teachers" in data and isinstance(data["teachers"], list):
            data = data["teachers"]
        else:
            report.error("Manifest must be a JSON array of teacher objects (or dict with 'teachers' key)")
            return []

    if len(data) == 0:
        report.error("Manifest is empty — no teachers defined")
        return []

    report.good(f"Manifest: {len(data)} teachers defined")

    for i, teacher in enumerate(data):
        name = teacher.get("name", f"teacher_{i}")
        model_path = teacher.get("model_path", "")

        if not name:
            report.warn(f"Teacher {i}: missing 'name' field")

        if model_path:
            mp = Path(model_path)
            if mp.exists():
                size_gb = mp.stat().st_size / (1024**3) if mp.is_file() else 0
                if mp.is_file():
                    report.good(f"Teacher '{name}': model file exists ({size_gb:.1f} GB)")
                elif mp.is_dir():
                    report.good(f"Teacher '{name}': model directory exists")
            else:
                report.error(f"Teacher '{name}': model_path not found: {model_path}")
        else:
            report.warn(f"Teacher '{name}': no model_path specified")

    return data


def check_prompts(path: Path, report: PreflightReport) -> int:
    """Validate prompts file format and content."""
    if not path.exists():
        report.error(f"Prompts file not found: {path}")
        return 0

    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []

    # Try JSONL
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            rows.append(row)
        except (json.JSONDecodeError, ValueError):
            # Try as full JSON array
            try:
                rows = json.loads(text)
                break
            except (json.JSONDecodeError, ValueError):
                report.error(f"Prompts file: malformed JSON at line {line_no}")
                return 0

    if not rows:
        report.error("Prompts file is empty — no prompts to process")
        return 0

    # Check required fields
    missing_id = sum(1 for r in rows if not r.get("id"))
    missing_prompt = sum(1 for r in rows if not r.get("prompt") and not r.get("instruction"))

    if missing_id > 0:
        report.warn(f"Prompts: {missing_id}/{len(rows)} rows missing 'id' field")
    if missing_prompt > 0:
        report.error(f"Prompts: {missing_prompt}/{len(rows)} rows missing 'prompt' field")

    if missing_prompt == 0:
        report.good(f"Prompts: {len(rows)} valid prompt rows")

    return len(rows)


def check_yaml_config(path: Path, report: PreflightReport) -> None:
    """Validate a LlamaFactory YAML training config."""
    if not _HAS_YAML:
        report.warn("PyYAML not installed — skipping YAML config validation")
        return

    if not path.exists():
        report.error(f"YAML config not found: {path}")
        return

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        report.error(f"YAML parse error in {path.name}: {exc}")
        return

    if not isinstance(config, dict):
        report.error(f"{path.name}: config is not a dict")
        return

    report.good(f"{path.name}: valid YAML ({len(config)} keys)")

    # Check model path
    model = config.get("model_name_or_path", "")
    if model:
        mp = Path(model)
        if not mp.exists() and "/" not in model:
            report.warn(f"{path.name}: model_name_or_path '{model}' not found locally (may be HF hub ID)")
    else:
        report.warn(f"{path.name}: no model_name_or_path specified")

    # Check dataset
    dataset = config.get("dataset", "")
    if not dataset:
        report.warn(f"{path.name}: no dataset specified")

    # Check for forbidden keys (known to crash LlamaFactory)
    forbidden = {"booster"}
    found_forbidden = forbidden & set(config.keys())
    if found_forbidden:
        report.error(f"{path.name}: forbidden keys {found_forbidden} — will crash LlamaFactory's parse_dict")


def check_forge_matrix(path: Path, report: PreflightReport) -> None:
    """Validate a forge matrix YAML."""
    if not _HAS_YAML:
        report.warn("PyYAML not installed — skipping matrix validation")
        return

    if not path.exists():
        report.error(f"Matrix file not found: {path}")
        return

    try:
        matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        report.error(f"Matrix YAML parse error: {exc}")
        return

    variants = matrix.get("variants", {})
    if not variants:
        report.error("Matrix has no variants defined")
        return

    report.good(f"Matrix: {len(variants)} variants defined")

    sft_data = matrix.get("sft_data", "")
    if sft_data and not Path(sft_data).exists():
        report.warn(f"Matrix sft_data '{sft_data}' not found (may be relative to data/)")

    for vid, v in variants.items():
        model = v.get("model", "")
        if not model:
            report.warn(f"Variant '{vid}': no model specified")


def check_disk_space(report: PreflightReport, min_gb: float = 5.0) -> None:
    """Check available disk space."""
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    if free_gb < min_gb:
        report.error(f"Low disk space: {free_gb:.1f} GB free (need {min_gb:.0f} GB minimum)")
    else:
        report.good(f"Disk space: {free_gb:.1f} GB free")


def check_dependencies(report: PreflightReport) -> None:
    """Check that critical Python packages are importable."""
    deps = {
        "torch": "PyTorch (training)",
        "transformers": "HuggingFace Transformers",
        "yaml": "PyYAML (config files)",
        "peft": "PEFT (LoRA)",
    }
    for mod, desc in deps.items():
        try:
            __import__(mod)
            report.good(f"Dependency: {desc}")
        except ImportError:
            report.warn(f"Dependency not installed: {desc} ({mod})")


def preflight(
    manifest: Path | None = None,
    prompts: Path | None = None,
    config: Path | None = None,
    matrix: Path | None = None,
    check_deps: bool = True,
    check_disk: bool = True,
) -> PreflightReport:
    """Run all preflight checks. Returns a PreflightReport."""
    report = PreflightReport()

    if manifest:
        check_manifest(manifest, report)
    if prompts:
        check_prompts(prompts, report)
    if config:
        check_yaml_config(config, report)
    if matrix:
        check_forge_matrix(matrix, report)
    if check_disk:
        check_disk_space(report)
    if check_deps:
        check_dependencies(report)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline preflight — validate environment before generation/training.",
        epilog="""\
examples:
  %(prog)s --manifest teachers.json --prompts prompts.jsonl
  %(prog)s --config examples/distillation/auto/zena007_sft.yaml
  %(prog)s --matrix data/forge_matrix/zena007_matrix.yaml
  %(prog)s --manifest teachers.json --prompts prompts.jsonl --no-deps
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", help="Path to teacher_manifest.json.")
    parser.add_argument("--prompts", help="Path to prompts file.")
    parser.add_argument("--config", help="Path to LlamaFactory YAML config.")
    parser.add_argument("--matrix", help="Path to forge_matrix YAML.")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency checks.")
    parser.add_argument("--no-disk", action="store_true", help="Skip disk space check.")
    parser.add_argument("--json-out", help="Write report as JSON.")
    args = parser.parse_args()

    if not any([args.manifest, args.prompts, args.config, args.matrix]):
        parser.error("Provide at least one of --manifest, --prompts, --config, --matrix")

    report = preflight(
        manifest=Path(args.manifest) if args.manifest else None,
        prompts=Path(args.prompts) if args.prompts else None,
        config=Path(args.config) if args.config else None,
        matrix=Path(args.matrix) if args.matrix else None,
        check_deps=not args.no_deps,
        check_disk=not args.no_disk,
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
