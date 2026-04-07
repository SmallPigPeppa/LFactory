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

"""Cross-Platform Pipeline Orchestrator — Python replacement for run_zena007_end_to_end.ps1.

Runs all distillation pipeline stages in sequence:
  1. Generate multi-teacher responses
  2. Purify into consensus (SFT) / conflict (DPO) splits
  3. Auto-generate training configs
  4. Register datasets
  5. Train SFT → [DPO if samples] → Merge
  OR: Student Forge Matrix (parallel training)

Every stage is idempotent — re-run after any crash.

Usage:
    python scripts/orchestrate_pipeline.py --tag zena007
    python scripts/orchestrate_pipeline.py --tag zena007 --skip-train
    python scripts/orchestrate_pipeline.py --tag zena007 --use-forge
    python scripts/orchestrate_pipeline.py --tag zena007 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _find_python() -> str:
    """Find the Python interpreter."""
    candidates = [
        ".venv-py314/Scripts/python.exe",
        ".venv-py314/bin/python",
        ".venv/Scripts/python.exe",
        ".venv/bin/python",
        sys.executable,
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return sys.executable


def _run_step(name: str, cmd: list[str], dry_run: bool = False) -> int:
    """Run a pipeline step, printing status."""
    print(f"\n{'='*60}")  # xray: ignore[PY-004]
    print(f"[STEP] {name}")  # xray: ignore[PY-004]
    print(f"  CMD: {' '.join(cmd)}")  # xray: ignore[PY-004]
    print(f"{'='*60}\n")  # xray: ignore[PY-004]

    if dry_run:
        print("  (dry-run — skipped)")  # xray: ignore[PY-004]
        return 0

    start = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - start
    print(f"\n  → {name}: {'OK' if result.returncode == 0 else 'FAILED'} ({elapsed:.0f}s)")  # xray: ignore[PY-004]
    return result.returncode


def _file_nonempty(path: str | Path) -> bool:
    """Check if a file exists and is non-empty."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def _read_yaml(path: str | Path) -> dict:
    """Read a YAML config file."""
    if not _HAS_YAML:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform pipeline orchestrator for multi-teacher distillation.",
        epilog="""\
examples:
  %(prog)s --tag zena007
  %(prog)s --tag zena007 --skip-train --skip-dpo
  %(prog)s --tag zena007 --use-forge
  %(prog)s --tag zena007 --dry-run
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", default="zena007", help="Pipeline run tag.")
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Student model name/path.")
    parser.add_argument("--py", default="", help="Python interpreter path (auto-detected if empty).")
    parser.add_argument("--skip-generate", action="store_true", help="Skip generation step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training steps.")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO training.")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step.")
    parser.add_argument("--use-forge", action="store_true", help="Use Student Forge Matrix (parallel).")
    parser.add_argument("--forge-matrix", default="", help="Forge matrix YAML path.")
    parser.add_argument("--cpu-safe", action="store_true", help="CPU-only mode (disable bf16).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    tag = args.tag
    py = args.py or _find_python()
    scripts = Path("scripts")
    dry = args.dry_run

    # Paths
    responses = f"data/{tag}/teacher_responses.jsonl"
    prompts = f"data/{tag}_prompts.jsonl"
    purified_dir = f"data/{tag}/purified"
    config_dir = "examples/distillation/auto"
    sft_cfg = f"{config_dir}/{tag}_sft.yaml"
    merge_cfg = f"{config_dir}/{tag}_merge.yaml"
    merged_dir = f"saves/{tag}/merged"
    forge_matrix = args.forge_matrix or f"data/forge_matrix/{tag}_matrix.yaml"

    steps_run = 0
    steps_skipped = 0
    failed = False

    # ── Stage 1: Generate ────────────────────────────────────────────────
    if not args.skip_generate:
        manifest = f"data/{tag}/teacher_manifest.json"
        if _file_nonempty(responses):
            print(f"[SKIP] Generation: {responses} already exists")  # xray: ignore[PY-004]
            steps_skipped += 1
        elif not Path(manifest).exists():
            print(f"[SKIP] Generation: manifest {manifest} not found")  # xray: ignore[PY-004]
            steps_skipped += 1
        else:
            code = _run_step("Generate teacher responses", [
                py, str(scripts / "multi_teacher_generate.py"),
                "--manifest", manifest,
                "--prompts", prompts,
                "--out", responses,
                "--dispatch-mode", "teacher-fifo",
                "--fifo-size", "0",
            ], dry_run=dry)
            if code != 0:
                print("FATAL: Generation failed. Check teacher GGUF paths in")  # xray: ignore[PY-004]
                print(f"  {manifest}")  # xray: ignore[PY-004]
                print("  Tip: re-run with --skip-generate to bypass if data exists.")  # xray: ignore[PY-004]
                return 1
            steps_run += 1

    # ── Stage 2: Purify ──────────────────────────────────────────────────
    report_path = f"{purified_dir}/purification_report.json"
    if _file_nonempty(report_path):
        print(f"[SKIP] Purification: report already exists")  # xray: ignore[PY-004]
        steps_skipped += 1
    else:
        code = _run_step("Purify teacher outputs", [
            py, str(scripts / "purify_teacher_outputs.py"),
            "--input", responses,
            "--out-dir", purified_dir,
            "--resume",
        ], dry_run=dry)
        if code != 0:
            print("FATAL: Purification failed. Check that the input file is valid JSONL:")  # xray: ignore[PY-004]
            print(f"  {responses}")  # xray: ignore[PY-004]
            print("  Tip: run validate_datasets.py --sft-data <file> for diagnostics.")  # xray: ignore[PY-004]
            return 1
        steps_run += 1

    # ── Stage 3: Validate datasets ───────────────────────────────────────
    sft_data = f"{purified_dir}/consensus_sft.jsonl"
    dpo_data = f"{purified_dir}/conflict_dpo.jsonl"
    if _file_nonempty(sft_data):
        val_cmd = [py, str(scripts / "validate_datasets.py"),
                   "--sft-data", sft_data, "--no-ds-info"]
        if _file_nonempty(dpo_data):
            val_cmd.extend(["--dpo-data", dpo_data])
        _run_step("Validate datasets", val_cmd, dry_run=dry)
        steps_run += 1

    # ── Stage 4: Config gen ──────────────────────────────────────────────
    if _file_nonempty(sft_cfg) and _file_nonempty(merge_cfg):
        print(f"[SKIP] Config gen: configs already exist")  # xray: ignore[PY-004]
        steps_skipped += 1
    else:
        cfg_cmd = [
            py, str(scripts / "gen_distill_configs.py"),
            "--student", args.student_model,
            "--data-dir", purified_dir,
            "--tag", tag,
            "--auto-register",
        ]
        if args.cpu_safe:
            cfg_cmd.append("--cpu-safe")
        code = _run_step("Generate training configs", cfg_cmd, dry_run=dry)
        if code != 0:
            print("FATAL: Config generation failed. Verify that purified data exists in:")  # xray: ignore[PY-004]
            print(f"  {purified_dir}/")  # xray: ignore[PY-004]
            print("  Tip: ensure consensus_sft.jsonl is present and non-empty.")  # xray: ignore[PY-004]
            return 1
        steps_run += 1

    # ── Stage 5: Train ───────────────────────────────────────────────────
    if args.use_forge:
        # Forge Matrix mode
        if args.skip_train:
            print("[SKIP] Training (forge): --skip-train")  # xray: ignore[PY-004]
            steps_skipped += 1
        else:
            forge_cmd = [
                py, str(scripts / "run_student_forge.py"),
                "--matrix", forge_matrix,
                "--tag", tag,
            ]
            code = _run_step("Student Forge (parallel training)", forge_cmd, dry_run=dry)
            if code != 0:
                print("WARNING: Forge had failures (check forge_results.jsonl)")  # xray: ignore[PY-004]
            steps_run += 1
    else:
        # Sequential mode
        if args.skip_train:
            print("[SKIP] Training: --skip-train")  # xray: ignore[PY-004]
            steps_skipped += 1
        else:
            # SFT
            if _file_nonempty(sft_cfg):
                code = _run_step("SFT Training", [
                    py, "-m", "llamafactory.cli", "train", sft_cfg,
                ], dry_run=dry)
                if code != 0:
                    print(f"FATAL: SFT training failed. Check config: {sft_cfg}")  # xray: ignore[PY-004]
                    print("  Common causes: OOM (reduce per_device_train_batch_size),")  # xray: ignore[PY-004]
                    print("  missing model (check model_name_or_path), or dataset not registered.")  # xray: ignore[PY-004]
                    print("  Tip: re-run — SFT auto-resumes from latest checkpoint.")  # xray: ignore[PY-004]
                    return 1
                steps_run += 1

            # DPO
            dpo_cfg = f"{config_dir}/{tag}_dpo.yaml"
            if not args.skip_dpo and _file_nonempty(dpo_cfg) and _file_nonempty(dpo_data):
                code = _run_step("DPO Training", [
                    py, "-m", "llamafactory.cli", "train", dpo_cfg,
                ], dry_run=dry)
                if code != 0:
                    print("WARNING: DPO training failed (continuing)")  # xray: ignore[PY-004]
                steps_run += 1

    # ── Stage 6: Merge ───────────────────────────────────────────────────
    if args.skip_merge:
        print("[SKIP] Merge: --skip-merge")  # xray: ignore[PY-004]
        steps_skipped += 1
    elif _file_nonempty(f"{merged_dir}/config.json"):
        print(f"[SKIP] Merge: {merged_dir}/config.json already exists")  # xray: ignore[PY-004]
        steps_skipped += 1
    elif _file_nonempty(merge_cfg):
        code = _run_step("Merge adapters", [
            py, "-m", "llamafactory.cli", "export", merge_cfg,
        ], dry_run=dry)
        steps_run += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")  # xray: ignore[PY-004]
    print(f"Pipeline complete: {steps_run} steps run, {steps_skipped} skipped")  # xray: ignore[PY-004]
    print(f"{'='*60}")  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
