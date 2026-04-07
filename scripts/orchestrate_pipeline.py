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
  2. Purify into consensus (SFT) / conflict (DPO) splits (+synthetic DPO)
  3. Validate datasets
  4. Auto-generate training configs
  5. Train SFT → [DPO if samples] (auto-resume incomplete training)
  6. Merge adapters
  7. Recover + synthesize forge results (bridge sequential → eval)
  8. Evaluate student (two-pass: quick quiz → deep exam) + GGUF teacher eval
  9. Graduation dashboard
  10. GGUF export + speed benchmark
  11. Qualitative eval (sample generation comparison)

  OR: Student Forge Matrix (parallel training) for stages 5-6.

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


def _find_latest_checkpoint(adapter_dir: str | Path) -> Path | None:
    """Find the highest checkpoint-N directory that contains adapter_model.safetensors."""
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        return None
    # First check if adapter exists in the root dir
    if (adapter_path / "adapter_model.safetensors").exists():
        return adapter_path
    # Scan for checkpoint-N/ subdirs
    checkpoints = sorted(
        (d for d in adapter_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-", 1)[1]) if d.name.split("-", 1)[1].isdigit() else 0,
    )
    for ckpt in reversed(checkpoints):
        if (ckpt / "adapter_model.safetensors").exists():
            return ckpt
    return None


def _get_final_loss(trainer_log: str | Path) -> float | None:
    """Read the last loss value from trainer_log.jsonl."""
    log_path = Path(trainer_log)
    if not log_path.exists():
        return None
    last_line = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            last_line = stripped
    if not last_line:
        return None
    try:
        entry = json.loads(last_line)
        return float(entry.get("loss", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _is_training_complete(adapter_dir: str | Path) -> bool:
    """Check if training ran to completion by reading trainer_log.jsonl."""
    log_path = Path(adapter_dir) / "trainer_log.jsonl"
    if not log_path.exists():
        return False
    last_line = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and stripped.startswith("{"):
            last_line = stripped
    if not last_line:
        return False
    try:
        entry = json.loads(last_line)
        current = entry.get("current_steps", 0)
        total = entry.get("total_steps", 0)
        return current >= total and total > 0
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def _synthesize_forge_results(
    tag: str,
    student_model: str,
    adapter_dir: str,
    forge_path: Path,
) -> bool:
    """Create a forge_results.jsonl from sequential training artifacts.

    Bridges the gap between sequential training and eval_student_panel.py
    which expects forge results.
    """
    adapter = _find_latest_checkpoint(adapter_dir)
    if adapter is None:
        print(f"WARNING: No adapter found in {adapter_dir}")
        return False

    trainer_log = Path(adapter_dir) / "trainer_log.jsonl"
    final_loss = _get_final_loss(trainer_log)

    result = {
        "variant_id": "B",
        "model": student_model,
        "sft_adapter_path": str(adapter),
        "sft_final_loss": final_loss,
        "ok": True,
    }

    forge_path.parent.mkdir(parents=True, exist_ok=True)
    forge_path.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Synthesized forge results → {forge_path}")
    print(f"    adapter: {adapter}")
    print(f"    final_loss: {final_loss}")
    return True


# ── Qualitative eval inline script ──────────────────────────────────────
_QUALITATIVE_EVAL_SCRIPT = r'''
import json, sys, collections
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

merged_dir = sys.argv[1]
probe_path = sys.argv[2]
out_path = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    merged_dir, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
)
model.eval()

# Group probes by category, pick up to 3 per category (max 12 total)
probes = []
for line in Path(probe_path).read_text(encoding="utf-8").splitlines():
    if line.strip():
        try:
            probes.append(json.loads(line))
        except json.JSONDecodeError:
            pass

by_cat = collections.defaultdict(list)
for p in probes:
    cat = p.get("category", "other")
    by_cat[cat].append(p)

selected = []
for cat, samples in sorted(by_cat.items()):
    selected.extend(samples[:3])
if len(selected) > 12:
    selected = selected[:12]

results = []
for sample in selected:
    instruction = sample.get("instruction", sample.get("prompt", ""))
    reference = sample.get("output", sample.get("response", ""))
    category = sample.get("category", "other")

    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    results.append({
        "id": sample.get("id", ""),
        "category": category,
        "prompt": instruction[:200],
        "reference": reference[:300],
        "generated": generated[:300],
    })

Path(out_path).write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
    encoding="utf-8",
)
print(f"Qualitative eval: {len(results)} samples written to {out_path}")
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform pipeline orchestrator for multi-teacher distillation.",
        epilog="""\
examples:
  %(prog)s --tag zena007
  %(prog)s --tag zena007 --skip-train --skip-dpo
  %(prog)s --tag zena007 --skip-train --skip-merge  (eval only)
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
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step.")
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip dashboard step.")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF export step.")
    parser.add_argument("--skip-qualitative", action="store_true", help="Skip qualitative eval step.")
    parser.add_argument("--gguf-quants", nargs="*", default=["Q4_K_M"], help="GGUF quant levels (default: Q4_K_M).")
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
        purify_cmd = [
            py, str(scripts / "purify_teacher_outputs.py"),
            "--input", responses,
            "--out-dir", purified_dir,
            "--resume",
            "--synthetic-dpo",
            "--synthetic-dpo-max", "200",
        ]
        code = _run_step("Purify teacher outputs", purify_cmd, dry_run=dry)
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
            # SFT — detect incomplete training and resume
            sft_adapter_dir_check = f"saves/{tag}/lora/sft"
            sft_done = (
                _find_latest_checkpoint(sft_adapter_dir_check) is not None
                and _is_training_complete(sft_adapter_dir_check)
            )
            if sft_done:
                print(f"[SKIP] SFT: training complete (all steps finished)")  # xray: ignore[PY-004]
                steps_skipped += 1
            elif _file_nonempty(sft_cfg):
                if _find_latest_checkpoint(sft_adapter_dir_check) is not None:
                    print(f"[RESUME] SFT: incomplete training detected — resuming from checkpoint")  # xray: ignore[PY-004]
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
    # Recovery: if adapter root dir lacks adapter_model.safetensors,
    # update the merge config to point to the latest checkpoint.
    sft_adapter_dir = f"saves/{tag}/lora/sft"
    effective_adapter = _find_latest_checkpoint(sft_adapter_dir)
    if effective_adapter and str(effective_adapter) != sft_adapter_dir:
        print(f"[RECOVER] Training adapter not in root dir; using {effective_adapter}")

    if args.skip_merge:
        print("[SKIP] Merge: --skip-merge")  # xray: ignore[PY-004]
        steps_skipped += 1
    elif _file_nonempty(f"{merged_dir}/config.json"):
        print(f"[SKIP] Merge: {merged_dir}/config.json already exists")  # xray: ignore[PY-004]
        steps_skipped += 1
    elif _file_nonempty(merge_cfg):
        # If adapter is in a checkpoint subdir, update merge config
        if effective_adapter and str(effective_adapter) != sft_adapter_dir and not dry:
            merge_conf = _read_yaml(merge_cfg)
            if merge_conf and _HAS_YAML:
                merge_conf["adapter_name_or_path"] = str(effective_adapter)
                Path(merge_cfg).write_text(
                    yaml.dump(merge_conf, default_flow_style=False, allow_unicode=True),
                    encoding="utf-8",
                )
                print(f"  Updated merge config adapter path → {effective_adapter}")
        code = _run_step("Merge adapters", [
            py, "-m", "llamafactory.cli", "export", merge_cfg,
        ], dry_run=dry)
        steps_run += 1

    # ── Stage 7: Synthesize forge results (sequential mode) ──────────────
    forge_results_path = Path(f"saves/{tag}/forge_results.jsonl")
    if not args.use_forge and not _file_nonempty(forge_results_path):
        print(f"\n[STAGE 7] Synthesize forge results for evaluation")
        if dry:
            print("  (dry-run — skipped)")
        elif effective_adapter:
            _synthesize_forge_results(tag, args.student_model, sft_adapter_dir, forge_results_path)
            steps_run += 1
        else:
            print(f"  WARNING: No adapter found — cannot synthesize forge results")
    elif _file_nonempty(forge_results_path):
        print(f"[SKIP] Forge results: {forge_results_path} already exists")
        steps_skipped += 1

    # ── Stage 8: Evaluation ──────────────────────────────────────────────
    eval_probes = f"{purified_dir}/eval_probes.jsonl"
    eval_scorecards = f"saves/{tag}/eval_scorecards.jsonl"

    if args.skip_eval:
        print("[SKIP] Evaluation: --skip-eval")
        steps_skipped += 1
    elif _file_nonempty(eval_scorecards):
        print(f"[SKIP] Evaluation: {eval_scorecards} already exists")
        steps_skipped += 1
    elif not _file_nonempty(eval_probes):
        print(f"[SKIP] Evaluation: no eval probes at {eval_probes}")
        steps_skipped += 1
    elif not _file_nonempty(forge_results_path):
        print(f"[SKIP] Evaluation: no forge_results.jsonl")
        steps_skipped += 1
    else:
        eval_cmd = [
            py, str(scripts / "eval_student_panel.py"),
            "--saves-tag", tag,
            "--probes", eval_probes,
        ]
        # Add teacher manifest for graduation exam if available
        manifest = f"data/{tag}/teacher_manifest.json"
        if Path(manifest).exists():
            eval_cmd.extend(["--teacher-manifest", manifest])
        code = _run_step("Student evaluation (two-pass)", eval_cmd, dry_run=dry)
        if code != 0:
            print("WARNING: Evaluation had failures (non-fatal — continuing)")
        steps_run += 1

    # ── Stage 9: Graduation Dashboard ────────────────────────────────────
    grad_report_path = f"saves/{tag}/graduation_report.json"

    if args.skip_dashboard:
        print("[SKIP] Dashboard: --skip-dashboard")
        steps_skipped += 1
    elif _file_nonempty(grad_report_path):
        dash_cmd = [
            py, str(scripts / "graduation_dashboard.py"),
            "--report", grad_report_path,
            "--saves-tag", tag,
            "--export-markdown",
        ]
        code = _run_step("Graduation dashboard", dash_cmd, dry_run=dry)
        steps_run += 1
    elif _file_nonempty(eval_scorecards):
        # No graduation report but scorecards exist — still show dashboard
        print(f"[INFO] No graduation report (no teacher manifest). Scorecards at {eval_scorecards}")
        steps_run += 1
    else:
        print("[SKIP] Dashboard: no evaluation results")
        steps_skipped += 1

    # ── Stage 10: GGUF Export + Speed Benchmark ──────────────────────────
    gguf_results = f"saves/{tag}/gguf/slim_down_results.jsonl"
    champion_path = f"saves/{tag}/champion.txt"

    if args.skip_gguf:
        print("[SKIP] GGUF export: --skip-gguf")
        steps_skipped += 1
    elif _file_nonempty(gguf_results):
        print(f"[SKIP] GGUF export: {gguf_results} already exists")
        steps_skipped += 1
    elif not _file_nonempty(champion_path):
        print(f"[SKIP] GGUF export: no champion.txt (run eval first)")
        steps_skipped += 1
    else:
        gguf_cmd = [
            py, str(scripts / "slim_down.py"),
            "--saves-tag", tag,
            "--quants", *args.gguf_quants,
        ]
        # Add speed benchmark probes if available
        if _file_nonempty(eval_probes):
            gguf_cmd.extend(["--probes", eval_probes, "--bench-count", "5"])
        code = _run_step("GGUF export + speed benchmark", gguf_cmd, dry_run=dry)
        if code != 0:
            print("WARNING: GGUF export had failures (non-fatal — continuing)")
        steps_run += 1

    # ── Stage 11: Qualitative Eval ───────────────────────────────────────
    qualitative_path = f"saves/{tag}/qualitative_eval.jsonl"

    if args.skip_qualitative:
        print("[SKIP] Qualitative eval: --skip-qualitative")
        steps_skipped += 1
    elif _file_nonempty(qualitative_path):
        print(f"[SKIP] Qualitative eval: {qualitative_path} already exists")
        steps_skipped += 1
    elif not _file_nonempty(f"{merged_dir}/config.json"):
        print(f"[SKIP] Qualitative eval: no merged model at {merged_dir}")
        steps_skipped += 1
    elif not _file_nonempty(eval_probes):
        print(f"[SKIP] Qualitative eval: no eval probes")
        steps_skipped += 1
    else:
        # Pick representative probes (up to 3 per category, max 12 total)
        qual_cmd = [
            py, "-c",
            _QUALITATIVE_EVAL_SCRIPT,
            merged_dir, eval_probes, qualitative_path,
        ]
        code = _run_step("Qualitative eval (sample generation)", qual_cmd, dry_run=dry)
        if code != 0:
            print("WARNING: Qualitative eval failed (non-fatal)")
        else:
            # Print a summary
            if _file_nonempty(qualitative_path) and not dry:
                try:
                    samples = [json.loads(l) for l in Path(qualitative_path).read_text(encoding="utf-8").splitlines() if l.strip()]
                    print(f"\n  Qualitative eval: {len(samples)} samples generated")
                    for s in samples[:3]:
                        prompt_preview = s.get("prompt", "")[:60]
                        gen_preview = s.get("generated", "")[:60]
                        print(f"    [{s.get('category', '?')}] {prompt_preview}...")
                        print(f"      → {gen_preview}...")
                except Exception:
                    pass
        steps_run += 1

    # ── Summary ──────────────────────────────────────────────────────────
    status = "FAILED" if failed else "complete"
    print(f"\n{'='*60}")  # xray: ignore[PY-004]
    print(f"Pipeline {status}: {steps_run} steps run, {steps_skipped} skipped")  # xray: ignore[PY-004]

    # Report final artifacts
    artifacts = [
        (f"saves/{tag}/forge_results.jsonl", "Forge results"),
        (f"saves/{tag}/eval_scorecards.jsonl", "Eval scorecards"),
        (f"saves/{tag}/champion.txt", "Champion"),
        (f"saves/{tag}/graduation_report.json", "Graduation report"),
        (f"{merged_dir}/config.json", "Merged model"),
        (f"saves/{tag}/gguf/slim_down_results.jsonl", "GGUF export"),
        (f"saves/{tag}/qualitative_eval.jsonl", "Qualitative eval"),
    ]
    found = [(label, path) for path, label in artifacts if _file_nonempty(path)]
    if found:
        print(f"\nArtifacts:")
        for label, path in found:
            print(f"  {label}: {path}")
    print(f"{'='*60}")  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
