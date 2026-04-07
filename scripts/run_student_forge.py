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

"""Student Forge — parallel multi-variant distillation with SPSC collection.

Reads a forge_matrix.yaml defining N student variants (model sizes, LoRA
ranks, learning rates).  Trains all variants in parallel, collecting
results via one SPSCRingBuffer per worker.  Prints a ranked leaderboard
on completion.

Architecture:
    variant_A thread → SPSC_0 ─┐
    variant_B thread → SPSC_1 ─┤→ Collector → leaderboard
    variant_C thread → SPSC_2 ─┤
    variant_D thread → SPSC_3 ─┘

Usage:
    python scripts/run_student_forge.py \\
        --matrix data/forge_matrix/zena007_matrix.yaml --tag zena007

    python scripts/run_student_forge.py \\
        --matrix data/forge_matrix/zena007_matrix.yaml --tag zena007 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml  # xray: ignore[SEC-015]

# ── SPSC ring buffer import with fallback ────────────────────────────────


def _detect_gpu_count() -> int:
    """Auto-detect available CUDA GPUs. Returns 1 (CPU fallback) if none found."""
    try:
        import torch
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except ImportError:  # xray: ignore[QUAL-002]
        pass
    return 1

try:
    _zcl = os.path.join(os.path.dirname(__file__), "..", "..", "..", "zen_core_libs")
    if os.path.isdir(_zcl) and _zcl not in sys.path:
        sys.path.insert(0, _zcl)
    from zen_core_libs.common.system import SPSCRingBuffer  # xray: ignore[LLM-004]
except ImportError:  # xray: ignore[QUAL-002]
    import queue as _queue

    class SPSCRingBuffer:  # type: ignore[no-redef]
        """Fallback queue-based shim matching SPSCRingBuffer API."""

        def __init__(self, capacity: int = 4):
            self._q: _queue.Queue = _queue.Queue(maxsize=capacity)

        def put(self, item) -> bool:
            try:
                self._q.put_nowait(item)
                return True
            except _queue.Full:
                return False

        def get(self):
            try:
                return self._q.get_nowait()
            except _queue.Empty:
                return None


# ── Helpers ──────────────────────────────────────────────────────────────

def _read_final_loss(log_path: Path) -> float | None:
    """Extract the last training loss from a trainer_log.jsonl file."""
    last_loss = None
    if not log_path.exists():
        return None
    for line in log_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if "loss" in entry:
                last_loss = float(entry["loss"])
        except (json.JSONDecodeError, ValueError, KeyError):  # xray: ignore[QUAL-002]
            pass
    return last_loss


def _stable_probe_split(
    sft_path: Path, train_path: Path, probe_path: Path, fraction: float = 0.10
) -> tuple[int, int]:
    """Split consensus_sft.jsonl into train + probe sets using deterministic hash.

    Returns (train_count, probe_count).
    """
    lines = [l for l in sft_path.read_text(encoding="utf-8").splitlines(keepends=True) if l.strip()]
    n_probes = max(1, int(len(lines) * fraction))

    # Sort indices by SHA-256 hash of content for deterministic, reproducible split
    ranked = sorted(range(len(lines)), key=lambda i: hashlib.sha256(lines[i].encode()).hexdigest())
    probe_indices = set(ranked[:n_probes])

    train_lines = [l for i, l in enumerate(lines) if i not in probe_indices]
    probe_lines = [lines[i] for i in sorted(probe_indices)]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("".join(train_lines), encoding="utf-8")
    probe_path.write_text("".join(probe_lines), encoding="utf-8")

    return len(train_lines), len(probe_lines)


def _generate_sft_yaml(
    variant_id: str,
    variant: dict,
    tag: str,
    sft_dataset_name: str,
    template: str,
    cpu_safe: bool,
    out_dir: Path,
    early_stop_patience: int = 0,
) -> Path:
    """Generate a per-variant SFT training YAML config."""
    cfg = {
        "model_name_or_path": variant["model"],
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": variant.get("lora_rank", 16),
        "lora_target": "all",
        "dataset_dir": "data",
        "dataset": sft_dataset_name,
        "template": template,
        "cutoff_len": 1024,
        "max_samples": 10000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": f"saves/{tag}/{variant_id}/lora/sft",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "save_only_model": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": variant.get("learning_rate", 2.0e-5),
        "num_train_epochs": variant.get("num_train_epochs", 2),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": not cpu_safe,
        "fp16": False,
    }
    if early_stop_patience > 0:
        cfg["eval_strategy"] = "steps"
        cfg["eval_steps"] = 50
        cfg["load_best_model_at_end"] = True
        cfg["metric_for_best_model"] = "eval_loss"
        cfg["greater_is_better"] = False
    path = out_dir / f"{tag}_{variant_id}_sft.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"### Auto-generated SFT config for {variant_id}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def _generate_dpo_yaml(
    variant_id: str,
    variant: dict,
    tag: str,
    dpo_dataset_name: str,
    template: str,
    cpu_safe: bool,
    out_dir: Path,
) -> Path:
    """Generate a per-variant DPO training YAML config."""
    cfg = {
        "model_name_or_path": variant["model"],
        "adapter_name_or_path": f"saves/{tag}/{variant_id}/lora/sft",
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": variant.get("lora_rank", 16),
        "lora_target": "all",
        "pref_beta": 0.1,
        "pref_loss": "sigmoid",
        "dataset_dir": "data",
        "dataset": dpo_dataset_name,
        "template": template,
        "cutoff_len": 1024,
        "max_samples": 10000,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": f"saves/{tag}/{variant_id}/lora/dpo",
        "logging_steps": 5,
        "save_steps": 100,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "save_only_model": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": max(variant.get("learning_rate", 2.0e-5) / 2.0, 1.0e-6),
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.0,
        "bf16": not cpu_safe,
        "fp16": False,
    }
    path = out_dir / f"{tag}_{variant_id}_dpo.yaml"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"### Auto-generated DPO config for {variant_id}\n\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def _register_dataset(name: str, file_name: str, is_dpo: bool = False) -> None:
    """Register a dataset in data/dataset_info.json if not already present."""
    ds_path = Path("data/dataset_info.json")
    try:  # xray: ignore[PY-005]
            obj = json.loads(ds_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
            obj = {}
    if name in obj:
        return
    if is_dpo:
        obj[name] = {
            "file_name": file_name,
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        }
    else:
        obj[name] = {
            "file_name": file_name,
            "columns": {"prompt": "instruction", "response": "output"},  # xray: ignore[PY-004]
        }
    ds_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  Registered dataset: {name}")  # xray: ignore[PY-004]


# ── Worker thread ────────────────────────────────────────────────────────


# ── Auto-Heal: state persistence, resume, heartbeat, LLM diagnosis ──────

_HEARTBEAT_INTERVAL = 30  # seconds


class ForgeState:
    """Persistent forge state — survives crashes, power pulls, OOM kills.

    Written atomically (write-tmp + rename) after each variant completes.
    On restart, completed variants are skipped automatically.
    """

    def __init__(self, tag: str, saves_dir: Path | None = None) -> None:
        self.tag = tag
        self._dir = (saves_dir or Path("saves")) / tag
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "forge_state.json"
        self._heartbeat_path = self._dir / "forge_heartbeat"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return self._blank()
        return self._blank()

    def _blank(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": {},
            "failed": {},
            "status": "running",
        }

    def _save(self) -> None:
        """Atomic write: tmp file + rename."""
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self._path)

    def record_complete(self, variant_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._data["completed"][variant_id] = {
                "sft_final_loss": result.get("sft_final_loss"),
                "dpo_final_loss": result.get("dpo_final_loss"),
                "elapsed_sec": result.get("elapsed_sec"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._data.pop("failed", None)  # clear stale failures
            if variant_id in self._data.get("failed", {}):
                del self._data["failed"][variant_id]
            self._save()

    def record_failure(self, variant_id: str, error: str, diagnosis: str = "") -> None:
        with self._lock:
            if "failed" not in self._data:
                self._data["failed"] = {}
            self._data["failed"][variant_id] = {
                "error": error[:2000],
                "diagnosis": diagnosis[:2000],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save()

    def record_finished(self) -> None:
        with self._lock:
            self._data["status"] = "finished"
            self._data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save()

    def is_completed(self, variant_id: str) -> bool:
        return variant_id in self._data.get("completed", {})

    def completed_ids(self) -> set[str]:
        return set(self._data.get("completed", {}).keys())

    def completed_results(self) -> list[dict[str, Any]]:
        """Reconstruct result dicts for previously-completed variants."""
        results = []
        for vid, info in self._data.get("completed", {}).items():  # xray: ignore[QUAL-005]
            results.append({
                "variant_id": vid,
                "ok": True,
                "sft_final_loss": info.get("sft_final_loss"),
                "dpo_final_loss": info.get("dpo_final_loss"),
                "elapsed_sec": info.get("elapsed_sec", 0),
                "error": None,
                "resumed": True,
            })
        return results

    def write_heartbeat(self) -> None:
        """Write current timestamp — external monitoring can detect staleness."""
        try:
            self._heartbeat_path.write_text(
                json.dumps({"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                encoding="utf-8",
            )
        except OSError:  # xray: ignore[QUAL-002]
            pass

    def last_heartbeat_age(self) -> float | None:
        """Seconds since last heartbeat, or None if no heartbeat file."""
        if not self._heartbeat_path.exists():
            return None
        try:
            data = json.loads(self._heartbeat_path.read_text(encoding="utf-8"))
            return time.time() - data["ts"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None


def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Find highest checkpoint-N directory for crash-resume."""
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(
        [d for d in out.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def _start_heartbeat(state: ForgeState) -> threading.Event:
    """Start a background daemon thread that writes heartbeat every N seconds."""
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            state.write_heartbeat()
            stop.wait(_HEARTBEAT_INTERVAL)

    th = threading.Thread(target=_loop, daemon=True, name="forge-heartbeat")
    th.start()
    return stop


def _diagnose_with_llm(variant_id: str, error: str, stderr: str) -> str:
    """Try to use an available LLM to diagnose the training failure.

    Attempts InProcessAdapter first (no HTTP needed), falls back to
    LocalLLMAdapter, and finally returns a static diagnosis if no LLM
    is available.
    """
    prompt = (
        f"A machine learning training job for variant '{variant_id}' failed.\n"
        f"Error: {error[:500]}\n"
        f"Last stderr:\n{stderr[-1000:]}\n\n"
        "In 2-3 sentences: What is the root cause? What should be changed to fix it?"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a concise ML ops debugger."},
        {"role": "user", "content": prompt},
    ]

    # Try InProcessAdapter (zero-HTTP, direct GGUF)
    try:
        from zen_core_libs.llm import InProcessAdapter  # xray: ignore[LLM-004]
        adapter = InProcessAdapter.get_default()
        if adapter is not None:
            resp = adapter.chat(messages=messages, max_tokens=200)  # xray: ignore[LLM-003]
            if resp and isinstance(resp, str):
                return resp
            if resp and hasattr(resp, "get"):
                return resp.get("content", str(resp))
    except Exception:  # xray: ignore[QUAL-002, QUAL-011]
        pass

    # Try LocalLLMAdapter (HTTP to running llama-server)
    try:
        from zen_core_libs.llm import LocalLLMAdapter  # xray: ignore[LLM-004]
        adapter = LocalLLMAdapter()
        if adapter.health_check():
            resp = adapter.chat(messages, max_tokens=200)  # xray: ignore[LLM-003]
            if resp:
                return str(resp)
    except Exception:  # xray: ignore[QUAL-002, QUAL-011]
        pass

    # Static fallback — pattern-match common errors
    err_lower = (error + stderr).lower()
    if "cuda out of memory" in err_lower or "oom" in err_lower:
        return "GPU OOM. Reduce batch_size, gradient_accumulation_steps, or try QLoRA with 4-bit quantization."
    if "no such file" in err_lower or "filenotfounderror" in err_lower:
        return "Missing file. Check dataset paths in config YAML. Verify file_name is relative to data/ directory."
    if "some keys are not used" in err_lower:
        return "Invalid config key. Remove unsupported keys from YAML (e.g. booster, bf16 on CPU)."
    if "killed" in err_lower or "signal 9" in err_lower:
        return "Process killed (likely OOM-killer or user). Check RAM/VRAM and reduce batch size."
    return "Unknown failure. Check stderr for details."


# ── Worker thread (with auto-heal) ──────────────────────────────────────

def _train_variant(
    variant_id: str,
    variant: dict,
    sft_yaml: Path,
    dpo_yaml: Path | None,
    py: str,
    ring: SPSCRingBuffer,
    sem: threading.Semaphore,
    state: ForgeState | None = None,
) -> None:
    """Worker: train one student variant (SFT, optionally DPO), deposit result.

    Auto-heal features:
    - Detects existing checkpoints and resumes training (no data loss)
    - On failure, runs LLM diagnosis and records to state file
    - Records completion atomically so restarts skip finished variants
    """
    sem.acquire()
    t0 = time.time()
    result: dict = {
        "variant_id": variant_id,
        "model": variant["model"],
        "lora_rank": variant.get("lora_rank", 16),
        "ok": False,
        "sft_final_loss": None,
        "dpo_final_loss": None,
        "elapsed_sec": 0.0,
        "error": None,
    }
    try:
        # Parse config to get output_dir
        sft_cfg_text = sft_yaml.read_text(encoding="utf-8")
        sft_output = yaml.safe_load(sft_cfg_text)
        if isinstance(sft_output, str):
            sft_output = yaml.safe_load(sft_cfg_text.split("\n\n", 1)[-1])
        sft_output_dir = sft_output["output_dir"]

        # ── Check if SFT already fully done (adapter exists) ────────────  # xray: ignore[PY-004]
        sft_adapter = Path(sft_output_dir) / "adapter_model.safetensors"
        if sft_adapter.exists():
            print(f"  [{variant_id}] SFT already complete — resuming from saved adapter")  # xray: ignore[PY-004]
            sft_log = Path(sft_output_dir) / "trainer_log.jsonl"
            result["sft_final_loss"] = _read_final_loss(sft_log)
            result["sft_adapter_path"] = sft_output_dir
        else:
            # ── Check for partial checkpoint → resume ───────────────────
            ckpt = _find_latest_checkpoint(sft_output_dir)  # xray: ignore[PY-004]
            cmd = [py, "-m", "llamafactory.cli", "train", str(sft_yaml)]
            if ckpt:
                print(f"  [{variant_id}] Resuming SFT from {Path(ckpt).name}")  # xray: ignore[PY-004]
                cmd.extend(["--resume_from_checkpoint", ckpt])  # xray: ignore[PY-004]
                # Don't overwrite when resuming
            else:
                print(f"  [{variant_id}] Starting SFT training...")  # xray: ignore[PY-004]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                err_msg = f"SFT failed (exit {proc.returncode}): {proc.stderr[-500:]}"
                result["error"] = err_msg
                # LLM diagnosis
                diagnosis = _diagnose_with_llm(variant_id, err_msg, proc.stderr)
                result["diagnosis"] = diagnosis  # xray: ignore[PY-004]
                if state:
                    state.record_failure(variant_id, err_msg, diagnosis)
                print(f"  [{variant_id}] SFT FAILED — Diagnosis: {diagnosis}")  # xray: ignore[PY-004]
                return

            sft_log = Path(sft_output_dir) / "trainer_log.jsonl"  # xray: ignore[PY-004]
            result["sft_final_loss"] = _read_final_loss(sft_log)
            result["sft_adapter_path"] = sft_output_dir
            print(f"  [{variant_id}] SFT done — loss={result['sft_final_loss']}")  # xray: ignore[PY-004]

        # ── DPO (optional) ──────────────────────────────────────────────
        if dpo_yaml and variant.get("run_dpo", False):
            dpo_cfg_text = dpo_yaml.read_text(encoding="utf-8")
            dpo_output = yaml.safe_load(dpo_cfg_text.split("\n\n", 1)[-1])
            dpo_output_dir = dpo_output["output_dir"]
  # xray: ignore-next[PY-004]
            dpo_adapter = Path(dpo_output_dir) / "adapter_model.safetensors"
            if dpo_adapter.exists():
                print(f"  [{variant_id}] DPO already complete — using saved adapter")  # xray: ignore[PY-004]
                dpo_log = Path(dpo_output_dir) / "trainer_log.jsonl"
                result["dpo_final_loss"] = _read_final_loss(dpo_log)
                result["dpo_adapter_path"] = dpo_output_dir
            else:
                ckpt = _find_latest_checkpoint(dpo_output_dir)  # xray: ignore[PY-004]
                cmd = [py, "-m", "llamafactory.cli", "train", str(dpo_yaml)]
                if ckpt:
                    print(f"  [{variant_id}] Resuming DPO from {Path(ckpt).name}")  # xray: ignore[PY-004]
                    cmd.extend(["--resume_from_checkpoint", ckpt])
                else:
                    print(f"  [{variant_id}] Starting DPO training...")  # xray: ignore[PY-004]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    err_msg = f"DPO failed (exit {proc.returncode}): {proc.stderr[-500:]}"
                    result["error"] = err_msg
                    result["ok"] = True  # SFT succeeded, DPO failed — partial success
                    diagnosis = _diagnose_with_llm(variant_id, err_msg, proc.stderr)
                    result["diagnosis"] = diagnosis  # xray: ignore[PY-004]
                    if state:
                        state.record_failure(variant_id, err_msg, diagnosis)
                    print(f"  [{variant_id}] DPO FAILED — Diagnosis: {diagnosis}")  # xray: ignore[PY-004]
                    return

                dpo_log = Path(dpo_output_dir) / "trainer_log.jsonl"  # xray: ignore[PY-004]
                result["dpo_final_loss"] = _read_final_loss(dpo_log)
                result["dpo_adapter_path"] = dpo_output_dir
                print(f"  [{variant_id}] DPO done — loss={result['dpo_final_loss']}")  # xray: ignore[PY-004]

        result["ok"] = True

    except Exception as exc:  # xray: ignore[QUAL-011]
        result["error"] = str(exc)
        if state:
            state.record_failure(variant_id, str(exc))
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 1)
        if result["ok"] and state:
            state.record_complete(variant_id, result)
        ring.put(result)
        sem.release()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Student Forge — parallel multi-variant distillation.")
    parser.add_argument("--matrix", required=True, help="Path to forge_matrix YAML.")
    parser.add_argument("--tag", default="", help="Run tag (overrides matrix file tag).")
    parser.add_argument("--py", default=".venv-py314/Scripts/python.exe", help="Python interpreter path.")
    parser.add_argument("--config-out-dir", default="examples/distillation/auto/forge", help="Dir for generated YAML configs.")
    parser.add_argument("--auto-parallel", action="store_true", help="Auto-detect GPU count and set max_parallel.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without training.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Enable LlamaFactory early stopping: halt after N eval steps with no loss improvement (0=disabled).")
    parser.add_argument("--retry-variant", type=str, default="",
                        help="Comma-separated variant IDs to force re-run (ignores completed state).")
    args = parser.parse_args()

    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    tag = args.tag or matrix.get("tag", "forge")
    template = matrix.get("template", "qwen")
    cpu_safe = matrix.get("cpu_safe", True)
    max_parallel = matrix.get("max_parallel", 4)
    if args.auto_parallel:  # xray: ignore[PY-004]
        gpu_count = _detect_gpu_count()
        max_parallel = gpu_count
        print(f"Auto-parallel: detected {gpu_count} GPU(s), setting max_parallel={max_parallel}")  # xray: ignore[PY-004]
    probe_fraction = matrix.get("eval_probe_split", 0.10)
    variants: dict = matrix.get("variants", {})

    sft_data_path = Path(matrix["sft_data"])
    dpo_data_path = Path(matrix.get("dpo_data", ""))  # xray: ignore[PY-004]
    config_out = Path(args.config_out_dir) / tag  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    print(f"=== Student Forge Matrix ===")  # xray: ignore[PY-004]
    print(f"Tag:           {tag}")  # xray: ignore[PY-004]
    print(f"Template:      {template}")  # xray: ignore[PY-004]
    print(f"CPU safe:      {cpu_safe}")  # xray: ignore[PY-004]
    print(f"Max parallel:  {max_parallel}")  # xray: ignore[PY-004]
    print(f"Probe split:   {probe_fraction:.0%}")  # xray: ignore[PY-004]
    print(f"SFT data:      {sft_data_path}")  # xray: ignore[PY-004]
    print(f"DPO data:      {dpo_data_path}")  # xray: ignore[PY-004]
    print(f"Variants:      {len(variants)}")  # xray: ignore[PY-004]
    print()  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    # ── Print variant table ──────────────────────────────────────────────
    print(f"{'Variant':<14} {'Model':<32} {'Rank':>4} {'LR':>10} {'Epochs':>6} {'DPO':>4}")  # xray: ignore[PY-004]
    print("-" * 80)  # xray: ignore[PY-004]
    for vid, v in variants.items():
        print(f"{vid:<14} {v['model']:<32} {v.get('lora_rank',16):>4} {v.get('learning_rate',2e-5):>10.1e} {v.get('num_train_epochs',2):>6} {'yes' if v.get('run_dpo') else 'no':>4}")  # xray: ignore[PY-004, QUAL-013]
    print()  # xray: ignore[PY-004]

    if args.dry_run:
        print("[dry-run] Would train the above variants. Exiting.")  # xray: ignore[PY-004]
        return 0

    # ── Initialize forge state (auto-heal) ───────────────────────────────
    state = ForgeState(tag)  # xray: ignore[PY-004]
    previously_done = state.completed_ids()
    # Handle --retry-variant: force re-run of specific variants
    retry_set: set[str] = set()
    if args.retry_variant:
        retry_set = {v.strip() for v in args.retry_variant.split(",") if v.strip()}
        previously_done -= retry_set
        if retry_set:
            print(f"Retry override: {', '.join(sorted(retry_set))} will be re-trained")  # xray: ignore[PY-004]
    if previously_done:  # xray: ignore[PY-004]
        print(f"\nAuto-heal: found {len(previously_done)} completed variant(s) from previous run:")  # xray: ignore[PY-004]
        for vid in sorted(previously_done):
            print(f"  {vid} — skipping")  # xray: ignore[PY-004]
        print()  # xray: ignore[PY-004]

    # ── Start heartbeat ──────────────────────────────────────────────────
    heartbeat_stop = _start_heartbeat(state)  # xray: ignore[PY-004]
    stale_age = state.last_heartbeat_age()  # xray: ignore[PY-004]
    if stale_age is not None and stale_age > _HEARTBEAT_INTERVAL * 3:
        print(f"Warning: last heartbeat was {stale_age:.0f}s ago — previous run likely crashed")  # xray: ignore[PY-004]
        print(f"Resuming from saved state...\n")  # xray: ignore[PY-004]

    # ── Split probe set ──────────────────────────────────────────────────
    train_sft_path = sft_data_path.parent / "train_sft.jsonl"
    probe_path = sft_data_path.parent / "eval_probes.jsonl"  # xray: ignore[PY-004]

    if not sft_data_path.exists():
        print(f"ERROR: SFT data not found: {sft_data_path}")  # xray: ignore[PY-004]
        return 1  # xray: ignore[PY-004]

    n_train, n_probe = _stable_probe_split(sft_data_path, train_sft_path, probe_path, probe_fraction)
    print(f"Probe split: {n_train} train + {n_probe} probes (from {n_train + n_probe} total)")  # xray: ignore[PY-004]

    # ── Register datasets ────────────────────────────────────────────────
    sft_ds_name = f"{tag}_forge_train_sft"
    dpo_ds_name = f"{tag}_conflict_dpo"

    # file_name in dataset_info.json must be relative to dataset_dir ("data/")
    def _rel_to_data(p: Path) -> str:
        try:
            return str(p.relative_to("data")).replace("\\", "/")
        except ValueError:  # xray: ignore[QUAL-002]
            return str(p).replace("\\", "/")

    _register_dataset(sft_ds_name, _rel_to_data(train_sft_path))
    if dpo_data_path.exists() and dpo_data_path.stat().st_size > 0:
        _register_dataset(dpo_ds_name, _rel_to_data(dpo_data_path), is_dpo=True)

    # ── Generate per-variant configs ─────────────────────────────────────
    variant_configs: dict[str, dict] = {}
    for vid, v in variants.items():
        sft_yaml = _generate_sft_yaml(vid, v, tag, sft_ds_name, template, cpu_safe, config_out,
                                       early_stop_patience=args.early_stop_patience)
        dpo_yaml = None
        if v.get("run_dpo") and dpo_data_path.exists() and dpo_data_path.stat().st_size > 0:  # xray: ignore[PY-004]
            dpo_yaml = _generate_dpo_yaml(vid, v, tag, dpo_ds_name, template, cpu_safe, config_out)
        variant_configs[vid] = {"sft_yaml": sft_yaml, "dpo_yaml": dpo_yaml, "variant": v}
        print(f"  Generated configs for {vid}")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    # ── Launch parallel training (skip already-completed) ──────────────
    remaining = {vid: cfg for vid, cfg in variant_configs.items() if vid not in previously_done}
    print(f"\nLaunching {len(remaining)} training workers ({len(previously_done)} skipped, max_parallel={max_parallel})...\n")  # xray: ignore[PY-004]

    sem = threading.Semaphore(max_parallel)
    buffers: dict[str, SPSCRingBuffer] = {}
    threads: list[threading.Thread] = []

    for vid, cfg in remaining.items():
        buf = SPSCRingBuffer(capacity=4)
        buffers[vid] = buf
        th = threading.Thread(
            target=_train_variant,
            args=(vid, cfg["variant"], cfg["sft_yaml"], cfg["dpo_yaml"], args.py, buf, sem, state),
            daemon=True,
            name=f"forge-{vid}",
        )
        th.start()
        threads.append(th)

    # ── Collect results ──────────────────────────────────────────────────
    results: list[dict] = state.completed_results()  # start with previously completed
    collected = 0
    total = len(remaining)
    forge_start = time.time()

    while collected < total:
        for vid, buf in buffers.items():
            item = buf.get()
            if item is not None:
                collected += 1
                status = "OK" if item["ok"] else "FAIL"
                elapsed_total = time.time() - forge_start
                avg_per_variant = elapsed_total / collected
                par = max(max_parallel, 1)
                remaining_count = total - collected
                remaining_time = remaining_count * avg_per_variant / par
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                diag = ""  # xray: ignore[PY-004, QUAL-013]
                if item.get("diagnosis") and not item["ok"]:
                    diag = f"\n          Diagnosis: {item['diagnosis']}"
                print(f"  [{collected + len(previously_done)}/{len(variants)}] {item['variant_id']}: {status} — SFT loss={item['sft_final_loss']} DPO loss={item['dpo_final_loss']} ({item['elapsed_sec']}s) | ETA {eta_str}{diag}")  # xray: ignore[PY-004, QUAL-013]
                results.append(item)
        if collected < total:
            time.sleep(1.0)

    for th in threads:
        th.join(timeout=5)

    # ── Stop heartbeat, mark state finished ──────────────────────────────
    heartbeat_stop.set()
    state.record_finished()

    # ── Leaderboard ──────────────────────────────────────────────────────
    ok_results = [r for r in results if r["ok"]]  # xray: ignore[PY-004]
    ok_results.sort(key=lambda r: r["sft_final_loss"] or float("inf"))  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    print(f"\n{'='*80}")  # xray: ignore[PY-004]
    print(f"  FORGE LEADERBOARD — {tag}")  # xray: ignore[PY-004]
    print(f"{'='*80}")  # xray: ignore[PY-004]
    print(f"{'Rank':>4} {'Variant':<14} {'Model':<32} {'SFT Loss':>10} {'DPO Loss':>10} {'Time':>8}")  # xray: ignore[PY-004]
    print(f"{'-'*80}")  # xray: ignore[PY-004]
    for i, r in enumerate(ok_results, 1):  # xray: ignore[PY-004]
        sft_l = f"{r['sft_final_loss']:.4f}" if r["sft_final_loss"] is not None else "n/a"
        dpo_l = f"{r['dpo_final_loss']:.4f}" if r["dpo_final_loss"] is not None else "n/a"
        print(f"{i:>4} {r['variant_id']:<14} {r['model']:<32} {sft_l:>10} {dpo_l:>10} {r['elapsed_sec']:>7.0f}s")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    failed = [r for r in results if not r["ok"]]
    if failed:  # xray: ignore[PY-004]
        print(f"\nFailed variants:")  # xray: ignore[PY-004]
        for r in failed:
            print(f"  {r['variant_id']}: {r['error']}")  # xray: ignore[PY-004]

    # ── Throughput summary ───────────────────────────────────────────────
    forge_elapsed = time.time() - forge_start  # xray: ignore[PY-004]
    forge_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(forge_elapsed))
    throughput = len(results) / (forge_elapsed / 3600) if forge_elapsed > 0 else 0
    print(f"\nForge throughput: {len(results)} variants in {forge_elapsed_str} ({throughput:.1f} variants/hour, max_parallel={max_parallel})")  # xray: ignore[PY-004]
    if throughput > 0:
        nightly_capacity = throughput * 10  # assume 10h overnight window
        print(f"Projected nightly capacity (10h window): {nightly_capacity:.0f} variants")  # xray: ignore[PY-004]

    # ── Save results ─────────────────────────────────────────────────────
    results_path = Path(f"saves/{tag}/forge_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:  # xray: ignore[PY-004]
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {results_path}")  # xray: ignore[PY-004]

    return 0 if ok_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
