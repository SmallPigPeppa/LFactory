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

"""Multi-teacher response generation via zen_core_libs llama.cpp adapters.

Each teacher answers the SAME prompts.  Output contains attributed responses
with answer + reasoning trace extraction for downstream purification.

Architecture (v4 -- SPSC ring-buffer FIFO, lock-free hot path):

  The key data structure is a per-teacher SPSC (Single Producer Single Consumer)
  ring buffer -- a true hardware-style FIFO:

      [Worker-Mistral] --produce--> RingBuffer[256] --consume--> |
      [Worker-Qwen]    --produce--> RingBuffer[256] --consume--> |- [Collector]
      [Worker-N]       --produce--> RingBuffer[256] --consume--> |    -> checkpoints

  Ring buffer = pre-allocated array + write_ptr (producer only) + read_ptr
  (consumer only).  Deposit = write slot + advance write_ptr.  Retrieval =
  read slot + advance read_ptr.  Both O(1).  In CPython the GIL guarantees
  that integer stores are atomic, so NO mutex / NO semaphore / NO condition
  variable is needed for SPSC.  Zero synchronization on the hot path.

  Buffer sized at 256 slots.  Inference takes seconds per item; checkpoint
  write takes milliseconds.  Consumer always drains faster than producers
  fill -- buffer effectively never full.

  Model pre-warming: all GGUF models loaded sequentially into InProcessAdapter
  cache BEFORE workers start.  Workers do pure inference from hot cache --
  llama-cpp releases the GIL during C++ matmuls -- true concurrency.

  RAMPressureThrottle: background thread with hysteresis thresholds gates
  workers via threading.Event when the system is under memory pressure.
  Safe for 7-8 concurrent GGUF-resident teachers on a 96 GB box.

Supports two backends (auto-detected):
  - InProcessAdapter  (direct GGUF loading -- no HTTP, lowest latency)
  - LlamaServerManager (HTTP to running llama-server -- good for large models)

Usage:
    python scripts/multi_teacher_generate.py \
        --manifest teacher_manifest.json \
        --prompts data/distill_prompts.jsonl \
        --out data/teacher_responses.jsonl

Manifest format (JSON):
    {
        "teachers": [
            {"name": "deepseek-r1", "gguf": "C:/AI/Models/deepseek-r1.gguf"},
            {"name": "qwen3-8b",    "gguf": "C:/AI/Models/qwen3-8b.gguf"}
        ]
    }
    OR with a running llama-server:
    {
        "backend": "server",
        "base_url": "http://localhost:8090/v1",
        "teachers": [
            {"name": "deepseek-r1", "model": "deepseek-r1"}
        ]
    }

Prompt JSONL rows must have:  {"id": "...", "prompt": "..."}
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# RAM-pressure throttle (zen_core_libs–inspired memory-budget pattern)
# ---------------------------------------------------------------------------

class RAMPressureThrottle:
    """Background monitor that pauses worker threads when system RAM is low.

    Uses a threading.Event as the gate — workers call ``wait_if_ok()`` before
    each inference call.  When available RAM drops below *pause_below_pct* the
    gate closes; it re-opens once RAM climbs above *resume_above_pct*.

    This 2-threshold (hysteresis) design prevents rapid pause/resume cycling
    when memory sits right at the boundary.

    Safe for 7-8 concurrent GGUF-resident teachers on a 96 GB box.
    """

    def __init__(
        self,
        pause_below_pct: float = 12.0,
        resume_above_pct: float = 22.0,
        poll_interval: float = 3.0,
    ) -> None:
        self._pause_pct = pause_below_pct
        self._resume_pct = resume_above_pct
        self._poll_s = poll_interval
        self._gate = threading.Event()
        self._gate.set()  # open = workers can proceed
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused_since: float | None = None
        self.pauses = 0
        self.total_paused_s = 0.0
        self._available_pct: float = 100.0

    # -- public API used by workers ---------------------------------------

    def wait_if_ok(self, timeout: float = 120.0) -> bool:
        """Block until RAM is OK (or timeout).  Returns True when cleared."""
        return self._gate.wait(timeout=timeout)

    @property
    def is_paused(self) -> bool:
        return not self._gate.is_set()

    @property
    def available_pct(self) -> float:
        return self._available_pct

    def stats(self) -> dict:
        return {
            "pauses": self.pauses,
            "total_paused_s": round(self.total_paused_s, 1),
            "available_pct": round(self._available_pct, 1),
            "is_paused": self.is_paused,
        }

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ram-throttle")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._gate.set()  # unblock any waiters
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # -- internal ---------------------------------------------------------

    def _loop(self) -> None:
        try:
            import psutil
        except ImportError:  # xray: ignore[QUAL-002]
            # Without psutil the throttle is a no-op (gate stays open).
            return

        while not self._stop.is_set():
            mem = psutil.virtual_memory()
            self._available_pct = mem.available / mem.total * 100.0

            if self._available_pct < self._pause_pct and self._gate.is_set():
                self._gate.clear()
                self._paused_since = time.monotonic()
                self.pauses += 1
                print(  # xray: ignore[PY-004]
                    f"\n  !! RAM throttle: {self._available_pct:.1f}% available "
                    f"(< {self._pause_pct}%) -- pausing workers",
                    flush=True,
                )

            elif self._available_pct >= self._resume_pct and not self._gate.is_set():
                if self._paused_since is not None:
                    self.total_paused_s += time.monotonic() - self._paused_since
                    self._paused_since = None
                self._gate.set()
                print(  # xray: ignore[PY-004]
                    f"\n  >> RAM recovered: {self._available_pct:.1f}% available "
                    f"(>= {self._resume_pct}%) -- resuming workers",
                    flush=True,
                )

            self._stop.wait(self._poll_s)

        # Final bookkeeping
        if self._paused_since is not None:
            self.total_paused_s += time.monotonic() - self._paused_since


# ---------------------------------------------------------------------------
# SPSC ring buffer — hardware-style FIFO, lock-free hot path
# ---------------------------------------------------------------------------

class SPSCRingBuffer:
    """Lock-free Single-Producer Single-Consumer ring buffer.

    Pre-allocated circular array with separate read/write indices.  CPython's
    GIL guarantees that integer stores are atomic, so one producer thread and
    one consumer thread can operate WITHOUT any mutex, semaphore, or condition
    variable.

    Deposit  = O(1): write slot, advance ``_write_idx``
    Retrieve = O(1): read slot,  advance ``_read_idx``
    """

    __slots__ = ("_buf", "_cap", "_write_idx", "_read_idx")

    def __init__(self, capacity: int = 256) -> None:
        self._cap = capacity
        self._buf: list = [None] * capacity
        self._write_idx = 0   # modified ONLY by producer
        self._read_idx = 0    # modified ONLY by consumer

    def put(self, item) -> bool:
        """Producer: deposit *item*.  Returns ``False`` if buffer is full."""
        nxt = (self._write_idx + 1) % self._cap
        if nxt == self._read_idx:
            return False
        self._buf[self._write_idx] = item
        self._write_idx = nxt          # store-release (GIL-atomic)
        return True

    def get(self):
        """Consumer: retrieve next item.  Returns ``None`` if empty."""
        if self._read_idx == self._write_idx:
            return None
        item = self._buf[self._read_idx]
        self._buf[self._read_idx] = None   # allow GC of payload
        self._read_idx = (self._read_idx + 1) % self._cap
        return item

    @property
    def is_empty(self) -> bool:
        return self._read_idx == self._write_idx

    @property
    def count(self) -> int:
        return (self._write_idx - self._read_idx) % self._cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: Path) -> list[dict]:
    """Load prompts from JSONL or JSON (alpaca) format.

    Supported formats:
      - JSONL with {"id": ..., "prompt": ...} rows
      - JSON array with {"instruction": ..., "input": ...} (alpaca format)
      - JSON array with {"prompt": ...} rows

    Returns list of {"id": str, "prompt": str} dicts.
    """
    raw_text = path.read_text(encoding="utf-8").strip()

    # Detect JSON array vs JSONL
    if raw_text.startswith("["):
        try:  # xray: ignore[PY-005]
                    rows = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
                    rows = {}
    else:
        rows = []
        for line in raw_text.splitlines():  # xray: ignore[PY-005]
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass  # skip malformed JSON line

    # Normalize each row to {id, prompt}
    normalized: list[dict] = []
    for i, row in enumerate(rows):
        if "prompt" in row:
            pid = row.get("id", f"p{i:04d}")
            prompt = row["prompt"]
        elif "instruction" in row:
            pid = row.get("id", f"p{i:04d}")
            inst = row["instruction"]
            inp = row.get("input", "")
            prompt = f"{inst}\n{inp}".strip() if inp else inst
        elif "messages" in row:
            pid = row.get("id", f"p{i:04d}")
            user_msgs = [m for m in row["messages"] if m.get("role") == "user"]
            if user_msgs:
                content = user_msgs[0].get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("value", c.get("text", ""))
                        for c in content if isinstance(c, dict)
                    )
                prompt = str(content)
            else:
                continue
        else:
            continue
        normalized.append({"id": str(pid), "prompt": prompt})
    return normalized


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))  # xray: ignore[PY-005]
    return rows


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _acquire_single_instance_lock(lock_path: Path) -> None:
    """Acquire a lock file to prevent duplicate generator runs.

    If a live PID is found in the lock file, abort with a clear message.
    Stale lock files are automatically replaced.
    """
    current_pid = os.getpid()

    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except Exception:  # xray: ignore[QUAL-011]
            existing_pid = 0

        if existing_pid and existing_pid != current_pid:
            try:
                os.kill(existing_pid, 0)
            except OSError:  # xray: ignore[QUAL-002]
                # Stale lock file from dead process
                pass
            else:
                raise RuntimeError(
                    f"Another multi_teacher_generate process is already running (pid={existing_pid}). "
                    f"If this is stale, delete lock file: {lock_path}"
                )

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(current_pid), encoding="utf-8")

    def _cleanup() -> None:
        try:
            if lock_path.exists():
                lock_pid = lock_path.read_text(encoding="utf-8").strip()
                if lock_pid == str(current_pid):
                    lock_path.unlink()
        except Exception:  # xray: ignore[QUAL-002, QUAL-011]
            pass

    atexit.register(_cleanup)


def _extract_thought_and_answer(text: str) -> tuple[str, str]:
    """Split a response into (thought, answer) if it contains think tags."""
    # Handle <think>...</think> or <|think|>...<|/think|>
    for open_tag, close_tag in [
        ("<think>", "</think>"),
        ("<|think|>", "<|/think|>"),
    ]:
        if close_tag in text:
            parts = text.split(close_tag, 1)
            thought = parts[0].replace(open_tag, "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
            return thought, answer

    # No think tags — entire text is the answer
    return "", text.strip()


def _infer_task_type(sample_id: str, prompt: str) -> str:
    """Infer a coarse task type for adaptive decoding budgets."""
    sid = (sample_id or "").lower()
    if sid.startswith("tr-"):
        return "translation"
    if sid.startswith("ocr-"):
        return "ocr"
    if sid.startswith("chat-"):
        return "chat"

    p = (prompt or "").lower()
    if "translate" in p or "translation" in p:
        return "translation"
    if "ocr" in p or "extract text" in p:
        return "ocr"
    return "general"


def _adaptive_budget(sample_id: str, prompt: str, base_max_tokens: int) -> int:
    """Compute per-sample max_tokens to reduce long-tail latency.

    The policy is intentionally conservative:
      - translation: short outputs, lower budget
      - OCR cleanup: medium outputs
      - chat: keep larger budget
    """
    t = _infer_task_type(sample_id, prompt)
    char_len = len(prompt or "")

    if t == "translation":
        # Most translation answers are short; keep bounded.
        cap = 192 if char_len < 500 else 256
    elif t == "ocr":
        cap = 256 if char_len < 800 else 320
    elif t == "chat":
        cap = 384 if char_len < 800 else 512
    else:
        cap = 320

    return max(64, min(base_max_tokens, cap))


def _adaptive_temperature(sample_id: str, prompt: str, base_temperature: float) -> float:
    """Use lower temperature for deterministic tasks, preserve chat creativity."""
    t = _infer_task_type(sample_id, prompt)
    if t == "translation":
        return min(base_temperature, 0.25)
    if t == "ocr":
        return min(base_temperature, 0.20)
    return base_temperature


def _resolve_decode_params(
    sample_id: str,
    prompt_text: str,
    base_max_tokens: int,
    base_temperature: float,
    adaptive_budgets: bool,
) -> tuple[int, float]:
    """Return effective decoding parameters for one sample."""
    if not adaptive_budgets:
        return base_max_tokens, base_temperature

    eff_max_tokens = _adaptive_budget(sample_id, prompt_text, base_max_tokens)
    eff_temperature = _adaptive_temperature(sample_id, prompt_text, base_temperature)
    return eff_max_tokens, eff_temperature


def _call_with_timeout(
    query_fn: Callable[[dict, str, int, float], str],
    teacher: dict,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> str:
    """Run query_fn in a daemon thread; raise TimeoutError if it doesn't finish."""
    result_box: list[str | BaseException] = []

    def _target() -> None:
        try:
            result_box.append(query_fn(teacher, prompt, max_tokens, temperature))
        except Exception as exc:  # xray: ignore[QUAL-011]
            result_box.append(exc)

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)

    if not result_box:
        raise TimeoutError(f"Teacher {teacher.get('name', '?')} timed out after {timeout_sec}s")
    if isinstance(result_box[0], BaseException):
        raise result_box[0]  # xray: ignore[QUAL-011]
    return str(result_box[0])


def _run_teacher_inference(
    query_fn: Callable[[dict, str, int, float], str],
    teacher: dict,
    sample_id: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: float = 0,
) -> tuple[dict, float]:  # xray: ignore[PY-004]
    """Run one teacher query and normalize to the response payload schema.

    If timeout_sec > 0, wraps the inference call in a thread with a deadline.
    A hung teacher will be abandoned after timeout_sec seconds.
    """
    t0 = time.time()
    try:
        if timeout_sec > 0:
            raw = _call_with_timeout(query_fn, teacher, prompt_text, max_tokens, temperature, timeout_sec)
        else:
            raw = query_fn(teacher, prompt_text, max_tokens, temperature)
    except TimeoutError:
        elapsed = time.time() - t0
        print(  # xray: ignore[PY-004]
            f"  TIMEOUT [{teacher['name']}] on '{sample_id}' after {elapsed:.0f}s",
            file=sys.stderr, flush=True,
        )
        raw = ""
    except Exception as exc:  # xray: ignore[QUAL-011]
        print(f"  ERROR [{teacher['name']}] on '{sample_id}': {exc}", file=sys.stderr, flush=True)  # xray: ignore[PY-004]
        raw = ""

    elapsed = time.time() - t0
    thought, answer = _extract_thought_and_answer(raw)
    return {
        "raw": raw,
        "thought": thought,
        "answer": answer,
        "elapsed_s": round(elapsed, 2),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, elapsed


def _auto_fifo_size_from_prompts(
    work_lists: dict[str, list[tuple[str, str]]],
    active_teachers: list[str],
    fixed_depth: int = 2048,
) -> tuple[int, float]:
    """Return fixed spare FIFO depth and prompt-byte telemetry.

    Auto mode intentionally provisions 2K slots per teacher for headroom.
    Average prompt bytes are still reported for observability.
    """
    total_bytes = 0
    total_prompts = 0
    for t_name in active_teachers:
        for _sample_id, prompt_text in work_lists[t_name]:
            total_bytes += len(prompt_text.encode("utf-8"))
            total_prompts += 1

    avg_prompt_bytes = (total_bytes / total_prompts) if total_prompts > 0 else 0.0
    return fixed_depth, avg_prompt_bytes


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

def _make_inprocess_backend(manifest: dict):
    """Return an InProcessAdapter-based query function."""
    # Add zen_core_libs to sys.path if needed
    import importlib
    try:
        llm_mod = importlib.import_module("zen_core_libs.llm")
    except ModuleNotFoundError:  # xray: ignore[QUAL-002]
        # Try adding the parent repo to path — walk up to find zen_core_libs
        import os
        zcl_path = os.environ.get("ZEN_CORE_LIBS", "")
        if not zcl_path:
            # Search upward from the script's location
            anchor = Path(__file__).resolve().parent
            for _ in range(8):
                anchor = anchor.parent
                candidate = anchor / "zen_core_libs"
                if (candidate / "zen_core_libs").is_dir() or (candidate / "__init__.py").is_file():
                    zcl_path = str(candidate)
                    break
        if zcl_path and zcl_path not in sys.path:
            sys.path.insert(0, zcl_path)
        llm_mod = importlib.import_module("zen_core_libs.llm")

    # Optional llama.cpp acceleration knobs from manifest and environment.
    # Manifest example:
    # {
    #   "llama_cpp": {
    #      "n_gpu_layers": -1,
    #      "main_gpu": 0,
    #      "n_threads": 24,
    #      "n_batch": 512,
    #      "n_ubatch": 512,
    #      "flash_attn": true,
    #      "offload_kqv": true
    #   }
    # }
    accel = dict(manifest.get("llama_cpp", {}) or {})

    # Env overrides for quick experiments without editing manifest.
    env_map = {
        "n_gpu_layers": os.environ.get("ZENA_N_GPU_LAYERS"),
        "main_gpu": os.environ.get("ZENA_MAIN_GPU"),
        "n_threads": os.environ.get("ZENA_N_THREADS"),
        "n_batch": os.environ.get("ZENA_N_BATCH"),
        "n_ubatch": os.environ.get("ZENA_N_UBATCH"),
    }
    for k, v in env_map.items():
        if v is not None and str(v).strip() != "":
            try:
                accel[k] = int(v)
            except ValueError:  # xray: ignore[QUAL-002]
                pass

    for flag_key, env_key in [
        ("flash_attn", "ZENA_FLASH_ATTN"),
        ("offload_kqv", "ZENA_OFFLOAD_KQV"),
    ]:
        ev = os.environ.get(env_key)
        if ev is not None and str(ev).strip() != "":
            accel[flag_key] = str(ev).strip().lower() in ("1", "true", "yes", "on")

    adapter = llm_mod.get_inprocess_adapter(
        max_models=manifest.get("max_models", 8),
        default_n_ctx=manifest.get("n_ctx", 4096),
        model_load_kwargs=accel,
    )
  # xray: ignore-next[LLM-003]
    def query(teacher: dict, prompt: str, max_tokens: int, temperature: float) -> str:
        gguf_path = teacher["gguf"]
        return adapter.chat(  # xray: ignore[LLM-003]
            gguf_path,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],  # xray: ignore[LLM-003]
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return query


def _make_server_backend(manifest: dict):
    """Return a requests-based query function for a running llama-server."""
    import requests

    base_url = manifest.get("base_url", "http://localhost:8090/v1")

    def query(teacher: dict, prompt: str, max_tokens: int, temperature: float) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": teacher.get("model", teacher["name"]),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    return query


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate responses from multiple teachers via zen_core_libs llama.cpp."
    )
    parser.add_argument("--manifest", required=True, help="Path to teacher_manifest.json.")
    parser.add_argument("--prompts", required=True, help="Path to prompts file (JSONL, JSON alpaca, or JSON messages).")
    parser.add_argument("--out", required=True, help="Output JSONL path for teacher_responses.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per teacher response.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--profile", default="", help="Path to teacher profile JSON (from teacher_profiler.py). Enables smart routing.")
    parser.add_argument("--route-threshold", type=float, default=0.3, help="Minimum capability score for a teacher to answer a prompt (default: 0.3).")
    parser.add_argument("--adaptive-budgets", action="store_true", help="Enable per-sample adaptive max_tokens/temperature for faster generation.")
    parser.add_argument(
        "--dispatch-mode",
        choices=["teacher-sequential", "teacher-fifo"],
        default="teacher-fifo",
        help=(
            "Dispatch strategy. 'teacher-fifo' pre-assigns work lists per teacher and "
            "runs concurrent workers (llama-cpp releases the GIL ⇒ true concurrency)."
        ),
    )
    parser.add_argument(
        "--fifo-size",
        type=int,
        default=256,
        help=(
            "Per-teacher SPSC ring buffer depth when using teacher-fifo. "
            "Use 0 for auto mode (fixed 2048 depth spare)."
        ),
    )  # xray: ignore[PY-004]
    parser.add_argument("--ram-pause-pct", type=float, default=12.0, help="Pause workers when available RAM drops below this %% (default 12).")
    parser.add_argument("--ram-resume-pct", type=float, default=22.0, help="Resume workers when available RAM rises above this %% (default 22).")
    parser.add_argument("--teacher-timeout", type=float, default=0, help="Per-prompt timeout in seconds per teacher (0 = no timeout). Kills hung inference.")
    args = parser.parse_args()

    if args.fifo_size < 0:
        print("ERROR: --fifo-size must be >= 0", file=sys.stderr)  # xray: ignore[PY-004]
        return 1

    manifest_path = Path(args.manifest)
    prompts_path = Path(args.prompts)
    out_path = Path(args.out)  # xray: ignore[PY-004]
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")

    try:
        _acquire_single_instance_lock(lock_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # xray: ignore[PY-004]
        return 2

    # Load manifest  # xray: ignore[PY-004]
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)  # xray: ignore[PY-005]

    teachers = manifest.get("teachers", [])
    if not teachers:  # xray: ignore[PY-004]
        print("ERROR: No teachers found in manifest.", file=sys.stderr)  # xray: ignore[PY-004]
        return 1

    teacher_count = len(teachers)
    if teacher_count % 2 == 0:
        print(f"WARNING: Even teacher count ({teacher_count}). Odd numbers recommended for majority rule.", file=sys.stderr)  # xray: ignore[PY-004]

    # Load capability profile if provided (enables smart routing)  # xray: ignore[PY-005]
    capability_matrix: dict[str, dict[str, float]] | None = None
    if args.profile:  # xray: ignore[PY-004]
        profile_path = Path(args.profile)
        if profile_path.is_file():  # xray: ignore[PY-004]
            with profile_path.open("r", encoding="utf-8") as f:
                profile_data = json.load(f)  # xray: ignore[PY-005]
            capability_matrix = profile_data.get("capability_matrix")
            print(f"Smart routing enabled -- {len(capability_matrix)} teachers profiled.", flush=True)  # xray: ignore[PY-004]
        else:
            print(f"WARNING: Profile file not found: {args.profile}. All teachers will answer all prompts.", file=sys.stderr)  # xray: ignore[PY-004]

    # Select backend
    backend_type = manifest.get("backend", "inprocess")  # xray: ignore[PY-004]
    if backend_type == "server":
        query_fn = _make_server_backend(manifest)
        print(f"Backend: llama-server at {manifest.get('base_url', 'http://localhost:8090/v1')}", flush=True)  # xray: ignore[PY-004]
    else:
        query_fn = _make_inprocess_backend(manifest)
        print("Backend: InProcessAdapter (direct GGUF, no HTTP)", flush=True)  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    if args.adaptive_budgets:
        print("Adaptive budgets: enabled (task-aware max_tokens + temperature)", flush=True)  # xray: ignore[PY-004]

    # Load prompts
    prompts = _load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts, {teacher_count} teachers.", flush=True)  # xray: ignore[PY-004]

    # Import routing if profile available
    route_fn = None
    classify_fn = None
    if capability_matrix:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from teacher_profiler import classify_prompt, route_prompt_to_teachers
        classify_fn = classify_prompt
        route_fn = lambda prompt_text: route_prompt_to_teachers(
            prompt_text, capability_matrix, threshold=args.route_threshold, min_teachers=1,
        )

    teacher_by_name = {t["name"]: t for t in teachers}

    # -------------------------------------------------------------------
    # Checkpoint / resume support
    # -------------------------------------------------------------------
    checkpoint_dir = out_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_EVERY = 10  # save every N prompts per teacher
    CHECKPOINT_BATCH_SIZE = 50  # buffer this many before flushing to disk

    def _ckpt_path(t_name: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", t_name)
        return checkpoint_dir / f"{safe}.jsonl"

    def _load_checkpoint(t_name: str) -> dict[str, dict]:
        """Load existing checkpoint for a teacher (sample_id -> response)."""
        p = _ckpt_path(t_name)
        if not p.is_file():  # xray: ignore[PY-005]
            return {}
        data: dict[str, dict] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                                    row = {}
                data[row["id"]] = row["response"]
        return data

    # Buffered checkpoint writes — reduce disk I/O from 1 write/prompt to 1 write/batch
    _ckpt_buffer: dict[str, list[str]] = {}  # teacher_name -> list of JSON lines

    def _append_checkpoint(t_name: str, sample_id: str, response: dict) -> None:
        """Buffer checkpoint writes and flush every CHECKPOINT_BATCH_SIZE items."""
        line = json.dumps({"id": sample_id, "response": response}, ensure_ascii=False) + "\n"
        buf = _ckpt_buffer.setdefault(t_name, [])
        buf.append(line)
        if len(buf) >= CHECKPOINT_BATCH_SIZE:
            _flush_checkpoint(t_name)

    def _flush_checkpoint(t_name: str) -> None:
        """Flush buffered checkpoint lines to disk."""
        buf = _ckpt_buffer.get(t_name, [])
        if not buf:
            return
        p = _ckpt_path(t_name)
        with p.open("a", encoding="utf-8") as f:
            f.writelines(buf)
        buf.clear()

    def _flush_all_checkpoints() -> None:
        """Flush all buffered checkpoints (call at end of generation)."""
        for t_name in list(_ckpt_buffer.keys()):
            _flush_checkpoint(t_name)

    # -------------------------------------------------------------------
    # Teacher-first generation: process ALL prompts per teacher to avoid
    # LRU cache thrashing (model reload on every prompt).  Only N model
    # loads needed (N = number of teachers) instead of N * num_prompts.
    # -------------------------------------------------------------------
    # Pre-classify prompts and build routing table
    prompt_routing: list[tuple[dict, str, list[str]]] = []   # (row, category, qualified_names)
    for row in prompts:
        prompt_text = row.get("prompt", "")
        if route_fn:
            qualified_names = route_fn(prompt_text)
            prompt_cap = classify_fn(prompt_text)
        else:
            qualified_names = [t["name"] for t in teachers]
            prompt_cap = "all"  # xray: ignore[PY-004]
        prompt_routing.append((row, prompt_cap, qualified_names))

    # Intermediate storage: teacher_name -> {sample_id -> response_dict}
    # Pre-load from checkpoints for resume
    all_teacher_responses: dict[str, dict[str, dict]] = {}
    for t in teachers:
        ckpt = _load_checkpoint(t["name"])
        if ckpt:
            print(f"  Resumed {len(ckpt)} cached responses for {t['name']}", flush=True)  # xray: ignore[PY-004]
        all_teacher_responses[t["name"]] = ckpt

    # Pre-compute assignment/resume/skip stats for deterministic accounting
    assigned_counts: dict[str, int] = {t["name"]: 0 for t in teachers}
    remaining_counts: dict[str, int] = {t["name"]: 0 for t in teachers}
    skipped_calls = 0

    for row, _prompt_cap, qualified_names in prompt_routing:
        skipped_calls += teacher_count - len(qualified_names)
        sample_id = row.get("id", "")
        qset = set(qualified_names)
        for t in teachers:
            t_name = t["name"]
            if t_name in qset:
                assigned_counts[t_name] += 1
                if sample_id not in all_teacher_responses[t_name]:
                    remaining_counts[t_name] += 1

    resumed_calls = sum(assigned_counts[t["name"]] - remaining_counts[t["name"]] for t in teachers)
    total_calls = 0

    use_parallel = backend_type == "inprocess" and args.dispatch_mode == "teacher-fifo"

    # Print per-teacher plan summary (sorted by model size for warm-up ordering)
    def _gguf_size_gb(teacher: dict) -> float:
        """Estimate model footprint for scheduling."""
        p = teacher.get("gguf", "")
        try:
            return os.path.getsize(p) / (1024 ** 3) if p and os.path.isfile(p) else 999.0
        except OSError:  # xray: ignore[QUAL-002]
            return 999.0
  # xray: ignore-next[PY-004]
    # Sort teachers: smallest model first → fastest warm-up, most concurrency sooner
    teachers_sorted = sorted(teachers, key=_gguf_size_gb)

    for t_idx, teacher in enumerate(teachers_sorted):
        t_name = teacher["name"]
        assigned = assigned_counts[t_name]  # xray: ignore[PY-004]
        remaining = remaining_counts[t_name]
        size_gb = _gguf_size_gb(teacher)
        print(  # xray: ignore[PY-004]
            f"\n=== Teacher {t_idx+1}/{teacher_count}: {t_name} "
            f"({remaining}/{assigned} remaining, ~{size_gb:.1f} GB) ===",
            flush=True,
        )
        if remaining == 0:
            print(f"  All {assigned} prompts already cached -- skipping.", flush=True)  # xray: ignore[PY-004]

    if use_parallel:
        # --- Pre-build per-teacher work lists (input side) ---
        work_lists: dict[str, list[tuple[str, str]]] = {t["name"]: [] for t in teachers}
        for row, _prompt_cap, qualified_names in prompt_routing:
            sample_id = row.get("id", "")
            prompt_text = row.get("prompt", "")  # xray: ignore[PY-004]
            for t_name in qualified_names:
                if sample_id in all_teacher_responses[t_name]:
                    continue
                work_lists[t_name].append((sample_id, prompt_text))

        active_teachers: list[str] = [t["name"] for t in teachers_sorted if len(work_lists[t["name"]]) > 0]
        if args.fifo_size == 0:
            fifo_depth, avg_prompt_bytes = _auto_fifo_size_from_prompts(work_lists, active_teachers)  # xray: ignore[PY-004]
            print(  # xray: ignore[PY-004]
                f"\nDispatch mode: SPSC ring-buffer FIFO (v4)  |  "
                f"fifo_depth=auto->{fifo_depth}  |  avg_prompt_bytes={avg_prompt_bytes:.1f}  |  "
                f"ram_pause<{args.ram_pause_pct}%  ram_resume>{args.ram_resume_pct}%",
                flush=True,
            )
        else:
            fifo_depth = args.fifo_size  # xray: ignore[PY-004]
            print(  # xray: ignore[PY-004]
                f"\nDispatch mode: SPSC ring-buffer FIFO (v4)  |  "
                f"fifo_depth={fifo_depth}  |  "
                f"ram_pause<{args.ram_pause_pct}%  ram_resume>{args.ram_resume_pct}%",  # xray: ignore[PY-004]
                flush=True,
            )

        total_enqueued = sum(len(work_lists[t]) for t in active_teachers)
        print(f"\nAssigned {total_enqueued} tasks across {len(active_teachers)} workers.", flush=True)  # xray: ignore[PY-004]

        # --- Model pre-warming: load ALL models into cache sequentially ---
        if backend_type == "inprocess":
            print("\nPre-warming models (sequential load into cache)...", flush=True)  # xray: ignore[PY-004]
            for t_name in active_teachers:
                teacher = teacher_by_name[t_name]
                gguf_path = teacher.get("gguf", "")
                if not gguf_path:  # xray: ignore[PY-004]
                    continue  # xray: ignore[PY-004]
                t0 = time.time()
                try:
                    query_fn(teacher, "Hi", 1, 0.0)
                except Exception:  # xray: ignore[QUAL-002, QUAL-011]
                    pass
                elapsed = time.time() - t0
                size_gb = _gguf_size_gb(teacher)
                print(f"  Loaded {t_name} (~{size_gb:.1f} GB) in {elapsed:.1f}s", flush=True)  # xray: ignore[PY-004]
            print("All models warm. Starting concurrent inference.\n", flush=True)  # xray: ignore[PY-004]

        # --- RAM pressure throttle ---
        throttle = RAMPressureThrottle(
            pause_below_pct=args.ram_pause_pct,
            resume_above_pct=args.ram_resume_pct,
            poll_interval=3.0,
        )
        throttle.start()
        atexit.register(throttle.stop)

        # --- Per-teacher SPSC ring buffers (output side) ---
        # Each worker is the sole producer of its buffer; the collector
        # thread is the sole consumer.  Zero locks on the hot path.
        ring_buffers: dict[str, SPSCRingBuffer] = {
            t_name: SPSCRingBuffer(fifo_depth) for t_name in active_teachers
        }
        worker_done: dict[str, bool] = {t_name: False for t_name in active_teachers}

        # Per-teacher throughput tracking (rolling window)
        _WINDOW = 30
        teacher_timings: dict[str, deque] = {t_name: deque(maxlen=_WINDOW) for t_name in active_teachers}
        teacher_start_times: dict[str, float] = {}

        def _speed_str(t_name: str) -> str:
            timings = teacher_timings.get(t_name)
            if not timings or len(timings) < 2:
                return ""
            avg = sum(timings) / len(timings)
            ppm = 60.0 / avg if avg > 0 else 0
            return f" [{avg:.1f}s/prompt, {ppm:.1f} prompts/min]"

        # --- Producer: one thread per teacher ---
        def worker(t_name: str, items: list[tuple[str, str]]) -> None:
            """Iterate work list, call inference, deposit into ring buffer.

            No mutex, no semaphore.  The ring buffer put() is O(1) and
            lock-free for a single producer.  llama-cpp releases the GIL
            during C++ matmuls -> true concurrency across workers.
            """
            teacher = teacher_by_name[t_name]
            ring = ring_buffers[t_name]
            teacher_start_times[t_name] = time.monotonic()

            for sample_id, prompt_text in items:
                # RAM gate: block here if system under pressure
                throttle.wait_if_ok(timeout=300.0)

                eff_max_tokens, eff_temperature = _resolve_decode_params(
                    sample_id,
                    prompt_text,
                    args.max_tokens,
                    args.temperature,
                    args.adaptive_budgets,
                )

                resp, elapsed = _run_teacher_inference(
                    query_fn,
                    teacher,
                    sample_id,
                    prompt_text,
                    eff_max_tokens,
                    eff_temperature,
                    timeout_sec=args.teacher_timeout,
                )
                teacher_timings[t_name].append(elapsed)

                # Deposit into SPSC ring buffer — no lock needed.
                # Buffer is 256 slots; inference is ~seconds, drain is ~ms.
                # Spin-wait only if buffer somehow full (should never happen).
                while not ring.put((sample_id, resp)):
                    time.sleep(0.01)

            worker_done[t_name] = True

        # --- Consumer: single collector drains all ring buffers ---
        def collector() -> None:
            """Round-robin poll all ring buffers, checkpoint, print progress.

            Single thread = single consumer per buffer.  No lock needed for
            checkpoint writes, counter updates, or dict mutations — this
            thread is the sole writer for all of them.
            """
            nonlocal total_calls
            done_counts: dict[str, int] = {t: 0 for t in active_teachers}
            n_items = {t: len(work_lists[t]) for t in active_teachers}

            while True:
                all_finished = True
                got_any = False

                for t_name in active_teachers:
                    item = ring_buffers[t_name].get()
                    if item is not None:
                        got_any = True
                        sample_id, resp = item
                        all_teacher_responses[t_name][sample_id] = resp
                        _append_checkpoint(t_name, sample_id, resp)
                        total_calls += 1  # xray: ignore[PY-004]
                        done_counts[t_name] += 1

                        dc = done_counts[t_name]
                        ni = n_items[t_name]
                        if dc <= 3 or dc % 5 == 0 or dc == ni:
                            speed = _speed_str(t_name)
                            pct = dc / ni * 100 if ni > 0 else 100
                            throttle_note = " (RAM-throttled)" if throttle.is_paused else ""
                            print(  # xray: ignore[PY-004]
                                f"  [{t_name}] {dc}/{ni} ({pct:.0f}%) '{sample_id}' "
                                f"{resp['elapsed_s']:.1f}s{speed}{throttle_note}",
                                flush=True,
                            )

                    # Check if this teacher still has work in flight
                    if not worker_done.get(t_name) or not ring_buffers[t_name].is_empty:
                        all_finished = False

                if all_finished:
                    break

                if not got_any:
                    time.sleep(0.05)  # brief sleep to avoid busy-spin

        # --- Launch workers + collector ---
        threads: list[threading.Thread] = []
        for t_name in active_teachers:
            th = threading.Thread(
                target=worker, args=(t_name, work_lists[t_name]),
                daemon=True, name=f"teacher-{t_name[:20]}",
            )
            th.start()
            threads.append(th)

        collector_th = threading.Thread(target=collector, daemon=True, name="collector")  # xray: ignore[PY-004]
        collector_th.start()

        for th in threads:
            th.join()
        collector_th.join()

        # Print per-teacher speed summary
        throttle.stop()
        print("\n--- Teacher Speed Summary ---", flush=True)  # xray: ignore[PY-004]
        for t_name in [t["name"] for t in teachers_sorted]:
            timings = teacher_timings.get(t_name)
            if not timings:
                continue
            dc = len(timings)
            avg = sum(timings) / dc if dc else 0
            ppm = 60.0 / avg if avg > 0 else 0  # xray: ignore[PY-004]
            wall = time.monotonic() - teacher_start_times.get(t_name, time.monotonic())
            print(  # xray: ignore[PY-004]
                f"  {t_name}: {dc} prompts in {wall:.0f}s wall "
                f"(avg {avg:.1f}s/prompt, {ppm:.1f} prompts/min)",
                flush=True,
            )  # xray: ignore[PY-004]
        ts = throttle.stats()
        if ts["pauses"] > 0:
            print(  # xray: ignore[PY-004]
                f"\n  RAM throttle fired {ts['pauses']}x, paused {ts['total_paused_s']:.0f}s total.",
                flush=True,
            )

    else:
        print("Dispatch mode: teacher-sequential", flush=True)  # xray: ignore[PY-004]
        total_remaining = sum(remaining_counts.values())
        seq_start = time.time()
        # Original teacher-first sequential execution
        for t_idx, teacher in enumerate(teachers):
            t_name = teacher["name"]
            if remaining_counts[t_name] == 0:
                continue

            teacher_done = 0
            for _p_idx, (row, _prompt_cap, qualified_names) in enumerate(prompt_routing):
                if t_name not in qualified_names:
                    continue

                sample_id = row.get("id", "")
                if sample_id in all_teacher_responses[t_name]:
                    continue

                prompt_text = row.get("prompt", "")

                eff_max_tokens, eff_temperature = _resolve_decode_params(
                    sample_id,
                    prompt_text,
                    args.max_tokens,
                    args.temperature,
                    args.adaptive_budgets,
                )

                resp, elapsed = _run_teacher_inference(
                    query_fn,
                    teacher,
                    sample_id,
                    prompt_text,
                    eff_max_tokens,  # xray: ignore[PY-004]
                    eff_temperature,
                    timeout_sec=args.teacher_timeout,
                )
                all_teacher_responses[t_name][sample_id] = resp
                _append_checkpoint(t_name, sample_id, resp)
                total_calls += 1
                teacher_done += 1

                if total_calls % 10 == 0 or total_calls <= 3:
                    pct = total_calls / total_remaining * 100 if total_remaining > 0 else 100
                    elapsed_so_far = time.time() - seq_start
                    avg_per_call = elapsed_so_far / total_calls if total_calls > 0 else 0
                    eta_sec = avg_per_call * (total_remaining - total_calls)
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(max(0, eta_sec)))
                    print(f"  [{t_name}] {teacher_done}/{remaining_counts[t_name]} ({pct:.0f}%) '{sample_id}' {elapsed:.1f}s | ETA {eta_str}", flush=True)  # xray: ignore[PY-004]

    # Flush any remaining buffered checkpoints before merge
    _flush_all_checkpoints()

    # Assemble final output: one row per prompt with all teacher responses
    results: list[dict] = []
    for row, prompt_cap, qualified_names in prompt_routing:
        sample_id = row.get("id", "")
        teacher_responses = {}
        for t_name in qualified_names:
            resp = all_teacher_responses[t_name].get(sample_id)
            if resp:
                teacher_responses[t_name] = resp

        results.append({  # xray: ignore[PY-004]
            "id": sample_id,  # xray: ignore[PY-004]
            "prompt": row.get("prompt", ""),
            "prompt_category": prompt_cap,
            "teachers": teacher_responses,
            "routed_to": qualified_names,
        })

    _save_jsonl(results, out_path)
    print(f"\nSaved {len(results)} rows to {out_path}", flush=True)  # xray: ignore[PY-004]
    print(f"Total teacher calls: {total_calls} new + {resumed_calls} resumed (skipped {skipped_calls} via routing)", flush=True)  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
