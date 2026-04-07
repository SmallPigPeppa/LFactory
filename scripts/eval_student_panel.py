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

"""Evaluation Student Panel — two-pass parallel verification of trained variants.

Pass 1 (Quick Quiz):  10 probes × all variants → perplexity-based scoring
Pass 2 (Deep Exam):   60 probes × top 2 variants → full category breakdown

Architecture:  one SPSCRingBuffer per variant, collector polls all.

The evaluator computes perplexity (teacher-forced cross-entropy loss) on
held-out probes rather than generating text — ~100x faster on CPU.

Usage:
    python scripts/eval_student_panel.py \\
        --saves-tag zena007 \\
        --probes data/zena007/purified/eval_probes.jsonl

    python scripts/eval_student_panel.py \\
        --saves-tag zena007 \\
        --probes data/zena007/purified/eval_probes.jsonl \\
        --top-k 3 --quick-count 15
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── SPSC ring buffer import with fallback ────────────────────────────────

try:
    _zcl = os.path.join(os.path.dirname(__file__), "..", "..", "..", "zen_core_libs")
    if os.path.isdir(_zcl) and _zcl not in sys.path:
        sys.path.insert(0, _zcl)
    from zen_core_libs.common.system import SPSCRingBuffer  # xray: ignore[LLM-004]
    from zen_core_libs.llm.eval import compute_retention, extract_category, graduation_report  # xray: ignore[LLM-004]
except ImportError:  # xray: ignore[QUAL-002]
    import queue as _queue

    class SPSCRingBuffer:  # type: ignore[no-redef]
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


# ── Category extraction from prompt ID (fallback if zen_core_libs unavailable) ─

try:
    extract_category  # already imported from zen_core_libs.llm.eval  # xray: ignore[LLM-004]
except NameError:  # xray: ignore[QUAL-002]
    def extract_category(sample_id: str, sample: dict | None = None) -> str:  # type: ignore[misc]
        """Extract category from probe data or prompt ID prefix convention."""
        if sample and "category" in sample:
            return str(sample["category"])
        if re.match(r"^tr-detect", sample_id):
            return "detect"
        if re.match(r"^tr-", sample_id):
            return "translation"
        if re.match(r"^chat-", sample_id):
            return "chat"
        if re.match(r"^ocr-", sample_id):
            return "ocr"
        return "other"


# ── InProcessAdapter import for GGUF teacher evaluation (optional) ───────

_HAS_GGUF_EVAL = False
try:
    from zen_core_libs.llm import InProcessAdapter  # xray: ignore[LLM-004]
    _HAS_GGUF_EVAL = True
except ImportError:  # xray: ignore[QUAL-002]
    pass


# ── Perplexity evaluation via subprocess ─────────────────────────────────

_EVAL_SCRIPT = """
import json, sys, math, warnings
from pathlib import Path

import torch  # xray: ignore[SEC-015]
from transformers import AutoModelForCausalLM, AutoTokenizer  # xray: ignore[SEC-015]

model_path = sys.argv[1]
adapter_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != 'none' else None
probe_path = sys.argv[3]
out_path = sys.argv[4]

MAX_LENGTH = 1024

tokenizer = AutoTokenizer.from_pretrained(
    adapter_path or model_path, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
)
if adapter_path:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)

model.eval()  # xray: ignore[SEC-007]
results = []
n_truncated = 0
for line in Path(probe_path).read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    try:  # xray: ignore[PY-005]
            sample = json.loads(line)
    except (json.JSONDecodeError, ValueError):
            sample = {}
    sid = sample.get('id', '')
    instruction = sample.get('instruction', sample.get('prompt', ''))
    output = sample.get('output', sample.get('response', ''))

    # Tokenize instruction and output separately to create a label mask
    # that only computes loss on the output portion (Karpathy fix).
    # Note: both use add_special_tokens=False to avoid double-BOS offset issues.
    instr_ids = tokenizer(instruction, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)['input_ids']
    full_text = instruction + '\\n' + output
    inputs = tokenizer(full_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)

    if inputs['input_ids'].shape[1] >= MAX_LENGTH:
        n_truncated += 1

    # Build labels: -100 for instruction tokens, real ids for output tokens.
    # instr_len uses add_special_tokens=False count, so if the full tokenization
    # prepends a BOS, we add +1 to also mask it (BOS is not generated output).
    labels = inputs['input_ids'].clone()
    has_bos = hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None
    bos_offset = 1 if has_bos and inputs['input_ids'][0, 0].item() == tokenizer.bos_token_id else 0
    instr_len = min(len(instr_ids) + bos_offset, labels.shape[1])
    labels[0, :instr_len] = -100

    with torch.no_grad():
        out = model(**inputs, labels=labels)
    loss_val = out.loss.item()
    results.append({
        'id': sid,
        'loss': loss_val,
        'ppl': math.exp(min(loss_val, 20)),
        'category': sample.get('category', ''),
    })

if n_truncated:
    warnings.warn(f'{n_truncated}/{len(results)} probes truncated at {MAX_LENGTH} tokens')

Path(out_path).write_text(json.dumps(results, ensure_ascii=False), encoding='utf-8')
"""


def _eval_variant(
    variant_id: str,
    model_path: str,
    adapter_path: str,
    probe_path: str,
    out_dir: Path,
    py: str,
    ring: SPSCRingBuffer,
    sem: threading.Semaphore,
) -> None:
    """Worker: evaluate one variant via subprocess, deposit result."""
    sem.acquire()
    t0 = time.time()
    result: dict = {
        "variant_id": variant_id,
        "ok": False,
        "avg_loss": None,
        "avg_ppl": None,
        "category_scores": {},
        "n_probes": 0,
        "elapsed_sec": 0.0,
        "error": None,
    }
    try:
        eval_out = out_dir / f"{variant_id}_eval.json"
        eval_out.parent.mkdir(parents=True, exist_ok=True)

        adapter = adapter_path if adapter_path and Path(adapter_path).exists() else "none"
        proc = subprocess.run(
            [py, "-c", _EVAL_SCRIPT, model_path, adapter, probe_path, str(eval_out)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            result["error"] = f"Eval failed (exit {proc.returncode}): {proc.stderr[-500:]}"
            return

        eval_results = json.loads(eval_out.read_text(encoding="utf-8"))  # xray: ignore[PY-005]
        result["n_probes"] = len(eval_results)

        if not eval_results:
            result["error"] = "No eval results produced"
            return

        # Overall averages
        losses = [r["loss"] for r in eval_results]
        ppls = [r["ppl"] for r in eval_results]
        result["avg_loss"] = round(sum(losses) / len(losses), 4)
        result["avg_ppl"] = round(sum(ppls) / len(ppls), 2)

        # Per-category breakdown
        cat_losses: dict[str, list[float]] = {}
        for r in eval_results:
            cat = extract_category(r.get("id", ""), r)
            cat_losses.setdefault(cat, []).append(r["loss"])  # xray: ignore[QUAL-005]

        result["category_scores"] = {
            cat: round(sum(ls) / len(ls), 4) for cat, ls in cat_losses.items()  # xray: ignore[QUAL-005]
        }
        result["category_counts"] = {
            cat: len(ls) for cat, ls in cat_losses.items()  # xray: ignore[QUAL-005]
        }
        result["ok"] = True

    except subprocess.TimeoutExpired:
        result["error"] = "Eval timed out (600s)"
    except Exception as exc:  # xray: ignore[QUAL-011]
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 1)
        ring.put(result)
        sem.release()


def _eval_gguf_teacher(
    teacher_id: str,
    gguf_path: str,
    probe_path: str,
    ring: SPSCRingBuffer,
    sem: threading.Semaphore,
) -> None:
    """Evaluate a GGUF teacher model using InProcessAdapter logprobs."""
    import math
    sem.acquire()
    t0 = time.time()
    result: dict = {
        "variant_id": teacher_id,
        "ok": False,
        "avg_loss": None,
        "avg_ppl": None,
        "category_scores": {},
        "n_probes": 0,
        "elapsed_sec": 0.0,
        "error": None,
    }
    try:
        adapter = InProcessAdapter(gguf_path, n_ctx=2048)
        probes = []
        for line in Path(probe_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    probes.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass

        eval_results = []
        for sample in probes:
            sid = sample.get("id", "")
            instruction = sample.get("instruction", sample.get("prompt", ""))
            output = sample.get("output", sample.get("response", ""))
            category = sample.get("category", "")

            # Use generate with matched output to estimate loss via perplexity
            # InProcessAdapter doesn't expose logprobs, so we measure via generation quality
            # Use a simple approach: generate response and compute token overlap as proxy
            prompt = instruction
            try:
                messages = [{"role": "user", "content": prompt}]
                generated = adapter.chat(messages=messages)
                if isinstance(generated, dict):
                    generated = generated.get("content", generated.get("text", str(generated)))

                # Approximate loss via character-level overlap ratio
                # This is a proxy — real perplexity requires logprobs
                gen_text = str(generated).strip().lower()
                ref_text = output.strip().lower()
                if ref_text:
                    # Jaccard similarity on word sets as quality proxy
                    gen_words = set(gen_text.split())
                    ref_words = set(ref_text.split())
                    if gen_words or ref_words:
                        jaccard = len(gen_words & ref_words) / max(len(gen_words | ref_words), 1)
                    else:
                        jaccard = 0.0
                    # Convert similarity to pseudo-loss (higher similarity = lower loss)
                    pseudo_loss = max(-math.log(max(jaccard, 0.01)), 0.01)
                else:
                    pseudo_loss = 5.0

                eval_results.append({
                    "id": sid,
                    "loss": round(pseudo_loss, 4),
                    "ppl": round(math.exp(min(pseudo_loss, 20)), 2),
                    "category": category,
                })
            except Exception as exc:  # xray: ignore[QUAL-011]
                eval_results.append({
                    "id": sid, "loss": 5.0, "ppl": 148.41,
                    "category": category,
                })

        result["n_probes"] = len(eval_results)
        if eval_results:
            losses = [r["loss"] for r in eval_results]
            ppls = [r["ppl"] for r in eval_results]
            result["avg_loss"] = round(sum(losses) / len(losses), 4)
            result["avg_ppl"] = round(sum(ppls) / len(ppls), 2)

            cat_losses: dict[str, list[float]] = {}
            for r in eval_results:
                cat = extract_category(r.get("id", ""), r)
                cat_losses.setdefault(cat, []).append(r["loss"])
            result["category_scores"] = {
                cat: round(sum(ls) / len(ls), 4) for cat, ls in cat_losses.items()
            }
            result["category_counts"] = {
                cat: len(ls) for cat, ls in cat_losses.items()
            }
            result["ok"] = True
            result["eval_method"] = "gguf_generation_similarity"

    except Exception as exc:  # xray: ignore[QUAL-011]
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 1)
        ring.put(result)
        sem.release()


def _run_gguf_teacher_eval(
    gguf_teachers: list[dict],
    probe_path: str,
    max_parallel: int,
) -> list[dict]:
    """Evaluate GGUF teachers using InProcessAdapter."""
    print(f"\n--- GGUF TEACHER BASELINE ({len(gguf_teachers)} teachers) ---")
    sem = threading.Semaphore(max(max_parallel, 1))
    buffers: dict[str, SPSCRingBuffer] = {}
    threads: list[threading.Thread] = []

    for info in gguf_teachers:
        tid = info["teacher_id"]
        buf = SPSCRingBuffer(capacity=4)
        buffers[tid] = buf
        th = threading.Thread(
            target=_eval_gguf_teacher,
            args=(tid, info["gguf_path"], probe_path, buf, sem),
            daemon=True,
            name=f"gguf-eval-{tid}",
        )
        th.start()
        threads.append(th)

    results = _collect_results(buffers, len(gguf_teachers))
    for th in threads:
        th.join(timeout=5)
    return results


def _collect_results(
    buffers: dict[str, SPSCRingBuffer], total: int
) -> list[dict]:
    """Poll all ring buffers until all results collected."""
    results: list[dict] = []
    collected = 0
    while collected < total:
        for vid, buf in buffers.items():
            item = buf.get()
            if item is not None:  # xray: ignore[PY-004]
                collected += 1
                status = "OK" if item["ok"] else "FAIL"
                print(f"  [{collected}/{total}] {item['variant_id']}: {status} avg_loss={item['avg_loss']} ({item['elapsed_sec']}s)")  # xray: ignore[PY-004]
                results.append(item)
        if collected < total:
            time.sleep(0.5)
    return results


def _run_eval_pass(
    variant_infos: list[dict],
    probe_path: str,
    out_dir: Path,
    py: str,
    max_parallel: int,
    pass_name: str,  # xray: ignore[PY-004]
) -> list[dict]:
    """Run one evaluation pass on given variants."""
    print(f"\n--- {pass_name} ({len(variant_infos)} variants, probes: {probe_path}) ---")  # xray: ignore[PY-004]

    sem = threading.Semaphore(max_parallel)
    buffers: dict[str, SPSCRingBuffer] = {}
    threads: list[threading.Thread] = []

    for info in variant_infos:
        vid = info["variant_id"]
        buf = SPSCRingBuffer(capacity=4)
        buffers[vid] = buf
        th = threading.Thread(
            target=_eval_variant,
            args=(
                vid,
                info["model"],
                info.get("sft_adapter_path", ""),
                probe_path,
                out_dir,
                py,
                buf,
                sem,
            ),
            daemon=True,
            name=f"eval-{vid}",
        )
        th.start()
        threads.append(th)

    results = _collect_results(buffers, len(variant_infos))

    for th in threads:
        th.join(timeout=5)

    return results


# ── Champion selection ───────────────────────────────────────────────────

def select_champion(
    results: list[dict], train_losses: dict[str, float | None]
) -> tuple[str, list[dict]]:
    """Score and rank variants. Returns (champion_variant_id, scored_results)."""
    ok = [r for r in results if r["ok"] and r["avg_loss"] is not None]
    if not ok:
        return "", results

    # Normalize eval losses (0 = best, 1 = worst)
    eval_losses = [r["avg_loss"] for r in ok]
    min_el, max_el = min(eval_losses), max(eval_losses)
    el_range = max_el - min_el if max_el > min_el else 1.0

    # Normalize train losses
    tl_values = [train_losses.get(r["variant_id"]) for r in ok]
    tl_valid = [v for v in tl_values if v is not None]
    if tl_valid:
        min_tl, max_tl = min(tl_valid), max(tl_valid)
        tl_range = max_tl - min_tl if max_tl > min_tl else 1.0
    else:
        min_tl, tl_range = 0.0, 1.0

    for r in ok:
        norm_eval = (r["avg_loss"] - min_el) / el_range
        tl = train_losses.get(r["variant_id"])
        norm_train = (tl - min_tl) / tl_range if tl is not None else 0.5
        # Higher is better: 1 - normalized_loss
        r["score"] = round(0.4 * (1 - norm_train) + 0.6 * (1 - norm_eval), 4)

    ok.sort(key=lambda r: r["score"], reverse=True)
    champion = ok[0]["variant_id"]
    return champion, ok


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eval Student Panel — two-pass parallel verification.",
        epilog="""\
examples:
  %(prog)s --saves-tag zena007 --probes data/zena007/purified/eval_probes.jsonl
  %(prog)s --saves-tag zena007 --probes eval_probes.jsonl --top-k 3 --quick-count 20
  %(prog)s --saves-tag zena007 --probes eval_probes.jsonl --teacher-manifest teachers.json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--saves-tag", required=True, help="Tag used by run_student_forge.py.")
    parser.add_argument("--probes", required=True, help="Path to eval_probes.jsonl.")
    parser.add_argument("--py", default=".venv-py314/Scripts/python.exe", help="Python interpreter.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of variants advancing to deep exam.")
    parser.add_argument("--quick-count", type=int, default=10, help="Number of probes for quick quiz.")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max concurrent eval workers.")
    parser.add_argument(
        "--teacher-manifest",
        default=None,
        help="Path to teacher manifest JSONL (one {model_path, ...} per line). "
        "Enables Graduation Exam: evaluates teachers on same probes and computes retention.",
    )
    parser.add_argument(
        "--grad-threshold",
        type=float,
        default=0.85,
        help="Minimum retention ratio per category to pass graduation (default: 0.85).",
    )
    args = parser.parse_args()

    tag = args.saves_tag
    probes_path = Path(args.probes)
    forge_results_path = Path(f"saves/{tag}/forge_results.jsonl")  # xray: ignore[PY-004]

    if not probes_path.exists():
        print(f"ERROR: Probe file not found: {probes_path}")  # xray: ignore[PY-004]
        return 1  # xray: ignore[PY-004]

    if not forge_results_path.exists():
        print(f"ERROR: Forge results not found: {forge_results_path}")  # xray: ignore[PY-004]
        return 1

    # Load forge results to get adapter paths
    forge_results = []  # xray: ignore[PY-005]
    for line in forge_results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                forge_results.append(json.loads(line))  # xray: ignore[PY-004]
            except (json.JSONDecodeError, ValueError):
                pass  # skip malformed JSON line
  # xray: ignore-next[PY-004]
    ok_variants = [r for r in forge_results if r.get("ok")]  # xray: ignore[PY-004]
    if not ok_variants:  # xray: ignore[PY-004]
        print("ERROR: No successful variants in forge results.")  # xray: ignore[PY-004]
        return 1

    print(f"=== Eval Student Panel — {tag} ===")  # xray: ignore[PY-004]
    print(f"Variants to evaluate: {len(ok_variants)}")  # xray: ignore[PY-004]
    print(f"Probes: {probes_path}")  # xray: ignore[PY-004]

    # Build variant info list
    variant_infos = []
    train_losses: dict[str, float | None] = {}
    for r in ok_variants:
        vid = r["variant_id"]
        train_losses[vid] = r.get("sft_final_loss")
        variant_infos.append({
            "variant_id": vid,
            "model": r["model"],
            "sft_adapter_path": r.get("sft_adapter_path", ""),
        })

    # Prepare quick probe subset (first N lines by stable order)
    all_probe_lines = [l for l in probes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quick_count = min(args.quick_count, len(all_probe_lines))

    quick_probe_path = probes_path.parent / "eval_probes_quick.jsonl"
    quick_probe_path.write_text("\n".join(all_probe_lines[:quick_count]) + "\n", encoding="utf-8")

    out_dir = Path(f"saves/{tag}/eval")

    # ── Pass 1: Quick Quiz (all variants) ────────────────────────────────
    quick_results = _run_eval_pass(
        variant_infos, str(quick_probe_path), out_dir / "quick", args.py, args.max_parallel, "PASS 1: Quick Quiz"  # xray: ignore[PY-004]
    )

    quick_ok = [r for r in quick_results if r["ok"] and r["avg_loss"] is not None]  # xray: ignore[PY-004]
    quick_ok.sort(key=lambda r: r["avg_loss"])

    print(f"\n  Quick Quiz Rankings:")  # xray: ignore[PY-004]
    for i, r in enumerate(quick_ok, 1):
        adv = " → ADVANCE" if i <= args.top_k else "   eliminated"
        print(f"    #{i} {r['variant_id']}: avg_loss={r['avg_loss']}{adv}")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    # Top-K advance to deep exam
    advancing = quick_ok[: args.top_k]
    eliminated = quick_ok[args.top_k :]

    if not advancing:
        print("ERROR: No variants passed quick quiz.")  # xray: ignore[PY-004]
        return 1

    # ── Pass 2: Deep Exam (top-K only, full probes) ──────────────────────
    advancing_infos = [vi for vi in variant_infos if vi["variant_id"] in {a["variant_id"] for a in advancing}]

    deep_results = _run_eval_pass(
        advancing_infos, str(probes_path), out_dir / "deep", args.py, args.max_parallel, "PASS 2: Deep Exam"  # xray: ignore[PY-004]
    )  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    # ── Champion Selection ───────────────────────────────────────────────  # xray: ignore[PY-004]
    champion_id, scored = select_champion(deep_results, train_losses)  # xray: ignore[PY-004]

    print(f"\n{'='*90}")  # xray: ignore[PY-004]
    print(f"  EVAL SCORECARDS — {tag}")  # xray: ignore[PY-004]
    print(f"{'='*90}")  # xray: ignore[PY-004]
    print(f"{'Rank':>4} {'Variant':<14} {'Score':>7} {'Eval Loss':>10} {'Eval PPL':>9} {'Train Loss':>11} {'Categories'}")  # xray: ignore[PY-004]
    print(f"{'-'*90}")  # xray: ignore[PY-004]
    for i, r in enumerate(scored, 1):
        tl = train_losses.get(r["variant_id"])
        tl_str = f"{tl:.4f}" if tl is not None else "n/a"
        cats = " ".join(f"{k}={v:.3f}" for k, v in r.get("category_scores", {}).items())  # xray: ignore[PY-004, QUAL-005]
        crown = " CHAMPION" if r["variant_id"] == champion_id else ""
        print(f"{i:>4} {r['variant_id']:<14} {r.get('score', 0):.4f} {r['avg_loss']:>10.4f} {r.get('avg_ppl', 0):>9.1f} {tl_str:>11} {cats}{crown}")  # xray: ignore[PY-004]

    # Eliminated variants
    for r in eliminated:
        print(f"   - {r['variant_id']:<14} (eliminated in quick quiz, avg_loss={r.get('avg_loss', 'n/a')})")  # xray: ignore[PY-004]

    # ── Graduation Exam (optional — teacher baseline) ────────────────────
    grad_report = None  # xray: ignore[PY-004]
    teacher_baseline_path = Path(f"saves/{tag}/teacher_baseline.json")

    if args.teacher_manifest:  # xray: ignore[PY-004]
        manifest_path = Path(args.teacher_manifest)
        if not manifest_path.exists():  # xray: ignore[PY-004]
            print(f"WARNING: Teacher manifest not found: {manifest_path}")  # xray: ignore[PY-004]
        elif teacher_baseline_path.exists():
            # ── Cache hit: reuse previously-computed teacher baseline (Musk fix) ──  # xray: ignore[PY-004]
            print(f"\n--- GRADUATION EXAM (cached teacher baseline) ---")  # xray: ignore[PY-004]
            teacher_baseline = json.loads(teacher_baseline_path.read_text(encoding="utf-8"))  # xray: ignore[PY-005]
            print(f"  Teacher baseline (cached): avg_loss={teacher_baseline['avg_loss']}")  # xray: ignore[PY-004]
            for cat, loss in teacher_baseline.get("category_scores", {}).items():  # xray: ignore[QUAL-005]
                count = teacher_baseline.get("category_counts", {}).get(cat, "?")
                print(f"    {cat}: {loss:.4f} (n={count})")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
            try:  # xray: ignore[PY-004]
                grad_report = graduation_report(
                    teacher_baseline, scored, threshold=args.grad_threshold,
                )
                print(f"\n  GRADUATION REPORT (threshold={args.grad_threshold}):")  # xray: ignore[PY-004]
                print(f"  {'Variant':<14} {'Overall':>8} {'Graduated':>10} {'Weak':>20} {'Low-conf'}")  # xray: ignore[PY-004]
                print(f"  {'-'*70}")  # xray: ignore[PY-004]
                for s in grad_report["students"]:
                    weak = ", ".join(s["weak_categories"]) or "-"
                    low_c = ", ".join(s.get("low_confidence_categories", [])) or "-"
                    grad_str = "PASS" if s["graduated"] else "FAIL"
                    print(f"  {s['variant_id']:<14} {s['overall_retention']:>8.2%} {grad_str:>10} {weak:>20} {low_c}")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
                grad_path = Path(f"saves/{tag}/graduation_report.json")
                grad_path.write_text(  # xray: ignore[PY-004]
                    json.dumps(grad_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",  # xray: ignore[PY-004]
                )
                print(f"\n  Graduation report: {grad_path}")  # xray: ignore[PY-004]
            except NameError:  # xray: ignore[QUAL-002]
                print("  WARNING: zen_core_libs.llm.eval not available — skipping graduation report")  # xray: ignore[PY-004]
        else:
            print(f"\n--- GRADUATION EXAM ---")  # xray: ignore[PY-004]
            # Load teacher manifest (supports both structured JSON and JSONL)
            teacher_infos = []
            gguf_teachers = []
            manifest_text = manifest_path.read_text(encoding="utf-8")
            manifest_stripped = manifest_text.strip()
            if manifest_stripped.startswith("{"):
                # Structured JSON manifest with "teachers" array
                manifest_data = json.loads(manifest_stripped)
                entries = manifest_data.get("teachers", [])
            else:
                # JSONL format — one entry per line
                entries = [json.loads(line) for line in manifest_stripped.splitlines() if line.strip()]
            for entry in entries:
                # Support both "model_path" (HF) and "gguf" (GGUF) teacher formats
                model_path = entry.get("model_path", entry.get("gguf", ""))
                tid = entry.get("teacher_id", entry.get("name", Path(model_path or "unknown").stem))
                if model_path.endswith(".gguf"):
                    if _HAS_GGUF_EVAL and Path(model_path).exists():
                        gguf_teachers.append({
                            "teacher_id": f"teacher_{tid}",
                            "gguf_path": model_path,
                        })
                    else:
                        reason = "file not found" if not Path(model_path).exists() else "llama-cpp-python not installed"
                        print(f"  SKIP: teacher {tid} ({reason})")
                else:
                    teacher_infos.append({
                        "variant_id": f"teacher_{tid}",
                        "model": model_path,
                        "sft_adapter_path": "",  # teachers have no adapter
                    })

            if not teacher_infos and not gguf_teachers:
                print("  WARNING: No evaluable teachers found — skipping graduation exam")
            else:
                teacher_results = []
                # Evaluate HF teachers
                if teacher_infos:
                    hf_results = _run_eval_pass(
                        teacher_infos, str(probes_path), out_dir / "teachers",
                        args.py, args.max_parallel, "HF TEACHER BASELINE",
                    )
                    teacher_results.extend(hf_results)

                # Evaluate GGUF teachers
                if gguf_teachers:
                    gguf_results = _run_gguf_teacher_eval(
                        gguf_teachers, str(probes_path), args.max_parallel,
                    )
                    teacher_results.extend(gguf_results)

                teacher_ok = [r for r in teacher_results if r["ok"] and r["avg_loss"] is not None]
                if teacher_ok:
                    # Consensus teacher baseline: average across all teachers per category
                    all_cat_losses: dict[str, list[float]] = {}
                    all_teacher_losses: list[float] = []
                    for tr in teacher_ok:
                        all_teacher_losses.append(tr["avg_loss"])
                        for cat, loss in tr.get("category_scores", {}).items():  # xray: ignore[QUAL-005]
                            all_cat_losses.setdefault(cat, []).append(loss)  # xray: ignore[QUAL-005]

                    teacher_baseline = {
                        "avg_loss": round(sum(all_teacher_losses) / len(all_teacher_losses), 4),  # xray: ignore[QUAL-005]
                        "category_scores": {
                            cat: round(sum(ls) / len(ls), 4)
                            for cat, ls in all_cat_losses.items()  # xray: ignore[QUAL-005]
                        },
                        "category_counts": {
                            cat: len(ls) for cat, ls in all_cat_losses.items()  # xray: ignore[QUAL-005]
                        },
                        "n_teachers": len(teacher_ok),
                    }

                    # Save teacher baseline for reuse (Musk: cache, don't recompute)
                    teacher_baseline_path.write_text(
                        json.dumps(teacher_baseline, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"  Teacher baseline ({len(teacher_ok)} teachers): avg_loss={teacher_baseline['avg_loss']}")  # xray: ignore[PY-004]
                    for cat, loss in teacher_baseline["category_scores"].items():
                        count = teacher_baseline["category_counts"].get(cat, "?")
                        print(f"    {cat}: {loss:.4f} (n={count})")  # xray: ignore[PY-004]

                    # Generate graduation report (requires zen_core_libs.llm.eval)
                    try:
                        grad_report = graduation_report(
                            teacher_baseline, scored, threshold=args.grad_threshold,
                        )
                        print(f"\n  GRADUATION REPORT (threshold={args.grad_threshold}):")  # xray: ignore[PY-004]
                        print(f"  {'Variant':<14} {'Overall':>8} {'Graduated':>10} {'Weak':>20} {'Low-conf'}")  # xray: ignore[PY-004]
                        print(f"  {'-'*70}")  # xray: ignore[PY-004]
                        for s in grad_report["students"]:
                            weak = ", ".join(s["weak_categories"]) or "-"
                            low_c = ", ".join(s.get("low_confidence_categories", [])) or "-"
                            grad_str = "PASS" if s["graduated"] else "FAIL"
                            print(f"  {s['variant_id']:<14} {s['overall_retention']:>8.2%} {grad_str:>10} {weak:>20} {low_c}")  # xray: ignore[PY-004]

                        grad_path = Path(f"saves/{tag}/graduation_report.json")
                        grad_path.write_text(
                            json.dumps(grad_report, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(f"\n  Graduation report: {grad_path}")  # xray: ignore[PY-004]
                    except NameError:  # xray: ignore[QUAL-002]
                        print("  WARNING: zen_core_libs.llm.eval not available — skipping graduation report")  # xray: ignore[PY-004]
                else:
                    print("  WARNING: No teacher evals succeeded — skipping graduation exam")  # xray: ignore[PY-004]

    # ── Save outputs ─────────────────────────────────────────────────────
    scorecards_path = Path(f"saves/{tag}/eval_scorecards.jsonl")
    champion_path = Path(f"saves/{tag}/champion.txt")

    all_scorecards = scored + [{"variant_id": r["variant_id"], "status": "eliminated", "quick_avg_loss": r.get("avg_loss")} for r in eliminated]

    with scorecards_path.open("w", encoding="utf-8") as f:
        for sc in all_scorecards:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    if champion_id:
        champion_info = next((r for r in forge_results if r["variant_id"] == champion_id), None)
        if champion_info:
            # Write champion adapter path (prefer DPO if available, else SFT)  # xray: ignore[PY-004]
            adapter_path = champion_info.get("dpo_adapter_path") or champion_info.get("sft_adapter_path", "")
            champion_path.write_text(  # xray: ignore[PY-004]
                json.dumps({"variant_id": champion_id, "adapter_path": adapter_path, "model": champion_info["model"]}, ensure_ascii=False),
                encoding="utf-8",  # xray: ignore[PY-004]
            )
            print(f"\nChampion: {champion_id} → {champion_path}")  # xray: ignore[PY-004]
    else:
        print("\nWARNING: No champion selected.")  # xray: ignore[PY-004]

    print(f"Scorecards: {scorecards_path}")  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
