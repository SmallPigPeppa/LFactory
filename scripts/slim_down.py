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

"""Slim-Down — parallel GGUF quantization export with speed benchmarking.

Exports the champion model (or any adapter) to multiple GGUF quant levels
in parallel, then runs a 5-probe speed bench on each exported file.

Architecture: one SPSCRingBuffer per quant worker, collector polls all.

Usage:
    python scripts/slim_down.py \\
        --saves-tag zena007 \\
        --quants Q4_K_M Q5_K_M Q8_0

    python scripts/slim_down.py \\
        --model-path Qwen/Qwen2.5-1.5B \\
        --adapter-path saves/zena007/B/sft \\
        --quants Q4_K_M \\
        --probes data/zena007/purified/eval_probes.jsonl \\
        --bench-count 5
"""

from __future__ import annotations

import argparse
import json
import os
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


# ── InProcessAdapter import for speed bench (optional) ───────────────────

_HAS_IN_PROCESS = False
try:
    from zen_core_libs.llm import InProcessAdapter  # xray: ignore[LLM-004]

    _HAS_IN_PROCESS = True
except ImportError:  # xray: ignore[QUAL-002]
    pass


# ── GGUF export worker ──────────────────────────────────────────────────

def _export_gguf(
    quant: str,
    model_path: str,
    adapter_path: str,
    output_dir: Path,
    py: str,
    ring: SPSCRingBuffer,
    sem: threading.Semaphore,
) -> None:
    """Merge adapter + export to GGUF at a given quantization level."""
    sem.acquire()
    t0 = time.time()
    result: dict = {
        "quant": quant,
        "ok": False,
        "gguf_path": "",
        "size_mb": 0.0,
        "elapsed_sec": 0.0,
        "error": None,
    }
    try:
        out_path = output_dir / quant
        out_path.mkdir(parents=True, exist_ok=True)

        export_yaml = output_dir / f"export_{quant}.yaml"
        config = {
            "model_name_or_path": model_path,
            "adapter_name_or_path": adapter_path,
            "template": "qwen",
            "export_dir": str(out_path),
            "export_size": 2,
            "export_quantization_bit": _quant_to_bits(quant),
            "export_legacy_format": False,
        }
        export_yaml.write_text(
            "\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()),
            encoding="utf-8",
        )

        proc = subprocess.run(
            [py, "-m", "llamafactory.cli", "export", str(export_yaml)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            result["error"] = f"Export failed (exit {proc.returncode}): {proc.stderr[-500:]}"
            return

        # Find the exported GGUF file
        gguf_files = list(out_path.glob("*.gguf"))
        if gguf_files:
            gguf_file = gguf_files[0]
            result["gguf_path"] = str(gguf_file)
            result["size_mb"] = round(gguf_file.stat().st_size / (1024 * 1024), 1)
        else:
            # Might be a merged model directory instead of GGUF
            result["gguf_path"] = str(out_path)
            total_size = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
            result["size_mb"] = round(total_size / (1024 * 1024), 1)

        result["ok"] = True

    except subprocess.TimeoutExpired:
        result["error"] = "Export timed out (1800s)"
    except Exception as exc:  # xray: ignore[QUAL-011]
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = round(time.time() - t0, 1)
        ring.put(result)
        sem.release()


def _quant_to_bits(quant: str) -> int:
    """Map GGUF quant name to approximate bit width for LlamaFactory export."""
    q = quant.upper()
    if "Q4" in q:
        return 4
    if "Q5" in q:
        return 5
    if "Q8" in q:
        return 8
    if "Q3" in q:
        return 3
    if "Q2" in q:
        return 2
    return 4


# ── Speed bench ──────────────────────────────────────────────────────────

def _bench_gguf(gguf_path: str, probes: list[str], bench_count: int) -> dict:
    """Run a quick speed bench on a GGUF file using InProcessAdapter."""
    if not _HAS_IN_PROCESS:
        return {"tok_per_sec": None, "note": "InProcessAdapter not available"}

    if not Path(gguf_path).exists() or not gguf_path.endswith(".gguf"):
        return {"tok_per_sec": None, "note": f"Not a GGUF file: {gguf_path}"}

    try:
        adapter = InProcessAdapter(gguf_path)
        total_tokens = 0
        total_time = 0.0
        count = min(bench_count, len(probes))

        for prompt in probes[:count]:
            t0 = time.time()
            resp = adapter.generate(prompt, max_tokens=64)
            elapsed = time.time() - t0
            tokens = len(resp.split())  # approximate
            total_tokens += tokens
            total_time += elapsed

        tok_sec = round(total_tokens / total_time, 1) if total_time > 0 else 0
        return {"tok_per_sec": tok_sec, "n_probes": count, "total_tokens": total_tokens}
    except Exception as exc:  # xray: ignore[QUAL-011]
        return {"tok_per_sec": None, "note": str(exc)}


# ── Collector ────────────────────────────────────────────────────────────

def _collect(buffers: dict[str, SPSCRingBuffer], total: int) -> list[dict]:
    results: list[dict] = []
    collected = 0
    while collected < total:
        for qname, buf in buffers.items():
            item = buf.get()
            if item is not None:
                collected += 1
                status = "OK" if item["ok"] else "FAIL"
                print(f"  [{collected}/{total}] {item['quant']}: {status} size={item['size_mb']}MB ({item['elapsed_sec']}s)")  # xray: ignore[PY-004]
                results.append(item)
        if collected < total:
            time.sleep(0.5)
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Slim-Down — parallel GGUF export + speed bench.")
    parser.add_argument("--saves-tag", help="Tag from forge pipeline (reads champion.txt).")
    parser.add_argument("--model-path", help="Base model path (overrides champion.txt).")
    parser.add_argument("--adapter-path", help="Adapter path (overrides champion.txt).")
    parser.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"], help="Quantization levels.")
    parser.add_argument("--probes", help="Probe file for speed bench.")
    parser.add_argument("--bench-count", type=int, default=5, help="Number of probes per bench.")
    parser.add_argument("--max-parallel", type=int, default=2, help="Max concurrent export workers.")
    parser.add_argument("--py", default=".venv-py314/Scripts/python.exe", help="Python interpreter.")
    parser.add_argument("--output-dir", help="Output directory for GGUF files.")
    parser.add_argument("--recommend-quant", action="store_true",
                        help="Analyze Pareto frontier of size vs speed and recommend optimal quant.")
    args = parser.parse_args()

    # Resolve model/adapter from champion.txt if using saves-tag
    model_path = args.model_path
    adapter_path = args.adapter_path

    if args.saves_tag and not (model_path and adapter_path):
        champion_file = Path(f"saves/{args.saves_tag}/champion.txt")
        if not champion_file.exists():
            print(f"ERROR: No champion.txt found at {champion_file}")  # xray: ignore[PY-004]
            return 1
        try:  # xray: ignore[PY-005]
                    champion = json.loads(champion_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
                    champion = {}
        model_path = model_path or champion.get("model", "")
        adapter_path = adapter_path or champion.get("adapter_path", "")  # xray: ignore[PY-004]

    if not model_path:
        print("ERROR: --model-path or --saves-tag with champion.txt required.")  # xray: ignore[PY-004]
        return 1
  # xray: ignore-next[PY-004]
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"saves/{args.saves_tag or 'export'}/gguf")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    print(f"=== Slim-Down — GGUF Export ===")  # xray: ignore[PY-004]
    print(f"Model: {model_path}")  # xray: ignore[PY-004]
    print(f"Adapter: {adapter_path or 'none'}")  # xray: ignore[PY-004]
    print(f"Quants: {', '.join(args.quants)}")  # xray: ignore[PY-004]
    print(f"Output: {out_dir}")  # xray: ignore[PY-004]

    # Launch parallel exports
    sem = threading.Semaphore(args.max_parallel)
    buffers: dict[str, SPSCRingBuffer] = {}
    threads: list[threading.Thread] = []

    for quant in args.quants:
        buf = SPSCRingBuffer(capacity=4)
        buffers[quant] = buf
        th = threading.Thread(
            target=_export_gguf,
            args=(quant, model_path, adapter_path or "", out_dir, args.py, buf, sem),
            daemon=True,
            name=f"export-{quant}",
        )
        th.start()
        threads.append(th)

    results = _collect(buffers, len(args.quants))

    for th in threads:
        th.join(timeout=5)

    # Speed bench (sequential, one GGUF at a time)
    if args.probes and Path(args.probes).exists():
        probe_lines = [l.strip() for l in Path(args.probes).read_text(encoding="utf-8").splitlines() if l.strip()]
        probes = []
        for line in probe_lines[: args.bench_count]:
            try:
                sample = json.loads(line)
                probes.append(sample.get("instruction", sample.get("prompt", line)))
            except json.JSONDecodeError:  # xray: ignore[PY-004]
                probes.append(line)

        print(f"\n--- Speed Bench ({args.bench_count} probes) ---")  # xray: ignore[PY-004]
        for r in results:
            if r["ok"] and r["gguf_path"]:
                bench = _bench_gguf(r["gguf_path"], probes, args.bench_count)  # xray: ignore[PY-004]
                r["bench"] = bench
                tok = bench.get("tok_per_sec")
                print(f"  {r['quant']}: {f'{tok} tok/s' if tok else bench.get('note', 'skipped')}")  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    # Summary table  # xray: ignore[PY-004]
    print(f"\n{'='*70}")  # xray: ignore[PY-004]
    print(f"  SLIM-DOWN RESULTS")  # xray: ignore[PY-004]
    print(f"{'='*70}")  # xray: ignore[PY-004]
    print(f"{'Quant':<12} {'Size (MB)':>10} {'Export (s)':>11} {'tok/s':>8} {'Status'}")  # xray: ignore[PY-004]
    print(f"{'-'*70}")  # xray: ignore[PY-004]
    for r in results:
        tok = r.get("bench", {}).get("tok_per_sec")  # xray: ignore[PY-004]
        tok_str = f"{tok}" if tok else "n/a"
        status = "OK" if r["ok"] else f"FAIL: {r.get('error', '')[:30]}"
        print(f"{r['quant']:<12} {r['size_mb']:>10.1f} {r['elapsed_sec']:>11.1f} {tok_str:>8} {status}")  # xray: ignore[PY-004]

    # Save results
    results_path = out_dir / "slim_down_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:  # xray: ignore[PY-004]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved: {results_path}")  # xray: ignore[PY-004]

    # Update registry if saves-tag provided
    if args.saves_tag:
        try:
            reg_path = Path("saves/student_registry.json")
            if reg_path.exists():
                reg = json.loads(reg_path.read_text(encoding="utf-8"))
                champion_file = Path(f"saves/{args.saves_tag}/champion.txt")
                if champion_file.exists():
                    champion = json.loads(champion_file.read_text(encoding="utf-8"))
                    mid = f"{args.saves_tag}/{champion['variant_id']}"
                    if mid in reg.get("students", {}):
                        reg["students"][mid]["gguf_variants"] = [
                            {"quant": r["quant"], "size_mb": r["size_mb"], "path": r["gguf_path"]}
                            for r in results
                            if r["ok"]  # xray: ignore[PY-004]
                        ]
                        reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")  # xray: ignore[PY-004]
                        print(f"Updated registry for {mid} with {len([r for r in results if r['ok']])} GGUF variants.")  # xray: ignore[PY-004]
        except Exception as exc:  # xray: ignore[QUAL-011]
            print(f"WARNING: Could not update registry: {exc}")  # xray: ignore[PY-004]

    # ── Pareto frontier analysis ─────────────────────────────────────────
    if args.recommend_quant:
        ok_results = [r for r in results if r["ok"] and r.get("bench", {}).get("tok_per_sec")]
        if len(ok_results) < 2:
            print("\nPareto analysis: need >=2 quants with successful benchmarks.")  # xray: ignore[PY-004]
        else:
            # Sort by size ascending → smaller is better
            ok_results.sort(key=lambda r: r["size_mb"])
            print(f"\n{'='*60}")  # xray: ignore[PY-004]
            print(f"  PARETO FRONTIER ANALYSIS (size vs speed)")
            print(f"{'='*60}")  # xray: ignore[PY-004]

            # Find Pareto-optimal points: no other point dominates on BOTH size and speed
            pareto: list[dict] = []
            for r in ok_results:
                dominated = False
                for p in ok_results:
                    if p is r:
                        continue
                    # p dominates r if p is <= size AND >= speed (with at least one strict)
                    p_speed = p["bench"]["tok_per_sec"]
                    r_speed = r["bench"]["tok_per_sec"]
                    if p["size_mb"] <= r["size_mb"] and p_speed >= r_speed:
                        if p["size_mb"] < r["size_mb"] or p_speed > r_speed:
                            dominated = True
                            break
                if not dominated:
                    pareto.append(r)

            print(f"\n  Pareto-optimal quants ({len(pareto)} of {len(ok_results)}):")  # xray: ignore[PY-004]
            for r in pareto:
                tok = r["bench"]["tok_per_sec"]
                print(f"    {r['quant']:<12} {r['size_mb']:>8.1f} MB  {tok:>6.1f} tok/s")  # xray: ignore[PY-004]

            # Recommend: best speed/size ratio among Pareto
            best = max(pareto, key=lambda r: r["bench"]["tok_per_sec"] / max(r["size_mb"], 0.1))
            print(f"\n  >>> RECOMMENDED: {best['quant']} (best speed/size ratio)")  # xray: ignore[PY-004]
            print(f"      Size: {best['size_mb']:.1f} MB, Speed: {best['bench']['tok_per_sec']:.1f} tok/s")  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
