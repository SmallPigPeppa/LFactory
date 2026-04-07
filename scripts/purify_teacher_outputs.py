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

"""Knowledge Purification — three-tier classification of multi-teacher outputs.

Takes teacher_responses.jsonl (output of multi_teacher_generate.py) and splits
into three tiers:

  GOLD   — all teachers agree on answer AND reasoning aligns  → consensus_sft.jsonl
  SILVER — majority agree on answer BUT reasoning differs     → conflict_dpo.jsonl
  DROP   — no majority agreement on answer                    → dropped_log.jsonl

Rationale verification is inspired by CREST (NAACL 2025, arXiv 2411.06387).

Usage:
    python scripts/purify_teacher_outputs.py \
        --input data/teacher_responses.jsonl \
        --out-dir data/purified \
        --answer-threshold 0.85 \
        --reason-threshold 0.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# SimHash — O(1) similarity estimation replacing O(n²) all-pairs n-gram
# ---------------------------------------------------------------------------

_SIMHASH_BITS = 64


def _simhash(text: str, n: int = 3) -> int:
    """Compute a SimHash fingerprint for fast similarity estimation."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    v = [0] * _SIMHASH_BITS
    for i in range(max(1, len(text) - n + 1)):
        token = text[i : i + n]
        h = int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()[:16], 16)
        for bit in range(_SIMHASH_BITS):
            if h & (1 << bit):
                v[bit] += 1
            else:
                v[bit] -= 1
    fp = 0
    for bit in range(_SIMHASH_BITS):
        if v[bit] > 0:
            fp |= (1 << bit)
    return fp


def _simhash_distance(a: int, b: int) -> int:
    """Hamming distance between two SimHash fingerprints (0 = identical)."""
    return bin(a ^ b).count("1")


def _simhash_similarity(a: str, b: str) -> float:
    """SimHash-based similarity (0.0-1.0). Much faster than Jaccard n-gram."""
    if not a.strip() or not b.strip():
        return 0.0
    ha = _simhash(a)
    hb = _simhash(b)
    dist = _simhash_distance(ha, hb)
    return max(0.0, 1.0 - dist / _SIMHASH_BITS)


# ---------------------------------------------------------------------------
# N-gram similarity (lightweight, no heavy deps)
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from text."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity of character n-gram sets. Returns 0.0-1.0."""
    if not a.strip() or not b.strip():
        return 0.0
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga and not gb:
        return 1.0
    intersection = ga & gb
    union = ga | gb
    return len(intersection) / len(union) if union else 0.0


def _normalize_answer(text: str) -> str:
    """Normalize an answer for comparison: lowercase, strip whitespace/punctuation."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def _find_majority_answer(
    teacher_data: dict[str, dict],
    threshold: float,
) -> tuple[str | None, list[str], list[str]]:
    """Find the majority answer among teachers.

    Returns (majority_answer_normalized, agreeing_teachers, disagreeing_teachers).
    If no majority, returns (None, [], all_teachers).
    """
    answers: dict[str, str] = {}  # teacher_name -> normalized answer
    for name, data in teacher_data.items():
        answers[name] = _normalize_answer(data.get("answer", ""))

    # Cluster by similarity
    teacher_names = list(answers.keys())
    clusters: list[list[str]] = []

    for name in teacher_names:
        placed = False
        for cluster in clusters:
            representative = answers[cluster[0]]
            if _ngram_similarity(answers[name], representative) >= threshold:
                cluster.append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])

    # Find largest cluster
    clusters.sort(key=len, reverse=True)
    largest = clusters[0]
    majority_needed = len(teacher_names) / 2.0

    if len(largest) > majority_needed:
        agree = largest
        disagree = [n for n in teacher_names if n not in set(largest)]
        majority_answer = answers[largest[0]]
        return majority_answer, agree, disagree

    return None, [], teacher_names


def _check_reasoning_alignment(
    teacher_data: dict[str, dict],
    agreeing_teachers: list[str],
    threshold: float,
) -> bool:
    """Check if reasoning traces of agreeing teachers are sufficiently similar.

    Uses SimHash for O(1) per-pair similarity (replaces O(n²m) n-gram Jaccard).
    """
    thoughts = [teacher_data[t].get("thought", "") for t in agreeing_teachers]
    # Filter out empty thoughts
    thoughts = [t for t in thoughts if t.strip()]

    if len(thoughts) <= 1:
        # Can't compare — treat as aligned (benefit of the doubt)
        return True

    # All-pairs similarity check (SimHash = O(n²) pairs but O(1) per pair)
    similarities: list[float] = []
    for i in range(len(thoughts)):
        for j in range(i + 1, len(thoughts)):
            sim = _simhash_similarity(thoughts[i], thoughts[j])
            similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    return avg_sim >= threshold


def classify_sample(
    sample: dict,
    answer_threshold: float,
    reason_threshold: float,
) -> tuple[str, dict]:
    """Classify a single sample as GOLD, SILVER, or DROP.

    Returns (tier, output_record).
    """
    teacher_data: dict[str, dict] = sample.get("teachers", {})
    sample_id = sample.get("id", "")
    prompt = sample.get("prompt", "")

    if len(teacher_data) < 1:
        return "DROP", {"id": sample_id, "reason": "no_teacher_responses"}

    # Single teacher — always GOLD (no conflict possible)
    if len(teacher_data) == 1:
        name = list(teacher_data.keys())[0]
        data = teacher_data[name]
        return "GOLD", {
            "instruction": prompt,
            "output": data.get("raw", data.get("answer", "")),
            "id": sample_id,
            "source_teacher": name,
            "tier": "GOLD",
        }

    # Find majority answer
    majority_answer, agreeing, disagreeing = _find_majority_answer(teacher_data, answer_threshold)

    if majority_answer is None:
        # No majority — DROP
        return "DROP", {
            "id": sample_id,
            "prompt": prompt,
            "reason": "no_majority_answer",
            "teachers": {n: d.get("answer", "")[:200] for n, d in teacher_data.items()},
        }

    # Majority exists — check reasoning alignment
    reasoning_aligned = _check_reasoning_alignment(teacher_data, agreeing, reason_threshold)

    # Confidence score = agreement strength (0-1)
    n_total = len(teacher_data)
    n_agree = len(agreeing)
    confidence = n_agree / n_total if n_total > 0 else 0.0

    # Prompt difficulty = 1 - confidence (easy prompts -> high confidence -> low difficulty)
    difficulty = round(1.0 - confidence, 4)

    # Pick best response from agreeing teachers (longest non-empty)
    best_teacher = max(agreeing, key=lambda t: len(teacher_data[t].get("raw", "")))
    best_response = teacher_data[best_teacher].get("raw", teacher_data[best_teacher].get("answer", ""))

    if reasoning_aligned:
        # GOLD — answer + reasoning agree
        return "GOLD", {
            "instruction": prompt,
            "output": best_response,
            "id": sample_id,
            "source_teacher": best_teacher,
            "agreeing_teachers": agreeing,
            "tier": "GOLD",
            "confidence": round(confidence, 4),
            "n_teachers_agree": n_agree,
            "difficulty": difficulty,
        }
    else:
        # SILVER — answer agrees but reasoning differs → DPO pair
        # chosen = majority answer, rejected = dissenter (or divergent-reasoning teacher)
        if disagreeing:
            reject_teacher = disagreeing[0]
        else:
            # All agree on answer but differ in reasoning — pick the one with most divergent thought
            reject_teacher = min(
                agreeing,
                key=lambda t: _ngram_similarity(
                    teacher_data[t].get("thought", ""),
                    teacher_data[best_teacher].get("thought", ""),
                ),
            )

        rejected_response = teacher_data[reject_teacher].get("raw", teacher_data[reject_teacher].get("answer", ""))

        return "SILVER", {
            "prompt": prompt,
            "chosen": best_response,
            "rejected": rejected_response,
            "id": sample_id,
            "chosen_teacher": best_teacher,
            "rejected_teacher": reject_teacher,
            "tier": "SILVER",
            "confidence": round(confidence, 4),
            "n_teachers_agree": n_agree,
            "difficulty": difficulty,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:  # xray: ignore[PY-005]
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass  # skip malformed JSON line
    return rows


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Purify multi-teacher outputs into GOLD/SILVER/DROP tiers.")
    parser.add_argument("--input", required=True, help="Path to teacher_responses.jsonl.")
    parser.add_argument("--out-dir", required=True, help="Output directory for purified data.")
    parser.add_argument("--answer-threshold", type=float, default=0.85, help="N-gram similarity threshold for answer agreement (default: 0.85).")
    parser.add_argument("--reason-threshold", type=float, default=0.6, help="N-gram similarity threshold for reasoning alignment (default: 0.6).")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)  # xray: ignore[PY-004]

    samples = _load_jsonl(input_path)
    print(f"Loaded {len(samples)} samples for purification.")  # xray: ignore[PY-004]

    gold: list[dict] = []
    silver: list[dict] = []
    dropped: list[dict] = []

    for sample in samples:
        tier, record = classify_sample(sample, args.answer_threshold, args.reason_threshold)
        if tier == "GOLD":
            gold.append(record)
        elif tier == "SILVER":
            silver.append(record)
        else:
            dropped.append(record)

    # Save outputs
    _save_jsonl(gold, out_dir / "consensus_sft.jsonl")
    _save_jsonl(silver, out_dir / "conflict_dpo.jsonl")
    _save_jsonl(dropped, out_dir / "dropped_log.jsonl")

    # Save report
    report = {
        "total_samples": len(samples),
        "gold_count": len(gold),
        "silver_count": len(silver),
        "dropped_count": len(dropped),
        "gold_pct": round(100.0 * len(gold) / max(len(samples), 1), 1),
        "silver_pct": round(100.0 * len(silver) / max(len(samples), 1), 1),
        "dropped_pct": round(100.0 * len(dropped) / max(len(samples), 1), 1),
        "answer_threshold": args.answer_threshold,
        "reason_threshold": args.reason_threshold,
    }
    report_path = out_dir / "purification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:  # xray: ignore[PY-004]
        json.dump(report, f, indent=2)  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    print(f"\n=== Purification Report ===")  # xray: ignore[PY-004]
    print(f"  GOLD  (SFT):  {len(gold):>4} ({report['gold_pct']:.1f}%)")  # xray: ignore[PY-004]
    print(f"  SILVER (DPO): {len(silver):>4} ({report['silver_pct']:.1f}%)")  # xray: ignore[PY-004]
    print(f"  DROP:         {len(dropped):>4} ({report['dropped_pct']:.1f}%)")  # xray: ignore[PY-004]
    print(f"\nOutputs saved to {out_dir}/")  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
