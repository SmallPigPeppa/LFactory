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
# Embedding-based similarity (optional — requires sentence-transformers)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_AVAILABLE = False


def _load_embedding_model() -> bool:
    """Lazily load a lightweight sentence-transformer for semantic similarity."""
    global _EMBED_MODEL, _EMBED_AVAILABLE
    if _EMBED_MODEL is not None:
        return _EMBED_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _EMBED_AVAILABLE = True
    except Exception:  # xray: ignore[QUAL-011]
        _EMBED_MODEL = False  # type: ignore[assignment]
        _EMBED_AVAILABLE = False
    return _EMBED_AVAILABLE


def _embedding_similarity(texts: list[str]) -> float:
    """Compute average pairwise cosine similarity using sentence embeddings.

    Returns 0.0-1.0.  Falls back to 0.0 if model unavailable.
    """
    if not _EMBED_AVAILABLE or _EMBED_MODEL is False:
        return 0.0
    try:
        embeddings = _EMBED_MODEL.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        import numpy as np

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T
        n = len(texts)
        if n < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(sim_matrix[i, j])
                count += 1
        return total / count if count > 0 else 0.0
    except Exception:  # xray: ignore[QUAL-011]
        return 0.0


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
    """Normalize an answer for comparison: extract numeric values, strip common
    prefixes ("The answer is", "It's"), lowercase, strip whitespace/punctuation.

    Examples:
        "The answer is 4."  → "4"
        "4.0"               → "4.0"
        "Four"              → "four"
        "It's blue."        → "blue"
        "  Hello World!!! " → "hello world"
    """
    text = re.sub(r"\s+", " ", text.lower().strip())

    # Strip common answer prefixes
    for prefix in (
        "the answer is", "the result is", "it is", "it's", "i think",
        "i believe", "that would be", "the correct answer is",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip().lstrip(":")

    # Strip surrounding punctuation/quotes
    text = re.sub(r"^[\s\"'`]+|[\s\"'`.,;:!?]+$", "", text)
    # Remove remaining non-word chars (keep dots for decimals)
    text = re.sub(r"[^\w\s.]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def _find_majority_answer(
    teacher_data: dict[str, dict],
    threshold: float,
    weights: dict[str, float] | None = None,
) -> tuple[str | None, list[str], list[str]]:
    """Find the majority answer among teachers.

    Args:
        weights: optional name→weight mapping.  When provided, cluster
            weight is sum of member weights rather than member count.

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

    # Find largest cluster (by weight if provided, else by count)
    def _cluster_weight(c: list[str]) -> float:
        if weights:
            return sum(weights.get(n, 1.0) for n in c)
        return float(len(c))

    clusters.sort(key=_cluster_weight, reverse=True)
    largest = clusters[0]

    total_weight = sum(weights.get(n, 1.0) for n in teacher_names) if weights else float(len(teacher_names))
    majority_needed = total_weight / 2.0

    if _cluster_weight(largest) > majority_needed:
        agree = largest
        disagree = [n for n in teacher_names if n not in set(largest)]
        majority_answer = answers[largest[0]]
        return majority_answer, agree, disagree

    return None, [], teacher_names


def _check_reasoning_alignment(
    teacher_data: dict[str, dict],
    agreeing_teachers: list[str],
    threshold: float,
    use_embeddings: bool = False,
) -> bool:
    """Check if reasoning traces of agreeing teachers are sufficiently similar.

    When ``use_embeddings=True`` and ``sentence-transformers`` is installed,
    uses all-MiniLM-L6-v2 cosine similarity (semantic).  Otherwise falls back
    to SimHash character-level similarity (lexical).
    """
    thoughts = [teacher_data[t].get("thought", "") for t in agreeing_teachers]
    # Filter out empty thoughts
    thoughts = [t for t in thoughts if t.strip()]

    if len(thoughts) <= 1:
        # Can't compare — treat as aligned (benefit of the doubt)
        return True

    # Try embedding-based similarity first
    if use_embeddings and _load_embedding_model():
        avg_sim = _embedding_similarity(thoughts)
        return avg_sim >= threshold

    # Fallback: SimHash for O(1) per-pair similarity
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
    use_embeddings: bool = False,
    teacher_weights: dict[str, float] | None = None,
) -> tuple[str, dict]:
    """Classify a single sample as GOLD, SILVER, or DROP.

    Args:
        teacher_weights: optional name→weight mapping for weighted voting.
            Higher weight = more influence in majority detection.

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

    # Find majority answer (weighted if weights provided)
    majority_answer, agreeing, disagreeing = _find_majority_answer(
        teacher_data, answer_threshold, teacher_weights,
    )

    if majority_answer is None:
        # No majority — DROP
        return "DROP", {
            "id": sample_id,
            "prompt": prompt,
            "reason": "no_majority_answer",
            "teachers": {n: d.get("answer", "")[:200] for n, d in teacher_data.items()},
        }

    # Majority exists — check reasoning alignment
    reasoning_aligned = _check_reasoning_alignment(
        teacher_data, agreeing, reason_threshold, use_embeddings,
    )

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
        # chosen = majority answer (best response), rejected = DISAGREEING teacher.
        # If no disagreeing teachers exist (all agreed on answer but reasoning
        # diverged), demote to GOLD instead — using a reasoning outlier as
        # "rejected" risks inverted preferences.
        if disagreeing:
            # Pick the disagreeing teacher whose answer is most different
            reject_teacher = disagreeing[0]
        else:
            # All teachers agree on answer; reasoning differs but answer is correct.
            # Safer to emit as GOLD than to fabricate a flawed DPO pair.
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
                "note": "reasoning_divergent_but_answer_unanimous",
            }

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
# Auto-tune thresholds (#3)
# ---------------------------------------------------------------------------


def auto_tune_thresholds(
    samples: list[dict],
    target_gold_pct: float = 60.0,
    answer_range: tuple[float, float, float] = (0.70, 0.95, 0.05),
    reason_range: tuple[float, float, float] = (0.40, 0.80, 0.05),
) -> tuple[float, float]:
    """Sweep answer/reason thresholds to get closest to ``target_gold_pct``.

    Returns (best_answer_threshold, best_reason_threshold).
    """
    import itertools

    best: tuple[float, float] = (0.85, 0.60)
    best_delta = float("inf")

    a_lo, a_hi, a_step = answer_range
    r_lo, r_hi, r_step = reason_range

    def _frange(lo: float, hi: float, step: float) -> list[float]:
        vals: list[float] = []
        v = lo
        while v <= hi + 1e-9:
            vals.append(round(v, 4))
            v += step
        return vals

    for at, rt in itertools.product(_frange(a_lo, a_hi, a_step), _frange(r_lo, r_hi, r_step)):
        gold = sum(1 for s in samples if classify_sample(s, at, rt)[0] == "GOLD")
        pct = 100.0 * gold / max(len(samples), 1)
        delta = abs(pct - target_gold_pct)
        if delta < best_delta:
            best_delta = delta
            best = (at, rt)

    return best


# ---------------------------------------------------------------------------
# Synthetic DPO pair generation (#4)
# ---------------------------------------------------------------------------


def generate_synthetic_dpo(
    gold_samples: list[dict],
    max_pairs: int = 0,
) -> list[dict]:
    """Generate synthetic DPO pairs from GOLD samples by pairing highest-
    confidence chosen responses with lowest-confidence ones as rejected.

    Pairs are formed across different prompts sharing similar difficulty.
    ``max_pairs=0`` means unlimited (generates as many as possible).
    """
    if len(gold_samples) < 2:
        return []

    # Sort by confidence descending
    scored = sorted(gold_samples, key=lambda s: s.get("confidence", 1.0), reverse=True)
    top_half = scored[: len(scored) // 2]
    bottom_half = scored[len(scored) // 2 :]

    pairs: list[dict] = []
    for chosen, rejected in zip(top_half, bottom_half):
        if chosen.get("instruction") == rejected.get("instruction"):
            continue  # skip same-prompt pairs
        pair = {
            "prompt": chosen["instruction"],
            "chosen": chosen["output"],
            "rejected": rejected["output"],
            "id": f"synth_{chosen.get('id', '')}_{rejected.get('id', '')}",
            "tier": "SYNTHETIC_DPO",
            "chosen_confidence": chosen.get("confidence", 1.0),
            "rejected_confidence": rejected.get("confidence", 1.0),
        }
        pairs.append(pair)
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break

    return pairs


# ---------------------------------------------------------------------------
# Curriculum learning — sort by difficulty (#7)
# ---------------------------------------------------------------------------


def curriculum_sort(samples: list[dict], reverse: bool = False) -> list[dict]:
    """Sort samples by difficulty (easy → hard by default).

    ``reverse=True`` gives hard → easy (anti-curriculum).
    Each sample must have a ``difficulty`` field (0.0-1.0).
    Samples without difficulty are placed at the end.
    """
    return sorted(
        samples,
        key=lambda s: (s.get("difficulty") is None, s.get("difficulty", 1.0)),
        reverse=reverse,
    )


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


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file (for resume tracking)."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _append_jsonl(row: dict, path: Path) -> None:
    """Append a single JSON line to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Purify multi-teacher outputs into GOLD/SILVER/DROP tiers.",
        epilog="""\
examples:
  %(prog)s --input data/teacher_responses.jsonl --out-dir data/purified
  %(prog)s --input data/teacher_responses.jsonl --out-dir data/purified --resume
  %(prog)s --input data/teacher_responses.jsonl --out-dir data/purified --answer-threshold 0.9
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to teacher_responses.jsonl.")
    parser.add_argument("--out-dir", required=True, help="Output directory for purified data.")
    parser.add_argument("--answer-threshold", type=float, default=0.85, help="N-gram similarity threshold for answer agreement (default: 0.85).")
    parser.add_argument("--reason-threshold", type=float, default=0.6, help="N-gram similarity threshold for reasoning alignment (default: 0.6).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from where last run left off (skip already-processed samples).")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Use sentence-transformer embeddings for reasoning alignment (requires sentence-transformers).")
    parser.add_argument("--teacher-weights", type=str, default=None,
                        help='JSON string or file path mapping teacher names to weights, e.g. \'{"qwen": 1.5, "llama": 1.0}\'.')
    parser.add_argument("--auto-tune", action="store_true",
                        help="Auto-tune answer/reason thresholds to target ~60%% GOLD.")
    parser.add_argument("--auto-tune-target", type=float, default=60.0,
                        help="Target GOLD percentage for auto-tune (default: 60.0).")
    parser.add_argument("--synthetic-dpo", action="store_true",
                        help="Generate synthetic DPO pairs from GOLD samples.")
    parser.add_argument("--synthetic-dpo-max", type=int, default=0,
                        help="Maximum synthetic DPO pairs (0 = unlimited).")
    parser.add_argument("--curriculum", action="store_true",
                        help="Sort GOLD output by difficulty (easy → hard curriculum learning).")
    args = parser.parse_args()

    # Parse teacher weights
    teacher_weights: dict[str, float] | None = None
    if args.teacher_weights:
        tw = args.teacher_weights
        if Path(tw).is_file():
            teacher_weights = json.loads(Path(tw).read_text(encoding="utf-8"))
        else:
            teacher_weights = json.loads(tw)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)  # xray: ignore[PY-004]
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = out_dir / "consensus_sft.jsonl"
    silver_path = out_dir / "conflict_dpo.jsonl"
    drop_path = out_dir / "dropped_log.jsonl"

    samples = _load_jsonl(input_path)
    total = len(samples)

    # T1 — reconcile: count lines parsed vs file lines to detect silent data loss
    file_lines = sum(1 for _ in open(input_path, encoding="utf-8") if _.strip())
    if total < file_lines:
        skipped = file_lines - total
        print(f"WARNING: {skipped}/{file_lines} input lines were malformed JSON and skipped.", file=sys.stderr)
        print(f"  Parsed: {total}, File lines: {file_lines}", file=sys.stderr)

    print(f"Loaded {total} samples for purification.")  # xray: ignore[PY-004]

    # Auto-tune thresholds if requested
    answer_threshold = args.answer_threshold
    reason_threshold = args.reason_threshold
    if args.auto_tune:
        print(f"Auto-tuning thresholds (target GOLD {args.auto_tune_target:.0f}%)...")  # xray: ignore[PY-004]
        answer_threshold, reason_threshold = auto_tune_thresholds(
            samples, target_gold_pct=args.auto_tune_target,
        )
        print(f"  Selected: answer={answer_threshold:.2f} reason={reason_threshold:.2f}")  # xray: ignore[PY-004]

    # Resume support: count already-processed lines across all 3 output files
    skip = 0
    if args.resume:
        skip = _count_lines(gold_path) + _count_lines(silver_path) + _count_lines(drop_path)
        if skip > 0:
            print(f"Resuming: skipping {skip} already-processed samples.")  # xray: ignore[PY-004]
        if skip >= total:
            print("All samples already processed — nothing to do.")  # xray: ignore[PY-004]
    else:
        # Fresh run: truncate output files
        for p in (gold_path, silver_path, drop_path):
            if p.exists():
                p.write_text("", encoding="utf-8")

    gold_count = _count_lines(gold_path) if args.resume else 0
    silver_count = _count_lines(silver_path) if args.resume else 0
    drop_count = _count_lines(drop_path) if args.resume else 0

    for idx, sample in enumerate(samples):
        if idx < skip:
            continue

        tier, record = classify_sample(
            sample, answer_threshold, reason_threshold,
            use_embeddings=args.use_embeddings,
            teacher_weights=teacher_weights,
        )
        if tier == "GOLD":
            _append_jsonl(record, gold_path)
            gold_count += 1
        elif tier == "SILVER":
            _append_jsonl(record, silver_path)
            silver_count += 1
        else:
            _append_jsonl(record, drop_path)
            drop_count += 1

        # Progress every 500 samples
        processed = idx + 1
        if processed % 500 == 0 or processed == total:
            print(f"  [{processed}/{total}] G={gold_count} S={silver_count} D={drop_count}", flush=True)  # xray: ignore[PY-004]

    # Curriculum sort: re-order GOLD file by difficulty (easy → hard)
    if args.curriculum and gold_path.exists() and gold_count > 0:
        gold_rows = _load_jsonl(gold_path)
        gold_rows = curriculum_sort(gold_rows)
        _save_jsonl(gold_rows, gold_path)
        print(f"  Curriculum sort applied to {gold_count} GOLD samples (easy → hard).")  # xray: ignore[PY-004]

    # Synthetic DPO generation from GOLD samples
    synth_count = 0
    if args.synthetic_dpo and gold_path.exists() and gold_count >= 2:
        gold_rows = _load_jsonl(gold_path) if not args.curriculum else gold_rows  # reuse if loaded
        synth_pairs = generate_synthetic_dpo(gold_rows, max_pairs=args.synthetic_dpo_max)
        if synth_pairs:
            synth_path = out_dir / "synthetic_dpo.jsonl"
            _save_jsonl(synth_pairs, synth_path)
            synth_count = len(synth_pairs)
            print(f"  Generated {synth_count} synthetic DPO pairs → {synth_path}")  # xray: ignore[PY-004]

    # Save report with checksum for idempotent resume validation
    processed_total = gold_count + silver_count + drop_count
    report = {
        "total_samples": total,
        "total_processed": processed_total,
        "gold_count": gold_count,
        "silver_count": silver_count,
        "dropped_count": drop_count,
        "gold_pct": round(100.0 * gold_count / max(total, 1), 1),
        "silver_pct": round(100.0 * silver_count / max(total, 1), 1),
        "dropped_pct": round(100.0 * drop_count / max(total, 1), 1),
        "answer_threshold": answer_threshold,
        "reason_threshold": reason_threshold,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "use_embeddings": args.use_embeddings,
        "teacher_weights": teacher_weights,
        "auto_tuned": args.auto_tune,
        "curriculum_sorted": args.curriculum,
        "synthetic_dpo_count": synth_count,
    }

    # Integrity check: processed rows should match input rows
    if processed_total != total:
        print(f"WARNING: processed {processed_total} != input {total}. Possible data loss.", file=sys.stderr)
    report_path = out_dir / "purification_report.json"
    with report_path.open("w", encoding="utf-8") as f:  # xray: ignore[PY-004]
        json.dump(report, f, indent=2)  # xray: ignore[PY-004]
  # xray: ignore-next[PY-004]
    print(f"\n=== Purification Report ===")  # xray: ignore[PY-004]
    print(f"  GOLD  (SFT):  {gold_count:>4} ({report['gold_pct']:.1f}%)")  # xray: ignore[PY-004]
    print(f"  SILVER (DPO): {silver_count:>4} ({report['silver_pct']:.1f}%)")  # xray: ignore[PY-004]
    print(f"  DROP:         {drop_count:>4} ({report['dropped_pct']:.1f}%)")  # xray: ignore[PY-004]
    print(f"\nOutputs saved to {out_dir}/")  # xray: ignore[PY-004]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
