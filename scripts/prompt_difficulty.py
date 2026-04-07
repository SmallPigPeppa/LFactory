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

"""Prompt Difficulty Scoring — score and filter datasets by teacher disagreement.

Uses teacher response data (from multi_teacher_generate.py output or purified data
with difficulty scores) to:
  1. Compute per-prompt difficulty (teacher variance / disagreement)
  2. Histogram of difficulty distribution
  3. Filter: remove easy prompts (all teachers agree trivially) to focus on hard ones
  4. Optionally output a difficulty-reweighted dataset

Difficulty = 1.0 - (n_agreeing_teachers / n_total_teachers)
  0.0 = trivially easy (100% agreement)
  1.0 = maximally hard (no agreement)

Usage:
    # Score raw teacher responses
    python scripts/prompt_difficulty.py \
        --teacher-responses data/zena007/teacher_responses.jsonl \
        --out-scored data/zena007/scored_prompts.jsonl

    # Filter purified SFT data by difficulty
    python scripts/prompt_difficulty.py \
        --sft-data data/zena007/purified/consensus_sft.jsonl \
        --min-difficulty 0.1 \
        --out-filtered data/zena007/purified/hard_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
    return rows


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_teacher_responses(rows: list[dict]) -> list[dict]:
    """Compute difficulty for each prompt from raw teacher_responses.jsonl.

    Each row has: {id, prompt, teachers: {name: {answer, thought, raw, ...}}}
    Difficulty = 1.0 - (max_cluster_size / total_teachers)
    """
    scored: list[dict] = []
    for row in rows:
        teachers = row.get("teachers", {})
        n_total = len(teachers)
        if n_total == 0:
            difficulty = 1.0
        elif n_total == 1:
            difficulty = 0.0
        else:
            # Simple clustering: normalize answers and count groups
            answers = []
            for t_data in teachers.values():
                ans = t_data.get("answer", "").strip().lower()[:200]
                answers.append(ans)
            counts = Counter(answers)
            max_agree = counts.most_common(1)[0][1] if counts else 0
            difficulty = round(1.0 - max_agree / n_total, 4)

        scored.append({
            "id": row.get("id", ""),
            "prompt": row.get("prompt", ""),
            "difficulty": difficulty,
            "n_teachers": n_total,
            "n_max_agree": n_total - int(n_total * difficulty),
        })

    return scored


def filter_by_difficulty(
    rows: list[dict],
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0,
) -> list[dict]:
    """Filter rows that have a 'difficulty' field."""
    return [r for r in rows if min_difficulty <= r.get("difficulty", 0.5) <= max_difficulty]


def print_histogram(scored: list[dict], bins: int = 10) -> None:
    """Print ASCII histogram of difficulty distribution."""
    if not scored:
        print("No data to histogram.")
        return

    difficulties = [r.get("difficulty", 0.5) for r in scored]
    bin_width = 1.0 / bins
    bin_counts = [0] * bins

    for d in difficulties:
        idx = min(int(d / bin_width), bins - 1)
        bin_counts[idx] += 1

    max_count = max(bin_counts) if bin_counts else 1
    bar_max = 40

    print(f"\nDifficulty Distribution ({len(scored)} prompts):")
    print(f"{'Range':>12} {'Count':>6}  Bar")
    print("-" * 65)
    for i, count in enumerate(bin_counts):
        lo = i * bin_width
        hi = lo + bin_width
        bar_len = int(count / max_count * bar_max) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{lo:.1f}-{hi:.1f}) {count:>6}  {bar}")

    avg_d = sum(difficulties) / len(difficulties) if difficulties else 0
    print(f"\n  Mean difficulty: {avg_d:.3f}")
    print(f"  Easy (<0.1):     {sum(1 for d in difficulties if d < 0.1)}")
    print(f"  Medium (0.1-0.5):{sum(1 for d in difficulties if 0.1 <= d < 0.5)}")
    print(f"  Hard (>=0.5):    {sum(1 for d in difficulties if d >= 0.5)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prompt difficulty scoring and filtering.",
        epilog="""\
examples:
  %(prog)s --teacher-responses data/teacher_responses.jsonl --out-scored scored.jsonl --histogram
  %(prog)s --sft-data data/purified/consensus_sft.jsonl --min-difficulty 0.2 --out-filtered hard_only.jsonl
  %(prog)s --teacher-responses data/teacher_responses.jsonl --histogram
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--teacher-responses", help="Raw teacher_responses.jsonl to score.")
    parser.add_argument("--sft-data", help="Purified SFT data with difficulty fields to filter.")
    parser.add_argument("--min-difficulty", type=float, default=0.0, help="Minimum difficulty threshold for filtering.")
    parser.add_argument("--max-difficulty", type=float, default=1.0, help="Maximum difficulty threshold for filtering.")
    parser.add_argument("--out-scored", help="Output path for scored prompts.")
    parser.add_argument("--out-filtered", help="Output path for filtered data.")
    parser.add_argument("--histogram", action="store_true", help="Print difficulty histogram.")
    args = parser.parse_args()

    if not args.teacher_responses and not args.sft_data:
        parser.error("Provide --teacher-responses or --sft-data")

    if args.teacher_responses:
        path = Path(args.teacher_responses)
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1
        rows = _load_jsonl(path)
        scored = score_teacher_responses(rows)
        print(f"Scored {len(scored)} prompts from teacher responses.")

        if args.histogram:
            print_histogram(scored)

        if args.out_scored:
            _save_jsonl(scored, Path(args.out_scored))
            print(f"Saved scored prompts to {args.out_scored}")

    if args.sft_data:
        path = Path(args.sft_data)
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1
        rows = _load_jsonl(path)

        if args.histogram:
            print_histogram(rows)

        filtered = filter_by_difficulty(rows, args.min_difficulty, args.max_difficulty)
        print(f"Filtered: {len(filtered)}/{len(rows)} samples (difficulty [{args.min_difficulty}, {args.max_difficulty}])")

        if args.out_filtered:
            _save_jsonl(filtered, Path(args.out_filtered))
            print(f"Saved filtered data to {args.out_filtered}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
