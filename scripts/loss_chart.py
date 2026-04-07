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

"""Loss Comparison Chart — overlay training curves from multiple forge variants.

Reads trainer_log.jsonl files from forge output directories and generates
a side-by-side text chart or HTML visualization of SFT and DPO loss curves.

Usage:
    python scripts/loss_chart.py --saves-tag zena007
    python scripts/loss_chart.py --saves-tag zena007 --html-out loss_chart.html
    python scripts/loss_chart.py --log-dirs saves/zena007/A/sft saves/zena007/B/sft
"""

from __future__ import annotations

import argparse
import json
from html import escape as _esc
from pathlib import Path


def _read_loss_curve(log_path: Path) -> list[dict]:
    """Read loss entries from a trainer_log.jsonl file.

    Returns list of {step, epoch, loss} dicts.
    """
    entries: list[dict] = []
    if not log_path.exists():
        return entries
    for line in log_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if "loss" in entry:
                entries.append({
                    "step": entry.get("current_steps", entry.get("step", len(entries))),
                    "epoch": entry.get("epoch", 0),
                    "loss": float(entry["loss"]),
                })
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return entries


def _discover_variants(saves_tag: str) -> dict[str, dict[str, Path]]:
    """Discover variant log files from saves/<tag>/<variant>/<stage>/trainer_log.jsonl."""
    saves_dir = Path(f"saves/{saves_tag}")
    variants: dict[str, dict[str, Path]] = {}

    if not saves_dir.exists():
        return variants

    for variant_dir in sorted(saves_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        vid = variant_dir.name
        if vid in ("merged", "forge_state.json"):
            continue

        logs: dict[str, Path] = {}
        for stage in ("sft", "dpo"):
            log_path = variant_dir / stage / "trainer_log.jsonl"
            if log_path.exists():
                logs[stage] = log_path
        if logs:
            variants[vid] = logs

    return variants


def predict_convergence(
    entries: list[dict],
    target_loss: float | None = None,
    window: int = 5,
) -> dict:
    """Predict when training will converge based on recent loss trend.

    Fits an exponential decay L(t) = a*exp(-b*t) + c to the last `window`
    points and extrapolates to the target loss (or to the asymptote c).

    Falls back to linear fit if exponential fit fails (e.g., too few points
    or non-monotonic data).

    Returns dict with prediction details.
    """
    import math

    if len(entries) < 3:
        return {"converged": False, "reason": "too_few_points", "points": len(entries)}

    recent = entries[-window:]
    losses = [e["loss"] for e in recent]
    steps = [e["step"] for e in recent]

    if steps[-1] == steps[0]:
        return {"converged": True, "reason": "single_step"}

    current_loss = losses[-1]
    current_step = steps[-1]

    # --- Attempt exponential decay fit: L(t) = asymptote + amplitude * exp(-rate * t) ---
    # Use first/last/min to estimate parameters
    asymptote_est = min(losses)  # floor estimate
    amplitude_est = losses[0] - asymptote_est
    exp_fit_ok = False
    exp_rate = 0.0

    if amplitude_est > 1e-6 and len(losses) >= 3:
        # Estimate decay rate from log-space linear regression on (L - asymptote)
        try:
            log_residuals = []
            log_steps = []
            for s, l in zip(steps, losses):
                residual = l - asymptote_est
                if residual > 1e-8:
                    log_residuals.append(math.log(residual))
                    log_steps.append(s)

            if len(log_steps) >= 2:
                n = len(log_steps)
                sx = sum(log_steps)
                sy = sum(log_residuals)
                sxy = sum(x * y for x, y in zip(log_steps, log_residuals))
                sx2 = sum(x * x for x in log_steps)
                denom = n * sx2 - sx * sx
                if abs(denom) > 1e-12:
                    exp_rate = -(n * sxy - sx * sy) / denom  # negative slope = positive rate
                    exp_fit_ok = exp_rate > 0  # only valid if decaying
        except (ValueError, OverflowError):
            exp_fit_ok = False

    result: dict = {
        "current_loss": round(current_loss, 6),
        "current_step": current_step,
        "fit_type": "exponential" if exp_fit_ok else "linear",
    }

    if exp_fit_ok:
        # Exponential model: loss approaching asymptote
        result["asymptote"] = round(asymptote_est, 6)
        result["decay_rate"] = round(exp_rate, 8)

        gap = current_loss - asymptote_est
        if gap < 0.001:
            result["converged"] = True
            result["reason"] = "near_asymptote"
            result["final_loss"] = round(current_loss, 6)
            return result

        if target_loss is not None and target_loss > asymptote_est:
            target_gap = target_loss - asymptote_est
            if target_gap > 0 and gap > target_gap:
                # steps_needed = ln(gap/target_gap) / rate
                steps_needed = math.log(gap / target_gap) / exp_rate
                result["target_loss"] = target_loss
                result["estimated_steps_remaining"] = max(0, int(steps_needed))
                result["converged"] = False
                result["reason"] = "improving_exp"
            else:
                result["converged"] = True
                result["reason"] = "target_reached"
                result["final_loss"] = round(current_loss, 6)
        else:
            # Estimate steps to near-asymptote (gap < 0.001)
            if gap > 0.001:
                steps_to_floor = math.log(gap / 0.001) / exp_rate
                result["estimated_steps_to_floor"] = max(0, int(steps_to_floor))
            result["converged"] = False
            result["reason"] = "improving_exp"
        return result

    # --- Fallback: linear fit ---
    n = len(steps)
    sum_x = sum(steps)
    sum_y = sum(losses)
    sum_xy = sum(x * y for x, y in zip(steps, losses))
    sum_x2 = sum(x * x for x in steps)
    denom = n * sum_x2 - sum_x * sum_x

    if abs(denom) < 1e-12:
        return {"converged": True, "reason": "flat_loss", "final_loss": round(losses[-1], 6)}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    result["slope"] = round(slope, 8)
    result["improvement_rate"] = round(-slope, 8)

    if abs(slope) < 1e-6:
        result["converged"] = True
        result["reason"] = "slope_near_zero"
        result["final_loss"] = round(current_loss, 6)
        return result

    if slope > 0:
        result["converged"] = False
        result["reason"] = "diverging"
        return result

    if target_loss is not None and target_loss < current_loss:
        est_steps = (target_loss - intercept) / slope
        remaining = max(0, est_steps - current_step)
        result["target_loss"] = target_loss
        result["estimated_steps_remaining"] = int(remaining)
        result["converged"] = False
        result["reason"] = "improving"
    else:
        result["converged"] = False
        result["reason"] = "improving"

    return result


def build_text_chart(
    curves: dict[str, list[dict]],
    width: int = 60,
    height: int = 15,
) -> str:
    """Build a simple text-based loss chart."""
    if not curves:
        return "No data."

    # Collect all loss values for scale
    all_losses = [e["loss"] for entries in curves.values() for e in entries]
    if not all_losses:
        return "No loss data found."

    min_loss = min(all_losses)
    max_loss = max(all_losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    lines: list[str] = []
    lines.append(f"  Loss range: {min_loss:.4f} — {max_loss:.4f}")
    lines.append("")

    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    legend = []
    for i, (name, entries) in enumerate(curves.items()):
        sym = symbols[i % len(symbols)]
        legend.append(f"  {sym} = {name} ({len(entries)} pts, final={entries[-1]['loss']:.4f})" if entries else f"  {sym} = {name}")

    # Build chart grid
    grid = [[" "] * width for _ in range(height)]

    for i, (name, entries) in enumerate(curves.items()):
        if not entries:
            continue
        sym = symbols[i % len(symbols)]
        max_step = max(e["step"] for e in entries) or 1

        for e in entries:
            x = int((e["step"] / max_step) * (width - 1))
            y = int(((e["loss"] - min_loss) / loss_range) * (height - 1))
            y = height - 1 - y  # invert y axis
            x = min(x, width - 1)
            y = min(max(y, 0), height - 1)
            grid[y][x] = sym

    # Render
    for row_idx, row in enumerate(grid):
        loss_label = max_loss - (row_idx / max(height - 1, 1)) * loss_range
        lines.append(f"  {loss_label:7.4f} |{''.join(row)}|")

    lines.append(f"          {'─' * width}")
    lines.append(f"          step →")
    lines.append("")
    lines.extend(legend)

    return "\n".join(lines)


def build_html_chart(curves: dict[str, list[dict]]) -> str:
    """Build a self-contained HTML chart using inline SVG."""
    w, h = 700, 350
    margin = 60

    if not curves:
        return "<p>No data.</p>"

    all_losses = [e["loss"] for entries in curves.values() for e in entries]
    all_steps = [e["step"] for entries in curves.values() for e in entries]
    if not all_losses or not all_steps:
        return "<p>No loss data.</p>"

    min_loss, max_loss = min(all_losses), max(all_losses)
    max_step = max(all_steps) or 1
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]

    svg_lines = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:system-ui,sans-serif;background:#fff;border:1px solid #e5e5e5;border-radius:8px">',
    ]

    # Y-axis labels
    for i in range(5):
        y = margin + (h - 2 * margin) * i / 4
        loss_val = max_loss - (loss_range * i / 4)
        svg_lines.append(
            f'<text x="{margin - 5}" y="{y + 4}" text-anchor="end" font-size="11" fill="#86868b">{loss_val:.3f}</text>'
        )
        svg_lines.append(
            f'<line x1="{margin}" y1="{y}" x2="{w - 20}" y2="{y}" stroke="#f0f0f0" stroke-width="1"/>'
        )

    # Plot lines
    legend_y = h - 15
    for idx, (name, entries) in enumerate(curves.items()):
        if not entries:
            continue
        color = colors[idx % len(colors)]
        points = []
        for e in entries:
            x = margin + (e["step"] / max_step) * (w - margin - 20)
            y = margin + ((max_loss - e["loss"]) / loss_range) * (h - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        svg_lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
        # Legend
        lx = margin + idx * 120
        svg_lines.append(f'<rect x="{lx}" y="{legend_y - 8}" width="12" height="12" rx="2" fill="{color}"/>')
        svg_lines.append(
            f'<text x="{lx + 16}" y="{legend_y + 2}" font-size="12" fill="#1d1d1f">{_esc(name)}</text>'
        )

    svg_lines.append("</svg>")
    svg = "\n".join(svg_lines)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Loss Chart</title></head>
<body style="font-family:system-ui;max-width:800px;margin:40px auto;padding:0 20px">
<h2>Training Loss Comparison</h2>
{svg}
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loss comparison chart for forge variants.",
        epilog="""\
examples:
  %(prog)s --saves-tag zena007
  %(prog)s --saves-tag zena007 --html-out loss_chart.html
  %(prog)s --log-dirs saves/zena007/A/sft saves/zena007/B/sft --labels A B
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--saves-tag", help="Tag from forge pipeline (auto-discovers variants).")
    parser.add_argument("--log-dirs", nargs="+", help="Explicit trainer_log.jsonl parent directories.")
    parser.add_argument("--labels", nargs="+", help="Labels for --log-dirs (default: directory names).")
    parser.add_argument("--html-out", help="Write HTML chart to this path.")
    parser.add_argument("--stage", choices=["sft", "dpo", "all"], default="all",
                        help="Which training stage to plot (default: all).")
    parser.add_argument("--predict", action="store_true",
                        help="Show convergence prediction for each curve.")
    parser.add_argument("--target-loss", type=float, default=None,
                        help="Target loss for convergence prediction.")
    args = parser.parse_args()

    curves: dict[str, list[dict]] = {}

    if args.saves_tag:
        variants = _discover_variants(args.saves_tag)
        if not variants:
            print(f"No training logs found in saves/{args.saves_tag}/")  # xray: ignore[PY-004]
            return 1
        for vid, logs in variants.items():
            for stage, log_path in logs.items():
                if args.stage != "all" and stage != args.stage:
                    continue
                label = f"{vid}/{stage}"
                curves[label] = _read_loss_curve(log_path)

    elif args.log_dirs:
        labels = args.labels or [Path(d).name for d in args.log_dirs]
        for log_dir, label in zip(args.log_dirs, labels):
            log_path = Path(log_dir) / "trainer_log.jsonl"
            if not log_path.exists():
                log_path = Path(log_dir)
            curves[label] = _read_loss_curve(log_path)
    else:
        parser.error("Provide --saves-tag or --log-dirs")

    if not curves:
        print("No loss data found.")  # xray: ignore[PY-004]
        return 1

    # Text chart to stdout
    print(f"\n=== Loss Comparison ===\n")  # xray: ignore[PY-004]
    print(build_text_chart(curves))  # xray: ignore[PY-004]

    # Convergence prediction
    if args.predict:
        print(f"\n=== Convergence Prediction ===\n")  # xray: ignore[PY-004]
        for name, entries in curves.items():
            pred = predict_convergence(entries, target_loss=args.target_loss)
            status = pred.get("reason", "unknown")
            loss = pred.get("current_loss", "?")
            slope = pred.get("slope", 0)
            print(f"  {name}: loss={loss}, slope={slope:+.6f}, status={status}")  # xray: ignore[PY-004]
            if "estimated_steps_remaining" in pred:
                print(f"    → ~{pred['estimated_steps_remaining']} steps to target {pred['target_loss']}")  # xray: ignore[PY-004]

    # Optional HTML export
    if args.html_out:
        html = build_html_chart(curves)
        out = Path(args.html_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"\nHTML chart saved to {out}")  # xray: ignore[PY-004]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
