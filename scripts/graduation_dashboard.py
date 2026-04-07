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

"""Graduation Dashboard — one glance, one decision.

Usage:
    python scripts/graduation_dashboard.py --report saves/zena007/graduation_report.json
    python scripts/graduation_dashboard.py --saves-tag zena007
"""

from __future__ import annotations

import argparse
import json
import math
from html import escape as _esc
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:  # xray: ignore[QUAL-002]
    raise SystemExit("graduation_dashboard requires gradio: pip install gradio")

# ── CSS ──────────────────────────────────────────────────────────────────

_CSS = """
.dash{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display",system-ui,sans-serif;
 max-width:720px;margin:0 auto;color:#1d1d1f}
.verdict{text-align:center;padding:48px 0 32px}
.verdict svg{display:block;margin:0 auto 16px}
.verdict .label{font-size:14px;letter-spacing:.08em;text-transform:uppercase;
 font-weight:600;margin:0}
.verdict .pct{font-size:64px;font-weight:700;letter-spacing:-.03em;
 line-height:1;margin:4px 0 0}
.alert{border-radius:8px;padding:12px 16px;margin:0 0 24px;font-size:13px;
 line-height:1.5}
.alert-ruin{background:#fee2e2;color:#991b1b}
.alert-emerge{background:#ede9fe;color:#5b21b6}
table.grid{width:100%;border-collapse:collapse;font-size:14px;margin:0 0 24px}
table.grid th{text-align:left;font-weight:600;padding:8px 12px;
 border-bottom:2px solid #e5e5e5;font-size:12px;letter-spacing:.04em;
 text-transform:uppercase;color:#86868b}
table.grid td{padding:8px 12px;border-bottom:1px solid #f0f0f0}
table.grid tr:last-child td{border-bottom:none}
.chip{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;
 font-weight:600;letter-spacing:.02em}
.chip-pass{background:#dcfce7;color:#166534}
.chip-fail{background:#fee2e2;color:#991b1b}
.cell-val{font-variant-numeric:tabular-nums}
.cell-weak{color:#dc2626;font-weight:600}
.cell-low{color:#9ca3af;font-style:italic}
.cell-good{color:#1d1d1f}
.cell-great{color:#16a34a;font-weight:600}
.foot{color:#86868b;font-size:12px;text-align:center;padding:16px 0}
"""


def _ring_svg(pct: float, color: str, size: int = 120, stroke: int = 8) -> str:
    """Minimal SVG ring showing a percentage."""
    r = (size - stroke) / 2
    c = math.pi * 2 * r
    offset = c * (1 - pct / 100)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{size // 2}" cy="{size // 2}" r="{r}" fill="none" '
        f'stroke="#f0f0f0" stroke-width="{stroke}"/>'
        f'<circle cx="{size // 2}" cy="{size // 2}" r="{r}" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{offset:.1f}" '
        f'transform="rotate(-90 {size // 2} {size // 2})"/>'
        f"</svg>"
    )


def _build_html(report: dict[str, Any]) -> str:
    """One HTML string.  One screen.  One decision."""
    threshold = report.get("threshold", 0.85)
    students = report.get("students", [])
    ruin_threshold = report.get("ruin_threshold", 0.50)

    # ── verdict ──────────────────────────────────────────────────────────
    n_grad = sum(1 for s in students if s.get("graduated"))
    n_total = len(students)

    if n_total == 0:
        return '<div class="dash"><p style="text-align:center;color:#86868b;padding:64px 0">No results.</p></div>'

    best = max(students, key=lambda s: s.get("overall_retention", 0))
    best_pct = round(best.get("overall_retention", 0) * 100)

    if n_grad == n_total:
        color, label = "#16a34a", "Ship it"
    elif n_grad > 0:
        color, label = "#d97706", "Review"
    else:
        color, label = "#dc2626", "Kill it"

    html = f'<div class="dash"><style>{_CSS}</style>'
    html += f'<div class="verdict">'
    html += _ring_svg(min(best_pct, 100), color)
    html += f'<p class="pct" style="color:{color}">{best_pct}%</p>'
    html += f'<p class="label" style="color:{color}">{label}</p>'
    html += f'<p style="color:#86868b;font-size:13px;margin:4px 0 0">'
    html += f'{n_grad}/{n_total} graduated &middot; threshold {threshold:.0%}</p>'
    html += "</div>"

    # ── ruin alert (only if triggered) ───────────────────────────────────
    ruin_lines: list[str] = []
    for s in students:
        for cat, info in s.get("retention_detail", {}).items():  # xray: ignore[QUAL-005]
            if not info.get("low_confidence") and info.get("value", 1.0) < ruin_threshold:
                ruin_lines.append(f"<b>{_esc(s['variant_id'])}</b> {_esc(cat)} {info['value']:.0%}")
    if ruin_lines:
        html += f'<div class="alert alert-ruin">Ruin: {" &middot; ".join(ruin_lines)}</div>'

    # ── emergence alert (only if triggered) ──────────────────────────────
    emerge_lines: list[str] = []
    for s in students:
        for cat in s.get("emergent_categories", []):
            emerge_lines.append(f"<b>{_esc(s['variant_id'])}</b> {_esc(cat)}")
    if emerge_lines:
        html += f'<div class="alert alert-emerge">Emergence: {" &middot; ".join(emerge_lines)}</div>'

    # ── table ────────────────────────────────────────────────────────────
    # Collect all categories across students
    cats: list[str] = []
    seen: set[str] = set()
    for s in students:
        for c in s.get("retention", {}):
            if c not in seen:
                cats.append(c)
                seen.add(c)

    html += '<table class="grid"><thead><tr>'
    html += "<th>Variant</th>"
    for c in cats:
        html += f"<th>{_esc(c)}</th>"
    html += "<th>Overall</th><th></th></tr></thead><tbody>"

    # Pre-compute best value per category for delta indicators
    best_per_cat: dict[str, float] = {}
    best_overall = 0.0
    for s in students:
        retention = s.get("retention", {})
        for c in cats:
            val = retention.get(c)
            if val is not None:
                best_per_cat[c] = max(best_per_cat.get(c, 0.0), val)
        best_overall = max(best_overall, s.get("overall_retention", 0))
    show_delta = len(students) > 1

    for s in students:
        retention = s.get("retention", {})
        detail = s.get("retention_detail", {})
        weak = set(s.get("weak_categories", []))
        low_conf = set(s.get("low_confidence_categories", []))
        graduated = s.get("graduated", False)

        html += f'<tr><td><b>{_esc(s["variant_id"])}</b></td>'
        for c in cats:
            val = retention.get(c)
            if val is None:
                html += '<td class="cell-val cell-low">&mdash;</td>'
                continue
            info = detail.get(c, {})
            ci = ""
            if "ci_lower" in info:
                ci = f'<br><span style="font-size:11px;color:#9ca3af">{info["ci_lower"]:.0%}&ndash;{info["ci_upper"]:.0%}</span>'
            # Delta indicator vs best in category
            delta = ""
            if show_delta and c in best_per_cat:
                diff = val - best_per_cat[c]
                if abs(diff) >= 0.005:
                    d_color = "#dc2626" if diff < 0 else "#16a34a"
                    delta = f'<br><span style="font-size:11px;color:{d_color}">{diff:+.0%}</span>'
            if c in low_conf:
                cls = "cell-low"
            elif c in weak:
                cls = "cell-weak"
            elif val >= 1.0:
                cls = "cell-great"
            else:
                cls = "cell-good"
            html += f'<td class="cell-val {cls}">{val:.0%}{ci}{delta}</td>'

        overall = s.get("overall_retention", 0)
        o_delta = ""
        if show_delta:
            diff = overall - best_overall
            if abs(diff) >= 0.005:
                d_color = "#dc2626" if diff < 0 else "#16a34a"
                o_delta = f' <span style="font-size:11px;color:{d_color}">({diff:+.0%})</span>'
        html += f'<td class="cell-val"><b>{overall:.0%}</b>{o_delta}</td>'
        chip = "chip-pass" if graduated else "chip-fail"
        word = "Pass" if graduated else "Fail"
        html += f'<td><span class="chip {chip}">{word}</span></td>'
        html += "</tr>"

    html += "</tbody></table>"
    html += f'<p class="foot">Threshold {threshold:.0%} &middot; Ruin floor {ruin_threshold:.0%}</p>'
    html += "</div>"
    return html


def _load_loss_curves(saves_tag: str) -> str:
    """Load trainer loss curves for all variants and render as HTML SVG chart."""
    try:
        from scripts.loss_chart import build_html_chart, _read_loss_curve
    except ImportError:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "loss_chart", Path(__file__).parent / "loss_chart.py",
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                build_html_chart = mod.build_html_chart
                _read_loss_curve = mod._read_loss_curve
            else:
                return "<p>loss_chart.py not found.</p>"
        except Exception:
            return "<p>Could not import loss_chart module.</p>"

    saves_dir = Path(f"saves/{saves_tag}")
    curves: dict[str, list[dict]] = {}

    # Scan for trainer_log.jsonl in variant subdirectories
    for log_file in saves_dir.rglob("trainer_log.jsonl"):
        variant = str(log_file.parent.relative_to(saves_dir)).replace("\\", "/")
        entries = _read_loss_curve(log_file)
        if entries:
            curves[variant] = entries

    if not curves:
        return '<p style="color:#86868b;text-align:center">No training logs found.</p>'

    return build_html_chart(curves)


def build_dashboard(report: dict[str, Any], saves_tag: str = "") -> gr.Blocks:
    with gr.Blocks(title="Graduation", css=".gradio-container{max-width:800px!important}") as demo:
        with gr.Tab("Graduation"):
            gr.HTML(_build_html(report))
        with gr.Tab("Loss Curves"):
            if saves_tag:
                gr.HTML(_load_loss_curves(saves_tag))
            else:
                gr.HTML('<p style="color:#86868b;text-align:center">Provide --saves-tag to see loss curves.</p>')
        with gr.Accordion("Raw JSON", open=False):
            gr.JSON(report)
    return demo


def build_markdown(report: dict[str, Any]) -> str:
    """Export graduation report as Markdown text."""
    threshold = report.get("threshold", 0.85)
    students = report.get("students", [])
    ruin_threshold = report.get("ruin_threshold", 0.50)

    if not students:
        return "# Graduation Report\n\nNo results.\n"

    best = max(students, key=lambda s: s.get("overall_retention", 0))
    best_pct = round(best.get("overall_retention", 0) * 100)
    n_grad = sum(1 for s in students if s.get("graduated"))
    n_total = len(students)

    if n_grad == n_total:
        verdict = "SHIP IT"
    elif n_grad > 0:
        verdict = "REVIEW"
    else:
        verdict = "KILL IT"

    lines: list[str] = []
    lines.append(f"# Graduation Report — {verdict} ({best_pct}%)")
    lines.append(f"\n{n_grad}/{n_total} graduated · threshold {threshold:.0%}\n")

    # Ruin alerts
    for s in students:
        for cat, info in s.get("retention_detail", {}).items():
            if not info.get("low_confidence") and info.get("value", 1.0) < ruin_threshold:
                lines.append(f"- **Ruin**: {s['variant_id']} {cat} {info['value']:.0%}")

    # Emergence alerts
    for s in students:
        for cat in s.get("emergent_categories", []):
            lines.append(f"- **Emergence**: {s['variant_id']} {cat}")

    # Table
    cats: list[str] = []
    seen: set[str] = set()
    for s in students:
        for c in s.get("retention", {}):
            if c not in seen:
                cats.append(c)
                seen.add(c)

    header = "| Variant | " + " | ".join(cats) + " | Overall | Status |"
    sep = "|" + "|".join(["---"] * (len(cats) + 3)) + "|"
    lines.append(f"\n{header}")
    lines.append(sep)

    for s in students:
        retention = s.get("retention", {})
        graduated = s.get("graduated", False)
        row = f"| **{s['variant_id']}** |"
        for c in cats:
            val = retention.get(c)
            row += f" {val:.0%} |" if val is not None else " — |"
        overall = s.get("overall_retention", 0)
        status = "Pass" if graduated else "Fail"
        row += f" **{overall:.0%}** | {status} |"
        lines.append(row)

    lines.append(f"\n*Threshold {threshold:.0%} · Ruin floor {ruin_threshold:.0%}*\n")
    return "\n".join(lines)


def _build_live_html(results: list[dict]) -> str:
    """Build a live progress view from forge_results.jsonl entries."""
    if not results:
        return '<div class="dash"><p style="text-align:center;color:#86868b;padding:64px 0">Waiting for results...</p></div>'

    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    html = f'<div class="dash"><style>{_CSS}</style>'
    html += f'<div class="verdict">'
    html += f'<p class="pct" style="color:#2563eb">{len(ok)}</p>'
    html += f'<p class="label" style="color:#2563eb">completed ({len(failed)} failed)</p>'
    html += "</div>"

    html += '<table class="grid"><thead><tr>'
    html += "<th>#</th><th>Variant</th><th>Model</th><th>SFT Loss</th><th>DPO Loss</th><th>Time</th><th>Status</th>"
    html += "</tr></thead><tbody>"

    sorted_results = sorted(results, key=lambda x: x.get("sft_final_loss") or float("inf"))
    best_loss = sorted_results[0].get("sft_final_loss") if sorted_results and sorted_results[0].get("ok") else None

    for rank, r in enumerate(sorted_results, 1):
        sft = f"{r['sft_final_loss']:.4f}" if r.get("sft_final_loss") is not None else "n/a"
        dpo = f"{r['dpo_final_loss']:.4f}" if r.get("dpo_final_loss") is not None else "n/a"
        elapsed = f"{r.get('elapsed_sec', 0):.0f}s"
        chip = "chip-pass" if r.get("ok") else "chip-fail"
        word = "OK" if r.get("ok") else "FAIL"
        # Highlight champion (rank 1 with ok status)
        is_champion = rank == 1 and r.get("ok")
        row_style = ' style="background:#f0fdf4"' if is_champion else ""
        rank_display = f"🏆 {rank}" if is_champion else str(rank)
        html += f'<tr{row_style}><td>{rank_display}</td><td><b>{_esc(str(r.get("variant_id", "?")))}</b></td>'
        html += f'<td>{_esc(str(r.get("model", "?")))}</td>'
        html += f'<td class="cell-val">{sft}</td>'
        html += f'<td class="cell-val">{dpo}</td>'
        html += f'<td>{elapsed}</td>'
        html += f'<td><span class="chip {chip}">{word}</span></td></tr>'

    html += "</tbody></table></div>"
    return html


def build_live_dashboard(results_path: Path, refresh_sec: int = 10) -> gr.Blocks:
    """Live dashboard that auto-refreshes from forge_results.jsonl."""

    def load_results() -> str:
        if not results_path.exists():
            return _build_live_html([])
        results = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
        return _build_live_html(results)

    with gr.Blocks(title="Forge Live", css=".gradio-container{max-width:900px!important}") as demo:
        html_out = gr.HTML(load_results())
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=load_results, outputs=html_out)
        # Auto-refresh via timer
        demo.load(fn=load_results, outputs=html_out, every=refresh_sec)

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Graduation dashboard.",
        epilog="""\
examples:
  %(prog)s --saves-tag zena007
  %(prog)s --report saves/zena007/graduation_report.json --port 8080
  %(prog)s --saves-tag zena007 --live --refresh-sec 5
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report", help="Path to graduation_report.json.")
    parser.add_argument("--saves-tag", help="Load from saves/<tag>/graduation_report.json.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="Live mode: watch forge_results.jsonl and auto-refresh.")
    parser.add_argument("--refresh-sec", type=int, default=10,
                        help="Auto-refresh interval in seconds for live mode (default: 10).")
    parser.add_argument("--export-markdown", help="Export report as Markdown to this path (no UI launched).")
    args = parser.parse_args()

    # Live mode — watch forge results
    if args.live:
        if not args.saves_tag:
            parser.error("--live requires --saves-tag")
            return 1
        results_path = Path(f"saves/{args.saves_tag}/forge_results.jsonl")
        print(f"Live dashboard watching: {results_path}")  # xray: ignore[PY-004]
        demo = build_live_dashboard(results_path, refresh_sec=args.refresh_sec)
        demo.launch(server_port=args.port, share=args.share)
        return 0

    if args.report:
        report_path = Path(args.report)
    elif args.saves_tag:
        report_path = Path(f"saves/{args.saves_tag}/graduation_report.json")
    else:
        parser.error("Provide --report or --saves-tag.")
        return 1

    if not report_path.exists():
        print(f"ERROR: Report not found: {report_path}")  # xray: ignore[PY-004]
        return 1

    try:  # xray: ignore[PY-005]
            report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
            report = {}

    # Markdown export (no UI)
    if args.export_markdown:
        md = build_markdown(report)
        md_path = Path(args.export_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        print(f"Markdown exported to {md_path}")  # xray: ignore[PY-004]
        return 0

    demo = build_dashboard(report, saves_tag=args.saves_tag or "")
    demo.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
