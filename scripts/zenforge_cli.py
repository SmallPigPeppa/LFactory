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

"""ZEN Forge — unified CLI entry point for multi-teacher distillation pipeline.

Usage:
    python scripts/zenforge_cli.py generate  [args]   # Multi-teacher generation
    python scripts/zenforge_cli.py purify    [args]   # Purify teacher outputs
    python scripts/zenforge_cli.py configure [args]   # Generate training configs
    python scripts/zenforge_cli.py train     [args]   # Run student forge training
    python scripts/zenforge_cli.py evaluate  [args]   # Eval student panel
    python scripts/zenforge_cli.py dashboard [args]   # Graduation dashboard
    python scripts/zenforge_cli.py slim      [args]   # GGUF export + bench
    python scripts/zenforge_cli.py validate  [args]   # Dataset validation
    python scripts/zenforge_cli.py preflight [args]   # Pipeline preflight checks
    python scripts/zenforge_cli.py loss      [args]   # Loss comparison chart
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_COMMANDS: dict[str, tuple[str, str]] = {
    "generate":  ("multi_teacher_generate", "Multi-teacher response generation"),
    "purify":    ("purify_teacher_outputs", "Purify into GOLD/SILVER/DROP tiers"),
    "configure": ("gen_distill_configs", "Generate SFT/DPO/merge YAML configs"),
    "train":     ("run_student_forge", "Run student forge (parallel training)"),
    "evaluate":  ("eval_student_panel", "Two-pass student evaluation"),
    "dashboard": ("graduation_dashboard", "Graduation dashboard (Gradio)"),
    "slim":      ("slim_down", "GGUF export + speed benchmark"),
    "validate":  ("validate_datasets", "Dataset validation checks"),
    "preflight": ("pipeline_preflight", "Pre-run pipeline validation"),
    "loss":      ("loss_chart", "Loss comparison chart"),
    "profile":   ("teacher_profile", "Per-teacher quality analysis"),
    "bayesian":  ("bayesian_forge", "Bayesian hyperparameter search"),
    "difficulty": ("prompt_difficulty", "Prompt difficulty scoring"),
}


def _print_help() -> None:
    print("ZEN Forge — multi-teacher distillation pipeline\n")
    print("Usage: python scripts/zenforge_cli.py <command> [args]\n")
    print("Commands:")
    max_len = max(len(c) for c in _COMMANDS)
    for cmd, (_, desc) in _COMMANDS.items():
        print(f"  {cmd:<{max_len + 2}} {desc}")
    print(f"\nRun 'python scripts/zenforge_cli.py <command> --help' for command-specific help.")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        _print_help()
        return 0

    cmd = sys.argv[1]
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd}")
        _print_help()
        return 1

    module_name, _ = _COMMANDS[cmd]

    # Remove the command name from argv so the sub-module sees clean args
    sys.argv = [f"zenforge {cmd}"] + sys.argv[2:]

    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        print(f"Error importing {module_name}: {exc}")
        return 1

    if hasattr(mod, "main"):
        return mod.main()
    else:
        print(f"Module {module_name} has no main() function.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
