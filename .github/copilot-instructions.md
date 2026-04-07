# GitHub Copilot Instructions for LLaMA Factory

## Project Overview

LLaMA Factory is an efficient fine-tuning framework for 100+ large language models (LLMs). It provides:
- Support for various models: LLaMA, LLaVA, Mistral, Qwen, DeepSeek, Yi, Gemma, ChatGLM, Phi, etc.
- Multiple training methods: pre-training, supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO
- Scalable resources: 16-bit full-tuning, freeze-tuning, LoRA and QLoRA variants
- Advanced algorithms: GaLore, BAdam, APOLLO, Adam-mini, Muon, OFT, DoRA, etc.
- Web UI (LLaMA Board) and CLI interfaces

### Architecture Versions

LLaMA Factory has two parallel architectures that can be switched via the `USE_V1` environment variable:

**v0 (default)** - File hierarchy:
- `api`, `webui` → `chat`, `eval`, `train` → `data`, `model` → `hparams` → `extras`

**v1** - File hierarchy:
- `trainers` → `core` → `accelerator`, `plugins`, `config` → `utils`

Set `USE_V1=1` to enable v1 architecture.

## Code Structure

### v0 Architecture (Default)

- `src/llamafactory/` - Main package directory
  - `api/` - OpenAI-style API implementation
  - `chat/` - Chat interface implementation
  - `cli.py` - Command-line interface
  - `data/` - Data processing and dataset handling
  - `eval/` - Model evaluation utilities
  - `extras/` - Additional utilities and helpers
  - `hparams/` - Hyperparameter definitions
  - `model/` - Model loading, patching, and utilities
  - `train/` - Training pipeline implementation
  - `webui/` - Gradio-based web interface
- `src/train.py` - Training entry script (delegates to `llamafactory.train.tuner`)
- `src/webui.py` - Web UI entry script (delegates to `llamafactory.webui.interface`)
- `src/api.py` - API server entry script (delegates to `llamafactory.api.app`)
- `tests/` - Test suite
- `examples/` - Example configurations for various training scenarios
- `data/` - Dataset definitions and examples

### v1 Architecture (USE_V1=1)

- `src/llamafactory/v1/` - Version 1 package directory
  - `trainers/` - Training implementations
  - `core/` - Core training utilities
  - `accelerator/` - Acceleration and distributed training
  - `plugins/` - Pluggable components (model, data, sampler, trainer)
  - `config/` - Configuration management
  - `utils/` - Utility functions

## Development Practices

### Code Style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use ruff for linting and formatting
- Line length: 119 characters
- Indentation: 4 spaces
- Quote style: double quotes
- Use Google-style docstrings for documentation

### Import Organization

- Known first-party: `llamafactory`
- Known third-party: `accelerate`, `datasets`, `gradio`, `numpy`, `peft`, `torch`, `transformers`, `trl`
- Use 2 blank lines after imports

### Quality Checks

Before committing code, run:
```bash
make style      # Auto-fix style issues
make quality    # Check code quality
make test       # Run test suite
```

Or use the combined command:
```bash
make commit     # Run pre-commit hooks
```

### Testing

- Use pytest for testing
- Tests are located in `tests/` and `tests_v1/` directories
- Run tests with: `make test` (which runs `WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/`)
- Disable wandb during testing to avoid external dependencies
- **Note**: Training configurations require GPU machines, so training is typically not tested end-to-end. Use `make test` to validate file-level functionality.

### Building

Build the package with:
```bash
pip3 install build && python3 -m build
```

### License

- All source files must include the Apache 2.0 license header
- Check license headers with: `make license`

## Common Patterns

### Configuration Files

- Training configurations are typically YAML or JSON files in `examples/` directory
- Hyperparameters are defined using dataclasses in `src/llamafactory/hparams/`

### Model Support

- New model support is added through model patches in `src/llamafactory/model/`
- Visual models use the visual utilities in `src/llamafactory/model/model_utils/visual.py`
- Quantization support is in `src/llamafactory/model/model_utils/quantization.py`

### Data Processing

- Dataset definitions are in `data/dataset_info.json`
- Data templates and processors are in `src/llamafactory/data/`

### Training

- Training pipelines are in `src/llamafactory/train/`
- Support for different training methods: SFT, DPO, PPO, RM, PT, KTO, ORPO

### Multi-Teacher Distillation & FIFO Dispatch

The multi-teacher distillation system generates training data by running multiple GGUF teachers
concurrently, then splitting outputs into consensus (SFT) and conflict (DPO) datasets.

#### Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/multi_teacher_generate.py` | Generate responses from multiple GGUF teachers |
| `scripts/purify_teacher_outputs.py` | Split responses into consensus (SFT) / conflict (DPO) |
| `scripts/gen_distill_configs.py` | Auto-generate SFT / DPO / merge YAML configs |
| `scripts/run_student_forge.py` | Parallel training of multiple student variants (Forge Matrix) |
| `scripts/eval_student_panel.py` | Two-pass evaluation: quick quiz → deep exam |
| `scripts/student_registry.py` | Persist eval scores and gap analysis across runs |
| `scripts/slim_down.py` | GGUF export + speed benchmark + Pareto frontier |
| `scripts/graduation_dashboard.py` | HTML dashboard with SVG verdict ring + live mode |
| `scripts/benchmark_multi_teacher_dispatch.py` | A/B benchmark for dispatch modes |
| `scripts/run_zena007_end_to_end.ps1` | Full pipeline orchestrator (sequential or Forge Matrix) |
| `scripts/validate_datasets.py` | Data validation: duplicates, leakage, balance, DPO validity |
| `scripts/prompt_difficulty.py` | Prompt difficulty scoring + histogram + filtering |
| `scripts/bayesian_forge.py` | Bayesian hyperparameter search (Optuna TPE sampler) |
| `scripts/teacher_profile.py` | Per-teacher quality analysis (agreement, GOLD/SILVER contribution) |
| `scripts/pipeline_preflight.py` | Pre-run validator (manifest, prompts, configs, disk, deps) |
| `scripts/loss_chart.py` | Loss comparison chart (text + SVG) with convergence prediction |
| `scripts/pipeline_events.py` | Structured JSON event logger for pipeline stages |
| `scripts/orchestrate_pipeline.py` | Cross-platform Python pipeline orchestrator (replaces PS1) |
| `scripts/zenforge_cli.py` | Unified CLI entry point (`python scripts/zenforge_cli.py <command>`) |

#### Generation architecture (v4 — SPSC Ring-Buffer FIFO)

- Per-teacher `SPSCRingBuffer` (lock-free, GIL-atomic integer stores)
- Default 2048-slot depth via `--fifo-size 0` (auto mode)
- `RAMPressureThrottle` with hysteresis (`--ram-pause-pct 12 --ram-resume-pct 22`)
- Adaptive decoding budgets per prompt category (`--adaptive-budgets`)
- Per-teacher JSONL checkpoints in `data/<tag>/checkpoints/` for crash-safe resume
- Backends: `InProcessAdapter` (direct GGUF, no HTTP) or `LlamaServerManager` (HTTP)

Shared library: `zen_core_libs` provides `SPSCRingBuffer`, `InProcessAdapter`,
and `RAMPressureThrottle` in `zen_core_libs.common.system` and `zen_core_libs.llm`.

`InProcessAdapter.chat()` signature: `messages: list[dict[str, Any]] | None = None` — always
use fully-typed dicts. The `_build_messages` helper also returns `list[dict[str, Any]]`.

#### Pipeline stages & crash-resume

Every stage is idempotent — re-run the same command after any crash:

| Stage | Skip condition | Resume behaviour |
|-------|---------------|-----------------|
| Generation | `teacher_responses.jsonl` exists with expected row count | Per-teacher checkpoints auto-merged; missing prompts refilled |
| Purification | `purified/purification_report.json` exists | Full skip |
| Config gen | Both `*_sft.yaml` and `*_merge.yaml` exist | Full skip |
| Dataset registration | Entry already in `dataset_info.json` | Idempotent write |
| SFT training | `adapter_model.safetensors` present in output dir | Auto-resumes from highest `checkpoint-N` |
| DPO training | Same as SFT, or no `conflict_dpo.jsonl` samples | Skipped entirely when no DPO data |
| Merge | `saves/<tag>/merged/config.json` exists | Full skip; auto-recovers adapter path from highest checkpoint |
| Forge results | `saves/<tag>/forge_results.jsonl` exists | Full skip; synthesized from training artifacts in sequential mode |
| Evaluation | `saves/<tag>/eval_scorecards.jsonl` exists | Full skip; two-pass: quick quiz → deep exam |
| Dashboard | `saves/<tag>/graduation_report.json` exists | Exports markdown + HTML |

#### Student Forge Matrix (parallel multi-variant training)

`run_zena007_end_to_end.ps1 -UseForge` activates Forge Matrix mode which trains N student
variants in parallel (controlled by `max_parallel` in the matrix YAML). Each variant can
differ in model, LoRA rank, learning rate, and epoch count. After training, an eval panel
scores all variants and selects the champion for merging.

Matrix config: `data/forge_matrix/<tag>_matrix.yaml` — defines `sft_data`, `dpo_data`,
`max_parallel`, `eval_probe_split`, and per-variant settings.

The forge automatically:
1. Splits `consensus_sft.jsonl` into `train_sft.jsonl` + `eval_probes.jsonl` (probe_fraction)
2. Registers `<tag>_forge_train_sft` in `dataset_info.json` (relative to `data/` dir)
3. Generates per-variant YAML configs in `examples/distillation/auto/forge/<tag>/`
4. Trains variants concurrently via SPSC ring-buffer result collection
5. Writes `saves/<tag>/forge_results.jsonl` and `saves/<tag>/champion.txt`

#### Student Forge Auto-Healing (`run_student_forge.py`)

`ForgeState` provides crash-safe state management for multi-day training runs:

- **Atomic state file** (`saves/<tag>/forge_state.json`) — records completed/failed variants,
  heartbeat timestamps, and partial results. Survives crashes.
- **Checkpoint resume** — `_find_latest_checkpoint(output_dir)` finds the highest
  `checkpoint-N` directory for automatic resume via `overwrite_output_dir: False`.
- **Heartbeat** — background thread writes timestamps every 30s; stale heartbeats
  (>120s) indicate a hung worker.
- **LLM diagnosis** — `_diagnose_with_llm(stderr)` sends error logs to a local GGUF
  model for root-cause analysis when a training variant fails.
- **Idempotent re-run** — ForgeState skips already-completed variants on restart.

Tests: `tests/test_forge_autoheal.py` (12 tests covering state, checkpoints, heartbeat,
atomic writes).

#### Graduation Dashboard (`graduation_dashboard.py`)

Generates a single-page HTML report from a graduation report JSON:

- SVG ring with pass/fail/review verdict and percentage
- Clean HTML table with per-category retention, confidence, and status
- Ruin and emergence alert banners
- Raw JSON in an accordion for debugging

Usage: `python scripts/graduation_dashboard.py --saves-tag zena007`

#### Graduation Exam (`zen_core_libs.llm.eval`)

Shared evaluation library in `zen_core_libs` (not in this repo). Key exports:

- `compute_retention()` — teacher-vs-student retention ratio per category
- `graduation_report()` — multi-student comparison with pass/fail verdicts
- `bootstrap_ci()` — confidence intervals via bootstrap resampling
- `detect_emergence()` — finds categories where student beats all teachers
- `RuinDetected` — exception raised when retention drops below threshold
- `extract_category()` — extracts category from prompt ID or explicit field

Tests: `zen_core_libs/llm/tests/test_eval.py` (42 tests).

#### Config generation rules (`gen_distill_configs.py`)

- `_merge_config(student, tag, has_dpo=True/False)` — only chains DPO adapter when
  `has_dpo=True`; if no conflict/DPO samples exist the merge config uses SFT adapter only
- **Do NOT add `booster: auto`** (or any non-HfArgumentParser key) to generated YAML configs —
  LlamaFactory's `parse_dict` will raise `ValueError: Some keys are not used` at startup
- `bf16` is set to `not cpu_safe` — always pass `--cpu-safe` on CPU-only machines

#### `dataset_info.json` file_name convention

Paths in `dataset_info.json` are **relative to `dataset_dir`** (default `"data"`). Always
strip the leading `data/` prefix. Use `_rel_to_data(path)` helper in `run_student_forge.py`
to compute the correct relative path. Wrong: `"data/zena007/purified/x.jsonl"` →
LlamaFactory joins `data/ + data/zena007/...` = double-path crash.

#### Advanced purification features (`purify_teacher_outputs.py`)

- **Embedding-based reasoning**: `--use-embeddings` uses all-MiniLM-L6-v2 for semantic
  similarity (requires `sentence-transformers`). Falls back to SimHash when unavailable.
- **Teacher weighting**: `--teacher-weights '{"qwen": 1.5, "llama": 1.0}'` — weighted
  majority voting. Higher weight = more influence in consensus detection.
- **Auto-tune thresholds**: `--auto-tune` sweeps answer/reason thresholds via grid search
  to hit a target GOLD percentage (default 60%). Controlled by `--auto-tune-target`.
- **Synthetic DPO**: `--synthetic-dpo` generates cross-prompt DPO pairs from GOLD samples
  by pairing highest-confidence chosen with lowest-confidence rejected.
- **Curriculum learning**: `--curriculum` sorts GOLD output by difficulty (easy → hard)
  for progressive training.

#### Early stopping (`gen_distill_configs.py`)

`--early-stopping-patience N` (N > 0) adds `eval_strategy`, `eval_steps`, and
`load_best_model_at_end` to SFT/DPO configs. Monitors `eval_loss` and stops when
it doesn't improve for N eval rounds.

#### Native GGUF export (`slim_down.py`)

`--llama-cpp-path /path/to/llama.cpp` enables direct GGUF conversion via llama.cpp's
`convert_hf_to_gguf.py` + `llama-quantize`. Falls back to LlamaFactory export CLI
when not specified.

#### CI/CD

`.github/workflows/ci.yml` runs on push/PR: 59 tests (3 suites), ruff lint, and script
smoke tests. No GPU required.

## Key Dependencies

- Python >= 3.9.0
- PyTorch and transformers for model handling
- datasets for data processing
- peft for parameter-efficient fine-tuning
- accelerate for distributed training
- gradio for web UI
- trl for reinforcement learning
- psutil for RAM-pressure throttling (multi-teacher generation)
- Optional: vllm/sglang for inference, flash-attention-2, unsloth, liger-kernel

## Entry Points

- **CLI Training**: `llamafactory-cli train --config examples/train_lora/llama3_lora_sft.yaml`
- **Web UI**: `llamafactory-cli webui` or `python src/webui.py`
- **API Server**: `llamafactory-cli api` or `python src/api.py`
- **Chat Interface**: `llamafactory-cli chat --model_name_or_path MODEL_PATH`
- **Multi-Teacher Generation**: `python scripts/multi_teacher_generate.py --manifest MANIFEST --prompts PROMPTS --dispatch-mode teacher-fifo --fifo-size 0`
- **Dispatch Benchmark**: `python scripts/benchmark_multi_teacher_dispatch.py --manifest MANIFEST --prompts PROMPTS --output-dir OUT`
- **End-to-End Distillation (sequential)**: `./scripts/run_zena007_end_to_end.ps1`
- **End-to-End Distillation (Forge Matrix)**: `./scripts/run_zena007_end_to_end.ps1 -UseForge`
- **Forge dry-run**: `python scripts/run_student_forge.py --matrix data/forge_matrix/zena007_matrix.yaml --tag zena007 --dry-run`
- **Eval panel only**: `python scripts/eval_student_panel.py --saves-tag zena007 --probes data/zena007/purified/eval_probes.jsonl`
- **Graduation dashboard**: `python scripts/graduation_dashboard.py --saves-tag zena007`

## Environment Setup

For development:
```bash
pip install -e ".[dev]"
```

## Important Notes

- The project supports multiple backends: default PyTorch, vLLM, SGLang
- Megatron-core training is supported via mcore_adapter
- SwanLab and W&B are supported for experiment tracking
- Docker support is available with pre-built images
- Day-0/Day-1 support for latest cutting-edge models
- Multi-modal support for vision and audio understanding tasks

## Contribution Guidelines

1. Fork the repository
2. Create a development branch
3. Set up development environment with `pip install -e ".[dev]"`
4. Make changes following the style guide
5. Run quality checks: `make style && make quality`
6. Run tests: `make test`
7. Submit a pull request

### WebUI Architecture

The Web UI is built with Gradio in `src/llamafactory/webui/`. Key files:

- `interface.py` — top-level `create_ui()` assembles tabs, JS injection, Zena menu
- `engine.py` — `Engine` class: state manager, coordinates manager/runner/chatter
- `runner.py` — `Runner` class: training/eval subprocess lifecycle
- `chatter.py` — `WebChatModel`: model load/unload, chat streaming
- `control.py` — dropdown/validation handlers (model info, checkpoints, auto-tune)
- `components/` — one file per tab (top, train, eval, infer, export, chatbot, data, footer)

All visible buttons are wired to real handlers. 6 help accordions (`*_help_tab`) are
`visible=False` — they have `lang.change` handlers but are never shown (JS `?` icons
serve the same purpose). No NOOP or dead handlers exist.

### Testing

Local test suites (no GPU required):

| Suite | Command | Tests |
|-------|---------|-------|
| Forge auto-heal | `pytest tests/test_forge_autoheal.py -v --noconftest` | 16 |
| Integration pipeline | `pytest tests/test_integration_pipeline.py -v --noconftest` | 35 |
| End-to-end toy | `pytest tests/test_end_to_end_toy.py -v --noconftest` | 8 |
| Graduation eval (zen_core_libs) | `pytest zen_core_libs/llm/tests/test_eval.py -v` | 42 |
| **Total** | | **101** |

## Common Commands

- `make style` - Format code
- `make quality` - Run linters
- `make test` - Run tests
- `make commit` - Install and run pre-commit hooks
- `make license` - Check license headers
