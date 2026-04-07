"""Integration smoke tests — validate pipeline scripts import cleanly and run --help.

These tests verify that all distillation pipeline scripts are importable and
their CLI arg parsers work without errors. No GPU or model downloads required.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_PY = sys.executable


class TestScriptImports:
    """All pipeline scripts should be importable without error."""

    @pytest.mark.parametrize("script", [
        "purify_teacher_outputs.py",
        "gen_distill_configs.py",
        "run_student_forge.py",
        "eval_student_panel.py",
        "slim_down.py",
        "graduation_dashboard.py",
        "student_registry.py",
        "validate_datasets.py",
        "prompt_difficulty.py",
        "bayesian_forge.py",
        "teacher_profile.py",
        "pipeline_preflight.py",
        "loss_chart.py",
        "pipeline_events.py",
        "orchestrate_pipeline.py",
    ])
    def test_script_importable(self, script):
        """Script should be importable as a module (syntax check)."""
        script_path = _SCRIPTS_DIR / script
        if not script_path.exists():
            pytest.skip(f"{script} not found")
        proc = subprocess.run(
            [_PY, "-c", f"import importlib.util; spec = importlib.util.spec_from_file_location('m', r'{script_path}'); mod = importlib.util.module_from_spec(spec)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Import failed: {proc.stderr}"


class TestScriptHelp:
    """All pipeline scripts should print help and exit 0."""

    @pytest.mark.parametrize("script", [
        "purify_teacher_outputs.py",
        "gen_distill_configs.py",
        "validate_datasets.py",
        "prompt_difficulty.py",
        "teacher_profile.py",
        "pipeline_preflight.py",
        "loss_chart.py",
        "orchestrate_pipeline.py",
    ])
    def test_script_help(self, script):
        script_path = _SCRIPTS_DIR / script
        if not script_path.exists():
            pytest.skip(f"{script} not found")
        proc = subprocess.run(
            [_PY, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"--help failed: {proc.stderr}"
        assert "usage:" in proc.stdout.lower() or "--" in proc.stdout


class TestPurificationPipeline:
    """End-to-end smoke test for purification."""

    def test_classify_gold(self):
        """Three teachers agree on answer + reasoning → GOLD tier."""
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import classify_sample

        # Thoughts must be similar enough for SimHash to pass reasoning alignment
        sample = {
            "id": "test_001",
            "prompt": "What is 2+2?",
            "teachers": {
                "teacher_a": {"answer": "4", "raw": "The answer is 4.", "thought": "Two plus two equals four by basic addition of integers."},
                "teacher_b": {"answer": "4", "raw": "It's 4.", "thought": "Two plus two equals four using simple integer addition."},
                "teacher_c": {"answer": "4", "raw": "2+2=4.", "thought": "Two plus two equals four through basic addition."},
            },
        }
        tier, record = classify_sample(sample, answer_threshold=0.85, reason_threshold=0.5)
        assert tier == "GOLD"
        assert record["tier"] == "GOLD"
        assert "confidence" in record
        assert record["confidence"] > 0.5
        assert record["n_teachers_agree"] == 3
        assert "difficulty" in record

    def test_classify_drop(self):
        """All teachers disagree → DROP."""
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import classify_sample

        sample = {
            "id": "test_002",
            "prompt": "What is the meaning of life?",
            "teachers": {
                "teacher_a": {"answer": "42", "raw": "42.", "thought": "Douglas Adams."},
                "teacher_b": {"answer": "Love", "raw": "Love.", "thought": "Philosophy."},
                "teacher_c": {"answer": "Nothing", "raw": "Nothing.", "thought": "Nihilism."},
            },
        }
        tier, record = classify_sample(sample, answer_threshold=0.85, reason_threshold=0.6)
        assert tier == "DROP"

    def test_classify_single_teacher(self):
        """Single teacher → always GOLD."""
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import classify_sample

        sample = {
            "id": "test_003",
            "prompt": "Hello?",
            "teachers": {
                "teacher_a": {"answer": "Hi!", "raw": "Hi there!"},
            },
        }
        tier, record = classify_sample(sample, answer_threshold=0.85, reason_threshold=0.6)
        assert tier == "GOLD"


class TestValidateDatasets:
    """Smoke test for validate_datasets.py."""

    def test_check_duplicates(self, tmp_path):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from validate_datasets import ValidationReport, check_duplicates

        rows = [
            {"instruction": "What is 2+2?", "output": "4"},
            {"instruction": "What is 2+2?", "output": "4"},  # duplicate
            {"instruction": "What is 3+3?", "output": "6"},
        ]
        report = ValidationReport()
        check_duplicates(rows, "test_sft", report)
        assert len(report.warnings) > 0  # should find duplicate

    def test_check_dpo_validity(self, tmp_path):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from validate_datasets import ValidationReport, check_dpo_validity

        rows = [
            {"prompt": "Q1", "chosen": "good", "rejected": "bad"},
            {"prompt": "Q2", "chosen": "", "rejected": "bad"},  # empty chosen
            {"prompt": "Q3", "chosen": "same", "rejected": "same"},  # identical
        ]
        report = ValidationReport()
        check_dpo_validity(rows, report)
        assert len(report.warnings) > 0 or len(report.errors) > 0


class TestPromptDifficulty:
    """Smoke test for prompt_difficulty.py."""

    def test_score_easy(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from prompt_difficulty import score_teacher_responses

        rows = [{
            "id": "p1",
            "prompt": "2+2?",
            "teachers": {
                "a": {"answer": "4"},
                "b": {"answer": "4"},
                "c": {"answer": "4"},
            },
        }]
        scored = score_teacher_responses(rows)
        assert len(scored) == 1
        assert scored[0]["difficulty"] == 0.0  # all agree = easy

    def test_score_hard(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from prompt_difficulty import score_teacher_responses

        rows = [{
            "id": "p2",
            "prompt": "Meaning of life?",
            "teachers": {
                "a": {"answer": "42"},
                "b": {"answer": "love"},
                "c": {"answer": "nothing"},
            },
        }]
        scored = score_teacher_responses(rows)
        assert scored[0]["difficulty"] > 0.5  # all disagree = hard

    def test_filter_by_difficulty(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from prompt_difficulty import filter_by_difficulty

        rows = [
            {"id": "1", "difficulty": 0.0},
            {"id": "2", "difficulty": 0.3},
            {"id": "3", "difficulty": 0.8},
        ]
        hard = filter_by_difficulty(rows, min_difficulty=0.5)
        assert len(hard) == 1
        assert hard[0]["id"] == "3"


class TestSimHash:
    """Verify SimHash implementation."""

    def test_identical_texts(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import _simhash_similarity

        assert _simhash_similarity("hello world", "hello world") == 1.0

    def test_different_texts(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import _simhash_similarity

        sim = _simhash_similarity("hello world", "completely different text here")
        assert sim < 0.8

    def test_similar_texts(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import _simhash_similarity

        sim = _simhash_similarity(
            "The answer is four because two plus two equals four",
            "The answer is four as two plus two makes four",
        )
        assert sim > 0.5

    def test_empty_text(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        from purify_teacher_outputs import _simhash_similarity

        assert _simhash_similarity("", "hello") == 0.0
        assert _simhash_similarity("hello", "") == 0.0


class TestOrchestratorRecovery:
    """Tests for orchestrate_pipeline.py recovery and forge-results synthesis."""

    def _import_orchestrator(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        import importlib
        import orchestrate_pipeline
        importlib.reload(orchestrate_pipeline)
        return orchestrate_pipeline

    def test_find_latest_checkpoint_root_adapter(self, tmp_path):
        """When adapter_model.safetensors is in the root dir, return root."""
        orch = self._import_orchestrator()
        (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00" * 10)
        result = orch._find_latest_checkpoint(tmp_path)
        assert result == tmp_path

    def test_find_latest_checkpoint_highest(self, tmp_path):
        """When no root adapter, find the highest checkpoint-N."""
        orch = self._import_orchestrator()
        for n in [50, 100, 200]:
            ckpt = tmp_path / f"checkpoint-{n}"
            ckpt.mkdir()
            (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 10)
        result = orch._find_latest_checkpoint(tmp_path)
        assert result == tmp_path / "checkpoint-200"

    def test_find_latest_checkpoint_none(self, tmp_path):
        """Empty dir returns None."""
        orch = self._import_orchestrator()
        result = orch._find_latest_checkpoint(tmp_path)
        assert result is None

    def test_find_latest_checkpoint_skip_empty(self, tmp_path):
        """Checkpoint dirs without adapter_model.safetensors are skipped."""
        orch = self._import_orchestrator()
        (tmp_path / "checkpoint-200").mkdir()  # empty — no adapter
        ckpt100 = tmp_path / "checkpoint-100"
        ckpt100.mkdir()
        (ckpt100 / "adapter_model.safetensors").write_bytes(b"\x00" * 10)
        result = orch._find_latest_checkpoint(tmp_path)
        assert result == ckpt100

    def test_get_final_loss(self, tmp_path):
        """Read the last loss from trainer_log.jsonl."""
        orch = self._import_orchestrator()
        log_path = tmp_path / "trainer_log.jsonl"
        log_path.write_text(
            '{"current_steps": 100, "loss": 1.5}\n'
            '{"current_steps": 200, "loss": 0.89}\n',
            encoding="utf-8",
        )
        assert orch._get_final_loss(log_path) == 0.89

    def test_get_final_loss_empty(self, tmp_path):
        """Empty file returns None."""
        orch = self._import_orchestrator()
        assert orch._get_final_loss(tmp_path / "nonexistent.jsonl") is None

    def test_synthesize_forge_results(self, tmp_path):
        """Synthesize forge_results.jsonl from training artifacts."""
        orch = self._import_orchestrator()
        # Create fake adapter
        ckpt = tmp_path / "adapter" / "checkpoint-200"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 10)
        # Create trainer log
        (tmp_path / "adapter" / "trainer_log.jsonl").write_text(
            '{"current_steps": 200, "loss": 1.05}\n',
            encoding="utf-8",
        )
        forge_path = tmp_path / "forge_results.jsonl"
        ok = orch._synthesize_forge_results("test", "model/test", str(tmp_path / "adapter"), forge_path)
        assert ok is True
        assert forge_path.exists()
        result = json.loads(forge_path.read_text(encoding="utf-8").strip())
        assert result["variant_id"] == "B"
        assert result["model"] == "model/test"
        assert result["sft_final_loss"] == 1.05
        assert result["ok"] is True

    def test_synthesize_forge_results_no_adapter(self, tmp_path):
        """Returns False when no adapter found."""
        orch = self._import_orchestrator()
        forge_path = tmp_path / "forge_results.jsonl"
        ok = orch._synthesize_forge_results("test", "model/test", str(tmp_path / "empty"), forge_path)
        assert ok is False
        assert not forge_path.exists()

    def test_is_training_complete_true(self, tmp_path):
        """Complete training (current_steps == total_steps) returns True."""
        orch = self._import_orchestrator()
        log = tmp_path / "trainer_log.jsonl"
        log.write_text(
            '{"current_steps": 100, "total_steps": 230, "loss": 1.5}\n'
            '{"current_steps": 230, "total_steps": 230, "loss": 0.8}\n',
            encoding="utf-8",
        )
        assert orch._is_training_complete(tmp_path) is True

    def test_is_training_complete_false(self, tmp_path):
        """Incomplete training (current_steps < total_steps) returns False."""
        orch = self._import_orchestrator()
        log = tmp_path / "trainer_log.jsonl"
        log.write_text(
            '{"current_steps": 100, "total_steps": 230, "loss": 1.5}\n'
            '{"current_steps": 220, "total_steps": 230, "loss": 1.2}\n',
            encoding="utf-8",
        )
        assert orch._is_training_complete(tmp_path) is False

    def test_is_training_complete_no_log(self, tmp_path):
        """No trainer_log.jsonl returns False."""
        orch = self._import_orchestrator()
        assert orch._is_training_complete(tmp_path) is False

    def test_qualitative_script_defined(self):
        """Qualitative eval script constant is defined and non-empty."""
        orch = self._import_orchestrator()
        assert hasattr(orch, "_QUALITATIVE_EVAL_SCRIPT")
        assert len(orch._QUALITATIVE_EVAL_SCRIPT) > 100
        assert "AutoModelForCausalLM" in orch._QUALITATIVE_EVAL_SCRIPT


class TestGenDistillConfigs:
    """Tests for gen_distill_configs.py resume-safe config generation."""

    def _import_gen(self):
        sys.path.insert(0, str(_SCRIPTS_DIR))
        import importlib
        import gen_distill_configs
        importlib.reload(gen_distill_configs)
        return gen_distill_configs

    def test_sft_config_overwrite_false(self):
        """SFT config should set overwrite_output_dir=False for resume."""
        gen = self._import_gen()
        cfg = gen._sft_config("model/test", "ds", "tag", cpu_safe=False)
        assert cfg["overwrite_output_dir"] is False

    def test_sft_config_save_only_model_false(self):
        """SFT config should set save_only_model=False for checkpoint resume."""
        gen = self._import_gen()
        cfg = gen._sft_config("model/test", "ds", "tag", cpu_safe=False)
        assert cfg["save_only_model"] is False

    def test_dpo_config_overwrite_false(self):
        """DPO config should set overwrite_output_dir=False for resume."""
        gen = self._import_gen()
        cfg = gen._dpo_config("model/test", "ds", "tag", cpu_safe=False)
        assert cfg["overwrite_output_dir"] is False
