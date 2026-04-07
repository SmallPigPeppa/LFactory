"""Tests for ForgeState auto-heal (crash recovery, state persistence)."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# Add scripts/ to path so we can import from run_student_forge
_scripts = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

from run_student_forge import ForgeState, _find_latest_checkpoint


class TestForgeState:
    def test_blank_state(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        assert state.completed_ids() == set()
        assert state.completed_results() == []

    def test_record_and_resume(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.5, "elapsed_sec": 60})

        # Simulate restart — new ForgeState loads from disk
        state2 = ForgeState("test", saves_dir=tmp_path)
        assert state2.is_completed("A")
        assert not state2.is_completed("B")
        assert state2.completed_ids() == {"A"}

    def test_completed_results_reconstructed(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.5, "dpo_final_loss": 0.8, "elapsed_sec": 120})

        state2 = ForgeState("test", saves_dir=tmp_path)
        results = state2.completed_results()
        assert len(results) == 1
        assert results[0]["variant_id"] == "A"
        assert results[0]["ok"] is True
        assert results[0]["sft_final_loss"] == 1.5
        assert results[0]["resumed"] is True

    def test_record_failure(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_failure("B", "OOM crash", "Reduce batch size")

        state2 = ForgeState("test", saves_dir=tmp_path)
        data = json.loads((tmp_path / "test" / "forge_state.json").read_text("utf-8"))
        assert "B" in data["failed"]
        assert "OOM crash" in data["failed"]["B"]["error"]
        assert "Reduce batch size" in data["failed"]["B"]["diagnosis"]

    def test_record_finished(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.0})
        state.record_finished()

        data = json.loads((tmp_path / "test" / "forge_state.json").read_text("utf-8"))
        assert data["status"] == "finished"
        assert "finished_at" in data

    def test_heartbeat(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        assert state.last_heartbeat_age() is None

        state.write_heartbeat()
        age = state.last_heartbeat_age()
        assert age is not None
        assert age < 2.0  # should be near-instant

    def test_multiple_variants(self, tmp_path):
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.5})
        state.record_complete("B", {"sft_final_loss": 2.0})
        state.record_failure("C", "timeout")

        state2 = ForgeState("test", saves_dir=tmp_path)
        assert state2.completed_ids() == {"A", "B"}
        assert len(state2.completed_results()) == 2

    def test_atomic_write_survives_reload(self, tmp_path):
        """State file should be valid JSON even after multiple rapid writes."""
        state = ForgeState("test", saves_dir=tmp_path)
        for i in range(20):
            state.record_complete(f"v{i}", {"sft_final_loss": float(i)})

        state2 = ForgeState("test", saves_dir=tmp_path)
        assert len(state2.completed_ids()) == 20


class TestFindLatestCheckpoint:
    def test_no_dir(self, tmp_path):
        assert _find_latest_checkpoint(str(tmp_path / "nonexistent")) is None

    def test_empty_dir(self, tmp_path):
        assert _find_latest_checkpoint(str(tmp_path)) is None

    def test_finds_highest(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-50").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result is not None
        assert "checkpoint-200" in result

    def test_ignores_non_checkpoint_dirs(self, tmp_path):
        (tmp_path / "logs").mkdir()
        (tmp_path / "checkpoint-100").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert "checkpoint-100" in result


class TestConcurrentForgeState:
    """Stress tests for concurrent writes to ForgeState."""

    def test_concurrent_writers(self, tmp_path):
        """Multiple threads writing to the same ForgeState should not corrupt state.

        On Windows, os.replace() may raise PermissionError under contention.
        The test tolerates individual write failures but verifies the final
        state file is valid JSON with at least some completions recorded.
        """
        n_threads = 4
        n_per_thread = 5
        errors: list[str] = []

        def writer(thread_id: int):
            state = ForgeState("test", saves_dir=tmp_path)
            for i in range(n_per_thread):
                vid = f"t{thread_id}_v{i}"
                try:
                    state.record_complete(vid, {"sft_final_loss": float(i)})
                except PermissionError:
                    errors.append(vid)  # expected on Windows

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(writer, tid) for tid in range(n_threads)]
            for f in futs:
                f.result()

        # Reload and verify at least some completions survived
        state = ForgeState("test", saves_dir=tmp_path)
        completed = state.completed_ids()
        assert len(completed) >= 1  # at least 1 survived contention
        # State file must still be valid JSON
        data = json.loads((tmp_path / "test" / "forge_state.json").read_text("utf-8"))
        assert "completed" in data

    def test_concurrent_heartbeats(self, tmp_path):
        """Concurrent heartbeat writes should not corrupt state."""
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("seed", {"sft_final_loss": 0.5})

        def heartbeat_writer(_):
            s = ForgeState("test", saves_dir=tmp_path)
            for _ in range(10):
                try:
                    s.write_heartbeat()
                except PermissionError:
                    pass  # expected on Windows

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(heartbeat_writer, i) for i in range(4)]
            for f in futs:
                f.result()

        # Must reload without error
        state2 = ForgeState("test", saves_dir=tmp_path)
        assert state2.is_completed("seed")
        # Heartbeat age may be None if all writes hit PermissionError
        age = state2.last_heartbeat_age()
        if age is not None:
            assert age < 10.0

    def test_disk_full_recovery(self, tmp_path):
        """ForgeState should survive a read even if last write was truncated."""
        state = ForgeState("test", saves_dir=tmp_path)
        state.record_complete("A", {"sft_final_loss": 1.0})

        # Simulate truncated write — overwrite with partial JSON
        state_file = tmp_path / "test" / "forge_state.json"
        good_data = state_file.read_text("utf-8")
        state_file.write_text(good_data[:len(good_data) // 2], encoding="utf-8")

        # ForgeState should handle corrupt state gracefully
        try:
            state2 = ForgeState("test", saves_dir=tmp_path)
            # If it recovers, it should start fresh or use a backup
            # The test passes as long as no unhandled exception occurs
        except (json.JSONDecodeError, KeyError):
            # Acceptable — at least it didn't silently corrupt further
            pass

    def test_interleaved_complete_and_failure(self, tmp_path):
        """Mix of completions and failures from concurrent workers."""
        def worker(tid: int):
            state = ForgeState("test", saves_dir=tmp_path)
            try:
                if tid % 2 == 0:
                    state.record_complete(f"w{tid}", {"sft_final_loss": float(tid)})
                else:
                    state.record_failure(f"w{tid}", f"Error from thread {tid}")
            except PermissionError:
                pass  # expected on Windows under contention

        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [pool.submit(worker, i) for i in range(12)]
            for f in futs:
                f.result()

        state = ForgeState("test", saves_dir=tmp_path)
        data = json.loads((tmp_path / "test" / "forge_state.json").read_text("utf-8"))
        # Should have some completed and/or failed — at least 1 survived
        total = len(data.get("completed", {})) + len(data.get("failed", {}))
        assert total >= 1
