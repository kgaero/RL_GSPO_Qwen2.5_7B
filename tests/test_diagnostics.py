"""Tests for diagnostics helpers."""

import tempfile
import unittest
from pathlib import Path

from staged_rl.diagnostics import build_post_training_diagnostics, write_fatal_error


class DiagnosticsTests(unittest.TestCase):
    def test_write_fatal_error_persists_traceback_and_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fatal_error.txt"
            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                write_fatal_error(output_path, exc, {"phase": "phase_a", "resume": None})

            text = output_path.read_text(encoding="utf-8")
            self.assertIn("exception_type: RuntimeError", text)
            self.assertIn("exception_message: boom", text)
            self.assertIn('"phase": "phase_a"', text)
            self.assertIn("traceback:", text)
            self.assertIn("RuntimeError: boom", text)

    def test_build_post_training_diagnostics_uses_primary_subset(self):
        registry_data = {"checkpoints": [], "aliases": {}}
        eval_results = {
            "phase_name": "phase_e",
            "subset_results": {
                "eval_overall_numeric": {
                    "all_sample_records": [{"failure_mode": "truncation"} for _ in range(3)]
                },
                "stage4_multi_choice": {
                    "all_sample_records": [{"failure_mode": "malformed_answer"} for _ in range(2)]
                },
            }
        }

        diagnostics = build_post_training_diagnostics(registry_data, eval_results)

        self.assertEqual(diagnostics["dominant_failure_modes"][0], ("malformed_answer", 2))


if __name__ == "__main__":
    unittest.main()
