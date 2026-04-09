"""Tests for checkpoint score and alias resolution."""

import tempfile
import unittest
from pathlib import Path

from staged_rl.checkpointing import CheckpointRegistry, build_resume_plan, compute_checkpoint_scores
from staged_rl.config import CheckpointScoreConfig


class CheckpointingTests(unittest.TestCase):
    def test_compute_checkpoint_scores(self):
        scores = compute_checkpoint_scores(
            {
                "normalized_exact_match": 0.5,
                "tolerance_accuracy": 0.6,
                "parseable_answer_rate": 0.9,
                "solution_tag_compliance": 0.95,
                "reasoning_tag_compliance": 0.94,
                "malformed_answer_rate": 0.05,
                "truncation_rate": 0.1,
            },
            CheckpointScoreConfig(),
        )
        self.assertIn("structure_score", scores)
        self.assertIn("correctness_score", scores)
        self.assertIn("composite_score", scores)

    def test_registry_and_resume_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "phase_a"
            registry = CheckpointRegistry(run_dir)
            registry.register(
                {
                    "checkpoint_path": str(run_dir / "checkpoint-10"),
                    "global_step": 10,
                    "phase_name": "phase_a",
                    "metrics": {"normalized_exact_match": 0.3},
                    "scores": {
                        "structure_score": 0.4,
                        "correctness_score": 0.3,
                        "composite_score": 0.35,
                    },
                }
            )
            registry.register(
                {
                    "checkpoint_path": str(run_dir / "checkpoint-20"),
                    "global_step": 20,
                    "phase_name": "phase_a",
                    "metrics": {"normalized_exact_match": 0.5},
                    "scores": {
                        "structure_score": 0.45,
                        "correctness_score": 0.5,
                        "composite_score": 0.48,
                    },
                }
            )
            (run_dir / "checkpoint-20").mkdir(parents=True, exist_ok=True)

            plan = build_resume_plan(
                selector="best_composite",
                current_phase="phase_b",
                current_phase_dir=Path(tmpdir) / "phase_b",
                search_dirs=[run_dir],
                default_model_name="base-model",
            )
            self.assertEqual(plan.model_load_path, "base-model")
            self.assertIsNone(plan.trainer_resume_path)
            self.assertEqual(plan.adapter_warm_start_path, str(run_dir / "checkpoint-20"))

            latest_plan = build_resume_plan(
                selector="latest",
                current_phase="phase_a",
                current_phase_dir=run_dir,
                search_dirs=[run_dir],
                default_model_name="base-model",
            )
            self.assertEqual(latest_plan.trainer_resume_path, str(run_dir / "checkpoint-20"))
            self.assertIsNone(latest_plan.adapter_warm_start_path)

            warm_start_plan = build_resume_plan(
                selector=str(run_dir / "checkpoint-20"),
                current_phase="phase_a",
                current_phase_dir=run_dir,
                search_dirs=[run_dir],
                default_model_name="base-model",
                force_warm_start=True,
            )
            self.assertEqual(warm_start_plan.model_load_path, "base-model")
            self.assertIsNone(warm_start_plan.trainer_resume_path)
            self.assertEqual(warm_start_plan.adapter_warm_start_path, str(run_dir / "checkpoint-20"))

    def test_best_composite_prefers_most_recent_prior_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            phase_a = tmpdir / "phase_a"
            phase_b = tmpdir / "phase_b"
            phase_c = tmpdir / "phase_c"
            phase_d = tmpdir / "phase_d"

            registry_a = CheckpointRegistry(phase_a)
            registry_a.register(
                {
                    "checkpoint_path": str(phase_a / "checkpoint-10"),
                    "global_step": 10,
                    "phase_name": "phase_a",
                    "metrics": {},
                    "scores": {
                        "structure_score": 1.0,
                        "correctness_score": 1.0,
                        "composite_score": 1.0,
                    },
                }
            )
            registry_b = CheckpointRegistry(phase_b)
            registry_b.register(
                {
                    "checkpoint_path": str(phase_b / "checkpoint-20"),
                    "global_step": 20,
                    "phase_name": "phase_b",
                    "metrics": {},
                    "scores": {
                        "structure_score": 2.0,
                        "correctness_score": 2.0,
                        "composite_score": 2.0,
                    },
                }
            )
            registry_d = CheckpointRegistry(phase_d)
            registry_d.register(
                {
                    "checkpoint_path": str(phase_d / "checkpoint-40"),
                    "global_step": 40,
                    "phase_name": "phase_d",
                    "metrics": {},
                    "scores": {
                        "structure_score": 9.0,
                        "correctness_score": 9.0,
                        "composite_score": 9.0,
                    },
                }
            )

            plan = build_resume_plan(
                selector="best_composite",
                current_phase="phase_c",
                current_phase_dir=phase_c,
                search_dirs=[phase_a, phase_b, phase_c, phase_d],
                default_model_name="base-model",
            )
            self.assertEqual(plan.model_load_path, "base-model")
            self.assertIsNone(plan.trainer_resume_path)
            self.assertEqual(plan.adapter_warm_start_path, str(phase_b / "checkpoint-20"))
            self.assertEqual(plan.phase_name, "phase_b")

    def test_latest_does_not_cross_phase_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            phase_a = tmpdir / "phase_a"
            registry_a = CheckpointRegistry(phase_a)
            registry_a.register(
                {
                    "checkpoint_path": str(phase_a / "checkpoint-10"),
                    "global_step": 10,
                    "phase_name": "phase_a",
                    "metrics": {},
                    "scores": {
                        "structure_score": 1.0,
                        "correctness_score": 1.0,
                        "composite_score": 1.0,
                    },
                }
            )

            plan = build_resume_plan(
                selector="latest",
                current_phase="phase_c",
                current_phase_dir=tmpdir / "phase_c",
                search_dirs=[phase_a],
                default_model_name="base-model",
            )
            self.assertEqual(plan.model_load_path, "base-model")
            self.assertIsNone(plan.trainer_resume_path)
            self.assertIsNone(plan.adapter_warm_start_path)
            self.assertIsNone(plan.phase_name)

    def test_best_composite_does_not_leak_into_future_phases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            phase_c = tmpdir / "phase_c"
            phase_d = tmpdir / "phase_d"

            registry_d = CheckpointRegistry(phase_d)
            registry_d.register(
                {
                    "checkpoint_path": str(phase_d / "checkpoint-40"),
                    "global_step": 40,
                    "phase_name": "phase_d",
                    "metrics": {},
                    "scores": {
                        "structure_score": 9.0,
                        "correctness_score": 9.0,
                        "composite_score": 9.0,
                    },
                }
            )

            plan = build_resume_plan(
                selector="best_composite",
                current_phase="phase_c",
                current_phase_dir=phase_c,
                search_dirs=[phase_c, phase_d],
                default_model_name="base-model",
            )
            self.assertEqual(plan.model_load_path, "base-model")
            self.assertIsNone(plan.trainer_resume_path)
            self.assertIsNone(plan.adapter_warm_start_path)
            self.assertIsNone(plan.phase_name)


if __name__ == "__main__":
    unittest.main()
