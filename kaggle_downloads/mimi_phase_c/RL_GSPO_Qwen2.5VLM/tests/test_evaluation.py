"""Tests for metric aggregation."""

import sys
import tempfile
import unittest
from pathlib import Path
import types

sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda _index: "stub",
        )
    ),
)

from rl_gspo_qwen2_5vlm_eval import apply_target_adapter_overrides
from staged_rl.config import build_default_run_config
from staged_rl.evaluation import (
    _prepare_prompt_for_generation,
    aggregate_subset_metrics,
    determine_failure_mode,
    select_overall_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_determine_failure_mode(self):
        record = {
            "truncation": True,
            "solution_tag_compliance": False,
            "reasoning_tag_compliance": False,
            "malformed_answer": True,
            "parseable_answer": False,
            "normalized_exact_match": False,
        }
        self.assertEqual(determine_failure_mode(record), "truncation")

    def test_aggregate_subset_metrics(self):
        per_prompt_records = [
            {
                "samples": [
                    {
                        "normalized_exact_match": True,
                        "tolerance_match": True,
                        "parseable_answer": True,
                        "solution_tag_compliance": True,
                        "reasoning_tag_compliance": True,
                        "malformed_answer": False,
                        "truncation": False,
                        "completion_tokens": 10,
                        "repetition_rate": 0.0,
                        "total_reward": 1.0,
                        "reward_component/format_reward": 1.0,
                    }
                ],
                "best_of_k_accuracy": True,
                "best_of_k_tolerance_accuracy": True,
                "sampled_answer_diversity": 0.5,
            }
        ]
        all_sample_records = per_prompt_records[0]["samples"]
        metrics = aggregate_subset_metrics(per_prompt_records, all_sample_records)
        self.assertEqual(metrics["normalized_exact_match"], 1.0)
        self.assertEqual(metrics["best_of_k_accuracy"], 1.0)
        self.assertIn("reward_component/format_reward_mean", metrics)

    def test_select_overall_metrics_prefers_full_split_when_present(self):
        subset_results = {
            "eval_full_split": {"metrics": {"normalized_exact_match": 0.5}},
            "stage1_easy_numeric": {"metrics": {"normalized_exact_match": 1.0}},
        }
        metrics = select_overall_metrics(subset_results)
        self.assertEqual(metrics["normalized_exact_match"], 0.5)

    def test_select_overall_metrics_prefers_multi_choice_stage_for_phase_e(self):
        subset_results = {
            "eval_overall_numeric": {"metrics": {"normalized_exact_match": 0.5}},
            "stage4_multi_choice": {"metrics": {"normalized_exact_match": 0.9}},
            "eval_full_split": {"metrics": {"normalized_exact_match": 0.1}},
        }
        metrics = select_overall_metrics(subset_results, phase_name="phase_e")
        self.assertEqual(metrics["normalized_exact_match"], 0.9)

    def test_select_overall_metrics_prefers_hard_numeric_stage_for_phase_d(self):
        subset_results = {
            "eval_overall_numeric": {"metrics": {"normalized_exact_match": 0.5}},
            "stage3_hard_numeric": {"metrics": {"normalized_exact_match": 0.8}},
            "stage4_multi_choice": {"metrics": {"normalized_exact_match": 0.9}},
        }
        metrics = select_overall_metrics(subset_results, phase_name="phase_d")
        self.assertEqual(metrics["normalized_exact_match"], 0.8)

    def test_select_overall_metrics_falls_back_to_single_subset(self):
        subset_results = {
            "custom_subset": {"metrics": {"normalized_exact_match": 0.25}},
        }
        metrics = select_overall_metrics(subset_results)
        self.assertEqual(metrics["normalized_exact_match"], 0.25)

    def test_prepare_prompt_for_generation_applies_chat_template(self):
        class _FakeTokenizer:
            def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True):
                return f"templated:{len(prompt)}"

        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hello"}]}]
        self.assertEqual(_prepare_prompt_for_generation(prompt, _FakeTokenizer()), "templated:1")

    def test_prepare_prompt_for_generation_leaves_string_prompts_untouched(self):
        self.assertEqual(_prepare_prompt_for_generation("hello", object()), "hello")

    def test_apply_target_adapter_overrides_requires_adapter_metadata(self):
        run_config = build_default_run_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            targets = [{"label": "checkpoint", "checkpoint": str(checkpoint_dir)}]

            with self.assertRaises(FileNotFoundError):
                apply_target_adapter_overrides(run_config, targets)


if __name__ == "__main__":
    unittest.main()
