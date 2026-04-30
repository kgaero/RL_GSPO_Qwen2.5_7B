"""Tests for the reward controller."""

import unittest

from staged_rl.config import RewardGateConfig
from staged_rl.controller import RewardController


class RewardControllerTests(unittest.TestCase):
    def make_controller(self) -> RewardController:
        return RewardController(
            gate_config=RewardGateConfig(),
            component_bounds={
                "format_reward": (0.25, 2.0),
                "parseable_reward": (0.25, 2.0),
                "finished_reward": (0.5, 2.0),
                "correctness_reward": (1.0, 8.0),
                "brevity_reward": (0.0, 1.0),
                "tolerance_reward": (0.0, 2.0),
            },
            initial_weights={
                "format_reward": 0.5,
                "parseable_reward": 0.5,
                "finished_reward": 0.5,
                "correctness_reward": 2.0,
                "brevity_reward": 0.2,
                "tolerance_reward": 0.0,
            },
        )

    def test_parseability_and_truncation_guards(self):
        controller = self.make_controller()

        updated = controller.update_from_metrics(
            {
                "parseable_answer_rate": 0.70,
                "solution_tag_compliance": 0.80,
                "reasoning_tag_compliance": 0.80,
                "malformed_answer_rate": 0.20,
                "truncation_rate": 0.30,
                "average_completion_tokens": 240.0,
                "normalized_exact_match": 0.10,
            },
            max_completion_length=256,
        )

        self.assertGreaterEqual(updated["parseable_reward"], 0.75)
        self.assertGreaterEqual(updated["format_reward"], 0.75)
        self.assertGreater(updated["finished_reward"], 0.5)
        decision = controller.latest_decision()
        self.assertTrue(decision["rule_status"]["parseable_guard"])
        self.assertTrue(decision["rule_status"]["format_guard"])
        self.assertTrue(decision["rule_status"]["finish_guard"])
        self.assertEqual(decision["weight_deltas"]["finished_reward"]["delta"], 0.25)
        self.assertIn("finished_reward", decision["changed_components"])

    def test_correctness_increase_after_stable_window(self):
        controller = self.make_controller()

        stable_metrics = {
            "parseable_answer_rate": 0.95,
            "solution_tag_compliance": 0.96,
            "reasoning_tag_compliance": 0.94,
            "malformed_answer_rate": 0.02,
            "truncation_rate": 0.03,
            "average_completion_tokens": 120.0,
            "normalized_exact_match": 0.40,
        }
        controller.update_from_metrics(stable_metrics, max_completion_length=256)
        updated = controller.update_from_metrics({**stable_metrics, "normalized_exact_match": 0.41}, max_completion_length=256)
        self.assertGreater(updated["correctness_reward"], 2.0)
        decision = controller.latest_decision()
        self.assertTrue(decision["rule_status"]["stable_structure"])
        self.assertTrue(decision["rule_status"]["stable_window_ready"])
        self.assertTrue(decision["rule_status"]["correctness_plateau"])
        self.assertTrue(decision["rule_status"]["correctness_escalation"])
        self.assertAlmostEqual(decision["exact_delta"], 0.01)

    def test_correctness_increase_on_regression_when_structure_stable(self):
        controller = self.make_controller()

        stable_metrics = {
            "parseable_answer_rate": 1.0,
            "solution_tag_compliance": 1.0,
            "reasoning_tag_compliance": 1.0,
            "malformed_answer_rate": 0.0,
            "truncation_rate": 0.0,
            "average_completion_tokens": 100.0,
            "normalized_exact_match": 0.75,
        }
        controller.update_from_metrics(stable_metrics, max_completion_length=256)
        updated = controller.update_from_metrics({**stable_metrics, "normalized_exact_match": 0.25}, max_completion_length=256)
        self.assertGreater(updated["correctness_reward"], 2.0)
        decision = controller.latest_decision()
        self.assertAlmostEqual(decision["exact_delta"], -0.5)
        self.assertTrue(decision["rule_status"]["correctness_escalation"])


if __name__ == "__main__":
    unittest.main()
