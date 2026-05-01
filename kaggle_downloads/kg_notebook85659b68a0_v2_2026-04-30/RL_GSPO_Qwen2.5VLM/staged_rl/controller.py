"""Metric-gated reward scheduling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import RewardGateConfig


@dataclass
class RewardControllerState:
    """Persisted reward-controller state."""

    history: list[dict[str, Any]] = field(default_factory=list)
    reward_weights: dict[str, float] = field(default_factory=dict)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    last_decision: dict[str, Any] = field(default_factory=dict)


class RewardController:
    """Update reward weights from recent checkpoint metrics."""

    def __init__(
        self,
        gate_config: RewardGateConfig,
        component_bounds: Mapping[str, tuple[float, float]],
        initial_weights: Mapping[str, float],
    ) -> None:
        self.gate_config = gate_config
        self.component_bounds = dict(component_bounds)
        self.state = RewardControllerState(
            history=[],
            reward_weights={name: float(weight) for name, weight in initial_weights.items()},
        )

    @classmethod
    def from_state(
        cls,
        gate_config: RewardGateConfig,
        component_bounds: Mapping[str, tuple[float, float]],
        initial_weights: Mapping[str, float],
        state_dict: Mapping[str, Any] | None,
    ) -> "RewardController":
        controller = cls(gate_config, component_bounds, initial_weights)
        if state_dict:
            controller.state = RewardControllerState(
                history=list(state_dict.get("history", [])),
                reward_weights={key: float(value) for key, value in state_dict.get("reward_weights", {}).items()},
                decision_history=list(state_dict.get("decision_history", [])),
                last_decision=dict(state_dict.get("last_decision", {})),
            )
            for name, weight in initial_weights.items():
                controller.state.reward_weights.setdefault(name, float(weight))
        return controller

    def to_dict(self) -> dict[str, Any]:
        """Serialize controller state."""

        return {
            "history": list(self.state.history),
            "reward_weights": dict(self.state.reward_weights),
            "decision_history": deepcopy(self.state.decision_history),
            "last_decision": deepcopy(self.state.last_decision),
        }

    def current_weights(self) -> dict[str, float]:
        """Return the current reward weights."""

        return dict(self.state.reward_weights)

    def latest_decision(self) -> dict[str, Any]:
        """Return the latest controller decision payload."""

        return deepcopy(self.state.last_decision)

    def update_from_metrics(self, metrics: Mapping[str, float], max_completion_length: int) -> dict[str, float]:
        """Apply gating rules after a checkpoint evaluation."""

        previous_weights = dict(self.state.reward_weights)
        history_length_before = len(self.state.history)
        history_entry = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        self.state.history.append(history_entry)
        weights = dict(self.state.reward_weights)
        cfg = self.gate_config

        parseable = metrics.get("parseable_answer_rate", 0.0)
        solution_ok = metrics.get("solution_tag_compliance", 0.0)
        reasoning_ok = metrics.get("reasoning_tag_compliance", 0.0)
        malformed = metrics.get("malformed_answer_rate", 1.0)
        truncation = metrics.get("truncation_rate", 1.0)
        avg_tokens = metrics.get("average_completion_tokens", float(max_completion_length))
        exact = metrics.get("normalized_exact_match", 0.0)

        stable_structure = (
            parseable >= cfg.parseable_stable_threshold
            and solution_ok >= cfg.solution_tag_stable_threshold
            and reasoning_ok >= cfg.reasoning_tag_stable_threshold
            and malformed <= cfg.malformed_stable_threshold
            and truncation <= cfg.truncation_stable_threshold
        )
        parseable_guard = parseable < cfg.parseable_floor_threshold
        format_guard = (
            solution_ok < cfg.solution_tag_floor_threshold
            or reasoning_ok < cfg.reasoning_tag_floor_threshold
            or malformed > cfg.malformed_ceiling_threshold
        )
        finish_guard = (
            truncation > cfg.truncation_ceiling_threshold
            or avg_tokens > cfg.average_token_fraction_threshold * max_completion_length
        )
        recent = self.state.history[-cfg.stable_window :] if cfg.stable_window > 0 else []
        stable_window_ready = len(recent) >= cfg.stable_window if cfg.stable_window > 0 else False
        previous_exact = None
        exact_delta = None
        correctness_plateau = False
        if stable_window_ready:
            previous_exact = recent[0].get("normalized_exact_match", exact)
            exact_delta = exact - previous_exact
            correctness_plateau = exact_delta < cfg.exact_match_plateau_delta

        decision = {
            "history_length_before": history_length_before,
            "history_length_after": len(self.state.history),
            "max_completion_length": int(max_completion_length),
            "metrics": dict(history_entry),
            "thresholds": {
                "parseable_floor_threshold": cfg.parseable_floor_threshold,
                "parseable_stable_threshold": cfg.parseable_stable_threshold,
                "solution_tag_floor_threshold": cfg.solution_tag_floor_threshold,
                "solution_tag_stable_threshold": cfg.solution_tag_stable_threshold,
                "reasoning_tag_floor_threshold": cfg.reasoning_tag_floor_threshold,
                "reasoning_tag_stable_threshold": cfg.reasoning_tag_stable_threshold,
                "malformed_ceiling_threshold": cfg.malformed_ceiling_threshold,
                "malformed_stable_threshold": cfg.malformed_stable_threshold,
                "truncation_ceiling_threshold": cfg.truncation_ceiling_threshold,
                "truncation_stable_threshold": cfg.truncation_stable_threshold,
                "average_token_fraction_threshold": cfg.average_token_fraction_threshold,
                "exact_match_plateau_delta": cfg.exact_match_plateau_delta,
                "stable_window": cfg.stable_window,
                "correctness_step": cfg.correctness_step,
                "finish_step": cfg.finish_step,
                "parseable_guard_weight": cfg.parseable_guard_weight,
                "formatting_guard_weight": cfg.formatting_guard_weight,
            },
            "pre_update_weights": dict(previous_weights),
            "post_update_weights": {},
            "rule_status": {
                "parseable_guard": parseable_guard,
                "format_guard": format_guard,
                "finish_guard": finish_guard,
                "stable_structure": stable_structure,
                "stable_window_ready": stable_window_ready,
                "correctness_plateau": correctness_plateau,
                "correctness_escalation": stable_structure and stable_window_ready and correctness_plateau,
            },
            "exact_previous": previous_exact,
            "exact_current": exact,
            "exact_delta": exact_delta,
            "avg_token_fraction": avg_tokens / max(max_completion_length, 1),
            "rule_events": [],
            "weight_deltas": {},
            "changed_components": [],
            "clamped_components": [],
        }

        if parseable_guard and "parseable_reward" in weights:
            before = weights["parseable_reward"]
            after = max(weights["parseable_reward"], cfg.parseable_guard_weight)
            weights["parseable_reward"] = after
            decision["rule_events"].append(
                {
                    "rule_key": "parseable_guard",
                    "component": "parseable_reward",
                    "before": before,
                    "after": after,
                    "changed": after != before,
                    "evidence": {
                        "parseable_answer_rate": parseable,
                        "parseable_floor_threshold": cfg.parseable_floor_threshold,
                        "parseable_guard_weight": cfg.parseable_guard_weight,
                    },
                }
            )

        if format_guard and "format_reward" in weights:
            before = weights["format_reward"]
            after = max(weights["format_reward"], cfg.formatting_guard_weight)
            weights["format_reward"] = after
            decision["rule_events"].append(
                {
                    "rule_key": "format_guard",
                    "component": "format_reward",
                    "before": before,
                    "after": after,
                    "changed": after != before,
                    "evidence": {
                        "solution_tag_compliance": solution_ok,
                        "solution_tag_floor_threshold": cfg.solution_tag_floor_threshold,
                        "reasoning_tag_compliance": reasoning_ok,
                        "reasoning_tag_floor_threshold": cfg.reasoning_tag_floor_threshold,
                        "malformed_answer_rate": malformed,
                        "malformed_ceiling_threshold": cfg.malformed_ceiling_threshold,
                        "formatting_guard_weight": cfg.formatting_guard_weight,
                    },
                }
            )

        if finish_guard and "finished_reward" in weights:
            before = weights["finished_reward"]
            after = before + cfg.finish_step
            weights["finished_reward"] = after
            decision["rule_events"].append(
                {
                    "rule_key": "finish_guard",
                    "component": "finished_reward",
                    "before": before,
                    "after": after,
                    "changed": after != before,
                    "evidence": {
                        "truncation_rate": truncation,
                        "truncation_ceiling_threshold": cfg.truncation_ceiling_threshold,
                        "average_completion_tokens": avg_tokens,
                        "average_token_fraction_threshold": cfg.average_token_fraction_threshold,
                        "max_completion_length": max_completion_length,
                        "finish_step": cfg.finish_step,
                    },
                }
            )

        if stable_structure and stable_window_ready and correctness_plateau and "correctness_reward" in weights:
            before = weights["correctness_reward"]
            after = before + cfg.correctness_step
            weights["correctness_reward"] = after
            decision["rule_events"].append(
                {
                    "rule_key": "correctness_escalation",
                    "component": "correctness_reward",
                    "before": before,
                    "after": after,
                    "changed": after != before,
                    "evidence": {
                        "exact_previous": previous_exact,
                        "exact_current": exact,
                        "exact_delta": exact_delta,
                        "exact_match_plateau_delta": cfg.exact_match_plateau_delta,
                        "stable_window": cfg.stable_window,
                        "correctness_step": cfg.correctness_step,
                    },
                }
            )

        weights["format_reward"] = max(weights.get("format_reward", 0.0), cfg.format_floor)
        weights["parseable_reward"] = max(weights.get("parseable_reward", 0.0), cfg.parseable_floor)
        weights["finished_reward"] = max(weights.get("finished_reward", 0.0), cfg.finish_floor)

        for name, weight in list(weights.items()):
            lower, upper = self.component_bounds[name]
            clamped = min(max(weight, lower), upper)
            if clamped != weight:
                decision["clamped_components"].append(
                    {
                        "component": name,
                        "before": weight,
                        "after": clamped,
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
            weights[name] = clamped

        decision["post_update_weights"] = dict(weights)
        for name, after in weights.items():
            before = previous_weights.get(name, 0.0)
            delta = after - before
            decision["weight_deltas"][name] = {
                "before": before,
                "after": after,
                "delta": delta,
            }
            if delta != 0.0:
                decision["changed_components"].append(name)

        self.state.last_decision = decision
        self.state.decision_history.append(deepcopy(decision))
        self.state.reward_weights = weights
        return dict(weights)
