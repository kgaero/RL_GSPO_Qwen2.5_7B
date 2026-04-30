"""Reward helpers and factories for staged RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .config import REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START, PhaseConfig
from .parsing import (
    completion_finished,
    compute_option_letter,
    extract_multichoice_option_letter,
    extract_single_solution_text,
    malformed_multichoice_answer,
    malformed_numeric_answer,
    normalize_numeric_string,
    normalized_exact_match,
    parse_float_safe,
    reasoning_tag_compliant,
    solution_tag_compliant,
    tolerance_match,
)


@dataclass
class RewardRuntimeContext:
    """Runtime dependencies used by reward functions."""

    tokenizer: Any
    max_completion_length: int
    phase_config: PhaseConfig

    def completion_token_count(self, text: str) -> int:
        """Count completion tokens using the tokenizer when available."""

        text = "" if text is None else str(text)
        if hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "encode"):
            return len(self.tokenizer.tokenizer.encode(text, add_special_tokens=False))
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())


def _exact_or_multichoice_match(completion: str, gold_answer: Any, answer_mode: str, choices: Optional[Iterable[str]]) -> bool:
    if answer_mode == "multi_choice":
        predicted = extract_multichoice_option_letter(completion)
        gold_letter = compute_option_letter(gold_answer, choices)
        return predicted is not None and predicted == gold_letter
    solution = extract_single_solution_text(completion)
    return solution is not None and normalized_exact_match(solution, gold_answer)


def _tolerance_match_for_record(
    completion: str,
    gold_answer: Any,
    answer_mode: str,
    precision: Optional[float],
) -> bool:
    if answer_mode != "numeric_free_form":
        return False
    solution = extract_single_solution_text(completion)
    if solution is None:
        return False
    if precision is None:
        return tolerance_match(solution, gold_answer)
    abs_tol = 10 ** (-int(precision))
    rel_tol = max(abs_tol, 1e-6)
    return tolerance_match(solution, gold_answer, abs_tol=abs_tol, rel_tol=rel_tol)


def _format_reward_single(completion: str, answer_mode: str) -> float:
    score = 0.0
    if reasoning_tag_compliant(completion):
        score += 1.0
    if solution_tag_compliant(completion):
        score += 1.0
    if answer_mode == "multi_choice" and malformed_multichoice_answer(completion):
        score -= 0.5
    if answer_mode == "numeric_free_form" and malformed_numeric_answer(completion):
        score -= 0.5
    return score


def _parseable_reward_single(completion: str, answer_mode: str) -> float:
    if answer_mode == "multi_choice":
        return 1.0 if not malformed_multichoice_answer(completion) else 0.0
    solution = extract_single_solution_text(completion)
    if solution is None:
        return 0.0
    return 1.0 if parse_float_safe(solution) is not None else 0.0


def _finished_reward_single(completion: str, answer_mode: str, token_count: int, max_completion_length: int) -> float:
    finished = completion_finished(completion, answer_mode=answer_mode)
    if finished and token_count < max_completion_length:
        return 1.0
    if token_count >= max_completion_length and not finished:
        return -1.0
    return 0.0


def _brevity_reward_single(completion: str, token_count: int, max_completion_length: int) -> float:
    if token_count >= max_completion_length:
        return -1.0
    ratio = token_count / max(max_completion_length, 1)
    if ratio <= 0.55:
        return 1.0
    if ratio <= 0.75:
        return 0.5
    if ratio <= 0.90:
        return 0.0
    return -0.5


def build_reward_functions(runtime: RewardRuntimeContext) -> list[Callable[..., list[float]]]:
    """Build the reward functions passed into GRPO."""

    def format_reward(completions, answer_mode=None, **kwargs):
        modes = answer_mode or ["numeric_free_form"] * len(completions)
        return [_format_reward_single(completion, mode) for completion, mode in zip(completions, modes)]

    def parseable_reward(completions, answer_mode=None, **kwargs):
        modes = answer_mode or ["numeric_free_form"] * len(completions)
        return [_parseable_reward_single(completion, mode) for completion, mode in zip(completions, modes)]

    def finished_reward(completions, answer_mode=None, **kwargs):
        modes = answer_mode or ["numeric_free_form"] * len(completions)
        scores = []
        for completion, mode in zip(completions, modes):
            token_count = runtime.completion_token_count(completion)
            scores.append(_finished_reward_single(completion, mode, token_count, runtime.max_completion_length))
        return scores

    def correctness_reward(completions, answer, answer_mode=None, choices=None, **kwargs):
        modes = answer_mode or ["numeric_free_form"] * len(completions)
        choices = choices or [None] * len(completions)
        return [
            1.0 if _exact_or_multichoice_match(completion, gold_answer, mode, record_choices) else 0.0
            for completion, gold_answer, mode, record_choices in zip(completions, answer, modes, choices)
        ]

    def brevity_reward(completions, **kwargs):
        return [
            _brevity_reward_single(completion, runtime.completion_token_count(completion), runtime.max_completion_length)
            for completion in completions
        ]

    def tolerance_reward(completions, answer, answer_mode=None, precision=None, **kwargs):
        modes = answer_mode or ["numeric_free_form"] * len(completions)
        precisions = precision or [None] * len(completions)
        scores = []
        for completion, gold_answer, mode, item_precision in zip(completions, answer, modes, precisions):
            if not runtime.phase_config.enable_tolerance_reward:
                scores.append(0.0)
                continue
            exact = _exact_or_multichoice_match(completion, gold_answer, mode, None)
            if exact:
                scores.append(0.0)
                continue
            scores.append(1.0 if _tolerance_match_for_record(completion, gold_answer, mode, item_precision) else 0.0)
        return scores

    format_reward.__name__ = "format_reward"
    parseable_reward.__name__ = "parseable_reward"
    finished_reward.__name__ = "finished_reward"
    correctness_reward.__name__ = "correctness_reward"
    brevity_reward.__name__ = "brevity_reward"
    tolerance_reward.__name__ = "tolerance_reward"

    return [
        format_reward,
        parseable_reward,
        finished_reward,
        correctness_reward,
        brevity_reward,
        tolerance_reward,
    ]


def phase_reward_weights(phase_config: PhaseConfig) -> dict[str, float]:
    """Return the initial reward weights for a phase."""

    return {
        name: component.initial_weight
        for name, component in phase_config.reward_components.items()
        if component.enabled
    }

