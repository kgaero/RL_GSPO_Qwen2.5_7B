"""Parsing and normalization helpers shared by rewards and evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Optional

from .config import REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START


OPTION_PATTERN = re.compile(r"\b([A-Z])\b")


def normalize_numeric_string(x: Any) -> Optional[str]:
    """
    Normalize numeric answers while preserving the current repo's core behavior.

    Examples:
    - ``2`` -> ``"2"``
    - ``2.000`` -> ``"2"``
    - ``0.5000`` -> ``"0.5"``
    """

    if x is None:
        return None

    s = str(x).strip()
    if not s:
        return None

    s = s.replace(",", "").strip()
    try:
        value = float(s)
    except Exception:
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    if value.is_integer():
        return str(int(round(value)))

    normalized = f"{value:.12g}"
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def parse_float_safe(x: Any) -> Optional[float]:
    """Parse a normalized numeric string into a finite float."""

    normalized = normalize_numeric_string(x)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except Exception:
        return None


def extract_reasoning_blocks(text: str) -> list[str]:
    """Return all reasoning blocks."""

    pattern = f"{re.escape(REASONING_START)}(.*?){re.escape(REASONING_END)}"
    return re.findall(pattern, text or "", re.DOTALL)


def extract_solution_blocks(text: str) -> list[str]:
    """Return all solution blocks."""

    pattern = f"{re.escape(SOLUTION_START)}(.*?){re.escape(SOLUTION_END)}"
    return re.findall(pattern, text or "", re.DOTALL)


def extract_single_solution_text(text: str) -> Optional[str]:
    """Return the single extracted solution block or ``None``."""

    matches = extract_solution_blocks(text or "")
    if len(matches) != 1:
        return None
    return matches[0].replace("\n", " ").strip()


def parse_numeric_solution(text: str) -> Optional[str]:
    """Extract and normalize a numeric free-form solution."""

    solution = extract_single_solution_text(text)
    if solution is None:
        return None
    return normalize_numeric_string(solution)


def extract_multichoice_option_letter(text: str) -> Optional[str]:
    """Extract a final multiple-choice option letter from the solution block."""

    solution = extract_single_solution_text(text)
    if solution is None:
        return None

    letters = OPTION_PATTERN.findall(solution.upper())
    if letters:
        return letters[-1]

    compact = solution.strip().upper()
    if compact in {"A", "B", "C", "D", "E", "F"}:
        return compact
    return None


def normalized_exact_match(pred: Any, gold: Any) -> bool:
    """Exact match after numeric normalization."""

    return normalize_numeric_string(pred) == normalize_numeric_string(gold) and normalize_numeric_string(pred) is not None


def tolerance_match(
    pred: Any,
    gold: Any,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> bool:
    """Tolerance-based numeric match."""

    pred_value = parse_float_safe(pred)
    gold_value = parse_float_safe(gold)
    if pred_value is None or gold_value is None:
        return False
    return math.isclose(pred_value, gold_value, rel_tol=rel_tol, abs_tol=abs_tol)


def solution_tag_compliant(text: str) -> bool:
    """Return whether the text has exactly one solution block."""

    return len(extract_solution_blocks(text or "")) == 1


def reasoning_tag_compliant(text: str) -> bool:
    """Return whether the text has exactly one reasoning block."""

    return len(extract_reasoning_blocks(text or "")) == 1


def malformed_numeric_answer(text: str) -> bool:
    """Return whether a numeric free-form completion is malformed."""

    solution = extract_solution_blocks(text or "")
    if len(solution) != 1:
        return True
    answer_text = solution[0].replace("\n", " ").strip()
    if not answer_text:
        return True
    return parse_float_safe(answer_text) is None


def malformed_multichoice_answer(text: str) -> bool:
    """Return whether a multi-choice completion fails basic extraction."""

    return extract_multichoice_option_letter(text) is None


def completion_finished(text: str, answer_mode: str = "numeric_free_form") -> bool:
    """Detect whether the completion appears to contain a finished answer."""

    if answer_mode == "multi_choice":
        return reasoning_tag_compliant(text) and solution_tag_compliant(text) and not malformed_multichoice_answer(text)
    return reasoning_tag_compliant(text) and solution_tag_compliant(text) and not malformed_numeric_answer(text)


def compute_repetition_rate(text: str, n: int = 3) -> float:
    """Whitespace n-gram repetition fraction."""

    tokens = (text or "").split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(ngrams), 1)


def infer_truncation(
    text: str,
    completion_tokens: int,
    max_completion_length: int,
    answer_mode: str = "numeric_free_form",
) -> bool:
    """Heuristic truncation detector."""

    if completion_tokens < max_completion_length:
        return False
    return not completion_finished(text, answer_mode=answer_mode)


def compute_option_letter(gold_answer: Any, choices: Optional[Iterable[str]]) -> Optional[str]:
    """Infer the gold option letter from answer text and choice list."""

    if not choices:
        return None
    normalized_choices = [str(choice).strip() for choice in choices]
    gold_text = str(gold_answer).strip()
    for index, choice in enumerate(normalized_choices):
        if gold_text == choice:
            return chr(ord("A") + index)
        if normalize_numeric_string(gold_text) is not None and normalize_numeric_string(gold_text) == normalize_numeric_string(choice):
            return chr(ord("A") + index)
    return None

