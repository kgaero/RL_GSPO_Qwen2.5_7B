"""Checkpoint evaluation and metric aggregation."""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Callable, Mapping, Optional

from .config import EvalConfig
from .parsing import (
    completion_finished,
    compute_repetition_rate,
    extract_multichoice_option_letter,
    extract_single_solution_text,
    infer_truncation,
    malformed_multichoice_answer,
    malformed_numeric_answer,
    normalized_exact_match,
    parse_float_safe,
    parse_numeric_solution,
    reasoning_tag_compliant,
    solution_tag_compliant,
    tolerance_match,
)
from .rewarding import RewardRuntimeContext


def _float_or_nan(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def determine_failure_mode(record: Mapping[str, Any]) -> str:
    """Collapse sample behavior into a single dominant failure mode."""

    if record["truncation"]:
        return "truncation"
    if not record["solution_tag_compliance"]:
        return "missing_solution_tag"
    if not record["reasoning_tag_compliance"]:
        return "missing_reasoning_tag"
    if record["malformed_answer"]:
        return "malformed_answer"
    if not record["parseable_answer"]:
        return "unparseable_answer"
    if not record["normalized_exact_match"]:
        return "parseable_but_wrong"
    return "exact_match"


def aggregate_subset_metrics(per_prompt_records: list[dict[str, Any]], all_sample_records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate sample-level and prompt-level metrics."""

    first_samples = [record["samples"][0] for record in per_prompt_records if record["samples"]]
    metrics = {
        "normalized_exact_match": _float_or_nan([float(item["normalized_exact_match"]) for item in first_samples]),
        "tolerance_accuracy": _float_or_nan([float(item["tolerance_match"]) for item in first_samples]),
        "best_of_k_accuracy": _float_or_nan([float(item["best_of_k_accuracy"]) for item in per_prompt_records]),
        "best_of_k_tolerance_accuracy": _float_or_nan(
            [float(item["best_of_k_tolerance_accuracy"]) for item in per_prompt_records]
        ),
        "parseable_answer_rate": _float_or_nan([float(item["parseable_answer"]) for item in all_sample_records]),
        "malformed_answer_rate": _float_or_nan([float(item["malformed_answer"]) for item in all_sample_records]),
        "reasoning_tag_compliance": _float_or_nan(
            [float(item["reasoning_tag_compliance"]) for item in all_sample_records]
        ),
        "solution_tag_compliance": _float_or_nan([float(item["solution_tag_compliance"]) for item in all_sample_records]),
        "truncation_rate": _float_or_nan([float(item["truncation"]) for item in all_sample_records]),
        "average_completion_tokens": _float_or_nan([float(item["completion_tokens"]) for item in all_sample_records]),
        "repetition_rate": _float_or_nan([float(item["repetition_rate"]) for item in all_sample_records]),
        "sampled_answer_diversity": _float_or_nan(
            [float(item["sampled_answer_diversity"]) for item in per_prompt_records]
        ),
        "sample_level_normalized_exact_match": _float_or_nan(
            [float(item["normalized_exact_match"]) for item in all_sample_records]
        ),
        "sample_level_tolerance_accuracy": _float_or_nan([float(item["tolerance_match"]) for item in all_sample_records]),
        "average_total_reward": _float_or_nan([float(item["total_reward"]) for item in all_sample_records]),
    }

    reward_component_keys = sorted(
        {
            key
            for record in all_sample_records
            for key in record.keys()
            if key.startswith("reward_component/")
        }
    )
    for key in reward_component_keys:
        metrics[f"{key}_mean"] = _float_or_nan([float(record[key]) for record in all_sample_records])

    return metrics


def select_overall_metrics(subset_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Choose the primary metric block from one or more evaluated subsets."""

    if "eval_overall_numeric" in subset_results:
        return dict(subset_results["eval_overall_numeric"].get("metrics", {}))
    if "eval_full_split" in subset_results:
        return dict(subset_results["eval_full_split"].get("metrics", {}))
    if len(subset_results) == 1:
        only_payload = next(iter(subset_results.values()))
        return dict(only_payload.get("metrics", {}))
    return {}


def _sample_reward_components(
    reward_funcs: list[Callable[..., list[float]]],
    reward_weights: Mapping[str, float],
    sample_kwargs: Mapping[str, list[Any]],
) -> tuple[dict[str, float], float]:
    component_scores = {}
    total = 0.0
    for reward_func in reward_funcs:
        name = reward_func.__name__
        score = float(reward_func(**sample_kwargs)[0])
        component_scores[f"reward_component/{name}"] = score
        total += score * reward_weights.get(name, 0.0)
    return component_scores, total


def evaluate_dataset_subset(
    model: Any,
    dataset: Any,
    lora_path: str,
    runtime: RewardRuntimeContext,
    reward_funcs: list[Callable[..., list[float]]],
    reward_weights: Mapping[str, float],
    eval_config: EvalConfig,
) -> dict[str, Any]:
    """Evaluate one dataset subset and return records plus aggregate metrics."""

    from vllm import SamplingParams  # pylint: disable=import-error

    eval_limit = eval_config.max_eval_examples_per_subset
    total = len(dataset) if eval_limit is None else min(len(dataset), eval_limit)
    subset = dataset.select(range(total))

    sampling_params = SamplingParams(
        temperature=eval_config.temperature,
        top_k=eval_config.top_k,
        max_tokens=runtime.max_completion_length,
        n=eval_config.num_samples_per_prompt,
    )

    lora_request = model.load_lora(lora_path)
    per_prompt_records: list[dict[str, Any]] = []
    all_sample_records: list[dict[str, Any]] = []

    for prompt_index in range(len(subset)):
        prompt = subset[prompt_index]["prompt"]
        prompt_messages = subset[prompt_index].get("prompt_messages")
        image = subset[prompt_index]["image"]
        gold_answer = subset[prompt_index]["answer"]
        answer_mode = subset[prompt_index]["answer_mode"]
        precision = subset[prompt_index].get("precision")
        choices = subset[prompt_index].get("choices")

        outputs = model.fast_generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params,
            lora_request=lora_request,
        )
        sampled_texts = [item.text for item in outputs[0].outputs]

        prompt_record = {
            "prompt_index": prompt_index,
            "pid": subset[prompt_index].get("pid"),
            "stage_name": subset[prompt_index].get("stage_name"),
            "answer_mode": answer_mode,
            "gold_answer": str(gold_answer),
            "question_type": subset[prompt_index].get("question_type"),
            "answer_type": subset[prompt_index].get("answer_type"),
            "source": subset[prompt_index].get("source"),
            "context": subset[prompt_index].get("context"),
            "context_family": subset[prompt_index].get("context_family"),
            "skills": subset[prompt_index].get("skills"),
            "samples": [],
        }

        normalized_candidates = []
        prompt_exact_hits = []
        prompt_tolerance_hits = []

        for sample_index, completion in enumerate(sampled_texts):
            solution_text = extract_single_solution_text(completion)
            if answer_mode == "multi_choice":
                parsed_answer = extract_multichoice_option_letter(completion)
                malformed = malformed_multichoice_answer(completion)
                parseable = parsed_answer is not None
                exact = parseable and parsed_answer == subset[prompt_index].get("gold_option_letter")
                tolerance_ok = exact
            else:
                parsed_answer = parse_numeric_solution(completion)
                malformed = malformed_numeric_answer(completion)
                parseable = parsed_answer is not None and parse_float_safe(parsed_answer) is not None
                exact = solution_text is not None and normalized_exact_match(solution_text, gold_answer)
                if precision is None:
                    tolerance_ok = solution_text is not None and tolerance_match(
                        solution_text,
                        gold_answer,
                        abs_tol=eval_config.abs_tol_default,
                        rel_tol=eval_config.rel_tol_default,
                    )
                else:
                    abs_tol = 10 ** (-int(precision))
                    rel_tol = max(abs_tol, eval_config.rel_tol_default)
                    tolerance_ok = solution_text is not None and tolerance_match(
                        solution_text,
                        gold_answer,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                    )

            completion_tokens = runtime.completion_token_count(completion)
            truncated = infer_truncation(
                completion,
                completion_tokens=completion_tokens,
                max_completion_length=runtime.max_completion_length,
                answer_mode=answer_mode,
            )
            sample_kwargs = {
                "prompts": [prompt_messages if prompt_messages is not None else prompt],
                "completions": [completion],
                "answer": [gold_answer],
                "answer_mode": [answer_mode],
                "precision": [precision],
                "choices": [choices],
            }
            reward_components, total_reward = _sample_reward_components(
                reward_funcs,
                reward_weights=reward_weights,
                sample_kwargs=sample_kwargs,
            )

            sample_record = {
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "completion": completion if eval_config.save_full_completion_text else None,
                "solution_text": solution_text,
                "parsed_answer": parsed_answer,
                "gold_answer": str(gold_answer),
                "normalized_exact_match": exact,
                "tolerance_match": tolerance_ok,
                "parseable_answer": parseable,
                "solution_tag_compliance": solution_tag_compliant(completion),
                "reasoning_tag_compliance": reasoning_tag_compliant(completion),
                "malformed_answer": malformed,
                "truncation": truncated,
                "completion_tokens": completion_tokens,
                "repetition_rate": compute_repetition_rate(completion),
                "answer_mode": answer_mode,
                "failure_mode": "",
                "total_reward": total_reward,
            }
            sample_record.update(reward_components)
            sample_record["failure_mode"] = determine_failure_mode(sample_record)

            prompt_record["samples"].append(sample_record)
            all_sample_records.append(sample_record)

            if parsed_answer is not None:
                normalized_candidates.append(parsed_answer)
            prompt_exact_hits.append(exact)
            prompt_tolerance_hits.append(tolerance_ok)

        unique_answers = len({candidate for candidate in normalized_candidates if candidate is not None})
        prompt_record["best_of_k_accuracy"] = any(prompt_exact_hits)
        prompt_record["best_of_k_tolerance_accuracy"] = any(prompt_tolerance_hits)
        prompt_record["sampled_answer_diversity"] = unique_answers / max(len(sampled_texts), 1)
        per_prompt_records.append(prompt_record)

    metrics = aggregate_subset_metrics(per_prompt_records, all_sample_records)
    return {
        "metrics": metrics,
        "per_prompt_records": per_prompt_records,
        "all_sample_records": all_sample_records,
    }


def evaluate_checkpoint(
    model: Any,
    eval_datasets: Mapping[str, Any],
    lora_path: str,
    runtime: RewardRuntimeContext,
    reward_funcs: list[Callable[..., list[float]]],
    reward_weights: Mapping[str, float],
    eval_config: EvalConfig,
) -> dict[str, Any]:
    """Evaluate all checkpoint subsets."""

    subset_results = {}
    for subset_name, subset in eval_datasets.items():
        subset_results[subset_name] = evaluate_dataset_subset(
            model=model,
            dataset=subset,
            lora_path=lora_path,
            runtime=runtime,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            eval_config=eval_config,
        )

    overall_metrics = select_overall_metrics(subset_results)
    subset_metrics = {name: payload["metrics"] for name, payload in subset_results.items()}
    return {
        "metrics": overall_metrics,
        "subset_metrics": subset_metrics,
        "subset_results": subset_results,
    }


def save_json_lines(records: list[dict[str, Any]], output_path) -> None:
    """Write JSONL diagnostics."""

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
