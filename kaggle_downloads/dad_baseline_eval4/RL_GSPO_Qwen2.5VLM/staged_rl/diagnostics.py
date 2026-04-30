"""Dataset-independent training diagnostics."""

from __future__ import annotations

import datetime as dt
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def summarize_training_logs(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize numeric trainer logs, including KL statistics."""

    numeric_by_key: dict[str, list[float]] = defaultdict(list)
    for row in log_history:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_by_key[key].append(float(value))

    kl_candidate_keys = [key for key in numeric_by_key if "kl" in key.lower()]
    summary = {
        "available_numeric_log_keys": sorted(numeric_by_key.keys()),
        "kl_key_used": None,
        "KL_mean": float("nan"),
        "KL_p95": float("nan"),
    }
    if kl_candidate_keys:
        best_key = max(kl_candidate_keys, key=lambda key: len(numeric_by_key[key]))
        values = numeric_by_key[best_key]
        sorted_values = sorted(values)
        summary["kl_key_used"] = best_key
        summary["KL_mean"] = sum(values) / max(len(values), 1)
        summary["KL_p95"] = sorted_values[int(0.95 * (len(sorted_values) - 1))] if sorted_values else float("nan")
    return summary


def build_post_training_diagnostics(
    registry_data: Mapping[str, Any],
    eval_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize failure modes and checkpoint ranking behavior."""

    checkpoint_entries = registry_data.get("checkpoints", [])
    latest_subset = eval_results.get("subset_results", {}).get("eval_overall_numeric", {})
    all_sample_records = latest_subset.get("all_sample_records", [])

    failure_counts = Counter(record.get("failure_mode") for record in all_sample_records)
    parseable_series = [entry["metrics"].get("parseable_answer_rate", 0.0) for entry in checkpoint_entries]
    exact_series = [entry["metrics"].get("normalized_exact_match", 0.0) for entry in checkpoint_entries]
    structure_series = [entry["scores"].get("structure_score", 0.0) for entry in checkpoint_entries]

    parseability_improved_first = False
    if len(parseable_series) >= 2 and len(exact_series) >= 2:
        parseability_improved_first = (parseable_series[1] - parseable_series[0]) > (exact_series[1] - exact_series[0])

    structure_regression_checkpoints = []
    for previous, current in zip(checkpoint_entries, checkpoint_entries[1:]):
        if (
            current["metrics"].get("normalized_exact_match", 0.0) > previous["metrics"].get("normalized_exact_match", 0.0)
            and current["metrics"].get("solution_tag_compliance", 1.0) < previous["metrics"].get("solution_tag_compliance", 1.0)
        ):
            structure_regression_checkpoints.append(current["checkpoint_path"])

    aliases = registry_data.get("aliases", {})
    ranking_differences = {
        alias: payload.get("checkpoint_path") if payload else None
        for alias, payload in aliases.items()
    }

    return {
        "dominant_failure_modes": failure_counts.most_common(),
        "parseability_improved_before_correctness": parseability_improved_first,
        "structure_regression_checkpoints": structure_regression_checkpoints,
        "checkpoint_ranking_differences": ranking_differences,
        "structure_score_series": structure_series,
        "parseable_answer_rate_series": parseable_series,
        "normalized_exact_match_series": exact_series,
    }


def save_json(data: Mapping[str, Any], output_path: Path) -> None:
    """Save a mapping to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_fatal_error(output_path: Path, exc: BaseException, context: Mapping[str, Any]) -> None:
    """Persist a fatal exception with traceback and run context."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    payload = [
        f"timestamp_utc: {timestamp}",
        f"exception_type: {type(exc).__name__}",
        f"exception_message: {exc}",
        "",
        "context:",
        json.dumps(dict(context), indent=2, ensure_ascii=False, default=str),
        "",
        "traceback:",
        trace,
    ]
    output_path.write_text("\n".join(payload), encoding="utf-8")
