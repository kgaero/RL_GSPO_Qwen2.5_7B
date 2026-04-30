"""Checkpoint artifact writing, ranking, and alias resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .config import CheckpointScoreConfig
from .evaluation import save_json_lines


BEST_ALIASES = ("latest", "best_structure", "best_correctness", "best_composite")


@dataclass
class ResumePlan:
    """Resolved checkpoint plan for a phase."""

    model_load_path: Optional[str]
    trainer_resume_path: Optional[str]
    adapter_warm_start_path: Optional[str]
    selector: Optional[str]
    phase_name: Optional[str]


def compute_weighted_score(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Compute a weighted checkpoint score."""

    total = 0.0
    for metric_name, weight in weights.items():
        total += float(metrics.get(metric_name, 0.0)) * float(weight)
    return total


def compute_checkpoint_scores(metrics: Mapping[str, float], config: CheckpointScoreConfig) -> dict[str, float]:
    """Compute structure/correctness/composite scores."""

    return {
        "structure_score": compute_weighted_score(metrics, config.structure_weights),
        "correctness_score": compute_weighted_score(metrics, config.correctness_weights),
        "composite_score": compute_weighted_score(metrics, config.composite_weights),
    }


class CheckpointRegistry:
    """Run-local checkpoint registry with best-alias tracking."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.registry_path = self.run_dir / "checkpoint_registry.json"
        self.alias_dir = self.run_dir / "aliases"
        self.data = {
            "checkpoints": [],
            "aliases": {alias: None for alias in BEST_ALIASES},
        }
        self.load()

    def load(self) -> None:
        """Load an existing registry if present."""

        if self.registry_path.exists():
            self.data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self.data.setdefault("checkpoints", [])
            aliases = self.data.setdefault("aliases", {})
            for alias in BEST_ALIASES:
                aliases.setdefault(alias, None)

    def save(self) -> None:
        """Persist the registry and alias files."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.alias_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        for alias, payload in self.data["aliases"].items():
            (self.alias_dir / f"{alias}.json").write_text(
                json.dumps(payload, indent=2) if payload is not None else "null",
                encoding="utf-8",
            )

    def _best_entry(self, score_key: str) -> Optional[dict[str, Any]]:
        entries = [entry for entry in self.data["checkpoints"] if entry["scores"].get(score_key) is not None]
        if not entries:
            return None
        return max(entries, key=lambda entry: entry["scores"][score_key])

    def register(self, entry: Mapping[str, Any]) -> None:
        """Append a checkpoint entry and refresh aliases."""

        self.data["checkpoints"] = [
            item for item in self.data["checkpoints"] if item.get("checkpoint_path") != entry.get("checkpoint_path")
        ]
        self.data["checkpoints"].append(dict(entry))
        self.data["checkpoints"].sort(key=lambda item: item.get("global_step", 0))
        self.data["aliases"]["latest"] = dict(entry)

        structure = self._best_entry("structure_score")
        correctness = self._best_entry("correctness_score")
        composite = self._best_entry("composite_score")
        self.data["aliases"]["best_structure"] = structure
        self.data["aliases"]["best_correctness"] = correctness
        self.data["aliases"]["best_composite"] = composite
        self.save()

    def resolve(self, selector: Optional[str]) -> Optional[dict[str, Any]]:
        """Resolve an alias to its registry entry."""

        if selector is None:
            return None
        if selector in self.data["aliases"]:
            return self.data["aliases"][selector]
        for entry in self.data["checkpoints"]:
            if Path(entry["checkpoint_path"]).name == selector:
                return entry
        return None


def write_checkpoint_artifacts(
    checkpoint_dir: Path,
    eval_results: Mapping[str, Any],
    reward_weights: Mapping[str, float],
    controller_state: Mapping[str, Any],
    checkpoint_info: Mapping[str, Any],
    score_config: CheckpointScoreConfig,
) -> dict[str, Any]:
    """Write checkpoint-side evaluation artifacts and return a registry entry."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(eval_results["metrics"])
    subset_metrics = dict(eval_results["subset_metrics"])
    scores = compute_checkpoint_scores(metrics, score_config)
    metrics.update(scores)

    (checkpoint_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (checkpoint_dir / "subset_metrics.json").write_text(json.dumps(subset_metrics, indent=2), encoding="utf-8")
    (checkpoint_dir / "reward_weights.json").write_text(json.dumps(dict(reward_weights), indent=2), encoding="utf-8")
    (checkpoint_dir / "controller_state.json").write_text(json.dumps(dict(controller_state), indent=2), encoding="utf-8")

    subset_results = eval_results.get("subset_results", {})
    prompt_payload = {
        subset_name: subset_result.get("per_prompt_records", [])
        for subset_name, subset_result in subset_results.items()
    }
    (checkpoint_dir / "per_prompt_records.json").write_text(
        json.dumps(prompt_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sample_records = []
    for subset_name, subset_result in subset_results.items():
        for sample in subset_result.get("all_sample_records", []):
            sample_with_subset = dict(sample)
            sample_with_subset["subset_name"] = subset_name
            sample_records.append(sample_with_subset)
    save_json_lines(sample_records, checkpoint_dir / "per_sample_records.jsonl")

    checkpoint_info_payload = dict(checkpoint_info)
    checkpoint_info_payload["metrics"] = metrics
    checkpoint_info_payload["scores"] = scores
    (checkpoint_dir / "checkpoint_info.json").write_text(
        json.dumps(checkpoint_info_payload, indent=2),
        encoding="utf-8",
    )

    summary_text = build_summary_text(metrics, reward_weights, checkpoint_info_payload)
    (checkpoint_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    return {
        "checkpoint_path": str(checkpoint_dir),
        "global_step": checkpoint_info.get("global_step"),
        "phase_name": checkpoint_info.get("phase_name"),
        "selector_phase_name": checkpoint_info.get("selector_phase_name"),
        "metrics": metrics,
        "scores": scores,
    }


def build_summary_text(metrics: Mapping[str, Any], reward_weights: Mapping[str, float], checkpoint_info: Mapping[str, Any]) -> str:
    """Create a short human-readable checkpoint summary."""

    lines = [
        f"Phase: {checkpoint_info.get('phase_name')}",
        f"Global step: {checkpoint_info.get('global_step')}",
        f"Checkpoint: {checkpoint_info.get('checkpoint_path', '')}",
        "",
        "Metrics:",
        f"  normalized_exact_match: {metrics.get('normalized_exact_match', float('nan')):.4f}",
        f"  tolerance_accuracy: {metrics.get('tolerance_accuracy', float('nan')):.4f}",
        f"  parseable_answer_rate: {metrics.get('parseable_answer_rate', float('nan')):.4f}",
        f"  solution_tag_compliance: {metrics.get('solution_tag_compliance', float('nan')):.4f}",
        f"  reasoning_tag_compliance: {metrics.get('reasoning_tag_compliance', float('nan')):.4f}",
        f"  malformed_answer_rate: {metrics.get('malformed_answer_rate', float('nan')):.4f}",
        f"  truncation_rate: {metrics.get('truncation_rate', float('nan')):.4f}",
        f"  average_completion_tokens: {metrics.get('average_completion_tokens', float('nan')):.2f}",
        "",
        "Scores:",
        f"  structure_score: {metrics.get('structure_score', float('nan')):.4f}",
        f"  correctness_score: {metrics.get('correctness_score', float('nan')):.4f}",
        f"  composite_score: {metrics.get('composite_score', float('nan')):.4f}",
        "",
        "Reward weights:",
    ]
    for name, weight in reward_weights.items():
        lines.append(f"  {name}: {weight:.4f}")
    return "\n".join(lines)


def read_checkpoint_info(checkpoint_path: Path) -> dict[str, Any]:
    """Read checkpoint-side info if present."""

    info_path = Path(checkpoint_path) / "checkpoint_info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def resolve_selector(selector: Optional[str], search_dirs: Sequence[Path]) -> Optional[dict[str, Any]]:
    """Resolve a selector across one or more phase output dirs."""

    if selector is None:
        return None

    candidate = Path(selector)
    if candidate.exists():
        info = read_checkpoint_info(candidate)
        return {
            "checkpoint_path": str(candidate),
            "phase_name": info.get("phase_name"),
            "metrics": info.get("metrics", {}),
            "scores": info.get("scores", {}),
        }

    for run_dir in search_dirs:
        registry = CheckpointRegistry(run_dir)
        resolved = registry.resolve(selector)
        if resolved is not None:
            return resolved
    return None


def build_resume_plan(
    selector: Optional[str],
    current_phase: str,
    current_phase_dir: Path,
    search_dirs: Sequence[Path],
    default_model_name: str,
    force_warm_start: bool = False,
) -> ResumePlan:
    """Resolve whether a selector should be treated as warm-start or trainer resume."""

    resolved = resolve_selector(selector, search_dirs)
    if resolved is None:
        return ResumePlan(
            model_load_path=default_model_name,
            trainer_resume_path=None,
            adapter_warm_start_path=None,
            selector=selector,
            phase_name=None,
        )

    checkpoint_path = str(resolved["checkpoint_path"])
    checkpoint_phase = resolved.get("phase_name")
    if not force_warm_start and (selector == "latest" or (checkpoint_phase is not None and checkpoint_phase == current_phase)):
        return ResumePlan(
            model_load_path=default_model_name,
            trainer_resume_path=checkpoint_path,
            adapter_warm_start_path=None,
            selector=selector,
            phase_name=checkpoint_phase,
        )

    return ResumePlan(
        model_load_path=default_model_name,
        trainer_resume_path=None,
        adapter_warm_start_path=checkpoint_path,
        selector=selector,
        phase_name=checkpoint_phase,
    )
