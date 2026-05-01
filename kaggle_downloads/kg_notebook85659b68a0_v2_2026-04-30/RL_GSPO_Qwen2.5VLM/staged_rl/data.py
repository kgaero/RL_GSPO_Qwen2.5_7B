"""Dataset loading, staging, prompt building, and metadata analysis."""

from __future__ import annotations

import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from packaging.version import Version

from .config import (
    DatasetFilterSpec,
    PhaseConfig,
    REASONING_END,
    REASONING_START,
    SUPPORTED_ANSWER_MODES,
    RunConfig,
    SOLUTION_END,
    SOLUTION_START,
    StageSpec,
    ensure_supported_answer_mode,
)
from .parsing import compute_option_letter


LOGGER = logging.getLogger(__name__)


def normalize_text_field(value: Any) -> str:
    """Normalize a metadata text field for filtering."""

    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_context_family(context: Any) -> str:
    """Collapse raw context labels into stable families."""

    normalized = normalize_text_field(context)
    if not normalized:
        return "unknown"
    if normalized.endswith("chart"):
        return "chart"
    if normalized.endswith("plot"):
        return "plot"
    return normalized


def normalize_skills(skills: Optional[Iterable[str]]) -> list[str]:
    """Normalize the skills list."""

    if not skills:
        return []
    return [normalize_text_field(skill) for skill in skills if normalize_text_field(skill)]


def determine_answer_mode(example: Mapping[str, Any]) -> str:
    """Infer the answer mode from question/answer metadata."""

    question_type = normalize_text_field(example.get("question_type"))
    answer_type = normalize_text_field(example.get("answer_type"))
    if question_type == "multi_choice":
        return "multi_choice"
    if question_type == "free_form" and answer_type in {"integer", "float"}:
        return "numeric_free_form"
    return "unsupported"


def flatten_metadata(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten the nested MathVista metadata block."""

    metadata = metadata or {}
    return {
        "category": normalize_text_field(metadata.get("category")),
        "context": normalize_text_field(metadata.get("context")),
        "context_family": normalize_context_family(metadata.get("context")),
        "grade": normalize_text_field(metadata.get("grade")),
        "language": normalize_text_field(metadata.get("language")),
        "skills": normalize_skills(metadata.get("skills")),
        "source": normalize_text_field(metadata.get("source")),
        "split": normalize_text_field(metadata.get("split")),
        "task": normalize_text_field(metadata.get("task")),
    }


def resize_and_convert_image(image: Any, image_size: int = 512) -> Any:
    """Resize and convert the decoded image to RGB."""

    if image is None:
        return None
    if hasattr(image, "resize"):
        image = image.resize((image_size, image_size))
    if getattr(image, "mode", None) != "RGB" and hasattr(image, "convert"):
        image = image.convert("RGB")
    return image


def _load_image_payload(image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> Any:
    """Best-effort decode for dataset image payloads."""

    try:
        from PIL import Image  # pylint: disable=import-error
    except Exception:
        return None

    if image_bytes:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            return opened.copy()
    if image_path:
        with Image.open(image_path) as opened:
            return opened.copy()
    return None


def normalize_image_payload(image: Any, image_size: int = 512) -> Any:
    """Normalize image payloads into processor-friendly objects."""

    if image is None:
        return None
    if isinstance(image, list):
        return [normalize_image_payload(item, image_size=image_size) for item in image]
    if hasattr(image, "resize"):
        return resize_and_convert_image(image, image_size=image_size)
    if isinstance(image, dict):
        decoded = _load_image_payload(
            image_path=image.get("path"),
            image_bytes=image.get("bytes"),
        )
        if decoded is not None:
            return resize_and_convert_image(decoded, image_size=image_size)
        if image.get("path"):
            return image["path"]
        return None
    if isinstance(image, str):
        decoded = _load_image_payload(image_path=image)
        if decoded is not None:
            return resize_and_convert_image(decoded, image_size=image_size)
        return image
    return image


def enrich_example(example: dict[str, Any], image_size: int = 512) -> dict[str, Any]:
    """Add flattened metadata fields used by filters, rewards, and diagnostics."""

    flattened = flatten_metadata(example.get("metadata"))
    updated = dict(example)
    updated.update(flattened)
    updated["question_type"] = normalize_text_field(example.get("question_type"))
    updated["answer_type"] = normalize_text_field(example.get("answer_type"))
    updated["answer_mode"] = determine_answer_mode(updated)
    updated["precision"] = example.get("precision")
    updated["unit"] = normalize_text_field(example.get("unit"))
    updated["image_path"] = example.get("image")
    updated["image"] = normalize_image_payload(
        example.get("decoded_image") if example.get("decoded_image") is not None else example.get("image"),
        image_size=image_size,
    )
    updated["gold_option_letter"] = compute_option_letter(example.get("answer"), example.get("choices"))
    return updated


def match_filter_spec(record: Mapping[str, Any], spec: DatasetFilterSpec) -> bool:
    """Return whether a flattened record matches a filter spec."""

    def _match_set(record_value: Any, values: Iterable[Any]) -> bool:
        values = tuple(values)
        if not values:
            return True
        return record_value in values

    if not _match_set(record.get("question_type"), spec.question_types):
        return False
    if not _match_set(record.get("answer_type"), spec.answer_types):
        return False
    if not _match_set(record.get("language"), spec.languages):
        return False
    if not _match_set(record.get("source"), spec.sources):
        return False
    if not _match_set(record.get("context"), spec.contexts):
        return False
    if not _match_set(record.get("context_family"), spec.context_families):
        return False
    if not _match_set(record.get("task"), spec.tasks):
        return False
    if not _match_set(record.get("category"), spec.categories):
        return False
    if not _match_set(record.get("grade"), spec.grades):
        return False
    if not _match_set(record.get("unit"), spec.unit_values):
        return False
    if not _match_set(record.get("answer_mode"), spec.answer_modes):
        return False

    precision = record.get("precision")
    if spec.precision_values and precision not in spec.precision_values:
        return False
    if spec.precision_min is not None and (precision is None or precision < spec.precision_min):
        return False
    if spec.precision_max is not None and (precision is None or precision > spec.precision_max):
        return False

    if spec.require_unit is True and not record.get("unit"):
        return False
    if spec.require_unit is False and record.get("unit"):
        return False

    skills = set(record.get("skills") or [])
    if spec.skills_any and not skills.intersection(spec.skills_any):
        return False
    if spec.skills_all and not set(spec.skills_all).issubset(skills):
        return False
    return True


def stage_priority(record: Mapping[str, Any], stage_spec: StageSpec) -> int:
    """Score a record within a stage so easier examples appear first."""

    score = 0
    if record.get("context_family") in stage_spec.priority_context_families:
        score += 3
    if record.get("context") in stage_spec.priority_contexts:
        score += 2
    if set(record.get("skills") or []).intersection(stage_spec.priority_skills):
        score += 2
    if record.get("grade") in stage_spec.priority_grades:
        score += 1
    if stage_spec.hard_only and record.get("grade") in {"high school", "college"}:
        score += 1
    return score


def build_prompt_text(record: Mapping[str, Any]) -> str:
    """Wrap the dataset query with a strict answer contract."""

    base_text = str(record.get("query") or record.get("question") or "").strip()
    answer_mode = ensure_supported_answer_mode(record.get("answer_mode"))

    if answer_mode == "multi_choice":
        contract = (
            f"{base_text}\n\n"
            "Respond using exactly the following structure.\n"
            f"{REASONING_START}\n"
            "Give concise reasoning.\n"
            f"{REASONING_END}\n"
            f"{SOLUTION_START}\n"
            "Write a single option letter only, such as A, B, C, or D.\n"
            f"{SOLUTION_END}\n"
            "Do not include text outside these tags."
        )
        return contract

    contract = (
        f"{base_text}\n\n"
        "Respond using exactly the following structure.\n"
        f"{REASONING_START}\n"
        "Give concise reasoning.\n"
        f"{REASONING_END}\n"
        f"{SOLUTION_START}\n"
        "Write a single numeric answer only.\n"
        f"{SOLUTION_END}\n"
        "Do not include text outside these tags."
    )
    return contract


def _maybe_apply_chat_template(dataset, tokenizer) -> Any:
    """Apply chat templates for older TRL versions."""

    try:
        import trl  # pylint: disable=import-error
    except Exception:
        return dataset

    if Version(getattr(trl, "__version__", "0.0.0")) >= Version("0.24.0"):
        return dataset

    return dataset.map(
        lambda example: {
            "prompt": tokenizer.apply_chat_template(
                example["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        }
    )


def _apply_runtime_image_transform(dataset, image_size: int = 512) -> Any:
    """Decode image payloads at access time so TRL sees PIL images, not dataset dicts."""

    def _transform(batch: Mapping[str, Any]) -> dict[str, Any]:
        materialized = dict(batch)
        if "image" not in materialized:
            return materialized
        images = materialized["image"]
        if isinstance(images, list):
            materialized["image"] = [normalize_image_payload(image, image_size=image_size) for image in images]
        else:
            materialized["image"] = normalize_image_payload(images, image_size=image_size)
        return materialized

    return dataset.with_transform(_transform)


def _materialize_image_column(dataset, image_size: int = 512) -> Any:
    """Eagerly normalize the image column so downstream trainers see concrete payloads."""

    def _transform(example: Mapping[str, Any]) -> dict[str, Any]:
        materialized = dict(example)
        if "image" not in materialized:
            return materialized
        normalized = normalize_image_payload(materialized["image"], image_size=image_size)
        if normalized is None:
            raise ValueError("Unable to normalize an image payload for a staged dataset example.")
        materialized["image"] = normalized
        return materialized

    dataset = dataset.map(_transform)

    cast_column = getattr(dataset, "cast_column", None)
    if cast_column is not None and "image" in getattr(dataset, "column_names", []):
        try:
            from datasets import Image  # pylint: disable=import-error

            dataset = cast_column("image", Image())
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.warning("Unable to cast materialized image column to datasets.Image: %s", exc)

    return dataset


def _assert_numeric_stage_records(dataset, stage_spec: StageSpec) -> None:
    if stage_spec.answer_mode != "numeric_free_form":
        return
    wrong_records = dataset.filter(
        lambda example: example["question_type"] != "free_form" or example["answer_type"] not in ("integer", "float")
    )
    if len(wrong_records) > 0:
        raise ValueError(f"Numeric stage {stage_spec.name} contains non-numeric or non-free-form rows.")


def _load_dataset_imports():
    try:
        from datasets import interleave_datasets, load_dataset
    except Exception as exc:  # pragma: no cover - exercised only in runtime envs with datasets missing
        raise RuntimeError("datasets is required to load MathVista splits.") from exc
    return load_dataset, interleave_datasets


def _active_training_stage_names(phase_config: PhaseConfig, stage_specs: Mapping[str, StageSpec]) -> list[str]:
    """Return the enabled stage names that should participate in a phase."""

    active_stage_names: list[str] = []
    skipped_stage_names: list[str] = []
    for stage_name in phase_config.stage_mix:
        stage_spec = stage_specs[stage_name]
        if not stage_spec.enabled:
            skipped_stage_names.append(stage_name)
            continue
        if stage_spec.answer_mode == "multi_choice" and not phase_config.allow_multichoice_training:
            raise ValueError("Phase E multi-choice training is scaffolded only and remains disabled.")
        active_stage_names.append(stage_name)

    if skipped_stage_names:
        LOGGER.info(
            "Skipping disabled training stages for phase %s: %s",
            phase_config.name,
            ", ".join(skipped_stage_names),
        )
    return active_stage_names


def _active_eval_stage_names(run_config: RunConfig) -> list[str]:
    """Return the enabled evaluation stage names for the current phase."""

    phase_config = run_config.phases[run_config.phase_name]
    requested_stage_names = list(phase_config.eval_stage_names)
    if not requested_stage_names:
        requested_stage_names = [stage_name for stage_name, stage_spec in run_config.stages.items() if stage_spec.enabled]

    active_stage_names: list[str] = []
    skipped_stage_names: list[str] = []
    for stage_name in requested_stage_names:
        stage_spec = run_config.stages[stage_name]
        if not stage_spec.enabled:
            skipped_stage_names.append(stage_name)
            continue
        active_stage_names.append(stage_name)

    if skipped_stage_names:
        LOGGER.info(
            "Skipping disabled eval stages for phase %s: %s",
            phase_config.name,
            ", ".join(skipped_stage_names),
        )
    return active_stage_names


def filter_supported_answer_mode_rows(dataset) -> Any:
    """Keep only rows that this pipeline can score reliably."""

    filtered = dataset.filter(lambda example: example.get("answer_mode") in SUPPORTED_ANSWER_MODES)
    skipped = len(dataset) - len(filtered)
    if skipped > 0:
        LOGGER.warning(
            "Skipping %s unsupported rows with answer_mode outside %s.",
            skipped,
            SUPPORTED_ANSWER_MODES,
        )
    return filtered


def load_mathvista_split(run_config: RunConfig, split_name: str):
    """Load and enrich a MathVista split."""

    load_dataset, _ = _load_dataset_imports()
    dataset = load_dataset(run_config.dataset_name, split=split_name)
    dataset = dataset.map(lambda example: enrich_example(example, image_size=run_config.model.image_size))
    return dataset


def build_stage_dataset(base_dataset, stage_spec: StageSpec, tokenizer, image_size: int = 512) -> Any:
    """Filter, score, and prompt-format a stage dataset."""

    dataset = base_dataset.filter(lambda example: match_filter_spec(example, stage_spec.filter_spec))

    def _format_example(example: Mapping[str, Any]) -> dict[str, Any]:
        prompt_text = build_prompt_text(example)
        prompt_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
        return {
            "prompt_text": prompt_text,
            "stage_name": stage_spec.name,
            "stage_priority": stage_priority(example, stage_spec),
            "prompt_messages": [prompt_message],
            "prompt": [prompt_message],
        }

    dataset = dataset.map(_format_example)
    dataset = dataset.sort("stage_priority", reverse=True)
    dataset = dataset.remove_columns([name for name in ("decoded_image",) if name in dataset.column_names])
    _assert_numeric_stage_records(dataset, stage_spec)
    dataset = _maybe_apply_chat_template(dataset, tokenizer)
    dataset = _materialize_image_column(dataset, image_size=image_size)
    return dataset


def build_phase_train_dataset(
    base_dataset,
    phase_config: PhaseConfig,
    stage_specs: Mapping[str, StageSpec],
    tokenizer,
    image_size: int = 512,
):
    """Build the training dataset for a phase."""

    _, interleave_datasets = _load_dataset_imports()
    active_stage_names = _active_training_stage_names(phase_config, stage_specs)
    stage_datasets = {}
    for stage_name in active_stage_names:
        stage_spec = stage_specs[stage_name]
        stage_datasets[stage_name] = build_stage_dataset(
            base_dataset,
            stage_spec,
            tokenizer,
            image_size=image_size,
        )

    if not stage_datasets:
        raise ValueError(f"No enabled training stages remain for phase {phase_config.name}.")
    if len(stage_datasets) == 1:
        only_stage = next(iter(stage_datasets.values()))
        return only_stage, stage_datasets

    probabilities = [phase_config.stage_mix[name] for name in stage_datasets.keys()]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    mixed_dataset = interleave_datasets(
        list(stage_datasets.values()),
        probabilities=probabilities,
        seed=3407,
        stopping_strategy="all_exhausted",
    )
    mixed_dataset = _materialize_image_column(mixed_dataset, image_size=image_size)
    return mixed_dataset, stage_datasets


def build_eval_datasets(base_eval_dataset, run_config: RunConfig, tokenizer) -> dict[str, Any]:
    """Build all default evaluation subsets."""

    eval_datasets = {}
    numeric_filter = DatasetFilterSpec(
        question_types=("free_form",),
        answer_types=("integer", "float"),
        languages=("english",),
        answer_modes=("numeric_free_form",),
    )
    numeric_stage = StageSpec(
        name="eval_overall_numeric",
        description="Overall free-form numeric english evaluation subset.",
        answer_mode="numeric_free_form",
        filter_spec=numeric_filter,
    )
    eval_datasets[numeric_stage.name] = build_stage_dataset(
        base_eval_dataset,
        numeric_stage,
        tokenizer,
        image_size=run_config.model.image_size,
    )

    for stage_name in _active_eval_stage_names(run_config):
        stage_spec = run_config.stages[stage_name]
        eval_datasets[stage_name] = build_stage_dataset(
            base_eval_dataset,
            stage_spec,
            tokenizer,
            image_size=run_config.model.image_size,
        )
    return eval_datasets


def dataset_to_records(dataset, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Convert a datasets Dataset into a list of dicts."""

    total = len(dataset) if limit is None else min(limit, len(dataset))
    return [dataset[index] for index in range(total)]


def analyze_dataset_records(records: Iterable[Mapping[str, Any]], stage_specs: Mapping[str, StageSpec]) -> dict[str, Any]:
    """Compute dataset-level diagnostics and stage recommendations."""

    field_counts = {
        "question_type": Counter(),
        "answer_type": Counter(),
        "answer_mode": Counter(),
        "language": Counter(),
        "context": Counter(),
        "context_family": Counter(),
        "source": Counter(),
        "task": Counter(),
        "category": Counter(),
        "grade": Counter(),
        "skills": Counter(),
        "precision_bucket": Counter(),
        "unit_bucket": Counter(),
    }

    stage_state = {
        stage_name: {
            "count": 0,
            "top_context_families": Counter(),
            "top_sources": Counter(),
            "examples": [],
        }
        for stage_name in stage_specs
    }

    total_rows = 0
    for record in records:
        total_rows += 1
        field_counts["question_type"][record.get("question_type")] += 1
        field_counts["answer_type"][record.get("answer_type")] += 1
        field_counts["answer_mode"][record.get("answer_mode")] += 1
        field_counts["language"][record.get("language")] += 1
        field_counts["context"][record.get("context")] += 1
        field_counts["context_family"][record.get("context_family")] += 1
        field_counts["source"][record.get("source")] += 1
        field_counts["task"][record.get("task")] += 1
        field_counts["category"][record.get("category")] += 1
        field_counts["grade"][record.get("grade")] += 1
        field_counts["skills"].update(skill for skill in (record.get("skills") or []))
        field_counts["precision_bucket"]["present" if record.get("precision") not in (None, "") else "missing"] += 1
        field_counts["unit_bucket"]["present" if record.get("unit") else "missing"] += 1

        for stage_name, stage_spec in stage_specs.items():
            if not match_filter_spec(record, stage_spec.filter_spec):
                continue
            state = stage_state[stage_name]
            state["count"] += 1
            state["top_context_families"][record.get("context_family")] += 1
            state["top_sources"][record.get("source")] += 1
            if len(state["examples"]) < 3:
                state["examples"].append(
                    {
                        "pid": record.get("pid"),
                        "question_type": record.get("question_type"),
                        "answer_type": record.get("answer_type"),
                        "answer_mode": record.get("answer_mode"),
                        "context": record.get("context"),
                        "source": record.get("source"),
                        "skills": record.get("skills"),
                        "answer": record.get("answer"),
                        "question": str(record.get("question", ""))[:180],
                    }
                )

    stage_summaries = {}
    warnings = []
    for stage_name, state in stage_state.items():
        heterogeneity = state["top_context_families"]
        stage_summaries[stage_name] = {
            "count": state["count"],
            "recommended_train_size": int(state["count"] * 0.8),
            "recommended_eval_size": int(state["count"] * 0.2),
            "top_context_families": heterogeneity.most_common(5),
            "top_sources": state["top_sources"].most_common(5),
            "examples": list(state["examples"]),
        }
        if state["count"] < 32:
            warnings.append(f"Stage {stage_name} is very small ({state['count']} examples).")
        if len(heterogeneity) > 6:
            warnings.append(f"Stage {stage_name} is heterogeneous across {len(heterogeneity)} context families.")

    return {
        "total_rows": total_rows,
        "field_counts": {key: dict(counter) for key, counter in field_counts.items()},
        "stage_summaries": stage_summaries,
        "warnings": warnings,
    }


def save_dataset_analysis(analysis: Mapping[str, Any], output_path: Path) -> None:
    """Write dataset diagnostics to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
