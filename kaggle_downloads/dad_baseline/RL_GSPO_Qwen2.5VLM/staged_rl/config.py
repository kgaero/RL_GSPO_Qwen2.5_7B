"""Typed configuration for staged RL training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


def _tuple(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(values)


@dataclass
class DatasetFilterSpec:
    """Declarative dataset filter for MathVista metadata."""

    question_types: tuple[str, ...] = ()
    answer_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    contexts: tuple[str, ...] = ()
    context_families: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    grades: tuple[str, ...] = ()
    skills_any: tuple[str, ...] = ()
    skills_all: tuple[str, ...] = ()
    precision_values: tuple[float, ...] = ()
    precision_min: Optional[float] = None
    precision_max: Optional[float] = None
    require_unit: Optional[bool] = None
    unit_values: tuple[str, ...] = ()
    answer_modes: tuple[str, ...] = ()


@dataclass
class StageSpec:
    """Named curriculum subset."""

    name: str
    description: str
    answer_mode: str
    filter_spec: DatasetFilterSpec
    priority_context_families: tuple[str, ...] = ()
    priority_contexts: tuple[str, ...] = ()
    priority_skills: tuple[str, ...] = ()
    priority_grades: tuple[str, ...] = ()
    hard_only: bool = False
    enabled: bool = True


@dataclass
class RewardComponentConfig:
    """Weight bounds for a single reward component."""

    name: str
    initial_weight: float
    min_weight: float
    max_weight: float
    enabled: bool = True


@dataclass
class RewardGateConfig:
    """Metric-gated reward control thresholds."""

    parseable_floor_threshold: float = 0.85
    parseable_stable_threshold: float = 0.92
    solution_tag_floor_threshold: float = 0.90
    solution_tag_stable_threshold: float = 0.95
    reasoning_tag_floor_threshold: float = 0.88
    reasoning_tag_stable_threshold: float = 0.93
    malformed_ceiling_threshold: float = 0.10
    malformed_stable_threshold: float = 0.05
    truncation_ceiling_threshold: float = 0.15
    truncation_stable_threshold: float = 0.08
    average_token_fraction_threshold: float = 0.85
    exact_match_plateau_delta: float = 0.02
    stable_window: int = 2
    correctness_step: float = 0.50
    finish_step: float = 0.25
    parseable_guard_weight: float = 0.75
    formatting_guard_weight: float = 0.75
    format_floor: float = 0.25
    parseable_floor: float = 0.25
    finish_floor: float = 0.50


@dataclass
class CheckpointScoreConfig:
    """Checkpoint ranking weights."""

    structure_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "parseable_answer_rate": 0.35,
            "solution_tag_compliance": 0.20,
            "reasoning_tag_compliance": 0.20,
            "malformed_answer_rate": -0.15,
            "truncation_rate": -0.10,
        }
    )
    correctness_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "normalized_exact_match": 0.70,
            "tolerance_accuracy": 0.20,
            "parseable_answer_rate": 0.10,
        }
    )
    composite_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "normalized_exact_match": 0.45,
            "tolerance_accuracy": 0.15,
            "parseable_answer_rate": 0.15,
            "solution_tag_compliance": 0.10,
            "reasoning_tag_compliance": 0.05,
            "malformed_answer_rate": -0.05,
            "truncation_rate": -0.05,
        }
    )


@dataclass
class ResumeSelector:
    """User-facing checkpoint selector."""

    selector: Optional[str] = None


@dataclass
class HardwareProfileSpec:
    """Named runtime profile used to adapt the same pipeline to smaller GPUs."""

    name: str
    description: str
    model_overrides: Mapping[str, Any] = field(default_factory=dict)
    trainer_overrides: Mapping[str, Any] = field(default_factory=dict)
    eval_overrides: Mapping[str, Any] = field(default_factory=dict)
    phase_trainer_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Evaluation generation settings."""

    num_samples_per_prompt: int = 4
    temperature: float = 1.0
    top_k: int = 50
    max_eval_examples_per_subset: Optional[int] = None
    max_eval_examples_overall: Optional[int] = None
    abs_tol_default: float = 1e-6
    rel_tol_default: float = 1e-6
    save_full_completion_text: bool = False


@dataclass
class ModelConfig:
    """Model loading settings."""

    base_model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct"
    max_seq_length: int = 16384
    image_size: int = 512
    load_in_4bit: bool = True
    fast_inference: bool = True
    fast_inference_kwargs: dict[str, Any] = field(default_factory=dict)
    gpu_memory_utilization: float = 0.8
    lora_rank: int = 16
    max_lora_rank: Optional[int] = None
    lora_alpha: int = 16
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict[str, Any]] = None
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class TrainerDefaults:
    """Default GRPO arguments shared across phases."""

    learning_rate: float = 3e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    log_completions: bool = False
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    temperature: float = 0.7
    num_train_epochs: float = 0.5
    save_steps: int = 60
    max_grad_norm: float = 0.1
    report_to: str = "none"
    importance_sampling_level: str = "sequence"
    mask_truncated_completions: bool = False
    loss_type: str = "dr_grpo"
    restore_callback_states_from_checkpoint: bool = True


@dataclass
class PhaseConfig:
    """Training phase configuration."""

    name: str
    description: str
    stage_mix: Mapping[str, float]
    output_subdir: str
    reward_components: Mapping[str, RewardComponentConfig]
    default_resume: ResumeSelector = field(default_factory=ResumeSelector)
    trainer_overrides: Mapping[str, Any] = field(default_factory=dict)
    enable_tolerance_reward: bool = False
    allow_multichoice_training: bool = False
    eval_stage_names: tuple[str, ...] = ()


@dataclass
class RunConfig:
    """Top-level run configuration."""

    dataset_name: str = "AI4Math/MathVista"
    train_split: str = "test"
    eval_split: str = "testmini"
    output_root: str = "outputs_staged"
    phase_name: str = "phase_a"
    hardware_profile_name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer_defaults: TrainerDefaults = field(default_factory=TrainerDefaults)
    eval: EvalConfig = field(default_factory=EvalConfig)
    reward_gate: RewardGateConfig = field(default_factory=RewardGateConfig)
    checkpoint_scores: CheckpointScoreConfig = field(default_factory=CheckpointScoreConfig)
    stages: Mapping[str, StageSpec] = field(default_factory=dict)
    phases: Mapping[str, PhaseConfig] = field(default_factory=dict)
    multichoice_scaffold_enabled: bool = False

    def output_dir_for_phase(self, phase_name: Optional[str] = None) -> Path:
        resolved_phase = phase_name or self.phase_name
        phase = self.phases[resolved_phase]
        return Path(self.output_root) / phase.output_subdir


def dataclass_to_dict(value: Any) -> Any:
    """Convert dataclasses into JSON-serializable dictionaries."""

    if is_dataclass(value):
        return {key: dataclass_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _reward_components(
    correctness: float,
    parseable: float,
    formatting: float,
    finished: float,
    brevity: float,
    tolerance: float,
) -> dict[str, RewardComponentConfig]:
    return {
        "format_reward": RewardComponentConfig("format_reward", formatting, 0.25, 2.0),
        "parseable_reward": RewardComponentConfig("parseable_reward", parseable, 0.25, 2.0),
        "finished_reward": RewardComponentConfig("finished_reward", finished, 0.50, 2.0),
        "correctness_reward": RewardComponentConfig("correctness_reward", correctness, 1.0, 8.0),
        "brevity_reward": RewardComponentConfig("brevity_reward", brevity, 0.0, 1.0, enabled=brevity > 0),
        "tolerance_reward": RewardComponentConfig("tolerance_reward", tolerance, 0.0, 2.0, enabled=tolerance > 0),
    }


def build_default_stage_specs() -> dict[str, StageSpec]:
    """Return default curriculum stage definitions."""

    return {
        "stage1_easy_numeric": StageSpec(
            name="stage1_easy_numeric",
            description="Free-form integer english subset emphasizing easy numeric tasks.",
            answer_mode="numeric_free_form",
            filter_spec=DatasetFilterSpec(
                question_types=_tuple(["free_form"]),
                answer_types=_tuple(["integer"]),
                languages=_tuple(["english"]),
                context_families=_tuple(["synthetic scene", "table", "natural image"]),
                skills_any=_tuple(["arithmetic reasoning", "statistical reasoning", "numeric commonsense"]),
                answer_modes=_tuple(["numeric_free_form"]),
            ),
            priority_context_families=_tuple(["synthetic scene", "table"]),
            priority_skills=_tuple(["arithmetic reasoning", "statistical reasoning", "numeric commonsense"]),
            priority_grades=_tuple(["elementary school", "daily life"]),
        ),
        "stage2_float_numeric": StageSpec(
            name="stage2_float_numeric",
            description="Free-form english subset with moderate precision numeric reasoning.",
            answer_mode="numeric_free_form",
            filter_spec=DatasetFilterSpec(
                question_types=_tuple(["free_form"]),
                answer_types=_tuple(["integer", "float"]),
                languages=_tuple(["english"]),
                context_families=_tuple(["table", "chart", "plot", "scientific figure", "natural image"]),
                skills_any=_tuple(
                    ["arithmetic reasoning", "statistical reasoning", "scientific reasoning", "numeric commonsense"]
                ),
                answer_modes=_tuple(["numeric_free_form"]),
            ),
            priority_context_families=_tuple(["table", "chart", "plot", "scientific figure"]),
            priority_skills=_tuple(["statistical reasoning", "scientific reasoning", "arithmetic reasoning"]),
        ),
        "stage3_hard_numeric": StageSpec(
            name="stage3_hard_numeric",
            description="Free-form english subset for harder multistep numeric reasoning.",
            answer_mode="numeric_free_form",
            filter_spec=DatasetFilterSpec(
                question_types=_tuple(["free_form"]),
                answer_types=_tuple(["integer", "float"]),
                languages=_tuple(["english"]),
                context_families=_tuple(["geometry diagram", "plot", "scientific figure", "abstract scene"]),
                skills_any=_tuple(
                    [
                        "geometry reasoning",
                        "algebraic reasoning",
                        "scientific reasoning",
                        "arithmetic reasoning",
                        "statistical reasoning",
                    ]
                ),
                answer_modes=_tuple(["numeric_free_form"]),
            ),
            priority_context_families=_tuple(["geometry diagram", "scientific figure", "plot"]),
            priority_skills=_tuple(["geometry reasoning", "algebraic reasoning", "scientific reasoning"]),
            priority_grades=_tuple(["high school", "college"]),
            hard_only=True,
        ),
        "stage4_multi_choice": StageSpec(
            name="stage4_multi_choice",
            description="Strictly separate multi-choice branch scaffold.",
            answer_mode="multi_choice",
            filter_spec=DatasetFilterSpec(
                question_types=_tuple(["multi_choice"]),
                answer_modes=_tuple(["multi_choice"]),
            ),
            priority_context_families=_tuple(["geometry diagram", "scientific figure", "plot"]),
            priority_skills=_tuple(["geometry reasoning", "algebraic reasoning"]),
        ),
        "stage5_robustness": StageSpec(
            name="stage5_robustness",
            description="Optional multilingual/noisy robustness subset.",
            answer_mode="mixed",
            filter_spec=DatasetFilterSpec(
                languages=_tuple(["chinese", "persian"]),
            ),
            enabled=False,
        ),
    }


def build_default_phase_specs() -> dict[str, PhaseConfig]:
    """Return default explicit training phases."""

    return {
        "phase_a": PhaseConfig(
            name="phase_a",
            description="Structure stabilization on Stage 1.",
            stage_mix={"stage1_easy_numeric": 1.0},
            output_subdir="phase_a",
            reward_components=_reward_components(2.0, 1.0, 1.0, 1.5, 0.25, 0.0),
            default_resume=ResumeSelector(None),
            eval_stage_names=("stage1_easy_numeric", "stage2_float_numeric", "stage3_hard_numeric"),
        ),
        "phase_b": PhaseConfig(
            name="phase_b",
            description="Correctness strengthening with Stage 1/2 mix.",
            stage_mix={"stage1_easy_numeric": 0.70, "stage2_float_numeric": 0.30},
            output_subdir="phase_b",
            reward_components=_reward_components(4.0, 0.75, 0.75, 1.0, 0.20, 0.0),
            default_resume=ResumeSelector("best_structure"),
            eval_stage_names=("stage1_easy_numeric", "stage2_float_numeric", "stage3_hard_numeric"),
        ),
        "phase_c": PhaseConfig(
            name="phase_c",
            description="Precision and harder reasoning with Stage 2/3 mix.",
            stage_mix={"stage2_float_numeric": 0.60, "stage3_hard_numeric": 0.40},
            output_subdir="phase_c",
            reward_components=_reward_components(5.0, 0.50, 0.50, 0.75, 0.20, 1.0),
            default_resume=ResumeSelector("best_composite"),
            enable_tolerance_reward=True,
            eval_stage_names=("stage1_easy_numeric", "stage2_float_numeric", "stage3_hard_numeric"),
        ),
        "phase_d": PhaseConfig(
            name="phase_d",
            description="Hard Stage 3 strengthening with longer completions allowed.",
            stage_mix={"stage3_hard_numeric": 1.0},
            output_subdir="phase_d",
            reward_components=_reward_components(5.0, 0.50, 0.50, 0.75, 0.30, 1.0),
            default_resume=ResumeSelector("best_composite"),
            trainer_overrides={"max_completion_length": 320},
            enable_tolerance_reward=True,
            eval_stage_names=("stage1_easy_numeric", "stage2_float_numeric", "stage3_hard_numeric"),
        ),
        "phase_e": PhaseConfig(
            name="phase_e",
            description="Scaffolded multi-choice branch.",
            stage_mix={"stage4_multi_choice": 1.0},
            output_subdir="phase_e",
            reward_components=_reward_components(4.0, 0.75, 0.75, 1.0, 0.20, 0.0),
            default_resume=ResumeSelector("best_composite"),
            allow_multichoice_training=False,
            eval_stage_names=("stage4_multi_choice",),
        ),
    }


def build_default_hardware_profiles() -> dict[str, HardwareProfileSpec]:
    """Return named hardware profiles for smaller or shared GPU environments."""

    return {
        "default": HardwareProfileSpec(
            name="default",
            description="Default profile with the standard staged RL settings.",
        ),
        "kaggle_t4": HardwareProfileSpec(
            name="kaggle_t4",
            description="Conservative single-GPU Kaggle T4 profile with shorter completions and lighter eval.",
            model_overrides={
                "max_seq_length": 1280,
                "image_size": 336,
                "gpu_memory_utilization": 0.65,
                "lora_rank": 8,
                "max_lora_rank": 8,
                "lora_alpha": 8,
                "fast_inference_kwargs": {
                    "compilation_config": {
                        "level": 3,
                        "cudagraph_mode": "PIECEWISE",
                    },
                },
            },
            trainer_overrides={
                "gradient_accumulation_steps": 4,
                "num_generations": 2,
                "max_prompt_length": 320,
                "max_completion_length": 64,
            },
            eval_overrides={
                "num_samples_per_prompt": 1,
                "max_eval_examples_per_subset": 2,
            },
            phase_trainer_overrides={
                "phase_d": {"max_completion_length": 96},
            },
        ),
    }


def _apply_simple_overrides(target: Any, overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        setattr(target, key, value)


def apply_hardware_profile(run_config: RunConfig, profile_name: Optional[str]) -> RunConfig:
    """Apply named hardware overrides while keeping CLI-specific overrides possible afterwards."""

    resolved_name = profile_name or "default"
    profiles = build_default_hardware_profiles()
    if resolved_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown hardware profile '{resolved_name}'. Available profiles: {available}")

    profile = profiles[resolved_name]
    run_config.hardware_profile_name = profile.name
    _apply_simple_overrides(run_config.model, profile.model_overrides)
    _apply_simple_overrides(run_config.trainer_defaults, profile.trainer_overrides)
    _apply_simple_overrides(run_config.eval, profile.eval_overrides)
    for phase_name, overrides in profile.phase_trainer_overrides.items():
        if phase_name not in run_config.phases:
            continue
        merged = dict(run_config.phases[phase_name].trainer_overrides)
        merged.update(dict(overrides))
        run_config.phases[phase_name].trainer_overrides = merged
    return run_config


def build_default_run_config(phase_name: str = "phase_a") -> RunConfig:
    """Build the default configuration used by the refactored runner."""

    stages = build_default_stage_specs()
    phases = build_default_phase_specs()
    return RunConfig(
        phase_name=phase_name,
        stages=stages,
        phases=phases,
        multichoice_scaffold_enabled=True,
    )
