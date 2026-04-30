"""Runtime integration with Unsloth and TRL GRPO."""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch

from .checkpointing import CheckpointRegistry, build_resume_plan, write_checkpoint_artifacts
from .config import RunConfig, dataclass_to_dict
from .controller import RewardController
from .data import (
    analyze_dataset_records,
    build_eval_datasets,
    build_phase_train_dataset,
    dataset_to_records,
    load_mathvista_split,
    save_dataset_analysis,
)
from .diagnostics import build_post_training_diagnostics, save_json, summarize_training_logs
from .evaluation import evaluate_checkpoint
from .rewarding import RewardRuntimeContext, build_reward_functions


LOGGER = logging.getLogger(__name__)


def _install_trl_prepare_peft_workaround(trl_model_utils: Any) -> bool:
    """Patch TRL's prepare_peft_model dataclass replacement bug for GRPO configs.

    TRL versions affected by issue #3980 call `dataclasses.replace(args, gradient_checkpointing=False)`
    inside `prepare_peft_model`. GRPOConfig may already have both `generation_batch_size` and
    `steps_per_generation` populated by the time that replacement happens, which causes a second
    validation failure during reconstruction. Upstream fixed this by mutating `args.gradient_checkpointing`
    in place instead of reconstructing the dataclass. We mirror that behavior locally.
    """

    if getattr(trl_model_utils, "_staged_rl_prepare_peft_workaround_installed", False):
        return False

    dataclasses_module = getattr(trl_model_utils, "dataclasses", None)
    replace_func = getattr(dataclasses_module, "replace", None)
    if replace_func is None:
        return False

    @functools.wraps(replace_func)
    def safe_replace(obj, /, **changes):
        if (
            changes == {"gradient_checkpointing": False}
            and hasattr(obj, "generation_batch_size")
            and hasattr(obj, "steps_per_generation")
            and getattr(obj, "generation_batch_size", None) is not None
            and getattr(obj, "steps_per_generation", None) is not None
        ):
            setattr(obj, "gradient_checkpointing", False)
            return obj
        return replace_func(obj, **changes)

    proxy = SimpleNamespace(**{name: getattr(dataclasses_module, name) for name in dir(dataclasses_module)})
    proxy.replace = safe_replace
    trl_model_utils.dataclasses = proxy
    trl_model_utils._staged_rl_prepare_peft_workaround_installed = True
    return True


def patch_trl_prepare_peft_workaround() -> None:
    """Install the GRPO PEFT workaround when running against older TRL builds."""

    try:
        from trl.models import utils as trl_model_utils  # pylint: disable=import-error
    except ImportError:
        return

    if _install_trl_prepare_peft_workaround(trl_model_utils):
        LOGGER.info(
            "Installed TRL GRPO PEFT workaround: avoid dataclasses.replace(args, gradient_checkpointing=False) "
            "for configs that already materialize both generation_batch_size and steps_per_generation."
        )


def log_cuda_environment() -> None:
    """Log visible CUDA devices and warn when multiple GPUs are present but unused."""

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available. This pipeline expects a GPU-backed runtime.")
        return

    device_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    LOGGER.info("Visible CUDA devices: %s | %s", device_count, device_names)
    if device_count > 1:
        LOGGER.warning(
            "Multiple GPUs are visible, but the current runner does not configure DDP or tensor parallelism. "
            "Training and vLLM evaluation remain effectively single-GPU unless the launch path is extended."
        )


def build_component_bounds(phase_config) -> dict[str, tuple[float, float]]:
    """Return min/max bounds for every reward component."""

    return {
        name: (component.min_weight, component.max_weight)
        for name, component in phase_config.reward_components.items()
    }


def build_initial_reward_weights(phase_config) -> dict[str, float]:
    """Return the initial per-component weights for a phase."""

    return {
        name: (component.initial_weight if component.enabled else 0.0)
        for name, component in phase_config.reward_components.items()
    }


def reward_weight_list(reward_funcs, reward_weights: Mapping[str, float]) -> list[float]:
    """Return weights aligned to the reward-function order."""

    return [float(reward_weights.get(func.__name__, 0.0)) for func in reward_funcs]


def apply_reward_weights(trainer, reward_funcs, reward_weights: Mapping[str, float]) -> None:
    """Update the trainer reward tensor in-place."""

    trainer.reward_weights = torch.tensor(
        reward_weight_list(reward_funcs, reward_weights),
        dtype=torch.float32,
        device=trainer.accelerator.device if hasattr(trainer, "accelerator") else None,
    )


def _count_trainable_parameters(model: Any) -> tuple[int, int]:
    """Return trainable and total parameter counts when available."""

    if not hasattr(model, "parameters"):
        return 0, 0
    trainable = 0
    total = 0
    for param in model.parameters():
        numel = int(param.numel()) if hasattr(param, "numel") else 0
        total += numel
        if getattr(param, "requires_grad", False):
            trainable += numel
    return trainable, total


def _has_active_peft_adapters(model: Any) -> bool:
    """Return whether the model already has live PEFT adapters attached."""

    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, Mapping):
        return bool(peft_config)
    return peft_config is not None


def _configure_generation_cache_behavior(model: Any) -> dict[str, Any]:
    """Clear static-cache settings on wrapped models before GRPO generation.

    Qwen2.5-VL under Unsloth/Transformers can fail inside cache update paths when
    generation uses a statically prepared cache. We explicitly reset cache-related
    generation config on the PEFT wrapper and nested base models so GRPO uses the
    default dynamic cache path instead.
    """

    patched = []
    visited: set[int] = set()
    pending = [model]

    while pending:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        generation_config = getattr(current, "generation_config", None)
        if generation_config is not None:
            if hasattr(generation_config, "cache_implementation"):
                generation_config.cache_implementation = None
            if hasattr(generation_config, "use_cache"):
                generation_config.use_cache = True
            patched.append(type(current).__name__)

        config = getattr(current, "config", None)
        if config is not None:
            if hasattr(config, "cache_implementation"):
                config.cache_implementation = None
            if hasattr(config, "use_cache"):
                config.use_cache = True

        for attr_name in ("base_model", "model", "language_model"):
            nested = getattr(current, attr_name, None)
            if nested is not None:
                pending.append(nested)

    root_generation_config = getattr(model, "generation_config", None)
    return {
        "patched_wrappers": patched,
        "cache_implementation": getattr(root_generation_config, "cache_implementation", None),
        "use_cache": getattr(root_generation_config, "use_cache", None),
    }


def _warm_start_peft_adapter(model: Any, adapter_path: Optional[str]) -> None:
    """Load checkpoint LoRA weights into the active Unsloth PEFT wrapper.

    Cross-phase continuation must keep the model object returned by
    `FastVisionModel.get_peft_model(...)` so Unsloth's GRPO hooks still expose
    methods like `load_lora(...)` for vLLM generation. Loading a checkpoint
    directly through `from_pretrained(checkpoint_path)` returns a plain PEFT
    wrapper and loses that method. We instead attach a fresh adapter to the base
    model, then warm-start its weights from the selected checkpoint.
    """

    if not adapter_path:
        return

    checkpoint_dir = Path(adapter_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Warm-start adapter path does not exist: {checkpoint_dir}")

    errors: list[str] = []

    try:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict  # pylint: disable=import-error

        adapter_state = load_peft_weights(str(checkpoint_dir), device="cpu")
        load_result = set_peft_model_state_dict(
            model,
            adapter_state,
            adapter_name="default",
            ignore_mismatched_sizes=False,
        )
        if hasattr(model, "set_adapter"):
            model.set_adapter("default")
        LOGGER.info(
            "Warm-started adapter from %s using PEFT state-dict load. Result=%s",
            checkpoint_dir,
            load_result,
        )
        return
    except Exception as exc:  # pragma: no cover - exercised in Kaggle runtime
        errors.append(f"peft_state_dict_load_failed={exc!r}")
        LOGGER.warning("PEFT state-dict warm start failed for %s: %s", checkpoint_dir, exc)

    if hasattr(model, "load_adapter"):
        try:
            if hasattr(model, "delete_adapter"):
                try:
                    model.delete_adapter("default")
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
            model.load_adapter(str(checkpoint_dir), adapter_name="default", is_trainable=True)
            if hasattr(model, "set_adapter"):
                model.set_adapter("default")
            LOGGER.info("Warm-started adapter from %s using model.load_adapter(...).", checkpoint_dir)
            return
        except Exception as exc:  # pragma: no cover - exercised in Kaggle runtime
            errors.append(f"model_load_adapter_failed={exc!r}")
            LOGGER.warning("Adapter warm start via load_adapter failed for %s: %s", checkpoint_dir, exc)

    raise RuntimeError(
        "Unable to warm-start LoRA adapter while preserving Unsloth runtime methods. "
        f"Checkpoint={checkpoint_dir}. Errors: {'; '.join(errors) or 'none recorded'}"
    )


def create_model_and_tokenizer(
    run_config: RunConfig,
    model_name_or_path: Optional[str] = None,
    adapter_warm_start_path: Optional[str] = None,
):
    """Load the model and tokenizer, then attach LoRA if needed."""

    from unsloth import FastVisionModel  # pylint: disable=import-error

    model_config = run_config.model
    resolved_name = model_name_or_path or model_config.base_model_name
    max_lora_rank = model_config.max_lora_rank or model_config.lora_rank
    LOGGER.info(
        "Loading model %s with max_seq_length=%s, fast_inference=%s, gpu_memory_utilization=%s, max_lora_rank=%s, fast_inference_kwargs=%s",
        resolved_name,
        model_config.max_seq_length,
        model_config.fast_inference,
        model_config.gpu_memory_utilization,
        max_lora_rank,
        model_config.fast_inference_kwargs,
    )
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=resolved_name,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        fast_inference=model_config.fast_inference,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        max_lora_rank=max_lora_rank,
        **model_config.fast_inference_kwargs,
    )

    if not _has_active_peft_adapters(model):
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=model_config.finetune_vision_layers,
            finetune_language_layers=model_config.finetune_language_layers,
            finetune_attention_modules=model_config.finetune_attention_modules,
            finetune_mlp_modules=model_config.finetune_mlp_modules,
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            bias=model_config.bias,
            random_state=model_config.random_state,
            use_rslora=model_config.use_rslora,
            loftq_config=model_config.loftq_config,
            use_gradient_checkpointing=model_config.use_gradient_checkpointing,
        )
    _warm_start_peft_adapter(model, adapter_warm_start_path)
    generation_runtime = _configure_generation_cache_behavior(model)
    LOGGER.info("Generation cache runtime after model prep: %s", generation_runtime)
    trainable_params, total_params = _count_trainable_parameters(model)
    LOGGER.info("Model parameter counts after PEFT prep: trainable=%s total=%s", trainable_params, total_params)
    if total_params > 0 and trainable_params == 0:
        raise RuntimeError(
            "Model has zero trainable parameters after PEFT preparation. "
            "LoRA adapters were not attached correctly."
        )
    return model, tokenizer


def build_grpo_args(run_config: RunConfig, phase_config, reward_funcs, output_dir: Path):
    """Create GRPOConfig for the current phase."""

    from trl import GRPOConfig  # pylint: disable=import-error

    defaults = dataclass_to_dict(run_config.trainer_defaults)
    defaults.update(dict(phase_config.trainer_overrides))
    defaults["output_dir"] = str(output_dir)
    defaults["max_completion_length"] = defaults.get("max_completion_length", run_config.trainer_defaults.max_completion_length)
    defaults["reward_weights"] = reward_weight_list(reward_funcs, build_initial_reward_weights(phase_config))
    return GRPOConfig(**defaults)


class MetricAwareGRPOTrainerMixin:
    """Mixin that evaluates and ranks every checkpoint."""

    eval_datasets: Mapping[str, Any]
    reward_runtime: RewardRuntimeContext
    reward_funcs_list: list[Any]
    reward_controller: RewardController
    checkpoint_registry: CheckpointRegistry
    run_config: RunConfig
    phase_name: str
    latest_eval_results: Optional[dict[str, Any]]

    def _metric_aware_save(self, checkpoint_dir: Path) -> None:
        if not checkpoint_dir.exists():
            return

        if hasattr(self.model, "for_inference"):
            self.model.for_inference()

        eval_results = evaluate_checkpoint(
            model=self.model,
            eval_datasets=self.eval_datasets,
            lora_path=str(checkpoint_dir),
            runtime=self.reward_runtime,
            reward_funcs=self.reward_funcs_list,
            reward_weights=self.reward_controller.current_weights(),
            eval_config=self.run_config.eval,
        )

        checkpoint_entry = write_checkpoint_artifacts(
            checkpoint_dir=checkpoint_dir,
            eval_results=eval_results,
            reward_weights=self.reward_controller.current_weights(),
            controller_state=self.reward_controller.to_dict(),
            checkpoint_info={
                "checkpoint_path": str(checkpoint_dir),
                "global_step": self.state.global_step,
                "phase_name": self.phase_name,
                "selector_phase_name": self.phase_name,
            },
            score_config=self.run_config.checkpoint_scores,
        )
        self.checkpoint_registry.register(checkpoint_entry)

        updated_weights = self.reward_controller.update_from_metrics(
            eval_results["metrics"],
            max_completion_length=self.reward_runtime.max_completion_length,
        )
        apply_reward_weights(self, self.reward_funcs_list, updated_weights)
        (checkpoint_dir / "controller_decision.json").write_text(
            json.dumps(self.reward_controller.latest_decision(), indent=2),
            encoding="utf-8",
        )
        (checkpoint_dir / "reward_weights.json").write_text(json.dumps(updated_weights, indent=2), encoding="utf-8")
        (checkpoint_dir / "controller_state.json").write_text(
            json.dumps(self.reward_controller.to_dict(), indent=2),
            encoding="utf-8",
        )
        self.latest_eval_results = eval_results

        if hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=self.run_config.model.use_gradient_checkpointing)


def build_metric_trainer_class(base_cls):
    """Create the concrete trainer subclass without importing TRL at module import time."""

    class MetricAwareGRPOTrainer(MetricAwareGRPOTrainerMixin, base_cls):
        """GRPOTrainer with checkpoint-side evaluation and reward control."""

        def __init__(
            self,
            *args,
            eval_datasets,
            reward_runtime,
            reward_funcs_list,
            reward_controller,
            checkpoint_registry,
            run_config,
            phase_name,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.eval_datasets = eval_datasets
            self.reward_runtime = reward_runtime
            self.reward_funcs_list = reward_funcs_list
            self.reward_controller = reward_controller
            self.checkpoint_registry = checkpoint_registry
            self.run_config = run_config
            self.phase_name = phase_name
            self.latest_eval_results = None

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            LOGGER.info("Running checkpoint evaluation for %s", checkpoint_dir)
            self._metric_aware_save(checkpoint_dir)

    return MetricAwareGRPOTrainer


def _load_controller_state_from_checkpoint(checkpoint_path: Optional[str], current_phase: str) -> Optional[dict[str, Any]]:
    if not checkpoint_path:
        return None
    checkpoint_dir = Path(checkpoint_path)
    info_path = checkpoint_dir / "checkpoint_info.json"
    state_path = checkpoint_dir / "controller_state.json"
    if not info_path.exists() or not state_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("phase_name") != current_phase:
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_phase(
    run_config: RunConfig,
    phase_name: Optional[str] = None,
    resume_selector: Optional[str] = None,
    warm_start_selector: Optional[str] = None,
) -> dict[str, Any]:
    """Run one explicit phase of staged RL training."""

    try:
        import unsloth  # pylint: disable=unused-import,import-error
    except ImportError:
        pass
    from trl import GRPOTrainer  # pylint: disable=import-error

    patch_trl_prepare_peft_workaround()
    log_cuda_environment()

    resolved_phase = phase_name or run_config.phase_name
    phase_config = run_config.phases[resolved_phase]
    output_dir = run_config.output_dir_for_phase(resolved_phase)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(dataclass_to_dict(run_config), output_dir / "run_config.json")

    search_dirs = [run_config.output_dir_for_phase(name) for name in run_config.phases]
    selector = warm_start_selector if warm_start_selector is not None else resume_selector
    if selector is None:
        selector = phase_config.default_resume.selector
    resume_plan = build_resume_plan(
        selector=selector,
        current_phase=resolved_phase,
        current_phase_dir=output_dir,
        search_dirs=search_dirs,
        default_model_name=run_config.model.base_model_name,
        force_warm_start=warm_start_selector is not None,
    )

    model, tokenizer = create_model_and_tokenizer(
        run_config,
        model_name_or_path=resume_plan.model_load_path,
        adapter_warm_start_path=resume_plan.adapter_warm_start_path,
    )
    train_base = load_mathvista_split(run_config, run_config.train_split)
    eval_base = load_mathvista_split(run_config, run_config.eval_split)

    save_dataset_analysis(
        analyze_dataset_records(dataset_to_records(train_base), run_config.stages),
        output_dir / "dataset_analysis_train.json",
    )
    save_dataset_analysis(
        analyze_dataset_records(dataset_to_records(eval_base), run_config.stages),
        output_dir / "dataset_analysis_eval.json",
    )

    train_dataset, stage_datasets = build_phase_train_dataset(
        train_base,
        phase_config,
        run_config.stages,
        tokenizer,
        image_size=run_config.model.image_size,
    )
    eval_datasets = build_eval_datasets(eval_base, run_config, tokenizer)

    reward_runtime = RewardRuntimeContext(
        tokenizer=tokenizer,
        max_completion_length=int(
            phase_config.trainer_overrides.get(
                "max_completion_length",
                run_config.trainer_defaults.max_completion_length,
            )
        ),
        phase_config=phase_config,
    )
    reward_funcs = build_reward_functions(reward_runtime)
    initial_weights = build_initial_reward_weights(phase_config)
    controller_state = _load_controller_state_from_checkpoint(resume_plan.trainer_resume_path, resolved_phase)
    reward_controller = RewardController.from_state(
        gate_config=run_config.reward_gate,
        component_bounds=build_component_bounds(phase_config),
        initial_weights=initial_weights,
        state_dict=controller_state,
    )

    args = build_grpo_args(run_config, phase_config, reward_funcs, output_dir)
    MetricAwareGRPOTrainer = build_metric_trainer_class(GRPOTrainer)
    trainer = MetricAwareGRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        reward_runtime=reward_runtime,
        reward_funcs_list=reward_funcs,
        reward_controller=reward_controller,
        checkpoint_registry=CheckpointRegistry(output_dir),
        run_config=run_config,
        phase_name=resolved_phase,
    )
    apply_reward_weights(trainer, reward_funcs, reward_controller.current_weights())

    train_result = trainer.train(resume_from_checkpoint=resume_plan.trainer_resume_path)

    final_lora_dir = output_dir / "final_lora"
    model.save_lora(str(final_lora_dir))

    log_summary = summarize_training_logs(trainer.state.log_history)
    save_json(log_summary, output_dir / "train_log_summary.json")
    registry = CheckpointRegistry(output_dir)
    diagnostics = build_post_training_diagnostics(registry.data, trainer.latest_eval_results or {})
    save_json(diagnostics, output_dir / "post_training_diagnostics.json")

    return {
        "train_result": train_result,
        "output_dir": str(output_dir),
        "final_lora_dir": str(final_lora_dir),
        "resume_plan": dataclass_to_dict(resume_plan),
        "stage_names": list(stage_datasets.keys()),
        "latest_eval_results": trainer.latest_eval_results,
        "train_log_summary": log_summary,
        "post_training_diagnostics": diagnostics,
    }
