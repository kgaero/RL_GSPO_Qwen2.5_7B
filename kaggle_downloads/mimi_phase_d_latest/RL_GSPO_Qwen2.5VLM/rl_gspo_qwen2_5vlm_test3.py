# -*- coding: utf-8 -*-
"""Notebook-friendly entrypoint for staged metric-gated GRPO training."""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from staged_rl.config import (
    apply_hardware_profile,
    build_default_hardware_profiles,
    build_default_run_config,
    ensure_supported_answer_mode,
)
from staged_rl.data import analyze_dataset_records, load_mathvista_split, save_dataset_analysis
from staged_rl.diagnostics import save_json, write_fatal_error


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure process-wide logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags while staying notebook-friendly."""

    parser = argparse.ArgumentParser(description="Run staged metric-gated GRPO training for Qwen2.5-VL.")
    parser.add_argument("--phase", default="phase_a", help="Phase to run: phase_a, phase_b, phase_c, phase_d, phase_e.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint selector: latest, best_structure, best_correctness, best_composite, checkpoint-* name, or path.",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        default=None,
        help="Explicit checkpoint path or selector to load adapter weights from without restoring trainer state.",
    )
    parser.add_argument("--output-root", default=None, help="Override the default output root.")
    parser.add_argument("--train-split", default=None, help="Override the training split name.")
    parser.add_argument("--eval-split", default=None, help="Override the evaluation split name.")
    parser.add_argument(
        "--hardware-profile",
        default="default",
        choices=sorted(build_default_hardware_profiles().keys()),
        help="Runtime profile for smaller GPUs, for example kaggle_t4.",
    )
    parser.add_argument(
        "--max-eval-examples-per-subset",
        type=int,
        default=None,
        help="Limit evaluation examples per subset for a lighter checkpoint pass.",
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Override the base model path or repo id (use a local path to avoid downloads).",
    )
    parser.add_argument(
        "--dataset-analysis-only",
        action="store_true",
        help="Only load the train/eval splits and save the dataset analysis outputs.",
    )
    parser.add_argument(
        "--disable-stage",
        action="append",
        default=[],
        help="Disable a named stage when building datasets and analysis outputs.",
    )
    parser.add_argument(
        "--enable-stage",
        action="append",
        default=[],
        help="Enable a named stage that is disabled by default, if its answer mode is supported by this pipeline.",
    )
    parser.add_argument(
        "--enable-multichoice-training",
        action="store_true",
        help="Allow Phase E multi-choice training. This stays disabled by default.",
    )
    return parser.parse_args()


def apply_cli_overrides(run_config, args: argparse.Namespace):
    """Update the default run config from CLI overrides."""

    run_config.phase_name = args.phase
    run_config = apply_hardware_profile(run_config, args.hardware_profile)
    if args.output_root:
        run_config.output_root = args.output_root
    if args.train_split:
        run_config.train_split = args.train_split
    if args.eval_split:
        run_config.eval_split = args.eval_split
    if args.max_eval_examples_per_subset is not None:
        run_config.eval.max_eval_examples_per_subset = args.max_eval_examples_per_subset
    if args.base_model_path:
        run_config.model.base_model_name = str(Path(args.base_model_path).expanduser())

    for stage_name in args.disable_stage:
        if stage_name not in run_config.stages:
            available = ", ".join(sorted(run_config.stages))
            raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {available}")
        run_config.stages[stage_name].enabled = False
    for stage_name in args.enable_stage:
        if stage_name not in run_config.stages:
            available = ", ".join(sorted(run_config.stages))
            raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {available}")
        stage_spec = run_config.stages[stage_name]
        try:
            ensure_supported_answer_mode(stage_spec.answer_mode)
        except ValueError as exc:
            raise ValueError(
                f"Stage '{stage_name}' is reserved and cannot be enabled because answer_mode "
                f"'{stage_spec.answer_mode}' is not supported by this pipeline."
            ) from exc
        stage_spec.enabled = True

    if args.enable_multichoice_training and "phase_e" in run_config.phases:
        run_config.phases["phase_e"].allow_multichoice_training = True
    return run_config


def dataset_analysis_only(run_config) -> dict:
    """Save train/eval dataset diagnostics without starting training."""

    output_dir = run_config.output_dir_for_phase(run_config.phase_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_base = load_mathvista_split(run_config, run_config.train_split)
    eval_base = load_mathvista_split(run_config, run_config.eval_split)
    train_analysis = analyze_dataset_records(train_base, run_config.stages)
    eval_analysis = analyze_dataset_records(eval_base, run_config.stages)
    save_dataset_analysis(train_analysis, output_dir / "dataset_analysis_train.json")
    save_dataset_analysis(eval_analysis, output_dir / "dataset_analysis_eval.json")
    return {
        "output_dir": str(output_dir),
        "dataset_analysis_train": train_analysis,
        "dataset_analysis_eval": eval_analysis,
    }


def main() -> None:
    """Entrypoint used by both CLI runs and notebook execution."""

    configure_logging()
    args = parse_args()

    run_config = apply_cli_overrides(build_default_run_config(phase_name=args.phase), args)
    output_dir = run_config.output_dir_for_phase(run_config.phase_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    fatal_error_path = output_dir / "fatal_error.txt"
    if fatal_error_path.exists():
        fatal_error_path.unlink()

    save_json(
        {
            "phase": run_config.phase_name,
            "resume": args.resume,
            "warm_start_checkpoint": args.warm_start_checkpoint,
            "dataset_analysis_only": args.dataset_analysis_only,
            "output_root": run_config.output_root,
            "train_split": run_config.train_split,
            "eval_split": run_config.eval_split,
            "hardware_profile": run_config.hardware_profile_name,
            "disabled_stages": args.disable_stage,
            "enabled_stages": args.enable_stage,
            "enable_multichoice_training": args.enable_multichoice_training,
        },
        output_dir / "run_request.json",
    )

    try:
        if args.dataset_analysis_only:
            results = dataset_analysis_only(run_config)
        else:
            from staged_rl.trainer_runtime import run_phase

            results = run_phase(
                run_config,
                phase_name=args.phase,
                resume_selector=args.resume,
                warm_start_selector=args.warm_start_checkpoint,
            )
    except Exception as exc:
        write_fatal_error(
            fatal_error_path,
            exc,
            {
                "phase": run_config.phase_name,
                "resume": args.resume,
                "warm_start_checkpoint": args.warm_start_checkpoint,
                "dataset_analysis_only": args.dataset_analysis_only,
                "output_root": run_config.output_root,
                "train_split": run_config.train_split,
                "eval_split": run_config.eval_split,
                "hardware_profile": run_config.hardware_profile_name,
                "disabled_stages": args.disable_stage,
                "enabled_stages": args.enable_stage,
                "enable_multichoice_training": args.enable_multichoice_training,
            },
        )
        LOGGER.exception("Fatal run error written to %s", fatal_error_path)
        raise

    print("=" * 80)
    print("STAGED RL RUN COMPLETE")
    print("=" * 80)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
