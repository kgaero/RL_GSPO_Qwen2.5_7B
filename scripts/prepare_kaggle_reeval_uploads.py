"""Prepare Kaggle upload bundles for full-testmini checkpoint reevaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_BUNDLE_ROOT = ROOT / "kaggle_uploads"
DEFAULT_BASELINE_DATASET_ROOT = Path("/tmp/rl-gspo-qwen2-5vlm-pre-refactor-baseline")
OWNER = "mcgmcg1"
CODE_DATASET = f"{OWNER}/rl-gspo-qwen2-5vlm-reeval-code-v2"
BASELINE_DATASET = f"{OWNER}/rl-gspo-qwen2-5vlm-pre-refactor-baseline"


@dataclass(frozen=True)
class KernelBundleSpec:
    bundle_name: str
    notebook_name: str
    kernel_slug: str
    title: str
    dataset_sources: tuple[str, ...]
    kernel_sources: tuple[str, ...]


KERNEL_SPECS = (
    KernelBundleSpec(
        bundle_name="full_testmini_reeval_baseline_and_smoke",
        notebook_name="kaggle_full_testmini_reeval_baseline_and_smoke.ipynb",
        kernel_slug="rl-gspo-qwen2-5vlm-testmini-reeval-smoke",
        title="RL GSPO Qwen2 5VLM Testmini Reeval Smoke",
        dataset_sources=(CODE_DATASET, BASELINE_DATASET),
        kernel_sources=(f"{OWNER}/rl-gspo-qwen2-5vlm-staged-train",),
    ),
    KernelBundleSpec(
        bundle_name="full_testmini_reeval_large_and_phase_d",
        notebook_name="kaggle_full_testmini_reeval_large_and_phase_d.ipynb",
        kernel_slug="rl-gspo-qwen2-5vlm-testmini-reeval-large-phased",
        title="RL GSPO Qwen2 5VLM Testmini Reeval Large PhaseD",
        dataset_sources=(CODE_DATASET,),
        kernel_sources=(
            f"{OWNER}/rl-gspo-qwen2-5vlm-large-split-continue",
            f"{OWNER}/rl-gspo-qwen2-5vlm-phase-d-large-continue",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle bundles for full-testmini reevaluation notebooks.")
    parser.add_argument(
        "--kernel-bundle-root",
        default=str(DEFAULT_KERNEL_BUNDLE_ROOT),
        help="Directory where notebook upload bundles should be written.",
    )
    parser.add_argument(
        "--baseline-dataset-root",
        default=str(DEFAULT_BASELINE_DATASET_ROOT),
        help="Directory where the packaged baseline dataset should be written.",
    )
    parser.add_argument(
        "--baseline-source-root",
        default=None,
        help=(
            "Directory containing grpo_lora/ and grpo_eval_outputs/ for the archived baseline. "
            "If omitted, the script auto-discovers a local checkout copy when present."
        ),
    )
    parser.add_argument(
        "--code-dataset-root",
        default="/tmp/rl-gspo-qwen2-5vlm-staged-code-reeval",
        help="Directory where the packaged reevaluation code dataset should be written.",
    )
    parser.add_argument(
        "--kaggle-bin",
        default="kaggle",
        help="Path to the Kaggle CLI executable used for optional publish operations.",
    )
    parser.add_argument(
        "--publish-baseline-dataset",
        action="store_true",
        help="Create or version the packaged baseline dataset on Kaggle.",
    )
    parser.add_argument(
        "--publish-code-dataset",
        action="store_true",
        help="Create or version the packaged reevaluation code dataset on Kaggle.",
    )
    parser.add_argument(
        "--push-kernels",
        action="store_true",
        help="Push the generated kernel bundles to Kaggle and start the runs.",
    )
    parser.add_argument(
        "--push-timeout",
        type=int,
        default=0,
        help="Optional timeout in seconds passed to 'kaggle kernels push'. Zero leaves the default unchanged.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Optional accelerator passed to 'kaggle kernels push', for example 'gpu'.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _baseline_source_has_artifacts(source_root: Path) -> bool:
    return (source_root / "grpo_lora").is_dir() and (source_root / "grpo_eval_outputs").is_dir()


def resolve_baseline_source_root(explicit_root: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())

    env_root = os.environ.get("BASELINE_SOURCE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.append(ROOT)

    for candidate in candidates:
        if _baseline_source_has_artifacts(candidate):
            return candidate.resolve()
    return None


def prepare_baseline_dataset_bundle(dataset_root: Path, source_root: Path) -> Path:
    source_root = source_root.resolve()
    required_paths = [source_root / "grpo_lora", source_root / "grpo_eval_outputs"]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Baseline source root {source_root} is missing required artifacts: {missing_text}"
        )

    ensure_clean_dir(dataset_root)

    grpo_lora_root = dataset_root / "grpo_lora"
    grpo_eval_root = dataset_root / "grpo_eval_outputs"
    shutil.copytree(source_root / "grpo_lora", grpo_lora_root)
    grpo_eval_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "grpo_eval_outputs" / "eval_metrics.json", grpo_eval_root / "eval_metrics.json")
    shutil.copy2(source_root / "grpo_eval_outputs" / "train_log_summary.json", grpo_eval_root / "train_log_summary.json")

    write_json(
        dataset_root / "dataset-metadata.json",
        {
            "title": "RL GSPO Qwen2.5VLM Pre-Refactor Baseline",
            "id": BASELINE_DATASET,
            "licenses": [{"name": "CC0-1.0"}],
        },
    )
    write_json(
        grpo_lora_root / "checkpoint_info.json",
        {
            "phase_name": "baseline",
            "checkpoint_name": "baseline_local_snapshot",
            "checkpoint_path": "grpo_lora",
            "source_phase_name": "baseline",
            "notes": "Pre-refactor local baseline adapter packaged for full-testmini Kaggle reevaluation.",
        },
    )
    write_json(
        dataset_root / "baseline_manifest.json",
        {
            "label": "baseline_local_snapshot",
            "dataset_slug": BASELINE_DATASET,
            "adapter_dir": "grpo_lora",
            "legacy_eval_metrics": "grpo_eval_outputs/eval_metrics.json",
            "legacy_train_log_summary": "grpo_eval_outputs/train_log_summary.json",
            "notes": [
                "This bundle exists only to make the pre-refactor baseline adapter available to reevaluation notebooks.",
                "It is not part of the staged checkpoint lineage.",
            ],
        },
    )
    (dataset_root / "README.md").write_text(
        "\n".join(
            [
                "# RL GSPO Qwen2.5VLM Pre-Refactor Baseline",
                "",
                "This private Kaggle dataset packages the local pre-refactor LoRA adapter used as the legacy baseline.",
                "It is intended only for the full-testmini reevaluation notebooks.",
                "",
                "Contents:",
                "- `grpo_lora/`: baseline LoRA adapter and a synthetic `checkpoint_info.json` for target discovery.",
                "- `grpo_eval_outputs/`: saved legacy evaluation metrics and train-log summary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_root


def prepare_code_dataset_bundle(dataset_root: Path) -> Path:
    ensure_clean_dir(dataset_root)
    shutil.copy2(ROOT / "rl_gspo_qwen2_5vlm_eval.py", dataset_root / "rl_gspo_qwen2_5vlm_eval.py")
    # Keep the staged package self-contained and small for fast Kaggle dataset propagation.
    shutil.make_archive(str(dataset_root / "staged_rl"), "tar", root_dir=ROOT, base_dir="staged_rl")
    write_json(
        dataset_root / "reeval_bundle_manifest.json",
        {
            "bundle_type": "full_testmini_reevaluation_code",
            "entrypoint": "rl_gspo_qwen2_5vlm_eval.py",
            "archives": ["staged_rl.tar"],
            "notes": [
                "This dataset is separate from the original staged training code bundle.",
                "It exists only to support inference-only reevaluation notebooks.",
            ],
        },
    )
    write_json(
        dataset_root / "dataset-metadata.json",
        {
            "title": "RL GSPO Qwen2.5VLM Reeval Code V2",
            "id": CODE_DATASET,
            "licenses": [{"name": "CC0-1.0"}],
        },
    )
    (dataset_root / "README.md").write_text(
        "\n".join(
            [
                "# RL GSPO Qwen2.5VLM Reevaluation Code V2",
                "",
                "This private Kaggle dataset packages the minimal code required to run full-testmini reevaluation notebooks.",
                "",
                "Contents:",
                "- `rl_gspo_qwen2_5vlm_eval.py`: evaluation-only entrypoint.",
                "- `staged_rl.tar`: staged RL package archive unpacked by the notebook at runtime.",
                "- `reeval_bundle_manifest.json`: bundle marker used for notebook discovery.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_root


def kernel_metadata(spec: KernelBundleSpec) -> dict[str, object]:
    return {
        "id": f"{OWNER}/{spec.kernel_slug}",
        "title": spec.title,
        "code_file": spec.notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": list(spec.dataset_sources),
        "competition_sources": [],
        "kernel_sources": list(spec.kernel_sources),
        "model_sources": [],
    }


def prepare_kernel_bundles(bundle_root: Path) -> list[Path]:
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_paths: list[Path] = []
    for spec in KERNEL_SPECS:
        bundle_dir = bundle_root / spec.bundle_name
        ensure_clean_dir(bundle_dir)
        shutil.copy2(ROOT / spec.notebook_name, bundle_dir / spec.notebook_name)
        write_json(bundle_dir / "kernel-metadata.json", kernel_metadata(spec))
        (bundle_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {spec.title}",
                    "",
                    "This bundle was generated locally and is safe to push as a fresh Kaggle notebook.",
                    "It does not overwrite the original staged training notebooks; it only attaches their outputs as inputs.",
                    "",
                    "Inputs:",
                    *[f"- dataset: `{item}`" for item in spec.dataset_sources],
                    *[f"- kernel output: `{item}`" for item in spec.kernel_sources],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        bundle_paths.append(bundle_dir)
    return bundle_paths


def kaggle_dataset_exists(kaggle_bin: str, dataset_ref: str) -> bool:
    command = [kaggle_bin, "datasets", "list", "--mine", "--search", dataset_ref.split("/", 1)[1], "--csv"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    return any(row.get("ref") == dataset_ref for row in rows)


def publish_baseline_dataset(kaggle_bin: str, dataset_root: Path) -> None:
    if kaggle_dataset_exists(kaggle_bin, BASELINE_DATASET):
        command = [
            kaggle_bin,
            "datasets",
            "version",
            "-p",
            str(dataset_root),
            "-m",
            "Refresh pre-refactor baseline bundle for full-testmini reevaluation.",
            "-r",
            "tar",
        ]
    else:
        command = [kaggle_bin, "datasets", "create", "-p", str(dataset_root), "-r", "tar"]
    subprocess.run(command, check=True)


def publish_code_dataset(kaggle_bin: str, dataset_root: Path) -> None:
    if kaggle_dataset_exists(kaggle_bin, CODE_DATASET):
        command = [
            kaggle_bin,
            "datasets",
            "version",
            "-p",
            str(dataset_root),
            "-m",
            "Refresh reevaluation code bundle.",
            "-r",
            "tar",
        ]
    else:
        command = [kaggle_bin, "datasets", "create", "-p", str(dataset_root), "-r", "tar"]
    subprocess.run(command, check=True)


def push_kernel_bundle(kaggle_bin: str, bundle_dir: Path, *, timeout: int, accelerator: str | None) -> None:
    command = [kaggle_bin, "kernels", "push", "-p", str(bundle_dir)]
    if timeout > 0:
        command.extend(["-t", str(timeout)])
    if accelerator:
        command.extend(["--accelerator", accelerator])
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.kernel_bundle_root).resolve()
    dataset_root = Path(args.baseline_dataset_root).resolve()
    code_dataset_root = Path(args.code_dataset_root).resolve()

    baseline_source_root = resolve_baseline_source_root(args.baseline_source_root)
    if args.publish_baseline_dataset and baseline_source_root is None:
        raise FileNotFoundError(
            "Could not find baseline source artifacts. Provide --baseline-source-root or set "
            "BASELINE_SOURCE_ROOT to a directory containing grpo_lora/ and grpo_eval_outputs/."
        )

    prepared_dataset_root: Path | None = None
    if baseline_source_root is not None:
        prepared_dataset_root = prepare_baseline_dataset_bundle(dataset_root, baseline_source_root)
    prepared_code_dataset_root = prepare_code_dataset_bundle(code_dataset_root)
    bundle_paths = prepare_kernel_bundles(bundle_root)

    if prepared_dataset_root is not None:
        print(f"Prepared baseline dataset bundle: {prepared_dataset_root} (source: {baseline_source_root})")
    else:
        print(
            "Skipped baseline dataset bundle: no local source artifacts were found. "
            "The kernel bundles were still generated and continue to reference the published Kaggle baseline dataset slug."
        )
    print(f"Prepared code dataset bundle: {prepared_code_dataset_root}")
    for bundle_dir in bundle_paths:
        print(f"Prepared kernel bundle: {bundle_dir}")

    if args.publish_baseline_dataset and prepared_dataset_root is not None:
        publish_baseline_dataset(args.kaggle_bin, prepared_dataset_root)
        print(f"Published baseline dataset: {BASELINE_DATASET}")

    if args.publish_code_dataset:
        publish_code_dataset(args.kaggle_bin, prepared_code_dataset_root)
        print(f"Published code dataset: {CODE_DATASET}")

    if args.push_kernels:
        for bundle_dir in bundle_paths:
            push_kernel_bundle(
                args.kaggle_bin,
                bundle_dir,
                timeout=args.push_timeout,
                accelerator=args.accelerator,
            )
            print(f"Pushed kernel bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
