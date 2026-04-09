"""Generate Kaggle notebooks for full-testmini reevaluation runs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }


def build_copy_and_catalog_cell() -> str:
    return """
    import glob, json, os, shutil, subprocess, tarfile

    WORKDIR = "/kaggle/working/RL_GSPO_Qwen2.5VLM"
    VENV_DIR = "/tmp/rl-gspo-venv"
    VENV_PYTHON = f"{VENV_DIR}/bin/python"


    def find_code_dataset_root():
        matches = []
        for root, _, files in os.walk("/kaggle/input"):
            if "reeval_bundle_manifest.json" in files and "rl_gspo_qwen2_5vlm_eval.py" in files:
                matches.append(root)
        if not matches:
            raise FileNotFoundError(
                f"Could not find attached code dataset. Visible inputs: {glob.glob('/kaggle/input/*')}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous code dataset attachment. Matches={matches}. Visible inputs: {glob.glob('/kaggle/input/*')}"
            )
        return matches[0]


    def load_json_if_exists(path):
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


    def catalog_attached_adapters():
        candidates = []
        for root, _, files in os.walk("/kaggle/input"):
            if "adapter_model.safetensors" not in files:
                continue
            checkpoint_info_path = os.path.join(root, "checkpoint_info.json")
            reward_weights_path = os.path.join(root, "reward_weights.json")
            checkpoint_info = load_json_if_exists(checkpoint_info_path)
            candidate = {
                "root": root,
                "checkpoint_name": os.path.basename(root),
                "phase_name": checkpoint_info.get("phase_name"),
                "source_phase_name": checkpoint_info.get("source_phase_name"),
                "checkpoint_path_from_info": checkpoint_info.get("checkpoint_path"),
                "reward_weights_path": reward_weights_path if os.path.exists(reward_weights_path) else None,
                "checkpoint_info_path": checkpoint_info_path if os.path.exists(checkpoint_info_path) else None,
                "metrics": checkpoint_info.get("metrics", {}),
            }
            haystack_parts = [
                candidate["root"],
                candidate["checkpoint_name"],
                candidate["phase_name"],
                candidate["source_phase_name"],
                candidate["checkpoint_path_from_info"],
            ]
            candidate["haystack"] = " ".join(str(part).lower() for part in haystack_parts if part)
            candidates.append(candidate)
        return sorted(candidates, key=lambda item: item["root"])


    CODE_DATASET_ROOT = find_code_dataset_root()
    CATALOG = catalog_attached_adapters()

    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    shutil.copytree(CODE_DATASET_ROOT, WORKDIR)

    for archive_name in ("staged_rl.tar",):
        archive_path = os.path.join(WORKDIR, archive_name)
        if os.path.exists(archive_path):
            with tarfile.open(archive_path, "r") as tf:
                tf.extractall(WORKDIR)
            os.remove(archive_path)
            print("Extracted", archive_name)

    for folder_name in ("staged_rl",):
        nested_path = os.path.join(WORKDIR, folder_name, folder_name)
        target_path = os.path.join(WORKDIR, folder_name)
        if os.path.isdir(nested_path):
            temp_path = f"{target_path}_flat"
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            shutil.move(nested_path, temp_path)
            shutil.rmtree(target_path)
            shutil.move(temp_path, target_path)
            print("Flattened", folder_name)

    print("Using code dataset", CODE_DATASET_ROOT)
    print("Copied code to", WORKDIR)
    print("Visible input roots:", glob.glob("/kaggle/input/*"))
    print("Discovered adapter candidates:")
    print(json.dumps(CATALOG, indent=2))
    """


def build_install_cell() -> str:
    return """
    subprocess.run(["python3", "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(["python3", "-m", "uv", "venv", "--seed", "--clear", VENV_DIR], check=True)
    install_commands = [
        [VENV_PYTHON, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel"],
        [
            VENV_PYTHON,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "numpy==1.26.4",
            "scipy==1.15.3",
            "scikit-learn==1.6.1",
        ],
        [
            VENV_PYTHON,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "pillow",
            "packaging",
            "safetensors",
            "torchvision",
            "bitsandbytes",
            "xformers",
            "huggingface_hub>=0.34.0",
            "datasets==4.3.0",
            "transformers==4.56.2",
            "trl==0.22.2",
            "unsloth",
        ],
        [
            VENV_PYTHON,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "vllm==0.10.2",
            "--extra-index-url",
            "https://wheels.vllm.ai/0.10.2/",
        ],
    ]
    for command in install_commands:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=WORKDIR)
    print("Venv ready:", VENV_PYTHON)
    """


def build_compat_cell() -> str:
    return r"""
    compat_script = r'''
    import numpy
    import scipy
    import sklearn
    import torch
    import transformers
    import trl
    import unsloth
    import vllm
    print({
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "vllm": vllm.__version__,
    })
    '''
    subprocess.run([VENV_PYTHON, "-c", compat_script], check=True, cwd=WORKDIR)
    """


def build_target_resolution_cell(output_root: str, target_specs: list[dict]) -> str:
    target_specs_json = json.dumps(target_specs, indent=2)
    return (
        f'HARDWARE_PROFILE = "kaggle_t4"\n'
        f'EVAL_SPLIT = "testmini"\n'
        f'OUTPUT_ROOT = "{output_root}"\n'
        "FULL_SPLIT = True\n"
        "MAX_EVAL_EXAMPLES_PER_SUBSET = None\n\n"
        f"TARGET_SPECS = {target_specs_json}\n\n"
        "def resolve_target(spec, catalog):\n"
        "    matches = []\n"
        "    for candidate in catalog:\n"
        '        haystack = candidate["haystack"]\n'
        '        if any(token.lower() not in haystack for token in spec.get("match_all", [])):\n'
        "            continue\n"
        '        match_any = spec.get("match_any", [])\n'
        '        if match_any and not any(token.lower() in haystack for token in match_any):\n'
        "            continue\n"
        "        matches.append(candidate)\n\n"
        "    if not matches:\n"
        '        raise RuntimeError(\n'
        '            "Could not resolve target "\n'
        '            + spec["label"]\n'
        '            + ". Edit TARGET_SPECS or attach the missing dataset.\\nCatalog="\n'
        "            + json.dumps(catalog, indent=2)\n"
        "        )\n"
        "    if len(matches) > 1:\n"
        '        raise RuntimeError(\n'
        '            "Target resolution is ambiguous for "\n'
        '            + spec["label"]\n'
        '            + ". Tighten the match tokens.\\nMatches="\n'
        "            + json.dumps(matches, indent=2)\n"
        "        )\n\n"
        "    match = matches[0]\n"
        "    resolved = {\n"
        '        "label": spec["label"],\n'
        '        "checkpoint": match["root"],\n'
        '        "phase": spec["phase"],\n'
        '        "max_completion_length": spec["max_completion_length"],\n'
        "    }\n"
        '    if match.get("reward_weights_path"):\n'
        '        resolved["reward_weights_json"] = match["reward_weights_path"]\n'
        "    return resolved\n\n"
        "RESOLVED_TARGETS = [resolve_target(spec, CATALOG) for spec in TARGET_SPECS]\n"
        'TARGET_SPEC_PATH = os.path.join(WORKDIR, f"{OUTPUT_ROOT}_targets.json")\n'
        'with open(TARGET_SPEC_PATH, "w", encoding="utf-8") as handle:\n'
        '    json.dump({"targets": RESOLVED_TARGETS}, handle, indent=2)\n\n'
        'print("Resolved targets:")\n'
        "print(json.dumps(RESOLVED_TARGETS, indent=2))\n"
        'print("Target spec path:", TARGET_SPEC_PATH)\n'
    )


def build_run_cell() -> str:
    return """
    cmd = [
        VENV_PYTHON,
        "rl_gspo_qwen2_5vlm_eval.py",
        "--target-spec-json", TARGET_SPEC_PATH,
        "--hardware-profile", HARDWARE_PROFILE,
        "--output-root", OUTPUT_ROOT,
        "--eval-split", EVAL_SPLIT,
        "--full-split",
    ]
    if MAX_EVAL_EXAMPLES_PER_SUBSET is not None:
        cmd.extend(["--max-eval-examples-per-subset", str(MAX_EVAL_EXAMPLES_PER_SUBSET)])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    reeval_log_path = os.path.join(WORKDIR, OUTPUT_ROOT, "reevaluation_subprocess.log")
    os.makedirs(os.path.dirname(reeval_log_path), exist_ok=True)
    print("Running:", " ".join(cmd))
    print("Subprocess log:", reeval_log_path)
    with open(reeval_log_path, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            check=False,
            cwd=WORKDIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        print(f"Reevaluation failed with return code {completed.returncode}. Last 200 log lines:")
        with open(reeval_log_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()[-200:]
        print("".join(lines))
        raise RuntimeError(f"Reevaluation subprocess failed with return code {completed.returncode}")
    print("Reevaluation finished successfully.")
    """


def build_summary_cell() -> str:
    return """
    summary_path = os.path.join(WORKDIR, OUTPUT_ROOT, "reevaluation_summary.json")
    csv_path = os.path.join(WORKDIR, OUTPUT_ROOT, "reevaluation_summary.csv")
    subset_sizes_path = os.path.join(WORKDIR, OUTPUT_ROOT, "eval_subset_sizes.json")

    with open(subset_sizes_path, "r", encoding="utf-8") as handle:
        print("Subset sizes:")
        print(handle.read())

    with open(summary_path, "r", encoding="utf-8") as handle:
        print("JSON summary:")
        print(handle.read())

    print("CSV summary path:", csv_path)

    collected = []
    for root, _, files in os.walk(os.path.join(WORKDIR, OUTPUT_ROOT)):
        for file_name in files:
            collected.append(os.path.join(root, file_name))
    for path in sorted(collected):
        print(path)
    """


def build_notebook(title: str, description: str, output_root: str, target_specs: list[dict]) -> dict:
    return {
        "cells": [
            md_cell(
                f"""
                # {title}

                {description}

                This notebook is separate from the original training notebooks. It does not overwrite any prior
                Kaggle run outputs. It expects:

                - the reevaluation code dataset attachment,
                - adapter/checkpoint attachments for the targets listed in `TARGET_SPECS`, and
                - enough GPU time to run a full `testmini` reevaluation pass.

                If target discovery is ambiguous or missing, edit the `TARGET_SPECS` cell and rerun.
                """
            ),
            code_cell(build_copy_and_catalog_cell()),
            code_cell(build_install_cell()),
            code_cell(build_compat_cell()),
            code_cell(build_target_resolution_cell(output_root, target_specs)),
            code_cell(build_run_cell()),
            code_cell(build_summary_cell()),
        ],
        "metadata": notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    baseline_smoke_targets = [
        {
            "label": "baseline_local_snapshot",
            "phase": "phase_a",
            "max_completion_length": 256,
            "match_any": ["grpo_lora", "baseline", "pre-refactor"],
        },
        {
            "label": "smoke_phase_b_best",
            "phase": "phase_b",
            "max_completion_length": 64,
            "match_all": ["phase_b", "checkpoint-60"],
            "match_any": ["outputs_staged", "phase_b", "smoke"],
        },
        {
            "label": "smoke_phase_c_best",
            "phase": "phase_c",
            "max_completion_length": 64,
            "match_all": ["phase_c", "checkpoint-119"],
            "match_any": ["outputs_staged", "phasec-best", "smoke"],
        },
    ]
    large_phase_d_targets = [
        {
            "label": "large_phase_c_best",
            "phase": "phase_c",
            "max_completion_length": 64,
            "match_all": ["phase_c", "checkpoint-120"],
            "match_any": ["outputs_staged_large_continue", "large-phasec", "phase_c_large_best"],
        },
        {
            "label": "dedicated_phase_d_best",
            "phase": "phase_d",
            "max_completion_length": 96,
            "match_all": ["phase_d", "checkpoint-130", "phase-d-large-continue"],
            "match_any": ["outputs_staged_phase_d_from_large_phase_c", "phase-d-large-continue"],
        },
    ]

    notebooks = {
        ROOT / "kaggle_full_testmini_reeval_baseline_and_smoke.ipynb": build_notebook(
            title="Full Testmini Reevaluation: Baseline and Smoke Checkpoints",
            description=(
                "Reevaluate the local pre-refactor baseline adapter plus the smoke Phase B and smoke Phase C "
                "best checkpoints on the full `testmini` split."
            ),
            output_root="outputs_full_testmini_reeval_baseline_and_smoke",
            target_specs=baseline_smoke_targets,
        ),
        ROOT / "kaggle_full_testmini_reeval_large_and_phase_d.ipynb": build_notebook(
            title="Full Testmini Reevaluation: Large Phase C and Dedicated Phase D",
            description=(
                "Reevaluate the recommended large Phase C checkpoint and the dedicated Phase D best checkpoint "
                "on the full `testmini` split."
            ),
            output_root="outputs_full_testmini_reeval_large_and_phase_d",
            target_specs=large_phase_d_targets,
        ),
    }

    for path, notebook in notebooks.items():
        write_notebook(path, notebook)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
