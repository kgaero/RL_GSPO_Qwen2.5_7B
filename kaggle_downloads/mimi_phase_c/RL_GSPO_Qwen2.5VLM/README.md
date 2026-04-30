# RL_GSPO_Qwen2.5VLM

Staged metric-gated GRPO training for `Qwen2.5-VL-7B-Instruct` on MathVista-style free-form numeric visual reasoning. The repository focuses on structure-first RL tuning: it first stabilizes parseable, non-truncated outputs with a strict `<REASONING>` / `<SOLUTION>` contract, then shifts reward pressure toward correctness through a metric-gated controller.

## Scope

This repository currently supports a reproducible case study, not a benchmark claim. The headline `0.75` milestone comes from archived checkpoint-side metrics collected during the original staged runs and summarized under [results/report_summary.md](results/report_summary.md) and [results/tables/master_table_milestones.csv](results/tables/master_table_milestones.csv).

The main supported path is:

- staged GRPO training through Phases A-D
- stage-filtered MathVista subset construction
- checkpoint-side evaluation and aliasing
- paper/report generation from archived artifacts

The repository does not currently establish broad empirical claims across seeds, models, or large ablation suites.

## Highlights

Practical repo highlights:

- Explicit staged curriculum over filtered numeric subsets of `AI4Math/MathVista`
- Metric-gated reward controller with structure floors and correctness escalation
- Named checkpoint aliases: `latest`, `best_structure`, `best_correctness`, `best_composite`
- Conservative `kaggle_t4` profile for single-GPU runs
- Paper-ready tables and plots under `results/` and `paper/`
- Lightweight test suite covering config, controller, parsing, evaluation, checkpointing, and runtime helpers

## Method Overview

Core components at a glance:

The staged pipeline is defined in [staged_rl/config.py](staged_rl/config.py) and executed by [rl_gspo_qwen2_5vlm_test3.py](rl_gspo_qwen2_5vlm_test3.py).

### Curriculum stages

- `stage1_easy_numeric`: free-form integer English problems emphasizing easier numeric reasoning
- `stage2_float_numeric`: free-form English problems with moderate-precision numeric answers
- `stage3_hard_numeric`: harder multistep free-form numeric reasoning with geometry/science-heavy contexts
- `stage4_multi_choice`: scaffolded multi-choice branch
- `stage5_robustness`: reserved multilingual/noisy robustness subset, disabled by default and not runnable yet

### Training phases

- `phase_a`: structure stabilization on Stage 1
- `phase_b`: correctness strengthening with a Stage 1/2 mix
- `phase_c`: precision and harder reasoning with a Stage 2/3 mix
- `phase_d`: hard Stage 3 continuation with longer completions
- `phase_e`: scaffolded multi-choice branch, disabled unless explicitly enabled

### Reward logic

Reward components are logged separately as:

- `format_reward`
- `parseable_reward`
- `finished_reward`
- `correctness_reward`
- `brevity_reward`
- `tolerance_reward`

After each checkpoint evaluation, the controller adjusts weights using held-out metrics. In broad terms:

- low parseability keeps structure rewards elevated
- missing tags keep formatting pressure elevated
- high truncation increases finish pressure
- stable structure with stalled accuracy increases correctness pressure
- structure weights retain nonzero floors rather than decaying fully away

## Archived Results

Archived run snapshot:

From [results/report_summary.md](results/report_summary.md):

- Recommended final checkpoint: `outputs_staged_large_continue/phase_c/checkpoint-120`
- Recommended final metrics: exact `0.75`, tolerance `0.75`, parseable `1.0`, malformed `0.0`, truncation `0.0`
- Dedicated Phase D best matched but did not exceed the recommended Phase C checkpoint
- Main qualitative read: staged RL fixed structure first; larger-split continuation produced the main correctness gain

Read these as archived internal run outcomes, not as fresh benchmark reevaluations.

## Repository Layout

```text
staged_rl/                      Core staged RL package
rl_gspo_qwen2_5vlm_test3.py     Main CLI entrypoint
results/                        Generated tables, plots, and summary artifacts
paper/                          ICLR-style paper sources and PDFs
scripts/                        Plot/report generation and Kaggle helper scripts
tests/                          Unit tests
README_staged_rl.md             Short runbook for staged phase commands
```

## Environment

What you need to run the project:

There is no pinned environment file or lockfile in the repository at the moment. The runtime code expects a Python environment with the following major packages available:

- `torch`
- `transformers`
- `trl`
- `unsloth`
- `vllm`
- `datasets`
- `pillow`
- `packaging`

Some reporting scripts also assume standard data/plotting packages such as `pandas` and `matplotlib`.

This codebase is oriented around GPU-backed execution. The training path assumes CUDA is available; when multiple GPUs are visible, the current runtime still behaves as a single-process, single-GPU pipeline.

## Data And Base Model

Primary upstream assets:

- Dataset: `AI4Math/MathVista`
  - https://huggingface.co/datasets/AI4Math/MathVista
- Base model: `Qwen2.5-VL-7B-Instruct`
  - https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

Default split settings in the refactored runner are:

- training split: `test`
- evaluation split: `testmini`

Those defaults are defined in [staged_rl/config.py](staged_rl/config.py).

## Quickstart 🚀

### 1. Run Phase A

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_a
```

### 2. Run Phase A with the conservative Kaggle T4 profile

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_a --hardware-profile kaggle_t4
```

### 3. Continue through later phases using checkpoint aliases

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_b --resume best_structure
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_c --resume best_composite
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_d --resume best_composite --hardware-profile kaggle_t4
```

### 4. Resume the current phase from its latest checkpoint

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_c --resume latest
```

### 5. Warm-start from a manual checkpoint path

```bash
python3 rl_gspo_qwen2_5vlm_test3.py \
  --phase phase_d \
  --warm-start-checkpoint outputs_staged/phase_c/checkpoint-120
```

### 6. Save dataset analysis without training

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_a --dataset-analysis-only
```

### 7. Enable the scaffolded multi-choice branch

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_e --enable-multichoice-training
```

## Kaggle T4 Profile 💻

The `kaggle_t4` profile keeps the same staged pipeline but reduces runtime pressure for smaller GPUs. In [staged_rl/config.py](staged_rl/config.py), it mainly changes:

- `max_seq_length: 16384 -> 1280`
- `image_size: 512 -> 336`
- `gpu_memory_utilization: 0.8 -> 0.65`
- `gradient_accumulation_steps: 2 -> 4`
- `num_generations: 4 -> 2`
- `max_prompt_length: 1024 -> 320`
- `max_completion_length: 256 -> 64`

LoRA is now pinned to rank 8 in the base config, so the profile keeps that setting unchanged.
- `num_samples_per_prompt: 4 -> 1`
- `max_eval_examples_per_subset: None -> 2`
- Phase D override: `max_completion_length: 320 -> 96`

## Outputs 📦

Each checkpoint evaluation writes artifacts such as:

- `eval_metrics.json`
- `subset_metrics.json`
- `reward_weights.json`
- `controller_state.json`
- `controller_decision.json`
- `summary.txt`
- `per_sample_records.jsonl`
- `per_prompt_records.json`

Run-level alias files are written under `outputs_staged/<phase>/aliases/` for:

- `latest`
- `best_structure`
- `best_correctness`
- `best_composite`

## Paper And Reporting 📝

Main paper sources and outputs live under `paper/iclr2026_template`.

Current paper/report entrypoints:

- arXiv-ready paper source:
  - `paper/iclr2026_template/staged_metric_gated_grpo_camera_ready.tex`
- compiled paper:
  - `paper/iclr2026_template/staged_metric_gated_grpo_camera_ready.pdf`
- results package summary:
  - `results/README.md`
- archived result summary:
  - `results/report_summary.md`

Regenerate plots and tables with:

```bash
python3 scripts/generate_results_report.py
```

## Tests ✅

Run the unit tests with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Current local status: `28` tests pass.

## Limitations

- The repository currently reflects a single primary model family and a narrow task slice.
- The headline `0.75` result comes from archived run artifacts, not a fresh full reevaluation campaign.
- The codebase includes reevaluation helpers and Kaggle notebook artifacts, but those are not part of the current core claim set.
- No pinned dependency lockfile is included yet.
- Multi-GPU execution is not configured automatically.
- Phase E multi-choice training is scaffolded, not part of the main reported result path.

## Citation

Citation information will be added after the paper is uploaded to arXiv.
