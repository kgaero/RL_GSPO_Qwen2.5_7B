# Staged Metric-Gated RL for Qwen2.5-VL

This repo now runs MathVista GRPO training through explicit phases with stage-aware data filtering, metric-gated reward weights, checkpoint-side evaluation, and named checkpoint aliases.

## Phase Runs

Run Phase A:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_a
```

Run Phase A with the conservative Kaggle T4 profile:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_a --hardware-profile kaggle_t4
```

Run Phase B from the best structure checkpoint found in Phase A:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_b --resume best_structure
```

Run Phase C from the best composite checkpoint found so far:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_c --resume best_composite
```

Resume the current phase from its latest checkpoint:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_c --resume latest
```

Warm-start from a manual checkpoint path:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_d --resume outputs_staged/phase_c/checkpoint-120
```

The `kaggle_t4` profile keeps the same staged pipeline but trims runtime pressure:

- `max_seq_length=1280`
- `image_size=336`
- `gpu_memory_utilization=0.65`
- `lora_rank=8`
- `max_lora_rank=8`
- `compilation_config={"level": 3, "cudagraph_mode": "PIECEWISE"}`
- `num_generations=2`
- `max_prompt_length=320`
- `max_completion_length=64`
- `num_samples_per_prompt=1`
- `max_eval_examples_per_subset=2`

If multiple GPUs are visible, the current runner logs that fact but still operates as a single-process, single-GPU pipeline. It does not set up DDP or tensor parallelism automatically.

## Stage Controls

Disable a stage:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_b --disable-stage stage3_hard_numeric
```

The multilingual robustness stage is reserved for future work. It is tracked in the curriculum docs, but it is not currently enableable because the trainer only supports `numeric_free_form` and `multi_choice` answer modes.

Phase E multi-choice training stays disabled by default. The scaffold is present, but training only runs if you opt in:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_e --enable-multichoice-training
```

## Reward Gating

Reward components are logged separately as:

- `format_reward`
- `parseable_reward`
- `finished_reward`
- `correctness_reward`
- `brevity_reward`
- `tolerance_reward`

After each checkpoint evaluation, the reward controller updates weights using held-out metrics from `testmini`.

- Low parseability keeps `parseable_reward` elevated.
- Missing or broken tags keep `format_reward` elevated.
- High truncation increases `finished_reward` and keeps brevity pressure on.
- Stable structure with stalled exact match increases `correctness_reward`.
- Structure rewards never decay fully to zero.

Checkpoint artifacts are written into each checkpoint directory:

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

## Dataset Analysis

Save train/eval split analysis without starting training:

```bash
python3 rl_gspo_qwen2_5vlm_test3.py --phase phase_a --dataset-analysis-only
```

This writes:

- `dataset_analysis_train.json`
- `dataset_analysis_eval.json`

The analysis includes counts by metadata field, stage sizes, sample examples, and warnings for tiny or heterogeneous stages.
