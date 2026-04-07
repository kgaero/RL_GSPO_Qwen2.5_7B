# Qwen2.5-VL Staged RL Diagram Walkthrough

This note follows the deep-dive pipeline image from left to right and explains which code implements each block.

Use it together with:

- `results/plots/phase_run_sequence.svg`
- `results/plots/reward_controller_flow.svg`
- `results/plots/checkpoint_artifact_map.svg`
- `results/plots/stage_filter_matrix.svg`
- `results/plots/resume_lineage.svg`

## How To Read The Repo

The cleanest top-down path is:

1. `rl_gspo_qwen2_5vlm_test3.py::main`
2. `staged_rl/trainer_runtime.py::run_phase`
3. `staged_rl/data.py` for dataset shaping
4. `staged_rl/rewarding.py` and `staged_rl/parsing.py` for reward semantics
5. `staged_rl/evaluation.py` for checkpoint-side eval
6. `staged_rl/controller.py` for metric-gated weight updates
7. `staged_rl/checkpointing.py` for artifact writes and alias selection

## Top Row

### 1. Base VLM

Image box:
- `Base VLM`
- `Unsloth / Qwen2.5-VL-7B-Instruct`

Primary code:
- `staged_rl/config.py:165` `ModelConfig`
- `staged_rl/trainer_runtime.py:272` `create_model_and_tokenizer`

What the code does:
- `ModelConfig` defines the base pretrained model, context length, image size, LoRA rank, quantization, and fast-inference settings.
- `create_model_and_tokenizer()` loads the base model through `FastVisionModel.from_pretrained(...)`.
- If the loaded object does not yet have active PEFT adapters, the function attaches a LoRA adapter through `FastVisionModel.get_peft_model(...)`.

### 2. MathVista Train Split

Image box:
- `MathVista Train Split (test or testmini)`

Primary code:
- `staged_rl/data.py:325` `load_mathvista_split`
- `staged_rl/data.py:144` `enrich_example`

What the code does:
- Loads the Hugging Face dataset split from `run_config.dataset_name`.
- Enriches each example with flattened metadata such as `context_family`, `skills`, `answer_mode`, `precision`, normalized image payloads, and inferred gold option letters.

### 3. MathVista Eval Split

Image box:
- `MathVista eval Split (eval subsets)`

Primary code:
- `staged_rl/data.py:410` `build_eval_datasets`
- `staged_rl/evaluation.py:284` `evaluate_checkpoint`

What the code does:
- Builds a numeric overall eval subset plus stage-specific eval subsets.
- These subsets are used every time a checkpoint is saved.

### 4. Phase Config A / B / C / D

Image box:
- `Phase config A / B / C / D`

Primary code:
- `staged_rl/config.py:376` `build_default_phase_specs`
- `staged_rl/config.py:291` `build_default_stage_specs`
- `staged_rl/config.py:73` `RewardGateConfig`
- `staged_rl/config.py:99` `CheckpointScoreConfig`

What the code does:
- Defines the curriculum phase sequence, stage mixes, initial reward weights, default resume selectors, controller thresholds, and checkpoint-scoring weights.

## Middle Loop

### 5. Model Prep / Resume Plan / Fresh LoRA / Trainer Resume / Adapter Warm Start

Image box:
- `Model prep/ resume plan fresh LoRA, trainer resume or adapter warm start`

Primary code:
- `staged_rl/checkpointing.py:242` `build_resume_plan`
- `staged_rl/checkpointing.py:218` `resolve_selector`
- `staged_rl/trainer_runtime.py:208` `_warm_start_peft_adapter`
- `staged_rl/trainer_runtime.py:445` `_load_controller_state_from_checkpoint`
- `staged_rl/trainer_runtime.py:459` `run_phase`

What the code does:
- Resolves selectors like `latest`, `best_structure`, `best_composite`, checkpoint names, or direct paths.
- Decides whether the selector means same-phase trainer resume or cross-phase warm-start of adapter weights only.

### 6. Stage / Prompt Builder Filters + Priorities `<REASONING>` `<SOLUTION>`

Image box:
- `Stage/prompt builder filters + priorities <REASONING> <SOLUTION>`

Primary code:
- `staged_rl/data.py:164` `match_filter_spec`
- `staged_rl/data.py:217` `stage_priority`
- `staged_rl/data.py:234` `build_prompt_text`
- `staged_rl/data.py:334` `build_stage_dataset`
- `staged_rl/data.py:371` `build_phase_train_dataset`

What the code does:
- Filters examples into curriculum stages, ranks them within a stage, wraps the strict answer contract, and interleaves stage datasets by phase mix.

### 7. GRPO Fine Tuning / TRL `GRPOTrainer` / Unsloth Runtime

Image box:
- `GRPO Fine Tuning`
- `TRL GRPOTrainer`
- `Unsloth runtime`

Primary code:
- `staged_rl/trainer_runtime.py:331` `build_grpo_args`
- `staged_rl/trainer_runtime.py:344` `MetricAwareGRPOTrainerMixin`
- `staged_rl/trainer_runtime.py:408` `build_metric_trainer_class`
- `staged_rl/trainer_runtime.py:459` `run_phase`

What the code does:
- Builds TRL GRPO arguments, creates the metric-aware trainer subclass, and runs `trainer.train(...)`.

### 8. Rule-Based Reward Stack

Image box:
- `Rule based reward stack`
- `Format | parseable | finished`
- `Correctness | brevity | tolerance`

Primary code:
- `staged_rl/rewarding.py:25` `RewardRuntimeContext`
- `staged_rl/rewarding.py:115` `build_reward_functions`
- `staged_rl/parsing.py:77` `extract_single_solution_text`
- `staged_rl/parsing.py:112` `normalized_exact_match`
- `staged_rl/parsing.py:118` `tolerance_match`
- `staged_rl/parsing.py:133` `solution_tag_compliant`
- `staged_rl/parsing.py:139` `reasoning_tag_compliant`
- `staged_rl/parsing.py:145` `malformed_numeric_answer`
- `staged_rl/parsing.py:183` `infer_truncation`

What the code does:
- Implements formatting, parseability, finished-answer, correctness, brevity, and tolerance reward functions on top of parsed completions.

### 9. Checkpoint Eval

Image box:
- `Checkpoint eval`
- `exact | tolerance | parseable | tag compliance | truncation`
- `Save / eval every 60 steps`

Primary code:
- `staged_rl/evaluation.py:122` `evaluate_dataset_subset`
- `staged_rl/evaluation.py:50` `aggregate_subset_metrics`
- `staged_rl/evaluation.py:32` `determine_failure_mode`
- `staged_rl/evaluation.py:284` `evaluate_checkpoint`
- `staged_rl/trainer_runtime.py:356` `_metric_aware_save`

What the code does:
- Runs generation on held-out subsets for each checkpoint, scores each sample, and aggregates the metrics that later drive both checkpoint ranking and controller updates.

### 10. Reward Controller / Metric-Gated Weight Updates

Image box:
- `Reward controller`
- `Metric-gated weight updates for next training segment`

Primary code:
- `staged_rl/controller.py:22` `RewardController`
- `staged_rl/controller.py:78` `update_from_metrics`
- `staged_rl/trainer_runtime.py:388` controller call inside `_metric_aware_save`

What the code does:
- Applies parseability, formatting, finish, and correctness-escalation rules to the current reward weights using held-out checkpoint metrics.

## Right Side

### 11. Checkpoint Artifacts / Metrics + Reward Weights + Controller State + Samples

Image box:
- `Checkpoint artifacts`
- `metrics + reward weights`
- `controller state + samples`

Primary code:
- `staged_rl/checkpointing.py:119` `write_checkpoint_artifacts`
- `staged_rl/evaluation.py:316` `save_json_lines`

What the code writes into each checkpoint:
- `eval_metrics.json`
- `subset_metrics.json`
- `reward_weights.json`
- `controller_state.json`
- `checkpoint_info.json`
- `summary.txt`
- `per_prompt_records.json`
- `per_sample_records.jsonl`

Important nuance:
- `reward_weights.json` and `controller_state.json` are written once inside `write_checkpoint_artifacts()`, then overwritten after the controller update with the new state.

### 12. Scores + Aliases / Structure | Correctness | Composite

Image box:
- `Scores + aliases`
- `latest | best_structure | best_correctness | best_composite`

Primary code:
- `staged_rl/checkpointing.py:37` `compute_checkpoint_scores`
- `staged_rl/checkpointing.py:47` `CheckpointRegistry`
- `staged_rl/checkpointing.py:88` `CheckpointRegistry.register`

What the code does:
- Computes ranking scores from eval metrics and refreshes phase-local alias files used by later resumes.

### 13. Recommended Final

Image box:
- `Recommended final`

Primary code and evidence:
- `results/README.md`
- `results/Chronological_Flow.txt`
- `results/plots/checkpoint_frontier_scatter.svg`
- `results/plots/resume_lineage.svg`

What it means in this repo:
- Final recommendation is a report-layer conclusion built from saved artifacts rather than a special training-time code path.

## Bottom Row: Phase-To-Phase Curriculum

### 14. Phase A

Primary code:
- `staged_rl/config.py:380`

Meaning:
- `stage1_easy_numeric` only, structure stabilization.

### 15. Phase B

Primary code:
- `staged_rl/config.py:389`

Meaning:
- `stage1_easy_numeric: 0.70` and `stage2_float_numeric: 0.30`, with default resume from `best_structure`.

### 16. Phase C

Primary code:
- `staged_rl/config.py:398`

Meaning:
- Repo reality is `stage2_float_numeric: 0.60` and `stage3_hard_numeric: 0.40`.
- This is where tolerance reward turns on.

### 17. Phase D

Primary code:
- `staged_rl/config.py:408`

Meaning:
- `stage3_hard_numeric` only, tolerance reward still enabled, and longer completion allowance.

## The Single Most Important Control Loop

The center of the repo is:

1. train for a segment
2. save checkpoint
3. evaluate checkpoint on held-out subsets
4. compute structure/correctness/composite scores
5. update aliases
6. update reward weights from eval metrics
7. continue training with the new weights

That loop is implemented mainly across:

- `staged_rl/trainer_runtime.py:356`
- `staged_rl/evaluation.py:284`
- `staged_rl/checkpointing.py:119`
- `staged_rl/controller.py:78`
