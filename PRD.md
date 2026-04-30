# PRD: Apple-to-Apple Baseline vs Staged RL Evaluation

## 1. Objective

Train `Qwen2.5-VL-7B-Instruct` using the existing staged metric-gated GRPO methodology, then produce a fair apple-to-apple comparison between the baseline LoRA adapter and phase-wise RL checkpoints.

The training methodology must remain intact:

- Phase A, Phase B, and Phase C staged training
- stage-filtered curriculum data
- metric-gated reward controller
- checkpoint-side evaluation
- checkpoint aliases such as `latest`, `best_structure`, `best_correctness`, and `best_composite`

Phase D is intentionally out of scope for this comparison to avoid mixing a hard-stage-specific evaluation with the common headline evaluation used for Baseline/A/B/C.

## 2. Core Fairness Principle

All headline comparisons must use the same evaluation subset:

```text
eval_split = testmini
headline eval subset = eval_overall_numeric
```

The headline comparison must include:

```text
Baseline
Phase A selected checkpoint
Phase B selected checkpoint
Phase C selected checkpoint
```

The headline table must not mix metrics from different eval subsets. Every metric in the headline table must come from `eval_overall_numeric`.

## 3. Dataset Splits

Use:

```text
training_split = test
eval_split = testmini
```

Training data comes from MathVista `test`.

Evaluation data comes from MathVista `testmini`.

Stages are filtered views over the selected split. They are not separate dataset files.

For example:

```text
Stage 1 training data = Stage 1 filter applied to MathVista test
Stage 1 eval data     = Stage 1 filter applied to MathVista testmini
```

## 4. Training Methodology

Training remains phase-wise and stage-based.

### Phase A

```text
Training data: Stage 1 easy numeric from MathVista test
Purpose: structure stabilization
```

### Phase B

```text
Training data: Stage 1 + Stage 2 from MathVista test
Purpose: correctness strengthening
```

### Phase C

```text
Training data: Stage 2 + Stage 3 from MathVista test
Purpose: harder numeric and precision reasoning
```

The training data mix remains phase-specific. This is expected and is part of the staged RL method.

The final evaluation must still use the same headline eval subset for every checkpoint.

## 5. Required Runtime Settings

Due to Kaggle memory limits, use the same constrained settings for baseline eval and phase-wise checkpoint evals:

```text
LoRA rank = 8
hardware_profile = kaggle_t4
max_eval_examples_per_subset = 4
num_samples_per_prompt = 1
max_completion_length = 64
```

These settings are acceptable for constrained Kaggle execution.

Important limitation: `max_eval_examples_per_subset = 4` means headline accuracy is coarse. Exact accuracy can only move in increments of `0.25`:

```text
0/4 = 0.00
1/4 = 0.25
2/4 = 0.50
3/4 = 0.75
4/4 = 1.00
```

The final results should therefore be described as Kaggle-constrained apple-to-apple evals, not full benchmark claims.

## 6. Kaggle Account Requirement

Use a single Kaggle account for all training and evaluation runs:

```text
access token file: access_token_KG
Kaggle account: KG / kgaero
```

All phase runs, baseline evals, checkpoint uploads, and continuation runs should use this same account.

This avoids cross-account artifact mismatch and makes checkpoint lineage easier to audit.

## 7. Checkpointing And Continuation Requirements

The code should support Kaggle's 12-hour session limit by saving checkpoints and allowing continuation from saved outputs.

Expected checkpoint artifacts per checkpoint:

```text
eval_metrics.json
subset_metrics.json
reward_weights.json
controller_state.json
controller_decision.json
checkpoint_info.json
summary.txt
per_sample_records.jsonl
per_prompt_records.json
```

Expected alias files:

```text
aliases/latest.json
aliases/best_structure.json
aliases/best_correctness.json
aliases/best_composite.json
```

Continuation should be possible by attaching prior Kaggle output as input and using one of:

```text
--resume latest
--resume best_structure
--resume best_composite
```

or by explicitly warm-starting from:

```text
--warm-start-checkpoint /kaggle/input/.../checkpoint-XYZ
```

Every Kaggle run must save checkpoints under the Kaggle working/output directory so that outputs can be attached to future runs.

## 8. Logging Requirements

Kaggle notebook logs are the primary progress UI.

Required logging behavior:

- print phase start and runtime configuration
- print checkpoint save path
- print checkpoint eval metrics
- print checkpoint subset metrics
- print reward controller decisions or updated reward weights
- print enough training progress to monitor from Kaggle notebook logs
- avoid relying on dataset-progress uploads for visibility

Previous progress-dataset upload logic was unreliable, so logs should be treated as the required progress surface.

## 9. Baseline Evaluation Protocol

Baseline evaluation must use the same final eval protocol as phase checkpoints.

Baseline target:

```text
source: baseline LoRA adapter
eval_split: testmini
headline subset: eval_overall_numeric
max_eval_examples_per_subset: 4
num_samples_per_prompt: 1
max_completion_length: 64
hardware_profile: kaggle_t4
LoRA rank: 8
```

Baseline must be evaluated in the same final re-evaluation job as the selected phase checkpoints, or with an identical configuration saved in `run_request.json`.

## 10. Final Re-Evaluation Protocol

After training completes, run one controlled re-evaluation job over:

```text
baseline
selected Phase A checkpoint
selected Phase B checkpoint
selected Phase C checkpoint
```

All targets must use:

```text
eval_split = testmini
headline subset = eval_overall_numeric
max_eval_examples_per_subset = 4
num_samples_per_prompt = 1
max_completion_length = 64
hardware_profile = kaggle_t4
LoRA rank = 8
```

The final headline table must be generated only from `eval_overall_numeric`.

Stage-wise results may be reported separately as diagnostic breakdowns.

## 11. Required Headline Eval Table

The required headline table is an apple-to-apple comparison.

Every metric in this table must be computed only from:

```text
eval subset = eval_overall_numeric
eval split = testmini
```

Required table:

| Phase | Checkpoint | Exact | Tol | Parseable | Avg Reward | Structure | Correctness | Composite |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | baseline_rank8 |  |  |  |  |  |  |  |
| Phase A | selected checkpoint |  |  |  |  |  |  |  |
| Phase B | selected checkpoint |  |  |  |  |  |  |  |
| Phase C | selected checkpoint |  |  |  |  |  |  |  |

The table must clearly state:

```text
Headline eval subset: eval_overall_numeric
Eval split: testmini
Max eval examples per subset: 4
Samples per prompt: 1
Max completion length: 64
Hardware profile: kaggle_t4
LoRA rank: 8
Kaggle account: KG / kgaero
```

## 12. Stage-Wise Diagnostic Tables

Stage-wise diagnostic tables are secondary analysis, not the headline comparison.

They use the actual stage filters from the staged curriculum, applied to `testmini`:

```text
stage1_easy_numeric eval = Stage 1 filter applied to MathVista testmini
stage2_float_numeric eval = Stage 2 filter applied to MathVista testmini
stage3_hard_numeric eval = Stage 3 filter applied to MathVista testmini
```

These diagnostic evals should be generated for the same selected checkpoints used in the headline table:

```text
Baseline
selected Phase A checkpoint
selected Phase B checkpoint
selected Phase C checkpoint
```

The diagnostic table should not include every checkpoint by default.

Required diagnostic table:

| Phase | Checkpoint | Eval Subset | Exact | Tol | Parseable | Malformed | Truncation | Composite |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | baseline_rank8 | stage1_easy_numeric |  |  |  |  |  |  |
| Baseline | baseline_rank8 | stage2_float_numeric |  |  |  |  |  |  |
| Baseline | baseline_rank8 | stage3_hard_numeric |  |  |  |  |  |  |
| Phase A | selected checkpoint | stage1_easy_numeric |  |  |  |  |  |  |
| Phase A | selected checkpoint | stage2_float_numeric |  |  |  |  |  |  |
| Phase A | selected checkpoint | stage3_hard_numeric |  |  |  |  |  |  |
| Phase B | selected checkpoint | stage1_easy_numeric |  |  |  |  |  |  |
| Phase B | selected checkpoint | stage2_float_numeric |  |  |  |  |  |  |
| Phase B | selected checkpoint | stage3_hard_numeric |  |  |  |  |  |  |
| Phase C | selected checkpoint | stage1_easy_numeric |  |  |  |  |  |  |
| Phase C | selected checkpoint | stage2_float_numeric |  |  |  |  |  |  |
| Phase C | selected checkpoint | stage3_hard_numeric |  |  |  |  |  |  |

## 13. Optional Checkpoint Audit Table

A separate checkpoint audit table may be generated for debugging and checkpoint selection.

This optional table may include every checkpoint.

It is not the final headline result table.

Suggested checkpoint audit table:

| Phase | Checkpoint | Primary Eval Subset | Exact | Tol | Parseable | Avg Reward | Structure | Correctness | Composite |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

The audit table should be clearly labeled as a checkpoint-selection diagnostic.

## 14. Known Pitfalls To Avoid

Do not mix primary metrics from different eval subsets.

Do not compare checkpoints generated with different eval settings.

Avoid mixing:

```text
max_eval_examples_per_subset = 4
max_eval_examples_per_subset = None
```

Do not mix different `max_completion_length` values in the same headline table.

Do not use cross-account checkpoint chains for final reporting.

Avoid:

```text
Phase A from KG
Phase B from DAD
Phase C from Mimi
```

For this PRD, use only KG.

## 15. Expected Code/Workflow Gaps To Review

The existing training methodology should remain intact, but the workflow likely needs tightening around final evaluation and reporting.

### 15.1 Force Headline Metrics From `eval_overall_numeric`

The current evaluation code can evaluate multiple subsets and then choose a primary metric block. For Baseline/A/B/C this usually selects `eval_overall_numeric`, but the final comparison should not rely on implicit primary-subset behavior.

Requirement:

- final report generation must explicitly read metrics from `subset_metrics["eval_overall_numeric"]`
- headline table must not use phase-specific primary metric selection

Reason:

- prevents accidental apples-to-oranges comparisons
- makes the table auditable

### 15.2 Single Re-Evaluation Job For Final Comparison

The current workflow has produced results across multiple Kaggle accounts and runs.

Requirement:

- create or use a final KG-only re-evaluation notebook/job
- evaluate baseline, selected Phase A, selected Phase B, and selected Phase C in the same job
- write one summary CSV/JSON for the final comparison

Reason:

- removes cross-account and cross-run setting drift
- ensures all targets use identical eval settings

### 15.3 Selected-Checkpoint Manifest

The final evaluation job needs an explicit manifest of selected targets.

Requirement:

- target manifest should include baseline and selected A/B/C checkpoints
- each target should include phase label, checkpoint path, expected LoRA rank, and intended max completion length

Reason:

- avoids accidentally evaluating `latest` when `best_composite` was intended
- makes result lineage reproducible

### 15.4 Reporting Tables

The existing artifacts include per-checkpoint metrics, subset metrics, and CSV summaries, but the final PRD requires two distinct output tables:

- headline table from `eval_overall_numeric`
- stage-wise diagnostic table from stage1/stage2/stage3 subsets

Requirement:

- add or update report generation to produce both tables
- clearly label eval subset, split, and runtime settings

Reason:

- separates the main improvement claim from diagnostic analysis

### 15.5 Logging Verification

The latest code logs checkpoint eval metrics and subset metrics.

Requirement:

- verify KG notebooks use the latest code containing checkpoint/subset metric logging
- ensure logs include checkpoint paths and selected aliases

Reason:

- Kaggle UI logs are the required progress surface
- continuation after 12-hour limits depends on knowing which checkpoint was saved and evaluated

## 16. Success Criteria

The project is successful when:

1. Baseline and selected Phase A/B/C checkpoints are evaluated under the same protocol.
2. The headline table uses only `eval_overall_numeric`.
3. The eval split is `testmini`.
4. The training split is `test`.
5. Runtime settings are identical across final eval targets.
6. All final runs use the KG Kaggle account.
7. Checkpoint lineage is auditable.
8. Checkpoint artifacts and logs are sufficient to resume after Kaggle timeout.
9. Results are clearly labeled as Kaggle-constrained apple-to-apple evals.

