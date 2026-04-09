# RL Training Curriculum for Qwen2.5-VL GSPO/GRPO

This document explains the implemented staged RL curriculum in detail, using the actual configuration and runtime behavior in this repository. It is not a generic RL note. The values below come from the code in:

- [staged_rl/config.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/config.py)
- [staged_rl/data.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/data.py)
- [staged_rl/rewarding.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/rewarding.py)
- [staged_rl/controller.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/controller.py)
- [staged_rl/checkpointing.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/checkpointing.py)
- [staged_rl/trainer_runtime.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/trainer_runtime.py)
- [staged_rl/evaluation.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/evaluation.py)

## 1. Why this curriculum exists

The RL pipeline is designed to solve a specific failure pattern seen in MathVista-style visual numeric QA:

- malformed outputs
- missing `<REASONING>` and `<SOLUTION>` tags
- truncation before the final answer
- answers that cannot be parsed as a number
- overly long completions that waste token budget
- correctness reward dominating too early
- weak policy updates on a mixed-difficulty dataset

The curriculum solves this by separating the problem into phases:

1. first make the model structurally reliable
2. then strengthen exact correctness
3. then add float tolerance and harder visual reasoning
4. only after that specialize on harder Stage 3 examples

This means the curriculum is not just "train on easier then harder data". It is a combined schedule over:

- dataset difficulty
- answer type
- reward emphasis
- checkpoint selection
- resume policy
- hardware profile

## 2. Core response contract

For numeric free-form tasks, the prompt forces the model to answer in this structure:

```text
<REASONING>
...
</REASONING>
<SOLUTION>
single numeric answer
</SOLUTION>
```

That contract is enforced in [staged_rl/data.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/data.py) by `build_prompt_text(...)`.

Why it matters:

- the reward functions depend on reliable extraction of the final answer
- the evaluator separately scores structure and correctness
- malformed or truncated outputs can be rewarded or penalized independently of exact correctness

Multi-choice is explicitly separated into its own answer mode and is not mixed with numeric free-form reward parsing.

## 3. Curriculum at a glance

The curriculum has two layers:

- `stages`: dataset subsets
- `phases`: training runs that mix one or more stages with specific reward settings

The stage defines what examples are eligible.
The phase defines how strongly each reward matters and how stages are mixed.

## 4. Stage definitions

### 4.1 Stage table

| Stage | Answer Mode | Main Filter | Context Bias | Skill Bias | Intended Difficulty | Why this stage exists |
| --- | --- | --- | --- | --- | --- | --- |
| `stage1_easy_numeric` | `numeric_free_form` | `free_form`, `integer`, `english` | `synthetic scene`, `table`, `natural image` | `arithmetic reasoning`, `statistical reasoning`, `numeric commonsense` | easiest | stabilize formatting, parseability, and short numeric answers |
| `stage2_float_numeric` | `numeric_free_form` | `free_form`, `integer/float`, `english` | `table`, `chart`, `plot`, `scientific figure`, `natural image` | `arithmetic reasoning`, `statistical reasoning`, `scientific reasoning`, `numeric commonsense` | medium | add moderate precision and broader visual formats |
| `stage3_hard_numeric` | `numeric_free_form` | `free_form`, `integer/float`, `english` | `geometry diagram`, `plot`, `scientific figure`, `abstract scene` | `geometry reasoning`, `algebraic reasoning`, `scientific reasoning`, `arithmetic reasoning`, `statistical reasoning` | hard | push multistep reasoning, geometry, algebra, harder figures |
| `stage4_multi_choice` | `multi_choice` | `multi_choice` only | `geometry diagram`, `scientific figure`, `plot` | `geometry reasoning`, `algebraic reasoning` | separate branch | keep multi-choice logic isolated from numeric extraction |
| `stage5_robustness` | `mixed` | non-English examples | none | none | reserved robustness | tracked for multilingual/noisy data, disabled by default and not runnable in the current trainer |

### 4.2 Exact implemented stage filters

| Stage | `question_types` | `answer_types` | `languages` | `context_families` | `skills_any` | `answer_modes` | Special flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stage1_easy_numeric` | `free_form` | `integer` | `english` | `synthetic scene`, `table`, `natural image` | `arithmetic reasoning`, `statistical reasoning`, `numeric commonsense` | `numeric_free_form` | easier-first priority |
| `stage2_float_numeric` | `free_form` | `integer`, `float` | `english` | `table`, `chart`, `plot`, `scientific figure`, `natural image` | `arithmetic reasoning`, `statistical reasoning`, `scientific reasoning`, `numeric commonsense` | `numeric_free_form` | moderate precision |
| `stage3_hard_numeric` | `free_form` | `integer`, `float` | `english` | `geometry diagram`, `plot`, `scientific figure`, `abstract scene` | `geometry reasoning`, `algebraic reasoning`, `scientific reasoning`, `arithmetic reasoning`, `statistical reasoning` | `numeric_free_form` | `hard_only=True` |
| `stage4_multi_choice` | `multi_choice` | not constrained here | not constrained here | not constrained here | not constrained here | `multi_choice` | separate scaffold only |
| `stage5_robustness` | none | none | `chinese`, `persian` | none | none | none | `enabled=False`, reserved |

### 4.3 Stage ordering logic

Examples inside a stage are not just filtered; they are also prioritized by `stage_priority(...)` in [staged_rl/data.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/data.py).

Priority score is increased when:

- the example belongs to a preferred context family
- the raw context is one of the preferred contexts
- the skills overlap with the stage's priority skills
- the grade is one of the stage's priority grades
- for hard stages, high-school or college level gets extra priority

This makes the curriculum "easy-first within stage", not only "easy stage before hard stage".

### 4.4 Stage-specific assumptions

| Stage | Main assumption | Main risk if skipped |
| --- | --- | --- |
| `stage1_easy_numeric` | the model must first learn the contract and terminate cleanly | correctness reward gets wasted on malformed outputs |
| `stage2_float_numeric` | precision and tables/charts should be introduced only after structure is mostly stable | float parsing and tolerance do not get enough signal |
| `stage3_hard_numeric` | hard visual math should not dominate early | sparse rewards and long completions reduce learning efficiency |
| `stage4_multi_choice` | multi-choice needs a separate parser and reward branch | numeric and option-letter parsing get entangled |
| `stage5_robustness` | robustness is late-stage work | multilingual/noisy data is reserved until mixed-mode parsing and scoring are implemented |

## 5. Phase definitions

### 5.1 Phase table

| Phase | Stage Mix | Primary Goal | Default Resume | Tolerance Reward | Multi-choice Allowed | Key change from previous phase |
| --- | --- | --- | --- | --- | --- | --- |
| `phase_a` | `stage1_easy_numeric=1.00` | structure stabilization | none | no | no | start from easiest integer numeric subset |
| `phase_b` | `stage1_easy_numeric=0.70`, `stage2_float_numeric=0.30` | correctness strengthening after structure | `best_structure` | no | no | add moderate-precision data while structure is preserved |
| `phase_c` | `stage2_float_numeric=0.60`, `stage3_hard_numeric=0.40` | precision and harder reasoning | `best_composite` | yes | no | introduce float tolerance and harder multistep tasks |
| `phase_d` | `stage3_hard_numeric=1.00` | hard-subset specialization | `best_composite` | yes | no | focus only on Stage 3 with longer completions allowed |
| `phase_e` | `stage4_multi_choice=1.00` | separate multi-choice branch | `best_composite` | no | scaffold only | isolated branch, intentionally disabled by default |

### 5.2 Exact phase configuration

| Phase | Description | Output Subdir | Eval Stages |
| --- | --- | --- | --- |
| `phase_a` | Structure stabilization on Stage 1. | `phase_a` | `stage1_easy_numeric`, `stage2_float_numeric`, `stage3_hard_numeric` |
| `phase_b` | Correctness strengthening with Stage 1/2 mix. | `phase_b` | `stage1_easy_numeric`, `stage2_float_numeric`, `stage3_hard_numeric` |
| `phase_c` | Precision and harder reasoning with Stage 2/3 mix. | `phase_c` | `stage1_easy_numeric`, `stage2_float_numeric`, `stage3_hard_numeric` |
| `phase_d` | Hard Stage 3 strengthening with longer completions allowed. | `phase_d` | `stage1_easy_numeric`, `stage2_float_numeric`, `stage3_hard_numeric` |
| `phase_e` | Scaffolded multi-choice branch. | `phase_e` | `stage4_multi_choice` |

### 5.3 Phase-by-phase intent

#### Phase A

Phase A is intentionally structure-heavy. It is not supposed to maximize exact match immediately. It is supposed to make the model:

- emit both tags
- terminate before max token cutoff
- place a parseable numeric answer in `<SOLUTION>`

If this phase fails, later phases waste reward budget correcting structural issues instead of improving reasoning.

#### Phase B

Phase B assumes the model is now structurally good enough to add more pressure on correctness. It keeps structure rewards alive, but not dominant. The 70/30 stage mix still anchors learning on Stage 1 so the model does not forget the answer contract.

#### Phase C

Phase C is the first phase where the system is clearly targeting real numeric reasoning gain, not only structure:

- `stage3_hard_numeric` enters the mix
- float tolerance reward becomes active
- correctness becomes the largest initial reward component

This is the phase that produced the best larger-split checkpoint in actual runs.

#### Phase D

Phase D specializes on hard Stage 3 examples only. It exists because some harder geometry/scientific tasks may need:

- longer completion budget
- less Stage 1/2 anchoring
- more direct pressure on hard examples

In the observed runs, Phase D recovered to match the best Phase C result, but did not exceed it.

#### Phase E

Phase E is intentionally scaffolded but disabled. It exists so the repo can later support multi-choice RL without contaminating numeric reward parsing.

## 6. Reward system

### 6.1 Reward components

| Reward Function | What it checks | Numeric score logic | Why it exists |
| --- | --- | --- | --- |
| `format_reward` | tag compliance and malformed structure | `+1` for reasoning tag, `+1` for solution tag, `-0.5` for malformed answer | teaches the output contract explicitly |
| `parseable_reward` | whether the answer can be parsed | `1` if parseable, else `0` | rewards answers that the evaluator can actually read |
| `finished_reward` | whether the model finished cleanly | `+1` if finished under budget, `-1` if maxed out and unfinished, else `0` | directly penalizes truncation |
| `correctness_reward` | exact match | `1` if exact, else `0` | core correctness signal |
| `brevity_reward` | completion length discipline | shorter is better; over-budget gets penalized | keeps verbosity under control |
| `tolerance_reward` | near-correct numeric answers for float tasks | `1` for tolerance match when exact is false, else `0` | gives denser signal on float tasks |

### 6.2 Initial reward weights by phase

| Phase | `correctness_reward` | `format_reward` | `parseable_reward` | `finished_reward` | `brevity_reward` | `tolerance_reward` |
| --- | --- | --- | --- | --- | --- | --- |
| `phase_a` | `2.0` | `1.0` | `1.0` | `1.5` | `0.25` | `0.0` |
| `phase_b` | `4.0` | `0.75` | `0.75` | `1.0` | `0.20` | `0.0` |
| `phase_c` | `5.0` | `0.50` | `0.50` | `0.75` | `0.20` | `1.0` |
| `phase_d` | `5.0` | `0.50` | `0.50` | `0.75` | `0.30` | `1.0` |
| `phase_e` | `4.0` | `0.75` | `0.75` | `1.0` | `0.20` | `0.0` |

### 6.3 Reward bounds

Reward weights are not unconstrained. Each component has bounds:

| Reward | Min Weight | Max Weight |
| --- | --- | --- |
| `format_reward` | `0.25` | `2.0` |
| `parseable_reward` | `0.25` | `2.0` |
| `finished_reward` | `0.50` | `2.0` |
| `correctness_reward` | `1.0` | `8.0` |
| `brevity_reward` | `0.0` | `1.0` |
| `tolerance_reward` | `0.0` | `2.0` |

This is important: structure rewards are never allowed to decay to zero.

## 7. Metric-gated reward control

The reward schedule is not epoch-based. It is checkpoint-metric-based.

The logic is in [staged_rl/controller.py](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/staged_rl/controller.py).

### 7.1 Gating rules

| Condition | Threshold | Action |
| --- | --- | --- |
| parseability too low | `parseable_answer_rate < 0.85` | keep `parseable_reward >= 0.75` |
| solution or reasoning tags weak | `solution_tag_compliance < 0.90` or `reasoning_tag_compliance < 0.88` | keep `format_reward >= 0.75` |
| malformed too high | `malformed_answer_rate > 0.10` | keep `format_reward >= 0.75` |
| truncation too high | `truncation_rate > 0.15` | increase `finished_reward` by `0.25` |
| completion length too close to max | `average_completion_tokens > 0.85 * max_completion_length` | increase `finished_reward` by `0.25` |
| structure stable but exact match stalled | stable window of `2` checkpoints and exact gain `< 0.02` | increase `correctness_reward` by `0.50` |

### 7.2 Stability thresholds

For correctness escalation, structure must already be stable:

| Metric | Stable threshold |
| --- | --- |
| `parseable_answer_rate` | `>= 0.92` |
| `solution_tag_compliance` | `>= 0.95` |
| `reasoning_tag_compliance` | `>= 0.93` |
| `malformed_answer_rate` | `<= 0.05` |
| `truncation_rate` | `<= 0.08` |

### 7.3 Why this matters

This is the heart of the curriculum logic.

Without metric-gated control:

- correctness can dominate too early
- malformed answers remain common
- the model may learn "try to be correct" before it learns "finish correctly in the required format"

With metric-gated control:

- structure rewards stay alive until structure is genuinely stable
- correctness weight increases only once the model is already reliable

## 8. Checkpoint scoring and selection

Every checkpoint is evaluated and scored by three objective views:

- `structure_score`
- `correctness_score`
- `composite_score`

### 8.1 Structure score

| Metric | Weight |
| --- | --- |
| `parseable_answer_rate` | `+0.35` |
| `solution_tag_compliance` | `+0.20` |
| `reasoning_tag_compliance` | `+0.20` |
| `malformed_answer_rate` | `-0.15` |
| `truncation_rate` | `-0.10` |

### 8.2 Correctness score

| Metric | Weight |
| --- | --- |
| `normalized_exact_match` | `+0.70` |
| `tolerance_accuracy` | `+0.20` |
| `parseable_answer_rate` | `+0.10` |

### 8.3 Composite score

| Metric | Weight |
| --- | --- |
| `normalized_exact_match` | `+0.45` |
| `tolerance_accuracy` | `+0.15` |
| `parseable_answer_rate` | `+0.15` |
| `solution_tag_compliance` | `+0.10` |
| `reasoning_tag_compliance` | `+0.05` |
| `malformed_answer_rate` | `-0.05` |
| `truncation_rate` | `-0.05` |

### 8.4 Alias system

Each phase maintains these aliases:

| Alias | Meaning |
| --- | --- |
| `latest` | most recent checkpoint |
| `best_structure` | highest `structure_score` |
| `best_correctness` | highest `correctness_score` |
| `best_composite` | highest `composite_score` |

This is necessary because RL progress was observed to be non-monotonic. Later checkpoints were not always better.

## 9. Default model configuration

These are the default settings before any hardware-profile override.

| Parameter | Value | What it controls | Why it matters |
| --- | --- | --- | --- |
| `base_model_name` | `unsloth/Qwen2.5-VL-7B-Instruct` | base VLM checkpoint | the model being RL-tuned |
| `max_seq_length` | `16384` | total model context budget | large by default, too heavy for Kaggle T4 |
| `image_size` | `512` | resized image resolution | larger images increase visual memory cost |
| `load_in_4bit` | `True` | 4-bit quantized model load | critical for memory savings |
| `fast_inference` | `True` | Unsloth/vLLM fast generation path | important for GRPO generation efficiency |
| `gpu_memory_utilization` | `0.8` | vLLM memory target | controls how aggressively GPU memory is used |
| `lora_rank` | `8` | adapter rank | pinned to a compact adapter for all phases |
| `max_lora_rank` | `8` | maximum adapter rank exposed to vLLM | kept aligned with `lora_rank` |
| `lora_alpha` | `8` | LoRA scaling | kept aligned with the fixed adapter rank |
| `finetune_vision_layers` | `False` | train vision tower or not | kept off for memory efficiency |
| `finetune_language_layers` | `True` | train language stack | main adaptation target |
| `finetune_attention_modules` | `True` | train attention modules | needed for reasoning adaptation |
| `finetune_mlp_modules` | `True` | train MLP modules | improves adaptation capacity |
| `bias` | `none` | LoRA bias strategy | standard low-overhead choice |
| `random_state` | `3407` | seed | makes curriculum mixing reproducible |
| `use_rslora` | `False` | RSLoRA mode | disabled in this pipeline |
| `loftq_config` | `None` | LoftQ support | unused here |
| `use_gradient_checkpointing` | `unsloth` | activation checkpointing mode | reduces memory during training |

## 10. Default trainer configuration

These are the default trainer settings before phase-level or hardware overrides.

| Parameter | Value | What it controls | Practical effect |
| --- | --- | --- | --- |
| `learning_rate` | `3e-5` | optimizer step size | moderate LoRA RL learning rate |
| `adam_beta1` | `0.9` | Adam momentum | standard |
| `adam_beta2` | `0.99` | Adam variance tracking | slightly conservative |
| `weight_decay` | `0.1` | regularization | helps prevent runaway updates |
| `warmup_ratio` | `0.1` | LR warmup fraction | smoother startup |
| `lr_scheduler_type` | `cosine` | LR schedule | gradual decay |
| `optim` | `adamw_8bit` | optimizer implementation | memory-efficient optimizer |
| `logging_steps` | `1` | logging frequency | high observability |
| `log_completions` | `False` | whether to log raw generations | kept off to reduce overhead |
| `per_device_train_batch_size` | `1` | micro-batch size | necessary for large VLM RL |
| `gradient_accumulation_steps` | `2` | effective batch-size multiplier | can be raised on smaller GPUs |
| `num_generations` | `4` | GRPO samples per prompt | more samples increase signal but cost memory |
| `max_prompt_length` | `1024` | maximum prompt tokens | large default |
| `max_completion_length` | `256` | maximum completion tokens | large default |
| `temperature` | `0.7` | generation randomness during training | keeps exploration while not too noisy |
| `num_train_epochs` | `0.5` | fractional epoch count | checkpoint-heavy training style |
| `save_steps` | `60` | checkpoint frequency | enables metric-aware aliasing |
| `max_grad_norm` | `0.1` | gradient clipping | stabilizes RL |
| `report_to` | `none` | external loggers | local artifact-first |
| `importance_sampling_level` | `sequence` | GRPO importance sampling mode | sequence-level weighting |
| `mask_truncated_completions` | `False` | truncation masking | truncation is scored explicitly instead |
| `loss_type` | `dr_grpo` | RL loss | default in this implementation |
| `restore_callback_states_from_checkpoint` | `True` | resume callback/controller behavior | helps exact restart semantics |

## 11. Evaluation configuration

| Parameter | Value | Meaning |
| --- | --- | --- |
| `num_samples_per_prompt` | `4` | number of completions sampled per eval prompt |
| `temperature` | `1.0` | eval sampling temperature |
| `top_k` | `50` | eval top-k sampling |
| `max_eval_examples_per_subset` | `None` | no subset cap by default |
| `max_eval_examples_overall` | `None` | no global cap by default |
| `abs_tol_default` | `1e-6` | default absolute tolerance for floats |
| `rel_tol_default` | `1e-6` | default relative tolerance for floats |
| `save_full_completion_text` | `False` | full text is not stored by default |

Metrics computed at checkpoint time include:

- exact match
- tolerance accuracy
- best-of-k accuracy
- parseable rate
- malformed rate
- reasoning tag compliance
- solution tag compliance
- truncation rate
- average completion tokens
- repetition rate
- reward component means

## 12. Hardware profiles

### 12.1 Default vs Kaggle T4

| Parameter | `default` | `kaggle_t4` | Why the override exists |
| --- | --- | --- | --- |
| `max_seq_length` | `16384` | `1280` | huge memory reduction |
| `image_size` | `512` | `336` | lower vision memory cost |
| `gpu_memory_utilization` | `0.8` | `0.65` | safer vLLM budget |
| `lora_rank` | `8` | `8` | pinned LoRA rank |
| `max_lora_rank` | `8` | `8` | keep vLLM LoRA budget bounded |
| `lora_alpha` | `8` | `8` | matched to the fixed rank |
| `gradient_accumulation_steps` | `2` | `4` | preserve effective optimization pressure with smaller safe runtime |
| `num_generations` | `4` | `2` | large generation-memory reduction while still valid for GRPO |
| `max_prompt_length` | `1024` | `320` | lower prompt memory |
| `max_completion_length` | `256` | `64` | lower generation memory |
| `num_samples_per_prompt` | `4` | `1` | much lighter eval |
| `max_eval_examples_per_subset` | unlimited | `2` | tiny smoke-test eval on Kaggle |
| `compilation_config.level` | none explicit | `3` | stable vLLM compile setting |
| `cudagraph_mode` | none explicit | `PIECEWISE` | fixes T4/runtime compatibility problems |

### 12.2 Phase-specific Kaggle override

| Phase | Override | Why |
| --- | --- | --- |
| `phase_d` | `max_completion_length=96` | hard Stage 3 may need slightly more completion budget even on T4 |

### 12.3 Resource tradeoff

The Kaggle T4 profile does not make the problem easier. It makes each training step cheaper.

Tradeoffs accepted:

- shorter context window
- smaller images
- fewer sampled generations
- shorter completions
- smaller eval sample count
- fixed LoRA rank 8

What was gained:

- the larger split became runnable on limited VRAM
- phase-by-phase continuation worked on Kaggle T4

What was sacrificed:

- slower optimization due to more gradient accumulation
- less exploration per step
- less evaluation coverage per checkpoint
- lower maximum reasoning budget per completion

## 13. Resume and continuation semantics

| Situation | How it works |
| --- | --- |
| same-phase restart from `latest` | resume trainer state directly from checkpoint |
| same-phase manual checkpoint path | resume trainer state if selector resolves to same phase |
| cross-phase start from `best_structure` or `best_composite` | load base model, attach fresh Unsloth LoRA, warm-start adapter weights from selected checkpoint |
| explicit warm start | force adapter warm-start instead of trainer resume |
| no selector found | load base model from scratch |

This distinction matters because cross-phase continuation must preserve Unsloth runtime methods like `load_lora(...)`.

## 14. Dataset and prompt behavior

### 14.1 Metadata normalization

The pipeline normalizes:

- `question_type`
- `answer_type`
- `language`
- `source`
- `context`
- `context_family`
- `task`
- `category`
- `grade`
- `skills`
- `precision`
- `unit`
- `answer_mode`

Context families are intentionally collapsed:

- anything ending with `chart` becomes `chart`
- anything ending with `plot` becomes `plot`

This prevents stage definitions from becoming brittle to raw label variations like `bar chart` vs `chart`.

### 14.2 Numeric-stage assertions

Numeric stages are guarded so they cannot silently include non-numeric or non-free-form rows.

If a numeric stage contains:

- `question_type != free_form`
- or `answer_type` not in `{integer, float}`

the build fails.

That prevents accidental leakage of multi-choice examples into numeric reward logic.

## 15. Actual executed curriculum lineage

This section describes how the implemented phases were actually used in the observed runs.

| Run | Train Split | Phase | Resume / Warm Start | Role in lineage |
| --- | --- | --- | --- | --- |
| smoke run | `testmini` | `phase_a` | from scratch | first structure stabilization run |
| smoke run | `testmini` | `phase_b` | resume `best_structure` from Phase A | structure-preserving correctness attempt |
| smoke run | `testmini` | `phase_c` | resume `best_composite` from Phase B | first Stage 2+3 run, still correctness plateau |
| smoke run | `testmini` | `phase_d` | resume `best_composite` from Phase C | hard specialization smoke run |
| larger-split continuation | `test` | `phase_c` | warm-start from smoke `phase_c/checkpoint-119` | run that broke the `0.5` exact plateau |
| larger-split continuation | `test` | `phase_d` | resume `best_composite` from same notebook Phase C | same-notebook Stage 3 specialization |
| dedicated Phase D notebook | `test` | `phase_d` | warm-start from larger-split `phase_c/checkpoint-120` | separate preserved specialization branch |

## 16. Practical interpretation of the curriculum

### What worked on small smoke runs

The smoke runs on `testmini` proved that the curriculum and checkpoint system worked operationally:

- structure became much better
- malformed and truncated outputs dropped sharply
- resume and checkpoint aliasing worked

But correctness stayed around `0.5`.

### What changed on the larger split

The larger split did not change the curriculum logic. It changed the amount of learning signal available once the structure problem was already mostly solved.

So the causal story is:

1. curriculum + metric-gated rewards made RL stable and structurally effective
2. checkpoint selection kept the right checkpoints when training was non-monotonic
3. larger split finally moved correctness beyond the `0.5` plateau
4. Kaggle T4 memory tuning made that continuation feasible

### Current best practical checkpoint

Based on the generated results package, the recommended final checkpoint remains:

- [results/report_summary.md](/home/kgaer/code/RL_GSPO_Qwen2.5VLM/results/report_summary.md)

That recommendation corresponds to the larger-split Phase C best composite checkpoint rather than the latest checkpoint by default.

## 17. One-sentence summary

This RL curriculum is a staged, metric-gated training system that first teaches the model to answer in a clean parseable structure, then gradually increases correctness pressure and task difficulty, while using checkpoint-aware selection and hardware-specific runtime tuning to keep multimodal RL feasible on constrained GPUs like Kaggle T4.
