# Speaker Notes

## Slide 1
- Today I will present a staged metric-gated GRPO fine-tuning pipeline for visual numeric reasoning.
- The goal is not to introduce a new GRPO loss.
- The goal is to make reinforcement learning more stable for a vision-language model on MathVista-style numeric tasks.
- The central idea is simple: first make the model answer in a reliable structure, then push harder on correctness.

## Slide 2
- This is the base model used in the work.
- We use the Unsloth release of Qwen2.5-VL-7B-Instruct.
- It is a vision-language model, so it can read both the image and the text question.
- All training and evaluation comparisons start from this same base model setup.

## Slide 3
- The model is loaded in a memory-efficient setup.
- Weights are quantized to 4-bit precision to fit within the available GPU memory.
- The LoRA adapter uses rank 8, with alpha 8.
- This means we train a small adapter instead of updating the full 7B model weights.
- That keeps the experiment feasible on the Kaggle T4 setting.

## Slide 4
- The dataset is MathVista from Hugging Face.
- MathVista combines visual perception with mathematical reasoning.
- The examples include charts, plots, natural images, scientific figures, and geometry-like visual questions.
- This makes it a good benchmark for visual numeric reasoning rather than pure text math.

## Slide 5
- This slide shows the Hugging Face dataset source.
- The work uses MathVista splits from AI4Math/MathVista.
- Training is drawn from the TEST split in this project setup.
- Final reporting uses TESTMINI, so the reported evaluation is separate from the training split.

## Slide 6
- This slide summarizes the main ingredients of the experiment.
- The model is a Qwen2.5-VL instruction model loaded through Unsloth.
- The data comes from MathVista, with TEST used for training and TESTMINI used for reporting.
- The training method is reinforcement learning from verifiable rewards.
- The implementation uses staged metric-gated GRPO to control reward pressure over time.

## Slide 7
- This is the headline result table.
- The trained adapter improves exact match and tolerance accuracy over the baseline.
- It also greatly improves parseability and tag compliance.
- That matters because a numeric answer is only useful if it can be reliably extracted and checked.
- The main takeaway is that staged RL improves both answer structure and numeric correctness.

## Slide 8
- This slide introduces the full staged fine-tuning pipeline.
- Each phase has its own data mix and reward-weight starting point.
- The model is periodically checkpointed and evaluated.
- Those checkpoint metrics decide whether reward weights should change for the next segment.
- The best checkpoints are then used for comparison and continuation.

## Slide 9
- The dataset is divided into stages by difficulty and answer type.
- Stage 1 is easier numeric reasoning, often integer-based.
- Stage 2 introduces medium difficulty or floating-point numeric answers.
- Stage 3 contains harder visual numeric reasoning, such as plots or more complex reasoning.
- The stages are mainly a curriculum and diagnostic tool.

## Slide 10
- This is a Stage 1 example.
- The image is a natural image, and the expected answer is an integer.
- The reasoning is mostly visual counting or simple arithmetic.
- This stage is useful for teaching the model to produce clean structured answers before harder tasks.

## Slide 11
- This is a Stage 2 example.
- The task uses a scientific-style figure and expects a floating-point answer.
- This is harder than simple counting because the model must interpret a plotted or measured quantity.
- Stage 2 helps bridge from easy numeric tasks to more difficult quantitative reasoning.

## Slide 12
- This is a Stage 3 example.
- The question requires interpreting a plot and applying algebraic or multi-step reasoning.
- The model must connect visual information with mathematical reasoning.
- This stage is used as the harder diagnostic subset.

## Slide 13
- This figure shows how phases and stages connect.
- Phase A uses Stage 1 only, so the model first learns reliable structure.
- Phase B mixes Stage 1 and Stage 2, so correctness pressure increases.
- Phase C mixes Stage 2 and Stage 3, so harder reasoning enters the curriculum.
- The headline evaluation is still the same eval_overall_numeric subset, so A, B, and C can be compared fairly.

## Slide 14
- This slide introduces the GRPO objective used in the training loop.
- GRPO compares sampled completions within a group and updates the model toward higher-reward completions.
- The KL term keeps the model from drifting too far from the reference behavior.
- In this project, the important design choice is how the reward is constructed and scheduled.

## Slide 15
- The reward is built from several measurable components.
- Correctness checks whether the extracted answer matches the gold answer.
- Formatting and parseability check whether the answer follows the required tags and can be read by the evaluator.
- Finished, tolerance, and brevity help discourage truncated, malformed, or unnecessarily long outputs.
- Together, these rewards make the output both checkable and correct.

## Slide 16
- The total reward is a weighted sum of the reward components.
- Changing the weights changes what the model is pushed to learn.
- Early in training, structure-related weights are more important.
- Later phases increase correctness pressure once the output format is stable.

## Slide 17
- This table shows the initial reward weights at the start of each phase.
- Phase A is structure-first, with strong pressure on parseability, formatting, and completion.
- Phase B increases correctness pressure while still keeping structure constraints active.
- Phase C increases pressure for harder numeric correctness and tolerance.
- These are starting weights; the controller can update them after checkpoint evaluation.

## Slide 18
- This slide shows the controller rules.
- At each checkpoint, evaluation metrics decide whether a reward weight should increase.
- For example, if structure is stable, the controller can shift pressure toward correctness.
- If outputs become malformed or truncated, the controller can protect formatting and completion behavior.
- This is why the training is metric-gated rather than using one fixed reward schedule.

## Slide 19
- This is the complete loop used in each phase.
- The base model is prepared with LoRA, then trained with GRPO on the staged data mix.
- Every checkpoint is evaluated on exact match, tolerance, parseability, tag compliance, and truncation.
- The controller uses those metrics to update reward weights for the next segment.
- Aliases such as best_structure, best_correctness, and best_composite identify which checkpoints matter most.

## Slide 20
- This plot shows reward across the cumulative training steps.
- Phase A reaches high reward quickly because structure rewards are easier to satisfy.
- Phases B and C have stronger correctness pressure, so their logged reward is not directly comparable to Phase A.
- The important interpretation is the trajectory: structure stabilizes first, then correctness improves.
- This is why the method is staged instead of using one reward setting throughout.

## Slide 21
- This panel plot shows how training behavior evolves across checkpoints.
- All panels share the same checkpoint x-axis, so changes can be correlated across metrics.
- The notebook evolution panel shows which checkpoint was used to restart the next phase.
- The dashed arrows show the handoff from A237 to Phase B and from B180 to Phase C.
- The plot is mainly a diagnostic view; the final comparison uses the full TESTMINI evaluation table.

## Slide 22
- Now we move from training diagnostics to final evaluation.
- The final evaluation is performed on the TESTMINI split.
- This split is separate from the TEST split used for training.
- The purpose is to compare baseline and trained checkpoints under the same evaluation setting.

## Slide 23
- The headline evaluation subset is eval_overall_numeric on TESTMINI.
- In simple terms, this subset is approximately the union of Stage 1, Stage 2, and Stage 3 numeric examples on TESTMINI.
- That is why it is suitable for the main Baseline, Phase A, Phase B, and Phase C comparison.
- Stage-specific evals are still useful, but they are diagnostic rather than the headline metric.

## Slide 24
- To conclude, the main result is that staged metric-gated GRPO improves visual numeric reasoning over the baseline.
- The method works by stabilizing output structure first, then increasing correctness pressure.
- The final comparison uses the same TESTMINI numeric subset for baseline and phase checkpoints.
- Thank you. I am happy to take questions.

## Slide 25
- This backup slide lists the runtime and evaluation settings.
- The LoRA rank is 8, and the model is loaded in 4-bit mode.
- Training uses a short generation budget to fit within memory limits.
- Final evaluation uses one completion per prompt and no subset cap.
- The primary reported subset is eval_overall_numeric with 456 examples.

## Slide 26
- This backup slide shows the Unsloth notebook source area.
- It is included to document the model-loading and notebook workflow used as a starting point.
- The project adapts this setup for staged GRPO training.
- This is useful if someone asks how the Kaggle runtime was configured.

## Slide 27
- This backup slide gives the high-level RL setup.
- Training data produces prompts, the policy generates answers, and a verifier assigns scalar rewards.
- GRPO then uses those rewards to update the policy.
- The key point is that the reward is verifiable because the final numeric answer can be checked.

## Slide 28
- This backup slide expands the GRPO computation.
- Multiple completions are sampled for a prompt during training.
- Rewards are compared within the group to estimate advantage.
- The update increases the probability of better completions while controlling drift with KL regularization.

## Slide 29
- This backup slide gives the detailed reward explanation.
- Total reward is a linear combination of component rewards and their weights.
- Those weights are initialized at each phase and can be adjusted after checkpoint evaluation.
- The objective is to stabilize formatting and parseability first, then push exact numeric correctness.
- This is the mechanism behind the structure-first, correctness-later behavior.

## Slide 30
- This backup slide shows the memory-related defaults.
- The settings were reduced to fit within the Kaggle T4 memory limit.
- Sequence length, image size, prompt length, and completion length all affect memory use.
- These constraints explain why the final experiments use compact LoRA and short completions.

## Slide 31
- This backup slide shows the split flow.
- Training uses the TEST split, rewards are computed per sample, and the model updates through GRPO.
- Checkpoint evaluation is used during training to update controller metrics and reward weights.
- Final reporting is performed separately on TESTMINI.
- This separation is important for fair comparison.

## Slide 32
- This backup slide explains where the metrics are used.
- Some metrics are per-sample rewards, such as correctness and formatting checks.
- Some are dataset-level averages, such as parseability and truncation rate.
- The controller uses these metrics to decide reward-weight updates.
- Selection scores combine multiple metrics to choose useful checkpoints.

## Slide 33
- This backup slide shows the composite scoring formulas.
- Structure score rewards parseable and well-formed outputs and penalizes truncation.
- Correctness score focuses on exact and tolerance accuracy.
- Composite score combines structure and correctness into one checkpoint-selection signal.
- This is how checkpoint aliases are assigned consistently.

## Slide 34
- This backup slide explains total sequence length.
- The total length is not just the generated answer length.
- It also includes prompt tokens, image tokens, chat template tokens, special tokens, and padding.
- That is why image size and prompt length are important memory knobs.
- Reducing these settings helped make the runs feasible on limited hardware.

## Slide 35
- This final backup slide is reserved for additional implementation details or questions.
- If there are no further technical questions, I would skip this slide during the talk.
- The main story remains the same: staged data, metric-gated rewards, checkpoint evaluation, and fair TESTMINI reporting.

