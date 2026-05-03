# Medium Upload Notes

Medium works best when the article is pasted into the native editor and the images are uploaded manually at their marked locations.

Best workflow:

1. Open `medium_article_for_paste.html` in Chrome.
2. Click **Copy rendered article**.
3. Paste into Medium.
4. Upload images manually if Medium skips local images.

Table note:

Medium does not reliably preserve HTML or Markdown tables when pasted into the editor. Use the table images in `images/` instead:

- `images/table_01_phase_curriculum.png`
- `images/table_02_primary_eval.png`
- `images/table_03_baseline_vs_phase_b.png`
- `images/table_04_stage_diagnostics.png`

In Medium, delete the pasted broken table text, add an image block, upload the matching table PNG, and paste the table caption underneath.

Fallback workflow:

1. Open `medium_article.md`.
2. Paste the title, subtitle/body, headings, tables, and captions into Medium.
3. At each Markdown image line, upload the matching file from `images/`.
4. Paste the italic caption below each uploaded image.
5. Medium may not preserve Markdown tables perfectly from one paste. If a table looks cramped, paste it as plain text or convert the most important one to an image using the included headline-results figure.

Image files:

- `images/figure_01_staged_metric_gated_grpo_loop.png`
- `images/figure_02_curriculum_and_eval_subsets.png`
- `images/figure_03_headline_results_table.png`
- `images/figure_04_training_reward_by_phase.png`
- `images/figure_05_training_evolution_panels.png`

Recommended Medium subtitle:

> A research breakdown of a structure-first, correctness-later GRPO pipeline for visual numeric reasoning with Qwen2.5-VL-7B.
