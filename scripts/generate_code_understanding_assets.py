#!/usr/bin/env python3
"""Generate code-understanding diagrams and entrypoint notes for the staged RL repo."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DOC_PATH = RESULTS_DIR / "code_understanding_entrypoints.md"
WALKTHROUGH_PATH = RESULTS_DIR / "deep_dive_diagram_walkthrough.md"


@dataclass(frozen=True)
class Rect:
    """Simple rectangle helper used by the SVG layout code."""

    x: float
    y: float
    w: float
    h: float

    def left(self) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    def right(self) -> tuple[float, float]:
        return (self.x + self.w, self.y + self.h / 2)

    def top(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y)

    def bottom(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h)

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


class SvgCanvas:
    """Small SVG writer for repository diagrams."""

    def __init__(self, width: int, height: int, title: str) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.elements: list[str] = []

    def add(self, raw: str) -> None:
        self.elements.append(raw)

    def rect(
        self,
        box: Rect,
        *,
        fill: str,
        stroke: str,
        stroke_width: float = 2.0,
        radius: float = 14.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            (
                f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" '
                f'rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width}"{dash_attr} />'
            )
        )

    def line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        stroke: str,
        stroke_width: float = 2.0,
        dash: str | None = None,
        marker_end: bool = False,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker_end else ""
        self.add(
            (
                f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}" fill="none"{dash_attr}{marker_attr} />'
            )
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 2.0,
        dash: str | None = None,
        marker_end: bool = False,
        fill: str = "none",
    ) -> None:
        points_attr = " ".join(f"{x},{y}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = ' marker-end="url(#arrow)"' if marker_end else ""
        self.add(
            (
                f'<polyline points="{points_attr}" stroke="{stroke}" stroke-width="{stroke_width}" '
                f'fill="{fill}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}{marker_attr} />'
            )
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        font_size: int = 18,
        weight: str = "400",
        anchor: str = "middle",
        fill: str = "#16202a",
        max_width: float | None = None,
        line_height: float = 1.25,
    ) -> None:
        lines: list[str] = []
        wrap_width = None
        if max_width is not None:
            wrap_width = max(int(max_width / max(font_size * 0.58, 1)), 10)
        for raw_line in content.splitlines():
            if not raw_line:
                lines.append("")
                continue
            if wrap_width is None:
                lines.append(raw_line)
                continue
            wrapped = textwrap.wrap(raw_line, width=wrap_width, break_long_words=False, break_on_hyphens=False)
            lines.extend(wrapped or [""])
        total_height = (len(lines) - 1) * font_size * line_height
        start_y = y - total_height / 2
        self.add(
            f'<text x="{x}" y="{start_y}" text-anchor="{anchor}" font-size="{font_size}" '
            f'font-weight="{weight}" fill="{fill}" font-family="DejaVu Sans, Arial, sans-serif">'
        )
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else font_size * line_height
            self.add(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
        self.add("</text>")

    def label_box(
        self,
        box: Rect,
        text: str,
        *,
        fill: str,
        stroke: str,
        font_size: int = 18,
        weight: str = "500",
        dash: str | None = None,
    ) -> Rect:
        self.rect(box, fill=fill, stroke=stroke, dash=dash)
        cx, cy = box.center()
        self.text(cx, cy, text, font_size=font_size, weight=weight, max_width=box.w - 24)
        return box

    def save(self, output_path: Path) -> None:
        defs = """
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L12,6 L0,12 z" fill="#25587a" />
  </marker>
</defs>
""".strip()
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}">',
            defs,
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#fbfcfe" />',
            f'<text x="36" y="48" font-size="26" font-weight="700" fill="#102030" '
            f'font-family="DejaVu Sans, Arial, sans-serif">{escape(self.title)}</text>',
            *self.elements,
            "</svg>",
        ]
        output_path.write_text("\n".join(svg), encoding="utf-8")


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _label_for_line(canvas: SvgCanvas, start: tuple[float, float], end: tuple[float, float], text: str, y_shift: float = -18) -> None:
    mx, my = _midpoint(start, end)
    label = Rect(mx - 92, my + y_shift - 18, 184, 36)
    canvas.rect(label, fill="#ffffff", stroke="#d7e2ea", stroke_width=1.2, radius=9)
    canvas.text(mx, my + y_shift, text, font_size=13, weight="500", max_width=168)


def generate_phase_run_sequence(output_path: Path) -> None:
    """Draw a single-phase run sequence/swimlane diagram."""

    canvas = SvgCanvas(1760, 1120, "One-Phase Run Sequence")
    lane_names = [
        ("CLI / main", 150, "#dceaf5"),
        ("Config / Resume", 430, "#f4e9d7"),
        ("Model Runtime", 710, "#dfeede"),
        ("Data Builder", 990, "#f5efd0"),
        ("Trainer Loop", 1270, "#dfe7f8"),
        ("Eval + Control", 1550, "#f8e3de"),
    ]

    for title, x, fill in lane_names:
        header = Rect(x - 100, 76, 200, 56)
        canvas.label_box(header, title, fill=fill, stroke="#6d8494", font_size=17, weight="600")
        canvas.line((x, 132), (x, 1040), stroke="#9eb4c2", stroke_width=2, dash="8 8")

    cli_box = canvas.label_box(Rect(60, 164, 180, 72), "parse_args()\napply_cli_overrides()", fill="#edf5fb", stroke="#6b8798", font_size=16)
    cfg_box = canvas.label_box(Rect(340, 268, 180, 92), "build_default_run_config()\napply_hardware_profile()\nbuild_resume_plan()", fill="#fff8e8", stroke="#8d7c4d", font_size=15)
    model_box = canvas.label_box(Rect(620, 390, 180, 92), "create_model_and_tokenizer()\nattach LoRA\noptional warm start", fill="#ebf7e9", stroke="#5e8b5b", font_size=15)
    data_box = canvas.label_box(Rect(900, 516, 180, 104), "load_mathvista_split(train/eval)\nbuild_phase_train_dataset()\nbuild_eval_datasets()", fill="#fff9de", stroke="#958146", font_size=15)
    trainer_box = canvas.label_box(Rect(1180, 650, 180, 80), "MetricAwareGRPOTrainer\ntrainer.train(...)", fill="#edf2ff", stroke="#637ab0", font_size=16)
    final_box = canvas.label_box(Rect(1180, 930, 180, 84), "save_lora()\nsummarize logs\nbuild diagnostics", fill="#edf2ff", stroke="#637ab0", font_size=15)
    result_box = canvas.label_box(Rect(1460, 930, 180, 84), "return results\noutput_dir\nlatest_eval_results", fill="#fff1ec", stroke="#a77062", font_size=15)

    loop = Rect(1100, 760, 580, 134)
    canvas.rect(loop, fill="#f9fbff", stroke="#9cb4c7", dash="10 8", radius=20)
    canvas.text(loop.x + 18, loop.y + 22, "Loop: every saved checkpoint", anchor="start", font_size=17, weight="700")
    eval_box = canvas.label_box(Rect(1200, 796, 160, 70), "evaluate_checkpoint()", fill="#fff3ef", stroke="#a77062", font_size=15)
    write_box = canvas.label_box(Rect(1390, 796, 160, 70), "write_checkpoint_artifacts()\nregister aliases", fill="#fff3ef", stroke="#a77062", font_size=14)
    control_box = canvas.label_box(Rect(1580, 796, 80, 70), "update\nweights", fill="#ffe9e0", stroke="#a77062", font_size=14)

    arrows = [
        (cli_box.bottom(), cfg_box.top(), "phase + profile + selectors"),
        (cfg_box.bottom(), model_box.top(), "resume plan"),
        (model_box.bottom(), data_box.top(), "model + tokenizer ready"),
        (data_box.bottom(), trainer_box.top(), "datasets + reward funcs + controller"),
    ]
    for start, end, label in arrows:
        canvas.line(start, end, stroke="#25587a", marker_end=True)
        _label_for_line(canvas, start, end, label)

    canvas.line(trainer_box.bottom(), eval_box.top(), stroke="#25587a", marker_end=True)
    _label_for_line(canvas, trainer_box.bottom(), eval_box.top(), "checkpoint-N saved")

    canvas.line(eval_box.right(), write_box.left(), stroke="#25587a", marker_end=True)
    _label_for_line(canvas, eval_box.right(), write_box.left(), "metrics + records", y_shift=-22)

    canvas.line(write_box.right(), control_box.left(), stroke="#25587a", marker_end=True)
    _label_for_line(canvas, write_box.right(), control_box.left(), "controller.update_from_metrics()", y_shift=-22)

    routed = [
        control_box.bottom(),
        (control_box.bottom()[0], 884),
        (1270, 884),
        trainer_box.bottom(),
    ]
    canvas.polyline(routed, stroke="#25587a", marker_end=True)
    canvas.text(1426, 872, "apply_reward_weights()\ncontinue training", font_size=13, weight="500", max_width=190)

    canvas.line(trainer_box.bottom(), final_box.top(), stroke="#25587a", marker_end=True)
    _label_for_line(canvas, trainer_box.bottom(), final_box.top(), "train complete")

    canvas.line(final_box.right(), result_box.left(), stroke="#25587a", marker_end=True)
    _label_for_line(canvas, final_box.right(), result_box.left(), "final payload", y_shift=-22)

    note = Rect(46, 1038, 1668, 50)
    canvas.rect(note, fill="#f3f7fb", stroke="#d3e0ea", stroke_width=1.6, radius=16)
    canvas.text(
        note.x + 18,
        note.y + 25,
        "Primary code anchors: rl_gspo_qwen2_5vlm_test3.py::main -> staged_rl.trainer_runtime::run_phase -> "
        "MetricAwareGRPOTrainerMixin::_metric_aware_save",
        anchor="start",
        font_size=15,
        weight="500",
        max_width=1630,
    )
    canvas.save(output_path)


def generate_reward_controller_flow(output_path: Path) -> None:
    """Draw a repo-specific reward-controller logic diagram."""

    canvas = SvgCanvas(1540, 1120, "Reward Controller Logic")

    input_box = canvas.label_box(
        Rect(350, 92, 840, 86),
        "Inputs: checkpoint metrics + prior reward_weights + history + RewardGateConfig thresholds",
        fill="#eef4fb",
        stroke="#6d8494",
        font_size=18,
        weight="600",
    )
    append_box = canvas.label_box(
        Rect(480, 230, 580, 78),
        "Append numeric metrics to history and copy the current weights",
        fill="#fff8e8",
        stroke="#8d7c4d",
        font_size=17,
    )
    derive_box = canvas.label_box(
        Rect(400, 360, 740, 96),
        "Derive signals from metrics:\nparseable, solution tags, reasoning tags, malformed, truncation, avg tokens, exact match",
        fill="#ebf7e9",
        stroke="#5e8b5b",
        font_size=17,
    )
    helper_box = canvas.label_box(
        Rect(400, 508, 740, 100),
        "Compute helper conditions:\nstable_structure, stable_window_ready, previous_exact, exact_delta, correctness_plateau",
        fill="#eef1ff",
        stroke="#637ab0",
        font_size=17,
    )

    parse_box = canvas.label_box(
        Rect(70, 690, 300, 126),
        "Parseable guard\nif parseable_answer_rate < parseable_floor_threshold:\nlift parseable_reward to parseable_guard_weight",
        fill="#edf8ee",
        stroke="#5e8b5b",
        font_size=15,
    )
    format_box = canvas.label_box(
        Rect(420, 690, 300, 126),
        "Format guard\nif bad tags or malformed_answer_rate is high:\nlift format_reward to formatting_guard_weight",
        fill="#fff8e8",
        stroke="#8d7c4d",
        font_size=15,
    )
    finish_box = canvas.label_box(
        Rect(770, 690, 300, 126),
        "Finish guard\nif truncation_rate is high or outputs run long:\nadd finish_step to finished_reward",
        fill="#fff1eb",
        stroke="#a77062",
        font_size=15,
    )
    correct_box = canvas.label_box(
        Rect(1120, 690, 350, 126),
        "Correctness escalation\nif stable structure + enough history + exact-match plateau:\nadd correctness_step to correctness_reward",
        fill="#edf2ff",
        stroke="#637ab0",
        font_size=15,
    )

    merge_box = canvas.label_box(
        Rect(400, 876, 740, 92),
        "Apply floors and bounds:\nformat_reward >= format_floor, parseable_reward >= parseable_floor, "
        "finished_reward >= finish_floor, then clamp every component to its min/max range",
        fill="#f8fbff",
        stroke="#7c92a3",
        font_size=16,
    )
    persist_box = canvas.label_box(
        Rect(430, 1016, 680, 66),
        "Persist last_decision, append decision_history, replace reward_weights, return updated weights",
        fill="#e8f0fb",
        stroke="#6d8494",
        font_size=16,
        weight="600",
    )
    note = canvas.label_box(
        Rect(1160, 232, 300, 126),
        "Threshold source:\nstaged_rl/config.py::RewardGateConfig\n\nUpdate logic:\nstaged_rl/controller.py::RewardController.update_from_metrics",
        fill="#f9fbff",
        stroke="#9cb4c7",
        font_size=15,
    )

    vertical_steps = [input_box, append_box, derive_box, helper_box]
    for prev_box, next_box in zip(vertical_steps, vertical_steps[1:]):
        canvas.line(prev_box.bottom(), next_box.top(), stroke="#25587a", marker_end=True)

    for rule_box in (parse_box, format_box, finish_box, correct_box):
        canvas.line(helper_box.bottom(), rule_box.top(), stroke="#25587a", marker_end=True)
        canvas.line(rule_box.bottom(), merge_box.top(), stroke="#25587a", marker_end=True)

    canvas.line(merge_box.bottom(), persist_box.top(), stroke="#25587a", marker_end=True)
    canvas.line(note.left(), (helper_box.right()[0] + 16, helper_box.right()[1]), stroke="#7c92a3", stroke_width=1.8, dash="6 6")

    canvas.save(output_path)


def generate_checkpoint_artifact_map(output_path: Path) -> None:
    """Draw the checkpoint artifact write/update flow."""

    canvas = SvgCanvas(1820, 1220, "Checkpoint Artifact Map")

    eval_results = canvas.label_box(
        Rect(70, 130, 270, 92),
        "evaluate_checkpoint()\nreturns metrics, subset_metrics,\nsubset_results",
        fill="#fff1eb",
        stroke="#a77062",
        font_size=17,
    )
    weights_input = canvas.label_box(
        Rect(70, 266, 270, 76),
        "reward_controller.current_weights()",
        fill="#edf8ee",
        stroke="#5e8b5b",
        font_size=16,
    )
    state_input = canvas.label_box(
        Rect(70, 378, 270, 76),
        "reward_controller.to_dict()",
        fill="#edf8ee",
        stroke="#5e8b5b",
        font_size=16,
    )
    info_input = canvas.label_box(
        Rect(70, 490, 270, 90),
        "checkpoint_info\ncheckpoint_path\nglobal_step\nphase_name",
        fill="#fff8e8",
        stroke="#8d7c4d",
        font_size=16,
    )
    write_box = canvas.label_box(
        Rect(430, 286, 330, 116),
        "write_checkpoint_artifacts()\ncompute_checkpoint_scores()\nreturn registry entry",
        fill="#eef4fb",
        stroke="#6d8494",
        font_size=19,
        weight="600",
    )

    checkpoint_container = Rect(860, 94, 880, 700)
    canvas.rect(checkpoint_container, fill="#fbfcfe", stroke="#9cb4c7", stroke_width=2.0, radius=18)
    canvas.text(checkpoint_container.x + 20, checkpoint_container.y + 22, "Inside checkpoint-N/", anchor="start", font_size=21, weight="700")

    pre_header = Rect(890, 144, 820, 46)
    canvas.label_box(pre_header, "Written before controller update", fill="#eef4fb", stroke="#c9d9e6", font_size=16, weight="600")
    pre_boxes = [
        Rect(900, 216, 180, 74),
        Rect(1100, 216, 180, 74),
        Rect(1300, 216, 180, 74),
        Rect(1500, 216, 180, 74),
        Rect(900, 318, 180, 74),
        Rect(1100, 318, 180, 74),
        Rect(1300, 318, 180, 74),
        Rect(1500, 318, 180, 74),
    ]
    pre_texts = [
        "eval_metrics.json",
        "subset_metrics.json",
        "per_prompt_records.json",
        "per_sample_records.jsonl",
        "checkpoint_info.json",
        "summary.txt",
        "reward_weights.json\ninitial write",
        "controller_state.json\ninitial write",
    ]
    for box, text in zip(pre_boxes, pre_texts):
        canvas.label_box(box, text, fill="#ffffff", stroke="#c8d5de", font_size=15)

    post_header = Rect(890, 448, 820, 46)
    canvas.label_box(post_header, "Written after controller.update_from_metrics()", fill="#fff1eb", stroke="#eed0c6", font_size=16, weight="600")
    post_boxes = [
        Rect(930, 522, 220, 86),
        Rect(1185, 522, 220, 86),
        Rect(1440, 522, 220, 86),
    ]
    post_texts = [
        "controller_decision.json",
        "reward_weights.json\noverwritten with updated weights",
        "controller_state.json\noverwritten with updated history + decision",
    ]
    for box, text in zip(post_boxes, post_texts):
        canvas.label_box(box, text, fill="#fffaf8", stroke="#dcb8ab", font_size=15)

    register_box = canvas.label_box(
        Rect(420, 708, 350, 100),
        "CheckpointRegistry.register(entry)\nrefresh latest / best_structure /\nbest_correctness / best_composite",
        fill="#fff8e8",
        stroke="#8d7c4d",
        font_size=18,
    )
    controller_box = canvas.label_box(
        Rect(420, 864, 350, 108),
        "RewardController.update_from_metrics()\nthen apply_reward_weights(trainer, updated_weights)",
        fill="#edf8ee",
        stroke="#5e8b5b",
        font_size=18,
    )

    phase_container = Rect(860, 850, 880, 276)
    canvas.rect(phase_container, fill="#fbfcfe", stroke="#9cb4c7", stroke_width=2.0, radius=18)
    canvas.text(phase_container.x + 20, phase_container.y + 22, "Inside phase output dir/", anchor="start", font_size=21, weight="700")
    phase_boxes = [
        Rect(905, 922, 240, 90),
        Rect(1180, 922, 240, 90),
        Rect(1455, 922, 240, 90),
    ]
    phase_texts = [
        "checkpoint_registry.json\nordered checkpoint entries",
        "aliases/latest.json\naliases/best_structure.json",
        "aliases/best_correctness.json\naliases/best_composite.json",
    ]
    for box, text in zip(phase_boxes, phase_texts):
        canvas.label_box(box, text, fill="#ffffff", stroke="#c8d5de", font_size=16)

    for source in (eval_results, weights_input, state_input, info_input):
        canvas.line(source.right(), write_box.left(), stroke="#25587a", marker_end=True)

    canvas.line(write_box.right(), (checkpoint_container.x, write_box.center()[1]), stroke="#25587a", marker_end=True)
    canvas.text(812, write_box.center()[1] - 22, "pre-update artifact write", font_size=14, weight="500")

    canvas.line(write_box.bottom(), register_box.top(), stroke="#25587a", marker_end=True)
    canvas.line(register_box.right(), (phase_container.x, register_box.center()[1]), stroke="#25587a", marker_end=True)
    canvas.text(804, register_box.center()[1] - 20, "registry + aliases", font_size=14, weight="500")

    canvas.line((1300, checkpoint_container.y + checkpoint_container.h), (1300, controller_box.top()[1]), stroke="#25587a", stroke_width=1.8, dash="6 6")
    canvas.line(register_box.bottom(), controller_box.top(), stroke="#25587a", marker_end=True)
    canvas.line(controller_box.right(), (checkpoint_container.x, 565), stroke="#25587a", marker_end=True)
    canvas.text(804, 796, "controller decision path", font_size=14, weight="500")
    canvas.text(800, 566, "post-update checkpoint files", font_size=14, weight="500")

    footnote = Rect(60, 1144, 1700, 48)
    canvas.rect(footnote, fill="#f3f7fb", stroke="#d3e0ea", stroke_width=1.6, radius=16)
    canvas.text(
        footnote.x + 18,
        footnote.y + 24,
        "Important nuance: reward_weights.json and controller_state.json are written twice per checkpoint. "
        "The first write captures pre-update state inside write_checkpoint_artifacts(); the later write stores the post-controller state.",
        anchor="start",
        font_size=15,
        weight="500",
        max_width=1660,
    )

    canvas.save(output_path)


def generate_entrypoint_doc(output_path: Path) -> None:
    """Write a repo-specific diagram-to-code map."""

    content = """# Code Understanding Entry Points

Generated by `scripts/generate_code_understanding_assets.py`.

## Deep-Dive Diagram To Code Map

| Diagram block | Primary code anchors | What happens here |
| --- | --- | --- |
| Base VLM | `staged_rl/config.py::ModelConfig` and `staged_rl/trainer_runtime.py::create_model_and_tokenizer` | Defines the base model name and attaches the trainable LoRA adapter through Unsloth. |
| Phase config A/B/C/D | `staged_rl/config.py::build_default_phase_specs` | Defines stage mixes, initial reward weights, default resume selectors, and phase-specific overrides. |
| MathVista train split | `staged_rl/data.py::load_mathvista_split` | Loads and enriches the train split with flattened metadata, answer mode, image normalization, and gold option letters. |
| Stage / prompt builder | `staged_rl/data.py::build_stage_dataset`, `staged_rl/data.py::build_prompt_text`, `staged_rl/data.py::build_phase_train_dataset` | Filters examples into curriculum stages, wraps the strict tag contract, and interleaves stage datasets using the phase mix. |
| Model prep / resume / warm start | `staged_rl/checkpointing.py::build_resume_plan`, `staged_rl/trainer_runtime.py::_warm_start_peft_adapter` | Decides whether a selector means trainer resume or cross-phase LoRA warm start, then loads the adapter accordingly. |
| GRPO fine-tuning | `staged_rl/trainer_runtime.py::run_phase`, `staged_rl/trainer_runtime.py::build_metric_trainer_class` | Instantiates the metric-aware GRPO trainer and runs the training loop. |
| Rule-based reward stack | `staged_rl/rewarding.py::build_reward_functions`, `staged_rl/parsing.py` helpers | Implements formatting, parseability, finished-answer, correctness, brevity, and tolerance rewards on top of parsed completions. |
| Checkpoint eval | `staged_rl/evaluation.py::evaluate_checkpoint`, `staged_rl/evaluation.py::evaluate_dataset_subset` | Runs generation on held-out subsets, scores each sampled completion, and aggregates prompt-level and sample-level metrics. |
| Reward controller | `staged_rl/controller.py::RewardController.update_from_metrics` | Applies metric-gated rules to modify reward weights after checkpoint evaluation. |
| Checkpoint artifacts | `staged_rl/checkpointing.py::write_checkpoint_artifacts` | Writes metrics, subset summaries, prompt/sample records, checkpoint info, and human-readable summaries into each checkpoint directory. |
| Scores + aliases | `staged_rl/checkpointing.py::compute_checkpoint_scores`, `staged_rl/checkpointing.py::CheckpointRegistry.register` | Computes structure/correctness/composite scores and refreshes `latest`, `best_structure`, `best_correctness`, and `best_composite`. |
| Recommended final / lineage | `results/Chronological_Flow.txt`, `results/plots/resume_lineage.svg`, `results/plots/checkpoint_frontier_scatter.svg` | Explains how the notebook-to-notebook continuation chain produced the final recommended checkpoint. |

## Recommended Reading Order

1. `rl_gspo_qwen2_5vlm_test3.py::main`
2. `staged_rl/config.py::build_default_stage_specs`
3. `staged_rl/config.py::build_default_phase_specs`
4. `staged_rl/checkpointing.py::build_resume_plan`
5. `staged_rl/trainer_runtime.py::create_model_and_tokenizer`
6. `staged_rl/data.py::load_mathvista_split`
7. `staged_rl/data.py::build_phase_train_dataset`
8. `staged_rl/rewarding.py::build_reward_functions`
9. `staged_rl/trainer_runtime.py::run_phase`
10. `staged_rl/evaluation.py::evaluate_checkpoint`
11. `staged_rl/controller.py::RewardController.update_from_metrics`
12. `staged_rl/checkpointing.py::CheckpointRegistry.register`

## New Understanding Assets

- `results/plots/phase_run_sequence.svg`: one end-to-end phase run with the checkpoint-eval loop.
- `results/plots/reward_controller_flow.svg`: exact logic flow for reward-weight updates.
- `results/plots/checkpoint_artifact_map.svg`: file-level write/update map for each checkpoint save.
    """
    output_path.write_text(content, encoding="utf-8")


def generate_deep_dive_walkthrough(output_path: Path) -> None:
    """Write the left-to-right image walkthrough note."""

    content = """# Qwen2.5-VL Staged RL Diagram Walkthrough

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
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_phase_run_sequence(PLOTS_DIR / "phase_run_sequence.svg")
    generate_reward_controller_flow(PLOTS_DIR / "reward_controller_flow.svg")
    generate_checkpoint_artifact_map(PLOTS_DIR / "checkpoint_artifact_map.svg")
    generate_entrypoint_doc(DOC_PATH)
    generate_deep_dive_walkthrough(WALKTHROUGH_PATH)
    print("Generated code-understanding assets:")
    print(f"- {PLOTS_DIR / 'phase_run_sequence.svg'}")
    print(f"- {PLOTS_DIR / 'reward_controller_flow.svg'}")
    print(f"- {PLOTS_DIR / 'checkpoint_artifact_map.svg'}")
    print(f"- {DOC_PATH}")
    print(f"- {WALKTHROUGH_PATH}")


if __name__ == "__main__":
    main()
