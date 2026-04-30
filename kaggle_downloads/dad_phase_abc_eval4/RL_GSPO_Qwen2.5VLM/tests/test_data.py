"""Tests for stage filters and dataset analysis."""

import sys
import types
import unittest
from unittest import mock

from staged_rl.config import build_default_run_config, build_default_stage_specs
from staged_rl.data import (
    _apply_runtime_image_transform,
    _active_eval_stage_names,
    _active_training_stage_names,
    _materialize_image_column,
    analyze_dataset_records,
    build_eval_datasets,
    build_phase_train_dataset,
    build_stage_dataset,
    build_prompt_text,
    filter_supported_answer_mode_rows,
    match_filter_spec,
    normalize_context_family,
    normalize_image_payload,
)


SAMPLE_NUMERIC = {
    "pid": "1",
    "question_type": "free_form",
    "answer_type": "integer",
    "language": "english",
    "context": "synthetic scene",
    "context_family": "synthetic scene",
    "source": "clevr-math",
    "task": "math word problem",
    "category": "math-targeted-vqa",
    "grade": "elementary school",
    "skills": ["arithmetic reasoning"],
    "unit": "",
    "precision": None,
    "answer_mode": "numeric_free_form",
    "answer": "4",
    "question": "How many objects are left?",
}

SAMPLE_MULTI = {
    "pid": "2",
    "question_type": "multi_choice",
    "answer_type": "text",
    "language": "english",
    "context": "geometry diagram",
    "context_family": "geometry diagram",
    "source": "geometry3k",
    "task": "geometry problem solving",
    "category": "math-targeted-vqa",
    "grade": "high school",
    "skills": ["geometry reasoning", "algebraic reasoning"],
    "unit": "",
    "precision": None,
    "answer_mode": "multi_choice",
    "answer": "97",
    "question": "Find m angle H",
}


class _FakeDataset:
    def __init__(self, rows):
        if isinstance(rows, dict):
            rows = [rows]
        self._rows = [dict(row) for row in rows]
        self.column_names = list(self._rows[0].keys()) if self._rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def filter(self, predicate):
        return _FakeDataset([row for row in self._rows if predicate(dict(row))])

    def map(self, transform):
        mapped_rows = []
        for row in self._rows:
            transformed = transform(dict(row))
            merged = dict(row)
            if transformed:
                merged.update(transformed)
            mapped_rows.append(merged)
        return _FakeDataset(mapped_rows)

    def sort(self, column_name, reverse=False):
        return _FakeDataset(sorted(self._rows, key=lambda row: row[column_name], reverse=reverse))

    def remove_columns(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]
        trimmed_rows = []
        for row in self._rows:
            trimmed = dict(row)
            for name in column_names:
                trimmed.pop(name, None)
            trimmed_rows.append(trimmed)
        return _FakeDataset(trimmed_rows)

    def cast_column(self, column_name, feature):
        return self

    def with_transform(self, transform):
        raise AssertionError("with_transform should not be used by the staged training builders.")


class DataTests(unittest.TestCase):
    class _FakeImage:
        def __init__(self, size=(16, 16), mode="L"):
            self.size = size
            self.mode = mode

        def resize(self, size):
            return DataTests._FakeImage(size=size, mode=self.mode)

        def convert(self, mode):
            return DataTests._FakeImage(size=self.size, mode=mode)

    def test_normalize_context_family(self):
        self.assertEqual(normalize_context_family("bar chart"), "chart")
        self.assertEqual(normalize_context_family("line plot"), "plot")
        self.assertEqual(normalize_context_family("geometry diagram"), "geometry diagram")

    def test_stage_filters(self):
        stages = build_default_stage_specs()
        self.assertTrue(match_filter_spec(SAMPLE_NUMERIC, stages["stage1_easy_numeric"].filter_spec))
        self.assertFalse(match_filter_spec(SAMPLE_MULTI, stages["stage1_easy_numeric"].filter_spec))
        self.assertTrue(match_filter_spec(SAMPLE_MULTI, stages["stage4_multi_choice"].filter_spec))

    def test_dataset_analysis(self):
        stages = build_default_stage_specs()
        analysis = analyze_dataset_records([SAMPLE_NUMERIC, SAMPLE_MULTI], stages)
        self.assertEqual(analysis["total_rows"], 2)
        self.assertEqual(analysis["field_counts"]["question_type"]["free_form"], 1)
        self.assertEqual(analysis["field_counts"]["answer_mode"]["numeric_free_form"], 1)
        self.assertEqual(analysis["field_counts"]["answer_mode"]["multi_choice"], 1)
        self.assertEqual(analysis["stage_summaries"]["stage1_easy_numeric"]["count"], 1)
        self.assertEqual(analysis["stage_summaries"]["stage4_multi_choice"]["count"], 1)

    def test_filter_supported_answer_mode_rows_drops_unsupported_rows(self):
        dataset = _FakeDataset(
            [
                SAMPLE_NUMERIC,
                {
                    **SAMPLE_NUMERIC,
                    "pid": "3",
                    "answer_mode": "unsupported",
                    "question_type": "free_form",
                    "answer_type": "text",
                },
            ]
        )

        filtered = filter_supported_answer_mode_rows(dataset)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["pid"], "1")

    def test_build_prompt_text_rejects_unsupported_answer_mode(self):
        with self.assertRaises(ValueError):
            build_prompt_text(
                {
                    "query": "What is the answer?",
                    "answer_mode": "unsupported",
                }
            )

    def test_normalize_image_payload_decodes_dataset_dict(self):
        fake_image = self._FakeImage()
        with mock.patch("staged_rl.data._load_image_payload", return_value=fake_image):
            normalized = normalize_image_payload({"path": "/tmp/example.png", "bytes": None}, image_size=64)

        self.assertIsInstance(normalized, self._FakeImage)
        self.assertEqual(normalized.size, (64, 64))
        self.assertEqual(normalized.mode, "RGB")

    def test_normalize_image_payload_prefers_existing_image_object(self):
        fake_image = self._FakeImage()
        normalized = normalize_image_payload(fake_image, image_size=32)

        self.assertIsInstance(normalized, self._FakeImage)
        self.assertEqual(normalized.size, (32, 32))
        self.assertEqual(normalized.mode, "RGB")

    def test_runtime_image_transform_normalizes_interleaved_dict_payloads(self):
        class _FakeDataset:
            def __init__(self, image):
                self.image = image

            def with_transform(self, transform):
                return transform({"image": self.image})

        fake_image = self._FakeImage()
        with mock.patch("staged_rl.data.normalize_image_payload", return_value=fake_image) as patched:
            transformed = _apply_runtime_image_transform(_FakeDataset({"path": "/tmp/example.png"}), image_size=48)

        patched.assert_called_once_with({"path": "/tmp/example.png"}, image_size=48)
        self.assertIs(transformed["image"], fake_image)

    def test_materialize_image_column_normalizes_eagerly(self):
        class _FakeDataset:
            def __init__(self, image):
                self.image = image

            def map(self, transform):
                return transform({"image": self.image})

        fake_image = self._FakeImage()
        with mock.patch("staged_rl.data.normalize_image_payload", return_value=fake_image) as patched:
            transformed = _materialize_image_column(_FakeDataset({"path": "/tmp/example.png"}), image_size=48)

        patched.assert_called_once_with({"path": "/tmp/example.png"}, image_size=48)
        self.assertIs(transformed["image"], fake_image)

    def test_materialize_image_column_casts_image_feature_when_available(self):
        class _FakeImageFeature:
            pass

        class _FakeDataset:
            def __init__(self, image):
                self.image = image
                self.column_names = ["image"]
                self.cast_calls = []

            def map(self, transform):
                transformed = transform({"image": self.image})
                self.image = transformed["image"]
                return self

            def cast_column(self, column_name, feature):
                self.cast_calls.append((column_name, feature))
                return self

        fake_image = self._FakeImage()
        fake_datasets_module = types.ModuleType("datasets")
        fake_datasets_module.Image = _FakeImageFeature
        dataset = _FakeDataset({"path": "/tmp/example.png"})

        with mock.patch("staged_rl.data.normalize_image_payload", return_value=fake_image):
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets_module}):
                transformed = _materialize_image_column(dataset, image_size=48)

        self.assertIs(transformed, dataset)
        self.assertEqual(dataset.cast_calls[0][0], "image")
        self.assertIsInstance(dataset.cast_calls[0][1], _FakeImageFeature)
        self.assertIs(dataset.image, fake_image)

    def test_active_training_stage_names_skip_disabled_stages(self):
        run_config = build_default_run_config("phase_b")
        run_config.stages["stage2_float_numeric"].enabled = False

        active_stage_names = _active_training_stage_names(run_config.phases["phase_b"], run_config.stages)

        self.assertEqual(active_stage_names, ["stage1_easy_numeric"])

    def test_active_eval_stage_names_use_phase_order_and_skip_disabled_stages(self):
        run_config = build_default_run_config("phase_b")
        run_config.stages["stage2_float_numeric"].enabled = False

        active_stage_names = _active_eval_stage_names(run_config)

        self.assertEqual(active_stage_names, ["stage1_easy_numeric", "stage3_hard_numeric"])

    def test_build_stage_dataset_materializes_images_eagerly(self):
        stage_spec = build_default_stage_specs()["stage1_easy_numeric"]
        base_dataset = _FakeDataset(
            {
                "pid": "1",
                "question_type": "free_form",
                "answer_type": "integer",
                "language": "english",
                "context": "synthetic scene",
                "context_family": "synthetic scene",
                "source": "clevr-math",
                "task": "math word problem",
                "category": "math-targeted-vqa",
                "grade": "elementary school",
                "skills": ["arithmetic reasoning"],
                "unit": "",
                "precision": None,
                "answer_mode": "numeric_free_form",
                "answer": "4",
                "question": "How many objects are left?",
                "image": {"path": "/tmp/example.png"},
                "decoded_image": {"path": "/tmp/example.png"},
            }
        )
        observed = []

        def _spy_materialize(dataset, image_size=512):
            observed.append((dataset, image_size))
            return dataset

        with mock.patch("staged_rl.data._maybe_apply_chat_template", side_effect=lambda dataset, tokenizer: dataset):
            with mock.patch("staged_rl.data._materialize_image_column", side_effect=_spy_materialize):
                with mock.patch(
                    "staged_rl.data._apply_runtime_image_transform",
                    side_effect=AssertionError("legacy runtime image transform should not be used"),
                ):
                    result = build_stage_dataset(base_dataset, stage_spec, tokenizer=object(), image_size=48)

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][1], 48)
        self.assertIsInstance(result, _FakeDataset)
        self.assertNotIn("decoded_image", result.column_names)

    def test_build_phase_train_dataset_skips_disabled_stages(self):
        run_config = build_default_run_config("phase_b")
        phase_config = run_config.phases["phase_b"]
        stage_specs = build_default_stage_specs()
        stage_specs["stage2_float_numeric"].enabled = False
        build_calls = []

        def _fake_build_stage_dataset(base_dataset, stage_spec, tokenizer, image_size=512):
            build_calls.append(stage_spec.name)
            return f"dataset:{stage_spec.name}"

        def _fake_interleave_datasets(datasets, probabilities, seed, stopping_strategy):
            raise AssertionError("interleave_datasets should not be called when only one stage remains active")

        with mock.patch("staged_rl.data.build_stage_dataset", side_effect=_fake_build_stage_dataset):
            with mock.patch("staged_rl.data._load_dataset_imports", return_value=(object(), _fake_interleave_datasets)):
                train_dataset, stage_datasets = build_phase_train_dataset(
                    object(),
                    phase_config,
                    stage_specs,
                    tokenizer=object(),
                    image_size=48,
                )

        self.assertEqual(build_calls, ["stage1_easy_numeric"])
        self.assertEqual(train_dataset, "dataset:stage1_easy_numeric")
        self.assertEqual(stage_datasets, {"stage1_easy_numeric": "dataset:stage1_easy_numeric"})

    def test_build_eval_datasets_uses_phase_eval_stage_names(self):
        run_config = build_default_run_config("phase_b")
        run_config.stages["stage2_float_numeric"].enabled = False
        observed = []

        def _fake_build_stage_dataset(base_dataset, stage_spec, tokenizer, image_size=512):
            observed.append(stage_spec.name)
            return f"dataset:{stage_spec.name}"

        with mock.patch("staged_rl.data.build_stage_dataset", side_effect=_fake_build_stage_dataset):
            eval_datasets = build_eval_datasets(object(), run_config, tokenizer=object())

        self.assertEqual(
            observed,
            ["eval_overall_numeric", "stage1_easy_numeric", "stage3_hard_numeric"],
        )
        self.assertEqual(
            list(eval_datasets.keys()),
            ["eval_overall_numeric", "stage1_easy_numeric", "stage3_hard_numeric"],
        )


if __name__ == "__main__":
    unittest.main()
