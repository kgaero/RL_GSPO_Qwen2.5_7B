"""Tests for Kaggle reevaluation upload bundle preparation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.prepare_kaggle_reeval_uploads as prepare_uploads


class PrepareKaggleReevalUploadsTests(unittest.TestCase):
    def test_resolve_baseline_source_root_returns_none_without_source_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_root = Path(tmpdir)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(prepare_uploads, "ROOT", fake_root):
                self.assertIsNone(prepare_uploads.resolve_baseline_source_root())

    def test_resolve_baseline_source_root_prefers_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_root = Path(tmpdir) / "baseline_source"
            (explicit_root / "grpo_lora").mkdir(parents=True)
            (explicit_root / "grpo_eval_outputs").mkdir(parents=True)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(prepare_uploads, "ROOT", Path(tmpdir) / "empty_root"):
                self.assertEqual(
                    prepare_uploads.resolve_baseline_source_root(str(explicit_root)),
                    explicit_root.resolve(),
                )

    def test_prepare_baseline_dataset_bundle_copies_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_root = tmp_path / "source"
            (source_root / "grpo_lora").mkdir(parents=True)
            (source_root / "grpo_lora" / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
            (source_root / "grpo_eval_outputs").mkdir(parents=True)
            (source_root / "grpo_eval_outputs" / "eval_metrics.json").write_text("{}", encoding="utf-8")
            (source_root / "grpo_eval_outputs" / "train_log_summary.json").write_text("{}", encoding="utf-8")

            dataset_root = tmp_path / "dataset"
            prepared_root = prepare_uploads.prepare_baseline_dataset_bundle(dataset_root, source_root)

            self.assertEqual(prepared_root, dataset_root)
            self.assertTrue((dataset_root / "grpo_lora" / "adapter_model.safetensors").exists())
            self.assertTrue((dataset_root / "grpo_lora" / "checkpoint_info.json").exists())
            self.assertTrue((dataset_root / "grpo_eval_outputs" / "eval_metrics.json").exists())
            self.assertTrue((dataset_root / "grpo_eval_outputs" / "train_log_summary.json").exists())

            baseline_manifest = json.loads((dataset_root / "baseline_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline_manifest["adapter_dir"], "grpo_lora")
            self.assertEqual(baseline_manifest["legacy_eval_metrics"], "grpo_eval_outputs/eval_metrics.json")
            self.assertEqual(baseline_manifest["legacy_train_log_summary"], "grpo_eval_outputs/train_log_summary.json")

    def test_main_skips_baseline_bundle_when_no_source_artifacts_are_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bundle_root = tmp_path / "bundles"
            baseline_root = tmp_path / "baseline"
            code_root = tmp_path / "code"

            with (
                mock.patch.object(prepare_uploads, "resolve_baseline_source_root", return_value=None),
                mock.patch.object(prepare_uploads, "prepare_code_dataset_bundle", return_value=code_root) as prepare_code,
                mock.patch.object(prepare_uploads, "prepare_kernel_bundles", return_value=[bundle_root / "bundle"]) as prepare_kernels,
                mock.patch.object(prepare_uploads, "publish_baseline_dataset") as publish_baseline,
                mock.patch.object(prepare_uploads, "publish_code_dataset") as publish_code,
                mock.patch.object(prepare_uploads, "push_kernel_bundle") as push_kernel,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_kaggle_reeval_uploads.py",
                        "--kernel-bundle-root",
                        str(bundle_root),
                        "--baseline-dataset-root",
                        str(baseline_root),
                        "--code-dataset-root",
                        str(code_root),
                    ],
                ),
            ):
                prepare_uploads.main()

            prepare_code.assert_called_once()
            prepare_kernels.assert_called_once()
            publish_baseline.assert_not_called()
            publish_code.assert_not_called()
            push_kernel.assert_not_called()

    def test_main_requires_a_baseline_source_when_publish_is_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with (
                mock.patch.object(prepare_uploads, "resolve_baseline_source_root", return_value=None),
                mock.patch.object(sys, "argv", ["prepare_kaggle_reeval_uploads.py", "--publish-baseline-dataset"]),
            ):
                with self.assertRaises(FileNotFoundError):
                    prepare_uploads.main()


if __name__ == "__main__":
    unittest.main()
