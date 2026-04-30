import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FairnessBenchmarkHarnessTests(unittest.TestCase):
    def test_synthetic_grouped_dataset_is_deterministic(self):
        from benchmarks.fairness_classification.datasets import make_dataset

        first = make_dataset("synthetic_grouped", seed=3)
        second = make_dataset("synthetic_grouped", seed=3)

        self.assertEqual(first.x_train.shape, second.x_train.shape)
        self.assertTrue(np.array_equal(first.y_train, second.y_train))
        self.assertTrue(np.array_equal(first.sensitive_train, second.sensitive_train))
        self.assertTrue(np.array_equal(first.y_test, second.y_test))
        self.assertTrue(np.array_equal(first.sensitive_test, second.sensitive_test))

    def test_sklearn_grouped_datasets_are_offline_binary_benchmarks(self):
        from benchmarks.fairness_classification.datasets import make_dataset

        for name in ("sklearn_breast_cancer_grouped", "sklearn_wine_grouped"):
            dataset = make_dataset(name, seed=4)

            self.assertEqual(dataset.name, name)
            self.assertEqual(set(np.unique(dataset.y_train)) | set(np.unique(dataset.y_test)), {0, 1})
            self.assertEqual(set(np.unique(dataset.sensitive_train)) | set(np.unique(dataset.sensitive_test)), {0, 1})
            self.assertEqual(dataset.x_train.shape[1], dataset.x_test.shape[1])
            self.assertEqual(dataset.x_train.shape[1] >= 2, True)

    def test_group_metrics_report_expected_keys(self):
        from benchmarks.fairness_classification.metrics import compute_metrics

        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        sensitive = np.array([0, 0, 1, 1])

        metrics = compute_metrics(y_true, y_pred, sensitive)

        self.assertGreaterEqual(
            set(metrics),
            {
                "accuracy",
                "demographic_parity_gap",
                "equalized_odds_gap",
                "fairness_score",
                "group_0_n",
                "group_1_n",
                "group_0_selection_rate",
                "group_1_selection_rate",
                "group_0_false_positive_rate",
                "group_1_false_positive_rate",
                "group_0_false_negative_rate",
                "group_1_false_negative_rate",
            },
        )

    def test_group_metrics_respect_configured_fairness_penalty(self):
        from benchmarks.fairness_classification.metrics import compute_metrics

        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1])
        sensitive = np.array([0, 0, 1, 1])

        low_penalty = compute_metrics(y_true, y_pred, sensitive, fairness_penalty=0.1)
        high_penalty = compute_metrics(y_true, y_pred, sensitive, fairness_penalty=0.9)

        self.assertGreater(low_penalty["fairness_score"], high_penalty["fairness_score"])

    def test_harness_runs_all_methods_for_seed(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        config = {
            "datasets": ["synthetic_grouped"],
            "methods": ["logistic_regression", "preference_cone_threshold"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }

        result = run_fairness_benchmark(config)
        rows = result["rows"]

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["status"] for row in rows}, {"ok"})
        self.assertEqual({row["dataset"] for row in rows}, {"synthetic_grouped"})
        self.assertEqual({row["seed"] for row in rows}, {0})
        self.assertEqual(
            {row["method"] for row in rows},
            {"logistic_regression", "preference_cone_threshold"},
        )

    def test_harness_passes_configured_fairness_penalty_to_metrics(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        base = {
            "datasets": ["synthetic_grouped"],
            "methods": ["logistic_regression"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }
        low_penalty = run_fairness_benchmark({**base, "fairness_penalty": 0.1})["rows"][0]["metrics"]
        high_penalty = run_fairness_benchmark({**base, "fairness_penalty": 0.9})["rows"][0]["metrics"]

        self.assertGreater(low_penalty["fairness_score"], high_penalty["fairness_score"])

    def test_harness_runs_multiple_offline_datasets(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        config = {
            "datasets": [
                "synthetic_grouped",
                "sklearn_breast_cancer_grouped",
                "sklearn_wine_grouped",
            ],
            "methods": ["logistic_regression", "preference_cone_threshold"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }

        result = run_fairness_benchmark(config)
        rows = result["rows"]

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["status"] for row in rows}, {"ok"})
        self.assertEqual(
            {row["dataset"] for row in rows},
            {"synthetic_grouped", "sklearn_breast_cancer_grouped", "sklearn_wine_grouped"},
        )

    def test_harness_records_row_error_without_crashing_suite(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        config = {
            "datasets": ["synthetic_grouped"],
            "methods": ["missing_method"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }

        result = run_fairness_benchmark(config)

        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["status"], "error")
        self.assertIn("unknown method", result["rows"][0]["error"])

    def test_preference_cone_penalty_variant_is_a_real_method(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        config = {
            "datasets": ["synthetic_grouped"],
            "methods": ["preference_cone_threshold_penalty_0.20"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }

        result = run_fairness_benchmark(config)

        self.assertEqual(result["rows"][0]["status"], "ok")
        self.assertIn("fairness_score", result["rows"][0]["metrics"])

    def test_validation_selected_preference_cone_is_a_real_method(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        config = {
            "datasets": ["sklearn_breast_cancer_grouped"],
            "methods": ["validation_selected_preference_cone"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }

        result = run_fairness_benchmark(config)

        self.assertEqual(result["rows"][0]["status"], "ok")
        self.assertIn("fairness_score", result["rows"][0]["metrics"])

    def test_validation_selected_fairlearn_baseline_is_a_real_method(self):
        from benchmarks.fairness_classification.harness import run_fairness_benchmark

        config = {
            "datasets": ["synthetic_grouped"],
            "methods": ["validation_selected_fairlearn_baseline"],
            "seeds": [0],
            "primary_metric": "fairness_score",
        }

        result = run_fairness_benchmark(config)

        self.assertEqual(result["rows"][0]["status"], "ok")
        self.assertIn("fairness_score", result["rows"][0]["metrics"])

    def test_preference_cone_threshold_grid_is_capped_for_large_datasets(self):
        from benchmarks.fairness_classification.methods import PreferenceConeThresholdClassifier

        scores = np.linspace(0.0, 1.0, 1000)
        thresholds = PreferenceConeThresholdClassifier._candidate_thresholds(scores)

        self.assertLessEqual(len(thresholds), 130)
        self.assertEqual(thresholds[0], 0.0)
        self.assertEqual(thresholds[-1], 1.0)

    def test_ordinary_predictive_features_exclude_sensitive_attribute(self):
        from benchmarks.fairness_classification.methods import ordinary_predictive_features

        x = np.zeros((5, 4))

        features = ordinary_predictive_features(x)

        self.assertEqual(features.shape, (5, 3))

    def test_openml_adult_dataset_uses_meaningful_sex_attribute(self):
        import pandas as pd
        from benchmarks.fairness_classification.datasets import make_dataset

        frame = pd.DataFrame({
            "age": [25, 45, 32, 52, 23, 39, 61, 48, 30, 44, 55, 27],
            "education": ["Bachelors", "HS", "Masters", "HS", "HS", "Bachelors", "Masters", "HS", "Bachelors", "HS", "Masters", "HS"],
            "sex": ["Male", "Female", "Female", "Male", "Female", "Male", "Male", "Female", "Male", "Female", "Male", "Female"],
            "class": [">50K", "<=50K", ">50K", "<=50K", "<=50K", ">50K", ">50K", "<=50K", ">50K", "<=50K", ">50K", "<=50K"],
        })

        with patch("benchmarks.fairness_classification.datasets.fetch_openml", return_value=SimpleNamespace(frame=frame), create=True):
            dataset = make_dataset("openml_adult_sex", seed=1)

        self.assertEqual(dataset.name, "openml_adult_sex")
        self.assertEqual(set(np.unique(dataset.y_train)) | set(np.unique(dataset.y_test)), {0, 1})
        self.assertEqual(set(np.unique(dataset.sensitive_train)) | set(np.unique(dataset.sensitive_test)), {0, 1})
        self.assertGreaterEqual(dataset.x_train.shape[1], 2)

    def test_openml_german_credit_dataset_uses_personal_status_sex_proxy(self):
        import pandas as pd
        from benchmarks.fairness_classification.datasets import make_dataset

        frame = pd.DataFrame({
            "duration": [6, 12, 24, 18, 10, 36, 8, 30, 14, 20, 16, 28],
            "purpose": ["radio", "car", "education", "car", "radio", "business", "car", "radio", "education", "car", "business", "radio"],
            "personal_status": [
                "male single", "female div/dep/mar", "male div/sep", "female div/dep/mar",
                "male single", "female div/dep/mar", "male mar/wid", "female div/dep/mar",
                "male single", "female div/dep/mar", "male div/sep", "female div/dep/mar",
            ],
            "class": ["good", "bad", "good", "bad", "good", "bad", "good", "bad", "good", "bad", "good", "bad"],
        })

        with patch("benchmarks.fairness_classification.datasets.fetch_openml", return_value=SimpleNamespace(frame=frame), create=True):
            dataset = make_dataset("openml_german_credit_sex", seed=2)

        self.assertEqual(dataset.name, "openml_german_credit_sex")
        self.assertEqual(set(np.unique(dataset.y_train)) | set(np.unique(dataset.y_test)), {0, 1})
        self.assertEqual(set(np.unique(dataset.sensitive_train)) | set(np.unique(dataset.sensitive_test)), {0, 1})

    def test_fairlearn_credit_card_dataset_uses_sex_attribute(self):
        import pandas as pd
        from benchmarks.fairness_classification.datasets import make_dataset

        frame = pd.DataFrame({
            "x1": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            "x2": [1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2],
            "x3": [1, 2, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })

        with patch("benchmarks.fairness_classification.datasets.fetch_credit_card", return_value=SimpleNamespace(frame=frame), create=True):
            dataset = make_dataset("fairlearn_credit_card_sex", seed=3)

        self.assertEqual(dataset.name, "fairlearn_credit_card_sex")
        self.assertEqual(set(np.unique(dataset.y_train)) | set(np.unique(dataset.y_test)), {0, 1})
        self.assertEqual(set(np.unique(dataset.sensitive_train)) | set(np.unique(dataset.sensitive_test)), {0, 1})

    def test_fairlearn_bank_marketing_dataset_uses_age_attribute(self):
        import pandas as pd
        from benchmarks.fairness_classification.datasets import make_dataset

        frame = pd.DataFrame({
            "V1": [23, 61, 45, 32, 71, 28, 55, 39, 68, 25, 48, 52],
            "V2": ["admin", "retired", "blue", "admin", "retired", "blue", "admin", "blue", "retired", "admin", "blue", "admin"],
            "Class": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        })

        with patch("benchmarks.fairness_classification.datasets.fetch_bank_marketing", return_value=SimpleNamespace(frame=frame), create=True):
            dataset = make_dataset("fairlearn_bank_marketing_age", seed=4)

        self.assertEqual(dataset.name, "fairlearn_bank_marketing_age")
        self.assertEqual(set(np.unique(dataset.y_train)) | set(np.unique(dataset.y_test)), {0, 1})
        self.assertEqual(set(np.unique(dataset.sensitive_train)) | set(np.unique(dataset.sensitive_test)), {0, 1})


if __name__ == "__main__":
    unittest.main()
