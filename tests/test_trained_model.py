import json
import unittest
from statistics import mean

import joblib
import numpy as np
from xgboost import XGBClassifier

from reservation import ARTIFACT_PATH, CATEGORICAL, FEATURES, LABELS
from test_reservation import PUBLISHED_EVALUATION, assert_ui_predictions
from train import holdout_metrics, load_data, split_data


class TrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARTIFACT_PATH.is_file():
            raise RuntimeError("Run python train.py before the trained-model tests")
        cls.artifact = joblib.load(ARTIFACT_PATH)
        X, y = load_data()
        _, test, _, _ = split_data(X, y)
        cls.X_test, cls.y_test = X.iloc[test], y.iloc[test]

    def test_selected_model_reproduces_published_holdout(self):
        recorded = json.loads(PUBLISHED_EVALUATION.read_text())
        generated = json.loads(ARTIFACT_PATH.with_name("evaluation.json").read_text())
        for key in (
            "data_sha256", "versions", "seed", "rows", "train_rows", "test_rows",
            "train_canceled", "test_canceled", "feature_groups", "shared_train_test_groups",
            "features", "target_encoding", "selection_metric", "selected_model",
        ):
            with self.subTest(field=key):
                self.assertEqual(generated[key], recorded[key])

        self.assertEqual(generated["models"].keys(), recorded["models"].keys())
        for name, result in generated["models"].items():
            with self.subTest(model=name):
                folds = result["cv_f1_canceled_folds"]
                self.assertEqual(len(folds), 4)
                self.assertTrue(all(0 <= score <= 1 for score in folds))
                self.assertAlmostEqual(mean(folds), result["cv_f1_canceled_mean"])
                holdout = result["holdout"]
                (tn, fp), (fn, tp) = holdout["confusion_matrix_actual_by_predicted_0_1"]
                self.assertEqual(tn + fp + fn + tp, recorded["test_rows"])
                self.assertEqual(fn + tp, recorded["test_canceled"])
                self.assertAlmostEqual((tn + tp) / recorded["test_rows"], holdout["accuracy"])
                self.assertAlmostEqual(2 * tp / (2 * tp + fp + fn), holdout["f1_canceled"])

        selected = max(generated["models"], key=lambda name: generated["models"][name]["cv_f1_canceled_mean"])
        self.assertEqual(self.artifact["selected_model"], selected)
        self.assertEqual(generated["selected_model"], selected)
        self.assertAlmostEqual(
            generated["models"][selected]["cv_f1_canceled_mean"],
            recorded["models"][selected]["cv_f1_canceled_mean"],
            places=4,
        )
        pipeline = self.artifact["pipeline"]
        self.assertIsInstance(pipeline.named_steps["model"], XGBClassifier)
        self.assertEqual(list(pipeline.feature_names_in_), FEATURES)
        self.assertEqual(list(pipeline.classes_), [LABELS["Not_Canceled"], LABELS["Canceled"]])
        encoder = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
        self.assertEqual(self.artifact["categories"], dict(zip(CATEGORICAL, (c.tolist() for c in encoder.categories_))))

        actual_holdout = holdout_metrics(pipeline, self.X_test, self.y_test)
        published_holdout = recorded["models"][selected]["holdout"]
        matrix_key = "confusion_matrix_actual_by_predicted_0_1"
        self.assertEqual(actual_holdout[matrix_key], published_holdout[matrix_key])
        self.assertEqual(actual_holdout[matrix_key], generated["models"][selected]["holdout"][matrix_key])
        self.assertEqual(f"{actual_holdout['accuracy']:.2%}", "85.68%")
        self.assertEqual(f"{actual_holdout['f1_canceled']:.4f}", "0.7552")
        self.assertAlmostEqual(actual_holdout["roc_auc"], published_holdout["roc_auc"], places=4)

    def test_real_model_labels_and_inputs_on_holdout_examples(self):
        sample = self.X_test
        for name, (lower, upper) in self.artifact["bounds"].items():
            sample = sample.loc[sample[name].between(lower, upper)]
        for name, options in self.artifact["categories"].items():
            sample = sample.loc[sample[name].isin(options)]
        self.assertGreaterEqual(len(sample), 300)
        self.assertTrue(np.isfinite(sample.select_dtypes("number").to_numpy()).all())
        assert_ui_predictions(self, ARTIFACT_PATH, self.artifact["pipeline"], sample.sample(300, random_state=88))
